"""
Apache AGE target for CocoIndex.

Two-level state system:
1. Table level — creates/drops AGE vertex/edge labels (via
   ``create_vlabel``/``create_elabel``), unique indexes on vertex properties
   (via ``agtype_access_operator`` expressions), and pgvector indexes on
   vector-valued properties.
2. Record level — upserts/deletes nodes via Cypher MERGE and edges via
   triple-MERGE (source, target, relationship), all wrapped in
   ``SELECT * FROM cypher(…)`` for AGE execution.

Multitenancy is by AGE graph name (one Postgres instance, many isolated
graphs); the graph name is part of the ``ConnectionFactory``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Generic,
    Literal,
    NamedTuple,
    Sequence,
)

from typing_extensions import TypeVar

try:
    import asyncpg as _asyncpg  # type: ignore[import-untyped]
except ImportError as e:
    raise ImportError(
        "asyncpg is required to use the AGE connector. "
        "Please install cocoindex[postgres]."
    ) from e

try:
    import age as _age  # type: ignore[import-untyped]
except ImportError as e:
    raise ImportError(
        "apache-age-python is required to use the AGE connector. "
        "Please install cocoindex[age]."
    ) from e


import msgspec
import numpy as np

import cocoindex as coco
from cocoindex._internal.context_keys import ContextKey, ContextProvider
from cocoindex._internal.datatype import (
    AnyType,
    MappingType,
    RecordType,
    SequenceType,
    TypeChecker,
    UnionType,
    analyze_type_info,
    is_record_type,
)
from cocoindex.connectorkits import statediff, target
from cocoindex.connectorkits.fingerprint import fingerprint_object
from cocoindex.resources import schema as res_schema

from . import _cypher

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Identifier validation
# ---------------------------------------------------------------------------

_IDENTIFIER_RE = _cypher.IDENTIFIER_RE
_validate_identifier = _cypher.validate_identifier


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The default graph OID used for vector index creation. AGE assigns graph OIDs
# sequentially starting from 1. We use 0 as a sentinel which works for the
# index expression because _label_name() resolves it at runtime.
_AGE_DEFAULT_GRAPH_OID: int = 0


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------


class ConnectionFactory:
    """Connection factory for Apache AGE.

    Holds connection parameters and creates an asyncpg connection pool on
    demand. The graph name is part of the factory (not the table key) —
    different graphs in the same Postgres instance are addressed via
    separate ``ConnectionFactory`` / ``ContextKey`` pairs.

    Example::

        factory = age.ConnectionFactory(
            dsn="postgresql://postgres:password@localhost:5432/postgres",
            graph="knowledge_graph",
        )
        builder.provide(AGE_DB, factory)
    """

    def __init__(
        self,
        dsn: str,
        *,
        graph: str = "default",
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        _validate_identifier(graph, "graph name")
        self._dsn = dsn
        self._graph = graph
        self._min_size = min_size
        self._max_size = max_size

    @property
    def graph(self) -> str:
        return self._graph

    async def acquire(self) -> Any:
        """Return a pool wrapper ready to issue Cypher/SQL queries."""
        pool = await _asyncpg.create_pool(
            self._dsn, min_size=self._min_size, max_size=self._max_size
        )
        return _GraphPool(pool, self._graph)


class _GraphPool:
    """Thin wrapper around an asyncpg pool that exposes ``query(sql, *args)``
    and ``execute(sql, *args)`` methods.

    Each call acquires a connection from the pool, initializes AGE on it if
    needed (``LOAD 'age'; SET search_path``), and releases it.
    """

    _pool: _asyncpg.Pool
    _graph: str

    def __init__(self, pool: _asyncpg.Pool, graph: str) -> None:
        self._pool = pool
        self._graph = graph

    @property
    def graph(self) -> str:
        return self._graph

    async def ensure_graph(self) -> None:
        """Ensure the AGE graph exists, creating it if necessary.

        Safe to call on every startup — checks existence first via
        ``ag_graph`` catalog.
        """
        async with self._pool.acquire() as conn:
            await self._init_connection(conn)
            exists = await conn.fetchval(_cypher.build_graph_exists_sql(), self._graph)
            if not exists:
                await conn.execute(_cypher.build_graph_create_sql(self._graph))

    async def query(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        """Execute a SQL statement and return rows as dicts."""
        async with self._pool.acquire() as conn:
            await self._init_connection(conn)
            if args:
                result = await conn.fetch(sql, *args)
            else:
                result = await conn.fetch(sql)
            return [dict(r) for r in result]

    async def execute(self, sql: str, *args: Any) -> str:
        """Execute a SQL statement that returns no rows."""
        async with self._pool.acquire() as conn:
            await self._init_connection(conn)
            if args:
                return await conn.execute(sql, *args)
            return await conn.execute(sql)

    async def close(self) -> None:
        """Close the underlying pool."""
        await self._pool.close()

    @staticmethod
    async def _init_connection(conn: _asyncpg.Connection) -> None:
        """Initialize AGE on a newly-acquired connection.

        The ``LOAD age`` and ``SET search_path`` are per-connection setup.
        We mark the connection attribute so subsequent calls are no-ops.
        """
        if getattr(conn, "_age_initialized", False):
            return
        await conn.execute("LOAD 'age'")
        await conn.execute("SET search_path TO ag_catalog, public")
        object.__setattr__(conn, "_age_initialized", True)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

_RowKey = tuple[Any, ...]
_ROW_KEY_CHECKER = TypeChecker(tuple[Any, ...])
_RowFingerprint = bytes


class _RelationRowValue(NamedTuple):
    """Value type for relation records.

    Endpoint metadata is structured (not pre-formatted strings) so values can
    bind via ``$row.``-parameters at query time rather than being string-interpolated.
    """

    from_label: str
    from_pk_field: str
    from_id: Any
    to_label: str
    to_pk_field: str
    to_id: Any
    fields: dict[str, Any]


_RowValue = dict[str, Any] | _RelationRowValue


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------


class AgeType(NamedTuple):
    """Annotation to override the default Python → AGE type mapping for a column.

    AGE (via Postgres) stores all property values as ``agtype``, which has
    native JSON-like types. Most values pass through transparently when
    serialized to JSON. Use with ``typing.Annotated``::

        from typing import Annotated
        from cocoindex.connectors.age import AgeType

        @dataclass
        class Row:
            id: str
            score: Annotated[float, AgeType("float8")]
    """

    age_type: str
    encoder: Any = None


def _decimal_str(value: Any) -> str:
    return str(value)


def _ndarray_to_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return value.tolist()  # type: ignore[no-any-return]


class _TypeMapping(NamedTuple):
    age_type: str
    encoder: Any = None


_LEAF_TYPE_MAPPINGS: dict[type, _TypeMapping] = {
    bool: _TypeMapping("boolean"),
    int: _TypeMapping("integer"),
    np.int8: _TypeMapping("integer"),
    np.int16: _TypeMapping("integer"),
    np.int32: _TypeMapping("integer"),
    np.int64: _TypeMapping("integer"),
    np.uint8: _TypeMapping("integer"),
    np.uint16: _TypeMapping("integer"),
    np.uint32: _TypeMapping("integer"),
    np.uint64: _TypeMapping("integer"),
    np.int_: _TypeMapping("integer"),
    np.uint: _TypeMapping("integer"),
    float: _TypeMapping("float"),
    np.float16: _TypeMapping("float"),
    np.float32: _TypeMapping("float"),
    np.float64: _TypeMapping("float"),
    str: _TypeMapping("string"),
    bytes: _TypeMapping("string"),
}

_OBJECT_MAPPING = _TypeMapping("map")
_ARRAY_MAPPING = _TypeMapping("array")


async def _get_type_mapping(
    python_type: Any, *, vector_schema: res_schema.VectorSchema | None = None
) -> _TypeMapping:
    type_info = analyze_type_info(python_type)
    for annotation in type_info.annotations:
        if isinstance(annotation, AgeType):
            return _TypeMapping(annotation.age_type, annotation.encoder)
    base_type = type_info.base_type
    if base_type in _LEAF_TYPE_MAPPINGS:
        return _LEAF_TYPE_MAPPINGS[base_type]

    if base_type is np.ndarray:
        if vector_schema is None:
            raise ValueError("VectorSchemaProvider is required for NumPy ndarray type.")
        if vector_schema.size <= 0:
            raise ValueError(f"Invalid vector dimension: {vector_schema.size}")
        return _TypeMapping(
            age_type=f"vector<float32, {vector_schema.size}>",
            encoder=_ndarray_to_list,
        )
    elif vector_schema is not None:
        raise ValueError(
            "VectorSchemaProvider is only supported for NumPy ndarray type. "
            f"Got type: {python_type}"
        )

    if isinstance(type_info.variant, (SequenceType,)):
        return _ARRAY_MAPPING
    if isinstance(type_info.variant, (MappingType, RecordType, UnionType, AnyType)):
        return _OBJECT_MAPPING
    return _OBJECT_MAPPING


# ---------------------------------------------------------------------------
# ColumnDef
# ---------------------------------------------------------------------------


class ColumnDef(NamedTuple):
    """Definition of a column (property) in an AGE label.

    ``type`` is metadata-only — AGE does not enforce per-property types
    server-side. The string contributes to the schema fingerprint and is
    surfaced in error messages, but no DDL is emitted from it.
    """

    type: str
    nullable: bool = True
    encoder: Any = None


# ---------------------------------------------------------------------------
# TableSchema
# ---------------------------------------------------------------------------

RowT = TypeVar("RowT", default=dict[str, Any])


@dataclass(slots=True)
class TableSchema(Generic[RowT]):
    """Schema definition for an AGE label (node label or relationship type).

    Single-field primary key (named via ``primary_key``, default ``"id"``).
    Compound primary keys are not supported in v1.0.
    """

    columns: dict[str, ColumnDef]
    primary_key: str
    row_type: type[RowT] | None

    def __init__(
        self,
        columns: dict[str, ColumnDef],
        *,
        primary_key: str = "id",
        row_type: type[RowT] | None = None,
    ) -> None:
        for col_name in columns:
            _validate_identifier(col_name, "column name")
        if primary_key not in columns:
            raise ValueError(
                f"primary_key {primary_key!r} not found in columns "
                f"({sorted(columns)!r})"
            )
        self.columns = columns
        self.primary_key = primary_key
        self.row_type = row_type

    @property
    def value_field_names(self) -> list[str]:
        return [c for c in self.columns if c != self.primary_key]

    @classmethod
    async def from_class(
        cls,
        record_type: type[RowT],
        *,
        primary_key: str = "id",
        column_overrides: dict[str, AgeType | res_schema.VectorSchemaProvider]
        | None = None,
    ) -> "TableSchema[RowT]":
        if not is_record_type(record_type):
            raise TypeError(
                f"record_type must be a record type (dataclass, NamedTuple, "
                f"Pydantic model), got {type(record_type)}"
            )
        columns = await cls._columns_from_record_type(record_type, column_overrides)
        return cls(columns, primary_key=primary_key, row_type=record_type)

    @staticmethod
    async def _columns_from_record_type(
        record_type: type,
        column_overrides: dict[str, AgeType | res_schema.VectorSchemaProvider] | None,
    ) -> dict[str, ColumnDef]:
        record_info = RecordType(record_type)
        columns: dict[str, ColumnDef] = {}

        for field in record_info.fields:
            type_info = analyze_type_info(field.type_hint)
            all_annotations: list[Any] = []
            if (override := column_overrides and column_overrides.get(field.name)) is not None:
                all_annotations.append(override)
            all_annotations.extend(type_info.annotations)

            age_type_annotation = next(
                (t for t in all_annotations if isinstance(t, AgeType)), None
            )
            vector_schema = None
            for annot in all_annotations:
                vs = await res_schema.get_vector_schema(annot)
                if vs is not None:
                    vector_schema = vs
                    break

            if age_type_annotation is not None:
                type_mapping = _TypeMapping(
                    age_type_annotation.age_type, age_type_annotation.encoder
                )
            else:
                type_mapping = await _get_type_mapping(
                    field.type_hint, vector_schema=vector_schema
                )

            columns[field.name] = ColumnDef(
                type=type_mapping.age_type.strip(),
                nullable=type_info.nullable,
                encoder=type_mapping.encoder,
            )

        return columns


# ---------------------------------------------------------------------------
# _RecordAction + _SharedRecordApplier
# ---------------------------------------------------------------------------


class _RecordAction(NamedTuple):
    """Action to perform on a record (upsert or delete)."""

    table_name: str
    is_relation: bool
    pk_field: str
    record_id: Any
    value: dict[str, Any] | None  # None = delete
    from_label: str | None
    from_pk_field: str | None
    from_id: Any | None
    to_label: str | None
    to_pk_field: str | None
    to_id: Any | None


class _SharedRecordApplier:
    """Owns a TargetActionSink shared by all record handlers for one graph.

    Actions are executed via ``SELECT * FROM cypher(…)`` against AGE.
    Each action's parameters are bundled as a JSON-serialized dict passed
    as the third argument to ``cypher()``.
    """

    _pool: _GraphPool
    sink: coco.TargetActionSink[_RecordAction, None]

    def __init__(self, pool: _GraphPool) -> None:
        self._pool = pool
        self.sink = coco.TargetActionSink.from_async_fn(self._apply_actions)

    async def _apply_actions(
        self, context_provider: ContextProvider, actions: Sequence[_RecordAction]
    ) -> None:
        if not actions:
            return

        upsert_normal: list[_RecordAction] = []
        upsert_relation: list[_RecordAction] = []
        delete_relation: list[_RecordAction] = []
        delete_normal: list[_RecordAction] = []

        for action in actions:
            if action.value is not None:
                if action.is_relation:
                    upsert_relation.append(action)
                else:
                    upsert_normal.append(action)
            else:
                if action.is_relation:
                    delete_relation.append(action)
                else:
                    delete_normal.append(action)

        for action in upsert_normal:
            await self._apply_node_upsert(action)
        for action in upsert_relation:
            await self._apply_relation_upsert(action)
        for action in delete_relation:
            await self._apply_relation_delete(action)
        for action in delete_normal:
            await self._apply_node_delete(action)

    async def _apply_node_upsert(self, action: _RecordAction) -> None:
        assert action.value is not None
        pk_value = action.value.get(action.pk_field, action.record_id)
        props = {k: v for k, v in action.value.items() if k != action.pk_field}
        cypher = _cypher.build_node_upsert(
            label=action.table_name,
            pk_fields=[action.pk_field],
            has_value_fields=bool(props),
        )
        params = {
            "key_0": pk_value,
        }
        if props:
            params["props"] = props
        sql = _cypher.wrap_cypher_sql(self._pool.graph, cypher)
        await self._pool.query(sql, json.dumps(params))

    async def _apply_node_delete(self, action: _RecordAction) -> None:
        cypher = _cypher.build_node_delete(
            label=action.table_name, pk_fields=[action.pk_field]
        )
        params = {"key_0": action.record_id}
        sql = _cypher.wrap_cypher_sql(self._pool.graph, cypher)
        await self._pool.query(sql, json.dumps(params))

    async def _apply_relation_upsert(self, action: _RecordAction) -> None:
        assert action.value is not None
        assert action.from_label is not None and action.from_pk_field is not None
        assert action.to_label is not None and action.to_pk_field is not None
        props = {k: v for k, v in action.value.items() if k != action.pk_field}
        cypher = _cypher.build_relationship_upsert(
            rel_type=action.table_name,
            from_label=action.from_label,
            from_pk_fields=[action.from_pk_field],
            to_label=action.to_label,
            to_pk_fields=[action.to_pk_field],
            rel_pk_fields=[action.pk_field],
            has_value_fields=bool(props),
        )
        params: dict[str, Any] = {
            "from_key_0": action.from_id,
            "to_key_0": action.to_id,
            "rel_key_0": action.record_id,
        }
        if props:
            params["props"] = props
        sql = _cypher.wrap_cypher_sql(self._pool.graph, cypher)
        await self._pool.query(sql, json.dumps(params))

    async def _apply_relation_delete(self, action: _RecordAction) -> None:
        cypher = _cypher.build_relationship_delete(
            rel_type=action.table_name, pk_fields=[action.pk_field]
        )
        params = {"key_0": action.record_id}
        sql = _cypher.wrap_cypher_sql(self._pool.graph, cypher)
        await self._pool.query(sql, json.dumps(params))


# ---------------------------------------------------------------------------
# Vector index
# ---------------------------------------------------------------------------


_METRIC_TO_PGVECTOR: dict[str, str] = {
    "cosine": "vector_cosine_ops",
    "l2": "vector_l2_ops",
    "ip": "vector_ip_ops",
}


class _VectorIndexSpec(NamedTuple):
    field: str
    metric: str
    dimension: int


_VectorIndexFingerprint = bytes


class _VectorIndexAction(NamedTuple):
    name: str
    table_name: str
    field: str
    spec: _VectorIndexSpec | None
    graph_name: str


_VectorIndexTrackingRecord = _VectorIndexSpec


class _VectorIndexHandler:
    """Attachment handler for vector indexes on an AGE node label."""

    _pool: _GraphPool
    _table_name: str
    _sink: coco.TargetActionSink[_VectorIndexAction, None]

    def __init__(self, pool: _GraphPool, table_name: str) -> None:
        self._pool = pool
        self._table_name = table_name
        self._sink = coco.TargetActionSink.from_async_fn(self._apply_actions)

    async def _apply_actions(
        self, context_provider: ContextProvider, actions: Sequence[_VectorIndexAction]
    ) -> None:
        for action in actions:
            if action.spec is None:
                sql = _cypher.build_vector_index_drop(
                    action.graph_name, action.name
                )
                try:
                    await self._pool.execute(sql)
                except Exception as e:  # noqa: BLE001
                    _logger.debug(
                        "AGE DROP INDEX %s (best-effort) failed: %s",
                        action.name,
                        e,
                    )
                continue

            # Drop-and-recreate so a metric/dimension change takes effect.
            try:
                await self._pool.execute(
                    _cypher.build_vector_index_drop(
                        action.graph_name, action.name
                    )
                )
            except Exception as e:  # noqa: BLE001
                _logger.debug(
                    "AGE DROP INDEX (pre-create best-effort) failed: %s",
                    e,
                )
            pg_metric = _METRIC_TO_PGVECTOR.get(action.spec.metric, action.spec.metric)
            sql = _cypher.build_vector_index_create(
                graph_name=action.graph_name,
                label=action.table_name,
                index_name=action.name,
                field=action.spec.field,
                vector_size=action.spec.dimension,
                metric=pg_metric,
            )
            await self._pool.execute(sql)

    def reconcile(
        self,
        key: coco.StableKey,
        desired_state: _VectorIndexSpec | coco.NonExistenceType,
        prev_possible_records: Collection[_VectorIndexTrackingRecord],
        prev_may_be_missing: bool,
        /,
    ) -> (
        coco.TargetReconcileOutput[_VectorIndexAction, _VectorIndexTrackingRecord, None]
        | None
    ):
        assert isinstance(key, str)
        if coco.is_non_existence(desired_state):
            if not prev_possible_records and not prev_may_be_missing:
                return None
            prev_field: str | None = None
            for prev in prev_possible_records:
                prev_field = prev.field
                break
            if prev_field is None:
                return None
            return coco.TargetReconcileOutput(
                action=_VectorIndexAction(
                    name=key,
                    table_name=self._table_name,
                    field=prev_field,
                    spec=None,
                    graph_name=self._pool.graph,
                ),
                sink=self._sink,
                tracking_record=coco.NON_EXISTENCE,
            )

        if not prev_may_be_missing and all(
            prev == desired_state for prev in prev_possible_records
        ):
            return None

        return coco.TargetReconcileOutput(
            action=_VectorIndexAction(
                name=key,
                table_name=self._table_name,
                field=desired_state.field,
                spec=desired_state,
                graph_name=self._pool.graph,
            ),
            sink=self._sink,
            tracking_record=desired_state,
        )


# ---------------------------------------------------------------------------
# _RecordHandler
# ---------------------------------------------------------------------------


class _RecordHandler(coco.TargetHandler[_RowValue, _RowFingerprint]):
    """Handler for record-level target states within an AGE label."""

    _table_name: str
    _is_relation: bool
    _pk_field: str
    _table_schema: TableSchema[Any] | None
    _pool: _GraphPool
    _sink: coco.TargetActionSink[_RecordAction, None]

    def __init__(
        self,
        table_name: str,
        is_relation: bool,
        pk_field: str,
        table_schema: TableSchema[Any] | None,
        pool: _GraphPool,
        sink: coco.TargetActionSink[_RecordAction, None],
    ) -> None:
        self._table_name = table_name
        self._is_relation = is_relation
        self._pk_field = pk_field
        self._table_schema = table_schema
        self._pool = pool
        self._sink = sink

    def attachments(self) -> dict[str, _VectorIndexHandler]:
        return {
            "vector_index": _VectorIndexHandler(self._pool, self._table_name),
        }

    def _encode_row(self, row_dict: dict[str, Any]) -> dict[str, Any]:
        if self._table_schema is None:
            return row_dict
        out: dict[str, Any] = {}
        for k, v in row_dict.items():
            col = self._table_schema.columns.get(k)
            if col is not None and col.encoder is not None and v is not None:
                out[k] = col.encoder(v)
            else:
                out[k] = v
        return out

    def reconcile(
        self,
        key: coco.StableKey,
        desired_state: _RowValue | coco.NonExistenceType,
        prev_possible_records: Collection[_RowFingerprint],
        prev_may_be_missing: bool,
        /,
    ) -> coco.TargetReconcileOutput[_RecordAction, _RowFingerprint, None] | None:
        key = _ROW_KEY_CHECKER.check(key)

        if coco.is_non_existence(desired_state):
            if not prev_possible_records and not prev_may_be_missing:
                return None
            return coco.TargetReconcileOutput(
                action=_RecordAction(
                    table_name=self._table_name,
                    is_relation=self._is_relation,
                    pk_field=self._pk_field,
                    record_id=key[0],
                    value=None,
                    from_label=None,
                    from_pk_field=None,
                    from_id=None,
                    to_label=None,
                    to_pk_field=None,
                    to_id=None,
                ),
                sink=self._sink,
                tracking_record=coco.NON_EXISTENCE,
            )

        target_fp = fingerprint_object(desired_state)
        if not prev_may_be_missing and all(
            prev == target_fp for prev in prev_possible_records
        ):
            return None

        if isinstance(desired_state, _RelationRowValue):
            from_label = desired_state.from_label
            from_pk_field = desired_state.from_pk_field
            from_id = desired_state.from_id
            to_label = desired_state.to_label
            to_pk_field = desired_state.to_pk_field
            to_id = desired_state.to_id
            encoded = self._encode_row(desired_state.fields)
        else:
            from_label = None
            from_pk_field = None
            from_id = None
            to_label = None
            to_pk_field = None
            to_id = None
            encoded = self._encode_row(desired_state)

        return coco.TargetReconcileOutput(
            action=_RecordAction(
                table_name=self._table_name,
                is_relation=self._is_relation,
                pk_field=self._pk_field,
                record_id=key[0],
                value=encoded,
                from_label=from_label,
                from_pk_field=from_pk_field,
                from_id=from_id,
                to_label=to_label,
                to_pk_field=to_pk_field,
                to_id=to_id,
            ),
            sink=self._sink,
            tracking_record=target_fp,
        )


# ---------------------------------------------------------------------------
# Table-level types
# ---------------------------------------------------------------------------


class _TableKey(NamedTuple):
    db_key: str
    table_name: str


_TABLE_KEY_CHECKER = TypeChecker(tuple[str, str])


@dataclass
class _TableSpec:
    table_schema: TableSchema[Any] | None
    primary_key: str
    is_relation: bool
    from_label: str | None
    from_pk_field: str | None
    to_label: str | None
    to_pk_field: str | None
    managed_by: target.ManagedBy = target.ManagedBy.SYSTEM


class _TableMainRecord(msgspec.Struct, frozen=True):
    """Tracking record for table-level properties."""

    has_schema: bool
    is_relation: bool
    primary_key: str
    pk_type: str | None
    from_label: str | None
    from_pk_field: str | None
    to_label: str | None
    to_pk_field: str | None


class _FieldTrackingRecord(msgspec.Struct, frozen=True):
    """Per-field tracking record."""

    age_type: str
    nullable: bool


_FIELD_SUBKEY_PREFIX: str = "field:"


def _field_subkey(name: str) -> str:
    return f"{_FIELD_SUBKEY_PREFIX}{name}"


class _TableAction(NamedTuple):
    key: _TableKey
    spec: _TableSpec | coco.NonExistenceType
    is_relation: bool
    main_action: statediff.DiffAction | None
    column_actions: dict[str, statediff.DiffAction]
    prev_pk_field: str | None
    prev_is_relation: bool


def _table_composite_tracking_record_from_spec(
    spec: _TableSpec,
) -> statediff.CompositeTrackingRecord[_TableMainRecord, str, _FieldTrackingRecord]:
    schema = spec.table_schema
    has_schema = schema is not None
    pk_type: str | None = None
    sub: dict[str, _FieldTrackingRecord] = {}

    if schema is not None:
        pk_col = schema.columns.get(spec.primary_key)
        if pk_col is not None:
            pk_type = pk_col.type
        for col_name, col_def in schema.columns.items():
            if col_name == spec.primary_key:
                continue
            sub[_field_subkey(col_name)] = _FieldTrackingRecord(
                age_type=col_def.type,
                nullable=col_def.nullable,
            )

    main = _TableMainRecord(
        has_schema=has_schema,
        is_relation=spec.is_relation,
        primary_key=spec.primary_key,
        pk_type=pk_type,
        from_label=spec.from_label,
        from_pk_field=spec.from_pk_field,
        to_label=spec.to_label,
        to_pk_field=spec.to_pk_field,
    )
    return statediff.CompositeTrackingRecord(main=main, sub=sub)


_TableTrackingRecord = statediff.MutualTrackingRecord[
    statediff.CompositeTrackingRecord[_TableMainRecord, str, _FieldTrackingRecord]
]


# ---------------------------------------------------------------------------
# _TableHandler
# ---------------------------------------------------------------------------


class _TableHandler(
    coco.TargetHandler[_TableSpec, _TableTrackingRecord, _RecordHandler]
):
    """Handler for table-level state — AGE label DDL + indexes."""

    _sink: coco.TargetActionSink[_TableAction, _RecordHandler]

    def __init__(self) -> None:
        self._sink = coco.TargetActionSink.from_async_fn(self._apply_actions)

    def reconcile(
        self,
        key: coco.StableKey,
        desired_state: _TableSpec | coco.NonExistenceType,
        prev_possible_records: Collection[_TableTrackingRecord],
        prev_may_be_missing: bool,
        /,
    ) -> (
        coco.TargetReconcileOutput[_TableAction, _TableTrackingRecord, _RecordHandler]
        | None
    ):
        key = _TableKey(*_TABLE_KEY_CHECKER.check(key))

        if coco.is_non_existence(desired_state):
            tracking_record: _TableTrackingRecord | coco.NonExistenceType = (
                coco.NON_EXISTENCE
            )
            is_relation = False
        else:
            tracking_record = statediff.MutualTrackingRecord(
                tracking_record=_table_composite_tracking_record_from_spec(
                    desired_state
                ),
                managed_by=desired_state.managed_by,
            )
            is_relation = desired_state.is_relation

        resolved = statediff.resolve_system_transition(
            statediff.TrackingRecordTransition(
                tracking_record, prev_possible_records, prev_may_be_missing
            )
        )
        main_action, column_transitions = statediff.diff_composite(resolved)

        column_actions: dict[str, statediff.DiffAction] = {}
        if main_action is None:
            for sub_key, t in column_transitions.items():
                action = statediff.diff(t)
                if action is not None:
                    column_actions[sub_key] = action

        if (
            main_action is None
            and not column_actions
            and coco.is_non_existence(desired_state)
        ):
            return None

        prev_pk_field: str | None = None
        prev_is_relation = False
        for prev in prev_possible_records:
            if prev.managed_by != target.ManagedBy.SYSTEM:
                continue
            prev_pk_field = prev.tracking_record.main.primary_key
            prev_is_relation = prev.tracking_record.main.is_relation
            break

        child_invalidation: Literal["destructive", "lossy"] | None = None
        if main_action == "replace":
            child_invalidation = "destructive"
        elif main_action is None and any(
            a != "insert" for a in column_actions.values()
        ):
            child_invalidation = "lossy"

        return coco.TargetReconcileOutput(
            action=_TableAction(
                key=key,
                spec=desired_state,
                is_relation=is_relation,
                main_action=main_action,
                column_actions=column_actions,
                prev_pk_field=prev_pk_field,
                prev_is_relation=prev_is_relation,
            ),
            sink=self._sink,
            tracking_record=tracking_record,
            child_invalidation=child_invalidation,
        )

    async def _apply_actions(
        self, context_provider: ContextProvider, actions: Sequence[_TableAction]
    ) -> list[coco.ChildTargetDef[_RecordHandler] | None]:
        actions_list = list(actions)
        outputs: list[coco.ChildTargetDef[_RecordHandler] | None] = [None] * len(
            actions_list
        )

        by_db: dict[str, list[int]] = {}
        for i, action in enumerate(actions_list):
            by_db.setdefault(action.key.db_key, []).append(i)

        for db_key, idxs in by_db.items():
            factory: ConnectionFactory = context_provider.get(db_key)  # type: ignore[assignment]
            pool = await factory.acquire()
            shared_applier = _SharedRecordApplier(pool)

            create_normal: list[int] = []
            create_relation: list[int] = []
            remove_relation: list[int] = []
            remove_normal: list[int] = []

            for i in idxs:
                action = actions_list[i]
                if coco.is_non_existence(action.spec):
                    if action.is_relation:
                        remove_relation.append(i)
                    else:
                        remove_normal.append(i)
                else:
                    if action.is_relation:
                        create_relation.append(i)
                    else:
                        create_normal.append(i)

            ordered = create_normal + create_relation + remove_relation + remove_normal

            for i in ordered:
                action = actions_list[i]
                spec = action.spec

                if action.main_action in ("replace", "delete"):
                    await self._drop_table_artifacts(pool, action.key.table_name, action)

                if coco.is_non_existence(spec):
                    outputs[i] = None
                    continue

                if action.main_action in ("insert", "upsert", "replace"):
                    await self._create_label(pool, action.key, spec)

                outputs[i] = coco.ChildTargetDef(
                    handler=_RecordHandler(
                        table_name=action.key.table_name,
                        is_relation=spec.is_relation,
                        pk_field=spec.primary_key,
                        table_schema=spec.table_schema,
                        pool=pool,
                        sink=shared_applier.sink,
                    )
                )

        return outputs

    @staticmethod
    async def _create_label(
        pool: _GraphPool, key: _TableKey, spec: _TableSpec
    ) -> None:
        """Create the AGE label and supporting unique index."""
        await pool.ensure_graph()

        # Create the label.
        try:
            sql = _cypher.build_label_create_sql(
                pool.graph, key.table_name, spec.is_relation
            )
            await pool.execute(sql)
        except Exception as e:  # noqa: BLE001
            _logger.debug(
                "AGE create_%slabel %s (best-effort) failed: %s",
                "e" if spec.is_relation else "v",
                key.table_name,
                e,
            )

        # Create a unique index on the primary key property.
        index_name = f"coco_uniq_{key.table_name}__{spec.primary_key}"
        try:
            sql = _cypher.build_node_index_create(
                pool.graph, key.table_name, index_name, [spec.primary_key]
            )
            await pool.execute(sql)
        except Exception as e:  # noqa: BLE001
            _logger.debug(
                "AGE CREATE UNIQUE INDEX %s (best-effort) failed: %s",
                index_name,
                e,
            )

    @staticmethod
    async def _drop_table_artifacts(
        pool: _GraphPool, table_name: str, action: _TableAction
    ) -> None:
        """Drop the supporting index and label on teardown."""
        pk_field = action.prev_pk_field
        if pk_field is None and isinstance(action.spec, _TableSpec):
            pk_field = action.spec.primary_key
        if pk_field is None:
            return

        # Drop the unique index.
        index_name = f"coco_uniq_{table_name}__{pk_field}"
        try:
            sql = _cypher.build_node_index_drop(pool.graph, index_name)
            await pool.execute(sql)
        except Exception as e:  # noqa: BLE001
            _logger.debug(
                "AGE DROP INDEX %s (best-effort) failed: %s",
                index_name,
                e,
            )

        # Drop the label.
        try:
            sql = _cypher.build_label_drop_sql(pool.graph, table_name)
            await pool.execute(sql)
        except Exception as e:  # noqa: BLE001
            _logger.debug(
                "AGE drop_label %s (best-effort) failed: %s",
                table_name,
                e,
            )


# ---------------------------------------------------------------------------
# Root provider registration
# ---------------------------------------------------------------------------

_table_provider = coco.register_root_target_states_provider(
    "cocoindex/age/table", _TableHandler()
)


# ---------------------------------------------------------------------------
# TableTarget
# ---------------------------------------------------------------------------


class TableTarget(
    Generic[RowT, coco.MaybePendingS], coco.ResolvesTo["TableTarget[RowT]"]
):
    """A target for writing records to an AGE node table."""

    _provider: coco.TargetStateProvider[_RowValue, None, coco.MaybePendingS]
    _table_schema: TableSchema[RowT] | None
    _table_name: str
    _primary_key: str

    def __init__(
        self,
        provider: coco.TargetStateProvider[_RowValue, None, coco.MaybePendingS],
        table_schema: TableSchema[RowT] | None,
        table_name: str,
        primary_key: str,
    ) -> None:
        self._provider = provider
        self._table_schema = table_schema
        self._table_name = table_name
        self._primary_key = primary_key

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def primary_key(self) -> str:
        return self._primary_key

    def declare_record(self: TableTarget[RowT], *, row: RowT) -> None:
        """Declare a record (node) to be upserted to this table."""
        row_dict = self._row_to_dict(row)
        if self._primary_key not in row_dict:
            raise ValueError(f"row is missing primary key field {self._primary_key!r}")
        pk_values = (row_dict[self._primary_key],)
        coco.declare_target_state(self._provider.target_state(pk_values, row_dict))

    declare_row = declare_record

    def _row_to_dict(self, row: RowT) -> dict[str, Any]:
        if self._table_schema is not None:
            out: dict[str, Any] = {}
            for col_name, col in self._table_schema.columns.items():
                if isinstance(row, dict):
                    value = row.get(col_name)
                else:
                    value = getattr(row, col_name)
                if value is not None and col.encoder is not None:
                    value = col.encoder(value)
                out[col_name] = value
            return out
        if isinstance(row, dict):
            return dict(row)
        record_info = RecordType(type(row))
        return {f.name: getattr(row, f.name) for f in record_info.fields}

    def declare_vector_index(
        self: TableTarget[RowT],
        *,
        name: str | None = None,
        field: str,
        metric: Literal["cosine", "l2", "ip"] = "cosine",
        dimension: int,
    ) -> None:
        """Declare a pgvector index on a column of this table."""
        _validate_identifier(field, "vector index field")
        if name is None:
            name = _cypher.vector_index_name("", self._table_name, field)
        _validate_identifier(name, "vector index name")
        if dimension <= 0:
            raise ValueError(f"Invalid vector dimension: {dimension}")
        spec = _VectorIndexSpec(field=field, metric=metric, dimension=dimension)
        att_provider = self._provider.attachment("vector_index")
        coco.declare_target_state(att_provider.target_state(name, spec))

    def __coco_memo_key__(self) -> str:
        return self._provider.memo_key


# ---------------------------------------------------------------------------
# RelationTarget
# ---------------------------------------------------------------------------


class RelationTarget(
    Generic[RowT, coco.MaybePendingS], coco.ResolvesTo["RelationTarget[RowT]"]
):
    """A target for writing relation records (edges) to an AGE relationship type."""

    _provider: coco.TargetStateProvider[_RowValue, None, coco.MaybePendingS]
    _table_name: str
    _table_schema: TableSchema[RowT] | None
    _primary_key: str
    _from_table: TableTarget[Any]
    _to_table: TableTarget[Any]

    def __init__(
        self,
        provider: coco.TargetStateProvider[_RowValue, None, coco.MaybePendingS],
        table_name: str,
        table_schema: TableSchema[RowT] | None,
        primary_key: str,
        from_table: TableTarget[Any],
        to_table: TableTarget[Any],
    ) -> None:
        self._provider = provider
        self._table_name = table_name
        self._table_schema = table_schema
        self._primary_key = primary_key
        self._from_table = from_table
        self._to_table = to_table

    def declare_relation(
        self: RelationTarget[RowT],
        *,
        from_id: Any,
        to_id: Any,
        record: RowT | None = None,
    ) -> None:
        """Declare a relation record (edge)."""
        from_label = self._from_table.table_name
        from_pk_field = self._from_table.primary_key
        to_label = self._to_table.table_name
        to_pk_field = self._to_table.primary_key

        if record is not None:
            if self._table_schema is not None:
                row_dict: dict[str, Any] = {}
                for col_name, col in self._table_schema.columns.items():
                    if col_name == self._primary_key:
                        continue
                    if isinstance(record, dict):
                        value = record.get(col_name)
                    else:
                        value = getattr(record, col_name)
                    if value is not None and col.encoder is not None:
                        value = col.encoder(value)
                    row_dict[col_name] = value
                record_id = (
                    record.get(self._primary_key)
                    if isinstance(record, dict)
                    else getattr(record, self._primary_key, None)
                )
            elif isinstance(record, dict):
                row_dict = {k: v for k, v in record.items() if k != self._primary_key}
                record_id = record.get(self._primary_key)
            else:
                record_info = RecordType(type(record))
                row_dict = {
                    f.name: getattr(record, f.name)
                    for f in record_info.fields
                    if f.name != self._primary_key
                }
                record_id = getattr(record, self._primary_key, None)
        else:
            row_dict = {}
            record_id = None

        if record_id is None:
            record_id = f"{from_label}_{from_id}_{to_label}_{to_id}"

        row_value: _RowValue = _RelationRowValue(
            from_label=from_label,
            from_pk_field=from_pk_field,
            from_id=from_id,
            to_label=to_label,
            to_pk_field=to_pk_field,
            to_id=to_id,
            fields=row_dict,
        )

        pk_values = (record_id,)
        coco.declare_target_state(self._provider.target_state(pk_values, row_value))

    def __coco_memo_key__(self) -> str:
        return self._provider.memo_key


# ---------------------------------------------------------------------------
# Module-level entry points
# ---------------------------------------------------------------------------


def table_target(
    db: ContextKey[ConnectionFactory],
    table_name: str,
    table_schema: TableSchema[RowT] | None = None,
    *,
    primary_key: str = "id",
    managed_by: target.ManagedBy = target.ManagedBy.SYSTEM,
) -> coco.TargetState[_RecordHandler]:
    """Create a ``TargetState`` for an AGE node table (label)."""
    _validate_identifier(table_name, "table name")
    _validate_identifier(primary_key, "primary key")
    if table_schema is not None and table_schema.primary_key != primary_key:
        raise ValueError(
            f"primary_key {primary_key!r} does not match the schema's "
            f"declared primary_key {table_schema.primary_key!r}"
        )
    key = _TableKey(db_key=db.key, table_name=table_name)
    spec = _TableSpec(
        table_schema=table_schema,
        primary_key=primary_key,
        is_relation=False,
        from_label=None,
        from_pk_field=None,
        to_label=None,
        to_pk_field=None,
        managed_by=managed_by,
    )
    return _table_provider.target_state(key, spec)


def declare_table_target(
    db: ContextKey[ConnectionFactory],
    table_name: str,
    table_schema: TableSchema[RowT] | None = None,
    *,
    primary_key: str = "id",
    managed_by: target.ManagedBy = target.ManagedBy.SYSTEM,
) -> TableTarget[RowT, coco.PendingS]:
    """Declare a node table target (no records flow into this declaration's
    own handler — typically used to register a table that is referenced as a
    relationship endpoint by other handlers)."""
    if table_schema is not None and table_schema.primary_key != primary_key:
        raise ValueError(
            f"primary_key {primary_key!r} does not match schema's {table_schema.primary_key!r}"
        )
    pk = table_schema.primary_key if table_schema is not None else primary_key
    provider = coco.declare_target_state_with_child(
        table_target(
            db, table_name, table_schema, primary_key=pk, managed_by=managed_by
        )
    )
    return TableTarget(provider, table_schema, table_name, pk)


async def mount_table_target(
    db: ContextKey[ConnectionFactory],
    table_name: str,
    table_schema: TableSchema[RowT] | None = None,
    *,
    primary_key: str = "id",
    managed_by: target.ManagedBy = target.ManagedBy.SYSTEM,
) -> TableTarget[RowT]:
    """Mount a node table target ready to receive ``declare_record`` calls."""
    if table_schema is not None and table_schema.primary_key != primary_key:
        raise ValueError(
            f"primary_key {primary_key!r} does not match schema's {table_schema.primary_key!r}"
        )
    pk = table_schema.primary_key if table_schema is not None else primary_key
    provider = await coco.mount_target(
        table_target(
            db, table_name, table_schema, primary_key=pk, managed_by=managed_by
        )
    )
    return TableTarget(provider, table_schema, table_name, pk)


def relation_target(
    db: ContextKey[ConnectionFactory],
    table_name: str,
    from_table: TableTarget[Any],
    to_table: TableTarget[Any],
    table_schema: TableSchema[RowT] | None = None,
    *,
    primary_key: str = "id",
    managed_by: target.ManagedBy = target.ManagedBy.SYSTEM,
) -> coco.TargetState[_RecordHandler]:
    """Create a ``TargetState`` for an AGE relationship type."""
    _validate_identifier(table_name, "relation table name")
    _validate_identifier(primary_key, "primary key")
    _validate_identifier(from_table.table_name, "from table name")
    _validate_identifier(to_table.table_name, "to table name")
    if table_schema is not None and table_schema.primary_key != primary_key:
        raise ValueError(
            f"primary_key {primary_key!r} does not match schema's {table_schema.primary_key!r}"
        )
    key = _TableKey(db_key=db.key, table_name=table_name)
    spec = _TableSpec(
        table_schema=table_schema,
        primary_key=primary_key,
        is_relation=True,
        from_label=from_table.table_name,
        from_pk_field=from_table.primary_key,
        to_label=to_table.table_name,
        to_pk_field=to_table.primary_key,
        managed_by=managed_by,
    )
    return _table_provider.target_state(key, spec)


def declare_relation_target(
    db: ContextKey[ConnectionFactory],
    table_name: str,
    from_table: TableTarget[Any],
    to_table: TableTarget[Any],
    table_schema: TableSchema[RowT] | None = None,
    *,
    primary_key: str = "id",
    managed_by: target.ManagedBy = target.ManagedBy.SYSTEM,
) -> RelationTarget[RowT, coco.PendingS]:
    """Declare a relation table target."""
    pk = table_schema.primary_key if table_schema is not None else primary_key
    provider = coco.declare_target_state_with_child(
        relation_target(
            db,
            table_name,
            from_table,
            to_table,
            table_schema,
            primary_key=pk,
            managed_by=managed_by,
        )
    )
    return RelationTarget(provider, table_name, table_schema, pk, from_table, to_table)


async def mount_relation_target(
    db: ContextKey[ConnectionFactory],
    table_name: str,
    from_table: TableTarget[Any],
    to_table: TableTarget[Any],
    table_schema: TableSchema[RowT] | None = None,
    *,
    primary_key: str = "id",
    managed_by: target.ManagedBy = target.ManagedBy.SYSTEM,
) -> RelationTarget[RowT]:
    """Mount a relation table target ready to receive ``declare_relation`` calls."""
    pk = table_schema.primary_key if table_schema is not None else primary_key
    provider = await coco.mount_target(
        relation_target(
            db,
            table_name,
            from_table,
            to_table,
            table_schema,
            primary_key=pk,
            managed_by=managed_by,
        )
    )
    return RelationTarget(provider, table_name, table_schema, pk, from_table, to_table)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "ColumnDef",
    "ConnectionFactory",
    "AgeType",
    "RelationTarget",
    "TableSchema",
    "TableTarget",
    "declare_relation_target",
    "declare_table_target",
    "mount_relation_target",
    "mount_table_target",
    "relation_target",
    "table_target",
]
