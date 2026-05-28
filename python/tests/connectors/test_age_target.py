"""Tests for the Apache AGE target connector.

Run with:
    uv run pytest python/tests/connectors/test_age_target.py -v

Unit tests run without a server. Integration tests require a running
PostgreSQL with Apache AGE extension (spun up via testcontainers when
``AGE_TEST_SERVER=1``).
"""

from __future__ import annotations

import os
import uuid as uuid_mod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
import pytest_asyncio

import cocoindex as coco
from cocoindex.connectors import age as ag
from cocoindex.connectors.age._cypher import (
    build_constraint_create,
    build_constraint_drop,
    build_edge_delete,
    build_edge_index_create,
    build_edge_index_drop,
    build_edge_upsert,
    build_vertex_delete,
    build_vertex_index_create,
    build_vertex_index_drop,
    build_vertex_upsert,
    validate_identifier,
)

from tests import common

coco_env = common.create_test_env(__file__)

KG_DB: coco.ContextKey[Any] = coco.ContextKey("test_age_kg")

_HAS_AGE_SERVER = bool(os.environ.get("AGE_TEST_SERVER"))

requires_age_server = pytest.mark.skipif(
    not _HAS_AGE_SERVER,
    reason="AGE_TEST_SERVER is not set",
)


# =============================================================================
# Unit tests — identifier validation (no DB)
# =============================================================================


class TestValidateIdentifier:
    @pytest.mark.parametrize(
        "name", ["users", "_private", "T1", "a_b_c", "X", "Document", "MENTION"]
    )
    def test_valid(self, name: str) -> None:
        validate_identifier(name, "test")

    @pytest.mark.parametrize(
        "name",
        ["my-table", "123abc", "", "has space", "ba`ck", "semi;colon", "a.b", "X-Y"],
    )
    def test_invalid(self, name: str) -> None:
        with pytest.raises(ValueError, match="Invalid AGE"):
            validate_identifier(name, "test")


# =============================================================================
# Unit tests — Cypher generation (no DB)
# =============================================================================


class TestVertexUpsertSql:
    def test_single_pk_with_props(self) -> None:
        sql = build_vertex_upsert("Document", ["filename"], ["title", "summary"])
        assert "CREATE" in sql
        assert "Document" in sql
        assert "$batch" in sql
        assert "title" in sql
        assert "summary" in sql

    def test_single_pk_no_value_fields(self) -> None:
        sql = build_vertex_upsert("Document", ["filename"], [])
        assert "CREATE" in sql

    def test_empty_pk_raises(self) -> None:
        with pytest.raises(ValueError, match="primary_key"):
            build_vertex_upsert("X", [], ["p"])

    def test_uses_unwound_batch(self) -> None:
        sql = build_vertex_upsert("Doc", ["id"], ["val"])
        assert "UNWIND $batch AS row" in sql


class TestVertexDeleteSql:
    def test_detach_delete(self) -> None:
        sql = build_vertex_delete("Document", ["filename"])
        assert "DETACH DELETE" in sql

    def test_unwinds_batch(self) -> None:
        sql = build_vertex_delete("Doc", ["id"])
        assert "UNWIND $batch AS row" in sql


class TestEdgeUpsertSql:
    def test_three_creates_with_props(self) -> None:
        sql = build_edge_upsert(
            "REL", "Entity", ["value"], "Entity", ["value"], ["id"],
            ["predicate"],
        )
        assert "UNWIND $batch AS row" in sql
        assert "row.from_key" in sql
        assert "row.to_key" in sql
        assert "row.rel_key" in sql
        assert "row.props" in sql

    def test_no_props(self) -> None:
        sql = build_edge_upsert(
            "REL", "A", ["x"], "B", ["y"], ["id"], [],
        )
        assert "row.props" not in sql

    def test_empty_pk_raises(self) -> None:
        with pytest.raises(ValueError):
            build_edge_upsert("REL", "A", ["x"], "B", ["y"], [], ["p"])


class TestEdgeDeleteSql:
    def test_unwinds_batch(self) -> None:
        sql = build_edge_delete("REL", ["id"])
        assert "DELETE" in sql
        assert "UNWIND $batch AS row" in sql


class TestIndexDdlSql:
    def test_vertex_index_create(self) -> None:
        sql = build_vertex_index_create("Document", ["filename"])
        assert "CREATE INDEX" in sql
        assert "Document" in sql

    def test_vertex_index_drop(self) -> None:
        sql = build_vertex_index_drop("Document", ["filename"])
        assert "DROP INDEX" in sql

    def test_edge_index_create(self) -> None:
        sql = build_edge_index_create("REL", ["id"])
        assert "CREATE INDEX" in sql

    def test_edge_index_drop(self) -> None:
        sql = build_edge_index_drop("REL", ["id"])
        assert "DROP INDEX" in sql


class TestConstraintDdlSql:
    def test_create(self) -> None:
        sql = build_constraint_create("Document", ["filename"])
        assert "CREATE" in sql
        assert "UNIQUE" in sql or "PRIMARY KEY" in sql

    def test_drop(self) -> None:
        sql = build_constraint_drop("Document", ["filename"])
        assert "DROP" in sql

    def test_empty_fields_raises(self) -> None:
        with pytest.raises(ValueError):
            build_constraint_create("X", [])


# =============================================================================
# Unit tests — TableSchema.from_class type mapping
# =============================================================================


class TestTableSchemaFromClass:
    @pytest.mark.asyncio
    async def test_basic_dataclass(self) -> None:
        @dataclass
        class Row:
            id: str
            count: int
            score: float
            flag: bool

        schema = await ag.TableSchema.from_class(Row, primary_key="id")
        assert schema.primary_key == "id"
        assert schema.columns["id"].type == "STRING"
        assert schema.columns["count"].type == "INTEGER"
        assert schema.columns["score"].type == "FLOAT"
        assert schema.columns["flag"].type == "BOOLEAN"
        assert schema.value_field_names == ["count", "score", "flag"]

    @pytest.mark.asyncio
    async def test_custom_pk(self) -> None:
        @dataclass
        class Doc:
            filename: str
            title: str

        schema = await ag.TableSchema.from_class(Doc, primary_key="filename")
        assert schema.primary_key == "filename"
        assert schema.value_field_names == ["title"]


# =============================================================================
# Identifier-validation-at-API-entry tests
# =============================================================================


class TestIdentifierValidationAtApiEntryPoints:
    def test_table_schema_invalid_column(self) -> None:
        with pytest.raises(ValueError, match="column name"):
            ag.TableSchema(
                columns={
                    "id": ag.ColumnDef(type="STRING"),
                    "bad-name": ag.ColumnDef(type="STRING"),
                },
                primary_key="id",
            )

    def test_table_schema_pk_must_exist_in_columns(self) -> None:
        with pytest.raises(ValueError, match="primary_key"):
            ag.TableSchema(
                columns={"id": ag.ColumnDef(type="STRING")},
                primary_key="missing",
            )

    def test_table_target_invalid_name(self) -> None:
        with pytest.raises(ValueError, match="table name"):
            ag.table_target(KG_DB, "bad-table")

    def test_relation_target_invalid_name(self) -> None:
        from typing import cast

        with pytest.raises(ValueError, match="relation table name"):
            ag.relation_target(
                KG_DB,
                "bad-rel",
                cast(Any, None),
                cast(Any, None),
            )

    def test_connection_factory_invalid_graph_name(self) -> None:
        with pytest.raises(ValueError, match="graph name"):
            ag.ConnectionFactory(
                dsn="postgresql://localhost:5432/postgres",
                graph="bad graph name",
            )


# =============================================================================
# Integration tests — require running PostgreSQL with AGE extension
# =============================================================================


@pytest.fixture(scope="module")
def age_conn_params() -> Any:
    """Spin up a PostgreSQL + AGE container once per test module.

    Returns a tuple (dsn, host, port, dbname, user, password) for the
    container's actual dynamically-allocated host/port.
    """
    if not _HAS_AGE_SERVER:
        pytest.skip("AGE_TEST_SERVER is not set")

    from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

    container = PostgresContainer(
        "apache/age:latest_PG17a",
        user="postgres",
        password="postgres",
        dbname="postgres",
    )
    container.start()
    try:
        dsn = container.get_connection_url().replace(
            "postgresql+psycopg2://", "postgresql://"
        )
        host = container.get_container_host_ip()
        mapped_port = container.get_exposed_port(5432)
        yield dsn, host, mapped_port, "postgres", "postgres", "postgres"
    finally:
        container.stop()


@pytest_asyncio.fixture
async def age_clean_env(
    age_conn_params: Any,
) -> AsyncIterator[tuple[str, str, str, str, str, str, str]]:
    """Create a unique graph per test, clean up at the end.

    Yields (dsn, host, port, dbname, user, password, graph_name).
    """
    import asyncpg  # type: ignore[import-untyped]
    from urllib.parse import urlparse

    dsn, host, port, dbname, user, password = age_conn_params
    graph_name = f"test_{uuid_mod.uuid4().hex[:8]}"

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1)
    async with pool.acquire() as conn:
        await conn.execute(f"SELECT * FROM ag_catalog.create_graph('{graph_name}')")
    await pool.close()

    yield dsn, host, port, dbname, user, password, graph_name

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1)
    async with pool.acquire() as conn:
        try:
            await conn.execute(
                f"SELECT * FROM ag_catalog.drop_graph('{graph_name}', true)"
            )
        except Exception:  # noqa: BLE001
            pass
    await pool.close()


async def _read_vertices(
    dsn: str, graph_name: str, label: str
) -> list[dict[str, Any]]:
    import asyncpg  # type: ignore[import-untyped]

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT * FROM cypher('{graph_name}', $$ "
            f"MATCH (n:`{label}`) RETURN n $$) AS (v agtype)"
        )
    await pool.close()
    out: list[dict[str, Any]] = []
    for row in rows:
        v = row["v"]
        if isinstance(v, dict):
            out.append(v)
        else:
            out.append({"v": str(v)})
    return out


async def _read_edges(
    dsn: str, graph_name: str, rel_type: str
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    import asyncpg  # type: ignore[import-untyped]

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT * FROM cypher('{graph_name}', $$ "
            f"MATCH (s)-[r:`{rel_type}`]->(t) "
            f"RETURN s, r, t $$) AS (s agtype, r agtype, t agtype)"
        )
    await pool.close()
    out: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        s = row["s"] if isinstance(row["s"], dict) else {"value": str(row["s"])}
        t = row["t"] if isinstance(row["t"], dict) else {"value": str(row["t"])}
        r = row["r"] if isinstance(row["r"], dict) else {"value": str(row["r"])}
        out.append((s, t, r))
    return out


# Per-test global state shared with the declare function.
_current_graph: str = ""
_node_rows: list[Any] = []
_rel_pairs: list[tuple[Any, Any]] = []


@dataclass
class Document:
    filename: str
    title: str
    summary: str


@dataclass
class Entity:
    value: str


@dataclass
class RelRow:
    id: str
    predicate: str


async def _declare_documents_only() -> None:
    schema = await ag.TableSchema.from_class(Document, primary_key="filename")
    table: Any = await coco.use_mount(  # type: ignore[call-overload]
        coco.component_subpath("setup", "doc_table"),
        ag.mount_table_target,  # type: ignore[arg-type]
        KG_DB,
        "Document",
        schema,
        primary_key="filename",
    )
    for row in _node_rows:
        table.declare_record(row=row)


async def _declare_entities_and_relationships() -> None:
    entity_schema = await ag.TableSchema.from_class(Entity, primary_key="value")
    rel_schema = await ag.TableSchema.from_class(RelRow, primary_key="id")
    entity_table: Any = await coco.use_mount(  # type: ignore[call-overload]
        coco.component_subpath("setup", "entity_table"),
        ag.mount_table_target,
        KG_DB,
        "Entity",
        entity_schema,
        primary_key="value",
    )
    rel_table: Any = await coco.use_mount(  # type: ignore[call-overload]
        coco.component_subpath("setup", "rel_table"),
        ag.mount_relation_target,
        KG_DB,
        "REL",
        entity_table,
        entity_table,
        rel_schema,
        primary_key="id",
    )
    seen_entities: set[str] = set()
    for from_id, to_id in _rel_pairs:
        for v in (from_id, to_id):
            if v not in seen_entities:
                entity_table.declare_record(row=Entity(value=v))
                seen_entities.add(v)
        rel_table.declare_relation(
            from_id=from_id,
            to_id=to_id,
            record=RelRow(id=f"{from_id}->{to_id}", predicate="connects"),
        )


@requires_age_server
@pytest.mark.asyncio
async def test_vertex_upsert_and_readback(age_clean_env: Any) -> None:
    global _current_graph, _node_rows
    dsn, host, port, dbname, user, password, graph_name = age_clean_env
    _current_graph = graph_name
    _node_rows = [
        Document(filename="a.md", title="A", summary="alpha"),
        Document(filename="b.md", title="B", summary="beta"),
    ]
    coco_env.context_provider.provide(
        KG_DB,
        ag.ConnectionFactory(
            dsn=dsn,
            graph=_current_graph,
        ),
    )
    app = coco.App(
        coco.AppConfig(name="test_age_vertex_upsert", environment=coco_env),
        _declare_documents_only,
    )
    await app.update()

    rows = await _read_vertices(dsn, _current_graph, "Document")
    by_fn = {r.get("filename", ""): r for r in rows}
    assert set(by_fn) == {"a.md", "b.md"}
    assert by_fn["a.md"]["title"] == "A"
    assert by_fn["a.md"]["summary"] == "alpha"


@requires_age_server
@pytest.mark.asyncio
async def test_reconcile_twice_is_noop(age_clean_env: Any) -> None:
    global _current_graph, _node_rows
    dsn, host, port, dbname, user, password, graph_name = age_clean_env
    _current_graph = graph_name
    _node_rows = [Document(filename="a.md", title="A", summary="alpha")]
    coco_env.context_provider.provide(
        KG_DB,
        ag.ConnectionFactory(
            dsn=dsn,
            graph=_current_graph,
        ),
    )
    app = coco.App(
        coco.AppConfig(name="test_age_noop", environment=coco_env),
        _declare_documents_only,
    )
    await app.update()
    rows1 = await _read_vertices(dsn, _current_graph, "Document")
    await app.update()
    rows2 = await _read_vertices(dsn, _current_graph, "Document")
    assert rows1 == rows2
    assert len(rows2) == 1


@requires_age_server
@pytest.mark.asyncio
async def test_modify_value_triggers_one_upsert(age_clean_env: Any) -> None:
    global _current_graph, _node_rows
    dsn, host, port, dbname, user, password, graph_name = age_clean_env
    _current_graph = graph_name
    _node_rows = [Document(filename="a.md", title="A", summary="alpha")]
    coco_env.context_provider.provide(
        KG_DB,
        ag.ConnectionFactory(
            dsn=dsn,
            graph=_current_graph,
        ),
    )
    app = coco.App(
        coco.AppConfig(name="test_age_modify", environment=coco_env),
        _declare_documents_only,
    )
    await app.update()
    _node_rows[0] = Document(filename="a.md", title="A", summary="ALPHA-2")
    await app.update()
    rows = await _read_vertices(dsn, _current_graph, "Document")
    assert len(rows) == 1
    assert rows[0]["summary"] == "ALPHA-2"


@requires_age_server
@pytest.mark.asyncio
async def test_delete_removes_vertex(age_clean_env: Any) -> None:
    global _current_graph, _node_rows
    dsn, host, port, dbname, user, password, graph_name = age_clean_env
    _current_graph = graph_name
    _node_rows = [
        Document(filename="a.md", title="A", summary="alpha"),
        Document(filename="b.md", title="B", summary="beta"),
    ]
    coco_env.context_provider.provide(
        KG_DB,
        ag.ConnectionFactory(
            dsn=dsn,
            graph=_current_graph,
        ),
    )
    app = coco.App(
        coco.AppConfig(name="test_age_delete", environment=coco_env),
        _declare_documents_only,
    )
    await app.update()
    assert {
        r.get("filename", "")
        for r in await _read_vertices(dsn, _current_graph, "Document")
    } == {"a.md", "b.md"}
    _node_rows.pop()
    await app.update()
    assert {
        r.get("filename", "")
        for r in await _read_vertices(dsn, _current_graph, "Document")
    } == {"a.md"}


@requires_age_server
@pytest.mark.asyncio
async def test_edge_upsert_with_endpoint_merge(age_clean_env: Any) -> None:
    global _current_graph, _rel_pairs
    dsn, host, port, dbname, user, password, graph_name = age_clean_env
    _current_graph = graph_name
    _rel_pairs = [("alice", "bob"), ("bob", "carol")]
    coco_env.context_provider.provide(
        KG_DB,
        ag.ConnectionFactory(
            dsn=dsn,
            graph=_current_graph,
        ),
    )
    app = coco.App(
        coco.AppConfig(name="test_age_rel_upsert", environment=coco_env),
        _declare_entities_and_relationships,
    )
    await app.update()

    nodes = await _read_vertices(dsn, _current_graph, "Entity")
    assert {n.get("value", "") for n in nodes} == {"alice", "bob", "carol"}

    edges = await _read_edges(dsn, _current_graph, "REL")
    assert len(edges) == 2
    pairs = {(s.get("value", ""), t.get("value", "")) for s, t, _ in edges}
    assert pairs == {("alice", "bob"), ("bob", "carol")}
    for _, _, rel in edges:
        assert rel.get("predicate") == "connects"
