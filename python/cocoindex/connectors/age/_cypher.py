"""
Pure Cypher/SQL generation for the Apache AGE connector.

This module has no runtime dependency on the ``age``, ``psycopg``, or
``asyncpg`` drivers and no I/O — every function returns a string suitable
for the caller to execute against a PostgreSQL+AGE backend.

All identifiers (graph names, labels, property names, index names) are
validated by the caller before being passed in; values always bind via
``$``-parameters that are resolved against the ``agtype`` third argument
to the ``cypher()`` function.
"""

from __future__ import annotations

import re
from typing import Sequence

__all__ = [
    "IDENTIFIER_RE",
    "validate_identifier",
    "build_node_upsert",
    "build_node_delete",
    "build_relationship_upsert",
    "build_relationship_delete",
    "build_node_index_create",
    "build_node_index_drop",
    "build_vector_index_create",
    "build_vector_index_drop",
    "build_label_create_sql",
    "build_label_drop_sql",
    "build_graph_create_sql",
    "build_graph_exists_sql",
    "wrap_cypher_sql",
    "vector_index_name",
]


IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_identifier(name: str, kind: str) -> None:
    """Reject anything that isn't ``[a-zA-Z_][a-zA-Z0-9_]*``.

    AGE identifiers (graph names, labels, property names, index names) cannot
    be parameter-bound, so untrusted names must be validated at API entry —
    never escaped at query construction time.
    """
    if not IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid AGE {kind}: {name!r}. Must match [a-zA-Z_][a-zA-Z0-9_]*."
        )


def _quote(name: str) -> str:
    """Double-quote an already-validated identifier for inline use in SQL/Cypher."""
    return f'"{name}"'


def _key_clause(prefix: str, fields: Sequence[str], var: str = "row") -> str:
    """Build ``{<f1>: $<prefix>_0, <f2>: $<prefix>_1, ...}`` for a MERGE/MATCH pattern."""
    parts = [f"{_quote(f)}: $row.{prefix}_{i}" for i, f in enumerate(fields)]
    return "{" + ", ".join(parts) + "}"


# ---------------------------------------------------------------------------
# Cypher DML — pure Cypher fragments for use inside SELECT * FROM cypher()
# ---------------------------------------------------------------------------


def build_node_upsert(
    label: str,
    pk_fields: Sequence[str],
    has_value_fields: bool,
) -> str:
    """``MERGE (n:`Label` {pk: $row.key_0, ...}) [SET n += $row.props]``."""
    if not pk_fields:
        raise ValueError("build_node_upsert requires at least one primary key field")
    cypher = f"MERGE (n:{_quote(label)} {_key_clause('key', pk_fields)})"
    if has_value_fields:
        cypher += " SET n += $row.props"
    return cypher


def build_node_delete(label: str, pk_fields: Sequence[str]) -> str:
    """``MATCH (n:`Label` {pk: $row.key_0, ...}) DETACH DELETE n``."""
    if not pk_fields:
        raise ValueError("build_node_delete requires at least one primary key field")
    return (
        f"MATCH (n:{_quote(label)} {_key_clause('key', pk_fields)}) "
        f"DETACH DELETE n"
    )


def build_relationship_upsert(
    rel_type: str,
    from_label: str,
    from_pk_fields: Sequence[str],
    to_label: str,
    to_pk_fields: Sequence[str],
    rel_pk_fields: Sequence[str],
    has_value_fields: bool,
) -> str:
    """Three MERGEs: source endpoint, target endpoint, then the relationship.

    Endpoint properties are NOT touched — they are owned by their table's own
    record handler. We only ``SET r += $row.props`` on the relationship itself.
    """
    if not from_pk_fields or not to_pk_fields or not rel_pk_fields:
        raise ValueError(
            "build_relationship_upsert requires PK fields for from, to, and the relationship"
        )
    cypher = (
        f"MERGE (s:{_quote(from_label)} {_key_clause('from_key', from_pk_fields, 'row')}) "
        f"MERGE (t:{_quote(to_label)} {_key_clause('to_key', to_pk_fields, 'row')}) "
        f"MERGE (s)-[r:{_quote(rel_type)} {_key_clause('rel_key', rel_pk_fields, 'row')}]->(t)"
    )
    if has_value_fields:
        cypher += " SET r += $row.props"
    return cypher


def build_relationship_delete(rel_type: str, pk_fields: Sequence[str]) -> str:
    """``MATCH ()-[r:`RelType` {pk: $row.key_0, ...}]->() DELETE r``.

    Endpoints are intentionally not deleted — they're tracked by their own
    table handlers and will be deleted by their own reconciler if orphaned.
    """
    if not pk_fields:
        raise ValueError(
            "build_relationship_delete requires at least one primary key field"
        )
    return (
        f"MATCH ()-[r:{_quote(rel_type)} "
        f"{_key_clause('key', pk_fields)}]->() DELETE r"
    )


# ---------------------------------------------------------------------------
# AGE DDL — SQL statements for label/index management
# ---------------------------------------------------------------------------


def build_graph_create_sql(graph_name: str) -> str:
    """``SELECT create_graph('graph_name')``."""
    return f"SELECT create_graph({_quote(graph_name)}::name)"


def build_graph_exists_sql() -> str:
    """``SELECT count(*) > 0 FROM ag_graph WHERE name = $1``.

    Caller binds the graph name as the first parameter.
    """
    return "SELECT count(*) > 0 FROM ag_catalog.ag_graph WHERE name = $1"


def build_label_create_sql(graph_name: str, label: str, is_edge: bool) -> str:
    """``SELECT create_vlabel('graph_name', 'Label')`` or
    ``SELECT create_elabel('graph_name', 'Label')``."""
    fn = "create_elabel" if is_edge else "create_vlabel"
    return f"SELECT {fn}({_quote(graph_name)}::name, {_quote(label)}::name)"


def build_label_drop_sql(graph_name: str, label: str) -> str:
    """``SELECT drop_label('graph_name', 'Label', cascading)``."""
    return (
        f"SELECT drop_label({_quote(graph_name)}::name, "
        f"{_quote(label)}::name, true::boolean)"
    )


def build_node_index_create(graph_name: str, label: str, index_name: str, fields: Sequence[str]) -> str:
    """``CREATE UNIQUE INDEX index_name ON graph_name."Label" (...)``.

    Creates a unique index on AGE vertex properties using ``agtype_access_operator``.
    """
    if not fields:
        raise ValueError("build_node_index_create requires at least one field")
    field_exprs = [
        f"(agtype_access_operator(properties, {_quote(f)}::agtype))"
        for f in fields
    ]
    col_expr = ", ".join(field_exprs)
    return (
        f"CREATE UNIQUE INDEX {_quote(index_name)} "
        f"ON {_quote(graph_name)}.{_quote(label)} ({col_expr})"
    )


def build_node_index_drop(graph_name: str, index_name: str) -> str:
    """``DROP INDEX IF EXISTS index_name``."""
    return f"DROP INDEX IF EXISTS {_quote(graph_name)}.{_quote(index_name)}"


def vector_index_name(graph_name: str, label: str, field: str) -> str:
    """Deterministic vector index name."""
    return f"coco_vec_{graph_name}__{label}__{field}"


def build_vector_index_create(
    graph_name: str,
    label: str,
    index_name: str,
    field: str,
    vector_size: int,
    metric: str,
    graph_oid: int = 0,
) -> str:
    """``CREATE INDEX ... USING hnsw (agtype_access_operator(...)::vector(n))``.

    The complex ``agtype_access_operator`` expression is required to index
    AGE vertex properties with pgvector's HNSW index.

    ``metric`` is the pgvector operator class suffix (``vector_cosine_ops``,
    ``vector_l2_ops``, ``vector_ip_ops``).
    """
    if vector_size <= 0:
        raise ValueError(f"Invalid vector size: {vector_size}")
    return (
        f"CREATE INDEX {_quote(index_name)} "
        f"ON {_quote(graph_name)}.{_quote(label)} "
        f"USING hnsw (("
        f"agtype_access_operator("
        f"VARIADIC ARRAY["
        f"_agtype_build_vertex(id, _label_name({graph_oid}, id), properties), "
        f"{_quote(field)}::agtype"
        f"]"
        f")::text"
        f")::vector({vector_size})) {metric}"
    )


def build_vector_index_drop(graph_name: str, index_name: str) -> str:
    """``DROP INDEX IF EXISTS index_name``."""
    return f"DROP INDEX IF EXISTS {_quote(graph_name)}.{_quote(index_name)}"


# ---------------------------------------------------------------------------
# SQL wrapper helpers
# ---------------------------------------------------------------------------


def wrap_cypher_sql(graph_name: str, cypher: str, columns: str = "v agtype") -> str:
    """Wrap a Cypher fragment in ``SELECT * FROM cypher(...)``.

    ``columns`` is the ``AS (... )`` column definition for the result set.
    Default ``"v agtype"`` is sufficient for most DML.
    """
    # The cypher is embedded as a dollar-quoted string to avoid escaping issues.
    return (
        f"SELECT * FROM cypher({_quote(graph_name)}::name, "
        f"$$ {cypher} $$, $batch::agtype) "
        f"AS ({columns})"
    )
