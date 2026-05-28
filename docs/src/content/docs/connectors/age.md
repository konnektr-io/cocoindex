---
title: "*Apache AGE* connector"
toc_max_heading_level: 4
description: >
  Write to Apache AGE — a graph extension for PostgreSQL — with support for
  vertex tables, edge tables (relationships), per-graph multitenancy,
  atomic-batch writes via asyncpg transactions, real label uniqueness
  constraints, and vector indexes via pgvector.
---

The `age` connector writes records to [Apache AGE](https://age.apache.org/), a property graph extension for PostgreSQL. It supports vertex tables (labels), edge tables (relationship types), per-graph multitenancy (one AGE instance, many isolated graphs), real constraint-backed uniqueness, and vector indexes via [pgvector](https://github.com/pgvector/pgvector).

```python
from cocoindex.connectors import age
```

:::note[Dependencies]
This connector requires additional dependencies. Install with:

```bash
pip install cocoindex[age]
```

Requires a PostgreSQL server with the Apache AGE extension loaded (`CREATE EXTENSION IF NOT EXISTS ag_catalog`). AGE's `cypher()` function is called via SQL, so no AGE-specific Python driver is needed — only `asyncpg`.
:::

## Connection setup

Create a `ConnectionFactory` and provide it via a `ContextKey`. The factory holds the PostgreSQL connection parameters and the target graph name; it lazily opens an `asyncpg` pool and returns a graph handle on demand.

:::note
The key name is load-bearing across runs — it's the stable identity CocoIndex uses to track managed rows. See [ContextKey as stable identity](../programming_guide/context#contextkey-as-stable-identity) before renaming.
:::

```python
from collections.abc import AsyncIterator
from cocoindex.connectors import age
import cocoindex as coco

KG_DB: coco.ContextKey[age.ConnectionFactory] = coco.ContextKey("kg_db")

@coco.lifespan
async def coco_lifespan(builder: coco.EnvironmentBuilder) -> AsyncIterator[None]:
    builder.provide(
        KG_DB,
        age.ConnectionFactory(
            host="localhost",
            port=5455,
            dbname="postgres",
            user="postgres",
            password="postgres",
            graph="my_graph",
        ),
    )
    yield
```

The graph must be created in advance:

```sql
SELECT * FROM ag_catalog.create_graph('my_graph');
```

### Multitenancy

A single AGE-enabled PostgreSQL instance can host many isolated graphs. Pair each graph with its own `ContextKey` and `ConnectionFactory(graph=...)`:

```python
KG_DB: coco.ContextKey[age.ConnectionFactory] = coco.ContextKey("kg_db")
APIS_DB: coco.ContextKey[age.ConnectionFactory] = coco.ContextKey("apis_db")

@coco.lifespan
async def coco_lifespan(builder: coco.EnvironmentBuilder) -> AsyncIterator[None]:
    conn = dict(host="localhost", port=5455, dbname="postgres", user="postgres", password="postgres")
    builder.provide(KG_DB, age.ConnectionFactory(**conn, graph="kg"))
    builder.provide(APIS_DB, age.ConnectionFactory(**conn, graph="apis"))
    yield
```

Different `ContextKey`s with different graph names produce fully separate target-state trees — changes to one never spill into the other.

## As target

The `age` connector provides target state APIs for writing records to vertex tables and edge tables. CocoIndex tracks what records should exist and automatically handles upserts and deletions.

Each apply batch is wrapped in a single PostgreSQL transaction (`COMMIT` on success, `ROLLBACK` on exception), so partial writes never leak into the database. Within a batch, writes are ordered as **vertex upserts → edge upserts → edge deletes → vertex deletes** so dependent edges always see their endpoints.

### Declaring target states

#### Vertex tables (parent state)

Declares a vertex label as a target state. Returns a `TableTarget` for declaring records.

```python
def declare_table_target(
    db: ContextKey,
    table_name: str,
    table_schema: TableSchema[RowT] | None = None,
    *,
    primary_key: str = "id",
    managed_by: Literal["system", "user"] = "system",
) -> TableTarget[RowT, coco.PendingS]
```

**Parameters:**

- `db` — A `ContextKey[age.ConnectionFactory]` for the AGE connection.
- `table_name` — The vertex label (e.g. `"Document"`). Must match `[a-zA-Z_][a-zA-Z0-9_]*`.
- `table_schema` — Optional schema definition (see [Table Schema](#table-schema-from-python-class)). The schema participates in CocoIndex's fingerprint; per-property type DDL is not emitted in v1.
- `primary_key` — Single property name used as the vertex's primary key. Defaults to `"id"`. Compound primary keys are not supported in v1.0.
- `managed_by` — Whether CocoIndex manages the table lifecycle (`"system"`) or assumes it exists (`"user"`).

**Returns:** A pending `TableTarget`. Use `await age.mount_table_target(KG_DB, ...)` to get a resolved target.

#### Records (child states)

Once a `TableTarget` is resolved, declare records to be upserted:

```python
def TableTarget.declare_record(
    self,
    *,
    row: RowT,
) -> None
```

**Parameters:**

- `row` — A row object (dict, dataclass, NamedTuple, or Pydantic model). Must include the `primary_key` field declared above.

`declare_row` is an alias for `declare_record`, for compatibility with Postgres and other RDBMS targets.

#### Edge tables (parent state)

Declares a relationship type as a target state. Returns a `RelationTarget` for declaring edges.

```python
def declare_relation_target(
    db: ContextKey,
    table_name: str,
    from_table: TableTarget,
    to_table: TableTarget,
    table_schema: TableSchema[RowT] | None = None,
    *,
    primary_key: str = "id",
    managed_by: Literal["system", "user"] = "system",
) -> RelationTarget[RowT, coco.PendingS]
```

**Parameters:**

- `db` — A `ContextKey[age.ConnectionFactory]` for the AGE connection.
- `table_name` — The relationship type (e.g. `"MENTION"`).
- `from_table` — The `TableTarget` whose vertices are the *source* endpoints of edges in this relationship.
- `to_table` — The `TableTarget` whose vertices are the *target* endpoints of edges in this relationship.
- `table_schema` — Optional schema for the relationship's own properties. The relationship's `primary_key` field uniquely identifies each edge.
- `primary_key` — Single property name used as the edge's primary key. Defaults to `"id"`.
- `managed_by` — Whether CocoIndex manages the relationship lifecycle (`"system"`) or assumes it exists (`"user"`).

**Returns:** A pending `RelationTarget`. Use `await age.mount_relation_target(KG_DB, ...)` to get a resolved target.

#### Relations (child states)

Once a `RelationTarget` is resolved, declare edges:

```python
def RelationTarget.declare_relation(
    self,
    *,
    from_id: Any,
    to_id: Any,
    record: RowT | None = None,
) -> None
```

**Parameters:**

- `from_id` — The source vertex's primary-key value.
- `to_id` — The target vertex's primary-key value.
- `record` — Optional row object whose fields populate the edge's properties. Must include the relationship's `primary_key` field if provided.

If `record` is omitted, the connector derives a deterministic edge id.

#### Vector indexes (attachment)

Declares a pgvector index on a property of a vertex label. Vector indexes are an [attachment](../advanced_topics/custom_target_connector#implementing-attachment-providers) to a `TableTarget`:

```python
def TableTarget.declare_vector_index(
    self,
    *,
    name: str | None = None,
    field: str,
    metric: Literal["cosine", "euclidean", "inner_product"] = "cosine",
    dimension: int,
) -> None
```

**Parameters:**

- `name` — Optional logical name for the index. Defaults to `f"coco_vec_{table_name}__{field}"`.
- `field` — The vertex property holding the vector.
- `metric` — Distance metric: `"cosine"`, `"euclidean"`, or `"inner_product"`.
- `dimension` — The vector's dimension. Required.

The connector creates a PostgreSQL index using AGE's `agtype_access_operator` to extract the vector field:

```sql
CREATE INDEX IF NOT EXISTS coco_vec_Document__embedding
ON "Document" USING ivfflat (
    (agtype_access_operator(VARIADIC ARRAY[_agtype_build_vertex(Document.*, '"embedding"'::agtype)])::text::vector(384))
) WITH (lists = 100);
```

### Table schema: from Python class

Build a `TableSchema` by introspecting a record type:

```python
@classmethod
async def TableSchema.from_class(
    cls,
    record_type: type[RowT],
    *,
    primary_key: str = "id",
    column_overrides: dict[str, AgeType | VectorSchemaProvider] | None = None,
) -> TableSchema[RowT]
```

**Parameters:**

- `record_type` — A dataclass, NamedTuple, or Pydantic model.
- `primary_key` — Field name to use as the table's primary key. Defaults to `"id"`.
- `column_overrides` — Optional dict mapping field names to `AgeType` or `VectorSchemaProvider` to override the default Python-to-AGE type mapping.

**Returns:** A `TableSchema[RowT]` populated from the class's fields.

### Table schema: explicit column definitions

Build a `TableSchema` directly from a dict of column definitions when the row type is dynamic:

```python
from cocoindex.connectors.age import TableSchema, ColumnDef

schema = TableSchema(
    columns={
        "filename": ColumnDef(type="STRING"),
        "title": ColumnDef(type="STRING"),
        "summary": ColumnDef(type="STRING", nullable=True),
    },
    primary_key="filename",
)
```

`ColumnDef` fields:

- `type` — The AGE type string (metadata only).
- `nullable` — Whether the column may be `None`. Defaults to `True`.
- `encoder` — Optional `Callable[[Any], Any]` applied to non-`None` values before they're sent to AGE.

### DDL: labels and indexes

For each managed vertex table, the connector creates a label definition with a uniqueness constraint on the primary key:

```sql
SELECT * FROM ag_catalog.create_vlabel('my_graph', 'Document');
```

For each managed edge table, the connector creates the edge label with an index on its primary key:

```sql
SELECT * FROM ag_catalog.create_elabel('my_graph', 'MENTION');
```

Additional property indexes are created as regular PostgreSQL indexes:

```sql
CREATE INDEX IF NOT EXISTS coco_idx_Document__filename ON "Document" ((agtype_access_operator(VARIADIC ARRAY[_agtype_build_vertex(Document.*, '"filename"'::agtype)])::text));
```

Indexes and labels are cleaned up on `cocoindex drop` or when the table is no longer declared.

When `managed_by="user"` is set, the connector skips DDL entirely — you're responsible for creating and dropping the schema. Record-level upserts and deletes still work.

### Example: Vertex tables

```python
from collections.abc import AsyncIterator
from dataclasses import dataclass
import cocoindex as coco
from cocoindex.connectors import age

KG_DB: coco.ContextKey[age.ConnectionFactory] = coco.ContextKey("kg_db")


@dataclass
class Document:
    filename: str
    title: str
    summary: str


@coco.lifespan
async def coco_lifespan(builder: coco.EnvironmentBuilder) -> AsyncIterator[None]:
    builder.provide(KG_DB, age.ConnectionFactory(
        host="localhost",
        port=5455,
        dbname="postgres",
        user="postgres",
        password="postgres",
        graph="my_graph",
    ))
    yield


@coco.fn
async def app_main() -> None:
    schema = await age.TableSchema.from_class(Document, primary_key="filename")
    documents = await age.mount_table_target(
        KG_DB, "Document", schema, primary_key="filename",
    )
    documents.declare_record(
        row=Document(
            filename="overview.md",
            title="Overview",
            summary="An overview of CocoIndex...",
        )
    )


app = coco.App(coco.AppConfig(name="docs_to_age"), app_main)
```

### Example: Edge tables (knowledge graph)

```python
@dataclass
class Entity:
    value: str


@dataclass
class RelationshipRow:
    id: str
    predicate: str


@coco.fn
async def kg_app_main() -> None:
    documents = await age.mount_table_target(
        KG_DB, "Document",
        await age.TableSchema.from_class(Document, primary_key="filename"),
        primary_key="filename",
    )
    entities = await age.mount_table_target(
        KG_DB, "Entity",
        await age.TableSchema.from_class(Entity, primary_key="value"),
        primary_key="value",
    )
    relationships = await age.mount_relation_target(
        KG_DB, "RELATIONSHIP",
        entities, entities,
        await age.TableSchema.from_class(RelationshipRow, primary_key="id"),
        primary_key="id",
    )

    documents.declare_record(row=Document(filename="overview.md", title="Overview", summary="..."))
    entities.declare_record(row=Entity(value="CocoIndex"))
    entities.declare_record(row=Entity(value="AGE"))
    relationships.declare_relation(
        from_id="CocoIndex",
        to_id="AGE",
        record=RelationshipRow(id="rel-1", predicate="writes_to"),
    )


kg_app = coco.App(coco.AppConfig(name="kg_app"), kg_app_main)
```

The `Entity` table is declared up-front so its label constraint is reconciled before any `RELATIONSHIP` edge upserts entity endpoints.
