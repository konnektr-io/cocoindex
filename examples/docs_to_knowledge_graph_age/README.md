# Build Real-Time Knowledge Graph For Documents with LLM (Apache AGE)

We will process a list of documents and use LLM to extract relationships between the concepts in each document.
We will generate two kinds of relationships:

1. Relationships between subjects and objects. E.g., "CocoIndex supports Incremental Processing"
2. Mentions of entities in a document. E.g., "core/basics.mdx" mentions `CocoIndex` and `Incremental Processing`.

This example demonstrates using **Apache AGE** (Graph extension for PostgreSQL) as the graph backend.

## Prerequisite

* [Install Postgres](https://cocoindex.io/docs/getting_started/installation#-install-postgres) and ensure the **AGE extension** is installed and enabled.
  * You usually need to run `CREATE EXTENSION age;` in your database.
  * You also need the `vector` extension: `CREATE EXTENSION vector;`.
* Install / configure LLM API. In this example we use Ollama, which runs LLM model locally. You need to get it ready following [this guide](https://cocoindex.io/docs/ai/llm#ollama).

## Run

### Build the index

Install dependencies:

```sh
pip install -e .
```

Update index:

```sh
cocoindex update main
```

### Browse the knowledge graph

After the knowledge graph is built, you can explore it using SQL queries with Cypher in your Postgres database.

Connect to your Postgres database (e.g., using `psql` or pgAdmin):

```sql
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- Get all relationships
SELECT * FROM cypher('coco_docs_graph', $$
    MATCH p=()-->() RETURN p
$$) as (p agtype);
```

## CocoInsight

I used CocoInsight (Free beta now) to troubleshoot the index generation and understand the data lineage of the pipeline.
It just connects to your local CocoIndex server, with Zero pipeline data retention. Run following command to start CocoInsight:

```sh
cocoindex server -ci main
```

And then open the url <https://cocoindex.io/cocoinsight>.
