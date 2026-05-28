# Build Meeting Notes Knowledge Graph from Google Drive — Apache AGE (CocoIndex v1)

Extract structured information from meeting notes stored in Google Drive and
build a knowledge graph in [Apache AGE](https://age.apache.org/). The flow
ingests Markdown notes, splits them by headings into per-meeting sections,
uses an LLM (via [LiteLLM](https://docs.litellm.ai/) +
[instructor](https://python.useinstructor.com/)) to parse participants,
organizer, time, and tasks, and writes vertices and edges into the graph.

Please drop [CocoIndex on Github](https://github.com/cocoindex-io/cocoindex) a
star to support us and stay tuned for more updates. Thank you so much 🥥🤗.
[![GitHub](https://img.shields.io/github/stars/cocoindex-io/cocoindex?color=5B5BD6)](https://github.com/cocoindex-io/cocoindex)

## What this builds

- `Meeting` vertices — one per meeting section, keyed by a stable integer id
  derived from `(note_file, date)`
- `Person` vertices — canonical organizers, participants, and task assignees,
  deduplicated by an embedding + LLM entity-resolution pass (so "Alice",
  "Alice Chen", and "alice c." collapse to a single vertex)
- `Task` vertices — tasks decided in meetings (keyed by description)
- Edges:
  - `ATTENDED` — `Person → Meeting` (with `is_organizer` flag)
  - `DECIDED` — `Meeting → Task`
  - `ASSIGNED_TO` — `Person → Task`

The source is one or more Google Drive folders shared with a service account.
The flow watches for changes and keeps the graph up to date incrementally.

## How it works

The pipeline runs in three phases:

1. **Per-file extraction.** Read each file from Google Drive, split it by
   Markdown headings (`#` / `##`) into meeting sections, and for each section
   extract a structured `Meeting` via LiteLLM + instructor (date, note,
   organizer, participants, tasks with assignees). `Meeting` and `Task`
   vertices plus `DECIDED` edges are declared in this phase. Raw person names
   are carried forward.
2. **Person entity resolution.** All raw person names from all files are
   deduplicated using sentence-transformer embeddings and an LLM pair resolver
   to produce a canonical-name mapping.
3. **Person-touching relations.** Canonical `Person` vertices are declared,
   then `ATTENDED` and `ASSIGNED_TO` edges are wired up using resolved names.

CocoIndex reconciles changes incrementally — re-running after editing one note
only re-processes the affected sections, and the resolution phase only re-runs
when the set of raw names changes.

## Prerequisites

- A running PostgreSQL instance with the Apache AGE extension:
  ```sh
  docker run -d \
    -p 5455:5432 \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=postgres \
    --name cocoindex-age \
    apache/age:latest_PG17a
  ```
  After starting, load the AGE extension:
  ```sh
  docker exec -it cocoindex-age psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS ag_catalog"
  ```

- An LLM key (defaults to OpenAI; configure via `LLM_MODEL` for other
  providers — see [LiteLLM providers](https://docs.litellm.ai/docs/providers)).
- A Google Cloud service account with read access to the source folders, and
  the folder IDs you want to ingest. See
  [Setup for Google Drive](https://cocoindex.io/docs/sources/googledrive#setup-for-google-drive).

## Environment

Set the following variables (copy `.env.example` to `.env` and fill in):

```sh
export OPENAI_API_KEY=sk-...
export GOOGLE_SERVICE_ACCOUNT_CREDENTIAL=/absolute/path/to/service_account.json
export GOOGLE_DRIVE_ROOT_FOLDER_IDS=folderId1,folderId2
export AGE_HOST=localhost
export AGE_PORT=5455
export AGE_DBNAME=postgres
export AGE_USER=postgres
export AGE_PASSWORD=postgres
export AGE_GRAPH=meeting_notes
export LLM_MODEL=openai/gpt-5.4
export RESOLUTION_LLM_MODEL=openai/gpt-5-mini   # used for entity resolution
```

Then:

```sh
set -a && source .env && set +a
```

## Run

Install dependencies:

```sh
uv pip install -e .
```

Build/update the graph:

```sh
cocoindex update main
```

## Browse the knowledge graph

Connect to the PostgreSQL instance and use AGE Cypher queries:

```sh
docker exec -it cocoindex-age psql -U postgres
```

```sql
-- Load AGE
LOAD 'age';
SET search_path TO ag_catalog;

-- Set graph path
SELECT * FROM ag_catalog.create_graph('meeting_notes');
-- Use the graph (after the pipeline has run)
SELECT * FROM cypher('meeting_notes', $$
  MATCH p=()-->() RETURN p
$$) AS (v agtype);

-- Who attended which meetings (including organizer)
SELECT * FROM cypher('meeting_notes', $$
  MATCH (p:Person)-[:ATTENDED]->(m:Meeting)
  RETURN p.name, m.note_file, m.time, m.id
$$) AS (name agtype, note_file agtype, time agtype, id agtype);

-- Tasks decided in meetings
SELECT * FROM cypher('meeting_notes', $$
  MATCH (m:Meeting)-[:DECIDED]->(t:Task)
  RETURN m.note_file, m.time, t.description
$$) AS (note_file agtype, time agtype, description agtype);

-- Task assignments
SELECT * FROM cypher('meeting_notes', $$
  MATCH (p:Person)-[:ASSIGNED_TO]->(t:Task)
  RETURN p.name, t.description
$$) AS (name agtype, description agtype);

-- Meetings someone organized
SELECT * FROM cypher('meeting_notes', $$
  MATCH (p:Person)-[r:ATTENDED]->(m:Meeting)
  WHERE r.is_organizer = true
  RETURN p.name, m.note_file, m.time
$$) AS (name agtype, note_file agtype, time agtype);
```

To wipe the graph between runs:

```sql
SELECT * FROM ag_catalog.drop_graph('meeting_notes', true);
```
