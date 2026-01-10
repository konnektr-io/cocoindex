"""
This example shows how to use the Apache AGE connector.
It reads lines from a text file and ingests them into an Apache AGE graph.
"""

import cocoindex
import os
from cocoindex.sources import LocalFile
from cocoindex.op import Nodes, TargetFieldMapping

# Define connection
# Ensure you have a running PostgreSQL instance with AGE extension.
age_conn_spec = cocoindex.add_auth_entry(
    "AgeConnection",
    cocoindex.targets.AgeConnection(
        host="localhost",
        port=5432,
        user="postgres",
        password=os.environ.get("PG_PASSWORD", "postgres"),
        database="postgres",
        graph_name="age_demo_graph"
    ),
)

flow = cocoindex.Flow(name="age_demo")

# Simple flow: Read lines -> Store as Nodes
flow.source(LocalFile(path="data.txt")) \
    .fn(cocoindex.functions.SplitBySeparators(separators=["\n"])) \
    .target(cocoindex.targets.Age(
        connection=age_conn_spec,
        mapping=Nodes(
            label="Line",
            fields=[
                TargetFieldMapping(source="text", target="content"),
            ]
        )
    ))

# Declaration (Optional but recommended for indexes)
flow.declare(cocoindex.targets.AgeDeclaration(
    connection=age_conn_spec,
    nodes_label="Line",
    primary_key_fields=["content"],
))

if __name__ == "__main__":
    cocoindex.run(flow)
