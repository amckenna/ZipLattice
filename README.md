# ZipLattice

A portable, JSON-backed knowledge graph manager built on [NetworkX](https://networkx.org/). ZipLattice is designed for ingesting technical documentation, building structured knowledge graphs, and supporting graph-based Retrieval-Augmented Generation (RAG) for LLM agents.

## Features

- **Flat-file persistence** -- Graphs are stored as plain JSON with no database required. Embeddings are stored in a companion JSON file.
- **Document ingestion** -- Parse Markdown files into structured sections and extract RDF-style triples via LLM integration.
- **Semantic search** -- Hybrid search combining vector embeddings with multi-hop graph expansion.
- **Relation proposals** -- Novel relation types discovered during extraction are tracked as proposals that can be reviewed, accepted, or rejected over time.
- **Source management** -- Ingested documents are stored with content-hash deduplication and version history.
- **Visualization** -- Export interactive HTML visualizations using Pyvis (quick overview) or Cytoscape.js (full-featured with filtering, search, and layout controls).
- **Graph analysis** -- Centrality metrics, connected components, shortest paths, and relation pattern analysis powered by NetworkX.

## Requirements

- Python 3.10 or later
- [networkx](https://pypi.org/project/networkx/)
- [pyvis](https://pypi.org/project/pyvis/) (optional, for Pyvis visualization)

## Installation

```bash
pip install networkx
pip install pyvis  # optional
```

There is no package to install for ZipLattice itself. Import the module directly or run the CLI from the project directory.

## Quick start

### As a library

```python
from knowledge_graph import KnowledgeGraph

# Create or load a graph
kg = KnowledgeGraph("my_graph.json")

# Add nodes
kg.add_node("python", type="technology", label="Python",
            properties={"domain": "programming"})
kg.add_node("asyncio", type="concept", label="asyncio",
            properties={"domain": "concurrency"})

# Add an edge
kg.add_edge("asyncio", "python", relation="part_of")

# Query
neighbors = kg.get_neighbors("python")
subgraph = kg.get_subgraph("python", depth=2)

# Save
kg.save()
```

### From the command line

```bash
# Show graph statistics
python knowledge_graph.py my_graph.json --stats

# Look up a node
python knowledge_graph.py my_graph.json --node python

# Explore neighbors (with depth)
python knowledge_graph.py my_graph.json --neighbors python --depth 2

# Export a Pyvis visualization
python knowledge_graph.py my_graph.json --pyvis output.html

# Export a Cytoscape.js visualization
python knowledge_graph.py my_graph.json --cytoscape output.html

# Center visualization on a specific node
python knowledge_graph.py my_graph.json --cytoscape output.html --center python --depth 3

# Parse a Markdown file and show its section breakdown
python knowledge_graph.py my_graph.json --ingest-md doc.md --sections

# List stored source documents
python knowledge_graph.py my_graph.json --sources

# Verify integrity of stored sources
python knowledge_graph.py my_graph.json --check-sources

# View and manage relation proposals
python knowledge_graph.py my_graph.json --proposals
python knowledge_graph.py my_graph.json --accept <relation_name>
python knowledge_graph.py my_graph.json --reject <relation_name>

# Analyze novel relation patterns
python knowledge_graph.py my_graph.json --patterns

# Export nodes and edges as separate files
python knowledge_graph.py my_graph.json --split output_dir/
```

## Architecture

ZipLattice is implemented as a single Python file (`knowledge_graph.py`). The main class, `KnowledgeGraph`, manages all operations.

### Data model

**Nodes** represent entities such as concepts, technologies, documents, people, and organizations. Each node has an ID (slugified string), a type, a label, arbitrary properties, a confidence score, and timestamps.

**Edges** represent directed relationships between nodes. Each edge has a source, target, relation type, optional properties, a confidence score, a weight, and timestamps.

**Relations** come from a built-in set of 25+ core types organized into categories:

| Category | Examples |
|---|---|
| Taxonomic | `is_a`, `part_of`, `has_part`, `subclass_of`, `instance_of` |
| Dependency | `depends_on`, `required_by`, `causes`, `caused_by` |
| Associative | `related_to`, `similar_to`, `references`, `implements`, `extends` |
| Documentation | `documents`, `documented_by`, `derived_from`, `supersedes` |
| Functional | `uses`, `used_by`, `configured_by`, `produces`, `consumes` |
| Contextual | `belongs_to`, `contains`, `tagged_with` |

Custom relations can be registered at runtime with `kg.register_relation()`.

### Internal representation

The graph maintains two synchronized representations:

1. **Python dicts** -- The canonical store for nodes and edges, used for JSON serialization.
2. **NetworkX DiGraph** -- Built from the dicts on load, used for graph algorithms (traversal, centrality, shortest paths, connected components).

All mutations go through class methods that update both representations and mark the graph as dirty for the next save.

### File layout

| File | Purpose |
|---|---|
| `<name>.json` | Nodes, edges, metadata, and relation proposals |
| `<name>_embeddings.json` | Node ID to vector mappings |
| `<name>_sources/` | Stored source documents with content-hash deduplication |

### Document ingestion pipeline

1. A Markdown file is parsed into sections using heading structure (ATX and Setext formats). Content annotations (code blocks, tables, lists, links) are detected per section.
2. Each section is passed to an LLM extraction function (user-provided callable) that returns RDF-style triples: `{source, target, relation, confidence, ...}`.
3. Triples are added as nodes and edges. If the LLM suggests a relation not in the schema, it is recorded as a `RelationProposal` for later review.
4. The original document text is stored with SHA-256 content hashing for deduplication and versioning.

### Embeddings and search

Embeddings are stored separately from the main graph and managed with their own dirty flag and save method. The `search()` method combines cosine similarity over embeddings with multi-hop graph expansion to provide context-rich results suitable for RAG.

## Node types

The following node types are available by default (advisory, not enforced):

`concept`, `entity`, `document`, `section`, `technology`, `tool`, `process`, `event`, `person`, `organization`, `code`, `configuration`, `artifact`, `custom`

## License

MIT -- see [LICENSE](LICENSE) for details.
