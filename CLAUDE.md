# CLAUDE.md

Project context for AI assistants working on ZipLattice.

## What is this project?

ZipLattice is a portable, JSON-backed knowledge graph manager built on NetworkX. It is a single-file Python library (`knowledge_graph.py`) designed to ingest technical documentation, build structured knowledge graphs, and support graph-based Retrieval-Augmented Generation (RAG) for LLM agents.

## Project layout

```
knowledge_graph.py       # Knowledge graph library and CLI (single-file)
convert_to_markdown.py   # Standalone document-to-Markdown converter (single-file)
README.md                # Project documentation
LICENSE                  # MIT license
.gitignore               # Standard Python ignores
```

At runtime, the library creates:
- A graph JSON file (e.g. `knowledge_graph.json`) for nodes, edges, metadata, and proposals
- An embeddings JSON file (e.g. `knowledge_graph_embeddings.json`) for vector data
- A sources directory (e.g. `knowledge_graph_sources/`) for stored document text

## Language and dependencies

- Python 3.10+ (uses `from __future__ import annotations` and modern type hints)
- **Required:** `networkx`
- **Optional:** `pyvis` (for Pyvis visualization export)
- **Optional (for converter):** `pymupdf4llm`, `mammoth`, `markdownify`
- Cytoscape.js is loaded from CDN in exported HTML files and does not need a local install

## Key classes

- `KnowledgeGraph` (line ~211) -- Primary class. Manages nodes, edges, embeddings, relation proposals, persistence, search, and visualization.
- `CoreRelation` (line ~51) -- Enum of 25+ built-in relation types (taxonomic, dependency, associative, documentation, functional, contextual).
- `RelationProposal` (line ~166) -- Dataclass tracking proposed new relation types with examples and confidence scores.
- `ProposalStatus` (line ~159) -- Enum: PENDING, ACCEPTED, REJECTED.
- `GraphEncoder` (line ~121) -- Custom JSON encoder for datetime, set, Enum, and Path objects.

## How to run

```bash
# Install dependencies
pip install networkx
pip install pyvis  # optional, for Pyvis visualization
pip install pymupdf4llm mammoth markdownify  # optional, for document converter

# Knowledge graph CLI
python knowledge_graph.py <path-to-graph.json> --stats
python knowledge_graph.py <path-to-graph.json> --node <id>
python knowledge_graph.py <path-to-graph.json> --neighbors <id> --depth 2
python knowledge_graph.py <path-to-graph.json> --pyvis output.html
python knowledge_graph.py <path-to-graph.json> --cytoscape output.html
python knowledge_graph.py <path-to-graph.json> --ingest-md doc.md --sections

# Document converter CLI
python convert_to_markdown.py document.pdf -o document.md
python convert_to_markdown.py file1.pdf file2.docx -d output_dir/

# Library usage
python -c "from knowledge_graph import KnowledgeGraph; print('OK')"
python -c "from convert_to_markdown import convert; print('OK')"
```

## Testing

There is no formal test suite. To verify the module loads correctly:

```bash
python -c "from knowledge_graph import KnowledgeGraph"
```

## Architecture notes

- The graph uses a **dual representation**: raw Python dicts for serialization and a NetworkX `DiGraph` for graph algorithms. These are kept in sync by the class methods.
- Node and edge IDs are slugified (lowercase, alphanumeric, hyphens).
- All nodes and edges carry `confidence` scores in the range [0, 1].
- Timestamps use ISO 8601 format with UTC timezone.
- The library tracks a `_dirty` flag to avoid unnecessary writes on `save()`.
- Relation proposals allow the schema to evolve: novel relations discovered during LLM extraction are proposed, accumulated across documents, and accepted or rejected.
- Source documents are stored with SHA-256 content hashing for deduplication and version tracking.

## Common patterns when modifying this code

- All public graph mutation methods (`add_node`, `add_edge`, `remove_node`, etc.) must update both the internal dict representation and the NetworkX DiGraph, and set `self._dirty = True`.
- The `GraphEncoder` class must handle any new types added to node/edge properties.
- The `save()`/`load()` round-trip must be lossless. If you add new fields, update both `to_dict()` and the `load()` constructor logic.
- Embedding operations use a separate dirty flag (`_dirty_embeddings`) and separate persistence file.
- The `main()` function at the bottom provides the CLI and uses argparse.
