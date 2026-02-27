# CLAUDE.md

Project context for AI assistants working on ZipLattice.

## What is this project?

ZipLattice is a portable, JSON-backed knowledge graph manager built on NetworkX. It is a single-file Python library (`knowledge_graph.py`) designed to ingest technical documentation, build structured knowledge graphs, and support graph-based Retrieval-Augmented Generation (RAG) for LLM agents.

## Project layout

```
knowledge_graph.py       # Knowledge graph library and CLI (single-file)
query_graph.py           # Knowledge graph query application (search, RAG, CLI)
web_app.py               # FastAPI web frontend (HTMX + Tailwind CSS)
templates/               # Jinja2 HTML templates for the web frontend
  base.html              #   Layout shell (nav, Tailwind/HTMX CDN)
  dashboard.html         #   Dashboard — lists all knowledge graphs
  graph_detail.html      #   Graph detail — Cytoscape.js visualization
  upload.html            #   File upload form
  query.html             #   Query form (search, context, ask)
  partials/              #   HTMX partial response fragments
test_query_graph.py      # Tests for query_graph.py
test_web_app.py          # Tests for web_app.py
benchmark_models.py      # Model comparison tool for extraction quality
convert_to_markdown.py   # Standalone document-to-Markdown converter (single-file)
README.md                # Project documentation
LICENSE                  # MIT license
.gitignore               # Standard Python ignores
```

At runtime, the library creates a dedicated directory named after the graph stem
containing all artifacts. For example, passing `knowledge_graph.json` produces:

```
knowledge_graph/                  # dedicated graph directory
  knowledge_graph.json            # nodes, edges, metadata, proposals
  knowledge_graph_embeddings.json # vector data
  knowledge_graph_sources/        # stored document text
  knowledge_graph_cytoscape.html  # Cytoscape visualization (auto-exported)
  knowledge_graph_pyvis.html      # Pyvis visualization (auto-exported)
```

## Language and dependencies

- Python 3.10+ (uses `from __future__ import annotations` and modern type hints)
- **Required:** `networkx`
- **Optional:** `pyvis` (for Pyvis visualization export)
- **Optional (for converter):** `pymupdf4llm`, `mammoth`, `markdownify`, `pymupdf_layout` (improves PDF layout analysis)
- **Optional (for web frontend):** `fastapi`, `uvicorn`, `python-multipart`, `jinja2`
- Cytoscape.js is loaded from CDN in exported HTML files and does not need a local install

## Key classes

- `KnowledgeGraph` (line ~211) -- Primary class. Manages nodes, edges, embeddings, relation proposals, persistence, search, and visualization.
- `CoreRelation` (line ~51) -- Enum of 25+ built-in relation types (taxonomic, dependency, associative, documentation, functional, contextual).
- `RelationProposal` (line ~166) -- Dataclass tracking proposed new relation types with examples and confidence scores.
- `ProposalStatus` (line ~159) -- Enum: PENDING, ACCEPTED, REJECTED.
- `GraphEncoder` (line ~121) -- Custom JSON encoder for datetime, set, Enum, and Path objects.
- `ollama_embed()` (module-level) -- Calls OpenAI-compatible `/v1/embeddings` endpoint. Works with Ollama, llama.cpp, vLLM, LocalAI, etc. Used during ingestion and by `query_graph.py` at query time.

## How to run

```bash
# Install dependencies
pip install networkx
pip install pyvis  # optional, for Pyvis visualization
pip install pymupdf4llm mammoth markdownify  # optional, for document converter
pip install pymupdf_layout  # optional, improves PDF page layout analysis

# Knowledge graph CLI
python knowledge_graph.py <path-to-graph.json> --stats
python knowledge_graph.py <path-to-graph.json> --node <id>
python knowledge_graph.py <path-to-graph.json> --neighbors <id> --depth 2
python knowledge_graph.py <path-to-graph.json> --pyvis output.html
python knowledge_graph.py <path-to-graph.json> --cytoscape output.html
python knowledge_graph.py <path-to-graph.json> --preview-md doc.md --sections
python knowledge_graph.py <path-to-graph.json> --ingest-md doc.md
python knowledge_graph.py <path-to-graph.json> --ingest-md docs/*.md --query-model qwen3-coder:30b --embed-model nomic-embed-text
python knowledge_graph.py <path-to-graph.json> --ingest-md doc.md --query-model qwen3-coder:30b --api-url http://exo:11434

# Query graph CLI
python query_graph.py <path-to-graph.json> search "synthetic aperture radar"
python query_graph.py <path-to-graph.json> context "how does SAR work?"
python query_graph.py <path-to-graph.json> ask "how does SAR work?" --query-model qwen3-coder:30b
python query_graph.py <path-to-graph.json> ask "how does SAR work?" --query-model qwen3-coder:30b --api-url http://exo:11434
python query_graph.py <path-to-graph.json> node <node-id>
python query_graph.py <path-to-graph.json> neighbors <node-id> --depth 2
python query_graph.py <path-to-graph.json> path <source-id> <target-id>
python query_graph.py <path-to-graph.json> stats

# Model benchmark CLI
python benchmark_models.py doc.md --models qwen3-coder:30b gemma3:27b llama3.1:70b
python benchmark_models.py docs/*.md --models modelA modelB --api-url http://exo:11434
python benchmark_models.py doc.md --models modelA modelB --json
python benchmark_models.py doc.md --models modelA modelB --max-sections 5  # quick test

# Document converter CLI
python convert_to_markdown.py document.pdf -o document.md
python convert_to_markdown.py file1.pdf file2.docx -d output_dir/

# Web frontend
pip install fastapi uvicorn python-multipart jinja2
uvicorn web_app:app --reload                    # http://localhost:8000
python web_app.py                               # same, starts on port 8000
ZIPLATTICE_GRAPHS_DIR=./my_graphs uvicorn web_app:app  # custom graphs directory

# Library usage
python -c "from knowledge_graph import KnowledgeGraph; print('OK')"
python -c "from convert_to_markdown import convert; print('OK')"
```

## Testing

```bash
# Run query_graph tests (no Ollama needed)
python -m pytest test_query_graph.py -v

# Run web app tests (no Ollama needed)
python -m pytest test_web_app.py -v

# Verify modules load
python -c "from knowledge_graph import KnowledgeGraph"
python -c "from query_graph import search_nodes, build_context, ask"
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
