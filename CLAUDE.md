# CLAUDE.md

Project context for AI assistants working on ZipLattice.

## What is this project?

ZipLattice is a portable, JSON-backed knowledge graph manager built on NetworkX. It is a single-file Python library (`knowledge_graph.py`) designed to ingest technical documentation, build structured knowledge graphs, and support graph-based Retrieval-Augmented Generation (RAG) for LLM agents.

## Project layout

```
knowledge_graph.py       # Knowledge graph library and CLI (single-file)
query_graph.py           # Knowledge graph query application (search, RAG, CLI)
web_app.py               # FastAPI web frontend (HTMX + Tailwind CSS)
mcp_server.py            # MCP server (FastMCP) — orchestrator-as-extractor pattern
templates/               # Jinja2 HTML templates for the web frontend
  base.html              #   Layout shell (nav, Tailwind/HTMX CDN)
  dashboard.html         #   Dashboard — lists all knowledge graphs
  graph_detail.html      #   Graph detail — Cytoscape.js visualization
  upload.html            #   File upload form
  query.html             #   Query form (search, context, ask)
  partials/              #   HTMX partial response fragments
test_knowledge_graph.py  # Tests for knowledge_graph.py
test_query_graph.py      # Tests for query_graph.py
test_web_app.py          # Tests for web_app.py
test_mcp_server.py       # Tests for mcp_server.py and ingest_triples
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
- **Optional (for MCP server):** `fastmcp`
- **Optional (for Bedrock provider):** `boto3`
- Cytoscape.js is loaded from CDN in exported HTML files and does not need a local install

## Key classes

- `KnowledgeGraph` (line ~611) -- Primary class. Manages nodes, edges, embeddings, relation proposals, persistence, search, and visualization.
- `CoreRelation` (line ~401) -- Enum of 25+ built-in relation types (taxonomic, dependency, associative, documentation, functional, contextual).
- `RelationProposal` (line ~566) -- Dataclass tracking proposed new relation types with examples and confidence scores.
- `ProposalStatus` (line ~559) -- Enum: PENDING, ACCEPTED, REJECTED.
- `GraphEncoder` (line ~521) -- Custom JSON encoder for datetime, set, Enum, and Path objects.
- `ollama_embed()` (module-level) -- Calls OpenAI-compatible `/v1/embeddings` endpoint. Works with Ollama, llama.cpp, vLLM, LocalAI, etc. Used during ingestion and by `query_graph.py` at query time.
- `claude_chat()` (module-level) -- Calls the Anthropic Messages API for chat/query. Used when `--provider anthropic` is set.
- `claude_extract()` (module-level) -- Calls the Anthropic Messages API for JSON entity/relation extraction during ingestion. Same three-tier JSON recovery as the local path.
- `bedrock_chat()` (module-level) -- Calls the AWS Bedrock Converse API for chat/query. Used when `--provider bedrock` is set.
- `bedrock_extract()` (module-level) -- Calls the AWS Bedrock Converse API for JSON entity/relation extraction. Same three-tier JSON recovery as the local path.
- `bedrock_embed()` (module-level) -- Calls AWS Bedrock for embeddings. Supports Titan (`invoke_model`) and Cohere (batched) embedding models.
- `KnowledgeGraph.ingest_triples()` -- Public method that accepts pre-extracted triples directly (no LLM call). Enables the orchestrator-as-extractor pattern used by the MCP server.

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
python knowledge_graph.py <path-to-graph.json> --ingest-md docs/*.md --query-model qwen3-coder:30b --embed-model qwen3-embedding
python knowledge_graph.py <path-to-graph.json> --ingest-md doc.md --query-model qwen3-coder:30b --api-url http://exo:11434
# Parallel ingestion: 4 concurrent LLM extraction threads, serial graph writes
python knowledge_graph.py <path-to-graph.json> --ingest-md docs/*.md --query-model qwen3-coder:30b -j 4

# Ingest with Claude API (Haiku for fast extraction, local embeddings)
python knowledge_graph.py <path-to-graph.json> --ingest-md doc.md --provider anthropic --extract-model claude-haiku-4-5 --embed-model qwen3-embedding

# Ingest with AWS Bedrock (Claude on Bedrock for extraction, local embeddings)
python knowledge_graph.py <path-to-graph.json> --ingest-md doc.md --provider bedrock --extract-model us.anthropic.claude-sonnet-4-20250514-v1:0 --embed-model qwen3-embedding
# Ingest with Bedrock for both extraction and embeddings (Titan embeddings)
python knowledge_graph.py <path-to-graph.json> --ingest-md doc.md --provider bedrock --extract-model us.anthropic.claude-sonnet-4-20250514-v1:0 --embed-model amazon.titan-embed-text-v2:0
# Ingest with Bedrock in a specific region
python knowledge_graph.py <path-to-graph.json> --ingest-md doc.md --provider bedrock --extract-model us.anthropic.claude-sonnet-4-20250514-v1:0 --embed-model qwen3-embedding --bedrock-region us-west-2

# Query graph CLI
python query_graph.py <path-to-graph.json> search "synthetic aperture radar"
python query_graph.py <path-to-graph.json> context "how does SAR work?"
python query_graph.py <path-to-graph.json> ask "how does SAR work?" --query-model qwen3-coder:30b
python query_graph.py <path-to-graph.json> ask "how does SAR work?" --query-model qwen3-coder:30b --api-url http://exo:11434

# Query with Claude API (Opus for smartest answers)
python query_graph.py <path-to-graph.json> ask "how does SAR work?" --provider anthropic --query-model claude-opus-4-6
# Query with AWS Bedrock
python query_graph.py <path-to-graph.json> ask "how does SAR work?" --provider bedrock --query-model us.anthropic.claude-sonnet-4-20250514-v1:0
python query_graph.py <path-to-graph.json> ask "how does SAR work?" --provider bedrock --query-model amazon.nova-pro-v1:0 --bedrock-region us-east-1
python query_graph.py <path-to-graph.json> node <node-id>
python query_graph.py <path-to-graph.json> neighbors <node-id> --depth 2
python query_graph.py <path-to-graph.json> path <source-id> <target-id>
python query_graph.py <path-to-graph.json> stats

# Model benchmark CLI
python benchmark_models.py doc.md --models qwen3-coder:30b gemma3:27b llama3.1:70b
python benchmark_models.py docs/*.md --models modelA modelB --api-url http://exo:11434
python benchmark_models.py doc.md --models modelA modelB --json
python benchmark_models.py doc.md --models modelA modelB --max-sections 5  # quick test
python benchmark_models.py doc.md --models claude-haiku-4-5 claude-sonnet-4-6 --provider anthropic

# Document converter CLI
python convert_to_markdown.py document.pdf -o document.md
python convert_to_markdown.py file1.pdf file2.docx -d output_dir/

# Web frontend
pip install fastapi uvicorn python-multipart jinja2
uvicorn web_app:app --reload                    # http://localhost:8000
python web_app.py                               # same, starts on port 8000
ZIPLATTICE_GRAPHS_DIR=./my_graphs uvicorn web_app:app  # custom graphs directory

# MCP server (orchestrator-as-extractor pattern)
pip install fastmcp
python mcp_server.py                            # stdio transport (for Claude Code)

# Library usage
python -c "from knowledge_graph import KnowledgeGraph; print('OK')"
python -c "from convert_to_markdown import convert; print('OK')"
```

## Testing

```bash
# Run all tests (no Ollama needed)
python -m pytest test_knowledge_graph.py test_query_graph.py test_web_app.py test_mcp_server.py -v

# Run knowledge_graph tests only
python -m pytest test_knowledge_graph.py -v

# Run query_graph tests only
python -m pytest test_query_graph.py -v

# Run web app tests only
python -m pytest test_web_app.py -v

# Run MCP server tests only
python -m pytest test_mcp_server.py -v

# Verify modules load
python -c "from knowledge_graph import KnowledgeGraph"
python -c "from query_graph import search_nodes, build_context, ask"
```

## Architecture notes

- The graph uses a **dual representation**: raw Python dicts for serialization and a NetworkX `MultiDiGraph` for graph algorithms (allowing multiple edges between the same node pair with different relations). These are kept in sync by the class methods.
- An **edge index** (`_edge_index: set[tuple[str, str, str]]`) provides O(1) duplicate edge detection during ingestion.
- Node and edge IDs are slugified (lowercase, alphanumeric, hyphens).
- All nodes and edges carry `confidence` scores in the range [0, 1].
- Timestamps use ISO 8601 format with UTC timezone.
- The library tracks a `_dirty` flag to avoid unnecessary writes on `save()`.
- Relation proposals allow the schema to evolve: novel relations discovered during LLM extraction are proposed, accumulated across documents, and accepted or rejected.
- Source documents are stored with SHA-256 content hashing for deduplication and version tracking.
- **LLM provider abstraction:** Chat/extraction functions accept callables (`llm_fn`, `llm_extract_fn`), making the core logic provider-agnostic. The `--provider` flag selects between `local` (OpenAI-compatible servers), `anthropic` (Claude API via `ANTHROPIC_API_KEY` env var), and `bedrock` (AWS Bedrock via `boto3` and standard AWS credentials). The Bedrock provider uses the Converse API for chat/extraction and `invoke_model` for embeddings (Titan and Cohere models). By default, embeddings use a local server unless `--embed-model` is explicitly set with the `bedrock` provider.
- **MCP server / orchestrator-as-extractor:** `mcp_server.py` exposes graph operations as MCP tools via FastMCP. The key insight is that the calling orchestrator (e.g. Claude Code) *is* an LLM, so it can perform entity extraction itself using `build_extraction_prompt` and pass structured triples to `ingest_triples`, eliminating the need for a second LLM backend. The `ingest_triples()` method on `KnowledgeGraph` is the public API for this pattern — it accepts pre-extracted triples and handles all validation, node/edge creation, and proposal management.
- **Pre-computed graph layout:** Cytoscape.js visualizations use server-side layout pre-computation via `nx.spring_layout()` (Fruchterman-Reingold). Node x/y positions are embedded in the Cytoscape element JSON and rendered with the `preset` layout, which is instant (no client-side force simulation). The `_compute_layout_positions()` static method handles this for both `cytoscape_elements()` (web frontend) and `export_cytoscape()` (standalone HTML export). Users can still switch to other layout algorithms (cose, circle, breadthfirst, grid, concentric) interactively.
- **Parallel extraction:** `ingest_markdown()` supports a `parallel_extractions` parameter (CLI: `-j N`, web UI: "Parallel Extraction Threads" field). When > 1, LLM extraction calls run concurrently in a `ThreadPoolExecutor` while graph writes (`ingest_triples`, `add_node`, `add_edge`) remain serial on the main thread. Phase 1 creates all structural nodes/edges (fast, serial), Phase 2 dispatches LLM calls in parallel and applies results serially as they complete. This is safe because `build_extraction_prompt()` only reads graph state and `llm_extract_fn` is a pure network call with no graph mutations.

## Common patterns when modifying this code

- All public graph mutation methods (`add_node`, `add_edge`, `remove_node`, etc.) must update the internal dict representation, the NetworkX MultiDiGraph, and the edge index (for edge mutations), and set `self._dirty = True`.
- The `GraphEncoder` class must handle any new types added to node/edge properties.
- The `save()`/`load()` round-trip must be lossless. If you add new fields, update both `to_dict()` and the `load()` constructor logic.
- Embedding operations use a separate dirty flag (`_dirty_embeddings`) and separate persistence file.
- The `main()` function at the bottom provides the CLI and uses argparse.
