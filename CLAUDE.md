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
  analytics.html         #   Quality analytics dashboard
  documents.html         #   Cross-graph document browser & transplant
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

## Key classes and functions

- `KnowledgeGraph` -- Primary class. Manages nodes, edges, embeddings, relation proposals, persistence, search, and visualization.
- `CoreRelation` -- Enum of 25+ built-in relation types (taxonomic, dependency, associative, documentation, functional, contextual).
- `RelationProposal` -- Dataclass tracking proposed new relation types with examples and confidence scores.
- `ProposalStatus` -- Enum: PENDING, ACCEPTED, REJECTED.
- `GraphEncoder` -- Custom JSON encoder for datetime, set, Enum, and Path objects.
- `ollama_embed()` -- Calls OpenAI-compatible `/v1/embeddings` endpoint. Works with Ollama, llama.cpp, vLLM, LocalAI, etc. Used during ingestion and by `query_graph.py` at query time.
- `claude_chat()` -- Calls the Anthropic Messages API for chat/query. Used when `--provider anthropic` is set.
- `claude_extract()` -- Calls the Anthropic Messages API for JSON entity/relation extraction during ingestion. Same three-tier JSON recovery as the local path. Accepts `temperature` (default 0.1).
- `bedrock_chat()` -- Calls the AWS Bedrock Converse API for chat/query. Used when `--provider bedrock` is set.
- `bedrock_extract()` -- Calls the AWS Bedrock Converse API for JSON entity/relation extraction. Same three-tier JSON recovery as the local path.
- `bedrock_embed()` -- Calls AWS Bedrock for embeddings. Supports Titan (`invoke_model`) and Cohere (batched) embedding models.
- `read_http_error_detail()` -- Shared utility for safely extracting response body text from `urllib.error.HTTPError` exceptions.
- `KnowledgeGraph.ingest_triples()` -- Public method that accepts pre-extracted triples directly (no LLM call). Enables the orchestrator-as-extractor pattern used by the MCP server.
- `KnowledgeGraph.extract_document_subgraph()` -- Extracts a document and all its associated nodes, edges, source text, embeddings, and proposals into a portable dict. Used for transplanting documents between graphs.
- `KnowledgeGraph.import_document_subgraph()` -- Imports a previously extracted document subgraph into this graph with smart-merge semantics (descriptions combined, confidence maximised, edges deduplicated). Records transplant provenance.
- `KnowledgeGraph.save_checkpoint()` -- Saves an ingestion checkpoint dict to `<graph_dir>/<stem>_checkpoint.json`. Written atomically (write-to-temp then rename). Used internally by `ingest_markdown()` when `checkpoint=True`.
- `KnowledgeGraph.load_checkpoint()` -- Loads an existing checkpoint file, or returns None. Handles corrupt files gracefully.
- `KnowledgeGraph.clear_checkpoint()` -- Removes the checkpoint file. Called automatically when ingestion completes successfully.
- `KnowledgeGraph.validate()` -- Runs read-only consistency checks on the graph. Returns a `ValidationReport` with errors (dangling edges, taxonomic cycles, sync issues), warnings (contradictory edges, orphan nodes, zero-confidence items), and info (missing embeddings). Available via CLI (`--validate`), web API (`GET /graphs/{name}/validate`), and MCP tool (`validate_graph`).
- `ValidationReport` -- Dataclass returned by `validate()` with `errors`, `warnings`, `info` lists and helper properties `is_valid` and `total_issues`.
- `DocumentDiff` -- Dataclass returned by `diff_document_versions()` with `added`, `removed`, `modified`, `unchanged` section lists and `has_changes`/`summary` helpers.
- `compute_section_hashes()` -- Module-level function that computes per-section SHA-256 hashes from a markdown document. Maps section heading to 12-char hash. Used during ingestion and stored in the source manifest for version comparison.
- `KnowledgeGraph.diff_document_versions()` -- Compares two versions of a document at section level using stored `section_hashes`. Returns a `DocumentDiff` identifying added/removed/modified/unchanged sections.
- `KnowledgeGraph.get_document_history()` -- Returns a rich version timeline for a document: version metadata, section counts, node/edge counts per ingestion, and diffs between consecutive versions. Available via CLI (`--doc-history`), web API (`GET /graphs/{name}/documents/{doc_id}/history`), and MCP tool (`document_history`).
- `KnowledgeGraph.analytics()` -- Computes comprehensive quality analytics: confidence distributions (10-bucket histograms for nodes/edges), per-relation and per-type stats, hub nodes (top-10 by degree), orphan nodes, source document coverage, embedding coverage, component sizes, and a composite quality score (0-100). Available via CLI (`--analytics`), web page (`GET /graphs/{name}/analytics`), JSON API (`GET /api/graphs/{name}/analytics`), and MCP tool (`graph_analytics`).
- `KnowledgeGraph._build_bm25_index()` -- Builds an in-memory BM25 inverted index from node labels, descriptions, body text, and properties. Uses standard BM25 parameters (k1=1.2, b=0.75). Lazily built, auto-invalidated when graph is dirty.
- `KnowledgeGraph.bm25_search()` -- Pure Python BM25 keyword search over node text. Tokenises query (lowercase, stopword removal), scores against inverted index, returns ranked `(node_id, score)` tuples.
- `KnowledgeGraph.hybrid_search()` -- Blends BM25 and semantic similarity with configurable alpha weight (0=pure BM25, 1=pure semantic, default 0.7). Normalises both score sets to [0,1] before combining.
- `KnowledgeGraph.search()` -- Now accepts `mode` parameter: `"semantic"` (default, embedding similarity), `"bm25"` (keyword), or `"hybrid"` (blended). Also accepts `alpha` for hybrid blending weight. Available via CLI (`--search-mode`), web UI (search mode radio buttons), and MCP tool (`semantic_search` with `search_mode` parameter).
- `GraphDiff` -- Dataclass returned by `diff()` with `nodes_added`, `nodes_removed`, `nodes_modified`, `edges_added`, `edges_removed`, `edges_modified`, `proposals_added`, `proposals_changed` lists and `has_changes`/`summary`/`to_dict()` helpers. Includes `counts` dict in serialized form.
- `KnowledgeGraph.snapshot()` -- Deep-copies the current graph state (data + proposals) for later comparison via `diff_from_snapshot()`.
- `KnowledgeGraph.diff(other)` -- Compares this graph (older) against another graph (newer) and returns a `GraphDiff` with field-level node/edge changes and proposal tracking.
- `KnowledgeGraph.diff_from_snapshot(snap)` -- Compares a previously captured snapshot against the current state. Used internally by `ingest_markdown()` to attach a diff summary to aggregate stats.
- `KnowledgeGraph.diff_from_file(path)` -- Loads a graph from a file and diffs it against the current state. Available via CLI (`--diff`), web API (`GET /graphs/{name}/diff?against={other}` and `GET /api/graphs/{name}/diff?against={other}`), and MCP tool (`diff_graphs`).

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
python knowledge_graph.py <path-to-graph.json> --validate
python knowledge_graph.py <path-to-graph.json> --diff <other-graph.json>
python knowledge_graph.py <path-to-graph.json> --analytics
python knowledge_graph.py <path-to-graph.json> --doc-history <doc-id>
python knowledge_graph.py <path-to-graph.json> --preview-md doc.md --sections
python knowledge_graph.py <path-to-graph.json> --ingest-md doc.md
python knowledge_graph.py <path-to-graph.json> --ingest-md notes.txt --query-model qwen3-coder:30b
python knowledge_graph.py <path-to-graph.json> --ingest-md docs/*.md --query-model qwen3-coder:30b --embed-model qwen3-embedding
python knowledge_graph.py <path-to-graph.json> --ingest-md doc.md --query-model qwen3-coder:30b --api-url http://exo:11434
# Parallel ingestion: 4 concurrent LLM extraction threads, serial graph writes
python knowledge_graph.py <path-to-graph.json> --ingest-md docs/*.md --query-model qwen3-coder:30b -j 4
# Incremental ingestion: only re-extract sections that changed since last ingestion
python knowledge_graph.py <path-to-graph.json> --ingest-md doc.md --query-model qwen3-coder:30b --incremental
# Checkpoint and resume: saves progress after each section, resumes on re-run
python knowledge_graph.py <path-to-graph.json> --ingest-md doc.md --query-model qwen3-coder:30b --checkpoint
# Custom temperature (0=deterministic, higher=more diverse)
python knowledge_graph.py <path-to-graph.json> --ingest-md doc.md --query-model qwen3-coder:30b --temperature 0.0
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
python query_graph.py <path-to-graph.json> search "radar detection" --search-mode bm25
python query_graph.py <path-to-graph.json> search "radar detection" --search-mode hybrid --alpha 0.5
python query_graph.py <path-to-graph.json> context "how does SAR work?"
python query_graph.py <path-to-graph.json> context "how does SAR work?" --search-mode hybrid
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
# Benchmark with custom temperature
python benchmark_models.py doc.md --models qwen3-coder:30b --temperature 0.0
# Benchmark with Bedrock
python benchmark_models.py doc.md --models us.anthropic.claude-haiku-4-5-20251001-v1:0 --provider bedrock

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
- **Incremental ingestion:** `ingest_markdown()` supports an `incremental` parameter (CLI: `--incremental`, web UI: "Incremental" checkbox). When enabled and the document has a previous version with section hashes, `diff_document_versions()` identifies which sections are unchanged. Unchanged sections skip LLM extraction entirely — structural nodes and edges are still created/updated (cheap), but the expensive LLM extraction call is avoided. This dramatically reduces LLM costs when re-ingesting documents where only a few sections changed. The aggregate stats include a `sections_skipped_incremental` counter. Requires `preserve_source=True` so that section hashes from the previous version are available for comparison.
- **Checkpoint and resume:** `ingest_markdown()` supports a `checkpoint` parameter (CLI: `--checkpoint`, web UI: "Checkpoint" checkbox). When enabled, after each section's LLM extraction completes, the graph is saved and a checkpoint file (`<stem>_checkpoint.json`) records which sections have been processed with their stats. If ingestion is interrupted (e.g. model server crash, process kill), re-running the same command with `--checkpoint` detects the checkpoint, skips completed sections (data already persisted in the graph), and resumes extraction from where it left off. The checkpoint is matched by `doc_id` and `content_hash` — stale checkpoints from different documents or versions are automatically discarded. On successful completion, the checkpoint file is deleted. The aggregate stats include a `sections_resumed` counter.
- **Progress bar and token throughput:** The CLI progress callback (`_make_progress_callback`) renders a visual progress bar (`[=====>          ] 3/12`) tracking sections completed, estimated tokens processed, and running average tokens/second. Token estimates use a `chars // 4` heuristic (consistent with `local_extract` debug logging). Progress events include `estimated_tokens` (in `doc_start`, `section_start`, `section_done`, `doc_done`) and `tokens_per_second` (in `section_done`, `doc_done`). The aggregate stats dict returned by `ingest_markdown()` also includes `estimated_tokens` and `tokens_per_second`. Per-section records in `aggregate_stats["sections"]` include both fields for detailed throughput analysis.
- **Document subgraph transplant:** `extract_document_subgraph()` extracts a document and all its associated nodes, edges, source text, embeddings, and relation proposals into a portable dict. `import_document_subgraph()` imports that dict into another graph with smart-merge semantics (same as merge). Transplant provenance is recorded in the source manifest as `transplanted_from` entries. The web UI at `/documents` provides a cross-graph document browser showing which graphs each document belongs to (primary vs transplanted) with node/edge counts, search, and one-click transplant.

## Common patterns when modifying this code

- All public graph mutation methods (`add_node`, `add_edge`, `remove_node`, etc.) must update the internal dict representation, the NetworkX MultiDiGraph, and the edge index (for edge mutations), and set `self._dirty = True`.
- The `GraphEncoder` class must handle any new types added to node/edge properties.
- The `save()`/`load()` round-trip must be lossless. If you add new fields, update both `to_dict()` and the `load()` constructor logic.
- Embedding operations use a separate dirty flag (`_dirty_embeddings`) and separate persistence file.
- The `main()` function at the bottom provides the CLI and uses argparse.
