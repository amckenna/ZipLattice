# ZipLattice

A portable, JSON-backed knowledge graph manager built on [NetworkX](https://networkx.org/). ZipLattice ingests technical documentation, builds structured knowledge graphs via LLM extraction, and supports graph-based Retrieval-Augmented Generation (RAG) — all backed by flat JSON files with no database required.

## Features

- **Zero-infrastructure persistence** — Graphs stored as plain JSON. No database, no server, fully portable.
- **LLM-powered ingestion** — Parse Markdown, PDF, DOCX, and HTML into structured sections, then extract entities and relationships via any OpenAI-compatible LLM, the Claude API, or AWS Bedrock.
- **Graph-based RAG** — Semantic search over node embeddings combined with multi-hop graph expansion to build rich context windows for LLM prompting.
- **Interactive web UI** — FastAPI + HTMX dashboard with Cytoscape.js graph visualization, file upload, streaming ingestion progress, multi-turn chat, and a query interface with markdown-rendered answers.
- **MCP server** — Expose graph operations as MCP tools via FastMCP. Implements an orchestrator-as-extractor pattern where the calling LLM performs entity extraction itself, eliminating the need for a second LLM backend.
- **Evolving schema** — Novel relation types discovered during extraction are tracked as proposals that can be reviewed, accepted, or rejected.
- **Hallucination detection** — Extracted entities are grounded against source text using word overlap, filtering out hallucinated nodes.
- **Model benchmarking** — Compare extraction quality across multiple LLMs side-by-side on the same documents.
- **Document conversion** — Built-in converter turns PDF, DOCX, and HTML into Markdown for ingestion. Pure Python, no system binaries.
- **Visualization** — Auto-exported interactive HTML visualizations using Cytoscape.js (full-featured, with server-side layout pre-computation) or Pyvis (quick overview).
- **Provenance tracking** — Source documents stored with SHA-256 content hashing, ingestion IDs link nodes back to specific runs, and timestamps track creation and updates.

## Requirements

- Python 3.10+
- [networkx](https://pypi.org/project/networkx/) (required)
- An LLM backend for ingestion and queries — any of:
  - [Ollama](https://ollama.ai/), [llama.cpp](https://github.com/ggerganov/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), [LocalAI](https://localai.io/) (OpenAI-compatible)
  - [Anthropic API](https://console.anthropic.com/) (Claude)
  - [AWS Bedrock](https://aws.amazon.com/bedrock/) (Claude, Titan, Nova, Cohere models)

## Installation

```bash
pip install networkx                                    # required
pip install pyvis                                       # optional: Pyvis visualization
pip install pymupdf4llm mammoth markdownify             # optional: document converter
pip install pymupdf_layout                              # optional: better PDF layout analysis
pip install fastapi uvicorn python-multipart jinja2      # optional: web frontend
pip install fastmcp                                     # optional: MCP server
pip install boto3                                       # optional: AWS Bedrock provider
```

There is no package to install for ZipLattice itself — run the CLI or import the module directly from the project directory.

## Quick start

### Web UI

```bash
# Start the web frontend
uvicorn web_app:app --reload    # http://localhost:8000

# Or with a custom graphs directory
ZIPLATTICE_GRAPHS_DIR=./my_graphs uvicorn web_app:app
```

The web UI provides:
- **Dashboard** — View, export, and delete knowledge graphs
- **Upload** — Drag-and-drop PDF, DOCX, HTML, or Markdown files for conversion and LLM ingestion with streaming progress
- **Graph detail** — Interactive Cytoscape.js visualization with search, layout controls, node type filters, and a detail panel
- **Query** — Search, build RAG context, or ask questions with markdown-rendered LLM answers
- **Chat** — Multi-turn follow-up conversations that preserve RAG context and message history
- **Export** — Download a graph as a portable .zip archive

### MCP server

The MCP server exposes graph operations as tools for LLM orchestrators like Claude Code. It uses the **orchestrator-as-extractor** pattern — the calling LLM extracts entities itself using `build_extraction_prompt` and passes structured triples to `ingest_triples`, so no second LLM backend is needed.

```bash
python mcp_server.py    # stdio transport (for Claude Code)
```

Available MCP tools include: `build_extraction_prompt`, `ingest_triples`, `parse_markdown_sections`, `store_source`, `add_node`, `get_node`, `remove_node`, `search_nodes`, `get_neighbors`, `get_subgraph`, `add_edge`, `get_edges`, `remove_edges`, `graph_stats`, `save_graph`, `list_proposals`, `accept_proposal`, `reject_proposal`, `embed_nodes`, `semantic_search`.

### Command line

```bash
# Ingest a Markdown document (uses local Ollama by default)
python knowledge_graph.py my_graph.json --ingest-md doc.md --query-model qwen3-coder:30b

# Ingest with Claude API (Haiku for fast extraction)
python knowledge_graph.py my_graph.json --ingest-md doc.md \
    --provider anthropic --extract-model claude-haiku-4-5

# Ingest with AWS Bedrock
python knowledge_graph.py my_graph.json --ingest-md doc.md \
    --provider bedrock --extract-model us.anthropic.claude-sonnet-4-20250514-v1:0

# Ingest with Bedrock embeddings (Titan)
python knowledge_graph.py my_graph.json --ingest-md doc.md \
    --provider bedrock --extract-model us.anthropic.claude-sonnet-4-20250514-v1:0 \
    --embed-model amazon.titan-embed-text-v2:0

# Ingest multiple files
python knowledge_graph.py my_graph.json --ingest-md docs/*.md

# Ask a question using RAG
python query_graph.py my_graph.json ask "how does SAR work?" --query-model qwen3-coder:30b

# Ask with Claude
python query_graph.py my_graph.json ask "how does SAR work?" \
    --provider anthropic --query-model claude-opus-4-6

# Ask with Bedrock
python query_graph.py my_graph.json ask "how does SAR work?" \
    --provider bedrock --query-model us.anthropic.claude-sonnet-4-20250514-v1:0

# Semantic search
python query_graph.py my_graph.json search "synthetic aperture radar"

# Build RAG context block (for pasting into your own LLM prompt)
python query_graph.py my_graph.json context "how does SAR work?"

# Graph inspection
python knowledge_graph.py my_graph.json --stats
python knowledge_graph.py my_graph.json --node <id>
python knowledge_graph.py my_graph.json --neighbors <id> --depth 2
```

### As a library

```python
from knowledge_graph import KnowledgeGraph

kg = KnowledgeGraph("my_graph.json")

# Add nodes and edges
kg.add_node("python", type="technology", label="Python",
            properties={"domain": "programming"})
kg.add_node("asyncio", type="concept", label="asyncio",
            properties={"domain": "concurrency"})
kg.add_edge("asyncio", "python", relation="part_of")

# Query
neighbors = kg.get_neighbors("python", max_depth=2)
path = kg.shortest_path("asyncio", "python")
central = kg.get_central_nodes(top_n=5, method="betweenness")

# Search (requires embeddings)
results = kg.search("async programming", embed_fn=my_embed_fn, top_k=10)

# Ingest pre-extracted triples (orchestrator-as-extractor pattern)
stats = kg.ingest_triples(triples, text=section_text, doc_id="my-doc")

kg.save()
```

## Architecture

For detailed architecture documentation including data flow diagrams, JSON schemas, and design decisions, see [ARCHITECTURE.md](ARCHITECTURE.md).

### Project layout

```
knowledge_graph.py        # Core library and CLI (single-file)
query_graph.py            # Query interface: search, context, ask (RAG)
web_app.py                # FastAPI web frontend (HTMX + Tailwind CSS)
mcp_server.py             # MCP server (FastMCP) — orchestrator-as-extractor
templates/                # Jinja2 templates for the web frontend
  base.html               #   Layout shell with dark/light theme toggle
  dashboard.html          #   Dashboard — lists all knowledge graphs
  graph_detail.html       #   Graph detail — Cytoscape.js visualization
  upload.html             #   File upload form
  query.html              #   Query form (search, context, ask)
  partials/               #   HTMX partial response fragments
benchmark_models.py       # Model comparison tool for extraction quality
convert_to_markdown.py    # Document-to-Markdown converter (single-file)
test_knowledge_graph.py   # Tests for knowledge_graph.py
test_query_graph.py       # Tests for query_graph.py
test_web_app.py           # Tests for web_app.py
test_mcp_server.py        # Tests for mcp_server.py
ARCHITECTURE.md           # Detailed architecture documentation
```

### Runtime file layout

For a graph named `my_graph.json`, ZipLattice creates a dedicated directory:

```
my_graph/
  my_graph.json                  # nodes, edges, metadata, proposals
  my_graph_embeddings.json       # node ID → vector mappings
  my_graph_sources/              # stored document text (SHA-256 deduplicated)
  my_graph_cytoscape.html        # auto-exported Cytoscape visualization
  my_graph_pyvis.html            # auto-exported Pyvis visualization
```

### Data model

**Nodes** represent entities (concepts, technologies, documents, people, etc.) with an ID, type, label, arbitrary properties, confidence score, and timestamps.

**Edges** are directed relationships between nodes. The graph uses a NetworkX `MultiDiGraph`, allowing multiple edges between the same node pair with different relation types.

**Relations** come from a built-in schema of 25+ core types:

| Category | Relations |
|---|---|
| Taxonomic | `is_a`, `part_of`, `has_part`, `subclass_of`, `instance_of` |
| Dependency | `depends_on`, `required_by`, `causes`, `caused_by` |
| Associative | `related_to`, `similar_to`, `references`, `implements`, `extends` |
| Documentation | `documents`, `documented_by`, `derived_from`, `supersedes` |
| Functional | `uses`, `used_by`, `configured_by`, `produces`, `consumes` |
| Contextual | `belongs_to`, `contains`, `tagged_with` |

Custom relations can be registered at runtime, and novel relations discovered during LLM extraction are tracked as proposals for review.

### Internal representation

The graph maintains two synchronized representations:

1. **Python dicts** — Canonical store for nodes and edges, used for JSON serialization.
2. **NetworkX MultiDiGraph** — Built from the dicts on load, used for graph algorithms (traversal, centrality, shortest paths, connected components).

An **edge index** (`set[tuple[str, str, str]]`) provides O(1) duplicate edge detection during ingestion. All mutations go through class methods that update both representations and set a dirty flag for the next save.

### LLM provider abstraction

Core functions accept callables (`llm_fn`, `llm_extract_fn`), making the logic provider-agnostic. The `--provider` flag selects between:

| Provider | Chat | Extract | Embed | Notes |
|----------|------|---------|-------|-------|
| **`local`** | OpenAI-compatible | OpenAI-compatible | OpenAI-compatible | Ollama, llama.cpp, vLLM, LocalAI |
| **`anthropic`** | Claude API | Claude API | Local server | Anthropic does not offer an embeddings API |
| **`bedrock`** | Converse API | Converse API | Titan / Cohere | AWS credentials via env vars, profiles, or IAM roles |

Bedrock embeddings support both Titan (`invoke_model`, one-at-a-time) and Cohere (batched) models with automatic retry and exponential backoff. By default, embeddings use a local server unless `--embed-model` is explicitly set to a Bedrock model ID.

### Extraction pipeline

1. **Convert** — PDF, DOCX, or HTML files are converted to Markdown (via `convert_to_markdown.py` or the web upload page).
2. **Parse** — Markdown is split into sections by heading structure (ATX and Setext formats). Content annotations (code blocks, tables, lists, links) are detected per section.
3. **Extract** — Each section is sent to an LLM that returns RDF-style triples: `{source, target, relation, confidence, ...}`. Output is parsed with three-tier JSON recovery (direct parse → boundary extraction → truncation salvage).
4. **Ground** — Extracted entities are validated against source text using word overlap to filter hallucinations.
5. **Build** — Triples are added as nodes and edges. Unknown relations become `RelationProposal` entries for later review.
6. **Embed** — All new nodes are embedded via a local or Bedrock embedding model.
7. **Store** — The original document text is stored with SHA-256 content hashing for deduplication and version tracking.

## Query modes

ZipLattice provides three query modes, available from both the CLI and web UI:

| Mode | What it does | Requires LLM? |
|------|-------------|----------------|
| **Search** | Semantic similarity search — returns ranked nodes by embedding similarity | No |
| **Context** | Builds a RAG context block — searches nodes, expands neighborhoods, formats structured text for pasting into an LLM prompt | No |
| **Ask** | Full RAG pipeline — builds context from the graph and sends it with your question to an LLM for a natural-language answer | Yes |

## CLI reference

### knowledge_graph.py

```bash
# Inspection
python knowledge_graph.py graph.json --stats
python knowledge_graph.py graph.json --node <id>
python knowledge_graph.py graph.json --neighbors <id> --depth 2
python knowledge_graph.py graph.json --sources
python knowledge_graph.py graph.json --check-sources
python knowledge_graph.py graph.json --verify-embeddings
python knowledge_graph.py graph.json --list-models

# Ingestion
python knowledge_graph.py graph.json --ingest-md doc.md --query-model qwen3-coder:30b
python knowledge_graph.py graph.json --ingest-md docs/*.md --embed-model qwen3-embedding
python knowledge_graph.py graph.json --ingest-md doc.md --provider anthropic --extract-model claude-haiku-4-5
python knowledge_graph.py graph.json --ingest-md doc.md --provider bedrock \
    --extract-model us.anthropic.claude-sonnet-4-20250514-v1:0 --bedrock-region us-west-2
python knowledge_graph.py graph.json --preview-md doc.md --sections    # dry run
python knowledge_graph.py graph.json --ingest-md doc.md --auto-accept  # auto-accept proposals
python knowledge_graph.py graph.json --ingest-md doc.md --no-viz       # skip visualization export

# Visualization
python knowledge_graph.py graph.json --cytoscape output.html
python knowledge_graph.py graph.json --pyvis output.html
python knowledge_graph.py graph.json --cytoscape output.html --center <node-id> --depth 3

# Relation proposals
python knowledge_graph.py graph.json --proposals
python knowledge_graph.py graph.json --accept <relation_name>
python knowledge_graph.py graph.json --reject <relation_name>
python knowledge_graph.py graph.json --accept-all
python knowledge_graph.py graph.json --patterns

# Export
python knowledge_graph.py graph.json --split output_dir/
```

Shared flags: `--api-url`, `--embed-url`, `--embed-model`, `--query-model`, `--extract-model`, `--provider {local,anthropic,bedrock}`, `--bedrock-region`, `--bedrock-profile`, `-v`, `-q`

### query_graph.py

```bash
python query_graph.py graph.json search "query text"
python query_graph.py graph.json context "query text"
python query_graph.py graph.json ask "question" --query-model qwen3-coder:30b
python query_graph.py graph.json ask "question" --provider anthropic --query-model claude-opus-4-6
python query_graph.py graph.json ask "question" --provider bedrock \
    --query-model us.anthropic.claude-sonnet-4-20250514-v1:0
python query_graph.py graph.json node <node-id>
python query_graph.py graph.json neighbors <node-id> --depth 2
python query_graph.py graph.json path <source-id> <target-id>
python query_graph.py graph.json stats
python query_graph.py graph.json list-models
```

Shared flags: `--api-url`, `--embed-url`, `--embed-model`, `--provider`, `--bedrock-region`, `--bedrock-profile`, `--top-k`, `--depth`, `--node-types`, `--json`, `-v`, `-q`

### benchmark_models.py

```bash
# Compare extraction quality across models
python benchmark_models.py doc.md --models qwen3-coder:30b gemma3:27b llama3.1:70b
python benchmark_models.py doc.md --models claude-haiku-4-5 claude-sonnet-4-6 --provider anthropic
python benchmark_models.py docs/*.md --models modelA modelB --api-url http://exo:11434
python benchmark_models.py doc.md --models modelA modelB --max-sections 5 --json
```

### convert_to_markdown.py

```bash
python convert_to_markdown.py document.pdf -o document.md
python convert_to_markdown.py file1.pdf file2.docx file3.html -d output_dir/
```

| Format | Library | Notes |
|--------|---------|-------|
| PDF | `pymupdf4llm` | Heading detection via font analysis. `pymupdf_layout` improves page layout. |
| DOCX | `mammoth` + `markdownify` | Semantic HTML intermediate, ATX headings |
| HTML | `markdownify` | Strips script/style/nav/footer before conversion |

## Testing

```bash
# Run all tests (no Ollama or API keys needed)
python -m pytest test_knowledge_graph.py test_query_graph.py test_web_app.py test_mcp_server.py -v

# Run individual test suites
python -m pytest test_knowledge_graph.py -v
python -m pytest test_query_graph.py -v
python -m pytest test_web_app.py -v
python -m pytest test_mcp_server.py -v
```

## License

MIT — see [LICENSE](LICENSE) for details.
