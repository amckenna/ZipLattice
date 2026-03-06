# ZipLattice

A portable, JSON-backed knowledge graph manager built on [NetworkX](https://networkx.org/). ZipLattice ingests technical documentation, builds structured knowledge graphs via LLM extraction, and supports graph-based Retrieval-Augmented Generation (RAG) — all backed by flat JSON files with no database required.

## Features

- **Zero-infrastructure persistence** — Graphs stored as plain JSON. No database, no server, fully portable.
- **LLM-powered ingestion** — Parse Markdown, PDF, DOCX, and HTML into structured sections, then extract entities and relationships via any OpenAI-compatible LLM or the Claude API.
- **Graph-based RAG** — Semantic search over node embeddings combined with multi-hop graph expansion to build rich context windows for LLM prompting.
- **Interactive web UI** — FastAPI + HTMX dashboard with Cytoscape.js graph visualization, file upload, and a query interface with markdown-rendered answers.
- **Evolving schema** — Novel relation types discovered during extraction are tracked as proposals that can be reviewed, accepted, or rejected.
- **Model benchmarking** — Compare extraction quality across multiple LLMs side-by-side on the same documents.
- **Document conversion** — Built-in converter turns PDF, DOCX, and HTML into Markdown for ingestion. Pure Python, no system binaries.
- **Visualization** — Auto-exported interactive HTML visualizations using Cytoscape.js (full-featured) or Pyvis (quick overview).

## Requirements

- Python 3.10+
- [networkx](https://pypi.org/project/networkx/) (required)
- An OpenAI-compatible LLM server ([Ollama](https://ollama.ai/), [llama.cpp](https://github.com/ggerganov/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), [LocalAI](https://localai.io/)) or an [Anthropic API key](https://console.anthropic.com/) for ingestion and queries

## Installation

```bash
pip install networkx                                    # required
pip install pyvis                                       # optional: Pyvis visualization
pip install pymupdf4llm mammoth markdownify             # optional: document converter
pip install pymupdf_layout                              # optional: better PDF layout analysis
pip install fastapi uvicorn python-multipart jinja2      # optional: web frontend
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

### Command line

```bash
# Ingest a Markdown document (uses local Ollama by default)
python knowledge_graph.py my_graph.json --ingest-md doc.md --query-model qwen3-coder:30b

# Ingest with Claude API
python knowledge_graph.py my_graph.json --ingest-md doc.md \
    --provider anthropic --extract-model claude-haiku-4-5

# Ingest multiple files
python knowledge_graph.py my_graph.json --ingest-md docs/*.md

# Ask a question using RAG
python query_graph.py my_graph.json ask "how does SAR work?" --query-model qwen3-coder:30b

# Ask with Claude
python query_graph.py my_graph.json ask "how does SAR work?" \
    --provider anthropic --query-model claude-opus-4-6

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

kg.save()
```

## Architecture

### Project layout

```
knowledge_graph.py        # Core library and CLI (single-file)
query_graph.py            # Query interface: search, context, ask (RAG)
web_app.py                # FastAPI web frontend
templates/                # Jinja2 templates (HTMX + Tailwind CSS)
  base.html               #   Layout shell with dark/light theme toggle
  dashboard.html           #   Dashboard — lists all knowledge graphs
  graph_detail.html        #   Graph detail — Cytoscape.js visualization
  upload.html              #   File upload form
  query.html               #   Query form (search, context, ask)
  partials/                #   HTMX partial response fragments
benchmark_models.py       # Model comparison tool for extraction quality
convert_to_markdown.py    # Document-to-Markdown converter (single-file)
test_knowledge_graph.py   # Tests for knowledge_graph.py
test_query_graph.py       # Tests for query_graph.py
test_web_app.py           # Tests for web_app.py
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

- **`local`** — Any OpenAI-compatible server (Ollama, llama.cpp, vLLM, LocalAI) for both chat and extraction
- **`anthropic`** — Claude API for chat and extraction (via `ANTHROPIC_API_KEY` env var). Embeddings always use a local server since Anthropic does not offer an embeddings API.

## Document ingestion pipeline

1. **Convert** — PDF, DOCX, or HTML files are converted to Markdown (via `convert_to_markdown.py` or the web upload page).
2. **Parse** — Markdown is split into sections by heading structure (ATX and Setext formats). Content annotations (code blocks, tables, lists, links) are detected per section.
3. **Extract** — Each section is sent to an LLM that returns RDF-style triples: `{source, target, relation, confidence, ...}`.
4. **Build** — Triples are added as nodes and edges. Unknown relations become `RelationProposal` entries for later review.
5. **Embed** — All new nodes are embedded via a local embedding model.
6. **Store** — The original document text is stored with SHA-256 content hashing for deduplication and version tracking.

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

# Ingestion
python knowledge_graph.py graph.json --ingest-md doc.md --query-model qwen3-coder:30b
python knowledge_graph.py graph.json --ingest-md docs/*.md --embed-model qwen3-embedding
python knowledge_graph.py graph.json --ingest-md doc.md --provider anthropic --extract-model claude-haiku-4-5
python knowledge_graph.py graph.json --preview-md doc.md --sections    # dry run

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

# Server
python knowledge_graph.py graph.json --list-models
```

### query_graph.py

```bash
python query_graph.py graph.json search "query text"
python query_graph.py graph.json context "query text"
python query_graph.py graph.json ask "question" --query-model qwen3-coder:30b
python query_graph.py graph.json ask "question" --provider anthropic --query-model claude-opus-4-6
python query_graph.py graph.json node <node-id>
python query_graph.py graph.json neighbors <node-id> --depth 2
python query_graph.py graph.json path <source-id> <target-id>
python query_graph.py graph.json stats
```

Shared flags: `--api-url`, `--embed-url`, `--embed-model`, `--provider`, `--top-k`, `--depth`, `--node-types`, `--json`, `-v`, `-q`

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
python -m pytest test_knowledge_graph.py test_query_graph.py test_web_app.py -v

# Run individual test suites
python -m pytest test_knowledge_graph.py -v
python -m pytest test_query_graph.py -v
python -m pytest test_web_app.py -v
```

## License

MIT — see [LICENSE](LICENSE) for details.
