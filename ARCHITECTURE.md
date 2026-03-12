# ZipLattice Architecture

This document describes the architecture of ZipLattice, a portable, JSON-backed
knowledge graph manager built on NetworkX. It ingests technical documentation,
builds structured knowledge graphs, and supports graph-based Retrieval-Augmented
Generation (RAG) for LLM agents.

---

## System Overview

ZipLattice is organized as a set of cooperating modules around a single-file core
library. Each module can be used independently (CLI or library import) or composed
through the web frontend or MCP server.

```
┌─────────────────────────────────────────────────────────────┐
│                        User / Client                        │
├──────────┬──────────┬───────────────┬───────────────────────┤
│  CLI     │  Web UI  │  MCP Server   │  Python import        │
│          │ (HTMX)   │ (FastMCP)     │                       │
├──────────┴──────────┴───────────────┴───────────────────────┤
│                    query_graph.py                            │
│              (search, context, ask — RAG)                    │
├─────────────────────────────────────────────────────────────┤
│                   knowledge_graph.py                        │
│         (core library — graph, ingestion, search,           │
│          embeddings, visualization, persistence)            │
├──────────┬──────────────────┬───────────────────────────────┤
│  Local   │   Anthropic API  │   AWS Bedrock                 │
│ (Ollama, │   (Claude)       │   (Converse API,              │
│  vLLM…)  │                  │    Titan/Cohere embeddings)   │
└──────────┴──────────────────┴───────────────────────────────┘
```

---

## Module Responsibilities

### knowledge_graph.py — Core Library

The heart of ZipLattice. A single-file library containing all graph management,
ingestion, search, embedding, and visualization logic.

**Key classes:**

| Class / Function | Purpose |
|---|---|
| `KnowledgeGraph` | Primary class — nodes, edges, embeddings, proposals, persistence, search, visualization |
| `CoreRelation` | Enum of 25+ built-in relation types (taxonomic, dependency, associative, etc.) |
| `RelationProposal` | Tracks proposed novel relation types with examples and confidence |
| `ProposalStatus` | Lifecycle enum: PENDING → ACCEPTED / REJECTED |
| `GraphEncoder` | Custom JSON encoder for datetime, set, Enum, Path |

**LLM provider functions** (module-level):

| Function | Provider | Purpose |
|---|---|---|
| `local_extract()` | Local (OpenAI-compat) | JSON entity/relation extraction |
| `ollama_embed()` | Local (OpenAI-compat) | Embedding via `/v1/embeddings` |
| `claude_chat()` | Anthropic | Chat / query completion |
| `claude_extract()` | Anthropic | JSON entity/relation extraction |
| `bedrock_chat()` | AWS Bedrock | Chat via Converse API |
| `bedrock_extract()` | AWS Bedrock | JSON extraction via Converse API |
| `bedrock_embed()` | AWS Bedrock | Embeddings (Titan `invoke_model`, Cohere batched) |

### query_graph.py — Query Application

CLI and library for querying an existing knowledge graph. Imports core functions
from `knowledge_graph.py`.

| Function | Purpose |
|---|---|
| `search_nodes()` | Semantic search with embedding similarity + optional graph expansion |
| `build_context()` | Constructs a RAG context window from search results |
| `ask()` | Full RAG pipeline: search → build context → LLM answer |

### web_app.py — Web Frontend

FastAPI application providing a browser-based UI for managing multiple knowledge
graphs. Uses HTMX for dynamic updates (no full-page reloads) and Tailwind CSS
for styling.

**Key routes:**

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard — lists all graphs with stats |
| `/graphs/{name}` | GET | Graph detail with Cytoscape.js visualization |
| `/upload` | GET/POST | Document upload (PDF, DOCX, HTML, MD) |
| `/ingest` | POST | LLM-powered ingestion of uploaded documents |
| `/query` | GET/POST | Search / context / ask queries |
| `/chat` | POST | Multi-turn chat follow-up |
| `/graphs/{name}` | DELETE | Delete a graph |
| `/graphs/{name}/export` | GET | Export as portable .zip archive |
| `/graphs/{name}/source/{doc_id}` | GET | Retrieve stored source text |

### mcp_server.py — MCP Server

Exposes graph operations as MCP tools via FastMCP for use by LLM orchestrators
(e.g., Claude Code). Implements the **orchestrator-as-extractor** pattern — the
calling LLM performs entity extraction itself rather than delegating to a second
LLM.

**MCP tools:** `build_extraction_prompt`, `ingest_triples`, `parse_markdown_sections`,
`store_source`, `add_node`, `get_node`, `remove_node`, `search_nodes`,
`get_neighbors`, `get_subgraph`, `add_edge`, `remove_edge`, `get_edges`,
`save_graph`, `list_graphs`, `graph_stats`.

### convert_to_markdown.py — Document Converter

Standalone single-file converter for PDF, DOCX, and HTML to Markdown. Used by
the web frontend during file upload and available as a CLI tool.

### benchmark_models.py — Model Comparison

CLI tool for comparing extraction quality across multiple LLM models on the same
document(s). Supports local and Anthropic providers.

---

## Architectural Patterns

### 1. Dual Representation

The graph maintains two synchronized representations:

- **Raw dict** (`_data`): JSON-serializable dictionary used for persistence.
- **NetworkX MultiDiGraph** (`_G`): In-memory graph used for algorithms
  (shortest path, neighborhood traversal, connected components).

All public mutation methods (`add_node`, `add_edge`, `remove_node`, etc.) update
both representations and set `self._dirty = True`. The MultiDiGraph allows
multiple edges between the same node pair with different relation types.

### 2. Edge Index for O(1) Deduplication

An edge index (`_edge_index: set[tuple[str, str, str]]`) tracks
`(source, target, relation)` triples. During bulk ingestion, this provides
constant-time duplicate detection. When a duplicate is found, existing edge
properties are merged and confidence is maximized rather than creating a
duplicate.

### 3. LLM Provider Abstraction

Core ingestion and query methods accept **callables** (`llm_fn`,
`llm_extract_fn`, `embed_fn`) rather than coupling to a specific provider. The
`--provider` CLI flag selects between:

- **local** — OpenAI-compatible servers (Ollama, llama.cpp, vLLM, LocalAI)
- **anthropic** — Claude API via `ANTHROPIC_API_KEY`
- **bedrock** — AWS Bedrock via `boto3` and standard AWS credentials

This makes the core graph logic completely provider-agnostic.

### 4. Orchestrator-as-Extractor (MCP)

Instead of a two-tier LLM pipeline (orchestrator calls graph, graph calls LLM
for extraction), the MCP server leverages the fact that the calling orchestrator
**is** an LLM:

```
Orchestrator (Claude Code)
    │
    ├── calls build_extraction_prompt() → schema-aware prompt
    ├── extracts entities/relations itself using that prompt
    └── calls ingest_triples() with structured JSON
         └── Graph validates, creates nodes/edges, manages proposals
```

This eliminates the need for a second LLM backend and reduces latency.

### 5. Three-Tier JSON Recovery

LLM-generated JSON is unreliable. Extraction uses a three-tier fallback:

1. **Direct parse** — `json.loads()` on the raw output.
2. **Boundary extraction** — Find the outermost `[…]` or `{…}` and parse.
3. **Truncation salvage** — Repair incomplete JSON objects from truncated output.

This same recovery logic is used across all providers (local, Anthropic, Bedrock).

### 6. Relation Proposal System

The schema evolves at runtime. When LLM extraction produces a relation type not
in `CoreRelation`, the system:

1. Creates a `RelationProposal` (status: PENDING) with the justification and
   example usage.
2. Adds the edge at low confidence.
3. Accumulates additional examples across documents.
4. Proposals can be accepted or rejected, promoting them to custom relations or
   discarding them.

### 7. Hallucination Detection

During ingestion, extracted entities are **grounded** against the source text
using word overlap. Entities with insufficient grounding are flagged or filtered,
reducing hallucinated nodes in the graph.

### 8. Provenance Tracking

- **Source documents** are stored in a `_sources/` directory with SHA-256 content
  hashing for deduplication and version tracking.
- **Ingestion IDs** link nodes and edges back to the specific ingestion run.
- **Source tags** distinguish `lm_extract` (LLM-generated) from `manual` entries.
- **Timestamps** (ISO 8601 UTC) track creation and last update.

---

## Data Flow

### Ingestion Pipeline

```
Document (PDF / DOCX / HTML / Markdown)
    │
    ▼
┌──────────────────────┐
│ convert_to_markdown   │  (if not already Markdown)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ parse_markdown_sections │  Split by headings
└──────────┬───────────┘
           ▼
    ┌──────┴──────┐
    │  Per section │ ◄── loop
    │              │
    │  build_extraction_prompt()  → schema-aware prompt
    │       │
    │       ▼
    │  LLM provider (local / anthropic / bedrock)
    │       │
    │       ▼
    │  Three-tier JSON recovery
    │       │
    │       ▼
    │  ingest_triples()
    │   ├── normalize triple keys
    │   ├── ground entities in source text
    │   ├── create/merge nodes
    │   ├── create/merge edges
    │   └── propose novel relations
    │
    └──────┬──────┘
           ▼
┌──────────────────────┐
│ Embedding pipeline    │
│  embed_fn(labels +    │
│  descriptions)        │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ save() + save_embed() │ → JSON files on disk
└──────────────────────┘
```

### Query Pipeline

```
User query (natural language)
    │
    ▼
┌──────────────────────┐
│ embed_fn(query)       │  → query vector
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ find_similar()        │  → top-K nodes by cosine similarity
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ get_neighbors() /     │  → expand graph neighborhood
│ get_subgraph()        │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ build_context()       │  → formatted Markdown context window
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ llm_fn(prompt +       │  → natural language answer
│ context)              │
└──────────────────────┘
```

### Visualization Pipeline

```
KnowledgeGraph instance
    │
    ├── export_cytoscape()  → standalone HTML with Cytoscape.js (CDN)
    │     Interactive: search, layout selection, confidence filter,
    │     detail panel, source preview
    │
    └── export_pyvis()      → standalone HTML with Pyvis / vis.js
          Physics-based force-directed layout
```

Both are auto-exported after ingestion to the graph's dedicated directory.

---

## Persistence and Disk Layout

Each graph gets a dedicated directory named after the graph file stem:

```
my_graph/
├── my_graph.json                # Main graph (nodes, edges, metadata, proposals)
├── my_graph_embeddings.json     # Node embedding vectors + model metadata
├── my_graph_sources/            # Stored source documents
│   ├── sha256-abc123…txt
│   └── sha256-def456…txt
├── my_graph_cytoscape.html      # Auto-exported Cytoscape visualization
└── my_graph_pyvis.html          # Auto-exported Pyvis visualization
```

**Dirty flags** control write behavior:
- `_dirty` — set by any graph mutation; cleared by `save()`
- `_dirty_embeddings` — set by embedding updates; cleared by `save_embeddings()`

Calling `save()` when no flag is set is a no-op, avoiding unnecessary I/O.

### Graph JSON Schema

```json
{
  "meta": {
    "version": "1.0.0",
    "created": "ISO-8601",
    "updated": "ISO-8601",
    "description": "…",
    "core_relations": ["is_a", "part_of", "…"],
    "custom_relations": ["…"],
    "node_types": ["concept", "entity", "…"],
    "sources": { "doc_id": { "hash": "sha256-…", "…" } }
  },
  "nodes": {
    "node-id": {
      "label": "Human Label",
      "type": "concept",
      "properties": { "description": "…" },
      "source": "ingestion",
      "confidence": 0.85,
      "created": "ISO-8601",
      "updated": "ISO-8601"
    }
  },
  "edges": [
    {
      "source": "node-a",
      "target": "node-b",
      "relation": "depends_on",
      "properties": { "context": "…" },
      "source_tag": "lm_extract",
      "confidence": 0.9,
      "weight": 1.0,
      "created": "ISO-8601",
      "updated": "ISO-8601"
    }
  ],
  "relation_proposals": [
    {
      "name": "influences",
      "status": "pending",
      "justification": "…",
      "examples": [{ "source": "A", "target": "B", "context": "…" }],
      "confidence": 0.6,
      "source_docs": ["doc_id"]
    }
  ]
}
```

### Embeddings JSON Schema

```json
{
  "_meta": {
    "model": "qwen3-embedding",
    "dimension": 1536,
    "created": "ISO-8601"
  },
  "node-id-1": [0.1, 0.2, "…", 0.9],
  "node-id-2": [0.05, 0.15, "…", 0.85]
}
```

---

## Type System

### Node Types

`concept`, `entity`, `document`, `section`, `technology`, `tool`, `process`,
`event`, `person`, `organization`, `code`, `configuration`, `artifact`, `custom`

### Core Relations (25+)

Organized by category:

- **Taxonomic:** `is_a`, `part_of`, `has_part`, `instance_of`
- **Dependency:** `depends_on`, `requires`, `uses`, `produces`
- **Associative:** `related_to`, `similar_to`, `contrasts_with`
- **Documentation:** `documents`, `references`, `defines`, `examples`
- **Functional:** `implements`, `extends`, `wraps`, `calls`
- **Contextual:** `tagged_with`, `belongs_to`, `applies_to`

Novel relations discovered during extraction become `RelationProposal` entries.

### IDs and Confidence

- **Node/edge IDs** are slugified: lowercase, alphanumeric, hyphens only.
- **Confidence scores** range from 0.0 to 1.0. Manual entries default to 1.0;
  LLM-extracted entries carry the model's assessed confidence; proposed relations
  start at lower confidence.

---

## Web Frontend Architecture

```
Browser
  │
  ├── Tailwind CSS (CDN) ─── styling + dark/light theme
  ├── HTMX (CDN) ─────────── dynamic partial updates (no JS framework)
  └── Cytoscape.js (CDN) ──── interactive graph visualization
        │
        ▼
FastAPI (web_app.py)
  │
  ├── Jinja2 templates
  │     ├── base.html          ← layout shell, nav, CDN links
  │     ├── dashboard.html     ← graph listing
  │     ├── graph_detail.html  ← Cytoscape visualization + controls
  │     ├── upload.html        ← file upload form
  │     ├── query.html         ← search / context / ask
  │     └── partials/          ← HTMX response fragments
  │
  ├── Factory helpers
  │     ├── _build_embed_fn()    ← creates embedding callable
  │     ├── _build_extract_fn()  ← creates extraction callable
  │     └── _build_llm_fn()      ← creates chat callable
  │
  └── knowledge_graph.py (library import)
```

The web app discovers graphs by scanning a configurable directory
(`ZIPLATTICE_GRAPHS_DIR` env var, defaults to current directory).

---

## Testing Strategy

All tests run without external dependencies (no Ollama or LLM server required).

| Test file | Covers |
|---|---|
| `test_knowledge_graph.py` | Core class: persistence, node/edge operations, extraction prompt building, triple ingestion, JSON recovery |
| `test_query_graph.py` | Query functions: search, context building, ask pipeline |
| `test_web_app.py` | Web routes: dashboard, upload, query, graph detail, export |
| `test_mcp_server.py` | MCP tools: ingestion, node/edge operations, orchestrator pattern |

```bash
# Run all tests
python -m pytest test_knowledge_graph.py test_query_graph.py test_web_app.py test_mcp_server.py -v
```

---

## Design Decisions

1. **Single-file core** — `knowledge_graph.py` is intentionally kept as one file
   for maximum portability. It can be dropped into any project without package
   installation.

2. **JSON over SQLite** — The graph is stored as plain JSON for human readability,
   easy diffing, and compatibility with version control. The trade-off is slower
   load times for very large graphs.

3. **NetworkX MultiDiGraph** — Chosen over DiGraph to allow multiple edges between
   the same node pair (e.g., A `depends_on` B and A `uses` B simultaneously).

4. **Separate embedding storage** — Embeddings are stored in a separate file with
   their own dirty flag to avoid rewriting the entire graph when only vectors
   change.

5. **HTMX over SPA** — The web frontend uses HTMX for dynamic updates instead of
   a JavaScript framework, keeping the frontend simple and the server in control
   of rendering.

6. **Provider callables** — LLM functions are passed as callables rather than
   using an abstract base class, keeping the abstraction lightweight and avoiding
   class hierarchy complexity.
