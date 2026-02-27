# Plan: FastAPI + HTMX + Tailwind Web Frontend for ZipLattice

## Overview

Add a web frontend (`web_app.py`) that provides a minimal black-and-white UI for managing multiple knowledge graphs: uploading documents, ingesting them, querying graphs, and visualizing each graph with Cytoscape.

## Architecture

- **Single new file**: `web_app.py` — FastAPI application with Jinja2 templates
- **No build step**: Tailwind CSS via CDN, HTMX via CDN, Cytoscape.js via CDN
- **Graph storage**: A configurable `GRAPHS_DIR` directory (default `./graphs/`) where each subdirectory is a knowledge graph (using existing `KnowledgeGraph` directory conventions)
- **File conversion**: Uses existing `convert_to_markdown.convert()` for PDF/DOCX/HTML→MD
- **New dependencies**: `pip install fastapi uvicorn python-multipart jinja2`

## UI Design (Black & White, Minimal)

```
┌─────────────────────────────────────────────────────────────┐
│  ZipLattice                     Dashboard | Upload | Query  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Page content (varies by route)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

- `bg-black text-white` everywhere
- Borders in `gray-800`, muted text in `gray-400`
- Monospace font for data/code
- Buttons: white border, black fill, white text (inverted on hover)

## Pages & Routes

| Route | Method | Description |
|-------|--------|-------------|
| `GET /` | GET | Dashboard — list all graphs with stats |
| `GET /graphs/{name}` | GET | Graph detail — Cytoscape viz + stats + sources |
| `GET /upload` | GET | Upload form — select/create graph, pick files |
| `POST /upload` | POST | HTMX — convert files, store markdown |
| `POST /ingest` | POST | HTMX — run LLM ingestion on stored docs |
| `GET /query` | GET | Query form — select graph, enter question |
| `POST /query` | POST | HTMX — run search/context/ask, return results |
| `GET /health` | GET | Health check |

## Implementation Steps

### Step 1: Create `templates/` directory with Jinja2 templates

Create these template files:

- **`templates/base.html`** — Layout shell:
  - `<head>`: Tailwind CDN (`https://cdn.tailwindcss.com`), HTMX CDN (`https://unpkg.com/htmx.org`), Tailwind config for black/white theme
  - `<nav>`: "ZipLattice" brand + links (Dashboard, Upload, Query)
  - `{% block content %}` for page body
  - Minimal footer

- **`templates/dashboard.html`** — Extends base:
  - Table/grid of graphs: name, node count, edge count, source count
  - Each row links to `/graphs/{name}`
  - Empty state message if no graphs found
  - "Create New Graph" form (text input + button)

- **`templates/graph_detail.html`** — Extends base:
  - Stats cards row (nodes, edges, components, proposals)
  - Full-width Cytoscape.js `<div>` with interactive controls (search, layout selector, type/relation filters) — reuse the styling and JS from the existing `_cytoscape_html_template()` but adapted to render inline
  - Collapsible list of ingested sources/documents

- **`templates/upload.html`** — Extends base:
  - `<select>` dropdown of existing graphs
  - "Or create new:" text input
  - File input (`.pdf, .docx, .md, .html`, multiple)
  - Submit button → HTMX POST to `/upload`
  - `<div id="upload-result">` target for HTMX swap
  - After upload: show converted files + "Ingest" button with model/URL config
  - `<div id="ingest-result">` target for ingestion results

- **`templates/query.html`** — Extends base:
  - `<select>` dropdown to choose graph
  - `<textarea>` for query text
  - Radio buttons: Search / Context / Ask
  - Optional model/URL fields (collapsible)
  - Submit button → HTMX POST to `/query`
  - `<div id="query-result">` target for HTMX swap

- **`templates/partials/upload_result.html`** — HTMX fragment:
  - List of converted files with status
  - "Ingest into {graph}" button + model config fields

- **`templates/partials/ingest_result.html`** — HTMX fragment:
  - Summary: sections processed, nodes added, edges added, errors

- **`templates/partials/query_result.html`** — HTMX fragment:
  - For Search: ranked result list (label, type, similarity, confidence)
  - For Context: formatted markdown context block (in `<pre>`)
  - For Ask: LLM answer text

### Step 2: Create `web_app.py` — FastAPI application

#### App setup
- FastAPI instance with title "ZipLattice"
- `Jinja2Templates` pointing at `./templates/`
- `GRAPHS_DIR` configurable via env var `ZIPLATTICE_GRAPHS_DIR` (default `./graphs/`)
- Helper: `list_graphs()` — scan `GRAPHS_DIR` for valid graph directories (look for `*.json` graph files)
- Helper: `load_graph(name)` — instantiate `KnowledgeGraph(GRAPHS_DIR/name/name.json)`

#### `GET /` — Dashboard
- Call `list_graphs()`
- For each graph, load it and get `stats()`
- Render `dashboard.html` with graph list

#### `POST /graphs/create` — Create new graph (HTMX)
- Accept graph name from form
- Slugify the name
- Create `KnowledgeGraph(GRAPHS_DIR/slug/slug.json)` and `save()`
- Redirect to dashboard (or HTMX refresh)

#### `GET /graphs/{name}` — Graph detail
- Load graph with `load_graph(name)`
- Get stats, sources list
- Serialize nodes/edges to Cytoscape elements JSON (reuse logic from `export_cytoscape` — call internal methods to get elements list, type/relation colors, metadata)
- Render `graph_detail.html` with elements JSON embedded in a `<script>` tag

#### `GET /upload` — Upload page
- Call `list_graphs()` for the dropdown
- Render `upload.html`

#### `POST /upload` — Handle file uploads (HTMX)
- Receive: graph name (existing or new), uploaded files
- For each file:
  - If `.md`: read content directly
  - If `.pdf`, `.docx`, `.html`, `.htm`: call `convert_to_markdown.convert()` to get markdown string
  - Store the markdown text in a server-side temp area (or directly as graph source)
- Return `partials/upload_result.html` with conversion summary
- Include hidden fields with converted content for the ingest step

#### `POST /ingest` — Run LLM ingestion (HTMX)
- Receive: graph name, markdown content(s), model config (query-model, api-url, embed-model, embed-url)
- Load graph
- For each document:
  - Build `llm_extract_fn` using the configured model (same pattern as CLI in `knowledge_graph.py main()`)
  - Call `kg.ingest_markdown(text, doc_id, llm_extract_fn=...)`
- After all docs: call `kg.embed_nodes()` with configured embed model
- Call `kg.save_all()`
- Auto-export cytoscape visualization
- Return `partials/ingest_result.html` with summary

#### `GET /query` — Query page
- Call `list_graphs()` for the dropdown
- Render `query.html`

#### `POST /query` — Run query (HTMX)
- Receive: graph name, query text, mode (search/context/ask), model config
- Load graph
- Build `embed_fn` using `ollama_embed` with configured model/URL
- Based on mode:
  - **search**: Call `search_nodes(kg, query, embed_fn, top_k=10)`
  - **context**: Call `build_context(kg, query, embed_fn)`
  - **ask**: Build `llm_fn` from model config, call `ask(kg, question, embed_fn, llm_fn)`
- Return `partials/query_result.html` with results

#### `GET /health` — Health check
- Return `{"status": "ok"}`

### Step 3: Cytoscape integration in graph detail page

The existing `_cytoscape_html_template()` generates a complete standalone HTML page. For the web app, we need to extract the Cytoscape initialization JS and embed it within our template. Approach:

- Add a method `cytoscape_elements_json(self, min_confidence=0.0)` to `KnowledgeGraph` (or just use the serialization logic in `web_app.py`) that returns the elements array + metadata as a Python dict
- In `graph_detail.html`, embed this as `<script>const elements = {{ elements_json|safe }};</script>`
- Include the Cytoscape.js CDN in the template
- Include the Cytoscape initialization JS (adapted from the existing template) — same styling, same interactivity (search, filter, layout switching, detail panel)
- The Cytoscape canvas occupies the main content area (~80vh height)

### Step 4: Update `CLAUDE.md`

- Add `web_app.py` and `templates/` to project layout
- Add `fastapi`, `uvicorn`, `python-multipart`, `jinja2` to dependencies
- Add run instructions: `uvicorn web_app:app --reload`

### Step 5: Add basic tests

- Create `test_web_app.py` with FastAPI `TestClient`
- Test `GET /` returns 200
- Test `GET /health` returns OK
- Test `GET /upload` returns 200
- Test `GET /query` returns 200
- Test `POST /upload` with a small `.md` file
- Test `POST /graphs/create` creates a new graph

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `web_app.py` | **CREATE** | FastAPI application (~400-500 lines) |
| `templates/base.html` | **CREATE** | Base layout template |
| `templates/dashboard.html` | **CREATE** | Dashboard page |
| `templates/graph_detail.html` | **CREATE** | Graph detail + Cytoscape |
| `templates/upload.html` | **CREATE** | Upload form |
| `templates/query.html` | **CREATE** | Query form |
| `templates/partials/upload_result.html` | **CREATE** | HTMX partial for upload response |
| `templates/partials/query_result.html` | **CREATE** | HTMX partial for query response |
| `templates/partials/ingest_result.html` | **CREATE** | HTMX partial for ingestion response |
| `knowledge_graph.py` | **EDIT** | Add `cytoscape_elements()` method for JSON serialization |
| `test_web_app.py` | **CREATE** | Web app endpoint tests |
| `CLAUDE.md` | **EDIT** | Add web_app docs, dependencies, run instructions |

## Dependencies

```bash
pip install fastapi uvicorn python-multipart jinja2
```

CDN (no install):
- Tailwind CSS: `https://cdn.tailwindcss.com`
- HTMX: `https://unpkg.com/htmx.org`
- Cytoscape.js: `https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js`

## Key Design Decisions

1. **Templates in `templates/` directory** — Separating HTML from Python is cleaner for this amount of UI, avoids massive inline strings
2. **HTMX over JS framework** — Server-rendered partials, minimal JS, fits the project's simplicity ethos
3. **Black & white theme** — Tailwind utility classes only, no custom CSS build needed
4. **Cytoscape embedded in page** — Reuse existing serialization logic but render inside the page layout; keep the dark canvas which fits the black theme naturally
5. **Upload vs Ingest separation** — Uploading/converting is fast (no LLM needed); ingestion is slow and requires model config, so they're separate user actions
6. **No auth** — Local tool, no authentication
7. **`GRAPHS_DIR` convention** — All graphs live under one directory, each in its own subdirectory following existing `KnowledgeGraph` conventions
8. **Add `cytoscape_elements()` to KnowledgeGraph** rather than duplicating serialization logic — keeps the single source of truth in the library
