# Plan: `query_graph.py` — Knowledge Graph Query Application

## Problem

The `KnowledgeGraph` class has a full query API (search, traversal, context building) but using it requires writing Python code. There's no standalone application that wires together search, context assembly, and LLM answering into a usable tool.

Additionally, node embeddings should be generated at ingestion time (when the graph is built), not deferred to query time. The Ollama server is already available during `--ingest-md` — we should embed nodes right after extracting triples.

## Architecture Overview

Two changes:

1. **`knowledge_graph.py`** (modify) — Add an `--embed-model` flag. After triple extraction during `--ingest-md`, call `kg.embed_nodes()` using Ollama's `/api/embed` endpoint. Embeddings are stored in the graph and saved alongside it.

2. **`query_graph.py`** (new) — Query application. Node embeddings already exist in the graph. Only the user's query needs embedding at search time.

```
knowledge_graph.py       # Modified: embed nodes during ingestion
query_graph.py           # New: query application (single file)
test_query_graph.py      # New: tests (single file)
```

---

## Part 1: Embed at ingestion time (`knowledge_graph.py`)

### Changes

Add an `ollama_embed` function (shared between ingestion and query) and wire it into the ingestion CLI.

**New CLI flag:**
```
--embed-model MODEL    Ollama embedding model to use during ingestion
                       (default: nomic-embed-text). Requires --ollama.
```

**Ingestion flow becomes:**
```
--ingest-md doc.md --ollama gpt-oss --embed-model nomic-embed-text
                          │                        │
                   triple extraction         node embedding
                   (via /api/chat)          (via /api/embed)
```

After `kg.ingest_document()` finishes for each section, call `kg.embed_nodes()` with the new nodes from that ingestion. Or simpler: call it once at the end after all sections are ingested, with `skip_existing=True` so only new nodes get embedded.

**`ollama_embed` function** (defined at module level so both files can use it):
```python
def ollama_embed(texts: list[str], *, model: str, url: str) -> list[list[float]]:
    """Call Ollama /api/embed for a batch of texts."""
    payload = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        f"{url}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    return body["embeddings"]
```

This function lives in `knowledge_graph.py` so `query_graph.py` can import it.

---

## Part 2: Query application (`query_graph.py`)

### Layer 1: Embedding (query only)

Node embeddings already exist in the graph. The only embedding call at query time is for the user's query string. Uses the same `ollama_embed` from `knowledge_graph.py`.

### Layer 2: Query Functions

Three functions that compose KnowledgeGraph methods:

**`search_nodes(kg, query, embed_fn, top_k=10, node_types=None, expand_depth=1) -> list[dict]`**
- Calls `kg.search()` with the embed function
- Returns ranked results with node data, similarity scores, and neighbor context

**`build_context(kg, query, embed_fn, max_nodes=30, depth=1) -> str`**
- Calls `search_nodes` → `kg.get_context_window()`
- Formats the result as a text block suitable for pasting into an LLM prompt
- This is the "retrieval" half of RAG

**`ask(kg, question, embed_fn, llm_fn, max_nodes=30) -> str`**
- Full RAG pipeline: `build_context()` → construct prompt → `llm_fn(prompt)` → return answer
- `llm_fn` is a callable (Ollama chat, or anything else)

### Layer 3: CLI

```
python query_graph.py <graph.json> search "synthetic aperture radar"
python query_graph.py <graph.json> context "how does SAR work?"
python query_graph.py <graph.json> ask "how does SAR work?"
python query_graph.py <graph.json> node <node-id>
python query_graph.py <graph.json> neighbors <node-id> --depth 2
python query_graph.py <graph.json> path <source-id> <target-id>
python query_graph.py <graph.json> stats
```

Subcommands:
| Command | Layer | What it does |
|---------|-------|-------------|
| `search` | search_nodes | Semantic search, print ranked results |
| `context` | build_context | Search + format context block for LLM |
| `ask` | ask | Full RAG: search → context → LLM → answer |
| `node` | kg.get_node | Look up a single node |
| `neighbors` | kg.get_neighbors | Traverse neighborhood |
| `path` | kg.shortest_path | Find shortest path between two nodes |
| `stats` | kg.stats | Print graph statistics |

Shared flags:
- `--ollama MODEL` / `--ollama-url URL` — Ollama model for chat (used by `ask`)
- `--embed-model MODEL` — embedding model for query embedding (default: nomic-embed-text)
- `--top-k N` — number of results
- `--depth N` — expansion depth for search/neighbors
- `--node-types TYPE [TYPE ...]` — filter by node type
- `--verbose` / `--quiet` — logging control
- `--json` — output as JSON instead of formatted text

### Logging

Python `logging` module with a single logger `query_graph`:
- Default: `INFO` to stderr (search results, timing)
- `--verbose`: `DEBUG` (embedding calls, similarity scores, context assembly)
- `--quiet`: `WARNING` only
- Format: `%(levelname)s: %(message)s` (same style as knowledge_graph.py)

No file logging — keep it simple.

---

## Part 3: Tests (`test_query_graph.py`)

All tests run without external services (no Ollama needed).

**Strategy:**
1. **Fixture**: Create a small KnowledgeGraph in /tmp with ~10 nodes and edges, set fake embeddings
2. **Mock `ollama_embed`**: Return deterministic vectors so similarity scores are predictable
3. **Mock `llm_fn`**: Return canned answers

**Test cases:**
- `test_search_nodes` — returns results sorted by similarity, respects top_k and node_types filters
- `test_search_no_embeddings` — graceful behavior when graph has no embeddings
- `test_build_context` — produces a non-empty string containing expected node labels
- `test_build_context_max_nodes` — respects the max_nodes limit
- `test_ask` — calls llm_fn with context that includes relevant nodes, returns the answer
- `test_ollama_embed_request` — mocks urllib to verify the correct API call is made
- `test_cli_search` — subprocess call to verify CLI wiring works
- `test_cli_node_lookup` — verify node lookup output
- `test_cli_json_output` — verify --json flag produces valid JSON

Run with: `python -m pytest test_query_graph.py -v`

---

## What this plan does NOT include

- No web UI or server — CLI only
- No streaming — full response returned at once
- No conversation memory — each `ask` is stateless
- No custom re-ranking or hybrid search — uses KnowledgeGraph.search() as-is

## File changes

| File | Action |
|------|--------|
| `knowledge_graph.py` | **Edit** — add `ollama_embed()` function and `--embed-model` flag; call `kg.embed_nodes()` after ingestion |
| `query_graph.py` | **Create** — query application |
| `test_query_graph.py` | **Create** — tests |
| `CLAUDE.md` | **Edit** — add query_graph.py to project layout and "How to run" section |
