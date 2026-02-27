"""ZipLattice web frontend — FastAPI + HTMX + Tailwind CSS.

A minimal black-and-white web UI for managing multiple knowledge graphs:
  - Dashboard listing all graphs with stats
  - Per-graph detail page with interactive Cytoscape.js visualization
  - Document upload with automatic PDF/DOCX/HTML→Markdown conversion
  - LLM-powered ingestion into any graph
  - Semantic search, RAG context, and RAG ask queries

Run with::

    uvicorn web_app:app --reload
    # or
    python web_app.py          # starts on http://localhost:8000
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from knowledge_graph import GraphEncoder, KnowledgeGraph, _strip_thinking, ollama_embed
from query_graph import ask, build_context, ollama_chat, search_nodes

logger = logging.getLogger("web_app")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRAPHS_DIR = Path(os.environ.get("ZIPLATTICE_GRAPHS_DIR", "./graphs"))
UPLOAD_TMP = Path(tempfile.gettempdir()) / "ziplattice_uploads"

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="ZipLattice")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """Convert a graph name to a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "untitled"


def _list_graphs() -> list[dict[str, Any]]:
    """Scan GRAPHS_DIR and return metadata for every graph found."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    graphs: list[dict[str, Any]] = []
    for entry in sorted(GRAPHS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        json_file = entry / f"{entry.name}.json"
        if not json_file.exists():
            continue
        try:
            kg = KnowledgeGraph(json_file)
            st = kg.stats()
            sources = kg._data.get("meta", {}).get("sources", {})
            graphs.append({
                "name": entry.name,
                "nodes": st.get("num_nodes", 0),
                "edges": st.get("num_edges", 0),
                "sources": len(sources),
            })
        except Exception:
            graphs.append({"name": entry.name, "nodes": "?", "edges": "?", "sources": "?"})
    return graphs


def _load_graph(name: str) -> KnowledgeGraph:
    """Load a KnowledgeGraph by directory name."""
    json_file = GRAPHS_DIR / name / f"{name}.json"
    return KnowledgeGraph(json_file)


def _build_embed_fn(
    embed_model: str, embed_url: str
) -> partial:
    """Build an embedding callable for search/query."""
    return partial(ollama_embed, model=embed_model, url=embed_url)


def _build_llm_extract_fn(model: str, url: str):
    """Build an LLM extraction callable matching knowledge_graph.py's pattern."""
    import urllib.request

    def _extract(prompt: str) -> list[dict[str, Any]]:
        payload = json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a JSON extraction engine. "
                        "Respond with ONLY a valid JSON array. "
                        "No explanations, no markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": 0.1,
            "max_tokens": 32768,
        }).encode()
        req = urllib.request.Request(
            f"{url.rstrip('/')}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=1200) as resp:
                body = json.loads(resp.read())
        except Exception as exc:
            logger.error("Extraction request failed: %s", exc)
            return []

        raw = body["choices"][0]["message"]["content"].strip()
        raw = _strip_thinking(raw)
        if not raw:
            return []

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None

        if parsed is None:
            start = raw.find("[")
            if start != -1:
                end = raw.rfind("]")
                if end > start:
                    try:
                        parsed = json.loads(raw[start : end + 1])
                    except json.JSONDecodeError:
                        pass

        if parsed is None:
            return []
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
            return []
        if isinstance(parsed, list):
            return parsed
        return []

    return _extract


def _convert_file(filename: str, content: bytes) -> tuple[str, str | None]:
    """Convert an uploaded file to markdown. Returns (markdown, error)."""
    ext = Path(filename).suffix.lower()
    if ext == ".md":
        return content.decode("utf-8", errors="replace"), None

    try:
        from convert_to_markdown import convert
    except ImportError:
        return "", "convert_to_markdown not available (install pymupdf4llm, mammoth, markdownify)"

    # Write to temp file for conversion
    tmp_path = UPLOAD_TMP / f"{uuid.uuid4().hex}{ext}"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(content)
    try:
        md = convert(tmp_path)
        return md, None
    except Exception as exc:
        return "", str(exc)
    finally:
        tmp_path.unlink(missing_ok=True)


# Temporary storage for converted documents (batch_id -> list of docs)
_upload_batches: dict[str, list[dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard — list all knowledge graphs."""
    graphs = _list_graphs()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "graphs": graphs,
    })


@app.post("/graphs/create", response_class=HTMLResponse)
async def create_graph(request: Request, name: str = Form(...)):
    """Create a new empty knowledge graph."""
    slug = _slugify(name)
    graph_dir = GRAPHS_DIR / slug
    graph_dir.mkdir(parents=True, exist_ok=True)
    kg = KnowledgeGraph(graph_dir / f"{slug}.json")
    kg.save()
    response = HTMLResponse(content="")
    response.headers["HX-Redirect"] = "/"
    return response


@app.get("/graphs/{name}", response_class=HTMLResponse)
async def graph_detail(request: Request, name: str):
    """Graph detail page with Cytoscape visualization."""
    json_file = GRAPHS_DIR / name / f"{name}.json"
    if not json_file.exists():
        return RedirectResponse(url="/", status_code=303)
    try:
        kg = _load_graph(name)
    except Exception:
        return RedirectResponse(url="/", status_code=303)

    cy = kg.cytoscape_elements()
    sources = kg._data.get("meta", {}).get("sources", {})

    return templates.TemplateResponse("graph_detail.html", {
        "request": request,
        "graph_name": name,
        "stats": cy["stats"],
        "sources": sources,
        "elements_json": json.dumps(cy["elements"], cls=GraphEncoder),
        "type_colors_json": json.dumps(cy["type_colors"]),
        "relation_colors_json": json.dumps(cy["relation_colors"]),
        "stats_json": json.dumps(cy["stats"]),
    })


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Upload form page."""
    graphs = _list_graphs()
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "graphs": graphs,
    })


@app.post("/upload", response_class=HTMLResponse)
async def upload_files(
    request: Request,
    graph_name: str = Form(""),
    new_graph_name: str = Form(""),
    files: list[UploadFile] = File(...),
):
    """Handle file uploads — convert to markdown."""
    # Determine target graph
    target = graph_name.strip()
    if new_graph_name.strip():
        target = _slugify(new_graph_name.strip())
        # Create the graph if it doesn't exist
        graph_dir = GRAPHS_DIR / target
        graph_dir.mkdir(parents=True, exist_ok=True)
        kg = KnowledgeGraph(graph_dir / f"{target}.json")
        kg.save()

    if not target:
        return templates.TemplateResponse("partials/upload_result.html", {
            "request": request,
            "error": "Please select an existing graph or enter a name for a new one.",
        })

    results = []
    batch_docs = []
    for f in files:
        content = await f.read()
        if not content:
            continue
        md_text, err = _convert_file(f.filename or "unknown", content)
        doc_id = Path(f.filename or "unknown").stem
        results.append({
            "name": f.filename,
            "chars": len(md_text),
            "error": err,
        })
        if not err:
            batch_docs.append({"doc_id": doc_id, "text": md_text, "filename": f.filename})

    batch_id = uuid.uuid4().hex[:12]
    _upload_batches[batch_id] = batch_docs

    return templates.TemplateResponse("partials/upload_result.html", {
        "request": request,
        "graph_name": target,
        "files": results,
        "batch_id": batch_id,
    })


@app.post("/ingest")
async def ingest_documents(
    request: Request,
    graph_name: str = Form(...),
    batch_id: str = Form(...),
    api_url: str = Form("http://localhost:11434"),
    query_model: str = Form("qwen3-coder:30b"),
    embed_url: str = Form(""),
    embed_model: str = Form("nomic-embed-text"),
):
    """Run LLM ingestion with streaming progress log."""
    batch = _upload_batches.pop(batch_id, None)
    if not batch:
        return HTMLResponse(
            json.dumps({"type": "error", "message": "Upload batch not found. Please re-upload your files."}) + "\n"
        )

    try:
        kg = _load_graph(graph_name)
    except Exception as exc:
        return HTMLResponse(
            json.dumps({"type": "error", "message": f"Cannot load graph '{graph_name}': {exc}"}) + "\n"
        )

    def _stream():
        extract_fn = _build_llm_extract_fn(query_model, api_url)
        results = []
        total_docs = len(batch)

        for di, doc in enumerate(batch):
            yield json.dumps({"type": "log", "message": f"[{di + 1}/{total_docs}] Ingesting '{doc['doc_id']}'..."}) + "\n"

            def _progress(event):
                pass  # progress events are yielded via the section callbacks below

            try:
                section_events: list[dict] = []

                def _capture_progress(event: dict):
                    section_events.append(event)

                stats = kg.ingest_markdown(
                    doc["text"],
                    doc["doc_id"],
                    llm_extract_fn=extract_fn,
                    original_path=doc.get("filename"),
                    progress_fn=_capture_progress,
                )
                results.append(stats)

                # Replay captured progress events as log lines
                for ev in section_events:
                    evt = ev.get("event", "")
                    idx = ev.get("index", 0) + 1
                    total = ev.get("total", "?")
                    heading = ev.get("heading", "")
                    if evt == "section_start":
                        yield json.dumps({"type": "log", "message": f"  section {idx}/{total}: {heading} ({ev.get('char_count', 0)} chars)..."}) + "\n"
                    elif evt == "section_done":
                        elapsed = ev.get("elapsed_seconds", 0)
                        triples = ev.get("triples_processed", 0)
                        nodes = ev.get("nodes_added", 0)
                        edges = ev.get("edges_added", 0)
                        yield json.dumps({"type": "log", "message": f"    +{triples} triples, +{nodes} nodes, +{edges} edges ({elapsed}s)"}) + "\n"
                    elif evt == "section_skip":
                        yield json.dumps({"type": "log", "message": f"  section {idx}/{total}: {heading} (skipped: {ev.get('reason', '')})"}) + "\n"

                yield json.dumps({"type": "log", "message": f"  done: {stats['total_triples']} triples, {stats['total_nodes_added']} nodes, {stats['total_edges_added']} edges"}) + "\n"

                # Auto-accept new relation proposals
                if stats.get("total_proposals_created"):
                    pending = kg.get_proposals()
                    accepted = 0
                    for p in pending:
                        kg.accept_proposal(p.name)
                        accepted += 1
                    if accepted:
                        yield json.dumps({"type": "log", "message": f"  auto-accepted {accepted} relation proposal(s)"}) + "\n"
            except Exception as exc:
                yield json.dumps({"type": "log", "message": f"  error: {exc}"}) + "\n"
                results.append({
                    "doc_id": doc["doc_id"],
                    "total_sections": 0,
                    "total_triples": 0,
                    "total_nodes_added": 0,
                    "total_edges_added": 0,
                    "error": str(exc),
                })

        # Embed new nodes
        _embed_url = embed_url.strip() or api_url
        embed_count = 0
        yield json.dumps({"type": "log", "message": "Embedding new nodes..."}) + "\n"
        try:
            efn = _build_embed_fn(embed_model, _embed_url)
            embed_stats = kg.embed_nodes(efn, skip_existing=True, model_name=embed_model)
            embed_count = embed_stats.get("nodes_embedded", 0)
            yield json.dumps({"type": "log", "message": f"  embedded {embed_count} nodes"}) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "log", "message": f"  embedding failed: {exc}"}) + "\n"

        kg.save()
        kg.save_embeddings()
        yield json.dumps({"type": "log", "message": "Graph saved."}) + "\n"

        # Render final result HTML
        tpl = templates.env.get_template("partials/ingest_result.html")
        html = tpl.render(graph_name=graph_name, results=results, embed_count=embed_count)
        yield json.dumps({"type": "done", "html": html}) + "\n"

    return StreamingResponse(
        _stream(),
        media_type="text/plain",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.get("/query", response_class=HTMLResponse)
async def query_page(request: Request):
    """Query form page."""
    graphs = _list_graphs()
    return templates.TemplateResponse("query.html", {
        "request": request,
        "graphs": graphs,
    })


@app.post("/query", response_class=HTMLResponse)
async def run_query(
    request: Request,
    graph_name: str = Form(...),
    query: str = Form(...),
    mode: str = Form("search"),
    api_url: str = Form("http://localhost:11434"),
    query_model: str = Form("qwen3-coder:30b"),
    embed_url: str = Form(""),
    embed_model: str = Form(""),
):
    """Execute a query against a knowledge graph."""
    try:
        kg = _load_graph(graph_name)
    except Exception as exc:
        return templates.TemplateResponse("partials/query_result.html", {
            "request": request,
            "error": f"Cannot load graph '{graph_name}': {exc}",
        })

    # Determine embed model
    _embed_model = embed_model.strip()
    if not _embed_model:
        _embed_model = kg._embed_meta.get("model", "nomic-embed-text")
    _embed_url = embed_url.strip() or api_url

    embed_fn = _build_embed_fn(_embed_model, _embed_url)

    try:
        if mode == "search":
            results = search_nodes(kg, query, embed_fn, top_k=10)
            return templates.TemplateResponse("partials/query_result.html", {
                "request": request,
                "mode": "search",
                "results": results,
                "graph_name": graph_name,
            })
        elif mode == "context":
            ctx = build_context(kg, query, embed_fn)
            return templates.TemplateResponse("partials/query_result.html", {
                "request": request,
                "mode": "context",
                "context": ctx,
            })
        elif mode == "ask":
            llm_fn = partial(ollama_chat, model=query_model, url=api_url)
            answer = ask(kg, query, embed_fn, llm_fn)
            return templates.TemplateResponse("partials/query_result.html", {
                "request": request,
                "mode": "ask",
                "answer": answer,
            })
        else:
            return templates.TemplateResponse("partials/query_result.html", {
                "request": request,
                "error": f"Unknown mode: {mode}",
            })
    except Exception as exc:
        return templates.TemplateResponse("partials/query_result.html", {
            "request": request,
            "error": str(exc),
        })


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=True)
