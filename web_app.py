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

import io
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from knowledge_graph import (
    GraphEncoder, KnowledgeGraph, ollama_embed, local_extract,
    claude_chat, claude_extract, _get_anthropic_api_key,
)
from query_graph import ask, build_context, ollama_chat, search_nodes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
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
    # Guard against path traversal from user-supplied graph names
    try:
        json_file.resolve().relative_to(GRAPHS_DIR.resolve())
    except ValueError:
        raise ValueError(f"Invalid graph name: {name}")
    return KnowledgeGraph(json_file)


def _build_embed_fn(
    embed_model: str, embed_url: str
) -> partial:
    """Build an embedding callable for search/query."""
    return partial(ollama_embed, model=embed_model, url=embed_url)


def _build_extract_fn(provider: str, model: str, api_url: str):
    """Build an extraction callable for the given provider."""
    if provider == "anthropic":
        api_key = _get_anthropic_api_key()
        return partial(claude_extract, model=model, api_key=api_key)
    return partial(local_extract, model=model, url=api_url)


def _build_llm_fn(provider: str, model: str, api_url: str):
    """Build a chat callable for the given provider (used by 'ask' mode)."""
    if provider == "anthropic":
        api_key = _get_anthropic_api_key()
        return partial(claude_chat, model=model, api_key=api_key)
    return partial(ollama_chat, model=model, url=api_url)


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


# Temporary storage for converted documents (batch_id -> {docs, created_at})
_upload_batches: dict[str, dict[str, Any]] = {}
_BATCH_TTL_SECONDS = 1800  # 30 minutes


def _evict_stale_batches() -> None:
    """Remove upload batches older than _BATCH_TTL_SECONDS."""
    now = time.time()
    stale = [k for k, v in _upload_batches.items()
             if now - v.get("created_at", 0) > _BATCH_TTL_SECONDS]
    for k in stale:
        _upload_batches.pop(k, None)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard — list all knowledge graphs."""
    graphs = _list_graphs()
    logger.info("GET / — dashboard, %d graph(s) found", len(graphs))
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "graphs": graphs,
    })



@app.get("/graphs/{name}", response_class=HTMLResponse)
async def graph_detail(request: Request, name: str):
    """Graph detail page with Cytoscape visualization."""
    logger.info("GET /graphs/%s — detail page", name)
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
    logger.info("GET /upload — upload page")
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
    filenames = [f.filename for f in files]
    logger.info(
        "POST /upload graph=%r new_graph=%r files=%s",
        graph_name, new_graph_name, filenames,
    )
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

    _evict_stale_batches()
    batch_id = uuid.uuid4().hex[:12]
    _upload_batches[batch_id] = {"docs": batch_docs, "created_at": time.time()}
    logger.info(
        "Upload batch %s created: %d doc(s) converted for graph %r",
        batch_id, len(batch_docs), target,
    )

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
    embed_model: str = Form("qwen3-embedding"),
    provider: str = Form("local"),
    extract_model: str = Form(""),
    verbose: str = Form(""),
):
    """Run LLM ingestion with streaming progress log."""
    _verbose = verbose.strip() == "1"
    logger.info(
        "POST /ingest graph=%s batch=%s provider=%s model=%s embed=%s verbose=%s",
        graph_name, batch_id, provider,
        extract_model.strip() or query_model, embed_model, _verbose,
    )

    batch_entry = _upload_batches.pop(batch_id, None)
    batch = batch_entry["docs"] if batch_entry else None
    if not batch:
        logger.warning("Batch %s not found for graph %s", batch_id, graph_name)
        return HTMLResponse(
            json.dumps({"type": "error", "message": "Upload batch not found. Please re-upload your files."}) + "\n"
        )

    try:
        kg = _load_graph(graph_name)
    except Exception as exc:
        logger.error("Cannot load graph '%s': %s", graph_name, exc)
        return HTMLResponse(
            json.dumps({"type": "error", "message": f"Cannot load graph '{graph_name}': {exc}"}) + "\n"
        )

    def _log(msg: str) -> str:
        """Build a JSON log line and also emit to server logger."""
        logger.info("[ingest %s] %s", graph_name, msg)
        return json.dumps({"type": "log", "message": msg}) + "\n"

    def _stream():
        _model = extract_model.strip() or query_model
        extract_fn = _build_extract_fn(provider, _model, api_url)
        results = []
        total_docs = len(batch)
        graph_stats_before = kg.stats()

        if _verbose:
            yield _log(
                f"Config: provider={provider} extract_model={_model} "
                f"embed_model={embed_model} api_url={api_url} "
                f"embed_url={embed_url.strip() or api_url}"
            )
            yield _log(
                f"Graph before: {graph_stats_before.get('num_nodes', 0)} nodes, "
                f"{graph_stats_before.get('num_edges', 0)} edges"
            )

        for di, doc in enumerate(batch):
            doc_chars = len(doc.get("text", ""))
            if _verbose:
                yield _log(f"[{di + 1}/{total_docs}] Ingesting '{doc['doc_id']}' ({doc_chars} chars)...")
            else:
                yield _log(f"[{di + 1}/{total_docs}] Ingesting '{doc['doc_id']}'...")

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
                        yield _log(f"  section {idx}/{total}: {heading} ({ev.get('char_count', 0)} chars)...")
                    elif evt == "section_done":
                        elapsed = ev.get("elapsed_seconds", 0)
                        triples = ev.get("triples_processed", 0)
                        nodes = ev.get("nodes_added", 0)
                        edges = ev.get("edges_added", 0)
                        yield _log(f"    +{triples} triples, +{nodes} nodes, +{edges} edges ({elapsed}s)")
                        if _verbose:
                            nodes_skipped = ev.get("nodes_skipped", 0)
                            edges_skipped = ev.get("edges_skipped", 0)
                            proposals = ev.get("proposals_created", 0)
                            extra_parts = []
                            if nodes_skipped:
                                extra_parts.append(f"{nodes_skipped} duplicate nodes skipped")
                            if edges_skipped:
                                extra_parts.append(f"{edges_skipped} duplicate edges skipped")
                            if proposals:
                                extra_parts.append(f"{proposals} new relation proposal(s)")
                            if extra_parts:
                                yield _log(f"    ({', '.join(extra_parts)})")
                    elif evt == "section_skip":
                        yield _log(f"  section {idx}/{total}: {heading} (skipped: {ev.get('reason', '')})")

                yield _log(f"  done: {stats['total_triples']} triples, {stats['total_nodes_added']} nodes, {stats['total_edges_added']} edges")

                if _verbose:
                    yield _log(
                        f"  detail: {stats.get('total_sections', 0)} sections processed, "
                        f"{stats.get('total_nodes_skipped', 0)} nodes skipped (dup), "
                        f"{stats.get('total_edges_skipped', 0)} edges skipped (dup), "
                        f"{stats.get('total_proposals_created', 0)} proposals"
                    )

                # Auto-accept new relation proposals
                if stats.get("total_proposals_created"):
                    pending = kg.get_proposals()
                    accepted = 0
                    for p in pending:
                        kg.accept_proposal(p.name)
                        accepted += 1
                    if accepted:
                        yield _log(f"  auto-accepted {accepted} relation proposal(s)")
            except Exception as exc:
                logger.error("Ingestion error for doc '%s': %s", doc["doc_id"], exc, exc_info=_verbose)
                yield _log(f"  error: {exc}")
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
        yield _log("Embedding new nodes...")
        if _verbose:
            yield _log(f"  embed config: model={embed_model} url={_embed_url}")
        try:
            efn = _build_embed_fn(embed_model, _embed_url)
            embed_stats = kg.embed_nodes(efn, skip_existing=True, model_name=embed_model)
            embed_count = embed_stats.get("nodes_embedded", 0)
            yield _log(f"  embedded {embed_count} nodes")
            if _verbose:
                skipped = embed_stats.get("nodes_skipped", 0)
                if skipped:
                    yield _log(f"  skipped {skipped} already-embedded nodes")
        except Exception as exc:
            logger.error("Embedding failed for graph '%s': %s", graph_name, exc)
            yield _log(f"  embedding failed: {exc}")

        kg.save()
        kg.save_embeddings()
        yield _log("Graph saved.")

        if _verbose:
            graph_stats_after = kg.stats()
            yield _log(
                f"Graph after: {graph_stats_after.get('num_nodes', 0)} nodes, "
                f"{graph_stats_after.get('num_edges', 0)} edges"
            )

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
    logger.info("GET /query — query page")
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
    provider: str = Form("local"),
):
    """Execute a query against a knowledge graph."""
    logger.info(
        "POST /query graph=%s mode=%s provider=%s model=%s query=%r",
        graph_name, mode, provider, query_model, query[:80],
    )
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
        _embed_model = kg._embed_meta.get("model", "qwen3-embedding")
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
            llm_fn = _build_llm_fn(provider, query_model, api_url)
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


@app.delete("/graphs/{name}")
async def delete_graph(name: str):
    """Delete a knowledge graph and all its artifacts."""
    logger.info("DELETE /graphs/%s — deleting graph", name)
    graph_dir = GRAPHS_DIR / name
    # Guard against path traversal
    try:
        graph_dir.resolve().relative_to(GRAPHS_DIR.resolve())
    except ValueError:
        return HTMLResponse(status_code=400, content="Invalid graph name")
    if not graph_dir.is_dir():
        return HTMLResponse(status_code=404, content="Graph not found")
    shutil.rmtree(graph_dir)
    response = HTMLResponse(content="")
    response.headers["HX-Redirect"] = "/"
    return response


@app.get("/graphs/{name}/export")
async def export_graph(name: str):
    """Export a knowledge graph as a portable .zip archive."""
    logger.info("GET /graphs/%s/export — exporting graph", name)
    graph_dir = GRAPHS_DIR / name
    try:
        graph_dir.resolve().relative_to(GRAPHS_DIR.resolve())
    except ValueError:
        return HTMLResponse(status_code=400, content="Invalid graph name")
    json_file = graph_dir / f"{name}.json"
    if not json_file.exists():
        return HTMLResponse(status_code=404, content="Graph not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(graph_dir.rglob("*")):
            if not file_path.is_file():
                continue
            arcname = str(Path(name) / file_path.relative_to(graph_dir))
            if file_path == json_file:
                # Rewrite absolute paths to relative for portability
                data = json.loads(json_file.read_text(encoding="utf-8"))
                sources = data.get("meta", {}).get("sources", {})
                for entry in sources.values():
                    entry["stored_path"] = _make_relative(
                        entry.get("stored_path", ""), graph_dir
                    )
                    for ver in entry.get("versions", []):
                        if "archived_to" in ver:
                            ver["archived_to"] = _make_relative(
                                ver["archived_to"], graph_dir
                            )
                zf.writestr(arcname, json.dumps(data, indent=2, cls=GraphEncoder))
            else:
                zf.write(file_path, arcname)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}.zip"'},
    )


def _make_relative(path_str: str, base: Path) -> str:
    """Convert an absolute path to be relative to base, if it's under base."""
    if not path_str:
        return path_str
    p = Path(path_str)
    if p.is_absolute():
        try:
            return str(p.relative_to(base))
        except ValueError:
            return path_str
    return path_str


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
