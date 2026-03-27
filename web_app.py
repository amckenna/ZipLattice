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
import threading
import zipfile
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from knowledge_graph import (
    GraphEncoder, KnowledgeGraph, ollama_embed, local_extract,
    claude_chat, claude_extract, _get_anthropic_api_key,
    bedrock_chat, bedrock_extract, bedrock_embed,
    slugify,
)
from query_graph import ask, build_context, ollama_chat, search_nodes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("web_app")

# ---------------------------------------------------------------------------
# Log-capture handler: captures knowledge_graph log records so they can be
# streamed to the web UI in verbose mode.
# ---------------------------------------------------------------------------

class _LogCaptureHandler(logging.Handler):
    """Accumulates formatted log records into a thread-safe list."""

    def __init__(self, level: int = logging.DEBUG):
        super().__init__(level)
        self._records: list[str] = []
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with self._lock:
                self._records.append(msg)
        except Exception:
            self.handleError(record)

    def drain(self) -> list[str]:
        """Return and clear all captured messages."""
        with self._lock:
            msgs = self._records[:]
            self._records.clear()
        return msgs

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
        except Exception as exc:
            logger.warning("Failed to load graph '%s': %s", entry.name, exc)
            graphs.append({"name": entry.name, "nodes": "?", "edges": "?", "sources": "?"})
    return graphs


def _load_graph(name: str) -> KnowledgeGraph:
    """Load a KnowledgeGraph by directory name."""
    json_file = GRAPHS_DIR / name / f"{name}.json"
    # Guard against path traversal from user-supplied graph names
    try:
        json_file.resolve().relative_to(GRAPHS_DIR.resolve())
    except ValueError:
        logger.warning("Path traversal attempt blocked: graph name=%r", name)
        raise ValueError(f"Invalid graph name: {name}")
    return KnowledgeGraph(json_file)


def _is_bedrock_embed_model(model: str) -> bool:
    """Check if the model name looks like a Bedrock model ID.

    Bedrock model IDs always contain a dot (e.g. ``amazon.titan-embed-text-v2:0``,
    ``cohere.embed-english-v3``), while local model names typically don't
    (e.g. ``qwen3-embedding``).
    """
    return "." in model


def _build_embed_fn(
    embed_model: str, embed_url: str, *, provider: str = "local",
    bedrock_region: str = "", bedrock_profile: str = "",
) -> partial:
    """Build an embedding callable for search/query."""
    if provider == "bedrock" and embed_model and _is_bedrock_embed_model(embed_model):
        region = bedrock_region.strip() or None
        profile = bedrock_profile.strip() or None
        return partial(bedrock_embed, model=embed_model, region=region, profile=profile)
    return partial(ollama_embed, model=embed_model, url=embed_url)


def _build_extract_fn(provider: str, model: str, api_url: str, *,
                      bedrock_region: str = "", bedrock_profile: str = ""):
    """Build an extraction callable for the given provider."""
    if provider == "anthropic":
        api_key = _get_anthropic_api_key()
        return partial(claude_extract, model=model, api_key=api_key)
    if provider == "bedrock":
        region = bedrock_region.strip() or None
        profile = bedrock_profile.strip() or None
        return partial(bedrock_extract, model=model, region=region, profile=profile)
    return partial(local_extract, model=model, url=api_url)


def _build_llm_fn(provider: str, model: str, api_url: str, *,
                  bedrock_region: str = "", bedrock_profile: str = ""):
    """Build a chat callable for the given provider (used by 'ask' mode)."""
    if provider == "anthropic":
        api_key = _get_anthropic_api_key()
        return partial(claude_chat, model=model, api_key=api_key)
    if provider == "bedrock":
        region = bedrock_region.strip() or None
        profile = bedrock_profile.strip() or None
        return partial(bedrock_chat, model=model, region=region, profile=profile)
    return partial(ollama_chat, model=model, url=api_url)


def _chat_multi_turn(
    messages: list[dict[str, str]],
    *,
    provider: str,
    model: str,
    api_url: str = "",
    bedrock_region: str = "",
    bedrock_profile: str = "",
) -> str:
    """Call a chat completions endpoint with full conversation history.

    Routes to the appropriate provider (local/anthropic/bedrock) and
    handles provider-specific message formatting.
    """
    if provider == "anthropic":
        import urllib.request
        from knowledge_graph import _ANTHROPIC_API_URL, _ANTHROPIC_VERSION

        api_key = _get_anthropic_api_key()
        # Anthropic requires alternating user/assistant messages.
        # System messages go via the ``system`` parameter.
        system_text = ""
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n"
            else:
                api_messages.append({"role": m["role"], "content": m["content"]})

        endpoint = f"{_ANTHROPIC_API_URL}/v1/messages"
        body_dict: dict[str, Any] = {
            "model": model, "max_tokens": 16384, "messages": api_messages,
        }
        if system_text.strip():
            body_dict["system"] = system_text.strip()

        payload = json.dumps(body_dict).encode()
        req = urllib.request.Request(
            endpoint, data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
            },
        )
        with urllib.request.urlopen(req, timeout=1800) as resp:
            body = json.loads(resp.read())
        return body["content"][0]["text"]

    if provider == "bedrock":
        from knowledge_graph import _get_bedrock_client

        region = bedrock_region.strip() or None
        profile = bedrock_profile.strip() or None
        client = _get_bedrock_client(region, profile=profile)

        system_parts: list[dict] = []
        api_messages_br: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append({"text": m["content"]})
            else:
                api_messages_br.append({
                    "role": m["role"],
                    "content": [{"text": m["content"]}],
                })

        kwargs: dict[str, Any] = {
            "modelId": model,
            "messages": api_messages_br,
            "inferenceConfig": {"maxTokens": 16384},
        }
        if system_parts:
            kwargs["system"] = system_parts

        resp = client.converse(**kwargs)
        return resp["output"]["message"]["content"][0]["text"].strip()

    # Default: local (OpenAI-compatible)
    import urllib.request
    endpoint = f"{api_url.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({
        "model": model, "messages": messages, "stream": False,
    }).encode()
    req = urllib.request.Request(
        endpoint, data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"]


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
        logger.error("File conversion failed for '%s': %s", filename, exc)
        return "", str(exc)
    finally:
        tmp_path.unlink(missing_ok=True)


# Temporary storage for converted documents (batch_id -> {docs, created_at})
_upload_batches: dict[str, dict[str, Any]] = {}
_BATCH_TTL_SECONDS = 1800  # 30 minutes

# Chat session storage (session_id -> {messages, config, created_at})
_chat_sessions: dict[str, dict[str, Any]] = {}
_CHAT_SESSION_TTL_SECONDS = 3600  # 1 hour


def _evict_stale_batches() -> None:
    """Remove upload batches older than _BATCH_TTL_SECONDS."""
    now = time.time()
    stale = [k for k, v in _upload_batches.items()
             if now - v.get("created_at", 0) > _BATCH_TTL_SECONDS]
    for k in stale:
        _upload_batches.pop(k, None)


def _evict_stale_chat_sessions() -> None:
    """Remove chat sessions older than _CHAT_SESSION_TTL_SECONDS."""
    now = time.time()
    stale = [k for k, v in _chat_sessions.items()
             if now - v.get("created_at", 0) > _CHAT_SESSION_TTL_SECONDS]
    for k in stale:
        _chat_sessions.pop(k, None)


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
    except Exception as exc:
        logger.error("Failed to load graph '%s': %s", name, exc)
        return RedirectResponse(url="/", status_code=303)

    cy = kg.cytoscape_elements()
    sources = kg._data.get("meta", {}).get("sources", {})

    # Compute per-source node and edge counts
    source_node_counts: dict[str, int] = {}
    source_edge_counts: dict[str, int] = {}
    for node in kg._data.get("nodes", {}).values():
        src = node.get("source", "")
        # source looks like "doc:<doc_id>" or "doc:<doc_id>::<section>"
        if src.startswith("doc:"):
            doc_key = slugify(src.split("::")[0].removeprefix("doc:"))
            source_node_counts[doc_key] = source_node_counts.get(doc_key, 0) + 1
    for edge in kg._data.get("edges", []):
        src_tag = edge.get("source_tag", "")
        if src_tag.startswith("doc:"):
            doc_key = slugify(src_tag.split("::")[0].removeprefix("doc:"))
            source_edge_counts[doc_key] = source_edge_counts.get(doc_key, 0) + 1

    # Enrich sources with counts and text availability
    sources_enriched = {}
    for doc_id, info in sources.items():
        sources_enriched[doc_id] = {
            **info,
            "node_count": source_node_counts.get(doc_id, 0),
            "edge_count": source_edge_counts.get(doc_id, 0),
            "has_text": kg.has_source(doc_id),
        }

    return templates.TemplateResponse("graph_detail.html", {
        "request": request,
        "graph_name": name,
        "stats": cy["stats"],
        "sources": sources_enriched,
        "elements_json": json.dumps(cy["elements"], cls=GraphEncoder),
        "type_colors_json": json.dumps(cy["type_colors"]),
        "relation_colors_json": json.dumps(cy["relation_colors"]),
        "stats_json": json.dumps(cy["stats"]),
        "has_positions": cy.get("has_positions", False),
    })


@app.get("/graphs/{name}/source/{doc_id}")
async def get_source_text(name: str, doc_id: str):
    """Return the stored source text for a document."""
    json_file = GRAPHS_DIR / name / f"{name}.json"
    if not json_file.exists():
        return JSONResponse({"error": "Graph not found"}, status_code=404)
    try:
        kg = _load_graph(name)
    except Exception as exc:
        logger.error("Failed to load graph '%s' for source text: %s", name, exc)
        return JSONResponse({"error": f"Cannot load graph: {exc}"}, status_code=500)
    text = kg.get_source_text(doc_id)
    if text is None:
        return JSONResponse({"error": "Source not found"}, status_code=404)
    return {"doc_id": doc_id, "text": text}


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request, graph: str = ""):
    """Upload form page."""
    logger.info("GET /upload — upload page (graph=%r)", graph)
    graphs = _list_graphs()
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "graphs": graphs,
        "selected_graph": graph.strip(),
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
    # Determine target graph ("__new__" is the sentinel from the dropdown)
    target = graph_name.strip()
    if target == "__new__":
        target = ""
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
        entry: dict[str, Any] = {
            "name": f.filename,
            "tokens": len(md_text) // 4,
            "error": err,
        }
        if not err:
            entry["doc_index"] = len(batch_docs)
            batch_docs.append({"doc_id": doc_id, "text": md_text, "filename": f.filename})
        results.append(entry)

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


@app.get("/download-markdown/{batch_id}/{doc_index}")
async def download_markdown(batch_id: str, doc_index: int):
    """Download a single converted markdown file from an upload batch."""
    batch_entry = _upload_batches.get(batch_id)
    if not batch_entry:
        return HTMLResponse("Batch not found or expired.", status_code=404)
    docs = batch_entry.get("docs", [])
    if doc_index < 0 or doc_index >= len(docs):
        return HTMLResponse("Document not found.", status_code=404)
    doc = docs[doc_index]
    stem = Path(doc["filename"]).stem
    filename = f"{stem}.md"
    return StreamingResponse(
        io.BytesIO(doc["text"].encode("utf-8")),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/download-markdown-zip/{batch_id}")
async def download_markdown_zip(batch_id: str):
    """Download all converted markdown files from an upload batch as a ZIP."""
    batch_entry = _upload_batches.get(batch_id)
    if not batch_entry:
        return HTMLResponse("Batch not found or expired.", status_code=404)
    docs = batch_entry.get("docs", [])
    if not docs:
        return HTMLResponse("No documents in batch.", status_code=404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for doc in docs:
            stem = Path(doc["filename"]).stem
            zf.writestr(f"{stem}.md", doc["text"])
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="converted_markdown.zip"'},
    )


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
    bedrock_region: str = Form(""),
    bedrock_profile: str = Form(""),
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
      # Attach a log-capture handler to surface knowledge_graph logs in verbose mode
      kg_logger = logging.getLogger("knowledge_graph")
      capture_handler: _LogCaptureHandler | None = None
      _prev_kg_level = kg_logger.level
      if _verbose:
          capture_handler = _LogCaptureHandler(level=logging.DEBUG)
          capture_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
          # Ensure the knowledge_graph logger emits DEBUG+ so the handler sees them
          kg_logger.setLevel(logging.DEBUG)
          kg_logger.addHandler(capture_handler)

      def _drain_captured():
          """Yield any captured knowledge_graph log messages."""
          if capture_handler is None:
              return
          for msg in capture_handler.drain():
              yield _log(f"  [kg] {msg}")

      try:
        _model = extract_model.strip() or query_model
        extract_fn = _build_extract_fn(provider, _model, api_url,
                                       bedrock_region=bedrock_region,
                                       bedrock_profile=bedrock_profile)
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
            doc_tokens = len(doc.get("text", "")) // 4
            if _verbose:
                yield _log(f"[{di + 1}/{total_docs}] Ingesting '{doc['doc_id']}' (~{doc_tokens:,} tokens)...")
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

                # Drain any captured knowledge_graph logs (verbose mode)
                yield from _drain_captured()

                # Replay captured progress events as log lines
                for ev in section_events:
                    evt = ev.get("event", "")
                    idx = ev.get("index", 0) + 1
                    total = ev.get("total", "?")
                    heading = ev.get("heading", "")
                    if evt == "doc_start":
                        secs = ev.get("total_sections", 0)
                        ccount = ev.get("char_count", 0)
                        yield _log(f"  Document: \"{ev.get('doc_id', '?')}\" (~{ccount // 4:,} tokens, {secs} sections)")
                    elif evt == "section_start":
                        yield _log(f"  section {idx}/{total}: {heading} (~{ev.get('char_count', 0) // 4:,} tokens)...")
                    elif evt == "extraction_done":
                        if _verbose:
                            n = ev.get("triples_returned", 0)
                            yield _log(f"    LLM returned {n} triples")
                    elif evt == "section_done":
                        elapsed = ev.get("elapsed_seconds", 0)
                        triples = ev.get("triples", 0)
                        nodes_added = ev.get("nodes_added", 0)
                        nodes_updated = ev.get("nodes_updated", 0)
                        edges_added = ev.get("edges_added", 0)
                        edges_updated = ev.get("edges_updated", 0)
                        # Build concise summary
                        n_parts = []
                        if nodes_added:
                            n_parts.append(f"+{nodes_added} new")
                        if nodes_updated:
                            n_parts.append(f"{nodes_updated} updated")
                        node_str = ", ".join(n_parts) if n_parts else "0"
                        e_parts = []
                        if edges_added:
                            e_parts.append(f"+{edges_added} new")
                        if edges_updated:
                            e_parts.append(f"{edges_updated} updated")
                        edge_str = ", ".join(e_parts) if e_parts else "0"
                        yield _log(f"    {triples} triples → {node_str} nodes, {edge_str} edges ({elapsed}s)")
                        if _verbose:
                            proposals = ev.get("proposals_created", 0)
                            augmented = ev.get("proposals_augmented", 0)
                            if proposals:
                                yield _log(f"    ({proposals} new relation proposal(s))")
                            if augmented:
                                yield _log(f"    ({augmented} existing proposal(s) augmented)")
                            errors = ev.get("errors", [])
                            for err in errors:
                                yield _log(f"    warning: {err}")
                    elif evt == "doc_done":
                        if _verbose:
                            yield _log(f"  Document complete: {ev.get('total_triples', 0)} triples, "
                                       f"{ev.get('total_nodes_added', 0)} nodes, {ev.get('total_edges_added', 0)} edges")
                    elif evt == "section_skip":
                        yield _log(f"  section {idx}/{total}: {heading} (skipped: {ev.get('reason', '')})")

                _na = stats['total_nodes_added']
                _nu = stats.get('total_nodes_updated', 0)
                _ea = stats['total_edges_added']
                _eu = stats.get('total_edges_updated', 0)
                _n_str = f"{_na} added"
                if _nu:
                    _n_str += f", {_nu} updated"
                _e_str = f"{_ea} added"
                if _eu:
                    _e_str += f", {_eu} updated"
                yield _log(f"  done: {stats['total_triples']} triples → nodes: {_n_str} | edges: {_e_str}")

                # Source version info
                src = stats.get("source")
                if src:
                    ver = src.get("version", 1)
                    if src.get("is_update"):
                        yield _log(f"  source updated to v{ver}")
                    elif src.get("is_duplicate"):
                        yield _log(f"  warning: duplicate content (matches '{src['existing_doc_id']}')")
                    elif ver > 1:
                        yield _log(f"  source unchanged (v{ver})")

                if _verbose:
                    yield _log(
                        f"  detail: {stats.get('total_sections', 0)} sections, "
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

                # Incremental save after each document so progress survives
                # client disconnects (e.g. network timeout during embedding)
                try:
                    kg.save()
                except Exception as save_exc:
                    logger.error("Incremental save failed for graph '%s': %s", graph_name, save_exc)
                    yield _log(f"  warning: incremental save failed: {save_exc}")
            except Exception as exc:
                logger.error("Ingestion error for doc '%s' (provider=%s, model=%s): %s", doc["doc_id"], provider, _model, exc, exc_info=_verbose)
                yield _log(f"  error (model={_model}): {exc}")
                results.append({
                    "doc_id": doc["doc_id"],
                    "total_sections": 0,
                    "total_triples": 0,
                    "total_nodes_added": 0,
                    "total_nodes_updated": 0,
                    "total_edges_added": 0,
                    "total_edges_updated": 0,
                    "error": str(exc),
                })

        # Embed new nodes
        _embed_url = embed_url.strip() or api_url
        embed_count = 0
        yield _log("Embedding new nodes...")
        if provider == "bedrock" and embed_model and _is_bedrock_embed_model(embed_model):
            yield _log(f"  embed config: model={embed_model} (bedrock, region={bedrock_region or 'default'}, profile={bedrock_profile or 'default'})")
        else:
            yield _log(f"  embed config: model={embed_model} url={_embed_url}")
        try:
            efn = _build_embed_fn(embed_model, _embed_url,
                                 provider=provider, bedrock_region=bedrock_region,
                                 bedrock_profile=bedrock_profile)
            embed_stats = kg.embed_nodes(efn, skip_existing=True, model_name=embed_model)
            embed_count = embed_stats.get("nodes_embedded", 0)
            # Save embeddings immediately so they survive client disconnects
            kg.save_embeddings()
            yield _log(f"  embedded {embed_count} nodes")
            if _verbose:
                skipped = embed_stats.get("nodes_skipped", 0)
                if skipped:
                    yield _log(f"  skipped {skipped} already-embedded nodes")
        except Exception as exc:
            logger.error("Embedding failed for graph '%s': %s", graph_name, exc)
            yield _log(f"  embedding failed: {exc}")
        yield from _drain_captured()

        try:
            kg.save()
            kg.save_embeddings()
            yield _log("Graph saved.")
        except Exception as exc:
            logger.error("Save failed for graph '%s': %s", graph_name, exc, exc_info=True)
            yield _log(f"  save failed: {exc}")

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
      except Exception as exc:
        logger.error(
            "Unhandled error in ingest stream (graph=%s, provider=%s, model=%s): %s",
            graph_name, provider, extract_model.strip() or query_model, exc,
            exc_info=True,
        )
        yield json.dumps({"type": "error", "message": f"Internal error: {exc}"}) + "\n"
      finally:
        if capture_handler is not None:
            kg_logger.removeHandler(capture_handler)
            kg_logger.setLevel(_prev_kg_level)

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
    response_mode: str = Form("single"),
    api_url: str = Form("http://localhost:11434"),
    query_model: str = Form("qwen3-coder:30b"),
    embed_url: str = Form(""),
    embed_model: str = Form(""),
    provider: str = Form("local"),
    bedrock_region: str = Form(""),
    bedrock_profile: str = Form(""),
):
    """Execute a query against a knowledge graph."""
    logger.info(
        "POST /query graph=%s mode=%s response_mode=%s provider=%s model=%s query=%r",
        graph_name, mode, response_mode, provider, query_model, query[:80],
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

    embed_fn = _build_embed_fn(_embed_model, _embed_url,
                               provider=provider, bedrock_region=bedrock_region,
                               bedrock_profile=bedrock_profile)

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
            llm_fn = _build_llm_fn(provider, query_model, api_url,
                                   bedrock_region=bedrock_region,
                                   bedrock_profile=bedrock_profile)
            answer = ask(kg, query, embed_fn, llm_fn)

            # If chat mode, create a session so the user can follow up
            chat_session_id = None
            if response_mode == "chat":
                _evict_stale_chat_sessions()
                # Build the RAG context that was used for this answer
                rag_context = build_context(kg, query, embed_fn)
                chat_session_id = uuid.uuid4().hex[:12]
                _chat_sessions[chat_session_id] = {
                    "messages": [
                        {"role": "system", "content": (
                            "You are a helpful assistant. Use the following knowledge graph "
                            "context to answer questions. If the context doesn't contain "
                            "enough information, say so.\n\n" + rag_context
                        )},
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": answer},
                    ],
                    "config": {
                        "provider": provider,
                        "query_model": query_model,
                        "api_url": api_url,
                        "bedrock_region": bedrock_region,
                        "bedrock_profile": bedrock_profile,
                    },
                    "created_at": time.time(),
                }
                logger.info("Chat session %s created for graph %s", chat_session_id, graph_name)

            return templates.TemplateResponse("partials/query_result.html", {
                "request": request,
                "mode": "ask",
                "answer": answer,
                "chat_session_id": chat_session_id,
            })
        else:
            return templates.TemplateResponse("partials/query_result.html", {
                "request": request,
                "error": f"Unknown mode: {mode}",
            })
    except Exception as exc:
        logger.error(
            "Query failed (graph=%s, mode=%s, provider=%s, model=%s): %s",
            graph_name, mode, provider, query_model, exc, exc_info=True,
        )
        return templates.TemplateResponse("partials/query_result.html", {
            "request": request,
            "error": str(exc),
        })


@app.post("/chat", response_class=HTMLResponse)
async def chat_follow_up(
    request: Request,
    session_id: str = Form(...),
    message: str = Form(...),
):
    """Handle a follow-up message in an active chat session."""
    logger.info("POST /chat session=%s message=%r", session_id, message[:80])

    session = _chat_sessions.get(session_id)
    if not session:
        return templates.TemplateResponse("partials/chat_message.html", {
            "request": request,
            "error": "Chat session expired. Please start a new query.",
        })

    # Append user message
    session["messages"].append({"role": "user", "content": message})

    cfg = session["config"]
    provider = cfg["provider"]
    model = cfg["query_model"]
    api_url = cfg["api_url"]
    bedrock_region = cfg.get("bedrock_region", "")
    bedrock_profile = cfg.get("bedrock_profile", "")

    try:
        answer = _chat_multi_turn(
            session["messages"],
            provider=provider, model=model, api_url=api_url,
            bedrock_region=bedrock_region, bedrock_profile=bedrock_profile,
        )
    except Exception as exc:
        # Remove the failed user message so they can retry
        session["messages"].pop()
        logger.error("Chat error (session=%s, provider=%s, model=%s): %s", session_id, provider, model, exc, exc_info=True)
        return templates.TemplateResponse("partials/chat_message.html", {
            "request": request,
            "error": str(exc),
        })

    # Append assistant response
    session["messages"].append({"role": "assistant", "content": answer})
    session["created_at"] = time.time()  # refresh TTL

    return templates.TemplateResponse("partials/chat_message.html", {
        "request": request,
        "user_message": message,
        "answer": answer,
        "session_id": session_id,
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
        logger.warning("Path traversal attempt blocked in delete: name=%r", name)
        return HTMLResponse(status_code=400, content="Invalid graph name")
    if not graph_dir.is_dir():
        return HTMLResponse(status_code=404, content="Graph not found")
    try:
        shutil.rmtree(graph_dir)
    except Exception as exc:
        logger.error("Failed to delete graph '%s': %s", name, exc)
        return HTMLResponse(status_code=500, content=f"Delete failed: {exc}")
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
        logger.warning("Path traversal attempt blocked in export: name=%r", name)
        return HTMLResponse(status_code=400, content="Invalid graph name")
    json_file = graph_dir / f"{name}.json"
    if not json_file.exists():
        return HTMLResponse(status_code=404, content="Graph not found")

    try:
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
    except Exception as exc:
        logger.error("Export failed for graph '%s': %s", name, exc, exc_info=True)
        return HTMLResponse(status_code=500, content=f"Export failed: {exc}")

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


@app.post("/import", response_class=HTMLResponse)
async def import_graph(
    request: Request,
    file: UploadFile = File(...),
):
    """Import a knowledge graph from a previously exported .zip archive."""
    filename = file.filename or "upload.zip"
    logger.info("POST /import file=%s", filename)

    if not filename.lower().endswith(".zip"):
        return templates.TemplateResponse("partials/import_result.html", {
            "request": request,
            "error": "Please upload a .zip file (exported from ZipLattice).",
        })

    content = await file.read()
    if not content:
        return templates.TemplateResponse("partials/import_result.html", {
            "request": request,
            "error": "Uploaded file is empty.",
        })

    try:
        buf = io.BytesIO(content)
        with zipfile.ZipFile(buf, "r") as zf:
            names = zf.namelist()
            if not names:
                return templates.TemplateResponse("partials/import_result.html", {
                    "request": request,
                    "error": "The zip archive is empty.",
                })

            # Determine the graph name from the top-level directory.
            # Export format: <name>/<name>.json plus other files under <name>/
            top_dirs = {n.split("/")[0] for n in names if "/" in n}
            if len(top_dirs) != 1:
                return templates.TemplateResponse("partials/import_result.html", {
                    "request": request,
                    "error": "Expected a single graph directory in the archive.",
                })

            graph_name = top_dirs.pop()
            # Validate the graph name is safe
            safe_name = _slugify(graph_name)
            if not safe_name:
                return templates.TemplateResponse("partials/import_result.html", {
                    "request": request,
                    "error": "Could not determine a valid graph name from the archive.",
                })

            # Check that the archive contains the expected JSON file
            json_arcname = f"{graph_name}/{graph_name}.json"
            if json_arcname not in names:
                return templates.TemplateResponse("partials/import_result.html", {
                    "request": request,
                    "error": f"Archive missing expected graph file: {json_arcname}",
                })

            # Guard against overwriting an existing graph
            target_dir = GRAPHS_DIR / safe_name
            if target_dir.exists():
                return templates.TemplateResponse("partials/import_result.html", {
                    "request": request,
                    "error": f"A graph named '{safe_name}' already exists. Delete it first or rename.",
                })

            # Validate all paths are safe (no path traversal)
            GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
            for member in names:
                dest = (GRAPHS_DIR / member).resolve()
                if not str(dest).startswith(str(GRAPHS_DIR.resolve())):
                    return templates.TemplateResponse("partials/import_result.html", {
                        "request": request,
                        "error": "Archive contains unsafe file paths.",
                    })

            # Extract — remap from archive name to safe_name if they differ
            for member in names:
                # Skip directory entries
                if member.endswith("/"):
                    continue
                parts = member.split("/", 1)
                if len(parts) < 2:
                    continue
                relative = parts[1]
                # Remap filenames that embed the graph name
                if graph_name != safe_name:
                    relative = relative.replace(graph_name, safe_name)
                dest = GRAPHS_DIR / safe_name / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(member))

            # If the archive graph name differs from the slugified name,
            # rename the main JSON file
            expected_json = GRAPHS_DIR / safe_name / f"{safe_name}.json"
            if not expected_json.exists():
                # Try the original name
                original_json = GRAPHS_DIR / safe_name / f"{graph_name}.json"
                if original_json.exists():
                    original_json.rename(expected_json)

            # Verify the graph loads
            kg = KnowledgeGraph(expected_json)
            st = kg.stats()

    except zipfile.BadZipFile:
        return templates.TemplateResponse("partials/import_result.html", {
            "request": request,
            "error": "The uploaded file is not a valid zip archive.",
        })
    except Exception as exc:
        logger.error("Import failed: %s", exc, exc_info=True)
        # Clean up partial extraction
        target_dir = GRAPHS_DIR / safe_name
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        return templates.TemplateResponse("partials/import_result.html", {
            "request": request,
            "error": f"Import failed: {exc}",
        })

    logger.info(
        "Imported graph '%s': %d nodes, %d edges",
        safe_name, st.get("num_nodes", 0), st.get("num_edges", 0),
    )
    return templates.TemplateResponse("partials/import_result.html", {
        "request": request,
        "graph_name": safe_name,
        "stats": st,
    })


@app.get("/merge", response_class=HTMLResponse)
async def merge_form(request: Request):
    """Show the merge graph form with checkboxes for each graph."""
    graphs = _list_graphs()
    return templates.TemplateResponse("merge.html", {
        "request": request,
        "graphs": graphs,
    })


@app.post("/merge", response_class=HTMLResponse)
async def merge_graphs_route(
    request: Request,
    graph_names: list[str] = Form(...),
    output_name: str = Form(...),
    strategy: str = Form("latest"),
):
    """Merge selected graphs into a new graph."""
    logger.info("POST /merge graphs=%s output=%s strategy=%s",
                graph_names, output_name, strategy)

    # Validate output name
    safe_name = slugify(output_name)
    if not safe_name:
        return templates.TemplateResponse("partials/merge_result.html", {
            "request": request,
            "error": "Invalid output graph name.",
        })

    if len(graph_names) < 2:
        return templates.TemplateResponse("partials/merge_result.html", {
            "request": request,
            "error": "Select at least two graphs to merge.",
        })

    # Check that output doesn't already exist
    output_dir = GRAPHS_DIR / safe_name
    if output_dir.exists():
        return templates.TemplateResponse("partials/merge_result.html", {
            "request": request,
            "error": f"A graph named '{safe_name}' already exists.",
        })

    if strategy not in ("latest", "first", "last"):
        strategy = "latest"

    try:
        source_kgs = [_load_graph(name) for name in graph_names]
        output_path = GRAPHS_DIR / safe_name / f"{safe_name}.json"
        merged = KnowledgeGraph.merge_graphs(
            source_kgs, output_path, prefer=strategy,
        )
        st = merged.stats()
    except Exception as exc:
        logger.error("Merge failed: %s", exc)
        return templates.TemplateResponse("partials/merge_result.html", {
            "request": request,
            "error": f"Merge failed: {exc}",
        })

    logger.info("Merged %d graphs → '%s': %d nodes, %d edges",
                len(graph_names), safe_name,
                st.get("num_nodes", 0), st.get("num_edges", 0))
    return templates.TemplateResponse("partials/merge_result.html", {
        "request": request,
        "graph_name": safe_name,
        "stats": st,
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
