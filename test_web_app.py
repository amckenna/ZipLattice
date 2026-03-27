"""Tests for the ZipLattice web frontend (web_app.py).

Run with::

    python -m pytest test_web_app.py -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Point GRAPHS_DIR to a temp directory before importing the app
_tmp_dir = tempfile.mkdtemp(prefix="ziplattice_test_")
os.environ["ZIPLATTICE_GRAPHS_DIR"] = _tmp_dir

from knowledge_graph import KnowledgeGraph  # noqa: E402
from web_app import app, GRAPHS_DIR, _upload_batches  # noqa: E402


def _create_graph(name: str) -> None:
    """Create a graph directly on disk (no route needed)."""
    graph_dir = GRAPHS_DIR / name
    graph_dir.mkdir(parents=True, exist_ok=True)
    kg = KnowledgeGraph(graph_dir / f"{name}.json")
    kg.save()


@pytest.fixture(autouse=True)
def _clean_graphs_dir():
    """Ensure each test starts with a clean graphs directory."""
    import shutil

    # Clean before test
    for item in GRAPHS_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    yield
    # Clean after test
    for item in GRAPHS_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


client = TestClient(app)


# ---------------------------------------------------------------------------
# Basic page tests
# ---------------------------------------------------------------------------


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_dashboard_empty():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "ZipLattice" in resp.text
    assert "No knowledge graphs found" in resp.text


def test_upload_page():
    resp = client.get("/upload")
    assert resp.status_code == 200
    assert "Upload" in resp.text


def test_query_page():
    resp = client.get("/query")
    assert resp.status_code == 200
    assert "Query" in resp.text


# ---------------------------------------------------------------------------
# Graph creation
# ---------------------------------------------------------------------------


def test_create_graph_via_upload():
    """Graphs are created through the upload page with new_graph_name."""
    md_content = b"# Test\n\nContent.\n"
    resp = client.post(
        "/upload",
        data={"graph_name": "", "new_graph_name": "Test Graph"},
        files=[("files", ("doc.md", md_content, "text/markdown"))],
    )
    assert resp.status_code == 200
    assert (GRAPHS_DIR / "test-graph").exists()

    # Verify it appears on dashboard
    resp = client.get("/")
    assert resp.status_code == 200
    assert "test-graph" in resp.text


# ---------------------------------------------------------------------------
# Graph detail page
# ---------------------------------------------------------------------------


def test_graph_detail_page():
    _create_graph("detail-test")
    resp = client.get("/graphs/detail-test")
    assert resp.status_code == 200
    assert "detail-test" in resp.text
    assert "cytoscape" in resp.text.lower()


def test_graph_detail_nonexistent():
    resp = client.get("/graphs/does-not-exist", follow_redirects=False)
    assert resp.status_code == 303  # redirect to dashboard


# ---------------------------------------------------------------------------
# Upload flow
# ---------------------------------------------------------------------------


def test_upload_md_file():
    """Upload a .md file and verify it gets processed."""
    _create_graph("upload-test")

    md_content = b"# Test\n\nThis is a test document.\n"
    resp = client.post(
        "/upload",
        data={"graph_name": "upload-test", "new_graph_name": ""},
        files=[("files", ("test.md", md_content, "text/markdown"))],
    )
    assert resp.status_code == 200
    assert "test.md" in resp.text
    assert "OK" in resp.text


def test_upload_creates_new_graph():
    """Upload with a new graph name creates the graph."""
    md_content = b"# Doc\n\nContent here.\n"
    resp = client.post(
        "/upload",
        data={"graph_name": "", "new_graph_name": "Brand New Graph"},
        files=[("files", ("doc.md", md_content, "text/markdown"))],
    )
    assert resp.status_code == 200
    assert "brand-new-graph" in resp.text
    assert (GRAPHS_DIR / "brand-new-graph").exists()


def test_upload_no_graph_selected():
    """Upload without selecting a graph shows error."""
    md_content = b"# Doc\n\nContent.\n"
    resp = client.post(
        "/upload",
        data={"graph_name": "", "new_graph_name": ""},
        files=[("files", ("doc.md", md_content, "text/markdown"))],
    )
    assert resp.status_code == 200
    assert "Error" in resp.text or "select" in resp.text.lower()


# ---------------------------------------------------------------------------
# Query (without LLM — tests the form rendering and error handling)
# ---------------------------------------------------------------------------


def test_query_no_graph():
    """Query with invalid graph returns error."""
    resp = client.post(
        "/query",
        data={
            "graph_name": "nonexistent",
            "query": "test query",
            "mode": "search",
            "api_url": "http://localhost:11434",
            "query_model": "test",
            "embed_url": "",
            "embed_model": "",
        },
    )
    assert resp.status_code == 200
    assert "Error" in resp.text or "Cannot load" in resp.text


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_graph():
    """Export a graph as a zip archive."""
    import zipfile as zf
    _create_graph("export-test")
    resp = client.get("/graphs/export-test/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert "export-test.zip" in resp.headers.get("content-disposition", "")

    buf = __import__("io").BytesIO(resp.content)
    with zf.ZipFile(buf) as z:
        names = z.namelist()
        assert any("export-test.json" in n for n in names)


def test_export_nonexistent():
    """Export of a nonexistent graph returns 404."""
    resp = client.get("/graphs/nonexistent/export")
    assert resp.status_code == 404


def test_export_rewrites_absolute_paths():
    """Absolute archived_to paths are made relative in the export."""
    import zipfile as zf
    _create_graph("path-test")
    # Inject an absolute path into the graph's source metadata
    graph_dir = GRAPHS_DIR / "path-test"
    json_file = graph_dir / "path-test.json"
    import json
    data = json.loads(json_file.read_text())
    data.setdefault("meta", {})["sources"] = {
        "doc1": {
            "stored_path": str(graph_dir / "path-test_sources" / "abc_doc1.md"),
            "versions": [
                {"archived_to": str(graph_dir / "path-test_sources" / "archive" / "v1_abc_doc1.md")}
            ],
        }
    }
    json_file.write_text(json.dumps(data))

    resp = client.get("/graphs/path-test/export")
    assert resp.status_code == 200
    buf = __import__("io").BytesIO(resp.content)
    with zf.ZipFile(buf) as z:
        exported = json.loads(z.read("path-test/path-test.json"))
    src = exported["meta"]["sources"]["doc1"]
    # Paths should now be relative (no leading /)
    assert not Path(src["stored_path"]).is_absolute()
    assert not Path(src["versions"][0]["archived_to"]).is_absolute()


# ---------------------------------------------------------------------------
# Dashboard lists graphs
# ---------------------------------------------------------------------------


def test_dashboard_lists_multiple_graphs():
    _create_graph("alpha")
    _create_graph("beta")
    resp = client.get("/")
    assert resp.status_code == 200
    assert "alpha" in resp.text
    assert "beta" in resp.text


# ---------------------------------------------------------------------------
# Provider parameter tests
# ---------------------------------------------------------------------------


def test_query_accepts_provider_param():
    """Query endpoint accepts the provider parameter without error."""
    resp = client.post(
        "/query",
        data={
            "graph_name": "nonexistent",
            "query": "test",
            "mode": "search",
            "api_url": "http://localhost:11434",
            "query_model": "test",
            "embed_url": "",
            "embed_model": "",
            "provider": "local",
        },
    )
    assert resp.status_code == 200


def test_query_accepts_anthropic_provider():
    """Query endpoint accepts provider=anthropic without a validation error."""
    resp = client.post(
        "/query",
        data={
            "graph_name": "nonexistent",
            "query": "test",
            "mode": "search",
            "api_url": "http://localhost:11434",
            "query_model": "claude-haiku-4-5",
            "embed_url": "",
            "embed_model": "",
            "provider": "anthropic",
        },
    )
    assert resp.status_code == 200
    # Should show an error (graph not found or embed connection), not a provider validation error
    assert "Error" in resp.text


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------


def test_build_extract_fn_local():
    """_build_extract_fn returns a callable for local provider."""
    from web_app import _build_extract_fn
    fn = _build_extract_fn("local", "test-model", "http://localhost:11434")
    assert callable(fn)


def test_build_extract_fn_anthropic():
    """_build_extract_fn returns a callable for anthropic provider."""
    import os
    from unittest.mock import patch
    from web_app import _build_extract_fn
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        fn = _build_extract_fn("anthropic", "claude-haiku-4-5", "http://localhost:11434")
    assert callable(fn)


def test_build_llm_fn_local():
    """_build_llm_fn returns a callable for local provider."""
    from web_app import _build_llm_fn
    fn = _build_llm_fn("local", "test-model", "http://localhost:11434")
    assert callable(fn)


def test_build_llm_fn_anthropic():
    """_build_llm_fn returns a callable for anthropic provider."""
    import os
    from unittest.mock import patch
    from web_app import _build_llm_fn
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        fn = _build_llm_fn("anthropic", "claude-haiku-4-5", "http://localhost:11434")
    assert callable(fn)


# ---------------------------------------------------------------------------
# Bedrock provider tests
# ---------------------------------------------------------------------------


def test_query_accepts_bedrock_provider():
    """Query endpoint accepts provider=bedrock without a validation error."""
    resp = client.post(
        "/query",
        data={
            "graph_name": "nonexistent",
            "query": "test",
            "mode": "search",
            "api_url": "http://localhost:11434",
            "query_model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "embed_url": "",
            "embed_model": "",
            "provider": "bedrock",
            "bedrock_region": "us-east-1",
        },
    )
    assert resp.status_code == 200
    # Should show an error (graph not found or embed connection), not a provider validation error
    assert "Error" in resp.text


def test_build_extract_fn_bedrock():
    """_build_extract_fn returns a callable for bedrock provider."""
    from web_app import _build_extract_fn
    fn = _build_extract_fn("bedrock", "us.anthropic.claude-sonnet-4-20250514-v1:0",
                           "http://localhost:11434", bedrock_region="us-west-2")
    assert callable(fn)


def test_build_extract_fn_bedrock_with_profile():
    """_build_extract_fn accepts bedrock_profile parameter."""
    from web_app import _build_extract_fn
    fn = _build_extract_fn("bedrock", "us.anthropic.claude-sonnet-4-20250514-v1:0",
                           "http://localhost:11434", bedrock_region="us-west-2",
                           bedrock_profile="my-profile")
    assert callable(fn)


def test_build_llm_fn_bedrock():
    """_build_llm_fn returns a callable for bedrock provider."""
    from web_app import _build_llm_fn
    fn = _build_llm_fn("bedrock", "us.anthropic.claude-sonnet-4-20250514-v1:0",
                        "http://localhost:11434", bedrock_region="us-west-2")
    assert callable(fn)


def test_build_llm_fn_bedrock_with_profile():
    """_build_llm_fn accepts bedrock_profile parameter."""
    from web_app import _build_llm_fn
    fn = _build_llm_fn("bedrock", "us.anthropic.claude-sonnet-4-20250514-v1:0",
                        "http://localhost:11434", bedrock_region="us-west-2",
                        bedrock_profile="my-profile")
    assert callable(fn)


def test_build_embed_fn_bedrock():
    """_build_embed_fn returns a Bedrock embed callable for Bedrock model IDs."""
    from web_app import _build_embed_fn
    fn = _build_embed_fn("amazon.titan-embed-text-v2:0", "http://localhost:11434",
                         provider="bedrock", bedrock_region="us-east-1")
    assert callable(fn)


def test_build_embed_fn_bedrock_with_profile():
    """_build_embed_fn accepts bedrock_profile parameter."""
    from web_app import _build_embed_fn
    fn = _build_embed_fn("amazon.titan-embed-text-v2:0", "http://localhost:11434",
                         provider="bedrock", bedrock_region="us-east-1",
                         bedrock_profile="my-profile")
    assert callable(fn)


def test_build_embed_fn_bedrock_local_fallback():
    """_build_embed_fn falls back to local embed for non-Bedrock model names."""
    from web_app import _build_embed_fn
    fn = _build_embed_fn("qwen3-embedding", "http://localhost:11434",
                         provider="bedrock", bedrock_region="us-east-1")
    assert callable(fn)


def test_ingest_accepts_bedrock_region():
    """Ingest endpoint accepts the bedrock_region parameter."""
    _create_graph("bedrock-ingest-test")
    # Upload a doc first to get a batch_id
    md_content = b"# Test\n\nBedrock test.\n"
    resp = client.post(
        "/upload",
        data={"graph_name": "bedrock-ingest-test", "new_graph_name": ""},
        files=[("files", ("test.md", md_content, "text/markdown"))],
    )
    assert resp.status_code == 200
    # Extract batch_id from response
    import re
    match = re.search(r'name="batch_id"\s+value="([^"]+)"', resp.text)
    assert match, "batch_id not found in upload response"
    batch_id = match.group(1)

    # Try ingest with bedrock provider — it will fail at the LLM call (no AWS creds)
    # but should NOT fail at parameter parsing
    resp = client.post(
        "/ingest",
        data={
            "graph_name": "bedrock-ingest-test",
            "batch_id": batch_id,
            "api_url": "http://localhost:11434",
            "query_model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "embed_url": "",
            "embed_model": "qwen3-embedding",
            "provider": "bedrock",
            "extract_model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "bedrock_region": "us-west-2",
            "bedrock_profile": "my-profile",
            "verbose": "",
        },
    )
    # Should return 200 (streaming response), the actual LLM error is in the stream
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def test_import_graph():
    """Import a previously exported graph from a zip archive."""
    import io
    import zipfile as zf

    _create_graph("to-export")
    # Add a node so we can verify round-trip
    kg = KnowledgeGraph(GRAPHS_DIR / "to-export" / "to-export.json")
    kg.add_node("test-node", type="concept", label="Test Node")
    kg.save()

    # Export it
    resp = client.get("/graphs/to-export/export")
    assert resp.status_code == 200
    export_data = resp.content

    # Delete the original so we can re-import
    import shutil
    shutil.rmtree(GRAPHS_DIR / "to-export")

    # Import
    resp = client.post(
        "/import",
        files=[("file", ("to-export.zip", export_data, "application/zip"))],
    )
    assert resp.status_code == 200
    assert "to-export" in resp.text
    assert "successfully" in resp.text.lower() or "Imported" in resp.text

    # Verify the graph exists and has the node
    assert (GRAPHS_DIR / "to-export" / "to-export.json").exists()
    kg2 = KnowledgeGraph(GRAPHS_DIR / "to-export" / "to-export.json")
    assert kg2.get_node("test-node") is not None


def test_import_rejects_non_zip():
    """Import rejects non-zip files."""
    resp = client.post(
        "/import",
        files=[("file", ("graph.txt", b"not a zip", "text/plain"))],
    )
    assert resp.status_code == 200
    assert "Error" in resp.text or ".zip" in resp.text


def test_import_rejects_invalid_zip():
    """Import rejects corrupt zip data."""
    resp = client.post(
        "/import",
        files=[("file", ("graph.zip", b"not a zip", "application/zip"))],
    )
    assert resp.status_code == 200
    assert "Error" in resp.text


def test_import_rejects_duplicate():
    """Import rejects if a graph with the same name already exists."""
    import io
    import zipfile as zf

    _create_graph("dup-test")

    # Build a minimal zip
    buf = io.BytesIO()
    with zf.ZipFile(buf, "w") as z:
        z.writestr("dup-test/dup-test.json", '{"nodes": {}, "edges": [], "meta": {}}')
    zip_data = buf.getvalue()

    resp = client.post(
        "/import",
        files=[("file", ("dup-test.zip", zip_data, "application/zip"))],
    )
    assert resp.status_code == 200
    assert "already exists" in resp.text


def test_import_rejects_missing_json():
    """Import rejects archives without the expected graph JSON."""
    import io
    import zipfile as zf

    buf = io.BytesIO()
    with zf.ZipFile(buf, "w") as z:
        z.writestr("my-graph/random.txt", "hello")
    zip_data = buf.getvalue()

    resp = client.post(
        "/import",
        files=[("file", ("my-graph.zip", zip_data, "application/zip"))],
    )
    assert resp.status_code == 200
    assert "Error" in resp.text or "missing" in resp.text.lower()


def test_query_bedrock_with_profile():
    """Query endpoint accepts bedrock_profile parameter."""
    resp = client.post(
        "/query",
        data={
            "graph_name": "nonexistent",
            "query": "test",
            "mode": "search",
            "api_url": "http://localhost:11434",
            "query_model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "embed_url": "",
            "embed_model": "",
            "provider": "bedrock",
            "bedrock_region": "us-east-1",
            "bedrock_profile": "my-profile",
        },
    )
    assert resp.status_code == 200
    assert "Error" in resp.text


# ---------------------------------------------------------------------------
# Merge routes
# ---------------------------------------------------------------------------


def test_merge_page():
    resp = client.get("/merge")
    assert resp.status_code == 200
    assert "Merge" in resp.text


def test_merge_graphs_success():
    """POST /merge with two graphs produces a merged graph."""
    for name in ("src1", "src2"):
        graph_dir = GRAPHS_DIR / name
        graph_dir.mkdir(parents=True, exist_ok=True)
        kg = KnowledgeGraph(graph_dir / f"{name}.json")
        kg.add_node(f"node-{name}", type="concept", label=f"Node {name}")
        kg.save()

    resp = client.post(
        "/merge",
        data={
            "graph_names": ["src1", "src2"],
            "output_name": "merged-test",
            "strategy": "latest",
        },
    )
    assert resp.status_code == 200
    assert "merged-test" in resp.text
    assert "successfully" in resp.text
    assert (GRAPHS_DIR / "merged-test" / "merged-test.json").exists()


def test_merge_graphs_too_few():
    """POST /merge with fewer than 2 graphs returns an error."""
    _create_graph("only-one")
    resp = client.post(
        "/merge",
        data={
            "graph_names": ["only-one"],
            "output_name": "bad-merge",
            "strategy": "latest",
        },
    )
    assert resp.status_code == 200
    assert "Error" in resp.text


def test_merge_graphs_existing_name():
    """POST /merge with an existing output name returns an error."""
    _create_graph("existing")
    _create_graph("other")
    resp = client.post(
        "/merge",
        data={
            "graph_names": ["existing", "other"],
            "output_name": "existing",
            "strategy": "latest",
        },
    )
    assert resp.status_code == 200
    assert "already exists" in resp.text


# ---------------------------------------------------------------------------
# Markdown download tests
# ---------------------------------------------------------------------------


def _seed_batch(batch_id: str, docs: list[dict]) -> None:
    """Seed _upload_batches with test data."""
    import time
    _upload_batches[batch_id] = {"docs": docs, "created_at": time.time()}


def test_download_markdown_single():
    """GET /download-markdown/<batch>/<index> returns the markdown file."""
    _seed_batch("testbatch1", [
        {"doc_id": "readme", "text": "# Hello\nWorld", "filename": "readme.pdf"},
    ])
    resp = client.get("/download-markdown/testbatch1/0")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    assert "readme.md" in resp.headers["content-disposition"]
    assert resp.text == "# Hello\nWorld"
    _upload_batches.pop("testbatch1", None)


def test_download_markdown_not_found():
    """GET /download-markdown with invalid batch returns 404."""
    resp = client.get("/download-markdown/nonexistent/0")
    assert resp.status_code == 404


def test_download_markdown_bad_index():
    """GET /download-markdown with out-of-range index returns 404."""
    _seed_batch("testbatch2", [
        {"doc_id": "doc", "text": "content", "filename": "doc.md"},
    ])
    resp = client.get("/download-markdown/testbatch2/5")
    assert resp.status_code == 404
    _upload_batches.pop("testbatch2", None)


def test_download_markdown_zip():
    """GET /download-markdown-zip/<batch> returns a ZIP with all markdown files."""
    import zipfile as zf
    import io
    _seed_batch("testbatch3", [
        {"doc_id": "a", "text": "AAA", "filename": "a.pdf"},
        {"doc_id": "b", "text": "BBB", "filename": "b.docx"},
    ])
    resp = client.get("/download-markdown-zip/testbatch3")
    assert resp.status_code == 200
    assert "application/zip" in resp.headers["content-type"]
    with zf.ZipFile(io.BytesIO(resp.content)) as z:
        names = sorted(z.namelist())
        assert names == ["a.md", "b.md"]
        assert z.read("a.md").decode() == "AAA"
        assert z.read("b.md").decode() == "BBB"
    _upload_batches.pop("testbatch3", None)


def test_download_markdown_zip_not_found():
    """GET /download-markdown-zip with invalid batch returns 404."""
    resp = client.get("/download-markdown-zip/nonexistent")
    assert resp.status_code == 404
