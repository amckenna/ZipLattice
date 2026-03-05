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

from web_app import app, GRAPHS_DIR  # noqa: E402


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


def test_create_graph():
    resp = client.post("/graphs/create", data={"name": "Test Graph"}, follow_redirects=False)
    assert resp.status_code == 200
    assert resp.headers.get("hx-redirect") == "/"

    # Verify it appears on dashboard
    resp = client.get("/")
    assert resp.status_code == 200
    assert "test-graph" in resp.text


def test_create_graph_slug():
    """Graph names are slugified."""
    client.post("/graphs/create", data={"name": "My Cool Project!!!"}, follow_redirects=False)
    graph_dir = GRAPHS_DIR / "my-cool-project"
    assert graph_dir.exists()
    assert (graph_dir / "my-cool-project.json").exists()


# ---------------------------------------------------------------------------
# Graph detail page
# ---------------------------------------------------------------------------


def test_graph_detail_page():
    client.post("/graphs/create", data={"name": "detail-test"}, follow_redirects=False)
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
    client.post("/graphs/create", data={"name": "upload-test"}, follow_redirects=False)

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
# Dashboard lists graphs
# ---------------------------------------------------------------------------


def test_dashboard_lists_multiple_graphs():
    client.post("/graphs/create", data={"name": "alpha"}, follow_redirects=False)
    client.post("/graphs/create", data={"name": "beta"}, follow_redirects=False)
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
            "query_model": "claude-haiku-4-5-20251001",
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
        fn = _build_extract_fn("anthropic", "claude-haiku-4-5-20251001", "http://localhost:11434")
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
        fn = _build_llm_fn("anthropic", "claude-haiku-4-5-20251001", "http://localhost:11434")
    assert callable(fn)
