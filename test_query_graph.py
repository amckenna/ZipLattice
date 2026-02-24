"""Tests for query_graph.py — all run without Ollama."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import urllib.error
from unittest.mock import patch, MagicMock

import pytest

from knowledge_graph import KnowledgeGraph, ollama_embed
from query_graph import search_nodes, build_context, ask, ollama_chat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(vec: list[float]) -> list[float]:
    """Normalize a vector to unit length."""
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


# Deterministic 4-dimensional embeddings for testing.
# Designed so "radar" is most similar to the query, then "signal", then others.
EMBEDDINGS = {
    "radar":      _unit([1.0, 0.0, 0.0, 0.0]),
    "signal":     _unit([0.8, 0.6, 0.0, 0.0]),
    "antenna":    _unit([0.5, 0.5, 0.5, 0.0]),
    "frequency":  _unit([0.3, 0.3, 0.3, 0.7]),
    "modulation": _unit([0.0, 0.0, 0.5, 0.8]),
}

# Query embedding: very close to "radar"
QUERY_EMBEDDING = _unit([0.95, 0.1, 0.0, 0.0])


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Mock embed function: returns QUERY_EMBEDDING for any input."""
    return [QUERY_EMBEDDING for _ in texts]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kg_path(tmp_path):
    """Create a small knowledge graph with nodes, edges, and embeddings."""
    path = str(tmp_path / "test_graph.json")
    kg = KnowledgeGraph(path)

    # Add nodes
    kg.add_node("radar", type="concept", label="Radar",
                properties={"description": "Radio detection and ranging"})
    kg.add_node("signal", type="concept", label="Signal Processing",
                properties={"description": "Processing of signals"})
    kg.add_node("antenna", type="component", label="Antenna",
                properties={"description": "Electromagnetic transducer"})
    kg.add_node("frequency", type="concept", label="Frequency",
                properties={"description": "Rate of oscillation"})
    kg.add_node("modulation", type="concept", label="Modulation",
                properties={"description": "Varying a carrier signal"})

    # Add edges
    kg.add_edge("radar", "signal", relation="uses")
    kg.add_edge("radar", "antenna", relation="has_component")
    kg.add_edge("signal", "frequency", relation="depends_on")
    kg.add_edge("modulation", "frequency", relation="depends_on")

    # Set embeddings
    for nid, emb in EMBEDDINGS.items():
        kg.set_embedding(nid, emb)

    kg.save_all()
    return path


@pytest.fixture
def kg(kg_path):
    """Load the test graph."""
    return KnowledgeGraph(kg_path)


# ---------------------------------------------------------------------------
# search_nodes tests
# ---------------------------------------------------------------------------


def test_search_nodes(kg):
    results = search_nodes(kg, "radar systems", fake_embed, top_k=3, expand_depth=0)
    assert len(results) <= 3
    assert results[0]["node_id"] == "radar"
    assert results[0]["similarity"] > results[1]["similarity"]


def test_search_nodes_type_filter(kg):
    results = search_nodes(kg, "radar", fake_embed, top_k=10,
                           node_types=["component"], expand_depth=0)
    for r in results:
        assert r["type"] == "component"


def test_search_nodes_expand(kg):
    results = search_nodes(kg, "radar", fake_embed, top_k=1, expand_depth=1)
    assert len(results) == 1
    assert results[0]["node_id"] == "radar"
    assert "neighbors" in results[0]
    neighbor_ids = {n["node_id"] for n in results[0]["neighbors"]}
    assert "signal" in neighbor_ids or "antenna" in neighbor_ids


def test_search_no_embeddings(tmp_path):
    """When the graph has no embeddings, search returns empty results."""
    path = str(tmp_path / "empty.json")
    kg = KnowledgeGraph(path)
    kg.add_node("orphan", type="concept", label="Orphan")
    kg.save()

    results = search_nodes(kg, "anything", fake_embed, top_k=5, expand_depth=0)
    assert results == []


# ---------------------------------------------------------------------------
# build_context tests
# ---------------------------------------------------------------------------


def test_build_context(kg):
    ctx = build_context(kg, "radar systems", fake_embed, max_nodes=5)
    assert "Radar" in ctx
    assert "Knowledge Graph Context" in ctx


def test_build_context_max_nodes(kg):
    ctx = build_context(kg, "radar", fake_embed, max_nodes=2)
    # Should still produce output but with limited nodes
    assert len(ctx) > 0
    assert "Knowledge Graph Context" in ctx


def test_build_context_no_results(tmp_path):
    path = str(tmp_path / "empty2.json")
    kg = KnowledgeGraph(path)
    kg.add_node("x", type="concept", label="X")
    kg.save()

    ctx = build_context(kg, "anything", fake_embed, max_nodes=5)
    assert "No relevant nodes" in ctx


# ---------------------------------------------------------------------------
# ask tests
# ---------------------------------------------------------------------------


def test_ask(kg):
    def mock_llm(prompt: str) -> str:
        assert "Radar" in prompt
        return "Radar uses radio waves for detection."

    answer = ask(kg, "What is radar?", fake_embed, mock_llm, max_nodes=5)
    assert answer == "Radar uses radio waves for detection."


def test_ask_includes_context(kg):
    """Verify the LLM receives context containing relevant nodes."""
    received_prompts = []

    def capture_llm(prompt: str) -> str:
        received_prompts.append(prompt)
        return "answer"

    ask(kg, "signal processing", fake_embed, capture_llm)
    assert len(received_prompts) == 1
    assert "Knowledge Graph Context" in received_prompts[0]


# ---------------------------------------------------------------------------
# ollama_embed tests
# ---------------------------------------------------------------------------


def test_ollama_embed_request():
    """Verify ollama_embed sends the correct HTTP request."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    }).encode()
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("knowledge_graph.urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        result = ollama_embed(["hello", "world"], model="test-model", url="http://localhost:11434")

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    # Check the request was made correctly
    call_args = mock_urlopen.call_args
    req = call_args[0][0]
    assert "/api/embed" in req.full_url
    body = json.loads(req.data)
    assert body["model"] == "test-model"
    assert body["input"] == ["hello", "world"]


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_cli_stats(kg_path):
    result = subprocess.run(
        [sys.executable, "query_graph.py", kg_path, "stats"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
    )
    assert result.returncode == 0
    assert "num_nodes" in result.stdout


def test_cli_node_lookup(kg_path):
    result = subprocess.run(
        [sys.executable, "query_graph.py", kg_path, "node", "radar"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
    )
    assert result.returncode == 0
    assert "Radar" in result.stdout


def test_cli_node_not_found(kg_path):
    result = subprocess.run(
        [sys.executable, "query_graph.py", kg_path, "node", "nonexistent"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
    )
    assert result.returncode == 0
    assert "not found" in result.stdout


def test_cli_json_output(kg_path):
    result = subprocess.run(
        [sys.executable, "query_graph.py", "--json", kg_path, "node", "radar"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["label"] == "Radar"


def test_cli_neighbors(kg_path):
    result = subprocess.run(
        [sys.executable, "query_graph.py", kg_path, "neighbors", "radar"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
    )
    assert result.returncode == 0
    assert "signal" in result.stdout.lower() or "antenna" in result.stdout.lower()


def test_cli_path(kg_path):
    result = subprocess.run(
        [sys.executable, "query_graph.py", kg_path, "path", "radar", "frequency"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
    )
    assert result.returncode == 0
    assert "radar" in result.stdout
    assert "frequency" in result.stdout


def test_cli_no_command(kg_path):
    result = subprocess.run(
        [sys.executable, "query_graph.py", kg_path],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
    )
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# ollama_chat tests
# ---------------------------------------------------------------------------


def _mock_urlopen_response(body_dict):
    """Create a mock urlopen response returning the given JSON body."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(body_dict).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_ollama_chat_basic():
    """Verify ollama_chat extracts response content correctly."""
    mock_resp = _mock_urlopen_response({
        "message": {"content": "Hello from Ollama"}
    })
    with patch("query_graph.urllib.request.urlopen", return_value=mock_resp):
        result = ollama_chat("test prompt", model="llama2", url="http://localhost:11434")
    assert result == "Hello from Ollama"


def test_ollama_chat_strips_whitespace():
    """Verify leading/trailing whitespace is stripped from the response."""
    mock_resp = _mock_urlopen_response({
        "message": {"content": "  spaced answer  \n"}
    })
    with patch("query_graph.urllib.request.urlopen", return_value=mock_resp):
        result = ollama_chat("test", model="m", url="http://localhost:11434")
    assert result == "spaced answer"


def test_ollama_chat_missing_content():
    """Returns empty string when content field is missing."""
    mock_resp = _mock_urlopen_response({"message": {}})
    with patch("query_graph.urllib.request.urlopen", return_value=mock_resp):
        result = ollama_chat("test", model="m", url="http://localhost:11434")
    assert result == ""


def test_ollama_chat_missing_message():
    """Returns empty string when message key doesn't exist."""
    mock_resp = _mock_urlopen_response({})
    with patch("query_graph.urllib.request.urlopen", return_value=mock_resp):
        result = ollama_chat("test", model="m", url="http://localhost:11434")
    assert result == ""


def test_ollama_chat_payload_format():
    """Verify the JSON payload sent to the server."""
    mock_resp = _mock_urlopen_response({"message": {"content": "ok"}})
    with patch("query_graph.urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        ollama_chat("test prompt", model="llama2", url="http://localhost:11434")

    req = mock_urlopen.call_args[0][0]
    assert req.full_url == "http://localhost:11434/api/chat"
    body = json.loads(req.data)
    assert body["model"] == "llama2"
    assert body["messages"] == [{"role": "user", "content": "test prompt"}]
    assert body["stream"] is False


def test_ollama_chat_trailing_slash():
    """Verify trailing slash in URL is handled correctly."""
    mock_resp = _mock_urlopen_response({"message": {"content": "ok"}})
    with patch("query_graph.urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        ollama_chat("test", model="m", url="http://localhost:11434/")

    req = mock_urlopen.call_args[0][0]
    assert req.full_url == "http://localhost:11434/api/chat"


def test_ollama_chat_http_error():
    """Verify HTTPError is wrapped in RuntimeError with helpful message."""
    exc = urllib.error.HTTPError(
        "http://localhost:11434/api/chat", 404, "Not Found", {}, None
    )
    with patch("query_graph.urllib.request.urlopen", side_effect=exc):
        with pytest.raises(RuntimeError, match="HTTP 404"):
            ollama_chat("test", model="bad-model", url="http://localhost:11434")


def test_ollama_chat_connection_error():
    """Verify URLError is wrapped in RuntimeError with helpful message."""
    exc = urllib.error.URLError("Connection refused")
    with patch("query_graph.urllib.request.urlopen", side_effect=exc):
        with pytest.raises(RuntimeError, match="Cannot connect"):
            ollama_chat("test", model="m", url="http://localhost:11434")


# ---------------------------------------------------------------------------
# Additional search_nodes edge-case tests
# ---------------------------------------------------------------------------


def test_search_nodes_nonexistent_type_filter(kg):
    """Search with a non-existent type filter returns empty results."""
    results = search_nodes(kg, "radar", fake_embed, top_k=10,
                           node_types=["nonexistent"], expand_depth=0)
    assert results == []


def test_search_nodes_top_k_exceeds_available(kg):
    """top_k larger than the graph still returns all matching nodes."""
    results = search_nodes(kg, "radar", fake_embed, top_k=1000, expand_depth=0)
    assert len(results) <= len(kg._data["nodes"])
    assert len(results) > 0


# ---------------------------------------------------------------------------
# Additional ask edge-case tests
# ---------------------------------------------------------------------------


def test_ask_empty_graph(tmp_path):
    """ask() works with an empty graph (LLM sees fallback context)."""
    path = str(tmp_path / "empty.json")
    empty_kg = KnowledgeGraph(path)
    empty_kg.add_node("x", type="concept", label="X")
    empty_kg.save()

    received = []

    def capture(prompt):
        received.append(prompt)
        return "No context available."

    answer = ask(empty_kg, "what is X?", fake_embed, capture)
    assert answer == "No context available."
    assert "No relevant nodes" in received[0]


def test_ask_llm_returns_empty(kg):
    """ask() returns empty string when LLM returns empty."""
    answer = ask(kg, "what is radar?", fake_embed, lambda _: "")
    assert answer == ""


def test_ask_llm_exception(kg):
    """ask() propagates LLM exceptions."""
    def failing_llm(prompt):
        raise RuntimeError("LLM service down")

    with pytest.raises(RuntimeError, match="LLM service down"):
        ask(kg, "question", fake_embed, failing_llm)


# ---------------------------------------------------------------------------
# Additional CLI tests
# ---------------------------------------------------------------------------


def _run_cli_with_mock_embed(kg_path, cli_args):
    """Run query_graph CLI in a subprocess with ollama_embed mocked out.

    The mock returns the QUERY_EMBEDDING for any input, matching the
    fake_embed helper used in the in-process tests.

    Args:
        kg_path: Path to the graph JSON file.
        cli_args: Full argument list as it would appear on the command
            line *after* ``query_graph.py``, including the graph path
            placeholder ``{graph}`` which will be replaced.
            Example: ``["--json", "{graph}", "search", "radar"]``
            If ``{graph}`` is absent, ``kg_path`` is prepended.
    """
    emb_json = json.dumps(QUERY_EMBEDDING)
    script = (
        "import sys, json; "
        "from unittest.mock import patch; "
        f"emb = {emb_json}; "
        "mock = lambda texts, **kw: [emb for _ in texts]; "
        "patcher = patch('query_graph.ollama_embed', side_effect=mock); "
        "patcher.start(); "
        "from query_graph import main; "
        "sys.argv = ['query_graph.py'] + sys.argv[1:]; "
        "main()"
    )
    # Replace {graph} placeholder, or prepend kg_path
    resolved = [kg_path if a == "{graph}" else a for a in cli_args]
    if kg_path not in resolved:
        resolved = [kg_path] + resolved
    cmd = [sys.executable, "-c", script] + resolved
    return subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=os.path.dirname(__file__) or ".",
    )


def test_cli_search(kg_path):
    """Verify search command produces formatted output."""
    result = _run_cli_with_mock_embed(kg_path, ["{graph}", "search", "radar"])
    assert result.returncode == 0, result.stderr
    assert "Radar" in result.stdout
    assert "1." in result.stdout


def test_cli_search_json(kg_path):
    """Verify search --json returns a JSON array of results."""
    result = _run_cli_with_mock_embed(kg_path, ["--json", "{graph}", "search", "radar"])
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "node_id" in data[0]
    assert "similarity" in data[0]


def test_cli_search_top_k(kg_path):
    """Verify --top-k limits results."""
    result = _run_cli_with_mock_embed(kg_path, ["--top-k", "2", "{graph}", "search", "radar"])
    assert result.returncode == 0, result.stderr
    # Count numbered result lines (e.g. "1. ...", "2. ...")
    numbered = [l for l in result.stdout.splitlines() if l and l[0].isdigit() and "." in l[:4]]
    assert len(numbered) <= 2


def test_cli_context(kg_path):
    """Verify context command produces formatted context block."""
    result = _run_cli_with_mock_embed(kg_path, ["{graph}", "context", "radar"])
    assert result.returncode == 0, result.stderr
    assert "Knowledge Graph Context" in result.stdout


def test_cli_context_json(kg_path):
    """Verify context --json returns structured data."""
    result = _run_cli_with_mock_embed(kg_path, ["--json", "{graph}", "context", "radar"])
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert "nodes" in data
    assert "edges" in data


def test_cli_ask_requires_query_model(kg_path):
    """Verify ask command fails gracefully without --query-model."""
    result = subprocess.run(
        [sys.executable, "query_graph.py", kg_path, "ask", "what is radar?"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
    )
    assert result.returncode == 0
    assert "Error" in result.stdout or "required" in result.stdout


def test_cli_path_no_path(kg_path):
    """Verify path command with non-existent nodes."""
    result = subprocess.run(
        [sys.executable, "query_graph.py", kg_path, "path", "nonexistent-a", "nonexistent-b"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
    )
    assert result.returncode == 0
    assert "No path" in result.stdout


def test_cli_path_json(kg_path):
    """Verify path --json returns structured data."""
    result = subprocess.run(
        [sys.executable, "query_graph.py", "--json", kg_path, "path", "radar", "frequency"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["source"] == "radar"
    assert data["target"] == "frequency"
    assert isinstance(data["path"], list)


def test_cli_neighbors_json(kg_path):
    """Verify neighbors --json returns structured data."""
    result = subprocess.run(
        [sys.executable, "query_graph.py", "--json", kg_path, "neighbors", "radar"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert len(data) > 0


def test_cli_neighbors_nonexistent(kg_path):
    """Verify neighbors handles non-existent nodes."""
    result = subprocess.run(
        [sys.executable, "query_graph.py", kg_path, "neighbors", "does-not-exist"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
    )
    assert result.returncode == 0
    assert "No neighbors" in result.stdout


def test_cli_stats_json(kg_path):
    """Verify stats --json returns valid JSON with expected keys."""
    result = subprocess.run(
        [sys.executable, "query_graph.py", "--json", kg_path, "stats"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "num_nodes" in data
    assert "num_edges" in data


def test_cli_verbose(kg_path):
    """Verify --verbose flag doesn't crash and produces debug output."""
    result = subprocess.run(
        [sys.executable, "query_graph.py", "--verbose", kg_path, "stats"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
    )
    assert result.returncode == 0


def test_cli_quiet(kg_path):
    """Verify --quiet flag suppresses info messages."""
    result = subprocess.run(
        [sys.executable, "query_graph.py", "--quiet", kg_path, "stats"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
    )
    assert result.returncode == 0
    # Quiet mode should suppress INFO-level messages from stderr
    assert "INFO" not in result.stderr


# ---------------------------------------------------------------------------
# Embed URL defaulting tests
# ---------------------------------------------------------------------------


def test_cli_embed_url_defaults_to_ollama_url(kg_path):
    """Verify --embed-url defaults to --ollama-url when not specified.

    We check this by running a search with --verbose and a custom
    --ollama-url. The embed config log line should show the custom URL.
    """
    result = subprocess.run(
        [sys.executable, "query_graph.py", "--verbose",
         "--ollama-url", "http://custom-host:9999",
         kg_path, "search", "radar"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
    )
    # The embed call will fail (no server at custom-host), but the
    # config log line should show the inherited URL.
    combined = result.stdout + result.stderr
    assert "http://custom-host:9999" in combined


def test_cli_embed_url_explicit_override(kg_path):
    """Verify --embed-url can be set independently of --ollama-url."""
    result = subprocess.run(
        [sys.executable, "query_graph.py", "--verbose",
         "--ollama-url", "http://llm-host:9999",
         "--embed-url", "http://embed-host:8888",
         kg_path, "search", "radar"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
    )
    combined = result.stdout + result.stderr
    # The embed config should show the explicit embed URL, not the LLM URL
    assert "http://embed-host:8888" in combined


# ---------------------------------------------------------------------------
# Rich context tests (edge context, section body, descriptions, boundary edges)
# ---------------------------------------------------------------------------


@pytest.fixture
def rich_kg(tmp_path):
    """Graph with edge context strings, section body text, and descriptions."""
    path = str(tmp_path / "rich_graph.json")
    kg = KnowledgeGraph(path)

    # Section node with body text
    kg.add_node("sec-polarization", type="section", label="Polarization",
                properties={
                    "body_text": "SAR sensors transmit linearly polarized waves. "
                                 "Polarization affects how radar signals interact "
                                 "with surface features.",
                    "path": ["SAR Guide", "Polarization"],
                })

    # Concept nodes with descriptions
    kg.add_node("sar", type="concept", label="Synthetic Aperture Radar",
                properties={"description": "An imaging radar using motion to synthesize a large antenna"})
    kg.add_node("polarization", type="concept", label="Polarization",
                properties={"description": "Orientation of the electromagnetic wave"})
    kg.add_node("hh-pol", type="concept", label="HH Polarization",
                properties={"description": "Horizontal transmit, horizontal receive"})

    # Node outside the search window (low similarity) to test boundary edges
    kg.add_node("vegetation", type="concept", label="Vegetation",
                properties={"description": "Plant cover on the surface"})

    # Edges with context sentences
    kg.add_edge("sar", "polarization", relation="uses",
                properties={"context": "SAR systems use polarization to distinguish surface types."})
    kg.add_edge("polarization", "hh-pol", relation="has_subtype",
                properties={"context": "HH is one of four standard polarization modes."})
    # Edge to a node that won't be in top search results
    kg.add_edge("polarization", "vegetation", relation="interacts_with",
                properties={"context": "Polarization affects backscatter from vegetation canopies."})
    # Structural edge to an external node — should be filtered from boundary edges
    kg.add_node("sar-doc", type="document", label="SAR Guide Document",
                properties={"description": "Full SAR guide"})
    kg.add_edge("sec-polarization", "sar-doc", relation="part_of")
    kg.add_edge("sar-doc", "sec-polarization", relation="contains")
    kg.set_embedding("sar-doc", _unit([0.05, 0.05, 0.05, 0.85]))

    # Embeddings: sar and sec-polarization close to query, vegetation far away
    kg.set_embedding("sec-polarization", _unit([0.9, 0.1, 0.0, 0.0]))
    kg.set_embedding("sar",              _unit([0.85, 0.15, 0.0, 0.0]))
    kg.set_embedding("polarization",     _unit([0.8, 0.2, 0.0, 0.0]))
    kg.set_embedding("hh-pol",           _unit([0.7, 0.3, 0.0, 0.0]))
    kg.set_embedding("vegetation",       _unit([0.1, 0.1, 0.8, 0.1]))

    kg.save_all()
    return kg


def test_build_context_includes_descriptions(rich_kg):
    """Verify node descriptions appear in the context output."""
    ctx = build_context(rich_kg, "SAR polarization", fake_embed, max_nodes=5)
    assert "An imaging radar using motion" in ctx
    assert "Orientation of the electromagnetic wave" in ctx


def test_build_context_includes_section_body(rich_kg):
    """Verify section body text is included and truncated."""
    ctx = build_context(rich_kg, "SAR polarization", fake_embed, max_nodes=5)
    assert "Content:" in ctx
    assert "linearly polarized waves" in ctx


def test_build_context_includes_section_body_truncated(rich_kg):
    """Verify very long body text gets truncated."""
    # Override the section body with something huge
    rich_kg._data["nodes"]["sec-polarization"]["properties"]["body_text"] = "x" * 2000
    rich_kg._rebuild_networkx()

    ctx = build_context(rich_kg, "SAR", fake_embed, max_nodes=5, max_body_chars=100)
    # Should be truncated with "..."
    assert "..." in ctx
    # Should not contain the full 2000 chars
    assert "x" * 200 not in ctx


def test_build_context_includes_edge_context(rich_kg):
    """Verify edge context sentences appear in the formatted context."""
    ctx = build_context(rich_kg, "SAR polarization", fake_embed, max_nodes=5)
    assert "SAR systems use polarization to distinguish surface types." in ctx
    assert "HH is one of four standard polarization modes." in ctx


def test_build_context_includes_boundary_edges(rich_kg):
    """Verify non-structural edges to external nodes are included."""
    # With max_nodes=4 and vegetation having low similarity, it should
    # fall outside the node set. But the edge polarization→vegetation
    # should still appear as a boundary edge (interacts_with is not structural).
    ctx = build_context(rich_kg, "SAR polarization", fake_embed, max_nodes=4)
    assert "Vegetation" in ctx
    assert "(external)" in ctx
    assert "backscatter from vegetation" in ctx


def test_build_context_boundary_edge_labels_external(rich_kg):
    """Verify boundary nodes are marked '(external)' and use labels."""
    ctx = build_context(rich_kg, "SAR polarization", fake_embed, max_nodes=4)
    # Find the boundary edge line — should use the label "Vegetation"
    for line in ctx.splitlines():
        if "Vegetation" in line and "--[" in line:
            assert "(external)" in line
            break
    else:
        pytest.fail("Expected a boundary edge line containing 'Vegetation'")


def test_build_context_filters_structural_boundary_edges(rich_kg):
    """Verify part_of/contains/documented_by boundary edges are excluded."""
    ctx = build_context(rich_kg, "SAR polarization", fake_embed, max_nodes=4)
    # The structural edges sec-polarization→sar-doc should NOT appear
    assert "part_of" not in ctx
    assert "contains" not in ctx
    assert "SAR Guide Document" not in ctx


def test_build_context_uses_labels_in_edges(rich_kg):
    """Verify edge lines use human-readable labels instead of raw IDs."""
    ctx = build_context(rich_kg, "SAR polarization", fake_embed, max_nodes=5)
    # Internal edge: sar→polarization should show labels
    assert "Synthetic Aperture Radar --[uses]--> Polarization" in ctx
    # Raw IDs should NOT appear in the Relationships section
    relationships_section = ctx.split("### Relationships")[1] if "### Relationships" in ctx else ""
    assert "sar --[" not in relationships_section
