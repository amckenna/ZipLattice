"""Tests for query_graph.py — all run without Ollama."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from knowledge_graph import KnowledgeGraph, ollama_embed
from query_graph import search_nodes, build_context, ask


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
        [sys.executable, "query_graph.py", kg_path, "node", "radar", "--json"],
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
