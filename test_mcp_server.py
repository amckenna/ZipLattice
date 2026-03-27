"""Tests for mcp_server.py and the ingest_triples refactoring."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from knowledge_graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_graph(tmp_path):
    """Create a fresh KnowledgeGraph in a temp directory."""
    return KnowledgeGraph(tmp_path / "test.json")


# ---------------------------------------------------------------------------
# ingest_triples — the public method extracted from ingest_document
# ---------------------------------------------------------------------------


class TestIngestTriples:
    """Tests for KnowledgeGraph.ingest_triples (orchestrator-as-extractor)."""

    def test_basic_triples(self, tmp_graph):
        """Standard dict triples are ingested via ingest_triples."""
        triples = [
            {
                "source": "Radar",
                "target": "Radio waves",
                "relation": "uses",
                "confidence": 0.9,
                "context": "Radar uses radio waves.",
            },
        ]
        stats = tmp_graph.ingest_triples(
            triples,
            text="Radar uses radio waves.",
            doc_id="test-doc",
        )
        assert stats["triples_processed"] == 1
        assert stats["nodes_added"] >= 2
        assert stats["edges_added"] >= 1
        assert not stats["errors"]

    def test_list_format_triples(self, tmp_graph):
        """List-format triples [source, relation, target] are accepted."""
        triples = [["Radar", "uses", "Radio waves"]]
        stats = tmp_graph.ingest_triples(
            triples,
            text="Radar uses radio waves.",
            doc_id="test-doc",
        )
        assert stats["triples_processed"] == 1
        assert stats["nodes_added"] >= 2

    def test_key_aliases(self, tmp_graph):
        """Triples with alias keys (subject/object/predicate) are normalized."""
        triples = [
            {
                "subject": "Radar",
                "object": "Radio waves",
                "predicate": "uses",
                "confidence": 0.9,
            },
        ]
        stats = tmp_graph.ingest_triples(
            triples,
            text="Radar uses radio waves.",
            doc_id="test-doc",
        )
        assert stats["triples_processed"] == 1
        assert stats["nodes_added"] >= 2

    def test_hallucination_detection(self, tmp_graph):
        """Triples with entities not grounded in source text are rejected."""
        triples = [
            {
                "source": "TotallyFakeEntityXYZ",
                "target": "AnotherFakeEntityABC",
                "relation": "uses",
                "confidence": 0.9,
            },
            {
                "source": "BogusThingOne",
                "target": "BogusThingTwo",
                "relation": "depends_on",
                "confidence": 0.9,
            },
        ]
        stats = tmp_graph.ingest_triples(
            triples,
            text="Radar uses radio waves for detection.",
            doc_id="test-doc",
        )
        assert stats["nodes_added"] == 0
        assert any("hallucinated" in str(e).lower() for e in stats["errors"])

    def test_off_topic_detection(self, tmp_graph):
        """Triples with unrecognized keys are rejected as off-topic."""
        triples = [{"foo": "bar", "baz": "qux"}]
        stats = tmp_graph.ingest_triples(
            triples,
            text="Radar uses radio waves.",
            doc_id="test-doc",
        )
        assert stats["nodes_added"] == 0
        assert any("recognized" in str(e).lower() for e in stats["errors"])

    def test_auto_add_doc_node(self, tmp_graph):
        """A document node is created and linked when auto_add_doc_node=True."""
        triples = [
            {
                "source": "Radar",
                "target": "Radio waves",
                "relation": "uses",
                "confidence": 0.9,
                "context": "Radar uses radio waves.",
            },
        ]
        tmp_graph.ingest_triples(
            triples,
            text="Radar uses radio waves.",
            doc_id="test-doc",
            auto_add_doc_node=True,
        )
        assert tmp_graph.has_node("test-doc")
        # Check documented_by edges exist
        edges = tmp_graph.get_edges("test-doc", direction="incoming")
        assert any(e["relation"] == "documented_by" for e in edges)

    def test_no_doc_node(self, tmp_graph):
        """No document node when auto_add_doc_node=False."""
        triples = [
            {
                "source": "Radar",
                "target": "Radio waves",
                "relation": "uses",
                "confidence": 0.9,
                "context": "Radar uses radio waves.",
            },
        ]
        tmp_graph.ingest_triples(
            triples,
            text="Radar uses radio waves.",
            doc_id="my-doc",
            auto_add_doc_node=False,
        )
        assert not tmp_graph.has_node("my-doc")

    def test_novel_relation_creates_proposal(self, tmp_graph):
        """Triples with is_new_relation=True create relation proposals."""
        triples = [
            {
                "source": "Radar",
                "target": "Weather",
                "relation": "monitors",
                "is_new_relation": True,
                "suggested_relation": "monitors",
                "justification": "Tracks weather patterns",
                "confidence": 0.8,
                "context": "Radar monitors weather.",
            },
        ]
        stats = tmp_graph.ingest_triples(
            triples,
            text="Radar monitors weather patterns.",
            doc_id="test-doc",
        )
        assert stats["proposals_created"] == 1
        proposals = tmp_graph.get_proposals(status="pending")
        assert any(p.name == "monitors" for p in proposals)

    def test_ingest_document_delegates_to_ingest_triples(self, tmp_graph):
        """ingest_document still works by delegating to ingest_triples."""
        triples = [
            {
                "source": "Radar",
                "target": "Radio waves",
                "relation": "uses",
                "confidence": 0.9,
                "context": "Radar uses radio waves.",
            },
        ]
        stats = tmp_graph.ingest_document(
            "Radar uses radio waves.",
            doc_id="test-doc",
            llm_extract_fn=lambda _: triples,
        )
        assert stats["triples_processed"] == 1
        assert stats["nodes_added"] >= 2
        assert stats["edges_added"] >= 1

    def test_empty_triples(self, tmp_graph):
        """Empty triples list produces zero stats with no errors."""
        stats = tmp_graph.ingest_triples(
            [],
            text="Some text.",
            doc_id="test-doc",
        )
        assert stats["triples_processed"] == 0
        assert stats["nodes_added"] == 0
        assert not stats["errors"]

    def test_ingestion_id_propagated(self, tmp_graph):
        """ingestion_id is propagated to created nodes."""
        triples = [
            {
                "source": "Radar",
                "target": "Radio waves",
                "relation": "uses",
                "confidence": 0.9,
                "context": "Radar uses radio waves.",
            },
        ]
        tmp_graph.ingest_triples(
            triples,
            text="Radar uses radio waves.",
            doc_id="test-doc",
            ingestion_id="ing-123",
        )
        node = tmp_graph.get_node("radar")
        assert node is not None
        assert node["properties"].get("ingestion_id") == "ing-123"


# ---------------------------------------------------------------------------
# MCP server tool tests (unit tests using direct function calls)
# ---------------------------------------------------------------------------


class TestMCPServerTools:
    """Test MCP server tool functions directly (no transport layer)."""

    def test_build_extraction_prompt(self, tmp_path):
        """build_extraction_prompt returns a schema-aware prompt."""
        from mcp_server import build_extraction_prompt
        graph_path = str(tmp_path / "test.json")
        prompt = build_extraction_prompt(
            graph_path=graph_path,
            text="Radar uses radio waves for detection.",
        )
        assert "source" in prompt
        assert "relation" in prompt
        assert "JSON" in prompt

    def test_ingest_triples_tool(self, tmp_path):
        """ingest_triples MCP tool ingests triples and returns stats."""
        from mcp_server import ingest_triples as mcp_ingest, _graph_cache
        graph_path = str(tmp_path / "test.json")
        _graph_cache.clear()

        result = mcp_ingest(
            graph_path=graph_path,
            triples=[
                {
                    "source": "Radar",
                    "target": "Radio waves",
                    "relation": "uses",
                    "confidence": 0.9,
                    "context": "Radar uses radio waves.",
                },
            ],
            text="Radar uses radio waves.",
            doc_id="test-doc",
        )
        stats = json.loads(result)
        assert stats["nodes_added"] >= 2
        assert stats["edges_added"] >= 1

    def test_add_get_node_tool(self, tmp_path):
        """add_node and get_node MCP tools work end-to-end."""
        from mcp_server import add_node, get_node, _graph_cache
        graph_path = str(tmp_path / "test.json")
        _graph_cache.clear()

        result = json.loads(add_node(
            graph_path=graph_path,
            node_id="radar",
            type="technology",
            label="Radar",
        ))
        assert result["node_id"] == "radar"

        node_result = json.loads(get_node(graph_path=graph_path, node_id="radar"))
        assert node_result["label"] == "Radar"
        assert node_result["type"] == "technology"

    def test_add_edge_tool(self, tmp_path):
        """add_edge MCP tool creates an edge between nodes."""
        from mcp_server import add_node, add_edge, get_edges, _graph_cache
        graph_path = str(tmp_path / "test.json")
        _graph_cache.clear()

        add_node(graph_path=graph_path, node_id="a", label="A")
        add_node(graph_path=graph_path, node_id="b", label="B")
        edge_result = json.loads(add_edge(
            graph_path=graph_path,
            source="a", target="b", relation="uses",
        ))
        assert edge_result["relation"] == "uses"

        edges = json.loads(get_edges(graph_path=graph_path, node_id="a"))
        assert len(edges) >= 1

    def test_graph_stats_tool(self, tmp_path):
        """graph_stats returns summary statistics."""
        from mcp_server import add_node, graph_stats, _graph_cache
        graph_path = str(tmp_path / "test.json")
        _graph_cache.clear()

        add_node(graph_path=graph_path, node_id="x", label="X")
        stats = json.loads(graph_stats(graph_path=graph_path))
        assert stats["num_nodes"] == 1

    def test_search_nodes_tool(self, tmp_path):
        """search_nodes MCP tool filters nodes."""
        from mcp_server import add_node, search_nodes, _graph_cache
        graph_path = str(tmp_path / "test.json")
        _graph_cache.clear()

        add_node(graph_path=graph_path, node_id="radar", type="technology", label="Radar System")
        add_node(graph_path=graph_path, node_id="lidar", type="technology", label="Lidar System")
        add_node(graph_path=graph_path, node_id="python", type="tool", label="Python")

        results = json.loads(search_nodes(
            graph_path=graph_path, type="technology",
        ))
        assert len(results) == 2

        results = json.loads(search_nodes(
            graph_path=graph_path, label_contains="radar",
        ))
        assert len(results) == 1

    def test_get_neighbors_tool(self, tmp_path):
        """get_neighbors returns connected nodes."""
        from mcp_server import add_node, add_edge, get_neighbors, _graph_cache
        graph_path = str(tmp_path / "test.json")
        _graph_cache.clear()

        add_node(graph_path=graph_path, node_id="a", label="A")
        add_node(graph_path=graph_path, node_id="b", label="B")
        add_node(graph_path=graph_path, node_id="c", label="C")
        add_edge(graph_path=graph_path, source="a", target="b", relation="uses")
        add_edge(graph_path=graph_path, source="b", target="c", relation="uses")

        neighbors = json.loads(get_neighbors(
            graph_path=graph_path, node_id="a", max_depth=1,
        ))
        assert len(neighbors) == 1
        assert neighbors[0]["node_id"] == "b"

        neighbors_2 = json.loads(get_neighbors(
            graph_path=graph_path, node_id="a", max_depth=2,
        ))
        assert len(neighbors_2) == 2

    def test_remove_node_tool(self, tmp_path):
        """remove_node removes the node."""
        from mcp_server import add_node, remove_node, get_node, _graph_cache
        graph_path = str(tmp_path / "test.json")
        _graph_cache.clear()

        add_node(graph_path=graph_path, node_id="x", label="X")
        result = json.loads(remove_node(graph_path=graph_path, node_id="x"))
        assert result["removed"] is True

        node = json.loads(get_node(graph_path=graph_path, node_id="x"))
        assert "error" in node

    def test_proposal_lifecycle(self, tmp_path):
        """list/accept/reject proposals work end-to-end."""
        from mcp_server import (
            ingest_triples as mcp_ingest, list_proposals,
            accept_proposal, reject_proposal, _graph_cache,
        )
        graph_path = str(tmp_path / "test.json")
        _graph_cache.clear()

        mcp_ingest(
            graph_path=graph_path,
            triples=[
                {
                    "source": "Radar",
                    "target": "Weather",
                    "relation": "monitors",
                    "is_new_relation": True,
                    "suggested_relation": "monitors",
                    "justification": "Tracks weather",
                    "confidence": 0.8,
                    "context": "Radar monitors weather.",
                },
            ],
            text="Radar monitors weather patterns.",
            doc_id="test-doc",
        )

        proposals = json.loads(list_proposals(graph_path=graph_path))
        assert len(proposals) >= 1
        assert any(p["name"] == "monitors" for p in proposals)

        result = json.loads(accept_proposal(
            graph_path=graph_path, name="monitors",
        ))
        assert result["accepted"] is True

    def test_parse_markdown_sections_tool(self):
        """parse_markdown_sections returns section list."""
        from mcp_server import parse_markdown_sections

        md = (
            "# Introduction\n\n"
            "This is a substantial introduction section with enough text to "
            "exceed the minimum section character threshold for parsing. It "
            "discusses important concepts in detail.\n\n"
            "## Technical Details\n\n"
            "This section covers the technical details of the system, "
            "including architecture decisions, implementation choices, "
            "and performance considerations that are relevant."
        )
        result = json.loads(parse_markdown_sections(text=md))
        assert isinstance(result, list)
        assert len(result) >= 2

    def test_store_source_tool(self, tmp_path):
        """store_source stores document text."""
        from mcp_server import store_source, _graph_cache
        graph_path = str(tmp_path / "test.json")
        _graph_cache.clear()

        result = json.loads(store_source(
            graph_path=graph_path,
            text="Hello world document.",
            doc_id="hello",
        ))
        assert result["content_hash"]
        assert result["stored_path"]

    def test_full_orchestrator_workflow(self, tmp_path):
        """End-to-end: build prompt → extract (mock) → ingest triples."""
        from mcp_server import (
            build_extraction_prompt, ingest_triples as mcp_ingest,
            graph_stats, _graph_cache,
        )
        graph_path = str(tmp_path / "test.json")
        _graph_cache.clear()

        doc_text = (
            "TensorFlow is a machine learning framework developed by Google. "
            "It supports GPU acceleration and distributed training."
        )

        # Step 1: Get the extraction prompt
        prompt = build_extraction_prompt(
            graph_path=graph_path, text=doc_text,
        )
        assert "source" in prompt
        assert "relation" in prompt

        # Step 2: Simulate orchestrator extraction (in real use, the
        # orchestrator LLM would produce these from the prompt + text)
        triples = [
            {
                "source": "TensorFlow",
                "source_type": "tool",
                "target": "Google",
                "target_type": "organization",
                "relation": "created_by",
                "confidence": 0.95,
                "context": "TensorFlow is a machine learning framework developed by Google.",
            },
            {
                "source": "TensorFlow",
                "source_type": "tool",
                "target": "GPU acceleration",
                "target_type": "concept",
                "relation": "supports",
                "confidence": 0.9,
                "context": "It supports GPU acceleration.",
            },
        ]

        # Step 3: Ingest the triples
        result = json.loads(mcp_ingest(
            graph_path=graph_path,
            triples=triples,
            text=doc_text,
            doc_id="tensorflow-doc",
        ))
        assert result["nodes_added"] >= 3  # TensorFlow, Google, GPU acceleration
        assert result["edges_added"] >= 2

        # Step 4: Verify graph state
        stats = json.loads(graph_stats(graph_path=graph_path))
        assert stats["num_nodes"] >= 3
        assert stats["num_edges"] >= 2


# ---------------------------------------------------------------------------
# merge_graphs MCP tool
# ---------------------------------------------------------------------------


class TestMergeGraphsTool:
    """Tests for the merge_graphs MCP tool."""

    def test_merge_two_graphs(self, tmp_path):
        """merge_graphs creates a combined graph from two sources."""
        from mcp_server import merge_graphs, _graph_cache

        # Create two source graphs
        kg1 = KnowledgeGraph(tmp_path / "mcp1.json")
        kg1.add_node("a", type="concept", label="Alpha")
        kg1.save()

        kg2 = KnowledgeGraph(tmp_path / "mcp2.json")
        kg2.add_node("b", type="concept", label="Beta")
        kg2.save()

        output = str(tmp_path / "mcp_merged.json")
        result = json.loads(merge_graphs(
            graph_paths=[str(kg1.graph_path), str(kg2.graph_path)],
            output_path=output,
        ))
        assert result["num_nodes"] == 2
        # Clean up cache
        _graph_cache.clear()

    def test_document_history(self, tmp_path):
        """document_history returns version timeline with section info."""
        from knowledge_graph import compute_section_hashes
        from mcp_server import document_history, _graph_cache

        kg = KnowledgeGraph(tmp_path / "mcp_hist.json")
        # Sections must be > 80 chars body for parse_markdown_sections
        body = "This is a sufficiently long section body. " * 5
        md = f"# Intro\n\n{body}\n\n# Methods\n\n{body}\n"
        hashes = compute_section_hashes(md)
        kg.store_source(md, "mydoc", section_hashes=hashes)
        kg.save()

        result = json.loads(document_history(
            graph_path=str(kg.graph_path),
            doc_id="mydoc",
        ))
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["version"] == 1
        assert result[0]["section_count"] >= 2
        _graph_cache.clear()

    def test_document_history_not_found(self, tmp_path):
        """document_history returns error for unknown doc."""
        from mcp_server import document_history, _graph_cache

        kg = KnowledgeGraph(tmp_path / "mcp_nohist.json")
        kg.save()

        result = json.loads(document_history(
            graph_path=str(kg.graph_path),
            doc_id="nonexistent",
        ))
        assert "error" in result
        _graph_cache.clear()
