"""Tests for knowledge_graph.py — core KnowledgeGraph class."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from knowledge_graph import (
    KnowledgeGraph,
    CoreRelation,
    ValidationReport,
    _merge_description,
    find_entity_spans,
    find_context_span,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_graph(tmp_path):
    """Create a fresh KnowledgeGraph in a temp directory."""
    return KnowledgeGraph(tmp_path / "test.json")


@pytest.fixture
def populated_graph(tmp_graph):
    """Graph with some nodes and edges already added."""
    kg = tmp_graph
    kg.add_node("radar", type="concept", label="Radar")
    kg.add_node("sar", type="technology", label="Synthetic Aperture Radar")
    kg.add_node("antenna", type="concept", label="Antenna")
    kg.add_edge("sar", "radar", relation="is_a")
    kg.add_edge("sar", "antenna", relation="depends_on")
    return kg


# ---------------------------------------------------------------------------
# Save / Load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    """Saving and loading produces identical graph data."""
    kg = KnowledgeGraph(tmp_path / "rt.json")
    kg.add_node("a", type="concept", label="Alpha", properties={"x": 1})
    kg.add_node("b", type="tool", label="Beta")
    kg.add_edge("a", "b", relation="uses", properties={"note": "test"})
    kg.save()

    kg2 = KnowledgeGraph(tmp_path / "rt.json")
    assert kg2.get_node("a")["label"] == "Alpha"
    assert kg2.get_node("a")["properties"]["x"] == 1
    assert kg2.get_node("b")["type"] == "tool"
    edges = kg2.get_edges("a", direction="outgoing")
    assert len(edges) == 1
    assert edges[0]["relation"] == "uses"
    assert edges[0]["properties"]["note"] == "test"


def test_save_load_embeddings_roundtrip(tmp_path):
    """Embeddings survive save/load cycle."""
    kg = KnowledgeGraph(tmp_path / "emb.json")
    kg.add_node("x", type="concept")
    kg.set_embedding("x", [1.0, 2.0, 3.0])
    kg.save()
    kg.save_embeddings()

    kg2 = KnowledgeGraph(tmp_path / "emb.json")
    assert kg2.get_embedding("x") == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# add_node
# ---------------------------------------------------------------------------


def test_add_node_basic(tmp_graph):
    nid = tmp_graph.add_node("foo", type="concept", label="Foo")
    assert nid == "foo"
    node = tmp_graph.get_node("foo")
    assert node["label"] == "Foo"
    assert node["type"] == "concept"


def test_add_node_merge(tmp_graph):
    """Merging updates properties and keeps the higher confidence."""
    tmp_graph.add_node("foo", type="concept", label="Foo", confidence=0.5, properties={"a": 1})
    tmp_graph.add_node("foo", type="tool", label="Foo2", confidence=0.8, properties={"b": 2})
    node = tmp_graph.get_node("foo")
    assert node["type"] == "tool"
    assert node["label"] == "Foo2"
    assert node["confidence"] == 0.8
    assert node["properties"]["a"] == 1
    assert node["properties"]["b"] == 2


def test_add_node_overwrite(tmp_graph):
    """With merge=False, properties are replaced entirely."""
    tmp_graph.add_node("foo", type="concept", properties={"a": 1})
    tmp_graph.add_node("foo", type="tool", properties={"b": 2}, merge=False)
    node = tmp_graph.get_node("foo")
    assert node["type"] == "tool"
    assert "a" not in node["properties"]
    assert node["properties"]["b"] == 2


def test_add_node_slugify(tmp_graph):
    """Node IDs with special characters are slugified."""
    nid = tmp_graph.add_node("Hello World!", type="concept")
    assert nid == "hello-world"


# ---------------------------------------------------------------------------
# add_edge
# ---------------------------------------------------------------------------


def test_add_edge_basic(populated_graph):
    edges = populated_graph.get_edges("sar", direction="outgoing")
    assert len(edges) == 2
    relations = {e["relation"] for e in edges}
    assert "is_a" in relations
    assert "depends_on" in relations


def test_add_edge_duplicate_updates(tmp_graph):
    """Adding a duplicate edge updates it rather than creating a new one."""
    tmp_graph.add_node("a", type="concept")
    tmp_graph.add_node("b", type="concept")
    tmp_graph.add_edge("a", "b", relation="uses", confidence=0.5)
    tmp_graph.add_edge("a", "b", relation="uses",
                       properties={"note": "updated"}, confidence=0.9)
    edges = [e for e in tmp_graph.get_edges("a", direction="outgoing")
             if e["relation"] == "uses"]
    assert len(edges) == 1
    assert edges[0]["properties"]["note"] == "updated"
    # confidence uses max(existing, new) → max(0.5, 0.9) = 0.9
    assert edges[0]["confidence"] == 0.9


def test_add_edge_allow_duplicate(populated_graph):
    """With allow_duplicate=True, a second edge is created."""
    populated_graph.add_edge("sar", "radar", relation="is_a",
                             allow_duplicate=True)
    edges = [e for e in populated_graph.get_edges("sar", direction="outgoing")
             if e["relation"] == "is_a"]
    assert len(edges) == 2


def test_add_edge_missing_node(tmp_graph):
    """Adding an edge with missing nodes raises KeyError."""
    tmp_graph.add_node("a", type="concept")
    with pytest.raises(KeyError):
        tmp_graph.add_edge("a", "nonexistent", relation="uses")


def test_add_edge_parallel_relations(tmp_graph):
    """Multiple edges between same pair with different relations."""
    tmp_graph.add_node("a", type="concept")
    tmp_graph.add_node("b", type="concept")
    tmp_graph.add_edge("a", "b", relation="uses")
    tmp_graph.add_edge("a", "b", relation="depends_on")
    edges = tmp_graph.get_edges("a", direction="outgoing")
    assert len(edges) == 2
    relations = {e["relation"] for e in edges}
    assert relations == {"uses", "depends_on"}


# ---------------------------------------------------------------------------
# remove_node
# ---------------------------------------------------------------------------


def test_remove_node(populated_graph):
    """Removing a node also removes connected edges and embeddings."""
    populated_graph.set_embedding("sar", [1.0, 2.0])
    assert populated_graph.remove_node("sar") is True
    assert populated_graph.get_node("sar") is None
    assert populated_graph.get_embedding("sar") is None
    # Edges involving 'sar' should be gone
    edges = populated_graph.get_edges("sar")
    assert len(edges) == 0


def test_remove_nonexistent_node(tmp_graph):
    assert tmp_graph.remove_node("doesnotexist") is False


# ---------------------------------------------------------------------------
# remove_edges
# ---------------------------------------------------------------------------


def test_remove_edges(populated_graph):
    removed = populated_graph.remove_edges("sar", "radar")
    assert removed == 1
    edges = populated_graph.get_edges("sar", direction="outgoing")
    assert len(edges) == 1
    assert edges[0]["relation"] == "depends_on"


def test_remove_edges_by_relation(populated_graph):
    removed = populated_graph.remove_edges("sar", "antenna", relation="depends_on")
    assert removed == 1
    removed = populated_graph.remove_edges("sar", "antenna", relation="depends_on")
    assert removed == 0


# ---------------------------------------------------------------------------
# Edge index
# ---------------------------------------------------------------------------


def test_edge_index_tracks_additions(tmp_graph):
    """The edge index contains all (source, target, relation) tuples."""
    tmp_graph.add_node("a", type="concept")
    tmp_graph.add_node("b", type="concept")
    tmp_graph.add_edge("a", "b", relation="uses")
    assert ("a", "b", "uses") in tmp_graph._edge_index


def test_edge_index_rebuilt_on_remove(populated_graph):
    """After removing edges, the index no longer contains the removed edge."""
    assert ("sar", "radar", "is_a") in populated_graph._edge_index
    populated_graph.remove_edges("sar", "radar", relation="is_a")
    assert ("sar", "radar", "is_a") not in populated_graph._edge_index


def test_edge_index_survives_save_load(tmp_path):
    """Edge index is rebuilt correctly after load."""
    kg = KnowledgeGraph(tmp_path / "idx.json")
    kg.add_node("a", type="concept")
    kg.add_node("b", type="concept")
    kg.add_edge("a", "b", relation="uses")
    kg.save()

    kg2 = KnowledgeGraph(tmp_path / "idx.json")
    assert ("a", "b", "uses") in kg2._edge_index


# ---------------------------------------------------------------------------
# MultiDiGraph parallel edges
# ---------------------------------------------------------------------------


def test_multidigraph_parallel_edges(tmp_graph):
    """NetworkX graph correctly stores multiple edges between same nodes."""
    tmp_graph.add_node("a", type="concept")
    tmp_graph.add_node("b", type="concept")
    tmp_graph.add_edge("a", "b", relation="uses")
    tmp_graph.add_edge("a", "b", relation="depends_on")
    # Both edges should exist in the NetworkX graph
    assert tmp_graph._G.has_edge("a", "b", key="uses")
    assert tmp_graph._G.has_edge("a", "b", key="depends_on")


# ---------------------------------------------------------------------------
# get_neighbors with relation filter
# ---------------------------------------------------------------------------


def test_get_neighbors_with_relation(populated_graph):
    """Relation filter correctly selects edges in MultiDiGraph."""
    neighbors = populated_graph.get_neighbors("sar", relation="is_a")
    neighbor_ids = {nid for nid, _ in neighbors}
    assert "radar" in neighbor_ids
    assert "antenna" not in neighbor_ids


def test_get_neighbors_without_filter(populated_graph):
    neighbors = populated_graph.get_neighbors("sar")
    neighbor_ids = {nid for nid, _ in neighbors}
    assert "radar" in neighbor_ids
    assert "antenna" in neighbor_ids


# ---------------------------------------------------------------------------
# _custom_relations is per-instance
# ---------------------------------------------------------------------------


def test_custom_relations_per_instance(tmp_path):
    """Custom relations registered on one instance don't affect another."""
    kg1 = KnowledgeGraph(tmp_path / "g1.json")
    kg2 = KnowledgeGraph(tmp_path / "g2.json")
    kg1.register_relation("my_custom_rel")
    assert "my_custom_rel" in kg1._custom_relations
    assert "my_custom_rel" not in kg2._custom_relations


# ---------------------------------------------------------------------------
# parse_markdown_sections
# ---------------------------------------------------------------------------


def test_parse_sections_atx_headings():
    md = "# Title\nIntro text.\n## Section A\nContent A.\n## Section B\nContent B."
    sections = KnowledgeGraph.parse_markdown_sections(md, min_section_chars=0)
    headings = [s["heading"] for s in sections]
    assert "Title" in headings
    assert "Section A" in headings
    assert "Section B" in headings


def test_parse_sections_setext_headings():
    md = "Title\n=====\nIntro text.\nSection A\n---------\nContent A."
    sections = KnowledgeGraph.parse_markdown_sections(md, min_section_chars=0)
    headings = [s["heading"] for s in sections]
    assert "Title" in headings
    assert "Section A" in headings


def test_parse_sections_short_merge():
    """Short sections are merged with next section by default."""
    md = "# Short\nABC\n# Long\n" + "x " * 100
    sections = KnowledgeGraph.parse_markdown_sections(md, min_section_chars=50)
    # "Short" should be merged into "Long"
    assert len(sections) == 1
    assert "ABC" in sections[0]["body"]


def test_parse_sections_oversized_split():
    """Long sections are split on paragraph boundaries."""
    paragraphs = [f"Paragraph {i}. " + "word " * 200 for i in range(10)]
    md = "# Long Section\n\n" + "\n\n".join(paragraphs)
    sections = KnowledgeGraph.parse_markdown_sections(
        md, min_section_chars=0, max_section_chars=2000
    )
    assert len(sections) > 1
    assert any("(part" in s["heading"] for s in sections)


def test_parse_sections_annotations():
    """Sections are annotated with code/list/table/link flags."""
    md = "# Code Section\n```python\nprint('hi')\n```\n- list item\n| a | b |\n|---|---|\n| 1 | 2 |\n[link](http://example.com)"
    sections = KnowledgeGraph.parse_markdown_sections(md, min_section_chars=0)
    s = sections[0]
    assert s["has_code"] is True
    assert s["has_list"] is True
    assert s["has_table"] is True
    assert len(s["links"]) == 1
    assert s["links"][0]["url"] == "http://example.com"


# ---------------------------------------------------------------------------
# ingest_document with various triple formats
# ---------------------------------------------------------------------------


def test_ingest_document_basic(tmp_graph):
    """Standard dict triples are ingested correctly."""
    triples = [
        {"source": "Radar", "target": "Radio waves", "relation": "uses",
         "confidence": 0.9, "context": "Radar uses radio waves."},
    ]
    stats = tmp_graph.ingest_document(
        "Radar uses radio waves.",
        doc_id="test-doc",
        llm_extract_fn=lambda _: triples,
    )
    assert stats["triples_processed"] == 1
    assert stats["nodes_added"] >= 2
    assert stats["edges_added"] >= 1


def test_ingest_document_list_triples(tmp_graph):
    """List-format triples [source, relation, target] are accepted."""
    triples = [["Radar", "uses", "Radio waves"]]
    stats = tmp_graph.ingest_document(
        "Radar uses radio waves.",
        doc_id="test-doc",
        llm_extract_fn=lambda _: triples,
    )
    assert stats["triples_processed"] == 1
    assert stats["nodes_added"] >= 2


def test_ingest_document_key_aliases(tmp_graph):
    """Triples with alias keys (subject/object/predicate) are normalized."""
    triples = [
        {"subject": "Radar", "object": "Radio waves", "predicate": "uses",
         "confidence": 0.9},
    ]
    stats = tmp_graph.ingest_document(
        "Radar uses radio waves.",
        doc_id="test-doc",
        llm_extract_fn=lambda _: triples,
    )
    assert stats["triples_processed"] == 1
    assert stats["nodes_added"] >= 2


def test_ingest_document_glossary(tmp_graph):
    """Glossary-style dicts {Concept, Definition} are converted to triples."""
    triples = [
        {"Concept": "Radar", "Definition": "A detection system using radio waves"},
    ]
    stats = tmp_graph.ingest_document(
        "Radar is a detection system using radio waves.",
        doc_id="test-doc",
        llm_extract_fn=lambda _: triples,
    )
    assert stats["triples_processed"] == 1
    assert stats["nodes_added"] >= 1


def test_ingest_document_hallucination_detection(tmp_graph):
    """Triples with entities not grounded in source text are rejected."""
    triples = [
        {"source": "TotallyFakeEntityXYZ", "target": "AnotherFakeEntityABC",
         "relation": "uses", "confidence": 0.9},
        {"source": "BogusThingOne", "target": "BogusThingTwo",
         "relation": "depends_on", "confidence": 0.9},
    ]
    stats = tmp_graph.ingest_document(
        "Radar uses radio waves for detection.",
        doc_id="test-doc",
        llm_extract_fn=lambda _: triples,
    )
    # Should detect hallucination and reject all
    assert stats["nodes_added"] == 0
    assert any("hallucinated" in str(e).lower() for e in stats["errors"])


# ---------------------------------------------------------------------------
# find_similar and dimension mismatch
# ---------------------------------------------------------------------------


def test_find_similar(tmp_graph):
    tmp_graph.add_node("a", type="concept")
    tmp_graph.add_node("b", type="concept")
    tmp_graph.set_embedding("a", [1.0, 0.0])
    tmp_graph.set_embedding("b", [0.0, 1.0])
    results = tmp_graph.find_similar([1.0, 0.0], top_k=2)
    assert results[0][0] == "a"
    assert results[0][1] > results[1][1]


def test_find_similar_dimension_mismatch(tmp_graph):
    tmp_graph.add_node("a", type="concept")
    tmp_graph.set_embedding("a", [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="dimension mismatch"):
        tmp_graph.find_similar([1.0, 0.0], top_k=1)


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def test_merge_nodes(tmp_path):
    kg1 = KnowledgeGraph(tmp_path / "m1.json")
    kg1.add_node("a", type="concept", label="Alpha")

    kg2 = KnowledgeGraph(tmp_path / "m2.json")
    kg2.add_node("b", type="tool", label="Beta")
    kg2.add_node("a", type="concept", label="Alpha Updated")

    stats = kg1.merge(kg2)
    assert stats["nodes_added"] == 1
    assert stats["nodes_updated"] == 1
    # prefer="other" (default) means kg2's version wins
    assert kg1.get_node("a")["label"] == "Alpha Updated"
    assert kg1.get_node("b")["label"] == "Beta"


def test_merge_edges(tmp_path):
    kg1 = KnowledgeGraph(tmp_path / "me1.json")
    kg1.add_node("a", type="concept")
    kg1.add_node("b", type="concept")
    kg1.add_edge("a", "b", relation="uses")

    kg2 = KnowledgeGraph(tmp_path / "me2.json")
    kg2.add_node("a", type="concept")
    kg2.add_node("b", type="concept")
    kg2.add_edge("a", "b", relation="depends_on")

    stats = kg1.merge(kg2)
    assert stats["edges_added"] == 1
    edges = kg1.get_edges("a", direction="outgoing")
    assert len(edges) == 2


def test_merge_embeddings_dirty(tmp_path):
    """Merging embeddings sets the dirty flag correctly."""
    kg1 = KnowledgeGraph(tmp_path / "md1.json")
    kg1.add_node("a", type="concept")

    kg2 = KnowledgeGraph(tmp_path / "md2.json")
    kg2.add_node("a", type="concept")
    kg2._embeddings["a"] = [1.0, 2.0]

    kg1.merge(kg2)
    assert kg1._dirty_embeddings is True
    assert kg1.get_embedding("a") == [1.0, 2.0]


def test_merge_smart_node_conflict(tmp_path):
    """Smart merge keeps max confidence and combines descriptions."""
    kg1 = KnowledgeGraph(tmp_path / "sc1.json")
    kg1.add_node("radar", type="concept", label="Radar",
                 properties={"description": "A detection system"},
                 source="doc:a", confidence=0.7)

    kg2 = KnowledgeGraph(tmp_path / "sc2.json")
    kg2.add_node("radar", type="system", label="RADAR",
                 properties={"description": "Radio detection and ranging"},
                 source="doc:b", confidence=0.9)

    stats = kg1.merge(kg2, prefer="other")
    node = kg1.get_node("radar")
    assert node["confidence"] == 0.9
    # Both descriptions should be present
    assert "detection system" in node["properties"]["description"]
    assert "Radio detection" in node["properties"]["description"]
    assert stats["nodes_updated"] == 1


def test_merge_smart_edge_conflict(tmp_path):
    """Smart merge keeps max confidence for duplicate edges."""
    kg1 = KnowledgeGraph(tmp_path / "ec1.json")
    kg1.add_node("a", type="concept")
    kg1.add_node("b", type="concept")
    kg1.add_edge("a", "b", relation="uses", confidence=0.5)

    kg2 = KnowledgeGraph(tmp_path / "ec2.json")
    kg2.add_node("a", type="concept")
    kg2.add_node("b", type="concept")
    kg2.add_edge("a", "b", relation="uses", confidence=0.9,
                 properties={"context": "high"})

    stats = kg1.merge(kg2)
    edges = kg1.get_edges("a", direction="outgoing")
    assert len(edges) == 1
    assert edges[0]["confidence"] == 0.9
    assert edges[0]["properties"]["context"] == "high"
    assert stats["edges_updated"] == 1


def test_merge_node_types(tmp_path):
    """Merge unions node_types from both graphs."""
    kg1 = KnowledgeGraph(tmp_path / "nt1.json")
    kg1._data["meta"]["node_types"] = ["concept", "tool"]

    kg2 = KnowledgeGraph(tmp_path / "nt2.json")
    kg2._data["meta"]["node_types"] = ["concept", "system", "person"]

    kg1.merge(kg2)
    types = set(kg1._data["meta"]["node_types"])
    assert types == {"concept", "tool", "system", "person"}


def test_merge_source_documents(tmp_path):
    """Merge copies source documents from the other graph."""
    kg1 = KnowledgeGraph(tmp_path / "ms1.json")
    kg1.store_source("Content A", "doc-a")

    kg2 = KnowledgeGraph(tmp_path / "ms2.json")
    kg2.store_source("Content B", "doc-b")

    stats = kg1.merge(kg2)
    assert stats["sources_added"] == 1
    manifest = kg1._data["meta"]["sources"]
    assert "doc-a" in manifest
    assert "doc-b" in manifest


def test_merge_source_documents_dedup(tmp_path):
    """Same content hash in both graphs is skipped."""
    kg1 = KnowledgeGraph(tmp_path / "sd1.json")
    kg1.store_source("Same content", "doc-1")

    kg2 = KnowledgeGraph(tmp_path / "sd2.json")
    kg2.store_source("Same content", "doc-1")

    stats = kg1.merge(kg2)
    assert stats["sources_skipped"] == 1


def test_merge_proposals_status(tmp_path):
    """Proposal status resolution: ACCEPTED > PENDING > REJECTED."""
    from knowledge_graph import RelationProposal, ProposalStatus

    kg1 = KnowledgeGraph(tmp_path / "ps1.json")
    p1 = RelationProposal(name="feeds_into", justification="test",
                          status=ProposalStatus.PENDING.value, confidence=0.5)
    kg1._proposals.append(p1)

    kg2 = KnowledgeGraph(tmp_path / "ps2.json")
    p2 = RelationProposal(name="feeds_into", justification="test",
                          status=ProposalStatus.ACCEPTED.value, confidence=0.8)
    kg2._proposals.append(p2)

    kg1.merge(kg2)
    merged_p = [p for p in kg1._proposals if p.name == "feeds_into"][0]
    assert merged_p.status == ProposalStatus.ACCEPTED.value
    assert merged_p.confidence == 0.8


def test_merge_embed_meta_warning(tmp_path, caplog):
    """Merging graphs with different embedding models logs a warning."""
    import logging

    kg1 = KnowledgeGraph(tmp_path / "ew1.json")
    kg1.add_node("a", type="concept")
    kg1._embed_meta = {"model": "model-a", "dimension": 128}

    kg2 = KnowledgeGraph(tmp_path / "ew2.json")
    kg2.add_node("a", type="concept")
    kg2._embed_meta = {"model": "model-b", "dimension": 128}
    kg2._embeddings["a"] = [1.0] * 128

    with caplog.at_level(logging.WARNING):
        kg1.merge(kg2)
    assert "different embedding models" in caplog.text


# ---------------------------------------------------------------------------
# merge_graphs classmethod
# ---------------------------------------------------------------------------


def test_merge_graphs_basic(tmp_path):
    """merge_graphs creates a new graph with nodes from both sources."""
    kg1 = KnowledgeGraph(tmp_path / "mg1.json")
    kg1.add_node("a", type="concept", label="Alpha")
    kg1.add_edge("a", "a", relation="self_ref")
    kg1.save()

    kg2 = KnowledgeGraph(tmp_path / "mg2.json")
    kg2.add_node("b", type="concept", label="Beta")
    kg2.save()

    merged = KnowledgeGraph.merge_graphs(
        [kg1, kg2], tmp_path / "merged.json",
    )
    assert merged.has_node("a")
    assert merged.has_node("b")
    assert merged.graph_path.exists()
    st = merged.stats()
    assert st["num_nodes"] == 2
    assert st["num_edges"] == 1


def test_merge_graphs_from_paths(tmp_path):
    """merge_graphs accepts file paths as strings."""
    kg1 = KnowledgeGraph(tmp_path / "p1.json")
    kg1.add_node("x", type="concept")
    kg1.save()

    kg2 = KnowledgeGraph(tmp_path / "p2.json")
    kg2.add_node("y", type="concept")
    kg2.save()

    merged = KnowledgeGraph.merge_graphs(
        [str(kg1.graph_path), str(kg2.graph_path)],
        tmp_path / "pm.json",
    )
    assert merged.has_node("x")
    assert merged.has_node("y")


def test_merge_graphs_prefer_first(tmp_path):
    """'first' strategy keeps first graph's data on conflict."""
    kg1 = KnowledgeGraph(tmp_path / "f1.json")
    kg1.add_node("n", type="concept", label="First", confidence=0.5)
    kg1.save()

    kg2 = KnowledgeGraph(tmp_path / "f2.json")
    kg2.add_node("n", type="concept", label="Second", confidence=0.9)
    kg2.save()

    merged = KnowledgeGraph.merge_graphs(
        [kg1, kg2], tmp_path / "first.json", prefer="first",
    )
    node = merged.get_node("n")
    # First graph's label wins
    assert node["label"] == "First"


def test_merge_graphs_prefer_last(tmp_path):
    """'last' strategy keeps last graph's data on conflict."""
    kg1 = KnowledgeGraph(tmp_path / "l1.json")
    kg1.add_node("n", type="concept", label="First", confidence=0.5)
    kg1.save()

    kg2 = KnowledgeGraph(tmp_path / "l2.json")
    kg2.add_node("n", type="concept", label="Second", confidence=0.9)
    kg2.save()

    merged = KnowledgeGraph.merge_graphs(
        [kg1, kg2], tmp_path / "last.json", prefer="last",
    )
    node = merged.get_node("n")
    assert node["label"] == "Second"


def test_merge_graphs_three_sources(tmp_path):
    """Merging three graphs works correctly."""
    graphs = []
    for i, name in enumerate(["t1", "t2", "t3"]):
        kg = KnowledgeGraph(tmp_path / f"{name}.json")
        kg.add_node(f"node-{i}", type="concept", label=f"Node {i}")
        kg.save()
        graphs.append(kg)

    merged = KnowledgeGraph.merge_graphs(
        graphs, tmp_path / "three.json",
    )
    assert merged.stats()["num_nodes"] == 3


def test_merge_graphs_roundtrip(tmp_path):
    """Merged graph survives save/load round-trip."""
    kg1 = KnowledgeGraph(tmp_path / "rt1.json")
    kg1.add_node("a", type="concept", label="Alpha",
                 properties={"description": "A"})
    kg1.add_node("b", type="concept", label="Beta")
    kg1.add_edge("a", "b", relation="uses")
    kg1._embeddings["a"] = [1.0, 2.0, 3.0]
    kg1.save_all()

    kg2 = KnowledgeGraph(tmp_path / "rt2.json")
    kg2.add_node("c", type="concept", label="Gamma")
    kg2.save()

    merged = KnowledgeGraph.merge_graphs(
        [kg1, kg2], tmp_path / "roundtrip.json",
    )

    # Reload from disk
    reloaded = KnowledgeGraph(merged.graph_path)
    assert reloaded.has_node("a")
    assert reloaded.has_node("b")
    assert reloaded.has_node("c")
    assert len(reloaded._data["edges"]) == 1
    assert reloaded.get_embedding("a") == [1.0, 2.0, 3.0]


def test_merge_graphs_empty(tmp_path):
    """Merging two empty graphs produces an empty graph."""
    kg1 = KnowledgeGraph(tmp_path / "e1.json")
    kg1.save()
    kg2 = KnowledgeGraph(tmp_path / "e2.json")
    kg2.save()

    merged = KnowledgeGraph.merge_graphs(
        [kg1, kg2], tmp_path / "empty.json",
    )
    st = merged.stats()
    assert st["num_nodes"] == 0
    assert st["num_edges"] == 0


def test_merge_graphs_requires_two(tmp_path):
    """merge_graphs raises ValueError with fewer than 2 sources."""
    import pytest
    kg = KnowledgeGraph(tmp_path / "one.json")
    with pytest.raises(ValueError, match="at least 2"):
        KnowledgeGraph.merge_graphs([kg], tmp_path / "out.json")


def test_merge_graphs_description(tmp_path):
    """Auto-generated description lists source graph names."""
    kg1 = KnowledgeGraph(tmp_path / "d1.json")
    kg1.save()
    kg2 = KnowledgeGraph(tmp_path / "d2.json")
    kg2.save()

    merged = KnowledgeGraph.merge_graphs(
        [kg1, kg2], tmp_path / "desc.json",
    )
    desc = merged._data["meta"]["description"]
    assert "d1" in desc
    assert "d2" in desc


# ---------------------------------------------------------------------------
# store_source
# ---------------------------------------------------------------------------


def test_store_source_basic(tmp_path):
    kg = KnowledgeGraph(tmp_path / "src.json")
    result = kg.store_source("Hello world", "doc1")
    assert result["is_duplicate"] is False
    assert result["version"] == 1
    assert Path(result["stored_path"]).exists()


def test_store_source_dedup(tmp_path):
    kg = KnowledgeGraph(tmp_path / "dedup.json")
    r1 = kg.store_source("Same content", "doc1")
    r2 = kg.store_source("Same content", "doc2")
    assert r2["is_duplicate"] is True
    assert r2["existing_doc_id"] == "doc1"


def test_store_source_update(tmp_path):
    kg = KnowledgeGraph(tmp_path / "upd.json")
    r1 = kg.store_source("Version 1 content", "mydoc")
    assert r1["version"] == 1
    r2 = kg.store_source("Version 2 content", "mydoc")
    assert r2["version"] == 2
    assert r2["is_update"] is True


# ---------------------------------------------------------------------------
# Document versioning (section hashes, diffs, history)
# ---------------------------------------------------------------------------


def _long_section(heading, body_seed):
    """Build a markdown section with body text > 80 chars for parse_markdown_sections."""
    body = f"{body_seed} " * 20  # ~200+ chars
    return f"# {heading}\n\n{body.strip()}\n"


def test_section_hashes_stored_in_manifest(tmp_path):
    """section_hashes are computed and stored when calling store_source with them."""
    from knowledge_graph import compute_section_hashes, content_hash

    kg = KnowledgeGraph(tmp_path / "sh.json")
    md = _long_section("Intro", "intro word") + "\n" + _long_section("Methods", "methods word")
    hashes = compute_section_hashes(md)
    result = kg.store_source(md, "doc1", section_hashes=hashes)

    assert "section_hashes" in result
    assert len(result["section_hashes"]) >= 2  # Intro + Methods (at minimum)

    # Verify manifest has section_hashes
    info = kg.get_source_info("doc1")
    assert "section_hashes" in info
    assert info["section_hashes"] == hashes


def test_section_hashes_in_version_history(tmp_path):
    """Archived versions preserve section_hashes from the old manifest entry."""
    from knowledge_graph import compute_section_hashes

    kg = KnowledgeGraph(tmp_path / "shv.json")

    md_v1 = _long_section("Alpha", "alpha content") + "\n" + _long_section("Beta", "beta content")
    hashes_v1 = compute_section_hashes(md_v1)
    kg.store_source(md_v1, "doc1", section_hashes=hashes_v1)

    md_v2 = _long_section("Alpha", "alpha changed") + "\n" + _long_section("Gamma", "gamma content")
    hashes_v2 = compute_section_hashes(md_v2)
    kg.store_source(md_v2, "doc1", section_hashes=hashes_v2)

    versions = kg.get_source_versions("doc1")
    assert len(versions) == 2

    v1 = versions[0]
    assert v1["version"] == 1
    assert v1["section_hashes"] == hashes_v1

    v2 = versions[1]
    assert v2["version"] == 2
    assert v2["section_hashes"] == hashes_v2


def test_diff_document_versions(tmp_path):
    """diff_document_versions identifies added/removed/modified/unchanged sections."""
    from knowledge_graph import compute_section_hashes, DocumentDiff

    kg = KnowledgeGraph(tmp_path / "diff.json")

    md_v1 = (_long_section("Intro", "intro text") + "\n"
             + _long_section("Methods", "methods text") + "\n"
             + _long_section("Results", "results text"))
    hashes_v1 = compute_section_hashes(md_v1)
    kg.store_source(md_v1, "doc1", section_hashes=hashes_v1)

    # v2: Intro unchanged, Methods modified, Results removed, Discussion added
    md_v2 = (_long_section("Intro", "intro text") + "\n"
             + _long_section("Methods", "methods CHANGED") + "\n"
             + _long_section("Discussion", "discussion new"))
    hashes_v2 = compute_section_hashes(md_v2)
    kg.store_source(md_v2, "doc1", section_hashes=hashes_v2)

    diff = kg.diff_document_versions("doc1", 1, 2)
    assert isinstance(diff, DocumentDiff)
    assert diff.doc_id == "doc1"
    assert diff.version_from == 1
    assert diff.version_to == 2
    assert "Discussion" in diff.added
    assert "Results" in diff.removed
    assert "Methods" in diff.modified
    assert "Intro" in diff.unchanged
    assert diff.has_changes is True
    assert "added" in diff.summary


def test_diff_document_versions_no_changes(tmp_path):
    """Diff shows unchanged + added when structure is extended."""
    from knowledge_graph import compute_section_hashes

    kg = KnowledgeGraph(tmp_path / "diffnc.json")

    md = _long_section("Intro", "intro text") + "\n" + _long_section("Methods", "methods text")
    hashes = compute_section_hashes(md)
    kg.store_source(md, "doc1", section_hashes=hashes)

    md_v2 = (md + "\n" + _long_section("Extra", "extra text"))
    hashes_v2 = compute_section_hashes(md_v2)
    kg.store_source(md_v2, "doc1", section_hashes=hashes_v2)

    diff = kg.diff_document_versions("doc1", 1, 2)
    assert "Intro" in diff.unchanged
    assert "Methods" in diff.unchanged
    assert "Extra" in diff.added


def test_diff_document_versions_missing_doc(tmp_path):
    """diff_document_versions raises KeyError for unknown doc."""
    kg = KnowledgeGraph(tmp_path / "noexist.json")
    with pytest.raises(KeyError):
        kg.diff_document_versions("nonexistent", 1, 2)


def test_diff_document_versions_missing_version(tmp_path):
    """diff_document_versions raises KeyError for unknown version number."""
    from knowledge_graph import compute_section_hashes

    kg = KnowledgeGraph(tmp_path / "nover.json")
    md = _long_section("Intro", "intro body text")
    hashes = compute_section_hashes(md)
    kg.store_source(md, "doc1", section_hashes=hashes)

    with pytest.raises(KeyError):
        kg.diff_document_versions("doc1", 1, 99)


def test_get_document_history(tmp_path):
    """get_document_history returns enriched version timeline with diffs."""
    from knowledge_graph import compute_section_hashes

    kg = KnowledgeGraph(tmp_path / "hist.json")

    md_v1 = _long_section("Intro", "intro text") + "\n" + _long_section("Methods", "methods text")
    hashes_v1 = compute_section_hashes(md_v1)
    kg.store_source(md_v1, "doc1", section_hashes=hashes_v1)

    md_v2 = (_long_section("Intro", "intro CHANGED") + "\n"
             + _long_section("Methods", "methods text") + "\n"
             + _long_section("Results", "results new"))
    hashes_v2 = compute_section_hashes(md_v2)
    kg.store_source(md_v2, "doc1", section_hashes=hashes_v2)

    history = kg.get_document_history("doc1")
    assert len(history) == 2

    # First version has no diff
    assert history[0]["version"] == 1
    assert history[0]["diff"] is None
    assert history[0]["section_count"] == len(hashes_v1)

    # Second version has diff
    assert history[1]["version"] == 2
    assert history[1]["diff"] is not None
    assert history[1]["diff"]["has_changes"] is True
    assert "Results" in history[1]["diff"]["added"]
    assert "Intro" in history[1]["diff"]["modified"]
    assert "Methods" in history[1]["diff"]["unchanged"]


def test_get_document_history_empty(tmp_path):
    """get_document_history returns empty list for unknown doc."""
    kg = KnowledgeGraph(tmp_path / "empty.json")
    assert kg.get_document_history("nonexistent") == []


def test_document_diff_to_dict(tmp_path):
    """DocumentDiff.to_dict() returns serializable representation."""
    from knowledge_graph import DocumentDiff

    diff = DocumentDiff(
        doc_id="test",
        version_from=1,
        version_to=2,
        added=["New Section"],
        removed=["Old Section"],
        modified=["Changed Section"],
        unchanged=["Same Section"],
    )
    d = diff.to_dict()
    assert d["doc_id"] == "test"
    assert d["has_changes"] is True
    assert "added" in d["summary"]
    assert "removed" in d["summary"]
    assert "modified" in d["summary"]
    assert "unchanged" in d["summary"]


# ---------------------------------------------------------------------------
# Proposal lifecycle
# ---------------------------------------------------------------------------


def test_proposal_lifecycle(tmp_graph):
    """Propose → accept → relation is registered."""
    p = tmp_graph.propose_relation(
        "validates",
        justification="X validates Y",
        source_entity="a",
        target_entity="b",
    )
    assert p.status == "pending"

    pending = tmp_graph.get_proposals()
    assert len(pending) == 1
    assert pending[0].name == "validates"

    assert tmp_graph.accept_proposal("validates") is True
    assert "validates" in tmp_graph._custom_relations

    # No more pending
    assert len(tmp_graph.get_proposals()) == 0


def test_proposal_reject(tmp_graph):
    tmp_graph.propose_relation("bad-rel")
    assert tmp_graph.reject_proposal("bad-rel") is True
    rejected = tmp_graph.get_proposals(status="rejected")
    assert len(rejected) == 1


def test_proposal_augment(tmp_graph):
    """Adding a second example to same proposal increases confidence."""
    p1 = tmp_graph.propose_relation("validates", source_entity="a", target_entity="b")
    c1 = p1.confidence
    p2 = tmp_graph.propose_relation("validates", source_entity="c", target_entity="d")
    assert p2.confidence > c1
    assert len(p2.examples) == 2


# ---------------------------------------------------------------------------
# save_all dirty flag handling
# ---------------------------------------------------------------------------


def test_save_all_with_dirty_embeddings(tmp_path):
    """save_all() saves embeddings when dirty flag is set."""
    kg = KnowledgeGraph(tmp_path / "da.json")
    kg.add_node("x", type="concept")
    kg.set_embedding("x", [1.0, 2.0])
    kg.save_all()
    assert kg.embeddings_path.exists()

    # Remove node (clears embedding and sets dirty)
    kg.remove_node("x")
    kg._dirty_embeddings = True
    kg.save_all()

    # Reload and verify embedding is gone
    kg2 = KnowledgeGraph(tmp_path / "da.json")
    assert kg2.get_embedding("x") is None


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats(populated_graph):
    s = populated_graph.stats()
    assert s["num_nodes"] == 3
    assert s["num_edges"] == 2


# ---------------------------------------------------------------------------
# Description merging — _merge_description()
# ---------------------------------------------------------------------------


def test_merge_description_basic():
    """Single description seeds both fields."""
    props: dict = {}
    _merge_description(props, "A detection system", "doc1", 0.9)
    assert props["description"] == "A detection system"
    assert len(props["description_sources"]) == 1
    assert props["description_sources"][0]["doc_id"] == "doc1"


def test_merge_description_accumulates():
    """Two different doc_ids produce a concatenated description."""
    props: dict = {}
    _merge_description(props, "Uses radio waves", "doc1", 0.9)
    _merge_description(props, "Detects objects at range", "doc2", 0.8)
    assert len(props["description_sources"]) == 2
    assert props["description"] == "Uses radio waves; Detects objects at range"


def test_merge_description_same_doc_update():
    """Same doc_id with different text updates in place."""
    props: dict = {}
    _merge_description(props, "Version 1", "doc1", 0.9)
    _merge_description(props, "Version 2", "doc1", 0.95)
    assert len(props["description_sources"]) == 1
    assert props["description_sources"][0]["text"] == "Version 2"
    assert props["description"] == "Version 2"


def test_merge_description_same_doc_skip():
    """Same doc_id with identical text is a no-op."""
    props: dict = {}
    _merge_description(props, "Same text", "doc1", 0.9)
    ts1 = props["description_sources"][0]["updated_at"]
    _merge_description(props, "Same text", "doc1", 0.9)
    assert len(props["description_sources"]) == 1
    # Timestamp unchanged since it was skipped
    assert props["description_sources"][0]["updated_at"] == ts1


def test_merge_description_dedup_text():
    """Two docs with identical text produce single entry in description."""
    props: dict = {}
    _merge_description(props, "Same description", "doc1", 0.9)
    _merge_description(props, "Same description", "doc2", 0.8)
    assert len(props["description_sources"]) == 2  # two sources
    assert props["description"] == "Same description"  # deduplicated text


def test_add_node_merge_description(tmp_graph):
    """Description merging works through add_node() merge path."""
    tmp_graph.add_node("radar", type="concept", label="Radar",
                       source="doc:doc1",
                       properties={"description": "A detection system"})
    tmp_graph.add_node("radar", type="concept", label="Radar",
                       source="doc:doc2",
                       properties={"description": "Uses radio waves"})
    node = tmp_graph.get_node("radar")
    assert len(node["properties"]["description_sources"]) == 2
    assert "A detection system" in node["properties"]["description"]
    assert "Uses radio waves" in node["properties"]["description"]


def test_ingest_description_seeded(tmp_graph):
    """New node from ingestion has description_sources."""
    triples = [
        {"source": "Radar", "target": "Radio waves", "relation": "uses",
         "source_description": "A detection system", "confidence": 0.9,
         "context": "Radar uses radio waves."},
    ]
    tmp_graph.ingest_document(
        "Radar uses radio waves for detection.",
        doc_id="test-doc",
        llm_extract_fn=lambda _: triples,
    )
    node = tmp_graph.get_node("radar")
    assert node is not None
    assert node["properties"]["description"] == "A detection system"
    assert len(node["properties"]["description_sources"]) == 1
    assert node["properties"]["description_sources"][0]["doc_id"] == "test-doc"


def test_ingest_description_multi_doc(tmp_graph):
    """Two ingestions of same entity accumulate descriptions."""
    triples1 = [
        {"source": "Radar", "target": "Radio waves", "relation": "uses",
         "source_description": "A detection system", "confidence": 0.9,
         "context": "Radar uses radio waves."},
    ]
    triples2 = [
        {"source": "Radar", "target": "Targets", "relation": "detects",
         "source_description": "Emits electromagnetic pulses", "confidence": 0.8,
         "context": "Radar detects targets."},
    ]
    tmp_graph.ingest_document(
        "Radar uses radio waves for detection.",
        doc_id="doc1",
        llm_extract_fn=lambda _: triples1,
    )
    tmp_graph.ingest_document(
        "Radar detects targets using electromagnetic pulses.",
        doc_id="doc2",
        llm_extract_fn=lambda _: triples2,
    )
    node = tmp_graph.get_node("radar")
    assert len(node["properties"]["description_sources"]) == 2
    assert "; " in node["properties"]["description"]


# ---------------------------------------------------------------------------
# Span mapping — find_entity_spans()
# ---------------------------------------------------------------------------


def test_find_entity_spans_exact():
    spans = find_entity_spans("Radar uses radio waves", "Radar")
    assert len(spans) >= 1
    assert spans[0]["start"] == 0
    assert spans[0]["end"] == 5
    assert spans[0]["match_type"] == "exact"


def test_find_entity_spans_case_insensitive():
    spans = find_entity_spans("Radar uses radio waves", "radar")
    assert len(spans) >= 1
    assert spans[0]["matched_text"] == "Radar"


def test_find_entity_spans_multiple():
    spans = find_entity_spans("Radar detects targets. Radar is useful.", "Radar")
    assert len(spans) == 2
    assert spans[0]["start"] == 0
    assert spans[1]["start"] == 23


def test_find_entity_spans_partial():
    """Multi-word entity with partial token match."""
    spans = find_entity_spans(
        "Synthetic aperture systems are advanced",
        "Synthetic Aperture Radar",
    )
    # Should match "Synthetic" or "aperture" via partial token
    assert len(spans) >= 1
    assert spans[0]["match_type"] in ("exact", "word_boundary", "partial_token")


def test_find_entity_spans_no_match():
    spans = find_entity_spans("Nothing relevant here", "Quantum Entanglement")
    assert spans == []


def test_find_entity_spans_empty():
    assert find_entity_spans("", "Radar") == []
    assert find_entity_spans("Radar", "") == []


# ---------------------------------------------------------------------------
# Span mapping — find_context_span()
# ---------------------------------------------------------------------------


def test_find_context_span_exact():
    text = "Radar uses radio waves for detection."
    span = find_context_span(text, "uses radio waves")
    assert span is not None
    assert span["start"] == 6
    assert span["end"] == 22
    assert span["match_type"] == "exact"


def test_find_context_span_case_insensitive():
    text = "Radar uses radio waves."
    span = find_context_span(text, "USES RADIO WAVES")
    assert span is not None
    assert span["match_type"] == "case_insensitive"


def test_find_context_span_prefix():
    text = "Radar uses radio waves for long-range detection of objects."
    context = "Radar uses radio waves for long-range detection of objects and more stuff added by LLM."
    span = find_context_span(text, context)
    assert span is not None
    assert span["match_type"] == "prefix"
    assert span["start"] == 0


def test_find_context_span_not_found():
    span = find_context_span("Nothing relevant here", "Quantum computing is fast")
    assert span is None


# ---------------------------------------------------------------------------
# Span tracking in ingestion
# ---------------------------------------------------------------------------


def test_ingest_creates_mentions(tmp_graph):
    """After ingestion, nodes have mentions with character offsets."""
    triples = [
        {"source": "Radar", "target": "Radio waves", "relation": "uses",
         "confidence": 0.9, "context": "Radar uses radio waves."},
    ]
    tmp_graph.ingest_document(
        "Radar uses radio waves for detection.",
        doc_id="test-doc",
        llm_extract_fn=lambda _: triples,
    )
    node = tmp_graph.get_node("radar")
    assert node is not None
    mentions = node["properties"].get("mentions", [])
    assert len(mentions) >= 1
    assert mentions[0]["doc_id"] == "test-doc"
    assert mentions[0]["start"] == 0
    assert mentions[0]["end"] == 5


def test_ingest_creates_context_span(tmp_graph):
    """After ingestion, edges have context_span with offsets."""
    context_text = "Radar uses radio waves"
    triples = [
        {"source": "Radar", "target": "Radio waves", "relation": "uses",
         "confidence": 0.9, "context": context_text},
    ]
    tmp_graph.ingest_document(
        "Radar uses radio waves for detection.",
        doc_id="test-doc",
        llm_extract_fn=lambda _: triples,
    )
    edges = tmp_graph.get_edges("radar", direction="outgoing")
    uses_edges = [e for e in edges if e["relation"] == "uses"]
    assert len(uses_edges) >= 1
    ctx_span = uses_edges[0]["properties"].get("context_span")
    assert ctx_span is not None
    assert ctx_span["doc_id"] == "test-doc"
    assert ctx_span["start"] == 0
    assert ctx_span["match_type"] == "exact"


def test_mentions_roundtrip(tmp_path):
    """Save/load preserves mentions and context_span."""
    kg = KnowledgeGraph(tmp_path / "spans.json")
    triples = [
        {"source": "Radar", "target": "Radio waves", "relation": "uses",
         "confidence": 0.9, "context": "Radar uses radio waves."},
    ]
    kg.ingest_document(
        "Radar uses radio waves for detection.",
        doc_id="test-doc",
        llm_extract_fn=lambda _: triples,
    )
    kg.save()

    kg2 = KnowledgeGraph(tmp_path / "spans.json")
    node = kg2.get_node("radar")
    assert len(node["properties"].get("mentions", [])) >= 1
    edges = kg2.get_edges("radar", direction="outgoing")
    uses_edges = [e for e in edges if e["relation"] == "uses"]
    assert uses_edges[0]["properties"].get("context_span") is not None


# ---------------------------------------------------------------------------
# Document subgraph extract / import
# ---------------------------------------------------------------------------


def _ingest_with_source(kg, text, doc_id, triples):
    """Helper: ingest a document and store its source text."""
    kg.store_source(text, doc_id)
    kg.ingest_document(text, doc_id=doc_id, llm_extract_fn=lambda _: triples)


def test_extract_document_subgraph(tmp_path):
    """Extract a document subgraph captures nodes, edges, source text."""
    kg = KnowledgeGraph(tmp_path / "extract.json")
    triples = [
        {"source": "Radar", "target": "Radio waves", "relation": "uses",
         "confidence": 0.9, "context": "Radar uses radio waves."},
        {"source": "Radar", "target": "Antenna", "relation": "depends_on",
         "confidence": 0.85, "context": "Radar depends on an antenna."},
    ]
    _ingest_with_source(kg, "Radar uses radio waves. Radar depends on an antenna.",
                        "radar-doc", triples)
    kg.save()

    subgraph = kg.extract_document_subgraph("radar-doc")
    assert subgraph["doc_id"] == "radar-doc"
    assert len(subgraph["nodes"]) > 0
    assert len(subgraph["edges"]) > 0
    assert subgraph["source_text"] is not None
    assert "Radar" in subgraph["source_text"]
    assert subgraph["source_info"]["content_hash"]
    assert subgraph["origin_graph"] == "extract"


def test_extract_document_subgraph_missing(tmp_graph):
    """Extracting a non-existent document raises KeyError."""
    with pytest.raises(KeyError):
        tmp_graph.extract_document_subgraph("nonexistent")


def test_import_document_subgraph(tmp_path):
    """Import a document subgraph into a different graph."""
    # Build source graph
    src_kg = KnowledgeGraph(tmp_path / "src.json")
    triples = [
        {"source": "Radar", "target": "Radio waves", "relation": "uses",
         "confidence": 0.9, "context": "Radar uses radio waves."},
    ]
    _ingest_with_source(src_kg, "Radar uses radio waves for detection.",
                        "radar-doc", triples)
    src_kg.save()

    # Extract subgraph
    subgraph = src_kg.extract_document_subgraph("radar-doc")

    # Import into destination graph
    dst_kg = KnowledgeGraph(tmp_path / "dst.json")
    dst_kg.add_node("existing", type="concept", label="Existing Node")
    stats = dst_kg.import_document_subgraph(subgraph)

    assert stats["nodes_added"] > 0
    assert stats["source_stored"] == 1

    # Verify nodes were imported
    assert dst_kg.get_node("radar") is not None
    # Verify source text was stored
    assert dst_kg.has_source("radar-doc")
    # Verify transplant provenance
    info = dst_kg.get_source_info("radar-doc")
    assert info is not None
    assert len(info.get("transplanted_from", [])) == 1
    assert info["transplanted_from"][0]["graph"] == "src"


def test_import_subgraph_smart_merge(tmp_path):
    """Importing into a graph with overlapping nodes does smart merge."""
    # Source graph
    src_kg = KnowledgeGraph(tmp_path / "src2.json")
    triples = [
        {"source": "Python", "target": "Language", "relation": "is_a",
         "confidence": 0.9, "context": "Python is a language."},
    ]
    _ingest_with_source(src_kg, "Python is a language.", "py-doc", triples)
    src_kg.save()

    # Destination graph with an overlapping node
    dst_kg = KnowledgeGraph(tmp_path / "dst2.json")
    dst_kg.add_node("python", type="concept", label="Python", source="manual",
                     properties={"description": "A great language"}, confidence=0.5)
    dst_kg.save()

    subgraph = src_kg.extract_document_subgraph("py-doc")
    stats = dst_kg.import_document_subgraph(subgraph)

    assert stats["nodes_updated"] >= 1
    # The existing node should still be there with merged data
    node = dst_kg.get_node("python")
    assert node is not None
    assert node["confidence"] >= 0.5  # should keep max confidence


def test_extract_import_roundtrip(tmp_path):
    """Extract → import → extract produces compatible subgraphs."""
    kg1 = KnowledgeGraph(tmp_path / "kg1.json")
    triples = [
        {"source": "A", "target": "B", "relation": "related_to",
         "confidence": 0.8, "context": "A relates to B."},
    ]
    _ingest_with_source(kg1, "A relates to B in this document.",
                        "test-doc", triples)
    kg1.save()

    sub1 = kg1.extract_document_subgraph("test-doc")
    kg2 = KnowledgeGraph(tmp_path / "kg2.json")
    kg2.import_document_subgraph(sub1)
    kg2.save()

    # The destination graph should now also be able to extract the same doc
    sub2 = kg2.extract_document_subgraph("test-doc")
    assert sub2["doc_id"] == sub1["doc_id"]
    assert sub2["source_text"] == sub1["source_text"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_clean_graph(populated_graph):
    """A well-formed graph should pass validation with no errors."""
    report = populated_graph.validate()
    assert report.is_valid
    assert len(report.errors) == 0


def test_validate_dangling_edge(tmp_graph):
    """Edges referencing non-existent nodes should be reported as errors."""
    kg = tmp_graph
    kg.add_node("a", type="concept", label="A")
    kg.add_node("b", type="concept", label="B")
    kg.add_edge("a", "b", relation="related_to")
    # Manually inject a dangling edge
    kg._data["edges"].append({
        "source": "a", "target": "ghost", "relation": "uses",
        "properties": {}, "source_tag": "test", "confidence": 0.8, "weight": 1.0,
    })
    report = kg.validate()
    assert not report.is_valid
    assert any("ghost" in e for e in report.errors)


def test_validate_taxonomic_cycle(tmp_graph):
    """Cycles in is_a relations should be flagged as errors."""
    kg = tmp_graph
    kg.add_node("a", type="concept", label="A")
    kg.add_node("b", type="concept", label="B")
    kg.add_node("c", type="concept", label="C")
    kg.add_edge("a", "b", relation="is_a")
    kg.add_edge("b", "c", relation="is_a")
    kg.add_edge("c", "a", relation="is_a")
    report = kg.validate()
    assert not report.is_valid
    assert any("cycle" in e.lower() for e in report.errors)


def test_validate_no_cycle_in_non_taxonomic(tmp_graph):
    """Cycles in non-taxonomic relations should NOT be flagged as errors."""
    kg = tmp_graph
    kg.add_node("a", type="concept", label="A")
    kg.add_node("b", type="concept", label="B")
    kg.add_edge("a", "b", relation="related_to")
    kg.add_edge("b", "a", relation="related_to")
    report = kg.validate()
    # No taxonomic cycle error
    assert not any("cycle" in e.lower() for e in report.errors)


def test_validate_contradictory_edges(tmp_graph):
    """Contradictory edges on the same pair should produce warnings."""
    kg = tmp_graph
    kg.add_node("a", type="concept", label="A")
    kg.add_node("b", type="concept", label="B")
    kg.add_edge("a", "b", relation="supersedes")
    kg.add_edge("a", "b", relation="superseded_by")
    report = kg.validate()
    assert any("contradictory" in w.lower() for w in report.warnings)


def test_validate_reflexive_contradiction(tmp_graph):
    """A is_a B and B is_a A should be flagged as a reflexive contradiction."""
    kg = tmp_graph
    kg.add_node("a", type="concept", label="A")
    kg.add_node("b", type="concept", label="B")
    kg.add_edge("a", "b", relation="is_a")
    kg.add_edge("b", "a", relation="is_a")
    report = kg.validate()
    # Should be caught as either a cycle error or a reflexive contradiction warning
    has_cycle = any("cycle" in e.lower() for e in report.errors)
    has_reflexive = any("reflexive" in w.lower() for w in report.warnings)
    assert has_cycle or has_reflexive


def test_validate_orphan_nodes(tmp_graph):
    """Nodes with no edges should be flagged as orphans (warnings)."""
    kg = tmp_graph
    kg.add_node("connected-a", type="concept", label="A")
    kg.add_node("connected-b", type="concept", label="B")
    kg.add_edge("connected-a", "connected-b", relation="related_to")
    kg.add_node("orphan", type="concept", label="Orphan")
    report = kg.validate()
    assert any("orphan" in w.lower() for w in report.warnings)
    assert any("'orphan'" in w for w in report.warnings)


def test_validate_orphan_document_nodes_excluded(tmp_graph):
    """Document nodes with no edges should NOT be flagged as orphans."""
    kg = tmp_graph
    kg.add_node("doc-node", type="document", label="My Doc")
    report = kg.validate()
    assert not any("orphan" in w.lower() for w in report.warnings)


def test_validate_zero_confidence(tmp_graph):
    """Nodes/edges with confidence=0 should be flagged as warnings."""
    kg = tmp_graph
    kg.add_node("a", type="concept", label="A", confidence=0.0)
    kg.add_node("b", type="concept", label="B")
    kg.add_edge("a", "b", relation="related_to", confidence=0.0)
    report = kg.validate()
    zero_warnings = [w for w in report.warnings if "confidence=0" in w]
    assert len(zero_warnings) >= 2  # one for node, one for edge


def test_validate_report_to_dict(populated_graph):
    """ValidationReport.to_dict() should return a well-structured dict."""
    report = populated_graph.validate()
    d = report.to_dict()
    assert "is_valid" in d
    assert "errors" in d
    assert "warnings" in d
    assert "info" in d
    assert "summary" in d
    assert d["summary"]["error_count"] == len(d["errors"])
    assert d["summary"]["warning_count"] == len(d["warnings"])


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


def test_analytics_empty_graph(tmp_graph):
    """Analytics on an empty graph should return valid structure with zeros."""
    a = tmp_graph.analytics()
    assert 0 <= a["quality_score"] <= 100
    assert a["confidence_distribution"]["node_counts"] == [0] * 10
    assert a["confidence_distribution"]["edge_counts"] == [0] * 10
    assert a["hub_nodes"] == []
    assert a["orphan_nodes"] == []
    assert a["embedding_coverage"]["embeddable"] == 0
    assert a["component_sizes"] == []


def test_analytics_populated_graph(populated_graph):
    """Analytics on a graph with nodes and edges returns expected structure."""
    a = populated_graph.analytics()

    # Quality score should be between 0 and 100
    assert 0 <= a["quality_score"] <= 100

    # Confidence distribution should have 10 buckets
    assert len(a["confidence_distribution"]["buckets"]) == 10
    assert len(a["confidence_distribution"]["node_counts"]) == 10
    assert len(a["confidence_distribution"]["edge_counts"]) == 10

    # Should sum to total node/edge counts
    assert sum(a["confidence_distribution"]["node_counts"]) == populated_graph.num_nodes
    assert sum(a["confidence_distribution"]["edge_counts"]) == populated_graph.num_edges

    # Mean confidence should be in [0, 1]
    assert 0 <= a["confidence_distribution"]["node_mean"] <= 1
    assert 0 <= a["confidence_distribution"]["edge_mean"] <= 1

    # Relation stats
    assert "is_a" in a["relation_stats"]
    assert "depends_on" in a["relation_stats"]
    assert a["relation_stats"]["is_a"]["count"] == 1
    assert a["relation_stats"]["depends_on"]["count"] == 1

    # Node type stats
    assert "concept" in a["node_type_stats"]
    assert "technology" in a["node_type_stats"]
    assert a["node_type_stats"]["concept"]["count"] == 2  # radar, antenna
    assert a["node_type_stats"]["technology"]["count"] == 1  # sar

    # Hub nodes
    assert len(a["hub_nodes"]) > 0
    assert "node_id" in a["hub_nodes"][0]
    assert "degree" in a["hub_nodes"][0]
    assert "in_degree" in a["hub_nodes"][0]

    # Component sizes
    assert a["component_sizes"] == [3]  # all nodes connected


def test_analytics_hub_nodes_sorted_by_degree(tmp_graph):
    """Hub nodes should be sorted by degree, highest first."""
    kg = tmp_graph
    kg.add_node("center", type="concept", label="Center")
    for i in range(5):
        nid = f"leaf-{i}"
        kg.add_node(nid, type="concept", label=f"Leaf {i}")
        kg.add_edge("center", nid, relation="related_to")
    a = kg.analytics()
    assert a["hub_nodes"][0]["node_id"] == "center"
    assert a["hub_nodes"][0]["degree"] == 5


def test_analytics_orphan_detection(tmp_graph):
    """Orphan nodes should be listed (excluding document/section types)."""
    kg = tmp_graph
    kg.add_node("connected-a", type="concept", label="A")
    kg.add_node("connected-b", type="concept", label="B")
    kg.add_edge("connected-a", "connected-b", relation="related_to")
    kg.add_node("orphan-1", type="concept", label="Orphan")
    kg.add_node("doc-orphan", type="document", label="Doc")
    a = kg.analytics()
    orphan_ids = [o["node_id"] for o in a["orphan_nodes"]]
    assert "orphan-1" in orphan_ids
    assert "doc-orphan" not in orphan_ids


def test_analytics_confidence_distribution(tmp_graph):
    """Confidence values should land in correct histogram buckets."""
    kg = tmp_graph
    kg.add_node("low", type="concept", label="Low", confidence=0.1)
    kg.add_node("mid", type="concept", label="Mid", confidence=0.5)
    kg.add_node("high", type="concept", label="High", confidence=0.9)
    kg.add_edge("low", "mid", relation="related_to", confidence=0.3)
    kg.add_edge("mid", "high", relation="related_to", confidence=0.8)
    a = kg.analytics()
    nc = a["confidence_distribution"]["node_counts"]
    ec = a["confidence_distribution"]["edge_counts"]
    # 0.1 -> bucket 1, 0.5 -> bucket 5, 0.9 -> bucket 9
    assert nc[1] >= 1  # 0.1
    assert nc[5] >= 1  # 0.5
    assert nc[9] >= 1  # 0.9
    # 0.3 -> bucket 3, 0.8 -> bucket 8
    assert ec[3] >= 1
    assert ec[8] >= 1


def test_analytics_embedding_coverage(tmp_graph):
    """Embedding coverage should reflect which nodes have embeddings."""
    kg = tmp_graph
    kg.add_node("a", type="concept", label="A")
    kg.add_node("b", type="concept", label="B")
    kg.add_node("c", type="concept", label="C")
    kg.set_embedding("a", [0.1, 0.2])
    a = kg.analytics()
    ec = a["embedding_coverage"]
    assert ec["embeddable"] == 3
    assert ec["embedded"] == 1
    assert ec["pct"] == pytest.approx(33.3, abs=0.1)


def test_analytics_quality_score_improves_with_embeddings(tmp_graph):
    """Quality score should improve when more nodes have embeddings."""
    kg = tmp_graph
    kg.add_node("a", type="concept", label="A")
    kg.add_node("b", type="concept", label="B")
    kg.add_edge("a", "b", relation="related_to")
    score_before = kg.analytics()["quality_score"]
    kg.set_embedding("a", [0.1, 0.2])
    kg.set_embedding("b", [0.3, 0.4])
    score_after = kg.analytics()["quality_score"]
    assert score_after > score_before


# ── Incremental ingestion tests ──────────────────────────────


def test_incremental_skips_unchanged_sections(tmp_path):
    """Incremental ingestion skips LLM extraction for sections unchanged since last version."""
    kg = KnowledgeGraph(tmp_path / "inc.json")

    # Track which sections the LLM was called on
    extracted_prompts = []

    def mock_extract(prompt):
        extracted_prompts.append(prompt)
        return [{"source": "EntityA", "target": "EntityB", "relation": "related_to"}]

    md_v1 = (_long_section("Intro", "introduction content here") + "\n"
             + _long_section("Methods", "methods description here") + "\n"
             + _long_section("Results", "results findings here"))

    # First ingestion: all sections should be extracted
    stats_v1 = kg.ingest_markdown(
        md_v1, "doc1", llm_extract_fn=mock_extract, incremental=True,
    )
    assert stats_v1["sections_skipped_incremental"] == 0
    calls_v1 = len(extracted_prompts)
    assert calls_v1 == 3  # All three sections extracted

    kg.save()
    extracted_prompts.clear()

    # v2: Intro unchanged, Methods changed, Results unchanged, Discussion added
    md_v2 = (_long_section("Intro", "introduction content here") + "\n"
             + _long_section("Methods", "methods COMPLETELY REWRITTEN") + "\n"
             + _long_section("Results", "results findings here") + "\n"
             + _long_section("Discussion", "discussion text added"))

    stats_v2 = kg.ingest_markdown(
        md_v2, "doc1", llm_extract_fn=mock_extract, incremental=True,
    )

    # Intro and Results are unchanged → should be skipped
    assert stats_v2["sections_skipped_incremental"] == 2
    # Methods (changed) + Discussion (new) → 2 LLM calls
    assert len(extracted_prompts) == 2


def test_incremental_false_extracts_all_sections(tmp_path):
    """When incremental=False (default), all sections are re-extracted on re-ingestion."""
    kg = KnowledgeGraph(tmp_path / "noinc.json")

    extracted_prompts = []

    def mock_extract(prompt):
        extracted_prompts.append(prompt)
        return [{"source": "X", "target": "Y", "relation": "related_to"}]

    md_v1 = (_long_section("Intro", "intro text words") + "\n"
             + _long_section("Methods", "methods description"))

    kg.ingest_markdown(md_v1, "doc1", llm_extract_fn=mock_extract)
    kg.save()
    extracted_prompts.clear()

    # Same doc, just add a section
    md_v2 = (md_v1 + "\n" + _long_section("Extra", "extra content"))

    stats_v2 = kg.ingest_markdown(
        md_v2, "doc1", llm_extract_fn=mock_extract, incremental=False,
    )

    # Without incremental, all 3 sections should be extracted
    assert stats_v2["sections_skipped_incremental"] == 0
    assert len(extracted_prompts) == 3


def test_incremental_first_ingestion_extracts_everything(tmp_path):
    """First-time ingestion with incremental=True still extracts everything."""
    kg = KnowledgeGraph(tmp_path / "first.json")

    call_count = [0]

    def mock_extract(prompt):
        call_count[0] += 1
        return [{"source": "A", "target": "B", "relation": "related_to"}]

    md = (_long_section("Alpha", "alpha body text") + "\n"
          + _long_section("Beta", "beta body text"))

    stats = kg.ingest_markdown(
        md, "doc1", llm_extract_fn=mock_extract, incremental=True,
    )
    assert stats["sections_skipped_incremental"] == 0
    assert call_count[0] == 2


def test_incremental_without_preserve_source_extracts_all(tmp_path):
    """Incremental requires preserve_source=True; without it, everything is extracted."""
    kg = KnowledgeGraph(tmp_path / "nops.json")

    call_count = [0]

    def mock_extract(prompt):
        call_count[0] += 1
        return []

    md = _long_section("Only", "only section content")

    # First pass with preserve_source
    kg.ingest_markdown(md, "doc1", llm_extract_fn=mock_extract, preserve_source=True)
    kg.save()
    call_count[0] = 0

    # Re-ingest with incremental=True but preserve_source=False
    kg.ingest_markdown(
        md, "doc1", llm_extract_fn=mock_extract,
        incremental=True, preserve_source=False,
    )
    # Without preserve_source, incremental can't compare → extracts everything
    assert call_count[0] == 1


def test_incremental_progress_events(tmp_path):
    """Incremental ingestion fires 'section_skip' with reason='unchanged' and 'incremental_skip_plan'."""
    kg = KnowledgeGraph(tmp_path / "prog.json")

    def mock_extract(prompt):
        return [{"source": "A", "target": "B", "relation": "related_to"}]

    events = []

    def capture_progress(event):
        events.append(event)

    md_v1 = (_long_section("Stable", "stable content here") + "\n"
             + _long_section("Changing", "changing content"))

    kg.ingest_markdown(md_v1, "doc1", llm_extract_fn=mock_extract,
                       progress_fn=capture_progress, incremental=True)
    kg.save()
    events.clear()

    md_v2 = (_long_section("Stable", "stable content here") + "\n"
             + _long_section("Changing", "completely different now"))

    kg.ingest_markdown(md_v2, "doc1", llm_extract_fn=mock_extract,
                       progress_fn=capture_progress, incremental=True)

    # Should have an incremental_skip_plan event
    skip_plan_events = [e for e in events if e["event"] == "incremental_skip_plan"]
    assert len(skip_plan_events) == 1
    assert "Stable" in skip_plan_events[0]["unchanged_sections"]

    # Should have a section_skip event with reason "unchanged"
    skip_events = [e for e in events if e.get("event") == "section_skip"
                   and e.get("reason") == "unchanged"]
    assert len(skip_events) == 1
    assert skip_events[0]["heading"] == "Stable"


def test_incremental_structural_nodes_still_updated(tmp_path):
    """Even when sections are skipped for extraction, structural nodes/edges are created."""
    kg = KnowledgeGraph(tmp_path / "struct.json")

    call_count = [0]

    def mock_extract(prompt):
        call_count[0] += 1
        return [{"source": "Entity1", "target": "Entity2", "relation": "related_to"}]

    md_v1 = (_long_section("Alpha", "alpha body content") + "\n"
             + _long_section("Beta", "beta body content"))

    kg.ingest_markdown(md_v1, "doc1", llm_extract_fn=mock_extract, incremental=True)
    kg.save()

    # Verify section nodes exist
    alpha_slug = "doc1-alpha"
    beta_slug = "doc1-beta"
    assert kg.get_node(alpha_slug) is not None
    assert kg.get_node(beta_slug) is not None

    call_count[0] = 0

    # v2: Alpha unchanged, Beta unchanged, Gamma added (triggers is_update)
    md_v2 = (md_v1 + "\n" + _long_section("Gamma", "gamma new content"))
    stats = kg.ingest_markdown(md_v2, "doc1", llm_extract_fn=mock_extract, incremental=True)
    # Alpha and Beta unchanged → skipped for extraction
    assert stats["sections_skipped_incremental"] == 2
    # Only Gamma extracted
    assert call_count[0] == 1

    # All section nodes still present (including the unchanged ones)
    assert kg.get_node(alpha_slug) is not None
    assert kg.get_node(beta_slug) is not None
    assert kg.get_node("doc1-gamma") is not None
    # Document node still present
    assert kg.get_node("doc1") is not None


# ---------------------------------------------------------------------------
# Graph diffing
# ---------------------------------------------------------------------------


def test_graph_diff_empty_graphs(tmp_path):
    """Diffing two empty graphs yields no changes."""
    from knowledge_graph import GraphDiff

    kg1 = KnowledgeGraph(tmp_path / "a.json")
    kg2 = KnowledgeGraph(tmp_path / "b.json")
    diff = kg1.diff(kg2)
    assert isinstance(diff, GraphDiff)
    assert not diff.has_changes
    assert diff.summary == "no changes"


def test_graph_diff_nodes_added(tmp_path):
    """Diff detects nodes present only in the newer graph."""
    from knowledge_graph import GraphDiff

    kg1 = KnowledgeGraph(tmp_path / "a.json")
    kg2 = KnowledgeGraph(tmp_path / "b.json")
    kg2.add_node("radar", type="concept", label="Radar")
    kg2.add_node("sar", type="technology", label="SAR")

    diff = kg1.diff(kg2)
    assert len(diff.nodes_added) == 2
    assert len(diff.nodes_removed) == 0
    added_ids = {n["node_id"] for n in diff.nodes_added}
    assert added_ids == {"radar", "sar"}
    assert diff.has_changes


def test_graph_diff_nodes_removed(tmp_path):
    """Diff detects nodes present only in the older graph."""
    kg1 = KnowledgeGraph(tmp_path / "a.json")
    kg1.add_node("radar", type="concept", label="Radar")

    kg2 = KnowledgeGraph(tmp_path / "b.json")

    diff = kg1.diff(kg2)
    assert len(diff.nodes_removed) == 1
    assert diff.nodes_removed[0]["node_id"] == "radar"


def test_graph_diff_nodes_modified(tmp_path):
    """Diff detects field-level changes for nodes in both graphs."""
    kg1 = KnowledgeGraph(tmp_path / "a.json")
    kg1.add_node("radar", type="concept", label="Radar", confidence=0.5)

    kg2 = KnowledgeGraph(tmp_path / "b.json")
    kg2.add_node("radar", type="concept", label="Radar System", confidence=0.9)

    diff = kg1.diff(kg2)
    assert len(diff.nodes_modified) == 1
    mod = diff.nodes_modified[0]
    assert mod["node_id"] == "radar"
    assert "label" in mod["changes"]
    assert mod["changes"]["label"]["old"] == "Radar"
    assert mod["changes"]["label"]["new"] == "Radar System"
    assert "confidence" in mod["changes"]


def test_graph_diff_edges(tmp_path):
    """Diff detects added, removed, and modified edges."""
    kg1 = KnowledgeGraph(tmp_path / "a.json")
    kg1.add_node("a", label="A")
    kg1.add_node("b", label="B")
    kg1.add_node("c", label="C")
    kg1.add_edge("a", "b", relation="is_a", confidence=0.5)
    kg1.add_edge("a", "c", relation="depends_on")

    kg2 = KnowledgeGraph(tmp_path / "b.json")
    kg2.add_node("a", label="A")
    kg2.add_node("b", label="B")
    kg2.add_node("d", label="D")
    kg2.add_edge("a", "b", relation="is_a", confidence=0.9)  # modified
    kg2.add_edge("a", "d", relation="uses")  # added

    diff = kg1.diff(kg2)
    assert len(diff.edges_added) == 1  # a->d uses
    assert len(diff.edges_removed) == 1  # a->c depends_on
    assert len(diff.edges_modified) == 1  # a->b is_a confidence change
    assert diff.edges_modified[0]["changes"]["confidence"]["old"] == 0.5
    assert diff.edges_modified[0]["changes"]["confidence"]["new"] == 0.9


def test_graph_diff_proposals(tmp_path):
    """Diff detects added and changed proposals."""
    kg1 = KnowledgeGraph(tmp_path / "a.json")
    kg1.propose_relation("validates", justification="test",
                         source_entity="a", target_entity="b")

    kg2 = KnowledgeGraph(tmp_path / "b.json")
    kg2.propose_relation("validates", justification="test",
                         source_entity="a", target_entity="b")
    kg2.accept_proposal("validates")
    kg2.propose_relation("measures", justification="test2",
                         source_entity="c", target_entity="d")

    diff = kg1.diff(kg2)
    assert len(diff.proposals_added) == 1
    assert diff.proposals_added[0]["name"] == "measures"
    assert len(diff.proposals_changed) == 1
    assert diff.proposals_changed[0]["name"] == "validates"
    assert diff.proposals_changed[0]["new_status"] == "accepted"


def test_snapshot_and_diff_from_snapshot(tmp_path):
    """snapshot() captures state; diff_from_snapshot() detects mutations."""
    kg = KnowledgeGraph(tmp_path / "a.json")
    kg.add_node("radar", label="Radar")
    snap = kg.snapshot()

    kg.add_node("sar", label="SAR")
    kg.add_edge("sar", "radar", relation="is_a")

    diff = kg.diff_from_snapshot(snap)
    assert len(diff.nodes_added) == 1
    assert diff.nodes_added[0]["node_id"] == "sar"
    assert len(diff.edges_added) == 1


def test_diff_from_file(tmp_path):
    """diff_from_file() loads a saved graph and diffs against current state."""
    kg1 = KnowledgeGraph(tmp_path / "base.json")
    kg1.add_node("a", label="A")
    kg1.save()

    kg2 = KnowledgeGraph(tmp_path / "current.json")
    kg2.add_node("a", label="A")
    kg2.add_node("b", label="B")

    diff = kg2.diff_from_file(tmp_path / "base.json")
    assert len(diff.nodes_added) == 1
    assert diff.nodes_added[0]["node_id"] == "b"


def test_graph_diff_to_dict(tmp_path):
    """GraphDiff.to_dict() returns serializable structure with counts."""
    from knowledge_graph import GraphDiff

    kg1 = KnowledgeGraph(tmp_path / "a.json")
    kg2 = KnowledgeGraph(tmp_path / "b.json")
    kg2.add_node("x", label="X")

    diff = kg1.diff(kg2)
    d = diff.to_dict()
    assert d["has_changes"] is True
    assert d["counts"]["nodes_added"] == 1
    assert isinstance(d["summary"], str)
    # Verify it's JSON-serializable
    json.dumps(d)


def test_ingest_markdown_includes_diff(tmp_path):
    """ingest_markdown() aggregate stats include a diff of changes."""
    kg = KnowledgeGraph(tmp_path / "a.json")

    def mock_extract(prompt):
        return [
            {"source": "Radar", "target": "Radio waves",
             "relation": "uses", "confidence": 0.9,
             "context": "Radar uses radio waves for detection and ranging."},
        ]

    md = _long_section("Radar Basics",
                       "Radar uses radio waves for detection and ranging.")
    stats = kg.ingest_markdown(md, "test-doc", llm_extract_fn=mock_extract)

    assert "diff" in stats
    diff = stats["diff"]
    assert diff["has_changes"] is True
    assert diff["counts"]["nodes_added"] > 0


def test_ingest_markdown_timing_metrics(tmp_path):
    """ingest_markdown() aggregate stats include timing metrics."""
    kg = KnowledgeGraph(tmp_path / "a.json")

    def mock_extract(prompt):
        return [
            {"source": "Radar", "target": "Radio waves",
             "relation": "uses", "confidence": 0.9,
             "context": "Radar uses radio waves."},
        ]

    md = _long_section("Radar Basics",
                       "Radar uses radio waves for detection and ranging.")
    stats = kg.ingest_markdown(md, "test-doc", llm_extract_fn=mock_extract)

    # Overall timing
    assert "elapsed_seconds" in stats
    assert isinstance(stats["elapsed_seconds"], float)
    assert stats["elapsed_seconds"] >= 0

    # Extraction timing
    assert "extraction_seconds" in stats
    assert isinstance(stats["extraction_seconds"], float)
    assert stats["extraction_seconds"] >= 0
    assert stats["extraction_seconds"] <= stats["elapsed_seconds"]

    # Per-section timing
    for section in stats["sections"]:
        assert "elapsed_seconds" in section
        assert section["elapsed_seconds"] >= 0


# ---------------------------------------------------------------------------
# BM25 & Hybrid search
# ---------------------------------------------------------------------------


def _make_search_graph(tmp_path):
    """Create a graph with varied text for BM25 testing."""
    kg = KnowledgeGraph(tmp_path / "search.json")
    kg.add_node("radar", type="concept", label="Radar",
                properties={"description": "Radio detection and ranging system"})
    kg.add_node("lidar", type="concept", label="LiDAR",
                properties={"description": "Light detection and ranging using laser pulses"})
    kg.add_node("sonar", type="concept", label="Sonar",
                properties={"description": "Sound navigation and ranging underwater"})
    kg.add_node("antenna", type="component", label="Antenna",
                properties={"description": "Electromagnetic transducer for radio waves"})
    kg.add_node("signal", type="concept", label="Signal Processing",
                properties={"description": "Mathematical analysis and transformation of signals"})
    return kg


def test_bm25_search_basic(tmp_path):
    """BM25 search finds nodes by keyword match."""
    kg = _make_search_graph(tmp_path)
    results = kg.bm25_search("radio detection ranging", top_k=5)
    assert len(results) > 0
    # "radar" should rank highest — its label and description contain all terms
    assert results[0][0] == "radar"
    # Score should be positive
    assert results[0][1] > 0


def test_bm25_search_exact_keyword(tmp_path):
    """BM25 finds exact keyword matches that semantic search might miss."""
    kg = _make_search_graph(tmp_path)
    results = kg.bm25_search("laser pulses", top_k=5)
    assert len(results) > 0
    assert results[0][0] == "lidar"


def test_bm25_search_type_filter(tmp_path):
    """BM25 respects node type filter."""
    kg = _make_search_graph(tmp_path)
    results = kg.bm25_search("radio waves", top_k=10, node_types=["component"])
    for nid, _ in results:
        assert kg.get_node(nid)["type"] == "component"


def test_bm25_search_confidence_filter(tmp_path):
    """BM25 respects min confidence filter."""
    kg = KnowledgeGraph(tmp_path / "conf.json")
    kg.add_node("low", label="Low confidence radar", confidence=0.2,
                properties={"description": "radar system"})
    kg.add_node("high", label="High confidence radar", confidence=0.9,
                properties={"description": "radar system"})
    results = kg.bm25_search("radar system", top_k=10, min_confidence=0.5)
    result_ids = {nid for nid, _ in results}
    assert "high" in result_ids
    assert "low" not in result_ids


def test_bm25_search_empty_query(tmp_path):
    """BM25 with stopwords-only query returns empty."""
    kg = _make_search_graph(tmp_path)
    results = kg.bm25_search("the and or", top_k=5)
    assert results == []


def test_bm25_search_no_match(tmp_path):
    """BM25 with unmatched terms returns empty."""
    kg = _make_search_graph(tmp_path)
    results = kg.bm25_search("quantum entanglement photon", top_k=5)
    assert results == []


def test_bm25_index_rebuild(tmp_path):
    """BM25 index auto-rebuilds when graph is dirty."""
    kg = _make_search_graph(tmp_path)
    # First search builds index
    results1 = kg.bm25_search("underwater", top_k=1)
    assert results1[0][0] == "sonar"

    # Add a node with "underwater" — makes graph dirty
    kg.add_node("submarine", label="Submarine",
                properties={"description": "Underwater vessel for naval operations"})
    # Search should find the new node
    results2 = kg.bm25_search("underwater", top_k=5)
    result_ids = {nid for nid, _ in results2}
    assert "submarine" in result_ids


def test_search_bm25_mode(tmp_path):
    """search() with mode='bm25' uses BM25 scoring."""
    kg = _make_search_graph(tmp_path)
    results = kg.search("laser pulses", mode="bm25", top_k=3)
    assert len(results) > 0
    assert results[0]["node_id"] == "lidar"
    assert "similarity" in results[0]


def test_search_hybrid_mode(tmp_path):
    """search() with mode='hybrid' blends BM25 and semantic scores."""
    import math
    kg = _make_search_graph(tmp_path)

    # Set up simple embeddings for hybrid search
    def _unit(vec):
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec

    kg.set_embedding("radar", _unit([1.0, 0.0, 0.0, 0.0]))
    kg.set_embedding("lidar", _unit([0.8, 0.6, 0.0, 0.0]))
    kg.set_embedding("sonar", _unit([0.5, 0.5, 0.5, 0.0]))
    kg.set_embedding("antenna", _unit([0.3, 0.3, 0.3, 0.7]))
    kg.set_embedding("signal", _unit([0.0, 0.0, 0.5, 0.8]))

    query_vec = _unit([0.95, 0.1, 0.0, 0.0])

    def fake_embed(texts):
        return [query_vec for _ in texts]

    results = kg.search("radio detection", fake_embed, mode="hybrid", top_k=5, alpha=0.5)
    assert len(results) > 0
    # Should get results from both BM25 (keyword match) and semantic (embedding similarity)
    result_ids = {r["node_id"] for r in results}
    assert "radar" in result_ids  # should be top — matches both BM25 and semantic


def test_search_hybrid_alpha_zero_is_pure_bm25(tmp_path):
    """Hybrid search with alpha=0 produces same ranking as pure BM25."""
    kg = _make_search_graph(tmp_path)
    bm25_results = kg.search("laser pulses", mode="bm25", top_k=5)
    hybrid_results = kg.search("laser pulses", mode="hybrid", top_k=5, alpha=0.0)
    # Same top result
    if bm25_results and hybrid_results:
        assert bm25_results[0]["node_id"] == hybrid_results[0]["node_id"]


def test_search_bm25_requires_string(tmp_path):
    """BM25 mode raises ValueError when given a vector."""
    kg = _make_search_graph(tmp_path)
    with pytest.raises(ValueError, match="text query"):
        kg.search([1.0, 0.0], mode="bm25")


# ---------------------------------------------------------------------------
# Bulk Proposal Management (Item 9)
# ---------------------------------------------------------------------------


def _make_proposals_graph(tmp_path):
    """Helper: create a graph with several proposals."""
    kg = KnowledgeGraph(tmp_path / "proposals.json")
    kg.propose_relation("validates", justification="X validates Y",
                        source_entity="a", target_entity="b")
    kg.propose_relation("monitors", justification="A monitors B",
                        source_entity="c", target_entity="d")
    kg.propose_relation("triggers", justification="X triggers Y",
                        source_entity="e", target_entity="f")
    # Add extra examples to boost confidence
    kg.propose_relation("validates", source_entity="g", target_entity="h")
    kg.propose_relation("validates", source_entity="i", target_entity="j")
    return kg


def test_bulk_accept_proposals(tmp_path):
    """bulk_accept_proposals accepts multiple at once."""
    kg = _make_proposals_graph(tmp_path)
    accepted = kg.bulk_accept_proposals(["validates", "monitors"])
    assert set(accepted) == {"validates", "monitors"}
    assert "validates" in kg._custom_relations
    assert "monitors" in kg._custom_relations
    # Only triggers should remain pending
    pending = kg.get_proposals()
    assert len(pending) == 1
    assert pending[0].name == "triggers"


def test_bulk_accept_with_review_note(tmp_path):
    """bulk_accept_proposals sets review_note on accepted proposals."""
    kg = _make_proposals_graph(tmp_path)
    kg.bulk_accept_proposals(["validates"], review_note="looks good")
    p = [p for p in kg._proposals if p.name == "validates"][0]
    assert p.review_note == "looks good"
    assert p.status == "accepted"


def test_bulk_accept_nonexistent(tmp_path):
    """bulk_accept_proposals skips names that don't exist."""
    kg = _make_proposals_graph(tmp_path)
    accepted = kg.bulk_accept_proposals(["validates", "nonexistent"])
    assert accepted == ["validates"]


def test_bulk_reject_proposals(tmp_path):
    """bulk_reject_proposals rejects multiple at once."""
    kg = _make_proposals_graph(tmp_path)
    rejected = kg.bulk_reject_proposals(["validates", "triggers"])
    assert set(rejected) == {"validates", "triggers"}
    pending = kg.get_proposals()
    assert len(pending) == 1
    assert pending[0].name == "monitors"


def test_bulk_reject_nonexistent(tmp_path):
    """bulk_reject_proposals skips names that don't exist."""
    kg = _make_proposals_graph(tmp_path)
    rejected = kg.bulk_reject_proposals(["nonexistent"])
    assert rejected == []


def test_accept_all_max_accept(tmp_path):
    """accept_all_proposals respects max_accept safety cap."""
    kg = _make_proposals_graph(tmp_path)
    # All three are pending; validates has 3 examples so high confidence
    accepted = kg.accept_all_proposals(min_confidence=0.0, min_examples=0, max_accept=2)
    assert len(accepted) == 2


def test_merge_proposals(tmp_path):
    """merge_proposals combines examples and rejects others."""
    kg = _make_proposals_graph(tmp_path)
    # Add a similar proposal
    kg.propose_relation("validating", justification="same as validates",
                        source_entity="k", target_entity="l")

    merged = kg.merge_proposals(["validates", "validating"], target_name="validates")
    assert merged.name == "validates"
    # Should have examples from both
    assert len(merged.examples) >= 4  # 3 from validates + 1 from validating
    assert merged.status == "pending"

    # The other proposal should be rejected
    validating = [p for p in kg._proposals if p.name == "validating"][0]
    assert validating.status == "rejected"
    assert "merged into" in validating.review_note


def test_merge_proposals_too_few(tmp_path):
    """merge_proposals raises ValueError with fewer than 2 proposals."""
    kg = _make_proposals_graph(tmp_path)
    with pytest.raises(ValueError, match="at least 2"):
        kg.merge_proposals(["validates"], target_name="validates")


def test_merge_proposals_new_target(tmp_path):
    """merge_proposals can create a new target proposal."""
    kg = _make_proposals_graph(tmp_path)
    merged = kg.merge_proposals(
        ["validates", "monitors"], target_name="checks"
    )
    assert merged.name == "checks"
    assert merged.status == "pending"
    # Both originals should be rejected
    for p in kg._proposals:
        if p.name in ("validates", "monitors"):
            assert p.status == "rejected"


def test_find_similar_proposals(tmp_path):
    """find_similar_proposals groups proposals with similar names."""
    kg = KnowledgeGraph(tmp_path / "sim.json")
    kg.propose_relation("related-to", source_entity="a", target_entity="b")
    kg.propose_relation("relates-to", source_entity="c", target_entity="d")
    kg.propose_relation("something-else", source_entity="e", target_entity="f")

    groups = kg.find_similar_proposals(threshold=0.7)
    assert len(groups) >= 1
    # related-to and relates-to should be in the same group
    group_names = [frozenset(p.name for p in g) for g in groups]
    assert any({"related-to", "relates-to"}.issubset(names) for names in group_names)


def test_find_similar_proposals_none(tmp_path):
    """find_similar_proposals returns empty when names are very different."""
    kg = KnowledgeGraph(tmp_path / "nosim.json")
    kg.propose_relation("abcdefgh", source_entity="a", target_entity="b")
    kg.propose_relation("xyz", source_entity="c", target_entity="d")
    groups = kg.find_similar_proposals(threshold=0.9)
    assert groups == []


def test_normalized_edit_distance():
    """Static method _normalized_edit_distance works correctly."""
    assert KnowledgeGraph._normalized_edit_distance("abc", "abc") == 0.0
    assert KnowledgeGraph._normalized_edit_distance("", "abc") == 1.0
    assert KnowledgeGraph._normalized_edit_distance("abc", "") == 1.0
    dist = KnowledgeGraph._normalized_edit_distance("kitten", "sitting")
    assert 0.0 < dist < 1.0


# ---------------------------------------------------------------------------
# Structured Graph Query Language (Item 5)
# ---------------------------------------------------------------------------

from knowledge_graph import (
    _parse_graph_query, QueryParseError, NodeFilter, EdgeFilter, QueryStep,
)


def _make_query_graph(tmp_path):
    """Build a small graph for pattern query tests."""
    kg = KnowledgeGraph(tmp_path / "query.json")
    kg.add_node("python", type="technology", label="Python", confidence=0.9)
    kg.add_node("javascript", type="technology", label="JavaScript", confidence=0.8)
    kg.add_node("fastapi", type="library", label="FastAPI", confidence=0.9)
    kg.add_node("react", type="library", label="React", confidence=0.85)
    kg.add_node("web-dev", type="concept", label="Web Development", confidence=0.95)
    kg.add_node("database", type="concept", label="Database", confidence=0.9)
    kg.add_node("postgres", type="technology", label="PostgreSQL", confidence=0.9)

    kg.add_edge("fastapi", "python", relation="depends_on", confidence=0.95)
    kg.add_edge("react", "javascript", relation="depends_on", confidence=0.9)
    kg.add_edge("fastapi", "web-dev", relation="related_to", confidence=0.8)
    kg.add_edge("react", "web-dev", relation="related_to", confidence=0.85)
    kg.add_edge("python", "web-dev", relation="is_a", confidence=0.7)
    kg.add_edge("postgres", "database", relation="is_a", confidence=0.9)
    kg.add_edge("fastapi", "postgres", relation="uses", confidence=0.7)
    return kg


# --- Parser tests ---


def test_parse_simple_two_hop():
    """Parse a basic two-hop pattern."""
    steps, conf, depth = _parse_graph_query('(type:library) -[depends_on]-> (*)')
    assert len(steps) == 2
    assert steps[0].node_filter.type == "library"
    assert steps[0].edge_filter.relations == ["depends_on"]
    assert steps[0].direction == "out"
    assert steps[1].node_filter.wildcard is True
    assert conf is None
    assert depth is None


def test_parse_wildcard_edge():
    """Parse pattern with wildcard edge."""
    steps, _, _ = _parse_graph_query('(*) -[*]-> (*)')
    assert steps[0].node_filter.wildcard is True
    assert steps[0].edge_filter.wildcard is True


def test_parse_edge_alternatives():
    """Parse pattern with alternative relations."""
    steps, _, _ = _parse_graph_query('(*) -[depends_on|uses]-> (*)')
    assert steps[0].edge_filter.relations == ["depends_on", "uses"]


def test_parse_label_exact():
    """Parse label:value node filter."""
    steps, _, _ = _parse_graph_query('(label:"FastAPI") -[*]-> (*)')
    assert steps[0].node_filter.label == "FastAPI"


def test_parse_label_glob():
    """Parse label:~"pattern" node filter."""
    steps, _, _ = _parse_graph_query('(label:~"Fast*") -[*]-> (*)')
    assert steps[0].node_filter.label_glob == "Fast*"


def test_parse_id_filter():
    """Parse id:value node filter."""
    steps, _, _ = _parse_graph_query('(id:"python") -[*]-> (*)')
    assert steps[0].node_filter.node_id == "python"


def test_parse_where_clause():
    """Parse WHERE confidence > N."""
    steps, conf, depth = _parse_graph_query(
        '(*) -[*]-> (*) WHERE confidence > 0.7'
    )
    assert conf == 0.7
    assert len(steps) == 2


def test_parse_depth_clause():
    """Parse DEPTH N modifier."""
    steps, conf, depth = _parse_graph_query(
        '(*) -[is_a]-> (*) DEPTH 3'
    )
    assert depth == 3
    assert conf is None


def test_parse_where_and_depth():
    """Parse both WHERE and DEPTH."""
    steps, conf, depth = _parse_graph_query(
        '(*) -[*]-> (*) WHERE confidence > 0.5 DEPTH 2'
    )
    assert conf == 0.5
    assert depth == 2


def test_parse_backward_edge():
    """Parse backward edge <-[rel]-."""
    steps, _, _ = _parse_graph_query('(*) <-[depends_on]- (*)')
    assert steps[0].direction == "in"
    assert steps[0].edge_filter.relations == ["depends_on"]


def test_parse_undirected_edge():
    """Parse undirected edge --[rel]--."""
    steps, _, _ = _parse_graph_query('(*) --[related_to]-- (*)')
    assert steps[0].direction == "any"


def test_parse_empty_query():
    """Empty query raises QueryParseError."""
    with pytest.raises(QueryParseError):
        _parse_graph_query("")


def test_parse_bad_syntax():
    """Malformed query raises QueryParseError with position."""
    with pytest.raises(QueryParseError):
        _parse_graph_query("not a query")


def test_parse_single_node():
    """Parse single-node pattern (no edge)."""
    steps, _, _ = _parse_graph_query('(type:technology)')
    assert len(steps) == 1
    assert steps[0].node_filter.type == "technology"


def test_parse_three_hop():
    """Parse a three-hop pattern."""
    steps, _, _ = _parse_graph_query(
        '(type:library) -[depends_on]-> (*) -[is_a]-> (type:concept)'
    )
    assert len(steps) == 3
    assert steps[0].node_filter.type == "library"
    assert steps[0].edge_filter.relations == ["depends_on"]
    assert steps[1].node_filter.wildcard is True
    assert steps[1].edge_filter.relations == ["is_a"]
    assert steps[2].node_filter.type == "concept"


# --- Execution tests ---


def test_query_single_node_type(tmp_path):
    """Single-node query matches by type."""
    kg = _make_query_graph(tmp_path)
    paths = kg.graph_query('(type:technology)')
    node_ids = [p[0]["node_id"] for p in paths]
    assert "python" in node_ids
    assert "javascript" in node_ids
    assert "postgres" in node_ids
    # libraries should NOT be in results
    assert "fastapi" not in node_ids


def test_query_two_hop_depends_on(tmp_path):
    """Two-hop query: library -[depends_on]-> technology."""
    kg = _make_query_graph(tmp_path)
    paths = kg.graph_query('(type:library) -[depends_on]-> (type:technology)')
    assert len(paths) >= 2
    # Each path: [node_dict, edge_dict, node_dict]
    for path in paths:
        assert len(path) == 3
        assert path[0]["type"] == "library"
        assert path[1]["relation"] == "depends_on"
        assert path[2]["type"] == "technology"


def test_query_wildcard_edge(tmp_path):
    """Wildcard edge matches any relation."""
    kg = _make_query_graph(tmp_path)
    paths = kg.graph_query('(id:"fastapi") -[*]-> (*)')
    # FastAPI has edges: depends_on->python, related_to->web-dev, uses->postgres
    assert len(paths) >= 3
    relations = {p[1]["relation"] for p in paths}
    assert "depends_on" in relations
    assert "related_to" in relations


def test_query_edge_alternatives(tmp_path):
    """Edge alternative filter [rel1|rel2] works."""
    kg = _make_query_graph(tmp_path)
    paths = kg.graph_query('(*) -[depends_on|uses]-> (*)')
    relations = {p[1]["relation"] for p in paths}
    assert relations <= {"depends_on", "uses"}
    assert len(paths) >= 3  # fastapi->python, react->js, fastapi->postgres


def test_query_label_glob(tmp_path):
    """label:~"pattern" glob matching works."""
    kg = _make_query_graph(tmp_path)
    paths = kg.graph_query('(label:~"*Script") -[*]-> (*)')
    # JavaScript matches *Script
    assert len(paths) == 0  # JavaScript has no outgoing edges

    paths = kg.graph_query('(*) -[*]-> (label:~"*Script")')
    # react -> javascript via depends_on
    assert len(paths) >= 1


def test_query_where_confidence(tmp_path):
    """WHERE confidence > N filters results."""
    kg = _make_query_graph(tmp_path)
    all_paths = kg.graph_query('(*) -[*]-> (*)')
    filtered = kg.graph_query('(*) -[*]-> (*) WHERE confidence > 0.85')
    assert len(filtered) <= len(all_paths)
    # All items in filtered paths should have confidence > 0.85
    for path in filtered:
        for item in path:
            assert item.get("confidence", 1.0) > 0.85


def test_query_depth(tmp_path):
    """DEPTH N for variable-length path traversal."""
    kg = _make_query_graph(tmp_path)
    # fastapi -[*]-> postgres -[is_a]-> database (2 hops)
    paths = kg.graph_query('(id:"fastapi") -[*]-> (id:"database") DEPTH 3')
    # Should find: fastapi -> postgres -> database (2 hops, within depth 3)
    assert len(paths) >= 1
    # Verify at least one path goes through postgres
    for path in paths:
        assert path[0]["node_id"] == "fastapi"
        assert path[-1]["node_id"] == "database"


def test_query_backward_edge(tmp_path):
    """Backward edge <-[rel]- traverses incoming edges."""
    kg = _make_query_graph(tmp_path)
    paths = kg.graph_query('(type:technology) <-[depends_on]- (type:library)')
    assert len(paths) >= 2
    for path in paths:
        assert path[0]["type"] == "technology"
        assert path[1]["relation"] == "depends_on"
        assert path[2]["type"] == "library"


def test_query_undirected_edge(tmp_path):
    """Undirected edge --[rel]-- matches both directions."""
    kg = _make_query_graph(tmp_path)
    paths = kg.graph_query('(id:"web-dev") --[related_to]-- (*)')
    # web-dev has incoming related_to from fastapi and react
    assert len(paths) >= 2


def test_query_three_hop(tmp_path):
    """Three-hop traversal: library -> technology -> concept."""
    kg = _make_query_graph(tmp_path)
    paths = kg.graph_query(
        '(type:library) -[depends_on]-> (*) -[is_a]-> (type:concept)'
    )
    # fastapi -> python -> web-dev (is_a concept)
    assert len(paths) >= 1
    for path in paths:
        assert len(path) == 5  # node, edge, node, edge, node
        assert path[0]["type"] == "library"
        assert path[4]["type"] == "concept"


def test_query_no_results(tmp_path):
    """Query that matches nothing returns empty list."""
    kg = _make_query_graph(tmp_path)
    paths = kg.graph_query('(type:nonexistent) -[*]-> (*)')
    assert paths == []


def test_query_limit(tmp_path):
    """Limit parameter caps results."""
    kg = _make_query_graph(tmp_path)
    paths = kg.graph_query('(*) -[*]-> (*)', limit=2)
    assert len(paths) <= 2


def test_query_parse_error(tmp_path):
    """Bad query raises QueryParseError."""
    kg = _make_query_graph(tmp_path)
    with pytest.raises(QueryParseError):
        kg.graph_query("this is not a valid query")


# ---------------------------------------------------------------------------
# Checkpoint and resume
# ---------------------------------------------------------------------------


def test_checkpoint_save_load_clear(tmp_path):
    """save_checkpoint / load_checkpoint / clear_checkpoint round-trip."""
    kg = KnowledgeGraph(tmp_path / "ck.json")
    assert kg.load_checkpoint() is None

    data = {"doc_id": "doc1", "content_hash": "abc123", "completed_sections": {"Intro": []}}
    kg.save_checkpoint(data)
    loaded = kg.load_checkpoint()
    assert loaded is not None
    assert loaded["doc_id"] == "doc1"
    assert "Intro" in loaded["completed_sections"]

    assert kg.clear_checkpoint() is True
    assert kg.load_checkpoint() is None
    assert kg.clear_checkpoint() is False


def test_checkpoint_corrupt_file(tmp_path):
    """load_checkpoint handles corrupt files gracefully."""
    kg = KnowledgeGraph(tmp_path / "ck.json")
    kg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    kg.checkpoint_path.write_text("NOT JSON", encoding="utf-8")
    assert kg.load_checkpoint() is None


def test_checkpoint_full_ingestion(tmp_path):
    """checkpoint=True saves checkpoints and clears on completion."""
    kg = KnowledgeGraph(tmp_path / "ck.json")

    call_count = 0

    def mock_extract(prompt):
        nonlocal call_count
        call_count += 1
        return [
            {"source": "Radar", "target": "Radio waves",
             "relation": "uses", "confidence": 0.9,
             "context": "Radar uses radio waves."},
        ]

    md = _long_section("Basics", "Radar uses radio waves for detection and ranging.")
    stats = kg.ingest_markdown(md, "doc1", llm_extract_fn=mock_extract, checkpoint=True)

    assert stats["total_triples"] > 0
    assert call_count == 1
    # Checkpoint should be cleared on success
    assert kg.load_checkpoint() is None


def test_checkpoint_resume_skips_completed(tmp_path):
    """Sections completed in a checkpoint are skipped on resume."""
    kg = KnowledgeGraph(tmp_path / "ck.json")

    call_count = 0

    def mock_extract(prompt):
        nonlocal call_count
        call_count += 1
        return [
            {"source": "Radar", "target": "Radio waves",
             "relation": "uses", "confidence": 0.9,
             "context": "Radar uses radio waves."},
        ]

    md = (_long_section("Intro", "Radar uses radio waves for detection.")
          + "\n" + _long_section("Details", "Details about radar systems and antennas."))

    # First ingestion — completes fully
    stats1 = kg.ingest_markdown(md, "doc1", llm_extract_fn=mock_extract, checkpoint=True)
    first_count = call_count
    assert first_count == 2  # Both sections extracted
    assert kg.load_checkpoint() is None  # cleared

    # Simulate a partially completed ingestion by manually writing a checkpoint
    from knowledge_graph import content_hash
    chash = content_hash(md)
    kg.save_checkpoint({
        "doc_id": "doc1",
        "content_hash": chash,
        "completed_sections": {"Intro": []},
        "section_stats": {
            "Intro": {
                "triples_processed": 1, "nodes_added": 2, "nodes_updated": 0,
                "edges_added": 1, "edges_updated": 0,
                "proposals_created": 0, "proposals_augmented": 0,
            },
        },
    })

    # Second ingestion with checkpoint — should skip "Intro", only extract "Details"
    call_count = 0
    stats2 = kg.ingest_markdown(md, "doc1", llm_extract_fn=mock_extract, checkpoint=True)

    assert call_count == 1  # Only "Details" was extracted
    assert stats2.get("sections_resumed", 0) == 1
    assert kg.load_checkpoint() is None  # cleared after success


def test_checkpoint_stale_discarded(tmp_path):
    """A checkpoint for a different doc/hash is discarded."""
    kg = KnowledgeGraph(tmp_path / "ck.json")

    def mock_extract(prompt):
        return [{"source": "A", "target": "B", "relation": "related_to",
                 "confidence": 0.8, "context": "A relates to B."}]

    # Write a checkpoint for a different document
    kg.save_checkpoint({
        "doc_id": "other-doc",
        "content_hash": "different",
        "completed_sections": {"Intro": []},
        "section_stats": {},
    })

    md = _long_section("Basics", "Information about basics of this topic.")
    stats = kg.ingest_markdown(md, "doc1", llm_extract_fn=mock_extract, checkpoint=True)

    # Should proceed normally (stale checkpoint discarded)
    assert stats["total_triples"] > 0
    assert stats.get("sections_resumed", 0) == 0
