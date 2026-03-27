"""Tests for knowledge_graph.py — core KnowledgeGraph class."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from knowledge_graph import (
    KnowledgeGraph,
    CoreRelation,
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
