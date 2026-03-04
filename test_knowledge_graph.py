"""Tests for knowledge_graph.py — core KnowledgeGraph class."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from knowledge_graph import KnowledgeGraph, CoreRelation


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
