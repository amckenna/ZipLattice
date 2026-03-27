"""
mcp_server.py — MCP Server for ZipLattice Knowledge Graphs

Exposes ZipLattice graph operations as MCP tools so that an AI orchestrator
(e.g. Claude Code) can manage knowledge graphs directly.  The key design
insight is that the orchestrator *is* an LLM, so it can perform entity
extraction itself rather than requiring a second LLM backend.  The server
provides ``build_extraction_prompt`` to give the orchestrator the schema-
aware prompt, and ``ingest_triples`` to accept the structured triples back.

Requires: ``pip install fastmcp networkx``

Usage:
    # stdio transport (for Claude Code / MCP clients)
    python mcp_server.py

    # or with uvicorn for SSE transport
    uvicorn mcp_server:mcp.sse_app() --host 0.0.0.0 --port 8100

Configure in Claude Code's MCP settings::

    {
      "mcpServers": {
        "ziplattice": {
          "command": "python",
          "args": ["mcp_server.py"],
          "cwd": "/path/to/ZipLattice"
        }
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from knowledge_graph import KnowledgeGraph, GraphEncoder, ollama_embed

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "ZipLattice",
    instructions=(
        "Knowledge graph management server.  Use build_extraction_prompt to "
        "get a schema-aware extraction prompt, extract entities/relations "
        "yourself from the document text, then call ingest_triples to add "
        "them to the graph.  Use the other tools for querying, navigating, "
        "and managing the graph."
    ),
)

# ---------------------------------------------------------------------------
# Graph instance cache — avoids reloading on every tool call
# ---------------------------------------------------------------------------

_graph_cache: dict[str, KnowledgeGraph] = {}


def _get_graph(graph_path: str) -> KnowledgeGraph:
    """Load (or retrieve from cache) a KnowledgeGraph by path."""
    graph_path = str(Path(graph_path).resolve())
    if graph_path not in _graph_cache:
        logger.info("Loading graph: %s", graph_path)
        try:
            _graph_cache[graph_path] = KnowledgeGraph(graph_path)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error("Failed to load graph '%s': %s", graph_path, exc)
            raise
    return _graph_cache[graph_path]


def _json(obj: Any) -> str:
    """Serialize to compact JSON using the project's custom encoder."""
    return json.dumps(obj, cls=GraphEncoder, default=str)


# =========================================================================
# Ingestion tools — orchestrator-as-extractor pattern
# =========================================================================


@mcp.tool()
def build_extraction_prompt(
    graph_path: str,
    text: str,
    focus_entities: list[str] | None = None,
    max_triples: int = 50,
) -> str:
    """Build a schema-aware entity extraction prompt for a document.

    Returns a prompt string that includes the graph's current relation
    schema, node types, and output format instructions.  The calling
    orchestrator should use this prompt (combined with the document text)
    to extract entities and relations, then pass the result to
    ``ingest_triples``.

    Args:
        graph_path: Path to the knowledge graph JSON file.
        text: Document text to extract from.
        focus_entities: Optional entity names to prioritize.
        max_triples: Maximum triples to request.
    """
    kg = _get_graph(graph_path)
    return kg.build_extraction_prompt(
        text, focus_entities=focus_entities, max_triples=max_triples,
    )


@mcp.tool()
def ingest_triples(
    graph_path: str,
    triples: list[dict[str, Any]],
    text: str,
    doc_id: str,
    auto_save: bool = True,
) -> str:
    """Ingest pre-extracted triples into a knowledge graph.

    This is the core tool for the orchestrator-as-extractor pattern.
    The orchestrator extracts entities/relations from document text
    (using the prompt from ``build_extraction_prompt``) and passes the
    structured triples here for graph mutation.

    Each triple should be a dict with at least ``source``, ``target``,
    and ``relation`` keys.  See ``build_extraction_prompt`` output for
    the full schema.

    Args:
        graph_path: Path to the knowledge graph JSON file.
        triples: List of triple dicts from the orchestrator's extraction.
        text: The original document text (for validation/span tracking).
        doc_id: Unique identifier for the source document.
        auto_save: Save the graph after ingestion (default True).
    """
    kg = _get_graph(graph_path)
    logger.info("ingest_triples: doc_id=%s, triples=%d", doc_id, len(triples))
    stats = kg.ingest_triples(
        triples,
        text=text,
        doc_id=doc_id,
    )
    logger.info(
        "ingest_triples: done — nodes_added=%d, nodes_updated=%d, "
        "edges_added=%d, edges_updated=%d",
        stats.get("nodes_added", 0), stats.get("nodes_updated", 0),
        stats.get("edges_added", 0), stats.get("edges_updated", 0),
    )
    if auto_save:
        kg.save()
    return _json(stats)


@mcp.tool()
def parse_markdown_sections(
    text: str,
    min_section_chars: int = 80,
    max_section_chars: int = 6000,
) -> str:
    """Parse a markdown document into sections split on headings.

    Returns a list of section dicts, each with heading, level, path,
    body, char_count, and content flags.  Useful for processing large
    documents section-by-section through build_extraction_prompt +
    ingest_triples.

    Args:
        text: Raw markdown text.
        min_section_chars: Minimum section length (shorter ones are merged).
        max_section_chars: Maximum section length (longer ones are split).
    """
    sections = KnowledgeGraph.parse_markdown_sections(
        text,
        min_section_chars=min_section_chars,
        max_section_chars=max_section_chars,
    )
    return _json(sections)


@mcp.tool()
def store_source(
    graph_path: str,
    text: str,
    doc_id: str,
    original_path: str | None = None,
    auto_save: bool = True,
) -> str:
    """Store a source document for provenance tracking.

    Stores the document text in the graph's managed sources directory
    with SHA-256 content hashing for deduplication.

    Args:
        graph_path: Path to the knowledge graph JSON file.
        text: Document text to store.
        doc_id: Unique identifier for the document.
        original_path: Original file path (for metadata).
        auto_save: Save after storing (default True).
    """
    kg = _get_graph(graph_path)
    result = kg.store_source(text, doc_id, original_path=original_path)
    if auto_save:
        kg.save()
    return _json(result)


# =========================================================================
# Node tools
# =========================================================================


@mcp.tool()
def add_node(
    graph_path: str,
    node_id: str,
    type: str = "concept",
    label: str | None = None,
    properties: dict[str, Any] | None = None,
    source: str = "manual",
    confidence: float = 1.0,
    auto_save: bool = True,
) -> str:
    """Add or update a node in the knowledge graph.

    Args:
        graph_path: Path to the knowledge graph JSON file.
        node_id: Unique identifier (will be slugified).
        type: Node type (concept, technology, tool, person, etc.).
        label: Human-readable label (defaults to node_id).
        properties: Arbitrary key-value metadata.
        source: Provenance tag.
        confidence: Confidence score [0, 1].
        auto_save: Save after mutation (default True).
    """
    kg = _get_graph(graph_path)
    nid = kg.add_node(
        node_id, type=type, label=label,
        properties=properties or {}, source=source, confidence=confidence,
    )
    if auto_save:
        kg.save()
    return _json({"node_id": nid, "node": kg.get_node(nid)})


@mcp.tool()
def get_node(graph_path: str, node_id: str) -> str:
    """Get a single node by ID.

    Args:
        graph_path: Path to the knowledge graph JSON file.
        node_id: The node ID to look up.
    """
    kg = _get_graph(graph_path)
    node = kg.get_node(node_id)
    if node is None:
        return _json({"error": f"Node '{node_id}' not found"})
    return _json({"node_id": node_id, **node})


@mcp.tool()
def remove_node(graph_path: str, node_id: str, auto_save: bool = True) -> str:
    """Remove a node and its connected edges.

    Args:
        graph_path: Path to the knowledge graph JSON file.
        node_id: The node ID to remove.
        auto_save: Save after mutation (default True).
    """
    kg = _get_graph(graph_path)
    removed = kg.remove_node(node_id)
    if auto_save and removed:
        kg.save()
    return _json({"removed": removed, "node_id": node_id})


@mcp.tool()
def search_nodes(
    graph_path: str,
    type: str | None = None,
    label_contains: str | None = None,
    min_confidence: float = 0.0,
) -> str:
    """Search nodes by type, label substring, or confidence threshold.

    Args:
        graph_path: Path to the knowledge graph JSON file.
        type: Filter by node type.
        label_contains: Filter by label substring (case-insensitive).
        min_confidence: Minimum confidence threshold.
    """
    kg = _get_graph(graph_path)
    results = kg.search_nodes(
        type=type, label_contains=label_contains,
        min_confidence=min_confidence,
    )
    return _json([{"node_id": nid, **data} for nid, data in results])


@mcp.tool()
def get_neighbors(
    graph_path: str,
    node_id: str,
    relation: str | None = None,
    direction: str = "both",
    max_depth: int = 1,
) -> str:
    """Get neighboring nodes up to max_depth hops away.

    Args:
        graph_path: Path to the knowledge graph JSON file.
        node_id: The center node ID.
        relation: Filter by relation type.
        direction: 'outgoing', 'incoming', or 'both'.
        max_depth: Maximum traversal depth.
    """
    kg = _get_graph(graph_path)
    neighbors = kg.get_neighbors(
        node_id, relation=relation, direction=direction, max_depth=max_depth,
    )
    return _json([{"node_id": nid, **data} for nid, data in neighbors])


@mcp.tool()
def get_subgraph(graph_path: str, node_id: str, depth: int = 2) -> str:
    """Extract a local subgraph around a node (for RAG context).

    Args:
        graph_path: Path to the knowledge graph JSON file.
        node_id: Center node ID.
        depth: Traversal depth.
    """
    kg = _get_graph(graph_path)
    subgraph = kg.get_subgraph(node_id, depth=depth)
    return _json(subgraph)


# =========================================================================
# Edge tools
# =========================================================================


@mcp.tool()
def add_edge(
    graph_path: str,
    source: str,
    target: str,
    relation: str = "related_to",
    properties: dict[str, Any] | None = None,
    confidence: float = 1.0,
    auto_save: bool = True,
) -> str:
    """Add an edge between two existing nodes.

    Args:
        graph_path: Path to the knowledge graph JSON file.
        source: Source node ID.
        target: Target node ID.
        relation: Relation type (e.g. depends_on, is_a, uses).
        properties: Arbitrary edge metadata.
        confidence: Confidence score [0, 1].
        auto_save: Save after mutation (default True).
    """
    kg = _get_graph(graph_path)
    edge = kg.add_edge(
        source, target, relation=relation,
        properties=properties or {}, confidence=confidence,
    )
    if auto_save:
        kg.save()
    return _json(edge)


@mcp.tool()
def get_edges(
    graph_path: str,
    node_id: str | None = None,
    relation: str | None = None,
    direction: str = "both",
) -> str:
    """Get edges, optionally filtered by node and/or relation.

    Args:
        graph_path: Path to the knowledge graph JSON file.
        node_id: Filter by this node as source, target, or both.
        relation: Filter by relation type.
        direction: 'outgoing', 'incoming', or 'both'.
    """
    kg = _get_graph(graph_path)
    edges = kg.get_edges(node_id=node_id, relation=relation, direction=direction)
    return _json(edges)


@mcp.tool()
def remove_edges(
    graph_path: str,
    source: str,
    target: str,
    relation: str | None = None,
    auto_save: bool = True,
) -> str:
    """Remove edges between two nodes.

    Args:
        graph_path: Path to the knowledge graph JSON file.
        source: Source node ID.
        target: Target node ID.
        relation: Optional relation type filter.
        auto_save: Save after mutation (default True).
    """
    kg = _get_graph(graph_path)
    count = kg.remove_edges(source, target, relation=relation)
    if auto_save and count > 0:
        kg.save()
    return _json({"removed_count": count})


# =========================================================================
# Graph-level tools
# =========================================================================


@mcp.tool()
def graph_stats(graph_path: str) -> str:
    """Get summary statistics about the knowledge graph.

    Returns node/edge counts, type distributions, connectivity metrics,
    and proposal counts.

    Args:
        graph_path: Path to the knowledge graph JSON file.
    """
    kg = _get_graph(graph_path)
    return _json(kg.stats())


@mcp.tool()
def graph_analytics(graph_path: str) -> str:
    """Get comprehensive quality analytics for a knowledge graph.

    Returns confidence distributions, relation/type stats, hub nodes,
    orphan nodes, embedding coverage, component sizes, and a composite
    quality score (0-100).

    Args:
        graph_path: Path to the knowledge graph JSON file.
    """
    kg = _get_graph(graph_path)
    return _json(kg.analytics())


@mcp.tool()
def validate_graph(graph_path: str) -> str:
    """Run consistency checks on the knowledge graph.

    Detects dangling edges, taxonomic cycles, contradictory edge pairs,
    orphan nodes, zero-confidence items, and missing embeddings.
    Returns a structured report with errors, warnings, and info.

    Args:
        graph_path: Path to the knowledge graph JSON file.
    """
    kg = _get_graph(graph_path)
    report = kg.validate()
    return _json(report.to_dict())


@mcp.tool()
def save_graph(graph_path: str) -> str:
    """Explicitly save the knowledge graph to disk.

    Args:
        graph_path: Path to the knowledge graph JSON file.
    """
    kg = _get_graph(graph_path)
    kg.save()
    return _json({"saved": True, "path": str(kg.graph_path)})


# =========================================================================
# Proposal management tools
# =========================================================================


@mcp.tool()
def list_proposals(
    graph_path: str,
    status: str | None = "pending",
) -> str:
    """List relation proposals (novel relations discovered during ingestion).

    Args:
        graph_path: Path to the knowledge graph JSON file.
        status: Filter by status: 'pending', 'accepted', 'rejected', or
                None for all.
    """
    kg = _get_graph(graph_path)
    proposals = kg.get_proposals(status=status)
    return _json([asdict(p) for p in proposals])


@mcp.tool()
def accept_proposal(
    graph_path: str,
    name: str,
    review_note: str = "",
    auto_save: bool = True,
) -> str:
    """Accept a pending relation proposal.

    Registers the relation type and boosts confidence on edges that
    used it.

    Args:
        graph_path: Path to the knowledge graph JSON file.
        name: The proposed relation name.
        review_note: Optional review note.
        auto_save: Save after acceptance (default True).
    """
    kg = _get_graph(graph_path)
    accepted = kg.accept_proposal(name, review_note=review_note)
    if auto_save and accepted:
        kg.save()
    return _json({"accepted": accepted, "relation": name})


@mcp.tool()
def reject_proposal(
    graph_path: str,
    name: str,
    review_note: str = "",
    auto_save: bool = True,
) -> str:
    """Reject a pending relation proposal.

    Args:
        graph_path: Path to the knowledge graph JSON file.
        name: The proposed relation name.
        review_note: Optional review note.
        auto_save: Save after rejection (default True).
    """
    kg = _get_graph(graph_path)
    rejected = kg.reject_proposal(name, review_note=review_note)
    if auto_save and rejected:
        kg.save()
    return _json({"rejected": rejected, "relation": name})


# =========================================================================
# Embedding tools (still uses ollama backend)
# =========================================================================


@mcp.tool()
def embed_nodes(
    graph_path: str,
    embed_model: str = "nomic-embed-text",
    api_url: str = "http://localhost:11434",
    node_types: list[str] | None = None,
    auto_save: bool = True,
) -> str:
    """Generate embeddings for graph nodes using a local embedding server.

    Requires a running Ollama / OpenAI-compatible embedding server.

    Args:
        graph_path: Path to the knowledge graph JSON file.
        embed_model: Embedding model name.
        api_url: Base URL of the embedding server.
        node_types: Only embed nodes of these types (None = all).
        auto_save: Save embeddings after generation (default True).
    """
    kg = _get_graph(graph_path)
    logger.info("embed_nodes: model=%s url=%s", embed_model, api_url)

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return ollama_embed(texts, model=embed_model, url=api_url)

    stats = kg.embed_nodes(embed_fn, node_types=node_types)
    logger.info("embed_nodes: done — %s", {k: v for k, v in stats.items() if isinstance(v, int)})
    if auto_save:
        kg.save_embeddings()
    return _json(stats)


@mcp.tool()
def semantic_search(
    graph_path: str,
    query: str,
    embed_model: str = "nomic-embed-text",
    api_url: str = "http://localhost:11434",
    top_k: int = 10,
    node_types: list[str] | None = None,
    expand_depth: int = 1,
) -> str:
    """Semantic search over the knowledge graph using embeddings.

    Requires pre-computed embeddings (see embed_nodes) and a running
    embedding server for the query vector.

    Args:
        graph_path: Path to the knowledge graph JSON file.
        query: Natural-language search query.
        embed_model: Embedding model name.
        api_url: Base URL of the embedding server.
        top_k: Number of results.
        node_types: Filter by node types.
        expand_depth: Neighborhood expansion depth.
    """
    kg = _get_graph(graph_path)
    logger.info("semantic_search: query=%r model=%s top_k=%d", query[:80], embed_model, top_k)

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return ollama_embed(texts, model=embed_model, url=api_url)

    results = kg.search(
        query, embed_fn, top_k=top_k,
        node_types=node_types, expand_depth=expand_depth,
    )
    logger.info("semantic_search: returned %d results", len(results))
    return _json(results)


@mcp.tool()
def merge_graphs(
    graph_paths: list[str],
    output_path: str,
    prefer: str = "latest",
    description: str = "",
) -> str:
    """Merge two or more knowledge graphs into a new combined graph.

    Creates a new graph at *output_path* containing all nodes, edges,
    embeddings, source documents, and relation proposals from the input
    graphs.  The source graphs are **not** modified.

    Args:
        graph_paths: Paths to existing graph JSON files (minimum 2).
        output_path: Path for the new merged graph JSON file.
        prefer: Conflict resolution: 'latest' (newest timestamp wins),
                'first' (first graph wins), 'last' (last graph wins).
        description: Optional description for the merged graph.
    """
    logger.info("merge_graphs: %d sources → %s (prefer=%s)",
                len(graph_paths), output_path, prefer)
    sources = [_get_graph(p) for p in graph_paths]
    merged = KnowledgeGraph.merge_graphs(
        sources, output_path, prefer=prefer, description=description,
    )
    # Cache the new graph
    _graph_cache[str(Path(output_path).resolve())] = merged
    return _json(merged.stats())


# =========================================================================
# Entry point
# =========================================================================

if __name__ == "__main__":
    mcp.run()
