"""
knowledge_graph.py — Flat-file Knowledge Graph Manager

A portable, JSON-backed knowledge graph built on top of networkx.
Designed for ingesting technical documentation and performing
graph-based RAG for LLM agents.

Usage:
    from knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph("my_graph.json")
    kg.add_node("python", type="technology", label="Python", properties={"domain": "programming"})
    kg.add_node("asyncio", type="concept", label="asyncio", properties={"domain": "concurrency"})
    kg.add_edge("asyncio", "python", relation="part_of")
    kg.save()

    # Query
    neighbors = kg.get_neighbors("python")
    subgraph = kg.get_subgraph("python", depth=2)

    # Embeddings (stored separately)
    kg.set_embedding("python", [0.1, 0.2, ...])
    kg.save_embeddings()
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import time
import urllib.request
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import networkx as nx

logger = logging.getLogger(__name__)


def ollama_embed(
    texts: list[str], *, model: str, url: str = "http://localhost:11434"
) -> list[list[float]]:
    """Call Ollama ``/api/embed`` for a batch of texts.

    Args:
        texts: List of strings to embed.
        model: Ollama model name (e.g. ``nomic-embed-text``).
        url: Ollama server base URL.

    Returns:
        One embedding vector per input text, in the same order.
    """
    endpoint = f"{url.rstrip('/')}/api/embed"
    payload = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    logger.debug("ollama_embed: POST %s  model=%s  texts=%d", endpoint, model, len(texts))
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Embedding request failed (HTTP {exc.code}): "
            f"POST {endpoint} with model '{model}'. "
            f"Check that the Ollama server is running at {url} "
            f"and the model '{model}' is available (ollama list)."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {endpoint}: {exc.reason}. "
            f"Is the server running?"
        ) from exc
    return body["embeddings"]


# ---------------------------------------------------------------------------
# Schema: Relation types
# ---------------------------------------------------------------------------

class CoreRelation(str, Enum):
    """
    Fixed set of common relation types.
    Extend by registering custom relations at runtime via
    KnowledgeGraph.register_relation().
    """
    # Taxonomic / hierarchical
    IS_A = "is_a"                        # "asyncio" is_a "python_module"
    PART_OF = "part_of"                  # "chapter_3" part_of "book"
    HAS_PART = "has_part"                # "book" has_part "chapter_3"
    SUBCLASS_OF = "subclass_of"          # "dict" subclass_of "mapping"
    INSTANCE_OF = "instance_of"          # "my_server" instance_of "linux_host"

    # Dependency / causal
    DEPENDS_ON = "depends_on"            # "app" depends_on "database"
    REQUIRED_BY = "required_by"          # "database" required_by "app"
    CAUSES = "causes"                    # "OOM" causes "crash"
    CAUSED_BY = "caused_by"             # "crash" caused_by "OOM"

    # Associative
    RELATED_TO = "related_to"            # general association
    SIMILAR_TO = "similar_to"            # semantic similarity
    REFERENCES = "references"            # "doc_a" references "doc_b"
    REFERENCED_BY = "referenced_by"      # inverse of references
    IMPLEMENTS = "implements"             # "module" implements "interface"
    EXTENDS = "extends"                  # "child_class" extends "parent_class"

    # Documentation / provenance
    DOCUMENTS = "documents"              # "page" documents "api_endpoint"
    DOCUMENTED_BY = "documented_by"      # "api_endpoint" documented_by "page"
    DERIVED_FROM = "derived_from"        # "summary" derived_from "source_doc"
    SUPERSEDES = "supersedes"            # "v2" supersedes "v1"
    SUPERSEDED_BY = "superseded_by"      # "v1" superseded_by "v2"

    # Functional
    USES = "uses"                        # "script" uses "library"
    USED_BY = "used_by"                  # "library" used_by "script"
    CONFIGURED_BY = "configured_by"      # "service" configured_by "config_file"
    PRODUCES = "produces"                # "pipeline" produces "artifact"
    CONSUMES = "consumes"                # "pipeline" consumes "input_data"

    # Contextual
    BELONGS_TO = "belongs_to"            # "file" belongs_to "project"
    CONTAINS = "contains"                # "project" contains "file"
    TAGGED_WITH = "tagged_with"          # "node" tagged_with "label"


# Default node types — advisory, not enforced unless you want strict mode
DEFAULT_NODE_TYPES = {
    "concept",
    "entity",
    "document",
    "section",
    "technology",
    "tool",
    "process",
    "event",
    "person",
    "organization",
    "code",
    "configuration",
    "artifact",
    "custom",
}


# ---------------------------------------------------------------------------
# JSON Encoder for graph-specific types
# ---------------------------------------------------------------------------

class GraphEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, set):
            return sorted(obj)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Convert text to a URL/ID-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")


def content_hash(text: str) -> str:
    """Deterministic short hash for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Relation Proposal system
# ---------------------------------------------------------------------------

class ProposalStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass
class RelationProposal:
    """
    A suggested new relation type awaiting review.

    Created during document ingestion when the extraction pipeline
    encounters a relationship that doesn't map cleanly to the existing
    schema. Proposals accumulate examples over time — if multiple
    documents independently suggest the same relation, confidence grows
    and the case for acceptance strengthens.
    """
    name: str                                           # snake_case relation name
    justification: str = ""                             # why this relation is needed
    examples: list[dict[str, str]] = field(default_factory=list)  # [{"source": ..., "target": ..., "context": ...}]
    source_docs: list[str] = field(default_factory=list)  # doc IDs that triggered this
    confidence: float = 0.5
    status: str = ProposalStatus.PENDING.value
    proposed_at: str = field(default_factory=now_iso)
    reviewed_at: str | None = None
    review_note: str = ""

    def add_example(self, source: str, target: str, context: str = "", doc_id: str = "") -> None:
        """Add a supporting example and boost confidence."""
        self.examples.append({
            "source": source,
            "target": target,
            "context": context,
        })
        if doc_id and doc_id not in self.source_docs:
            self.source_docs.append(doc_id)
        # More independent examples = higher confidence (diminishing returns)
        n = len(self.examples)
        self.confidence = min(0.95, 0.3 + 0.15 * n)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RelationProposal":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """
    A portable, JSON-backed knowledge graph.

    Data layout on disk:
        graph_path          → nodes + edges + metadata  (single JSON file)
        embeddings_path     → node_id → vector          (separate JSON file)

    In memory the graph is held as a networkx DiGraph for fast traversal,
    with the raw dict kept in sync for serialization.
    """

    # Class-level registry of valid relations (core + user-registered)
    _custom_relations: set[str] = set()

    def __init__(
        self,
        graph_path: str | Path = "knowledge_graph.json",
        embeddings_path: str | Path | None = None,
        sources_dir: str | Path | None = None,
        *,
        strict_relations: bool = False,
        strict_node_types: bool = False,
        auto_timestamp: bool = True,
    ):
        """
        Args:
            graph_path: Path to the main graph JSON file.  All runtime
                        artifacts (embeddings, sources, visualizations)
                        are placed in a dedicated directory named after
                        the graph stem.  For example, ``my_graph.json``
                        becomes ``my_graph/my_graph.json``.
            embeddings_path: Path to the embeddings JSON file.
                             Defaults to ``<graph_dir>/<stem>_embeddings.json``.
            sources_dir: Directory for storing ingested source files.
                         Defaults to ``<graph_dir>/<stem>_sources/``.
            strict_relations: If True, only allow CoreRelation values and
                              explicitly registered custom relations.
            strict_node_types: If True, only allow DEFAULT_NODE_TYPES.
            auto_timestamp: Automatically add created/updated timestamps.
        """
        raw = Path(graph_path)
        # Place the graph file inside a dedicated directory named after
        # the graph stem, unless it already lives in one.
        if raw.parent.name != raw.stem:
            graph_dir = raw.parent / raw.stem
            self.graph_path = graph_dir / raw.name
        else:
            self.graph_path = raw
        self.embeddings_path = Path(
            embeddings_path
            or self.graph_path.with_name(
                f"{self.graph_path.stem}_embeddings.json"
            )
        )
        self.sources_dir = Path(
            sources_dir
            or self.graph_path.with_name(
                f"{self.graph_path.stem}_sources"
            )
        )
        self.strict_relations = strict_relations
        self.strict_node_types = strict_node_types
        self.auto_timestamp = auto_timestamp

        # Internal state
        self._data: dict[str, Any] = self._empty_graph_data()
        self._embeddings: dict[str, list[float]] = {}
        self._embed_meta: dict[str, Any] = {}
        self._proposals: list[RelationProposal] = []
        self._G: nx.DiGraph = nx.DiGraph()
        self._dirty = False
        self._dirty_embeddings = False

        # Load existing files if present
        if self.graph_path.exists():
            self.load()
        if self.embeddings_path.exists():
            self.load_embeddings()

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_graph_data() -> dict[str, Any]:
        return {
            "meta": {
                "version": "1.0.0",
                "created": now_iso(),
                "updated": now_iso(),
                "description": "",
                "core_relations": [r.value for r in CoreRelation],
                "custom_relations": [],
                "node_types": sorted(DEFAULT_NODE_TYPES),
            },
            "nodes": {},
            "edges": [],
            "relation_proposals": [],
        }

    @classmethod
    def register_relation(cls, name: str) -> None:
        """Register a custom relation type available to all instances."""
        cls._custom_relations.add(name)

    def _valid_relations(self) -> set[str]:
        core = {r.value for r in CoreRelation}
        custom_meta = set(self._data.get("meta", {}).get("custom_relations", []))
        return core | self._custom_relations | custom_meta

    def _validate_relation(self, relation: str, *, _skip_auto_register: bool = False) -> str:
        """Normalize and optionally validate a relation string."""
        relation = relation.strip().lower()
        if self.strict_relations and relation not in self._valid_relations():
            raise ValueError(
                f"Unknown relation '{relation}'. "
                f"Register it with register_relation() or set strict_relations=False. "
                f"Valid: {sorted(self._valid_relations())}"
            )
        # Track newly seen custom relations in meta (unless during ingestion)
        if not _skip_auto_register:
            core_vals = {r.value for r in CoreRelation}
            if relation not in core_vals:
                customs = self._data["meta"].setdefault("custom_relations", [])
                if relation not in customs:
                    customs.append(relation)
                    customs.sort()
        return relation

    def _validate_node_type(self, node_type: str) -> str:
        node_type = node_type.strip().lower()
        if self.strict_node_types and node_type not in DEFAULT_NODE_TYPES:
            raise ValueError(
                f"Unknown node type '{node_type}'. Valid: {sorted(DEFAULT_NODE_TYPES)}"
            )
        return node_type

    # ------------------------------------------------------------------
    # Persistence — Load / Save
    # ------------------------------------------------------------------

    def load(self, path: str | Path | None = None) -> None:
        """Load graph from JSON file."""
        path = Path(path) if path else self.graph_path
        with open(path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        # Hydrate proposals
        self._proposals = [
            RelationProposal.from_dict(p)
            for p in self._data.get("relation_proposals", [])
        ]
        self._rebuild_networkx()
        self._dirty = False
        logger.info("Loaded graph from %s (%d nodes, %d edges, %d proposals)",
                     path, len(self._data["nodes"]), len(self._data["edges"]),
                     len(self._proposals))

    def save(self, path: str | Path | None = None) -> None:
        """Save graph to JSON file with sorted keys for stable diffs."""
        path = Path(path) if path else self.graph_path
        self._data["meta"]["updated"] = now_iso()
        # Sync proposals back into data dict before serializing
        self._data["relation_proposals"] = [p.to_dict() for p in self._proposals]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, sort_keys=True, cls=GraphEncoder)
        self._dirty = False
        logger.info("Saved graph to %s", path)

    def load_embeddings(self, path: str | Path | None = None) -> None:
        """Load embeddings from separate JSON file."""
        path = Path(path) if path else self.embeddings_path
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Support files with _meta key (new format) or flat dict (legacy)
        if isinstance(raw, dict) and "_meta" in raw:
            self._embed_meta = raw.pop("_meta")
            self._embeddings = raw
        else:
            self._embeddings = raw
        self._dirty_embeddings = False
        logger.info("Loaded %d embeddings from %s", len(self._embeddings), path)

    def save_embeddings(self, path: str | Path | None = None) -> None:
        """Save embeddings to separate JSON file."""
        path = Path(path) if path else self.embeddings_path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(self._embeddings)
        data["_meta"] = self._embed_meta
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        self._dirty_embeddings = False
        logger.info("Saved %d embeddings to %s", len(self._embeddings), path)

    def save_all(self) -> None:
        """Save both graph and embeddings."""
        self.save()
        if self._embeddings:
            self.save_embeddings()

    # ------------------------------------------------------------------
    # Source file management
    # ------------------------------------------------------------------

    def store_source(
        self,
        text: str,
        doc_id: str,
        *,
        original_path: str | Path | None = None,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        """
        Store a source document's text in the managed sources directory.

        Files are stored with content-hash filenames for deduplication.
        When a document is re-ingested with different content, the previous
        version is archived to ``sources/archive/`` with a version number.

        Args:
            text: The full document text.
            doc_id: The document identifier (matches the graph node ID).
            original_path: The original file path (for metadata only).
            encoding: File encoding for writing.

        Returns:
            A dict with storage details:
              - stored_path: Path to the stored file
              - content_hash: SHA-256 content hash (12 chars)
              - is_duplicate: True if identical content was already stored
              - existing_doc_id: If duplicate, the doc_id of the existing copy
              - is_update: True if this replaces a previous version
              - version: Current version number (1-based)
              - ingestion_id: Unique identifier for this ingestion run
        """
        self.sources_dir.mkdir(parents=True, exist_ok=True)

        chash = content_hash(text)
        doc_slug = slugify(doc_id)
        ts = now_iso()
        ingestion_id = f"{doc_slug}_{chash}_{ts[:19].replace(':', '').replace('-', '')}"

        # Check for duplicate content across all stored sources
        manifest = self._data["meta"].setdefault("sources", {})
        for existing_id, entry in manifest.items():
            if entry.get("content_hash") == chash and existing_id != doc_slug:
                logger.info(
                    "Duplicate content detected: '%s' matches existing '%s'",
                    doc_id, existing_id,
                )
                return {
                    "stored_path": str(Path(entry["stored_path"])),
                    "content_hash": chash,
                    "is_duplicate": True,
                    "existing_doc_id": existing_id,
                    "is_update": False,
                    "version": entry.get("version", 1),
                    "ingestion_id": ingestion_id,
                }

        # Check if same content was already stored for this doc_id
        if doc_slug in manifest:
            old_entry = manifest[doc_slug]
            if old_entry.get("content_hash") == chash:
                # Same content, same doc — not a duplicate cross-doc, just a re-ingest
                return {
                    "stored_path": str(Path(old_entry["stored_path"])),
                    "content_hash": chash,
                    "is_duplicate": False,
                    "existing_doc_id": None,
                    "is_update": False,
                    "version": old_entry.get("version", 1),
                    "ingestion_id": ingestion_id,
                }

        # Determine filename: {hash}_{slug}.md
        stored_name = f"{chash}_{doc_slug}.md"
        stored_path = self.sources_dir / stored_name

        # Handle version archival if this doc_id was previously stored
        is_update = False
        version = 1
        if doc_slug in manifest:
            old_entry = manifest[doc_slug]
            if old_entry.get("content_hash") != chash:
                is_update = True
                version = old_entry.get("version", 1) + 1

                # Archive the old version
                archive_dir = self.sources_dir / "archive"
                archive_dir.mkdir(exist_ok=True)

                old_stored = Path(old_entry["stored_path"])
                if not old_stored.is_absolute():
                    old_stored = self.graph_path.parent / old_stored
                if old_stored.exists():
                    old_version = old_entry.get("version", 1)
                    archive_name = f"v{old_version}_{old_entry.get('content_hash', 'unknown')}_{doc_slug}.md"
                    archive_path = archive_dir / archive_name
                    old_stored.rename(archive_path)
                    logger.info(
                        "Archived previous version of '%s' → %s",
                        doc_id, archive_path,
                    )

                logger.info(
                    "Content update for '%s': v%d (%s) → v%d (%s)",
                    doc_id, version - 1, old_entry.get("content_hash"),
                    version, chash,
                )
            else:
                version = old_entry.get("version", 1)
        else:
            pass

        # Write the current version
        stored_path.write_text(text, encoding=encoding)

        # Build manifest entry
        entry: dict[str, Any] = {
            "stored_path": str(stored_path.relative_to(self.sources_dir.parent)
                              if self.sources_dir.parent == self.graph_path.parent
                              else stored_path),
            "content_hash": chash,
            "original_path": str(original_path) if original_path else None,
            "char_count": len(text),
            "stored_at": ts,
            "version": version,
            "ingestion_id": ingestion_id,
        }

        # Build version history: carry over old history + append the just-archived version
        if doc_slug in manifest:
            old_entry = manifest[doc_slug]
            entry["versions"] = list(old_entry.get("versions", []))
            if is_update:
                old_version_num = old_entry.get("version", 1)
                archive_name = f"v{old_version_num}_{old_entry.get('content_hash', 'unknown')}_{doc_slug}.md"
                entry["versions"].append({
                    "version": old_version_num,
                    "content_hash": old_entry.get("content_hash"),
                    "char_count": old_entry.get("char_count", 0),
                    "stored_at": old_entry.get("stored_at", ""),
                    "ingestion_id": old_entry.get("ingestion_id", ""),
                    "archived_to": str(self.sources_dir / "archive" / archive_name),
                })

        manifest[doc_slug] = entry
        self._dirty = True

        logger.info("Stored source '%s' v%d → %s (%d chars, hash=%s)",
                     doc_id, version, stored_path, len(text), chash)

        return {
            "stored_path": str(stored_path),
            "content_hash": chash,
            "is_duplicate": False,
            "existing_doc_id": None,
            "is_update": is_update,
            "version": version,
            "ingestion_id": ingestion_id,
        }

    def get_source_path(self, doc_id: str) -> Path | None:
        """Get the stored file path for a document, or None if not stored."""
        manifest = self._data["meta"].get("sources", {})
        doc_slug = slugify(doc_id)
        entry = manifest.get(doc_slug)
        if not entry:
            return None
        stored = Path(entry["stored_path"])
        # Resolve relative paths against the graph directory
        if not stored.is_absolute():
            stored = self.graph_path.parent / stored
        return stored if stored.exists() else None

    def get_source_text(self, doc_id: str, encoding: str = "utf-8") -> str | None:
        """Read and return the stored source text for a document."""
        path = self.get_source_path(doc_id)
        if path is None:
            return None
        return path.read_text(encoding=encoding)

    def has_source(self, doc_id: str) -> bool:
        """Check if a source file is stored for this document."""
        return self.get_source_path(doc_id) is not None

    def get_source_info(self, doc_id: str) -> dict[str, Any] | None:
        """Get the manifest entry for a stored source."""
        manifest = self._data["meta"].get("sources", {})
        return manifest.get(slugify(doc_id))

    def list_sources(self) -> list[dict[str, Any]]:
        """
        List all stored sources with their metadata.

        Returns:
            List of dicts with doc_id, content_hash, stored_path,
            char_count, stored_at, version, ingestion_id, num_versions,
            and file_exists.
        """
        manifest = self._data["meta"].get("sources", {})
        results = []
        for doc_slug, entry in sorted(manifest.items()):
            stored = Path(entry.get("stored_path", ""))
            if not stored.is_absolute():
                stored = self.graph_path.parent / stored
            versions = entry.get("versions", [])
            results.append({
                "doc_id": doc_slug,
                "content_hash": entry.get("content_hash", ""),
                "stored_path": str(entry.get("stored_path", "")),
                "original_path": entry.get("original_path"),
                "char_count": entry.get("char_count", 0),
                "stored_at": entry.get("stored_at", ""),
                "version": entry.get("version", 1),
                "ingestion_id": entry.get("ingestion_id", ""),
                "num_versions": len(versions) + 1,  # current + archived
                "file_exists": stored.exists(),
            })
        return results

    def get_source_versions(self, doc_id: str) -> list[dict[str, Any]]:
        """
        Get the full version history for a stored source.

        Returns:
            List of version dicts (oldest first), each with:
              version, content_hash, char_count, stored_at, ingestion_id,
              archived_to (path), and file_exists.
            The last entry is the current (active) version.
        """
        manifest = self._data["meta"].get("sources", {})
        doc_slug = slugify(doc_id)
        entry = manifest.get(doc_slug)
        if not entry:
            return []

        versions: list[dict[str, Any]] = []

        # Archived versions
        for v in entry.get("versions", []):
            archived = Path(v.get("archived_to", ""))
            if not archived.is_absolute():
                archived = self.graph_path.parent / archived
            versions.append({
                "version": v.get("version", 0),
                "content_hash": v.get("content_hash", ""),
                "char_count": v.get("char_count", 0),
                "stored_at": v.get("stored_at", ""),
                "ingestion_id": v.get("ingestion_id", ""),
                "archived_to": str(v.get("archived_to", "")),
                "file_exists": archived.exists(),
                "is_current": False,
            })

        # Current version
        stored = Path(entry.get("stored_path", ""))
        if not stored.is_absolute():
            stored = self.graph_path.parent / stored
        versions.append({
            "version": entry.get("version", 1),
            "content_hash": entry.get("content_hash", ""),
            "char_count": entry.get("char_count", 0),
            "stored_at": entry.get("stored_at", ""),
            "ingestion_id": entry.get("ingestion_id", ""),
            "stored_path": str(entry.get("stored_path", "")),
            "file_exists": stored.exists(),
            "is_current": True,
        })

        return sorted(versions, key=lambda v: v["version"])

    def get_nodes_by_ingestion(self, ingestion_id: str) -> list[tuple[str, dict]]:
        """
        Find all nodes created during a specific ingestion run.

        Args:
            ingestion_id: The ingestion run identifier.

        Returns:
            List of (node_id, node_data) tuples.
        """
        results = []
        for nid, node in self._data["nodes"].items():
            props = node.get("properties", {})
            if props.get("ingestion_id") == ingestion_id:
                results.append((nid, deepcopy(node)))
        return results

    def get_edges_by_ingestion(self, ingestion_id: str) -> list[dict]:
        """
        Find all edges created during a specific ingestion run.

        Args:
            ingestion_id: The ingestion run identifier.

        Returns:
            List of edge dicts.
        """
        return [
            deepcopy(e) for e in self._data["edges"]
            if e.get("properties", {}).get("ingestion_id") == ingestion_id
        ]

    def get_nodes_by_content_hash(self, chash: str) -> list[tuple[str, dict]]:
        """
        Find all nodes created from a specific version of a source document.

        Args:
            chash: The content hash of the source version.

        Returns:
            List of (node_id, node_data) tuples.
        """
        results = []
        for nid, node in self._data["nodes"].items():
            props = node.get("properties", {})
            if props.get("content_hash") == chash:
                results.append((nid, deepcopy(node)))
        return results

    def remove_source(self, doc_id: str, *, delete_file: bool = True) -> bool:
        """
        Remove a stored source file and its manifest entry.

        Args:
            doc_id: Document identifier.
            delete_file: If True, also delete the file from disk.

        Returns:
            True if the source was found and removed.
        """
        manifest = self._data["meta"].get("sources", {})
        doc_slug = slugify(doc_id)
        entry = manifest.get(doc_slug)
        if not entry:
            return False

        if delete_file:
            stored = Path(entry["stored_path"])
            if not stored.is_absolute():
                stored = self.graph_path.parent / stored
            if stored.exists():
                stored.unlink()
                logger.info("Deleted source file: %s", stored)

        del manifest[doc_slug]
        self._dirty = True
        return True

    def check_source_integrity(self) -> list[dict[str, Any]]:
        """
        Verify that all stored sources match their recorded hashes
        and that files exist on disk.

        Returns:
            List of issues found (empty = all good). Each issue is a dict
            with doc_id, issue type, and details.
        """
        issues: list[dict[str, Any]] = []
        manifest = self._data["meta"].get("sources", {})

        for doc_slug, entry in manifest.items():
            stored = Path(entry.get("stored_path", ""))
            if not stored.is_absolute():
                stored = self.graph_path.parent / stored

            if not stored.exists():
                issues.append({
                    "doc_id": doc_slug,
                    "issue": "file_missing",
                    "expected_path": str(stored),
                })
                continue

            # Verify content hash
            actual_text = stored.read_text(encoding="utf-8")
            actual_hash = content_hash(actual_text)
            expected_hash = entry.get("content_hash", "")
            if actual_hash != expected_hash:
                issues.append({
                    "doc_id": doc_slug,
                    "issue": "hash_mismatch",
                    "expected_hash": expected_hash,
                    "actual_hash": actual_hash,
                    "path": str(stored),
                })

            # Verify char count
            expected_chars = entry.get("char_count", 0)
            if len(actual_text) != expected_chars:
                issues.append({
                    "doc_id": doc_slug,
                    "issue": "size_mismatch",
                    "expected_chars": expected_chars,
                    "actual_chars": len(actual_text),
                })

        return issues

    def source_stats(self) -> dict[str, Any]:
        """Summary statistics about stored sources."""
        manifest = self._data["meta"].get("sources", {})
        total_chars = sum(e.get("char_count", 0) for e in manifest.values())
        hashes = [e.get("content_hash") for e in manifest.values()]
        unique_hashes = len(set(hashes))

        return {
            "total_sources": len(manifest),
            "unique_content": unique_hashes,
            "total_chars": total_chars,
            "sources_dir": str(self.sources_dir),
            "dir_exists": self.sources_dir.exists(),
        }

    # ------------------------------------------------------------------
    # Export / Import helpers
    # ------------------------------------------------------------------

    def export_split(self, directory: str | Path) -> tuple[Path, Path]:
        """
        Export nodes and edges into separate files for the 'split later' option.

        Returns (nodes_path, edges_path).
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        nodes_path = directory / "nodes.json"
        edges_path = directory / "edges.json"

        nodes_data = {
            "meta": deepcopy(self._data["meta"]),
            "nodes": deepcopy(self._data["nodes"]),
        }
        edges_data = {
            "meta": deepcopy(self._data["meta"]),
            "edges": deepcopy(self._data["edges"]),
        }

        with open(nodes_path, "w", encoding="utf-8") as f:
            json.dump(nodes_data, f, indent=2, sort_keys=True, cls=GraphEncoder)
        with open(edges_path, "w", encoding="utf-8") as f:
            json.dump(edges_data, f, indent=2, sort_keys=True, cls=GraphEncoder)

        logger.info("Exported split files to %s", directory)
        return nodes_path, edges_path

    @classmethod
    def from_split(
        cls,
        nodes_path: str | Path,
        edges_path: str | Path,
        output_path: str | Path = "knowledge_graph.json",
        **kwargs: Any,
    ) -> "KnowledgeGraph":
        """Reconstruct a graph from previously split node/edge files."""
        with open(nodes_path, "r", encoding="utf-8") as f:
            nodes_data = json.load(f)
        with open(edges_path, "r", encoding="utf-8") as f:
            edges_data = json.load(f)

        kg = cls(output_path, **kwargs)
        kg._data["nodes"] = nodes_data.get("nodes", {})
        kg._data["edges"] = edges_data.get("edges", [])
        # Merge meta
        for key in ("custom_relations", "node_types"):
            existing = set(kg._data["meta"].get(key, []))
            existing.update(nodes_data.get("meta", {}).get(key, []))
            existing.update(edges_data.get("meta", {}).get(key, []))
            kg._data["meta"][key] = sorted(existing)
        kg._rebuild_networkx()
        kg._dirty = True
        return kg

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the raw graph data."""
        return deepcopy(self._data)

    # ------------------------------------------------------------------
    # Internal: networkx sync
    # ------------------------------------------------------------------

    def _rebuild_networkx(self) -> None:
        """Rebuild the networkx DiGraph from the raw dict."""
        self._G = nx.DiGraph()
        for nid, node in self._data["nodes"].items():
            self._G.add_node(nid, **node)
        for edge in self._data["edges"]:
            self._G.add_edge(
                edge["source"],
                edge["target"],
                **{k: v for k, v in edge.items() if k not in ("source", "target")},
            )

    @property
    def graph(self) -> nx.DiGraph:
        """Direct access to the underlying networkx DiGraph (read-friendly)."""
        return self._G

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        *,
        type: str = "concept",
        label: str | None = None,
        properties: dict[str, Any] | None = None,
        source: str = "manual",
        confidence: float = 1.0,
        merge: bool = True,
    ) -> str:
        """
        Add or update a node.

        Args:
            node_id: Unique identifier (will be slugified if needed).
            type: Node type from DEFAULT_NODE_TYPES or custom.
            label: Human-readable label (defaults to node_id).
            properties: Arbitrary key-value metadata.
            source: Provenance tag (e.g., 'manual', 'llm_extraction', 'doc_ingest').
            confidence: Confidence score [0, 1].
            merge: If True, merge properties with existing node instead of overwriting.

        Returns:
            The (possibly slugified) node_id.
        """
        node_id = slugify(node_id) if node_id != slugify(node_id) else node_id
        type = self._validate_node_type(type)
        label = label or node_id
        properties = properties or {}

        ts = now_iso()
        new_node: dict[str, Any] = {
            "type": type,
            "label": label,
            "properties": properties,
            "source": source,
            "confidence": confidence,
        }

        if node_id in self._data["nodes"] and merge:
            existing = self._data["nodes"][node_id]
            existing["properties"].update(properties)
            existing["type"] = type
            existing["label"] = label
            existing["source"] = source
            existing["confidence"] = max(existing.get("confidence", 0), confidence)
            if self.auto_timestamp:
                existing["updated"] = ts
        else:
            if self.auto_timestamp:
                new_node["created"] = ts
                new_node["updated"] = ts
            self._data["nodes"][node_id] = new_node

        # Sync networkx
        self._G.add_node(node_id, **self._data["nodes"][node_id])
        self._dirty = True
        return node_id

    def remove_node(self, node_id: str, *, remove_orphan_edges: bool = True) -> bool:
        """Remove a node and optionally its connected edges."""
        if node_id not in self._data["nodes"]:
            return False
        del self._data["nodes"][node_id]
        if remove_orphan_edges:
            self._data["edges"] = [
                e for e in self._data["edges"]
                if e["source"] != node_id and e["target"] != node_id
            ]
        if node_id in self._G:
            self._G.remove_node(node_id)
        # Clean up embedding if present
        self._embeddings.pop(node_id, None)
        self._dirty = True
        return True

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Return node data or None."""
        return deepcopy(self._data["nodes"].get(node_id))

    def has_node(self, node_id: str) -> bool:
        return node_id in self._data["nodes"]

    def search_nodes(
        self,
        *,
        type: str | None = None,
        label_contains: str | None = None,
        property_filter: dict[str, Any] | None = None,
        source: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Search nodes by various criteria. Returns list of (node_id, node_data)."""
        results = []
        for nid, node in self._data["nodes"].items():
            if type and node.get("type") != type:
                continue
            if label_contains and label_contains.lower() not in node.get("label", "").lower():
                continue
            if source and node.get("source") != source:
                continue
            if node.get("confidence", 1.0) < min_confidence:
                continue
            if property_filter:
                props = node.get("properties", {})
                if not all(props.get(k) == v for k, v in property_filter.items()):
                    continue
            results.append((nid, deepcopy(node)))
        return results

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str = "related_to",
        *,
        properties: dict[str, Any] | None = None,
        source_tag: str = "manual",
        confidence: float = 1.0,
        weight: float = 1.0,
        allow_duplicate: bool = False,
        _skip_auto_register: bool = False,
    ) -> dict[str, Any]:
        """
        Add an edge between two nodes.

        Both source and target nodes must already exist. Use add_node() first
        or pass allow_missing=True (not implemented — explicit is better).

        Args:
            source: Source node ID.
            target: Target node ID.
            relation: Relation type (from CoreRelation or custom).
            properties: Arbitrary metadata on the edge.
            source_tag: Provenance.
            confidence: Confidence score [0, 1].
            weight: Edge weight for graph algorithms.
            allow_duplicate: If False, skip if an identical edge exists.

        Returns:
            The edge dict that was added.
        """
        if source not in self._data["nodes"]:
            raise KeyError(f"Source node '{source}' does not exist. Add it first.")
        if target not in self._data["nodes"]:
            raise KeyError(f"Target node '{target}' does not exist. Add it first.")

        relation = self._validate_relation(relation, _skip_auto_register=_skip_auto_register)

        # Check for duplicates
        if not allow_duplicate:
            for e in self._data["edges"]:
                if e["source"] == source and e["target"] == target and e["relation"] == relation:
                    # Update existing edge
                    e["properties"] = {**e.get("properties", {}), **(properties or {})}
                    e["confidence"] = max(e.get("confidence", 0), confidence)
                    e["weight"] = weight
                    if self.auto_timestamp:
                        e["updated"] = now_iso()
                    self._G[source][target].update(e)
                    self._dirty = True
                    return e

        ts = now_iso()
        edge: dict[str, Any] = {
            "source": source,
            "target": target,
            "relation": relation,
            "properties": properties or {},
            "source_tag": source_tag,
            "confidence": confidence,
            "weight": weight,
        }
        if self.auto_timestamp:
            edge["created"] = ts
            edge["updated"] = ts

        self._data["edges"].append(edge)
        self._G.add_edge(source, target, **edge)
        self._dirty = True
        return edge

    def remove_edges(
        self, source: str, target: str, relation: str | None = None
    ) -> int:
        """Remove edges between source and target (optionally filtered by relation). Returns count removed."""
        before = len(self._data["edges"])
        self._data["edges"] = [
            e for e in self._data["edges"]
            if not (
                e["source"] == source
                and e["target"] == target
                and (relation is None or e["relation"] == relation)
            )
        ]
        removed = before - len(self._data["edges"])
        if removed > 0:
            self._rebuild_networkx()
            self._dirty = True
        return removed

    def get_edges(
        self,
        node_id: str | None = None,
        relation: str | None = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """
        Get edges, optionally filtered by node and/or relation.

        Args:
            node_id: Filter by this node as source, target, or both.
            relation: Filter by relation type.
            direction: 'outgoing', 'incoming', or 'both'.
        """
        results = []
        for e in self._data["edges"]:
            if relation and e["relation"] != relation:
                continue
            if node_id:
                if direction == "outgoing" and e["source"] != node_id:
                    continue
                elif direction == "incoming" and e["target"] != node_id:
                    continue
                elif direction == "both" and node_id not in (e["source"], e["target"]):
                    continue
            results.append(deepcopy(e))
        return results

    # ------------------------------------------------------------------
    # Graph queries (leverage networkx)
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        node_id: str,
        relation: str | None = None,
        direction: str = "both",
        max_depth: int = 1,
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Get neighboring nodes up to max_depth hops.

        Returns list of (node_id, node_data) tuples.
        """
        if node_id not in self._G:
            return []

        visited: set[str] = set()
        frontier = {node_id}

        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                if direction in ("outgoing", "both"):
                    for succ in self._G.successors(nid):
                        if relation:
                            edge_data = self._G.edges[nid, succ]
                            if edge_data.get("relation") != relation:
                                continue
                        next_frontier.add(succ)
                if direction in ("incoming", "both"):
                    for pred in self._G.predecessors(nid):
                        if relation:
                            edge_data = self._G.edges[pred, nid]
                            if edge_data.get("relation") != relation:
                                continue
                        next_frontier.add(pred)
            visited.update(frontier)
            frontier = next_frontier - visited

        visited.update(frontier)
        visited.discard(node_id)

        return [
            (nid, deepcopy(self._data["nodes"][nid]))
            for nid in sorted(visited)
            if nid in self._data["nodes"]
        ]

    def get_subgraph(
        self, node_id: str, depth: int = 2
    ) -> dict[str, Any]:
        """
        Extract a local subgraph around a node (useful for RAG context windows).

        Returns a graph_data dict in the same schema as the full graph,
        ready to be serialized or fed to an LLM.
        """
        neighbor_ids = {nid for nid, _ in self.get_neighbors(node_id, max_depth=depth)}
        neighbor_ids.add(node_id)

        subgraph_data: dict[str, Any] = {
            "meta": {
                "extracted_from": str(self.graph_path),
                "center_node": node_id,
                "depth": depth,
                "extracted_at": now_iso(),
            },
            "nodes": {
                nid: deepcopy(self._data["nodes"][nid])
                for nid in neighbor_ids
                if nid in self._data["nodes"]
            },
            "edges": [
                deepcopy(e)
                for e in self._data["edges"]
                if e["source"] in neighbor_ids and e["target"] in neighbor_ids
            ],
        }
        return subgraph_data

    def shortest_path(self, source: str, target: str) -> list[str] | None:
        """Return shortest path as list of node IDs, or None if no path exists."""
        try:
            return nx.shortest_path(self._G, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_connected_components(self) -> list[set[str]]:
        """Return connected components (treating graph as undirected)."""
        return [
            comp for comp in nx.weakly_connected_components(self._G)
        ]

    def get_central_nodes(self, top_n: int = 10, method: str = "degree") -> list[tuple[str, float]]:
        """
        Return the most central nodes.

        Args:
            method: 'degree', 'betweenness', 'pagerank', or 'eigenvector'.
        """
        if method == "degree":
            centrality = nx.degree_centrality(self._G)
        elif method == "betweenness":
            centrality = nx.betweenness_centrality(self._G)
        elif method == "pagerank":
            centrality = nx.pagerank(self._G)
        elif method == "eigenvector":
            try:
                centrality = nx.eigenvector_centrality(self._G, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                centrality = nx.eigenvector_centrality_numpy(self._G)
        else:
            raise ValueError(f"Unknown centrality method: {method}")

        return sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # ------------------------------------------------------------------
    # Embedding operations
    # ------------------------------------------------------------------

    def set_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Store an embedding vector for a node."""
        if node_id not in self._data["nodes"]:
            raise KeyError(f"Node '{node_id}' does not exist.")
        self._embeddings[node_id] = embedding
        self._dirty_embeddings = True

    def get_embedding(self, node_id: str) -> list[float] | None:
        """Retrieve the embedding for a node."""
        return self._embeddings.get(node_id)

    def nodes_with_embeddings(self) -> list[str]:
        """List all node IDs that have embeddings."""
        return sorted(self._embeddings.keys())

    def nodes_without_embeddings(self) -> list[str]:
        """List node IDs that lack embeddings (useful for batch embedding jobs)."""
        return sorted(set(self._data["nodes"].keys()) - set(self._embeddings.keys()))

    @property
    def embed_model(self) -> str | None:
        """Return the embedding model name stored in the embeddings file, or None."""
        return self._embed_meta.get("model")

    @property
    def embed_dim(self) -> int | None:
        """Return the embedding dimension stored in the embeddings file, or None."""
        return self._embed_meta.get("dim")

    def find_similar(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Find the most similar nodes by cosine similarity.

        Returns list of (node_id, similarity_score) tuples, descending.
        Uses pure Python — for large-scale use, replace with numpy/faiss.

        Raises ValueError if the query embedding dimension doesn't match
        the stored embeddings.
        """
        # Check for dimension mismatch (catches wrong embedding model)
        if self._embeddings:
            stored_dim = len(next(iter(self._embeddings.values())))
            query_dim = len(query_embedding)
            if query_dim != stored_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: query has {query_dim} dims "
                    f"but graph embeddings have {stored_dim} dims. "
                    f"This usually means the query is using a different "
                    f"embedding model than was used during ingestion"
                    f"{f' ({self.embed_model})' if self.embed_model else ''}."
                )

        def cosine_sim(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        if not self._embeddings:
            logger.debug("find_similar: no embeddings loaded, returning empty")
            return []

        scores = []
        for nid, emb in self._embeddings.items():
            sim = cosine_sim(query_embedding, emb)
            scores.append((nid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        logger.debug(
            "find_similar: scored %d nodes, top=%.4f, bottom=%.4f",
            len(scores),
            scores[0][1] if scores else 0.0,
            scores[-1][1] if scores else 0.0,
        )
        return scores[:top_k]

    def build_embedding_text(
        self,
        node_id: str,
        *,
        include_neighbors: bool = True,
        max_neighbor_context: int = 10,
        max_chars: int = 4000,
    ) -> str:
        """
        Build a text representation of a node suitable for embedding.

        The strategy varies by node type:
          - **section**: Uses stored body text (if available), heading path,
            and content flags.
          - **document**: Uses label, description properties, and a summary
            of contained sections.
          - **entity/concept/tool/etc.**: Uses label, type, properties, and
            a structured summary of relationships from the graph.

        Args:
            node_id: The node to build text for.
            include_neighbors: Include relationship context from connected nodes.
            max_neighbor_context: Max number of relationships to include.
            max_chars: Truncate the final text to this length.

        Returns:
            A text string ready for an embedding model.
        """
        node = self._data["nodes"].get(node_id)
        if not node:
            raise KeyError(f"Node '{node_id}' does not exist.")

        ntype = node.get("type", "concept")
        label = node.get("label", node_id)
        props = node.get("properties", {})

        parts: list[str] = []

        if ntype == "section":
            parts.append(self._build_section_embedding_text(node_id, node))
        elif ntype == "document":
            parts.append(self._build_document_embedding_text(node_id, node))
        else:
            parts.append(self._build_entity_embedding_text(node_id, node))

        # Add neighbor/relationship context
        if include_neighbors:
            rel_text = self._build_relationship_context(
                node_id, max_relations=max_neighbor_context
            )
            if rel_text:
                parts.append(rel_text)

        text = "\n\n".join(parts)

        # Truncate if needed
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."

        return text

    def _build_section_embedding_text(self, node_id: str, node: dict) -> str:
        """Build embedding text for a section node."""
        props = node.get("properties", {})
        label = node.get("label", node_id)

        parts: list[str] = []

        # Heading hierarchy as context
        path = props.get("path", [])
        if path:
            parts.append(f"Section: {' > '.join(path)}")
        else:
            parts.append(f"Section: {label}")

        # Body text if stored
        body = props.get("body_text", "")
        if body:
            parts.append(body)
        else:
            # Reconstruct from available metadata
            desc_parts = [f"This section covers {label}."]
            flags = []
            if props.get("has_code"):
                flags.append("code examples")
            if props.get("has_table"):
                flags.append("tables")
            if props.get("has_list"):
                flags.append("lists")
            if flags:
                desc_parts.append(f"Contains: {', '.join(flags)}.")
            parts.append(" ".join(desc_parts))

        return "\n".join(parts)

    def _build_document_embedding_text(self, node_id: str, node: dict) -> str:
        """Build embedding text for a document node."""
        props = node.get("properties", {})
        label = node.get("label", node_id)

        parts: list[str] = [f"Document: {label}"]

        # Include description if available
        desc = props.get("description", "")
        if desc:
            parts.append(desc)

        # Summarize sections
        section_edges = [
            e for e in self._data["edges"]
            if e["source"] == node_id and e["relation"] == "contains"
        ]
        if section_edges:
            section_labels = []
            for e in section_edges:
                sec_node = self._data["nodes"].get(e["target"], {})
                sec_label = sec_node.get("label", e["target"])
                section_labels.append(sec_label)
            parts.append(f"Sections: {', '.join(section_labels)}")

        return "\n".join(parts)

    def _build_entity_embedding_text(self, node_id: str, node: dict) -> str:
        """Build embedding text for entity/concept/tool/etc. nodes."""
        label = node.get("label", node_id)
        ntype = node.get("type", "concept")
        props = node.get("properties", {})

        parts: list[str] = [f"{label} ({ntype})"]

        # Include meaningful properties
        skip_keys = {"body_text"}  # internal, already handled
        for k, v in props.items():
            if k in skip_keys:
                continue
            if isinstance(v, (str, int, float, bool)):
                parts.append(f"{k}: {v}")
            elif isinstance(v, list) and len(v) <= 5:
                parts.append(f"{k}: {', '.join(str(i) for i in v)}")

        # Include edge context snippets
        edge_contexts: list[str] = []
        for e in self._data["edges"]:
            ctx = e.get("properties", {}).get("context", "")
            if ctx and (e["source"] == node_id or e["target"] == node_id):
                edge_contexts.append(ctx)
        if edge_contexts:
            # Deduplicate and take best
            unique = list(dict.fromkeys(edge_contexts))[:5]
            parts.append("Context: " + " | ".join(unique))

        return "\n".join(parts)

    def _build_relationship_context(
        self, node_id: str, max_relations: int = 10
    ) -> str:
        """Build a text summary of a node's relationships."""
        outgoing: list[str] = []
        incoming: list[str] = []

        for e in self._data["edges"]:
            if e["source"] == node_id:
                target_label = self._data["nodes"].get(
                    e["target"], {}
                ).get("label", e["target"])
                outgoing.append(f"{e['relation']} → {target_label}")
            elif e["target"] == node_id:
                source_label = self._data["nodes"].get(
                    e["source"], {}
                ).get("label", e["source"])
                incoming.append(f"{source_label} → {e['relation']}")

        lines: list[str] = []
        all_rels = outgoing[:max_relations // 2] + incoming[:max_relations // 2]
        if all_rels:
            lines.append("Relationships: " + "; ".join(all_rels))

        return "\n".join(lines)

    def embed_nodes(
        self,
        embed_fn: Callable[[list[str]], list[list[float]]],
        *,
        node_ids: list[str] | None = None,
        node_types: list[str] | None = None,
        skip_existing: bool = True,
        batch_size: int = 32,
        include_neighbors: bool = True,
        max_chars: int = 4000,
        store_text: bool = False,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate embeddings for nodes in batch.

        Args:
            embed_fn: A callable that takes a list of text strings and returns
                      a list of embedding vectors (list[list[float]]). This is
                      where you plug in your embedding API. The function receives
                      a batch of texts and must return one embedding per text,
                      in the same order.

                      Example with OpenAI:
                          def embed_fn(texts):
                              resp = client.embeddings.create(
                                  model="text-embedding-3-small", input=texts
                              )
                              return [d.embedding for d in resp.data]

                      Example with sentence-transformers:
                          model = SentenceTransformer("all-MiniLM-L6-v2")
                          def embed_fn(texts):
                              return model.encode(texts).tolist()

            node_ids: Specific node IDs to embed. If None, embeds all nodes
                      matching other filters.
            node_types: Only embed nodes of these types. If None, all types.
                        Common patterns:
                          - ["section"] for document-level RAG
                          - ["concept", "technology", "tool"] for entity-level RAG
                          - None for everything
            skip_existing: Skip nodes that already have embeddings.
            batch_size: Number of texts to embed per API call.
            include_neighbors: Include relationship context in embedding text.
            max_chars: Max text length per node.
            store_text: If True, store the generated embedding text in the
                        node's properties under 'embedding_text' (useful for
                        debugging and inspection).

        Returns:
            Stats dict: nodes_embedded, nodes_skipped, batches, errors.
        """
        stats: dict[str, Any] = {
            "nodes_embedded": 0,
            "nodes_skipped": 0,
            "batches": 0,
            "errors": [],
        }

        # Determine which nodes to embed
        if node_ids is not None:
            candidates = node_ids
        else:
            candidates = list(self._data["nodes"].keys())

        # Apply type filter
        if node_types is not None:
            type_set = set(node_types)
            candidates = [
                nid for nid in candidates
                if self._data["nodes"].get(nid, {}).get("type") in type_set
            ]

        # Skip existing
        if skip_existing:
            existing = set(self._embeddings.keys())
            before = len(candidates)
            candidates = [nid for nid in candidates if nid not in existing]
            stats["nodes_skipped"] = before - len(candidates)

        if not candidates:
            logger.info("No nodes to embed (all skipped or no matches).")
            return stats

        # Build embedding texts
        texts: list[tuple[str, str]] = []  # (node_id, text)
        for nid in candidates:
            try:
                text = self.build_embedding_text(
                    nid,
                    include_neighbors=include_neighbors,
                    max_chars=max_chars,
                )
                texts.append((nid, text))
                if store_text:
                    self._data["nodes"][nid].setdefault("properties", {})["embedding_text"] = text
            except Exception as e:
                stats["errors"].append(f"Text build failed for '{nid}': {e}")
                logger.warning("Failed to build embedding text for '%s': %s", nid, e)

        # Process in batches
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start:batch_start + batch_size]
            batch_texts = [t for _, t in batch]
            batch_ids = [nid for nid, _ in batch]

            try:
                embeddings = embed_fn(batch_texts)

                if len(embeddings) != len(batch_ids):
                    msg = (
                        f"Batch {stats['batches']}: expected {len(batch_ids)} "
                        f"embeddings, got {len(embeddings)}"
                    )
                    stats["errors"].append(msg)
                    logger.warning("Embedding batch size mismatch: %s", msg)
                    continue

                for nid, emb in zip(batch_ids, embeddings):
                    self._embeddings[nid] = emb
                    stats["nodes_embedded"] += 1

                self._dirty_embeddings = True
                stats["batches"] += 1

            except Exception as e:
                stats["errors"].append(
                    f"Batch {stats['batches']} failed: {e} "
                    f"(nodes: {batch_ids[:3]}...)"
                )
                logger.error("Embedding batch failed: %s", e)

        # Record embedding metadata when we successfully embedded at least one node
        if stats["nodes_embedded"] > 0:
            if model_name:
                self._embed_meta["model"] = model_name
            # Infer dimension from first available embedding
            for emb in self._embeddings.values():
                self._embed_meta["dim"] = len(emb)
                break

        logger.info(
            "Embedded %d nodes in %d batches (%d skipped, %d errors)",
            stats["nodes_embedded"], stats["batches"],
            stats["nodes_skipped"], len(stats["errors"]),
        )
        return stats

    def embed_query(
        self,
        query: str,
        embed_fn: Callable[[list[str]], list[list[float]]],
    ) -> list[float]:
        """
        Embed a query string using the same embedding function.
        Convenience wrapper for search workflows.

        Args:
            query: The query text to embed.
            embed_fn: Same embedding function used for embed_nodes.

        Returns:
            The embedding vector.
        """
        result = embed_fn([query])
        return result[0]

    def search(
        self,
        query: str | list[float],
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        *,
        top_k: int = 5,
        node_types: list[str] | None = None,
        min_confidence: float = 0.0,
        expand_depth: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Semantic search over the knowledge graph.

        Combines embedding similarity with optional graph expansion for
        graph-RAG workflows.

        Args:
            query: Either a text string (requires embed_fn) or a
                   pre-computed embedding vector.
            embed_fn: Embedding function, required if query is a string.
            top_k: Number of top results to return.
            node_types: Filter results to these node types.
            min_confidence: Minimum node confidence.
            expand_depth: If > 0, expand each result's neighborhood to
                          this depth and include connected nodes.

        Returns:
            List of result dicts, each with:
              - node_id, label, type, confidence, similarity
              - neighbors (if expand_depth > 0)
              - context (subgraph data if expand_depth > 0)
        """
        # Get query embedding
        if isinstance(query, str):
            if embed_fn is None:
                raise ValueError(
                    "embed_fn is required when query is a string."
                )
            query_vec = self.embed_query(query, embed_fn)
        else:
            query_vec = query

        # Find similar nodes
        candidates = self.find_similar(query_vec, top_k=top_k * 3)  # over-fetch for filtering

        results: list[dict[str, Any]] = []
        for nid, similarity in candidates:
            if len(results) >= top_k:
                break

            node = self._data["nodes"].get(nid)
            if not node:
                continue

            # Apply filters
            if node_types and node.get("type") not in node_types:
                continue
            if node.get("confidence", 1.0) < min_confidence:
                continue

            result: dict[str, Any] = {
                "node_id": nid,
                "label": node.get("label", nid),
                "type": node.get("type", "concept"),
                "confidence": node.get("confidence", 1.0),
                "similarity": round(similarity, 4),
                "properties": node.get("properties", {}),
            }

            # Graph expansion
            if expand_depth > 0:
                neighbors = self.get_neighbors(nid, max_depth=expand_depth)
                result["neighbors"] = [
                    {"node_id": n_id, "label": n_data.get("label", n_id),
                     "type": n_data.get("type", "concept")}
                    for n_id, n_data in neighbors
                ]
                result["context"] = self.get_subgraph(nid, depth=expand_depth)

            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Relation proposal management
    # ------------------------------------------------------------------

    def propose_relation(
        self,
        name: str,
        *,
        justification: str = "",
        source_entity: str = "",
        target_entity: str = "",
        context: str = "",
        doc_id: str = "",
        confidence: float = 0.5,
    ) -> RelationProposal:
        """
        Propose a new relation type. If a proposal with the same name
        already exists, the new example is appended to it (strengthening
        the case for acceptance).

        Returns the proposal (new or existing).
        """
        name = slugify(name)

        # Check if already a valid relation — no proposal needed
        if name in self._valid_relations():
            logger.debug("Relation '%s' already exists, skipping proposal.", name)
            # Return a dummy accepted proposal for API consistency
            return RelationProposal(
                name=name,
                status=ProposalStatus.ACCEPTED.value,
                justification="Already in schema.",
            )

        # Check for existing pending proposal with same name
        for proposal in self._proposals:
            if proposal.name == name and proposal.status == ProposalStatus.PENDING.value:
                proposal.add_example(source_entity, target_entity, context, doc_id)
                if justification and not proposal.justification:
                    proposal.justification = justification
                self._dirty = True
                logger.info("Augmented existing proposal '%s' (now %d examples)",
                           name, len(proposal.examples))
                return proposal

        # Create new proposal
        proposal = RelationProposal(
            name=name,
            justification=justification,
            confidence=confidence,
            source_docs=[doc_id] if doc_id else [],
        )
        if source_entity or target_entity:
            proposal.examples.append({
                "source": source_entity,
                "target": target_entity,
                "context": context,
            })
        self._proposals.append(proposal)
        self._dirty = True
        logger.info("Created new relation proposal: '%s'", name)
        return proposal

    def get_proposals(
        self,
        status: str | None = ProposalStatus.PENDING.value,
        min_confidence: float = 0.0,
        min_examples: int = 0,
    ) -> list[RelationProposal]:
        """
        Retrieve proposals filtered by status, confidence, and example count.

        Args:
            status: Filter by status ('pending', 'accepted', 'rejected') or None for all.
            min_confidence: Minimum confidence threshold.
            min_examples: Minimum number of supporting examples.
        """
        results = []
        for p in self._proposals:
            if status and p.status != status:
                continue
            if p.confidence < min_confidence:
                continue
            if len(p.examples) < min_examples:
                continue
            results.append(p)
        return sorted(results, key=lambda p: (-p.confidence, p.name))

    def accept_proposal(
        self,
        name: str,
        *,
        review_note: str = "",
        boost_edge_confidence: float = 0.7,
    ) -> bool:
        """
        Accept a proposed relation: register it and optionally boost
        confidence on edges that were added with this relation at low confidence.

        Returns True if a matching pending proposal was found and accepted.
        """
        name = slugify(name)
        for proposal in self._proposals:
            if proposal.name == name and proposal.status == ProposalStatus.PENDING.value:
                proposal.status = ProposalStatus.ACCEPTED.value
                proposal.reviewed_at = now_iso()
                proposal.review_note = review_note
                self.register_relation(name)

                # Boost confidence on existing edges that used this relation
                if boost_edge_confidence > 0:
                    for edge in self._data["edges"]:
                        if (edge["relation"] == name
                                and edge.get("confidence", 1.0) < boost_edge_confidence):
                            edge["confidence"] = boost_edge_confidence
                            if self.auto_timestamp:
                                edge["updated"] = now_iso()

                self._rebuild_networkx()
                self._dirty = True
                logger.info("Accepted relation proposal: '%s'", name)
                return True
        return False

    def reject_proposal(self, name: str, *, review_note: str = "") -> bool:
        """
        Reject a proposed relation. Associated edges remain but are not
        boosted in confidence.

        Returns True if a matching pending proposal was found and rejected.
        """
        name = slugify(name)
        for proposal in self._proposals:
            if proposal.name == name and proposal.status == ProposalStatus.PENDING.value:
                proposal.status = ProposalStatus.REJECTED.value
                proposal.reviewed_at = now_iso()
                proposal.review_note = review_note
                self._dirty = True
                logger.info("Rejected relation proposal: '%s'", name)
                return True
        return False

    def accept_all_proposals(
        self, min_confidence: float = 0.7, min_examples: int = 2
    ) -> list[str]:
        """
        Bulk-accept proposals that meet confidence and example thresholds.
        Returns list of accepted relation names.
        """
        accepted = []
        for p in self.get_proposals(min_confidence=min_confidence, min_examples=min_examples):
            if self.accept_proposal(p.name, review_note="auto-accepted by threshold"):
                accepted.append(p.name)
        return accepted

    def purge_rejected_proposals(self) -> int:
        """Remove all rejected proposals from the list. Returns count removed."""
        before = len(self._proposals)
        self._proposals = [
            p for p in self._proposals
            if p.status != ProposalStatus.REJECTED.value
        ]
        removed = before - len(self._proposals)
        if removed:
            self._dirty = True
        return removed

    # ------------------------------------------------------------------
    # LLM relation extraction & document ingestion
    # ------------------------------------------------------------------

    def build_extraction_prompt(
        self,
        text: str,
        *,
        focus_entities: list[str] | None = None,
        max_triples: int = 50,
    ) -> str:
        """
        Build a prompt for an LLM to extract knowledge graph triples from text.

        The prompt includes the current relation schema so the LLM can reuse
        existing types and only propose new ones when genuinely needed.

        Args:
            text: Document text to extract from.
            focus_entities: Optional list of entity names/IDs to focus extraction on.
            max_triples: Maximum triples to request from the LLM.

        Returns:
            A prompt string ready to send to an LLM.
        """
        existing_relations = sorted(self._valid_relations())
        pending_proposals = [p.name for p in self.get_proposals()]

        focus_section = ""
        if focus_entities:
            focus_section = f"""
FOCUS ENTITIES (prioritize relationships involving these):
{json.dumps(focus_entities, indent=2)}
"""

        system = f"""You are a knowledge graph extraction engine.
Extract entity relationships from text and return ONLY a JSON array.
Do NOT include any explanation, reasoning, or text outside the JSON.

INSTRUCTIONS:
1. Identify entities (concepts, technologies, tools, processes, people, organizations, etc.)
2. Identify relationships between entity pairs.
3. For each relationship, use an EXISTING relation type if it fits reasonably well.
4. Only propose a NEW relation type when no existing type captures the semantics.
5. Return up to {max_triples} triples, prioritized by importance and confidence.

EXISTING RELATION TYPES (prefer these):
{json.dumps(existing_relations, indent=2)}

RECENTLY PROPOSED (not yet accepted — reuse if applicable):
{json.dumps(pending_proposals, indent=2)}

NODE TYPES (assign one to each entity):
{json.dumps(sorted(DEFAULT_NODE_TYPES), indent=2)}

OUTPUT FORMAT — a JSON array of objects:
[
  {{
    "source": "entity name (human-readable)",
    "source_type": "node type",
    "target": "entity name (human-readable)",
    "target_type": "node type",
    "relation": "relation_type (snake_case, existing or new)",
    "is_new_relation": false,
    "suggested_relation": null,
    "justification": null,
    "confidence": 0.85,
    "context": "brief quote or paraphrase from text supporting this triple"
  }}
]

EXAMPLES:

Input: "The Kalman filter is widely used in navigation systems. It requires a state-space model and produces optimal estimates under Gaussian noise."
Output:
[
  {{"source": "Kalman filter", "source_type": "concept", "target": "navigation systems", "target_type": "concept", "relation": "used_in", "is_new_relation": false, "suggested_relation": null, "justification": null, "confidence": 0.92, "context": "The Kalman filter is widely used in navigation systems."}},
  {{"source": "Kalman filter", "source_type": "concept", "target": "state-space model", "target_type": "concept", "relation": "depends_on", "is_new_relation": false, "suggested_relation": null, "justification": null, "confidence": 0.90, "context": "It requires a state-space model"}},
  {{"source": "Kalman filter", "source_type": "concept", "target": "Gaussian noise", "target_type": "concept", "relation": "assumes", "is_new_relation": true, "suggested_relation": "assumes", "justification": "Captures a precondition or assumption dependency not covered by depends_on.", "confidence": 0.85, "context": "produces optimal estimates under Gaussian noise"}}
]

Input: "TensorFlow was developed by Google Brain. It supports GPU acceleration and is commonly compared to PyTorch."
Output:
[
  {{"source": "TensorFlow", "source_type": "tool", "target": "Google Brain", "target_type": "organization", "relation": "created_by", "is_new_relation": false, "suggested_relation": null, "justification": null, "confidence": 0.95, "context": "TensorFlow was developed by Google Brain."}},
  {{"source": "TensorFlow", "source_type": "tool", "target": "GPU acceleration", "target_type": "concept", "relation": "supports", "is_new_relation": false, "suggested_relation": null, "justification": null, "confidence": 0.90, "context": "It supports GPU acceleration"}},
  {{"source": "TensorFlow", "source_type": "tool", "target": "PyTorch", "target_type": "tool", "relation": "alternative_to", "is_new_relation": false, "suggested_relation": null, "justification": null, "confidence": 0.75, "context": "commonly compared to PyTorch"}}
]

Input: "Convolutional layers extract spatial features. Pooling reduces dimensionality before the fully connected layer classifies the output."
Output:
[
  {{"source": "convolutional layers", "source_type": "concept", "target": "spatial features", "target_type": "concept", "relation": "produces", "is_new_relation": false, "suggested_relation": null, "justification": null, "confidence": 0.90, "context": "Convolutional layers extract spatial features."}},
  {{"source": "pooling", "source_type": "concept", "target": "dimensionality", "target_type": "concept", "relation": "reduces", "is_new_relation": true, "suggested_relation": "reduces", "justification": "Captures a quantitative reduction relationship not covered by existing types.", "confidence": 0.88, "context": "Pooling reduces dimensionality"}},
  {{"source": "fully connected layer", "source_type": "concept", "target": "convolutional layers", "target_type": "concept", "relation": "depends_on", "is_new_relation": false, "suggested_relation": null, "justification": null, "confidence": 0.70, "context": "before the fully connected layer classifies the output"}}
]

RULES:
- relation names must be snake_case.
- is_new_relation = true ONLY when no existing relation fits. In that case,
  populate suggested_relation (snake_case) and justification (one sentence).
- confidence ranges: 0.9+ explicit statement, 0.7-0.9 strong implication,
  0.5-0.7 reasonable inference, <0.5 speculative.
- Prefer specific relations (e.g. "depends_on") over generic ones (e.g. "related_to").
{focus_section}"""

        user = f"""Extract entity relationship triples from the following text as a JSON array.

TEXT:
---
{text}
---"""

        return system + "\n\n" + user

    def ingest_document(
        self,
        text: str,
        doc_id: str,
        *,
        llm_extract_fn: Callable[[str], list[dict[str, Any]]],
        focus_entities: list[str] | None = None,
        max_triples: int = 50,
        low_confidence_threshold: float = 0.3,
        auto_add_doc_node: bool = True,
        ingestion_id: str | None = None,
        content_hash: str | None = None,
    ) -> dict[str, Any]:
        """
        Process a document through the extraction pipeline:
          1. Build a schema-aware prompt
          2. Call the LLM to extract triples
          3. Add nodes and edges for known relations
          4. Queue proposals for novel relations (edges added at low confidence)
          5. Optionally add a document node linking to all extracted entities

        Args:
            text: Full document text.
            doc_id: Unique identifier for the document (used for provenance).
            llm_extract_fn: A callable that takes a prompt string and returns
                            a list of triple dicts (matching the JSON schema in
                            the extraction prompt). This is where you plug in
                            your LLM API call + JSON parsing.
            focus_entities: Optional entities to focus extraction on.
            max_triples: Max triples to request.
            low_confidence_threshold: Confidence assigned to edges with novel relations.
            auto_add_doc_node: If True, create a 'document' node and link extracted
                               entities to it with 'documented_by' edges.
            ingestion_id: Unique identifier for this ingestion run (propagated to
                          all created nodes and edges for provenance tracking).
            content_hash: Content hash of the source document (propagated to all
                          created nodes and edges).

        Returns:
            Stats dict: nodes_added, edges_added, proposals_created, proposals_augmented,
                        triples_processed, errors.
        """
        stats = {
            "triples_processed": 0,
            "nodes_added": 0,
            "edges_added": 0,
            "proposals_created": 0,
            "proposals_augmented": 0,
            "errors": [],
        }

        prompt = self.build_extraction_prompt(
            text, focus_entities=focus_entities, max_triples=max_triples
        )

        try:
            triples = llm_extract_fn(prompt)
        except Exception as e:
            stats["errors"].append(f"LLM extraction failed: {e}")
            logger.error("LLM extraction failed for doc '%s': %s", doc_id, e)
            return stats

        if not isinstance(triples, list):
            stats["errors"].append(f"LLM returned non-list: {type(triples)}")
            logger.warning(
                "LLM returned %s instead of list for doc '%s': %s",
                type(triples).__name__, doc_id, str(triples)[:200],
            )
            return stats

        if triples:
            logger.debug(
                "First triple from doc '%s' (type=%s): %s",
                doc_id, type(triples[0]).__name__,
                str(triples[0])[:300],
            )

        known_relations = self._valid_relations()
        doc_slug = slugify(doc_id)
        entity_ids: set[str] = set()

        # Optionally add a document node
        if auto_add_doc_node:
            if not self.has_node(doc_slug):
                self.add_node(
                    doc_slug,
                    type="document",
                    label=doc_id,
                    properties={"text_length": len(text)},
                    source=f"doc_ingest",
                )

        for triple in triples:
            stats["triples_processed"] += 1
            try:
                # Normalize list-format triples into dicts.
                # Some LLMs return [source, relation, target, context] arrays
                # instead of the requested dict format.
                if isinstance(triple, (list, tuple)):
                    if len(triple) >= 3:
                        triple = {
                            "source": str(triple[0]),
                            "relation": str(triple[1]),
                            "target": str(triple[2]),
                            "context": str(triple[3]) if len(triple) > 3 else "",
                        }
                    else:
                        stats["errors"].append(f"Triple too short ({len(triple)} elements): {triple}")
                        continue

                # Attempt to parse bare-string triples returned by some LLMs.
                # Common patterns: "Subject has Object", "A is connected to B",
                # "X provides Y for Z".
                if isinstance(triple, str):
                    parsed = _parse_string_triple(triple)
                    if parsed is not None:
                        triple = parsed
                        logger.debug(
                            "Parsed string triple from doc '%s': %s", doc_id, parsed
                        )
                    else:
                        stats["errors"].append(
                            f"Triple processing error: could not parse string — {triple}"
                        )
                        logger.warning(
                            "Skipping unparseable string triple from doc '%s': %s",
                            doc_id, triple,
                        )
                        continue

                # Skip non-dict items (e.g. bare ints, bools, etc.).
                if not isinstance(triple, dict):
                    stats["errors"].append(
                        f"Triple processing error: expected dict, got {type(triple).__name__} — {triple}"
                    )
                    logger.warning(
                        "Skipping non-dict triple from doc '%s': %s", doc_id, triple
                    )
                    continue

                # Normalize common LLM key aliases so that triples
                # returned as {"subject": ..., "object": ...} or
                # {"head": ..., "tail": ...} are accepted.
                _KEY_ALIASES = {
                    "subject": "source",
                    "head": "source",
                    "from": "source",
                    "entity1": "source",
                    "object": "target",
                    "tail": "target",
                    "to": "target",
                    "entity2": "target",
                    "predicate": "relation",
                    "relationship": "relation",
                    "rel": "relation",
                    "type": "relation",
                }
                for old_key, new_key in _KEY_ALIASES.items():
                    if old_key in triple and new_key not in triple:
                        triple[new_key] = triple.pop(old_key)

                source_label = triple.get("source", "").strip()
                target_label = triple.get("target", "").strip()
                if not source_label or not target_label:
                    stats["errors"].append(
                        f"Triple missing source/target: keys={list(triple.keys())} — {str(triple)[:200]}"
                    )
                    logger.debug(
                        "Skipping triple with empty source/target: %s",
                        str(triple)[:200],
                    )
                    continue

                source_id = slugify(source_label)
                target_id = slugify(target_label)
                relation = slugify(triple.get("relation", "related_to"))
                conf = float(triple.get("confidence", 0.5))
                context = triple.get("context", "")

                # Ensure nodes exist
                for nid, label, ntype in [
                    (source_id, source_label, triple.get("source_type", "concept")),
                    (target_id, target_label, triple.get("target_type", "concept")),
                ]:
                    if not self.has_node(nid):
                        node_props: dict[str, Any] = {}
                        if ingestion_id:
                            node_props["ingestion_id"] = ingestion_id
                        if content_hash:
                            node_props["content_hash"] = content_hash
                        self.add_node(
                            nid,
                            type=ntype,
                            label=label,
                            properties=node_props if node_props else {},
                            source=f"doc:{doc_id}",
                            confidence=conf,
                        )
                        stats["nodes_added"] += 1
                    entity_ids.add(nid)

                # Handle novel vs known relations
                is_new = triple.get("is_new_relation", False)
                suggested = slugify(triple.get("suggested_relation", "")) if triple.get("suggested_relation") else ""

                effective_relation = suggested if (is_new and suggested) else relation

                if is_new and effective_relation not in known_relations:
                    proposal = self.propose_relation(
                        effective_relation,
                        justification=triple.get("justification", ""),
                        source_entity=source_id,
                        target_entity=target_id,
                        context=context,
                        doc_id=doc_id,
                        confidence=conf,
                    )
                    if proposal.status == ProposalStatus.PENDING.value:
                        if len(proposal.examples) <= 1:
                            stats["proposals_created"] += 1
                        else:
                            stats["proposals_augmented"] += 1

                    # Still add the edge but at low confidence
                    edge_conf = min(conf, low_confidence_threshold)
                    skip_register = True
                else:
                    edge_conf = conf
                    skip_register = False

                edge_props: dict[str, Any] = {}
                if context:
                    edge_props["context"] = context
                if ingestion_id:
                    edge_props["ingestion_id"] = ingestion_id
                if content_hash:
                    edge_props["content_hash"] = content_hash

                self.add_edge(
                    source_id,
                    target_id,
                    relation=effective_relation,
                    properties=edge_props,
                    source_tag=f"doc:{doc_id}",
                    confidence=edge_conf,
                    _skip_auto_register=skip_register,
                )
                stats["edges_added"] += 1

            except Exception as e:
                stats["errors"].append(f"Triple processing error: {e} — {triple}")
                logger.warning("Error processing triple from doc '%s': %s", doc_id, e)

        # Link extracted entities to the document node
        if auto_add_doc_node and entity_ids:
            for eid in entity_ids:
                self.add_edge(
                    eid, doc_slug,
                    relation="documented_by",
                    source_tag="doc_ingest",
                    confidence=0.9,
                )

        n_errors = len(stats["errors"])
        if n_errors and stats["nodes_added"] == 0 and stats["triples_processed"] > 0:
            logger.warning(
                "All %d triples from doc '%s' failed to produce nodes "
                "(%d errors). Run with --verbose to see error details.",
                stats["triples_processed"], doc_id, n_errors,
            )
        logger.info(
            "Ingested doc '%s': %d triples → %d nodes, %d edges, %d proposals",
            doc_id, stats["triples_processed"], stats["nodes_added"],
            stats["edges_added"], stats["proposals_created"],
        )
        return stats

    # ------------------------------------------------------------------
    # Markdown ingestion
    # ------------------------------------------------------------------

    @staticmethod
    def parse_markdown_sections(
        text: str,
        *,
        min_section_chars: int = 80,
        max_section_chars: int = 6000,
        combine_short_sections: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Parse a markdown document into semantically meaningful sections
        split on headings. Each section preserves its heading hierarchy
        for context.

        Args:
            text: Raw markdown text.
            min_section_chars: Sections shorter than this may be combined
                               with the next section.
            max_section_chars: Sections longer than this are split on
                               paragraph boundaries.
            combine_short_sections: If True, merge very short sections
                                    into their next sibling.

        Returns:
            List of section dicts, each with:
              - heading: The section heading text (empty for preamble)
              - level: Heading level (0 for preamble, 1-6 for h1-h6)
              - path: List of ancestor headings (breadcrumb)
              - body: The section body text (may include sub-content)
              - char_count: Length of the body
              - has_code: Whether the section contains code blocks
              - has_list: Whether the section contains list items
              - has_table: Whether the section contains tables
              - links: List of markdown links found [text](url)
        """
        lines = text.split("\n")
        raw_sections: list[dict[str, Any]] = []
        current_heading = ""
        current_level = 0
        current_body_lines: list[str] = []
        heading_stack: list[tuple[int, str]] = []  # (level, heading)

        def _flush_section():
            body = "\n".join(current_body_lines).strip()
            if not body and not current_heading:
                return
            # Build breadcrumb path from heading stack
            path = [h for _, h in heading_stack]
            raw_sections.append({
                "heading": current_heading,
                "level": current_level,
                "path": path.copy(),
                "body": body,
            })

        for line in lines:
            # Detect ATX headings: # Heading
            heading_match = re.match(r"^(#{1,6})\s+(.+?)(?:\s*#*\s*)?$", line)
            if heading_match:
                # Flush previous section
                _flush_section()

                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()

                # Update heading stack — pop deeper/equal levels
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading))

                current_heading = heading
                current_level = level
                current_body_lines = []
                continue

            # Detect setext headings: Heading\n======= or -------
            if (current_body_lines
                    and re.match(r"^[=]{3,}\s*$", line)
                    and current_body_lines[-1].strip()):
                prev = current_body_lines.pop()
                _flush_section()
                level = 1
                heading = prev.strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading))
                current_heading = heading
                current_level = level
                current_body_lines = []
                continue

            if (current_body_lines
                    and re.match(r"^[-]{3,}\s*$", line)
                    and current_body_lines[-1].strip()):
                prev = current_body_lines.pop()
                _flush_section()
                level = 2
                heading = prev.strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading))
                current_heading = heading
                current_level = level
                current_body_lines = []
                continue

            current_body_lines.append(line)

        # Flush final section
        _flush_section()

        # --- Annotate sections ---
        link_re = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
        code_block_re = re.compile(r"```")
        table_row_re = re.compile(r"^\s*\|.+\|")
        list_re = re.compile(r"^\s*[-*+]|\s*\d+\.")

        for section in raw_sections:
            body = section["body"]
            section["char_count"] = len(body)
            section["has_code"] = bool(code_block_re.search(body))
            section["has_list"] = bool(list_re.search(body))
            section["has_table"] = bool(table_row_re.search(body))
            section["links"] = [
                {"text": m.group(1), "url": m.group(2)}
                for m in link_re.finditer(body)
            ]

        # --- Combine short sections ---
        if combine_short_sections and len(raw_sections) > 1:
            merged: list[dict[str, Any]] = []
            carry: dict[str, Any] | None = None

            for section in raw_sections:
                if carry is not None:
                    # Merge carry into this section
                    combined_body = carry["body"]
                    if section["heading"]:
                        combined_body += f"\n\n{'#' * section['level']} {section['heading']}\n\n"
                    combined_body += section["body"]

                    section = {
                        **section,
                        "heading": carry["heading"],
                        "level": carry["level"],
                        "path": carry["path"],
                        "body": combined_body.strip(),
                        "char_count": len(combined_body.strip()),
                        "has_code": carry["has_code"] or section["has_code"],
                        "has_list": carry["has_list"] or section["has_list"],
                        "has_table": carry["has_table"] or section["has_table"],
                        "links": carry["links"] + section["links"],
                    }
                    carry = None

                if section["char_count"] < min_section_chars:
                    carry = section
                    continue

                merged.append(section)

            # Don't drop trailing carry
            if carry is not None:
                if merged:
                    last = merged[-1]
                    combined_body = last["body"]
                    if carry["heading"]:
                        combined_body += f"\n\n{'#' * carry['level']} {carry['heading']}\n\n"
                    combined_body += carry["body"]
                    merged[-1] = {
                        **last,
                        "body": combined_body.strip(),
                        "char_count": len(combined_body.strip()),
                        "has_code": last["has_code"] or carry["has_code"],
                        "has_list": last["has_list"] or carry["has_list"],
                        "has_table": last["has_table"] or carry["has_table"],
                        "links": last["links"] + carry["links"],
                    }
                else:
                    merged.append(carry)

            raw_sections = merged

        # --- Split oversized sections ---
        final_sections: list[dict[str, Any]] = []
        for section in raw_sections:
            if section["char_count"] <= max_section_chars:
                final_sections.append(section)
                continue

            # Split on double newlines (paragraph boundaries)
            paragraphs = re.split(r"\n\n+", section["body"])
            chunk_lines: list[str] = []
            chunk_len = 0
            part_num = 0

            for para in paragraphs:
                para_len = len(para)
                if chunk_len + para_len > max_section_chars and chunk_lines:
                    part_num += 1
                    chunk_body = "\n\n".join(chunk_lines).strip()
                    final_sections.append({
                        **section,
                        "heading": f"{section['heading']} (part {part_num})" if section["heading"] else f"(part {part_num})",
                        "body": chunk_body,
                        "char_count": len(chunk_body),
                        "has_code": bool(code_block_re.search(chunk_body)),
                        "has_list": bool(list_re.search(chunk_body)),
                        "has_table": bool(table_row_re.search(chunk_body)),
                        "links": [
                            {"text": m.group(1), "url": m.group(2)}
                            for m in link_re.finditer(chunk_body)
                        ],
                    })
                    chunk_lines = []
                    chunk_len = 0

                chunk_lines.append(para)
                chunk_len += para_len

            # Remaining content
            if chunk_lines:
                part_num += 1
                chunk_body = "\n\n".join(chunk_lines).strip()
                final_sections.append({
                    **section,
                    "heading": f"{section['heading']} (part {part_num})" if part_num > 1 and section["heading"] else section["heading"],
                    "body": chunk_body,
                    "char_count": len(chunk_body),
                    "has_code": bool(code_block_re.search(chunk_body)),
                    "has_list": bool(list_re.search(chunk_body)),
                    "has_table": bool(table_row_re.search(chunk_body)),
                    "links": [
                        {"text": m.group(1), "url": m.group(2)}
                        for m in link_re.finditer(chunk_body)
                    ],
                })

        return final_sections

    def ingest_markdown(
        self,
        text: str,
        doc_id: str,
        *,
        llm_extract_fn: Callable[[str], list[dict[str, Any]]],
        max_triples_per_section: int = 30,
        min_section_chars: int = 80,
        max_section_chars: int = 6000,
        low_confidence_threshold: float = 0.3,
        add_structure_nodes: bool = True,
        add_structure_edges: bool = True,
        store_body_text: bool = True,
        preserve_source: bool = True,
        original_path: str | Path | None = None,
        doc_properties: dict[str, Any] | None = None,
        progress_fn: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """
        Ingest a markdown document with structure-aware chunking.

        The document is parsed into sections by heading, each section is
        sent through the LLM extraction pipeline independently (with
        heading hierarchy as context), and the resulting triples are
        merged into the graph. Optionally, document and section nodes are
        created to preserve the structural hierarchy.

        Args:
            text: Raw markdown text.
            doc_id: Unique identifier for the document.
            llm_extract_fn: Callable that takes a prompt string and returns
                            a list of triple dicts.
            max_triples_per_section: Max triples to request per section.
            min_section_chars: Sections shorter than this are merged with
                               the next sibling.
            max_section_chars: Sections longer than this are split on
                               paragraph boundaries.
            low_confidence_threshold: Confidence for novel-relation edges.
            add_structure_nodes: Create document and section nodes.
            add_structure_edges: Create structural edges (part_of, contains)
                                 between document/section/entity nodes.
            store_body_text: Store the section body text in the section node's
                             properties (under 'body_text'). This is used by
                             embed_nodes() to build high-quality section embeddings.
                             Increases JSON file size but produces much better
                             embeddings. Set False to keep the file lean.
            preserve_source: Copy the original markdown into the managed
                             sources directory for later reference. Uses
                             content-hashing for deduplication.
            original_path: Original file path for provenance tracking.
            doc_properties: Extra properties to attach to the document node.
            progress_fn: Optional callback for real-time progress reporting.
                         Called with event dicts containing at least an "event"
                         key. Events: "section_start", "section_done",
                         "section_skip".

        Returns:
            Aggregate stats dict with per-section breakdown.
        """
        aggregate_stats: dict[str, Any] = {
            "doc_id": doc_id,
            "total_sections": 0,
            "total_triples": 0,
            "total_nodes_added": 0,
            "total_edges_added": 0,
            "total_proposals_created": 0,
            "total_proposals_augmented": 0,
            "source": None,
            "sections": [],
            "errors": [],
        }

        # Store source file
        ingestion_id = None
        source_content_hash = None
        if preserve_source:
            source_result = self.store_source(
                text, doc_id, original_path=original_path,
            )
            aggregate_stats["source"] = source_result
            ingestion_id = source_result.get("ingestion_id")
            source_content_hash = source_result["content_hash"]

            if source_result["is_duplicate"]:
                logger.warning(
                    "Document '%s' has identical content to '%s'. "
                    "Proceeding with ingestion (may create duplicate nodes).",
                    doc_id, source_result["existing_doc_id"],
                )

            # Add source info to doc properties
            if doc_properties is None:
                doc_properties = {}
            doc_properties["content_hash"] = source_content_hash
            doc_properties["source_stored"] = True
            doc_properties["stored_path"] = source_result["stored_path"]
            doc_properties["ingestion_id"] = ingestion_id
            doc_properties["source_version"] = source_result.get("version", 1)
        else:
            # Generate an ingestion_id even without source storage
            ts = now_iso()
            chash = content_hash(text)
            doc_slug = slugify(doc_id)
            ingestion_id = f"{doc_slug}_{chash}_{ts[:19].replace(':', '').replace('-', '')}"
            source_content_hash = chash
            if doc_properties is None:
                doc_properties = {}
            doc_properties["content_hash"] = source_content_hash
            doc_properties["ingestion_id"] = ingestion_id

        aggregate_stats["ingestion_id"] = ingestion_id
        aggregate_stats["content_hash"] = source_content_hash

        # Parse into sections
        sections = self.parse_markdown_sections(
            text,
            min_section_chars=min_section_chars,
            max_section_chars=max_section_chars,
        )
        aggregate_stats["total_sections"] = len(sections)

        if not sections:
            aggregate_stats["errors"].append("No sections found in markdown.")
            return aggregate_stats

        doc_slug = slugify(doc_id)

        # Create document node
        if add_structure_nodes:
            doc_props = {
                "format": "markdown",
                "total_sections": len(sections),
                "char_count": len(text),
            }
            if doc_properties:
                doc_props.update(doc_properties)

            self.add_node(
                doc_slug,
                type="document",
                label=doc_id,
                properties=doc_props,
                source="markdown_ingest",
            )

        # Track all section node IDs for structural edges
        section_ids: list[str] = []
        prev_section_id: str | None = None

        for i, section in enumerate(sections):
            heading = section["heading"] or f"Section {i + 1}"
            section_slug = slugify(f"{doc_id}-{heading}")
            body = section["body"]

            # Build context prefix from heading hierarchy
            breadcrumb = " > ".join(section["path"]) if section["path"] else doc_id
            context_prefix = f"Document: {doc_id}\nSection: {breadcrumb}\n"
            if section["heading"]:
                context_prefix += f"Heading: {section['heading']}\n"
            context_prefix += "---\n"

            # Create section node
            if add_structure_nodes:
                section_props: dict[str, Any] = {
                    "heading": heading,
                    "level": section["level"],
                    "path": section["path"],
                    "char_count": section["char_count"],
                    "has_code": section["has_code"],
                    "has_list": section["has_list"],
                    "has_table": section["has_table"],
                    "section_index": i,
                    "ingestion_id": ingestion_id,
                    "content_hash": source_content_hash,
                }
                if section["links"]:
                    section_props["link_count"] = len(section["links"])

                if store_body_text:
                    section_props["body_text"] = body

                self.add_node(
                    section_slug,
                    type="section",
                    label=heading,
                    properties=section_props,
                    source=f"doc:{doc_id}",
                )
                section_ids.append(section_slug)

                if add_structure_edges:
                    # Section → Document
                    self.add_edge(
                        section_slug, doc_slug,
                        relation="part_of",
                        source_tag="markdown_ingest",
                        confidence=1.0,
                    )
                    # Document → Section
                    self.add_edge(
                        doc_slug, section_slug,
                        relation="contains",
                        source_tag="markdown_ingest",
                        confidence=1.0,
                    )

                    # Build heading hierarchy edges
                    # Find parent section: the most recent section at a lower level
                    if section["level"] > 0:
                        for prev_idx in range(len(section_ids) - 2, -1, -1):
                            prev_sec = sections[prev_idx] if prev_idx < len(sections) else None
                            if prev_sec and prev_sec["level"] < section["level"]:
                                parent_slug = section_ids[prev_idx]
                                self.add_edge(
                                    section_slug, parent_slug,
                                    relation="part_of",
                                    source_tag="markdown_ingest",
                                    confidence=1.0,
                                )
                                break

            # Skip extraction on very short sections (just structural nodes)
            if section["char_count"] < 40:
                skip_info = {
                    "heading": heading,
                    "skipped": True,
                    "reason": "too_short",
                }
                aggregate_stats["sections"].append(skip_info)
                if progress_fn:
                    progress_fn({
                        "event": "section_skip",
                        "index": i,
                        "total": len(sections),
                        "heading": heading,
                        "reason": "too_short",
                        "char_count": section["char_count"],
                    })
                continue

            # Notify progress callback before LLM extraction
            if progress_fn:
                progress_fn({
                    "event": "section_start",
                    "index": i,
                    "total": len(sections),
                    "heading": heading,
                    "char_count": section["char_count"],
                })

            # Run LLM extraction on this section
            section_text = context_prefix + body
            t0 = time.monotonic()
            section_stats = self.ingest_document(
                section_text,
                doc_id=f"{doc_id}::{heading}",
                llm_extract_fn=llm_extract_fn,
                max_triples=max_triples_per_section,
                low_confidence_threshold=low_confidence_threshold,
                auto_add_doc_node=False,  # we handle doc nodes ourselves
                ingestion_id=ingestion_id,
                content_hash=source_content_hash,
            )
            elapsed = time.monotonic() - t0

            # Link extracted entities to the section node
            if add_structure_nodes and add_structure_edges:
                for nid in list(self._data["nodes"].keys()):
                    node = self._data["nodes"][nid]
                    if node.get("source", "").startswith(f"doc:{doc_id}::"):
                        # Check this entity was just added (has this section's source tag)
                        if node.get("source") == f"doc:{doc_id}::{heading}":
                            self.add_edge(
                                nid, section_slug,
                                relation="documented_by",
                                source_tag="markdown_ingest",
                                confidence=0.9,
                            )

            # Accumulate stats
            aggregate_stats["total_triples"] += section_stats["triples_processed"]
            aggregate_stats["total_nodes_added"] += section_stats["nodes_added"]
            aggregate_stats["total_edges_added"] += section_stats["edges_added"]
            aggregate_stats["total_proposals_created"] += section_stats["proposals_created"]
            aggregate_stats["total_proposals_augmented"] += section_stats["proposals_augmented"]
            section_record = {
                "heading": heading,
                "char_count": section["char_count"],
                "elapsed_seconds": round(elapsed, 1),
                **section_stats,
            }
            aggregate_stats["sections"].append(section_record)
            if section_stats["errors"]:
                aggregate_stats["errors"].extend(section_stats["errors"])

            # Notify progress callback after extraction
            if progress_fn:
                progress_fn({
                    "event": "section_done",
                    "index": i,
                    "total": len(sections),
                    "heading": heading,
                    "char_count": section["char_count"],
                    "elapsed_seconds": round(elapsed, 1),
                    "triples": section_stats["triples_processed"],
                    "nodes_added": section_stats["nodes_added"],
                    "errors": section_stats["errors"],
                })

        # Add links found in the document as lightweight reference edges
        all_links: list[dict[str, str]] = []
        for section in sections:
            all_links.extend(section.get("links", []))

        if all_links and add_structure_nodes:
            unique_urls = {}
            for link in all_links:
                url = link["url"]
                if url.startswith("http") and url not in unique_urls:
                    unique_urls[url] = link["text"]

            if unique_urls:
                self._data["nodes"][doc_slug].setdefault("properties", {})["external_links"] = [
                    {"text": text, "url": url} for url, text in unique_urls.items()
                ]

        logger.info(
            "Markdown ingest '%s': %d sections, %d triples → %d nodes, %d edges",
            doc_id, aggregate_stats["total_sections"],
            aggregate_stats["total_triples"],
            aggregate_stats["total_nodes_added"],
            aggregate_stats["total_edges_added"],
        )
        return aggregate_stats

    def ingest_markdown_file(
        self,
        file_path: str | Path,
        *,
        llm_extract_fn: Callable[[str], list[dict[str, Any]]],
        doc_id: str | None = None,
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Convenience method to ingest a markdown file from disk.

        The source file is automatically copied into the managed sources
        directory (unless preserve_source=False is passed in kwargs).

        Args:
            file_path: Path to the .md file.
            llm_extract_fn: LLM extraction callable.
            doc_id: Override document ID (defaults to filename stem).
            encoding: File encoding.
            **kwargs: Passed through to ingest_markdown().

        Returns:
            Aggregate stats dict from ingest_markdown().
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        text = file_path.read_text(encoding=encoding)
        doc_id = doc_id or file_path.stem

        return self.ingest_markdown(
            text,
            doc_id=doc_id,
            llm_extract_fn=llm_extract_fn,
            original_path=file_path.resolve(),
            doc_properties={"file_path": str(file_path), "file_size": file_path.stat().st_size},
            **kwargs,
        )

    def analyze_relation_patterns(
        self,
        min_occurrences: int = 3,
        status_filter: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Analyze the graph for recurring novel relation patterns.

        Examines edges using non-core relations and returns usage statistics
        to help decide which proposed relations deserve formal adoption.

        Args:
            min_occurrences: Minimum edge count to include a relation.
            status_filter: Only count edges whose relation matches a proposal
                           with this status (e.g., 'pending'). None = all.

        Returns:
            Dict mapping relation name → {count, avg_confidence, sources, proposal_status}.
        """
        core = {r.value for r in CoreRelation}
        proposal_map = {p.name: p for p in self._proposals}

        relation_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_confidence": 0.0, "source_docs": set(), "example_pairs": []}
        )

        for edge in self._data["edges"]:
            rel = edge.get("relation", "")
            if rel in core:
                continue

            if status_filter:
                proposal = proposal_map.get(rel)
                if not proposal or proposal.status != status_filter:
                    continue

            stats = relation_stats[rel]
            stats["count"] += 1
            stats["total_confidence"] += edge.get("confidence", 0.5)
            tag = edge.get("source_tag", "")
            if tag.startswith("doc:"):
                stats["source_docs"].add(tag[4:])
            if len(stats["example_pairs"]) < 5:  # keep up to 5 examples
                stats["example_pairs"].append((edge["source"], edge["target"]))

        # Finalize and filter
        results = {}
        for rel, s in relation_stats.items():
            if s["count"] < min_occurrences:
                continue
            proposal = proposal_map.get(rel)
            results[rel] = {
                "count": s["count"],
                "avg_confidence": round(s["total_confidence"] / s["count"], 3),
                "source_docs": sorted(s["source_docs"]),
                "num_source_docs": len(s["source_docs"]),
                "example_pairs": s["example_pairs"],
                "proposal_status": proposal.status if proposal else "untracked",
                "proposal_justification": proposal.justification if proposal else "",
            }

        return dict(sorted(results.items(), key=lambda x: -x[1]["count"]))

    # ------------------------------------------------------------------
    # Merge / combine graphs
    # ------------------------------------------------------------------

    def merge(self, other: "KnowledgeGraph", *, prefer: str = "other") -> dict[str, int]:
        """
        Merge another KnowledgeGraph into this one.

        Args:
            other: The graph to merge in.
            prefer: On conflict, prefer 'self' or 'other' node data.

        Returns:
            Stats dict with counts of nodes_added, nodes_updated, edges_added.
        """
        stats = {"nodes_added": 0, "nodes_updated": 0, "edges_added": 0}

        for nid, node in other._data["nodes"].items():
            if nid in self._data["nodes"]:
                if prefer == "other":
                    self._data["nodes"][nid] = deepcopy(node)
                stats["nodes_updated"] += 1
            else:
                self._data["nodes"][nid] = deepcopy(node)
                stats["nodes_added"] += 1

        existing_edge_keys = {
            (e["source"], e["target"], e["relation"])
            for e in self._data["edges"]
        }
        for edge in other._data["edges"]:
            key = (edge["source"], edge["target"], edge["relation"])
            if key not in existing_edge_keys:
                self._data["edges"].append(deepcopy(edge))
                existing_edge_keys.add(key)
                stats["edges_added"] += 1

        # Merge custom relations
        my_customs = set(self._data["meta"].get("custom_relations", []))
        other_customs = set(other._data["meta"].get("custom_relations", []))
        self._data["meta"]["custom_relations"] = sorted(my_customs | other_customs)

        # Merge embeddings
        for nid, emb in other._embeddings.items():
            if nid not in self._embeddings or prefer == "other":
                self._embeddings[nid] = emb

        # Merge proposals
        my_proposal_names = {p.name for p in self._proposals}
        for op in other._proposals:
            if op.name not in my_proposal_names:
                self._proposals.append(deepcopy(op))
            else:
                # Augment existing proposal with new examples
                for mp in self._proposals:
                    if mp.name == op.name:
                        for ex in op.examples:
                            if ex not in mp.examples:
                                mp.examples.append(ex)
                        for doc in op.source_docs:
                            if doc not in mp.source_docs:
                                mp.source_docs.append(doc)
                        mp.confidence = max(mp.confidence, op.confidence)
                        break

        self._rebuild_networkx()
        self._dirty = True
        self._dirty_embeddings = bool(other._embeddings)
        return stats

    # ------------------------------------------------------------------
    # RAG helpers
    # ------------------------------------------------------------------

    def get_context_window(
        self,
        node_ids: list[str],
        depth: int = 1,
        max_nodes: int = 50,
        include_embeddings: bool = False,
        boundary_edges: bool = False,
    ) -> dict[str, Any]:
        """
        Build a context payload for LLM consumption.

        Gathers local subgraphs around the given seed nodes, deduplicates,
        and returns a serializable dict sized for a context window.

        Args:
            node_ids: Seed node IDs (e.g., from embedding similarity search).
            depth: How many hops to traverse from each seed.
            max_nodes: Cap on total nodes returned.
            include_embeddings: Whether to include embedding vectors.
            boundary_edges: If True, include edges where *at least one*
                endpoint is in the node set (not just edges where both
                endpoints are present).  This surfaces relationships to
                nodes just outside the context window.

        Returns:
            A dict ready for json.dumps() and injection into an LLM prompt.
        """
        all_node_ids: set[str] = set()
        for nid in node_ids:
            all_node_ids.add(nid)
            neighbors = self.get_neighbors(nid, max_depth=depth)
            all_node_ids.update(n[0] for n in neighbors)

        # Prioritize seed nodes, then sort by confidence
        sorted_ids = sorted(
            all_node_ids,
            key=lambda nid: (
                0 if nid in node_ids else 1,  # seeds first
                -(self._data["nodes"].get(nid, {}).get("confidence", 0)),
            ),
        )[:max_nodes]

        node_set = set(sorted_ids)
        if boundary_edges:
            edge_filter = lambda e: (
                e["source"] in node_set or e["target"] in node_set
            )
        else:
            edge_filter = lambda e: (
                e["source"] in node_set and e["target"] in node_set
            )
        context: dict[str, Any] = {
            "nodes": {
                nid: deepcopy(self._data["nodes"][nid])
                for nid in sorted_ids
                if nid in self._data["nodes"]
            },
            "edges": [
                deepcopy(e)
                for e in self._data["edges"]
                if edge_filter(e)
            ],
        }

        if include_embeddings:
            context["embeddings"] = {
                nid: self._embeddings[nid]
                for nid in sorted_ids
                if nid in self._embeddings
            }

        return context

    # ------------------------------------------------------------------
    # Stats / info
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return len(self._data["nodes"])

    @property
    def num_edges(self) -> int:
        return len(self._data["edges"])

    @property
    def is_dirty(self) -> bool:
        return self._dirty or self._dirty_embeddings

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about the graph."""
        type_counts: dict[str, int] = defaultdict(int)
        for node in self._data["nodes"].values():
            type_counts[node.get("type", "unknown")] += 1

        relation_counts: dict[str, int] = defaultdict(int)
        for edge in self._data["edges"]:
            relation_counts[edge.get("relation", "unknown")] += 1

        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_embeddings": len(self._embeddings),
            "nodes_without_embeddings": len(self.nodes_without_embeddings()),
            "node_type_distribution": dict(type_counts),
            "relation_distribution": dict(relation_counts),
            "connected_components": nx.number_weakly_connected_components(self._G),
            "is_dag": nx.is_directed_acyclic_graph(self._G),
            "density": nx.density(self._G) if self.num_nodes > 0 else 0,
            "proposals_pending": len(self.get_proposals(status=ProposalStatus.PENDING.value)),
            "proposals_accepted": len(self.get_proposals(status=ProposalStatus.ACCEPTED.value)),
            "proposals_rejected": len(self.get_proposals(status=ProposalStatus.REJECTED.value)),
            "sources": self.source_stats(),
        }

    def __repr__(self) -> str:
        pending = len(self.get_proposals(status=ProposalStatus.PENDING.value))
        return (
            f"KnowledgeGraph(nodes={self.num_nodes}, edges={self.num_edges}, "
            f"embeddings={len(self._embeddings)}, pending_proposals={pending}, "
            f"path='{self.graph_path}')"
        )

    # ------------------------------------------------------------------
    # Visualization — Color & style configuration
    # ------------------------------------------------------------------

    # Node type → color mapping (used by both Pyvis and Cytoscape exports)
    NODE_TYPE_COLORS: dict[str, str] = {
        "concept":       "#6366f1",  # indigo
        "entity":        "#8b5cf6",  # violet
        "document":      "#f59e0b",  # amber
        "section":       "#fbbf24",  # yellow
        "technology":    "#3b82f6",  # blue
        "tool":          "#10b981",  # emerald
        "process":       "#f97316",  # orange
        "event":         "#ef4444",  # red
        "person":        "#ec4899",  # pink
        "organization":  "#14b8a6",  # teal
        "code":          "#64748b",  # slate
        "configuration": "#a78bfa",  # light violet
        "artifact":      "#06b6d4",  # cyan
        "custom":        "#9ca3af",  # gray
    }
    DEFAULT_NODE_COLOR = "#9ca3af"

    # Relation → edge color mapping (for the most common core relations)
    RELATION_COLORS: dict[str, str] = {
        "is_a":          "#6366f1",
        "part_of":       "#3b82f6",
        "has_part":      "#3b82f6",
        "depends_on":    "#ef4444",
        "required_by":   "#ef4444",
        "uses":          "#10b981",
        "used_by":       "#10b981",
        "related_to":    "#9ca3af",
        "similar_to":    "#a78bfa",
        "references":    "#f59e0b",
        "implements":    "#14b8a6",
        "extends":       "#8b5cf6",
        "documented_by": "#fbbf24",
        "documents":     "#fbbf24",
        "derived_from":  "#f97316",
        "causes":        "#dc2626",
        "caused_by":     "#dc2626",
    }
    DEFAULT_EDGE_COLOR = "#94a3b8"

    def _node_color(self, node_type: str) -> str:
        return self.NODE_TYPE_COLORS.get(node_type, self.DEFAULT_NODE_COLOR)

    def _edge_color(self, relation: str) -> str:
        return self.RELATION_COLORS.get(relation, self.DEFAULT_EDGE_COLOR)

    # ------------------------------------------------------------------
    # Visualization — Pyvis (quick interactive view)
    # ------------------------------------------------------------------

    def export_pyvis(
        self,
        output_path: str | Path = "graph_pyvis.html",
        *,
        center_node: str | None = None,
        depth: int | None = None,
        height: str = "900px",
        width: str = "100%",
        physics: bool = True,
        show_edge_labels: bool = True,
        node_size_by: str = "degree",
        min_confidence: float = 0.0,
        notebook: bool = False,
    ) -> Path:
        """
        Export an interactive HTML visualization using Pyvis.

        Args:
            output_path: Where to write the HTML file.
            center_node: If set, only show the subgraph around this node.
            depth: Depth for subgraph extraction (requires center_node).
            height: CSS height of the visualization.
            width: CSS width of the visualization.
            physics: Enable physics simulation (draggable layout).
            show_edge_labels: Display relation names on edges.
            node_size_by: Size nodes by 'degree', 'confidence', or 'fixed'.
            min_confidence: Only show edges above this confidence.
            notebook: Set True if running inside a Jupyter notebook.

        Returns:
            Path to the generated HTML file.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError(
                "pyvis is required for Pyvis export. Install it with: "
                "pip install pyvis"
            )

        output_path = Path(output_path)

        # Determine which nodes/edges to render
        if center_node and depth:
            subgraph = self.get_subgraph(center_node, depth=depth)
            render_nodes = subgraph["nodes"]
            render_edges = subgraph["edges"]
        else:
            render_nodes = self._data["nodes"]
            render_edges = self._data["edges"]

        # Filter edges by confidence
        if min_confidence > 0:
            render_edges = [
                e for e in render_edges
                if e.get("confidence", 1.0) >= min_confidence
            ]

        # Compute degree for sizing
        degree: dict[str, int] = defaultdict(int)
        for e in render_edges:
            degree[e["source"]] += 1
            degree[e["target"]] += 1

        net = Network(
            height=height,
            width=width,
            directed=True,
            notebook=notebook,
            cdn_resources="in_line",
        )

        # Configure physics
        if physics:
            net.force_atlas_2based(
                gravity=-80,
                central_gravity=0.01,
                spring_length=150,
                spring_strength=0.05,
                damping=0.4,
            )
        else:
            net.toggle_physics(False)

        # Add nodes
        for nid, node in render_nodes.items():
            label = node.get("label", nid)
            ntype = node.get("type", "custom")
            color = self._node_color(ntype)
            conf = node.get("confidence", 1.0)

            if node_size_by == "degree":
                size = 10 + min(degree.get(nid, 0) * 4, 50)
            elif node_size_by == "confidence":
                size = 10 + int(conf * 30)
            else:
                size = 20

            # Build tooltip
            props = node.get("properties", {})
            tooltip_lines = [
                f"<b>{label}</b>",
                f"ID: {nid}",
                f"Type: {ntype}",
                f"Confidence: {conf:.2f}",
                f"Source: {node.get('source', 'unknown')}",
            ]
            for k, v in props.items():
                tooltip_lines.append(f"{k}: {v}")

            net.add_node(
                nid,
                label=label,
                color=color,
                size=size,
                title="<br>".join(tooltip_lines),
                shape="dot",
                font={"size": 12, "color": "#333333"},
                borderWidth=2,
                borderWidthSelected=4,
            )

        # Add edges
        for edge in render_edges:
            src = edge["source"]
            tgt = edge["target"]
            if src not in render_nodes or tgt not in render_nodes:
                continue

            relation = edge.get("relation", "related_to")
            color = self._edge_color(relation)
            conf = edge.get("confidence", 1.0)

            # Tooltip for edge
            edge_tooltip_lines = [
                f"<b>{relation}</b>",
                f"Confidence: {conf:.2f}",
                f"Source: {edge.get('source_tag', 'unknown')}",
            ]
            edge_props = edge.get("properties", {})
            for k, v in edge_props.items():
                if v:
                    edge_tooltip_lines.append(f"{k}: {v}")

            edge_kwargs: dict[str, Any] = {
                "color": color,
                "title": "<br>".join(edge_tooltip_lines),
                "width": max(1, conf * 3),
                "arrows": "to",
                "smooth": {"type": "curvedCW", "roundness": 0.1},
            }
            if show_edge_labels:
                edge_kwargs["label"] = relation
                edge_kwargs["font"] = {"size": 9, "color": "#666666", "align": "middle"}

            net.add_edge(src, tgt, **edge_kwargs)

        # Write HTML
        net.save_graph(str(output_path))

        # Inject a legend and stats banner into the HTML
        self._inject_pyvis_legend(output_path, render_nodes, render_edges)

        logger.info("Pyvis export: %s (%d nodes, %d edges)",
                     output_path, len(render_nodes), len(render_edges))
        return output_path

    def _inject_pyvis_legend(
        self, path: Path, nodes: dict, edges: list
    ) -> None:
        """Inject a floating legend and stats bar into the Pyvis HTML."""
        # Collect types present in this render
        types_present = sorted({
            n.get("type", "custom") for n in nodes.values()
        })
        relations_present = sorted({
            e.get("relation", "related_to") for e in edges
        })

        legend_items = "".join(
            f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'border-radius:50%;background:{self._node_color(t)}"></span>'
            f'<span style="font-size:12px">{t}</span></div>'
            for t in types_present
        )

        edge_legend_items = "".join(
            f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
            f'<span style="display:inline-block;width:18px;height:3px;'
            f'background:{self._edge_color(r)}"></span>'
            f'<span style="font-size:12px">{r}</span></div>'
            for r in relations_present[:15]  # cap to avoid huge legend
        )

        legend_html = f"""
<div id="kg-legend" style="position:fixed;top:10px;right:10px;background:rgba(255,255,255,0.95);
  border:1px solid #ddd;border-radius:8px;padding:12px 16px;z-index:9999;
  font-family:system-ui,-apple-system,sans-serif;max-height:80vh;overflow-y:auto;
  box-shadow:0 2px 8px rgba(0,0,0,0.1);min-width:160px">
  <div style="font-weight:600;font-size:13px;margin-bottom:8px;color:#333">
    {len(nodes)} nodes · {len(edges)} edges
  </div>
  <div style="font-weight:600;font-size:11px;color:#666;margin-bottom:4px">NODE TYPES</div>
  {legend_items}
  <div style="font-weight:600;font-size:11px;color:#666;margin:8px 0 4px">RELATIONS</div>
  {edge_legend_items}
  <div style="margin-top:8px;font-size:10px;color:#999">
    Scroll to zoom · Drag to pan · Click nodes to select
  </div>
</div>
"""
        html = path.read_text(encoding="utf-8")
        html = html.replace("</body>", f"{legend_html}</body>")
        path.write_text(html, encoding="utf-8")

    # ------------------------------------------------------------------
    # Visualization — Cytoscape.js (detailed interactive view)
    # ------------------------------------------------------------------

    def export_cytoscape(
        self,
        output_path: str | Path = "graph_cytoscape.html",
        *,
        center_node: str | None = None,
        depth: int | None = None,
        min_confidence: float = 0.0,
        layout: str = "cose",
        title: str = "Knowledge Graph",
    ) -> Path:
        """
        Export a detailed interactive HTML visualization using Cytoscape.js.

        The output is a single self-contained HTML file with:
          - Cytoscape.js loaded from CDN
          - Full graph data embedded as JSON
          - Interactive controls: layout switching, search, filtering,
            node/edge detail panel, confidence slider, type toggles
          - Color coding by node type and relation type
          - Click-to-focus neighborhood exploration

        Args:
            output_path: Where to write the HTML file.
            center_node: If set, only show the subgraph around this node.
            depth: Depth for subgraph extraction (requires center_node).
            min_confidence: Only include edges above this confidence.
            layout: Initial layout algorithm ('cose', 'circle', 'grid',
                    'breadthfirst', 'concentric').
            title: Page title.

        Returns:
            Path to the generated HTML file.
        """
        output_path = Path(output_path)

        # Determine which nodes/edges to render
        if center_node and depth:
            subgraph = self.get_subgraph(center_node, depth=depth)
            render_nodes = subgraph["nodes"]
            render_edges = subgraph["edges"]
        else:
            render_nodes = self._data["nodes"]
            render_edges = self._data["edges"]

        # Filter edges by confidence
        if min_confidence > 0:
            render_edges = [
                e for e in render_edges
                if e.get("confidence", 1.0) >= min_confidence
            ]

        # Build Cytoscape elements
        elements = []

        # Compute degree for sizing
        degree: dict[str, int] = defaultdict(int)
        for e in render_edges:
            degree[e["source"]] += 1
            degree[e["target"]] += 1

        for nid, node in render_nodes.items():
            ntype = node.get("type", "custom")
            elements.append({
                "group": "nodes",
                "data": {
                    "id": nid,
                    "label": node.get("label", nid),
                    "type": ntype,
                    "color": self._node_color(ntype),
                    "confidence": node.get("confidence", 1.0),
                    "source": node.get("source", "unknown"),
                    "degree": degree.get(nid, 0),
                    "properties": node.get("properties", {}),
                },
            })

        for i, edge in enumerate(render_edges):
            src = edge["source"]
            tgt = edge["target"]
            if src not in render_nodes or tgt not in render_nodes:
                continue
            relation = edge.get("relation", "related_to")
            elements.append({
                "group": "edges",
                "data": {
                    "id": f"e{i}",
                    "source": src,
                    "target": tgt,
                    "relation": relation,
                    "color": self._edge_color(relation),
                    "confidence": edge.get("confidence", 1.0),
                    "source_tag": edge.get("source_tag", "unknown"),
                    "weight": edge.get("weight", 1.0),
                    "properties": edge.get("properties", {}),
                },
            })

        # Collect types and relations for controls
        types_present = sorted({n.get("type", "custom") for n in render_nodes.values()})
        relations_present = sorted({e.get("relation", "related_to") for e in render_edges})

        type_colors_json = json.dumps({t: self._node_color(t) for t in types_present})
        relation_colors_json = json.dumps({r: self._edge_color(r) for r in relations_present})
        elements_json = json.dumps(elements, cls=GraphEncoder)

        # Proposal data for the panel
        pending_proposals = [
            {"name": p.name, "confidence": p.confidence,
             "justification": p.justification, "num_examples": len(p.examples)}
            for p in self.get_proposals(status=ProposalStatus.PENDING.value)
        ]
        proposals_json = json.dumps(pending_proposals)

        stats = {
            "nodes": len(render_nodes),
            "edges": len(render_edges),
            "components": nx.number_weakly_connected_components(self._G),
            "pending_proposals": len(pending_proposals),
        }
        stats_json = json.dumps(stats)

        html = self._cytoscape_html_template(
            title=title,
            elements_json=elements_json,
            type_colors_json=type_colors_json,
            relation_colors_json=relation_colors_json,
            proposals_json=proposals_json,
            stats_json=stats_json,
            initial_layout=layout,
            types_present=types_present,
            relations_present=relations_present,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

        logger.info("Cytoscape export: %s (%d nodes, %d edges)",
                     output_path, len(render_nodes), len(render_edges))
        return output_path

    def _cytoscape_html_template(
        self,
        *,
        title: str,
        elements_json: str,
        type_colors_json: str,
        relation_colors_json: str,
        proposals_json: str,
        stats_json: str,
        initial_layout: str,
        types_present: list[str],
        relations_present: list[str],
    ) -> str:
        """Generate the full self-contained Cytoscape.js HTML page."""

        # Build checkbox HTML for node type filters
        type_checkboxes = "".join(
            f'<label class="filter-item">'
            f'<input type="checkbox" checked data-type="{t}" onchange="applyFilters()">'
            f'<span class="color-dot" style="background:{self._node_color(t)}"></span>'
            f'{t}</label>'
            for t in types_present
        )

        # Build checkbox HTML for relation filters
        relation_checkboxes = "".join(
            f'<label class="filter-item">'
            f'<input type="checkbox" checked data-relation="{r}" onchange="applyFilters()">'
            f'<span class="color-line" style="background:{self._edge_color(r)}"></span>'
            f'{r}</label>'
            for r in relations_present
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; overflow: hidden; }}

  #cy {{ width: 100vw; height: 100vh; position: absolute; top: 0; left: 0; }}

  /* --- Stats banner --- */
  #stats-bar {{
    position: fixed; top: 0; left: 0; right: 0; height: 42px; z-index: 100;
    background: rgba(15, 23, 42, 0.92); border-bottom: 1px solid #334155;
    display: flex; align-items: center; padding: 0 16px; gap: 24px;
    backdrop-filter: blur(8px); font-size: 13px;
  }}
  #stats-bar .stat {{ color: #94a3b8; }}
  #stats-bar .stat b {{ color: #e2e8f0; }}
  #stats-bar .title {{ font-weight: 700; color: #f8fafc; margin-right: 12px; font-size: 14px; }}

  /* --- Control panel --- */
  #controls {{
    position: fixed; top: 52px; left: 10px; width: 260px; z-index: 100;
    background: rgba(15, 23, 42, 0.92); border: 1px solid #334155;
    border-radius: 8px; backdrop-filter: blur(8px);
    max-height: calc(100vh - 64px); overflow-y: auto;
  }}
  .panel-section {{
    padding: 10px 14px; border-bottom: 1px solid #1e293b;
  }}
  .panel-section:last-child {{ border-bottom: none; }}
  .panel-title {{
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; color: #64748b; margin-bottom: 6px;
  }}

  /* Search */
  #search-input {{
    width: 100%; padding: 6px 10px; background: #1e293b; border: 1px solid #334155;
    border-radius: 5px; color: #e2e8f0; font-size: 13px; outline: none;
  }}
  #search-input:focus {{ border-color: #6366f1; }}
  #search-results {{
    margin-top: 4px; max-height: 120px; overflow-y: auto; font-size: 12px;
  }}
  .search-result {{
    padding: 3px 6px; cursor: pointer; border-radius: 3px; color: #94a3b8;
  }}
  .search-result:hover {{ background: #1e293b; color: #e2e8f0; }}

  /* Layout selector */
  .layout-btn {{
    padding: 4px 10px; background: #1e293b; border: 1px solid #334155;
    border-radius: 4px; color: #94a3b8; font-size: 11px; cursor: pointer;
    transition: all 0.15s;
  }}
  .layout-btn:hover {{ border-color: #6366f1; color: #e2e8f0; }}
  .layout-btn.active {{ background: #6366f1; border-color: #6366f1; color: white; }}
  .layout-grid {{ display: flex; flex-wrap: wrap; gap: 4px; }}

  /* Confidence slider */
  .slider-row {{ display: flex; align-items: center; gap: 8px; }}
  .slider-row input[type=range] {{ flex: 1; accent-color: #6366f1; }}
  .slider-val {{ font-size: 12px; color: #94a3b8; min-width: 32px; text-align: right; }}

  /* Filter checkboxes */
  .filter-item {{
    display: flex; align-items: center; gap: 6px; font-size: 12px;
    color: #94a3b8; cursor: pointer; padding: 1px 0;
  }}
  .filter-item input {{ accent-color: #6366f1; }}
  .color-dot {{
    display: inline-block; width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
  }}
  .color-line {{
    display: inline-block; width: 14px; height: 3px; border-radius: 1px; flex-shrink: 0;
  }}
  .filter-scroll {{ max-height: 140px; overflow-y: auto; }}

  /* --- Detail panel --- */
  #detail-panel {{
    position: fixed; top: 52px; right: 10px; width: 320px; z-index: 100;
    background: rgba(15, 23, 42, 0.92); border: 1px solid #334155;
    border-radius: 8px; backdrop-filter: blur(8px);
    max-height: calc(100vh - 64px); overflow-y: auto;
    display: none;
  }}
  #detail-panel.visible {{ display: block; }}
  .detail-header {{
    padding: 12px 14px; border-bottom: 1px solid #1e293b;
    display: flex; justify-content: space-between; align-items: center;
  }}
  .detail-header h3 {{ font-size: 15px; font-weight: 600; color: #f8fafc; }}
  .detail-close {{
    background: none; border: none; color: #64748b; font-size: 18px;
    cursor: pointer; padding: 2px 6px; border-radius: 4px;
  }}
  .detail-close:hover {{ background: #1e293b; color: #e2e8f0; }}
  .detail-row {{
    display: flex; justify-content: space-between; padding: 4px 0;
    font-size: 12px; border-bottom: 1px solid #1e293b;
  }}
  .detail-key {{ color: #64748b; }}
  .detail-val {{ color: #e2e8f0; text-align: right; max-width: 180px; word-break: break-word; }}
  .detail-section {{ padding: 10px 14px; }}
  .neighbors-list {{ margin-top: 4px; }}
  .neighbor-item {{
    font-size: 12px; color: #94a3b8; padding: 2px 0; cursor: pointer;
  }}
  .neighbor-item:hover {{ color: #6366f1; }}
  .badge {{
    display: inline-block; padding: 1px 7px; border-radius: 10px;
    font-size: 10px; font-weight: 600;
  }}

  /* Scrollbar styling */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: #475569; }}
</style>
</head>
<body>

<!-- Stats Banner -->
<div id="stats-bar">
  <span class="title">{title}</span>
  <span class="stat"><b id="stat-nodes">0</b> nodes</span>
  <span class="stat"><b id="stat-edges">0</b> edges</span>
  <span class="stat"><b id="stat-visible-nodes">0</b> visible</span>
  <span class="stat" id="proposals-stat" style="display:none">
    <b id="stat-proposals">0</b> pending proposals
  </span>
</div>

<!-- Control Panel -->
<div id="controls">
  <!-- Search -->
  <div class="panel-section">
    <div class="panel-title">Search Nodes</div>
    <input type="text" id="search-input" placeholder="Type to search..." oninput="onSearch(this.value)">
    <div id="search-results"></div>
  </div>

  <!-- Layout -->
  <div class="panel-section">
    <div class="panel-title">Layout</div>
    <div class="layout-grid">
      <button class="layout-btn" data-layout="cose" onclick="changeLayout('cose')">Force</button>
      <button class="layout-btn" data-layout="circle" onclick="changeLayout('circle')">Circle</button>
      <button class="layout-btn" data-layout="breadthfirst" onclick="changeLayout('breadthfirst')">Tree</button>
      <button class="layout-btn" data-layout="grid" onclick="changeLayout('grid')">Grid</button>
      <button class="layout-btn" data-layout="concentric" onclick="changeLayout('concentric')">Radial</button>
    </div>
  </div>

  <!-- Confidence filter -->
  <div class="panel-section">
    <div class="panel-title">Min Confidence</div>
    <div class="slider-row">
      <input type="range" id="confidence-slider" min="0" max="100" value="0"
             oninput="onConfidenceChange(this.value)">
      <span class="slider-val" id="confidence-val">0.00</span>
    </div>
  </div>

  <!-- Node type filters -->
  <div class="panel-section">
    <div class="panel-title">Node Types</div>
    <div class="filter-scroll" id="type-filters">
      {type_checkboxes}
    </div>
  </div>

  <!-- Relation filters -->
  <div class="panel-section">
    <div class="panel-title">Relations</div>
    <div class="filter-scroll" id="relation-filters">
      {relation_checkboxes}
    </div>
  </div>

  <!-- Actions -->
  <div class="panel-section">
    <div class="panel-title">Actions</div>
    <div class="layout-grid">
      <button class="layout-btn" onclick="cy.fit(undefined, 40)">Fit View</button>
      <button class="layout-btn" onclick="resetFilters()">Reset Filters</button>
      <button class="layout-btn" onclick="highlightHighDegree()">Hub Nodes</button>
      <button class="layout-btn" onclick="exportPNG()">Export PNG</button>
    </div>
  </div>
</div>

<!-- Detail Panel -->
<div id="detail-panel">
  <div class="detail-header">
    <h3 id="detail-title">—</h3>
    <button class="detail-close" onclick="closeDetail()">×</button>
  </div>
  <div class="detail-section" id="detail-content"></div>
  <div class="detail-section" id="detail-neighbors">
    <div class="panel-title">Connections</div>
    <div class="neighbors-list" id="neighbors-list"></div>
  </div>
</div>

<!-- Cytoscape container -->
<div id="cy"></div>

<script>
// --- Data ---
const elements = {elements_json};
const typeColors = {type_colors_json};
const relationColors = {relation_colors_json};
const proposals = {proposals_json};
const graphStats = {stats_json};

// --- Init Cytoscape ---
const cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: elements,
  style: [
    {{
      selector: 'node',
      style: {{
        'background-color': 'data(color)',
        'label': 'data(label)',
        'color': '#cbd5e1',
        'font-size': '11px',
        'text-valign': 'bottom',
        'text-margin-y': 6,
        'text-outline-color': '#0f172a',
        'text-outline-width': 2,
        'width': function(ele) {{ return 16 + Math.min(ele.data('degree') * 3, 40); }},
        'height': function(ele) {{ return 16 + Math.min(ele.data('degree') * 3, 40); }},
        'border-width': 2,
        'border-color': '#0f172a',
        'opacity': 1,
        'transition-property': 'opacity, border-color, border-width, width, height',
        'transition-duration': '0.2s',
      }}
    }},
    {{
      selector: 'node:selected',
      style: {{
        'border-color': '#f8fafc',
        'border-width': 3,
        'font-weight': 'bold',
      }}
    }},
    {{
      selector: 'node.dimmed',
      style: {{ 'opacity': 0.15 }}
    }},
    {{
      selector: 'node.highlighted',
      style: {{
        'border-color': '#fbbf24',
        'border-width': 4,
        'font-size': '13px',
        'font-weight': 'bold',
        'z-index': 999,
      }}
    }},
    {{
      selector: 'edge',
      style: {{
        'line-color': 'data(color)',
        'target-arrow-color': 'data(color)',
        'target-arrow-shape': 'triangle',
        'arrow-scale': 0.8,
        'width': function(ele) {{ return Math.max(1, ele.data('confidence') * 3); }},
        'curve-style': 'bezier',
        'opacity': 0.7,
        'label': 'data(relation)',
        'font-size': '9px',
        'color': '#64748b',
        'text-rotation': 'autorotate',
        'text-outline-color': '#0f172a',
        'text-outline-width': 1.5,
        'transition-property': 'opacity, line-color, width',
        'transition-duration': '0.2s',
      }}
    }},
    {{
      selector: 'edge.dimmed',
      style: {{ 'opacity': 0.05 }}
    }},
    {{
      selector: 'edge.highlighted',
      style: {{ 'opacity': 1, 'width': 3, 'z-index': 999 }}
    }},
    {{
      selector: 'node.hidden, edge.hidden',
      style: {{ 'display': 'none' }}
    }},
  ],
  layout: {{ name: '{initial_layout}', animate: true, animationDuration: 600,
             nodeRepulsion: function(){{ return 8000; }}, idealEdgeLength: function(){{ return 100; }},
             padding: 60 }},
  minZoom: 0.1,
  maxZoom: 5,
  wheelSensitivity: 0.3,
}});

// --- Stats ---
function updateStats() {{
  const visibleNodes = cy.nodes(':visible').length;
  const visibleEdges = cy.edges(':visible').length;
  document.getElementById('stat-nodes').textContent = graphStats.nodes;
  document.getElementById('stat-edges').textContent = graphStats.edges;
  document.getElementById('stat-visible-nodes').textContent = visibleNodes;
  if (graphStats.pending_proposals > 0) {{
    document.getElementById('proposals-stat').style.display = '';
    document.getElementById('stat-proposals').textContent = graphStats.pending_proposals;
  }}
}}
updateStats();

// --- Layout ---
let currentLayout = '{initial_layout}';
document.querySelector(`[data-layout="${{currentLayout}}"]`)?.classList.add('active');

function changeLayout(name) {{
  document.querySelectorAll('.layout-btn[data-layout]').forEach(b => b.classList.remove('active'));
  document.querySelector(`[data-layout="${{name}}"]`)?.classList.add('active');
  currentLayout = name;
  const opts = {{ name, animate: true, animationDuration: 500, padding: 60 }};
  if (name === 'cose') {{
    opts.nodeRepulsion = function() {{ return 8000; }};
    opts.idealEdgeLength = function() {{ return 100; }};
  }}
  if (name === 'breadthfirst') {{
    opts.spacingFactor = 1.2;
  }}
  if (name === 'concentric') {{
    opts.concentric = function(n) {{ return n.degree(); }};
    opts.levelWidth = function() {{ return 2; }};
  }}
  cy.layout(opts).run();
}}

// --- Search ---
function onSearch(query) {{
  const results = document.getElementById('search-results');
  if (!query || query.length < 2) {{
    results.innerHTML = '';
    cy.nodes().removeClass('highlighted dimmed');
    cy.edges().removeClass('highlighted dimmed');
    return;
  }}
  const q = query.toLowerCase();
  const matches = cy.nodes().filter(n =>
    n.data('label').toLowerCase().includes(q) ||
    n.data('id').toLowerCase().includes(q) ||
    n.data('type').toLowerCase().includes(q)
  );
  results.innerHTML = matches.map(n =>
    `<div class="search-result" onclick="focusNode('${{n.id()}}')">${{n.data('label')}} <span style="color:#64748b">(${{n.data('type')}})</span></div>`
  ).join('');

  // Visual highlight
  cy.nodes().addClass('dimmed');
  cy.edges().addClass('dimmed');
  matches.removeClass('dimmed').addClass('highlighted');
  matches.connectedEdges().removeClass('dimmed');
}}

function focusNode(nodeId) {{
  const node = cy.getElementById(nodeId);
  if (!node.length) return;
  cy.nodes().removeClass('highlighted dimmed');
  cy.edges().removeClass('highlighted dimmed');
  showNodeNeighborhood(node);
  cy.animate({{ center: {{ eles: node }}, zoom: 1.5 }}, {{ duration: 400 }});
  showDetail(node);
}}

// --- Confidence filter ---
let minConfidence = 0;
function onConfidenceChange(val) {{
  minConfidence = val / 100;
  document.getElementById('confidence-val').textContent = minConfidence.toFixed(2);
  applyFilters();
}}

// --- Filters ---
function applyFilters() {{
  // Get active types
  const activeTypes = new Set();
  document.querySelectorAll('#type-filters input:checked').forEach(cb => activeTypes.add(cb.dataset.type));

  // Get active relations
  const activeRelations = new Set();
  document.querySelectorAll('#relation-filters input:checked').forEach(cb => activeRelations.add(cb.dataset.relation));

  // Apply node visibility
  cy.nodes().forEach(node => {{
    const visible = activeTypes.has(node.data('type')) &&
                    node.data('confidence') >= minConfidence;
    node.toggleClass('hidden', !visible);
  }});

  // Apply edge visibility
  cy.edges().forEach(edge => {{
    const srcVisible = !edge.source().hasClass('hidden');
    const tgtVisible = !edge.target().hasClass('hidden');
    const relVisible = activeRelations.has(edge.data('relation'));
    const confVisible = edge.data('confidence') >= minConfidence;
    edge.toggleClass('hidden', !(srcVisible && tgtVisible && relVisible && confVisible));
  }});

  updateStats();
}}

function resetFilters() {{
  document.querySelectorAll('#type-filters input, #relation-filters input').forEach(cb => cb.checked = true);
  document.getElementById('confidence-slider').value = 0;
  document.getElementById('confidence-val').textContent = '0.00';
  minConfidence = 0;
  cy.nodes().removeClass('hidden highlighted dimmed');
  cy.edges().removeClass('hidden highlighted dimmed');
  document.getElementById('search-input').value = '';
  document.getElementById('search-results').innerHTML = '';
  updateStats();
}}

// --- Node interaction ---
function showNodeNeighborhood(node) {{
  const neighborhood = node.neighborhood().add(node);
  cy.elements().addClass('dimmed');
  neighborhood.removeClass('dimmed');
  neighborhood.edges().addClass('highlighted');
  node.addClass('highlighted');
}}

cy.on('tap', 'node', function(evt) {{
  const node = evt.target;
  cy.elements().removeClass('highlighted dimmed');
  showNodeNeighborhood(node);
  showDetail(node);
}});

cy.on('tap', 'edge', function(evt) {{
  const edge = evt.target;
  cy.elements().removeClass('highlighted dimmed');
  cy.elements().addClass('dimmed');
  edge.removeClass('dimmed').addClass('highlighted');
  edge.source().removeClass('dimmed').addClass('highlighted');
  edge.target().removeClass('dimmed').addClass('highlighted');
  showEdgeDetail(edge);
}});

cy.on('tap', function(evt) {{
  if (evt.target === cy) {{
    cy.elements().removeClass('highlighted dimmed');
    closeDetail();
  }}
}});

// --- Detail panel ---
function showDetail(node) {{
  const panel = document.getElementById('detail-panel');
  const d = node.data();
  document.getElementById('detail-title').textContent = d.label;

  let html = '';
  html += detailRow('ID', d.id);
  html += detailRow('Type', `<span class="badge" style="background:${{d.color}};color:#fff">${{d.type}}</span>`);
  html += detailRow('Confidence', d.confidence.toFixed(2));
  html += detailRow('Source', d.source);
  html += detailRow('Degree', d.degree);

  const props = d.properties || {{}};
  Object.entries(props).forEach(([k, v]) => {{
    html += detailRow(k, typeof v === 'object' ? JSON.stringify(v) : v);
  }});

  document.getElementById('detail-content').innerHTML = html;

  // Neighbors
  const neighbors = node.neighborhood('node');
  let nhtml = '';
  neighbors.forEach(n => {{
    // Find the connecting edge to show the relation
    const edges = node.edgesWith(n);
    const relations = edges.map(e => e.data('relation')).join(', ');
    nhtml += `<div class="neighbor-item" onclick="focusNode('${{n.id()}}')">
      <span class="color-dot" style="background:${{n.data('color')}};display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px"></span>
      ${{n.data('label')}} <span style="color:#64748b;font-size:11px">(${{relations}})</span>
    </div>`;
  }});
  document.getElementById('neighbors-list').innerHTML = nhtml || '<span style="color:#64748b;font-size:12px">No connections</span>';

  panel.classList.add('visible');
}}

function showEdgeDetail(edge) {{
  const panel = document.getElementById('detail-panel');
  const d = edge.data();
  document.getElementById('detail-title').textContent = d.relation;

  let html = '';
  html += detailRow('Source', d.source);
  html += detailRow('Target', d.target);
  html += detailRow('Relation', `<span style="color:${{d.color}};font-weight:600">${{d.relation}}</span>`);
  html += detailRow('Confidence', d.confidence.toFixed(2));
  html += detailRow('Provenance', d.source_tag);
  html += detailRow('Weight', d.weight);

  const props = d.properties || {{}};
  Object.entries(props).forEach(([k, v]) => {{
    if (v) html += detailRow(k, typeof v === 'object' ? JSON.stringify(v) : v);
  }});

  document.getElementById('detail-content').innerHTML = html;
  document.getElementById('neighbors-list').innerHTML = '';
  panel.classList.add('visible');
}}

function detailRow(key, val) {{
  return `<div class="detail-row"><span class="detail-key">${{key}}</span><span class="detail-val">${{val}}</span></div>`;
}}

function closeDetail() {{
  document.getElementById('detail-panel').classList.remove('visible');
}}

// --- Actions ---
function highlightHighDegree() {{
  cy.elements().removeClass('highlighted dimmed');
  const nodes = cy.nodes(':visible');
  if (!nodes.length) return;
  const degrees = nodes.map(n => n.degree());
  const maxDeg = Math.max(...degrees);
  const threshold = Math.max(maxDeg * 0.5, 2);
  cy.elements().addClass('dimmed');
  nodes.forEach(n => {{
    if (n.degree() >= threshold) {{
      n.removeClass('dimmed').addClass('highlighted');
      n.connectedEdges().removeClass('dimmed');
      n.neighborhood('node').removeClass('dimmed');
    }}
  }});
}}

function exportPNG() {{
  const png = cy.png({{ full: true, scale: 2, bg: '#0f172a' }});
  const link = document.createElement('a');
  link.href = png;
  link.download = 'knowledge_graph.png';
  link.click();
}}

// --- Keyboard shortcuts ---
document.addEventListener('keydown', function(e) {{
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'Escape') {{
    cy.elements().removeClass('highlighted dimmed');
    closeDetail();
  }}
  if (e.key === 'f' || e.key === 'F') {{
    cy.fit(undefined, 40);
  }}
  if (e.key === '/') {{
    e.preventDefault();
    document.getElementById('search-input').focus();
  }}
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# String-triple parser
# ---------------------------------------------------------------------------

# Relation phrases that can appear in bare-string triples returned by LLMs.
# Ordered longest-first so greedy matching picks the most specific phrase.
_STRING_TRIPLE_VERBS: list[str] = sorted(
    [
        " is a ",
        " is an ",
        " is ",
        " are ",
        " has ",
        " have ",
        " had ",
        " uses ",
        " used by ",
        " provides ",
        " contains ",
        " implements ",
        " runs ",
        " runs on ",
        " is part of ",
        " is located in ",
        " is connected to ",
        " is based on ",
        " is used by ",
        " is used for ",
        " is composed of ",
        " is derived from ",
        " is related to ",
        " is associated with ",
        " depends on ",
        " belongs to ",
        " supports ",
        " enables ",
        " produces ",
        " processes ",
        " communicates with ",
        " interfaces with ",
        " connects to ",
        " operates on ",
        " controls ",
        " manages ",
        " generates ",
        " receives ",
        " transmits ",
        " stores ",
        " loads ",
        " reads ",
        " writes ",
        " extends ",
        " inherits from ",
        " calls ",
        " invokes ",
        " wraps ",
    ],
    key=len,
    reverse=True,
)


def _parse_string_triple(text: str) -> dict[str, str] | None:
    """Try to parse a bare-string triple like ``'A has B'`` into a dict.

    Returns ``{"source": ..., "relation": ..., "target": ...}`` on success,
    or ``None`` if the string cannot be split into a recognisable triple.
    """
    text = text.strip()
    if not text:
        return None

    for verb in _STRING_TRIPLE_VERBS:
        idx = text.lower().find(verb.lower())
        if idx > 0:
            source = text[:idx].strip()
            target = text[idx + len(verb):].strip()
            if source and target:
                relation = verb.strip()
                return {
                    "source": source,
                    "relation": relation,
                    "target": target,
                }
    return None


# ---------------------------------------------------------------------------
# JSON salvage helper
# ---------------------------------------------------------------------------

def _salvage_truncated_json(raw: str) -> list[dict[str, Any]] | None:
    """Try to recover complete objects from a truncated JSON array.

    When the LLM hits its token limit the JSON is cut off mid-object,
    e.g. ``[{...}, {... <eof>``. This finds the last complete object
    boundary and closes the array so the valid prefix can be parsed.

    Returns the list of recovered dicts, or None if recovery fails.
    """
    # Must start with '['
    if not raw.lstrip().startswith("["):
        return None

    # Walk backwards from the end to find the last '}' that could
    # close a complete object inside the top-level array.
    last_brace = raw.rfind("}")
    if last_brace == -1:
        return None

    # Close the array right after that brace
    candidate = raw[: last_brace + 1].rstrip().rstrip(",") + "]"
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, list) and all(isinstance(o, dict) for o in parsed):
        return parsed
    return None


# ---------------------------------------------------------------------------
# CLI for quick inspection
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Graph Inspector")
    parser.add_argument("path", help="Path to graph JSON file")
    parser.add_argument("--stats", action="store_true", help="Print graph statistics")
    parser.add_argument("--node", help="Look up a specific node by ID")
    parser.add_argument("--neighbors", help="Get neighbors of a node")
    parser.add_argument("--depth", type=int, default=1, help="Neighbor search depth")
    parser.add_argument("--split", help="Export split files to this directory")
    parser.add_argument("--proposals", action="store_true",
                        help="Show pending relation proposals")
    parser.add_argument("--accept", help="Accept a proposed relation by name")
    parser.add_argument("--accept-all", action="store_true",
                        help="Accept all pending relation proposals")
    parser.add_argument("--reject", help="Reject a proposed relation by name")
    parser.add_argument("--patterns", action="store_true",
                        help="Analyze novel relation patterns in the graph")
    parser.add_argument("--pyvis", nargs="?", const="graph_pyvis.html",
                        help="Export Pyvis visualization (optionally specify output path)")
    parser.add_argument("--cytoscape", nargs="?", const="graph_cytoscape.html",
                        help="Export Cytoscape visualization (optionally specify output path)")
    parser.add_argument("--center", help="Center visualization on this node (use with --pyvis/--cytoscape)")
    parser.add_argument("--preview-md", help="Preview section breakdown of a markdown file (dry run)")
    parser.add_argument("--ingest-md", help="Ingest a markdown file into the graph (stores source, creates structure nodes)")
    parser.add_argument("--sections", action="store_true",
                        help="Show full section details when used with --preview-md or --ingest-md")
    parser.add_argument("--query-model", "--ollama", nargs="?", const="qwen3-coder:30b",
                        metavar="MODEL", dest="query_model",
                        help="Ollama model for LLM extraction during ingestion (default: qwen3-coder:30b)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama server URL for LLM extraction (default: http://localhost:11434)")
    parser.add_argument("--embed-url", default=None, metavar="URL",
                        help="Ollama server URL for embeddings (default: same as --ollama-url)")
    parser.add_argument("--embed-model", default=None, metavar="MODEL",
                        help="Ollama embedding model for node embeddings during ingestion "
                             "(default: auto-detect from graph, or nomic-embed-text)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip automatic visualization export after ingestion")
    parser.add_argument("--auto-accept", action="store_true",
                        help="Automatically accept all new relation proposals created during ingestion")
    parser.add_argument("--sources", action="store_true",
                        help="List all stored source files")
    parser.add_argument("--check-sources", action="store_true",
                        help="Verify integrity of stored source files")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed progress, timing, and debug info")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress per-section output; only show final summary and errors")
    args = parser.parse_args()

    # Default --embed-url to --ollama-url when not explicitly set
    if args.embed_url is None:
        args.embed_url = args.ollama_url

    # Configure logging level from CLI flags
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    elif args.quiet:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    kg = KnowledgeGraph(args.path)

    if args.stats:
        import pprint
        pprint.pprint(kg.stats())

    if args.node:
        node = kg.get_node(args.node)
        if node:
            print(json.dumps(node, indent=2, cls=GraphEncoder))
        else:
            print(f"Node '{args.node}' not found.")

    if args.neighbors:
        neighbors = kg.get_neighbors(args.neighbors, max_depth=args.depth)
        for nid, data in neighbors:
            print(f"  {nid}: {data.get('label', nid)} ({data.get('type', '?')})")

    if args.split:
        nodes_path, edges_path = kg.export_split(args.split)
        print(f"Exported: {nodes_path}, {edges_path}")

    if args.proposals:
        proposals = kg.get_proposals()
        if not proposals:
            print("No pending proposals.")
        for p in proposals:
            print(f"\n  [{p.status.upper()}] {p.name} (confidence: {p.confidence:.2f})")
            print(f"    Justification: {p.justification or '(none)'}")
            print(f"    Examples: {len(p.examples)} | Source docs: {len(p.source_docs)}")
            for ex in p.examples[:3]:
                print(f"      {ex.get('source', '?')} → {ex.get('target', '?')}"
                      f"  ({ex.get('context', '')[:60]})")

    if args.accept:
        if kg.accept_proposal(args.accept):
            kg.save()
            print(f"Accepted and registered relation: '{args.accept}'")
        else:
            print(f"No pending proposal named '{args.accept}'.")

    if args.accept_all:
        pending = kg.get_proposals()
        if not pending:
            print("No pending proposals to accept.")
        else:
            for p in pending:
                kg.accept_proposal(p.name)
                print(f"  Accepted: '{p.name}'")
            kg.save()
            print(f"\nAccepted {len(pending)} proposal(s).")

    if args.reject:
        if kg.reject_proposal(args.reject):
            kg.save()
            print(f"Rejected proposal: '{args.reject}'")
        else:
            print(f"No pending proposal named '{args.reject}'.")

    if args.patterns:
        patterns = kg.analyze_relation_patterns(min_occurrences=1)
        if not patterns:
            print("No novel relation patterns found.")
        for rel, info in patterns.items():
            print(f"\n  {rel}: {info['count']} edges, avg confidence {info['avg_confidence']}")
            print(f"    Status: {info['proposal_status']} | Docs: {info['num_source_docs']}")
            if info.get("proposal_justification"):
                print(f"    Justification: {info['proposal_justification']}")
            for src, tgt in info["example_pairs"][:3]:
                print(f"      {src} → {tgt}")

    if args.pyvis:
        path = kg.export_pyvis(
            args.pyvis,
            center_node=args.center,
            depth=args.depth if args.center else None,
        )
        print(f"Pyvis visualization: {path}")

    if args.cytoscape:
        path = kg.export_cytoscape(
            args.cytoscape,
            center_node=args.center,
            depth=args.depth if args.center else None,
        )
        print(f"Cytoscape visualization: {path}")

    if args.preview_md:
        from pathlib import Path as _P
        md_path = _P(args.preview_md)
        if not md_path.exists():
            print(f"File not found: {md_path}")
        else:
            text = md_path.read_text(encoding="utf-8")
            sections = KnowledgeGraph.parse_markdown_sections(text)
            print(f"\nMarkdown: {md_path.name}")
            print(f"  Total characters: {len(text):,}")
            print(f"  Sections parsed: {len(sections)}")
            print()
            for i, sec in enumerate(sections):
                prefix = "  " * sec["level"] if sec["level"] > 0 else ""
                heading = sec["heading"] or "(preamble)"
                flags = []
                if sec["has_code"]:
                    flags.append("code")
                if sec["has_table"]:
                    flags.append("table")
                if sec["has_list"]:
                    flags.append("list")
                if sec["links"]:
                    flags.append(f"{len(sec['links'])} links")
                flag_str = f"  [{', '.join(flags)}]" if flags else ""
                print(f"  {prefix}{'#' * sec['level']} {heading}  "
                      f"({sec['char_count']:,} chars){flag_str}")
                if args.sections:
                    # Show first 200 chars of body
                    preview = sec["body"][:200].replace("\n", " ")
                    if len(sec["body"]) > 200:
                        preview += "..."
                    print(f"  {prefix}  → {preview}")
            print(f"\n  To ingest, run: --ingest-md {args.preview_md}")

    if args.ingest_md:
        from pathlib import Path as _P
        md_path = _P(args.ingest_md)
        if not md_path.exists():
            print(f"File not found: {md_path}")
        else:
            text = md_path.read_text(encoding="utf-8")
            doc_id = md_path.stem
            file_path = md_path.resolve()

            # Build the LLM extraction function
            if args.query_model:
                ollama_model = args.query_model
                ollama_url = args.ollama_url.rstrip("/")

                # Build verbosity flags early so ollama_extract can use them
                _quiet = args.quiet
                _verbose = args.verbose

                def ollama_extract(prompt: str) -> list[dict[str, Any]]:
                    """Call Ollama /api/chat and parse the JSON response."""
                    if _verbose:
                        logger.debug("Prompt length: %d chars", len(prompt))

                    payload = json.dumps({
                        "model": ollama_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a JSON extraction engine. "
                                    "Respond with ONLY a valid JSON array. "
                                    "No thinking, no explanations, no markdown."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0.1, "num_predict": 32768},
                    }).encode()
                    req = urllib.request.Request(
                        f"{ollama_url}/api/chat",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=1200) as resp:
                        body = json.loads(resp.read())

                    raw = body.get("message", {}).get("content", "").strip()
                    if _verbose:
                        logger.debug("Raw response length: %d chars", len(raw))
                    if not raw:
                        logger.warning("LLM returned empty response")
                        return []

                    # Try direct parse first (works when format:json is honored)
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        parsed = None

                    # If direct parse failed, try to extract JSON from the text
                    # (model may have produced thinking/prose around the JSON)
                    if parsed is None:
                        start = raw.find("[")
                        if start != -1:
                            end = raw.rfind("]")
                            if end > start:
                                try:
                                    parsed = json.loads(raw[start:end + 1])
                                except json.JSONDecodeError:
                                    pass

                    # Last resort: salvage truncated JSON array
                    if parsed is None:
                        salvaged = _salvage_truncated_json(raw)
                        if salvaged is not None:
                            logger.warning(
                                "JSON truncated, salvaged %d complete triple(s)",
                                len(salvaged),
                            )
                            return salvaged
                        logger.error(
                            "JSON parse failed (%d chars): %s",
                            len(raw), raw[:500],
                        )
                        return []

                    # Handle models that wrap the array in an object
                    if isinstance(parsed, dict):
                        for v in parsed.values():
                            if isinstance(v, list):
                                return v
                        return []
                    if isinstance(parsed, list):
                        return parsed
                    return []

                extract_fn: Callable[[str], list[dict[str, Any]]] = ollama_extract
                print(f"  Using Ollama model: {ollama_model} at {ollama_url}")
            else:
                # Structure-only ingestion: no LLM, build document/section
                # nodes and store the source. Entity extraction is skipped.
                _quiet = args.quiet
                _verbose = args.verbose
                extract_fn = lambda _text: []
            _ingest_t0 = time.monotonic()

            def _progress(event: dict[str, Any]) -> None:
                if _quiet:
                    return
                ev = event["event"]
                idx = event.get("index", 0)
                total = event.get("total", 0)
                heading = event.get("heading", "?")
                chars = event.get("char_count", 0)
                tag = f"[{idx + 1}/{total}]"

                if ev == "section_skip":
                    print(f"  {tag} Skip: \"{heading}\" ({chars:,} chars, {event.get('reason', 'skipped')})")
                elif ev == "section_start":
                    print(f"  {tag} Extracting: \"{heading}\" ({chars:,} chars)...", end="", flush=True)
                elif ev == "section_done":
                    elapsed = event.get("elapsed_seconds", 0)
                    triples = event.get("triples", 0)
                    nodes_added = event.get("nodes_added", 0)
                    errors = event.get("errors", [])
                    if errors and nodes_added == 0 and triples > 0:
                        print(f" {triples} triples → 0 nodes ({len(errors)} errors, {elapsed}s)")
                        if _verbose:
                            for err in errors[:5]:
                                print(f"         {err}")
                            if len(errors) > 5:
                                print(f"         ... and {len(errors) - 5} more")
                    elif errors:
                        print(f" {triples} triples → {nodes_added} nodes ({len(errors)} errors, {elapsed}s)")
                        if _verbose:
                            for err in errors[:5]:
                                print(f"         {err}")
                            if len(errors) > 5:
                                print(f"         ... and {len(errors) - 5} more")
                    else:
                        print(f" {triples} triples → {nodes_added} nodes ({elapsed}s)")

            stats = kg.ingest_markdown(
                text,
                doc_id=doc_id,
                llm_extract_fn=extract_fn,
                original_path=file_path,
                doc_properties={
                    "file_path": str(md_path),
                    "file_size": md_path.stat().st_size,
                },
                progress_fn=_progress,
            )
            _total_elapsed = time.monotonic() - _ingest_t0

            # Embed nodes using Ollama if available
            _embed_stats = None
            if args.query_model:
                # Resolve embed model: explicit flag > graph metadata > fallback
                if args.embed_model is not None:
                    _embed_model = args.embed_model
                elif kg.embed_model:
                    _embed_model = kg.embed_model
                    if not _quiet:
                        print(f"  Using embed model '{_embed_model}' from graph metadata")
                else:
                    _embed_model = "nomic-embed-text"
                _embed_url = args.embed_url.rstrip("/")

                def _embed_fn(batch: list[str]) -> list[list[float]]:
                    return ollama_embed(batch, model=_embed_model, url=_embed_url)

                if not _quiet:
                    print(f"  Embedding nodes with {_embed_model}...", end="", flush=True)
                _embed_t0 = time.monotonic()
                _embed_stats = kg.embed_nodes(_embed_fn, skip_existing=True, model_name=_embed_model)
                _embed_elapsed = time.monotonic() - _embed_t0
                if not _quiet:
                    print(f" {_embed_stats['nodes_embedded']} nodes ({_embed_elapsed:.1f}s)")

            kg.save_all()

            # Print summary
            graph_stats = kg.stats()
            print(f"\nIngested: {md_path.name}")
            print(f"  Document ID: {stats['doc_id']}")
            print(f"  Sections: {stats['total_sections']}")
            print(f"  Triples extracted: {stats['total_triples']}")
            print(f"  Nodes added: {stats['total_nodes_added']}, Edges added: {stats['total_edges_added']}")
            print(f"  Total time: {_total_elapsed:.1f}s")
            print(f"  Graph totals: {graph_stats['num_nodes']} nodes, {graph_stats['num_edges']} edges")
            if _embed_stats and _embed_stats["nodes_embedded"]:
                print(f"  Nodes embedded: {_embed_stats['nodes_embedded']} "
                      f"(skipped {_embed_stats['nodes_skipped']})")
            if stats.get("total_proposals_created"):
                print(f"  New relation proposals: {stats['total_proposals_created']}")
                if args.auto_accept:
                    pending = kg.get_proposals()
                    for p in pending:
                        kg.accept_proposal(p.name)
                    if pending:
                        kg.save()
                        print(f"  Auto-accepted {len(pending)} proposal(s)")
            if stats.get("source"):
                src = stats["source"]
                if src.get("is_duplicate"):
                    print(f"  Warning: duplicate content (matches '{src['existing_doc_id']}')")
                else:
                    print(f"  Source stored: {src['stored_path']}")
            print()

            # Show section breakdown (verbose only — real-time output already shown)
            if _verbose:
                print("  Section details:")
                for sec_stat in stats.get("sections", []):
                    heading = sec_stat.get("heading", "?")
                    elapsed = sec_stat.get("elapsed_seconds", "")
                    elapsed_str = f", {elapsed}s" if elapsed else ""
                    if sec_stat.get("skipped"):
                        print(f"    [skip] {heading} ({sec_stat.get('reason', '')})")
                    else:
                        triples_info = ""
                        n_triples = sec_stat.get("triples_processed", 0)
                        n_nodes = sec_stat.get("nodes_added", 0)
                        n_errors = len(sec_stat.get("errors", []))
                        if n_triples:
                            triples_info = f", {n_triples} triples"
                        if n_triples and n_nodes == 0 and n_errors:
                            tag = "WARN"
                            triples_info += f", 0 nodes, {n_errors} errors"
                        else:
                            tag = "ok"
                        print(f"    [{tag}]   {heading} ({sec_stat.get('char_count', 0):,} chars{triples_info}{elapsed_str})")

            if stats["errors"]:
                print(f"\n  Errors ({len(stats['errors'])}):")
                for err in stats["errors"]:
                    print(f"    - {err}")

            print(f"\n  Graph saved to {kg.graph_path}")

            # Auto-export visualizations
            if not args.no_viz:
                graph_dir = kg.graph_path.parent
                base_name = kg.graph_path.stem

                cyto_path = graph_dir / f"{base_name}_cytoscape.html"
                try:
                    kg.export_cytoscape(cyto_path)
                    print(f"  Cytoscape visualization: {cyto_path}")
                except Exception as e:
                    logger.error("Cytoscape export failed: %s", e)

                try:
                    pyvis_path = graph_dir / f"{base_name}_pyvis.html"
                    kg.export_pyvis(pyvis_path)
                    print(f"  Pyvis visualization: {pyvis_path}")
                except Exception as e:
                    logger.error("Pyvis export skipped: %s", e)

    if args.sources:
        sources = kg.list_sources()
        if not sources:
            print("No stored sources.")
        else:
            ss = kg.source_stats()
            print(f"\nStored sources: {ss['total_sources']} files, "
                  f"{ss['total_chars']:,} total chars")
            print(f"Directory: {ss['sources_dir']}")
            print()
            for s in sources:
                status = "✓" if s["file_exists"] else "✗ MISSING"
                print(f"  [{status}] {s['doc_id']}")
                print(f"      hash: {s['content_hash']}  |  "
                      f"{s['char_count']:,} chars  |  {s['stored_at'][:10]}")
                if s.get("original_path"):
                    print(f"      original: {s['original_path']}")

    if args.check_sources:
        issues = kg.check_source_integrity()
        if not issues:
            ss = kg.source_stats()
            print(f"All {ss['total_sources']} sources OK.")
        else:
            print(f"Found {len(issues)} issue(s):")
            for issue in issues:
                print(f"  {issue['doc_id']}: {issue['issue']}")
                for k, v in issue.items():
                    if k not in ('doc_id', 'issue'):
                        print(f"    {k}: {v}")

    if not any([args.stats, args.node, args.neighbors, args.split,
                args.proposals, args.accept, args.accept_all, args.reject,
                args.patterns, args.pyvis, args.cytoscape,
                args.preview_md, args.ingest_md,
                args.sources, args.check_sources]):
        print(kg)
        print(f"\nUse --stats, --node, --neighbors, --split, --proposals, "
              f"--accept, --accept-all, --reject, --patterns, --pyvis, --cytoscape, "
              f"--preview-md, --ingest-md, --sources, --check-sources for details.")


if __name__ == "__main__":
    main()
