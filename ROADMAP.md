# ZipLattice Feature Roadmap

Implementation plan for 9 planned features (items 1-5, 7-10).

Each item follows the pattern: **core method -> CLI flag -> web endpoint -> template -> MCP tool -> tests**.

---

## Recommended Implementation Order

1. **Item 1** (Validation) -- foundational, no deps, improves all later work
2. **Item 4** (Analytics) -- builds on stats infrastructure, helps measure quality
3. **Item 7** (Document versioning) -- extends existing manifest, needed by items 3 and 10
4. **Item 10** (Incremental ingestion) -- depends on section hashing from item 7
5. **Item 3** (Graph diffing) -- benefits from versioning infrastructure
6. **Item 2** (Hybrid search) -- independent, benefits from analytics to measure improvement
7. **Item 9** (Bulk proposals) -- independent, extends existing proposal system
8. **Item 5** (Query language) -- most complex, benefits from all prior infrastructure
9. **Item 8** (Inline editing) -- most UI-heavy, best done last when API surface is stable

---

## Item 1: Graph Validation & Consistency

**Goal:** Detect logical inconsistencies (cycles in taxonomic relations, contradictory edges, orphan nodes, confidence anomalies).

**Where:** New `validate()` method on `KnowledgeGraph` in `knowledge_graph.py`.

### Checklist

- [ ] **1a.** Add `validate()` method returning a `ValidationReport` dataclass with `errors`, `warnings`, `info` lists
- [ ] **1b.** Cycle detection for taxonomic relations (`is_a`, `subclass_of`, `instance_of`, `part_of`) using `nx.simple_cycles()` on relation-filtered subgraph
- [ ] **1c.** Contradictory edge detection -- find pairs like `A is_a B` + `B is_a A`, or `A supersedes B` + `B supersedes A`
- [ ] **1d.** Orphan node detection -- nodes with degree 0 in `_G` (excluding document nodes)
- [ ] **1e.** Dangling edge detection -- edges referencing node IDs not in `_data["nodes"]`
- [ ] **1f.** Confidence anomaly detection -- nodes/edges with confidence=0 or extreme outliers
- [ ] **1g.** Missing embedding detection -- non-document/section nodes without embeddings
- [ ] **1h.** Add `--validate` CLI flag to `main()`
- [ ] **1i.** Add `GET /graphs/{name}/validate` endpoint in `web_app.py` returning JSON report
- [ ] **1j.** Add validation summary card to `graph_detail.html` (error/warning counts, expandable details)
- [ ] **1k.** Add MCP tool `validate_graph` in `mcp_server.py`
- [ ] **1l.** Tests: create graphs with known cycles, contradictions, orphans; verify detection

### Design Decisions

- `validate()` is read-only, never mutates the graph
- Returns structured report, not exceptions -- users decide what to fix
- Taxonomic cycle check uses `nx.DiGraph` subgraph filtered to only taxonomic edges
- Contradictory pairs defined as a config dict of `{(rel_a, rel_b)}` mutually exclusive pairs

---

## Item 2: Hybrid Search (BM25 + Semantic)

**Goal:** Combine BM25 full-text scoring with existing vector similarity for better retrieval.

**Where:** New methods in `KnowledgeGraph`, updated `search()`, updated `query_graph.py`.

### Checklist

- [x] **2a.** Implement `_build_bm25_index()` -- in-memory inverted index from node labels, properties, body text. Standard BM25 formula (k1=1.2, b=0.75). Lazily built, invalidated when `_dirty=True`
- [x] **2b.** Implement `bm25_search(query, top_k) -> list[tuple[str, float]]` -- tokenize query, score indexed nodes, return ranked results
- [x] **2c.** Add `hybrid_search()` combining BM25 + cosine similarity with configurable alpha weight (0=pure BM25, 1=pure semantic, default 0.7)
- [x] **2d.** Update `search()` to accept `mode: Literal["semantic", "bm25", "hybrid"]` parameter
- [x] **2e.** Update `search_nodes()` and `build_context()` in `query_graph.py` to pass through search mode
- [x] **2f.** Add `--search-mode` CLI flag to `query_graph.py`
- [x] **2g.** Add search mode dropdown to `query.html` (semantic / keyword / hybrid)
- [x] **2h.** Add MCP tool parameter for search mode in `mcp_server.py`
- [x] **2i.** Tests: verify BM25 finds exact keyword matches that semantic search misses; verify hybrid ranking

### Design Decisions

- Pure Python BM25 -- no new dependencies
- Tokenization: lowercase, split on non-alphanumeric, remove stopwords (small built-in set)
- Index stored in memory only (rebuilt on load), not persisted
- Score normalization: both BM25 and cosine normalized to [0, 1] before combining

---

## Item 3: Graph Diffing & Change Tracking

**Goal:** Show what changed between two graph states or after an ingestion run.

**Where:** New `diff()` static method on `KnowledgeGraph`, new CLI command, web UI additions.

### Checklist

- [x] **3a.** Add `GraphDiff` dataclass: `nodes_added`, `nodes_removed`, `nodes_modified` (with field-level diffs), `edges_added`, `edges_removed`, `edges_modified`, `proposals_added`, `proposals_changed`
- [x] **3b.** Implement `KnowledgeGraph.diff(other) -> GraphDiff` -- compares two graph instances
- [x] **3c.** Implement `KnowledgeGraph.diff_from_file(path) -> GraphDiff` -- loads snapshot and diffs against current state
- [x] **3d.** Add `snapshot()` method -- deep copy of current `_data` for before/after comparison within a session
- [x] **3e.** After `ingest_markdown()` completes, compute diff from pre-ingestion snapshot and attach summary to aggregate stats
- [x] **3f.** Add `--diff <other_graph.json>` CLI flag
- [x] **3g.** Add `GET /graphs/{name}/diff?against={other_name}` endpoint in `web_app.py`
- [x] **3h.** Add `templates/partials/diff_result.html` -- styled add/remove/modify with green/red/yellow highlighting
- [x] **3i.** Show "changes since last ingestion" summary on `graph_detail.html` sources table
- [x] **3j.** Add MCP tool `diff_graphs` in `mcp_server.py`
- [x] **3k.** Tests: create two graph states, verify diff correctly identifies additions, removals, modifications

### Design Decisions

- Diff operates on `_data` dict level (nodes by ID, edges by `(source, target, relation)` tuple)
- Node modifications tracked at field level (e.g., "confidence changed from 0.5 to 0.8")
- Edge identity matches existing `_edge_index`
- No automatic snapshots on disk -- snapshots are ephemeral or explicit

---

## Item 4: Analytics & Quality Reporting

**Goal:** Surface extraction quality metrics, confidence distributions, and structural health as a dashboard.

**Where:** New `analytics()` method on `KnowledgeGraph`, new web dashboard page.

### Checklist

- [ ] **4a.** Add `analytics()` method returning:
  - Confidence distribution (histogram buckets) for nodes and edges separately
  - Relation type frequency with confidence stats per relation
  - Orphan node count and list
  - Hub nodes (top-10 by degree)
  - Source document coverage (nodes per doc, edges per doc)
  - Embedding coverage percentage
  - Component size distribution
  - Average path length (sampled for large graphs)
- [ ] **4b.** Add `--analytics` CLI flag with formatted terminal output
- [ ] **4c.** Add `GET /graphs/{name}/analytics` endpoint returning JSON
- [ ] **4d.** Create `templates/analytics.html` page:
  - Confidence distribution charts (CSS bar charts or Chart.js from CDN)
  - Relation type frequency table
  - Quality score summary (composite: embedding coverage * avg confidence * (1 - orphan_ratio))
  - Hub nodes table with degree counts
  - Source document health table
- [ ] **4e.** Add "Analytics" link to `graph_detail.html` stats section
- [ ] **4f.** Add MCP tool `graph_analytics` in `mcp_server.py`
- [ ] **4g.** Tests: verify analytics on populated graph returns expected structure and reasonable values

### Design Decisions

- No new dependencies -- CSS bar charts or Chart.js from CDN
- Confidence histogram: 10 buckets (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
- Quality score is advisory, displayed as "graph health" percentage
- Sampling for expensive metrics (avg path length) on graphs > 1000 nodes

---

## Item 5: Structured Graph Query Language

**Goal:** Lightweight pattern-matching query language for multi-hop graph traversal.

**Where:** New `graph_query()` method, new parser, CLI and web integration.

### Checklist

- [ ] **5a.** Design query syntax:
  ```
  (type:technology) -[depends_on]-> (*) -[is_a]-> (label:"database")
  (label:~"SAR*") -[*]-> (*) WHERE confidence > 0.7
  (*) -[is_a]-> (id:"python") DEPTH 2
  ```
- [ ] **5b.** Implement `_parse_graph_query(query_str) -> list[QueryStep]` -- tokenizer + parser. `QueryStep` dataclass with `node_filter`, `edge_filter`, `direction`
- [ ] **5c.** Implement `graph_query(query_str, limit=50) -> list[list[dict]]` -- execute parsed query against `_G`, return matching paths
- [ ] **5d.** Node filters: `type:X`, `label:X`, `label:~"pattern"` (glob), `id:X`, `*` (any)
- [ ] **5e.** Edge filters: `[relation_name]`, `[*]` (any), `[rel1|rel2]` (alternatives)
- [ ] **5f.** `WHERE` clause for confidence filtering on final result set
- [ ] **5g.** `DEPTH N` modifier for variable-length paths
- [ ] **5h.** Add `--query "pattern"` CLI flag
- [ ] **5i.** Add "Pattern Query" tab/mode to `query.html` with syntax help tooltip
- [ ] **5j.** Add `POST /pattern-query` endpoint in `web_app.py`
- [ ] **5k.** Add MCP tool `pattern_query` in `mcp_server.py`
- [ ] **5l.** Tests: multi-hop queries, wildcard edges, type filters, WHERE clauses, DEPTH modifier

### Design Decisions

- Pure Python parser -- syntax is simple enough, no grammar library needed
- Arrow direction: `->` follows edge, `<-` reverses, `--` ignores direction
- Results capped at `limit` to prevent runaway queries
- Uses `nx.all_simple_paths()` for multi-hop, filtered by pattern constraints
- Error messages include parse position for malformed queries

---

## Item 7: Document Versioning

**Goal:** Track document versions and show what changed when a document is re-ingested.

**Where:** Extends existing `store_source()` and `ingest_markdown()`, new web UI views.

### Checklist

- [x] **7a.** Extend source manifest with `section_hashes: dict[str, str]` -- maps section heading to SHA-256 of body, computed during ingestion
- [x] **7b.** Add `diff_document_versions(doc_id, v1, v2) -> DocumentDiff` -- compares archived versions' section hashes to identify added/removed/modified sections
- [x] **7c.** Add `get_document_history(doc_id) -> list[dict]` -- version timeline with dates, hashes, section counts, node/edge counts per version
- [x] **7d.** During re-ingestion, compute and log section-level diffs as part of progress events
- [x] **7e.** Add `--doc-history <doc_id>` CLI flag
- [x] **7f.** Add `GET /graphs/{name}/documents/{doc_id}/history` endpoint
- [x] **7g.** Add version history view to `graph_detail.html` sources table -- expandable timeline
- [x] **7h.** Add MCP tool `document_history` in `mcp_server.py`
- [x] **7i.** Tests: ingest doc v1, modify, re-ingest as v2, verify version history and section diffs

### Design Decisions

- Builds on existing `versions` list in source manifest
- Section hashes enable comparison without storing full text of every version
- Archived versions already saved to `sources/archive/` -- this surfaces that data
- Diff is between section sets, not line-by-line

---

## Item 8: Web UI Inline Graph Editing

**Goal:** Edit nodes and edges directly from the Cytoscape visualization.

**Where:** New API endpoints in `web_app.py`, updated `graph_detail.html` JavaScript.

### Checklist

- [ ] **8a.** Add REST API endpoints:
  - `PUT /api/graphs/{name}/nodes/{node_id}` -- update label, description, confidence, type, properties
  - `DELETE /api/graphs/{name}/nodes/{node_id}` -- remove node and edges
  - `PUT /api/graphs/{name}/edges` -- update edge (identified by source+target+relation)
  - `DELETE /api/graphs/{name}/edges` -- remove edge
  - `POST /api/graphs/{name}/nodes` -- add new node
  - `POST /api/graphs/{name}/edges` -- add new edge
- [ ] **8b.** Extend `#detail-panel` with edit mode: "Edit" toggles fields to inputs, Save/Cancel buttons, Delete with confirmation
- [ ] **8c.** Add "Add Node" floating button with modal form (id, label, type dropdown, confidence slider)
- [ ] **8d.** Add "Add Edge" mode: click source, click target, select relation, confirm
- [ ] **8e.** Cytoscape visual updates after save via `cy.getElementById(id).data({...})` without full reload
- [ ] **8f.** Proposal management in detail panel -- accept/reject buttons for proposed relations
- [ ] **8g.** Optimistic UI: update Cytoscape immediately, revert on API error
- [ ] **8h.** Tests: all CRUD endpoints, verify graph state after mutations

### Design Decisions

- REST JSON API for mutations (not HTMX partials) -- cleaner for Cytoscape JS integration
- All mutations go through existing `KnowledgeGraph` methods -- no direct dict manipulation
- Auto-save after each edit (calls `kg.save()`)
- Edge identity: `{source, target, relation}` tuple matching `_edge_index`

---

## Item 9: Bulk Proposal Management

**Goal:** Accept/reject proposals in batches, auto-accept by threshold, merge similar proposals.

**Where:** Extends proposal methods in `KnowledgeGraph`, new web UI page.

### Checklist

- [ ] **9a.** Add `bulk_accept_proposals(names, review_note="") -> list[str]`
- [ ] **9b.** Add `bulk_reject_proposals(names, review_note="") -> list[str]`
- [ ] **9c.** Extend `accept_all_proposals()` with `max_accept: int` safety parameter
- [ ] **9d.** Add `merge_proposals(names, target_name) -> RelationProposal` -- combine examples/docs, reject others
- [ ] **9e.** Add `find_similar_proposals(threshold=0.8) -> list[list[RelationProposal]]` -- group by edit distance
- [ ] **9f.** Add `GET /graphs/{name}/proposals` returning all proposals as JSON
- [ ] **9g.** Add `POST /graphs/{name}/proposals/bulk` accepting `{action, names, review_note}`
- [ ] **9h.** Add `POST /graphs/{name}/proposals/auto-accept` accepting `{min_confidence, min_examples}`
- [ ] **9i.** Create `templates/proposals.html`:
  - Table with status badges, confidence, example count, source docs
  - Checkboxes + bulk action bar (Accept/Reject Selected)
  - Auto-accept form (confidence slider, min examples)
  - Similar proposal groups with "Merge" button
- [ ] **9j.** Add "Proposals" link to navigation in `base.html`
- [ ] **9k.** Add MCP tools: `bulk_manage_proposals`, `auto_accept_proposals`
- [ ] **9l.** Tests: bulk accept/reject, merge, auto-accept threshold, similar detection

### Design Decisions

- `bulk_*` methods call individual accept/reject in a loop but only rebuild NetworkX once at the end
- Similar proposal detection uses normalized edit distance (pure Python)
- Merge combines all examples, takes highest confidence
- Web UI uses standard form submission with HTMX

---

## Item 10: Incremental Ingestion

**Goal:** Skip re-extraction for unchanged sections when re-ingesting a document.

**Where:** Modifies `ingest_markdown()` in `KnowledgeGraph`.

### Checklist

- [ ] **10a.** Store per-section content hashes in source manifest: `section_hashes: {section_slug: sha256}` (shared with item 7)
- [ ] **10b.** Add `incremental: bool = False` parameter to `ingest_markdown()` -- compare section hashes against previous ingestion
- [ ] **10c.** Unchanged sections (hash match): skip LLM extraction, retain existing nodes/edges, emit `section_skip` event
- [ ] **10d.** Changed sections: run LLM extraction normally, update hash
- [ ] **10e.** New sections: run LLM extraction normally
- [ ] **10f.** Removed sections: mark associated nodes/edges as stale (`stale: true` property) rather than deleting
- [ ] **10g.** Add `--incremental` / `-i` CLI flag
- [ ] **10h.** Add "Incremental" checkbox to ingest settings in `upload_result.html`
- [ ] **10i.** Report incremental stats: sections skipped vs. re-extracted
- [ ] **10j.** Add `incremental` parameter to `ingest_triples` MCP tool context
- [ ] **10k.** Tests: ingest doc, re-ingest unchanged (verify 0 LLM calls), modify one section, re-ingest (verify only changed section triggers extraction)

### Design Decisions

- Content hash is per-section, not per-document -- enables surgical re-extraction
- Hash includes heading + body (heading change = re-extract even if body unchanged)
- Structural nodes/edges always refreshed (cheap, no LLM)
- "Stale" marking is soft-delete -- avoids breaking references
- Default `incremental=False` to match current behavior -- opt-in
