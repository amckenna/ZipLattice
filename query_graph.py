"""
query_graph.py — Knowledge Graph Query Application

A CLI tool for querying a ZipLattice knowledge graph. Supports semantic
search, context building for RAG, and full question-answering via Ollama.

Usage:
    python query_graph.py <graph.json> search "synthetic aperture radar"
    python query_graph.py <graph.json> context "how does SAR work?"
    python query_graph.py <graph.json> ask "how does SAR work?"
    python query_graph.py <graph.json> node <node-id>
    python query_graph.py <graph.json> neighbors <node-id> --depth 2
    python query_graph.py <graph.json> path <source-id> <target-id>
    python query_graph.py <graph.json> stats
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from knowledge_graph import (
    KnowledgeGraph, GraphEncoder, ollama_embed, _strip_thinking,
    claude_chat, _get_anthropic_api_key,
    bedrock_chat, bedrock_embed,
    read_http_error_detail,
)

logger = logging.getLogger("query_graph")


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------


def search_nodes(
    kg: KnowledgeGraph,
    query: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    *,
    top_k: int = 10,
    node_types: list[str] | None = None,
    expand_depth: int = 1,
    search_mode: str = "semantic",
    alpha: float = 0.7,
) -> list[dict[str, Any]]:
    """Search over the knowledge graph.

    Args:
        kg: A loaded KnowledgeGraph with pre-computed node embeddings.
        query: Natural-language query string.
        embed_fn: Embedding function for the query text (not required
            for ``"bm25"`` mode).
        top_k: Number of results to return.
        node_types: Optional filter by node type.
        expand_depth: Neighborhood expansion depth for each result.
        search_mode: ``"semantic"``, ``"bm25"``, or ``"hybrid"``.
        alpha: Hybrid blending weight (0 = pure BM25, 1 = pure semantic).

    Returns:
        List of result dicts from ``kg.search()``.
    """
    logger.debug(
        "search_nodes: mode=%s %d embeddings loaded, %d nodes in graph",
        search_mode, len(kg._embeddings), len(kg._data["nodes"]),
    )
    results = kg.search(
        query,
        embed_fn,
        top_k=top_k,
        node_types=node_types,
        expand_depth=expand_depth,
        mode=search_mode,
        alpha=alpha,
    )
    logger.debug("search_nodes: %d results returned", len(results))
    for r in results[:5]:
        logger.debug(
            "  %.4f  %s  (%s)", r["similarity"], r["label"], r["node_id"],
        )
    return results


def build_context(
    kg: KnowledgeGraph,
    query: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    *,
    max_nodes: int = 30,
    depth: int = 1,
    max_body_chars: int = 500,
    search_mode: str = "semantic",
    alpha: float = 0.7,
) -> str:
    """Search the graph and format a context block for an LLM prompt.

    Args:
        kg: A loaded KnowledgeGraph with pre-computed node embeddings.
        query: Natural-language query string.
        embed_fn: Embedding function for the query text.
        max_nodes: Maximum number of nodes in the context window.
        depth: Neighborhood expansion depth.
        max_body_chars: Truncate section body text to this many characters.
        search_mode: ``"semantic"``, ``"bm25"``, or ``"hybrid"``.
        alpha: Hybrid blending weight.

    Returns:
        A formatted text block suitable for injecting into an LLM prompt.
    """
    results = search_nodes(
        kg, query, embed_fn, top_k=max_nodes, expand_depth=0,
        search_mode=search_mode, alpha=alpha,
    )
    if not results:
        logger.debug("build_context: no search results for query")
        return "(No relevant nodes found in the knowledge graph.)"

    seed_ids = [r["node_id"] for r in results]
    context = kg.get_context_window(
        seed_ids, depth=depth, max_nodes=max_nodes, boundary_edges=True,
    )
    logger.debug(
        "build_context: %d seed nodes expanded to %d nodes, %d edges (depth=%d)",
        len(seed_ids), len(context["nodes"]), len(context["edges"]), depth,
    )

    node_set = set(context["nodes"])

    # Structural relations are useful between nodes that are *both* in the
    # context window (navigation) but pure noise as boundary edges.
    _STRUCTURAL_RELATIONS = {"part_of", "contains", "documented_by"}

    # Helper to resolve a human-readable label for any node ID
    def _label(nid: str) -> str:
        # Prefer label from context nodes (already fetched)
        if nid in context["nodes"]:
            return context["nodes"][nid].get("label", nid)
        # Fall back to the full graph for external nodes
        full = kg._data["nodes"].get(nid, {})
        return full.get("label", nid)

    lines: list[str] = []
    lines.append(f"## Knowledge Graph Context ({len(context['nodes'])} nodes, "
                 f"{len(context['edges'])} edges)")
    lines.append("")

    # Build a map of edge context sentences per node so we can synthesize
    # a description for nodes that don't have one stored.
    _edge_contexts: dict[str, list[str]] = {}
    for edge in context["edges"]:
        ctx = edge.get("properties", {}).get("context", "")
        if ctx:
            for endpoint in (edge.get("source"), edge.get("target")):
                if endpoint and endpoint in node_set:
                    _edge_contexts.setdefault(endpoint, []).append(ctx)

    # Nodes — include descriptions and section body text
    for nid, ndata in context["nodes"].items():
        label = ndata.get("label", nid)
        ntype = ndata.get("type", "concept")
        conf = ndata.get("confidence", 1.0)
        props = ndata.get("properties", {})
        desc = props.get("description", "")
        # Fall back to the first edge context sentence if no description
        if not desc and nid in _edge_contexts:
            desc = _edge_contexts[nid][0]
        line = f"- [{ntype}] {label} (id={nid}, confidence={conf:.2f})"
        if desc:
            line += f"\n    {desc}"
        # Include section body text (truncated) so the LLM gets real content
        if ntype == "section":
            body = props.get("body_text", "")
            if body:
                if len(body) > max_body_chars:
                    body = body[:max_body_chars] + "..."
                line += f"\n    Content: {body}"
        lines.append(line)

    # Edges — include context sentences and label boundary nodes
    edge_lines: list[str] = []
    for edge in context["edges"]:
        src = edge.get("source", "?")
        tgt = edge.get("target", "?")
        rel = edge.get("relation", "related_to")
        is_boundary = src not in node_set or tgt not in node_set

        # Drop structural scaffolding from boundary edges — they add
        # noise without helping the LLM reason about the domain.
        if is_boundary and rel in _STRUCTURAL_RELATIONS:
            continue

        # Use human-readable labels instead of raw slugified IDs
        src_display = _label(src)
        tgt_display = _label(tgt)
        if src not in node_set:
            src_display += " (external)"
        if tgt not in node_set:
            tgt_display += " (external)"

        edge_line = f"- {src_display} --[{rel}]--> {tgt_display}"
        # Append the source sentence so the LLM gets real evidence
        ctx = edge.get("properties", {}).get("context", "")
        if ctx:
            edge_line += f"\n    \"{ctx}\""
        edge_lines.append(edge_line)

    if edge_lines:
        lines.append("")
        lines.append("### Relationships")
        lines.extend(edge_lines)

    return "\n".join(lines)


def ask(
    kg: KnowledgeGraph,
    question: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    llm_fn: Callable[[str], str],
    *,
    max_nodes: int = 30,
    search_mode: str = "semantic",
    alpha: float = 0.7,
) -> str:
    """Full RAG pipeline: search the graph, build context, call an LLM.

    Args:
        kg: A loaded KnowledgeGraph with pre-computed node embeddings.
        question: The user's question.
        embed_fn: Embedding function for the query text.
        llm_fn: A callable that takes a prompt string and returns the
                 LLM's response string.
        max_nodes: Maximum number of nodes in the context window.
        search_mode: ``"semantic"``, ``"bm25"``, or ``"hybrid"``.
        alpha: Hybrid blending weight.

    Returns:
        The LLM's answer string.
    """
    t0 = time.monotonic()
    context = build_context(
        kg, question, embed_fn, max_nodes=max_nodes,
        search_mode=search_mode, alpha=alpha,
    )
    ctx_elapsed = time.monotonic() - t0
    logger.debug("Context length: %d chars (%.2fs)", len(context), ctx_elapsed)

    prompt = (
        f"You are a helpful assistant. Use the following knowledge graph "
        f"context to answer the question. If the context doesn't contain "
        f"enough information, say so.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    logger.debug("Full prompt (%d chars):\n%s", len(prompt), prompt)

    t1 = time.monotonic()
    answer = llm_fn(prompt)
    llm_elapsed = time.monotonic() - t1
    logger.debug("LLM answer: %d chars (%.1fs)", len(answer), llm_elapsed)
    return answer


# ---------------------------------------------------------------------------
# Chat helper (OpenAI-compatible)
# ---------------------------------------------------------------------------


def ollama_chat(prompt: str, *, model: str, url: str) -> str:
    """Call an OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Works with Ollama (>=0.1.14), llama.cpp, vLLM, LocalAI, and any
    other server that implements the OpenAI chat completions API.
    """
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    logger.debug("ollama_chat: POST %s  model=%s  prompt=%d chars", endpoint, model, len(prompt))
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = read_http_error_detail(exc)
        msg = (
            f"Chat request failed (HTTP {exc.code}): "
            f"POST {endpoint} with model '{model}'."
        )
        if detail:
            msg += f"\nServer response: {detail}"
        if exc.code == 400:
            msg += (
                f"\nHint: the model name '{model}' may not match what the "
                f"server expects. Run: python query_graph.py <graph> "
                f"list-models --api-url {url}"
            )
        else:
            msg += (
                f"\nCheck that the server is running at {url} "
                f"and the model '{model}' is available."
            )
        raise RuntimeError(msg) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot connect to {endpoint}: {exc.reason}. "
            f"Is the server running?"
        ) from exc
    elapsed = time.monotonic() - t0
    # OpenAI format: {"choices": [{"message": {"content": "..."}}]}
    answer = _strip_thinking(body["choices"][0]["message"]["content"].strip())
    logger.debug("ollama_chat: response=%d chars (%.1fs)", len(answer), elapsed)
    return answer


# ---------------------------------------------------------------------------
# Server introspection helpers
# ---------------------------------------------------------------------------


def list_models(url: str) -> list[dict[str, Any]]:
    """Query ``/v1/models`` and return the list of available models.

    Works with Ollama, llama.cpp (router mode), vLLM, LocalAI, exo, etc.
    Returns a list of model dicts (each has at least an ``id`` key).
    Falls back to ``/models`` if ``/v1/models`` returns no data.
    """
    base = url.rstrip("/")
    for path in ("/v1/models", "/models"):
        endpoint = f"{base}{path}"
        req = urllib.request.Request(endpoint, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            logger.debug("list_models: %s failed: %s", endpoint, exc)
            continue
        logger.debug("list_models: GET %s returned: %s", endpoint, json.dumps(body, indent=2)[:2000])
        # OpenAI format: {"data": [{"id": ..., ...}, ...]}
        if isinstance(body, dict) and body.get("data"):
            return body["data"]
        # Some servers return a bare list
        if isinstance(body, list) and body:
            return body
        # Try "models" key (some servers use this)
        if isinstance(body, dict) and body.get("models"):
            return body["models"]

    # If we get here, all attempts returned empty or failed
    raise RuntimeError(
        f"Could not list models from {base}. "
        f"Tried /v1/models and /models. "
        f"Is the server running and does it have models loaded?"
    )


def _list_models(url: str) -> None:
    """Print available models from the server."""
    try:
        models = list_models(url)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return
    print(f"Models available at {url}:")
    for m in models:
        if isinstance(m, dict):
            model_id = m.get("id", m.get("name", "?"))
            owned_by = m.get("owned_by", "")
            extra = f"  (owned_by: {owned_by})" if owned_by else ""
            print(f"  {model_id}{extra}")
        else:
            # Bare string model name
            print(f"  {m}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    # Shared flags as a parent parser so they work with subcommands
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--query-model", "--ollama", nargs="?", const="qwen3-coder:30b",
                        metavar="MODEL", dest="query_model",
                        help="Model for LLM query/chat (default: qwen3-coder:30b)")
    shared.add_argument("--api-url", "--ollama-url", default="http://localhost:11434",
                        dest="ollama_url",
                        help="OpenAI-compatible API server URL (works with Ollama, llama.cpp, vLLM, etc.) "
                             "(default: http://localhost:11434)")
    shared.add_argument("--embed-url", default=None, metavar="URL",
                        help="API server URL for embeddings (default: same as --api-url)")
    shared.add_argument("--embed-model", default=None, metavar="MODEL",
                        help="Embedding model for query embedding "
                             "(default: auto-detect from graph, or qwen3-embedding)")
    shared.add_argument("--provider", choices=["local", "anthropic", "bedrock"], default="local",
                        help="LLM provider: 'local' for OpenAI-compatible servers, "
                             "'anthropic' for the Claude API, 'bedrock' for AWS Bedrock (default: local)")
    shared.add_argument("--bedrock-region", default=None, metavar="REGION",
                        help="AWS region for Bedrock (default: AWS_DEFAULT_REGION or us-east-1)")
    shared.add_argument("--bedrock-profile", default=None, metavar="PROFILE",
                        help="AWS profile name from ~/.aws/credentials for Bedrock")
    shared.add_argument("--search-mode", choices=["semantic", "bm25", "hybrid"],
                        default="semantic", dest="search_mode",
                        help="Search mode: 'semantic' (embedding similarity), 'bm25' "
                             "(keyword), or 'hybrid' (blended) (default: semantic)")
    shared.add_argument("--alpha", type=float, default=0.7,
                        help="Hybrid search blending weight: 0=pure BM25, 1=pure semantic "
                             "(default: 0.7, only used with --search-mode hybrid)")
    shared.add_argument("--top-k", type=int, default=10, help="Number of search results")
    shared.add_argument("--depth", type=int, default=1, help="Neighborhood expansion depth")
    shared.add_argument("--node-types", nargs="+", metavar="TYPE",
                        help="Filter by node type(s)")
    shared.add_argument("--json", action="store_true", dest="json_output",
                        help="Output as JSON instead of formatted text")
    shared.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed debug info")
    shared.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress info messages; warnings only")

    parser = argparse.ArgumentParser(
        description="Query a ZipLattice knowledge graph",
        parents=[shared],
    )
    parser.add_argument("graph", metavar="GRAPH_JSON",
                        help="Path to knowledge graph JSON file")

    sub = parser.add_subparsers(dest="command")

    # search
    p_search = sub.add_parser("search", help="Semantic search over graph nodes")
    p_search.add_argument("query", help="Search query text")

    # context
    p_ctx = sub.add_parser("context", help="Build RAG context block from graph")
    p_ctx.add_argument("query", help="Query text")

    # ask
    p_ask = sub.add_parser("ask", help="Full RAG: search + context + LLM answer")
    p_ask.add_argument("question", help="Question to answer")

    # node
    p_node = sub.add_parser("node", help="Look up a single node by ID")
    p_node.add_argument("node_id", help="Node ID")

    # neighbors
    p_nbr = sub.add_parser("neighbors", help="Get neighbors of a node")
    p_nbr.add_argument("node_id", help="Node ID")

    # path
    p_path = sub.add_parser("path", help="Shortest path between two nodes")
    p_path.add_argument("source", help="Source node ID")
    p_path.add_argument("target", help="Target node ID")

    # stats
    sub.add_parser("stats", help="Print graph statistics")

    # list-models
    sub.add_parser("list-models", help="List models available on the API server")

    args = parser.parse_args()

    # Default --embed-url to --ollama-url when not explicitly set
    if args.embed_url is None:
        args.embed_url = args.ollama_url

    # Logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    elif args.quiet:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.command:
        parser.print_help()
        return

    # Commands that don't need the graph loaded
    if args.command == "list-models":
        _list_models(args.ollama_url)
        return

    kg = KnowledgeGraph(args.graph)

    # Build embed function for query-time use
    embed_url = args.embed_url.rstrip("/")

    # Resolve embed model: prefer explicit CLI flag, then graph metadata, then fallback
    if args.embed_model is not None:
        embed_model = args.embed_model
        # Warn if explicitly set model doesn't match what was used to build the graph
        if kg.embed_model and kg.embed_model != embed_model:
            logger.warning(
                "Embedding model mismatch: graph was embedded with '%s' "
                "but query is using '%s'. Results will be unreliable. "
                "Use --embed-model %s to match.",
                kg.embed_model, embed_model, kg.embed_model,
            )
    elif kg.embed_model:
        embed_model = kg.embed_model
        logger.info("Using embed model '%s' from graph metadata", embed_model)
    else:
        embed_model = "qwen3-embedding"
        logger.info("No embed model in graph metadata, falling back to '%s'", embed_model)

    if args.provider == "bedrock" and args.embed_model is not None:
        _br_region = args.bedrock_region
        _br_profile = args.bedrock_profile
        logger.info("Embed config: model='%s' (bedrock, region=%s, profile=%s)", embed_model, _br_region or "default", _br_profile or "default")

        def embed_fn(texts: list[str]) -> list[list[float]]:
            return bedrock_embed(texts, model=embed_model, region=_br_region, profile=_br_profile)
    else:
        logger.info("Embed config: model='%s' url='%s'", embed_model, embed_url)

        def embed_fn(texts: list[str]) -> list[list[float]]:
            return ollama_embed(texts, model=embed_model, url=embed_url)

    # Dispatch
    if args.command == "search":
        results = search_nodes(
            kg, args.query, embed_fn,
            top_k=args.top_k,
            node_types=args.node_types,
            expand_depth=args.depth,
            search_mode=args.search_mode,
            alpha=args.alpha,
        )
        if args.json_output:
            print(json.dumps(results, indent=2, cls=GraphEncoder))
        else:
            if not results:
                print("No results found.")
            for i, r in enumerate(results, 1):
                print(f"{i}. [{r['type']}] {r['label']}  "
                      f"(similarity={r['similarity']}, confidence={r['confidence']:.2f})")
                desc = r.get("properties", {}).get("description", "")
                if desc:
                    print(f"   {desc[:120]}")
                if r.get("neighbors"):
                    nbr_labels = [n["label"] for n in r["neighbors"][:5]]
                    print(f"   neighbors: {', '.join(nbr_labels)}")

    elif args.command == "context":
        ctx = build_context(
            kg, args.query, embed_fn,
            max_nodes=args.top_k,
            depth=args.depth,
            search_mode=args.search_mode,
            alpha=args.alpha,
        )
        if args.json_output:
            # For JSON mode, return the raw context window dict
            results = search_nodes(kg, args.query, embed_fn, top_k=args.top_k, expand_depth=0)
            seed_ids = [r["node_id"] for r in results]
            context_data = kg.get_context_window(seed_ids, depth=args.depth, max_nodes=args.top_k)
            print(json.dumps(context_data, indent=2, cls=GraphEncoder))
        else:
            print(ctx)

    elif args.command == "ask":
        if not args.query_model:
            print("Error: --query-model MODEL is required for the 'ask' command.")
            return

        chat_model = args.query_model

        if args.provider == "anthropic":
            _api_key = _get_anthropic_api_key()

            def llm_fn(prompt: str) -> str:
                return claude_chat(prompt, model=chat_model, api_key=_api_key)
        elif args.provider == "bedrock":
            _br_region = args.bedrock_region
            _br_profile = args.bedrock_profile

            def llm_fn(prompt: str) -> str:
                return bedrock_chat(prompt, model=chat_model, region=_br_region, profile=_br_profile)
        else:
            chat_url = args.ollama_url.rstrip("/")

            def llm_fn(prompt: str) -> str:
                return ollama_chat(prompt, model=chat_model, url=chat_url)

        answer = ask(kg, args.question, embed_fn, llm_fn, max_nodes=args.top_k,
                     search_mode=args.search_mode, alpha=args.alpha)
        if args.json_output:
            print(json.dumps({"question": args.question, "answer": answer}, indent=2))
        else:
            print(answer)

    elif args.command == "node":
        node = kg.get_node(args.node_id)
        if node:
            if args.json_output:
                print(json.dumps(node, indent=2, cls=GraphEncoder))
            else:
                label = node.get("label", args.node_id)
                ntype = node.get("type", "?")
                conf = node.get("confidence", 1.0)
                print(f"[{ntype}] {label}  (id={args.node_id}, confidence={conf:.2f})")
                for k, v in node.get("properties", {}).items():
                    print(f"  {k}: {v}")
        else:
            print(f"Node '{args.node_id}' not found.")

    elif args.command == "neighbors":
        neighbors = kg.get_neighbors(args.node_id, max_depth=args.depth)
        if args.json_output:
            data = [{"node_id": nid, **ndata} for nid, ndata in neighbors]
            print(json.dumps(data, indent=2, cls=GraphEncoder))
        else:
            if not neighbors:
                print(f"No neighbors found for '{args.node_id}'.")
            for nid, ndata in neighbors:
                print(f"  {nid}: {ndata.get('label', nid)} ({ndata.get('type', '?')})")

    elif args.command == "path":
        path = kg.shortest_path(args.source, args.target)
        if args.json_output:
            print(json.dumps({"source": args.source, "target": args.target, "path": path}, indent=2))
        else:
            if path is None:
                print(f"No path found between '{args.source}' and '{args.target}'.")
            else:
                print(" -> ".join(path))

    elif args.command == "stats":
        import pprint
        s = kg.stats()
        if args.json_output:
            print(json.dumps(s, indent=2, cls=GraphEncoder))
        else:
            pprint.pprint(s)


if __name__ == "__main__":
    main()
