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

from knowledge_graph import KnowledgeGraph, GraphEncoder, ollama_embed

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
) -> list[dict[str, Any]]:
    """Semantic search over the knowledge graph.

    Args:
        kg: A loaded KnowledgeGraph with pre-computed node embeddings.
        query: Natural-language query string.
        embed_fn: Embedding function for the query text.
        top_k: Number of results to return.
        node_types: Optional filter by node type.
        expand_depth: Neighborhood expansion depth for each result.

    Returns:
        List of result dicts from ``kg.search()``.
    """
    logger.debug(
        "search_nodes: %d embeddings loaded, %d nodes in graph",
        len(kg._embeddings), len(kg._data["nodes"]),
    )
    results = kg.search(
        query,
        embed_fn,
        top_k=top_k,
        node_types=node_types,
        expand_depth=expand_depth,
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
) -> str:
    """Search the graph and format a context block for an LLM prompt.

    Args:
        kg: A loaded KnowledgeGraph with pre-computed node embeddings.
        query: Natural-language query string.
        embed_fn: Embedding function for the query text.
        max_nodes: Maximum number of nodes in the context window.
        depth: Neighborhood expansion depth.

    Returns:
        A formatted text block suitable for injecting into an LLM prompt.
    """
    results = search_nodes(kg, query, embed_fn, top_k=max_nodes, expand_depth=0)
    if not results:
        logger.debug("build_context: no search results for query")
        return "(No relevant nodes found in the knowledge graph.)"

    seed_ids = [r["node_id"] for r in results]
    context = kg.get_context_window(seed_ids, depth=depth, max_nodes=max_nodes)
    logger.debug(
        "build_context: %d seed nodes expanded to %d nodes, %d edges (depth=%d)",
        len(seed_ids), len(context["nodes"]), len(context["edges"]), depth,
    )

    lines: list[str] = []
    lines.append(f"## Knowledge Graph Context ({len(context['nodes'])} nodes, "
                 f"{len(context['edges'])} edges)")
    lines.append("")

    # Nodes
    for nid, ndata in context["nodes"].items():
        label = ndata.get("label", nid)
        ntype = ndata.get("type", "concept")
        conf = ndata.get("confidence", 1.0)
        desc = ndata.get("properties", {}).get("description", "")
        line = f"- [{ntype}] {label} (id={nid}, confidence={conf:.2f})"
        if desc:
            line += f"\n    {desc}"
        lines.append(line)

    # Edges
    if context["edges"]:
        lines.append("")
        lines.append("### Relationships")
        for edge in context["edges"]:
            src = edge.get("source", "?")
            tgt = edge.get("target", "?")
            rel = edge.get("relation", "related_to")
            lines.append(f"- {src} --[{rel}]--> {tgt}")

    return "\n".join(lines)


def ask(
    kg: KnowledgeGraph,
    question: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    llm_fn: Callable[[str], str],
    *,
    max_nodes: int = 30,
) -> str:
    """Full RAG pipeline: search the graph, build context, call an LLM.

    Args:
        kg: A loaded KnowledgeGraph with pre-computed node embeddings.
        question: The user's question.
        embed_fn: Embedding function for the query text.
        llm_fn: A callable that takes a prompt string and returns the
                 LLM's response string.
        max_nodes: Maximum number of nodes in the context window.

    Returns:
        The LLM's answer string.
    """
    t0 = time.monotonic()
    context = build_context(kg, question, embed_fn, max_nodes=max_nodes)
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
# Ollama chat helper
# ---------------------------------------------------------------------------


def ollama_chat(prompt: str, *, model: str, url: str) -> str:
    """Call Ollama ``/api/chat`` with a single user message."""
    endpoint = f"{url.rstrip('/')}/api/chat"
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
        with urllib.request.urlopen(req, timeout=1200) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Chat request failed (HTTP {exc.code}): "
            f"POST {endpoint} with model '{model}'. "
            f"Check that the Ollama server is running at {url} "
            f"and the model '{model}' is available (ollama list)."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {endpoint}: {exc.reason}. "
            f"Is the server running?"
        ) from exc
    elapsed = time.monotonic() - t0
    answer = body.get("message", {}).get("content", "").strip()
    logger.debug("ollama_chat: response=%d chars (%.1fs)", len(answer), elapsed)
    return answer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    # Shared flags as a parent parser so they work with subcommands
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--query-model", "--ollama", nargs="?", const="qwen3-coder:30b",
                        metavar="MODEL", dest="query_model",
                        help="Ollama model for LLM query/chat (default: qwen3-coder:30b)")
    shared.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama server URL (default: http://localhost:11434)")
    shared.add_argument("--embed-url", default=None, metavar="URL",
                        help="Ollama server URL for embeddings (default: same as --ollama-url)")
    shared.add_argument("--embed-model", default=None, metavar="MODEL",
                        help="Ollama embedding model for query embedding "
                             "(default: auto-detect from graph, or nomic-embed-text)")
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
        embed_model = "nomic-embed-text"
        logger.info("No embed model in graph metadata, falling back to '%s'", embed_model)

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
        chat_url = args.ollama_url.rstrip("/")

        def llm_fn(prompt: str) -> str:
            return ollama_chat(prompt, model=chat_model, url=chat_url)

        answer = ask(kg, args.question, embed_fn, llm_fn, max_nodes=args.top_k)
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
