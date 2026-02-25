#!/usr/bin/env python3
"""
benchmark_models.py — Compare LLM extraction models for knowledge graph ingestion.

Runs the same markdown file(s) through multiple models, each into an isolated
temporary graph, and produces a side-by-side comparison of extraction quality.

Usage:
    # Compare three models on one document
    python benchmark_models.py ingest/converted/sar_manual.md \
        --models qwen3-coder:30b gemma3:27b llama3.1:70b \
        --ollama-url http://exo:11434

    # Compare on multiple documents
    python benchmark_models.py ingest/converted/*.md \
        --models qwen3-coder:30b gemma3:27b \
        --ollama-url http://exo:11434

    # JSON output for further analysis
    python benchmark_models.py doc.md \
        --models modelA modelB --json

    # Limit to first N sections (quick smoke test)
    python benchmark_models.py doc.md \
        --models modelA modelB --max-sections 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from knowledge_graph import KnowledgeGraph, _salvage_truncated_json

logger = logging.getLogger(__name__)


def _make_extract_fn(
    model: str,
    ollama_url: str,
    verbose: bool = False,
) -> Callable[[str], list[dict[str, Any]]]:
    """Build an LLM extraction function for the given model."""

    def extract(prompt: str) -> list[dict[str, Any]]:
        if verbose:
            logger.debug("[%s] Prompt length: %d chars", model, len(prompt))

        payload = json.dumps({
            "model": model,
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
        if verbose:
            logger.debug("[%s] Raw response length: %d chars", model, len(raw))
        if not raw:
            logger.warning("[%s] Empty response", model)
            return []

        # Try direct parse
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None

        # Try bracket extraction
        if parsed is None:
            start = raw.find("[")
            if start != -1:
                end = raw.rfind("]")
                if end > start:
                    try:
                        parsed = json.loads(raw[start:end + 1])
                    except json.JSONDecodeError:
                        pass

        # Salvage truncated JSON
        if parsed is None:
            salvaged = _salvage_truncated_json(raw)
            if salvaged is not None:
                logger.warning("[%s] Salvaged %d items from truncated JSON", model, len(salvaged))
                return salvaged
            logger.error("[%s] JSON parse failed (%d chars)", model, len(raw))
            return []

        # Unwrap dict-wrapped arrays
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
            return []
        if isinstance(parsed, list):
            return parsed
        return []

    return extract


def run_benchmark(
    md_files: list[Path],
    models: list[str],
    ollama_url: str,
    max_sections: int | None = None,
    verbose: bool = False,
    quiet: bool = False,
) -> list[dict[str, Any]]:
    """Run each model against the same file(s) and collect stats.

    Returns a list of result dicts, one per model.
    """
    results: list[dict[str, Any]] = []

    # Read all files upfront
    documents: list[dict[str, Any]] = []
    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        documents.append({
            "path": md_path,
            "text": text,
            "doc_id": md_path.stem,
        })

    total_chars = sum(len(d["text"]) for d in documents)

    for model_idx, model in enumerate(models):
        if not quiet:
            print(f"\n{'='*60}")
            print(f"  Model {model_idx + 1}/{len(models)}: {model}")
            print(f"{'='*60}")

        extract_fn = _make_extract_fn(model, ollama_url, verbose=verbose)

        model_result: dict[str, Any] = {
            "model": model,
            "files": [],
            "total_sections": 0,
            "total_triples": 0,
            "total_nodes_added": 0,
            "total_edges_added": 0,
            "total_proposals_created": 0,
            "total_errors": 0,
            "total_hallucinated": 0,
            "total_elapsed": 0.0,
            "section_times": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, f"bench_{model.replace(':', '_').replace('/', '_')}.json")
            kg = KnowledgeGraph(graph_path)

            for doc in documents:
                md_path = doc["path"]
                text = doc["text"]
                doc_id = doc["doc_id"]

                if not quiet:
                    print(f"\n  File: {md_path.name}")

                # Track per-section timing
                section_times: list[float] = []

                def _progress(event: dict[str, Any]) -> None:
                    if quiet:
                        return
                    ev = event["event"]
                    idx = event.get("index", 0)
                    total = event.get("total", 0)
                    heading = event.get("heading", "?")
                    chars = event.get("char_count", 0)
                    tag = f"[{idx + 1}/{total}]"

                    if ev == "section_skip":
                        print(f"    {tag} Skip: \"{heading}\" ({chars:,} chars)")
                    elif ev == "section_start":
                        print(f"    {tag} \"{heading}\" ({chars:,} chars)...",
                              end="", flush=True)
                    elif ev == "section_done":
                        elapsed = event.get("elapsed_seconds", 0)
                        triples = event.get("triples", 0)
                        nodes = event.get("nodes_added", 0)
                        errors = event.get("errors", [])
                        section_times.append(elapsed)
                        if errors:
                            print(f" {triples} triples → {nodes} nodes "
                                  f"({len(errors)} errors, {elapsed}s)")
                        else:
                            print(f" {triples} triples → {nodes} nodes ({elapsed}s)")

                t0 = time.monotonic()

                # Optionally limit sections for quick comparison
                extra_kwargs: dict[str, Any] = {}
                if max_sections is not None:
                    extra_kwargs["max_section_chars"] = 6000  # default

                stats = kg.ingest_markdown(
                    text,
                    doc_id=doc_id,
                    llm_extract_fn=extract_fn,
                    original_path=md_path.resolve(),
                    doc_properties={
                        "file_path": str(md_path),
                        "file_size": md_path.stat().st_size,
                    },
                    progress_fn=_progress,
                    preserve_source=False,  # no need to store sources for benchmark
                )
                elapsed = time.monotonic() - t0

                # Count hallucinated sections
                hallucinated = sum(
                    1 for s in stats.get("sections", [])
                    if any("hallucinated" in str(e).lower() for e in s.get("errors", []))
                )

                file_result = {
                    "file": md_path.name,
                    "sections": stats["total_sections"],
                    "triples": stats["total_triples"],
                    "nodes_added": stats["total_nodes_added"],
                    "edges_added": stats["total_edges_added"],
                    "proposals": stats["total_proposals_created"],
                    "errors": len(stats["errors"]),
                    "hallucinated": hallucinated,
                    "elapsed": round(elapsed, 1),
                    "section_times": [round(t, 1) for t in section_times],
                }
                model_result["files"].append(file_result)
                model_result["total_sections"] += stats["total_sections"]
                model_result["total_triples"] += stats["total_triples"]
                model_result["total_nodes_added"] += stats["total_nodes_added"]
                model_result["total_edges_added"] += stats["total_edges_added"]
                model_result["total_proposals_created"] += stats["total_proposals_created"]
                model_result["total_errors"] += len(stats["errors"])
                model_result["total_hallucinated"] += hallucinated
                model_result["total_elapsed"] += elapsed
                model_result["section_times"].extend(section_times)

                if not quiet:
                    print(f"    → {stats['total_triples']} triples, "
                          f"{stats['total_nodes_added']} nodes, "
                          f"{len(stats['errors'])} errors, {elapsed:.1f}s")

            # Compute graph-level stats
            graph_stats = kg.stats()
            model_result["graph_nodes"] = graph_stats["num_nodes"]
            model_result["graph_edges"] = graph_stats["num_edges"]

            # Compute timing stats
            times = model_result["section_times"]
            if times:
                model_result["avg_section_time"] = round(sum(times) / len(times), 1)
                model_result["median_section_time"] = round(sorted(times)[len(times) // 2], 1)
                model_result["max_section_time"] = round(max(times), 1)
                model_result["min_section_time"] = round(min(times), 1)
            else:
                model_result["avg_section_time"] = 0
                model_result["median_section_time"] = 0
                model_result["max_section_time"] = 0
                model_result["min_section_time"] = 0

            model_result["total_elapsed"] = round(model_result["total_elapsed"], 1)

            # Compute triples per 1K chars
            if total_chars > 0:
                model_result["triples_per_1k_chars"] = round(
                    model_result["total_triples"] / (total_chars / 1000), 1
                )
            else:
                model_result["triples_per_1k_chars"] = 0

            # Error rate
            if model_result["total_sections"] > 0:
                model_result["error_rate"] = round(
                    model_result["total_errors"] / max(model_result["total_triples"], 1) * 100, 1
                )
            else:
                model_result["error_rate"] = 0

        results.append(model_result)

    return results


def print_comparison(results: list[dict[str, Any]], total_chars: int) -> None:
    """Print a formatted comparison table."""

    if not results:
        print("No results to compare.")
        return

    # Determine column widths
    model_names = [r["model"] for r in results]
    max_model_len = max(len(m) for m in model_names)
    col_w = max(max_model_len + 2, 20)

    # Header
    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON — {len(results)} models, "
          f"{total_chars:,} chars input")
    print(f"{'='*70}")

    # Table header
    header = f"  {'Metric':<28}"
    for r in results:
        header += f"  {r['model']:>{col_w}}"
    print(f"\n{header}")
    print(f"  {'-'*28}" + f"  {'-'*col_w}" * len(results))

    # Rows
    rows = [
        ("Sections processed", "total_sections", "d"),
        ("Triples extracted", "total_triples", "d"),
        ("Nodes added", "total_nodes_added", "d"),
        ("Edges added", "total_edges_added", "d"),
        ("New relations proposed", "total_proposals_created", "d"),
        ("Errors", "total_errors", "d"),
        ("Hallucinated sections", "total_hallucinated", "d"),
        ("Error rate (%)", "error_rate", ".1f"),
        ("Triples / 1K chars", "triples_per_1k_chars", ".1f"),
        ("Total time (s)", "total_elapsed", ".1f"),
        ("Avg section time (s)", "avg_section_time", ".1f"),
        ("Median section time (s)", "median_section_time", ".1f"),
        ("Max section time (s)", "max_section_time", ".1f"),
        ("Final graph nodes", "graph_nodes", "d"),
        ("Final graph edges", "graph_edges", "d"),
    ]

    # Track best values for highlighting
    for label, key, fmt in rows:
        values = [r.get(key, 0) for r in results]
        row = f"  {label:<28}"
        for val in values:
            formatted = f"{val:{fmt}}"
            row += f"  {formatted:>{col_w}}"
        print(row)

    # Winner summary
    print(f"\n  {'-'*28}" + f"  {'-'*col_w}" * len(results))

    # Most triples
    best_triples_idx = max(range(len(results)), key=lambda i: results[i]["total_triples"])
    print(f"  Most triples:      {results[best_triples_idx]['model']}"
          f" ({results[best_triples_idx]['total_triples']})")

    # Fewest errors
    best_errors_idx = min(range(len(results)), key=lambda i: results[i]["total_errors"])
    print(f"  Fewest errors:     {results[best_errors_idx]['model']}"
          f" ({results[best_errors_idx]['total_errors']})")

    # Fastest
    best_time_idx = min(range(len(results)), key=lambda i: results[i]["total_elapsed"])
    print(f"  Fastest:           {results[best_time_idx]['model']}"
          f" ({results[best_time_idx]['total_elapsed']}s)")

    # Best triples/second
    best_tps_idx = max(
        range(len(results)),
        key=lambda i: (results[i]["total_triples"] / results[i]["total_elapsed"])
        if results[i]["total_elapsed"] > 0 else 0,
    )
    if results[best_tps_idx]["total_elapsed"] > 0:
        tps = results[best_tps_idx]["total_triples"] / results[best_tps_idx]["total_elapsed"]
        print(f"  Best triples/sec:  {results[best_tps_idx]['model']}"
              f" ({tps:.2f} t/s)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM extraction models for knowledge graph ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python benchmark_models.py doc.md --models qwen3-coder:30b gemma3:27b
  python benchmark_models.py docs/*.md --models modelA modelB --ollama-url http://exo:11434
  python benchmark_models.py doc.md --models modelA modelB --json
  python benchmark_models.py doc.md --models modelA modelB --max-sections 5
""",
    )
    parser.add_argument("files", nargs="+", metavar="FILE",
                        help="Markdown file(s) to benchmark")
    parser.add_argument("--models", nargs="+", required=True, metavar="MODEL",
                        help="Ollama model names to compare")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--max-sections", type=int, default=None, metavar="N",
                        help="Limit to first N sections per file (quick test)")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output results as JSON")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show debug output")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress progress output (only show final comparison)")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    elif not args.quiet:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # Validate files
    md_files: list[Path] = []
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)
        md_files.append(p)

    total_chars = sum(p.stat().st_size for p in md_files)

    if not args.quiet:
        print(f"Benchmark: {len(md_files)} file(s), {len(args.models)} model(s)")
        print(f"  Files: {', '.join(p.name for p in md_files)}")
        print(f"  Models: {', '.join(args.models)}")
        print(f"  Server: {args.ollama_url}")
        if args.max_sections:
            print(f"  Max sections per file: {args.max_sections}")

    # Apply max_sections by monkey-patching parse_markdown_sections if needed
    _orig_parse = KnowledgeGraph.parse_markdown_sections
    if args.max_sections is not None:
        max_s = args.max_sections

        @staticmethod
        def _limited_parse(text, **kwargs):
            sections = _orig_parse(text, **kwargs)
            return sections[:max_s]

        KnowledgeGraph.parse_markdown_sections = _limited_parse

    try:
        results = run_benchmark(
            md_files=md_files,
            models=args.models,
            ollama_url=args.ollama_url,
            max_sections=args.max_sections,
            verbose=args.verbose,
            quiet=args.quiet,
        )
    finally:
        # Restore original method
        KnowledgeGraph.parse_markdown_sections = _orig_parse

    if args.json_output:
        # Strip section_times from JSON for cleaner output (they're verbose)
        output = []
        for r in results:
            r_copy = {k: v for k, v in r.items() if k != "section_times"}
            for f in r_copy.get("files", []):
                f.pop("section_times", None)
            output.append(r_copy)
        print(json.dumps(output, indent=2))
    else:
        total_chars = sum(len(p.read_text(encoding="utf-8")) for p in md_files)
        print_comparison(results, total_chars)


if __name__ == "__main__":
    main()
