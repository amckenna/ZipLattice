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
import math
import os
import re
import time
import urllib.error
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


# ---------------------------------------------------------------------------
# Shared HTTP error handling
# ---------------------------------------------------------------------------


def read_http_error_detail(exc: urllib.error.HTTPError) -> str:
    """Safely extract the response body text from an ``HTTPError``.

    Returns the decoded body string (stripped), or ``""`` if the body
    cannot be read (e.g. the stream was already consumed).
    """
    try:
        return exc.read().decode(errors="replace").strip()
    except OSError:
        logger.debug("Could not read error detail from HTTP %d response", exc.code)
        return ""


def ollama_embed(
    texts: list[str], *, model: str, url: str = "http://localhost:11434"
) -> list[list[float]]:
    """Call an OpenAI-compatible ``/v1/embeddings`` endpoint.

    Works with Ollama (>=0.1.14), llama.cpp, vLLM, LocalAI, and any
    other server that implements the OpenAI embeddings API.

    Args:
        texts: List of strings to embed.
        model: Model name (e.g. ``qwen3-embedding``).
        url: Server base URL.

    Returns:
        One embedding vector per input text, in the same order.
    """
    endpoint = f"{url.rstrip('/')}/v1/embeddings"
    payload = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    logger.debug("ollama_embed: POST %s  model=%s  texts=%d", endpoint, model, len(texts))
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = read_http_error_detail(exc)
        msg = (
            f"Embedding request failed (HTTP {exc.code}): "
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
    # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
    data = sorted(body["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in data]


# ---------------------------------------------------------------------------
# Shared LLM response parsing
# ---------------------------------------------------------------------------


def _parse_llm_json(raw: str) -> list[dict[str, Any]]:
    """Three-tier JSON recovery for LLM extraction responses.

    Handles the common failure modes when asking an LLM to return a JSON
    array of triples:

    1. **Direct parse** — works when the model returns clean JSON.
    2. **Bracket extraction** — finds ``[…]`` inside surrounding prose.
    3. **Truncation salvage** — recovers complete objects when the model
       hit its token limit mid-array.

    Also unwraps dict-wrapped arrays (e.g. ``{"entities": [...]}``) into
    a plain list.

    Returns an empty list if all recovery strategies fail.
    """
    # 1. Direct parse
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    # 2. Bracket extraction
    if parsed is None:
        start = raw.find("[")
        if start != -1:
            end = raw.rfind("]")
            if end > start:
                try:
                    parsed = json.loads(raw[start:end + 1])
                except json.JSONDecodeError:
                    pass

    # 3. Salvage truncated JSON
    if parsed is None:
        salvaged = _salvage_truncated_json(raw)
        if salvaged is not None:
            logger.warning(
                "JSON truncated, salvaged %d complete triple(s)", len(salvaged),
            )
            return salvaged
        logger.error("JSON parse failed (%d chars): %s", len(raw), raw[:500])
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


_EXTRACTION_SYSTEM_PROMPT = (
    "You are a JSON extraction engine. "
    "Respond with ONLY a valid JSON array. "
    "No explanations, no markdown."
)


def _parse_extraction_response(raw: str, *, model: str, label: str) -> list[dict[str, Any]]:
    """Shared post-processing for all extraction functions.

    Strips thinking tags, checks for empty responses, and parses JSON.
    """
    raw = _strip_thinking(raw)
    if not raw:
        logger.warning("%s returned empty response (model=%s)", label, model)
        return []
    return _parse_llm_json(raw)


def local_extract(
    prompt: str, *, model: str, url: str = "http://localhost:11434"
) -> list[dict[str, Any]]:
    """Call an OpenAI-compatible ``/v1/chat/completions`` endpoint for extraction.

    Sends the extraction system prompt, parses the JSON response with
    :func:`_parse_llm_json`, and returns a list of triple dicts.

    Args:
        prompt: The user-facing extraction prompt (section text).
        model: Model name (e.g. ``qwen3-coder:30b``).
        url: Server base URL.

    Returns:
        List of extracted triple dicts, or ``[]`` on failure.
    """
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 32768,
    }).encode()
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    logger.debug("local_extract: POST %s  model=%s  prompt=~%d tokens", endpoint, model, len(prompt) // 4)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = read_http_error_detail(exc)
        logger.error(
            "Extraction request failed (HTTP %d, model=%s): %s",
            exc.code, model, detail or "(no detail)",
        )
        return []
    except urllib.error.URLError as exc:
        logger.error("Cannot connect to %s (model=%s): %s", url, model, exc.reason)
        return []

    elapsed = time.monotonic() - t0
    # OpenAI format: choices[0].message.content
    raw = body["choices"][0]["message"]["content"].strip()
    logger.debug("local_extract: response=~%d tokens (%.1fs)", len(raw) // 4, elapsed)
    return _parse_extraction_response(raw, model=model, label="LLM")


# ---------------------------------------------------------------------------
# Claude (Anthropic) API helpers
# ---------------------------------------------------------------------------

_ANTHROPIC_API_URL = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"


def _get_anthropic_api_key() -> str:
    """Return the Anthropic API key from the environment.

    Raises ``RuntimeError`` with a clear message if the key is not set.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it to your Anthropic API key to use the 'anthropic' provider."
        )
    return key


def _rate_limit_wait(detail: str, attempt: int) -> float:
    """Compute wait time for a 429 retry.

    Tries to parse ``retry-after`` from the error body, otherwise uses
    exponential backoff: 30s, 60s, 120s, 240s, ...
    """
    # Try to extract retry-after hint from JSON error body
    try:
        err = json.loads(detail) if detail else {}
        msg = err.get("error", {}).get("message", "")
        m = re.search(r"retry.after\D*(\d+)", msg, re.IGNORECASE)
        if m:
            return float(m.group(1))
    except (json.JSONDecodeError, KeyError, AttributeError):
        pass
    return min(30 * (2 ** attempt), 300)


def _anthropic_request(
    payload_dict: dict[str, Any],
    *,
    api_key: str,
    model: str,
    label: str,
    on_error: str = "raise",
) -> dict[str, Any] | None:
    """Shared Anthropic HTTP request with 429 retry logic.

    Args:
        payload_dict: JSON-serializable request body.
        api_key: Anthropic API key.
        model: Model ID (for logging and error messages).
        label: Human label for log messages (e.g. ``"chat"``, ``"extract"``).
        on_error: ``"raise"`` to raise RuntimeError on non-retryable errors,
            ``"return"`` to log and return None.

    Returns:
        Parsed response body dict, or None if *on_error* is ``"return"``
        and the request failed.
    """
    endpoint = f"{_ANTHROPIC_API_URL}/v1/messages"
    payload = json.dumps(payload_dict).encode()
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
    }
    req = urllib.request.Request(endpoint, data=payload, headers=headers)
    logger.debug("claude_%s: POST %s  model=%s  prompt payload=%d bytes",
                 label, endpoint, model, len(payload))

    max_retries = 5
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = read_http_error_detail(exc)
            if exc.code == 429 and attempt < max_retries - 1:
                wait = _rate_limit_wait(detail, attempt)
                logger.warning(
                    "Claude %s rate-limited (429, model=%s), retry %d/%d in %.0fs",
                    label, model, attempt + 1, max_retries - 1, wait,
                )
                time.sleep(wait)
                req = urllib.request.Request(endpoint, data=payload, headers=headers)
                continue
            if on_error == "return":
                logger.error(
                    "Claude %s request failed (HTTP %d, model=%s): %s",
                    label, exc.code, model, detail or "(no detail)",
                )
                return None
            msg = (
                f"Claude {label} request failed (HTTP {exc.code}): "
                f"POST {endpoint} with model '{model}'."
            )
            if detail:
                msg += f"\nServer response: {detail}"
            if exc.code == 401:
                msg += "\nHint: check that ANTHROPIC_API_KEY is valid."
            elif exc.code == 404:
                msg += f"\nHint: model '{model}' may not exist. Check the model ID."
            elif exc.code == 429:
                msg += "\nHint: rate limited. Wait and retry."
            raise RuntimeError(msg) from exc
        except urllib.error.URLError as exc:
            if on_error == "return":
                logger.error("Cannot connect to %s: %s", endpoint, exc.reason)
                return None
            raise RuntimeError(
                f"Cannot connect to {endpoint}: {exc.reason}."
            ) from exc
    # All retries exhausted (all were 429s)
    if on_error == "return":
        return None
    raise RuntimeError(f"Claude {label} failed after {max_retries} retries.")


def claude_chat(prompt: str, *, model: str, api_key: str | None = None) -> str:
    """Call the Anthropic Messages API and return the assistant's text.

    Args:
        prompt: The user message to send.
        model: Anthropic model ID (e.g. ``claude-haiku-4-5``).
        api_key: API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.

    Returns:
        The assistant's response text.
    """
    if not api_key:
        api_key = _get_anthropic_api_key()

    t0 = time.monotonic()
    body = _anthropic_request(
        {"model": model, "max_tokens": 16384,
         "messages": [{"role": "user", "content": prompt}]},
        api_key=api_key, model=model, label="chat",
    )
    elapsed = time.monotonic() - t0
    answer = body["content"][0]["text"].strip()
    logger.debug("claude_chat: response=%d chars (%.1fs)", len(answer), elapsed)
    return answer


def claude_extract(prompt: str, *, model: str, api_key: str | None = None) -> list[dict[str, Any]]:
    """Call the Anthropic Messages API for JSON extraction.

    Mirrors the extraction pattern used by the OpenAI-compatible path:
    sends a system prompt requesting pure JSON, then parses the response
    with the same three-tier recovery logic.

    Args:
        prompt: The user-facing extraction prompt (section text).
        model: Anthropic model ID (e.g. ``claude-haiku-4-5``).
        api_key: API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.

    Returns:
        List of extracted triple dicts, or ``[]`` on failure.
    """
    if not api_key:
        api_key = _get_anthropic_api_key()

    t0 = time.monotonic()
    body = _anthropic_request(
        {"model": model, "max_tokens": 32768, "temperature": 0.1,
         "system": _EXTRACTION_SYSTEM_PROMPT,
         "messages": [{"role": "user", "content": prompt}]},
        api_key=api_key, model=model, label="extract", on_error="return",
    )
    if body is None:
        return []

    elapsed = time.monotonic() - t0
    raw = body["content"][0]["text"].strip()
    logger.debug("claude_extract: raw response=%d chars (%.1fs)", len(raw), elapsed)
    return _parse_extraction_response(raw, model=model, label="Claude")


# ---------------------------------------------------------------------------
# AWS Bedrock API helpers
# ---------------------------------------------------------------------------


def _get_bedrock_client(region: str | None = None, profile: str | None = None) -> Any:
    """Return a ``boto3`` Bedrock Runtime client.

    Relies on standard AWS credential resolution (env vars, shared
    credentials file, IAM role, etc.).  The *region* falls back to
    ``AWS_DEFAULT_REGION`` / ``AWS_REGION`` env vars, then ``us-east-1``.

    When *profile* is given, a ``boto3.Session`` is created with that
    profile name, allowing use of named profiles from
    ``~/.aws/credentials`` and ``~/.aws/config``.

    Raises ``RuntimeError`` with a clear message if ``boto3`` is not
    installed or credentials cannot be resolved.
    """
    try:
        import boto3  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "boto3 is required for the 'bedrock' provider.  "
            "Install it with:  pip install boto3"
        )
    if region is None:
        region = (
            os.environ.get("AWS_DEFAULT_REGION")
            or os.environ.get("AWS_REGION")
            or "us-east-1"
        )
    try:
        if profile:
            session = boto3.Session(profile_name=profile, region_name=region)
            client = session.client("bedrock-runtime")
        else:
            client = boto3.client("bedrock-runtime", region_name=region)
        # Force credential resolution so we fail fast with a clear message.
        client.meta.region_name  # noqa: B018 — triggers lazy init
    except Exception as exc:
        raise RuntimeError(
            f"Cannot create Bedrock client (region={region}, profile={profile}): {exc}.  "
            "Ensure AWS credentials are configured (AWS_ACCESS_KEY_ID / "
            "AWS_SECRET_ACCESS_KEY, ~/.aws/credentials, or an IAM role)."
        ) from exc
    return client


def _is_bedrock_retryable(exc: Exception) -> bool:
    """Return True if *exc* is a transient Bedrock error worth retrying.

    Covers throttling, network stream errors ("Error in input stream"),
    connection resets, read timeouts, and generic ``botocore`` transport
    failures.
    """
    msg = str(exc).lower()
    retryable_phrases = [
        "error in input stream",
        "throttl",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "broken pipe",
        "network",
        "endpoint url",
        "read timeout",
        "connect timeout",
        "service unavailable",
        "internal server error",
        "bad gateway",
        "too many requests",
    ]
    if any(phrase in msg for phrase in retryable_phrases):
        return True
    # botocore-specific exceptions
    exc_type = type(exc).__name__
    if exc_type in (
        "ThrottlingException",
        "ReadTimeoutError",
        "ConnectTimeoutError",
        "EndpointConnectionError",
        "ConnectionClosedError",
        "EventStreamError",
    ):
        return True
    return False


def _bedrock_converse_with_retries(
    converse_kwargs: dict[str, Any],
    *,
    model: str,
    region: str | None = None,
    profile: str | None = None,
    label: str,
    on_error: str = "raise",
) -> dict[str, Any] | None:
    """Shared Bedrock Converse call with transient-error retry logic.

    Args:
        converse_kwargs: Keyword arguments for ``client.converse()``.
        model: Model ID (for logging).
        region: AWS region.
        profile: AWS profile name.
        label: Human label for log messages (e.g. ``"chat"``, ``"extract"``).
        on_error: ``"raise"`` to raise RuntimeError on non-retryable errors,
            ``"return"`` to log and return None.

    Returns:
        Converse response dict, or None if *on_error* is ``"return"``
        and the request failed.
    """
    client = _get_bedrock_client(region, profile=profile)
    logger.debug("bedrock_%s: model=%s", label, model)

    max_retries = 5
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return client.converse(**converse_kwargs)
        except Exception as exc:
            last_exc = exc
            if _is_bedrock_retryable(exc) and attempt < max_retries - 1:
                wait = min(30 * (2 ** attempt), 300)
                logger.warning(
                    "Bedrock %s error (model=%s, %s), retry %d/%d in %.0fs",
                    label, model, exc, attempt + 1, max_retries - 1, wait,
                )
                time.sleep(wait)
                client = _get_bedrock_client(region, profile=profile)
                continue
            if on_error == "return":
                logger.error(
                    "Bedrock %s failed after %d attempt(s) (model=%s): %s",
                    label, attempt + 1, model, exc,
                )
                return None
            raise RuntimeError(
                f"Bedrock {label} failed (model={model}): {exc}"
            ) from exc

    # All retries exhausted
    if on_error == "return":
        return None
    raise RuntimeError(
        f"Bedrock {label} failed after {max_retries} retries: {last_exc}"
    )


def bedrock_chat(
    prompt: str,
    *,
    model: str,
    region: str | None = None,
    profile: str | None = None,
) -> str:
    """Call AWS Bedrock Converse API and return the assistant's text.

    Args:
        prompt: The user message to send.
        model: Bedrock model ID (e.g.
            ``us.anthropic.claude-sonnet-4-20250514``,
            ``amazon.nova-pro-v1:0``).
        region: AWS region.  Falls back to env / ``us-east-1``.
        profile: AWS profile name from ``~/.aws/credentials``.

    Returns:
        The assistant's response text.
    """
    t0 = time.monotonic()
    body = _bedrock_converse_with_retries(
        {"modelId": model,
         "messages": [{"role": "user", "content": [{"text": prompt}]}],
         "inferenceConfig": {"maxTokens": 16384}},
        model=model, region=region, profile=profile, label="chat",
    )
    elapsed = time.monotonic() - t0
    answer = body["output"]["message"]["content"][0]["text"].strip()
    logger.debug("bedrock_chat: response=%d chars (%.1fs)", len(answer), elapsed)
    return answer


def bedrock_extract(
    prompt: str,
    *,
    model: str,
    region: str | None = None,
    profile: str | None = None,
) -> list[dict[str, Any]]:
    """Call AWS Bedrock Converse API for JSON extraction.

    Mirrors the extraction pattern of :func:`claude_extract` and
    :func:`local_extract`:  sends a system prompt requesting pure JSON,
    then parses the response with the three-tier recovery logic.

    Args:
        prompt: The user-facing extraction prompt.
        model: Bedrock model ID.
        region: AWS region.
        profile: AWS profile name from ``~/.aws/credentials``.

    Returns:
        List of extracted triple dicts, or ``[]`` on failure.
    """
    t0 = time.monotonic()
    body = _bedrock_converse_with_retries(
        {"modelId": model,
         "messages": [{"role": "user", "content": [{"text": prompt}]}],
         "system": [{"text": _EXTRACTION_SYSTEM_PROMPT}],
         "inferenceConfig": {"maxTokens": 32768, "temperature": 0.1}},
        model=model, region=region, profile=profile,
        label="extract", on_error="return",
    )
    if body is None:
        return []

    elapsed = time.monotonic() - t0
    raw = body["output"]["message"]["content"][0]["text"].strip()
    logger.debug("bedrock_extract: raw response=%d chars (%.1fs)", len(raw), elapsed)
    return _parse_extraction_response(raw, model=model, label="Bedrock")


def bedrock_embed(
    texts: list[str],
    *,
    model: str,
    region: str | None = None,
    profile: str | None = None,
) -> list[list[float]]:
    """Generate embeddings via AWS Bedrock.

    Uses the Bedrock ``invoke_model`` API for embedding models such as
    ``amazon.titan-embed-text-v2:0`` or ``cohere.embed-english-v3``.

    For **Titan** models the request/response format is::

        Request:  {"inputText": "...", "dimensions": 1024, "normalize": true}
        Response: {"embedding": [...]}

    For **Cohere** models the format is::

        Request:  {"texts": [...], "input_type": "search_document"}
        Response: {"embeddings": [[...], ...]}

    Args:
        texts: List of strings to embed.
        model: Bedrock embedding model ID.
        region: AWS region.
        profile: AWS profile name from ``~/.aws/credentials``.

    Returns:
        List of embedding vectors (one per input text).
    """
    if not texts:
        return []

    client = _get_bedrock_client(region, profile=profile)
    logger.debug("bedrock_embed: model=%s  texts=%d", model, len(texts))

    is_cohere = "cohere" in model.lower()

    max_retries = 5

    if is_cohere:
        # Cohere supports batched embedding
        payload = json.dumps({
            "texts": texts,
            "input_type": "search_document",
        }).encode()
        for attempt in range(max_retries):
            try:
                resp = client.invoke_model(modelId=model, body=payload)
                result = json.loads(resp["body"].read())
                return result["embeddings"]
            except Exception as exc:
                if _is_bedrock_retryable(exc) and attempt < max_retries - 1:
                    wait = min(30 * (2 ** attempt), 300)
                    logger.warning(
                        "Bedrock embed error (%s), retry %d/%d in %.0fs",
                        exc, attempt + 1, max_retries - 1, wait,
                    )
                    time.sleep(wait)
                    client = _get_bedrock_client(region, profile=profile)
                    continue
                raise RuntimeError(
                    f"Bedrock embedding failed (model={model}): {exc}"
                ) from exc
    else:
        # Titan and other models — one text at a time
        embeddings: list[list[float]] = []
        for text in texts:
            payload = json.dumps({
                "inputText": text,
                "dimensions": 1024,
                "normalize": True,
            }).encode()
            for attempt in range(max_retries):
                try:
                    resp = client.invoke_model(modelId=model, body=payload)
                    result = json.loads(resp["body"].read())
                    embeddings.append(result["embedding"])
                    break
                except Exception as exc:
                    if _is_bedrock_retryable(exc) and attempt < max_retries - 1:
                        wait = min(30 * (2 ** attempt), 300)
                        logger.warning(
                            "Bedrock embed error (%s), retry %d/%d in %.0fs",
                            exc, attempt + 1, max_retries - 1, wait,
                        )
                        time.sleep(wait)
                        client = _get_bedrock_client(region, profile=profile)
                        continue
                    raise RuntimeError(
                        f"Bedrock embedding failed (model={model}): {exc}"
                    ) from exc
        return embeddings


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
# LLM extraction constants (used by ingest_document)
# ---------------------------------------------------------------------------

# Keys that indicate a valid extraction triple
_EXPECTED_KEYS = {
    "source", "subject", "head", "from", "entity1",
    "target", "object", "tail", "to", "entity2",
    "relation", "predicate", "relationship", "rel",
    "Concept", "concept", "Term", "term", "Name", "name",
}

# Key alias mapping for normalizing LLM output
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

# Source/target key sets for hallucination detection
_SRC_KEYS = {
    "source", "subject", "head", "from", "entity1",
    "Concept", "concept", "Term", "term", "Name", "name",
}
_TGT_KEYS = {"target", "object", "tail", "to", "entity2"}

# Stopwords for entity grounding checks
_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "are", "was",
    "were", "been", "being", "have", "has", "had", "does", "did",
    "will", "would", "could", "should", "may", "might", "can",
    "shall", "not", "but", "its", "from", "they", "them",
    "their", "there", "then", "than", "other", "which", "what",
    "when", "where", "who", "how", "all", "each", "every",
    "both", "few", "more", "most", "some", "such", "only",
    "also", "into", "over", "after", "before", "between",
    "under", "about", "these", "those", "through", "during",
    "while", "used", "using",
}


# ---------------------------------------------------------------------------
# Description merging helper
# ---------------------------------------------------------------------------


def _merge_description(
    existing_props: dict[str, Any],
    new_text: str,
    doc_id: str,
    confidence: float = 1.0,
) -> None:
    """Merge a new description into a node's properties.

    Maintains ``description_sources`` (list of per-document entries) and
    rebuilds ``description`` as a concatenation of all unique description
    texts joined with ``"; "``.
    """
    sources: list[dict[str, Any]] = existing_props.setdefault("description_sources", [])

    # Check if this doc_id already contributed
    for entry in sources:
        if entry["doc_id"] == doc_id:
            if entry["text"] == new_text:
                return  # identical — nothing to do
            # Same doc, different text — update in place
            entry["text"] = new_text
            entry["confidence"] = confidence
            entry["updated_at"] = now_iso()
            break
    else:
        # New doc_id
        sources.append({
            "text": new_text,
            "doc_id": doc_id,
            "confidence": confidence,
            "updated_at": now_iso(),
        })

    # Rebuild concatenated description, deduplicating by text content
    seen_texts: list[str] = []
    for entry in sources:
        if entry["text"] not in seen_texts:
            seen_texts.append(entry["text"])
    existing_props["description"] = "; ".join(seen_texts)


def _copy_source_file(
    src_kg: "KnowledgeGraph",
    dst_kg: "KnowledgeGraph",
    manifest_entry: dict[str, Any],
) -> None:
    """Copy a source file from one graph's sources directory to another.

    Updates *manifest_entry*'s ``stored_path`` to point to the new location
    inside *dst_kg*'s sources directory.  If the source file does not exist
    on disk the entry is kept but no file is copied.
    """
    stored = Path(manifest_entry.get("stored_path", ""))
    if not stored.is_absolute():
        stored = src_kg.graph_path.parent / stored
    if not stored.exists():
        logger.debug("Source file not found during merge copy: %s", stored)
        return

    dst_kg.sources_dir.mkdir(parents=True, exist_ok=True)
    dest = dst_kg.sources_dir / stored.name
    if not dest.exists() or dest.read_bytes() != stored.read_bytes():
        import shutil
        shutil.copy2(str(stored), str(dest))

    # Rewrite stored_path relative to the destination graph dir
    try:
        rel = dest.relative_to(dst_kg.graph_path.parent)
    except ValueError:
        rel = dest
    manifest_entry["stored_path"] = str(rel)


# ---------------------------------------------------------------------------
# Span-finding helpers
# ---------------------------------------------------------------------------


def find_entity_spans(
    text: str,
    entity_label: str,
    *,
    case_sensitive: bool = False,
) -> list[dict[str, Any]]:
    """Find character spans where *entity_label* appears in *text*.

    Uses tiered matching:
      1. Exact substring (case-insensitive by default).
      2. Word-boundary regex match.
      3. Longest-token partial match (token ≥ 4 chars, not a stopword).

    Returns a list of ``{start, end, matched_text, match_type}`` dicts.
    """
    if not text or not entity_label:
        return []

    results: list[dict[str, Any]] = []

    # --- Tier 1: exact substring ---
    haystack = text if case_sensitive else text.lower()
    needle = entity_label if case_sensitive else entity_label.lower()

    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        results.append({
            "start": idx,
            "end": idx + len(needle),
            "matched_text": text[idx : idx + len(needle)],
            "match_type": "exact",
        })
        start = idx + 1

    if results:
        return results

    # --- Tier 2: word-boundary regex ---
    try:
        pattern = re.compile(r"\b" + re.escape(entity_label) + r"\b",
                             0 if case_sensitive else re.IGNORECASE)
        for m in pattern.finditer(text):
            results.append({
                "start": m.start(),
                "end": m.end(),
                "matched_text": m.group(),
                "match_type": "word_boundary",
            })
    except re.error:
        pass

    if results:
        return results

    # --- Tier 3: longest-token partial match ---
    tokens = entity_label.split()
    # Pick the longest token that isn't a stopword and is ≥ 4 chars
    candidates = sorted(
        [t for t in tokens if len(t) >= 4 and t.lower() not in _STOPWORDS],
        key=len,
        reverse=True,
    )
    if candidates:
        token = candidates[0]
        pattern = re.compile(re.escape(token), 0 if case_sensitive else re.IGNORECASE)
        for m in pattern.finditer(text):
            results.append({
                "start": m.start(),
                "end": m.end(),
                "matched_text": m.group(),
                "match_type": "partial_token",
            })

    return results


def find_context_span(
    text: str,
    context: str,
) -> dict[str, Any] | None:
    """Find the character span of a *context* quote within *text*.

    Uses tiered matching:
      1. Exact substring.
      2. Case-insensitive exact.
      3. Prefix match (first 40+ chars).

    Returns ``{start, end, matched_text, match_type}`` or ``None``.
    """
    if not text or not context:
        return None

    # --- Tier 1: exact substring ---
    idx = text.find(context)
    if idx != -1:
        return {
            "start": idx,
            "end": idx + len(context),
            "matched_text": context,
            "match_type": "exact",
        }

    # --- Tier 1b: strip trailing punctuation and retry ---
    stripped = context.rstrip(".,;:!?")
    if stripped != context and stripped:
        idx = text.find(stripped)
        if idx != -1:
            return {
                "start": idx,
                "end": idx + len(stripped),
                "matched_text": text[idx : idx + len(stripped)],
                "match_type": "exact",
            }

    # --- Tier 2: case-insensitive ---
    lower_text = text.lower()
    lower_ctx = context.lower()
    idx = lower_text.find(lower_ctx)
    if idx != -1:
        return {
            "start": idx,
            "end": idx + len(context),
            "matched_text": text[idx : idx + len(context)],
            "match_type": "case_insensitive",
        }

    # --- Tier 2b: case-insensitive with stripped punctuation ---
    if stripped != context and stripped:
        idx = lower_text.find(stripped.lower())
        if idx != -1:
            return {
                "start": idx,
                "end": idx + len(stripped),
                "matched_text": text[idx : idx + len(stripped)],
                "match_type": "case_insensitive",
            }

    # --- Tier 3: prefix match (first 40+ chars) ---
    prefix_len = min(len(context), max(40, len(context) // 2))
    prefix = context[:prefix_len]
    idx = lower_text.find(prefix.lower())
    if idx != -1:
        # Extend to end of sentence or paragraph if possible
        end = idx + len(context)
        if end > len(text):
            end = len(text)
        return {
            "start": idx,
            "end": end,
            "matched_text": text[idx:end],
            "match_type": "prefix",
        }

    return None


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
# Graph Validation
# ---------------------------------------------------------------------------

# Taxonomic relations that should not form cycles
TAXONOMIC_RELATIONS = {"is_a", "subclass_of", "instance_of", "part_of"}

# Pairs of relations that are contradictory on the same (source, target) pair
CONTRADICTORY_PAIRS: set[tuple[str, str]] = {
    ("is_a", "has_part"),
    ("supersedes", "superseded_by"),
    ("causes", "caused_by"),
    ("depends_on", "required_by"),
    ("uses", "used_by"),
    ("documents", "documented_by"),
    ("references", "referenced_by"),
    ("contains", "part_of"),
}


@dataclass
class ValidationReport:
    """Result of graph validation checks.

    Attributes:
        errors: Serious issues that indicate data corruption or logical
                inconsistency (e.g., dangling edges, taxonomic cycles).
        warnings: Potential problems worth investigating (e.g., orphan
                  nodes, contradictory edge pairs, zero-confidence items).
        info: Informational observations (e.g., missing embeddings,
              graph structure notes).
    """
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no errors were found."""
        return len(self.errors) == 0

    @property
    def total_issues(self) -> int:
        return len(self.errors) + len(self.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "summary": {
                "error_count": len(self.errors),
                "warning_count": len(self.warnings),
                "info_count": len(self.info),
            },
        }


@dataclass
class DocumentDiff:
    """Result of comparing two document versions at section level.

    Attributes:
        doc_id: The document identifier.
        version_from: The older version number.
        version_to: The newer version number.
        added: Section headings present only in the newer version.
        removed: Section headings present only in the older version.
        modified: Section headings present in both but with different hashes.
        unchanged: Section headings present in both with identical hashes.
    """
    doc_id: str
    version_from: int
    version_to: int
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.modified)

    @property
    def summary(self) -> str:
        parts = []
        if self.added:
            parts.append(f"{len(self.added)} added")
        if self.removed:
            parts.append(f"{len(self.removed)} removed")
        if self.modified:
            parts.append(f"{len(self.modified)} modified")
        if self.unchanged:
            parts.append(f"{len(self.unchanged)} unchanged")
        return ", ".join(parts) if parts else "no sections"

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "version_from": self.version_from,
            "version_to": self.version_to,
            "added": self.added,
            "removed": self.removed,
            "modified": self.modified,
            "unchanged": self.unchanged,
            "has_changes": self.has_changes,
            "summary": self.summary,
        }


@dataclass
class GraphDiff:
    """Result of comparing two graph states.

    Tracks nodes and edges that were added, removed, or modified between
    two ``KnowledgeGraph`` snapshots.  Proposals are tracked separately.

    Attributes:
        nodes_added: List of node dicts present only in the *newer* graph.
        nodes_removed: List of node dicts present only in the *older* graph.
        nodes_modified: List of dicts describing field-level changes for
            nodes present in both graphs but with different data.
        edges_added: List of edge dicts present only in the newer graph.
        edges_removed: List of edge dicts present only in the older graph.
        edges_modified: List of dicts describing field-level changes for
            edges present in both graphs but with different data.
        proposals_added: List of proposal dicts added in the newer graph.
        proposals_changed: List of dicts describing proposal status changes.
    """

    nodes_added: list[dict[str, Any]] = field(default_factory=list)
    nodes_removed: list[dict[str, Any]] = field(default_factory=list)
    nodes_modified: list[dict[str, Any]] = field(default_factory=list)
    edges_added: list[dict[str, Any]] = field(default_factory=list)
    edges_removed: list[dict[str, Any]] = field(default_factory=list)
    edges_modified: list[dict[str, Any]] = field(default_factory=list)
    proposals_added: list[dict[str, Any]] = field(default_factory=list)
    proposals_changed: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(
            self.nodes_added or self.nodes_removed or self.nodes_modified
            or self.edges_added or self.edges_removed or self.edges_modified
            or self.proposals_added or self.proposals_changed
        )

    @property
    def summary(self) -> str:
        parts: list[str] = []
        if self.nodes_added:
            parts.append(f"{len(self.nodes_added)} nodes added")
        if self.nodes_removed:
            parts.append(f"{len(self.nodes_removed)} nodes removed")
        if self.nodes_modified:
            parts.append(f"{len(self.nodes_modified)} nodes modified")
        if self.edges_added:
            parts.append(f"{len(self.edges_added)} edges added")
        if self.edges_removed:
            parts.append(f"{len(self.edges_removed)} edges removed")
        if self.edges_modified:
            parts.append(f"{len(self.edges_modified)} edges modified")
        if self.proposals_added:
            parts.append(f"{len(self.proposals_added)} proposals added")
        if self.proposals_changed:
            parts.append(f"{len(self.proposals_changed)} proposals changed")
        return ", ".join(parts) if parts else "no changes"

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes_added": self.nodes_added,
            "nodes_removed": self.nodes_removed,
            "nodes_modified": self.nodes_modified,
            "edges_added": self.edges_added,
            "edges_removed": self.edges_removed,
            "edges_modified": self.edges_modified,
            "proposals_added": self.proposals_added,
            "proposals_changed": self.proposals_changed,
            "has_changes": self.has_changes,
            "summary": self.summary,
            "counts": {
                "nodes_added": len(self.nodes_added),
                "nodes_removed": len(self.nodes_removed),
                "nodes_modified": len(self.nodes_modified),
                "edges_added": len(self.edges_added),
                "edges_removed": len(self.edges_removed),
                "edges_modified": len(self.edges_modified),
                "proposals_added": len(self.proposals_added),
                "proposals_changed": len(self.proposals_changed),
            },
        }


def compute_section_hashes(
    text: str,
    *,
    min_section_chars: int = 80,
    max_section_chars: int = 6000,
) -> dict[str, str]:
    """Compute SHA-256 hashes for each section in a markdown document.

    Args:
        text: Raw markdown text.
        min_section_chars: Passed through to ``parse_markdown_sections``.
        max_section_chars: Passed through to ``parse_markdown_sections``.

    Returns:
        Dict mapping section heading (or ``"__preamble__"`` for the
        untitled preamble) to the 12-char SHA-256 hash of that
        section's body text.
    """
    sections = KnowledgeGraph.parse_markdown_sections(
        text,
        min_section_chars=min_section_chars,
        max_section_chars=max_section_chars,
    )
    hashes: dict[str, str] = {}
    for section in sections:
        heading = section["heading"] or "__preamble__"
        hashes[heading] = content_hash(section["body"])
    return hashes


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
        self._custom_relations: set[str] = set()
        self._data: dict[str, Any] = self._empty_graph_data()
        self._embeddings: dict[str, list[float]] = {}
        self._embed_meta: dict[str, Any] = {}
        self._proposals: list[RelationProposal] = []
        self._G: nx.MultiDiGraph = nx.MultiDiGraph()
        self._edge_index: set[tuple[str, str, str]] = set()
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

    def register_relation(self, name: str) -> None:
        """Register a custom relation type on this graph instance."""
        self._custom_relations.add(name)

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
        if self._dirty_embeddings or self._embeddings:
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
        section_hashes: dict[str, str] | None = None,
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
                    "section_hashes": entry.get("section_hashes", {}),
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
                    "section_hashes": old_entry.get("section_hashes", {}),
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
            "section_hashes": section_hashes or {},
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
                    "section_hashes": old_entry.get("section_hashes", {}),
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
            "section_hashes": section_hashes or {},
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
                "section_hashes": v.get("section_hashes", {}),
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
            "section_hashes": entry.get("section_hashes", {}),
        })

        return sorted(versions, key=lambda v: v["version"])

    def diff_document_versions(
        self, doc_id: str, v1: int, v2: int,
    ) -> DocumentDiff:
        """Compare two versions of a document at section level.

        Uses ``section_hashes`` stored in the source manifest to identify
        which sections were added, removed, modified, or unchanged between
        two versions.

        Args:
            doc_id: The document identifier.
            v1: The older version number.
            v2: The newer version number.

        Returns:
            A :class:`DocumentDiff` describing changes between the versions.

        Raises:
            KeyError: If the document or either version is not found.
        """
        versions = self.get_source_versions(doc_id)
        if not versions:
            raise KeyError(f"Document '{doc_id}' not found in source manifest")

        v1_entry = None
        v2_entry = None
        for v in versions:
            if v["version"] == v1:
                v1_entry = v
            if v["version"] == v2:
                v2_entry = v

        if v1_entry is None:
            raise KeyError(f"Version {v1} not found for document '{doc_id}'")
        if v2_entry is None:
            raise KeyError(f"Version {v2} not found for document '{doc_id}'")

        hashes_v1: dict[str, str] = v1_entry.get("section_hashes", {})
        hashes_v2: dict[str, str] = v2_entry.get("section_hashes", {})

        keys_v1 = set(hashes_v1.keys())
        keys_v2 = set(hashes_v2.keys())

        added = sorted(keys_v2 - keys_v1)
        removed = sorted(keys_v1 - keys_v2)
        common = keys_v1 & keys_v2
        modified = sorted(h for h in common if hashes_v1[h] != hashes_v2[h])
        unchanged = sorted(h for h in common if hashes_v1[h] == hashes_v2[h])

        return DocumentDiff(
            doc_id=doc_id,
            version_from=v1,
            version_to=v2,
            added=added,
            removed=removed,
            modified=modified,
            unchanged=unchanged,
        )

    def get_document_history(self, doc_id: str) -> list[dict[str, Any]]:
        """Get a rich version timeline for a document.

        Extends :meth:`get_source_versions` with per-version section counts
        and node/edge counts from the graph.

        Args:
            doc_id: The document identifier.

        Returns:
            List of version dicts (oldest first), each augmented with:
              - ``section_count``: number of sections (from section_hashes)
              - ``node_count``: number of nodes created during this ingestion
              - ``edge_count``: number of edges created during this ingestion
              - ``diff``: if not the first version, a :class:`DocumentDiff`
                dict showing changes from the previous version
        """
        versions = self.get_source_versions(doc_id)
        if not versions:
            return []

        result: list[dict[str, Any]] = []
        prev_version_num: int | None = None

        for v in versions:
            entry = dict(v)  # shallow copy
            shashes = v.get("section_hashes", {})
            entry["section_count"] = len(shashes)

            # Count nodes and edges from this ingestion run
            iid = v.get("ingestion_id", "")
            if iid:
                entry["node_count"] = len(self.get_nodes_by_ingestion(iid))
                entry["edge_count"] = len(self.get_edges_by_ingestion(iid))
            else:
                entry["node_count"] = 0
                entry["edge_count"] = 0

            # Compute diff from previous version
            if prev_version_num is not None:
                try:
                    diff = self.diff_document_versions(
                        doc_id, prev_version_num, v["version"],
                    )
                    entry["diff"] = diff.to_dict()
                except KeyError:
                    entry["diff"] = None
            else:
                entry["diff"] = None

            prev_version_num = v["version"]
            result.append(entry)

        return result

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
    # Snapshot & diff
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a deep copy of the current graph state for later diffing.

        The returned dict captures ``_data`` and ``_proposals`` so that
        :meth:`diff` can compare before/after states within a session.
        """
        return {
            "data": deepcopy(self._data),
            "proposals": [p.to_dict() for p in self._proposals],
        }

    @staticmethod
    def _diff_node_fields(
        node_id: str,
        old: dict[str, Any],
        new: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Compare two node dicts and return field-level changes, or None."""
        changes: dict[str, Any] = {}
        all_keys = set(old) | set(new)
        for key in all_keys:
            old_val = old.get(key)
            new_val = new.get(key)
            if old_val != new_val:
                changes[key] = {"old": old_val, "new": new_val}
        if changes:
            return {"node_id": node_id, "label": new.get("label", old.get("label", node_id)), "changes": changes}
        return None

    @staticmethod
    def _edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
        return (edge["source"], edge["target"], edge.get("relation", "related_to"))

    @staticmethod
    def _diff_edge_fields(
        edge_key: tuple[str, str, str],
        old: dict[str, Any],
        new: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Compare two edge dicts and return field-level changes, or None."""
        changes: dict[str, Any] = {}
        all_keys = set(old) | set(new)
        for key in all_keys:
            old_val = old.get(key)
            new_val = new.get(key)
            if old_val != new_val:
                changes[key] = {"old": old_val, "new": new_val}
        if changes:
            return {"source": edge_key[0], "target": edge_key[1], "relation": edge_key[2], "changes": changes}
        return None

    def diff(self, other: "KnowledgeGraph") -> GraphDiff:
        """Compare this graph against *other* and return a :class:`GraphDiff`.

        ``self`` is treated as the **older** state and *other* as the
        **newer** state.  The result describes what changed to go from
        ``self`` to *other*.
        """
        return self._diff_data(self._data, [p.to_dict() for p in self._proposals],
                               other._data, [p.to_dict() for p in other._proposals])

    def diff_from_snapshot(self, snap: dict[str, Any]) -> GraphDiff:
        """Compare a previously captured :meth:`snapshot` against the current state.

        The snapshot is the **older** state; the current graph is the
        **newer** state.
        """
        return self._diff_data(snap["data"], snap["proposals"],
                               self._data, [p.to_dict() for p in self._proposals])

    def diff_from_file(self, path: str | Path) -> GraphDiff:
        """Load a graph from *path* and diff it against the current state.

        The file is the **older** state; the current graph is the
        **newer** state.
        """
        other = KnowledgeGraph(path)
        return other.diff(self)

    @classmethod
    def _diff_data(
        cls,
        old_data: dict[str, Any],
        old_proposals: list[dict[str, Any]],
        new_data: dict[str, Any],
        new_proposals: list[dict[str, Any]],
    ) -> GraphDiff:
        """Core diff logic operating on raw data dicts."""
        result = GraphDiff()

        # --- Nodes ---
        old_nodes = old_data.get("nodes", {})
        new_nodes = new_data.get("nodes", {})
        old_ids = set(old_nodes)
        new_ids = set(new_nodes)

        for nid in sorted(new_ids - old_ids):
            result.nodes_added.append({"node_id": nid, **new_nodes[nid]})
        for nid in sorted(old_ids - new_ids):
            result.nodes_removed.append({"node_id": nid, **old_nodes[nid]})
        for nid in sorted(old_ids & new_ids):
            mod = cls._diff_node_fields(nid, old_nodes[nid], new_nodes[nid])
            if mod:
                result.nodes_modified.append(mod)

        # --- Edges ---
        old_edges_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        for edge in old_data.get("edges", []):
            old_edges_by_key[cls._edge_key(edge)] = edge
        new_edges_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        for edge in new_data.get("edges", []):
            new_edges_by_key[cls._edge_key(edge)] = edge

        old_ekeys = set(old_edges_by_key)
        new_ekeys = set(new_edges_by_key)

        for ek in sorted(new_ekeys - old_ekeys):
            result.edges_added.append(new_edges_by_key[ek])
        for ek in sorted(old_ekeys - new_ekeys):
            result.edges_removed.append(old_edges_by_key[ek])
        for ek in sorted(old_ekeys & new_ekeys):
            mod = cls._diff_edge_fields(ek, old_edges_by_key[ek], new_edges_by_key[ek])
            if mod:
                result.edges_modified.append(mod)

        # --- Proposals ---
        old_prop_names = {p["name"] for p in old_proposals}
        new_prop_names = {p["name"] for p in new_proposals}
        new_prop_map = {p["name"]: p for p in new_proposals}
        old_prop_map = {p["name"]: p for p in old_proposals}

        for name in sorted(new_prop_names - old_prop_names):
            result.proposals_added.append(new_prop_map[name])
        for name in sorted(old_prop_names & new_prop_names):
            old_p = old_prop_map[name]
            new_p = new_prop_map[name]
            if old_p.get("status") != new_p.get("status"):
                result.proposals_changed.append({
                    "name": name,
                    "old_status": old_p.get("status"),
                    "new_status": new_p.get("status"),
                })

        return result

    # ------------------------------------------------------------------
    # Internal: networkx sync
    # ------------------------------------------------------------------

    def _rebuild_networkx(self) -> None:
        """Rebuild the networkx MultiDiGraph and edge index from the raw dict."""
        self._G = nx.MultiDiGraph()
        self._edge_index = set()
        for nid, node in self._data["nodes"].items():
            self._G.add_node(nid, **node)
        for edge in self._data["edges"]:
            rel = edge.get("relation", "related_to")
            self._G.add_edge(
                edge["source"],
                edge["target"],
                key=rel,
                **{k: v for k, v in edge.items() if k not in ("source", "target")},
            )
            self._edge_index.add((edge["source"], edge["target"], rel))

    @property
    def graph(self) -> nx.MultiDiGraph:
        """Direct access to the underlying networkx MultiDiGraph (read-friendly)."""
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

        # Seed description_sources for new nodes with a description
        if "description" in properties and "description_sources" not in properties:
            properties["description_sources"] = [{
                "text": properties["description"],
                "doc_id": source or "unknown",
                "confidence": confidence,
                "updated_at": ts,
            }]

        if node_id in self._data["nodes"] and merge:
            existing = self._data["nodes"][node_id]
            # Handle description merging separately to avoid overwrite
            _desc = properties.pop("description", None)
            _desc_sources = properties.pop("description_sources", None)
            existing["properties"].update(properties)
            if _desc:
                _merge_description(
                    existing["properties"],
                    _desc,
                    doc_id=source or "unknown",
                    confidence=confidence,
                )
            elif _desc_sources:
                # Caller provided pre-built sources (e.g. from merge())
                for ds in _desc_sources:
                    _merge_description(
                        existing["properties"],
                        ds["text"],
                        doc_id=ds.get("doc_id", "unknown"),
                        confidence=ds.get("confidence", confidence),
                    )
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

        # Check for duplicates using the edge index (O(1) lookup)
        if not allow_duplicate and (source, target, relation) in self._edge_index:
            for e in self._data["edges"]:
                if e["source"] == source and e["target"] == target and e["relation"] == relation:
                    # Update existing edge
                    e["properties"] = {**e.get("properties", {}), **(properties or {})}
                    e["confidence"] = max(e.get("confidence", 0), confidence)
                    e["weight"] = weight
                    if self.auto_timestamp:
                        e["updated"] = now_iso()
                    if self._G.has_edge(source, target, key=relation):
                        self._G[source][target][relation].update(e)
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
        self._edge_index.add((source, target, relation))
        self._G.add_edge(source, target, key=relation, **edge)
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
                            if not self._G.has_edge(nid, succ, key=relation):
                                continue
                        next_frontier.add(succ)
                if direction in ("incoming", "both"):
                    for pred in self._G.predecessors(nid):
                        if relation:
                            if not self._G.has_edge(pred, nid, key=relation):
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
        progress_fn: Callable[[dict[str, Any]], None] | None = None,
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
            progress_fn: Optional callback for real-time progress reporting.
                         Events: "embed_start", "embed_batch_done",
                         "embed_done".

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

        total_to_embed = len(candidates)
        logger.info("Preparing to embed %d nodes...", total_to_embed)

        if progress_fn:
            progress_fn({
                "event": "embed_start",
                "total_nodes": total_to_embed,
                "nodes_skipped": stats["nodes_skipped"],
            })

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
        total_texts = len(texts)
        total_batches = (total_texts + batch_size - 1) // batch_size
        for batch_start in range(0, total_texts, batch_size):
            batch = texts[batch_start:batch_start + batch_size]
            batch_texts = [t for _, t in batch]
            batch_ids = [nid for nid, _ in batch]
            batch_num = stats["batches"] + 1

            logger.info(
                "Embedding batch %d/%d (%d nodes)...",
                batch_num, total_batches, len(batch_ids),
            )

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
                    logger.info(
                        "  Embedded node %d/%d: '%s' (dim=%d)",
                        stats["nodes_embedded"], total_texts, nid, len(emb),
                    )

                self._dirty_embeddings = True
                stats["batches"] += 1

                if progress_fn:
                    progress_fn({
                        "event": "embed_batch_done",
                        "batch": stats["batches"],
                        "total_batches": total_batches,
                        "nodes_embedded": stats["nodes_embedded"],
                        "total_nodes": total_to_embed,
                    })

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

        if progress_fn:
            progress_fn({
                "event": "embed_done",
                "nodes_embedded": stats["nodes_embedded"],
                "nodes_skipped": stats["nodes_skipped"],
                "batches": stats["batches"],
                "errors": len(stats["errors"]),
            })

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
    "source_description": "one-sentence description of the source entity",
    "target": "entity name (human-readable)",
    "target_type": "node type",
    "target_description": "one-sentence description of the target entity",
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
  {{"source": "Kalman filter", "source_type": "concept", "source_description": "Recursive algorithm for estimating the state of a linear dynamic system from noisy measurements", "target": "navigation systems", "target_type": "concept", "target_description": "Systems that determine position and guide movement", "relation": "used_in", "is_new_relation": false, "suggested_relation": null, "justification": null, "confidence": 0.92, "context": "The Kalman filter is widely used in navigation systems."}},
  {{"source": "Kalman filter", "source_type": "concept", "source_description": "Recursive algorithm for estimating the state of a linear dynamic system from noisy measurements", "target": "state-space model", "target_type": "concept", "target_description": "Mathematical representation of a system using state variables", "relation": "depends_on", "is_new_relation": false, "suggested_relation": null, "justification": null, "confidence": 0.90, "context": "It requires a state-space model"}},
  {{"source": "Kalman filter", "source_type": "concept", "source_description": "Recursive algorithm for estimating the state of a linear dynamic system from noisy measurements", "target": "Gaussian noise", "target_type": "concept", "target_description": "Statistical noise with a normal probability distribution", "relation": "assumes", "is_new_relation": true, "suggested_relation": "assumes", "justification": "Captures a precondition or assumption dependency not covered by depends_on.", "confidence": 0.85, "context": "produces optimal estimates under Gaussian noise"}}
]

Input: "TensorFlow was developed by Google Brain. It supports GPU acceleration and is commonly compared to PyTorch."
Output:
[
  {{"source": "TensorFlow", "source_type": "tool", "source_description": "Open-source machine learning framework", "target": "Google Brain", "target_type": "organization", "target_description": "AI research team at Google", "relation": "created_by", "is_new_relation": false, "suggested_relation": null, "justification": null, "confidence": 0.95, "context": "TensorFlow was developed by Google Brain."}},
  {{"source": "TensorFlow", "source_type": "tool", "source_description": "Open-source machine learning framework", "target": "GPU acceleration", "target_type": "concept", "target_description": "Using graphics processing units to speed up computation", "relation": "supports", "is_new_relation": false, "suggested_relation": null, "justification": null, "confidence": 0.90, "context": "It supports GPU acceleration"}},
  {{"source": "TensorFlow", "source_type": "tool", "source_description": "Open-source machine learning framework", "target": "PyTorch", "target_type": "tool", "target_description": "Open-source machine learning framework by Meta", "relation": "alternative_to", "is_new_relation": false, "suggested_relation": null, "justification": null, "confidence": 0.75, "context": "commonly compared to PyTorch"}}
]

Input: "Convolutional layers extract spatial features. Pooling reduces dimensionality before the fully connected layer classifies the output."
Output:
[
  {{"source": "convolutional layers", "source_type": "concept", "source_description": "Neural network layers that apply convolution filters to detect patterns", "target": "spatial features", "target_type": "concept", "target_description": "Location-dependent patterns in input data such as edges and textures", "relation": "produces", "is_new_relation": false, "suggested_relation": null, "justification": null, "confidence": 0.90, "context": "Convolutional layers extract spatial features."}},
  {{"source": "pooling", "source_type": "concept", "source_description": "Downsampling operation that reduces spatial dimensions of feature maps", "target": "dimensionality", "target_type": "concept", "target_description": "The number of features or spatial dimensions in a representation", "relation": "reduces", "is_new_relation": true, "suggested_relation": "reduces", "justification": "Captures a quantitative reduction relationship not covered by existing types.", "confidence": 0.88, "context": "Pooling reduces dimensionality"}},
  {{"source": "fully connected layer", "source_type": "concept", "source_description": "Neural network layer where every neuron connects to all neurons in the previous layer", "target": "convolutional layers", "target_type": "concept", "target_description": "Neural network layers that apply convolution filters to detect patterns", "relation": "depends_on", "is_new_relation": false, "suggested_relation": null, "justification": null, "confidence": 0.70, "context": "before the fully connected layer classifies the output"}}
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
        progress_fn: Callable[[dict[str, Any]], None] | None = None,
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
            progress_fn: Optional callback for real-time progress reporting.
                         Called with event dicts containing at least an "event"
                         key. Events: "extraction_start", "extraction_done",
                         "triple_done".

        Returns:
            Stats dict: nodes_added, edges_added, proposals_created, proposals_augmented,
                        triples_processed, errors.
        """
        stats = {
            "triples_processed": 0,
            "nodes_added": 0,
            "nodes_updated": 0,
            "edges_added": 0,
            "edges_updated": 0,
            "proposals_created": 0,
            "proposals_augmented": 0,
            "errors": [],
        }

        prompt = self.build_extraction_prompt(
            text, focus_entities=focus_entities, max_triples=max_triples
        )

        if progress_fn:
            progress_fn({"event": "extraction_start", "doc_id": doc_id})

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

        if progress_fn:
            progress_fn({
                "event": "extraction_done",
                "doc_id": doc_id,
                "triples_returned": len(triples),
            })

        return self.ingest_triples(
            triples,
            text=text,
            doc_id=doc_id,
            low_confidence_threshold=low_confidence_threshold,
            auto_add_doc_node=auto_add_doc_node,
            ingestion_id=ingestion_id,
            content_hash=content_hash,
            progress_fn=progress_fn,
        )

    def ingest_triples(
        self,
        triples: list[dict[str, Any]],
        *,
        text: str,
        doc_id: str,
        low_confidence_threshold: float = 0.3,
        auto_add_doc_node: bool = True,
        ingestion_id: str | None = None,
        content_hash: str | None = None,
        progress_fn: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """
        Process pre-extracted triples into the knowledge graph.

        This is the second half of the ingestion pipeline — it takes a list
        of triple dicts (as produced by an LLM or an MCP orchestrator) and
        adds the corresponding nodes, edges, and relation proposals to the
        graph.  Unlike ``ingest_document``, this method does **not** call an
        LLM; the caller is responsible for producing the triples.

        This enables an "orchestrator-as-extractor" pattern where the calling
        LLM (e.g. Claude Code via MCP) performs entity extraction itself and
        passes the structured triples directly, eliminating the need for a
        second LLM backend.

        Args:
            triples: List of triple dicts matching the extraction schema::

                    [{"source": "...", "source_type": "concept",
                      "source_description": "...",
                      "target": "...", "target_type": "concept",
                      "target_description": "...",
                      "relation": "depends_on",
                      "is_new_relation": false,
                      "suggested_relation": null,
                      "justification": null,
                      "confidence": 0.85,
                      "context": "supporting quote"}, ...]

                    List/tuple and string formats are also accepted and
                    normalized automatically.
            text: The original document text (used for validation and
                  entity span tracking).
            doc_id: Unique identifier for the source document.
            low_confidence_threshold: Confidence assigned to edges with
                novel (proposed) relations.
            auto_add_doc_node: If True, create a 'document' node and link
                extracted entities to it with 'documented_by' edges.
            ingestion_id: Unique identifier for this ingestion run
                (propagated to all created nodes and edges).
            content_hash: Content hash of the source document.
            progress_fn: Optional callback for real-time progress reporting.
                         Called with ``{"event": "triple_done", ...}`` after
                         each triple is processed.

        Returns:
            Stats dict: nodes_added, nodes_updated, edges_added,
            edges_updated, proposals_created, proposals_augmented,
            triples_processed, errors.
        """
        stats: dict[str, Any] = {
            "triples_processed": 0,
            "nodes_added": 0,
            "nodes_updated": 0,
            "edges_added": 0,
            "edges_updated": 0,
            "proposals_created": 0,
            "proposals_augmented": 0,
            "errors": [],
        }

        # Detect off-topic responses: if no triple has any recognized key,
        # the model returned garbage unrelated to the extraction prompt.
        if triples:
            _has_valid = any(
                (isinstance(t, dict) and _EXPECTED_KEYS & t.keys())
                or (isinstance(t, (list, tuple)) and len(t) >= 3)
                for t in triples
            )
            if not _has_valid:
                sample_keys = set()
                for t in triples[:3]:
                    if isinstance(t, dict):
                        sample_keys.update(t.keys())
                stats["errors"].append(
                    f"LLM returned {len(triples)} item(s) with no recognized triple keys "
                    f"(got keys: {sorted(sample_keys)[:10]}). "
                    f"Model may have ignored the extraction prompt."
                )
                logger.warning(
                    "Off-topic response for doc '%s': %d items, keys=%s, first=%s",
                    doc_id, len(triples), sorted(sample_keys)[:10],
                    str(triples[0])[:200] if triples else "?",
                )
                return stats

        # Detect hallucinated responses: if entity names have no lexical
        # overlap with the source text, the model fabricated content.
        # Extract entity names from all recognized key aliases and check
        # whether their words appear in the input text.
        if triples:
            _text_lower = text.lower()
            _text_words = set(re.findall(r"[a-z]{3,}", _text_lower)) - _STOPWORDS
            _grounded = 0
            _checked = 0
            for t in triples:
                if not isinstance(t, dict):
                    continue
                for _kset in (_SRC_KEYS, _TGT_KEYS):
                    for _k in _kset:
                        val = t.get(_k, "")
                        if not isinstance(val, str) or not val.strip():
                            continue
                        _checked += 1
                        # Entity is grounded if any of its non-stopword tokens
                        # (3+ chars) appear in the source text
                        entity_words = set(re.findall(r"[a-z]{3,}", val.lower())) - _STOPWORDS
                        if entity_words & _text_words:
                            _grounded += 1
                        break  # only check first matching key per set

            if _checked >= 4 and _grounded == 0:
                sample_entities = []
                for t in triples[:3]:
                    if isinstance(t, dict):
                        for _k in ("source", "subject", "head", "Concept",
                                   "target", "object", "tail"):
                            if _k in t:
                                sample_entities.append(str(t[_k])[:50])
                stats["errors"].append(
                    f"LLM hallucinated: {_checked} entity mentions checked, "
                    f"0 grounded in source text. "
                    f"Sample entities: {sample_entities[:4]}. "
                    f"Model fabricated content unrelated to the input."
                )
                logger.warning(
                    "Hallucinated response for doc '%s': 0/%d entities grounded, "
                    "samples=%s",
                    doc_id, _checked, sample_entities[:4],
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

        _total_triples = len(triples)
        for _ti, triple in enumerate(triples):
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
                for old_key, new_key in _KEY_ALIASES.items():
                    if old_key in triple and new_key not in triple:
                        triple[new_key] = triple.pop(old_key)

                # Convert glossary-style dicts into triples.
                # Some models return {Concept, Definition, Example} or
                # {term, definition} instead of source/target triples.
                # We convert these into "concept -[defined_in]-> doc" triples
                # with the definition stored as the entity description.
                _concept_key = None
                for _ck in ("Concept", "concept", "Term", "term", "Name", "name"):
                    if _ck in triple and "source" not in triple:
                        _concept_key = _ck
                        break
                if _concept_key is not None:
                    _concept_name = str(triple[_concept_key]).strip()
                    _definition = ""
                    for _dk in ("Definition", "definition", "Description", "description"):
                        if _dk in triple:
                            _definition = str(triple[_dk]).strip()
                            break
                    _example = ""
                    for _ek in ("Example", "example", "Examples", "examples"):
                        if _ek in triple:
                            _example = str(triple[_ek]).strip()
                            break
                    if _concept_name:
                        triple = {
                            "source": _concept_name,
                            "source_type": triple.get("source_type", "concept"),
                            "source_description": _definition,
                            "target": doc_id.split("::")[-1] if "::" in doc_id else doc_id,
                            "target_type": "section" if "::" in doc_id else "document",
                            "relation": "defined_in",
                            "confidence": 0.7,
                            "context": _example or _definition,
                        }
                        logger.debug(
                            "Converted glossary entry from doc '%s': %s → %s",
                            doc_id, _concept_name, triple["target"],
                        )

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
                for nid, label, ntype, desc_key in [
                    (source_id, source_label, triple.get("source_type", "concept"), "source_description"),
                    (target_id, target_label, triple.get("target_type", "concept"), "target_description"),
                ]:
                    desc = triple.get(desc_key, "").strip()
                    if not self.has_node(nid):
                        node_props: dict[str, Any] = {}
                        if desc:
                            node_props["description"] = desc
                            node_props["description_sources"] = [{
                                "text": desc,
                                "doc_id": doc_id,
                                "confidence": conf,
                                "updated_at": now_iso(),
                            }]
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
                    else:
                        # Node already exists — merge description if available
                        if desc:
                            existing = self._data["nodes"].get(nid, {})
                            existing.setdefault("properties", {})
                            _merge_description(
                                existing["properties"], desc, doc_id, conf,
                            )
                            # Sync networkx
                            self._G.nodes[nid].update(existing)
                            self._dirty = True
                        stats["nodes_updated"] += 1

                    # --- Entity span tracking ---
                    spans = find_entity_spans(text, label)
                    if spans:
                        node_data = self._data["nodes"].get(nid)
                        if node_data is not None:
                            mentions: list[dict[str, Any]] = node_data.setdefault(
                                "properties", {},
                            ).setdefault("mentions", [])
                            existing_keys = {
                                (m["doc_id"], m["start"], m["end"])
                                for m in mentions
                            }
                            for span in spans:
                                entry = {
                                    "doc_id": doc_id,
                                    "start": span["start"],
                                    "end": span["end"],
                                    "matched_text": span["matched_text"],
                                    "match_type": span["match_type"],
                                }
                                if (doc_id, span["start"], span["end"]) not in existing_keys:
                                    mentions.append(entry)
                            # Sync networkx
                            self._G.nodes[nid].update(node_data)
                            self._dirty = True

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
                    # --- Edge context span tracking ---
                    ctx_span = find_context_span(text, context)
                    if ctx_span:
                        ctx_span["doc_id"] = doc_id
                        edge_props["context_span"] = ctx_span
                if ingestion_id:
                    edge_props["ingestion_id"] = ingestion_id
                if content_hash:
                    edge_props["content_hash"] = content_hash

                _edge_existed = (source_id, target_id, effective_relation) in self._edge_index
                self.add_edge(
                    source_id,
                    target_id,
                    relation=effective_relation,
                    properties=edge_props,
                    source_tag=f"doc:{doc_id}",
                    confidence=edge_conf,
                    _skip_auto_register=skip_register,
                )
                if _edge_existed:
                    stats["edges_updated"] += 1
                else:
                    stats["edges_added"] += 1

            except Exception as e:
                stats["errors"].append(f"Triple processing error: {e} — {triple}")
                logger.warning("Error processing triple from doc '%s': %s", doc_id, e)

            if progress_fn:
                progress_fn({
                    "event": "triple_done",
                    "index": _ti,
                    "total": _total_triples,
                    "doc_id": doc_id,
                    "nodes_added": stats["nodes_added"],
                    "nodes_updated": stats["nodes_updated"],
                    "edges_added": stats["edges_added"],
                    "edges_updated": stats["edges_updated"],
                })

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

        def _flush_section() -> None:
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
        table_row_re = re.compile(r"^\s*\|.+\|", re.MULTILINE)
        list_re = re.compile(r"^\s*[-*+]\s|\s*\d+\.", re.MULTILINE)

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
        parallel_extractions: int = 1,
        incremental: bool = False,
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
                         key.  Document-level events: "doc_start", "doc_done".
                         Section-level events: "section_start", "section_done",
                         "section_skip".  Extraction-level events (fired per
                         section): "extraction_start", "extraction_done".
                         Triple-level events: "triple_done" (fired for each
                         triple processed, includes section_index/section_total/
                         section_heading for correlation).
            parallel_extractions: Number of parallel LLM extraction threads.
                                  When > 1, LLM calls run concurrently in a
                                  thread pool while graph writes remain serial.
                                  Defaults to 1 (sequential extraction).
            incremental: When True and the document has a previous version
                         with section hashes, skip LLM extraction for sections
                         whose content has not changed.  Structural nodes and
                         edges are always created/updated regardless.  This can
                         dramatically reduce LLM calls when re-ingesting a
                         document where only a few sections changed.  Requires
                         ``preserve_source=True`` so that section hashes are
                         available for comparison.

        Returns:
            Aggregate stats dict with per-section breakdown.
            Includes a ``"diff"`` key with a :class:`GraphDiff` dict
            summarising changes made during this ingestion.
        """
        # Capture pre-ingestion state for post-ingestion diff
        _pre_snapshot = self.snapshot()

        aggregate_stats: dict[str, Any] = {
            "doc_id": doc_id,
            "total_sections": 0,
            "total_triples": 0,
            "total_nodes_added": 0,
            "total_nodes_updated": 0,
            "total_edges_added": 0,
            "total_edges_updated": 0,
            "total_proposals_created": 0,
            "total_proposals_augmented": 0,
            "sections_skipped_incremental": 0,
            "source": None,
            "sections": [],
            "errors": [],
        }

        # Compute per-section hashes for version tracking
        section_hashes = compute_section_hashes(
            text,
            min_section_chars=min_section_chars,
            max_section_chars=max_section_chars,
        )

        # Store source file
        ingestion_id = None
        source_content_hash = None
        if preserve_source:
            source_result = self.store_source(
                text, doc_id, original_path=original_path,
                section_hashes=section_hashes,
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

            # Log section-level diffs on re-ingestion
            if source_result.get("is_update") and progress_fn:
                prev_version = source_result["version"] - 1
                try:
                    diff = self.diff_document_versions(
                        doc_id, prev_version, source_result["version"],
                    )
                    progress_fn({
                        "event": "version_diff",
                        "doc_id": doc_id,
                        "version_from": prev_version,
                        "version_to": source_result["version"],
                        "sections_added": diff.added,
                        "sections_removed": diff.removed,
                        "sections_modified": diff.modified,
                        "sections_unchanged": diff.unchanged,
                        "summary": diff.summary,
                    })
                except KeyError:
                    pass  # no section hashes for previous version

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

        # ── Incremental ingestion: determine unchanged sections ───
        # When incremental=True and we have a previous version with section
        # hashes, build a set of section headings whose content has not
        # changed so we can skip their (expensive) LLM extraction.
        _unchanged_headings: set[str] = set()
        if incremental and preserve_source and source_result is not None:
            prev_version = source_result.get("version", 1) - 1
            if source_result.get("is_update") and prev_version >= 1:
                try:
                    diff = self.diff_document_versions(
                        doc_id, prev_version, source_result["version"],
                    )
                    _unchanged_headings = set(diff.unchanged)
                    if _unchanged_headings:
                        logger.info(
                            "[%s] Incremental mode: %d unchanged sections will "
                            "skip LLM extraction",
                            doc_id, len(_unchanged_headings),
                        )
                        if progress_fn:
                            progress_fn({
                                "event": "incremental_skip_plan",
                                "doc_id": doc_id,
                                "unchanged_sections": sorted(_unchanged_headings),
                                "changed_sections": sorted(
                                    set(diff.added) | set(diff.modified)
                                ),
                                "removed_sections": diff.removed,
                            })
                except KeyError:
                    pass  # no section hashes for previous version

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

        # Notify progress callback of document ingestion start
        if progress_fn:
            progress_fn({
                "event": "doc_start",
                "doc_id": doc_id,
                "total_sections": len(sections),
                "char_count": len(text),
            })

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

        # ── Phase 1: Create structural nodes and edges ──────────────
        # This is fast (no LLM calls) and must be serial because it
        # mutates the graph.  We also collect the sections that need
        # LLM extraction for Phase 2.

        section_ids: list[str] = []
        # Each entry: (index, section_dict, section_slug, section_text, heading)
        extractable: list[tuple[int, dict[str, Any], str, str, str]] = []

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
            context_prefix_len = len(context_prefix)

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
                    "context_prefix_len": context_prefix_len,
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

            # Incremental mode: skip LLM extraction for unchanged sections
            _section_hash_key = heading if heading != f"Section {i + 1}" else "__preamble__"
            if _unchanged_headings and _section_hash_key in _unchanged_headings:
                aggregate_stats["sections_skipped_incremental"] += 1
                skip_info = {
                    "heading": heading,
                    "skipped": True,
                    "reason": "unchanged",
                }
                aggregate_stats["sections"].append(skip_info)
                if progress_fn:
                    progress_fn({
                        "event": "section_skip",
                        "index": i,
                        "total": len(sections),
                        "heading": heading,
                        "reason": "unchanged",
                        "char_count": section["char_count"],
                    })
                logger.info(
                    "[%s] [%d/%d] Skipping '%s' (unchanged)",
                    doc_id, i + 1, len(sections), heading,
                )
                continue

            section_text = context_prefix + body
            extractable.append((i, section, section_slug, section_text, heading))

        # ── Phase 2: LLM extraction + graph writes ────────────────
        # Helper to process one section's extraction result into the
        # graph (must be called from a single thread).
        def _write_section(
            idx: int, section: dict[str, Any], section_slug: str,
            section_text: str, heading: str, section_stats: dict[str, Any],
            elapsed: float,
        ) -> None:
            """Apply extraction results to the graph (serial)."""
            # Link extracted entities to the section node
            if add_structure_nodes and add_structure_edges:
                for nid in list(self._data["nodes"].keys()):
                    node = self._data["nodes"][nid]
                    if node.get("source", "").startswith(f"doc:{doc_id}::"):
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
            aggregate_stats["total_nodes_updated"] += section_stats["nodes_updated"]
            aggregate_stats["total_edges_added"] += section_stats["edges_added"]
            aggregate_stats["total_edges_updated"] += section_stats["edges_updated"]
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

            logger.info(
                "[%s] [%d/%d] Done '%s': %d triples → "
                "%d nodes, %d edges (%.1fs)",
                doc_id, idx + 1, len(sections), heading,
                section_stats["triples_processed"],
                section_stats["nodes_added"],
                section_stats["edges_added"],
                elapsed,
            )

            if progress_fn:
                progress_fn({
                    "event": "section_done",
                    "index": idx,
                    "total": len(sections),
                    "heading": heading,
                    "char_count": section["char_count"],
                    "elapsed_seconds": round(elapsed, 1),
                    "triples": section_stats["triples_processed"],
                    "nodes_added": section_stats["nodes_added"],
                    "nodes_updated": section_stats["nodes_updated"],
                    "edges_added": section_stats["edges_added"],
                    "edges_updated": section_stats["edges_updated"],
                    "proposals_created": section_stats["proposals_created"],
                    "proposals_augmented": section_stats["proposals_augmented"],
                    "errors": section_stats["errors"],
                })

        _parallel = max(1, parallel_extractions)

        if _parallel <= 1:
            # ── Serial extraction (original behaviour) ─────────
            for idx, section, section_slug, section_text, heading in extractable:
                logger.info(
                    "[%s] [%d/%d] Extracting '%s' (~%s chars)...",
                    doc_id, idx + 1, len(sections), heading,
                    f"{section['char_count']:,}",
                )
                if progress_fn:
                    progress_fn({
                        "event": "section_start",
                        "index": idx,
                        "total": len(sections),
                        "heading": heading,
                        "char_count": section["char_count"],
                    })

                t0 = time.monotonic()
                _section_pfn: Callable[[dict[str, Any]], None] | None = None
                if progress_fn:
                    _sec_i = idx
                    _sec_total = len(sections)
                    _sec_heading = heading

                    def _section_pfn(event: dict[str, Any],
                                     _i: int = _sec_i,
                                     _t: int = _sec_total,
                                     _h: str = _sec_heading) -> None:
                        event.setdefault("section_index", _i)
                        event.setdefault("section_total", _t)
                        event.setdefault("section_heading", _h)
                        progress_fn(event)  # type: ignore[misc]

                section_stats = self.ingest_document(
                    section_text,
                    doc_id=f"{doc_id}::{heading}",
                    llm_extract_fn=llm_extract_fn,
                    max_triples=max_triples_per_section,
                    low_confidence_threshold=low_confidence_threshold,
                    auto_add_doc_node=False,
                    ingestion_id=ingestion_id,
                    content_hash=source_content_hash,
                    progress_fn=_section_pfn,
                )
                elapsed = time.monotonic() - t0
                _write_section(idx, section, section_slug, section_text, heading,
                               section_stats, elapsed)

        else:
            # ── Parallel extraction, serial graph writes ───────
            # LLM calls run concurrently in a thread pool.  As each
            # extraction completes, the triples are written to the
            # graph serially (from the main thread) to avoid races.
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _llm_extract(
                idx: int, section: dict[str, Any], section_slug: str,
                section_text: str, heading: str,
            ) -> tuple[int, dict[str, Any], str, str, str, list[dict[str, Any]] | None, str | None, float]:
                """Run LLM extraction for one section (thread-safe)."""
                t0 = time.monotonic()
                prompt = self.build_extraction_prompt(
                    section_text,
                    max_triples=max_triples_per_section,
                )
                if progress_fn:
                    progress_fn({
                        "event": "section_start",
                        "index": idx,
                        "total": len(sections),
                        "heading": heading,
                        "char_count": section["char_count"],
                    })
                try:
                    triples = llm_extract_fn(prompt)
                except Exception as exc:
                    return (idx, section, section_slug, section_text, heading,
                            None, f"LLM extraction failed: {exc}",
                            time.monotonic() - t0)
                if progress_fn:
                    progress_fn({
                        "event": "extraction_done",
                        "doc_id": f"{doc_id}::{heading}",
                        "triples_returned": len(triples) if isinstance(triples, list) else 0,
                    })
                return (idx, section, section_slug, section_text, heading,
                        triples, None, time.monotonic() - t0)

            logger.info(
                "[%s] Parallel extraction with %d threads for %d sections",
                doc_id, _parallel, len(extractable),
            )

            with ThreadPoolExecutor(max_workers=_parallel) as executor:
                futures = {
                    executor.submit(_llm_extract, *args): args
                    for args in extractable
                }

                for future in as_completed(futures):
                    (idx, section, section_slug, section_text, heading,
                     triples, error, elapsed) = future.result()

                    if error:
                        section_stats: dict[str, Any] = {
                            "triples_processed": 0, "nodes_added": 0,
                            "nodes_updated": 0, "edges_added": 0,
                            "edges_updated": 0, "proposals_created": 0,
                            "proposals_augmented": 0, "errors": [error],
                        }
                        logger.error(
                            "[%s] [%d/%d] %s",
                            doc_id, idx + 1, len(sections), error,
                        )
                    elif not isinstance(triples, list):
                        section_stats = {
                            "triples_processed": 0, "nodes_added": 0,
                            "nodes_updated": 0, "edges_added": 0,
                            "edges_updated": 0, "proposals_created": 0,
                            "proposals_augmented": 0,
                            "errors": [f"LLM returned non-list: {type(triples)}"],
                        }
                    else:
                        # Serial graph write — safe, no concurrent mutation
                        section_stats = self.ingest_triples(
                            triples,
                            text=section_text,
                            doc_id=f"{doc_id}::{heading}",
                            low_confidence_threshold=low_confidence_threshold,
                            auto_add_doc_node=False,
                            ingestion_id=ingestion_id,
                            content_hash=source_content_hash,
                        )

                    _write_section(idx, section, section_slug, section_text,
                                   heading, section_stats, elapsed)

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

        _skipped_inc = aggregate_stats["sections_skipped_incremental"]
        if _skipped_inc:
            logger.info(
                "Markdown ingest '%s': %d sections (%d skipped, incremental), "
                "%d triples → %d nodes added (%d updated), "
                "%d edges added (%d updated)",
                doc_id, aggregate_stats["total_sections"], _skipped_inc,
                aggregate_stats["total_triples"],
                aggregate_stats["total_nodes_added"],
                aggregate_stats["total_nodes_updated"],
                aggregate_stats["total_edges_added"],
                aggregate_stats["total_edges_updated"],
            )
        else:
            logger.info(
                "Markdown ingest '%s': %d sections, %d triples → "
                "%d nodes added (%d updated), %d edges added (%d updated)",
                doc_id, aggregate_stats["total_sections"],
                aggregate_stats["total_triples"],
                aggregate_stats["total_nodes_added"],
                aggregate_stats["total_nodes_updated"],
                aggregate_stats["total_edges_added"],
                aggregate_stats["total_edges_updated"],
            )

        # Notify progress callback of document ingestion completion
        if progress_fn:
            progress_fn({
                "event": "doc_done",
                "doc_id": doc_id,
                "total_sections": aggregate_stats["total_sections"],
                "sections_skipped_incremental": _skipped_inc,
                "total_triples": aggregate_stats["total_triples"],
                "total_nodes_added": aggregate_stats["total_nodes_added"],
                "total_nodes_updated": aggregate_stats["total_nodes_updated"],
                "total_edges_added": aggregate_stats["total_edges_added"],
                "total_edges_updated": aggregate_stats["total_edges_updated"],
                "total_proposals_created": aggregate_stats["total_proposals_created"],
                "total_proposals_augmented": aggregate_stats["total_proposals_augmented"],
            })

        # Compute diff from pre-ingestion snapshot
        ingestion_diff = self.diff_from_snapshot(_pre_snapshot)
        aggregate_stats["diff"] = ingestion_diff.to_dict()

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
            prefer: On conflict, prefer ``'self'`` or ``'other'`` node data.
                    When ``'other'``, conflicting nodes are smart-merged:
                    confidence is maximised, descriptions are combined via
                    ``_merge_description``, and timestamps keep the earliest
                    ``created`` and latest ``updated``.

        Returns:
            Stats dict with merge counts.
        """
        stats: dict[str, int] = {
            "nodes_added": 0,
            "nodes_updated": 0,
            "edges_added": 0,
            "edges_updated": 0,
            "sources_added": 0,
            "sources_skipped": 0,
            "proposals_added": 0,
            "proposals_updated": 0,
        }

        # -- Nodes --------------------------------------------------------
        for nid, node in other._data["nodes"].items():
            if nid in self._data["nodes"]:
                if prefer == "other":
                    existing = self._data["nodes"][nid]
                    incoming = deepcopy(node)
                    # Smart merge: combine properties, keep best metadata
                    merged_props = {**existing.get("properties", {})}
                    inc_props = incoming.get("properties", {})
                    # Handle description merging via description_sources
                    inc_desc_sources = inc_props.pop("description_sources", [])
                    inc_desc = inc_props.pop("description", None)
                    # Pop description from merged too — it will be rebuilt
                    merged_props.pop("description", None)
                    merged_props.update(inc_props)
                    # Merge description sources
                    if inc_desc_sources:
                        for ds in inc_desc_sources:
                            _merge_description(
                                merged_props,
                                ds["text"],
                                doc_id=ds.get("doc_id", "unknown"),
                                confidence=ds.get("confidence", 1.0),
                            )
                    elif inc_desc:
                        _merge_description(
                            merged_props,
                            inc_desc,
                            doc_id=incoming.get("source", "unknown"),
                            confidence=incoming.get("confidence", 1.0),
                        )
                    existing["properties"] = merged_props
                    existing["confidence"] = max(
                        existing.get("confidence", 0),
                        incoming.get("confidence", 0),
                    )
                    # Keep earliest created, latest updated
                    if incoming.get("created") and existing.get("created"):
                        existing["created"] = min(existing["created"], incoming["created"])
                    if incoming.get("updated") and existing.get("updated"):
                        existing["updated"] = max(existing["updated"], incoming["updated"])
                    elif incoming.get("updated"):
                        existing["updated"] = incoming["updated"]
                    # Take label/type from higher-confidence source
                    if incoming.get("confidence", 0) >= existing.get("confidence", 0):
                        existing["label"] = incoming.get("label", existing.get("label"))
                        existing["type"] = incoming.get("type", existing.get("type"))
                stats["nodes_updated"] += 1
            else:
                self._data["nodes"][nid] = deepcopy(node)
                stats["nodes_added"] += 1

        # -- Edges --------------------------------------------------------
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
            else:
                # Smart-merge duplicate edges: max confidence, merge properties
                for e in self._data["edges"]:
                    if (e["source"], e["target"], e["relation"]) == key:
                        e["properties"] = {
                            **e.get("properties", {}),
                            **edge.get("properties", {}),
                        }
                        e["confidence"] = max(
                            e.get("confidence", 0),
                            edge.get("confidence", 0),
                        )
                        if edge.get("created") and e.get("created"):
                            e["created"] = min(e["created"], edge["created"])
                        if edge.get("updated") and e.get("updated"):
                            e["updated"] = max(e["updated"], edge["updated"])
                        elif edge.get("updated"):
                            e["updated"] = edge["updated"]
                        stats["edges_updated"] += 1
                        break

        # -- Metadata: custom relations + node types ----------------------
        my_customs = set(self._data["meta"].get("custom_relations", []))
        other_customs = set(other._data["meta"].get("custom_relations", []))
        self._data["meta"]["custom_relations"] = sorted(my_customs | other_customs)

        my_types = set(self._data["meta"].get("node_types", []))
        other_types = set(other._data["meta"].get("node_types", []))
        self._data["meta"]["node_types"] = sorted(my_types | other_types)

        # -- Source documents ---------------------------------------------
        my_sources = self._data["meta"].setdefault("sources", {})
        for doc_slug, entry in other._data["meta"].get("sources", {}).items():
            if doc_slug not in my_sources:
                my_sources[doc_slug] = deepcopy(entry)
                # Copy the actual source file
                _copy_source_file(other, self, my_sources[doc_slug])
                stats["sources_added"] += 1
            elif my_sources[doc_slug].get("content_hash") == entry.get("content_hash"):
                stats["sources_skipped"] += 1
            elif prefer == "other":
                my_sources[doc_slug] = deepcopy(entry)
                _copy_source_file(other, self, my_sources[doc_slug])
                stats["sources_added"] += 1
            else:
                stats["sources_skipped"] += 1

        # -- Embeddings ---------------------------------------------------
        _emb_changed = False
        # Warn about incompatible embedding models
        if (other._embed_meta and self._embed_meta
                and other._embed_meta.get("model")
                and self._embed_meta.get("model")
                and other._embed_meta["model"] != self._embed_meta["model"]):
            logger.warning(
                "Merging graphs with different embedding models: %s vs %s",
                self._embed_meta.get("model"), other._embed_meta.get("model"),
            )
        elif other._embed_meta and not self._embed_meta:
            self._embed_meta = deepcopy(other._embed_meta)

        for nid, emb in other._embeddings.items():
            if nid not in self._embeddings or prefer == "other":
                self._embeddings[nid] = emb
                _emb_changed = True
        if _emb_changed:
            self._dirty_embeddings = True

        # -- Proposals ----------------------------------------------------
        my_proposal_names = {p.name for p in self._proposals}
        for op in other._proposals:
            if op.name not in my_proposal_names:
                self._proposals.append(deepcopy(op))
                stats["proposals_added"] += 1
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
                        # Status resolution: ACCEPTED > PENDING > REJECTED
                        _status_order = {
                            ProposalStatus.REJECTED.value: 0,
                            ProposalStatus.PENDING.value: 1,
                            ProposalStatus.ACCEPTED.value: 2,
                        }
                        mp.status = max(
                            mp.status, op.status,
                            key=lambda s: _status_order.get(s, 1),
                        )
                        stats["proposals_updated"] += 1
                        break

        self._rebuild_networkx()
        self._dirty = True
        return stats

    @classmethod
    def merge_graphs(
        cls,
        sources: list["KnowledgeGraph | str | Path"],
        output_path: str | Path,
        *,
        prefer: str = "latest",
        description: str = "",
    ) -> "KnowledgeGraph":
        """Create a new KnowledgeGraph by merging two or more existing graphs.

        The source graphs are **not** modified.  A fresh graph is created at
        *output_path* and each source is merged into it sequentially.

        Args:
            sources: KnowledgeGraph instances or paths to graph JSON files.
                     At least two sources are required.
            output_path: File path for the new combined graph.
            prefer: Conflict resolution strategy:
                ``'latest'`` — node/edge with the most recent ``updated``
                timestamp wins (default).
                ``'first'`` — first graph's data wins on conflicts.
                ``'last'``  — last graph's data wins on conflicts.
            description: Optional description for the merged graph.

        Returns:
            The newly created (and saved) KnowledgeGraph instance.
        """
        if len(sources) < 2:
            raise ValueError("merge_graphs requires at least 2 source graphs")

        # Resolve sources: load from path if needed
        loaded: list[KnowledgeGraph] = []
        names: list[str] = []
        for src in sources:
            if isinstance(src, (str, Path)):
                kg = cls(src)
                loaded.append(kg)
                names.append(str(Path(src).stem))
            else:
                loaded.append(src)
                names.append(str(src.graph_path.stem))

        output = cls(output_path)
        if not description:
            description = f"Merged from: {', '.join(names)}"
        output._data["meta"]["description"] = description

        # For 'latest' strategy we sort each merge so nodes with newer
        # timestamps overwrite older ones.  We merge all sources with
        # prefer='other' so that the later source always updates; for
        # 'latest' we sort ascending by newest updated timestamp so the
        # newest graph is merged last (and thus wins).
        if prefer == "latest":
            def _newest_ts(kg: KnowledgeGraph) -> str:
                return kg._data["meta"].get("updated", "")
            ordered = sorted(loaded, key=_newest_ts)
            merge_prefer = "other"
        elif prefer == "first":
            ordered = loaded
            # First graph populates empty output with prefer='other',
            # subsequent graphs use prefer='self' so first data wins.
            merge_prefer = "self"
        else:  # "last"
            ordered = loaded
            merge_prefer = "other"

        all_stats: list[dict[str, int]] = []
        for i, src_kg in enumerate(ordered):
            # For 'first' strategy: first merge uses 'other' to populate
            # the empty output, subsequent merges use 'self' to keep first data
            if prefer == "first" and i == 0:
                st = output.merge(src_kg, prefer="other")
            else:
                st = output.merge(src_kg, prefer=merge_prefer)
            all_stats.append(st)

        # Aggregate stats
        agg: dict[str, int] = {}
        for st in all_stats:
            for k, v in st.items():
                agg[k] = agg.get(k, 0) + v
        agg["graphs_merged"] = len(loaded)

        logger.info(
            "Merged %d graphs → %s  (nodes_added=%d, nodes_updated=%d, "
            "edges_added=%d, edges_updated=%d)",
            len(loaded), output.graph_path,
            agg.get("nodes_added", 0), agg.get("nodes_updated", 0),
            agg.get("edges_added", 0), agg.get("edges_updated", 0),
        )

        output.save_all()
        return output

    # ------------------------------------------------------------------
    # Document subgraph extract / import (transplant)
    # ------------------------------------------------------------------

    def extract_document_subgraph(self, doc_id: str) -> dict[str, Any]:
        """Extract a document and all its associated nodes, edges, source text,
        and embeddings into a portable dict that can be imported into another
        graph via :meth:`import_document_subgraph`.

        The subgraph includes:
        - All nodes whose ``source`` field references this document
          (``doc:<doc_id>`` or ``doc:<doc_id>::<section>``).
        - All edges where **both** endpoints are in the extracted node set,
          or whose ``source_tag`` references this document.
        - The source manifest entry and the raw source text (if stored).
        - Embeddings for extracted nodes.
        - Relation proposals that reference this document.

        Args:
            doc_id: The document identifier (will be slugified).

        Returns:
            A dict with keys ``doc_id``, ``nodes``, ``edges``, ``source_info``,
            ``source_text``, ``embeddings``, ``proposals``,
            ``origin_graph`` and ``custom_relations``.

        Raises:
            KeyError: If the document is not found in the graph sources.
        """
        doc_slug = slugify(doc_id)
        manifest = self._data["meta"].get("sources", {})
        if doc_slug not in manifest:
            raise KeyError(f"Document '{doc_id}' not found in graph sources")

        # --- Collect nodes attributed to this document --------------------
        node_ids: set[str] = set()
        nodes: dict[str, dict] = {}
        for nid, node in self._data["nodes"].items():
            src = node.get("source", "")
            # Match  doc:<slug>  or  doc:<slug>::<section>
            if src.startswith("doc:"):
                src_slug = slugify(src.split("::")[0].removeprefix("doc:"))
                if src_slug == doc_slug:
                    node_ids.add(nid)
                    nodes[nid] = deepcopy(node)

        # --- Collect edges ------------------------------------------------
        edges: list[dict] = []
        for edge in self._data["edges"]:
            # Include edge if both endpoints are in the node set
            in_subgraph = (edge["source"] in node_ids and
                           edge["target"] in node_ids)
            # Also include if source_tag references this doc (even if one
            # endpoint is outside the extracted set — we still carry the edge
            # so the relationship is preserved on import)
            tag = edge.get("source_tag", "")
            tag_match = False
            if tag.startswith("doc:"):
                tag_slug = slugify(tag.split("::")[0].removeprefix("doc:"))
                tag_match = tag_slug == doc_slug
            if in_subgraph or tag_match:
                edges.append(deepcopy(edge))
                # Ensure referenced endpoints are in node_ids so they get
                # included if they exist (cross-doc shared nodes)
                for endpoint in ("source", "target"):
                    eid = edge[endpoint]
                    if eid not in nodes and eid in self._data["nodes"]:
                        nodes[eid] = deepcopy(self._data["nodes"][eid])
                        node_ids.add(eid)

        # --- Source text and manifest entry --------------------------------
        source_info = deepcopy(manifest[doc_slug])
        source_text = self.get_source_text(doc_slug)

        # --- Embeddings for extracted nodes --------------------------------
        embeddings: dict[str, list[float]] = {}
        for nid in node_ids:
            if nid in self._embeddings:
                embeddings[nid] = list(self._embeddings[nid])

        # --- Relation proposals referencing this doc -----------------------
        proposals: list[dict] = []
        for p in self._proposals:
            if doc_slug in p.source_docs:
                proposals.append({
                    "name": p.name,
                    "justification": p.justification,
                    "examples": deepcopy(p.examples),
                    "source_docs": list(p.source_docs),
                    "confidence": p.confidence,
                    "status": p.status if isinstance(p.status, str) else p.status.value,
                    "proposed_at": p.proposed_at,
                    "reviewed_at": p.reviewed_at,
                    "review_note": p.review_note,
                })

        # Custom relations used by extracted edges
        custom_rels = set(self._data["meta"].get("custom_relations", []))
        used_rels = {e["relation"] for e in edges}
        relevant_customs = sorted(custom_rels & used_rels)

        origin_graph = str(self.graph_path.stem)

        logger.info(
            "Extracted subgraph for doc '%s': %d nodes, %d edges, "
            "%d embeddings, %d proposals",
            doc_slug, len(nodes), len(edges), len(embeddings), len(proposals),
        )

        return {
            "doc_id": doc_slug,
            "nodes": nodes,
            "edges": edges,
            "source_info": source_info,
            "source_text": source_text,
            "embeddings": embeddings,
            "proposals": proposals,
            "custom_relations": relevant_customs,
            "origin_graph": origin_graph,
        }

    def import_document_subgraph(
        self,
        subgraph: dict[str, Any],
        *,
        overwrite: bool = False,
    ) -> dict[str, int]:
        """Import a document subgraph previously extracted with
        :meth:`extract_document_subgraph` into this graph.

        Nodes are smart-merged (descriptions combined, confidence maximised)
        when they already exist.  Edges are deduplicated by
        ``(source, target, relation)``.

        Args:
            subgraph: The dict returned by ``extract_document_subgraph()``.
            overwrite: If *True*, overwrite the source entry and text even if
                the document already exists in this graph.

        Returns:
            Stats dict with counts of nodes/edges/sources added or updated.
        """
        doc_slug = subgraph["doc_id"]
        stats: dict[str, int] = {
            "nodes_added": 0,
            "nodes_updated": 0,
            "edges_added": 0,
            "edges_updated": 0,
            "embeddings_added": 0,
            "source_stored": 0,
            "proposals_added": 0,
        }

        # --- Import nodes (smart-merge) -----------------------------------
        for nid, node in subgraph["nodes"].items():
            if nid in self._data["nodes"]:
                existing = self._data["nodes"][nid]
                incoming = deepcopy(node)
                merged_props = {**existing.get("properties", {})}
                inc_props = incoming.get("properties", {})
                inc_desc_sources = inc_props.pop("description_sources", [])
                inc_desc = inc_props.pop("description", None)
                merged_props.pop("description", None)
                merged_props.update(inc_props)
                if inc_desc_sources:
                    for ds in inc_desc_sources:
                        _merge_description(
                            merged_props,
                            ds["text"],
                            doc_id=ds.get("doc_id", "unknown"),
                            confidence=ds.get("confidence", 1.0),
                        )
                elif inc_desc:
                    _merge_description(
                        merged_props,
                        inc_desc,
                        doc_id=incoming.get("source", "unknown"),
                        confidence=incoming.get("confidence", 1.0),
                    )
                existing["properties"] = merged_props
                existing["confidence"] = max(
                    existing.get("confidence", 0),
                    incoming.get("confidence", 0),
                )
                if incoming.get("created") and existing.get("created"):
                    existing["created"] = min(existing["created"], incoming["created"])
                if incoming.get("updated") and existing.get("updated"):
                    existing["updated"] = max(existing["updated"], incoming["updated"])
                elif incoming.get("updated"):
                    existing["updated"] = incoming["updated"]
                if incoming.get("confidence", 0) >= existing.get("confidence", 0):
                    existing["label"] = incoming.get("label", existing.get("label"))
                    existing["type"] = incoming.get("type", existing.get("type"))
                stats["nodes_updated"] += 1
            else:
                self._data["nodes"][nid] = deepcopy(node)
                stats["nodes_added"] += 1

        # --- Import edges (dedup) -----------------------------------------
        existing_edge_keys = {
            (e["source"], e["target"], e["relation"])
            for e in self._data["edges"]
        }
        for edge in subgraph["edges"]:
            key = (edge["source"], edge["target"], edge["relation"])
            if key not in existing_edge_keys:
                self._data["edges"].append(deepcopy(edge))
                existing_edge_keys.add(key)
                stats["edges_added"] += 1
            else:
                for e in self._data["edges"]:
                    if (e["source"], e["target"], e["relation"]) == key:
                        e["properties"] = {
                            **e.get("properties", {}),
                            **edge.get("properties", {}),
                        }
                        e["confidence"] = max(
                            e.get("confidence", 0),
                            edge.get("confidence", 0),
                        )
                        stats["edges_updated"] += 1
                        break

        # --- Store source text --------------------------------------------
        manifest = self._data["meta"].setdefault("sources", {})
        if subgraph.get("source_text") and (doc_slug not in manifest or overwrite):
            result = self.store_source(subgraph["source_text"], doc_slug)
            if not result.get("is_duplicate"):
                stats["source_stored"] = 1
            # Record transplant provenance
            entry = manifest.get(doc_slug, {})
            transplant_history = entry.get("transplanted_from", [])
            if subgraph.get("origin_graph"):
                transplant_history.append({
                    "graph": subgraph["origin_graph"],
                    "transplanted_at": now_iso(),
                })
            entry["transplanted_from"] = transplant_history
            manifest[doc_slug] = entry

        # --- Import embeddings --------------------------------------------
        for nid, emb in subgraph.get("embeddings", {}).items():
            if nid not in self._embeddings:
                self._embeddings[nid] = emb
                self._dirty_embeddings = True
                stats["embeddings_added"] += 1

        # --- Import custom relations --------------------------------------
        my_customs = set(self._data["meta"].get("custom_relations", []))
        for rel in subgraph.get("custom_relations", []):
            my_customs.add(rel)
        self._data["meta"]["custom_relations"] = sorted(my_customs)

        # --- Import relation proposals ------------------------------------
        my_proposal_names = {p.name for p in self._proposals}
        for pd in subgraph.get("proposals", []):
            if pd["name"] not in my_proposal_names:
                self._proposals.append(RelationProposal(
                    name=pd["name"],
                    justification=pd.get("justification", ""),
                    examples=pd.get("examples", []),
                    source_docs=pd.get("source_docs", []),
                    confidence=pd.get("confidence", 0.5),
                    status=pd.get("status", ProposalStatus.PENDING.value),
                    proposed_at=pd.get("proposed_at", now_iso()),
                    reviewed_at=pd.get("reviewed_at"),
                    review_note=pd.get("review_note", ""),
                ))
                stats["proposals_added"] += 1

        self._rebuild_networkx()
        self._dirty = True

        logger.info(
            "Imported subgraph for doc '%s': "
            "nodes +%d ~%d, edges +%d ~%d, embeddings +%d, source=%d",
            doc_slug,
            stats["nodes_added"], stats["nodes_updated"],
            stats["edges_added"], stats["edges_updated"],
            stats["embeddings_added"], stats["source_stored"],
        )

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
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
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

    def analytics(self) -> dict[str, Any]:
        """Compute comprehensive quality and structural analytics.

        Returns a dict with sections:
          - **confidence_distribution**: histogram of node and edge
            confidences in 10 buckets from [0.0, 0.1) to [0.9, 1.0].
          - **relation_stats**: per-relation counts, mean/min/max confidence.
          - **node_type_stats**: per-type counts, mean confidence, embedding
            coverage.
          - **hub_nodes**: top-10 nodes by total degree.
          - **orphan_nodes**: nodes with degree 0 (excluding document/section).
          - **source_coverage**: per-source-document node and edge counts.
          - **embedding_coverage**: fraction of embeddable nodes that have
            embeddings.
          - **component_sizes**: list of weakly-connected-component sizes.
          - **quality_score**: composite 0-100 health metric.
        """
        nodes = self._data["nodes"]
        edges = self._data["edges"]

        # -- Confidence distributions ------------------------------------
        node_buckets = [0] * 10
        edge_buckets = [0] * 10
        node_confs: list[float] = []
        edge_confs: list[float] = []

        for node in nodes.values():
            c = node.get("confidence", 1.0)
            node_confs.append(c)
            idx = min(int(c * 10), 9)
            node_buckets[idx] += 1

        for edge in edges:
            c = edge.get("confidence", 1.0)
            edge_confs.append(c)
            idx = min(int(c * 10), 9)
            edge_buckets[idx] += 1

        bucket_labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]

        # -- Relation stats -----------------------------------------------
        rel_data: dict[str, list[float]] = defaultdict(list)
        for edge in edges:
            rel = edge.get("relation", "unknown")
            rel_data[rel].append(edge.get("confidence", 1.0))

        relation_stats = {}
        for rel, confs in sorted(rel_data.items()):
            relation_stats[rel] = {
                "count": len(confs),
                "mean_confidence": sum(confs) / len(confs),
                "min_confidence": min(confs),
                "max_confidence": max(confs),
            }

        # -- Node type stats -----------------------------------------------
        type_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "confidences": [], "with_embedding": 0}
        )
        for nid, node in nodes.items():
            ntype = node.get("type", "unknown")
            entry = type_data[ntype]
            entry["count"] += 1
            entry["confidences"].append(node.get("confidence", 1.0))
            if nid in self._embeddings:
                entry["with_embedding"] += 1

        node_type_stats = {}
        for ntype, entry in sorted(type_data.items()):
            confs = entry["confidences"]
            node_type_stats[ntype] = {
                "count": entry["count"],
                "mean_confidence": sum(confs) / len(confs) if confs else 0,
                "embedding_count": entry["with_embedding"],
                "embedding_pct": (
                    entry["with_embedding"] / entry["count"] * 100
                    if entry["count"] > 0 else 0
                ),
            }

        # -- Hub nodes (top 10 by degree) ----------------------------------
        if self._G.number_of_nodes() > 0:
            degree_list = sorted(
                self._G.degree(), key=lambda x: x[1], reverse=True
            )[:10]
            hub_nodes = [
                {
                    "node_id": nid,
                    "label": nodes.get(nid, {}).get("label", nid),
                    "type": nodes.get(nid, {}).get("type", "unknown"),
                    "degree": deg,
                    "in_degree": self._G.in_degree(nid),
                    "out_degree": self._G.out_degree(nid),
                }
                for nid, deg in degree_list
            ]
        else:
            hub_nodes = []

        # -- Orphan nodes --------------------------------------------------
        structural_types = {"document", "section"}
        orphan_nodes = [
            {
                "node_id": nid,
                "label": nodes[nid].get("label", nid),
                "type": nodes[nid].get("type", "unknown"),
            }
            for nid in nodes
            if self._G.degree(nid) == 0
            and nodes[nid].get("type") not in structural_types
        ]

        # -- Source coverage ------------------------------------------------
        source_node_counts: dict[str, int] = defaultdict(int)
        source_edge_counts: dict[str, int] = defaultdict(int)
        for node in nodes.values():
            src = node.get("source", "")
            if src.startswith("doc:"):
                doc_key = src.split("::")[0].removeprefix("doc:")
                source_node_counts[doc_key] += 1
        for edge in edges:
            src_tag = edge.get("source_tag", "")
            if src_tag.startswith("doc:"):
                doc_key = src_tag.split("::")[0].removeprefix("doc:")
                source_edge_counts[doc_key] += 1

        manifest = self._data["meta"].get("sources", {})
        source_coverage = []
        for doc_id, info in sorted(manifest.items()):
            source_coverage.append({
                "doc_id": doc_id,
                "node_count": source_node_counts.get(doc_id, 0),
                "edge_count": source_edge_counts.get(doc_id, 0),
                "char_count": info.get("char_count", 0),
                "version": info.get("version", 1),
            })

        # -- Embedding coverage ---------------------------------------------
        embeddable = {
            nid for nid, node in nodes.items()
            if node.get("type") not in structural_types
        }
        embedded = embeddable & set(self._embeddings.keys())
        embed_pct = len(embedded) / len(embeddable) * 100 if embeddable else 0

        # -- Component sizes ------------------------------------------------
        comp_sizes = sorted(
            [len(c) for c in nx.weakly_connected_components(self._G)],
            reverse=True,
        )

        # -- Quality score (0-100) ------------------------------------------
        # Composite metric:
        #   40% embedding coverage
        #   30% average confidence (nodes + edges)
        #   20% connectivity (1 - orphan_ratio)
        #   10% source coverage (nodes that trace back to a document)
        avg_conf = 0.0
        if node_confs or edge_confs:
            all_confs = node_confs + edge_confs
            avg_conf = sum(all_confs) / len(all_confs)

        orphan_ratio = len(orphan_nodes) / len(nodes) if nodes else 0
        sourced_nodes = sum(
            1 for n in nodes.values() if n.get("source", "").startswith("doc:")
        )
        source_ratio = sourced_nodes / len(nodes) if nodes else 0

        quality_score = round(
            (embed_pct / 100) * 40
            + avg_conf * 30
            + (1 - orphan_ratio) * 20
            + source_ratio * 10,
            1,
        )

        return {
            "confidence_distribution": {
                "buckets": bucket_labels,
                "node_counts": node_buckets,
                "edge_counts": edge_buckets,
                "node_mean": sum(node_confs) / len(node_confs) if node_confs else 0,
                "edge_mean": sum(edge_confs) / len(edge_confs) if edge_confs else 0,
            },
            "relation_stats": relation_stats,
            "node_type_stats": node_type_stats,
            "hub_nodes": hub_nodes,
            "orphan_nodes": orphan_nodes,
            "source_coverage": source_coverage,
            "embedding_coverage": {
                "embeddable": len(embeddable),
                "embedded": len(embedded),
                "pct": round(embed_pct, 1),
            },
            "component_sizes": comp_sizes,
            "quality_score": quality_score,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> "ValidationReport":
        """Run consistency checks on the graph. Read-only — never mutates.

        Checks performed:
          - **Dangling edges** (error): edges referencing non-existent nodes.
          - **Taxonomic cycles** (error): cycles in is_a / subclass_of /
            instance_of / part_of subgraph.
          - **Contradictory edges** (warning): pairs like A is_a B and
            B is_a A, or A supersedes B and B supersedes A.
          - **Orphan nodes** (warning): nodes with no edges (degree 0),
            excluding document nodes.
          - **Zero-confidence items** (warning): nodes or edges with
            confidence == 0.
          - **Missing embeddings** (info): entity/concept nodes that
            lack an embedding vector.
        """
        report = ValidationReport()
        node_ids = set(self._data["nodes"].keys())

        # -- Dangling edges --------------------------------------------------
        for i, edge in enumerate(self._data["edges"]):
            src, tgt = edge.get("source"), edge.get("target")
            if src not in node_ids:
                report.errors.append(
                    f"Edge [{i}] references non-existent source node '{src}' "
                    f"(relation: {edge.get('relation')}, target: {tgt})"
                )
            if tgt not in node_ids:
                report.errors.append(
                    f"Edge [{i}] references non-existent target node '{tgt}' "
                    f"(relation: {edge.get('relation')}, source: {src})"
                )

        # -- Taxonomic cycles ------------------------------------------------
        tax_graph = nx.DiGraph()
        for edge in self._data["edges"]:
            rel = edge.get("relation", "")
            if rel in TAXONOMIC_RELATIONS:
                src, tgt = edge["source"], edge["target"]
                if src in node_ids and tgt in node_ids:
                    tax_graph.add_edge(src, tgt, relation=rel)
        try:
            cycles = list(nx.simple_cycles(tax_graph))
            for cycle in cycles:
                path_str = " -> ".join(cycle + [cycle[0]])
                report.errors.append(
                    f"Taxonomic cycle detected: {path_str}"
                )
        except nx.NetworkXError:
            pass  # empty graph

        # -- Contradictory edges ---------------------------------------------
        # Build a lookup: (src, tgt) -> set of relations
        pair_rels: dict[tuple[str, str], set[str]] = defaultdict(set)
        for edge in self._data["edges"]:
            src, tgt = edge.get("source", ""), edge.get("target", "")
            rel = edge.get("relation", "")
            pair_rels[(src, tgt)].add(rel)

        # Check for same-direction contradictions
        for (src, tgt), rels in pair_rels.items():
            for rel_a, rel_b in CONTRADICTORY_PAIRS:
                if rel_a in rels and rel_b in rels:
                    report.warnings.append(
                        f"Contradictory edges on ({src}, {tgt}): "
                        f"'{rel_a}' and '{rel_b}' both present"
                    )

        # Check for reflexive contradictions (A rel B and B rel A for
        # relations that form contradictory inverse pairs)
        checked_pairs: set[tuple[str, str]] = set()
        for (src, tgt), rels in pair_rels.items():
            if (tgt, src) in checked_pairs:
                continue
            reverse_rels = pair_rels.get((tgt, src), set())
            if not reverse_rels:
                continue
            for rel in rels:
                # Self-contradictory: A rel B and B rel A for hierarchical rels
                if rel in TAXONOMIC_RELATIONS and rel in reverse_rels:
                    report.warnings.append(
                        f"Reflexive contradiction: '{src}' {rel} '{tgt}' "
                        f"and '{tgt}' {rel} '{src}'"
                    )
                # Cross-contradictory: A supersedes B and B supersedes A
                for rel_a, rel_b in CONTRADICTORY_PAIRS:
                    if rel == rel_a and rel_b in reverse_rels:
                        report.warnings.append(
                            f"Contradictory pair: '{src}' {rel_a} '{tgt}' "
                            f"and '{tgt}' {rel_b} '{src}'"
                        )
            checked_pairs.add((src, tgt))

        # -- Orphan nodes ----------------------------------------------------
        structural_types = {"document", "section"}
        orphans = []
        for nid in node_ids:
            if self._G.degree(nid) == 0:
                ntype = self._data["nodes"][nid].get("type", "unknown")
                if ntype not in structural_types:
                    orphans.append(nid)
        if orphans:
            if len(orphans) <= 10:
                report.warnings.append(
                    f"{len(orphans)} orphan node(s) with no edges: "
                    + ", ".join(f"'{n}'" for n in orphans)
                )
            else:
                report.warnings.append(
                    f"{len(orphans)} orphan node(s) with no edges "
                    f"(showing first 10): "
                    + ", ".join(f"'{n}'" for n in orphans[:10])
                    + ", ..."
                )

        # -- Zero-confidence items -------------------------------------------
        zero_conf_nodes = [
            nid for nid, node in self._data["nodes"].items()
            if node.get("confidence", 1.0) == 0
        ]
        if zero_conf_nodes:
            report.warnings.append(
                f"{len(zero_conf_nodes)} node(s) with confidence=0: "
                + ", ".join(f"'{n}'" for n in zero_conf_nodes[:5])
                + ("..." if len(zero_conf_nodes) > 5 else "")
            )

        zero_conf_edges = []
        for i, edge in enumerate(self._data["edges"]):
            if edge.get("confidence", 1.0) == 0:
                zero_conf_edges.append(
                    f"{edge.get('source')}-[{edge.get('relation')}]->{edge.get('target')}"
                )
        if zero_conf_edges:
            report.warnings.append(
                f"{len(zero_conf_edges)} edge(s) with confidence=0: "
                + ", ".join(zero_conf_edges[:5])
                + ("..." if len(zero_conf_edges) > 5 else "")
            )

        # -- Missing embeddings (info) ---------------------------------------
        embeddable_types = node_ids - {
            nid for nid, node in self._data["nodes"].items()
            if node.get("type") in structural_types
        }
        missing_embeds = embeddable_types - set(self._embeddings.keys())
        if missing_embeds and self._embeddings:
            pct = len(missing_embeds) / len(embeddable_types) * 100 if embeddable_types else 0
            report.info.append(
                f"{len(missing_embeds)} of {len(embeddable_types)} "
                f"embeddable nodes ({pct:.0f}%) lack embeddings"
            )

        # -- NetworkX / dict sync check (info) -------------------------------
        nx_nodes = set(self._G.nodes())
        dict_nodes = node_ids
        if nx_nodes != dict_nodes:
            only_nx = nx_nodes - dict_nodes
            only_dict = dict_nodes - nx_nodes
            if only_nx:
                report.errors.append(
                    f"Nodes in NetworkX but not in data dict: {sorted(only_nx)[:5]}"
                )
            if only_dict:
                report.errors.append(
                    f"Nodes in data dict but not in NetworkX: {sorted(only_dict)[:5]}"
                )

        # -- Summary info -----------------------------------------------------
        if report.is_valid and not report.warnings:
            report.info.append("All validation checks passed.")
        else:
            report.info.append(
                f"Validation complete: {len(report.errors)} error(s), "
                f"{len(report.warnings)} warning(s)."
            )

        return report

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

    @staticmethod
    def _compute_layout_positions(
        render_nodes: dict,
        render_edges: list[dict],
        *,
        scale: float = 1000.0,
    ) -> dict[str, dict[str, float]]:
        """Pre-compute force-directed node positions using NetworkX.

        Uses Fruchterman-Reingold (``nx.spring_layout``) to compute stable
        positions server-side.  The resulting coordinates are scaled to
        *scale* × *scale* pixels so they map directly to Cytoscape.js
        ``preset`` layout positions, which renders instantly with no
        client-side simulation.

        Disconnected components are laid out independently and then tiled
        left-to-right so that outlier components cannot compress the main
        cluster via ``rescale_layout``.

        Returns a mapping of ``{node_id: {"x": float, "y": float}}``.
        """
        G = nx.Graph()
        G.add_nodes_from(render_nodes.keys())
        for e in render_edges:
            src, tgt = e["source"], e["target"]
            if src in render_nodes and tgt in render_nodes:
                G.add_edge(src, tgt)

        if len(G) == 0:
            return {}

        # --- Per-component layout to avoid rescale compression -----------
        components = list(nx.connected_components(G))

        # Single component: standard spring layout is fine
        if len(components) == 1:
            pos = nx.spring_layout(
                G,
                k=None,
                iterations=100,
                seed=42,
                scale=scale / 2,
            )
            return {
                nid: {"x": float(xy[0]), "y": float(xy[1])}
                for nid, xy in pos.items()
            }

        # Multiple components: layout each separately, then tile
        # Sort largest-first so the biggest cluster gets the most space.
        components.sort(key=len, reverse=True)

        combined: dict[str, tuple[float, float]] = {}
        half = scale / 2
        x_cursor = 0.0
        gap = scale * 0.05  # spacing between tiled components

        for comp in components:
            sub = G.subgraph(comp)
            n = len(sub)

            if n == 1:
                nid = next(iter(sub.nodes()))
                combined[nid] = (x_cursor, 0.0)
                x_cursor += gap
                continue

            # Scale each component proportionally to sqrt(n) so that
            # larger clusters get more room.
            comp_radius = half * math.sqrt(n / len(G))
            comp_radius = max(comp_radius, scale * 0.03)

            raw = nx.spring_layout(
                sub, k=None, iterations=100, seed=42, scale=comp_radius,
            )

            # Shift component so its left edge starts at x_cursor
            xs = [float(xy[0]) for xy in raw.values()]
            min_x = min(xs)
            width = max(xs) - min_x

            for nid, xy in raw.items():
                combined[nid] = (float(xy[0]) - min_x + x_cursor, float(xy[1]))

            x_cursor += width + gap

        # --- Center and rescale to [-half, half] -------------------------
        all_x = [p[0] for p in combined.values()]
        all_y = [p[1] for p in combined.values()]
        cx = (max(all_x) + min(all_x)) / 2
        cy = (max(all_y) + min(all_y)) / 2

        # Center around origin
        combined = {nid: (x - cx, y - cy) for nid, (x, y) in combined.items()}

        lim = max(
            max((abs(x) for x, _ in combined.values()), default=1.0),
            max((abs(y) for _, y in combined.values()), default=1.0),
        )
        if lim > 0:
            s = half / lim
            combined = {nid: (x * s, y * s) for nid, (x, y) in combined.items()}

        return {nid: {"x": xy[0], "y": xy[1]} for nid, xy in combined.items()}

    def cytoscape_elements(
        self,
        *,
        center_node: str | None = None,
        depth: int | None = None,
        min_confidence: float = 0.0,
        precompute_layout: bool = True,
    ) -> dict:
        """Return Cytoscape.js data needed to render an interactive graph.

        Returns a dict with keys:
          - ``elements``: list of Cytoscape element dicts (nodes + edges)
          - ``type_colors``: mapping of node type → hex colour
          - ``relation_colors``: mapping of relation → hex colour
          - ``types``: sorted list of node types present
          - ``relations``: sorted list of relation types present
          - ``proposals``: list of pending relation proposals
          - ``stats``: dict with node/edge/component/proposal counts
          - ``has_positions``: bool indicating whether elements contain
            pre-computed layout positions (``preset`` layout ready)
        """
        if center_node and depth:
            subgraph = self.get_subgraph(center_node, depth=depth)
            render_nodes = subgraph["nodes"]
            render_edges = subgraph["edges"]
        else:
            render_nodes = self._data["nodes"]
            render_edges = self._data["edges"]

        if min_confidence > 0:
            render_edges = [
                e for e in render_edges
                if e.get("confidence", 1.0) >= min_confidence
            ]

        # Pre-compute positions with NetworkX spring_layout
        positions: dict[str, dict[str, float]] = {}
        if precompute_layout and render_nodes:
            positions = self._compute_layout_positions(render_nodes, render_edges)

        elements: list[dict] = []
        degree: dict[str, int] = defaultdict(int)
        for e in render_edges:
            degree[e["source"]] += 1
            degree[e["target"]] += 1

        for nid, node in render_nodes.items():
            ntype = node.get("type", "custom")
            elem: dict = {
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
            }
            if nid in positions:
                elem["position"] = positions[nid]
            elements.append(elem)

        for i, edge in enumerate(render_edges):
            src, tgt = edge["source"], edge["target"]
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

        types_present = sorted({n.get("type", "custom") for n in render_nodes.values()})
        relations_present = sorted({e.get("relation", "related_to") for e in render_edges})

        pending_proposals = [
            {"name": p.name, "confidence": p.confidence,
             "justification": p.justification, "num_examples": len(p.examples)}
            for p in self.get_proposals(status=ProposalStatus.PENDING.value)
        ]

        return {
            "elements": elements,
            "type_colors": {t: self._node_color(t) for t in types_present},
            "relation_colors": {r: self._edge_color(r) for r in relations_present},
            "types": types_present,
            "relations": relations_present,
            "proposals": pending_proposals,
            "has_positions": bool(positions),
            "stats": {
                "nodes": len(render_nodes),
                "edges": len(render_edges),
                "components": nx.number_weakly_connected_components(self._G),
                "pending_proposals": len(pending_proposals),
                "relation_types": len(relations_present),
            },
        }

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

        When *layout* is ``"cose"`` (the default), node positions are
        pre-computed server-side with ``nx.spring_layout()`` and the
        Cytoscape ``preset`` layout is used for instant rendering.
        Other layout algorithms are still executed client-side.

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

        # Pre-compute positions for instant preset layout
        positions: dict[str, dict[str, float]] = {}
        if render_nodes:
            positions = self._compute_layout_positions(render_nodes, render_edges)
        has_positions = bool(positions)

        # Build Cytoscape elements
        elements = []

        # Compute degree for sizing
        degree: dict[str, int] = defaultdict(int)
        for e in render_edges:
            degree[e["source"]] += 1
            degree[e["target"]] += 1

        for nid, node in render_nodes.items():
            ntype = node.get("type", "custom")
            elem: dict = {
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
            }
            if nid in positions:
                elem["position"] = positions[nid]
            elements.append(elem)

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
            "relation_types": len(relations_present),
        }
        stats_json = json.dumps(stats)

        # Use preset layout for instant render when positions are pre-computed
        effective_layout = "preset" if has_positions else layout

        html = self._cytoscape_html_template(
            title=title,
            elements_json=elements_json,
            type_colors_json=type_colors_json,
            relation_colors_json=relation_colors_json,
            proposals_json=proposals_json,
            stats_json=stats_json,
            initial_layout=effective_layout,
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
  <span class="stat">
    <b id="stat-relations">0</b> relation types
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
  document.getElementById('stat-relations').textContent = graphStats.relation_types || 0;
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
# ---------------------------------------------------------------------------
# LLM response helpers
# ---------------------------------------------------------------------------

def _strip_thinking(raw: str) -> str:
    """Remove ``<think>…</think>`` blocks from LLM responses.

    Thinking models (Qwen3, DeepSeek-R1, etc.) wrap their chain-of-thought
    reasoning in ``<think>`` tags. This strips those blocks so we only
    parse the final answer. Handles:
    - Complete blocks: ``<think>…</think>``
    - Unclosed blocks (truncated): ``<think>…`` with no closing tag
    - Multiple blocks and nested whitespace
    """
    # Strip complete <think>...</think> blocks (DOTALL for newlines)
    result = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    # Strip unclosed <think> block (model hit token limit mid-thought)
    result = re.sub(r"<think>.*", "", result, flags=re.DOTALL)
    return result.strip()


# ---------------------------------------------------------------------------
# JSON salvage helper
# ---------------------------------------------------------------------------

def _salvage_truncated_json(raw: str) -> list[dict[str, Any]] | None:
    """Try to recover complete objects from a truncated JSON array.

    When the LLM hits its token limit the JSON is cut off mid-object,
    e.g. ``[{...}, {... <eof>``. This finds the last complete object
    boundary and closes the array so the valid prefix can be parsed.

    Also handles dict-wrapped arrays (e.g. ``{"entities": [{...}, {... <eof>``).
    In that case, the inner array is located and salvaged.

    Returns the list of recovered dicts, or None if recovery fails.
    """
    stripped = raw.lstrip()

    # Handle dict-wrapped arrays: find the first '[' inside the object
    if stripped.startswith("{"):
        arr_start = raw.find("[")
        if arr_start == -1:
            return None
        # Extract from the array start and recurse on the inner array
        return _salvage_truncated_json(raw[arr_start:])

    if not stripped.startswith("["):
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

def _cmd_preview_md(args: Any, kg: "KnowledgeGraph") -> None:
    """Handle --preview-md: show section breakdown of a markdown file."""
    from pathlib import Path as _P

    md_path = _P(args.preview_md)
    if not md_path.exists():
        print(f"File not found: {md_path}")
        return

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
            preview = sec["body"][:200].replace("\n", " ")
            if len(sec["body"]) > 200:
                preview += "..."
            print(f"  {prefix}  → {preview}")
    print(f"\n  To ingest, run: --ingest-md {args.preview_md}")


def _make_progress_callback(
    *, quiet: bool, verbose: bool,
) -> Callable[[dict[str, Any]], None]:
    """Build the CLI progress callback for ingestion and embedding events."""

    def _progress(event: dict[str, Any]) -> None:
        if quiet:
            return
        ev = event["event"]
        idx = event.get("index", 0)
        total = event.get("total", 0)
        heading = event.get("heading", "?")
        chars = event.get("char_count", 0)
        tag = f"[{idx + 1}/{total}]"

        if ev == "doc_start":
            doc = event.get("doc_id", "?")
            secs = event.get("total_sections", 0)
            ccount = event.get("char_count", 0)
            print(f"  Document: \"{doc}\" ({ccount:,} chars, {secs} sections)")
        elif ev == "section_skip":
            print(f"  {tag} Skip: \"{heading}\" ({chars:,} chars, {event.get('reason', 'skipped')})")
        elif ev == "section_start":
            print(f"  {tag} Extracting: \"{heading}\" ({chars:,} chars)...", end="", flush=True)
        elif ev == "extraction_done":
            n = event.get("triples_returned", 0)
            if verbose:
                print(f" {n} triples returned, processing...", end="", flush=True)
        elif ev == "triple_done":
            if verbose:
                ti = event.get("index", 0)
                tt = event.get("total", 0)
                if tt > 5 and (ti + 1) % 5 == 0:
                    print(".", end="", flush=True)
        elif ev == "section_done":
            elapsed = event.get("elapsed_seconds", 0)
            triples = event.get("triples", 0)
            nodes_added = event.get("nodes_added", 0)
            nodes_updated = event.get("nodes_updated", 0)
            edges_added = event.get("edges_added", 0)
            edges_updated = event.get("edges_updated", 0)
            errors = event.get("errors", [])
            parts = []
            if nodes_added:
                parts.append(f"{nodes_added} new")
            if nodes_updated:
                parts.append(f"{nodes_updated} updated")
            node_summary = "+".join(parts) + " nodes" if parts else "0 nodes"
            edge_parts = []
            if edges_added:
                edge_parts.append(f"{edges_added} new")
            if edges_updated:
                edge_parts.append(f"{edges_updated} updated")
            edge_summary = ("+".join(edge_parts) + " edges") if edge_parts else ""
            result_str = node_summary
            if edge_summary:
                result_str += f", {edge_summary}"
            if errors:
                print(f" {triples} triples → {result_str} ({len(errors)} errors, {elapsed}s)")
                if verbose:
                    for err in errors[:5]:
                        print(f"         {err}")
                    if len(errors) > 5:
                        print(f"         ... and {len(errors) - 5} more")
            else:
                print(f" {triples} triples → {result_str} ({elapsed}s)")
        elif ev == "incremental_skip_plan":
            unchanged = event.get("unchanged_sections", [])
            changed = event.get("changed_sections", [])
            removed = event.get("removed_sections", [])
            print(f"  Incremental: {len(unchanged)} unchanged, "
                  f"{len(changed)} changed, {len(removed)} removed")
            if verbose:
                for s in unchanged[:10]:
                    print(f"    skip: {s}")
                if len(unchanged) > 10:
                    print(f"    ... and {len(unchanged) - 10} more")
        elif ev == "version_diff":
            summary = event.get("summary", "")
            if verbose:
                print(f"  Version diff (v{event.get('version_from')}→v{event.get('version_to')}): {summary}")
        elif ev == "doc_done":
            if verbose:
                secs = event.get("total_sections", 0)
                triples = event.get("total_triples", 0)
                na = event.get("total_nodes_added", 0)
                ea = event.get("total_edges_added", 0)
                skipped_inc = event.get("sections_skipped_incremental", 0)
                inc_note = f", {skipped_inc} skipped (unchanged)" if skipped_inc else ""
                print(f"  Document complete: {secs} sections{inc_note}, "
                      f"{triples} triples, {na} nodes, {ea} edges")
        elif ev == "embed_start":
            tn = event.get("total_nodes", 0)
            sk = event.get("nodes_skipped", 0)
            print(f"  Embedding {tn} nodes ({sk} skipped)...", end="", flush=True)
        elif ev == "embed_batch_done":
            b = event.get("batch", 0)
            tb = event.get("total_batches", 0)
            ne = event.get("nodes_embedded", 0)
            tn = event.get("total_nodes", 0)
            if verbose:
                print(f" batch {b}/{tb} ({ne}/{tn})", end="", flush=True)
        elif ev == "embed_done":
            ne = event.get("nodes_embedded", 0)
            nb = event.get("batches", 0)
            print(f" {ne} nodes embedded in {nb} batches")

    return _progress


def _print_file_summary(
    stats: dict[str, Any],
    md_path: "Path",
    elapsed: float,
    kg: "KnowledgeGraph",
    embed_stats: dict[str, Any] | None,
    *,
    verbose: bool,
    auto_accept: bool,
) -> None:
    """Print the per-file ingestion summary."""
    graph_stats = kg.stats()
    print(f"\nIngested: {md_path.name}")
    print(f"  Document ID: {stats['doc_id']}")
    _skipped_inc = stats.get("sections_skipped_incremental", 0)
    if _skipped_inc:
        print(f"  Sections: {stats['total_sections']} ({_skipped_inc} skipped, unchanged)")
    else:
        print(f"  Sections: {stats['total_sections']}")
    print(f"  Triples extracted: {stats['total_triples']}")
    _n_added = stats["total_nodes_added"]
    _n_updated = stats["total_nodes_updated"]
    _e_added = stats["total_edges_added"]
    _e_updated = stats["total_edges_updated"]
    node_str = f"{_n_added} added"
    if _n_updated:
        node_str += f", {_n_updated} updated"
    edge_str = f"{_e_added} added"
    if _e_updated:
        edge_str += f", {_e_updated} updated"
    print(f"  Nodes: {node_str}")
    print(f"  Edges: {edge_str}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Graph totals: {graph_stats['num_nodes']} nodes, "
          f"{graph_stats['num_edges']} edges")
    if embed_stats and embed_stats["nodes_embedded"]:
        print(f"  Nodes embedded: {embed_stats['nodes_embedded']} "
              f"(skipped {embed_stats['nodes_skipped']})")
    if stats.get("total_proposals_created"):
        print(f"  New relation proposals: {stats['total_proposals_created']}")
        if auto_accept:
            print(f"  Auto-accepted {stats['total_proposals_created']} proposal(s)")
    if stats.get("source"):
        src = stats["source"]
        if src.get("is_duplicate"):
            print(f"  Warning: duplicate content (matches '{src['existing_doc_id']}')")
        elif src.get("is_update"):
            print(f"  Source updated: v{src['version']} ({src['stored_path']})")
        else:
            ver = src.get("version", 1)
            if ver > 1:
                print(f"  Source unchanged: v{ver} ({src['stored_path']})")
            else:
                print(f"  Source stored: v{ver} ({src['stored_path']})")

    if verbose:
        print("\n  Section details:")
        for sec_stat in stats.get("sections", []):
            _heading = sec_stat.get("heading", "?")
            sec_elapsed = sec_stat.get("elapsed_seconds", "")
            elapsed_str = f", {sec_elapsed}s" if sec_elapsed else ""
            if sec_stat.get("skipped"):
                print(f"    [skip] {_heading} ({sec_stat.get('reason', '')})")
            else:
                triples_info = ""
                n_triples = sec_stat.get("triples_processed", 0)
                n_nodes = sec_stat.get("nodes_added", 0)
                n_nodes_upd = sec_stat.get("nodes_updated", 0)
                n_edges = sec_stat.get("edges_added", 0)
                n_edges_upd = sec_stat.get("edges_updated", 0)
                n_errors = len(sec_stat.get("errors", []))
                if n_triples:
                    triples_info = f", {n_triples} triples"
                if n_nodes:
                    triples_info += f", {n_nodes} new nodes"
                if n_nodes_upd:
                    triples_info += f", {n_nodes_upd} updated nodes"
                if n_edges:
                    triples_info += f", {n_edges} new edges"
                if n_edges_upd:
                    triples_info += f", {n_edges_upd} updated edges"
                if n_triples and n_nodes == 0 and n_nodes_upd == 0 and n_errors:
                    sec_tag = "WARN"
                    triples_info += f", {n_errors} errors"
                else:
                    sec_tag = "ok"
                print(f"    [{sec_tag}]   {_heading} "
                      f"({sec_stat.get('char_count', 0):,} chars{triples_info}{elapsed_str})")

    if stats["errors"]:
        print(f"\n  Errors ({len(stats['errors'])}):")
        for err in stats["errors"]:
            print(f"    - {err}")


def _cmd_ingest_md(args: Any, kg: "KnowledgeGraph") -> None:
    """Handle --ingest-md: ingest markdown files into the graph."""
    from pathlib import Path as _P

    _quiet = args.quiet
    _verbose = args.verbose

    # Resolve which model to use for extraction
    _has_model = args.query_model or args.extract_model
    if _has_model:
        _extract_model = args.extract_model or args.query_model
        _provider = args.provider

        if _provider == "anthropic":
            _api_key = _get_anthropic_api_key()
            extract_fn: Callable[[str], list[dict[str, Any]]] = (
                lambda prompt: claude_extract(prompt, model=_extract_model, api_key=_api_key)
            )
            print(f"  Using model: {_extract_model} (provider: anthropic)")
        elif _provider == "bedrock":
            _bedrock_region = args.bedrock_region
            _bedrock_profile = args.bedrock_profile
            extract_fn = (
                lambda prompt: bedrock_extract(prompt, model=_extract_model, region=_bedrock_region, profile=_bedrock_profile)
            )
            print(f"  Using model: {_extract_model} (provider: bedrock, "
                  f"region: {_bedrock_region or 'default'}, profile: {_bedrock_profile or 'default'})")
        else:
            _extract_url = args.ollama_url.rstrip("/")
            extract_fn = (
                lambda prompt: local_extract(prompt, model=_extract_model, url=_extract_url)
            )
            print(f"  Using model: {_extract_model} at {_extract_url}")
    else:
        extract_fn = lambda _text: []

    if args.parallel > 1 and not _quiet:
        print(f"  Parallel extractions: {args.parallel} threads")
    if args.incremental and not _quiet:
        print("  Incremental mode: unchanged sections will skip LLM extraction")

    _progress = _make_progress_callback(quiet=_quiet, verbose=_verbose)

    # Resolve embed model once (shared across all files)
    _embed_model = None
    _embed_url = None
    if _has_model:
        if args.embed_model is not None:
            _embed_model = args.embed_model
        elif kg.embed_model:
            _embed_model = kg.embed_model
            if not _quiet:
                print(f"  Using embed model '{_embed_model}' from graph metadata")
        else:
            _embed_model = "qwen3-embedding"
        _embed_url = args.embed_url.rstrip("/")

    _all_stats: list[dict[str, Any]] = []
    _batch_t0 = time.monotonic()

    for md_file_arg in args.ingest_md:
        md_path = _P(md_file_arg)
        if not md_path.exists():
            print(f"File not found: {md_path}")
            continue

        text = md_path.read_text(encoding="utf-8")
        doc_id = md_path.stem
        file_path = md_path.resolve()

        if not _quiet and len(args.ingest_md) > 1:
            print(f"\n{'='*60}")
            print(f"  File: {md_path.name}")
            print(f"{'='*60}")

        _ingest_t0 = time.monotonic()

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
            parallel_extractions=args.parallel,
            incremental=args.incremental,
        )
        _total_elapsed = time.monotonic() - _ingest_t0

        # Embed nodes if an embedding model is configured
        _embed_stats = None
        if _embed_model:
            if args.provider == "bedrock" and args.embed_model is not None:
                _br = args.bedrock_region
                _bp = args.bedrock_profile
                def _embed_fn(batch: list[str]) -> list[list[float]]:
                    return bedrock_embed(batch, model=_embed_model, region=_br, profile=_bp)
                if not _quiet:
                    print(f"  Embed config: model='{_embed_model}' (bedrock, "
                          f"region={_br or 'default'}, profile={_bp or 'default'})")
            else:
                def _embed_fn(batch: list[str]) -> list[list[float]]:
                    return ollama_embed(batch, model=_embed_model, url=_embed_url)
                if not _quiet:
                    print(f"  Embed config: model='{_embed_model}' url='{_embed_url}'")

            _embed_t0 = time.monotonic()
            _embed_stats = kg.embed_nodes(
                _embed_fn, skip_existing=True, model_name=_embed_model,
                progress_fn=_progress,
            )
            _embed_elapsed = time.monotonic() - _embed_t0

        kg.save_all()

        # Auto-accept proposals after each file
        if stats.get("total_proposals_created") and args.auto_accept:
            pending = kg.get_proposals()
            for p in pending:
                kg.accept_proposal(p.name)
            if pending:
                kg.save()

        _all_stats.append({
            "stats": stats,
            "embed_stats": _embed_stats,
            "elapsed": _total_elapsed,
            "md_path": md_path,
        })

        _print_file_summary(
            stats, md_path, _total_elapsed, kg, _embed_stats,
            verbose=_verbose, auto_accept=args.auto_accept,
        )

    # Batch summary for multi-file ingestion
    if len(_all_stats) > 1:
        _batch_elapsed = time.monotonic() - _batch_t0
        total_triples = sum(s["stats"]["total_triples"] for s in _all_stats)
        total_nodes_added = sum(s["stats"]["total_nodes_added"] for s in _all_stats)
        total_nodes_updated = sum(s["stats"]["total_nodes_updated"] for s in _all_stats)
        total_edges_added = sum(s["stats"]["total_edges_added"] for s in _all_stats)
        total_edges_updated = sum(s["stats"]["total_edges_updated"] for s in _all_stats)
        graph_stats = kg.stats()
        print(f"\n{'='*60}")
        print(f"  Batch complete: {len(_all_stats)} files ingested")
        print(f"  Total triples: {total_triples}")
        _bn = f"{total_nodes_added} added"
        if total_nodes_updated:
            _bn += f", {total_nodes_updated} updated"
        _be = f"{total_edges_added} added"
        if total_edges_updated:
            _be += f", {total_edges_updated} updated"
        print(f"  Total nodes: {_bn}")
        print(f"  Total edges: {_be}")
        print(f"  Graph totals: {graph_stats['num_nodes']} nodes, "
              f"{graph_stats['num_edges']} edges")
        print(f"  Total time: {_batch_elapsed:.1f}s")
        print(f"{'='*60}")

    if _all_stats:
        print(f"\n  Graph saved to {kg.graph_path}")

        # Auto-export visualizations (once at the end)
        if not args.no_viz:
            graph_dir = kg.graph_path.parent
            base_name = kg.graph_path.stem

            cyto_path = graph_dir / f"{base_name}_cytoscape.html"
            try:
                kg.export_cytoscape(cyto_path)
                print(f"  Cytoscape visualization: {cyto_path}")
            except (OSError, ValueError) as e:
                logger.error("Cytoscape export failed: %s", e)

            try:
                pyvis_path = graph_dir / f"{base_name}_pyvis.html"
                kg.export_pyvis(pyvis_path)
                print(f"  Pyvis visualization: {pyvis_path}")
            except (ImportError, OSError, ValueError) as e:
                logger.error("Pyvis export skipped: %s", e)


def _cmd_verify_embeddings(kg: "KnowledgeGraph") -> None:
    """Handle --verify-embeddings: check embedding integrity."""
    emb_count = len(kg._embeddings)
    if emb_count == 0:
        print("No embeddings found.")
        return

    model = kg.embed_model or "(unknown)"
    dim = kg.embed_dim
    missing = kg.nodes_without_embeddings()

    zero_vectors: list[str] = []
    dim_mismatches: list[tuple[str, int]] = []
    first_dim: int | None = None
    for nid, vec in kg._embeddings.items():
        if first_dim is None:
            first_dim = len(vec)
        if all(v == 0.0 for v in vec):
            zero_vectors.append(nid)
        if len(vec) != first_dim:
            dim_mismatches.append((nid, len(vec)))

    print(f"  Embeddings: {emb_count}")
    print(f"  Model: {model}")
    print(f"  Dimension: {dim or first_dim}")
    print(f"  Nodes without embeddings: {len(missing)}")
    if missing and len(missing) <= 10:
        for nid in missing:
            print(f"    - {nid}")
    elif missing:
        for nid in missing[:5]:
            print(f"    - {nid}")
        print(f"    ... and {len(missing) - 5} more")

    if zero_vectors:
        print(f"  WARNING: {len(zero_vectors)} zero vector(s) detected "
              "(embedding may have failed):")
        for nid in zero_vectors[:5]:
            print(f"    - {nid}")
        if len(zero_vectors) > 5:
            print(f"    ... and {len(zero_vectors) - 5} more")

    if dim_mismatches:
        print(f"  WARNING: {len(dim_mismatches)} dimension mismatch(es):")
        for nid, d in dim_mismatches[:5]:
            print(f"    - {nid}: {d} (expected {first_dim})")

    if not zero_vectors and not dim_mismatches:
        sample_nid = next(iter(kg._embeddings))
        sample_vec = kg._embeddings[sample_nid]
        preview = ", ".join(f"{v:.4f}" for v in sample_vec[:5])
        print(f"  Sample ({sample_nid}): [{preview}, ...] (len={len(sample_vec)})")
        print("  OK — all embeddings look valid.")


def main() -> None:
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
    parser.add_argument("--ingest-md", nargs="+", metavar="FILE",
                        help="Ingest one or more markdown files into the graph (supports globs)")
    parser.add_argument("--sections", action="store_true",
                        help="Show full section details when used with --preview-md or --ingest-md")
    parser.add_argument("--query-model", "--ollama", nargs="?", const="qwen3-coder:30b",
                        metavar="MODEL", dest="query_model",
                        help="Model for LLM extraction during ingestion (default: qwen3-coder:30b)")
    parser.add_argument("--api-url", "--ollama-url", default="http://localhost:11434",
                        dest="ollama_url",
                        help="OpenAI-compatible API server URL (works with Ollama, llama.cpp, vLLM, etc.) "
                             "(default: http://localhost:11434)")
    parser.add_argument("--embed-url", default=None, metavar="URL",
                        help="API server URL for embeddings (default: same as --api-url)")
    parser.add_argument("--embed-model", default=None, metavar="MODEL",
                        help="Embedding model for node embeddings during ingestion "
                             "(default: auto-detect from graph, or qwen3-embedding)")
    parser.add_argument("--provider", choices=["local", "anthropic", "bedrock"], default="local",
                        help="LLM provider: 'local' for OpenAI-compatible servers (Ollama, llama.cpp, etc.), "
                             "'anthropic' for the Claude API, 'bedrock' for AWS Bedrock (default: local)")
    parser.add_argument("--bedrock-region", default=None, metavar="REGION",
                        help="AWS region for Bedrock (default: AWS_DEFAULT_REGION or us-east-1)")
    parser.add_argument("--bedrock-profile", default=None, metavar="PROFILE",
                        help="AWS profile name from ~/.aws/credentials for Bedrock")
    parser.add_argument("--extract-model", default=None, metavar="MODEL",
                        help="Dedicated model for entity/relation extraction during ingestion. "
                             "When set, --query-model is used only for querying. "
                             "Useful for using a fast model (e.g. Haiku) for extraction.")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip automatic visualization export after ingestion")
    parser.add_argument("-j", "--parallel", type=int, default=1, metavar="N",
                        dest="parallel",
                        help="Number of parallel LLM extraction threads during ingestion. "
                             "LLM calls run concurrently while graph writes remain serial. "
                             "(default: 1, sequential)")
    parser.add_argument("--incremental", action="store_true",
                        help="Skip LLM extraction for sections unchanged since the last "
                             "ingestion. Uses section-level content hashes to detect changes. "
                             "Structural nodes/edges are always updated. Dramatically reduces "
                             "LLM calls when re-ingesting documents with minor edits.")
    parser.add_argument("--auto-accept", action="store_true",
                        help="Automatically accept all new relation proposals created during ingestion")
    parser.add_argument("--doc-history", metavar="DOC_ID",
                        help="Show version history for a document")
    parser.add_argument("--sources", action="store_true",
                        help="List all stored source files")
    parser.add_argument("--check-sources", action="store_true",
                        help="Verify integrity of stored source files")
    parser.add_argument("--verify-embeddings", action="store_true",
                        help="Check embedding integrity: dimensions, zero vectors, coverage")
    parser.add_argument("--validate", action="store_true",
                        help="Run consistency checks: dangling edges, taxonomic cycles, "
                             "contradictions, orphan nodes, confidence anomalies")
    parser.add_argument("--analytics", action="store_true",
                        help="Show quality analytics: confidence distributions, hub nodes, "
                             "orphan detection, embedding coverage, quality score")
    parser.add_argument("--diff", metavar="OTHER_GRAPH",
                        help="Diff this graph against another graph file and show changes")
    parser.add_argument("--list-models", action="store_true",
                        help="List models available on the API server and exit")
    parser.add_argument("--merge", nargs="+", metavar="GRAPH",
                        help="Merge two or more existing graphs into the target graph "
                             "(specified by the positional 'path' argument)")
    parser.add_argument("--merge-strategy", choices=["latest", "first", "last"],
                        default="latest", dest="merge_strategy",
                        help="Conflict resolution strategy for merge (default: latest)")
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
        # Default: INFO messages print bare, WARNING+ get a level prefix
        # so retry/error messages are clearly visible in console output.
        class _DefaultFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                if record.levelno >= logging.WARNING:
                    self._style._fmt = "%(levelname)s: %(message)s"
                else:
                    self._style._fmt = "%(message)s"
                return super().format(record)

        handler = logging.StreamHandler()
        handler.setFormatter(_DefaultFormatter())
        logging.basicConfig(level=logging.INFO, handlers=[handler])

    # Commands that don't need the graph
    if args.list_models:
        from query_graph import _list_models
        _list_models(args.ollama_url)
        return

    if args.merge:
        import pprint as _pp
        source_graphs = [KnowledgeGraph(p) for p in args.merge]
        merged = KnowledgeGraph.merge_graphs(
            source_graphs,
            args.path,
            prefer=args.merge_strategy,
        )
        st = merged.stats()
        print(f"Merged {len(source_graphs)} graphs → {merged.graph_path}")
        _pp.pprint(st)
        return

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
        _cmd_preview_md(args, kg)

    if args.ingest_md:
        _cmd_ingest_md(args, kg)

    if args.doc_history:
        history = kg.get_document_history(args.doc_history)
        if not history:
            print(f"No history found for document '{args.doc_history}'.")
        else:
            print(f"\nVersion history for '{args.doc_history}' "
                  f"({len(history)} version(s)):\n")
            for v in history:
                current = " (current)" if v.get("is_current") else ""
                print(f"  v{v['version']}{current}  {v.get('stored_at', '')[:10]}")
                print(f"      hash: {v.get('content_hash', '')}  |  "
                      f"{v.get('char_count', 0):,} chars  |  "
                      f"{v.get('section_count', 0)} sections")
                print(f"      nodes: {v.get('node_count', 0)}  |  "
                      f"edges: {v.get('edge_count', 0)}")
                if v.get("diff") and v["diff"].get("has_changes"):
                    d = v["diff"]
                    print(f"      diff: {d['summary']}")
                    if d.get("added"):
                        for s in d["added"]:
                            print(f"        + {s}")
                    if d.get("removed"):
                        for s in d["removed"]:
                            print(f"        - {s}")
                    if d.get("modified"):
                        for s in d["modified"]:
                            print(f"        ~ {s}")
                print()

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

    if args.verify_embeddings:
        _cmd_verify_embeddings(kg)

    if args.validate:
        report = kg.validate()
        if report.errors:
            print(f"\n  ERRORS ({len(report.errors)}):")
            for msg in report.errors:
                print(f"    [ERROR] {msg}")
        if report.warnings:
            print(f"\n  WARNINGS ({len(report.warnings)}):")
            for msg in report.warnings:
                print(f"    [WARN]  {msg}")
        if report.info:
            print(f"\n  INFO ({len(report.info)}):")
            for msg in report.info:
                print(f"    [INFO]  {msg}")
        if report.is_valid:
            print(f"\n  Result: VALID ({report.total_issues} warning(s))")
        else:
            print(f"\n  Result: INVALID ({len(report.errors)} error(s), "
                  f"{len(report.warnings)} warning(s))")

    if args.analytics:
        a = kg.analytics()
        print(f"\n  Quality Score: {a['quality_score']}/100")

        cd = a["confidence_distribution"]
        print(f"\n  Confidence Distribution (nodes mean={cd['node_mean']:.2f}, "
              f"edges mean={cd['edge_mean']:.2f}):")
        print(f"    {'Bucket':<10} {'Nodes':>6} {'Edges':>6}")
        for label, nc, ec in zip(cd["buckets"], cd["node_counts"], cd["edge_counts"]):
            bar_n = "#" * min(nc, 40)
            print(f"    {label:<10} {nc:>6} {ec:>6}  {bar_n}")

        print(f"\n  Hub Nodes (top {len(a['hub_nodes'])}):")
        for h in a["hub_nodes"]:
            print(f"    {h['label']:<30} type={h['type']:<12} "
                  f"degree={h['degree']} (in={h['in_degree']}, out={h['out_degree']})")

        print(f"\n  Relation Types ({len(a['relation_stats'])}):")
        for rel, rs in a["relation_stats"].items():
            print(f"    {rel:<25} count={rs['count']:>4}  "
                  f"conf={rs['mean_confidence']:.2f} "
                  f"[{rs['min_confidence']:.2f}-{rs['max_confidence']:.2f}]")

        ec = a["embedding_coverage"]
        print(f"\n  Embedding Coverage: {ec['embedded']}/{ec['embeddable']} ({ec['pct']:.1f}%)")

        print(f"\n  Components: {len(a['component_sizes'])} "
              f"(sizes: {a['component_sizes'][:10]}{'...' if len(a['component_sizes']) > 10 else ''})")

        if a["orphan_nodes"]:
            print(f"\n  Orphan Nodes ({len(a['orphan_nodes'])}):")
            for o in a["orphan_nodes"][:10]:
                print(f"    {o['node_id']:<30} type={o['type']}")
            if len(a["orphan_nodes"]) > 10:
                print(f"    ... and {len(a['orphan_nodes']) - 10} more")

    if args.diff:
        other_path = Path(args.diff)
        if not other_path.exists():
            print(f"Error: '{args.diff}' not found.")
        else:
            diff = kg.diff_from_file(other_path)
            if not diff.has_changes:
                print("\n  No changes between the two graphs.")
            else:
                print(f"\n  Graph Diff: {kg.graph_path} vs {other_path}")
                print(f"  Summary: {diff.summary}\n")
                if diff.nodes_added:
                    print(f"  Nodes Added ({len(diff.nodes_added)}):")
                    for n in diff.nodes_added[:20]:
                        print(f"    + {n.get('node_id', '?')}: "
                              f"{n.get('label', '')} ({n.get('type', '?')})")
                    if len(diff.nodes_added) > 20:
                        print(f"    ... and {len(diff.nodes_added) - 20} more")
                if diff.nodes_removed:
                    print(f"  Nodes Removed ({len(diff.nodes_removed)}):")
                    for n in diff.nodes_removed[:20]:
                        print(f"    - {n.get('node_id', '?')}: "
                              f"{n.get('label', '')} ({n.get('type', '?')})")
                    if len(diff.nodes_removed) > 20:
                        print(f"    ... and {len(diff.nodes_removed) - 20} more")
                if diff.nodes_modified:
                    print(f"  Nodes Modified ({len(diff.nodes_modified)}):")
                    for n in diff.nodes_modified[:20]:
                        fields = ", ".join(n.get("changes", {}).keys())
                        print(f"    ~ {n.get('node_id', '?')}: {fields}")
                    if len(diff.nodes_modified) > 20:
                        print(f"    ... and {len(diff.nodes_modified) - 20} more")
                if diff.edges_added:
                    print(f"  Edges Added ({len(diff.edges_added)}):")
                    for e in diff.edges_added[:20]:
                        print(f"    + {e.get('source', '?')} "
                              f"-[{e.get('relation', '?')}]-> {e.get('target', '?')}")
                    if len(diff.edges_added) > 20:
                        print(f"    ... and {len(diff.edges_added) - 20} more")
                if diff.edges_removed:
                    print(f"  Edges Removed ({len(diff.edges_removed)}):")
                    for e in diff.edges_removed[:20]:
                        print(f"    - {e.get('source', '?')} "
                              f"-[{e.get('relation', '?')}]-> {e.get('target', '?')}")
                    if len(diff.edges_removed) > 20:
                        print(f"    ... and {len(diff.edges_removed) - 20} more")
                if diff.edges_modified:
                    print(f"  Edges Modified ({len(diff.edges_modified)}):")
                    for e in diff.edges_modified[:20]:
                        fields = ", ".join(e.get("changes", {}).keys())
                        print(f"    ~ {e.get('source', '?')} "
                              f"-[{e.get('relation', '?')}]-> {e.get('target', '?')}: {fields}")
                    if len(diff.edges_modified) > 20:
                        print(f"    ... and {len(diff.edges_modified) - 20} more")
                if diff.proposals_added:
                    print(f"  Proposals Added ({len(diff.proposals_added)}):")
                    for p in diff.proposals_added:
                        print(f"    + {p.get('name', '?')} "
                              f"(confidence: {p.get('confidence', '?')})")
                if diff.proposals_changed:
                    print(f"  Proposals Changed ({len(diff.proposals_changed)}):")
                    for p in diff.proposals_changed:
                        print(f"    ~ {p.get('name', '?')}: "
                              f"{p.get('old_status', '?')} -> {p.get('new_status', '?')}")

    if not any([args.stats, args.node, args.neighbors, args.split,
                args.proposals, args.accept, args.accept_all, args.reject,
                args.patterns, args.pyvis, args.cytoscape,
                args.preview_md, args.ingest_md,
                args.doc_history,
                args.sources, args.check_sources, args.verify_embeddings,
                args.validate, args.analytics, args.diff]):
        print(kg)
        print(f"\nUse --stats, --node, --neighbors, --split, --proposals, "
              f"--accept, --accept-all, --reject, --patterns, --pyvis, --cytoscape, "
              f"--preview-md, --ingest-md, --doc-history, --sources, --check-sources, "
              f"--verify-embeddings, --validate, --analytics, --diff for details.")


if __name__ == "__main__":
    main()
