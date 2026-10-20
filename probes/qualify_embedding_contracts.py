#!/usr/bin/env python3
"""Bounded opt-in qualification of total embedding budgets and exact preparation.

Uses externally loaded models only. No admin endpoint, consumer job or database
is touched. Timings describe public calls and HTTP, never pure GPU execution.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import struct
import sys
from threading import Lock
import time
from urllib.parse import urlsplit

MAX_ABS = 1e-5
MIN_COSINE = 0.999999
MAX_NORM_ERROR = 1e-6
MAX_RAW_PROJECTION_ERROR = 1e-12
DIMENSION = 768
PREFIXES = {
    "document": "title: none | text: ",
    "query": "task: search result | query: ",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def normalized(values):
    norm = math.hypot(*values)
    require(norm > 0 and math.isfinite(norm), "invalid vector norm")
    return [v / norm for v in values]


def raw_vectors(payload, count):
    data = payload.get("data") if isinstance(payload, dict) else None
    require(
        isinstance(data, list) and len(data) == count, "native vector count mismatch"
    )
    result = {}
    for fallback, item in enumerate(data):
        require(isinstance(item, dict), "invalid native response item")
        index = item.get("index", fallback)
        require(
            type(index) is int and 0 <= index < count and index not in result,
            "invalid native response index",
        )
        vector = item.get("embedding")
        require(
            isinstance(vector, list) and len(vector) == DIMENSION,
            "native dimension mismatch",
        )
        require(
            all(type(v) in (int, float) and math.isfinite(v) for v in vector),
            "invalid native vector value",
        )
        result[index] = [float(v) for v in vector]
    return [result[i] for i in range(count)]


def compare(left, right):
    require(len(left) == len(right) == DIMENSION, "comparison dimension mismatch")
    absolute = max(abs(a - b) for a, b in zip(left, right))
    nl, nr = math.hypot(*left), math.hypot(*right)
    cosine = math.fsum(a * b for a, b in zip(left, right)) / (nl * nr)
    norm_error = max(abs(nl - 1), abs(nr - 1))
    require(
        absolute <= MAX_ABS, "maximum absolute difference exceeded unchanged threshold"
    )
    require(cosine >= MIN_COSINE, "cosine below unchanged threshold")
    require(
        norm_error <= MAX_NORM_ERROR, "unit norm error exceeded unchanged threshold"
    )
    return {
        "maximum_absolute_error": absolute,
        "minimum_cosine": cosine,
        "maximum_norm_error": norm_error,
    }


def vector_digest(vectors):
    digest = hashlib.sha256()
    for vector in vectors:
        digest.update(struct.pack("<" + "d" * len(vector), *vector))
    return digest.hexdigest()


def profile_snapshot(catalog, props, aliases):
    result = {}
    entries = {entry["id"]: entry for entry in catalog["data"]}
    for alias in aliases:
        entry, prop = entries[alias], props[alias]
        require(
            entry.get("status", {}).get("value") == "loaded",
            "selected replica not loaded",
        )
        args = entry["status"].get("args")
        require(
            isinstance(args, list) and all(isinstance(arg, str) for arg in args),
            "structured loaded args required",
        )

        def option(flag):
            require(args.count(flag) == 1, "required loaded option missing or repeated")
            index = args.index(flag)
            require(index + 1 < len(args), "loaded option has no value")
            return args[index + 1]

        result[alias] = {
            "tags": sorted(entry.get("tags", [])),
            "loaded_context": prop["default_generation_settings"]["n_ctx"],
            "total_slots": prop["total_slots"],
            "native_batch": int(option("--batch-size")),
            "native_microbatch": int(option("--ubatch-size")),
            "flash_attention": option("--flash-attn"),
            "build": prop.get("build_info"),
        }
        require(
            result[alias]["loaded_context"] == 2048,
            "loaded context differs from planned profile2048",
        )
        require(
            result[alias]["native_batch"] == 2048,
            "native computation batch differs from planned profile2048",
        )
        require(
            result[alias]["total_slots"] == 1,
            "loaded slots differ from planned serial replica profile",
        )
        require(
            result[alias]["flash_attention"] == "on",
            "Flash Attention differs from observed plan; no administration allowed",
        )
    return result


class ObservedClient:
    """Observe public client seam without changing transport or request policy."""

    def __init__(self, client, deadline, wire):
        self.client = client
        self.deadline = deadline
        self.wire = wire
        self.lock = Lock()
        self.phase = "preflight"
        self.calls = []
        self.active = Counter()
        self.peaks = Counter()
        self.alias_active = Counter()
        self.alias_peaks = Counter()
        self.counts = {}
        self.raw = defaultdict(deque)
        self.maximum_payload_tokens = 0
        self.projection_error = 0.0
        self.output_count = 0
        self.maximum_norm_error = 0.0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def cancelled(self):
        return time.monotonic() >= self.deadline

    def _invoke(self, kind, path, function, args, kwargs, texts=None):
        require(not self.cancelled(), "qualification wall deadline exceeded")
        model = kwargs.get("model")
        started = time.monotonic()
        with self.lock:
            self.active[kind] += 1
            self.peaks[kind] = max(self.peaks[kind], self.active[kind])
            if kind == "embedding":
                self.alias_active[model] += 1
                self.alias_peaks[model] = max(
                    self.alias_peaks[model], self.alias_active[model]
                )
            record = {
                "request_id": len(self.calls),
                "phase": self.phase,
                "path": path,
                "kind": kind,
                "model": model,
                "started_monotonic": started,
            }
            self.calls.append(record)
        if texts is not None:
            record["text_count"] = len(texts)
            record["text_hashes"] = [sha(text.encode()) for text in texts]
        try:
            payload = function(*args, **kwargs)
            if kind == "tokenize":
                tokens = payload.get("tokens") if isinstance(payload, dict) else None
                require(
                    isinstance(tokens, list)
                    and all(type(t) is int and t >= 0 for t in tokens),
                    "invalid exact token response",
                )
                content = args[0]
                count = len(tokens)
                require(
                    kwargs.get("add_special") is True
                    and kwargs.get("parse_special") is True,
                    "tokenizer special policy mismatch",
                )
                with self.lock:
                    self.counts[content] = count
                record.update(
                    token_count=count,
                    text_hash=sha(content.encode()),
                    add_special=True,
                    parse_special=True,
                )
            elif kind == "embedding":
                vectors = raw_vectors(payload, len(texts))
                with self.lock:
                    counts = [self.counts.get(text) for text in texts]
                require(
                    all(type(n) is int and 0 <= n <= 2048 for n in counts),
                    "embedding lacks exact in-context preparation",
                )
                total = sum(counts)
                prompt_tokens = payload.get("usage", {}).get("prompt_tokens")
                if prompt_tokens is not None:
                    require(
                        prompt_tokens == total,
                        "native prompt_tokens differs from exact prepared counts",
                    )
                with self.lock:
                    self.maximum_payload_tokens = max(
                        self.maximum_payload_tokens, total
                    )
                    self.output_count += len(vectors)
                    for text, value in zip(texts, vectors):
                        self.raw[text].append(normalized(value))
                record.update(
                    token_counts=counts,
                    aggregate_tokens=total,
                    native_prompt_tokens=prompt_tokens,
                    ordered_raw_vector_sha256=vector_digest(vectors),
                )
            record["status"] = "passed"
            return payload
        except BaseException as exc:
            record.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            record["finished_monotonic"] = time.monotonic()
            record["elapsed_seconds"] = record["finished_monotonic"] - started
            with self.lock:
                self.active[kind] -= 1
                if kind == "embedding":
                    self.alias_active[model] -= 1
                self.wire.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.wire.flush()

    def health(self, *args, **kwargs):
        return self._invoke("metadata", "/health", self.client.health, args, kwargs)

    def models(self, *args, **kwargs):
        return self._invoke("metadata", "/v1/models", self.client.models, args, kwargs)

    def props(self, *args, **kwargs):
        return self._invoke("metadata", "/props", self.client.props, args, kwargs)

    def tokenize(self, content, **kwargs):
        return self._invoke(
            "tokenize", "/tokenize", self.client.tokenize, (content,), kwargs
        )

    def embeddings(self, input, **kwargs):
        texts = list(input)
        return self._invoke(
            "embedding",
            "/v1/embeddings",
            self.client.embeddings,
            (texts,),
            kwargs,
            texts,
        )

    def consume(self, prepared_texts, public):
        require(len(prepared_texts) == len(public), "public result count mismatch")
        for text, projected in zip(prepared_texts, public):
            with self.lock:
                require(bool(self.raw[text]), "public result lacks native association")
                expected = self.raw[text].popleft()
            require(
                len(projected) == DIMENSION
                and all(math.isfinite(v) for v in projected),
                "invalid public vector",
            )
            error = max(abs(a - b) for a, b in zip(projected, expected))
            self.projection_error = max(self.projection_error, error)
            self.maximum_norm_error = max(
                self.maximum_norm_error, abs(math.hypot(*projected) - 1)
            )
            require(
                error <= MAX_RAW_PROJECTION_ERROR,
                "public raw projection error exceeded unchanged threshold",
            )
            require(
                abs(math.hypot(*projected) - 1) <= MAX_NORM_ERROR,
                "public unit norm error exceeded unchanged threshold",
            )

    def stats(self, phase=None):
        calls = [r for r in self.calls if phase is None or r["phase"] == phase]
        kinds = Counter(r["kind"] for r in calls)
        durations = {
            kind: math.fsum(r["elapsed_seconds"] for r in calls if r["kind"] == kind)
            for kind in kinds
        }
        embedding = [r for r in calls if r["kind"] == "embedding"]
        aliases = Counter()
        for record in embedding:
            aliases[record["model"]] += record["text_count"]
        return {
            "http_requests_by_kind": dict(kinds),
            "sum_http_wall_seconds_by_kind": durations,
            "embedding_request_text_counts": [r["text_count"] for r in embedding],
            "embedding_request_aggregate_tokens": [
                r["aggregate_tokens"] for r in embedding if "aggregate_tokens" in r
            ],
            "texts_by_alias": dict(aliases),
        }


def corpus(index):
    return f"Synthetic contract document {index:03d}: archive worker preserves ordered results and exact token budgets. Unicode città 🔬."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--install", type=Path)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--wall-timeout", type=float, default=180)
    args = parser.parse_args()
    if not args.allow_live:
        parser.error("live inference requires --allow-live")
    if any(not math.isfinite(v) or v <= 0 for v in (args.timeout, args.wall_timeout)):
        parser.error("timeouts must be finite positive numbers")
    parsed = urlsplit(args.base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        parser.error("base URL requires HTTP(S) without credentials/query/fragment")
    if args.install:
        sys.path.insert(0, str(args.install.resolve()))
    import runtime_llama
    from runtime_llama import build_embedding_adapter
    from runtime_llama.client import LlamaCppClient
    from runtime_llama.errors import LLMBackendError

    args.output.parent.mkdir(parents=True, exist_ok=True)
    wire_path = args.output.with_suffix(".http.jsonl")
    started = time.monotonic()
    report = {
        "schema": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model_selector": args.model,
        "version": runtime_llama.__version__,
        "module": runtime_llama.__file__,
        "distribution_version": importlib.metadata.version("runtime-llama"),
        "scope": "Limited synthetic HTTP/preparation qualification; no GPU-only timing or retrieval quality claim",
        "runtime_admin_actions": False,
        "consumer_changes": False,
        "database_or_jobs_touched": False,
        "maximum_planned_output_vectors": 551,
        "thresholds": {
            "dimension": DIMENSION,
            "maximum_absolute_error": MAX_ABS,
            "minimum_cosine": MIN_COSINE,
            "maximum_norm_error": MAX_NORM_ERROR,
            "maximum_raw_projection_error": MAX_RAW_PROJECTION_ERROR,
        },
        "phases": [],
    }
    adapter = None
    client = LlamaCppClient(args.base_url, timeout=args.timeout)
    code = 1
    with wire_path.open("w", encoding="utf-8") as wire:
        observed = ObservedClient(client, started + args.wall_timeout, wire)
        try:
            configuration = {
                "base_url": args.base_url,
                "model": args.model,
                "embedding_replica_discovery": True,
                "batch_size": 256,
                "embedding_request_parallelism": 4,
                "embedding_queue_size": 2,
                "token_count_parallelism": 2,
                "timeout": args.timeout,
                "dimension": DIMENSION,
                "normalize": True,
                "embedding_prefixes": PREFIXES,
            }
            adapter = build_embedding_adapter(configuration, client=observed)
            status = adapter.health_status()
            require(status.get("model_ready"), "embedding model not ready")
            require(
                status["embedding_replica_discovery"]["identity_verified"],
                "content identity not verified",
            )
            aliases = sorted(status["replicas"])
            require(
                len(aliases) == 2,
                "planned live qualification requires exactly two loaded replicas",
            )
            before = profile_snapshot(
                observed.models(timeout=args.timeout),
                {
                    alias: observed.props(model=alias, timeout=args.timeout)
                    for alias in aliases
                },
                aliases,
            )
            report.update(
                configuration=configuration,
                aliases=aliases,
                before=before,
                identity=status["embedding_replica_discovery"],
            )

            incoming = [corpus(index) for index in range(256)]
            observed.phase = "prepare-and-reuse256"
            phase_started = time.monotonic()
            prepared = adapter.prepare_texts(
                incoming, timeout=args.timeout, cancel=observed.cancelled
            )
            prep_seconds = time.monotonic() - phase_started
            require(
                len(prepared) == 256
                and all(item.fits_input is True for item in prepared),
                "prepared corpus does not fit loaded limits",
            )
            require(
                all(
                    item.context_token_limit == item.max_input_tokens == 2048
                    and item.batch_token_limit is None
                    for item in prepared
                ),
                "public effective limits differ from loaded profile",
            )
            tokenizer_before = len(
                [r for r in observed.calls if r["kind"] == "tokenize"]
            )
            embedding_started = time.monotonic()
            public = adapter.embed_prepared(
                prepared, timeout=args.timeout, cancel=observed.cancelled
            )
            embed_seconds = time.monotonic() - embedding_started
            require(
                len([r for r in observed.calls if r["kind"] == "tokenize"])
                == tokenizer_before,
                "safe reuse duplicated exact counts",
            )
            observed.consume([item.text for item in prepared], public)
            stats = observed.stats(observed.phase)
            require(
                sorted(stats["texts_by_alias"].values()) == [128, 128],
                "256 total text budget not distributed128+128",
            )
            require(
                stats["http_requests_by_kind"].get("tokenize") == 256,
                "prepare did not count each corpus item exactly once",
            )
            require(
                stats["http_requests_by_kind"].get("embedding", 0) == 2,
                "native computation budget incorrectly split the two HTTP quotas",
            )
            require(
                max(stats["embedding_request_aggregate_tokens"]) > 2048,
                "fixture did not exercise aggregate HTTP tokens above native n_batch",
            )
            report["phases"].append(
                {
                    "name": observed.phase,
                    "input_count": 256,
                    "preparation_wall_seconds": prep_seconds,
                    "embed_prepared_wall_seconds": embed_seconds,
                    "wall_seconds": time.monotonic() - phase_started,
                    "ordered_public_vector_sha256": vector_digest(public),
                    **stats,
                }
            )

            observed.phase = "concurrent255-documents-and3-queries"
            phase_started = time.monotonic()
            queries = [
                "Which worker preserves order?",
                "How is the token budget enforced?",
                "Where are replicas managed?",
            ]
            with ThreadPoolExecutor(max_workers=2) as executor:
                docs_job = executor.submit(
                    adapter.embed_texts,
                    incoming[:255],
                    timeout=args.timeout,
                    cancel=observed.cancelled,
                )
                query_job = executor.submit(
                    adapter.embed_texts,
                    queries,
                    input_type="query",
                    timeout=args.timeout,
                    cancel=observed.cancelled,
                )
                documents, query_vectors = docs_job.result(), query_job.result()
            observed.consume(
                [PREFIXES["document"] + text for text in incoming[:255]], documents
            )
            observed.consume(
                [PREFIXES["query"] + text for text in queries], query_vectors
            )
            metrics = [
                compare(public[index], value) for index, value in enumerate(documents)
            ]
            report["phases"].append(
                {
                    "name": observed.phase,
                    "input_count": 258,
                    "wall_seconds": time.monotonic() - phase_started,
                    "ordered_document_vector_sha256": vector_digest(documents),
                    "comparisons": 255,
                    "maximum_absolute_error": max(
                        m["maximum_absolute_error"] for m in metrics
                    ),
                    "minimum_cosine": min(m["minimum_cosine"] for m in metrics),
                    **observed.stats(observed.phase),
                }
            )

            observed.phase = "cross-replica-references"
            phase_started = time.monotonic()
            comparisons = []
            # Sixteen representative documents per alias; no additional full-corpus soak.
            sample_indices = [index * 17 for index in range(16)]
            for alias in aliases:
                sample = [prepared[index].text for index in sample_indices]
                native = observed.embeddings(
                    sample,
                    model=alias,
                    timeout=args.timeout,
                    cancel=observed.cancelled,
                    cancel_wait=True,
                )
                references = [
                    normalized(value) for value in raw_vectors(native, len(sample))
                ]
                observed.consume(sample, references)
                for index, reference in zip(sample_indices, references):
                    comparisons.append(compare(public[index], reference))
            report["phases"].append(
                {
                    "name": observed.phase,
                    "input_count": 32,
                    "wall_seconds": time.monotonic() - phase_started,
                    "comparisons": 32,
                    "maximum_absolute_error": max(
                        m["maximum_absolute_error"] for m in comparisons
                    ),
                    "minimum_cosine": min(m["minimum_cosine"] for m in comparisons),
                    **observed.stats(observed.phase),
                }
            )

            observed.phase = "exact-boundary-and-recovery"
            phase_started = time.monotonic()
            boundaries = {}
            # Prior same-identity tokenizer evidence chooses candidates only.
            # Every target is authorized exclusively by current public exact counts.
            for target, repetitions in ((2048, 2039), (2049, 2040)):
                for _attempt in range(3):
                    value = adapter.prepare_texts(
                        [" a" * repetitions],
                        timeout=args.timeout,
                        cancel=observed.cancelled,
                    )[0]
                    if value.token_count == target:
                        boundaries[target] = value
                        break
                    repetitions += target - value.token_count
                    require(repetitions > 0, "boundary search produced invalid seed")
                require(
                    target in boundaries,
                    "exact boundary not found within bounded native counts",
                )
            exact, over = boundaries[2048], boundaries[2049]
            require(
                exact.fits_input is True and over.fits_input is False,
                "public boundary fit flags incorrect",
            )
            exact_vector = adapter.embed_prepared(
                [exact], timeout=args.timeout, cancel=observed.cancelled
            )
            observed.consume([exact.text], exact_vector)
            embedding_before = len(
                [r for r in observed.calls if r["kind"] == "embedding"]
            )
            try:
                adapter.embed_prepared(
                    [over], timeout=args.timeout, cancel=observed.cancelled
                )
            except LLMBackendError:
                pass
            else:
                raise AssertionError("measured over-context input was admitted")
            require(
                len([r for r in observed.calls if r["kind"] == "embedding"])
                == embedding_before,
                "over-context input caused embedding HTTP",
            )
            recovered = adapter.embed_query(
                "short recovery", timeout=args.timeout, cancel=observed.cancelled
            )
            observed.consume([PREFIXES["query"] + "short recovery"], [recovered])
            report["phases"].append(
                {
                    "name": observed.phase,
                    "input_count": 2,
                    "wall_seconds": time.monotonic() - phase_started,
                    "exact_boundary": exact.token_count,
                    "oversize_measured": over.token_count,
                    "effective_limit": over.max_input_tokens,
                    "oversize_embedding_http_requests": 0,
                    **observed.stats(observed.phase),
                }
            )

            observed.phase = "raw-query-document-parity"
            phase_started = time.monotonic()
            parity = []
            for task in ("raw", "query", "document"):
                value = adapter.prepare_texts(
                    ["  città 🔬 <bos>\n"],
                    input_type=task,
                    timeout=args.timeout,
                    cancel=observed.cancelled,
                )[0]
                require(
                    value.text
                    == ("" if task == "raw" else PREFIXES[task]) + "  città 🔬 <bos>\n",
                    "task prefix applied inconsistently",
                )
                got = adapter.embed_prepared(
                    [value], timeout=args.timeout, cancel=observed.cancelled
                )
                observed.consume([value.text], got)
                parity.append(
                    {
                        "task": task,
                        "token_count": value.token_count,
                        "prepared_text_sha256": sha(value.text.encode()),
                        "vector_sha256": vector_digest(got),
                    }
                )
            report["phases"].append(
                {
                    "name": observed.phase,
                    "input_count": 3,
                    "wall_seconds": time.monotonic() - phase_started,
                    "tasks": parity,
                    **observed.stats(observed.phase),
                }
            )

            observed.phase = "after"
            after_status = adapter.health_status()
            after = profile_snapshot(
                observed.models(timeout=args.timeout),
                {
                    alias: observed.props(model=alias, timeout=args.timeout)
                    for alias in aliases
                },
                aliases,
            )
            require(
                before == after,
                "loaded runtime profile or identity changed during qualification",
            )
            require(
                observed.peaks["embedding"] <= 2
                and all(n <= 1 for n in observed.alias_peaks.values()),
                "embedding HTTP concurrency exceeded verified serial replicas",
            )
            require(
                observed.peaks["tokenize"] <= 2,
                "tokenizer admission exceeded shared adapter limit",
            )
            require(observed.output_count <= 551, "live output budget exceeded")
            require(not any(observed.active.values()), "HTTP work not drained")
            require(
                not any(observed.raw.values()),
                "native vectors not associated with public output",
            )
            require(
                after_status["scheduler"]["in_flight"]
                == after_status["scheduler"]["queued"]
                == 0,
                "scheduler not drained",
            )
            report.update(
                result="passed",
                after=after,
                profile_unchanged=True,
                observed_output_vectors=observed.output_count,
                total_http=observed.stats(),
                peak_http_by_kind=dict(observed.peaks),
                peak_embedding_http_by_alias=dict(observed.alias_peaks),
                maximum_payload_tokens=observed.maximum_payload_tokens,
                maximum_raw_projection_error=observed.projection_error,
                maximum_norm_error=observed.maximum_norm_error,
                scheduler=after_status["scheduler"],
            )
            code = 0
        except BaseException as exc:
            report.update(result="failed", error_type=type(exc).__name__)
            # Assertion text is local and never contains a remote body or secret.
            if isinstance(exc, AssertionError):
                report["reason"] = str(exc)
            report.update(
                observed_output_vectors=observed.output_count,
                total_http=observed.stats(),
                peak_http_by_kind=dict(observed.peaks),
                peak_embedding_http_by_alias=dict(observed.alias_peaks),
            )
        finally:
            close_started = time.monotonic()
            if adapter is not None:
                adapter.close()
            client.close()
            report.update(
                closed=True,
                close_wall_seconds=time.monotonic() - close_started,
                total_wall_seconds=time.monotonic() - started,
                finished_at=datetime.now(timezone.utc).isoformat(),
                active_http_at_finish=dict(observed.active),
                wire_reference=str(wire_path),
            )
    report["wire_sha256"] = sha(wire_path.read_bytes())
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "result": report["result"],
                "outputs": report.get("observed_output_vectors"),
                "output": str(args.output),
            }
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
