#!/usr/bin/env python3
"""Opt-in installed-wheel qualification of hard calibration and native isolation.

Uses one already loaded, verified Nomic embedding replica. No model or consumer
administration, source injection, implicit retry, raw text or raw vector output.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from threading import Lock
import time
from urllib.parse import urlsplit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()
    ).hexdigest()


def compare(left, right):
    require(len(left) == len(right) > 0, "vector dimension mismatch")
    require(all(math.isfinite(x) for x in (*left, *right)), "nonfinite vector")
    a = math.sqrt(math.fsum(x * x for x in left))
    b = math.sqrt(math.fsum(x * x for x in right))
    require(a > 0 and b > 0, "zero vector norm")
    absolute = max(abs(x - y) for x, y in zip(left, right))
    cosine = math.fsum(x * y for x, y in zip(left, right)) / (a * b)
    norm = max(abs(a - 1), abs(b - 1))
    return {
        "max_abs": absolute,
        "cosine": cosine,
        "max_norm_error": norm,
        "passed": absolute <= 1e-5 and cosine >= 0.999999 and norm <= 1e-6,
    }


def normalize(vector):
    norm = math.sqrt(math.fsum(x * x for x in vector))
    require(norm > 0, "zero native vector")
    return [x / norm for x in vector]


def unpack(payload, count):
    rows = payload.get("data", [])
    require(len(rows) == count, "native result cardinality mismatch")
    indexed = {row["index"]: normalize(row["embedding"]) for row in rows}
    require(set(indexed) == set(range(count)), "native result indices mismatch")
    return [indexed[index] for index in range(count)]


def run(args, report):
    import runtime_llama
    from runtime_llama import (
        AutomaticEmbeddingConfig,
        AutomaticEmbeddingService,
        EmbeddingChunkSplitRequiredError,
    )
    from runtime_llama.adapters.embedding import LlamaCppEmbeddingAdapter
    from runtime_llama.client import ClientHTTPError, LlamaCppClient
    from runtime_llama.embedding_metadata import loaded_execution_metadata
    from runtime_llama.replica_discovery import (
        discover_embedding_pool,
        is_loaded,
        parse_replica,
    )

    dist = importlib.metadata.distribution("runtime-llama")
    source = Path(__file__).resolve().parents[1] / "src"
    require(
        not Path(runtime_llama.__file__).resolve().is_relative_to(source),
        "installed wheel required, checkout source imported",
    )
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    require(not direct.get("dir_info", {}).get("editable"), "editable install rejected")
    require(
        dist.version == runtime_llama.__version__ == "0.13.0a1",
        "candidate version mismatch",
    )
    if args.reuse_calibration:
        require(
            args.cache.is_dir() and any(args.cache.iterdir()), "warm cache must exist"
        )
    else:
        require(
            not args.cache.exists() or not any(args.cache.iterdir()),
            "cold cache must be empty",
        )
    report.update(
        {
            "version": dist.version,
            "installed_archive_sha256": direct.get("archive_info", {})
            .get("hashes", {})
            .get("sha256"),
            "scope": "One loaded Nomic replica; no cross-replica numerical finding closure",
            "runtime_admin_actions": False,
            "consumer_changes": False,
            "calibration_mode": "explicit_warm_reuse"
            if args.reuse_calibration
            else "cold_2500",
            "thresholds": {"max_abs": 1e-5, "min_cosine": 0.999999, "norm": 1e-6},
            "timing_scope": "Client wall and overlapping worker/HTTP sums, never pure GPU time",
            "phases": {},
            "http": [],
        }
    )
    started = time.monotonic()
    deadline = started + args.wall_timeout
    lock, active, peaks, worker = Lock(), Counter(), Counter(), {}
    phase = "before"
    allowed_retry_hashes = set()

    class ObservedClient(LlamaCppClient):
        def request(self, method, path, *positional, **kwargs):
            nonlocal phase
            remaining = deadline - time.monotonic()
            require(remaining > 0, "campaign wall deadline reached")
            kwargs["timeout"] = min(
                kwargs.get("timeout") or args.timeout, args.timeout, remaining
            )
            previous = kwargs.get("cancel")
            kwargs["cancel"] = lambda: (
                time.monotonic() >= deadline or bool(previous and previous())
            )
            clean_path = urlsplit(path).path
            body = kwargs.get("json")
            values = body.get("input") if isinstance(body, dict) else None
            input_count = len(values) if isinstance(values, list) else None
            kind = (
                "tokenize"
                if clean_path == "/tokenize"
                else ("embedding" if clean_path == "/v1/embeddings" else "metadata")
            )
            with lock:
                current = phase
                if current == "only_halves_retry" and kind == "embedding":
                    require(
                        isinstance(values, list)
                        and all(
                            isinstance(value, str)
                            and digest(value) in allowed_retry_hashes
                            for value in values
                        ),
                        "retry sent an original, completed or unrequested source row",
                    )
                require(
                    len(report["http"]) + sum(active.values()) < 4000,
                    "HTTP request cap reached",
                )
                active[current] += 1
                active[(current, kind)] += 1
                peaks[current] = max(peaks[current], active[current])
                peaks[(current, kind)] = max(
                    peaks[(current, kind)], active[(current, kind)]
                )
            begin, status, error_type = time.monotonic(), None, None
            try:
                result = super().request(method, path, *positional, **kwargs)
                status = 200
                return result
            except ClientHTTPError as exc:
                status, error_type = exc.status_code, exc.error_type
                raise
            finally:
                elapsed = time.monotonic() - begin
                with lock:
                    active[current] -= 1
                    active[(current, kind)] -= 1
                    report["http"].append(
                        {
                            "phase": current,
                            "method": method,
                            "path": clean_path,
                            "kind": kind,
                            "status": status,
                            "error_type": error_type,
                            "wall_sec": elapsed,
                            "input_count": input_count,
                        }
                    )

    original_prepare = LlamaCppEmbeddingAdapter._engine_prepare

    def measured_prepare(self, *positional, **kwargs):
        current, begin = phase, time.monotonic()
        try:
            return original_prepare(self, *positional, **kwargs)
        finally:
            with lock:
                item = worker.setdefault(
                    current, {"records": 0, "worker_wall_sum_sec": 0.0}
                )
                item["records"] += 1
                item["worker_wall_sum_sec"] += time.monotonic() - begin

    def measured(name, operation):
        nonlocal phase
        phase, begin = name, time.monotonic()
        try:
            return operation()
        finally:
            events = [event for event in report["http"] if event["phase"] == name]
            report["phases"][name] = {
                "client_wall_sec": time.monotonic() - begin,
                "http_counts": dict(Counter(event["kind"] for event in events)),
                "http_statuses": dict(
                    Counter(str(event["status"]) for event in events)
                ),
                "http_wall_sum_sec": math.fsum(event["wall_sec"] for event in events),
                "peak_http": peaks[name],
                "peak_embedding_http": peaks[(name, "embedding")],
                "preparation": dict(worker.get(name, {})),
            }

    client = ObservedClient(args.base_url, timeout=args.timeout, trust_env=False)
    LlamaCppEmbeddingAdapter._engine_prepare = measured_prepare
    service = None
    try:

        def snapshot():
            catalog = client.models()
            selector = args.model
            if selector is None:
                candidates = [
                    parse_replica(entry)
                    for entry in catalog.get("data", [])
                    if is_loaded(entry)
                ]
                candidates = [
                    item
                    for item in candidates
                    if item.identity is not None
                    and "nomic-embed-text-v2" in (item.model_name or "").lower()
                ]
                require(
                    len(candidates) == 1,
                    "one unambiguous loaded Nomic replica required",
                )
                selector = "sha256:" + candidates[0].identity.content_sha256
            pool = discover_embedding_pool(catalog, selector)
            require(
                pool.identity is not None and len(pool.replicas) == 1,
                "exactly one verified loaded Nomic replica required",
            )
            selected = pool.replicas[0]
            require(
                "nomic-embed-text-v2" in (selected.model_name or "").lower(),
                "loaded model is not Nomic v2",
            )
            entry = next(
                entry for entry in catalog["data"] if entry["id"] == selected.model
            )
            props = client.props(model=selected.model)
            metadata = loaded_execution_metadata(props, entry)
            context = props.get("default_generation_settings", {}).get("n_ctx")
            require(
                context == 512, "probe requires the observed 512-token Nomic profile"
            )
            args_hash = digest(entry.get("status", {}).get("args", []))
            return selected.model, {
                "alias_sha256": digest(selected.model),
                "content_sha256": pool.identity.content_sha256,
                "embedding_sha256": pool.identity.embedding_sha256,
                "loaded_args_sha256": args_hash,
                "build_sha256": digest(props.get("build_info")),
                "context_tokens": context,
                **asdict(metadata),
            }

        alias, report["before"] = measured("before", snapshot)
        service = AutomaticEmbeddingService(
            AutomaticEmbeddingConfig(
                base_url=args.base_url,
                cache_dir=args.cache,
                model=args.model,
                timeout=args.timeout,
                calibration_timeout=min(180, args.wall_timeout),
                batch_size=256,
                request_parallelism=2,
                calibration_workers=2,
            ),
            client=client,
        )
        calibration_phase = (
            "warm_calibration" if args.reuse_calibration else "cold_calibration"
        )

        def initial_budget():
            if args.reuse_calibration:
                current = service.refresh_character_budget(category="mixed")
                require(
                    current.available and current.max_input_chars is not None,
                    "warm cache has no matching completed calibration; no cold pass allowed",
                )
            return service.character_budget(category="mixed")

        initial = measured(calibration_phase, initial_budget)
        require(
            report["phases"][calibration_phase]["http_counts"].get("tokenize", 0)
            == (0 if args.reuse_calibration else 2500),
            "calibration tokenization count does not match the explicit cold/warm mode",
        )
        report["initial_budget"] = asdict(initial)
        categories = (
            "code",
            "documentation",
            "prose",
            "structured",
            "multilingual",
            "mixed",
        )
        budgets = measured(
            "cached_categories",
            lambda: {
                category: asdict(service.character_budget(category=category))
                for category in categories
            },
        )
        require(
            report["phases"]["cached_categories"]["http_counts"].get("tokenize", 0)
            == 0,
            "category reuse unexpectedly tokenized",
        )
        report["category_budgets"] = budgets
        profiles = list(args.cache.glob("*.json"))
        require(len(profiles) == 1, "one completed calibration profile expected")
        statistics = json.loads(profiles[0].read_text(encoding="utf-8"))["statistics"]
        report["calibration_statistics"] = statistics
        for category, status in budgets.items():
            factor = 0.9 if category == "mixed" else 0.8
            snapshot_budget = status["budget"]
            expected = math.floor(
                snapshot_budget["effective_context_tokens"]
                * statistics[category]["chars_per_token"]
                * factor
            )
            require(
                status["max_input_chars"] == expected, "mean safety factor mismatch"
            )
        corpus = [
            f"Queue document {index:03d}: worker {(index * 17) % 97:02d} preserves "
            f"ordered results, exact tokens and bounded preparation. Unicode città 🔬 {index:03d}."
            for index in range(511)
        ]
        report["corpus_sha256"] = digest(corpus)
        report["performance_comparison"] = (
            "No speed comparison with earlier two-replica runs; this run uses one replica"
        )
        vectors = measured("direct_511", lambda: service.embed_texts(corpus))
        require(len(vectors) == 511, "511 direct result cardinality mismatch")
        require(
            report["phases"]["direct_511"]["http_counts"].get("tokenize", 0) == 0,
            "ordinary embedding tokenized",
        )
        require(
            all(compare(vector, vector)["passed"] for vector in vectors),
            "direct vector norm failure",
        )
        report["direct_511_sha256"] = digest(vectors)
        good = ["First valid reference document.", "Second valid reference document."]
        oversized = "hello " * 520
        saved = {}

        def isolate():
            try:
                service.embed_texts([good[0], oversized, good[1]])
            except EmbeddingChunkSplitRequiredError as exc:
                require(
                    exc.input_count == 3 and exc.failed_input_indices == (1,),
                    "incorrect isolated input indices",
                )
                require(
                    tuple(item.input_index for item in exc.completed) == (0, 2),
                    "successful rows were not retained",
                )
                item = exc.inputs[0]
                require(
                    item.text_sha256 == hashlib.sha256(oversized.encode()).hexdigest(),
                    "halving source identity mismatch",
                )
                pieces = [
                    oversized[span.char_start : span.char_end] for span in item.halves
                ]
                require(
                    "".join(pieces) == oversized
                    and all(0 < len(piece) < len(oversized) for piece in pieces),
                    "halving does not make lossless strict progress",
                )
                saved["completed"] = [list(item.embedding) for item in exc.completed]
                saved["pieces"] = pieces
                report["isolated_halves"] = [asdict(span) for span in item.halves]
                return
            raise ValueError("expected native 500 and tokenizer-free halving advice")

        measured("native_500_isolation", isolate)
        isolation = report["phases"]["native_500_isolation"]
        require(
            isolation["http_counts"].get("tokenize", 0) == 0,
            "isolation unexpectedly tokenized",
        )
        require(
            int(isolation["http_statuses"].get("500", 0)) >= 1,
            "no actual native 500 observed",
        )

        def retry_halves():
            nonlocal allowed_retry_hashes
            pending, retained = list(saved["pieces"]), []
            rounds, submitted = [], 0
            for _ in range(8):
                require(time.monotonic() < deadline, "consumer retry deadline reached")
                submitted += len(pending)
                allowed_retry_hashes = {
                    digest("search_document: " + source) for source in pending
                }
                try:
                    result = service.embed_texts(pending)
                except EmbeddingChunkSplitRequiredError as exc:
                    require(
                        exc.input_count == len(pending), "retry input count mismatch"
                    )
                    retained.extend(list(item.embedding) for item in exc.completed)
                    next_pending = []
                    for item in exc.inputs:
                        source = pending[item.input_index]
                        require(
                            item.text_sha256
                            == hashlib.sha256(source.encode()).hexdigest(),
                            "retry source identity mismatch",
                        )
                        pieces = [
                            source[span.char_start : span.char_end]
                            for span in item.halves
                        ]
                        require(
                            "".join(pieces) == source
                            and all(0 < len(piece) < len(source) for piece in pieces),
                            "recursive halving does not make lossless strict progress",
                        )
                        next_pending.extend(pieces)
                    rounds.append(
                        {
                            "submitted": len(pending),
                            "completed": len(exc.completed),
                            "rejected": len(exc.inputs),
                            "next_halves": len(next_pending),
                        }
                    )
                    pending = next_pending
                else:
                    require(
                        len(result) == len(pending), "retry result cardinality mismatch"
                    )
                    retained.extend(result)
                    rounds.append(
                        {
                            "submitted": len(pending),
                            "completed": len(result),
                            "rejected": 0,
                            "next_halves": 0,
                        }
                    )
                    report["consumer_halving_retry"] = {
                        "rounds": rounds,
                        "logical_rows_submitted": submitted,
                        "retained_final_fragments": len(retained),
                        "max_rounds": 8,
                    }
                    return retained
            raise ValueError("consumer halving retry round bound reached")

        halves = measured("only_halves_retry", retry_halves)
        require(len(halves) >= 2, "recursive halves result cardinality mismatch")
        half_posts = [
            event
            for event in report["http"]
            if event["phase"] == "only_halves_retry" and event["kind"] == "embedding"
        ]
        require(
            sum(event["input_count"] for event in half_posts if event["status"] == 200)
            == len(halves),
            "completed descendant rows were repeated or omitted",
        )
        require(
            report["phases"]["only_halves_retry"]["http_counts"].get("tokenize", 0)
            == 0,
            "halves retry unexpectedly tokenized",
        )
        reference = measured("good_references", lambda: service.embed_texts(good))
        report["retained_good_equivalence"] = [
            compare(a, b) for a, b in zip(saved["completed"], reference)
        ]
        later = measured(
            "budget_after", lambda: service.refresh_character_budget(category="mixed")
        )
        require(
            asdict(later) == asdict(initial),
            "isolation changed the fixed calibrated budget",
        )
        report["budget_unchanged"] = True
        sample = ["search_document: " + text for text in corpus[:4]]

        def text_ids():
            tokens = [
                client.tokenize(
                    text, model=alias, add_special=True, parse_special=True
                )["tokens"]
                for text in sample
            ]
            native_text = unpack(client.embeddings(sample, model=alias), len(sample))
            native_ids = unpack(client.embeddings(tokens, model=alias), len(sample))
            return [compare(a, b) for a, b in zip(native_text, native_ids)]

        report["same_alias_text_id_equivalence"] = measured(
            "text_vs_exact_ids", text_ids
        )
        _, report["after"] = measured("after", snapshot)
        require(report["before"] == report["after"], "loaded profile changed")
        report["functional_passed"] = True
        report["numerical_passed"] = all(
            item["passed"]
            for item in (
                *report["retained_good_equivalence"],
                *report["same_alias_text_id_equivalence"],
            )
        )
        report["passed"] = report["functional_passed"] and report["numerical_passed"]
    finally:
        if service is not None:
            service.close()
        client.close()
        LlamaCppEmbeddingAdapter._engine_prepare = original_prepare
        report["client_wall_sec"] = time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--reuse-calibration", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--wall-timeout", type=float, default=180)
    args = parser.parse_args()
    if not args.allow_live:
        parser.error(
            "explicit --allow-live is required before importing the runtime or making HTTP requests"
        )
    if not all(
        math.isfinite(value) and value > 0
        for value in (args.timeout, args.wall_timeout)
    ):
        parser.error("timeouts must be positive and finite")
    if args.output.exists():
        parser.error("output must be a new evidence file")
    report = {
        "schema": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "passed": False,
        "functional_passed": False,
        "numerical_passed": False,
    }
    try:
        run(args, report)
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__}
        if type(exc) is ValueError:
            report["failure"]["reason"] = str(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print("PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
