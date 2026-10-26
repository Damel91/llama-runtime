#!/usr/bin/env python3
"""Bounded opt-in queue qualification against an installed wheel and loaded models.

No source-path injection, server administration, consumer jobs or retries.
A single installed candidate uses eight source windows with no wait override;
loaded LLM work is allowed. Timings include client/HTTP work and instrumentation,
never pure GPU time or a throughput comparison.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from threading import Lock
import time
from urllib.parse import urlencode, urlsplit

DIMENSION = 768
WINDOW_SIZE, PENDING_WINDOWS = 64, 4
MAX_ABS, MIN_COSINE, MAX_NORM, MAX_PROJECTION = 1e-5, 0.999999, 1e-6, 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def catalog_profile_digest(catalog):
    # Native /v1/models computes this observation timestamp on every request.
    stable = {
        **catalog,
        "data": [
            {key: value for key, value in entry.items() if key != "created"}
            for entry in catalog["data"]
        ],
    }
    return digest(stable)


def corpus(prefix=""):
    return [
        prefix + f"Queue document {i:03d}: worker {(i * 17) % 97:02d} preserves "
        f"ordered results, exact tokens and bounded preparation. Unicode città 🔬 {i:03d}."
        for i in range(511)
    ]


def source_windows(texts):
    return [
        texts[start : start + WINDOW_SIZE]
        for start in range(0, len(texts), WINDOW_SIZE)
    ]


def audit_windows(expected, observed, requests):
    require(
        observed == [len(window) for window in expected],
        "source window cardinality mismatch",
    )
    pipeline = [
        request
        for request in requests
        if request["phase"] == "pipeline" and request["kind"] == "embedding"
    ]
    mixed = [
        request for request in pipeline if len(request.get("source_windows", [])) > 1
    ]
    require(mixed, "no embedding HTTP batch mixed source windows")
    return {
        "source_window_sizes": observed,
        "delivered_windows": len(observed),
        "mixed_source_window_posts": len(mixed),
        "maximum_source_windows_per_post": max(
            len(request["source_windows"]) for request in pipeline
        ),
    }


def normalized(vector):
    require(len(vector) == DIMENSION, "dimension mismatch")
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in vector),
        "invalid vector",
    )
    norm = math.hypot(*vector)
    require(norm > 0, "zero vector norm")
    return [v / norm for v in vector]


def compare(left, right):
    require(len(left) == len(right) == DIMENSION, "dimension mismatch")
    nl, nr = math.hypot(*left), math.hypot(*right)
    require(nl > 0 and nr > 0, "zero vector norm")
    maximum = max(abs(a - b) for a, b in zip(left, right))
    cosine = math.fsum(a * b for a, b in zip(left, right)) / (nl * nr)
    norm_error = max(abs(nl - 1), abs(nr - 1))
    return {
        "maximum_absolute_error": maximum,
        "cosine": cosine,
        "maximum_norm_error": norm_error,
        "passed": maximum <= MAX_ABS
        and cosine >= MIN_COSINE
        and norm_error <= MAX_NORM,
    }


def raw_vectors(payload, count):
    rows = payload.get("data") if isinstance(payload, dict) else None
    require(
        isinstance(rows, list) and len(rows) == count, "response cardinality mismatch"
    )
    result = {}
    for row in rows:
        index = row.get("index") if isinstance(row, dict) else None
        require(
            type(index) is int and 0 <= index < count and index not in result,
            "invalid response index",
        )
        vector = row.get("embedding")
        normalized(vector)
        result[index] = vector
    return [result[i] for i in range(count)]


class Budget:
    def __init__(self, wall=180, limits=None):
        self.started = time.monotonic()
        self.deadline = self.started + wall
        # Retain finite diagnostic headroom even if batching regresses to one
        # POST per text. Such fragmentation remains visible in the report.
        self.limits = limits or {"embedding": 600, "tokenize": 550, "metadata": 1200}
        self.lock, self.counts, self.active, self.peaks = (
            Lock(),
            Counter(),
            Counter(),
            Counter(),
        )
        self.records, self.ids, self.raw = [], {}, {}
        self.phase_raw = {"pipeline": self.raw}
        self.source_positions = {}
        self.phase_positions = {}
        self.compatible_aliases = frozenset()
        self.phase = "preflight"

    def text_for_ids(self, model, ids):
        key = model, tuple(ids)
        if key in self.ids:
            return self.ids[key]
        # An audit lookup, not authorization to run inference: the adapter
        # independently verifies reuse. Only the pinned equivalent aliases
        # with an equal observed tokenizer build may share this correlation.
        require(model in self.compatible_aliases, "uncorrelated token IDs")
        candidates = {
            text
            for (alias, tokens), text in self.ids.items()
            if alias in self.compatible_aliases and tokens == key[1]
        }
        require(len(candidates) == 1, "ambiguous or uncorrelated token IDs")
        return candidates.pop()

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        require(remaining > 0, "absolute wall budget exhausted")
        return remaining

    def enter(self, method, path):
        endpoint = urlsplit(path).path
        allowed = {
            "GET": {"/v1/models", "/props", "/slots", "/health"},
            "POST": {"/tokenize", "/v1/embeddings"},
        }
        require(
            endpoint in allowed.get(method, set()), "probe endpoint is not permitted"
        )
        kind = (
            "embedding"
            if endpoint == "/v1/embeddings"
            else "tokenize"
            if endpoint == "/tokenize"
            else "metadata"
        )
        self.remaining()
        with self.lock:
            require(
                self.counts[kind] < self.limits[kind],
                f"{kind} request budget exhausted",
            )
            self.counts[kind] += 1
            self.counts[endpoint] += 1
            self.counts["total"] += 1
            self.active[kind] += 1
            self.active["all"] += 1
            self.peaks[kind] = max(self.peaks[kind], self.active[kind])
            self.peaks["all"] = max(self.peaks["all"], self.active["all"])
        return kind, endpoint


def observed_client_type(base):
    class ObservedClient(base):
        def __init__(self, *args, budget, **kwargs):
            self.observed = budget
            super().__init__(*args, **kwargs)

        def request(self, method, path, json=None, *args, **kwargs):
            budget = self.observed
            kind, endpoint = budget.enter(method, path)
            started = time.monotonic()
            record = {
                "kind": kind,
                "endpoint": endpoint,
                "phase": budget.phase,
                "start_sec": started - budget.started,
                "model": (json or {}).get("model"),
                "request_json_bytes": 0,
            }
            try:
                record["request_json_bytes"] = (
                    len(canonical(json)) if json is not None else 0
                )
                original_cancel = kwargs.get("cancel")
                kwargs["cancel"] = lambda: (
                    time.monotonic() >= budget.deadline
                    or bool(original_cancel and original_cancel())
                )
                kwargs["timeout"] = min(
                    kwargs.get("timeout") or 20, 20, budget.remaining()
                )
                kwargs["max_response_bytes"] = min(
                    kwargs.get("max_response_bytes") or 8 * 1024 * 1024,
                    8 * 1024 * 1024,
                )
                result = super().request(method, path, json, *args, **kwargs)
                record["client_http_wall_sec"] = time.monotonic() - started
                model = (json or {}).get("model")
                if endpoint == "/tokenize":
                    ids = result.get("tokens") if isinstance(result, dict) else None
                    require(
                        isinstance(ids, list) and all(type(i) is int for i in ids),
                        "invalid tokenization",
                    )
                    with budget.lock:
                        budget.ids[model, tuple(ids)] = json["content"]
                if kind == "embedding":
                    rows = json["input"]
                    rows = (
                        [rows]
                        if isinstance(rows, str) or (rows and type(rows[0]) is int)
                        else rows
                    )
                    with budget.lock:
                        texts = [
                            row
                            if isinstance(row, str)
                            else budget.text_for_ids(model, row)
                            for row in rows
                        ]
                    vectors = raw_vectors(result, len(rows))
                    record.update(
                        model=model,
                        texts=len(rows),
                        tokens=sum(len(row) for row in rows)
                        if rows and not isinstance(rows[0], str)
                        else None,
                        input_format="text"
                        if rows and isinstance(rows[0], str)
                        else "tokens",
                    )
                    phase = record["phase"]
                    if phase in budget.phase_raw:
                        with budget.lock:
                            positions = budget.phase_positions.get(
                                phase, budget.source_positions
                            )
                            record["source_windows"] = sorted(
                                {
                                    positions[text][0]
                                    for text in texts
                                    if text in positions
                                }
                            )
                            if positions:
                                require(
                                    all(text in positions for text in texts),
                                    "unknown source-window input",
                                )
                            raw = budget.phase_raw[phase]
                            for text, vector in zip(texts, vectors):
                                require(
                                    text not in raw,
                                    "duplicate phase inference",
                                )
                                raw[text] = {
                                    "model": model,
                                    "vector": vector,
                                    "input_format": record["input_format"],
                                }
                                if text in positions:
                                    raw[text]["source_position"] = positions[text]
                return result
            except BaseException as exc:
                record["error_type"] = type(exc).__name__
                raise
            finally:
                record["wrapper_wall_sec"] = time.monotonic() - started
                with budget.lock:
                    budget.active[kind] -= 1
                    budget.active["all"] -= 1
                    budget.records.append(record)

    return ObservedClient


def snapshot(client, aliases, catalog=None):
    catalog = client.models() if catalog is None else catalog
    entries = {e["id"]: e for e in catalog["data"]}
    result = {
        "catalog_profile_sha256": catalog_profile_digest(catalog),
        "catalog_excluded_fields": ["data[].created"],
        "models": {},
        "slots": {},
    }
    for alias in aliases:
        entry = entries[alias]
        require(
            entry.get("status", {}).get("value") == "loaded", "replica is not loaded"
        )
        props = client.props(model=alias)
        arguments = entry["status"].get("args", [])
        options = {}
        for flag in (
            "--ctx-size",
            "--batch-size",
            "--ubatch-size",
            "--parallel",
            "--flash-attn",
            "--pooling",
            "--embd-normalize",
        ):
            options[flag] = [
                arguments[i + 1] for i, arg in enumerate(arguments[:-1]) if arg == flag
            ]
        result["models"][alias] = {
            "identity_tags": sorted(entry.get("tags", [])),
            "loaded_args_sha256": digest(arguments),
            "loaded_options": options,
            "props_sha256": digest(props),
            "build": props.get("build_info"),
            "context": props.get("default_generation_settings", {}).get("n_ctx"),
            "total_slots": props.get("total_slots"),
        }
    for alias, entry in entries.items():
        if entry.get("status", {}).get("value") != "loaded":
            continue
        try:
            slots = client.request("GET", "/slots?" + urlencode({"model": alias}))
            result["slots"][alias] = {
                "is_processing": [s.get("is_processing") for s in slots]
            }
        except Exception as exc:
            result["slots"][alias] = {"unavailable": type(exc).__name__}
    return result


def validate_pipeline(texts, outputs, raw):
    require(len(outputs) == len(texts) == len(raw), "pipeline cardinality mismatch")
    maximum, norm_error = 0.0, 0.0
    for text, output in zip(texts, outputs):
        reference = normalized(raw[text]["vector"])
        normalized(output)
        difference = max(abs(a - b) for a, b in zip(output, reference))
        maximum = max(maximum, difference)
        norm_error = max(norm_error, abs(math.hypot(*output) - 1))
        require(
            difference <= MAX_PROJECTION and math.isfinite(difference),
            "pipeline order or raw projection mismatch",
        )
        require(norm_error <= MAX_NORM, "pipeline norm mismatch")
        require(
            raw[text]["input_format"] == "tokens", "native token-ID correlation missing"
        )
    return {
        "correlations": len(texts),
        "maximum_raw_projection_error": maximum,
        "maximum_norm_error": norm_error,
    }


def oversize_canary(adapter, budget, text, backend_error):
    prepared = adapter.prepare_texts(
        [text], input_type="raw", timeout=min(30, budget.remaining())
    )[0]
    require(
        prepared.max_input_tokens is not None
        and prepared.token_count > prepared.max_input_tokens,
        "oversize canary did not exceed the exact loaded/configured input limit",
    )
    remote_counts = {
        len(ids) for (_, ids), content in budget.ids.items() if content == text
    }
    require(
        remote_counts == {prepared.token_count},
        "oversize count did not match remote tokenization",
    )
    before = budget.counts["embedding"]
    try:
        adapter.embed_prepared([prepared], timeout=min(30, budget.remaining()))
    except backend_error as exc:
        require(
            "input_exceeds_context" in str(exc),
            "oversize failed for an unrelated reason",
        )
    else:
        raise ValueError("oversize input was admitted to inference")
    require(
        budget.counts["embedding"] == before, "oversize rejection sent embedding HTTP"
    )
    return {
        "exact_token_count": prepared.token_count,
        "max_input_tokens": prepared.max_input_tokens,
        "remote_count_matches": True,
        "rejected_without_embedding_http": True,
    }


def phase_counts(budget, phase):
    return dict(
        Counter(record["kind"] for record in budget.records if record["phase"] == phase)
    )


def qualify_reuse(runtime, config, client, budget, report, artifacts):
    """A fresh adapter after cold qualification, never a cold-cache benchmark."""
    from runtime_llama.adapters import embedding as embedding_module

    texts = [
        f"Reuse document {i}: whitespace \t Unicode città 🔬 end {i}." for i in range(8)
    ]
    reuse_config = dict(config, embedding_preparation_cache_enabled=True)
    adapter = runtime.build_embedding_adapter(reuse_config, client=client)
    report.update(
        mode="separate prepared/raw reuse; fresh adapter; not cold timing", phases={}
    )
    artifacts.update(texts=texts, phases={})
    cache = adapter._preparation_cache
    report["cache_before"] = cache.snapshot()
    require(report["cache_before"]["entries"] == 0, "reuse adapter was not fresh")
    hook_costs = {}

    def measured(original, name):
        def invoke(*args, **kwargs):
            phase, started = budget.phase, time.monotonic()
            try:
                return original(*args, **kwargs)
            finally:
                with budget.lock:
                    costs = hook_costs.setdefault(phase, {}).setdefault(
                        name, {"calls": 0, "sum_worker_wall_sec": 0.0}
                    )
                    costs["calls"] += 1
                    costs["sum_worker_wall_sec"] += time.monotonic() - started

        return invoke

    original_prepare = adapter._engine_prepare
    originals = {
        name: getattr(embedding_module, name)
        for name in ("issue_preparation", "valid_preparation")
    }
    adapter._engine_prepare = measured(original_prepare, "_engine_prepare")
    for name, original in originals.items():
        setattr(embedding_module, name, measured(original, name))
    try:
        budget.phase = "reuse_prepare"
        before = budget.counts["tokenize"]
        started = time.monotonic()
        prepared = adapter.prepare_texts(
            texts, input_type="raw", timeout=min(60, budget.remaining())
        )
        require(len(prepared) == 8, "reuse preparation cardinality mismatch")
        require(
            budget.counts["tokenize"] - before == 8,
            "fresh reuse preparation did not tokenize exactly eight inputs",
        )
        report["prepare_wall_sec"] = time.monotonic() - started
        report["prepare_request_counts"] = phase_counts(budget, budget.phase)
        require(
            report["prepare_request_counts"].get("embedding", 0) == 0,
            "public preparation sent embedding HTTP",
        )
        artifacts["token_counts"] = [item.token_count for item in prepared]
        reference = None
        for phase, public_records in (
            ("reuse_prepared_1", True),
            ("reuse_prepared_2", True),
            ("reuse_raw_1", False),
            ("reuse_raw_2", False),
        ):
            budget.phase = phase
            raw = budget.phase_raw[phase] = {}
            budget.phase_positions[phase] = {
                text: (0, i) for i, text in enumerate(texts)
            }
            before, started = budget.counts["tokenize"], time.monotonic()
            outputs = (
                adapter.embed_prepared(prepared, timeout=min(60, budget.remaining()))
                if public_records
                else adapter.embed_texts(
                    texts, input_type="raw", timeout=min(60, budget.remaining())
                )
            )
            require(
                budget.counts["tokenize"] == before,
                "prepared/raw reuse repeated tokenization",
            )
            result = validate_pipeline(texts, outputs, raw)
            comparisons = [compare(a, b) for a, b in zip(reference or outputs, outputs)]
            require(
                all(item["passed"] for item in comparisons),
                "reuse numerical thresholds failed",
            )
            reference = outputs if reference is None else reference
            result.update(
                wall_sec=time.monotonic() - started,
                request_counts=phase_counts(budget, phase),
                comparisons=comparisons,
            )
            report["phases"][phase] = result
            artifacts["phases"][phase] = {"outputs": outputs, "raw_by_text": raw}
        report["cache_after_reuse"] = cache.snapshot()
    finally:
        adapter._engine_prepare = original_prepare
        for name, original in originals.items():
            setattr(embedding_module, name, original)
        adapter.close()
        report["cache_after_close"] = cache.snapshot()
        require(
            report["cache_after_close"]["closed"]
            and report["cache_after_close"]["entries"]
            == report["cache_after_close"]["bytes"]
            == 0,
            "reuse cache did not drain on close",
        )
        report["hook_costs"] = hook_costs


def run(args):
    import runtime_llama
    from runtime_llama.client import LlamaCppClient
    from runtime_llama.replica_discovery import discover_embedding_pool

    distribution = importlib.metadata.distribution("runtime-llama")
    module = Path(runtime_llama.__file__).resolve()
    require(
        not module.is_relative_to(Path(__file__).resolve().parents[1] / "src"),
        "installed wheel required; checkout source imported",
    )
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    require(
        not direct.get("dir_info", {}).get("editable"),
        "editable installation is not a wheel qualification",
    )
    require(
        distribution.version == runtime_llama.__version__,
        "distribution and import versions differ",
    )
    report = {
        "schema": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "version": runtime_llama.__version__,
        "distribution_version": distribution.version,
        "http_connection_policy": getattr(args, "connection_policy", "close"),
        "installed_archive_sha256": direct.get("archive_info", {})
        .get("hashes", {})
        .get("sha256"),
        "thresholds": {
            "dimension": DIMENSION,
            "max_abs": MAX_ABS,
            "min_cosine": MIN_COSINE,
            "norm": MAX_NORM,
            "raw_projection": MAX_PROJECTION,
        },
        "scope": "Synthetic client/HTTP qualification; no pure GPU timing or retrieval-quality claim",
        "runtime_admin_actions": False,
        "consumer_changes": False,
        "qualification_mode": "single-candidate functional; active LLM allowed; no version throughput comparison",
        "canary": [],
    }
    budget = Budget(args.wall_timeout)
    client = observed_client_type(LlamaCppClient)(
        args.base_url,
        timeout=20,
        budget=budget,
        connection_policy=getattr(args, "connection_policy", "close"),
    )
    adapter, outputs, aliases, data = None, [], [], corpus(args.prefix)
    windows = source_windows(data)
    budget.source_positions = {
        text: (window_index, input_index)
        for window_index, window in enumerate(windows)
        for input_index, text in enumerate(window)
    }
    delivered_sizes = []
    artifacts = {"texts": data, "outputs": outputs, "raw_by_text": budget.raw}
    try:
        catalog = client.models()
        pool = discover_embedding_pool(catalog, args.model)
        require(
            pool.identity is not None and len(pool.replicas) == 2,
            "two content-verified loaded replicas required",
        )
        aliases = [r.model for r in pool.replicas]
        report["identity"] = {
            "content_sha256": pool.identity.content_sha256,
            "embedding_sha256": pool.identity.embedding_sha256,
        }
        report["before"] = snapshot(client, aliases, catalog)
        builds = {
            digest(item["build"]) if item["build"] is not None else None
            for item in report["before"]["models"].values()
        }
        if len(builds) == 1 and None not in builds:
            budget.compatible_aliases = frozenset(aliases)
        budget.phase = "canary"
        canaries = []
        for alias in aliases:
            ids = [
                client.tokenize(
                    text, model=alias, add_special=True, parse_special=True
                )["tokens"]
                for text in data[:8]
            ]
            for label, inputs in (("text", data[:8]), ("tokens", ids)):
                vectors = raw_vectors(client.embeddings(inputs, model=alias), 8)
                canaries.append((alias, label, vectors))
        artifacts["canary"] = canaries
        reference = [normalized(v) for v in canaries[0][2]]
        for alias, label, vectors in canaries[1:]:
            comparisons = [
                compare(a, normalized(b)) for a, b in zip(reference, vectors)
            ]
            report["canary"].append(
                {"model": alias, "input_format": label, "comparisons": comparisons}
            )
        require(
            all(c["passed"] for row in report["canary"] for c in row["comparisons"]),
            "canary numerical thresholds failed",
        )
        config = {
            "base_url": args.base_url,
            "model": args.model,
            "http_connection_policy": getattr(args, "connection_policy", "close"),
            "embedding_replica_discovery": True,
            "batch_size": 256,
            "embedding_request_parallelism": 2,
            "embedding_queue_size": 2,
            "token_count_parallelism": 2,
            "normalize": True,
            "dimension": DIMENSION,
            "timeout": 150,
        }
        adapter = runtime_llama.build_embedding_adapter(config, client=client)
        report["effective_batch_wait_sec"] = adapter.embedding_batch_wait_sec
        require(
            adapter.embedding_batch_wait_sec == 0,
            "functional queue qualification requires default zero wait",
        )
        budget.phase = "oversize"
        report["oversize"] = oversize_canary(
            adapter, budget, data[-1] * 100, runtime_llama.LLMBackendError
        )
        prep = {"calls": 0, "sum_worker_wall_sec": 0.0}
        name = (
            "_engine_prepare" if hasattr(adapter, "_engine_prepare") else "_prepare_job"
        )
        original = getattr(adapter, name)

        def prepare(*positional, **kwargs):
            started = time.monotonic()
            try:
                return original(*positional, **kwargs)
            finally:
                with budget.lock:
                    prep["calls"] += 1
                    prep["sum_worker_wall_sec"] += time.monotonic() - started

        setattr(adapter, name, prepare)
        if hasattr(adapter, "_preparation_cache"):
            report["preparation_cache_before_pipeline"] = (
                adapter._preparation_cache.snapshot()
            )
        budget.phase = "pipeline"

        async def ingest():
            async def incoming():
                for window in windows:
                    yield window

            async with adapter.astream_texts(
                incoming(),
                input_type="raw",
                max_pending_windows=PENDING_WINDOWS,
                timeout=min(150, budget.remaining()),
            ) as stream:
                async for window in stream:
                    delivered_sizes.append(len(window))
                    outputs.extend(window)

        started = time.monotonic()
        asyncio.run(ingest())
        report["pipeline_wall_sec"] = time.monotonic() - started
        report["preparation"] = {"hook": name, **prep}
        report["pipeline"] = validate_pipeline(data, outputs, budget.raw)
        comparisons = [compare(a, b) for a, b in zip(reference, outputs[:8])]
        require(
            all(item["passed"] for item in comparisons),
            "pipeline/canary numerical thresholds failed",
        )
        report["pipeline"]["canary_comparisons"] = comparisons
        report["pipeline"]["request_counts"] = phase_counts(budget, "pipeline")
        require(
            report["pipeline"]["request_counts"].get("tokenize", 0) == 511,
            "cold pipeline did not tokenize exactly 511 unique inputs",
        )
        require(
            report["pipeline"]["request_counts"].get("metadata", 0) < 50,
            "cold pipeline metadata did not improve below baseline 50",
        )
        report["pipeline"].update(
            audit_windows(windows, delivered_sizes, budget.records)
        )
        require(
            {row["model"] for row in budget.raw.values()} == set(aliases),
            "pipeline did not use both replicas",
        )
        if hasattr(adapter, "_engine"):
            report["engine"] = adapter._engine.snapshot()
        report["scheduler"] = adapter._scheduler.snapshot()
        if hasattr(adapter, "_preparation_cache"):
            report["preparation_cache"] = adapter._preparation_cache.snapshot()
        report["cold_counts_before_reuse"] = dict(budget.counts)
        if getattr(args, "qualify_reuse", False):
            report["reuse"], artifacts["reuse"] = {}, {}
            qualify_reuse(
                runtime_llama,
                config,
                client,
                budget,
                report["reuse"],
                artifacts["reuse"],
            )
        report["result"] = "passed"
    except Exception as exc:
        report.update(
            result="failed", failure={"type": type(exc).__name__, "message": str(exc)}
        )
    finally:
        if adapter is not None:
            adapter.close()
        try:
            budget.phase = "postflight"
            report["after"] = snapshot(client, aliases) if aliases else None
            report["profiles_unchanged"] = bool(
                aliases
                and report["before"]["models"] == report["after"]["models"]
                and report["before"]["catalog_profile_sha256"]
                == report["after"]["catalog_profile_sha256"]
            )
            if not report["profiles_unchanged"]:
                report["result"] = "failed"
        except Exception as exc:
            report.update(result="failed", postflight_failure=type(exc).__name__)
        client.close()
    report.update(
        total_wall_sec=time.monotonic() - budget.started,
        corpus_sha256=digest(data),
        wall_timeout_sec=args.wall_timeout,
        budgets=budget.limits,
        request_counts=dict(budget.counts),
        request_peaks=dict(budget.peaks),
        requests=budget.records,
        active_at_finish=dict(budget.active),
    )
    return report, artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument(
        "--connection-policy",
        choices=("close", "reuse"),
        default="close",
        help="qualify the selected HTTP policy without fallback or changing defaults",
    )
    parser.add_argument(
        "--qualify-reuse",
        action="store_true",
        help="after cold 511, separately verify prepared and raw ID reuse",
    )
    parser.add_argument("--wall-timeout", type=float, default=180)
    args = parser.parse_args(argv)
    if not args.allow_live:
        parser.error("explicit --allow-live is required")
    url = urlsplit(args.base_url)
    try:
        port = url.port
    except ValueError:
        parser.error("invalid HTTP(S) endpoint")
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or "?" in args.base_url
        or "#" in args.base_url
        or "\\" in args.base_url
        or any(c.isspace() for c in args.base_url)
    ):
        parser.error("HTTP(S) endpoint must omit credentials, query and fragment")
    del port
    if not math.isfinite(args.wall_timeout) or not 0 < args.wall_timeout <= 180:
        parser.error("wall timeout must be finite and within (0, 180]")
    vectors = args.output.with_suffix(".vectors.json")
    if args.output.exists() or vectors.exists():
        parser.error("output files must not already exist")
    report, artifacts = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vectors.write_bytes(canonical(artifacts))
    report["vectors_file"] = vectors.name
    report["vectors_file_sha256"] = hashlib.sha256(vectors.read_bytes()).hexdigest()
    args.output.write_bytes(canonical(report))
    print(report["result"], report["version"], args.output)
    return 0 if report["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
