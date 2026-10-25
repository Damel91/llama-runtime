#!/usr/bin/env python3
"""Explicitly opted-in qualification of externally loaded embedding replicas.

Imports before the --live gate use the standard library only. No runtime
administration, consumer modification, automatic retry or server bootstrap is
performed. The corpus and prompts are public synthetic qualification inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
from threading import Event, Lock, Thread
import time
from typing import Any
from urllib.parse import urlencode, urlsplit


DIMENSION = 768
MAX_ABS_ERROR = 1e-5
MIN_COSINE = 0.999999
CONTEXT = 512
SLOTS = 4
BATCH_SIZE = 8
BATCH_TOKEN_LIMIT = 480
_ROW = re.compile(r"Synthetic document ([0-9]+):")


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def base_url(value: str) -> str:
    raw = value.strip()
    try:
        parsed = urlsplit(raw)
        hostname = parsed.hostname
        parsed.port
    except ValueError:
        raise argparse.ArgumentTypeError("invalid HTTP(S) base URL") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or "?" in raw
        or "#" in raw
        or "\\" in raw
        or any(character.isspace() or ord(character) < 32 for character in raw)
    ):
        raise argparse.ArgumentTypeError(
            "base URL requires HTTP(S), without credentials/query/fragment"
        )
    return raw.rstrip("/")


def positive(value: str) -> int:
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("a positive integer is required") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("a positive integer is required")
    return result


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def corpus_text(index: int) -> str:
    revision = (index * 7919 + 104729) % 1_000_003
    return (
        f"Synthetic document {index:05d}: archive revision {revision} stores "
        f"packet {index % 997} in queue {index % 31}. The worker validates "
        "ordered records, checks finite vectors, and commits complete results. "
        "Preserve each document identity through concurrent processing."
    )


def corpus_index(text: str) -> int:
    match = _ROW.match(text)
    require(match is not None, "non-corpus embedding input")
    index = int(match[1])
    require(text == corpus_text(index), "corpus identity mismatch")
    return index


def validate_vector(vector: Any) -> list[float]:
    require(isinstance(vector, list) and len(vector) == DIMENSION, "vector dimension")
    require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in vector
        ),
        "non-finite embedding vector",
    )
    require(
        abs(math.sqrt(math.fsum(value * value for value in vector)) - 1) <= 1e-6,
        "embedding normalization",
    )
    return vector


def measure_vectors(reference: list[float], candidate: list[float]) -> dict:
    """Measure first; structural errors still fail immediately."""
    validate_vector(reference)
    validate_vector(candidate)
    absolute = max(abs(a - b) for a, b in zip(reference, candidate))
    norm_a = math.sqrt(math.fsum(value * value for value in reference))
    norm_b = math.sqrt(math.fsum(value * value for value in candidate))
    cosine = math.fsum(a * b for a, b in zip(reference, candidate)) / (norm_a * norm_b)
    cosine = min(1.0, cosine)
    return {
        "maximum_absolute_error": absolute,
        "cosine": cosine,
        "absolute_error_limit": MAX_ABS_ERROR,
        "cosine_minimum": MIN_COSINE,
        "passed": absolute <= MAX_ABS_ERROR and cosine >= MIN_COSINE,
    }


class VectorComparisonError(ValueError):
    def __init__(self, measurements: dict):
        self.measurements = measurements
        condition = (
            "absolute-error"
            if measurements["maximum_absolute_error"] > MAX_ABS_ERROR
            else "cosine"
        )
        super().__init__(
            f"embedding {condition} tolerance: "
            f"max_abs={measurements['maximum_absolute_error']:.12g}; "
            f"cosine={measurements['cosine']:.12g}"
        )


def compare_vectors(
    reference: list[float], candidate: list[float]
) -> tuple[float, float]:
    measured = measure_vectors(reference, candidate)
    if not measured["passed"]:
        raise VectorComparisonError(measured)
    return measured["maximum_absolute_error"], measured["cosine"]


def same_request_projection(raw: Any, projected: list[float]) -> None:
    """Verify association against this request, independently of GPU repeatability."""
    require(
        isinstance(raw, list) and len(raw) == DIMENSION, "same-request raw dimension"
    )
    require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in raw
        ),
        "same-request raw finite values",
    )
    validate_vector(projected)
    norm = math.sqrt(math.fsum(value * value for value in raw))
    require(math.isfinite(norm) and norm > 0, "same-request raw norm")
    require(
        max(abs(actual - expected / norm) for actual, expected in zip(projected, raw))
        <= 1e-12,
        "same-request output order or projection mismatch",
    )


def snapshot(client: Any, aliases: list[str], replicas: list[str], build: str) -> dict:
    health = client.health(timeout=30, cancel_wait=True)
    require(isinstance(health, dict), "health payload")
    catalog = client.models(timeout=30, cancel_wait=True)
    data = catalog.get("data") if isinstance(catalog, dict) else None
    require(isinstance(data, list), "catalog payload")
    selected = {}
    for alias in aliases:
        matches = [
            item for item in data if isinstance(item, dict) and item.get("id") == alias
        ]
        require(len(matches) == 1, "selected alias must be unique")
        item = matches[0]
        require(
            item.get("status", {}).get("value") == "loaded", "selected alias not loaded"
        )
        props = client.props(model=alias, timeout=30, cancel_wait=True)
        require(isinstance(props, dict), "props payload")
        require(props.get("build_info") == build, "native runtime build mismatch")
        context = props.get("default_generation_settings", {}).get("n_ctx")
        slots = client.request(
            "GET", "/slots?" + urlencode({"model": alias}), timeout=30, cancel_wait=True
        )
        require(isinstance(slots, list) and bool(slots), "slots payload")
        require(
            all(
                isinstance(slot, dict) and slot.get("is_processing") is False
                for slot in slots
            ),
            "selected model has busy or unknown slots",
        )
        if alias in replicas:
            require(context == CONTEXT, "embedding loaded context mismatch")
            require(
                props.get("total_slots") == SLOTS and len(slots) == SLOTS,
                "embedding slot count",
            )
            require(
                item.get("meta", {}).get("n_embd") == DIMENSION,
                "embedding metadata dimension",
            )
        selected[alias] = {
            "status": "loaded",
            "build": props["build_info"],
            "context": context,
            "total_slots": props.get("total_slots"),
            "idle_slots": len(slots),
            "catalog_meta": item.get("meta"),
            "props_sha256": sha(canonical(props)),
            "model_path_sha256": sha(str(props.get("model_path", "")).encode()),
            "chat_template_sha256": sha(str(props.get("chat_template", "")).encode()),
        }
    return {"health_reachable": True, "models": selected}


def invariant_snapshots(before: dict, after: dict) -> None:
    require(set(before["models"]) == set(after["models"]), "selected model set changed")
    for alias, original in before["models"].items():
        current = after["models"][alias]
        require(
            original["props_sha256"] == current["props_sha256"],
            "model properties changed",
        )
        require(
            original["catalog_meta"] == current["catalog_meta"],
            "model metadata changed",
        )
        require(current["status"] == "loaded", "model unloaded")
        require(current["idle_slots"] == original["idle_slots"], "model slots changed")


class Activity:
    """Client-call intervals, not a claim about simultaneous GPU computation."""

    def __init__(self, path: Path, caps: dict[str, int]):
        self.lock = Lock()
        self.file = path.open("x")
        self.caps = caps
        self.active = Counter()
        self.maximum = Counter()
        self.global_maximum = 0
        self.llm_active = 0
        self.mixed_overlap = 0
        self.calls = Counter()
        self.tokens = Counter()
        self.token_calls = 0
        self.minimum_input_tokens: int | None = None
        self.maximum_input_tokens = 0
        self.embedding_items = 0
        self.coverage_sum = 0
        self.coverage_xor = 0
        self.sequence = 0
        self.raw_vectors = {}
        self.same_request_checks = 0

    def begin(self, kind: str, model: str, indices=()) -> tuple[int, float]:
        now = time.monotonic()
        with self.lock:
            self.sequence += 1
            sequence = self.sequence
            if kind == "embedding":
                require(
                    model in self.caps, "embedding request selected unexpected alias"
                )
                self.active[model] += 1
                self.maximum[model] = max(self.maximum[model], self.active[model])
                self.global_maximum = max(
                    self.global_maximum, sum(self.active.values())
                )
                require(self.active[model] <= self.caps[model], "replica request limit")
                require(sum(self.active.values()) <= 4, "global request limit")
                self.calls[model] += 1
                if self.llm_active:
                    self.mixed_overlap += 1
            else:
                self.llm_active += 1
                self.mixed_overlap += int(bool(sum(self.active.values())))
            self.file.write(
                json.dumps(
                    {
                        "event": "start",
                        "id": sequence,
                        "kind": kind,
                        "model": model,
                        "indices": list(indices),
                        "monotonic": now,
                    }
                )
                + "\n"
            )
            self.file.flush()
        return sequence, now

    def end(
        self, token: tuple[int, float], kind: str, model: str, success: bool
    ) -> None:
        now = time.monotonic()
        with self.lock:
            if kind == "embedding":
                self.active[model] -= 1
            else:
                self.llm_active -= 1
            self.file.write(
                json.dumps(
                    {
                        "event": "end",
                        "id": token[0],
                        "kind": kind,
                        "model": model,
                        "success": success,
                        "elapsed_seconds": now - token[1],
                        "monotonic": now,
                    }
                )
                + "\n"
            )
            self.file.flush()

    def summary(self) -> dict:
        with self.lock:
            return {
                "embedding_requests": dict(self.calls),
                "maximum_embedding_requests": self.global_maximum,
                "maximum_per_replica": dict(self.maximum),
                "four_requests_overlap_observed": self.global_maximum >= 4,
                "mixed_overlap_observations": self.mixed_overlap,
                "tokenization_requests": self.token_calls,
                "minimum_input_tokens": self.minimum_input_tokens,
                "maximum_input_tokens": self.maximum_input_tokens,
                "response_associations_checked": self.embedding_items,
                "coverage_index_sum": self.coverage_sum,
                "coverage_index_xor": self.coverage_xor,
                "active_at_finish": sum(self.active.values()) + self.llm_active,
                "same_request_order_projection_checks": self.same_request_checks,
                "unconsumed_raw_vectors": len(self.raw_vectors),
            }

    def close(self) -> None:
        self.file.close()


class ObservedClient:
    """Wrap the public injection seam; preserve native response fields."""

    def __init__(self, client: Any, activity: Activity):
        self.client, self.activity = client, activity

    def __getattr__(self, name):
        return getattr(self.client, name)

    def tokenize(self, content, **kwargs):
        payload = self.client.tokenize(content, **kwargs)
        count = len(payload["tokens"])
        require(0 < count <= CONTEXT, "tokenized input context")
        with self.activity.lock:
            self.activity.tokens[(kwargs["model"], content)] = count
            self.activity.token_calls += 1
            current = self.activity.minimum_input_tokens
            self.activity.minimum_input_tokens = (
                count if current is None else min(current, count)
            )
            self.activity.maximum_input_tokens = max(
                self.activity.maximum_input_tokens, count
            )
        return payload

    def embeddings(self, input, **kwargs):
        texts = list(input)
        indices = [corpus_index(text) for text in texts]
        model = kwargs["model"]
        with self.activity.lock:
            count = sum(self.activity.tokens[(model, text)] for text in texts)
        require(count <= BATCH_TOKEN_LIMIT, "embedding packing token budget")
        token = self.activity.begin("embedding", model, indices)
        success = False
        try:
            payload = self.client.embeddings(texts, **kwargs)
            data = payload.get("data") if isinstance(payload, dict) else None
            require(
                isinstance(data, list) and len(data) == len(texts),
                "raw embedding cardinality",
            )
            response_indices = [
                item.get("index", position) for position, item in enumerate(data)
            ]
            require(
                all(
                    isinstance(index, int) and not isinstance(index, bool)
                    for index in response_indices
                )
                and sorted(response_indices) == list(range(len(texts))),
                "raw embedding input association",
            )
            with self.activity.lock:
                for item, position in zip(data, response_indices):
                    corpus_position = indices[position]
                    require(
                        corpus_position not in self.activity.raw_vectors,
                        "duplicate same-request corpus position",
                    )
                    self.activity.raw_vectors[corpus_position] = item.get("embedding")
                self.activity.embedding_items += len(indices)
                self.activity.coverage_sum += sum(indices)
                for index in indices:
                    self.activity.coverage_xor ^= index
                for text in texts:
                    del self.activity.tokens[(model, text)]
            success = True
            return payload
        finally:
            self.activity.end(token, "embedding", model, success)

    def chat_completion(self, *args, **kwargs):
        upstream = self.client.chat_completion(*args, **kwargs)
        model = kwargs["model"]

        def stream():
            token = self.activity.begin("llm", model)
            success = False
            try:
                yield from upstream
                success = True
            finally:
                close = getattr(upstream, "close", None)
                if callable(close):
                    close()
                self.activity.end(token, "llm", model, success)

        require(
            kwargs.get("stream") is True,
            "qualification LLM must use streaming transport",
        )
        return stream()


def embedding_profile(args, caps: dict[str, int]) -> dict:
    return {
        "model": next(iter(caps)),
        "base_url": args.base_url,
        "role_id": "replica-qualification",
        "batch_size": BATCH_SIZE,
        "batch_token_limit": BATCH_TOKEN_LIMIT,
        "max_input_tokens": CONTEXT,
        "dimension": DIMENSION,
        "normalize": True,
        "token_count_parallelism": 1,
        # Pin the historical qualification profile across library defaults.
        "embedding_batch_wait_sec": 0.5,
        "embedding_request_parallelism": 4,
        "embedding_queue_size": 8,
        "embedding_replicas": [
            {"model": alias, "max_in_flight": cap} for alias, cap in caps.items()
        ],
        "stream_idle_timeout_sec": 30,
    }


def llm_output(result: Any, *, records: list | None = None) -> dict:
    record = {
        "text": result.text,
        "terminal_reason": result.terminal_reason,
        "usage": asdict(result.usage) if result.usage is not None else None,
        "usage_available": result.usage is not None,
        "usage_unavailable_reason": (
            "client_visible_output_cap_before_provider_terminal"
            if result.usage is None and result.terminal_reason == "visible_output_cap"
            else "provider_usage_not_observed"
            if result.usage is None
            else None
        ),
    }
    if records is not None:
        records.append(record)
    require(
        result.terminal_reason in {"completed", "length_cap", "visible_output_cap"},
        "LLM terminal reason",
    )
    require(bool(result.text.strip()), "LLM visible output absent")
    reported_usage = (
        result.usage is not None and result.usage.completion_tokens is not None
    )
    require(
        reported_usage
        or (result.terminal_reason == "visible_output_cap" and result.usage is None),
        "LLM usage absent",
    )
    require(
        "<think>" not in result.text and "</think>" not in result.text,
        "thinking leaked into visible output",
    )
    return record


def run_phase(
    args,
    root: Path,
    report: dict,
    label: str,
    caps: dict[str, int],
    *,
    baseline: bool,
    mixed: bool,
    runtime: dict,
) -> dict:
    activity = Activity(root / (label + "-activity.jsonl"), caps)
    client = runtime["client"](args.base_url, timeout=30)
    adapter = runtime["embedding"](
        embedding_profile(args, caps), client=ObservedClient(client, activity)
    )
    phase = {
        "name": label,
        "configuration": embedding_profile(args, caps),
        "processed": 0,
        "result": "running",
        "maximum_absolute_error": 0.0,
        "minimum_cosine": 1.0,
        "oracle": "baseline" if baseline else "same_corpus_replica_equivalence",
        "numeric_failures": 0,
        "numeric_failure_samples": [],
        "llm_calls": [],
    }
    report["phases"].append(phase)
    error = []
    llm_client = None
    llm_adapter = None
    llm_thread = None
    start_llm = Event()
    stop_llm = Event()

    def mixed_worker():
        try:
            require(start_llm.wait(timeout=30), "mixed load did not start")
            for index in range(3):
                if stop_llm.is_set():
                    return
                result = llm_adapter.generate_with_metadata(
                    f"For synthetic pipeline case {index}, describe the validation, indexing, recovery and publication steps in one clear paragraph of about one hundred words. Use distinct complete sentences.",
                    num_predict=128,
                    timeout=30,
                    cancellation_probe=stop_llm.is_set,
                )
                llm_output(result, records=phase["llm_calls"])
        except Exception as exc:
            error.append(exc)

    started = time.monotonic()
    reference_path = root / "alias0-reference.f64le"
    mode = "xb" if baseline else "rb"
    try:
        if mixed:
            llm_client = runtime["client"](args.base_url, timeout=30)
            llm_adapter = runtime["llm"](
                {
                    "model": args.llm_model,
                    "base_url": args.base_url,
                    "role_id": "mixed-qualification",
                    "reasoning": "off",
                    "temperature": 0.0,
                    "stream_idle_timeout_sec": 30,
                },
                client=ObservedClient(llm_client, activity),
            )
            llm_thread = Thread(
                target=mixed_worker, name="qualification-mixed-llm", daemon=True
            )
            llm_thread.start()
        with reference_path.open(mode) as reference:
            for offset in range(0, args.count, args.window):
                stop = min(args.count, offset + args.window)
                texts = [corpus_text(index) for index in range(offset, stop)]
                if mixed:
                    start_llm.set()
                vectors = adapter.embed_texts(texts, timeout=30)
                require(len(vectors) == len(texts), "embedding output cardinality")
                for index, vector in enumerate(vectors, offset):
                    validate_vector(vector)
                    with activity.lock:
                        require(
                            index in activity.raw_vectors, "same-request vector missing"
                        )
                        raw_vector = activity.raw_vectors.pop(index)
                    same_request_projection(raw_vector, vector)
                    with activity.lock:
                        activity.same_request_checks += 1
                    require(
                        corpus_index(texts[index - offset]) == index,
                        "window input association",
                    )
                    if baseline:
                        reference.write(struct.pack("<" + "d" * DIMENSION, *vector))
                    else:
                        raw = reference.read(DIMENSION * 8)
                        require(len(raw) == DIMENSION * 8, "reference vector missing")
                        original = list(struct.unpack("<" + "d" * DIMENSION, raw))
                        measured = measure_vectors(original, vector)
                        absolute, cosine = (
                            measured["maximum_absolute_error"],
                            measured["cosine"],
                        )
                        phase["maximum_absolute_error"] = max(
                            phase["maximum_absolute_error"], absolute
                        )
                        phase["minimum_cosine"] = min(phase["minimum_cosine"], cosine)
                        report["compared_vectors"] = (
                            report.get("compared_vectors", 0) + 1
                        )
                        if not measured["passed"]:
                            phase["numeric_failures"] += 1
                            report["numeric_failures"] = (
                                report.get("numeric_failures", 0) + 1
                            )
                            if len(phase["numeric_failure_samples"]) < 10:
                                phase["numeric_failure_samples"].append(
                                    {"index": index, **measured}
                                )
                phase["processed"] = stop
                if error:
                    raise error[0]
                phase["elapsed_seconds"] = time.monotonic() - started
                write_json(args.output, report)
                print(
                    json.dumps(
                        {
                            "phase": label,
                            "processed": stop,
                            "count": args.count,
                            "elapsed_seconds": round(phase["elapsed_seconds"], 3),
                        }
                    ),
                    flush=True,
                )
            require(baseline or not reference.read(1), "reference cardinality mismatch")
        if llm_thread is not None:
            llm_thread.join()
            if error:
                raise error[0]
            require(len(phase["llm_calls"]) == 3, "mixed LLM call count")
        phase["result"] = "processed"
    except BaseException:
        phase["result"] = "failed"
        raise
    finally:
        stop_llm.set()
        start_llm.set()
        adapter.close()
        client.close()
        if llm_thread is not None:
            llm_thread.join()
        if llm_adapter is not None:
            llm_adapter.close()
        if llm_client is not None:
            llm_client.close()
        phase["elapsed_seconds"] = time.monotonic() - started
        phase["documents_per_second_end_to_end"] = phase["processed"] / max(
            phase["elapsed_seconds"], 1e-9
        )
        phase["activity"] = activity.summary()
        activity.close()
    require(phase["activity"]["active_at_finish"] == 0, "active transport at phase end")
    require(
        phase["activity"]["response_associations_checked"] == args.count,
        "embedding association coverage",
    )
    require(
        phase["activity"]["same_request_order_projection_checks"] == args.count
        and phase["activity"]["unconsumed_raw_vectors"] == 0,
        "same-request projection coverage",
    )
    require(
        phase["activity"]["coverage_index_sum"] == args.count * (args.count - 1) // 2,
        "embedding index coverage sum",
    )
    expected_xor = 0
    for index in range(args.count):
        expected_xor ^= index
    require(
        phase["activity"]["coverage_index_xor"] == expected_xor,
        "embedding index coverage xor",
    )
    # Native slots do not multiply HTTP admission on a single replica.
    require(
        phase["activity"]["maximum_embedding_requests"] == min(len(caps), 4),
        "embedding overlap does not match the selected replica capacity",
    )
    if mixed:
        require(
            phase["activity"]["mixed_overlap_observations"] > 0,
            "no LLM and embedding overlap observed",
        )
    phase["result"] = "failed" if phase["numeric_failures"] else "passed"
    return phase


def reference_anchors(args, root: Path, runtime: dict) -> list[dict]:
    """Single-input requests independently anchor three baseline positions."""
    adapter = runtime["embedding"](embedding_profile(args, {args.replica[0]: 4}))
    records = []
    try:
        with (root / "alias0-reference.f64le").open("rb") as reference:
            for index in sorted({0, args.count // 2, args.count - 1}):
                reference.seek(index * DIMENSION * 8)
                raw = reference.read(DIMENSION * 8)
                require(len(raw) == DIMENSION * 8, "reference anchor missing")
                expected = list(struct.unpack("<" + "d" * DIMENSION, raw))
                observed = adapter.embed_query(corpus_text(index), timeout=30)
                measured = measure_vectors(expected, observed)
                records.append(
                    {
                        "index": index,
                        "text_sha256": sha(corpus_text(index).encode()),
                        "oracle": "independent_single_input_vs_batch_association_anchor",
                        "reference_batch_size_configured": BATCH_SIZE,
                        "candidate_batch_size": 1,
                        "result": "passed" if measured["passed"] else "failed",
                        **measured,
                    }
                )
    finally:
        adapter.close()
    return records


def validate_thinking(record: dict, *, source: str, model: str, role: str) -> None:
    diagnostic = record["diagnostic"]
    require(record["sink_records"] == 1, "thinking sink cardinality")
    require(
        diagnostic["model"] == model and diagnostic["role_id"] == role,
        "thinking role/model correlation",
    )
    require(bool(diagnostic["call_id"]), "thinking call identity absent")
    require(
        source in diagnostic["sources"] and bool(diagnostic["text"]),
        "requested thinking channel unsupported or empty",
    )
    require(
        diagnostic["captured_chars"] == len(diagnostic["text"]) <= 256,
        "thinking capture bound",
    )
    require(
        diagnostic["observed_chars"] >= diagnostic["captured_chars"],
        "thinking observed/captured mismatch",
    )
    require(
        diagnostic["truncated"] is (diagnostic["observed_chars"] > 256),
        "thinking truncation declaration",
    )
    require(
        diagnostic["terminal_reason"] == record["terminal_reason"],
        "thinking terminal correlation",
    )
    expected_status = (
        "completed" if record["terminal_reason"] == "completed" else "interrupted"
    )
    require(diagnostic["status"] == expected_status, "thinking status")
    require(
        "<think>" not in record["text"] and "</think>" not in record["text"],
        "visible inline thinking leaked",
    )
    if source == "inline":
        require(
            record["reasoning_text"] == "", "inline changed reasoning_text contract"
        )
    else:
        require(bool(record["reasoning_text"]), "separated reasoning_text absent")


def thinking_cases(args, runtime: dict, report: dict) -> None:
    call_ids = set()
    for reasoning_format, source in (("deepseek", "separated"), ("none", "inline")):
        for streaming in (False, True):
            role = f"thinking-{source}-{'stream' if streaming else 'blocking'}"
            adapter = runtime["llm"](
                {
                    "model": args.llm_model,
                    "base_url": args.base_url,
                    "role_id": role,
                    "reasoning": "on",
                    "temperature": 0.0,
                    "stream_idle_timeout_sec": 30,
                }
            )
            sink = []
            options = {
                "num_predict": 256,
                "timeout": 30,
                "request_options": {"reasoning_format": reasoning_format},
                "thinking_diagnostics": runtime["thinking"](
                    enabled=True, max_chars=256
                ),
                "thinking_sink": sink.append,
            }
            prompt = "A synthetic archive has 17 packets with 23 records each, and 59 records are removed. Work through the arithmetic carefully, then give the remaining record count in one short final sentence."
            try:
                if streaming:
                    stream = adapter.stream_with_metadata(prompt, **options)
                    visible, reasoning, terminals = [], [], []
                    try:
                        for event in stream:
                            if event.kind == "content":
                                visible.append(event.text)
                            elif event.kind == "reasoning":
                                reasoning.append(event.text)
                            elif event.kind == "terminal":
                                terminals.append(event)
                    finally:
                        stream.close()
                    require(len(terminals) == 1, "thinking stream terminal cardinality")
                    terminal = terminals[0]
                    require(
                        terminal.thinking_diagnostic is not None,
                        "thinking stream diagnostic absent",
                    )
                    record = {
                        "text": "".join(visible),
                        "reasoning_text": "".join(reasoning),
                        "terminal_reason": terminal.terminal_reason,
                        "usage": asdict(terminal.usage) if terminal.usage else None,
                        "diagnostic": terminal.thinking_diagnostic.to_dict(),
                    }
                else:
                    result = adapter.generate_with_metadata(prompt, **options)
                    require(
                        result.thinking_diagnostic is not None,
                        "thinking result diagnostic absent",
                    )
                    record = {
                        **llm_output(result),
                        "reasoning_text": result.reasoning_text,
                        "diagnostic": result.thinking_diagnostic.to_dict(),
                    }
                record.update(
                    {
                        "reasoning_format": reasoning_format,
                        "streaming": streaming,
                        "sink_records": len(sink),
                    }
                )
                report["thinking"].append(record)
                validate_thinking(
                    record, source=source, model=args.llm_model, role=role
                )
                require(
                    record["diagnostic"]["call_id"] not in call_ids,
                    "thinking invocation identity reused",
                )
                call_ids.add(record["diagnostic"]["call_id"])
                require(
                    sink[0].to_dict() == record["diagnostic"],
                    "thinking sink/result mismatch",
                )
                record["usage_available"] = record["usage"] is not None
                record["usage_unavailable_reason"] = (
                    "client_visible_output_cap_before_provider_terminal"
                    if record["usage"] is None
                    and record["terminal_reason"] == "visible_output_cap"
                    else "provider_usage_not_observed"
                    if record["usage"] is None
                    else None
                )
                require(
                    record["usage"] is not None
                    or record["terminal_reason"] == "visible_output_cap",
                    "thinking usage absent",
                )
            finally:
                adapter.close()
            write_json(args.output, report)


def run(args) -> int:
    # These imports are deliberately after CLI opt-in and argument validation.
    import runtime_llama
    from runtime_llama import build_adapter, build_embedding_adapter
    from runtime_llama.client import LlamaCppClient
    from runtime_llama.diagnostics import ThinkingDiagnosticsConfig
    from runtime_llama.errors import sanitize_diagnostic

    require(not args.output.exists(), "refusing to overwrite qualification report")
    root = args.output.parent / (args.output.stem + "-artifacts")
    require(not root.exists(), "refusing to overwrite qualification artifacts")
    root.mkdir(parents=True)
    corpus_digest = hashlib.sha256()
    with (root / "corpus.jsonl").open("xb") as corpus:
        for index in range(args.count):
            data = canonical({"index": index, "text": corpus_text(index)}) + b"\n"
            corpus.write(data)
            corpus_digest.update(data)
    report = {
        "schema": 1,
        "result": "running",
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_package_file": str(Path(runtime_llama.__file__).resolve()),
        "base_url": args.base_url,
        "llm_model": args.llm_model,
        "replicas": args.replica,
        "expected_build": args.expected_build,
        "count": args.count,
        "window": args.window,
        "corpus_sha256": corpus_digest.hexdigest(),
        "vector_comparison": {
            "reference": "alias0-reference.f64le",
            "format": "little-endian float64",
            "dimension": DIMENSION,
            "maximum_absolute_error": MAX_ABS_ERROR,
            "minimum_cosine": MIN_COSINE,
            "every_index_compared": True,
        },
        "measurement_scope": "end-to-end client calls including tokenization and metadata; overlap is HTTP call overlap, not GPU compute overlap",
        "consumer_integration": "not_run",
        "phases": [],
        "thinking": [],
        "numeric_failures": 0,
        "compared_vectors": 0,
    }
    runtime = {
        "client": LlamaCppClient,
        "embedding": build_embedding_adapter,
        "llm": build_adapter,
        "thinking": ThinkingDiagnosticsConfig,
    }
    client = LlamaCppClient(args.base_url, timeout=30)
    aliases = [args.llm_model, *args.replica]
    started = time.monotonic()
    try:
        report["before"] = snapshot(client, aliases, args.replica, args.expected_build)
        write_json(args.output, report)
        phases = [
            ("alias0", {args.replica[0]: 4}, True, False),
            ("alias1", {args.replica[1]: 4}, False, False),
            ("pooled", dict.fromkeys(args.replica, 2), False, False),
            ("pooled-mixed", dict.fromkeys(args.replica, 2), False, True),
        ]
        for label, caps, baseline, mixed in phases:
            run_phase(
                args,
                root,
                report,
                label,
                caps,
                baseline=baseline,
                mixed=mixed,
                runtime=runtime,
            )
            if baseline:
                report["independent_single_input_anchors"] = reference_anchors(
                    args, root, runtime
                )
                report["numeric_failures"] += sum(
                    not record["passed"]
                    for record in report["independent_single_input_anchors"]
                )
        report["reference_sha256"] = file_sha(root / "alias0-reference.f64le")
        thinking_cases(args, runtime, report)
        report["result"] = "failed" if report["numeric_failures"] else "passed"
        if report["numeric_failures"]:
            report["failure"] = {
                "class": "NumericOracleFailure",
                "message": "unchanged strict numerical tolerances exceeded; remaining phases were measured without retry",
                "count": report["numeric_failures"],
            }
    except Exception as exc:
        report["result"] = "failed"
        report["failure"] = {
            "class": type(exc).__name__,
            "message": sanitize_diagnostic(exc, limit=300),
        }
        if isinstance(exc, VectorComparisonError):
            report["failure"]["measurements"] = exc.measurements
    finally:
        try:
            report["after"] = snapshot(
                client, aliases, args.replica, args.expected_build
            )
            if "before" in report:
                invariant_snapshots(report["before"], report["after"])
                report["server_configuration_unchanged"] = True
        except Exception as exc:
            report["result"] = "failed"
            report["after_failure"] = {
                "class": type(exc).__name__,
                "message": sanitize_diagnostic(exc, limit=300),
            }
        finally:
            client.close()
        report["elapsed_seconds"] = time.monotonic() - started
        report["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(args.output, report)
    print(
        json.dumps({"result": report["result"], "report": str(args.output)}), flush=True
    )
    return 0 if report["result"] == "passed" else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="explicitly authorize HTTP inference on the supplied endpoint",
    )
    parser.add_argument("--base-url", type=base_url, required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--replica", action="append", required=True)
    parser.add_argument("--expected-build", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=positive, default=10_000)
    parser.add_argument("--window", type=positive, default=256)
    args = parser.parse_args(argv)
    if not args.live:
        parser.error(
            "explicit --live is required before importing the runtime or accessing HTTP"
        )
    if len(args.replica) != 2 or len(set(args.replica)) != 2:
        parser.error("exactly two distinct --replica aliases are required")
    if args.llm_model in args.replica:
        parser.error("LLM and embedding aliases must be distinct")
    for alias in [args.llm_model, *args.replica, args.expected_build]:
        if (
            not alias.strip()
            or alias != alias.strip()
            or any(ord(c) < 32 for c in alias)
        ):
            parser.error(
                "model aliases and expected build must be nonempty and free of control characters"
            )
    if args.count < 32 or args.window < 32:
        parser.error(
            "count and window must be at least 32 for concurrency qualification"
        )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
