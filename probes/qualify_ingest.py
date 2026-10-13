#!/usr/bin/env python3
"""Opt-in live embedding soak. Preparation performs no network requests.

The selected server and model are externally managed. This program never loads,
unloads, restarts, retries an HTTP failure, or changes inference configuration.
It calls the library's public embedding factory and client injection seam.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None
import struct
import subprocess
import sys
from threading import Event, Lock, Thread, active_count
import time
import traceback
from typing import Any
from urllib.parse import parse_qs, urlsplit


DEFAULT_ROOT = Path.cwd() / "qualification-ingest"
DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
EXPECTED_SLOTS = 4
MODEL = "nomic-embed-text-v2-moe.Q8_0.gguf"
EXPECTED_BUILD = "b11482-b86d2f075"
EXPECTED_CONTEXT = 512
DIMENSION = 768
COUNT = 10_000
WINDOW = 128
BATCH_SIZE = 32
BATCH_TOKEN_LIMIT = 480
TOKEN_PARALLELISM = 4


def validate_base_url(value: str) -> str:
    """Reject credentials and URL suffixes before any evidence is written."""
    raw = value.strip()
    try:
        parsed = urlsplit(raw)
        hostname = parsed.hostname
        parsed.port
    except ValueError:
        raise argparse.ArgumentTypeError(
            "base URL must be a valid HTTP(S) URL"
        ) from None
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
            "base URL must use HTTP(S) without credentials, query or fragment"
        )
    return raw


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def vector_bytes(vector: list[float]) -> bytes:
    return struct.pack("<" + "d" * len(vector), *vector)


def normalized(vector: list[float]) -> list[float]:
    scale = max(abs(value) for value in vector)
    if scale == 0:
        return vector
    scaled = [value / scale for value in vector]
    norm = math.sqrt(math.fsum(value * value for value in scaled))
    return [value / norm for value in scaled]


def rss_bytes() -> int:
    # POSIX ps reports resident KiB on both macOS and Linux; no optional psutil.
    return (
        int(
            subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(os.getpid())]
            ).strip()
        )
        * 1024
    )


def process_measurement() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    return {
        "monotonic": time.monotonic(),
        "rss_bytes": rss_bytes(),
        "peak_rss_bytes": int(peak),
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "python_threads": active_count(),
    }


def corpus_row(index: int) -> dict[str, Any]:
    number = (index * 7919 + 104729) % 1_000_003
    category = index % 10
    texts = [
        (
            "italian",
            f"documento {index:05d}: il servizio archivio conserva la revisione {number}. "
            f"La coda elabora {index % 29 + 1} elementi; ogni risultato deve mantenere l'ordine originale. "
            "Un errore transitorio viene registrato senza perdere il lavoro già verificato.",
        ),
        (
            "english",
            f"record {index:05d}: worker {number} validates the manifest and writes an atomic checkpoint. "
            f"The queue contains {index % 47 + 1} items. A failed connection must not publish incomplete vectors "
            "or silently duplicate the document.",
        ),
        (
            "python",
            f"# synthetic case {index:05d}\ndef transform_{number}(items):\n"
            f"    revision = {index % 97}\n    return [{{'id': i, 'value': x, 'rev': revision}} "
            "for i, x in enumerate(items) if x is not None]\n",
        ),
        (
            "javascript",
            f"// synthetic case {index:05d}\nexport const packet{number} = async (items) => {{\n"
            f"  const revision = {index % 89};\n  return Promise.all(items.map(async (x, i) => "
            "({ index: i, key: x.key, revision })));\n};",
        ),
        (
            "sql",
            f"-- synthetic case {index:05d}\nSELECT document_id, revision, status FROM ledger_{index % 31} "
            f"WHERE project_id = {number} AND status IN ('verified', 'pending') "
            "ORDER BY revision DESC, document_id ASC LIMIT 64;",
        ),
        (
            "json",
            json.dumps(
                {
                    "case": f"{index:05d}",
                    "project": number,
                    "limits": {"timeout": 30, "batch": 16},
                    "features": ["stream", "embedding", "unicode"],
                    "label": "qualifica sintetica",
                },
                ensure_ascii=False,
            ),
        ),
        (
            "yaml",
            f"# synthetic case {index:05d}\nservice: archivio-{number}\ntransport: http\n"
            f"queue:\n  workers: {index % 4 + 1}\n  preserve_order: true\n"
            "recovery:\n  checkpoint: atomic\n  publish_partial: false\n",
        ),
        (
            "unicode",
            f"caso {index:05d} / révision {number}: città, perché, naïve, über, España. "
            "日本語の文書を検証します。中文记录保持顺序。 Ελληνικό αρχείο. "
            f"Emoji 🔬 🧪 ✅; identificatore {number}; nessun contenuto privato.",
        ),
        (
            "log",
            f"2031-01-{index % 28 + 1:02d}T12:{index % 60:02d}:00Z INFO case={index:05d} "
            f"request={number} component=synthetic-ingest state=validated count={index % 128 + 1}\n"
            "INFO checkpoint=committed outcome=complete order=preserved retry=none",
        ),
        (
            "mixed",
            f"case {index:05d}: requisito REQ-{number:07d}, endpoint /v1/embeddings. "
            "Invariant: output_count == input_count; dimension == 768; all values finite. "
            f"Nota italiana: verificare l'integrità della sequenza {index % 113}. "
            "No server lifecycle operation is part of this ingest.",
        ),
    ]
    kind, text = texts[category]
    return {
        "index": index,
        "category": kind,
        "text": text,
        "text_sha256": sha(text.encode()),
    }


def prepare(root: Path) -> None:
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("Preparation requires a new or empty evidence directory")
    root.mkdir(parents=True, exist_ok=True)
    corpus = root / "corpus.jsonl"
    categories: Counter[str] = Counter()
    lengths = []
    with corpus.open("xb") as output:
        for index in range(COUNT):
            row = corpus_row(index)
            categories[row["category"]] += 1
            lengths.append(len(row["text"]))
            output.write(json_bytes(row) + b"\n")
    oracle = {
        "schema": 1,
        "prepared_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint_default": DEFAULT_ENDPOINT,
        "expected_model": MODEL,
        "expected_build": EXPECTED_BUILD,
        "expected_loaded_context": EXPECTED_CONTEXT,
        "expected_slots": EXPECTED_SLOTS,
        "expected_dimension": DIMENSION,
        "input_count": COUNT,
        "corpus_sha256": sha(corpus.read_bytes()),
        "category_counts": dict(categories),
        "character_length": {
            "minimum": min(lengths),
            "maximum": max(lengths),
            "mean": sum(lengths) / COUNT,
        },
        "configuration": {
            "window_size": WINDOW,
            "batch_size": BATCH_SIZE,
            "batch_token_limit": BATCH_TOKEN_LIMIT,
            "token_count_parallelism": TOKEN_PARALLELISM,
            "normalize": True,
            "max_input_tokens": EXPECTED_CONTEXT,
            "timeout_seconds": 30,
        },
        "oracles": {
            "integrity": "exactly10000 unique indexed inputs, exact ordered10000 vectors, manifest and binary SHA256",
            "vectors": f"all{DIMENSION} dimensions finite, nonzero, normalized abs(norm-1)<=1e-10",
            "http_projection": "each emitted output exactly matches independently normalized raw payload at its provider index",
            "capacity": f"every token count<={EXPECTED_CONTEXT}; every actual provider request sum<={BATCH_TOKEN_LIMIT}, all request counts recorded",
            "memory": "post-warmup RSS growth<=128MiB; peak RSS increase<=256MiB; warmupafter1024items",
            "transport": "no implicit retries by probe; fail preserves diagnostics and processed-prefix evidence",
            "sample_queries": "8 sampled corpus inputs; raw and adapter use the same input batch; compare shapes and report variation",
            "server_identity": "build/model path/config/meta stable before/after; dynamic catalog created excluded",
        },
        "nonclaims": [
            "retrieval quality",
            "bitwise equivalence across separate server requests",
            "four simultaneous GPU slots",
            "absence of leaks beyond this bounded workload",
            "consumer integration",
            "automatic recovery after transport failures",
        ],
        "load_estimate_before_tokenization": {
            "token_range": [500000, 1200000],
            "tokenize_requests": COUNT,
            "props_requests": math.ceil(COUNT / WINDOW),
            "embedding_requests_estimate": [1200, 3500],
        },
    }
    write_json(root / "oracle.json", oracle)
    print(
        json.dumps(
            {
                "prepared": str(root),
                "count": COUNT,
                "corpus_sha256": oracle["corpus_sha256"],
                "categories": dict(categories),
                "no_network_requests": True,
            },
            ensure_ascii=False,
        )
    )


class TransportAudit:
    """Observe the public requests.Session seam without retries or pool changes."""

    def __init__(self, root: Path, indexes: dict[str, int]):
        self.root, self.indexes = root, indexes
        self.expected_model = MODEL
        self.lock = Lock()
        self.sequence = 0

    @staticmethod
    def exceptions(exc: BaseException) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        seen: set[int] = set()

        def visit(value: Any, relation: str, depth: int = 0) -> None:
            if depth > 8 or id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, BaseException):
                item: dict[str, Any] = {
                    "class": type(value).__module__ + "." + type(value).__name__,
                    "relation": relation,
                }
                if isinstance(getattr(value, "errno", None), int):
                    item["errno"] = value.errno
                item["frames"] = [
                    {
                        "file": Path(frame.filename).name,
                        "line": frame.lineno,
                        "function": frame.name,
                    }
                    for frame in traceback.extract_tb(value.__traceback__)[-12:]
                ]
                entries.append(item)
                visit(value.__cause__, "cause", depth + 1)
                visit(value.__context__, "context", depth + 1)
                for argument in value.args:
                    visit(argument, "argument", depth + 1)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child, relation, depth + 1)

        visit(exc, "root")
        return entries

    def new_session(self, label: str, *, model_scoped: bool = True):
        """Use an unscoped session only for intentional router metadata reads."""
        import requests

        audit = self

        class AuditedSession(requests.Session):
            def request(self, *args, **kwargs):
                started = time.monotonic()
                method = kwargs.get("method", args[0] if args else None)
                url = kwargs.get("url", args[1] if len(args) > 1 else "")
                body = kwargs.get("json") or {}
                content, inputs = body.get("content"), body.get("input")
                route = urlsplit(url)
                model_query = parse_qs(route.query, keep_blank_values=True).get("model")
                texts = (
                    [content]
                    if isinstance(content, str)
                    else inputs
                    if isinstance(inputs, list)
                    else []
                )
                with audit.lock:
                    audit.sequence += 1
                    record = {
                        "request_index": audit.sequence,
                        "session": label,
                        "method": method,
                        "path": urlsplit(url).path,
                        "model": body.get("model"),
                        "query_model": model_query,
                        "input_positions": [
                            audit.indexes.get(sha(text.encode()))
                            for text in texts
                            if isinstance(text, str)
                        ],
                        "input_count": len(texts),
                        "started_monotonic": started,
                    }
                try:
                    if route.path.endswith(("/v1/embeddings", "/tokenize")):
                        if body.get("model") != audit.expected_model:
                            raise ValueError(
                                "Qualification body model selector mismatch"
                            )
                    elif route.path.endswith("/props") and (
                        model_scoped or model_query is not None
                    ):
                        if model_query != [audit.expected_model]:
                            raise ValueError(
                                "Qualification props model selector mismatch"
                            )
                    response = super().request(*args, **kwargs)
                    record["status_code"] = response.status_code
                    return response
                except BaseException as exc:
                    record["exception_chain"] = audit.exceptions(exc)
                    raise
                finally:
                    record["elapsed_seconds"] = time.monotonic() - started
                    with (
                        audit.lock,
                        (audit.root / "transport-audit.jsonl").open("ab") as output,
                    ):
                        output.write(json_bytes(record) + b"\n")

        return AuditedSession()


class Monitor:
    def __init__(self, root: Path):
        self.root, self.stop = root, Event()
        self.lock = Lock()
        self.state = {"phase": "starting", "processed": 0, "input_count": COUNT}
        self.samples = []
        self.thread = Thread(
            target=self._run, name="qualification-monitor", daemon=True
        )

    def update(self, **values: Any) -> None:
        with self.lock:
            self.state.update(values)

    def _run(self) -> None:
        while not self.stop.is_set():
            measurement = process_measurement()
            with self.lock:
                state = {**self.state, **measurement}
            self.samples.append(state)
            write_json(self.root / "heartbeat.json", state)
            with (self.root / "resources.jsonl").open("ab") as output:
                output.write(json_bytes(state) + b"\n")
            self.stop.wait(2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(5)


def run(root: Path, endpoint: str, source: Path | None) -> None:
    # Explicit source path makes the evaluated checkout reviewable; network is
    # only entered by --run --acknowledge-live in the CLI below.
    if source is not None:
        sys.path.insert(0, str(source / "src"))
    from runtime_llama import LlamaCppClient, build_embedding_adapter

    oracle = json.loads((root / "oracle.json").read_text())
    for key, expected in (
        ("expected_model", MODEL),
        ("expected_build", EXPECTED_BUILD),
        ("expected_loaded_context", EXPECTED_CONTEXT),
        ("expected_slots", EXPECTED_SLOTS),
        ("expected_dimension", DIMENSION),
    ):
        if oracle[key] != expected:
            raise ValueError(f"Prepared oracle disagrees with CLI: {key}")
    expected_configuration = {
        "window_size": WINDOW,
        "batch_size": BATCH_SIZE,
        "batch_token_limit": BATCH_TOKEN_LIMIT,
        "token_count_parallelism": TOKEN_PARALLELISM,
        "normalize": True,
        "max_input_tokens": EXPECTED_CONTEXT,
        "timeout_seconds": 30,
    }
    if oracle["configuration"] != expected_configuration:
        raise ValueError("Prepared oracle configuration disagrees with CLI")
    assert sha((root / "corpus.jsonl").read_bytes()) == oracle["corpus_sha256"]
    rows = [
        json.loads(line) for line in (root / "corpus.jsonl").read_text().splitlines()
    ]
    assert len(rows) == COUNT and [row["index"] for row in rows] == list(range(COUNT))
    assert len({row["text_sha256"] for row in rows}) == COUNT
    assert all(sha(row["text"].encode()) == row["text_sha256"] for row in rows)
    if (root / "result.json").exists() or (root / "vectors.f64le").exists():
        raise FileExistsError(
            "Refusing to overwrite a live run; prepare a fresh evidence directory"
        )

    transport_audit = TransportAudit(
        root, {row["text_sha256"]: row["index"] for row in rows}
    )

    class AuditClient(LlamaCppClient):
        def __init__(self):
            self.audit_session = transport_audit.new_session("adapter")
            super().__init__(endpoint, timeout=30, session=self.audit_session)
            self.counts: dict[str, int] = {}
            self.pending: list[tuple[str, list[float]]] = []
            self.tokenize_count = self.embedding_count = self.provider_token_count = 0
            self.min_input_tokens, self.max_input_tokens = 1_000_000, 0
            self.max_provider_tokens = self.max_provider_items = 0
            self.tokens_lock = Lock()

        def close(self):
            super().close()
            self.audit_session.close()

        def tokenize(self, content, **kwargs):
            response = super().tokenize(content, **kwargs)
            count = len(response["tokens"])
            assert 0 < count <= EXPECTED_CONTEXT, ("input_capacity", count)
            with self.tokens_lock:
                self.counts[sha(content.encode())] = count
                self.tokenize_count += 1
                self.min_input_tokens = min(self.min_input_tokens, count)
                self.max_input_tokens = max(self.max_input_tokens, count)
            return response

        def embeddings(self, input, **kwargs):
            texts = list(input)
            counts = [self.counts[sha(text.encode())] for text in texts]
            assert sum(counts) <= BATCH_TOKEN_LIMIT, ("provider_capacity", counts)
            response = super().embeddings(texts, **kwargs)
            data = response["data"]
            assert len(data) == len(texts)
            assert sorted(
                item.get("index", pos) for pos, item in enumerate(data)
            ) == list(range(len(texts)))
            indexed = {
                item.get("index", pos): item["embedding"]
                for pos, item in enumerate(data)
            }
            records = []
            for position, text in enumerate(texts):
                raw = indexed[position]
                assert len(raw) == DIMENSION and all(
                    not isinstance(x, bool)
                    and isinstance(x, (int, float))
                    and math.isfinite(x)
                    for x in raw
                )
                projected = normalized([float(value) for value in raw])
                self.pending.append((sha(text.encode()), projected))
                records.append(
                    {
                        "input_sha256": sha(text.encode()),
                        "tokens": counts[position],
                        "raw_vector_sha256": sha(vector_bytes(raw)),
                        "projected_sha256": sha(vector_bytes(projected)),
                    }
                )
            self.embedding_count += 1
            self.provider_token_count += sum(counts)
            self.max_provider_tokens = max(self.max_provider_tokens, sum(counts))
            self.max_provider_items = max(self.max_provider_items, len(texts))
            with (root / "provider-batches.jsonl").open("ab") as output:
                output.write(
                    json_bytes(
                        {
                            "request_index": self.embedding_count,
                            "items": records,
                            "sum_tokens": sum(counts),
                            "observed_response_indices": [
                                item.get("index", pos) for pos, item in enumerate(data)
                            ],
                        }
                    )
                    + b"\n"
                )
            return response

    def server_state(client):
        catalog = client.models()
        selected = next(item for item in catalog["data"] if item["id"] == MODEL)
        assert selected["status"]["value"] == "loaded", (
            "Selected model must already be loaded"
        )
        props = client.props(model=MODEL)
        assert props["build_info"] == EXPECTED_BUILD
        assert props["default_generation_settings"]["n_ctx"] == EXPECTED_CONTEXT
        assert props["total_slots"] == EXPECTED_SLOTS
        assert selected["meta"]["n_embd"] == DIMENSION
        return {
            "catalog": catalog,
            "props": props,
            "identity": {
                "model": MODEL,
                "model_path": props["model_path"],
                "build": props["build_info"],
                "context": props["default_generation_settings"]["n_ctx"],
                "slots": props["total_slots"],
                "meta": selected["meta"],
                "args": selected["status"]["args"],
                "chat_template_sha256": sha(props.get("chat_template", "").encode()),
            },
        }

    started = time.monotonic()
    result: dict[str, Any] = {
        "status": "running",
        "processed": 0,
        "oracle_sha256": sha((root / "oracle.json").read_bytes()),
        "source": str(source) if source is not None else "installed-package",
        "resource_scope": "POSIX ps/resource; Windows resource qualification not supported",
        "transport_diagnostics": "passive Session.request audit, no headers/content, no retries/pool changes",
        "endpoint": endpoint,
        "pid": os.getpid(),
    }
    client = AuditClient()
    adapter = None
    try:
        before = server_state(client)
        write_json(root / "server-before.json", before)
        configuration = {
            "model": MODEL,
            "base_url": endpoint,
            "batch_size": BATCH_SIZE,
            "batch_token_limit": BATCH_TOKEN_LIMIT,
            "token_count_parallelism": TOKEN_PARALLELISM,
            "max_input_tokens": EXPECTED_CONTEXT,
            "dimension": DIMENSION,
            "normalize": True,
            "timeout": 30,
        }
        adapter = build_embedding_adapter(configuration, client=client)
        initial_resource = process_measurement()
        warm_resource = None
        vector_digest = hashlib.sha256()
        ordering_digest = hashlib.sha256()
        norm_min, norm_max = 1.0, 1.0
        with (
            Monitor(root) as monitor,
            (root / "vectors.f64le").open("wb") as vectors_file,
            (root / "vectors-manifest.jsonl").open("wb") as manifest,
        ):
            for start in range(0, COUNT, WINDOW):
                window = rows[start : start + WINDOW]
                client.pending = []
                monitor.update(phase="ingest", window_start=start, processed=start)
                vectors = adapter.embed_texts([row["text"] for row in window])
                assert len(vectors) == len(window) == len(client.pending)
                for row, vector, (input_hash, expected) in zip(
                    window, vectors, client.pending
                ):
                    assert row["text_sha256"] == input_hash, (
                        "order_input_hash",
                        row["index"],
                    )
                    assert len(vector) == DIMENSION and all(
                        math.isfinite(value) for value in vector
                    )
                    norm = math.sqrt(math.fsum(value * value for value in vector))
                    assert norm > 0 and abs(norm - 1.0) <= 1e-10, (
                        "normalization",
                        row["index"],
                        norm,
                    )
                    assert vector_bytes(vector) == vector_bytes(expected), (
                        "response_projection",
                        row["index"],
                    )
                    encoded = vector_bytes(vector)
                    vectors_file.write(encoded)
                    vector_digest.update(encoded)
                    entry = {
                        "index": row["index"],
                        "input_sha256": input_hash,
                        "vector_sha256": sha(encoded),
                        "norm": norm,
                        "tokens": client.counts[input_hash],
                    }
                    manifest.write(json_bytes(entry) + b"\n")
                    ordering_digest.update(
                        json_bytes(
                            {
                                key: entry[key]
                                for key in ("index", "input_sha256", "vector_sha256")
                            }
                        )
                    )
                    norm_min, norm_max = min(norm_min, norm), max(norm_max, norm)
                result["processed"] = start + len(window)
                vectors_file.flush()
                manifest.flush()
                del vectors
                client.pending = []
                gc.collect()
                observed = process_measurement()
                if warm_resource is None and result["processed"] >= 1024:
                    warm_resource = observed
                assert (
                    observed["peak_rss_bytes"] - initial_resource["rss_bytes"]
                    <= 256 * 1024 * 1024
                ), "Peak RSS budget"
                if warm_resource is not None:
                    assert (
                        observed["rss_bytes"] - warm_resource["rss_bytes"]
                        <= 128 * 1024 * 1024
                    ), "Post-warmup RSS budget"
                monitor.update(
                    processed=result["processed"],
                    embedding_requests=client.embedding_count,
                    tokenization_requests=client.tokenize_count,
                    elapsed_seconds=time.monotonic() - started,
                )
            monitor.update(phase="sample_queries", processed=COUNT)
            # These are separate requests and may differ numerically. The main
            # ingest projection oracle above compares the exact same response.
            sample_indices = [0, 137, 999, 2501, 4999, 6503, 7997, 9999]
            samples = []
            raw_session = transport_audit.new_session("raw_samples")
            plain = LlamaCppClient(endpoint, timeout=30, session=raw_session)
            try:
                for position in range(0, len(sample_indices), 2):
                    indices = sample_indices[position : position + 2]
                    texts = [rows[index]["text"] for index in indices]
                    client.pending = []
                    adapter_vectors = adapter.embed_texts(texts)
                    raw = plain.embeddings(texts, model=MODEL)
                    ordered = {
                        item["index"]: normalized(item["embedding"])
                        for item in raw["data"]
                    }
                    assert len(ordered) == len(texts)
                    for offset, index in enumerate(indices):
                        actual, direct = adapter_vectors[offset], ordered[offset]
                        assert len(direct) == DIMENSION and all(
                            math.isfinite(value) for value in direct
                        )
                        samples.append(
                            {
                                "corpus_index": index,
                                "adapter_sha256": sha(vector_bytes(actual)),
                                "raw_normalized_sha256": sha(vector_bytes(direct)),
                                "max_abs_difference": max(
                                    abs(a - b) for a, b in zip(actual, direct)
                                ),
                                "cosine": math.fsum(
                                    a * b for a, b in zip(actual, direct)
                                ),
                            }
                        )
            finally:
                plain.close()
                raw_session.close()
            write_json(
                root / "sample-queries.json",
                {
                    "comparison": "separate_requests_same_two_item_batch",
                    "numerical_equivalence_required": False,
                    "samples": samples,
                },
            )
            final_resource = process_measurement()
            result["resources"] = {
                "initial": initial_resource,
                "warm_after1024": warm_resource,
                "final": final_resource,
                "monitor_samples": len(monitor.samples),
            }
        after = server_state(client)
        write_json(root / "server-after.json", after)
        assert before["identity"] == after["identity"], "Server configuration changed"
        assert result["processed"] == COUNT
        assert (root / "vectors.f64le").stat().st_size == COUNT * DIMENSION * 8
        result.update(
            status="passed",
            vector_binary_sha256=vector_digest.hexdigest(),
            ordered_manifest_sha256=ordering_digest.hexdigest(),
            norm_range=[norm_min, norm_max],
            exact_vector_count=COUNT,
            stored_vector_bytes=COUNT * DIMENSION * 8,
            tokenizer_requests=client.tokenize_count,
            embedding_requests=client.embedding_count,
            provider_token_count_including_samples=client.provider_token_count,
            input_token_range=[client.min_input_tokens, client.max_input_tokens],
            maximum_provider_tokens=client.max_provider_tokens,
            maximum_provider_items=client.max_provider_items,
            final_adapter_health=adapter.health_status(),
            sample_queries=8,
            no_implicit_probe_retry=True,
        )
    except BaseException as exc:
        result.update(
            status="failed", failure_type=type(exc).__name__, failure=str(exc)
        )
        raise
    finally:
        if adapter is not None:
            adapter.close()
        client.close()
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(root / "result.json", result)
        write_json(
            root / "heartbeat.json",
            {
                "phase": "finished",
                "status": result["status"],
                "processed": result["processed"],
            },
        )
        print(json.dumps(result, ensure_ascii=False))


def main() -> None:
    global MODEL, EXPECTED_BUILD, EXPECTED_CONTEXT, EXPECTED_SLOTS, DIMENSION
    global BATCH_SIZE, BATCH_TOKEN_LIMIT, TOKEN_PARALLELISM, DEFAULT_ENDPOINT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", "--output", dest="root", type=Path, default=DEFAULT_ROOT
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Write deterministic corpus/oracle without contacting any server",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the already prepared corpus after explicit opt-in",
    )
    parser.add_argument("--acknowledge-live", action="store_true")
    parser.add_argument(
        "--base-url",
        "--endpoint",
        dest="endpoint",
        type=validate_base_url,
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-build", required=True)
    parser.add_argument("--expected-context", type=int, default=EXPECTED_CONTEXT)
    parser.add_argument("--expected-slots", type=int, default=EXPECTED_SLOTS)
    parser.add_argument("--dimension", type=int, default=DIMENSION)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--batch-token-limit", type=int, default=BATCH_TOKEN_LIMIT)
    parser.add_argument(
        "--token-count-parallelism", type=int, default=TOKEN_PARALLELISM
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Optional source checkout; otherwise import installed runtime_llama",
    )
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("Choose exactly one of --prepare or --run")
    if args.run and not args.acknowledge_live:
        parser.error("Live execution requires --acknowledge-live")
    if args.run and (os.name != "posix" or resource is None):
        parser.error(
            "This probe's process/resource measurements require POSIX; Windows execution is outside its qualification scope"
        )
    if any(
        value <= 0
        for value in (
            args.expected_context,
            args.expected_slots,
            args.dimension,
            args.batch_size,
            args.batch_token_limit,
            args.token_count_parallelism,
        )
    ):
        parser.error(
            "Context, slot, dimension and batching declarations must be positive"
        )
    if args.batch_token_limit > args.expected_context:
        parser.error(
            "Conservative provider token budget cannot exceed the expected per-slot context"
        )
    MODEL, EXPECTED_BUILD = args.model, args.expected_build
    EXPECTED_CONTEXT, EXPECTED_SLOTS, DIMENSION = (
        args.expected_context,
        args.expected_slots,
        args.dimension,
    )
    BATCH_SIZE, BATCH_TOKEN_LIMIT = args.batch_size, args.batch_token_limit
    TOKEN_PARALLELISM, DEFAULT_ENDPOINT = args.token_count_parallelism, args.endpoint
    if args.prepare:
        prepare(args.root)
    else:
        run(args.root, args.endpoint, args.source)


if __name__ == "__main__":
    main()
