"""Socket-level recovery qualification, with explicitly synthetic faults."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import socket
import sys
from threading import Lock, Thread
import time
from urllib.parse import parse_qs, urlencode, urlsplit

import requests

from runtime_llama import build_adapter, build_embedding_adapter
from runtime_llama.client import ClientHTTPError
from runtime_llama.errors import LLMBackendError, LLMTimeoutError


ROOT = Path(__file__).resolve().parent
PROMPT_SECRET = "QUALIFICATION_PRIVATE_PROMPT_c4729f"
KEY_SECRET = "QUALIFICATION_FAKE_KEY_342a8e"
PATH_SECRET = "/home/qualification-private/model-secret.gguf"
LLM = "Qwen3.8-27B-Q6_K.gguf"
EMBED = "nomic-embed-text-v2-moe.Q8_0.gguf"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def chunk(*, content=None, finish=None, usage=None):
    delta = {} if content is None else {"content": content}
    result = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        result["usage"] = usage
    return result


def sse(payloads, done=True):
    result = b"".join(
        b"data: " + json.dumps(item).encode() + b"\n\n" for item in payloads
    )
    return result + (b"data: [DONE]\n\n" if done else b"")


class ProxyState:
    def __init__(self, mode, upstream, *, llm_model, embedding_model):
        self.mode, self.upstream = mode, upstream
        self.llm_model, self.embedding_model = llm_model, embedding_model
        self.lock = Lock()
        self.events = []
        self.pending = None

    def arm(self, path, fault):
        with self.lock:
            assert self.pending is None
            self.pending = (path, fault)

    def dispatch(self, path, body, raw):
        parsed = urlsplit(path)
        route = parsed.path
        with self.lock:
            event = {
                "sequence": len(self.events),
                "path": route,
                "body_sha256": sha(raw),
                "injected_fault": None,
                "upstream_forwarded": False,
                "input_count": len(body.get("input", []))
                if isinstance(body.get("input"), list)
                else None,
            }
            self.events.append(event)
            expected = {
                "/v1/chat/completions": self.llm_model,
                "/v1/embeddings": self.embedding_model,
                "/tokenize": self.embedding_model,
                "/props": self.embedding_model,
            }.get(route)
            if expected is not None:
                if route == "/props":
                    observed = parse_qs(parsed.query, keep_blank_values=True).get(
                        "model", []
                    )
                    event["model_source"] = "query"
                    event["observed_model"] = observed
                    matches = observed == [expected]
                else:
                    observed = body.get("model")
                    event["model_source"] = "body"
                    event["observed_model"] = (
                        observed if isinstance(observed, str) else None
                    )
                    matches = observed == expected
                event["expected_model"] = expected
                event["model_matches"] = matches
                if not matches:
                    event["model_guard_failure"] = "exact_selected_alias_required"
                    return event, None
            fault = None
            if self.pending is not None and self.pending[0] == route:
                fault = self.pending[1]
                self.pending = None
                event["injected_fault"] = fault
        return event, fault

    def selected(self, start, path):
        with self.lock:
            return [dict(item) for item in self.events[start:] if item["path"] == path]


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def send_bytes(self, status, payload, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            self.close_connection = True

        def respond_json(self, status, payload):
            self.send_bytes(status, json.dumps(payload).encode())

        def do_GET(self):
            self.handle_call()

        def do_POST(self):
            self.handle_call()

        def handle_call(self):
            size = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(size) if size else b""
            body = json.loads(raw) if raw else {}
            event, fault = state.dispatch(self.path, body, raw)
            route = urlsplit(self.path).path
            try:
                if event.get("model_guard_failure"):
                    self.respond_json(
                        400, {"error": {"type": "qualification_model_guard"}}
                    )
                elif fault is not None:
                    self.inject(fault, body)
                elif state.mode == "live_proxy":
                    assert route in {
                        "/health",
                        "/props",
                        "/tokenize",
                        "/v1/embeddings",
                        "/v1/chat/completions",
                    }
                    event["upstream_forwarded"] = True
                    started = time.monotonic()
                    with requests.request(
                        self.command,
                        state.upstream + self.path,
                        data=raw if raw else None,
                        headers={"Content-Type": "application/json"},
                        timeout=90,
                        stream=True,
                        allow_redirects=False,
                    ) as response:
                        event["upstream_status"] = response.status_code
                        # Bounded responses are forwarded byte for byte. Buffering
                        # is deliberate: this qualifies error recovery, not TTFT.
                        content = response.content
                        event["upstream_body_sha256"] = sha(content)
                        event["elapsed_ms"] = round(
                            (time.monotonic() - started) * 1000, 3
                        )
                        self.send_bytes(
                            response.status_code,
                            content,
                            response.headers.get("Content-Type", "application/json"),
                        )
                else:
                    self.fixture(route, body)
            except (BrokenPipeError, ConnectionResetError):
                event["consumer_disconnected"] = True
            except Exception as exc:
                event["proxy_failure_type"] = type(exc).__name__
                raise

        def inject(self, fault, body):
            if fault.startswith("http_") or fault == "capacity":
                status = 400 if fault == "capacity" else int(fault.split("_")[1])
                self.respond_json(
                    status,
                    {
                        "error": {
                            "type": "exceed_context_size_error"
                            if fault == "capacity"
                            else "qualification_error",
                            "code": status,
                            "message": PROMPT_SECRET + KEY_SECRET + PATH_SECRET,
                        }
                    },
                )
            elif fault == "duplicate_index":
                self.respond_json(
                    200,
                    {
                        "data": [
                            {"index": 0, "embedding": [1.0, 2.0, 3.0, 4.0]}
                            for _ in body["input"]
                        ]
                    },
                )
            elif fault == "nonfinite_vector":
                self.send_bytes(200, b'{"data":[{"index":0,"embedding":[NaN,1,2,3]}]}')
            elif fault == "malformed_sse":
                self.send_bytes(
                    200,
                    ("data: {" + PROMPT_SECRET + "}\n\n").encode(),
                    "text/event-stream",
                )
            elif fault == "early_eof":
                self.send_bytes(
                    200,
                    sse([chunk(content="partial")], done=False),
                    "text/event-stream",
                )
            elif fault == "invalid_usage":
                self.send_bytes(
                    200,
                    sse(
                        [
                            chunk(content="partial"),
                            chunk(finish="stop", usage={"completion_tokens": -1}),
                        ]
                    ),
                    "text/event-stream",
                )
            elif fault == "missing_usage":
                self.send_bytes(
                    200,
                    sse([chunk(content="RECOVERED"), chunk(finish="stop")]),
                    "text/event-stream",
                )
            elif fault == "delayed_response":
                time.sleep(0.2)
                self.fixture(urlsplit(self.path).path, body)
            elif fault == "socket_disconnect":
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                self.close_connection = True
            else:
                raise AssertionError(f"unknown controlled fault {fault}")

        def fixture(self, route, body):
            if route == "/health":
                self.respond_json(200, {"status": "ok"})
            elif route == "/props":
                self.respond_json(200, {"default_generation_settings": {"n_ctx": 512}})
            elif route == "/tokenize":
                self.respond_json(
                    200, {"tokens": list(range(len(body["content"].split()) + 2))}
                )
            elif route == "/v1/embeddings":
                vectors = []
                for index, text in enumerate(body["input"]):
                    digest = hashlib.sha256(text.encode()).digest()
                    vectors.append(
                        {
                            "index": index,
                            "embedding": [(value + 1) / 256 for value in digest[:4]],
                        }
                    )
                self.respond_json(200, {"data": vectors})
            elif route == "/v1/chat/completions":
                self.send_bytes(
                    200,
                    sse(
                        [
                            chunk(content="RECOVERED"),
                            chunk(
                                finish="stop",
                                usage={
                                    "prompt_tokens": 7,
                                    "completion_tokens": 4,
                                    "total_tokens": 11,
                                },
                            ),
                        ]
                    ),
                    "text/event-stream",
                )
            else:
                raise AssertionError(route)

    return Handler


def validate_vectors(vectors, count, dimension):
    assert len(vectors) == count, (len(vectors), count)
    assert all(len(vector) == dimension for vector in vectors)
    assert all(math.isfinite(value) for vector in vectors for value in vector)


def exception_evidence(exc):
    diagnostics = getattr(exc, "diagnostics", None)
    payload = {
        "type": type(exc).__name__,
        "message": str(exc),
        "terminal_reason": getattr(exc, "terminal_reason", None),
        "status_code": getattr(exc, "status_code", None),
        "error_type": getattr(exc, "error_type", None),
        "diagnostics": diagnostics.to_dict()
        if hasattr(diagnostics, "to_dict")
        else diagnostics,
        "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
        "cause_message": str(exc.__cause__) if exc.__cause__ else None,
    }
    serialized = json.dumps(payload)
    assert all(
        secret not in serialized for secret in [PROMPT_SECRET, KEY_SECRET, PATH_SECRET]
    )
    return payload


def main():
    global LLM, EMBED
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fixture", "live_proxy"], default="fixture")
    parser.add_argument("--base-url", dest="upstream")
    parser.add_argument("--llm-model")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-dimension", type=int)
    parser.add_argument("--expected-build")
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path)
    parser.add_argument("--acknowledge-live", action="store_true")
    args = parser.parse_args()
    preflight = None
    if args.mode == "live_proxy":
        if not args.acknowledge_live:
            parser.error(
                "live_proxy requires --acknowledge-live; faults are synthetic and valid requests reach the selected external server"
            )
        for field in (
            "upstream",
            "llm_model",
            "embedding_model",
            "expected_build",
            "embedding_dimension",
        ):
            if not getattr(args, field):
                flag = "base-url" if field == "upstream" else field.replace("_", "-")
                parser.error("live_proxy requires --" + flag)
        route = urlsplit(args.upstream)
        if (
            route.scheme not in {"http", "https"}
            or not route.hostname
            or route.username
            or route.password
            or route.query
            or route.fragment
        ):
            parser.error(
                "base URL must be HTTP(S) without credentials, query or fragment"
            )
        if args.embedding_dimension <= 0:
            parser.error("embedding dimension must be positive")
        LLM, EMBED = args.llm_model, args.embedding_model
    else:
        LLM, EMBED = (
            args.llm_model or "fixture-llm",
            args.embedding_model or "fixture-embedding",
        )
        args.upstream = None
    output_dir = args.output_dir or ROOT / (args.mode + "-" + str(time.time_ns()))
    output_dir.mkdir(parents=True, exist_ok=False)
    if args.mode == "live_proxy":
        args.upstream = args.upstream.rstrip("/")
        with requests.get(
            args.upstream + "/props", timeout=10, allow_redirects=False
        ) as response:
            response.raise_for_status()
            props = response.json()
        if (
            not isinstance(props, dict)
            or props.get("build_info") != args.expected_build
        ):
            raise RuntimeError("preflight_build_mismatch_no_inference_sent")
        with requests.get(
            args.upstream + "/models", timeout=10, allow_redirects=False
        ) as response:
            response.raise_for_status()
            models = response.json()
        catalog = {
            item.get("id"): item
            for item in models.get("data", [])
            if isinstance(item, dict)
        }
        loaded = {
            alias: catalog.get(alias, {}).get("status", {}).get("value")
            for alias in (LLM, EMBED)
        }
        if any(value != "loaded" for value in loaded.values()):
            raise RuntimeError("preflight_selected_model_not_loaded_no_inference_sent")
        preflight = {
            "observed_build": props["build_info"],
            "expected_build": args.expected_build,
            "loaded_aliases": loaded,
            "read_only_http_requests": 2,
        }
        (output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))
    state = ProxyState(args.mode, args.upstream, llm_model=LLM, embedding_model=EMBED)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    dimension = 4 if args.mode == "fixture" else args.embedding_dimension
    results = []

    def run(name, action):
        before = len(state.events)
        started = time.monotonic()
        try:
            detail = action()
            result = {"case": name, "outcome": "passed", "detail": detail}
        except Exception as exc:
            result = {
                "case": name,
                "outcome": "failed",
                "failure_type": type(exc).__name__,
                "failure": str(exc),
            }
        result["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
        result["events"] = [dict(item) for item in state.events[before:]]
        guard_failures = [
            item for item in result["events"] if item.get("model_guard_failure")
        ]
        if guard_failures:
            result.update(
                outcome="failed",
                failure_type="ModelSelectionViolation",
                failure="exact_selected_alias_required_before_forwarding",
            )
        if state.pending is not None:
            result.update(
                outcome="failed",
                failure_type="HarnessFaultNotConsumed",
                unconsumed_injection=state.pending,
            )
            state.pending = None
        results.append(result)
        (output_dir / (name + ".json")).write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {key: result[key] for key in ["case", "outcome", "duration_ms"]}
            ),
            flush=True,
        )

    def embedding():
        return build_embedding_adapter(
            {
                "model": EMBED,
                "base_url": base_url,
                "batch_size": 4,
                "normalize": True,
                "timeout": 30,
                "dimension": dimension,
            }
        )

    def llm():
        return build_adapter(
            {
                "model": LLM,
                "base_url": base_url,
                "reasoning": "off",
                "temperature": 0,
                "num_predict": 32,
                "stream_idle_timeout_sec": 30,
                "api_key": KEY_SECRET,
            }
        )

    def adaptive():
        adapter = embedding()
        start = len(state.events)
        state.arm("/v1/embeddings", "capacity")
        try:
            initial = adapter.embed_texts(
                [f"qualification initial document {i}" for i in range(4)]
            )
            validate_vectors(initial, 4, dimension)
            reduced = adapter.health_status()
            assert reduced["effective_batch_size"] == 2
            assert reduced["batch_recovery"]["successes"] == 0
            middle = adapter.embed_texts(
                [f"qualification recovery document {i}" for i in range(64)]
            )
            validate_vectors(middle, 64, dimension)
            recovered = adapter.health_status()
            assert recovered["effective_batch_size"] == 4
            assert recovered["batch_recovery"]["successes"] == 0
            final = adapter.embed_texts(
                [f"qualification final document {i}" for i in range(4)]
            )
            validate_vectors(final, 4, dimension)
            events = state.selected(start, "/v1/embeddings")
            assert [item["input_count"] for item in events] == [4, 2, 2] + [2] * 32 + [
                4
            ]
            return {
                "initial_health": reduced,
                "final_health": recovered,
                "embedding_http_request_count": len(events),
                "input_outputs_count": 72,
                "dimension": dimension,
                "order_checked": "Transport input slices and output count; direct numeric order is independently qualified by embedding campaign.",
            }
        finally:
            adapter.close()

    run("embedding_adaptive_split_and_recovery", adaptive)

    def embed_fault(fault):
        adapter = embedding()
        try:
            start = len(state.events)
            state.arm("/v1/embeddings", fault)
            texts = ["qualification private document " + PROMPT_SECRET]
            if fault == "duplicate_index":
                texts.append("second qualification document")
            try:
                adapter.embed_texts(texts)
            except LLMBackendError as exc:
                evidence = exception_evidence(exc)
                if fault.startswith("http_"):
                    assert isinstance(exc, ClientHTTPError) and exc.status_code == int(
                        fault.split("_")[1]
                    )
                if fault == "capacity":
                    assert (
                        isinstance(exc, ClientHTTPError)
                        and exc.error_type == "exceed_context_size_error"
                    )
                if fault in {"duplicate_index", "nonfinite_vector"}:
                    assert str(exc) == "llamacpp_embedding_invalid_response"
            else:
                raise AssertionError("fault unexpectedly succeeded")
            failed = state.selected(start, "/v1/embeddings")
            assert len(failed) == 1
            recovered = adapter.embed_texts(["qualification valid recovered document"])
            validate_vectors(recovered, 1, dimension)
            assert len(state.selected(start, "/v1/embeddings")) == 2
            return {
                "failure": evidence,
                "failing_http_requests": 1,
                "recovery_success": True,
                "dimension": len(recovered[0]),
            }
        finally:
            adapter.close()

    for fault in [
        "http_401",
        "http_429",
        "http_500",
        "duplicate_index",
        "nonfinite_vector",
        "capacity",
    ]:
        run("embedding_" + fault, lambda fault=fault: embed_fault(fault))

    def llm_fault(fault):
        adapter = llm()
        try:
            warmup = adapter.generate_with_metadata("Reply exactly RECOVERED.")
            assert warmup.text.strip() == "RECOVERED" and warmup.usage is not None
            start = len(state.events)
            state.arm("/v1/chat/completions", fault)
            try:
                adapter.generate_with_metadata(
                    "Private qualification marker: " + PROMPT_SECRET
                )
            except LLMBackendError as exc:
                evidence = exception_evidence(exc)
                if fault.startswith("http_"):
                    assert isinstance(exc, ClientHTTPError) and exc.status_code == int(
                        fault.split("_")[1]
                    )
                else:
                    expected = {
                        "malformed_sse": "llama_server_malformed_sse_json",
                        "early_eof": "llama_stream_ended_before_finish_reason",
                        "invalid_usage": "llama_malformed_usage_counter",
                        "socket_disconnect": "llama_server_transport_failure",
                    }[fault]
                    assert str(exc) == expected, str(exc)
                if fault == "early_eof":
                    assert exc.diagnostics.terminal_reason == "early_eof"
            else:
                raise AssertionError("fault unexpectedly succeeded")
            assert adapter.get_last_usage() is None
            assert adapter.get_last_terminal_reason() == "backend_error"
            assert len(state.selected(start, "/v1/chat/completions")) == 1
            recovered = adapter.generate_with_metadata("Reply exactly RECOVERED.")
            assert recovered.text.strip() == "RECOVERED"
            assert recovered.terminal_reason == "completed"
            assert recovered.usage is not None
            assert adapter.get_last_usage() == recovered.usage
            assert adapter.get_last_terminal_reason() == "completed"
            assert len(state.selected(start, "/v1/chat/completions")) == 2
            return {
                "failure": evidence,
                "failing_chat_requests": 1,
                "recovery_success": True,
                "fresh_recovery_usage": asdict(recovered.usage),
            }
        finally:
            adapter.close()

    for fault in [
        "http_401",
        "http_429",
        "http_500",
        "malformed_sse",
        "early_eof",
        "invalid_usage",
        "socket_disconnect",
    ]:
        run("llm_" + fault, lambda fault=fault: llm_fault(fault))

    def no_usage():
        adapter = llm()
        try:
            state.arm("/v1/chat/completions", "missing_usage")
            missing = adapter.generate_with_metadata("Reply exactly RECOVERED.")
            assert missing.text == "RECOVERED" and missing.usage is None
            assert adapter.get_last_usage() is None
            fresh = adapter.generate_with_metadata("Reply exactly RECOVERED.")
            assert fresh.usage is not None and adapter.get_last_usage() == fresh.usage
            return {
                "missing_usage_preserved_unknown": True,
                "following_usage": asdict(fresh.usage),
            }
        finally:
            adapter.close()

    run("llm_missing_usage_then_fresh_usage", no_usage)

    def timeout_then_recover(kind):
        adapter = llm() if kind == "llm" else embedding()
        path = "/v1/chat/completions" if kind == "llm" else "/v1/embeddings"
        try:
            start = len(state.events)
            state.arm(path, "delayed_response")
            started = time.monotonic()
            try:
                if kind == "llm":
                    adapter.generate_with_metadata(
                        "Reply exactly RECOVERED.", timeout=0.04
                    )
                else:
                    adapter.embed_texts(["qualification timed embedding"], timeout=0.04)
            except LLMTimeoutError as exc:
                evidence = exception_evidence(exc)
            else:
                raise AssertionError("delayed response did not timeout")
            elapsed = time.monotonic() - started
            assert elapsed < 1.0
            assert len(state.selected(start, path)) == 1
            if kind == "llm":
                assert adapter.get_last_usage() is None
                assert adapter.get_last_terminal_reason() == "idle_timeout"
                fresh = adapter.generate_with_metadata("Reply exactly RECOVERED.")
                assert fresh.text.strip() == "RECOVERED" and fresh.usage is not None
                time.sleep(0.25)
                assert adapter.get_last_usage() == fresh.usage
                assert adapter.get_last_terminal_reason() == "completed"
            else:
                fresh = adapter.embed_texts(["qualification recovered embedding"])
                validate_vectors(fresh, 1, dimension)
            assert len(state.selected(start, path)) == 2
            return {
                "failure": evidence,
                "timeout_elapsed_ms": round(elapsed * 1000, 3),
                "failing_requests": 1,
                "same_adapter_recovery": True,
                "remote_cancellation_proven": False,
            }
        finally:
            adapter.close()

    for kind in ["llm", "embedding"]:
        run(
            kind + "_injected_timeout_then_recovery",
            lambda kind=kind: timeout_then_recover(kind),
        )

    if args.mode == "fixture":

        def model_guards():
            # Both proxy and upstream are loopback fixtures. A guard regression
            # can never cause this negative self-test to reach an external URL.
            guarded = ProxyState(
                "live_proxy", base_url, llm_model=LLM, embedding_model=EMBED
            )
            guard_server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(guarded))
            guard_thread = Thread(target=guard_server.serve_forever, daemon=True)
            guard_thread.start()
            guard_url = f"http://127.0.0.1:{guard_server.server_port}"
            invalid = [
                ("POST", "/v1/chat/completions", {"messages": []}),
                ("POST", "/v1/chat/completions", {"model": EMBED, "messages": []}),
                ("POST", "/v1/embeddings", {"input": ["fixture"]}),
                ("POST", "/v1/embeddings", {"model": LLM, "input": ["fixture"]}),
                ("POST", "/tokenize", {"content": "fixture"}),
                ("POST", "/tokenize", {"model": LLM, "content": "fixture"}),
                ("GET", "/props", None),
                ("GET", "/props?" + urlencode({"model": LLM}), None),
                (
                    "GET",
                    "/props?" + urlencode([("model", EMBED), ("model", EMBED)]),
                    None,
                ),
            ]
            upstream_before = len(state.events)
            try:
                for method, path, body in invalid:
                    with requests.request(
                        method, guard_url + path, json=body, timeout=2
                    ) as response:
                        assert response.status_code == 400
                        assert (
                            response.json()["error"]["type"]
                            == "qualification_model_guard"
                        )
                assert len(state.events) == upstream_before
                assert all(
                    item.get("model_guard_failure") and not item["upstream_forwarded"]
                    for item in guarded.events
                )
                valid = [
                    ("POST", "/v1/chat/completions", {"model": LLM, "messages": []}),
                    ("POST", "/v1/embeddings", {"model": EMBED, "input": ["fixture"]}),
                    ("POST", "/tokenize", {"model": EMBED, "content": "fixture"}),
                    ("GET", "/props?" + urlencode({"model": EMBED}), None),
                ]
                for method, path, body in valid:
                    with requests.request(
                        method, guard_url + path, json=body, timeout=2
                    ) as response:
                        assert response.status_code == 200
                assert len(state.events) == upstream_before + len(valid)
                assert all(
                    item["model_matches"] and item["upstream_forwarded"]
                    for item in guarded.events[len(invalid) :]
                )
                return {
                    "rejected_before_forwarding": len(invalid),
                    "accepted_exact_model": len(valid),
                    "guard_events": guarded.events,
                    "upstream_scope": "loopback fixture only",
                }
            finally:
                guard_server.shutdown()
                guard_server.server_close()
                guard_thread.join(timeout=2)

        run("fixture_model_selection_guards", model_guards)

    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    summary = {
        "mode": args.mode,
        "fault_provenance": "All failures injected synthetically by local proxy.",
        "success_provenance": "Synthetic loopback fixture"
        if args.mode == "fixture"
        else "Responses forwarded byte for byte from Noxi remote server",
        "upstream": args.upstream if args.mode == "live_proxy" else None,
        "preflight": preflight,
        "source_script_sha256": sha(Path(__file__).read_bytes()),
        "oracle_sha256": sha((ROOT / "recovery-oracle.json").read_bytes()),
        "passed": sum(item["outcome"] == "passed" for item in results),
        "total": len(results),
        "cases": [
            {key: item[key] for key in ["case", "outcome", "duration_ms"]}
            for item in results
        ],
        "http_requests": len(state.events),
        "forwarded_requests": sum(item["upstream_forwarded"] for item in state.events),
        "proxy_failures": [
            item for item in state.events if "proxy_failure_type" in item
        ],
        "model_guard_failures": [
            item for item in state.events if item.get("model_guard_failure")
        ],
    }
    (output_dir / "events.json").write_text(json.dumps(state.events, indent=2))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return (
        0
        if summary["passed"] == summary["total"]
        and not summary["proxy_failures"]
        and not summary["model_guard_failures"]
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
