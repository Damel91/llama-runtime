"""TCP relay qualification against deterministic native HTTP fixtures.

These are transport/lifecycle checks, not live-model numerical qualification
or measurements of GPU execution. No tools proposed by fixtures are executed.
"""

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import gc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import chain
import json
import select
import socket
from threading import Event, Lock, Thread, enumerate as threads
import time
from urllib.parse import urlsplit
import weakref

import httpx
import pytest
import uvicorn

from runtime_llama import build_embedding_adapter
from runtime_llama.relay.app import create_app
from runtime_llama.relay.config import RelayConfigStore, default_config
from runtime_llama.relay.service import RelayService
from test_generation import _chunk


INSPECT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    }
]


def eventually(predicate, timeout=5):
    """Finite fixture assertion, not an inference policy or admission limit."""
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "fixture did not finish draining"
        time.sleep(0.005)


def vector(text):
    return [float(text.split(":")[1]), float(len(text))]


@contextmanager
def native_server(kind="llamacpp", chat_responses=None):
    state = {
        "requests": Counter(),
        "batches": [],
        "texts": 0,
        "chat_requests": [],
        "active": 0,
        "quiet_started": Event(),
        "peer_closed": Event(),
        "finish": Event(),
    }
    lock = Lock()
    model = "qualification-embedding" if kind == "llamacpp" else "qualification-chat"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def reply(self, value, *, status=200):
            wire = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(wire)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(wire)
            self.close_connection = True

        def stream_frames(self, frames):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            wire = (
                b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames
            )
            for event in chain(wire, (b"data: [DONE]\n\n",)):
                self.wfile.write(f"{len(event):x}\r\n".encode() + event + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.close_connection = True

        def do_GET(self):
            path = urlsplit(self.path).path
            with lock:
                state["requests"]["GET " + path] += 1
            if path == "/health":
                body = {"status": "ok"}
                if kind == "strata":
                    body.update(service="strata", loaded=True, model=model)
            elif path == "/v1/models":
                body = {
                    "data": [
                        {
                            "id": model,
                            "status": {"value": "loaded"},
                            "capabilities": {
                                "chat": kind == "strata",
                                "embeddings": kind == "llamacpp",
                            },
                        }
                    ]
                }
            elif path == "/props":
                body = {
                    "build_info": "b12345-aabbcc"
                    if kind == "llamacpp"
                    else "Strata 0.1.52",
                    "model_alias": model,
                    "is_sleeping": False,
                    "total_slots": 4,
                    "default_generation_settings": {"n_ctx": 2048},
                }
            else:
                self.reply({"error": "unexpected fixture endpoint"}, status=404)
                return
            self.reply(body)

        def quiet(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.flush()
            state["quiet_started"].set()
            while not state["finish"].is_set():
                readable, _, _ = select.select([self.connection], [], [], 0.02)
                if readable and not self.connection.recv(1, socket.MSG_PEEK):
                    state["peer_closed"].set()
                    break
            self.close_connection = True

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state["requests"]["POST " + self.path] += 1
                state["active"] += 1
            try:
                if self.path == "/v1/embeddings":
                    # Only aggregate counters survive the native response. The
                    # fixture itself does not retain these 10,000 input texts.
                    with lock:
                        state["batches"].append(len(body["input"]))
                        state["texts"] += len(body["input"])
                    self.reply(
                        {
                            "data": [
                                {"index": index, "embedding": vector(text)}
                                for index, text in reversed(
                                    list(enumerate(body["input"]))
                                )
                            ]
                        }
                    )
                elif self.path == "/v1/chat/completions":
                    with lock:
                        index = len(state["chat_requests"])
                        state["chat_requests"].append(body)
                    frames = chat_responses(index, body)
                    if frames == "quiet":
                        self.quiet()
                    else:
                        self.stream_frames(frames)
                else:
                    self.reply({"error": "unexpected fixture endpoint"}, status=404)
            finally:
                with lock:
                    state["active"] -= 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    thread = Thread(
        target=lambda: server.serve_forever(poll_interval=0.01),
        name="relay-qualification-native",
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", model, state
    finally:
        state["finish"].set()
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive() and state["active"] == 0


@contextmanager
def tcp_relay(config, **factories):
    """Actual Uvicorn TCP listener, with the admitted app shutdown hook."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    configuration = deepcopy(config)
    configuration["listener"]["port"] = listener.getsockname()[1]
    service = RelayService(RelayConfigStore(initial_config=configuration), **factories)
    app = create_app(service=service)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=listener.getsockname()[1],
            log_level="critical",
            access_log=False,
        )
    )
    failures = []

    def run():
        try:
            server.run(sockets=[listener])
        except BaseException as error:
            failures.append(error)

    thread = Thread(target=run, name="relay-qualification-uvicorn")
    thread.start()
    try:
        eventually(lambda: server.started or failures)
        assert not failures
        with httpx.Client(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}",
            timeout=20,
            trust_env=False,
        ) as api:
            yield api, app, service
    finally:
        app.state.relay_cancel_all()
        server.should_exit = True
        thread.join(10)
        listener.close()
        assert not thread.is_alive(), "TCP relay did not drain before shutdown"
        assert not failures
        assert not app.state.relay_bridges
        assert service.registry.snapshot()["active"] == []


def configuration(url, *, embedding=False, batch_size=None):
    config = default_config()
    config["runtime_endpoints"] = [{"name": "qualification", "base_url": url}]
    config["llm_defaults"] = {"stream_idle_timeout_sec": None}
    if embedding:
        config["embedding_defaults"] = {
            "input_type": "raw",
            "normalize": False,
            "timeout": 20,
            "batch_size": batch_size,
        }
    return config


def admin_headers(api):
    origin = str(api.base_url).rstrip("/")
    response = api.get("/api/admin/session", headers={"Origin": origin})
    assert response.status_code == 200
    return {"Origin": origin, "X-Relay-Admin-Token": response.json()["admin_token"]}


@pytest.mark.parametrize(
    "batch_size", [None, 10000], ids=["progressive-default", "explicit-large-batch"]
)
def test_tcp_10000_direct_texts_release_records_owners_handoff_and_transports(
    batch_size,
):
    adapters, records = [], []
    before = {thread.ident for thread in threads()}

    def factory(profile):
        adapter = build_embedding_adapter(profile)
        adapters.append(weakref.ref(adapter))
        original = adapter._engine_prepare

        def observed(*args, **kwargs):
            record = original(*args, **kwargs)
            records.append(weakref.ref(record))
            return record

        adapter._engine_prepare = observed
        return adapter

    with native_server() as (url, model, native):
        with tcp_relay(
            configuration(url, embedding=True, batch_size=batch_size),
            embedding_factory=factory,
        ) as (api, app, service):
            texts = [f"row:{index}:東京🙂" for index in range(10000)]
            started = time.monotonic()
            response = api.post("/v1/embeddings", json={"model": model, "input": texts})
            elapsed = time.monotonic() - started
            assert response.status_code == 200, response.text
            payload = response.json()
            assert [row["index"] for row in payload["data"]] == list(range(10000))
            assert [row["embedding"] for row in payload["data"]] == [
                vector(text) for text in texts
            ]
            assert "usage" not in payload
            assert native["texts"] == 10000 and len(records) == 10000
            assert sum(native["batches"]) == 10000
            if batch_size == 10000:
                assert native["batches"] == [10000]
            assert native["requests"]["POST /tokenize"] == 0
            adapter = adapters[0]()
            assert adapter is not None and len(adapters) == 1
            snapshot = adapter._engine.snapshot()
            for key in (
                "pending_bytes",
                "pending_tokens",
                "retained_result_bytes",
                "result_reserved_bytes",
                "active_plans",
                "embedding_in_flight",
                "control_in_flight",
                "original_leases_active",
            ):
                assert snapshot[key] == 0, (key, snapshot)
            assert snapshot["handoff"]["used"] == snapshot["handoff"]["tickets"] == 0
            assert adapter._intake.snapshot()["owners"] == 0 and not adapter._calls
            assert (
                adapter._scheduler.snapshot()["queued"]
                == adapter._scheduler.snapshot()["in_flight"]
                == 0
            )
            assert not adapter._engine._states and not adapter._engine._incoming
            assert (
                not service.registry.snapshot()["active"]
                and not app.state.relay_bridges
            )
            gc.collect()
            assert all(record() is None for record in records)
            print(
                "RELAY_QUALIFICATION "
                + json.dumps(
                    {
                        "scenario": "10000-direct-texts",
                        "configured_batch_size": batch_size,
                        "native_http_requests": dict(native["requests"]),
                        "native_embedding_batches": native["batches"],
                        "total_texts": native["texts"],
                        "http_roundtrip_seconds": elapsed,
                        "gpu_time_measured": False,
                        "owned_records_after_gc": sum(
                            record() is not None for record in records
                        ),
                        "handoff_used_after_call": snapshot["handoff"]["used"],
                    }
                )
            )
        assert adapter._closed and adapter._client._closed
        assert not adapter._client._streams and not adapter._client._session_pool
        assert not adapter._engine._thread.is_alive()
        assert not service._cached and not service._retired
        assert service.generations.wait_drained(0)
        del adapter
        gc.collect()
        assert all(reference() is None for reference in adapters)
    eventually(
        lambda: (
            not [
                thread
                for thread in threads()
                if thread.ident not in before
                and (
                    thread.name.startswith("runtime_llama_")
                    or thread.name == "runtime-llama-relay-call"
                )
            ]
        )
    )


def test_tcp_tool_sse_streams_guarded_text_before_tool_validation(monkeypatch):
    monkeypatch.setattr("runtime_llama.relay.app._SSE_KEEPALIVE_SECONDS", 0.02)
    release = Event()
    call = {
        "index": 0,
        "id": "accepted-after-validation",
        "type": "function",
        "function": {"name": "inspect", "arguments": '{"key":"fixture"}'},
    }

    def responses(_index, _request):
        yield _chunk(reasoning="held until validation")
        yield _chunk(tools=[call])
        yield _chunk(content="<think>INLINE-PRIVATE</think>Visible ")
        yield _chunk(content="before validation.")
        if release.wait(5):
            yield _chunk(finish="tool_calls")

    with native_server("strata", responses) as (url, model, native):
        with tcp_relay(configuration(url)) as (api, _app, service):
            try:
                with api.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "synthetic request"}],
                        "tools": INSPECT_TOOLS,
                        "stream": True,
                    },
                ) as response:
                    assert response.status_code == 200
                    lines = response.iter_lines()
                    before = []
                    for line in lines:
                        if line.startswith("data: "):
                            assert line != "data: [DONE]"
                            choice = json.loads(line[6:])["choices"][0]
                            assert choice["finish_reason"] is None
                            before.append(choice["delta"])
                        if line == ": keep-alive" and any(
                            delta.get("content") == "before validation."
                            for delta in before
                        ):
                            break
                    assert before == [
                        {"role": "assistant"},
                        {"content": "Visible "},
                        {"content": "before validation."},
                    ]
                    assert native["active"] == 1
                    assert not release.is_set()
                    release.set()
                    after = list(lines)
            finally:
                release.set()

            assert after[-2] == "data: [DONE]"
            choices = [
                json.loads(line[6:])["choices"][0]
                for line in after
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            assert [choice["delta"] for choice in choices] == [
                {"reasoning_content": "held until validation"},
                {"tool_calls": [call]},
                {},
            ]
            assert [choice["finish_reason"] for choice in choices] == [
                None,
                None,
                "tool_calls",
            ]
            assert native["requests"]["POST /v1/chat/completions"] == 1
            assert service.registry.snapshot()["active"] == []


def test_tcp_tool_only_fragmented_multiple_calls_and_tool_result_history():
    def responses(index, _request):
        if index:
            return [_chunk(content="roundtrip-ok"), _chunk(finish="stop")]
        return [
            _chunk(reasoning="accepted-reasoning"),
            _chunk(
                tools=[
                    {
                        "index": 0,
                        "id": "call-first",
                        "type": "function",
                        "function": {"name": "inspect", "arguments": '{"key":'},
                    }
                ]
            ),
            _chunk(
                tools=[
                    {
                        "index": 1,
                        "id": "call-second",
                        "type": "function",
                        "function": {
                            "name": "inspect",
                            "arguments": '{"key":"second"}',
                        },
                    }
                ]
            ),
            _chunk(tools=[{"index": 0, "function": {"arguments": '"first"}'}}]),
            _chunk(finish="tool_calls"),
        ]

    with native_server("strata", responses) as (url, model, native):
        with tcp_relay(configuration(url)) as (api, _app, service):
            initial = [{"role": "user", "content": "synthetic request"}]
            with api.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": initial,
                    "tools": INSPECT_TOOLS,
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200
                lines = list(response.iter_lines())
            assert lines[-2] == "data: [DONE]"
            chunks = [
                json.loads(line[6:])
                for line in lines
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            calls = {}
            reasoning = ""
            for chunk in chunks:
                for choice in chunk["choices"]:
                    delta = choice["delta"]
                    reasoning += delta.get("reasoning_content", "")
                    assert not delta.get("content")
                    for part in delta.get("tool_calls", []):
                        call = calls.setdefault(
                            part["index"],
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        call["id"] += part.get("id", "")
                        for name in ("name", "arguments"):
                            call["function"][name] += part.get("function", {}).get(
                                name, ""
                            )
            structured = [calls[index] for index in sorted(calls)]
            assert [call["id"] for call in structured] == ["call-first", "call-second"]
            assert [
                json.loads(call["function"]["arguments"])["key"] for call in structured
            ] == ["first", "second"]
            assert reasoning == "accepted-reasoning"
            assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
            history = [
                *initial,
                {"role": "assistant", "content": None, "tool_calls": structured},
                *[
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": "synthetic result",
                    }
                    for call in structured
                ],
            ]
            followup = api.post(
                "/v1/chat/completions",
                json={"model": model, "messages": history, "tools": INSPECT_TOOLS},
            )
            assert (
                followup.status_code == 200
                and followup.json()["choices"][0]["message"]["content"]
                == "roundtrip-ok"
            )
            assert native["chat_requests"][1]["messages"] == history
            assert native["requests"]["POST /v1/chat/completions"] == 2
            assert (
                service.registry.snapshot()["last_completed"]["corrections_used"] == 0
            )


@pytest.mark.parametrize("mode", ["disconnect", "stop", "shutdown"])
def test_tcp_quiet_stream_cancellation_closes_native_socket_and_drains(mode):
    with native_server("strata", lambda *_args: "quiet") as (url, model, native):
        with tcp_relay(configuration(url)) as (api, app, service):
            headers = admin_headers(api)
            with api.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "quiet fixture"}],
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200
                assert native["quiet_started"].wait(5)
                if mode == "disconnect":
                    response.close()
                elif mode == "stop":
                    active = api.get("/api/admin/requests", headers=headers).json()[
                        "active"
                    ]
                    request_id = next(
                        request["request_id"]
                        for request in active
                        if request["kind"] == "chat"
                    )
                    stopped = api.post(
                        f"/api/admin/requests/{request_id}/cancel", headers=headers
                    )
                    assert stopped.status_code == 200
                else:
                    app.state.relay_cancel_all()
                assert native["peer_closed"].wait(5)
                eventually(
                    lambda: (
                        not app.state.relay_bridges
                        and not service.registry.snapshot()["active"]
                    )
                )
            assert native["requests"]["POST /v1/chat/completions"] == 1
            assert (
                service.registry.snapshot()["last_completed"]["status"] == "cancelled"
            )
            assert not service._retired
            print(
                "RELAY_QUALIFICATION "
                + json.dumps(
                    {
                        "scenario": "quiet-" + mode,
                        "native_http_requests": dict(native["requests"]),
                        "native_peer_closed": native["peer_closed"].is_set(),
                        "active_relay_requests_after_cancel": len(
                            service.registry.snapshot()["active"]
                        ),
                    }
                )
            )
