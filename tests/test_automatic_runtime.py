from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Event, Thread

import pytest

from runtime_llama import AdapterProfile, LLMBackendError, LLMCancelledError
from runtime_llama.adapters.automatic import AutoRuntimeAdapter
from runtime_llama.client import LlamaCppClient
from runtime_llama.runtime_discovery import RuntimeCatalog


@pytest.fixture
def runtime():
    running = []

    def start(backend="strata", *, model="shared", loaded=True, answer=None):
        state = {
            "backend": backend,
            "model": model,
            "loaded": loaded,
            "answer": answer or backend,
            "gets": [],
            "posts": [],
            "post_started": Event(),
            "get_started": Event(),
            "release": Event(),
            "hold_post": False,
            "hold_get": False,
        }

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                state["gets"].append((self.path, self.headers.get("Authorization")))
                state["get_started"].set()
                if state["hold_get"]:
                    state["release"].wait(2)
                if self.path == "/health":
                    payload = {
                        "status": "ok",
                        "service": backend,
                        "loaded": state["loaded"],
                        "model": model,
                    }
                elif self.path == "/props":
                    payload = {
                        "build_info": "Strata 0.1.52"
                        if backend == "strata"
                        else "b12345-aabbccd",
                        "model_alias": model,
                        "is_sleeping": not state["loaded"],
                        "default_generation_settings": {"n_ctx": 4096},
                        "total_slots": 1,
                    }
                elif self.path == "/v1/models":
                    payload = {
                        "data": [
                            {
                                "id": model,
                                "aliases": ["requested-alias"],
                                "status": {
                                    "value": "loaded" if state["loaded"] else "unloaded"
                                },
                                "meta": {"n_ctx": 4096},
                            }
                        ]
                    }
                else:
                    self.send_error(404)
                    return
                wire = json.dumps(payload).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(wire)))
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(wire)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                state["posts"].append(
                    (self.path, body, self.headers.get("Authorization"))
                )
                state["post_started"].set()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def send(payload):
                    wire = ("data: " + json.dumps(payload) + "\n\n").encode()
                    self.wfile.write(f"{len(wire):x}\r\n".encode() + wire + b"\r\n")
                    self.wfile.flush()

                try:
                    send(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": state["answer"]},
                                    "finish_reason": None,
                                }
                            ]
                        }
                    )
                    if state["hold_post"]:
                        state["release"].wait(2)
                    send(
                        {
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "stop"}
                            ]
                        }
                    )
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
        )
        thread.start()
        running.append((server, thread, state))
        return {
            "name": backend + str(server.server_port),
            "base_url": f"http://127.0.0.1:{server.server_port}",
        }, state

    yield start
    for server, thread, state in running:
        state["release"].set()
        server.shutdown()
        server.server_close()
        thread.join(1)


def adapter(endpoints, **options):
    profile = AdapterProfile.from_config(
        "test", {"model": "shared", "runtime_endpoints": endpoints, **options}
    )
    return AutoRuntimeAdapter(profile)


def test_construction_is_lazy_and_catalog_lists_every_runtime(runtime):
    strata, a = runtime()
    native, b = runtime("llamacpp", model="other")
    api = adapter([strata, native])
    assert not a["gets"] and not b["gets"]
    catalog = api.discover_models()
    assert [
        (item.backend, [model.id for model in item.loaded_models])
        for item in catalog.runtimes
    ] == [("strata", ["shared"]), ("llamacpp", ["other"])]
    assert catalog.http_requests == 6
    assert not a["posts"] and not b["posts"]
    api.close()


@pytest.mark.parametrize("backend", ["llamacpp", "strata"])
def test_generation_selects_backend_and_returns_request_selection(runtime, backend):
    endpoint, state = runtime(backend)
    api = adapter([endpoint])
    result = api.generate_with_metadata("hello")
    assert result.text == backend
    assert result.runtime_selection.backend == backend
    assert result.runtime_selection.model.id == "shared"
    assert api.selected_runtime == result.runtime_selection
    assert state["posts"][0][1]["model"] == "shared"
    assert not api._runtime_operations
    api.close()


def test_alias_is_resolved_to_canonical_wire_model(runtime):
    endpoint, state = runtime()
    api = adapter([endpoint], model="requested-alias")
    result = api.generate_with_metadata("hello")
    assert result.model == "requested-alias"
    assert result.runtime_selection.model.id == "shared"
    assert state["posts"][0][1]["model"] == "shared"
    api.close()


def test_every_call_reprobes_and_never_uses_stale_route(runtime):
    endpoint, state = runtime()
    api = adapter([endpoint])
    assert api.generate("first") == "strata"
    state["loaded"] = False
    with pytest.raises(LLMBackendError, match="runtime_model_unavailable"):
        api.generate("second")
    assert len(state["gets"]) == 6
    assert len(state["posts"]) == 1
    assert api.selected_runtime is None
    assert not api._runtime_operations
    api.close()


def test_concurrent_stream_keeps_its_route_when_model_moves(runtime):
    first, a = runtime("strata", answer="first")
    second, b = runtime("llamacpp", answer="second", loaded=False)
    api = adapter([first, second])
    stream = api.stream_with_metadata("first")
    event = next(stream)
    assert event.text == "first"
    assert event.runtime_selection.backend == "strata"
    a["loaded"], b["loaded"] = False, True
    with ThreadPoolExecutor(1) as workers:
        result = workers.submit(api.generate_with_metadata, "second").result(2)
    assert result.text == "second"
    assert result.runtime_selection.backend == "llamacpp"
    terminal = list(stream)[-1]
    assert terminal.kind == "terminal"
    assert terminal.runtime_selection.backend == "strata"
    assert len(a["posts"]) == len(b["posts"]) == 1
    assert not api._runtime_operations
    api.close()


def test_endpoint_credentials_do_not_cross_routes(runtime):
    first, a = runtime("strata", model="other")
    second, b = runtime("llamacpp")
    first["api_key"], second["api_key"] = "first-secret", "second-secret"
    api = adapter([first, second], api_key="unused-profile-secret")
    api.generate("hello")
    assert {auth for _, auth in a["gets"]} == {"Bearer first-secret"}
    assert {auth for _, auth in b["gets"]} == {"Bearer second-secret"}
    assert b["posts"][0][2] == "Bearer second-secret"
    assert not a["posts"]
    api.close()


def test_cancellation_before_discovery_sends_nothing(runtime):
    endpoint, state = runtime()
    api = adapter([endpoint])
    with pytest.raises(LLMCancelledError):
        api.generate("hello", cancellation_probe=lambda: True)
    assert not state["gets"] and not state["posts"]
    assert not api._runtime_operations
    api.close()


def test_close_drains_http_while_stream_is_paused(runtime):
    endpoint, state = runtime()
    state["hold_post"] = True
    api = adapter([endpoint], stream_idle_timeout_sec=None)
    stream = api.stream_with_metadata("hello")
    assert next(stream).text == "strata"
    operation = next(iter(api._runtime_operations))
    client = operation.client
    api.close()
    assert client._closed
    assert not client._session_pool
    assert not api._runtime_operations
    with pytest.raises(LLMCancelledError):
        list(stream)
    api.close()


def test_close_cancels_and_drains_in_progress_discovery(runtime):
    endpoint, state = runtime()
    state["hold_get"] = True
    api = adapter([endpoint], runtime_discovery_timeout=0.15)
    with ThreadPoolExecutor(1) as workers:
        result = workers.submit(api.discover_models)
        assert state["get_started"].wait(1)
        api.close()
        with pytest.raises(LLMCancelledError):
            result.result(1)
    assert not state["posts"]
    assert not api._runtime_operations


def test_auto_uses_strata_schema_and_rejects_invalid_success(runtime):
    endpoint, state = runtime(answer='{"wrong": 1}')
    api = adapter(
        [endpoint],
        output_mode="json",
        json_schema={"type": "object", "required": ["ok"]},
    )
    with pytest.raises(LLMBackendError, match="strata_json_schema_mismatch"):
        api.generate("hello")
    assert state["posts"][0][1]["response_format"]["type"] == "json_schema"
    assert not api._runtime_operations
    state["answer"] = '{"ok": true}'
    assert api.generate("hello") == '{"ok": true}'
    api.close()


def test_auto_context_uses_selected_loaded_model(runtime):
    endpoint, state = runtime()
    api = adapter([endpoint])
    with pytest.raises(
        LLMBackendError, match="requested_context_exceeds_loaded_context"
    ):
        api.generate("hello", num_ctx=8192)
    assert not state["posts"]
    api.close()


def test_single_base_url_auto_mode_and_health(runtime):
    endpoint, state = runtime()
    profile = AdapterProfile.from_config(
        "test", {"backend": "auto", "model": "shared", "base_url": endpoint["base_url"]}
    )
    api = AutoRuntimeAdapter(profile)
    health = api.health_status()
    assert health["backend"] == "strata"
    assert health["model_ready"] is True
    assert health["loaded_context_length"] == 4096
    assert api.generate("hello") == "strata"
    assert len(state["posts"]) == 1
    api.close()


def test_close_waits_for_an_operation_already_draining(runtime, monkeypatch):
    endpoint, _ = runtime()
    api = adapter([endpoint])
    drain_entered, allow_drain, closed = Event(), Event(), Event()
    original = LlamaCppClient.close

    def delayed_close(client):
        with api._runtime_lock:
            owned = any(
                operation.client is client for operation in api._runtime_operations
            )
        if owned:
            drain_entered.set()
            assert allow_drain.wait(2)
        original(client)

    monkeypatch.setattr(LlamaCppClient, "close", delayed_close)

    def close():
        api.close()
        closed.set()

    with ThreadPoolExecutor(2) as workers:
        result = workers.submit(api.generate, "hello")
        assert drain_entered.wait(1)
        shutdown = workers.submit(close)
        try:
            assert not closed.wait(0.03)
        finally:
            allow_drain.set()
        assert result.result(1) == "strata"
        shutdown.result(1)
    assert not api._runtime_operations


def test_healthy_runtime_without_requested_model_reports_health_separately(runtime):
    endpoint, _ = runtime(model="other")
    api = adapter([endpoint])
    status = api.health_status()
    assert status["healthy"] is True
    assert status["model_available"] is False
    assert status["model_ready"] is False
    api.close()


def test_concurrent_route_publication_keeps_catalog_and_selection_coherent(
    runtime, monkeypatch
):
    first, a = runtime("strata")
    second, b = runtime("llamacpp", loaded=False)
    api = adapter([first, second])
    observed, release = Event(), Event()
    original = api._discover

    def discover(operation, cancel, timeout=None):
        catalog = original(operation, cancel, timeout)
        if not observed.is_set():
            observed.set()
            assert release.wait(2)
        return catalog

    monkeypatch.setattr(api, "_discover", discover)
    with ThreadPoolExecutor(2) as workers:
        pending = workers.submit(api.generate_with_metadata, "first")
        assert observed.wait(1)
        a["loaded"], b["loaded"] = False, True
        second_result = workers.submit(api.generate_with_metadata, "second").result(1)
        assert second_result.runtime_selection.backend == "llamacpp"
        release.set()
        first_result = pending.result(1)
    assert first_result.runtime_selection.backend == "strata"
    assert api.selected_runtime == first_result.runtime_selection
    assert api.selected_runtime.runtime in api.current_catalog.runtimes
    assert api.current_catalog.select("shared").backend == "strata"
    api.close()


def test_health_cannot_republish_selection_after_close(runtime, monkeypatch):
    endpoint, _ = runtime()
    api = adapter([endpoint])
    selecting, release = Event(), Event()
    original = RuntimeCatalog.select

    def select(catalog, *args, **kwargs):
        selecting.set()
        assert release.wait(2)
        return original(catalog, *args, **kwargs)

    monkeypatch.setattr(RuntimeCatalog, "select", select)
    with ThreadPoolExecutor(1) as workers:
        result = workers.submit(api.health_status)
        assert selecting.wait(1)
        api.close()
        release.set()
        with pytest.raises(LLMCancelledError):
            result.result(1)
    assert api.selected_runtime is None
    assert api.current_catalog is None
    assert not api._runtime_operations
