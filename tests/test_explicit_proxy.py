from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import select
import socket
from threading import Barrier, Event, Lock, Thread
import time
import traceback
from urllib.parse import urlsplit

import pytest
import requests

from runtime_llama import (
    AdapterProfile,
    LlamaCppAdapter,
    LlamaCppClient,
    LlamaServerConfig,
    build_adapter,
    build_embedding_adapter,
    discover_models,
)
from runtime_llama._proxy import normalize_proxy_url
from runtime_llama.client import ClientHTTPError
from runtime_llama.errors import LLMBackendError, LLMCancelledError
from test_generation import _Client, _chunk
from test_optional_stream_idle_transport import (
    endpoint as endpoint,
    headers,
    tls_context as tls_context,
)

TOKEN = "fixture_urlsafe_capability_" + "x" * 32
BASIC = "Basic " + base64.b64encode(f"runtime:{TOKEN}".encode()).decode()
CHAT = (
    b"data: " + json.dumps(_chunk(content="ready")).encode() + b"\n\n"
    b"data: " + json.dumps(_chunk(finish="stop")).encode() + b"\n\n"
    b"data: [DONE]\n\n"
)


def send_json(handler, payload, *, status=200):
    wire = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(wire)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(wire)
    handler.wfile.flush()


@pytest.fixture
def proxy():
    """Authenticated loopback proxy; destinations are mapped to loopback only."""
    calls, stop = [], Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def authorize(self):
            calls.append(
                (self.command, self.path, self.headers.get("Proxy-Authorization"))
            )
            if self.headers.get("Proxy-Authorization") != BASIC:
                send_json(self, {}, status=407)
                self.close_connection = True
                return False
            return True

        def forward(self):
            self.close_connection = True
            if not self.authorize():
                return
            target = urlsplit(self.path)
            assert target.scheme == "http" and target.port
            assert target.hostname in {"127.0.0.1", "tailnet-target.invalid"}
            connection = HTTPConnection("127.0.0.1", target.port, timeout=3)
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                forwarded = {
                    key: value
                    for key, value in self.headers.items()
                    if key.lower()
                    not in {"proxy-authorization", "proxy-connection", "connection"}
                }
                forwarded["Connection"] = "close"
                path = target.path or "/"
                if target.query:
                    path += "?" + target.query
                connection.request(
                    self.command, path, body=body or None, headers=forwarded
                )
                response = connection.getresponse()
                self.send_response(response.status)
                for name, value in response.getheaders():
                    if name.lower() not in {"connection", "transfer-encoding"}:
                        self.send_header(name, value)
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                connection.close()

        do_GET = forward
        do_POST = forward

        def do_CONNECT(self):
            self.close_connection = True
            if not self.authorize():
                return
            target = urlsplit("//" + self.path)
            assert (
                target.hostname in {"127.0.0.1", "tailnet-target.invalid"}
                and target.port
            )
            upstream = socket.create_connection(("127.0.0.1", target.port), timeout=1)
            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.flush()
                peers = [self.connection, upstream]
                while not stop.is_set():
                    readable, _, _ = select.select(peers, [], [], 0.05)
                    for source in readable:
                        data = source.recv(65536)
                        if not data:
                            return
                        destination = (
                            upstream if source is self.connection else self.connection
                        )
                        destination.sendall(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                upstream.close()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    url = f"http://runtime:{TOKEN}@127.0.0.1:{server.server_port}"
    yield url, calls
    stop.set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=1)


def remote_url(origin):
    return origin.replace("127.0.0.1", "tailnet-target.invalid")


def test_config_and_factory_routes_chat_discovery_embeddings_and_tokenization(
    endpoint, proxy, monkeypatch, tmp_path
):
    proxy_url, proxy_calls = proxy
    auth, leaked_proxy_auth = [], []

    def respond(handler, body):
        auth.append(handler.headers.get("Authorization"))
        leaked_proxy_auth.append(handler.headers.get("Proxy-Authorization"))
        path = urlsplit(handler.path).path
        if path == "/v1/chat/completions":
            headers(handler, body_length=len(CHAT))
            handler.wfile.write(CHAT)
        elif path == "/v1/models":
            send_json(
                handler, {"data": [{"id": "chosen", "status": {"value": "loaded"}}]}
            )
        elif path == "/props":
            send_json(
                handler,
                {
                    "model_alias": "chosen",
                    "default_generation_settings": {"n_ctx": 128},
                    "n_batch": 256,
                    "total_slots": 1,
                },
            )
        elif path == "/tokenize":
            send_json(handler, {"tokens": list(range(len(body["content"]) + 1))})
        elif path == "/v1/embeddings":
            send_json(
                handler,
                {
                    "data": [
                        {"index": index, "embedding": [1.0, float(len(text))]}
                        for index, text in enumerate(body["input"])
                    ]
                },
            )
        else:
            pytest.fail("unexpected endpoint")

    origin, origin_calls = endpoint(respond)
    netrc = tmp_path / "netrc"
    netrc.write_text("machine tailnet-target.invalid login wrong password wrong\n")
    monkeypatch.setenv("NETRC", str(netrc))
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "*")
    config = {
        "model": "chosen",
        "base_url": remote_url(origin),
        "proxy_url": proxy_url,
        "api_key": "origin-bearer",
    }
    chat = build_adapter({**config, "stream_idle_timeout_sec": None})
    embedding = build_embedding_adapter({**config, "dimension": 2, "normalize": False})
    try:
        assert chat.generate("fixture") == "ready"
        vectors = embedding.embed_texts(["a", "bb"])
        assert vectors == [[1.0, 1.0], [1.0, 2.0]]
        catalog = discover_models(
            remote_url(origin), proxy_url=proxy_url, api_key="origin-bearer"
        )
        assert [model.id for model in catalog.loaded_models] == ["chosen"]
        assert chat.role_profile.model_config.proxy_url == proxy_url
        assert TOKEN not in repr(chat.role_profile)
        assert TOKEN not in repr(chat._client)
        assert chat._client.trust_env is False
        assert auth and set(auth) == {"Bearer origin-bearer"}
        assert set(leaked_proxy_auth) == {None}
        assert len(origin_calls) == len(proxy_calls)
        assert {path.split("?")[0] for path, _, _ in origin_calls} >= {
            "/v1/chat/completions",
            "/v1/models",
            "/props",
            "/tokenize",
            "/v1/embeddings",
        }
        assert all(value == BASIC for _, _, value in proxy_calls)
        assert all(
            "proxy_url" not in body for _, body, _ in origin_calls if body is not None
        )
    finally:
        chat.close()
        embedding.close()


def test_every_owned_concurrent_lease_uses_explicit_proxy(endpoint, proxy):
    proxy_url, calls = proxy
    barrier, lock, state = Barrier(2), Lock(), {"active": 0, "peak": 0}

    def respond(handler, _body):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            barrier.wait(timeout=1)
            send_json(handler, {})
        finally:
            with lock:
                state["active"] -= 1

    origin, origin_calls = endpoint(respond)
    with LlamaCppClient(remote_url(origin), proxy_url=proxy_url, timeout=1) as client:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = (
                executor.submit(client.models),
                executor.submit(client.props),
            )
            assert first.result(timeout=2) == second.result(timeout=2) == {}
    assert state["peak"] == 2 and len(calls) == len(origin_calls) == 2
    assert all(auth == BASIC for _, _, auth in calls)


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_no_idle_proxy_prefill_and_cancellation_keep_finite_connect_budget(
    endpoint, proxy, phase
):
    proxy_url, calls = proxy
    entered, release = Event(), Event()
    endpoint.release_on_cleanup(release)

    def respond(handler, _body):
        if phase == "body":
            headers(handler)
        entered.set()
        release.wait(timeout=3)
        if phase == "headers":
            headers(handler)
        handler.wfile.write(CHAT)
        handler.wfile.flush()

    origin, origin_calls = endpoint(respond)
    client = LlamaCppClient(remote_url(origin), proxy_url=proxy_url, timeout=0.04)
    adapter = build_adapter(
        {
            "model": "chosen",
            "base_url": remote_url(origin),
            "stream_idle_timeout_sec": None,
        },
        client=client,
    )
    cancel = Event()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                adapter.generate_with_metadata,
                "fixture",
                cancellation_probe=cancel.is_set,
            )
            assert entered.wait(timeout=1)
            time.sleep(0.1)
            assert not pending.done()
            cancel.set()
            with pytest.raises(LLMCancelledError) as caught:
                pending.result(timeout=1)
        assert caught.value.diagnostics.idle_timeout_sec is None
        started = time.monotonic()
        client.close()
        assert time.monotonic() - started < 0.5
        assert len(calls) == len(origin_calls) == 1
    finally:
        release.set()
        adapter.close()
        client.close()


def test_proxy_preserves_typed_http_error_codes_and_sanitized_diagnostics(
    endpoint, proxy
):
    proxy_url, calls = proxy
    origin, origin_calls = endpoint(
        lambda h, _b: send_json(
            h,
            {
                "error": {
                    "code": "context_length_exceeded",
                    "type": "invalid_request_error",
                    "message": TOKEN,
                }
            },
            status=400,
        )
    )
    adapter = build_adapter(
        {
            "model": "chosen",
            "base_url": remote_url(origin),
            "proxy_url": proxy_url,
            "stream_idle_timeout_sec": None,
        }
    )
    try:
        with pytest.raises(ClientHTTPError) as caught:
            adapter.generate("fixture")
        assert caught.value.status_code == 400
        assert caught.value.error_code == "context_length_exceeded"
        assert caught.value.error_type == "invalid_request_error"
        assert caught.value.diagnostics.idle_timeout_sec is None
        assert TOKEN not in "".join(traceback.format_exception(caught.value))
        assert len(calls) == len(origin_calls) == 1
    finally:
        adapter.close()


@pytest.mark.parametrize("transport", ["json", "sse"])
@pytest.mark.parametrize("phase", ["headers", "body"])
@pytest.mark.parametrize("action", ["cancel", "close"])
def test_finite_proxy_calls_interrupt_without_waiting_for_the_read_budget(
    endpoint, proxy, transport, phase, action
):
    proxy_url, calls = proxy
    entered, release, cancel = Event(), Event(), Event()
    endpoint.release_on_cleanup(release)

    def respond(handler, _body):
        if phase == "body":
            if transport == "sse":
                headers(handler)
                handler.wfile.write(b'data: {"part":1}\n\n')
            else:
                handler.send_response(200)
                handler.send_header("Content-Type", "application/json")
                handler.send_header("Content-Length", "11")
                handler.end_headers()
                handler.wfile.write(b'{"data":')
            handler.wfile.flush()
        entered.set()
        release.wait(timeout=3)
        if phase == "headers":
            if transport == "sse":
                headers(handler)
            else:
                send_json(handler, {"data": []})
                return
        handler.wfile.write(b"data: [DONE]\n\n" if transport == "sse" else b"[]}")
        handler.wfile.flush()

    origin, origin_calls = endpoint(respond)
    client = LlamaCppClient(remote_url(origin), proxy_url=proxy_url, timeout=5)

    def invoke():
        if transport == "json":
            return client.models(cancel=cancel.is_set, cancel_wait=True)
        return list(
            client.complete(
                "fixture", stream=True, cancel=cancel.is_set, cancel_wait=True
            )
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(invoke)
            assert entered.wait(timeout=1)
            started = time.monotonic()
            if action == "cancel":
                cancel.set()
                expected = LLMCancelledError
            else:
                client.close()
                expected = LLMBackendError
            if action == "close" and transport == "sse":
                assert pending.result(timeout=1) in ([], [{"part": 1}])
            else:
                with pytest.raises(expected):
                    pending.result(timeout=1)
            assert time.monotonic() - started < 0.5
        assert len(calls) == len(origin_calls) == 1
    finally:
        release.set()
        client.close()


@pytest.mark.filterwarnings("ignore:Unverified HTTPS request")
def test_https_connect_keeps_tls_verification_and_requires_explicit_opt_out(
    endpoint, proxy
):
    proxy_url, calls = proxy

    def respond(handler, _body):
        headers(handler, body_length=len(CHAT))
        handler.wfile.write(CHAT)

    origin, origin_calls = endpoint(respond, tls=True)
    config = {
        "model": "chosen",
        "base_url": remote_url(origin),
        "proxy_url": proxy_url,
        "stream_idle_timeout_sec": None,
    }
    strict = build_adapter(config)
    try:
        with pytest.raises(LLMBackendError) as caught:
            strict.generate("fixture")
        assert TOKEN not in "".join(traceback.format_exception(caught.value))
        assert origin_calls == []
    finally:
        strict.close()
    explicit = build_adapter({**config, "verify_ssl": False})
    try:
        assert explicit.generate("fixture") == "ready"
    finally:
        explicit.close()
    assert [method for method, _, _ in calls] == ["CONNECT", "CONNECT"]
    assert len(origin_calls) == 1


def test_unreachable_proxy_never_falls_back_to_reachable_origin(endpoint):
    origin, origin_calls = endpoint(lambda h, _b: send_json(h, {"data": []}))
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]
    proxy_url = f"http://runtime:{TOKEN}@127.0.0.1:{port}"
    with LlamaCppClient(origin, proxy_url=proxy_url, timeout=0.1) as client:
        with pytest.raises(LLMBackendError) as caught:
            client.models()
        assert TOKEN not in "".join(traceback.format_exception(caught.value))
    assert origin_calls == []


@pytest.mark.parametrize(
    "value",
    [
        True,
        "https://127.0.0.1:8080",
        "http://localhost:8080",
        "http://127.0.0.2:8080",
        "http://127.0.0.1",
        "http://127.0.0.1:0",
        "http://127.0.0.1:65536",
        "http://127.0.0.1:8080/",
        "http://127.0.0.1:8080?",
        "http://127.0.0.1:8080#",
        "http://127.0.0.1:8080\\",
        "http://127.0.0.1:8080\n",
        f"http://wrong:{TOKEN}@127.0.0.1:8080",
        "http://runtime@127.0.0.1:8080",
        "http://runtime:short@127.0.0.1:8080",
        f"http://runtime:{TOKEN}%2F@127.0.0.1:8080",
    ],
)
def test_invalid_proxy_urls_are_sanitized_and_never_dispatched(value):
    for construct in (
        lambda: LlamaCppClient("http://origin.invalid", proxy_url=value),
        lambda: LlamaServerConfig("chosen", proxy_url=value),
        lambda: discover_models("http://origin.invalid", proxy_url=value),
    ):
        with pytest.raises(ValueError, match="proxy_url") as caught:
            construct()
        assert TOKEN not in "".join(traceback.format_exception(caught.value))


def test_proxy_borrowed_session_and_injected_route_mismatch_are_rejected():
    required = f"http://runtime:{TOKEN}@127.0.0.1:8080"
    session = requests.Session()
    try:
        with pytest.raises(ValueError, match="owned session"):
            LlamaCppClient("http://origin.invalid", session=session, proxy_url=required)
        assert session.trust_env is True and session.proxies == {}
    finally:
        session.close()
    with LlamaCppClient("http://origin.invalid", proxy_url=required) as client:
        for build in (build_adapter, build_embedding_adapter):
            with pytest.raises(ValueError, match="required proxy route"):
                build(
                    {"model": "chosen", "proxy_url": "http://127.0.0.1:8081"},
                    client=client,
                )
    with LlamaCppClient("http://origin.invalid") as direct:
        with pytest.raises(ValueError, match="required proxy route"):
            build_adapter({"model": "chosen", "proxy_url": required}, client=direct)


def test_none_defaults_and_injected_matching_route_preserve_compatibility():
    assert normalize_proxy_url(None) is None
    assert normalize_proxy_url("http://127.0.0.1:08080") == "http://127.0.0.1:8080"
    with LlamaCppClient("http://origin.invalid") as direct:
        assert direct.proxy_url is None and direct.trust_env is True
    proxy_url = f"http://runtime:{TOKEN}@127.0.0.1:8080"
    with LlamaCppClient("http://origin.invalid", proxy_url=proxy_url) as client:
        profile = AdapterProfile.from_config(
            "chat", {"model": "chosen", "proxy_url": proxy_url}
        )
        adapter = build_adapter(profile.to_llm_config(), client=client)
        adapter.close()
        assert client.proxy_url == proxy_url and client.trust_env is False


@pytest.mark.parametrize("configured_route", [None, "http://127.0.0.1:8081"])
def test_direct_proxy_cannot_be_silently_overridden_by_model_config(configured_route):
    required = f"http://runtime:{TOKEN}@127.0.0.1:8080"
    client = _Client()
    with pytest.raises(ValueError, match="explicit proxy conflicts") as caught:
        LlamaCppAdapter(
            model_config=LlamaServerConfig("chosen", proxy_url=configured_route),
            proxy_url=required,
            client=client,
        )
    assert TOKEN not in str(caught.value)
    assert client.calls == []


def test_proxy_route_is_managed_and_capability_secrets_are_redacted_from_capture():
    proxy_url = f"http://runtime:{TOKEN}@127.0.0.1:8080"
    client = _Client(
        [
            _chunk(reasoning=proxy_url + " " + TOKEN),
            _chunk(content="ready"),
            _chunk(finish="stop"),
        ]
    )
    client.proxy_url = proxy_url
    adapter = build_adapter(
        {
            "model": "chosen",
            "proxy_url": proxy_url,
            "thinking_diagnostics": {"enabled": True},
        },
        client=client,
    )
    result = adapter.generate_with_metadata("fixture")
    assert TOKEN not in result.thinking_diagnostic.text
    assert TOKEN not in repr(adapter.role_profile)
    with pytest.raises(ValueError, match="overrides_managed_fields"):
        adapter.generate("fixture", request_options={"proxy_url": proxy_url})
