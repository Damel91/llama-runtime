from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Barrier, Event, Thread, enumerate as threads
import time
import traceback

import pytest
import requests

from runtime_llama.client import ClientHTTPError, LlamaCppClient
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError


@pytest.fixture
def endpoint():
    servers = []

    def start(respond):
        observed = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                observed.append((self.path, self.headers.get("Authorization")))
                try:
                    respond(self)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        serving = Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
        )
        serving.start()
        servers.append((server, serving))
        return f"http://127.0.0.1:{server.server_port}", observed

    yield start
    for server, serving in servers:
        server.shutdown()
        server.server_close()
        serving.join(timeout=1)


def send(handler, body, *, status=200, chunked=False, compressed=False, extra=()):
    wire = gzip.compress(body) if compressed else body
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Connection", "close")
    if compressed:
        handler.send_header("Content-Encoding", "gzip")
    if chunked:
        handler.send_header("Transfer-Encoding", "chunked")
    else:
        handler.send_header("Content-Length", str(len(wire)))
    for name, value in extra:
        handler.send_header(name, value)
    handler.end_headers()
    if chunked:
        for index in range(0, len(wire), 7):
            chunk = wire[index : index + 7]
            handler.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
        handler.wfile.write(b"0\r\n\r\n")
    else:
        handler.wfile.write(wire)
    handler.wfile.flush()


def wait_for(predicate, timeout=1):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert predicate()


@pytest.mark.parametrize("method", ["request", "models", "props"])
@pytest.mark.parametrize(
    "chunked,compressed", [(False, False), (True, False), (True, True)]
)
def test_json_bound_counts_decompressed_utf8_bytes_at_exact_boundary(
    endpoint, method, chunked, compressed
):
    payload = {"data": [{"id": "modèle-🦙"}]}
    body = json.dumps(payload, ensure_ascii=False).encode()
    base_url, observed = endpoint(
        lambda handler: send(handler, body, chunked=chunked, compressed=compressed)
    )
    with LlamaCppClient(base_url, timeout=1, trust_env=False) as api:

        def invoke(**kwargs):
            if method == "request":
                return api.request("GET", "/v1/models", **kwargs)
            return getattr(api, method)(**kwargs)

        assert invoke(max_response_bytes=len(body)) == payload
        with pytest.raises(LLMBackendError, match="json_response_too_large"):
            invoke(max_response_bytes=len(body) - 1)
        # A rejected response leaves the client usable, without implicit retry.
        assert invoke(max_response_bytes=len(body)) == payload
        assert not api._streams
    assert len(observed) == 3


@pytest.mark.parametrize("chunked", [False, True])
def test_gzip_expansion_is_bounded_even_when_compressed_body_fits(endpoint, chunked):
    body = json.dumps({"data": [{"id": "x" * 10000}]}).encode()
    assert len(gzip.compress(body)) < 128 < len(body)
    base_url, observed = endpoint(
        lambda handler: send(handler, body, compressed=True, chunked=chunked)
    )
    with LlamaCppClient(base_url, timeout=1, trust_env=False) as api:
        with pytest.raises(LLMBackendError, match="json_response_too_large"):
            api.models(max_response_bytes=128)
        assert not api._streams
    assert len(observed) == 1


def test_oversized_http_error_keeps_status_without_echoing_body(endpoint):
    secret = "fixture-private-provider-message"
    body = json.dumps(
        {"error": {"code": "unauthorized", "message": secret * 100}}
    ).encode()
    base_url, observed = endpoint(lambda handler: send(handler, body, status=401))
    with LlamaCppClient(base_url, timeout=1, trust_env=False) as api:
        with pytest.raises(ClientHTTPError) as caught:
            api.models(max_response_bytes=64)
        assert caught.value.status_code == 401
        assert caught.value.error_code is None
        assert caught.value.error_type is None
        assert secret not in "".join(traceback.format_exception(caught.value))
        assert not api._streams
    assert len(observed) == 1


def test_redirect_is_rejected_without_forwarding_credentials(endpoint):
    target, forwarded = endpoint(lambda handler: send(handler, b"{}"))
    base_url, observed = endpoint(
        lambda handler: send(
            handler, b"{}", status=302, extra=[("Location", target + "/v1/models")]
        )
    )
    with LlamaCppClient(
        base_url, api_key="fixture-bearer", timeout=1, trust_env=False
    ) as api:
        with pytest.raises(ClientHTTPError) as caught:
            api.models(max_response_bytes=64)
    assert caught.value.status_code == 302
    assert observed == [("/v1/models", "Bearer fixture-bearer")]
    assert forwarded == []


@pytest.mark.parametrize("trust_env", [False, True])
def test_explicit_environment_policy_controls_ambient_netrc_auth(
    endpoint, monkeypatch, trust_env
):
    monkeypatch.setattr(
        requests.sessions,
        "get_netrc_auth",
        lambda _url: ("fixture-netrc-user", "fixture-netrc-password"),
    )
    base_url, observed = endpoint(lambda handler: send(handler, b"{}"))
    with LlamaCppClient(
        base_url, api_key="fixture-bearer", timeout=1, trust_env=trust_env
    ) as api:
        assert api.models(max_response_bytes=64) == {}
    auth = observed[0][1]
    assert auth.startswith("Basic ") if trust_env else auth == "Bearer fixture-bearer"


def test_trust_env_does_not_reconfigure_or_close_borrowed_session(
    endpoint, monkeypatch
):
    monkeypatch.setattr(
        requests.sessions,
        "get_netrc_auth",
        lambda _url: ("fixture-netrc-user", "fixture-netrc-password"),
    )
    base_url, observed = endpoint(lambda handler: send(handler, b"{}"))
    session = requests.Session()
    closed = []
    close = session.close
    monkeypatch.setattr(session, "close", lambda: closed.append(True))
    try:
        with LlamaCppClient(
            base_url, session=session, timeout=1, trust_env=False
        ) as api:
            assert api.models(max_response_bytes=64) == {}
            assert session.trust_env is True
        assert session.trust_env is True
        assert closed == []
        assert observed[0][1].startswith("Basic ")
    finally:
        close()


def test_all_owned_pool_sessions_use_policy_and_bound_failures_are_isolated(
    endpoint, monkeypatch
):
    rendezvous = Barrier(2, timeout=2)
    sessions = []
    original_session = requests.Session

    class TrackedSession(original_session):
        def __init__(self):
            super().__init__()
            self.close_count = 0
            sessions.append(self)

        def close(self):
            self.close_count += 1
            super().close()

    monkeypatch.setattr(requests, "Session", TrackedSession)
    monkeypatch.setattr(
        requests.sessions,
        "get_netrc_auth",
        lambda _url: ("fixture-netrc-user", "fixture-netrc-password"),
    )

    def respond(handler):
        rendezvous.wait()
        if handler.path == "/v1/models":
            send(handler, json.dumps({"data": ["x" * 1000]}).encode())
        else:
            send(handler, b'{"n_ctx":2048}')

    base_url, observed = endpoint(respond)
    with LlamaCppClient(
        base_url, api_key="fixture-bearer", timeout=1, trust_env=False
    ) as api:
        with ThreadPoolExecutor(max_workers=2) as executor:
            oversized = executor.submit(api.models, max_response_bytes=64)
            healthy = executor.submit(api.props, max_response_bytes=64)
            with pytest.raises(LLMBackendError, match="json_response_too_large"):
                oversized.result(timeout=2)
            assert healthy.result(timeout=2) == {"n_ctx": 2048}
        assert len(sessions) == 2
        assert all(session.trust_env is False for session in sessions)
        assert not api._streams
    assert all(session.close_count == 1 for session in sessions)
    assert len(observed) == 2
    assert all(auth == "Bearer fixture-bearer" for _path, auth in observed)


@pytest.mark.parametrize("termination", ["cancel", "timeout"])
def test_bounded_json_idle_wait_cancels_and_drains(endpoint, termination):
    entered, release, requested = Event(), Event(), Event()

    def respond(handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", "100")
        handler.send_header("Connection", "close")
        handler.end_headers()
        entered.set()
        release.wait(timeout=2)

    base_url, observed = endpoint(respond)
    api = LlamaCppClient(base_url, timeout=0.1, trust_env=False)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                api.models,
                cancel=requested.is_set,
                cancel_wait=True,
                max_response_bytes=128,
            )
            assert entered.wait(timeout=1)
            if termination == "cancel":
                requested.set()
            expected = LLMCancelledError if termination == "cancel" else LLMTimeoutError
            with pytest.raises(expected):
                pending.result(timeout=0.75)
        assert not api._streams
        assert len(observed) == 1
    finally:
        release.set()
        api.close()


def test_tls_verification_and_response_limit_are_not_request_payload_fields(
    monkeypatch,
):
    seen = []

    class Session(requests.Session):
        def request(self, **kwargs):
            seen.append(kwargs)
            response = requests.Response()
            response.status_code = 200
            response._content = b"{}"
            return response

    monkeypatch.setattr(requests, "Session", Session)
    with LlamaCppClient("https://chosen.example", trust_env=False) as api:
        assert api.models(max_response_bytes=2) == {}
        assert api.props("model name", max_response_bytes=2) == {}
    assert len(seen) == 2
    assert all(call["verify"] is True for call in seen)
    assert all(call["allow_redirects"] is False for call in seen)
    assert all(call["json"] is None for call in seen)
    assert all("max_response_bytes" not in call for call in seen)
    assert seen[1]["url"] == "https://chosen.example/props?model=model+name"


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "128"])
@pytest.mark.parametrize("method", ["request", "models", "props"])
def test_response_limit_requires_positive_integer_before_dispatch(value, method):
    with LlamaCppClient("http://chosen.example", trust_env=False) as api:
        with pytest.raises(ValueError, match="max_response_bytes"):
            if method == "request":
                api.request("GET", "/v1/models", max_response_bytes=value)
            else:
                getattr(api, method)(max_response_bytes=value)
        assert not api._streams


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_environment_policy_requires_boolean(value):
    with pytest.raises(ValueError, match="trust_env"):
        LlamaCppClient("http://chosen.example", trust_env=value)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://fixture-user:fixture-password@chosen.example",
        "http://@chosen.example",
        "http://:fixture-password@chosen.example",
        "http://chosen.example:fixture-invalid-port",
        "http://chosen.example:65536",
        "http://chosen.example:0",
        "http://[invalid-address",
        "http://chosen.example\n",
        "http://chosen.example/has space",
        "http://chosen.example/\tmetadata",
        "http://chosen.example/\x00metadata",
        "http://chosen.example/\x7fmetadata",
        "http://chosen.example/\x80metadata",
        "http://chosen.example/\x9fmetadata",
        "http://chosen.example/\u2003metadata",
        "http://chosen.example\\other.example",
        "http://chosen.example?fixture-query=secret",
        "http://chosen.example#fixture-fragment",
        "ftp://chosen.example",
        "http:///missing-host",
        None,
    ],
)
def test_invalid_endpoint_is_rejected_before_session_creation_without_echo(
    monkeypatch, base_url
):
    created = []
    monkeypatch.setattr(requests, "Session", lambda: created.append(True))
    with pytest.raises(ValueError) as caught:
        LlamaCppClient(base_url)
    assert created == []
    message = str(caught.value)
    assert "fixture-" not in message
    assert "chosen.example" not in message
    assert "base_url must be" in message


def test_reader_threads_are_drained_after_bounded_calls(endpoint):
    base_url, _observed = endpoint(lambda handler: send(handler, b"{}"))
    with LlamaCppClient(base_url, timeout=1, trust_env=False) as api:
        assert api.models(max_response_bytes=2) == {}
    names = {
        "runtime_llama_json_reader",
        "runtime_llama_http_close",
    }
    wait_for(lambda: not any(thread.name in names for thread in threads()))
