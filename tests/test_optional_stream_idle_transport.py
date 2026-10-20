from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import shutil
import ssl
import subprocess
from threading import Event, Thread
import time

import pytest
import requests

from runtime_llama.client import ClientHTTPError, LlamaCppClient
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError
from runtime_llama._http_interrupt import InterruptibleAdapter, RequestInterrupt


def wait_for(predicate, timeout=1):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert predicate()


@pytest.fixture(scope="module")
def tls_context(tmp_path_factory):
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("TLS loopback fixture requires the openssl executable")
    directory = tmp_path_factory.mktemp("idle-tls")
    key, cert = directory / "fixture.key", directory / "fixture.crt"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    return context


@pytest.fixture
def endpoint(request):
    servers, releases = [], []

    def start(respond, *, tls=False, keepalive=False):
        observed = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                observed.append((self.path, body, self.client_address))
                try:
                    respond(self, body)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                    pass
                if not keepalive:
                    self.close_connection = True

            def do_GET(self):
                observed.append((self.path, None, self.client_address))
                respond(self, None)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        if tls:
            context = request.getfixturevalue("tls_context")
            server.socket = context.wrap_socket(server.socket, server_side=True)
        serving = Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
        )
        serving.start()
        servers.append((server, serving))
        scheme = "https" if tls else "http"
        return f"{scheme}://127.0.0.1:{server.server_port}", observed

    start.release_on_cleanup = releases.append
    yield start
    for release in releases:
        release.set()
    for server, serving in servers:
        server.shutdown()
        server.server_close()
        serving.join(timeout=1)


FIRST = b'data: {"part":1}\n\n'
SECOND = b'data: {"part":2}\n\ndata: [DONE]\n\n'


def headers(handler, *, status=200, body_length=None, keepalive=False):
    handler.send_response(status)
    handler.send_header("Content-Type", "text/event-stream")
    if body_length is not None:
        handler.send_header("Content-Length", str(body_length))
    if not keepalive:
        handler.send_header("Connection", "close")
    handler.end_headers()


def stalled(endpoint, phase, *, tls=False):
    entered, release = Event(), Event()
    endpoint.release_on_cleanup(release)

    def respond(handler, _body):
        if phase == "headers":
            entered.set()
            release.wait(timeout=3)
        headers(handler)
        if phase == "between":
            handler.wfile.write(FIRST)
            handler.wfile.flush()
        if phase != "headers":
            entered.set()
            release.wait(timeout=3)
        handler.wfile.write(SECOND)
        handler.wfile.flush()

    base_url, observed = endpoint(respond, tls=tls)
    return base_url, observed, entered, release


@pytest.mark.parametrize("phase", ["headers", "first", "between"])
def test_none_waits_past_finite_connect_budget_without_retry(endpoint, phase):
    base_url, observed, entered, release = stalled(endpoint, phase)
    with LlamaCppClient(base_url, timeout=0.04, trust_env=False) as api:
        stream = api.complete("fixture", stream=True, idle_timeout=None)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(list, stream)
            assert entered.wait(timeout=1)
            time.sleep(0.08)
            assert not pending.done()
            release.set()
            expected = (
                [{"part": 1}, {"part": 2}] if phase == "between" else [{"part": 2}]
            )
            assert pending.result(timeout=1) == expected
        assert not api._streams
    assert len(observed) == 1
    assert "idle_timeout" not in observed[0][1]


@pytest.mark.filterwarnings("ignore:Unverified HTTPS request")
@pytest.mark.parametrize("tls", [False, True])
@pytest.mark.parametrize("phase", ["headers", "first", "between"])
@pytest.mark.parametrize("cancel_wait", [False, True])
def test_none_cancellation_aborts_quiet_socket_and_drains(
    endpoint, tls, phase, cancel_wait
):
    base_url, observed, entered, release = stalled(endpoint, phase, tls=tls)
    requested = Event()
    api = LlamaCppClient(base_url, timeout=0.2, verify_ssl=not tls, trust_env=False)
    stream = api.complete(
        "fixture",
        stream=True,
        idle_timeout=None,
        cancel=requested.is_set,
        cancel_wait=cancel_wait,
    )
    try:
        if phase == "between":
            assert next(stream) == {"part": 1}
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(next, stream)
            assert entered.wait(timeout=1)
            requested.set()
            with pytest.raises(LLMCancelledError):
                pending.result(timeout=0.75)
        wait_for(lambda: not api._streams)
        started = time.monotonic()
        api.close()
        assert time.monotonic() - started < 0.5
        assert len(observed) == 1
    finally:
        release.set()
        stream.close()
        api.close()


@pytest.mark.parametrize("phase", ["headers", "first", "between"])
def test_client_close_drains_none_stream_without_server_release(endpoint, phase):
    base_url, observed, entered, release = stalled(endpoint, phase)
    api = LlamaCppClient(base_url, timeout=0.2, trust_env=False)
    stream = api.complete("fixture", stream=True, idle_timeout=None)
    try:
        if phase == "between":
            assert next(stream) == {"part": 1}
        assert entered.wait(timeout=1)
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(api.close).result(timeout=0.75)
        assert not api._streams
        assert list(stream) == []
        assert len(observed) == 1
    finally:
        release.set()
        api.close()


def test_concurrent_none_cancel_does_not_abort_other_lease(endpoint):
    entered = {name: Event() for name in ("cancel", "healthy")}
    release = {name: Event() for name in entered}
    for event in release.values():
        endpoint.release_on_cleanup(event)

    def respond(handler, body):
        name = body["prompt"]
        headers(handler)
        entered[name].set()
        release[name].wait(timeout=3)
        handler.wfile.write(SECOND)
        handler.wfile.flush()

    base_url, observed = endpoint(respond)
    requested = Event()
    with LlamaCppClient(base_url, timeout=0.2, trust_env=False) as api:
        cancel = api.complete(
            "cancel",
            stream=True,
            idle_timeout=None,
            cancel=requested.is_set,
            cancel_wait=True,
        )
        healthy = api.complete("healthy", stream=True, idle_timeout=None)
        with ThreadPoolExecutor(max_workers=2) as executor:
            cancelled = executor.submit(list, cancel)
            successful = executor.submit(list, healthy)
            assert all(event.wait(timeout=1) for event in entered.values())
            requested.set()
            with pytest.raises(LLMCancelledError):
                cancelled.result(timeout=0.75)
            assert not successful.done()
            release["healthy"].set()
            assert successful.result(timeout=1) == [{"part": 2}]
        assert len(api._session_pool) == 2
    assert len(observed) == 2


def test_keepalive_reuse_and_stale_close_cannot_abort_new_lease(endpoint):
    entered, release = Event(), Event()
    endpoint.release_on_cleanup(release)

    def respond(handler, body):
        if body["prompt"] == "new":
            entered.set()
            release.wait(timeout=3)
        headers(handler, body_length=len(SECOND), keepalive=True)
        handler.wfile.write(SECOND)
        handler.wfile.flush()

    base_url, observed = endpoint(respond, keepalive=True)
    with LlamaCppClient(
        base_url, timeout=0.2, trust_env=False, connection_policy="reuse"
    ) as api:
        old = api.complete("old", stream=True, idle_timeout=None)
        assert next(old) == {"part": 2}
        wait_for(lambda: not api._streams)
        new = api.complete("new", stream=True, idle_timeout=None)
        try:
            assert entered.wait(timeout=1)
            old.close()
            release.set()
            assert list(new) == [{"part": 2}]
        finally:
            old.close()
            new.close()
    assert len(observed) == 2
    assert observed[0][2] == observed[1][2]


def test_disabled_idle_error_status_has_finite_json_body_budget(endpoint):
    release = Event()
    endpoint.release_on_cleanup(release)

    def respond(handler, _body):
        headers(handler, status=503, body_length=100)
        release.wait(timeout=3)

    base_url, observed = endpoint(respond)
    with LlamaCppClient(base_url, timeout=0.2, trust_env=False) as api:
        stream = api.complete("fixture", stream=True, idle_timeout=None)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(list, stream)
            with pytest.raises(ClientHTTPError) as caught:
                pending.result(timeout=0.75)
        assert caught.value.status_code == 503
        assert caught.value.error_code is None and caught.value.error_type is None
        assert not api._streams
    assert len(observed) == 1


@pytest.mark.parametrize("method", ["complete", "chat_completion", "request"])
def test_none_keeps_finite_connect_and_control_out_of_payload(monkeypatch, method):
    seen = []

    def request(_session, **kwargs):
        seen.append(kwargs)
        response = requests.Response()
        response.status_code = 200
        response._content = SECOND
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "request", request)
    with LlamaCppClient("https://chosen.example", timeout=0.07, trust_env=False) as api:
        if method == "complete":
            result = api.complete("fixture", stream=True, idle_timeout=None)
        elif method == "chat_completion":
            result = api.chat_completion([], stream=True, idle_timeout=None)
        else:
            result = api.request(
                "POST",
                "/completion",
                json={"prompt": "fixture"},
                stream=True,
                idle_timeout=None,
            )
        assert list(result) == [{"part": 2}]
    assert len(seen) == 1
    call = seen[0]
    assert call["timeout"] == (0.07, None)
    assert call["verify"] is True and call["allow_redirects"] is False
    assert "idle_timeout" not in call["json"]


def test_omitted_idle_preserves_default_and_positive_override_is_finite(endpoint):
    base_url, observed, entered, release = stalled(endpoint, "headers")
    with LlamaCppClient(base_url, timeout=0.05, trust_env=False) as api:
        with pytest.raises(LLMTimeoutError):
            list(api.complete("fixture", stream=True))
        assert entered.is_set()
    release.set()
    base_url, observed = endpoint(
        lambda handler, _body: (headers(handler), handler.wfile.write(SECOND))
    )
    with LlamaCppClient(base_url, timeout=0.05, trust_env=False) as api:
        assert list(api.complete("fixture", stream=True, idle_timeout=0.3)) == [
            {"part": 2}
        ]
    assert len(observed) == 1


@pytest.mark.parametrize(
    "idle", [False, True, 0, -1, float("nan"), float("inf"), "30", 10**1000]
)
def test_invalid_idle_fails_before_dispatch(idle):
    with LlamaCppClient("https://chosen.example", trust_env=False) as api:
        with pytest.raises(ValueError, match="idle_timeout"):
            api.complete("fixture", stream=True, idle_timeout=idle)
        assert not api._streams


@pytest.mark.parametrize("idle", [None, 1])
def test_json_request_rejects_idle_control(idle):
    with LlamaCppClient("https://chosen.example", trust_env=False) as api:
        with pytest.raises(ValueError, match="only supported"):
            api.request("GET", "/v1/models", idle_timeout=idle)
        assert not api._streams


@pytest.mark.parametrize(
    "method,args",
    [
        ("embeddings", (["fixture"],)),
        ("tokenize", ("fixture",)),
        ("detokenize", ([1],)),
    ],
)
@pytest.mark.parametrize("idle", [None, 1])
def test_json_helpers_reject_idle_control_instead_of_sending_it(method, args, idle):
    with LlamaCppClient("https://chosen.example", trust_env=False) as api:
        with pytest.raises(ValueError, match="only supported"):
            getattr(api, method)(*args, idle_timeout=idle)
        assert not api._streams


def test_none_rejects_borrowed_session_without_modifying_or_dispatching(monkeypatch):
    session = requests.Session()
    seen = []
    monkeypatch.setattr(session, "request", lambda **kwargs: seen.append(kwargs))
    before = dict(session.adapters)
    try:
        with LlamaCppClient("https://chosen.example", session=session) as api:
            with pytest.raises(ValueError, match="owned session"):
                api.complete("fixture", stream=True, idle_timeout=None)
        assert seen == []
        assert session.adapters == before
    finally:
        session.close()


def test_json_keeps_finite_timeout_after_disabled_stream(endpoint):
    release = Event()
    endpoint.release_on_cleanup(release)

    def respond(handler, body):
        if body is not None:
            headers(handler)
            handler.wfile.write(SECOND)
            handler.wfile.flush()
        else:
            release.wait(timeout=3)

    base_url, observed = endpoint(respond)
    with LlamaCppClient(base_url, timeout=0.05, trust_env=False) as api:
        assert list(api.complete("fixture", stream=True, idle_timeout=None)) == [
            {"part": 2}
        ]
        with pytest.raises(LLMTimeoutError):
            api.models()
    assert len(observed) == 2


def test_adapter_pool_maps_are_scoped_and_proxy_connection_types_preserved():
    from urllib3.poolmanager import pool_classes_by_scheme

    before = dict(pool_classes_by_scheme)
    adapter = InterruptibleAdapter()
    try:
        assert pool_classes_by_scheme == before
        assert adapter.poolmanager.pool_classes_by_scheme is not pool_classes_by_scheme
        proxy = adapter.proxy_manager_for("http://fixture-proxy.example:8080")
        assert proxy.pool_classes_by_scheme is not pool_classes_by_scheme
        for scheme, pool in proxy.pool_classes_by_scheme.items():
            assert issubclass(pool.ConnectionCls, before[scheme].ConnectionCls)
        assert pool_classes_by_scheme == before
    finally:
        adapter.close()


def test_http_proxy_idle_cancel_preserves_proxy_and_auth_policy(endpoint, monkeypatch):
    entered, release, requested = Event(), Event(), Event()
    proxy_auth = []
    endpoint.release_on_cleanup(release)

    def respond(handler, _body):
        proxy_auth.append(handler.headers.get("Proxy-Authorization"))
        entered.set()
        release.wait(timeout=3)

    proxy, observed = endpoint(respond)
    monkeypatch.setenv(
        "HTTP_PROXY", proxy.replace("://", "://fixture-user:fixture-password@")
    )
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setattr(requests.sessions, "get_netrc_auth", lambda _url: None)
    with LlamaCppClient("http://upstream.invalid", timeout=0.2) as api:
        stream = api.complete(
            "fixture",
            stream=True,
            idle_timeout=None,
            cancel=requested.is_set,
            cancel_wait=True,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(list, stream)
            assert entered.wait(timeout=1)
            requested.set()
            with pytest.raises(LLMCancelledError):
                pending.result(timeout=0.75)
        assert not api._streams
    assert len(observed) == 1
    assert observed[0][0] == "http://upstream.invalid/completion"
    assert proxy_auth[0].startswith("Basic ")


def test_reused_keepalive_socket_is_interruptible_before_new_headers(endpoint):
    entered, release, requested = Event(), Event(), Event()
    endpoint.release_on_cleanup(release)

    def respond(handler, body):
        if body["prompt"] == "new":
            entered.set()
            release.wait(timeout=3)
        headers(handler, body_length=len(SECOND), keepalive=True)
        handler.wfile.write(SECOND)
        handler.wfile.flush()

    base_url, observed = endpoint(respond, keepalive=True)
    with LlamaCppClient(
        base_url, timeout=0.2, trust_env=False, connection_policy="reuse"
    ) as api:
        assert list(api.complete("old", stream=True, idle_timeout=None)) == [
            {"part": 2}
        ]
        stream = api.complete(
            "new",
            stream=True,
            idle_timeout=None,
            cancel=requested.is_set,
            cancel_wait=True,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(list, stream)
            assert entered.wait(timeout=1)
            requested.set()
            with pytest.raises(LLMCancelledError):
                pending.result(timeout=0.75)
        assert not api._streams
    assert len(observed) == 2
    assert observed[0][2] == observed[1][2]


def test_cancel_during_connect_aborts_late_socket_before_header_wait():
    class Socket:
        def __init__(self):
            self.aborted = False

        def shutdown(self, _how):
            self.aborted = True

    class Connection:
        sock = None

    requested = Event()
    connection = Connection()
    interrupt = RequestInterrupt(requested)
    interrupt.capture(connection)
    requested.set()
    interrupt.abort()
    connection.sock = Socket()
    with pytest.raises(OSError, match="request_aborted"):
        interrupt.capture(connection)
    assert connection.sock.aborted


def test_tls_in_tls_interrupt_reaches_known_underlying_socket():
    from urllib3.util.ssltransport import SSLTransport

    class Socket:
        def __init__(self):
            self.interrupts = []

        def shutdown(self, how):
            self.interrupts.append(how)

    sock = Socket()
    transport = object.__new__(SSLTransport)
    transport.socket = sock
    RequestInterrupt._shutdown(transport)
    assert len(sock.interrupts) == 1


def test_tls_default_rejects_untrusted_fixture_certificate(endpoint):
    base_url, observed, _entered, release = stalled(endpoint, "headers", tls=True)
    with LlamaCppClient(base_url, timeout=0.2, trust_env=False) as api:
        with pytest.raises(LLMBackendError):
            list(api.complete("fixture", stream=True, idle_timeout=None))
    release.set()
    assert observed == []


def test_truncated_valid_json_is_a_typed_transport_failure(endpoint):
    def respond(handler, _body):
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", "20")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(b"{}")
        handler.wfile.flush()
        handler.close_connection = True

    base_url, observed = endpoint(respond)
    with LlamaCppClient(base_url, timeout=0.2, trust_env=False) as api:
        with pytest.raises(LLMBackendError, match="transport_failure"):
            api.models()
    assert len(observed) == 1
