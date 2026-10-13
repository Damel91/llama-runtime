from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
from threading import Event, Thread, enumerate as threads
import time
import traceback

import pytest
import requests
from urllib3.util import connection as urllib3_connection

from runtime_llama.adapters.llama import LlamaCppAdapter
from runtime_llama.client import ClientHTTPError, LlamaCppClient
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError


class Response:
    def __init__(self, payload=None, *, status=200, chunks=()):
        self.payload = {} if payload is None else payload
        self.status_code = status
        self.chunks = chunks
        self.closed = Event()
        self.reading = Event()

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def iter_content(self, chunk_size=1):
        self.reading.set()
        yield from self.chunks

    def close(self):
        self.closed.set()


class BlockingResponse(Response):
    def iter_content(self, chunk_size=1):
        self.reading.set()
        yield from self.chunks
        self.closed.wait(timeout=3)


class Session:
    def __init__(self, response=None, error=None):
        self.response = response or Response()
        self.error = error
        self.calls = []
        self.closed = False
        self.mounts = {}

    def request(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response

    def mount(self, prefix, adapter):
        self.mounts[prefix] = adapter

    def close(self):
        self.closed = True


def client(response=None, *, error=None, **kwargs):
    session = Session(response, error)
    return LlamaCppClient(
        "https://chosen.example/proxy", session=session, **kwargs
    ), session


def readers():
    return [
        thread
        for thread in threads()
        if thread.name == "runtime_llama_sse_reader" and thread.is_alive()
    ]


def wait_for(predicate, timeout=1):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert predicate()


@pytest.mark.parametrize(
    ("method", "args", "path", "body"),
    [
        (
            "chat_completion",
            ([{"role": "user", "content": "hello"}],),
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hello"}], "stream": False},
        ),
        ("complete", ("hello",), "/completion", {"prompt": "hello", "stream": False}),
        ("embeddings", (["one", "two"],), "/v1/embeddings", {"input": ["one", "two"]}),
        ("tokenize", ("one",), "/tokenize", {"content": "one"}),
        ("detokenize", ([12, 34],), "/detokenize", {"tokens": [12, 34]}),
        ("health", (), "/health", None),
        ("models", (), "/v1/models", None),
    ],
)
def test_helpers_preserve_raw_fields_without_implicit_sampling(
    method, args, path, body
):
    raw = {"unknown_upstream_field": {"value": 42}}
    response = Response(raw)
    api, session = client(response)
    assert getattr(api, method)(*args) is raw
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == "https://chosen.example/proxy" + path
    assert call["json"] == body
    assert "usage" not in raw
    assert response.closed.is_set()


@pytest.mark.parametrize(
    "method,args",
    [
        ("chat_completion", ([],)),
        ("complete", ("prompt",)),
        ("embeddings", (["input"],)),
        ("tokenize", ("input",)),
    ],
)
def test_per_call_timeout_is_http_only_and_request_fields_remain_explicit(method, args):
    api, session = client(api_key="private-key", verify_ssl=False, timeout=12)
    getattr(api, method)(*args, timeout=0.75, seed=7, model="selected")
    call = session.calls[0]
    assert call["timeout"] == 0.75
    assert call["verify"] is False
    assert call["headers"]["Authorization"] == "Bearer private-key"
    assert call["allow_redirects"] is False
    assert call["json"]["seed"] == 7
    assert call["json"]["model"] == "selected"
    assert "timeout" not in call["json"]
    assert "autoload" not in call["url"]


def test_props_model_is_query_encoded_and_native_arrays_remain_raw():
    raw = [{"embedding": [1, 2], "future_field": True}]
    api, session = client(Response(raw))
    assert api.props("org/model with spaces") is raw
    assert session.calls[0]["url"].endswith("/props?model=org%2Fmodel+with+spaces")
    assert (
        api.request("POST", "/embedding?autoload=false", json={"content": "test"})
        is raw
    )
    assert session.calls[1]["url"].endswith("/embedding?autoload=false")


@pytest.mark.parametrize(
    "path",
    [
        "https://elsewhere/health",
        "//elsewhere/health",
        "/health#fragment",
        "/\\elsewhere",
    ],
)
def test_raw_request_cannot_send_credentials_to_another_host(path):
    api, session = client(api_key="secret")
    with pytest.raises(ValueError):
        api.request("GET", path)
    assert session.calls == []


def test_http_error_keeps_only_structured_short_codes_and_closes_response():
    secret = "DO-NOT-EXPOSE-KEY-OR-PROMPT"
    response = Response(
        {
            "error": {
                "code": 400,
                "type": "exceed_context_size_error",
                "message": secret,
            }
        },
        status=400,
    )
    api, session = client(response, api_key=secret)
    with pytest.raises(ClientHTTPError) as caught:
        api.embeddings([secret])
    assert caught.value.status_code == 400
    assert caught.value.error_code == "400"
    assert caught.value.error_type == "exceed_context_size_error"
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert response.closed.is_set()
    assert len(session.calls) == 1


def test_untrusted_structured_code_is_not_echoed():
    response = Response(
        {"error": {"code": "private prompt with spaces", "type": "x" * 100}}, status=400
    )
    api, _ = client(response)
    with pytest.raises(ClientHTTPError) as caught:
        api.health()
    assert caught.value.error_code is None
    assert caught.value.error_type is None


@pytest.mark.parametrize(
    "exception,error_type",
    [
        (requests.exceptions.ReadTimeout, LLMTimeoutError),
        (requests.exceptions.ConnectionError, LLMBackendError),
    ],
)
def test_transport_errors_have_sanitized_causes_and_no_retry(exception, error_type):
    secret = "DO-NOT-EXPOSE-KEY-OR-PROMPT"
    api, session = client(error=exception("https://server/?api_key=" + secret))
    with pytest.raises(error_type) as caught:
        api.complete(secret)
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert type(caught.value.__cause__) is RuntimeError
    assert len(session.calls) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_wrapped_connection_refusal_is_backend_failure_without_retry(stream):
    from urllib3.exceptions import NewConnectionError

    secret = "DO-NOT-EXPOSE-CONNECTION-PROMPT-OR-KEY"
    api, session = client(
        error=requests.ConnectionError(NewConnectionError(None, secret))
    )
    with pytest.raises(LLMBackendError, match="transport_failure") as caught:
        response = api.complete(secret, stream=stream)
        if stream:
            list(response)
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert len(session.calls) == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("timeout_kind", ["connect", "read"])
def test_wrapped_actual_timeouts_keep_timeout_classification(stream, timeout_kind):
    from urllib3.exceptions import ConnectTimeoutError, ReadTimeoutError

    secret = "DO-NOT-EXPOSE-TIMEOUT-PROMPT-OR-KEY"
    timeout = (
        ConnectTimeoutError(None, secret)
        if timeout_kind == "connect"
        else ReadTimeoutError(None, None, secret)
    )
    api, session = client(error=requests.ConnectionError(timeout))
    with pytest.raises(LLMTimeoutError) as caught:
        response = api.complete(secret, stream=stream)
        if stream:
            list(response)
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert len(session.calls) == 1


@pytest.mark.parametrize("payload", [42, "bad shape", ValueError("private prompt")])
def test_nonstream_invalid_json_or_shape_closes_response(payload):
    response = Response(payload)
    api, _ = client(response)
    with pytest.raises(LLMBackendError) as caught:
        api.health()
    assert "private prompt" not in "".join(traceback.format_exception(caught.value))
    assert response.closed.is_set()


def test_cancelled_before_request_performs_no_http():
    api, session = client()
    with pytest.raises(LLMCancelledError):
        api.complete("secret", stream=True, cancel=lambda: True)
    assert not session.calls


def test_fragmented_sse_preserves_extra_fields_multiline_unicode_and_final_usage():
    first = {
        "choices": [{"delta": {"content": "caffè ☕"}}],
        "reasoning_content": "raw",
    }
    last = {"choices": [], "usage": {"completion_tokens": 3, "future_usage": 9}}
    frame = (
        "\ufeff: keepalive\r\nevent: message\r\n"
        "data: " + json.dumps(first, ensure_ascii=False) + "\r\n\r\n"
        "data: {\n" + 'data: "choices": [],\n'
        'data: "usage": {"completion_tokens": 3, "future_usage": 9}}\n\n'
        "data: [DONE]\n\n" + "data: {bad trailing frame}\n\n"
    ).encode()
    response = Response(chunks=[bytes([value]) for value in frame])
    api, _ = client(response)
    assert list(api.chat_completion([], stream=True)) == [first, last]
    assert response.closed.is_set()


@pytest.mark.parametrize(
    "frame,message",
    [
        (b"data: {private prompt}\n\n", "malformed_sse_json"),
        (b"data: []\n\n", "sse_payload_not_object"),
        (b'data: {"text":"\xff"}\n\n', "sse_invalid_utf8"),
    ],
)
def test_invalid_sse_is_strict_sanitized_and_closes(frame, message):
    response = Response(chunks=[frame])
    api, _ = client(response)
    with pytest.raises(LLMBackendError, match=message) as caught:
        list(api.complete("secret", stream=True))
    assert "private prompt" not in "".join(traceback.format_exception(caught.value))
    assert response.closed.is_set()


def test_done_is_transport_only_and_does_not_invent_finish_reason_or_usage():
    response = Response(chunks=[b'data: {"content":"part"}\n\ndata: [DONE]\n\n'])
    api, _ = client(response)
    assert list(api.complete("hello", stream=True)) == [{"content": "part"}]


def test_early_iterator_close_unblocks_reader():
    response = BlockingResponse(chunks=[b'data: {"content":"part"}\n\n'])
    api, _ = client(response)
    stream = api.complete("hello", stream=True)
    assert next(stream) == {"content": "part"}
    stream.close()
    assert response.closed.is_set()
    wait_for(lambda: not readers())


def test_idle_cancellation_closes_response_without_waiting_for_http_timeout():
    response = BlockingResponse()
    requested = Event()
    api, _ = client(response, timeout=2)
    stream = api.complete("hello", stream=True, cancel=requested.is_set)
    assert response.reading.wait(timeout=1)
    trigger = Thread(target=lambda: (time.sleep(0.02), requested.set()))
    trigger.start()
    started = time.monotonic()
    with pytest.raises(LLMCancelledError):
        next(stream)
    trigger.join(timeout=1)
    assert time.monotonic() - started < 0.5
    assert response.closed.is_set()
    wait_for(lambda: not readers())


def test_client_close_cleans_unconsumed_stream_and_does_not_close_injected_session():
    response = BlockingResponse()
    api, session = client(response)
    stream = api.complete("hello", stream=True)
    assert response.reading.wait(timeout=1)
    api.close()
    assert response.closed.is_set()
    assert list(stream) == []
    assert session.closed is False
    with pytest.raises(LLMBackendError, match="client_closed"):
        api.health()
    wait_for(lambda: not readers())


def test_owned_session_explicitly_disables_retries_and_is_closed(monkeypatch):
    session = Session()
    monkeypatch.setattr(requests, "Session", lambda: session)
    api = LlamaCppClient("http://localhost:8000")
    assert all(adapter.max_retries.total == 0 for adapter in session.mounts.values())
    api.close()
    assert session.closed


def test_loopback_sse_idle_cancel_with_real_requests():
    first_sent = Event()
    release = Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b'data: {"content":"hello"}\n\n')
            self.wfile.flush()
            first_sent.set()
            release.wait(timeout=2)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    serving = Thread(target=server.serve_forever, daemon=True)
    serving.start()
    requested = Event()
    api = LlamaCppClient(f"http://127.0.0.1:{server.server_port}", timeout=0.3)
    stream = api.complete("local fixture only", stream=True, cancel=requested.is_set)
    try:
        assert next(stream) == {"content": "hello"}
        assert first_sent.wait(timeout=1)
        requested.set()
        started = time.monotonic()
        with pytest.raises(LLMCancelledError):
            next(stream)
        assert time.monotonic() - started < 0.5
        release.set()
        wait_for(lambda: not readers(), timeout=1)
    finally:
        release.set()
        stream.close()
        api.close()
        server.shutdown()
        server.server_close()
        serving.join(timeout=1)


def test_loopback_truncated_sse_body_recovers_same_adapter_without_retry():
    """A local fixture breaks the HTTP body; no remote runtime is touched."""
    first_sent, break_connection = Event(), Event()
    received = []

    def wire(content, *, finish=False):
        payload = {
            "choices": [
                {
                    "index": 0,
                    "delta": {} if finish else {"content": content},
                    "finish_reason": "stop" if finish else None,
                }
            ]
        }
        if finish:
            payload["usage"] = {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            }
        return b"data: " + json.dumps(payload).encode() + b"\n\n"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received.append(self.path)
            broken = len(received) == 1
            body = (
                wire("partial ") + b'data: {"choices":[{"delta":{"content":"broken'
                if broken
                else wire("RECOVERED") + wire("", finish=True) + b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body) + (100 if broken else 0)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            if broken:
                first_sent.set()
                break_connection.wait(timeout=3)
                self.connection.shutdown(socket.SHUT_RDWR)
            self.close_connection = True

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    serving = Thread(target=server.serve_forever, daemon=True)
    serving.start()
    api = LlamaCppClient(f"http://127.0.0.1:{server.server_port}", timeout=2)
    adapter = LlamaCppAdapter(model="fixture-only", client=api)
    stream = adapter.stream_with_metadata("local fixture only")
    try:
        assert next(stream).text == "partial "
        assert first_sent.wait(timeout=1)
        break_connection.set()
        with pytest.raises(LLMBackendError, match="transport_failure"):
            list(stream)
        assert received == ["/v1/chat/completions"]
        assert adapter.get_last_usage() is None
        assert adapter.get_last_terminal_reason() == "backend_error"

        recovered = adapter.generate_with_metadata("local fixture only")
        assert recovered.text == "RECOVERED"
        assert recovered.terminal_reason == "completed"
        assert recovered.usage.prompt_tokens == 7
        assert adapter.get_last_usage() == recovered.usage
        assert adapter.get_last_terminal_reason() == "completed"
        assert received == ["/v1/chat/completions"] * 2
        wait_for(lambda: not readers())
    finally:
        break_connection.set()
        stream.close()
        adapter.close()
        api.close()
        server.shutdown()
        server.server_close()
        serving.join(timeout=1)


def test_loopback_fixture_restart_reuses_client_for_tokenization_without_retry(
    monkeypatch,
):
    """Restart a local fixture on the same port, including its pooled sockets."""
    received, connections, connection_attempts = [], [], []
    create_connection = urllib3_connection.create_connection

    def counted_connection(address, *args, **kwargs):
        connection_attempts.append(address)
        return create_connection(address, *args, **kwargs)

    monkeypatch.setattr(urllib3_connection, "create_connection", counted_connection)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            connections.append(self.connection)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received.append((self.path, self.server.fixture_token))
            body = json.dumps({"tokens": [self.server.fixture_token]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def log_message(self, *_args):
            pass

    def start_server(port, token):
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        server.daemon_threads = True
        server.fixture_token = token
        serving = Thread(target=server.serve_forever, daemon=True)
        serving.start()
        return server, serving

    def stop_server(server, serving):
        server.shutdown()
        server.server_close()
        serving.join(timeout=1)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        connections.clear()

    server, serving = start_server(0, 11)
    port = server.server_port
    api = LlamaCppClient(f"http://127.0.0.1:{port}", timeout=1)
    try:
        assert api.tokenize("local fixture before restart") == {"tokens": [11]}
        stop_server(server, serving)
        server = serving = None

        attempts_before_failure = len(connection_attempts)
        with pytest.raises(LLMBackendError, match="transport_failure"):
            api.tokenize("local fixture while stopped")
        assert len(connection_attempts) == attempts_before_failure + 1
        assert received == [("/tokenize", 11)]

        server, serving = start_server(port, 22)
        assert api.tokenize("local fixture after restart") == {"tokens": [22]}
        assert received == [("/tokenize", 11), ("/tokenize", 22)]
    finally:
        api.close()
        if server is not None:
            stop_server(server, serving)


@pytest.mark.parametrize("separator", ["\n", "\r", "\r\n"])
@pytest.mark.parametrize("chunk_size", [1, 3, 17, 4096])
def test_sse_line_endings_and_chunk_sizes(separator, chunk_size):
    wire = (
        separator.join(
            [
                ": comment",
                'data: {"one": 1}',
                "",
                'data: {"two": "é"}',
                "",
                "data: [DONE]",
                "",
                "",
            ]
        )
    ).encode()
    response = Response(
        chunks=[
            wire[index : index + chunk_size]
            for index in range(0, len(wire), chunk_size)
        ]
    )
    api, _ = client(response)
    assert list(api.complete("hello", stream=True)) == [{"one": 1}, {"two": "é"}]
    assert response.closed.is_set()


def test_cancellation_before_headers_closes_response_when_it_arrives():
    entered, release, requested = Event(), Event(), Event()
    response = Response(chunks=[b'data: {"content":"must not escape"}\n\n'])

    class DelayedSession(Session):
        def request(self, **kwargs):
            entered.set()
            release.wait(timeout=2)
            return super().request(**kwargs)

    session = DelayedSession(response)
    api = LlamaCppClient("http://chosen.example", session=session, timeout=2)
    stream = api.complete("hello", stream=True, cancel=requested.is_set)
    assert entered.wait(timeout=1)
    requested.set()
    started = time.monotonic()
    with pytest.raises(LLMCancelledError):
        next(stream)
    assert time.monotonic() - started < 0.5
    release.set()
    wait_for(response.closed.is_set)
    wait_for(lambda: not readers())


def test_idle_timeout_is_a_typed_error_and_cleans_reader():
    response = BlockingResponse()
    api, _ = client(response, timeout=0.03)
    stream = api.complete("hello", stream=True)
    with pytest.raises(LLMTimeoutError, match="stream_idle_timeout"):
        next(stream)
    assert response.closed.is_set()
    wait_for(lambda: not readers())


def test_cancel_before_queued_payload_does_not_project_it():
    requested = Event()

    class CancellingResponse(Response):
        def iter_content(self, chunk_size=1):
            requested.set()
            yield b'data: {"content":"must not escape"}\n\n'

    response = CancellingResponse()
    api, _ = client(response)
    stream = api.complete("hello", stream=True, cancel=requested.is_set)
    with pytest.raises(LLMCancelledError):
        next(stream)
    assert response.closed.is_set()


def test_nonfinite_sse_json_number_is_rejected():
    response = Response(chunks=[b'data: {"value": NaN}\n\n'])
    api, _ = client(response)
    with pytest.raises(LLMBackendError, match="malformed_sse_json"):
        list(api.complete("hello", stream=True))
    assert response.closed.is_set()


def test_stream_requests_wrapped_read_timeout_keeps_timeout_classification():
    from urllib3.exceptions import ReadTimeoutError

    class TimedOutResponse(Response):
        def iter_content(self, chunk_size=1):
            raise requests.exceptions.ConnectionError(
                ReadTimeoutError(None, None, "private URL/prompt")
            )
            yield

    response = TimedOutResponse()
    api, _ = client(response)
    with pytest.raises(LLMTimeoutError) as caught:
        list(api.complete("hello", stream=True))
    assert "private URL/prompt" not in "".join(traceback.format_exception(caught.value))
    assert response.closed.is_set()


@pytest.mark.parametrize("stream", [False, True])
def test_cancellation_probe_failure_is_typed_sanitized_and_performs_no_http(stream):
    secret = "TEST_PRIVATE_CALLBACK_SENTINEL"

    def failed_probe():
        raise RuntimeError(secret)

    api, session = client()
    with pytest.raises(LLMBackendError, match="cancellation_probe_failed") as caught:
        api.complete("hello", stream=stream, cancel=failed_probe)
    assert session.calls == []
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert str(caught.value.__cause__) == "RuntimeError"


def test_idle_cancellation_probe_failure_closes_reader_without_echoing_exception():
    secret = "TEST_PRIVATE_CALLBACK_SENTINEL"
    response = BlockingResponse()

    def failed_probe():
        if response.reading.is_set():
            raise RuntimeError(secret)
        return False

    api, _ = client(response)
    stream = api.complete("hello", stream=True, cancel=failed_probe)
    assert response.reading.wait(timeout=1)
    with pytest.raises(LLMBackendError, match="cancellation_probe_failed") as caught:
        next(stream)
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert response.closed.is_set()
    wait_for(lambda: not readers())
