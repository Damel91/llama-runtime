from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Event, Lock, Thread, enumerate as threads
import time

import pytest
import requests

from runtime_llama.client import LlamaCppClient
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError


def wait_for(predicate, timeout=1):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert predicate()


def readers():
    return [
        thread
        for thread in threads()
        if thread.name
        in {
            "runtime_llama_json_reader",
            "runtime_llama_sse_reader",
            "runtime_llama_http_close",
        }
        and thread.is_alive()
    ]


class Response:
    status_code = 200

    def __init__(self, payload=None, *, entered=None, release=None):
        self.payload = {"tokens": [7]} if payload is None else payload
        self.entered = entered or Event()
        self.release = release
        self.closed = Event()
        self.close_count = 0

    def json(self):
        self.entered.set()
        if self.release is not None:
            assert self.release.wait(timeout=3)
        return self.payload

    def iter_content(self, chunk_size=1):
        self.entered.set()
        self.closed.wait(timeout=3)
        return
        yield

    def close(self):
        self.close_count += 1
        self.closed.set()


class Session:
    def __init__(self, response=None):
        self.response = response or Response()
        self.calls = []
        self.mounts = {}
        self.close_count = 0

    def request(self, **kwargs):
        self.calls.append(kwargs)
        return self.response

    def mount(self, prefix, adapter):
        self.mounts[prefix] = adapter

    def close(self):
        self.close_count += 1


def test_owned_sessions_are_exclusive_until_body_cleanup_and_are_reused(monkeypatch):
    sessions, body_entered, body_release = [], [], Event()

    def make_session():
        entered = Event()
        body_entered.append(entered)
        session = Session(Response(entered=entered, release=body_release))
        sessions.append(session)
        return session

    monkeypatch.setattr(requests, "Session", make_session)
    api = LlamaCppClient("http://chosen.example", timeout=2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = [executor.submit(api.tokenize, text) for text in ("one", "two")]
        try:
            wait_for(lambda: len(body_entered) == 2)
            assert all(entered.wait(timeout=1) for entered in body_entered)
            assert len(sessions) == 2
            assert [len(session.calls) for session in sessions] == [1, 1]
        finally:
            body_release.set()
        assert [future.result(timeout=1) for future in pending] == [
            {"tokens": [7]},
            {"tokens": [7]},
        ]
    api.health()
    assert len(sessions) == 2
    assert sorted(len(session.calls) for session in sessions) == [1, 2]
    assert all(call["stream"] is True for s in sessions for call in s.calls)
    assert all(
        adapter.max_retries.total == 0
        for session in sessions
        for adapter in session.mounts.values()
    )
    api.close()
    assert [session.close_count for session in sessions] == [1, 1]
    assert not readers()


def test_borrowed_session_serializes_dispatch_but_not_independent_body_reads():
    first_dispatch, release_dispatch, release_bodies = Event(), Event(), Event()
    body_events = [Event(), Event()]

    class GuardedSession(Session):
        def __init__(self):
            super().__init__()
            self.active = self.maximum = 0
            self.guard = Lock()

        def request(self, **kwargs):
            with self.guard:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                index = len(self.calls)
                self.calls.append(kwargs)
            if index == 0:
                first_dispatch.set()
                assert release_dispatch.wait(timeout=3)
            with self.guard:
                self.active -= 1
            return Response(entered=body_events[index], release=release_bodies)

    session = GuardedSession()
    api = LlamaCppClient("http://chosen.example", session=session, timeout=2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = [executor.submit(api.tokenize, text) for text in ("one", "two")]
        try:
            assert first_dispatch.wait(timeout=1)
            time.sleep(0.03)
            assert len(session.calls) == 1
            release_dispatch.set()
            assert all(event.wait(timeout=1) for event in body_events)
            assert session.maximum == 1
        finally:
            release_dispatch.set()
            release_bodies.set()
        assert all(future.result(timeout=1) == {"tokens": [7]} for future in pending)
    api.close()
    assert session.close_count == 0
    assert not readers()


@pytest.mark.parametrize("cancel_wait", [False, True])
def test_json_cancel_before_headers_tracks_and_closes_late_response(cancel_wait):
    dispatch, release, requested = Event(), Event(), Event()
    response = Response()

    class DelayedSession(Session):
        def request(self, **kwargs):
            self.calls.append(kwargs)
            dispatch.set()
            assert release.wait(timeout=3)
            return self.response

    session = DelayedSession(response)
    api = LlamaCppClient("http://chosen.example", session=session, timeout=2)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            api.embeddings,
            ["one"],
            cancel=requested.is_set,
            cancel_wait=cancel_wait,
        )
        try:
            assert dispatch.wait(timeout=1)
            requested.set()
            if cancel_wait:
                time.sleep(0.15)
                assert not pending.done()
            else:
                with pytest.raises(LLMCancelledError):
                    pending.result(timeout=0.5)
                assert len(api._streams) == 1
                assert not response.closed.is_set()
        finally:
            release.set()
        with pytest.raises(LLMCancelledError):
            pending.result(timeout=1)
    wait_for(response.closed.is_set)
    assert response.close_count == 1
    assert len(session.calls) == 1
    assert "cancel_wait" not in session.calls[0]["json"]
    api.close()
    assert not readers()


def test_json_body_cancellation_closes_response_and_can_recover_same_client():
    requested = Event()

    class BlockingResponse(Response):
        def json(self):
            self.entered.set()
            assert self.closed.wait(timeout=3)
            return self.payload

    response = BlockingResponse()
    session = Session(response)
    api = LlamaCppClient("http://chosen.example", session=session, timeout=2)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(api.tokenize, "one", cancel=requested.is_set)
        assert response.entered.wait(timeout=1)
        requested.set()
        with pytest.raises(LLMCancelledError):
            pending.result(timeout=0.5)
    assert response.closed.is_set()
    session.response = Response({"tokens": [8]})
    assert api.tokenize("two") == {"tokens": [8]}
    assert len(session.calls) == 2
    api.close()
    assert session.close_count == 0
    assert not readers()


def test_parallel_json_cancellation_does_not_close_other_response(monkeypatch):
    sessions, requested, release_other = [], Event(), Event()

    class IndependentSession(Session):
        def request(self, **kwargs):
            self.calls.append(kwargs)
            cancelled = kwargs["json"]["content"] == "cancelled"
            self.response = Response(
                {"tokens": [1 if cancelled else 2]},
                release=None if cancelled else release_other,
            )
            if cancelled:
                self.response.release = self.response.closed
            return self.response

    def make_session():
        session = IndependentSession()
        sessions.append(session)
        return session

    monkeypatch.setattr(requests, "Session", make_session)
    api = LlamaCppClient("http://chosen.example", timeout=2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        cancelled = executor.submit(api.tokenize, "cancelled", cancel=requested.is_set)
        other = executor.submit(api.tokenize, "other")
        try:
            wait_for(lambda: len(sessions) == 2)
            assert all(session.response.entered.wait(timeout=1) for session in sessions)
            requested.set()
            with pytest.raises(LLMCancelledError):
                cancelled.result(timeout=0.5)
            live = next(
                session.response
                for session in sessions
                if session.calls[0]["json"]["content"] == "other"
            )
            assert not live.closed.is_set()
            assert not other.done()
        finally:
            requested.set()
            release_other.set()
        assert other.result(timeout=1) == {"tokens": [2]}
    api.close()
    assert not readers()


def test_client_close_drains_active_json_and_sse_and_owned_sessions(monkeypatch):
    sessions = []

    class BlockingResponse(Response):
        def json(self):
            self.entered.set()
            self.closed.wait(timeout=3)
            return self.payload

    def make_session():
        session = Session(BlockingResponse())
        sessions.append(session)
        return session

    monkeypatch.setattr(requests, "Session", make_session)
    api = LlamaCppClient("http://chosen.example", timeout=2)
    stream = api.complete("one", stream=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(api.embeddings, ["two"])
        wait_for(lambda: len(sessions) == 2)
        assert all(session.response.entered.wait(timeout=1) for session in sessions)
        api.close()
        with pytest.raises(LLMBackendError, match="client_closed"):
            pending.result(timeout=1)
    assert list(stream) == []
    assert all(session.response.closed.is_set() for session in sessions)
    assert all(session.close_count == 1 for session in sessions)
    assert not readers()
    with pytest.raises(LLMBackendError, match="client_closed"):
        api.health()
    api.close()
    assert all(session.close_count == 1 for session in sessions)


def test_client_close_waits_for_preheader_request_cleanup():
    dispatch, release, closed = Event(), Event(), Event()
    response = Response()

    class DelayedSession(Session):
        def request(self, **kwargs):
            dispatch.set()
            assert release.wait(timeout=3)
            return super().request(**kwargs)

    session = DelayedSession(response)
    api = LlamaCppClient("http://chosen.example", session=session, timeout=2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(api.tokenize, "one")
        assert dispatch.wait(timeout=1)
        closer = executor.submit(lambda: (api.close(), closed.set()))
        try:
            time.sleep(0.1)
            assert not closed.is_set()
            with pytest.raises(LLMBackendError, match="client_closed"):
                pending.result(timeout=0.5)
        finally:
            release.set()
        closer.result(timeout=1)
    assert response.closed.is_set()
    assert not readers()
    assert session.close_count == 0


def test_cancel_wait_retains_cleanup_on_json_timeout():
    dispatch, release = Event(), Event()

    class DelayedSession(Session):
        def request(self, **kwargs):
            dispatch.set()
            assert release.wait(timeout=3)
            return super().request(**kwargs)

    session = DelayedSession()
    api = LlamaCppClient("http://chosen.example", session=session, timeout=0.03)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(api.tokenize, "one", cancel_wait=True)
        assert dispatch.wait(timeout=1)
        try:
            time.sleep(0.1)
            assert not pending.done()
        finally:
            release.set()
        with pytest.raises(LLMTimeoutError, match="json_response_timeout"):
            pending.result(timeout=1)
    api.close()
    assert session.response.closed.is_set()
    assert not readers()


@pytest.mark.parametrize(
    "method,args",
    [
        ("chat_completion", ([],)),
        ("complete", ("one",)),
        ("embeddings", (["one"],)),
        ("tokenize", ("one",)),
        ("detokenize", ([1],)),
        ("health", ()),
        ("props", ("alias",)),
        ("models", ()),
    ],
)
def test_cancel_wait_is_transport_only_for_every_helper(method, args):
    session = Session()
    with LlamaCppClient("http://chosen.example", session=session) as api:
        getattr(api, method)(*args, cancel_wait=True)
    assert len(session.calls) == 1
    body = session.calls[0]["json"]
    assert body is None or "cancel_wait" not in body


def test_cancel_wait_requires_explicit_boolean():
    session = Session()
    with LlamaCppClient("http://chosen.example", session=session) as api:
        with pytest.raises(ValueError, match="cancel_wait"):
            api.health(cancel_wait=1)
    assert session.calls == []


def test_loopback_json_idle_cancellation_and_recovery_without_retry():
    entered, release, requested = Event(), Event(), Event()
    received = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received.append(self.path)
            body = json.dumps({"tokens": [7]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if len(received) == 1:
                entered.set()
                release.wait(timeout=3)
            try:
                self.wfile.write(body)
                self.wfile.flush()
            except OSError:
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    serving = Thread(target=server.serve_forever, daemon=True)
    serving.start()
    api = LlamaCppClient(f"http://127.0.0.1:{server.server_port}", timeout=0.3)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(api.tokenize, "one", cancel=requested.is_set)
            assert entered.wait(timeout=1)
            requested.set()
            with pytest.raises(LLMCancelledError):
                pending.result(timeout=0.5)
        assert received == ["/tokenize"]
        release.set()
        wait_for(lambda: not readers())
        assert api.tokenize("two") == {"tokens": [7]}
        assert received == ["/tokenize", "/tokenize"]
    finally:
        release.set()
        api.close()
        server.shutdown()
        server.server_close()
        serving.join(timeout=1)
    assert not readers()


@pytest.mark.parametrize("status", [200, 429])
@pytest.mark.parametrize("termination", ["timeout", "cancel"])
def test_loopback_json_slow_drip_does_not_extend_cleanup_forever(status, termination):
    entered, release, requested = Event(), Event(), Event()
    received = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received.append(self.path)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "1000000")
            self.end_headers()
            try:
                self.wfile.write(b'{"tokens":[')
                self.wfile.flush()
                entered.set()
                while not release.wait(timeout=0.01):
                    self.wfile.write(b" ")
                    self.wfile.flush()
            except OSError:
                pass
            self.close_connection = True

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    serving = Thread(target=server.serve_forever, daemon=True)
    serving.start()
    api = LlamaCppClient(
        f"http://127.0.0.1:{server.server_port}",
        timeout=0.1 if termination == "timeout" else 5,
    )
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        pending = executor.submit(
            api.tokenize, "one", cancel=requested.is_set, cancel_wait=True
        )
        assert entered.wait(timeout=1)
        if termination == "cancel":
            requested.set()
        error_type = LLMTimeoutError if termination == "timeout" else LLMCancelledError
        with pytest.raises(error_type):
            pending.result(timeout=0.75)
        assert not readers()
        assert received == ["/tokenize"]
        api.close()
        assert not readers()
    finally:
        release.set()
        api.close()
        executor.shutdown(wait=True)
        server.shutdown()
        server.server_close()
        serving.join(timeout=1)
