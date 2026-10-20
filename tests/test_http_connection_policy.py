from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from threading import Event, Lock, Thread
import time

import pytest
import requests
from urllib3.response import HTTPResponse

from runtime_llama import AdapterProfile, LlamaCppClient, LlamaServerConfig
import runtime_llama.client as http_client
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError
from test_optional_stream_idle_transport import (
    endpoint as endpoint,
    tls_context as tls_context,
)


def send(handler, body, *, content_type="application/json", encoding=None, extra=0):
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body) + extra))
    if encoding:
        handler.send_header("Content-Encoding", encoding)
    if extra or handler.headers.get("Connection") == "close":
        handler.send_header("Connection", "close")
    if extra:
        handler.close_connection = True
    handler.end_headers()
    handler.wfile.write(body)
    handler.wfile.flush()


@pytest.mark.parametrize("policy", [None, "", "keep-alive", "CLOSE", 1, True, [], {}])
def test_invalid_connection_policy_fails_before_session_creation(monkeypatch, policy):
    created = []
    monkeypatch.setattr(requests, "Session", lambda: created.append(True))
    with pytest.raises(ValueError, match="connection_policy"):
        LlamaCppClient("http://chosen.example", connection_policy=policy)
    assert created == []
    with pytest.raises(ValueError, match="http_connection_policy"):
        LlamaServerConfig("chosen", http_connection_policy=policy)
    with pytest.raises(ValueError, match="http_connection_policy"):
        AdapterProfile.from_config(
            "role", {"model": "chosen", "http_connection_policy": policy}
        )


def test_connection_policy_configuration_defaults_and_identity():
    default = AdapterProfile.from_config("role", {"model": "chosen"})
    reuse = AdapterProfile.from_config(
        "role", {"model": "chosen"}, {"http_connection_policy": "reuse"}
    )
    overridden = AdapterProfile.from_config(
        "role",
        {"model": "chosen", "http_connection_policy": "close"},
        {"http_connection_policy": "reuse"},
    )
    assert default.model_config.http_connection_policy == "close"
    assert reuse.model_config.http_connection_policy == "reuse"
    assert overridden.model_config.http_connection_policy == "close"
    assert (
        default.model_config.physical_identity == reuse.model_config.physical_identity
    )


@pytest.mark.parametrize("policy", ["close", "reuse"])
def test_connection_policy_controls_native_socket_reuse_without_retries(
    endpoint, policy
):
    request_headers = []

    def respond(handler, _body):
        request_headers.append(handler.headers.get("Connection"))
        send(handler, b'{"status":"ok"}')

    base_url, observed = endpoint(respond, keepalive=True)
    with LlamaCppClient(base_url, trust_env=False, connection_policy=policy) as api:
        for _ in range(3):
            assert api.health() == {"status": "ok"}
        assert all(a.max_retries.total == 0 for a in api._session.adapters.values())
    assert len(observed) == 3
    assert request_headers == ["close" if policy == "close" else "keep-alive"] * 3
    peers = {row[2] for row in observed}
    assert len(peers) == (3 if policy == "close" else 1)


@pytest.mark.parametrize("policy", ["close", "reuse"])
@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_owned_close_opens_fresh_sockets_when_peer_ignores_close(
    endpoint, policy, borrowed, streaming
):
    request_headers = []
    body = b'data: {"part":1}\n\ndata: [DONE]\n\n' if streaming else b'{"status":"ok"}'

    def respond(handler, _body):
        request_headers.append(handler.headers.get("Connection"))
        handler.send_response(200)
        handler.send_header(
            "Content-Type", "text/event-stream" if streaming else "application/json"
        )
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
        handler.wfile.flush()
        # Deliberately contradict the request. The owned close policy must
        # still guarantee a fresh connection without relying on this peer.
        handler.close_connection = False

    base_url, observed = endpoint(respond, keepalive=True)
    session = requests.Session() if borrowed else None
    if session is not None:
        session.trust_env = False
    try:
        with LlamaCppClient(
            base_url,
            trust_env=False,
            connection_policy=policy,
            session=session,
        ) as api:
            for _ in range(3):
                if streaming:
                    assert list(api.complete("fixture", stream=True)) == [{"part": 1}]
                else:
                    assert api.health() == {"status": "ok"}
            if session is not None:
                # A borrowed owner retains its pools and can keep using them.
                with session.get(base_url + "/health", timeout=2) as response:
                    assert response.content == body
    finally:
        if session is not None:
            session.close()
    assert request_headers[:3] == ["close" if policy == "close" else "keep-alive"] * 3
    assert len(observed) == (4 if borrowed else 3)
    assert len({row[2] for row in observed}) == (
        3 if policy == "close" and not borrowed else 1
    )


@pytest.mark.parametrize("policy", ["close", "reuse"])
def test_connection_policy_preserves_distinct_concurrent_sockets(endpoint, policy):
    entered, release, lock = Event(), Event(), Lock()
    endpoint.release_on_cleanup(release)
    pending_headers = []

    def respond(handler, body):
        with lock:
            pending_headers.append(handler.headers.get("Connection"))
            if len(pending_headers) == 2:
                entered.set()
        assert release.wait(timeout=3)
        send(handler, json.dumps({"model": body["model"]}).encode())

    base_url, observed = endpoint(respond, keepalive=True)
    with LlamaCppClient(
        base_url, timeout=2, trust_env=False, connection_policy=policy
    ) as api:
        with ThreadPoolExecutor(max_workers=2) as callers:
            futures = [
                callers.submit(api.embeddings, ["one", "two"], model=alias)
                for alias in ("copy_a", "copy_b")
            ]
            try:
                assert entered.wait(timeout=1)
                assert len({row[2] for row in observed}) == 2
                assert all(not future.done() for future in futures)
            finally:
                release.set()
            assert [future.result(timeout=2) for future in futures] == [
                {"model": "copy_a"},
                {"model": "copy_b"},
            ]
    assert len(observed) == 2
    assert all(row[1]["input"] == ["one", "two"] for row in observed)
    assert all("connection_policy" not in row[1] for row in observed)


@pytest.mark.parametrize("policy", ["close", "reuse"])
def test_connection_policy_keeps_sse_open_until_all_frames_are_read(endpoint, policy):
    first = b'data: {"part":1}\n\n'
    last = b'data: {"part":2}\n\ndata: [DONE]\n\n'
    release = Event()
    endpoint.release_on_cleanup(release)

    def respond(handler, _body):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Content-Length", str(len(first) + len(last)))
        if handler.headers.get("Connection") == "close":
            handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(first)
        handler.wfile.flush()
        assert release.wait(timeout=3)
        handler.wfile.write(last)
        handler.wfile.flush()

    base_url, observed = endpoint(respond, keepalive=True)
    with LlamaCppClient(
        base_url, timeout=0.1, trust_env=False, connection_policy=policy
    ) as api:
        stream = api.complete(
            "fixture", stream=True, idle_timeout=None, cancel_wait=True
        )
        try:
            assert next(stream) == {"part": 1}
            with ThreadPoolExecutor(max_workers=1) as callers:
                tail = callers.submit(list, stream)
                assert not tail.done()
                release.set()
                assert tail.result(timeout=1) == [{"part": 2}]
        finally:
            release.set()
            stream.close()
    assert len(observed) == 1


@pytest.mark.parametrize("policy", ["close", "reuse"])
@pytest.mark.parametrize("borrowed", [False, True])
def test_preheader_cancellation_drains_local_transport_not_remote_computation(
    endpoint, policy, borrowed
):
    entered, release, remote_finished = Event(), Event(), Event()
    requested, noticed = Event(), Event()
    endpoint.release_on_cleanup(release)

    def respond(handler, body):
        if body["prompt"] == "first":
            entered.set()
            assert release.wait(timeout=3)
            remote_finished.set()
            # The peer cancelled this request. Do not let the HTTP fixture
            # try to parse another request from the interrupted socket.
            handler.close_connection = True
        send(handler, json.dumps({"prompt": body["prompt"]}).encode())

    def cancel():
        if requested.is_set():
            noticed.set()
            return True
        return False

    base_url, observed = endpoint(respond, keepalive=True)
    session = requests.Session() if borrowed else None
    if session is not None:
        session.trust_env = False
    try:
        with LlamaCppClient(
            base_url,
            timeout=2,
            trust_env=False,
            connection_policy=policy,
            session=session,
        ) as api:
            with ThreadPoolExecutor(max_workers=1) as callers:
                pending = callers.submit(
                    api.complete, "first", cancel=cancel, cancel_wait=True
                )
                try:
                    assert entered.wait(timeout=1)
                    call = next(iter(api._streams))
                    requested.set()
                    assert noticed.wait(timeout=1)
                    if borrowed:
                        # The owner has not supplied a socket interrupt. A
                        # preheader read keeps the lease until its I/O ends.
                        assert not pending.done()
                        assert not call._drained.is_set()
                        release.set()
                    with pytest.raises(LLMCancelledError):
                        pending.result(timeout=1)
                    assert call._drained.is_set()
                    assert not api._streams
                    if not borrowed:
                        # Local draining does not claim that a remote worker
                        # has stopped. This fixture deliberately ignores EOF.
                        assert not remote_finished.is_set()
                        assert not release.is_set()
                    assert api.complete("second") == {"prompt": "second"}
                finally:
                    release.set()
        assert len(observed) == 2
        assert observed[0][2] != observed[1][2]
    finally:
        if session is not None:
            session.close()


@pytest.mark.parametrize("policy", ["close", "reuse"])
def test_json_bulk_fallback_is_bounded_after_gzip_decoding(
    endpoint, monkeypatch, policy
):
    monkeypatch.setattr(HTTPResponse, "read1", None, raising=False)
    sizes = []
    original = requests.Response.iter_content

    def read(response, chunk_size=1, **kwargs):
        sizes.append(chunk_size)
        yield from original(response, chunk_size=chunk_size, **kwargs)

    monkeypatch.setattr(requests.Response, "iter_content", read)
    payload = {"value": "é" * 10000}
    compressed = gzip.compress(json.dumps(payload).encode())
    base_url, observed = endpoint(
        lambda handler, _body: send(handler, compressed, encoding="gzip"),
        keepalive=True,
    )
    with LlamaCppClient(base_url, trust_env=False, connection_policy=policy) as api:
        with pytest.raises(LLMBackendError, match="json_response_too_large"):
            api.models(max_response_bytes=128)
        assert api.models(max_response_bytes=100000) == payload
    assert sizes == [65536, 65536]
    assert len(observed) == 2


def test_borrowed_json_session_keeps_cooperative_fallback_reads(endpoint, monkeypatch):
    monkeypatch.setattr(HTTPResponse, "read1", None, raising=False)
    sizes = []
    original = requests.Response.iter_content

    def read(response, chunk_size=1, **kwargs):
        sizes.append(chunk_size)
        yield from original(response, chunk_size=chunk_size, **kwargs)

    monkeypatch.setattr(requests.Response, "iter_content", read)
    base_url, observed = endpoint(lambda handler, _body: send(handler, b'{"part":1}'))
    with requests.Session() as borrowed:
        borrowed.trust_env = False
        with LlamaCppClient(base_url, timeout=1, session=borrowed) as api:
            assert api.models() == {"part": 1}
    assert sizes == [1]
    assert len(observed) == 1


@pytest.mark.parametrize("policy", ["close", "reuse"])
@pytest.mark.parametrize("streaming", [False, True])
def test_bulk_json_does_not_change_sse_fallback_or_hide_truncation(
    endpoint, monkeypatch, policy, streaming
):
    monkeypatch.setattr(HTTPResponse, "read1", None, raising=False)
    sizes = []
    original = requests.Response.iter_content

    def read(response, chunk_size=1, **kwargs):
        sizes.append(chunk_size)
        yield from original(response, chunk_size=chunk_size, **kwargs)

    monkeypatch.setattr(requests.Response, "iter_content", read)
    body = b'data: {"part":1}\n\n' if streaming else b'{"part":1}'
    base_url, observed = endpoint(lambda handler, _body: send(handler, body, extra=10))
    with LlamaCppClient(
        base_url, timeout=1, trust_env=False, connection_policy=policy
    ) as api:
        with pytest.raises(LLMBackendError):
            if streaming:
                list(api.complete("fixture", stream=True, idle_timeout=None))
            else:
                api.models()
    assert sizes == [1 if streaming else 65536]
    assert len(observed) == 1


@pytest.mark.parametrize("policy", ["close", "reuse"])
@pytest.mark.parametrize("tls", [False, True])
@pytest.mark.parametrize("termination", ["timeout", "cancel"])
def test_bulk_json_slow_drip_cannot_extend_deadline_or_cancellation(
    endpoint, monkeypatch, policy, tls, termination
):
    monkeypatch.setattr(HTTPResponse, "read1", None, raising=False)
    entered, release, requested = Event(), Event(), Event()
    endpoint.release_on_cleanup(release)

    def respond(handler, _body):
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", "1000000")
        if handler.headers.get("Connection") == "close":
            handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(b'{"part":')
        handler.wfile.flush()
        entered.set()
        try:
            while not release.wait(timeout=0.005):
                handler.wfile.write(b" ")
                handler.wfile.flush()
        finally:
            handler.close_connection = True

    base_url, observed = endpoint(respond, tls=tls, keepalive=True)
    with LlamaCppClient(
        base_url,
        timeout=0.2 if termination == "timeout" else 5,
        verify_ssl=not tls,
        trust_env=False,
        connection_policy=policy,
    ) as api:
        with ThreadPoolExecutor(max_workers=1) as callers:
            pending = callers.submit(
                api.models, cancel=requested.is_set, cancel_wait=True
            )
            assert entered.wait(timeout=1)
            if termination == "cancel":
                requested.set()
            expected = (
                LLMTimeoutError if termination == "timeout" else LLMCancelledError
            )
            try:
                with pytest.raises(expected):
                    pending.result(timeout=1)
                assert not api._streams
            finally:
                release.set()
    assert len(observed) == 1


@pytest.mark.parametrize("idle", [None, 1.0])
def test_natural_sse_completion_does_not_start_a_cleanup_thread(
    endpoint, monkeypatch, idle
):
    started = []

    def thread(*args, **kwargs):
        started.append(kwargs.get("name"))
        return Thread(*args, **kwargs)

    monkeypatch.setattr(http_client, "Thread", thread)
    body = b'data: {"part":1}\n\ndata: [DONE]\n\n'
    base_url, _ = endpoint(
        lambda handler, _body: send(handler, body, content_type="text/event-stream")
    )
    with LlamaCppClient(base_url, trust_env=False) as api:
        stream = api.complete(
            "fixture", stream=True, idle_timeout=idle, cancel_wait=True
        )
        assert list(stream) == [{"part": 1}]
        assert stream._drained.is_set()
        stream.close()
        deadline = time.monotonic() + 1
        while api._streams and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not api._streams
    assert started.count("runtime_llama_sse_reader") == 1
    assert "runtime_llama_http_close" not in started
