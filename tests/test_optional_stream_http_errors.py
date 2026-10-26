from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from threading import Event, enumerate as threads
import time
import traceback

import pytest

from runtime_llama.adapters.llama import LlamaCppAdapter
from runtime_llama.client import ClientHTTPError, LlamaCppClient
from runtime_llama.errors import LLMCancelledError
from test_optional_stream_idle_transport import (
    SECOND,
    endpoint as endpoint,
    headers,
    wait_for,
)


OVERFLOW = {"error": {"code": 400, "type": "exceed_context_size_error"}}


def watchdogs():
    return [
        thread
        for thread in threads()
        if thread.name == "runtime_llama_http_error_watchdog"
    ]


def error_headers(handler, *, length, encoding=None, keepalive=False):
    handler.send_response(400)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(length))
    if encoding is not None:
        handler.send_header("Content-Encoding", encoding)
    if not keepalive:
        handler.send_header("Connection", "close")
    handler.end_headers()


def send_error(handler, payload=OVERFLOW, *, keepalive=False):
    body = json.dumps(payload).encode()
    error_headers(handler, length=len(body), keepalive=keepalive)
    handler.wfile.write(body)
    handler.wfile.flush()


def assert_status_only(error):
    assert error.status_code == 400
    assert error.error_code is None and error.error_type is None


@pytest.mark.parametrize(
    "payload,code,kind",
    [
        (OVERFLOW, "400", "exceed_context_size_error"),
        (
            {"code": "context_length_exceeded", "type": "invalid_request_error"},
            "context_length_exceeded",
            "invalid_request_error",
        ),
        ({"error": {"code": False, "type": "untrusted type with spaces"}}, None, None),
    ],
)
def test_timely_sanitized_error_codes_preserved_without_retry(
    endpoint, payload, code, kind
):
    base_url, observed = endpoint(lambda handler, _body: send_error(handler, payload))
    with LlamaCppClient(base_url, timeout=0.2, trust_env=False) as api:
        with pytest.raises(ClientHTTPError) as caught:
            list(api.complete("fixture", stream=True, idle_timeout=None))
        assert caught.value.status_code == 400
        assert caught.value.error_code == code and caught.value.error_type == kind
        assert not api._streams and not watchdogs()
    assert len(observed) == 1


def test_adapter_preserves_overflow_codes_and_nullable_idle_diagnostics(endpoint):
    secret = "DO-NOT-EXPOSE-PROVIDER-OR-AUTH-MESSAGE"
    payload = {
        "error": {
            "code": "context_length_exceeded",
            "type": "exceed_context_size_error",
            "message": secret,
        }
    }
    base_url, observed = endpoint(lambda handler, _body: send_error(handler, payload))
    with LlamaCppClient(base_url, api_key=secret, timeout=0.2, trust_env=False) as api:
        adapter = LlamaCppAdapter(
            model="fixture", client=api, stream_idle_timeout_sec=None
        )
        try:
            with pytest.raises(ClientHTTPError) as caught:
                adapter.generate("fixture")
            assert caught.value.status_code == 400
            assert caught.value.error_code == "context_length_exceeded"
            assert caught.value.error_type == "exceed_context_size_error"
            assert caught.value.diagnostics.idle_timeout_sec is None
            assert secret not in "".join(traceback.format_exception(caught.value))
        finally:
            adapter.close()
    assert len(observed) == 1 and not watchdogs()


@pytest.mark.parametrize("style", ["quiet", "drip", "gzip-header"])
def test_total_error_body_budget_cannot_be_extended_by_slow_bytes(endpoint, style):
    entered, release = Event(), Event()
    endpoint.release_on_cleanup(release)

    def respond(handler, _body):
        error_headers(
            handler, length=1000000, encoding="gzip" if style == "gzip-header" else None
        )
        if style == "gzip-header":
            handler.wfile.write(b"\x1f\x8b\x08\x08" + b"\x00" * 6)
            handler.wfile.flush()
        entered.set()
        while not release.wait(timeout=0.01):
            if style != "quiet":
                handler.wfile.write(b"x" if style == "gzip-header" else b" ")
                handler.wfile.flush()

    base_url, observed = endpoint(respond)
    with LlamaCppClient(base_url, timeout=0.12, trust_env=False) as api:
        stream = api.complete("fixture", stream=True, idle_timeout=None)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(list, stream)
            assert entered.wait(timeout=1)
            with pytest.raises(ClientHTTPError) as caught:
                pending.result(timeout=0.75)
        assert_status_only(caught.value)
        assert not api._streams and not watchdogs()
    assert len(observed) == 1


def test_error_body_uses_original_deadline_without_hidden_five_second_cap(endpoint):
    def respond(handler, _body):
        body = json.dumps(OVERFLOW).encode()
        error_headers(handler, length=len(body))
        time.sleep(5.1)
        handler.wfile.write(body)
        handler.wfile.flush()

    base_url, observed = endpoint(respond)
    with LlamaCppClient(base_url, timeout=8, trust_env=False) as api:
        stream = api.complete("fixture", stream=True, idle_timeout=None)
        with ThreadPoolExecutor(max_workers=1) as executor:
            with pytest.raises(ClientHTTPError) as caught:
                executor.submit(list, stream).result(timeout=7)
        assert caught.value.error_code == "400"
        assert caught.value.error_type == "exceed_context_size_error"
        assert not api._streams and not watchdogs()
    assert len(observed) == 1


@pytest.mark.parametrize(
    "variant",
    [
        "oversize",
        "chunked-oversize",
        "gzip-oversize",
        "gzip-corrupt",
        "caller-limit",
        "malformed",
        "truncated",
    ],
)
def test_unusable_or_excessive_body_keeps_status_and_discards_partial_codes(
    endpoint, variant
):
    body = json.dumps(
        {
            "error": {
                "code": 400,
                "type": "exceed_context_size_error",
                "message": "x" * 65536,
            }
        }
    ).encode()
    kwargs = {"max_response_bytes": 65536}
    if variant in {"caller-limit", "truncated"}:
        body = json.dumps(OVERFLOW).encode()
    if variant == "caller-limit":
        kwargs["max_response_bytes"] = len(body) - 1
    if variant == "malformed":
        body = b'{"error":{"code":400,"type":"exceed_context_size_error"}'
    encoded = gzip.compress(body) if variant == "gzip-oversize" else body
    if variant == "gzip-corrupt":
        encoded = b"not a gzip stream"

    def respond(handler, _body):
        if variant == "chunked-oversize":
            handler.send_response(400)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Transfer-Encoding", "chunked")
            handler.send_header("Connection", "close")
            handler.end_headers()
            for index in range(0, len(encoded), 4096):
                chunk = encoded[index : index + 4096]
                handler.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            handler.wfile.write(b"0\r\n\r\n")
            handler.wfile.flush()
            return
        error_headers(
            handler,
            length=len(encoded) + (10 if variant == "truncated" else 0),
            encoding="gzip" if variant in {"gzip-oversize", "gzip-corrupt"} else None,
        )
        handler.wfile.write(encoded)
        handler.wfile.flush()

    base_url, observed = endpoint(respond)
    # Allow urllib3 1.26's one-byte reader to exercise the byte/format bound,
    # without a tiny body deadline masking a missing size check.
    with LlamaCppClient(base_url, timeout=2, trust_env=False) as api:
        with pytest.raises(ClientHTTPError) as caught:
            list(
                api.request(
                    "POST",
                    "/completion",
                    json={"prompt": "fixture"},
                    stream=True,
                    idle_timeout=None,
                    **kwargs,
                )
            )
        assert_status_only(caught.value)
        assert not api._streams and not watchdogs()
    assert len(observed) == 1


def test_exact_error_body_limit_keeps_codes(endpoint):
    body = json.dumps(OVERFLOW).encode()

    def respond(handler, _body):
        error_headers(handler, length=len(body))
        handler.wfile.write(body)

    base_url, observed = endpoint(respond)
    with LlamaCppClient(base_url, timeout=0.2, trust_env=False) as api:
        with pytest.raises(ClientHTTPError) as caught:
            list(
                api.request(
                    "POST",
                    "/completion",
                    json={},
                    stream=True,
                    idle_timeout=None,
                    max_response_bytes=len(body),
                )
            )
        assert caught.value.error_code == "400"
        assert caught.value.error_type == "exceed_context_size_error"
    assert len(observed) == 1 and not watchdogs()


@pytest.mark.parametrize("cancel_wait", [False, True])
def test_user_cancellation_during_error_body_wins_over_http_fallback(
    endpoint, cancel_wait
):
    entered, release, requested = Event(), Event(), Event()
    endpoint.release_on_cleanup(release)

    def respond(handler, _body):
        error_headers(handler, length=100)
        entered.set()
        release.wait(timeout=3)

    base_url, observed = endpoint(respond)
    with LlamaCppClient(base_url, timeout=1, trust_env=False) as api:
        stream = api.complete(
            "fixture",
            stream=True,
            idle_timeout=None,
            cancel=requested.is_set,
            cancel_wait=cancel_wait,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(list, stream)
            assert entered.wait(timeout=1)
            requested.set()
            with pytest.raises(LLMCancelledError):
                pending.result(timeout=0.75)
        wait_for(lambda: not api._streams and not watchdogs())
    assert len(observed) == 1


def test_client_close_during_error_body_drains_watchdog(endpoint):
    entered, release = Event(), Event()
    endpoint.release_on_cleanup(release)

    def respond(handler, _body):
        error_headers(handler, length=100)
        entered.set()
        release.wait(timeout=3)

    base_url, observed = endpoint(respond)
    api = LlamaCppClient(base_url, timeout=1, trust_env=False)
    stream = api.complete("fixture", stream=True, idle_timeout=None)
    try:
        assert entered.wait(timeout=1)
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(api.close).result(timeout=0.75)
        assert not api._streams and not watchdogs()
        assert list(stream) == []
    finally:
        release.set()
        api.close()
    assert len(observed) == 1


def test_error_body_budget_runs_without_consumer_iteration(endpoint):
    release = Event()
    endpoint.release_on_cleanup(release)

    def respond(handler, _body):
        error_headers(handler, length=100)
        release.wait(timeout=3)

    base_url, observed = endpoint(respond)
    with LlamaCppClient(base_url, timeout=0.12, trust_env=False) as api:
        stream = api.complete("fixture", stream=True, idle_timeout=None)
        wait_for(lambda: not api._streams, timeout=0.75)
        assert not watchdogs()
        with pytest.raises(ClientHTTPError) as caught:
            next(stream)
        assert_status_only(caught.value)
    assert len(observed) == 1


def test_error_budget_does_not_interrupt_concurrent_healthy_stream(endpoint):
    entered = {name: Event() for name in ("error", "healthy")}
    release = Event()
    endpoint.release_on_cleanup(release)

    def respond(handler, body):
        name = body["prompt"]
        if name == "error":
            error_headers(handler, length=100)
        else:
            headers(handler)
        entered[name].set()
        release.wait(timeout=3)
        if name == "healthy":
            handler.wfile.write(SECOND)
            handler.wfile.flush()

    base_url, observed = endpoint(respond)
    with LlamaCppClient(base_url, timeout=0.12, trust_env=False) as api:
        error = api.complete("error", stream=True, idle_timeout=None)
        healthy = api.complete("healthy", stream=True, idle_timeout=None)
        with ThreadPoolExecutor(max_workers=2) as executor:
            failed = executor.submit(list, error)
            succeeded = executor.submit(list, healthy)
            assert all(event.wait(timeout=1) for event in entered.values())
            with pytest.raises(ClientHTTPError) as caught:
                failed.result(timeout=0.75)
            assert_status_only(caught.value)
            assert not succeeded.done()
            release.set()
            assert succeeded.result(timeout=1) == [{"part": 2}]
        assert not api._streams and not watchdogs()
    assert len(observed) == 2


def test_completed_error_watchdog_and_stale_stream_cannot_hit_next_keepalive_lease(
    endpoint,
):
    entered, release = Event(), Event()
    endpoint.release_on_cleanup(release)

    def respond(handler, body):
        if body["prompt"] == "error":
            send_error(handler, keepalive=True)
        else:
            entered.set()
            release.wait(timeout=3)
            headers(handler, body_length=len(SECOND), keepalive=True)
            handler.wfile.write(SECOND)
            handler.wfile.flush()

    base_url, observed = endpoint(respond, keepalive=True)
    with LlamaCppClient(
        base_url, timeout=0.12, trust_env=False, connection_policy="reuse"
    ) as api:
        old = api.complete("error", stream=True, idle_timeout=None)
        wait_for(lambda: not api._streams)
        assert not watchdogs()
        new = api.complete("healthy", stream=True, idle_timeout=None)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(list, new)
            assert entered.wait(timeout=1)
            time.sleep(0.2)
            with pytest.raises(ClientHTTPError) as caught:
                next(old)
            assert caught.value.error_type == "exceed_context_size_error"
            assert not pending.done()
            release.set()
            assert pending.result(timeout=1) == [{"part": 2}]
        assert not api._streams and not watchdogs()
    assert len(observed) == 2
    assert observed[0][2] == observed[1][2]
