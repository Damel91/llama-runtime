"""Partial-result formatting remains owned through close, cancel and deadline."""

import asyncio
from threading import Event
import time

import pytest

import runtime_llama.automatic_embedding as module
from runtime_llama.client import ClientHTTPError
from runtime_llama.embedding_halving import _rejected_embedding
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError

from test_automatic_embedding import make_service as make_service
from test_automatic_embedding import remote
from test_embedding_async_stream import WindowSource, checkpoint, run, wait_until


def gated_conversion(monkeypatch):
    entered, release = Event(), Event()
    convert = module._halving_error
    captured = []

    def held(result, *args):
        captured.append(result)
        entered.set()
        if not release.wait(3):
            raise AssertionError("test did not release partial-result conversion")
        return convert(result, *args)

    monkeypatch.setattr(module, "_halving_error", held)
    return entered, release, captured


def rejecting_service(make_service, **changes):
    client = remote(replicas=1)
    client.failure = lambda *_: ClientHTTPError(500)
    service, _ = make_service(client, **changes)
    return service, client


def test_service_close_drains_conversion_before_completing(make_service, monkeypatch):
    async def scenario():
        service, client = rejecting_service(make_service)
        entered, release, _ = gated_conversion(monkeypatch)
        stream = service.astream_texts(WindowSource([["rejected"]]))
        pending = asyncio.create_task(anext(stream))
        closing = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            closing = asyncio.create_task(service.aclose())
            await wait_until(lambda: stream._closed)
            await asyncio.wait({closing}, timeout=0.1)
            assert not closing.done(), "close abandoned its partial-result worker"
            assert not pending.done()
            release.set()
            await asyncio.wait_for(closing, 2)
            with pytest.raises(
                (LLMBackendError, LLMCancelledError, StopAsyncIteration)
            ) as stopped:
                await asyncio.wait_for(pending, 2)
            assert type(stopped.value).__name__ != "EmbeddingChunkSplitRequiredError"
            assert not client.token_calls and not service._streams
        finally:
            release.set()
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
            await asyncio.gather(pending, return_exceptions=True)
            await service.aclose()

    run(scenario)


def test_task_cancellation_during_conversion_waits_for_worker_and_drains(
    make_service, monkeypatch
):
    async def scenario():
        service, client = rejecting_service(make_service)
        entered, release, _ = gated_conversion(monkeypatch)
        stream = service.astream_texts(WindowSource([["rejected"]]))
        pending = asyncio.create_task(anext(stream))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            pending.cancel()
            await checkpoint()
            assert not pending.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 2)
            assert stream._closed and stream._delegate._closed
            assert not service._streams and not client.token_calls
            assert service._adapter._engine.snapshot()["result_reserved_bytes"] == 0
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await service.aclose()

    run(scenario)


def test_consumer_stop_during_conversion_has_priority_over_halving_advice(
    make_service, monkeypatch
):
    async def scenario():
        service, client = rejecting_service(make_service)
        entered, release, _ = gated_conversion(monkeypatch)
        cancel = Event()
        stream = service.astream_texts(
            WindowSource([["rejected"]]), cancel=cancel.is_set
        )
        pending = asyncio.create_task(anext(stream))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            cancel.set()
            release.set()
            with pytest.raises(LLMCancelledError):
                await asyncio.wait_for(pending, 2)
            assert stream._closed and not service._streams
            assert not client.token_calls
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await service.aclose()

    run(scenario)


def test_conversion_preserves_original_window_deadline(make_service, monkeypatch):
    async def scenario():
        service, client = rejecting_service(make_service, timeout=0.5)
        entered, release, captured = gated_conversion(monkeypatch)
        stream = service.astream_texts(WindowSource([["rejected"]]))
        pending = asyncio.create_task(anext(stream))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            deadline = captured[0].deadline
            assert deadline is not None and deadline < time.monotonic() + 0.5
            await asyncio.sleep(max(0, deadline - time.monotonic()) + 0.01)
            release.set()
            with pytest.raises(LLMTimeoutError):
                await asyncio.wait_for(pending, 2)
            assert stream._closed and not service._streams
            assert not client.token_calls
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await service.aclose()

    run(scenario)


def test_recoverable_marker_cause_has_no_http_traceback_or_private_context():
    parent = RuntimeError("private parent source")
    original = ClientHTTPError(500, error_code="too_large", error_type="server_error")
    original.__cause__ = parent
    original.__context__ = parent
    try:
        raise original
    except ClientHTTPError as caught:
        marker = _rejected_embedding("private source", "document", caught)
        assert marker.original is not caught
        assert marker.original.__traceback__ is None
        assert marker.original.__cause__ is None
        assert marker.original.__context__ is None
        assert marker.original.status_code == 500
        assert marker.original.error_code == "too_large"
        assert marker.original.error_type == "server_error"
        assert "private" not in str(marker.original)
