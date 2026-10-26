"""Actual shared-engine lifecycle of the progressive record facade."""

import asyncio
import gc
from threading import Event
import weakref

from runtime_llama.embedding_records import stream_text_records
from runtime_llama.errors import LLMCancelledError

from test_embedding_async_stream import wait_thread, wait_until
from test_embedding_contracts import ContractClient, configured, vector


def adapter_for(client):
    # Explicit capacities isolate lifecycle tests from configuration-default
    # migration. They are test choices, not defaults added to the new API.
    return configured(
        client,
        batch_size=8,
        embedding_batch_wait_sec=0.0,
        embedding_max_pending_owners=6,
        embedding_max_pending_bytes=64 * 1024 * 1024,
        embedding_max_pending_tokens=1024 * 1024,
        embedding_max_request_bytes=8 * 1024 * 1024,
        embedding_max_response_bytes=8 * 1024 * 1024,
        embedding_isolation_max_requests=512,
        embedding_preparation_cache_entries=0,
        embedding_preparation_cache_bytes=0,
        timeout=10,
    )


def test_actual_engine_delivers_before_the_flat_source_is_exhausted():
    async def scenario():
        client = ContractClient()
        adapter = adapter_for(client)
        more = asyncio.Event()
        exhausted = False

        async def source():
            nonlocal exhausted
            yield "first"
            await more.wait()
            yield "second"
            exhausted = True

        try:
            async with stream_text_records(
                adapter, source(), close_source=True
            ) as stream:
                result = await asyncio.wait_for(anext(stream), 4)
                assert result.index == 0 and result.embedding == vector("first")
                assert not exhausted
                more.set()
                second = await asyncio.wait_for(anext(stream), 4)
                assert second.index == 1 and second.embedding == vector("second")
            assert not adapter._calls
            assert adapter._async_window_bridge.used == 0
            assert not adapter._async_window_bridge.streams
        finally:
            more.set()
            await adapter.aclose()

    asyncio.run(scenario())


def test_dropped_facade_stops_inflight_http_but_waits_for_actual_drain():
    class Client(ContractClient):
        def __init__(self):
            super().__init__()
            self.later_started = Event()
            self.later_release = Event()
            self.later_drained = Event()

        def embeddings(self, input, *, cancel=None, **kwargs):
            if "later" in input:
                self.later_started.set()
                try:
                    # Deliberately ignore Stop until the external resource
                    # finishes: a completed waiter is not transport drain.
                    assert self.later_release.wait(5)
                    if cancel is not None and cancel():
                        raise LLMCancelledError()
                finally:
                    self.later_drained.set()
            return super().embeddings(input, cancel=cancel, **kwargs)

    async def scenario():
        client = Client()
        adapter = adapter_for(client)
        more = asyncio.Event()

        async def source():
            yield "first"
            await more.wait()
            yield "later"

        stream = stream_text_records(adapter, source(), close_source=True)
        stream_ref = weakref.ref(stream)
        try:
            result = await asyncio.wait_for(anext(stream), 4)
            assert result.embedding == vector("first")
            del result
            more.set()
            await wait_thread(client.later_started)
            del stream
            gc.collect()
            assert stream_ref() is None
            await asyncio.sleep(0)
            assert adapter._async_window_bridge.used > 0
            assert not client.later_drained.is_set()
            client.later_release.set()
            await wait_thread(client.later_drained)
            await wait_until(lambda: not adapter._calls)
            await wait_until(lambda: adapter._async_window_bridge.used == 0)
            assert not adapter._async_window_bridge.streams
            assert adapter._engine.snapshot()["retained_result_bytes"] == 0
        finally:
            more.set()
            client.later_release.set()
            await adapter.aclose()

    asyncio.run(scenario())


def test_dropped_facade_releases_completed_undelivered_windows():
    async def scenario():
        client = ContractClient()
        adapter = adapter_for(client)
        stream = stream_text_records(adapter, ["first", "second", "third"])
        try:
            result = await asyncio.wait_for(anext(stream), 4)
            assert result.index == 0
            del result
            inner = stream._stream
            await wait_until(
                lambda inner=inner: (
                    inner._exhausted
                    and all(window.future.done() for window in inner._pending)
                )
            )
            assert inner._pending
            assert adapter._engine.snapshot()["retained_result_bytes"] > 0
            inner_ref, stream_ref = weakref.ref(inner), weakref.ref(stream)
            del inner, stream
            gc.collect()
            assert stream_ref() is None
            await wait_until(lambda: adapter._async_window_bridge.used == 0)
            await asyncio.sleep(0)
            gc.collect()
            assert inner_ref() is None
            assert adapter._engine.snapshot()["retained_result_bytes"] == 0
            assert not adapter._async_window_bridge.streams
        finally:
            await adapter.aclose()

    asyncio.run(scenario())


def test_actual_engine_accepts_ten_thousand_flat_occurrences_without_owner_caps():
    class Texts(list):
        pass

    async def scenario():
        client = ContractClient(2)
        adapter = configured(
            client,
            batch_size=256,
            embedding_batch_wait_sec=0.0,
            embedding_queue_size=None,
            embedding_max_pending_owners=None,
            timeout=120,
        )
        values = Texts(f"flat-{index}" for index in range(10000))
        source_ref = weakref.ref(values)
        stream = stream_text_records(adapter, values)
        del values
        try:
            seen = 0
            async with stream:
                async for result in stream:
                    assert result.index == seen
                    assert result.embedding == vector(f"flat-{seen}")
                    seen += 1
            assert seen == 10000
            del result
            gc.collect()
            assert source_ref() is None
            assert not adapter._calls
            assert adapter._intake.snapshot()["owners"] == 0
            assert adapter._async_window_bridge.used == 0
            assert not adapter._async_window_bridge.streams
            assert adapter._engine.snapshot()["retained_result_bytes"] == 0
            assert adapter._scheduler.snapshot()["in_flight"] == 0
        finally:
            await adapter.aclose()

    asyncio.run(scenario())
