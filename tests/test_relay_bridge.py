"""Real invocation threads, transport handoff and cooperative abandonment."""

import asyncio
from threading import Event, Lock
import time
import weakref
import gc

import pytest

from runtime_llama.errors import LLMCancelledError
from runtime_llama.relay.bridge import ThreadBridge, run_blocking


async def until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "fixture did not advance"
        await asyncio.sleep(0.001)


def test_bridge_preserves_order_and_final_result_without_pool():
    async def scenario():
        def produce(emit, _cancel):
            for value in range(100):
                emit(value)
            return "complete"

        async with ThreadBridge(produce) as bridge:
            assert [value async for value in bridge] == list(range(100))
            assert await bridge.result() == "complete"
        assert not bridge._thread.is_alive()
        assert bridge._producer is None and bridge._completion is None

    asyncio.run(scenario())


def test_abandonment_unblocks_a_full_handoff_and_runs_producer_cleanup():
    async def scenario():
        attempted, drained = Event(), Event()

        def produce(emit, cancel):
            try:
                emit("first")
                attempted.set()
                emit("second")
                pytest.fail("the abandoned handoff must be cancelled")
            finally:
                assert cancel.is_set()
                drained.set()

        bridge = ThreadBridge(produce)
        await until(attempted.is_set)
        await bridge.aclose()
        assert drained.is_set() and not bridge._thread.is_alive()
        assert bridge._items.empty()
        await bridge.aclose()

    asyncio.run(scenario())


def test_cancel_token_set_wakes_pending_handoff_and_notice():
    async def scenario():
        waiting = Event()

        def produce(emit, _cancel):
            emit(1)
            waiting.set()
            emit(2)

        bridge = ThreadBridge(produce)
        await until(waiting.is_set)
        bridge.cancel_event.set()
        await asyncio.wait_for(bridge.wait_cancelled(), 1)
        with pytest.raises(LLMCancelledError):
            await bridge.result()
        await bridge.aclose()
        assert not bridge._thread.is_alive()

    asyncio.run(scenario())


def test_cancelled_async_task_drains_a_quiet_cooperative_worker():
    async def scenario():
        started, drained = Event(), Event()

        def produce(_emit, cancel):
            started.set()
            try:
                assert cancel.wait(2)
                raise LLMCancelledError()
            finally:
                drained.set()

        bridge = ThreadBridge(produce)
        task = asyncio.create_task(bridge.result())
        await until(started.is_set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await bridge.aclose()
        assert drained.is_set() and not bridge._thread.is_alive()

    asyncio.run(scenario())


def test_more_than_framework_default_workers_can_start_together():
    async def scenario():
        lock, release = Lock(), Event()
        started = 0

        def produce(_emit, cancel):
            nonlocal started
            with lock:
                started += 1
            assert release.wait(3)
            return not cancel.is_set()

        bridges = [ThreadBridge(produce) for _ in range(48)]
        try:
            await until(lambda: started == 48)
            release.set()
            assert all(await asyncio.gather(*(bridge.result() for bridge in bridges)))
        finally:
            release.set()
            for bridge in bridges:
                await bridge.aclose()
        assert all(not bridge._thread.is_alive() for bridge in bridges)

    asyncio.run(scenario())


def test_failed_worker_does_not_retain_its_stack_payload_after_close():
    class Payload:
        pass

    async def scenario():
        references = []

        def produce(_emit, _cancel):
            payload = Payload()
            references.append(weakref.ref(payload))
            raise ValueError("fixture")

        bridge = ThreadBridge(produce)
        with pytest.raises(ValueError):
            await bridge.result()
        await bridge.aclose()
        gc.collect()
        assert references[0]() is None
        assert not bridge._thread.is_alive()

    asyncio.run(scenario())


def test_run_blocking_returns_value_and_closes_owned_thread():
    assert asyncio.run(run_blocking(lambda cancel: (42, cancel.is_set()))) == (
        42,
        False,
    )
