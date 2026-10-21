"""Shared native HTTP batches keep independent window lifecycles and real drain."""

import asyncio
from collections import Counter
from dataclasses import replace
import math
from threading import Event
import time

import pytest

from runtime_llama.adapters.embedding import _Call
from runtime_llama.client import ClientHTTPError
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError

from test_embedding_async_stream import (
    WindowSource,
    assert_drained,
    checkpoint,
    run,
    wait_thread,
    wait_until,
)
from test_embedding_contracts import ContractClient, configured, vector


def test_batch_group_freezes_unique_members_and_longest_initial_deadline():
    adapter = configured(ContractClient(1))
    try:
        first, second = _Call(adapter, None, 2), _Call(adapter, None, 10)
        members = [first, second, first]
        group = adapter._engine_batch_call(members)
        initial_deadline = second.deadline
        members.append(_Call(adapter, None, 100))
        second.deadline += 100
        assert group.calls == (first, second)
        assert group.deadline == initial_deadline
        assert math.isfinite(group.remaining())
        assert 5 < group.remaining() <= 10
    finally:
        adapter.close()


@pytest.mark.parametrize("first_failure", ["cancel", "probe", "timeout"])
def test_batch_group_retiring_one_owner_preserves_surviving_owner(first_failure):
    adapter = configured(ContractClient(1))
    try:

        def failed_probe():
            raise ValueError("private sentinel")

        first = _Call(adapter, failed_probe if first_failure == "probe" else None, 10)
        second = _Call(adapter, None, 5)
        group = adapter._engine_batch_call([first, second])
        if first_failure == "cancel":
            first.stop.set()
            expected = LLMCancelledError
        elif first_failure == "timeout":
            first.deadline = time.monotonic() - 1
            expected = LLMTimeoutError
        else:
            expected = LLMBackendError
        assert not group.probe()
        assert isinstance(first.failure, expected)
        assert second.failure is None
        group.check()
        assert 5 < group.remaining() <= 10
        assert "private sentinel" not in str(first.failure)
    finally:
        adapter.close()


def test_batch_group_all_retired_raises_first_error_without_extending_short_survivor():
    adapter = configured(ContractClient(1))
    try:
        first, second = _Call(adapter, None, 10), _Call(adapter, None, 2)
        group = adapter._engine_batch_call([first, second])
        first.fail(LLMCancelledError())
        assert not group.probe()
        second.deadline = time.monotonic() - 1
        assert group.probe()
        assert isinstance(second.failure, LLMTimeoutError)
        # The fixed ten-second socket bound does not keep an expired last owner
        # alive; the original first error survives the shared transport cleanup.
        with pytest.raises(LLMCancelledError):
            group.remaining()
    finally:
        adapter.close()


class SharedClient(ContractClient):
    """Expose actual native POST lifetime independently from cancellation detection."""

    supports_embedding_token_ids = True

    def __init__(self):
        super().__init__(1)
        self.prime_started = Event()
        self.prime_gate = Event()
        self.shared_started = Event()
        self.shared_gate = Event()
        self.transport_cancelled = Event()
        self.drain_gate = Event()
        self.drain_gate.set()
        self.retired_owner_polled = Event()
        self.retired_owner = lambda: False
        self.shared_timeout = None
        self.shared_error = None
        self.shared_response = None
        self.shared_posts = []

    def release_all(self):
        self.prime_gate.set()
        self.shared_gate.set()
        self.drain_gate.set()

    def embeddings(self, input, *, model, cancel=None, timeout=None, **_kwargs):
        texts = tuple(input)
        shared = any(text.startswith(("a-", "b-")) for text in texts)
        with self.lock:
            self.embed_active += 1
            self.embed_peak = max(self.embed_peak, self.embed_active)
            self.by_alias[model] += 1
            self.alias_peak[model] = max(self.alias_peak[model], self.by_alias[model])
            self.embedding_calls.append((model, texts))
        try:
            if texts == ("prime",):
                self.prime_started.set()
                self._wait(self.prime_gate, cancel=cancel, timeout=timeout)
            elif shared:
                self.shared_posts.append(texts)
                self.shared_timeout = timeout
                self.shared_started.set()
                deadline = time.monotonic() + timeout
                while True:
                    cancelled = bool(cancel and cancel())
                    if self.retired_owner() and not cancelled:
                        self.retired_owner_polled.set()
                    if cancelled:
                        self.transport_cancelled.set()
                        # A cancelled POST still owns its scheduler lease and all
                        # selected input credits until the real worker exits.
                        if not self.drain_gate.wait(3):
                            raise AssertionError("shared drain was not released")
                        raise LLMCancelledError()
                    if time.monotonic() >= deadline:
                        raise LLMTimeoutError("test_shared_transport_timeout")
                    if self.shared_gate.wait(0.002):
                        break
                if self.shared_error is not None:
                    raise self.shared_error
                if self.shared_response is not None:
                    return self.shared_response
            return {
                "data": [
                    {"index": index, "embedding": vector(text)}
                    for index, text in reversed(list(enumerate(texts)))
                ]
            }
        finally:
            with self.lock:
                self.embed_active -= 1
                self.by_alias[model] -= 1


class SharedScenario:
    """Fill a single occupied lane, then combine two ready independent owners."""

    def __init__(
        self,
        *,
        cancel_a=None,
        cancel_b=None,
        timeout_a=5,
        timeout_b=10,
        invalidate=None,
    ):
        self.client = SharedClient()
        self.adapter = configured(
            self.client,
            batch_size=4,
            dimension=2,
            embedding_input_format="text",
            embedding_batch_wait_sec=0,
            embedding_request_parallelism=1,
            embedding_queue_size=4,
            token_count_parallelism=2,
            timeout=10,
        )
        self.calls = {}
        original = self.adapter._stream_submit

        def capture(window, *, prepared, input_type, call):
            self.calls[window[0][0]] = call
            return original(window, prepared=prepared, input_type=input_type, call=call)

        self.adapter._stream_submit = capture
        if invalidate is not None:
            original_run = self.adapter._engine_run_shared
            if invalidate == "context":
                self.client.counts.update({"b-0": 1, "b-1": 1})

            def invalidate_owner(model, records, calls, group, observation):
                if not any(record.text.startswith("a-") for record in records):
                    return original_run(model, records, calls, group, observation)
                if invalidate == "context":
                    # Loaded capacity decreased after exact preparation. The
                    # small B inputs still fit; this is not a transport error.
                    observation = replace(
                        observation, context_length=3, context_token_limit=3
                    )
                else:
                    records = [
                        replace(record, _tokenizer_build="stale-tokenizer-build")
                        if owner is self.calls["a"]
                        else record
                        for record, owner in zip(records, calls)
                    ]
                return original_run(model, records, calls, group, observation)

            self.adapter._engine_run_shared = invalidate_owner
        self.expected_shared = (
            ["b-0", "b-1"] if invalidate is not None else ["a-0", "a-1", "b-0", "b-1"]
        )
        self.sources = [WindowSource([["a-0", "a-1"]]), WindowSource([["b-0", "b-1"]])]
        self.streams = [
            self.adapter.astream_texts(
                source, cancel=cancel, timeout=timeout, close_source=True
            )
            for source, cancel, timeout in zip(
                self.sources, (cancel_a, cancel_b), (timeout_a, timeout_b)
            )
        ]
        self.tasks = []
        self.prime = None

    async def __aenter__(self):
        try:
            self.prime = asyncio.create_task(
                asyncio.to_thread(self.adapter.embed_texts, ["prime"])
            )
            await wait_thread(self.client.prime_started)
            for stream in self.streams:
                await stream.__aenter__()
                self.tasks.append(asyncio.create_task(anext(stream)))
            await wait_until(
                lambda: (
                    sum(self.adapter._engine.snapshot()["ready_by_replica"].values())
                    == 4
                )
            )
            self.client.prime_gate.set()
            assert await self.prime == [vector("prime")]
            await wait_thread(self.client.shared_started)
            assert Counter(self.client.shared_posts[0]) == Counter(self.expected_shared)
            return self
        except BaseException:
            await self.close()
            raise

    async def close(self):
        self.client.release_all()
        pending = [*self.tasks, *([self.prime] if self.prime else [])]
        await asyncio.gather(*pending, return_exceptions=True)
        for stream in self.streams:
            await stream.aclose()
        await self.adapter.aclose()

    async def __aexit__(self, *args):
        await self.close()

    def assert_one_shared_post(self):
        assert len(self.client.shared_posts) == 1
        assert Counter(
            text for _, batch in self.client.embedding_calls for text in batch
        ) == Counter(["prime", "a-0", "a-1", "b-0", "b-1"])
        assert self.client.embed_peak == 1


@pytest.mark.parametrize("cancellation", ["probe", "task"])
def test_cancel_one_shared_owner_preserves_survivor_and_waits_actual_drain(
    cancellation,
):
    async def scenario():
        stop = Event()
        async with SharedScenario(cancel_a=stop.is_set) as shared:
            owner = shared.calls["a"]
            shared.client.retired_owner = owner.stop.is_set
            if cancellation == "probe":
                stop.set()
                expected = LLMCancelledError
            else:
                shared.tasks[0].cancel()
                expected = asyncio.CancelledError
            await wait_thread(shared.client.retired_owner_polled)
            await checkpoint()
            assert not shared.tasks[0].done()
            assert not shared.tasks[1].done()
            assert not shared.client.transport_cancelled.is_set()
            assert shared.adapter._scheduler.snapshot()["in_flight"] == 1
            shared.client.shared_gate.set()
            with pytest.raises(expected):
                await shared.tasks[0]
            assert await shared.tasks[1] == [vector("b-0"), vector("b-1")]
            shared.assert_one_shared_post()
            await shared.streams[1].aclose()
            assert_drained(shared.adapter, shared.client)

    run(scenario)


def test_shared_transport_uses_finite_longest_deadline_and_owner_timeout_is_local():
    async def scenario():
        async with SharedScenario(timeout_a=3, timeout_b=10) as shared:
            assert math.isfinite(shared.client.shared_timeout)
            assert 5 < shared.client.shared_timeout <= 10
            owner = shared.calls["a"]
            shared.client.retired_owner = owner.stop.is_set
            # A controlled work deadline expires while its shared POST is active.
            # Avoid a timing race or a multi-second sleep in this contract test.
            owner.deadline = time.monotonic() - 1
            await wait_thread(shared.client.retired_owner_polled)
            assert isinstance(owner.failure, LLMTimeoutError)
            assert not shared.tasks[0].done()
            assert not shared.client.transport_cancelled.is_set()
            shared.client.shared_gate.set()
            with pytest.raises(LLMTimeoutError):
                await shared.tasks[0]
            assert await shared.tasks[1] == [vector("b-0"), vector("b-1")]
            shared.assert_one_shared_post()

    run(scenario)


def test_all_shared_owners_cancel_transport_once_and_hold_lease_until_real_drain():
    async def scenario():
        stop_a, stop_b = Event(), Event()
        async with SharedScenario(
            cancel_a=stop_a.is_set, cancel_b=stop_b.is_set
        ) as shared:
            shared.client.drain_gate.clear()
            before = shared.adapter._engine.snapshot()
            stop_a.set()
            stop_b.set()
            await wait_thread(shared.client.transport_cancelled)
            assert not any(task.done() for task in shared.tasks)
            assert shared.adapter._scheduler.snapshot()["in_flight"] == 1
            assert (
                shared.adapter._engine.snapshot()["result_reserved_bytes"]
                == before["result_reserved_bytes"]
            )
            recovery = asyncio.create_task(
                asyncio.to_thread(shared.adapter.embed_query, "recovery")
            )
            await wait_until(lambda: len(shared.adapter._calls) == 3)
            assert len(shared.client.embedding_calls) == 2
            shared.client.drain_gate.set()
            for task in shared.tasks:
                with pytest.raises(LLMCancelledError):
                    await task
            assert await recovery == vector("recovery")
            assert len(shared.client.shared_posts) == 1
            assert shared.client.embed_peak == 1
            assert_drained(shared.adapter, shared.client)
            assert shared.adapter._engine.snapshot()["result_reserved_bytes"] == 0

    run(scenario)


def test_shared_native_http_error_preserves_cancelled_owner_first_failure_and_no_replay():
    async def scenario():
        stop = Event()
        async with SharedScenario(cancel_a=stop.is_set) as shared:
            owner = shared.calls["a"]
            shared.client.retired_owner = owner.stop.is_set
            shared.client.shared_error = ClientHTTPError(500, error_type="server_error")
            stop.set()
            await wait_thread(shared.client.retired_owner_polled)
            shared.client.shared_gate.set()
            with pytest.raises(LLMCancelledError):
                await shared.tasks[0]
            with pytest.raises(LLMBackendError, match="status=500"):
                await shared.tasks[1]
            shared.assert_one_shared_post()
            assert await asyncio.to_thread(
                shared.adapter.embed_query, "recovery"
            ) == vector("recovery")
            assert_drained(shared.adapter, shared.client)

    run(scenario)


def test_failing_owner_cancellation_probe_does_not_abort_shared_survivor():
    async def scenario():
        fail_probe = Event()

        def probe():
            if fail_probe.is_set():
                raise ValueError("private probe detail")
            return False

        async with SharedScenario(cancel_a=probe) as shared:
            owner = shared.calls["a"]
            shared.client.retired_owner = owner.stop.is_set
            fail_probe.set()
            await wait_thread(shared.client.retired_owner_polled)
            assert not shared.client.transport_cancelled.is_set()
            shared.client.shared_gate.set()
            with pytest.raises(
                LLMBackendError, match="cancellation_probe_failed"
            ) as error:
                await shared.tasks[0]
            assert "private probe detail" not in str(error.value)
            assert await shared.tasks[1] == [vector("b-0"), vector("b-1")]
            shared.assert_one_shared_post()

    run(scenario)


@pytest.mark.parametrize("invalid", ["context", "tokenizer_build"])
def test_pre_post_owner_validation_prunes_only_invalid_rows_and_scatter_is_not_rebased(
    invalid,
):
    async def scenario():
        async with SharedScenario(invalidate=invalid) as shared:
            assert shared.calls["a"].failure is not None
            assert shared.calls["b"].failure is None
            assert not shared.tasks[0].done()
            shared.client.shared_gate.set()
            with pytest.raises(LLMBackendError):
                await shared.tasks[0]
            assert await shared.tasks[1] == [vector("b-0"), vector("b-1")]
            assert shared.client.shared_posts == [("b-0", "b-1")]
            assert shared.client.embedding_calls == [
                ("embed_0", ("prime",)),
                ("embed_0", ("b-0", "b-1")),
            ]
            assert_drained(shared.adapter, shared.client)
            assert shared.adapter._engine.snapshot()["result_reserved_bytes"] == 0

    run(scenario)


@pytest.mark.parametrize("malformation", ["missing", "duplicate"])
def test_shared_native_response_cardinality_failure_fails_live_owners_without_replay(
    malformation,
):
    async def scenario():
        async with SharedScenario() as shared:
            rows = [
                {"index": index, "embedding": vector(text)}
                for index, text in enumerate(shared.client.shared_posts[0])
            ]
            if malformation == "missing":
                rows.pop()
            else:
                rows[-1]["index"] = rows[0]["index"]
            shared.client.shared_response = {"data": rows}
            shared.client.shared_gate.set()
            for task in shared.tasks:
                with pytest.raises(LLMBackendError):
                    await task
            shared.assert_one_shared_post()
            assert_drained(shared.adapter, shared.client)
            assert shared.adapter._engine.snapshot()["result_reserved_bytes"] == 0

    run(scenario)


def test_adapter_close_cancels_shared_transport_and_releases_results_after_drain():
    async def scenario():
        async with SharedScenario() as shared:
            shared.client.drain_gate.clear()
            closing = asyncio.create_task(shared.adapter.aclose())
            await wait_thread(shared.client.transport_cancelled)
            assert not closing.done()
            assert not any(task.done() for task in shared.tasks)
            assert shared.adapter._scheduler.snapshot()["in_flight"] == 1
            shared.client.drain_gate.set()
            await closing
            for task in shared.tasks:
                # Adapter close explicitly closes its streams; existing public
                # iteration semantics report exhaustion after their real drain.
                with pytest.raises(StopAsyncIteration):
                    await task
            shared.assert_one_shared_post()
            assert_drained(shared.adapter, shared.client)
            assert shared.adapter._engine.snapshot()["result_reserved_bytes"] == 0
            assert all(source.close_calls == 1 for source in shared.sources)

    run(scenario)


def test_shared_completed_windows_retain_only_undelivered_result_credit():
    async def scenario():
        client = SharedClient()
        adapter = configured(
            client,
            batch_size=4,
            dimension=2,
            embedding_input_format="text",
            embedding_batch_wait_sec=0,
            embedding_request_parallelism=1,
            embedding_queue_size=4,
            embedding_max_pending_bytes=8192,
            timeout=10,
        )
        source = WindowSource([["a-0", "a-1"], ["b-0", "b-1"]])
        stream = adapter.astream_texts(source, max_pending_windows=2, close_source=True)
        prime = asyncio.create_task(asyncio.to_thread(adapter.embed_texts, ["prime"]))
        first = None
        try:
            await wait_thread(client.prime_started)
            await stream.__aenter__()
            first = asyncio.create_task(anext(stream))
            await wait_until(
                lambda: (
                    sum(adapter._engine.snapshot()["ready_by_replica"].values()) == 4
                )
            )
            client.prime_gate.set()
            assert await prime == [vector("prime")]
            await wait_thread(client.shared_started)
            assert len(client.shared_posts[0]) == 4
            assert adapter._engine.snapshot()["result_reserved_bytes"] == 832
            client.shared_gate.set()
            assert await first == [vector("a-0"), vector("a-1")]
            await wait_until(
                lambda: adapter._engine.snapshot()["retained_result_bytes"] == 416
            )
            assert adapter._engine.snapshot()["result_reserved_bytes"] == 416
            assert not adapter._calls
            assert len(client.shared_posts) == 1
            await stream.aclose()
            await wait_until(
                lambda: adapter._engine.snapshot()["result_reserved_bytes"] == 0
            )
            assert_drained(adapter, client)
            assert source.close_calls == 1
        finally:
            client.release_all()
            await asyncio.gather(
                prime, *([first] if first else []), return_exceptions=True
            )
            await stream.aclose()
            await adapter.aclose()

    run(scenario)
