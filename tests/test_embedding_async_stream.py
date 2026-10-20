"""Async window contracts using stdlib asyncio and cooperative offline fixtures."""

import asyncio
from collections import Counter
from threading import Event, enumerate as threads
from typing import get_type_hints

import pytest

import runtime_llama as runtime
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError

from test_embedding_contracts import (
    CONTENT,
    ContractClient,
    configured,
    eventually,
    vector,
)


class WindowSource:
    """Track source ownership, concurrent pulls and cancellation independently."""

    def __init__(self, windows, *, block_at=None, error_at=None, error=None):
        self.windows = windows
        self.block_at = block_at
        self.error_at = error_at
        self.error = error or RuntimeError("source sentinel")
        self.position = 0
        self.pulls = 0
        self.active = self.peak = 0
        self.close_calls = self.cancelled_pulls = 0
        self.changed = asyncio.Event()
        self.release = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        index = self.position
        self.pulls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.changed.set()
        try:
            if index == self.block_at:
                await self.release.wait()
            if index == self.error_at:
                raise self.error
            if index == len(self.windows):
                raise StopAsyncIteration
            self.position += 1
            return self.windows[index]
        except asyncio.CancelledError:
            self.cancelled_pulls += 1
            raise
        finally:
            self.active -= 1
            self.changed.set()

    async def wait_pulls(self, count):
        while self.pulls < count:
            self.changed.clear()
            if self.pulls >= count:
                break
            await self.changed.wait()

    async def aclose(self):
        self.close_calls += 1


def run(scenario):
    existing = {thread.ident for thread in threads()}
    asyncio.run(scenario())
    assert not [
        thread.name
        for thread in threads()
        if thread.ident not in existing and thread.name.startswith("runtime_llama_")
    ]


async def wait_thread(event):
    assert await asyncio.wait_for(asyncio.to_thread(event.wait, 2), 3)


async def wait_until(predicate):
    await asyncio.wait_for(asyncio.to_thread(eventually, predicate), 4)


async def checkpoint():
    """An event-loop scheduling barrier, not a throughput or sleep assertion."""
    future = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(future.set_result, None)
    await future


async def collect(stream):
    return [window async for window in stream]


def assert_drained(adapter, client):
    assert not adapter._calls
    snapshot = adapter._scheduler.snapshot()
    assert snapshot["queued"] == snapshot["in_flight"] == 0
    assert client.token_active == client.embed_active == 0


def test_async_factories_are_inert_without_an_event_loop(monkeypatch):
    client = ContractClient()
    adapter = configured(client)
    source = WindowSource([["one"]])

    def forbidden(*args, **kwargs):
        raise AssertionError("factory performed remote work")

    for name in ("health", "models", "props", "tokenize", "embeddings"):
        monkeypatch.setattr(client, name, forbidden)
    before = {thread.ident for thread in threads()}
    try:
        assert get_type_hints(runtime.EmbeddingWindowStream.__init__)
        assert (
            get_type_hints(adapter.astream_texts)["return"]
            is runtime.EmbeddingWindowStream
        )
        text_stream = adapter.astream_texts(source)
        prepared_stream = adapter.astream_prepared(source)
        assert isinstance(text_stream, runtime.EmbeddingWindowStream)
        assert isinstance(prepared_stream, runtime.EmbeddingWindowStream)
        assert source.pulls == source.close_calls == 0
        assert {thread.ident for thread in threads()} == before
    finally:
        adapter.close()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_pending_window_budget_is_rejected_before_source_pull(value):
    async def scenario():
        client = ContractClient()
        adapter = configured(client)
        source = WindowSource([["one"]])
        try:
            with pytest.raises(ValueError):
                stream = adapter.astream_texts(source, max_pending_windows=value)
                async with stream:
                    await anext(stream)
            assert source.pulls == 0
            assert not client.token_calls and not client.embedding_calls
        finally:
            await adapter.aclose()

    run(scenario)


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_invalid_window_deadline_is_rejected_before_source_pull(value):
    async def scenario():
        client = ContractClient()
        adapter = configured(client)
        source = WindowSource([["one"]])
        try:
            with pytest.raises(ValueError):
                stream = adapter.astream_texts(source, timeout=value)
                async with stream:
                    await anext(stream)
            assert source.pulls == 0
            assert not client.token_calls and not client.embedding_calls
        finally:
            await adapter.aclose()

    run(scenario)


def test_order_empty_windows_prefixes_and_multiple_internal_batches():
    async def scenario():
        client = ContractClient(2)
        adapter = configured(
            client,
            batch_size=2,
            embedding_request_parallelism=2,
            embedding_queue_size=1,
            embedding_prefixes={"query": "Q: ", "document": "D: "},
        )
        incoming = [[], ["one", "two", "three", "four", "five"], ["six"], []]
        source = WindowSource(incoming)
        stream = adapter.astream_texts(source, max_pending_windows=3, close_source=True)
        try:
            async with stream:
                result = await asyncio.wait_for(collect(stream), 4)
            assert result == [
                [vector("D: " + text) for text in window] for window in incoming
            ]
            assert source.peak == 1 and source.close_calls == 1
            observed = [text for _, batch in client.embedding_calls for text in batch]
            assert Counter(observed) == Counter(
                "D: " + text for window in incoming for text in window
            )
            await stream.aclose()
            assert source.close_calls == 1
            assert_drained(adapter, client)
        finally:
            await adapter.aclose()

    run(scenario)


def test_prepared_stream_reuses_safe_counts_and_task_prefixes_once():
    async def scenario():
        client = ContractClient(2)
        adapter = configured(
            client, embedding_prefixes={"query": "Q: ", "document": "D: "}
        )
        try:
            prepared = await asyncio.to_thread(
                adapter.prepare_texts, ["one", "two"], input_type="query"
            )
            counted = len(client.token_calls)
            source = WindowSource([prepared[:1], (), prepared[1:]])
            async with adapter.astream_prepared(source) as stream:
                assert await collect(stream) == [
                    [vector("Q: one")],
                    [],
                    [vector("Q: two")],
                ]
            assert len(client.token_calls) == counted
            assert [text for _, batch in client.embedding_calls for text in batch] == [
                "Q: one",
                "Q: two",
            ]
            assert source.close_calls == 0
        finally:
            await adapter.aclose()

    run(scenario)


def test_foreign_preparation_and_measured_oversize_do_not_reach_http():
    async def scenario():
        client = ContractClient(1)
        foreign_client = ContractClient(1)
        adapter, foreign = configured(client), configured(foreign_client)
        try:
            record = await asyncio.to_thread(foreign.prepare_query, "foreign")
            async with adapter.astream_prepared(WindowSource([[record]])) as stream:
                with pytest.raises(LLMBackendError):
                    await anext(stream)
            client.counts["oversize"] = 2049
            oversize = await asyncio.to_thread(adapter.prepare_query, "oversize")
            assert oversize.token_count == 2049 and oversize.fits_input is False
            async with adapter.astream_prepared(WindowSource([[oversize]])) as stream:
                with pytest.raises(LLMBackendError, match="exceeds_context"):
                    await anext(stream)
            assert not client.embedding_calls
            assert await asyncio.to_thread(adapter.embed_query, "healthy") == vector(
                "healthy"
            )
        finally:
            await adapter.aclose()
            await foreign.aclose()

    run(scenario)


def test_window_is_copied_before_producer_can_mutate_admitted_list():
    async def scenario():
        client = ContractClient(1)
        client.token_gate.clear()
        adapter = configured(client)
        original = ["original"]
        stream = adapter.astream_texts(WindowSource([original]), max_pending_windows=1)
        try:
            async with stream:
                pending = asyncio.create_task(anext(stream))
                await wait_thread(client.token_started)
                original[0] = "mutated"
                client.token_gate.set()
                assert await asyncio.wait_for(pending, 3) == [vector("original")]
            assert client.embedding_calls == [("embed_0", ("original",))]
        finally:
            client.token_gate.set()
            await adapter.aclose()

    run(scenario)


@pytest.mark.parametrize("phase", ["source", "preparation", "http"])
def test_one_window_lease_is_held_through_source_preparation_and_http(phase):
    async def scenario():
        client = ContractClient(1)
        adapter = configured(
            client, embedding_request_parallelism=1, embedding_queue_size=0, timeout=5
        )
        source = WindowSource(
            [["first"], ["second"], ["third"]],
            block_at=0 if phase == "source" else None,
        )
        if phase == "preparation":
            client.token_gate.clear()
        elif phase == "http":
            client.embed_gate.clear()
        stream = adapter.astream_texts(source, max_pending_windows=1)
        pending = None
        try:
            pending = asyncio.create_task(anext(stream))
            if phase == "source":
                await asyncio.wait_for(source.wait_pulls(1), 2)
            else:
                await wait_thread(
                    client.token_started
                    if phase == "preparation"
                    else client.embed_started
                )
            await checkpoint()
            assert (
                not pending.done()
            )  # The heartbeat ran while remote/source work waits.
            assert source.pulls == 1 and source.peak == 1
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 3)
            assert source.active == 0
            assert_drained(adapter, client)
        finally:
            source.release.set()
            client.token_gate.set()
            client.embed_gate.set()
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await stream.aclose()
            await adapter.aclose()

    run(scenario)


def test_out_of_order_ready_result_keeps_its_lease_and_preserves_window_order(
    monkeypatch,
):
    async def scenario():
        client = ContractClient(2)
        adapter = configured(
            client, embedding_request_parallelism=2, embedding_queue_size=0, timeout=5
        )
        source = WindowSource([["first"], ["second"], ["third"]])
        first_entered, second_finished, release_first = Event(), Event(), Event()
        original = client.embeddings

        def embeddings(input, **kwargs):
            result = original(input, **kwargs)
            if input == ["first"]:
                first_entered.set()
                client._wait(
                    release_first, cancel=kwargs["cancel"], timeout=kwargs["timeout"]
                )
            elif input == ["second"]:
                second_finished.set()
            return result

        monkeypatch.setattr(client, "embeddings", embeddings)
        stream = adapter.astream_texts(source, max_pending_windows=2)
        try:
            async with stream:
                first = asyncio.create_task(anext(stream))
                await wait_thread(first_entered)
                await wait_thread(second_finished)
                await checkpoint()
                assert not first.done()
                assert source.pulls == 2
                release_first.set()
                assert await first == [vector("first")]
                assert await collect(stream) == [[vector("second")], [vector("third")]]
            assert Counter(
                text for _, batch in client.embedding_calls for text in batch
            ) == Counter(["first", "second", "third"])
        finally:
            release_first.set()
            await stream.aclose()
            await adapter.aclose()

    run(scenario)


def test_slow_consumer_cannot_release_completed_but_undelivered_windows(monkeypatch):
    async def scenario():
        client = ContractClient(1)
        adapter = configured(
            client, embedding_request_parallelism=1, embedding_queue_size=1, timeout=5
        )
        source = WindowSource([[f"window-{index}"] for index in range(8)])
        third_finished = Event()
        original = client.embeddings

        def embeddings(input, **kwargs):
            result = original(input, **kwargs)
            if input == ["window-2"]:
                third_finished.set()
            return result

        monkeypatch.setattr(client, "embeddings", embeddings)
        stream = adapter.astream_texts(source, max_pending_windows=2)
        try:
            async with stream:
                assert await anext(stream) == [vector("window-0")]
                # Delivery releases exactly one lease: two further results may
                # finish, but a fourth source window has no admission budget.
                await wait_thread(third_finished)
                await checkpoint()
                assert source.pulls == 3
                assert await collect(stream) == [
                    [vector(f"window-{index}")] for index in range(1, 8)
                ]
        finally:
            await stream.aclose()
            await adapter.aclose()

    run(scenario)


def test_global_intake_budget_is_reserved_before_blocking_source_pulls():
    async def scenario():
        client = ContractClient(1)
        adapter = configured(
            client, embedding_request_parallelism=1, embedding_queue_size=0, timeout=5
        )
        sources = [WindowSource([[str(index)]], block_at=0) for index in range(5)]
        streams = [
            adapter.astream_texts(source, max_pending_windows=3) for source in sources
        ]
        pending = [asyncio.create_task(anext(stream)) for stream in streams]
        try:
            await asyncio.wait_for(sources[0].wait_pulls(1), 2)
            await checkpoint()
            assert sum(source.pulls for source in sources) == 1
            assert sum(source.active for source in sources) == 1
            assert not client.token_calls and not client.embedding_calls
            for task in pending:
                task.cancel()
            results = await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True), 3
            )
            assert all(isinstance(result, asyncio.CancelledError) for result in results)
            assert all(source.active == 0 for source in sources)
        finally:
            for source in sources:
                source.release.set()
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await asyncio.gather(*(stream.aclose() for stream in streams))
            await adapter.aclose()

    run(scenario)


def test_partial_window_failure_keeps_delivered_results_without_replay(monkeypatch):
    async def scenario():
        client = ContractClient(2)
        adapter = configured(
            client, embedding_request_parallelism=2, embedding_queue_size=0
        )
        source = WindowSource([["delivered"], ["part", "bad"], ["never"]])
        part_finished = Event()
        original = client.embeddings
        error = LLMBackendError("window backend sentinel")

        def embeddings(input, **kwargs):
            result = original(input, **kwargs)
            if input == ["part"]:
                part_finished.set()
            elif input == ["bad"]:
                client._wait(
                    part_finished, cancel=kwargs["cancel"], timeout=kwargs["timeout"]
                )
                raise error
            return result

        monkeypatch.setattr(client, "embeddings", embeddings)
        stream = adapter.astream_texts(source, max_pending_windows=1)
        try:
            async with stream:
                assert await anext(stream) == [vector("delivered")]
                with pytest.raises(LLMBackendError) as failure:
                    await anext(stream)
                assert failure.value is error
            assert Counter(
                text for _, batch in client.embedding_calls for text in batch
            ) == Counter(["delivered", "part", "bad"])
            assert source.position == 2
            assert_drained(adapter, client)
            assert await asyncio.to_thread(adapter.embed_query, "healthy") == vector(
                "healthy"
            )
        finally:
            part_finished.set()
            await stream.aclose()
            await adapter.aclose()

    run(scenario)


def test_source_exception_stays_original_and_owned_source_closes_once():
    async def scenario():
        client = ContractClient(1)
        adapter = configured(client)
        error = LookupError("source failure sentinel")
        source = WindowSource([["first"], ["never"]], error_at=1, error=error)
        stream = adapter.astream_texts(source, max_pending_windows=1, close_source=True)
        try:
            async with stream:
                assert await anext(stream) == [vector("first")]
                with pytest.raises(LookupError) as failure:
                    await anext(stream)
                assert failure.value is error
            await stream.aclose()
            assert source.close_calls == 1 and source.active == 0
            assert client.embedding_calls == [("embed_0", ("first",))]
            assert_drained(adapter, client)
        finally:
            await adapter.aclose()

    run(scenario)


@pytest.mark.parametrize("owned", [False, True])
def test_cancellation_of_pending_source_pull_obeys_source_ownership(owned):
    async def scenario():
        client = ContractClient(1)
        adapter = configured(client)
        source = WindowSource([["never"]], block_at=0)
        stream = adapter.astream_texts(source, close_source=owned)
        pending = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(source.wait_pulls(1), 2)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 3)
            await stream.aclose()
            await stream.aclose()
            assert source.active == 0 and source.cancelled_pulls == 1
            assert source.close_calls == int(owned)
            assert not client.token_calls and not client.embedding_calls
        finally:
            source.release.set()
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await stream.aclose()
            await adapter.aclose()

    run(scenario)


def test_repeated_task_cancellation_waits_for_actual_http_worker_drain(monkeypatch):
    async def scenario():
        client = ContractClient(1)
        client.embed_gate.clear()
        adapter = configured(client, timeout=5)
        draining, release_drain = Event(), Event()
        original = client.embeddings

        def embeddings(input, **kwargs):
            try:
                return original(input, **kwargs)
            except LLMCancelledError:
                draining.set()
                assert release_drain.wait(3)
                raise

        monkeypatch.setattr(client, "embeddings", embeddings)
        stream = adapter.astream_texts(WindowSource([["held"]]), max_pending_windows=1)
        pending = asyncio.create_task(anext(stream))
        try:
            await wait_thread(client.embed_started)
            pending.cancel()
            await wait_thread(draining)
            pending.cancel()
            await checkpoint()
            assert not pending.done()
            release_drain.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 3)
            assert_drained(adapter, client)
        finally:
            release_drain.set()
            client.embed_gate.set()
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await stream.aclose()
            await adapter.aclose()

    run(scenario)


@pytest.mark.parametrize("phase", ["source", "http"])
def test_explicit_probe_cancellation_stays_typed_and_drains(phase):
    async def scenario():
        client = ContractClient(1)
        adapter = configured(client, timeout=5)
        source = WindowSource([["held"]], block_at=0 if phase == "source" else None)
        cancel = Event()
        if phase == "http":
            client.embed_gate.clear()
        stream = adapter.astream_texts(source, cancel=cancel.is_set)
        pending = asyncio.create_task(anext(stream))
        try:
            if phase == "source":
                await asyncio.wait_for(source.wait_pulls(1), 2)
            else:
                await wait_thread(client.embed_started)
            cancel.set()
            with pytest.raises(LLMCancelledError):
                await asyncio.wait_for(pending, 3)
            assert source.active == 0
            assert_drained(adapter, client)
        finally:
            cancel.set()
            source.release.set()
            client.embed_gate.set()
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await stream.aclose()
            await adapter.aclose()

    run(scenario)


def test_early_break_closes_only_stream_and_owned_source():
    async def scenario():
        client = ContractClient(1)
        adapter = configured(client)
        source = WindowSource([["first"], ["next"], ["last"]], block_at=1)
        stream = adapter.astream_texts(source, max_pending_windows=2, close_source=True)
        try:
            async with stream:
                async for result in stream:
                    assert result == [vector("first")]
                    break
            await stream.aclose()
            assert source.close_calls == 1 and source.active == 0
            assert await asyncio.to_thread(adapter.embed_query, "healthy") == vector(
                "healthy"
            )
            assert not client.closed
        finally:
            source.release.set()
            await adapter.aclose()

    run(scenario)


def test_concurrent_streams_and_sync_caller_share_caps_and_cancel_locally():
    async def scenario():
        client = ContractClient(2)
        client.embed_gate.clear()
        adapter = configured(
            client, embedding_request_parallelism=2, embedding_queue_size=0, timeout=5
        )
        a = adapter.astream_texts(WindowSource([["async-a"]]))
        b = adapter.astream_texts(WindowSource([["async-b"]]))
        pending_a, pending_b = (
            asyncio.create_task(anext(a)),
            asyncio.create_task(anext(b)),
        )
        sync = None
        try:
            await wait_until(lambda: client.embed_active == 2)
            sync = asyncio.create_task(asyncio.to_thread(adapter.embed_texts, ["sync"]))
            await wait_until(lambda: len(adapter._calls) == 3)
            assert client.embed_peak == 2
            assert all(peak == 1 for peak in client.alias_peak.values())
            pending_a.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending_a, 3)
            assert not pending_b.done()
            client.embed_gate.set()
            assert await pending_b == [vector("async-b")]
            assert await sync == [vector("sync")]
            assert client.embed_peak <= 2
            assert all(peak <= 1 for peak in client.alias_peak.values())
            assert Counter(
                text for _, batch in client.embedding_calls for text in batch
            ) == Counter(["async-a", "async-b", "sync"])
        finally:
            client.embed_gate.set()
            for pending in [pending_a, pending_b, sync]:
                if pending is not None and not pending.done():
                    pending.cancel()
            await asyncio.gather(
                *(
                    pending
                    for pending in [pending_a, pending_b, sync]
                    if pending is not None
                ),
                return_exceptions=True,
            )
            await a.aclose()
            await b.aclose()
            await adapter.aclose()

    run(scenario)


def test_adapter_async_close_cancels_source_and_http_without_blocking_loop():
    async def scenario():
        client = ContractClient(1)
        client.embed_gate.clear()
        adapter = configured(
            client, embedding_request_parallelism=2, embedding_queue_size=0, timeout=5
        )
        source = WindowSource([["never"]], block_at=0)
        remote = adapter.astream_texts(WindowSource([["held"]]))
        blocked = adapter.astream_texts(source, close_source=True)
        jobs = [asyncio.create_task(anext(remote)), asyncio.create_task(anext(blocked))]
        try:
            await wait_thread(client.embed_started)
            await asyncio.wait_for(source.wait_pulls(1), 2)
            closing = asyncio.create_task(adapter.aclose())
            await checkpoint()
            await asyncio.wait_for(closing, 3)
            await adapter.aclose()
            outcomes = await asyncio.gather(*jobs, return_exceptions=True)
            assert all(
                isinstance(
                    value, (LLMBackendError, StopAsyncIteration, asyncio.CancelledError)
                )
                for value in outcomes
            )
            assert source.close_calls == 1 and source.active == 0
            assert_drained(adapter, client)
            with pytest.raises(LLMBackendError, match="closed"):
                await asyncio.to_thread(adapter.embed_query, "closed")
        finally:
            client.embed_gate.set()
            source.release.set()
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            await adapter.aclose()

    run(scenario)


def test_adapter_async_close_rejects_new_intake_while_owned_source_drains():
    async def scenario():
        close_entered, release_close = asyncio.Event(), asyncio.Event()

        class SlowClosingSource(WindowSource):
            async def aclose(self):
                await super().aclose()
                close_entered.set()
                await release_close.wait()

        client = ContractClient(1)
        adapter = configured(client, timeout=5)
        source = SlowClosingSource([["never"]], block_at=0)
        first = adapter.astream_texts(source, close_source=True)
        pending = asyncio.create_task(anext(first))
        closing = None
        rejected = None
        try:
            await asyncio.wait_for(source.wait_pulls(1), 2)
            closing = asyncio.create_task(adapter.aclose())
            await asyncio.wait_for(close_entered.wait(), 2)
            assert not closing.done()
            assert source.active == 0

            late_source = WindowSource([["late-async"]])
            with pytest.raises(LLMBackendError, match="closed"):
                rejected = adapter.astream_texts(late_source)
                await anext(rejected)
            with pytest.raises(LLMBackendError, match="closed"):
                await asyncio.to_thread(adapter.embed_query, "late-sync")
            assert late_source.pulls == 0
            assert not client.token_calls and not client.embedding_calls

            release_close.set()
            await asyncio.wait_for(closing, 3)
            await asyncio.gather(pending, return_exceptions=True)
            assert source.close_calls == 1
            assert_drained(adapter, client)
        finally:
            release_close.set()
            source.release.set()
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            if rejected is not None:
                await rejected.aclose()
            await adapter.aclose()

    run(scenario)


@pytest.mark.parametrize(
    "primary_kind", ["timeout", "backend", "source", "task_cancel"]
)
@pytest.mark.parametrize("use_context", [False, True])
def test_primary_error_survives_failing_owned_source_cleanup(primary_kind, use_context):
    async def scenario():
        cleanup_error = OSError("owned cleanup sentinel")

        class FailingCloseSource(WindowSource):
            async def aclose(self):
                await super().aclose()
                raise cleanup_error

        client = ContractClient(1)
        adapter = configured(client, timeout=5)
        if primary_kind == "backend":
            primary = LLMBackendError("embedding primary sentinel")
            client.failure = lambda _alias, _texts: primary
            source = FailingCloseSource([["bad"]])
            expected = LLMBackendError
        elif primary_kind == "source":
            primary = LookupError("source primary sentinel")
            source = FailingCloseSource([["bad"]], error_at=0, error=primary)
            expected = LookupError
        else:
            primary = None
            source = FailingCloseSource([["never"]], block_at=0)
            expected = (
                asyncio.CancelledError
                if primary_kind == "task_cancel"
                else LLMTimeoutError
            )
        stream = adapter.astream_texts(
            source,
            max_pending_windows=1,
            close_source=True,
            timeout=0.05 if primary_kind == "timeout" else 5,
        )

        async def consume():
            if use_context:
                async with stream:
                    return await anext(stream)
            return await anext(stream)

        pending = asyncio.create_task(consume())
        try:
            if primary_kind == "task_cancel":
                await asyncio.wait_for(source.wait_pulls(1), 2)
                pending.cancel()
            with pytest.raises(expected) as observed:
                await asyncio.wait_for(pending, 3)
            if primary is not None:
                assert observed.value is primary
            assert observed.value is not cleanup_error
            assert source.close_calls == 1 and source.active == 0
            assert_drained(adapter, client)
        finally:
            source.release.set()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            await asyncio.gather(stream.aclose(), return_exceptions=True)
            await adapter.aclose()

    run(scenario)


def test_explicit_close_without_primary_reports_owned_source_cleanup_error():
    async def scenario():
        cleanup_error = OSError("owned cleanup sentinel")

        class FailingCloseSource(WindowSource):
            async def aclose(self):
                await super().aclose()
                raise cleanup_error

        client = ContractClient(1)
        adapter = configured(client, timeout=5)
        source = FailingCloseSource([["never"]], block_at=0)
        stream = adapter.astream_texts(source, close_source=True)
        pending = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(source.wait_pulls(1), 2)
            with pytest.raises(OSError) as observed:
                await asyncio.wait_for(stream.aclose(), 3)
            assert observed.value is cleanup_error
            assert source.close_calls == 1 and source.active == 0
            await asyncio.gather(pending, return_exceptions=True)
            assert_drained(adapter, client)
        finally:
            source.release.set()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, stream.aclose(), return_exceptions=True)
            await adapter.aclose()

    run(scenario)


@pytest.mark.parametrize("termination", ["aclose", "timeout"])
def test_global_intake_lease_survives_cancellation_until_source_pull_drains(
    monkeypatch, termination
):
    async def scenario():
        draining, release_drain = asyncio.Event(), asyncio.Event()

        class SlowCancelledPull(WindowSource):
            async def __anext__(self):
                try:
                    return await super().__anext__()
                except asyncio.CancelledError:
                    draining.set()
                    await release_drain.wait()
                    raise

        client = ContractClient(1)
        adapter = configured(
            client, embedding_request_parallelism=1, embedding_queue_size=0, timeout=5
        )
        source_a = SlowCancelledPull([["never"]], block_at=0)
        source_b = WindowSource([["after-drain"]])
        a = adapter.astream_texts(
            source_a, timeout=0.05 if termination == "timeout" else 5
        )
        b = adapter.astream_texts(source_b, max_pending_windows=1)
        first = asyncio.create_task(anext(a))
        closing = second = None
        try:
            await asyncio.wait_for(source_a.wait_pulls(1), 2)
            if termination == "aclose":
                closing = asyncio.create_task(a.aclose())
            await asyncio.wait_for(draining.wait(), 2)

            # Observe arrival at global admission, independently of the source:
            # a free lease would let acquire return in that same task turn.
            attempted, admitted = asyncio.Event(), asyncio.Event()
            bridge = adapter._async_window_bridge
            acquire = bridge.acquire

            async def observed_acquire(call):
                attempted.set()
                await acquire(call)
                admitted.set()

            monkeypatch.setattr(bridge, "acquire", observed_acquire)
            second = asyncio.create_task(anext(b))
            await asyncio.wait_for(attempted.wait(), 2)
            await checkpoint()
            assert not admitted.is_set()
            assert source_b.pulls == 0 and not second.done()
            assert not first.done()

            release_drain.set()
            if closing is not None:
                await asyncio.wait_for(closing, 3)
                with pytest.raises(StopAsyncIteration):
                    await first
            else:
                with pytest.raises(LLMTimeoutError):
                    await first
            assert await asyncio.wait_for(second, 3) == [vector("after-drain")]
            assert admitted.is_set()
            assert source_a.cancelled_pulls == 1
        finally:
            release_drain.set()
            source_a.release.set()
            for job in [first, second]:
                if job is not None and not job.done():
                    job.cancel()
            await asyncio.gather(
                *(job for job in [first, second, closing] if job is not None),
                return_exceptions=True,
            )
            await a.aclose()
            await b.aclose()
            await adapter.aclose()

    run(scenario)


@pytest.mark.parametrize("change", ["topology", "identity", "context"])
def test_prepared_windows_refresh_topology_identity_and_loaded_limits(change):
    async def scenario():
        client = ContractClient(1)
        client.counts["same"] = 4
        adapter = configured(client)
        try:
            prepared = await asyncio.to_thread(adapter.prepare_query, "same")
            counted = len(client.token_calls)
            source = WindowSource([[prepared], [prepared]], block_at=1)
            async with adapter.astream_prepared(
                source, max_pending_windows=1
            ) as stream:
                assert await anext(stream) == [vector("same")]
                await asyncio.wait_for(source.wait_pulls(2), 2)
                if change == "topology":
                    client.set_aliases(["renamed"])
                elif change == "identity":
                    client.catalog[0]["tags"][0] = "noxi-content-sha256-v1:" + (
                        "c" * 64 if CONTENT != "c" * 64 else "d" * 64
                    )
                else:
                    client.contexts["embed_0"] = 2
                source.release.set()
                if change == "topology":
                    assert await anext(stream) == [vector("same")]
                    assert len(client.token_calls) == counted
                    assert [alias for alias, _ in client.embedding_calls] == [
                        "embed_0",
                        "renamed",
                    ]
                else:
                    with pytest.raises(LLMBackendError):
                        await anext(stream)
                    assert client.embedding_calls == [("embed_0", ("same",))]
        finally:
            await adapter.aclose()

    run(scenario)


def test_window_timeout_includes_source_pull_and_cleans_pending_pull():
    async def scenario():
        client = ContractClient(1)
        adapter = configured(client)
        source = WindowSource([["never"]], block_at=0)
        stream = adapter.astream_texts(source, timeout=0.05)
        try:
            with pytest.raises(LLMTimeoutError):
                await asyncio.wait_for(anext(stream), 3)
            assert source.pulls == 1 and source.active == 0
            assert not client.token_calls and not client.embedding_calls
        finally:
            source.release.set()
            await stream.aclose()
            await adapter.aclose()

    run(scenario)


def test_window_timeout_starts_before_global_admission_without_cancelling_owner():
    async def scenario():
        client = ContractClient(1)
        adapter = configured(
            client, embedding_request_parallelism=1, embedding_queue_size=0, timeout=5
        )
        first_source = WindowSource([["owner"]], block_at=0)
        second_source = WindowSource([["waiting"]])
        first = adapter.astream_texts(first_source, timeout=5)
        second = adapter.astream_texts(second_source, timeout=0.05)
        owner = asyncio.create_task(anext(first))
        try:
            await asyncio.wait_for(first_source.wait_pulls(1), 2)
            with pytest.raises(LLMTimeoutError):
                await asyncio.wait_for(anext(second), 3)
            assert second_source.pulls == 0
            assert first_source.active == 1 and not owner.done()
            owner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(owner, 3)
            assert not client.embedding_calls
        finally:
            first_source.release.set()
            if not owner.done():
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
            await first.aclose()
            await second.aclose()
            await adapter.aclose()

    run(scenario)


def test_completed_result_does_not_expire_while_consumer_is_paused(monkeypatch):
    async def scenario():
        client = ContractClient(1)
        adapter = configured(client, timeout=5)
        ready = Event()
        original = client.embeddings

        def embeddings(input, **kwargs):
            result = original(input, **kwargs)
            if input == ["second"]:
                ready.set()
            return result

        monkeypatch.setattr(client, "embeddings", embeddings)
        stream = adapter.astream_texts(WindowSource([["first"], ["second"]]), timeout=1)
        try:
            async with stream:
                assert await anext(stream) == [vector("first")]
                await wait_thread(ready)
                # Intentionally advance real time past the documented budget;
                # this tests deadline semantics, not a performance threshold.
                expired = asyncio.Event()
                timer = asyncio.get_running_loop().call_later(1.05, expired.set)
                try:
                    await asyncio.wait_for(expired.wait(), 2)
                finally:
                    timer.cancel()
                assert await anext(stream) == [vector("second")]
                with pytest.raises(StopAsyncIteration):
                    await anext(stream)
        finally:
            await stream.aclose()
            await adapter.aclose()

    run(scenario)


def test_active_adapter_rejects_a_second_event_loop_before_source_pull():
    async def scenario():
        client = ContractClient(1)
        adapter = configured(client, timeout=5)
        source = WindowSource([["first"]], block_at=0)
        foreign_source = WindowSource([["foreign"]])
        first = adapter.astream_texts(source)
        pending = asyncio.create_task(anext(first))

        def foreign_loop():
            async def attempt():
                other = adapter.astream_texts(foreign_source)
                try:
                    await anext(other)
                except (RuntimeError, LLMBackendError, ValueError) as error:
                    assert "loop" in str(error).lower()
                    return
                raise AssertionError("foreign loop was admitted")

            asyncio.run(attempt())

        try:
            await asyncio.wait_for(source.wait_pulls(1), 2)
            await asyncio.wait_for(asyncio.to_thread(foreign_loop), 3)
            assert foreign_source.pulls == 0
            assert not client.token_calls and not client.embedding_calls
        finally:
            source.release.set()
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            await first.aclose()
            await adapter.aclose()

    run(scenario)
