"""Model-level packing across small source windows without a fill timer."""

import asyncio
from collections import Counter

import pytest

from runtime_llama.embedding_scheduler import EmbeddingScheduler
from runtime_llama.errors import LLMBackendError

from test_embedding_async_stream import WindowSource, run, wait_thread, wait_until
from test_embedding_contracts import configured, ContractClient, texts, vector


@pytest.mark.parametrize(
    ("sizes", "replicas", "parallelism", "budget", "expected"),
    [
        ([64] * 4, 2, 2, 256, [128, 128]),
        ([64] * 7 + [63], 2, 2, 256, [127, 128, 128, 128]),
        ([1] * 8, 4, 4, 8, [2, 2, 2, 2]),
    ],
)
def test_small_source_windows_fill_the_configured_model_budget(
    sizes, replicas, parallelism, budget, expected
):
    async def scenario():
        client = ContractClient(replicas)
        client.token_gate.clear()
        client.embed_gate.clear()
        adapter = configured(
            client,
            batch_size=budget,
            dimension=2,
            embedding_batch_wait_sec=0,
            embedding_request_parallelism=parallelism,
            embedding_queue_size=len(sizes) - parallelism,
            token_count_parallelism=parallelism,
            timeout=8,
        )
        # A frozen coalescing clock demonstrates that filling advances through
        # preparation/capacity events, rather than an elapsed fill interval.
        adapter._batch_now = lambda: 0.0
        incoming = texts(sum(sizes))
        windows, offset = [], 0
        for size in sizes:
            windows.append(incoming[offset : offset + size])
            offset += size
        source = WindowSource(windows)
        stream = adapter.astream_texts(source, max_pending_windows=len(sizes))
        first = asyncio.create_task(anext(stream))
        try:
            await wait_thread(client.token_started)
            await wait_until(
                lambda: (
                    len(adapter._engine._states) == len(sizes)
                    and all(
                        state.context is not None for state in adapter._engine._states
                    )
                )
            )
            assert not client.embedding_calls
            client.token_gate.set()
            await wait_until(lambda: client.embed_active == parallelism)
            await wait_until(lambda: len(client.token_calls) == len(incoming))
            await wait_until(
                lambda: adapter._engine.snapshot()["tokenization_in_flight"] == 0
            )
            snapshot = adapter._engine.snapshot()
            cap = adapter._scheduler.text_cap(budget)
            assert all(len(batch) == cap for _, batch in client.embedding_calls)
            assert (
                snapshot["peak_active_plans"]
                <= parallelism + adapter.embedding_queue_size
            )
            assert snapshot["retained_plans"] >= len(sizes)
            assert snapshot["pending_bytes"] > 0
            assert adapter._scheduler.snapshot()["queued"] == 0
            client.embed_gate.set()
            outputs = [await asyncio.wait_for(first, 3)]
            outputs.extend([window async for window in stream])
            assert outputs == [[vector(text) for text in window] for window in windows]
            assert sorted(len(batch) for _, batch in client.embedding_calls) == expected
            assert Counter(
                text for _, batch in client.embedding_calls for text in batch
            ) == Counter(incoming)
            assert client.embed_peak == parallelism
            assert adapter._engine.snapshot()["retained_plans"] == 0
        finally:
            client.token_gate.set()
            client.embed_gate.set()
            await asyncio.gather(first, return_exceptions=True)
            await stream.aclose()
            await adapter.aclose()

    run(scenario)


@pytest.mark.parametrize("pressure", ["bytes", "tokens"])
def test_resource_pressure_flushes_partial_model_batch_without_a_timer(pressure):
    client = ContractClient(1)
    incoming = texts(8)
    client.counts.update(dict.fromkeys(incoming, 2))
    limits = (
        {"embedding_max_pending_bytes": 1024}
        if pressure == "bytes"
        else {"embedding_max_pending_tokens": 3}
    )
    adapter = configured(
        client,
        batch_size=8,
        embedding_request_parallelism=1,
        embedding_queue_size=0,
        embedding_batch_wait_sec=0,
        **limits,
    )
    adapter._batch_now = lambda: 0.0
    try:
        assert adapter.embed_texts(incoming) == [vector(text) for text in incoming]
        assert len(client.embedding_calls) > 1
        assert all(len(batch) < 8 for _, batch in client.embedding_calls)
        snapshot = adapter._engine.snapshot()
        assert snapshot["peak_pending_bytes"] <= adapter.embedding_max_pending_bytes
        assert snapshot["peak_pending_tokens"] <= adapter.embedding_max_pending_tokens
        assert Counter(
            text for _, batch in client.embedding_calls for text in batch
        ) == Counter(incoming)
    finally:
        adapter.close()


def test_explicit_token_budget_splits_a_batch_gathered_across_owners():
    async def scenario():
        client = ContractClient(1)
        client.token_gate.clear()
        # Unequal windows require cross-window fill even when packing
        # completes the oldest ready window before taking the next one.
        windows = [
            [f"w{owner}-{index}" for index in range(size)]
            for owner, size in enumerate((1, 3, 1, 3))
        ]
        incoming = [text for window in windows for text in window]
        client.counts.update(dict.fromkeys(incoming, 2))
        adapter = configured(
            client,
            batch_size=8,
            dimension=2,
            batch_token_limit=4,
            embedding_request_parallelism=1,
            embedding_queue_size=3,
            embedding_batch_wait_sec=0,
        )
        stream = adapter.astream_texts(WindowSource(windows), max_pending_windows=4)
        first = asyncio.create_task(anext(stream))
        try:
            await wait_thread(client.token_started)
            await wait_until(
                lambda: (
                    len(adapter._engine._states) == 4
                    and all(
                        state.context is not None for state in adapter._engine._states
                    )
                )
            )
            client.token_gate.set()
            outputs = [await first]
            outputs.extend([window async for window in stream])
            assert outputs == [[vector(text) for text in window] for window in windows]
            assert [len(batch) for _, batch in client.embedding_calls] == [2] * 4
            assert any(
                len({text[:2] for text in batch}) > 1
                for _, batch in client.embedding_calls
            )
        finally:
            client.token_gate.set()
            await asyncio.gather(first, return_exceptions=True)
            await stream.aclose()
            await adapter.aclose()

    run(scenario)


def test_text_cap_is_pure_and_small_windows_keep_their_distribution_credits():
    scheduler = EmbeddingScheduler(2, 0, {"a": 1, "b": 1, "c": 1})
    try:
        assert scheduler.text_cap(256) == 128
        assert scheduler.text_cap(3) == 2
        assert scheduler.text_cap(1) == 1
        # Reading the configured ceiling cannot consume the rotation used to
        # distribute actual source windows over the verified destinations.
        assert scheduler.plan_window(1) == (("a", 1),)
        assert scheduler.text_cap(256) == 128
        assert scheduler.plan_window(1) == (("b", 1),)
    finally:
        scheduler.close()


def test_worker_rejected_owner_releases_its_unsent_backlog_after_shared_drain():
    async def scenario():
        client = ContractClient(1)
        client.embed_gate.clear()
        first_inputs, second_inputs = texts(10), ["b0", "b1"]
        client.counts.update(dict.fromkeys(first_inputs, 4))
        client.counts.update(dict.fromkeys(["prime", *second_inputs, "ok"], 2))
        adapter = configured(
            client,
            batch_size=4,
            dimension=2,
            embedding_request_parallelism=1,
            embedding_queue_size=1,
            embedding_batch_wait_sec=0,
        )
        prime = asyncio.create_task(asyncio.to_thread(adapter.embed_query, "prime"))
        pending = []
        try:
            await wait_thread(client.embed_started)
            pending = [
                asyncio.create_task(asyncio.to_thread(adapter.embed_texts, inputs))
                for inputs in (first_inputs, second_inputs)
            ]
            await wait_until(lambda: len(client.token_calls) == 13)
            await wait_until(
                lambda: (
                    adapter._engine.snapshot()["ready_by_replica"].get("embed_0") == 12
                )
            )
            # Only A's prepared rows exceed the newly observed input limit.
            # The shared sender fails A on its worker before POSTing B's rows.
            client.contexts["embed_0"] = 2
            client.embed_gate.set()
            assert await prime == vector("prime")
            with pytest.raises(LLMBackendError, match="exceeds"):
                await pending[0]
            assert await pending[1] == [vector(text) for text in second_inputs]
            assert client.embedding_calls == [
                ("embed_0", ("prime",)),
                ("embed_0", tuple(second_inputs)),
            ]
            snapshot = adapter._engine.snapshot()
            assert snapshot["pending_bytes"] == snapshot["pending_tokens"] == 0
            assert (
                snapshot["retained_plans"] == snapshot["active_preparation_plans"] == 0
            )
            assert adapter._engine._descriptor_count == 0
            assert await asyncio.to_thread(adapter.embed_query, "ok") == vector("ok")
        finally:
            client.embed_gate.set()
            await asyncio.gather(prime, *pending, return_exceptions=True)
            await adapter.aclose()

    run(scenario)
