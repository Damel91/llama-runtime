"""Recoverable later windows preserve completed and prefetched window order."""

import asyncio

import pytest

from runtime_llama import EmbeddingChunkSplitRequiredError

from test_automatic_embedding import make_service as make_service, remote
from test_automatic_embedding_recovery import assert_halves, failing_native
from test_embedding_async_stream import WindowSource, run, wait_thread
from test_embedding_contracts import vector


def test_second_window_rejection_preserves_first_and_allows_later_window(
    make_service,
):
    client = remote(replicas=1, context=64)
    original = failing_native(client)
    service, _ = make_service(client, batch_size=3)
    first, second, third = ["first"], ["α🙂漢字" * 20], ["third"]
    source = WindowSource([first, second, third])

    async def scenario():
        async with service.astream_texts(
            source, max_pending_windows=1, close_source=True
        ) as stream:
            assert await anext(stream) == [vector("search_document: first")]
            with pytest.raises(EmbeddingChunkSplitRequiredError) as caught:
                await anext(stream)
            failure = caught.value
            assert failure.window_index == 1
            assert failure.failed_input_indices == (0,)
            assert failure.__cause__.status_code == original.status_code
            assert failure.__cause__.error_type == original.error_type
            assert not failure.completed
            assert_halves(failure.inputs[0], second[0])
            assert await anext(stream) == [vector("search_document: third")]
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
        await service.aclose()

    run(scenario)
    assert len(client.embedding_calls) == 3
    assert [rows for _, rows in client.embedding_calls] == [
        tuple("search_document: " + text for text in window)
        for window in (first, second, third)
    ]
    assert not client.token_calls
    assert source.close_calls == 1
    assert client.token_active == client.embed_active == 0
    assert not service._operations and not service._streams


def test_buffered_earlier_results_are_delivered_before_recoverable_later_error(
    make_service,
):
    client = remote(replicas=1, context=64)
    failing_native(client)
    client.embed_gate.clear()
    service, _ = make_service(client, batch_size=3)
    source = WindowSource([["first"], ["x" * 80, "second fit"], ["last"]])

    async def scenario():
        async with service.astream_texts(
            source, max_pending_windows=3, close_source=True
        ) as stream:
            waiting = asyncio.create_task(anext(stream))
            await wait_thread(client.embed_started)
            await source.wait_pulls(3)
            client.embed_gate.set()
            assert await waiting == [vector("search_document: first")]
            with pytest.raises(EmbeddingChunkSplitRequiredError) as caught:
                await anext(stream)
            failure = caught.value
            assert failure.window_index == 1 and failure.failed_input_indices == (0,)
            assert [(row.input_index, row.embedding) for row in failure.completed] == [
                (1, tuple(vector("search_document: second fit")))
            ]
            assert_halves(failure.inputs[0], "x" * 80)
            assert await anext(stream) == [vector("search_document: last")]
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
        await service.aclose()

    run(scenario)
    assert not client.token_calls and not service.config.cache_dir.exists()
    assert source.close_calls == 1
    assert client.token_active == client.embed_active == 0
    assert not service._operations and not service._streams
