"""A failed later window preserves already delivered results and provenance."""

import pytest

from runtime_llama import EmbeddingInputOverflowError

from test_automatic_embedding import make_service as make_service, remote
from test_automatic_embedding_recovery import assert_partition, failing_native
from test_embedding_async_stream import WindowSource, run
from test_embedding_contracts import vector


def test_second_window_overflow_diagnoses_only_its_actual_post(make_service):
    client = remote(replicas=1, context=64)
    original = failing_native(client)
    service, _ = make_service(client, batch_size=3)
    first = ["first"]
    oversized = "α🙂漢字" * 25
    second = [oversized]
    source = WindowSource([first, second])

    async def scenario():
        async with service.astream_texts(
            source, max_pending_windows=1, close_source=True
        ) as stream:
            assert await anext(stream) == [
                vector("search_document: " + text) for text in first
            ]
            assert not client.token_calls
            with pytest.raises(EmbeddingInputOverflowError) as caught:
                await anext(stream)
            failure = caught.value
            assert failure.window_index == 1
            assert failure.failed_input_indices == (0,)
            assert failure.__cause__ is original
            assert len(failure.inputs) == 1
            assert failure.inputs[0].input_index == 0
            assert_partition(failure.inputs[0], oversized, client)
        await service.aclose()

    run(scenario)
    assert len(client.embedding_calls) == 2
    assert all(model == "embed_0" for model, _ in client.embedding_calls)
    assert client.embedding_calls[0][1] == tuple(
        "search_document: " + text for text in first
    )
    assert client.embedding_calls[1][1] == tuple(
        "search_document: " + text for text in second
    )
    # The earlier successful window is neither diagnosed nor replayed.
    measured = [text for text, _ in client.token_calls]
    assert measured[0] == "search_document: " + oversized
    assert "search_document: first" not in measured
    assert source.close_calls == 1
    assert client.token_active == client.embed_active == 0
    assert not service._operations and not service._streams
