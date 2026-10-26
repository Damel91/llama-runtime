"""Retaining a closed, never-started iterator must not retain its input source."""

import asyncio
import gc
import weakref

import pytest

from runtime_llama import AutomaticEmbeddingService
from test_embedding_contracts import ContractClient, configured


class UntouchedSource:
    def __aiter__(self):
        raise AssertionError("closing an inert stream must not acquire its source")


@pytest.mark.parametrize("kind", ["adapter", "automatic", "flat"])
def test_inert_closed_stream_releases_borrowed_source_and_service(kind):
    async def scenario():
        client = ContractClient(1)
        service = (
            AutomaticEmbeddingService(
                {
                    "base_url": "http://fixture.example",
                    "embedding_prefixes": {"document": "", "query": ""},
                },
                client=client,
            )
            if kind == "automatic"
            else configured(client)
        )
        source = UntouchedSource()
        reference = weakref.ref(source)
        stream = (
            service.aiter_texts(source)
            if kind == "flat"
            else service.astream_texts(source)
        )
        del source
        try:
            await stream.aclose()
            await stream.aclose()
            gc.collect()
            assert reference() is None
            assert stream._closed
            assert not client.token_calls and not client.embedding_calls
        finally:
            await service.aclose()

    asyncio.run(scenario())
