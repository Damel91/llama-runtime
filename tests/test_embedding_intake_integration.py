"""Mixed sync/async owners enter one fixed budget before copying or pulling."""

import asyncio
from collections.abc import Sequence

import pytest

from runtime_llama.embedding_resources import EmbeddingResourceLimitError

from test_embedding_async_stream import WindowSource, wait_until
from test_embedding_contracts import ContractClient, configured, vector


def test_sync_owner_prevents_async_source_pull_until_result_transfer():
    async def scenario():
        client = ContractClient(1)
        client.embed_gate.clear()
        adapter = configured(
            client,
            embedding_request_parallelism=1,
            embedding_queue_size=0,
            embedding_max_pending_owners=1,
            embedding_batch_wait_sec=0,
            timeout=5,
        )
        source = WindowSource([["async"]])
        stream = adapter.astream_texts(source)
        sync = asyncio.create_task(asyncio.to_thread(adapter.embed_texts, ["sync"]))
        pending = None
        try:
            await wait_until(lambda: client.embed_active == 1)
            pending = asyncio.create_task(anext(stream))
            await wait_until(lambda: adapter._intake.snapshot()["owner_waiters"] == 1)
            assert source.pulls == 0
            assert len(adapter._calls) == 1
            assert adapter._intake.snapshot()["owners"] == 1
            client.embed_gate.set()
            assert await asyncio.wait_for(sync, 3) == [vector("sync")]
            assert await asyncio.wait_for(pending, 3) == [vector("async")]
            await stream.aclose()
            view = adapter._intake.snapshot()
            assert view["peak_owners"] == 1
            assert view["owners"] == view["snapshot_bytes"] == 0
        finally:
            client.embed_gate.set()
            for task in (sync, pending):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (sync, pending) if task is not None),
                return_exceptions=True,
            )
            await stream.aclose()
            await adapter.aclose()

    asyncio.run(scenario())


def test_completed_async_result_keeps_sync_caller_outside_registration():
    async def scenario():
        client = ContractClient(1)
        adapter = configured(
            client,
            embedding_request_parallelism=1,
            embedding_queue_size=0,
            embedding_max_pending_owners=1,
            embedding_batch_wait_sec=0,
            timeout=5,
        )
        stream = adapter.astream_texts(WindowSource([["held"]]), max_pending_windows=1)
        sync = None
        try:
            await stream.__aenter__()
            await wait_until(
                lambda: bool(stream._pending) and stream._pending[0].future.done()
            )
            assert not adapter._calls
            assert adapter._intake.snapshot()["owners"] == 1
            sync = asyncio.create_task(
                asyncio.to_thread(adapter.embed_texts, ["waiting"])
            )
            await wait_until(lambda: adapter._intake.snapshot()["owner_waiters"] == 1)
            assert not adapter._calls
            assert len(client.embedding_calls) == 1
            assert await anext(stream) == [vector("held")]
            assert await asyncio.wait_for(sync, 3) == [vector("waiting")]
            await stream.aclose()
            assert adapter._intake.snapshot()["owners"] == 0
        finally:
            if sync is not None:
                await asyncio.gather(sync, return_exceptions=True)
            await stream.aclose()
            await adapter.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["embed_texts", "prepare_texts", "embed_prepared"])
def test_impossible_sync_snapshot_fails_before_source_reads_or_registration(method):
    class ImpossibleWindow(Sequence):
        def __len__(self):
            return 10**9

        def __getitem__(self, index):
            raise AssertionError("input copied before its finite bound was checked")

    client = ContractClient(1)
    adapter = configured(client, embedding_max_pending_bytes=512)
    try:
        with pytest.raises(EmbeddingResourceLimitError, match="snapshot_bytes_limit"):
            getattr(adapter, method)(ImpossibleWindow())
        assert not adapter._calls
        assert not client.token_calls and not client.embedding_calls
        assert adapter._intake.snapshot()["owners"] == 0
        assert adapter._intake.snapshot()["snapshot_bytes"] == 0
    finally:
        adapter.close()


def test_oversized_async_snapshot_does_not_register_or_send_embedding():
    async def scenario():
        client = ContractClient(1)
        adapter = configured(client, embedding_max_pending_bytes=512)
        stream = adapter.astream_texts(WindowSource([["large" * 1000]]))
        try:
            async with stream:
                with pytest.raises(
                    EmbeddingResourceLimitError, match="snapshot_bytes_limit"
                ):
                    await anext(stream)
            assert not adapter._calls
            assert not client.token_calls and not client.embedding_calls
            assert adapter._intake.snapshot()["owners"] == 0
            assert adapter._intake.snapshot()["snapshot_bytes"] == 0
            assert adapter._async_window_bridge.used == 0
        finally:
            await stream.aclose()
            await adapter.aclose()

    asyncio.run(scenario())
