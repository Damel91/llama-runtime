"""A stream's packing identity never replaces its window ownership."""

import asyncio
from threading import Event
from types import SimpleNamespace

import pytest

from runtime_llama.automatic_embedding import AutomaticEmbeddingService
from runtime_llama.embedding_stream import EmbeddingWindowStream
from runtime_llama.errors import LLMCancelledError

from test_automatic_embedding import remote
from test_embedding_async_stream import WindowSource, run, wait_until
from test_embedding_contracts import ContractClient, configured, vector


def observe_submissions(adapter, monkeypatch):
    submissions = []
    original = adapter._stream_submit

    def submit(window, *, call, **kwargs):
        future = original(window, call=call, **kwargs)
        submissions.append((call, future))
        return future

    monkeypatch.setattr(adapter, "_stream_submit", submit)
    return submissions


@pytest.mark.parametrize("kind", ["exact", "prepared", "direct"])
def test_constructor_creates_only_an_inert_unique_packing_identity(kind):
    adapter = SimpleNamespace(profile=SimpleNamespace(timeout=5))
    first_source, second_source = WindowSource([["one"]]), WindowSource([["two"]])
    policy = object() if kind == "direct" else None
    scope = object() if kind == "direct" else None
    first = EmbeddingWindowStream(
        adapter,
        first_source,
        prepared=kind == "prepared",
        _direct_policy=policy,
        _admission_scope=scope,
    )
    second = EmbeddingWindowStream(
        adapter,
        second_source,
        prepared=kind == "prepared",
        _direct_policy=policy,
        _admission_scope=scope,
    )
    assert type(first._packing_group) is object
    assert first._packing_group is not second._packing_group
    assert first._packing_group is not scope
    assert first_source.pulls == second_source.pulls == 0
    assert first._producer is second._producer is None


@pytest.mark.parametrize("kind", ["exact", "prepared", "direct"])
def test_each_window_has_one_shared_group_but_its_own_call_and_future(
    kind, monkeypatch
):
    async def scenario():
        client = remote(replicas=1) if kind == "direct" else ContractClient(1)
        service = None
        if kind == "direct":
            service = AutomaticEmbeddingService(
                {"base_url": "http://runtime.example:8080", "normalize": False},
                client=client,
            )
            await asyncio.to_thread(service.initialize, input_type="raw")
            adapter = service._adapter
        else:
            adapter = configured(client, batch_size=4, dimension=2)
        submissions = observe_submissions(adapter, monkeypatch)
        windows = [["one", "two"], [], ["three"]]
        if kind == "prepared":
            prepared = [
                await asyncio.to_thread(adapter.prepare_texts, window, input_type="raw")
                for window in windows
            ]
            stream = adapter.astream_prepared(WindowSource(prepared))
        elif kind == "direct":
            stream = service.astream_texts(WindowSource(windows), input_type="raw")
        else:
            stream = adapter.astream_texts(WindowSource(windows), input_type="raw")
        try:
            async with stream:
                delegate = stream._delegate if kind == "direct" else stream
                packing_group = delegate._packing_group
                result = [window async for window in stream]
            assert result == [[vector(text) for text in window] for window in windows]
            calls = [call for call, _ in submissions]
            futures = [future for _, future in submissions]
            assert len(calls) == len(windows)
            assert len(set(calls)) == len({id(call.stop) for call in calls}) == 3
            assert len(set(futures)) == 3 and all(future.done() for future in futures)
            assert [call._window_index for call in calls] == [0, 1, 2]
            assert all(call._packing_group is packing_group for call in calls)
            assert delegate._packing_group is None
            assert delegate._adapter is delegate._bridge is None
            if kind == "direct":
                assert stream._delegate is None
            if kind == "direct":
                assert all(
                    call._packing_group is not call._admission_scope for call in calls
                )
            else:
                assert all(not hasattr(call, "_admission_scope") for call in calls)
            assert not adapter._calls and adapter._pool_users == 0
        finally:
            await stream.aclose()
            await (service.aclose() if service is not None else adapter.aclose())

    run(scenario)


def test_mixed_streams_share_adapter_but_never_a_packing_identity(monkeypatch):
    async def scenario():
        client = remote(replicas=2)
        service = AutomaticEmbeddingService(
            {
                "base_url": "http://runtime.example:8080",
                "normalize": False,
                "batch_size": 4,
            },
            client=client,
        )
        await asyncio.to_thread(service.initialize, input_type="raw")
        adapter = service._adapter
        submissions = observe_submissions(adapter, monkeypatch)
        exact = adapter.astream_texts(
            WindowSource([["exact0"], ["exact1"]]), input_type="raw"
        )
        direct = service.astream_texts(
            WindowSource([["direct0"], ["direct1"]]), input_type="raw"
        )

        async def consume(stream):
            async with stream:
                delegate = stream._delegate if stream is direct else stream
                group = delegate._packing_group
                return [window async for window in stream], group

        try:
            outputs = await asyncio.gather(consume(exact), consume(direct))
            assert [result for result, _ in outputs] == [
                [[vector("exact0")], [vector("exact1")]],
                [[vector("direct0")], [vector("direct1")]],
            ]
            groups = {call._packing_group for call, _ in submissions}
            assert groups == {group for _, group in outputs}
            assert exact._packing_group is None and direct._delegate is None
            assert len(groups) == 2
            assert len({call for call, _ in submissions}) == 4
            assert not adapter._calls and adapter._pool_users == 0
        finally:
            await exact.aclose()
            await direct.aclose()
            await service.aclose()

    run(scenario)


def test_cancelling_one_group_preserves_other_stream_call_ownership(monkeypatch):
    async def scenario():
        client = ContractClient(1)
        client.embed_gate.clear()
        adapter = configured(
            client,
            batch_size=4,
            dimension=2,
            embedding_batch_wait_sec=0,
            embedding_queue_size=4,
            timeout=5,
        )
        submissions = observe_submissions(adapter, monkeypatch)
        cancelled = Event()
        first_source = WindowSource([["cancel0"], ["cancel1"]])
        second_source = WindowSource([["survive0"], ["survive1"]])
        first = adapter.astream_texts(
            first_source,
            input_type="raw",
            cancel=cancelled.is_set,
            close_source=True,
        )
        second = adapter.astream_texts(
            second_source, input_type="raw", close_source=True
        )
        failed = asyncio.create_task(anext(first))

        async def consume_survivor():
            async with second:
                return [window async for window in second]

        survivor = asyncio.create_task(consume_survivor())
        try:
            await wait_until(lambda: len(submissions) == 4)
            cancelled.set()
            await wait_until(
                lambda: any(
                    call.failure is not None
                    for call, _ in submissions
                    if call._packing_group is first._packing_group
                )
            )
            others = [
                call
                for call, _ in submissions
                if call._packing_group is second._packing_group
            ]
            assert len(others) == 2
            assert all(
                not call.stop.is_set() and call.failure is None for call in others
            )
            client.embed_gate.set()
            with pytest.raises(LLMCancelledError):
                await failed
            assert await survivor == [[vector("survive0")], [vector("survive1")]]
            assert client.embed_active == adapter._pool_users == 0
            assert not adapter._calls and first_source.close_calls == 1
            assert second_source.close_calls == 1
        finally:
            client.embed_gate.set()
            await first.aclose()
            await second.aclose()
            await asyncio.gather(failed, survivor, return_exceptions=True)
            await adapter.aclose()

    run(scenario)
