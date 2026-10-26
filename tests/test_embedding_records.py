"""Flat progressive input, correlation and facade ownership without HTTP."""

import asyncio
import gc
import weakref

import pytest

from runtime_llama.client import ClientHTTPError
from runtime_llama.embedding_halving import (
    EmbeddingChunkSplitRequiredError,
    _rejected_embedding,
)
from runtime_llama.embedding_records import (
    EmbeddingRecordResult,
    stream_text_records,
)


class Vector(list):
    pass


class Source:
    def __init__(self, values):
        self.values = iter(values)
        self.closed = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.values)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self):
        self.closed += 1


class Windows:
    def __init__(self, service, windows):
        self.service = service
        self.source = aiter(windows)
        self._closed = False

    def __aiter__(self):
        return self

    async def __aenter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        values = await anext(self.source)
        assert len(values) == 1
        text = values[0]
        if text == "rejected" and self.service.rejection is not None:
            raise self.service.rejection
        vector = Vector([float(len(text)), 1.0])
        self.service.vectors.append(weakref.ref(vector))
        return [vector]

    async def aclose(self):
        if self._closed:
            return
        self._closed = True
        self.service.close_started.set()
        if self.service.close_gate is not None:
            await self.service.close_gate.wait()
        try:
            await self.source.aclose()
            if self.service.close_error is not None:
                raise self.service.close_error
        finally:
            self.source = None
            self.service.closed.set()


class Service:
    def __init__(self):
        self.requests = []
        self.vectors = []
        self.closed = asyncio.Event()
        self.close_started = asyncio.Event()
        self.close_gate = self.close_error = self.rejection = None

    def astream_texts(self, windows, **kwargs):
        self.requests.append(kwargs)
        return Windows(self, windows)


def test_flat_factory_is_inert_and_does_not_impose_a_pending_count():
    service = Service()
    source = Source(["one"])
    stream = stream_text_records(service, source)
    assert service.requests == []

    async def scenario():
        async with stream:
            assert await anext(stream) == EmbeddingRecordResult(0, [3.0, 1.0])
        assert service.requests[0]["max_pending_windows"] is None
        assert service.requests[0]["close_source"] is True
        # The internally owned windows close, while the original source stays
        # borrowed unless the caller explicitly transfers its close ownership.
        assert source.closed == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("value", ["text", b"text", object(), 1])
def test_flat_factory_rejects_non_sources_without_starting_work(value):
    service = Service()
    with pytest.raises(ValueError):
        stream_text_records(service, value)
    assert not service.requests


def test_ten_thousand_occurrences_transfer_without_retaining_past_vectors():
    async def scenario():
        service = Service()
        source = Source("same" if index % 2 else "other" for index in range(10000))
        source_ref = weakref.ref(source)
        stream = stream_text_records(service, source, close_source=True)
        del source
        seen = 0
        async with stream:
            async for result in stream:
                assert result.index == seen
                assert result.embedding == ([5.0, 1.0] if seen % 2 == 0 else [4.0, 1.0])
                seen += 1
        assert seen == 10000  # Test size, not a maximum of the API.
        del result
        gc.collect()
        assert source_ref() is None
        assert all(ref() is None for ref in service.vectors)
        assert stream._service is stream._texts is stream._stream is None
        assert stream._cleanup_task is None

    asyncio.run(scenario())


def test_first_result_does_not_wait_for_async_source_exhaustion():
    async def scenario():
        service = Service()
        more = asyncio.Event()
        exhausted = False

        async def source():
            nonlocal exhausted
            yield "first"
            await more.wait()
            yield "second"
            exhausted = True

        async with stream_text_records(service, source(), close_source=True) as stream:
            assert (await anext(stream)).index == 0
            assert not exhausted
            more.set()
            assert (await anext(stream)).index == 1
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
        assert exhausted

    asyncio.run(scenario())


def test_typed_recovery_keeps_global_occurrence_and_continues():
    async def scenario():
        service = Service()
        marker = _rejected_embedding("rejected", "document", ClientHTTPError(500))
        error = EmbeddingChunkSplitRequiredError(
            input_count=1,
            completed=(),
            inputs=(marker.for_input(0),),
            representation_id="test-representation",
            input_type="document",
        )
        service.rejection = error
        async with stream_text_records(service, ["one", "rejected", "three"]) as stream:
            assert (await anext(stream)).index == 0
            with pytest.raises(EmbeddingChunkSplitRequiredError) as caught:
                await anext(stream)
            assert caught.value is error
            assert error.window_index == 1
            assert error.failed_input_indices == (0,)
            assert error.inputs[0].halves == marker.halves
            assert (await anext(stream)).index == 2

    asyncio.run(scenario())


def test_two_close_waiters_both_wait_for_the_owned_cleanup():
    async def scenario():
        service = Service()
        service.close_gate = asyncio.Event()
        source = Source(["one", "two"])
        stream = stream_text_records(service, source, close_source=True)
        await anext(stream)
        first = asyncio.create_task(stream.aclose())
        await service.close_started.wait()
        second = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
        service.close_gate.set()
        await asyncio.gather(first, second)
        assert source.closed == 1
        assert stream._cleanup_task is None

    asyncio.run(scenario())


def test_dropped_facade_schedules_cleanup_without_retaining_itself():
    async def scenario():
        service = Service()
        service.close_gate = asyncio.Event()
        source = Source(["one", "two"])
        source_ref = weakref.ref(source)
        stream = stream_text_records(service, source, close_source=True)
        del source
        await anext(stream)
        stream_ref = weakref.ref(stream)
        del stream
        gc.collect()
        assert stream_ref() is None
        await asyncio.wait_for(service.close_started.wait(), 2)
        assert source_ref() is not None  # Owned cleanup has not drained yet.
        service.close_gate.set()
        await asyncio.wait_for(service.closed.wait(), 2)
        await asyncio.sleep(0)
        gc.collect()
        assert source_ref() is None

    asyncio.run(scenario())


def test_consumer_exception_is_preserved_when_close_also_fails():
    async def scenario():
        service = Service()
        service.close_error = RuntimeError("cleanup sentinel")
        primary = ValueError("consumer sentinel")
        stream = stream_text_records(service, ["one"])
        with pytest.raises(ValueError) as caught:
            async with stream:
                await anext(stream)
                raise primary
        assert caught.value is primary
        assert service.closed.is_set()
        assert stream._stream is stream._service is None

    asyncio.run(scenario())


def test_source_validation_failure_closes_and_releases_owned_source():
    async def scenario():
        service = Service()
        source = Source(["valid", object()])
        stream = stream_text_records(service, source, close_source=True)
        await anext(stream)
        with pytest.raises(ValueError):
            await anext(stream)
        assert source.closed == 1
        assert service.closed.is_set()
        assert stream._stream is stream._service is None

    asyncio.run(scenario())
