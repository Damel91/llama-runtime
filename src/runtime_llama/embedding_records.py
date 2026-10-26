"""Progressive indexed text results over the existing embedding window engine.

The adapter still owns discovery, preparation, batching and HTTP scheduling.
Singleton source windows are correlation units, not an HTTP batch-size cap:
compatible records are packed across these windows by the shared engine.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Sequence
from dataclasses import dataclass

from runtime_llama.embedding_halving import EmbeddingChunkSplitRequiredError
from runtime_llama.embedding_stream import _drain
from runtime_llama.errors import LLMBackendError


@dataclass(frozen=True, slots=True, weakref_slot=True)
class EmbeddingRecordResult:
    """One source occurrence; its vector belongs to the consumer on transfer."""

    index: int
    embedding: list[float]


async def _text_windows(texts, *, close_source):
    """Own only an acquired text, without a reference to the public facade."""
    iterator = None
    try:
        if isinstance(texts, Sequence):
            iterator = iter(texts)
            for text in iterator:
                if not isinstance(text, str):
                    raise ValueError("texts must contain only strings")
                yield (text,)
        else:
            iterator = aiter(texts)
            while True:
                try:
                    text = await anext(iterator)
                except StopAsyncIteration:
                    break
                if not isinstance(text, str):
                    raise ValueError("texts must contain only strings")
                yield (text,)
    finally:
        if close_source:
            close = getattr(iterator, "aclose", None)
            if callable(close):
                await close()


def _signal_stop(stream):
    """Withdraw actual work before the async cleanup task gets another turn."""
    stop = getattr(stream, "_stop", None)
    if stop is not None:
        stop.set()
    inner = getattr(stream, "_delegate", None) or getattr(stream, "_stream", None)
    request = getattr(inner if inner is not None else stream, "_request_stop", None)
    if callable(request):
        request()


def _observe_cleanup(task):
    if not task.cancelled():
        task.exception()


def _schedule_cleanup(stream):
    task = asyncio.create_task(_drain(stream.aclose()))
    task.add_done_callback(_observe_cleanup)


class EmbeddingRecordStream:
    """Source-ordered vectors from flat text input, with no pending-count default.

    Use an async context manager or ``aclose`` for deterministic early Stop.
    A discarded facade also signals Stop and schedules cleanup on its live
    owning loop. Cleanup retains owned work until the transport actually drains;
    an already closed loop cannot execute asynchronous cleanup.

    A recoverable ``EmbeddingChunkSplitRequiredError`` is delivered unchanged.
    Its ``window_index`` identifies the global record index; its inner input
    indices remain relative to the singleton source window. The next iteration
    can continue with later occurrences, as on the ordinary automatic stream.
    """

    def __init__(
        self,
        service,
        texts: Sequence[str] | AsyncIterable[str],
        *,
        input_type="document",
        timeout=None,
        cancel=None,
        close_source=False,
    ):
        if isinstance(texts, (str, bytes)) or not (
            isinstance(texts, Sequence) or callable(getattr(texts, "__aiter__", None))
        ):
            raise ValueError("texts must be a finite sequence or async iterable")
        if not isinstance(close_source, bool):
            raise ValueError("close_source must be a boolean")
        self._service, self._texts = service, texts
        self._options = {
            "input_type": input_type,
            "timeout": timeout,
            "cancel": cancel,
        }
        self._close_source = close_source
        self._stream = self._loop = None
        self._cleanup_task = None
        self._next_index = 0
        self._active = self._closed = False

    def __aiter__(self):
        return self

    async def _start(self):
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise LLMBackendError("llamacpp_embedding_async_event_loop_mismatch")
        if self._closed or self._stream is not None:
            return
        self._loop = loop
        # The producer owns this standalone iterator. It never refers back to
        # the facade, allowing a dropped facade to stop an in-flight producer.
        windows = _text_windows(self._texts, close_source=self._close_source)
        try:
            self._stream = self._service.astream_texts(
                windows,
                **self._options,
                max_pending_windows=None,
                close_source=True,
            )
        except BaseException:
            await windows.aclose()
            raise
        self._texts = None

    async def __anext__(self) -> EmbeddingRecordResult:
        if self._closed:
            raise StopAsyncIteration
        if self._active:
            raise RuntimeError("concurrent iteration of one embedding stream")
        self._active = True
        try:
            await self._start()
            try:
                rows = await anext(self._stream)
            except EmbeddingChunkSplitRequiredError as exc:
                # Keep the typed recovery payload and its relative source
                # spans. Singleton windows make the global correlation exact.
                exc.window_index = self._next_index
                self._next_index += 1
                raise
            except StopAsyncIteration:
                await self.aclose()
                raise
            if not isinstance(rows, list) or len(rows) != 1:
                raise LLMBackendError("llamacpp_embedding_count_or_response_mismatch")
            result = EmbeddingRecordResult(self._next_index, rows[0])
            self._next_index += 1
            return result
        except EmbeddingChunkSplitRequiredError:
            raise
        except BaseException as exc:
            await _drain(self.aclose(), primary=exc)
            raise
        finally:
            self._active = False

    async def __aenter__(self):
        try:
            await self._start()
            if self._stream is not None:
                enter = getattr(self._stream, "__aenter__", None)
                if callable(enter):
                    await enter()
            return self
        except BaseException as exc:
            await _drain(self.aclose(), primary=exc)
            raise

    async def __aexit__(self, exc_type, exc, traceback):
        if exc is not None:
            await _drain(self.aclose(), primary=exc)
        else:
            await self.aclose()

    async def aclose(self):
        if self._cleanup_task is None:
            if self._closed:
                return
            self._closed = True
            self._cleanup_task = asyncio.create_task(self._cleanup())
        task = self._cleanup_task
        try:
            await _drain(task)
        finally:
            if task.done() and self._cleanup_task is task:
                self._cleanup_task = None

    async def _cleanup(self):
        stream = self._stream
        try:
            if stream is not None:
                _signal_stop(stream)
                await _drain(stream.aclose())
        finally:
            self._service = self._texts = self._stream = self._loop = None
            self._options = None

    def __del__(self):
        try:
            stream, loop = self._stream, self._loop
            if not self._closed and stream is not None:
                _signal_stop(stream)
                if loop is not None and not loop.is_closed():
                    # The scheduled callback owns the inner stream, not this
                    # facade. Do not create an unawaited coroutine on a dead loop.
                    loop.call_soon_threadsafe(_schedule_cleanup, stream)
        except (AttributeError, RuntimeError):
            pass


def stream_text_records(
    service,
    texts: Sequence[str] | AsyncIterable[str],
    *,
    input_type="document",
    timeout=None,
    cancel=None,
    close_source=False,
) -> EmbeddingRecordStream:
    """Build an inert flat-input facade over a service's existing scheduler."""
    return EmbeddingRecordStream(
        service,
        texts,
        input_type=input_type,
        timeout=timeout,
        cancel=cancel,
        close_source=close_source,
    )
