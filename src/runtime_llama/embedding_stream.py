"""Bounded async window intake over the shared persistent embedding engine."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import math
from threading import Event
from typing import Any, AsyncIterable, Callable, Literal, Sequence

from runtime_llama.embedding_preparation import PreparedEmbeddingInput
from runtime_llama.errors import LLMBackendError


async def _drain(awaitable, *, primary: BaseException | None = None):
    """Do not abandon owned cleanup when its waiter is cancelled repeatedly."""
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:
            if task.done() and task.cancelled():
                if cancelled:
                    outgoing = (
                        primary
                        if isinstance(primary, asyncio.CancelledError)
                        else asyncio.CancelledError()
                    )
                    raise outgoing from exc
                if primary is not None:
                    raise primary from exc
                raise
            cancelled = True
        except BaseException as exc:
            if cancelled:
                outgoing = (
                    primary
                    if isinstance(primary, asyncio.CancelledError)
                    else asyncio.CancelledError()
                )
                raise outgoing from exc
            if primary is not None:
                raise primary from exc
            raise
    if cancelled:
        if isinstance(primary, asyncio.CancelledError):
            raise primary
        raise asyncio.CancelledError
    if primary is not None:
        raise primary
    return result


@dataclass
class _Window:
    call: Any
    future: asyncio.Future


class _EmbeddingWindowBridge:
    """Async intake leases and result ownership, without coordinator workers."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.capacity = max(
            1, adapter.embedding_request_parallelism + adapter.embedding_queue_size
        )
        self.loop = None
        self.streams = set()
        self.used = 0
        self.changed = None

    def activate(self, stream, loop):
        with self.adapter._calls_lock:
            if self.adapter._closed or self.adapter._closing.is_set():
                raise LLMBackendError("llamacpp_embedding_adapter_closed")
            if self.loop is not None and self.loop is not loop:
                raise LLMBackendError("llamacpp_embedding_async_event_loop_mismatch")
            if self.loop is None:
                self.loop = loop
                self.changed = asyncio.Event()
            self.streams.add(stream)

    def retire(self, stream):
        with self.adapter._calls_lock:
            self.streams.discard(stream)
            if not self.streams:
                self.loop = None
                self.changed = None

    def _notify(self):
        if self.changed is not None:
            previous, self.changed = self.changed, asyncio.Event()
            previous.set()

    async def acquire(self, call):
        while True:
            call.check()
            with self.adapter._calls_lock:
                if self.used < self.capacity:
                    self.used += 1
                    return
                changed = self.changed
            try:
                await asyncio.wait_for(
                    changed.wait(), timeout=min(0.05, call.remaining())
                )
            except TimeoutError:
                pass

    def release(self):
        with self.adapter._calls_lock:
            self.used -= 1
        self._notify()

    def submit(self, window, *, prepared, input_type, call):
        # Registration is nonblocking and belongs to the shared engine. Its
        # Future completes only after preparation, HTTP and their leases drain.
        return self.adapter._stream_submit(
            window, prepared=prepared, input_type=input_type, call=call
        )

    def signal_close(self):
        with self.adapter._calls_lock:
            streams = tuple(self.streams)
        for stream in streams:
            stream._adapter_closed()


class EmbeddingWindowStream:
    """Ordered async embedding windows with bounded, shared intake leases.

    Use ``async with`` or ``aclose`` after an early break. Source iterators are
    borrowed unless ``close_source=True`` is requested explicitly.
    """

    def __init__(
        self,
        adapter,
        windows: AsyncIterable[Sequence[str] | Sequence[PreparedEmbeddingInput]],
        *,
        prepared: bool = False,
        input_type: Literal["document", "query", "raw"] = "document",
        max_pending_windows: int = 2,
        timeout: float | None = None,
        cancel: Callable[[], bool] | None = None,
        close_source: bool = False,
        _direct_policy=None,
        _admission_scope=None,
    ):
        if (
            isinstance(max_pending_windows, bool)
            or not isinstance(max_pending_windows, int)
            or max_pending_windows <= 0
        ):
            raise ValueError("max_pending_windows must be a positive integer")
        if not callable(getattr(windows, "__aiter__", None)):
            raise ValueError("windows must be an async iterable")
        if not isinstance(close_source, bool):
            raise ValueError("close_source must be a boolean")
        value = adapter.profile.timeout if timeout is None else timeout
        if isinstance(value, bool):
            raise ValueError("timeout must be a positive finite number")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("timeout must be a positive finite number") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError("timeout must be a positive finite number")
        if prepared and _direct_policy is not None:
            raise ValueError("direct streams cannot contain exact preparations")
        if _admission_scope is not None and (prepared or _direct_policy is None):
            raise ValueError("admission scopes belong to direct streams")
        self._direct_policy = _direct_policy
        self._admission_scope = _admission_scope
        self._packing_group = object()
        self._adapter, self._source = adapter, windows
        self._prepared, self._input_type = prepared, input_type
        self._maximum, self._timeout, self._cancel = (
            max_pending_windows,
            value,
            cancel,
        )
        self._close_source = close_source
        self._loop = self._bridge = self._iterator = None
        self._producer = self._pull = self._pull_call = None
        self._pull_lease = False
        self._cleanup_task = self._changed = None
        self._pending = deque()
        self._stop = Event()
        self._failure = None
        self._error_reported = self._closed = self._exhausted = False
        self._next_active = False
        self._window_index = 0

    def __aiter__(self) -> EmbeddingWindowStream:
        return self

    def _check_loop(self):
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise LLMBackendError("llamacpp_embedding_async_event_loop_mismatch")
        return loop

    async def _start(self):
        loop = self._check_loop()
        if self._loop is not None or self._closed:
            return
        bridge = self._adapter._stream_bridge()
        bridge.activate(self, loop)
        self._loop, self._bridge = loop, bridge
        self._changed = asyncio.Event()
        try:
            self._iterator = aiter(self._source)
        except BaseException:
            bridge.retire(self)
            self._closed = True
            raise
        self._producer = loop.create_task(self._produce())

    def _notify(self):
        if self._changed is not None:
            previous, self._changed = self._changed, asyncio.Event()
            previous.set()

    def _request_stop(self):
        self._stop.set()
        if self._pull_call is not None:
            self._pull_call.stop.set()
        for window in self._pending:
            window.call.stop.set()
        self._adapter._engine.wake()
        self._notify()

    def _fail(self, exc):
        if self._failure is None and not self._stop.is_set():
            self._failure = exc
        self._request_stop()
        self._ensure_cleanup()

    def _adapter_closed(self):
        self._stop.set()
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._closed_on_loop)
            except RuntimeError:
                pass

    def _closed_on_loop(self):
        if not self._closed:
            if self._failure is None:
                self._failure = LLMBackendError("llamacpp_embedding_adapter_closed")
            self._request_stop()
            self._ensure_cleanup()

    async def _wait_pull(self, call):
        while not self._pull.done():
            call.check()
            await asyncio.wait([self._pull], timeout=min(0.05, call.remaining()))
        call.check()
        return self._pull.result()

    async def _produce(self):
        try:
            while not self._stop.is_set():
                changed = self._changed
                if len(self._pending) >= self._maximum:
                    await changed.wait()
                    continue
                call = self._adapter._stream_call(self._cancel, self._timeout)
                if self._direct_policy is not None:
                    call._direct_policy = self._direct_policy
                    if self._admission_scope is not None:
                        call._admission_scope = self._admission_scope
                self._pull_call = call
                await self._bridge.acquire(call)
                self._pull_lease = True
                self._pull = asyncio.ensure_future(anext(self._iterator))
                try:
                    value = await self._wait_pull(call)
                except StopAsyncIteration:
                    self._exhausted = True
                    break
                if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                    raise ValueError("each window must be a finite sequence")
                copied = list(value)
                if not self._prepared and any(
                    not isinstance(text, str) for text in copied
                ):
                    raise ValueError("text windows must contain only strings")
                call.check()
                # Preserve source-window provenance without changing dispatch
                # or result ordering. Recovery never guesses it from text.
                call._window_index = self._window_index
                call._packing_group = self._packing_group
                future = self._bridge.submit(
                    copied,
                    prepared=self._prepared,
                    input_type=self._input_type,
                    call=call,
                )
                self._window_index += 1
                wrapped = asyncio.wrap_future(future, loop=self._loop)
                self._pending.append(_Window(call, wrapped))
                self._pull_lease = False
                self._pull = self._pull_call = None
                wrapped.add_done_callback(self._completed)
                self._notify()
        except BaseException as exc:
            if not self._stop.is_set():
                self._fail(exc)
        finally:
            if self._pull_lease and (self._pull is None or self._pull.done()):
                self._pull_lease = False
                self._bridge.release()
            # A cancelled source pull can still be executing its async finally.
            # Cleanup owns that lease until the actual pull task has drained.
            self._pull_call = None
            self._notify()

    def _completed(self, future):
        if not self._stop.is_set():
            if future.cancelled():
                self._fail(LLMBackendError("llamacpp_embedding_adapter_closed"))
            elif future.exception() is not None:
                self._fail(future.exception())
        self._notify()

    def _release_window(self, window):
        # A completed result can still occupy the engine's memory budget while
        # it waits for ordered delivery or a slow consumer. Transfer/discard it
        # before releasing the source/window admission lease.
        try:
            self._adapter._stream_release_result(window.call)
        finally:
            self._bridge.release()

    def _ensure_cleanup(self):
        if self._cleanup_task is None and self._loop is not None:
            self._cleanup_task = self._loop.create_task(self._cleanup())
            # Retain the exception for awaiters without an unobserved-task warning.
            self._cleanup_task.add_done_callback(
                lambda task: None if task.cancelled() else task.exception()
            )
        return self._cleanup_task

    async def _cleanup(self):
        self._request_stop()
        try:
            if self._pull is not None and not self._pull.done():
                self._pull.cancel()
            if self._producer is not None and not self._producer.done():
                self._producer.cancel()
            owned = [item for item in (self._pull, self._producer) if item is not None]
            if owned:
                await asyncio.gather(*owned, return_exceptions=True)
            if self._pull_lease:
                self._pull_lease = False
                self._bridge.release()
            if self._pending:
                await asyncio.gather(
                    *(asyncio.shield(item.future) for item in self._pending),
                    return_exceptions=True,
                )
            while self._pending:
                self._release_window(self._pending.popleft())
            if self._close_source:
                close = getattr(self._iterator, "aclose", None)
                if callable(close):
                    await close()
        finally:
            self._closed = True
            self._notify()
            self._bridge.retire(self)

    async def __anext__(self) -> list[list[float]]:
        self._check_loop()
        if self._next_active:
            raise RuntimeError("concurrent iteration of one embedding stream")
        self._next_active = True
        try:
            await self._start()
            while True:
                changed = self._changed
                if self._failure is not None and not self._error_reported:
                    self._error_reported = True
                    await _drain(self._ensure_cleanup(), primary=self._failure)
                if self._closed or self._stop.is_set():
                    task = self._ensure_cleanup()
                    if task is not None:
                        await _drain(task)
                    raise StopAsyncIteration
                if self._pending and self._pending[0].future.done():
                    window = self._pending[0]
                    try:
                        result = window.future.result()
                    except BaseException as exc:
                        self._fail(exc)
                        continue
                    self._pending.popleft()
                    self._release_window(window)
                    self._notify()
                    return result
                if self._exhausted and not self._pending:
                    await _drain(self._ensure_cleanup())
                    raise StopAsyncIteration
                await changed.wait()
        except asyncio.CancelledError as exc:
            self._request_stop()
            task = self._ensure_cleanup()
            if task is not None:
                await _drain(task, primary=exc)
            raise
        finally:
            self._next_active = False

    async def __aenter__(self) -> EmbeddingWindowStream:
        await self._start()
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        if exc is not None and self._loop is not None:
            self._request_stop()
            await _drain(self._ensure_cleanup(), primary=exc)
        await self.aclose()

    async def aclose(self) -> None:
        self._check_loop()
        if self._loop is None:
            self._closed = True
            return
        self._request_stop()
        await _drain(self._ensure_cleanup())


async def _close_embedding_adapter(adapter):
    loop = asyncio.get_running_loop()
    with adapter._calls_lock:
        bridge = adapter._async_window_bridge
        if bridge is not None and bridge.loop is not None and bridge.loop is not loop:
            raise LLMBackendError("llamacpp_embedding_async_event_loop_mismatch")
        # Close admission atomically with the stream snapshot. The synchronous
        # close still owns resource teardown and does not return on this flag.
        adapter._closing.set()
        streams = () if bridge is None else tuple(bridge.streams)

    async def cleanup():
        errors = await asyncio.gather(
            *(stream.aclose() for stream in streams), return_exceptions=True
        )
        await asyncio.to_thread(adapter.close)
        for error in errors:
            if isinstance(error, BaseException):
                raise error

    await _drain(cleanup())
