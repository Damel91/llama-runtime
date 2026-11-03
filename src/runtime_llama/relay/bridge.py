"""Dedicated invocation workers with cooperative cancellation and real drain.

The single-item handoff is transport backpressure, not an admission quota on
requests, texts, models or retained results. No framework/global thread pool
is used to run synchronous SDK inference.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError
from threading import Event, Lock, Thread

from runtime_llama.errors import LLMCancelledError


def _clear_tracebacks(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        error.__traceback__ = None
        error = error.__cause__ or error.__context__


class ThreadBridge:
    """A producer receives ``emit(value), cancel_event`` and returns its result."""

    def __init__(self, producer, *, cancel_event=None):
        self._loop = asyncio.get_running_loop()
        self._stop = cancel_event if cancel_event is not None else Event()
        self.cancel_event = _Cancellation(self)
        self._cancel_notice = asyncio.Event()
        self._producer = producer
        self._items = asyncio.Queue(maxsize=1)
        self._completion = self._loop.create_future()
        self._pending = None
        self._lock = Lock()
        self._closed = False
        self._thread = Thread(
            target=self._run, name="runtime-llama-relay-call", daemon=False
        )
        self._thread.start()

    def _emit(self, value):
        if self.cancel_event.is_set():
            raise LLMCancelledError()
        pending = asyncio.run_coroutine_threadsafe(self._items.put(value), self._loop)
        with self._lock:
            self._pending = pending
            if self.cancel_event.is_set():
                pending.cancel()
        try:
            pending.result()
        except CancelledError:
            raise LLMCancelledError() from None
        finally:
            with self._lock:
                if self._pending is pending:
                    self._pending = None
        if self.cancel_event.is_set():
            raise LLMCancelledError()

    def _run(self):
        value = error = None
        try:
            value = self._producer(self._emit, self.cancel_event)
        except BaseException as failure:
            _clear_tracebacks(failure)
            error = failure
        finally:
            self._producer = None
            self._loop.call_soon_threadsafe(self._complete, value, error)

    def _complete(self, value, error):
        if not self._completion.done():
            self._completion.set_result((value, error))

    def cancel(self):
        self._stop.set()
        self._loop.call_soon_threadsafe(self._cancel_notice.set)
        with self._lock:
            if self._pending is not None:
                self._pending.cancel()

    async def wait_cancelled(self):
        await self._cancel_notice.wait()

    async def result(self):
        completion = self._completion
        if completion is None:
            raise RuntimeError("relay_bridge_closed")
        try:
            value, error = await asyncio.shield(completion)
        except asyncio.CancelledError:
            self.cancel()
            raise
        if error is not None:
            raise error
        return value

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        completion = self._completion
        if not self._items.empty():
            return self._items.get_nowait()
        if completion.done():
            _value, error = completion.result()
            if error is not None:
                raise error
            raise StopAsyncIteration
        getter = asyncio.create_task(self._items.get())
        try:
            await asyncio.wait(
                (getter, completion), return_when=asyncio.FIRST_COMPLETED
            )
            if getter.done():
                return getter.result()
            if not self._items.empty():
                return self._items.get_nowait()
            _value, error = completion.result()
            if error is not None:
                raise error
            raise StopAsyncIteration
        except asyncio.CancelledError:
            self.cancel()
            raise
        finally:
            if not getter.done():
                getter.cancel()
            await asyncio.gather(getter, return_exceptions=True)

    async def aclose(self):
        if self._closed:
            return
        self.cancel()
        completion = self._completion
        # A parent cancellation must not orphan the worker. Keep draining even
        # if the caller's task already received CancelledError.
        while not completion.done():
            try:
                await asyncio.shield(completion)
            except asyncio.CancelledError:
                self.cancel()
        # _complete is scheduled as the worker's final operation. It cannot be
        # waiting on this event loop now, so joining does not wait on SDK I/O.
        self._thread.join()
        while not self._items.empty():
            self._items.get_nowait()
        self._completion = None
        self._closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        await self.aclose()


async def run_blocking(function):
    """Run one synchronous callable(cancel_event) without a shared pool cap."""
    async with ThreadBridge(lambda _emit, cancel: function(cancel)) as bridge:
        return await bridge.result()


class _Cancellation:
    """Event-shaped token whose set also releases a blocked transport handoff."""

    def __init__(self, bridge):
        self._bridge = bridge

    def is_set(self):
        return self._bridge._stop.is_set()

    def wait(self, timeout=None):
        return self._bridge._stop.wait(timeout)

    def set(self):
        self._bridge.cancel()
