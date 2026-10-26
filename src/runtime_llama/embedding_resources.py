"""Adapter-wide intake and byte leases, independent of replica topology.

These counters bound library ownership, not the caller's source or process RSS.
No admission waiter creates a worker thread; async waiters use loop notifications.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sequence
from threading import Condition, get_ident
import struct
import sys

from runtime_llama.embedding_preparation import PreparedEmbeddingInput
from runtime_llama.errors import LLMBackendError

_REFERENCE_BYTES = struct.calcsize("P")


class EmbeddingResourceLimitError(LLMBackendError):
    """An indivisible input cannot fit its finite ownership budget."""

    def __init__(self, resource, limit, requested):
        super().__init__(f"llamacpp_embedding_{resource}_limit")
        self.resource = resource
        self.limit = limit
        self.requested = requested
        self.remedy = "reduce the input window or increase the resource limit"


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_positive(value, name):
    return None if value is None else _positive(value, name)


class ByteLease:
    """One idempotent byte reservation, released only after real ownership ends."""

    def __init__(self, budget, amount, family):
        self.budget, self.amount, self.family = budget, amount, family
        self.released = False

    def release(self):
        with self.budget._condition:
            if self.released:
                return
            self.released = True
            self.budget._used -= self.amount
            self.budget._leases -= 1
            self.budget._condition.notify_all()
        if self.budget._notify is not None:
            self.budget._notify()


class ByteBudget:
    """Nonblocking coordinator admission for one distinct byte-pool family.

    Use separate instances for embedding handoff and control/recovery progress.
    A request larger than the whole pool fails; aggregate pressure returns None.
    """

    def __init__(self, limit, *, name="handoff_bytes", notify=None):
        self.limit = _optional_positive(limit, "limit")
        self.name, self._notify = name, notify
        self._condition = Condition()
        self._used = self._peak = self._leases = 0
        self._closed = False

    def try_acquire(self, amount, family="embedding"):
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError("amount must be a nonnegative integer")
        if self.limit is not None and amount > self.limit:
            raise EmbeddingResourceLimitError(self.name, self.limit, amount)
        with self._condition:
            if self._closed:
                raise LLMBackendError("llamacpp_embedding_adapter_closed")
            if self.limit is not None and self._used + amount > self.limit:
                return None
            self._used += amount
            self._leases += 1
            self._peak = max(self._peak, self._used)
            return ByteLease(self, amount, family)

    def snapshot(self):
        with self._condition:
            return {
                "limit": self.limit,
                "used": self._used,
                "peak": self._peak,
                "leases": self._leases,
                "closed": self._closed,
            }

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._notify is not None:
            self._notify()


class EmbeddingIntake:
    """Finite owner count and immutable snapshots shared by sync/async callers.

    The count limit resolves once for the adapter. Topology changes never create
    a second admission pool. A serial snapshot builder prevents multiple partial
    windows from consuming all bytes before any of them can be registered.
    """

    def __init__(self, owner_limit, max_snapshot_bytes, *, notify=None):
        self.owner_limit = _optional_positive(owner_limit, "owner_limit")
        self.max_snapshot_bytes = _optional_positive(
            max_snapshot_bytes, "max_snapshot_bytes"
        )
        self._notify = notify
        self._condition = Condition()
        self._owners = self._peak_owners = self._bytes = self._peak_bytes = 0
        self._leases = set()
        self._owner_waiters = deque()
        self._builder_waiters = deque()
        self._builder = None
        self._signals = set()
        self._closed = False

    def set_notify(self, notify):
        self._notify = notify

    def _changed_locked(self):
        self._condition.notify_all()
        for loop, event in tuple(self._signals):
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                # The loop owner must still drain/retire its lease explicitly.
                pass

    def _changed(self):
        if self._notify is not None:
            self._notify()

    def _check_locked(self, lease=None):
        if self._closed:
            raise LLMBackendError("llamacpp_embedding_adapter_closed")
        if lease is not None and lease.released:
            raise LLMBackendError("llamacpp_embedding_intake_lease_released")

    def _existing(self, call):
        lease = getattr(call, "_intake_lease", None)
        if lease is not None and not lease.released:
            if lease.intake is not self:
                raise LLMBackendError("llamacpp_embedding_intake_lease_mismatch")
            return lease
        return None

    def _new_owner_locked(self, call, *, asynchronous=False):
        lease = OwnerLease(self, call, asynchronous=asynchronous)
        call._intake_lease = lease
        self._leases.add(lease)
        self._owners += 1
        self._peak_owners = max(self._peak_owners, self._owners)
        return lease

    def acquire_owner(self, call):
        call.check()
        ticket = object()
        with self._condition:
            self._check_locked()
            existing = self._existing(call)
            if existing is not None:
                return existing
            self._owner_waiters.append(ticket)
        try:
            while True:
                call.check()
                with self._condition:
                    self._check_locked()
                    if self._owner_waiters[0] is ticket and (
                        self.owner_limit is None or self._owners < self.owner_limit
                    ):
                        self._owner_waiters.popleft()
                        lease = self._new_owner_locked(call)
                        self._changed_locked()
                        return lease
                    self._condition.wait(0.05)
        finally:
            with self._condition:
                if ticket in self._owner_waiters:
                    self._owner_waiters.remove(ticket)
                    self._changed_locked()

    async def _wait_async(self, call, predicate):
        loop = asyncio.get_running_loop()
        while True:
            call.check()
            with self._condition:
                self._check_locked()
                result = predicate()
                if result is not None:
                    return result
                event = asyncio.Event()
                signal = (loop, event)
                self._signals.add(signal)
            try:
                try:
                    await asyncio.wait_for(event.wait(), min(0.05, call.remaining()))
                except TimeoutError:
                    pass
            finally:
                with self._condition:
                    self._signals.discard(signal)

    async def acquire_owner_async(self, call):
        call.check()
        ticket = object()
        with self._condition:
            self._check_locked()
            existing = self._existing(call)
            if existing is not None:
                return existing
            self._owner_waiters.append(ticket)

        def admitted():
            if self._owner_waiters[0] is ticket and (
                self.owner_limit is None or self._owners < self.owner_limit
            ):
                self._owner_waiters.popleft()
                lease = self._new_owner_locked(call, asynchronous=True)
                self._changed_locked()
                return lease
            return None

        try:
            return await self._wait_async(call, admitted)
        finally:
            with self._condition:
                if ticket in self._owner_waiters:
                    self._owner_waiters.remove(ticket)
                    self._changed_locked()

    def snapshot(self):
        with self._condition:
            return {
                "owner_limit": self.owner_limit,
                "owners": self._owners,
                "peak_owners": self._peak_owners,
                "owner_waiters": len(self._owner_waiters),
                "snapshot_bytes_limit": self.max_snapshot_bytes,
                "snapshot_bytes": self._bytes,
                "peak_snapshot_bytes": self._peak_bytes,
                "snapshot_waiters": len(self._builder_waiters),
                "snapshot_building": self._builder is not None,
                "snapshot_waiting_bytes": bool(
                    self._builder is not None and self._builder._waiting_bytes
                ),
                "closed": self._closed,
            }

    def assert_can_drain_sync(self):
        """Reject self-wait before the adapter changes its closed state."""
        identity = get_ident()
        with self._condition:
            if any(
                not lease.asynchronous and lease.owner_thread == identity
                for lease in self._leases
            ):
                raise LLMBackendError("llamacpp_embedding_close_from_active_intake")

    def drain_sync(self):
        """Drain sync ownership, including snapshots not registered in the engine.

        Async source cleanup belongs to ``adapter.aclose``. Waiting for it from
        synchronous close on its own event loop would prevent that cleanup.
        Repeated interruption never turns a live lease into a completed one.
        """
        self.assert_can_drain_sync()
        primary = None
        with self._condition:
            if not self._closed:
                raise RuntimeError("close intake admission before draining it")
            while any(not lease.asynchronous for lease in self._leases):
                try:
                    self._condition.wait()
                except BaseException as exc:
                    if primary is None:
                        primary = exc
        if primary is not None:
            raise primary

    def wait_drained(self):
        """Compatibility spelling for the synchronous ownership drain."""
        self.drain_sync()

    def close(self):
        """Stop admission and wake waiters; existing owners still owe cleanup."""
        with self._condition:
            self._closed = True
            self._changed_locked()
        self._changed()


class OwnerLease:
    """Count and snapshot ownership through ordered transfer or actual cleanup."""

    def __init__(self, intake, call, *, asynchronous=False):
        self.intake, self.call = intake, call
        self.asynchronous = asynchronous
        self.owner_thread = None if asynchronous else get_ident()
        self.values = None
        self.released = False
        self._bytes = 0
        self._waiting_bytes = False

    def _setup(self, values):
        if values is self.values and self.values is not None:
            return None
        if self.values is not None:
            raise LLMBackendError("llamacpp_embedding_intake_snapshot_already_set")
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError("each window must be a finite sequence")
        size = len(values)
        # Both the building list and the frozen tuple are covered at transfer.
        base = sys.getsizeof([]) + sys.getsizeof(()) + 2 * _REFERENCE_BYTES * size
        if (
            self.intake.max_snapshot_bytes is not None
            and base > self.intake.max_snapshot_bytes
        ):
            raise EmbeddingResourceLimitError(
                "snapshot_bytes", self.intake.max_snapshot_bytes, base
            )
        return size, base

    @staticmethod
    def _item(value, *, prepared, coerce_text):
        if prepared:
            if type(value) is PreparedEmbeddingInput:
                ids = value._token_ids
                return value, (
                    640
                    + sys.getsizeof(value.text)
                    + (40 * len(ids) if ids is not None else 0)
                )
            # Preserve the adapter's typed validation of foreign preparations.
            return value, sys.getsizeof(value)
        if coerce_text:
            value = str(value)
        elif not isinstance(value, str):
            raise ValueError("text windows must contain only strings")
        return value, sys.getsizeof(value)

    def _builder_ready(self):
        self.intake._check_locked(self)
        if self.intake.max_snapshot_bytes is None:
            if self in self.intake._builder_waiters:
                self.intake._builder_waiters.remove(self)
            return True
        if self.intake._builder is None and self.intake._builder_waiters[0] is self:
            self.intake._builder_waiters.popleft()
            self.intake._builder = self
            return True
        return None

    def _reserve_ready(self, extra):
        self.intake._check_locked(self)
        wanted = self._bytes + extra
        if (
            self.intake.max_snapshot_bytes is not None
            and wanted > self.intake.max_snapshot_bytes
        ):
            raise EmbeddingResourceLimitError(
                "snapshot_bytes", self.intake.max_snapshot_bytes, wanted
            )
        if (
            self.intake.max_snapshot_bytes is not None
            and self.intake._bytes + extra > self.intake.max_snapshot_bytes
        ):
            self._waiting_bytes = True
            return None
        self._waiting_bytes = False
        self._bytes = wanted
        self.intake._bytes += extra
        self.intake._peak_bytes = max(self.intake._peak_bytes, self.intake._bytes)
        return True

    def _reserve(self, extra):
        while True:
            self.call.check()
            with self.intake._condition:
                if self._reserve_ready(extra):
                    return
                self.intake._condition.wait(0.05)

    def _finish(self, items):
        frozen = tuple(items)
        with self.intake._condition:
            temporary = sys.getsizeof([]) + _REFERENCE_BYTES * len(items)
            self._bytes -= temporary
            self.intake._bytes -= temporary
            self.values = frozen
            self.intake._changed_locked()
        return frozen

    def _end_builder(self, *, success):
        with self.intake._condition:
            if self.intake._builder is self:
                self.intake._builder = None
            if self in self.intake._builder_waiters:
                self.intake._builder_waiters.remove(self)
            self._waiting_bytes = False
            if not success:
                self.intake._bytes -= self._bytes
                self._bytes = 0
            self.intake._changed_locked()
        self.intake._changed()

    def snapshot(self, values, *, prepared=False, coerce_text=False):
        setup = self._setup(values)
        if setup is None:
            return self.values
        size, base = setup
        with self.intake._condition:
            self.intake._check_locked(self)
            self.intake._builder_waiters.append(self)
        success = False
        try:
            while True:
                self.call.check()
                with self.intake._condition:
                    if self._builder_ready():
                        break
                    self.intake._condition.wait(0.05)
            self._reserve(base)
            items = [None] * size
            for index in range(size):
                self.call.check()
                value, cost = self._item(
                    values[index], prepared=prepared, coerce_text=coerce_text
                )
                self._reserve(cost)
                items[index] = value
            self.call.check()
            result = self._finish(items)
            success = True
            return result
        finally:
            self._end_builder(success=success)

    async def snapshot_async(self, values, *, prepared=False, coerce_text=False):
        setup = self._setup(values)
        if setup is None:
            return self.values
        size, base = setup
        with self.intake._condition:
            self.intake._check_locked(self)
            self.intake._builder_waiters.append(self)
        success = False
        try:
            await self.intake._wait_async(self.call, self._builder_ready)
            await self.intake._wait_async(self.call, lambda: self._reserve_ready(base))
            items = [None] * size
            for index in range(size):
                self.call.check()
                value, cost = self._item(
                    values[index], prepared=prepared, coerce_text=coerce_text
                )
                await self.intake._wait_async(
                    self.call, lambda cost=cost: self._reserve_ready(cost)
                )
                items[index] = value
                if index % 64 == 63:
                    await asyncio.sleep(0)
            self.call.check()
            result = self._finish(items)
            success = True
            return result
        finally:
            self._end_builder(success=success)

    def release(self):
        with self.intake._condition:
            if self.released:
                return
            self.released = True
            self.intake._leases.discard(self)
            self.intake._owners -= 1
            self.intake._bytes -= self._bytes
            self._bytes = 0
            self.values = None
            self.intake._changed_locked()
        self.intake._changed()
