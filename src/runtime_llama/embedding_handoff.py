"""Track response ownership with an optional independent byte policy.

Every original POST retains its ticket through response transfer or discard.
Without explicit capacity opt-in, accounting observes ownership and never
creates an admission quota from another resource's settings.
"""

from __future__ import annotations

from threading import Lock

from runtime_llama.embedding_resources import (
    EmbeddingResourceLimitError,
    _optional_positive,
)
from runtime_llama.errors import LLMBackendError


class HandoffTicket:
    """Retain transport/body/materialized ownership until output transfer."""

    def __init__(self, reservations, amount, family, response_limit):
        self.reservations = reservations
        self.amount, self.family = amount, family
        self.response_limit = response_limit
        self.released = False

    def release(self):
        with self.reservations._lock:
            if self.released:
                return
            self.released = True
            self.reservations._used[self.family] -= self.amount
            self.reservations._tickets -= 1
        if self.reservations._notify is not None:
            self.reservations._notify()


class HandoffReservations:
    """Adapter-wide ownership, optionally bounded by an explicit byte policy.

    Pending inputs and response-reader limits do not derive this pool's size.
    Root and recovery descendants retain the original ticket until every
    response is transferred or discarded. Accounting is not a measurement of
    process RSS or a promise about caller-owned output retention.
    """

    def __init__(self, max_bytes, configured_response_bytes, *, notify=None):
        self.limit = _optional_positive(max_bytes, "max_bytes")
        self.normal_limit = self.limit
        self.progress_limit = None
        self.max_batch_bytes = self.limit
        self.response_limit = _optional_positive(
            configured_response_bytes, "configured_response_bytes"
        )
        self._notify = notify
        self._lock = Lock()
        self._used = {"normal": 0, "progress": 0}
        self._peak = {"normal": 0, "progress": 0}
        self._peak_total = self._tickets = 0
        self._closed = False

    @staticmethod
    def _wire_cost(record):
        # Also accept ready entries, without walking or serializing their text.
        record = getattr(record, "record", record)
        sizes = (
            getattr(record, "_text_wire_bytes", 0),
            getattr(record, "_tokens_wire_bytes", None) or 0,
        )
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in sizes
        ):
            raise ValueError("record wire sizes must be nonnegative integers")
        return max(sizes) + 1

    def _envelope(self, records, dimension=None, *, wire_overhead=64):
        if dimension is not None and (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise ValueError("dimension must be None or a positive integer")
        if not records:
            raise ValueError("handoff candidates must contain records")
        if (
            isinstance(wire_overhead, bool)
            or not isinstance(wire_overhead, int)
            or wire_overhead < 0
        ):
            raise ValueError("wire_overhead must be nonnegative")
        request_wire = wire_overhead + sum(
            self._wire_cost(record) for record in records
        )
        # Observational accounting is independent of admission. No capacity
        # policy is inferred from working bytes, response bytes or cold width.
        materialized = (
            0 if dimension is None else 64 + (40 * dimension + 96) * len(records)
        )
        return request_wire + materialized, self.response_limit

    def estimate(self, records, dimension=None, *, wire_overhead=64):
        """Return the estimated owned-byte accounting for a candidate."""
        return self._envelope(records, dimension, wire_overhead=wire_overhead)[0]

    def try_acquire(self, records, dimension=None, *, progress=False, wire_overhead=64):
        amount, response_limit = self._envelope(
            records, dimension, wire_overhead=wire_overhead
        )
        # Only a deliberately configured policy can constrain a candidate.
        # An indivisible row exceeding that policy fails explicitly.
        if self.max_batch_bytes is not None and amount > self.max_batch_bytes:
            raise EmbeddingResourceLimitError(
                "handoff_bytes", self.max_batch_bytes, amount
            )
        with self._lock:
            if self._closed:
                raise LLMBackendError("llamacpp_embedding_adapter_closed")
            if (
                self.normal_limit is None
                or self._used["normal"] + amount <= self.normal_limit
            ):
                family = "normal"
            else:
                return None
            self._used[family] += amount
            self._tickets += 1
            self._peak[family] = max(self._peak[family], self._used[family])
            self._peak_total = max(self._peak_total, sum(self._used.values()))
            return HandoffTicket(self, amount, family, response_limit)

    def snapshot(self):
        with self._lock:
            return {
                "limit": self.limit,
                "used": sum(self._used.values()),
                "peak": self._peak_total,
                "normal_limit": self.normal_limit,
                "normal_used": self._used["normal"],
                "normal_peak": self._peak["normal"],
                "progress_limit": self.progress_limit,
                "progress_used": self._used["progress"],
                "progress_peak": self._peak["progress"],
                "response_limit": self.response_limit,
                "tickets": self._tickets,
                "closed": self._closed,
            }

    def close(self):
        """Close admission; held tickets still require actual transfer/cleanup."""
        with self._lock:
            self._closed = True
        if self._notify is not None:
            self._notify()
