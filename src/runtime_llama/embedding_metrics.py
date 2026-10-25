"""Fixed-cardinality HTTP diagnostics without request data or event histories.

Wall intervals include network waiting, JSON parsing and transport cleanup as
covered by the caller's measurement. Their union is not summed replica time.
Thread CPU is inclusive for completed measurements and can count nested scopes
more than once; neither quantity is GPU execution time.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
import time
from typing import Iterator

HTTP_KINDS = ("health", "catalog", "props", "tokenize", "embedding")
HTTP_PHASES = ("admission", "preflight", "recovery", "readiness", "preparation")
__all__ = ["EmbeddingHTTPMetrics", "HTTP_KINDS", "HTTP_PHASES"]


@dataclass
class _Counter:
    count: int = 0
    failed: int = 0
    active: int = 0
    peak: int = 0
    wall_seconds: float = 0.0
    union_seconds: float = 0.0
    active_started_sum: float = 0.0
    union_started: float | None = None
    cpu_seconds: float = 0.0
    cpu_samples: int = 0

    def begin(self, now: float) -> None:
        self.count += 1
        if not self.active:
            self.union_started = now
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.active_started_sum += now

    def finish(
        self, started: float, ended: float, cpu: float | None, failed: bool
    ) -> None:
        self.failed += int(failed)
        self.wall_seconds += max(0.0, ended - started)
        self.active -= 1
        self.active_started_sum -= started
        if not self.active:
            assert self.union_started is not None
            self.union_seconds += max(0.0, ended - self.union_started)
            self.union_started = None
            # Avoid retaining floating-point cancellation noise between calls.
            self.active_started_sum = 0.0
        if cpu is not None:
            self.cpu_seconds += max(0.0, cpu)
            self.cpu_samples += 1

    def snapshot(self, now: float) -> dict[str, int | float | None]:
        active_wall = max(0.0, self.active * now - self.active_started_sum)
        active_union = (
            max(0.0, now - self.union_started)
            if self.union_started is not None
            else 0.0
        )
        return {
            "count": self.count,
            "failed": self.failed,
            "active": self.active,
            "peak": self.peak,
            "wall_seconds": self.wall_seconds + active_wall,
            "wall_union_seconds": self.union_seconds + active_union,
            "inclusive_thread_cpu_seconds": (
                self.cpu_seconds if self.cpu_samples else None
            ),
            "thread_cpu_samples": self.cpu_samples,
        }


class EmbeddingHTTPMetrics:
    """Thread-safe constant-space counters for fixed HTTP kind/phase labels.

    ``count`` includes active calls. Wall sums/unions include the active portion
    at snapshot time. CPU samples include only completed calls on their worker
    thread; unavailable clocks are reported explicitly. The helper does not
    perform HTTP, store aliases/payloads, or change scheduler admission.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._clock = time.monotonic
        self._cpu_clock = getattr(time, "thread_time", None)
        self._cpu_supported = callable(self._cpu_clock)
        self._total = _Counter()
        self._kinds = {kind: _Counter() for kind in HTTP_KINDS}
        self._phases = {phase: _Counter() for phase in HTTP_PHASES}
        self._pairs = {
            (kind, phase): _Counter() for kind in HTTP_KINDS for phase in HTTP_PHASES
        }

    def _cpu(self) -> float | None:
        if not self._cpu_supported:
            return None
        try:
            return float(self._cpu_clock())
        except (NotImplementedError, OSError):
            with self._lock:
                self._cpu_supported = False
            return None

    @contextmanager
    def measure(self, kind: str, phase: str) -> Iterator[None]:
        """Measure one call including failures, without retaining its outcome."""
        if type(kind) is not str or kind not in HTTP_KINDS:
            raise ValueError("unsupported embedding HTTP metric kind")
        if type(phase) is not str or phase not in HTTP_PHASES:
            raise ValueError("unsupported embedding HTTP metric phase")
        cpu_started = self._cpu()
        with self._lock:
            started = self._clock()
            counters = (
                self._total,
                self._kinds[kind],
                self._phases[phase],
                self._pairs[kind, phase],
            )
            for counter in counters:
                counter.begin(started)
        failed = True
        try:
            yield
            failed = False
        finally:
            cpu_ended = self._cpu()
            cpu = (
                cpu_ended - cpu_started
                if cpu_started is not None and cpu_ended is not None
                else None
            )
            with self._lock:
                ended = self._clock()
                for counter in counters:
                    counter.finish(started, ended, cpu, failed)

    def snapshot(self) -> dict:
        """Return bounded aggregate state; no event list or private values."""
        with self._lock:
            now = self._clock()
            return {
                "thread_cpu_supported": self._cpu_supported,
                "total": self._total.snapshot(now),
                "by_kind": {
                    key: counter.snapshot(now) for key, counter in self._kinds.items()
                },
                "by_phase": {
                    key: counter.snapshot(now) for key, counter in self._phases.items()
                },
                "by_kind_phase": {
                    f"{kind}:{phase}": counter.snapshot(now)
                    for (kind, phase), counter in self._pairs.items()
                },
            }
