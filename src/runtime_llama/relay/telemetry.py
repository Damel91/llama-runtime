"""Local telemetry projections for the relay console."""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from threading import Lock

from runtime_llama.embedding_metrics import HTTP_KINDS


INFERENCE_KINDS = ("chat", "embedding")
OUTCOMES = ("completed", "failed", "cancelled")


def model_activity(active: list[dict], recent: list[dict]) -> list[dict]:
    """Group current and retained observations, without a lifetime label store."""
    groups = {}
    for rows, running in ((active, True), (recent, False)):
        for row in rows:
            backend, endpoint = row.get("backend"), row.get("endpoint")
            model, kind = row.get("selected_model") or row["model"], row["kind"]
            key = (backend, endpoint, model, kind)
            if key not in groups:
                groups[key] = {
                    "backend": backend,
                    "endpoint": endpoint,
                    "model": model,
                    "kind": kind,
                    "active": 0,
                    "counters": dict.fromkeys(OUTCOMES, 0),
                    "samples": 0,
                    "average_elapsed_ms": None,
                }
            group = groups[key]
            if running:
                group["active"] += 1
            else:
                group["counters"][row["status"]] += 1
                samples = group["samples"] + 1
                average = group["average_elapsed_ms"] or 0.0
                group["average_elapsed_ms"] = (
                    average + (row["elapsed_ms"] - average) / samples
                )
                group["samples"] = samples
    return list(groups.values())


@dataclass
class _HTTPActivity:
    count: int = 0
    failed: int = 0
    active: int = 0
    peak: int = 0

    def begin(self):
        self.count += 1
        self.active += 1
        self.peak = max(self.peak, self.active)

    def finish(self, failed):
        self.failed += int(failed)
        self.active -= 1

    def snapshot(self):
        return {
            "count": self.count,
            "failed": self.failed,
            "active": self.active,
            "peak": self.peak,
        }


class RelayEmbeddingTelemetry:
    """Observe shared HTTP calls once at the adapter boundary, across owners."""

    def __init__(self):
        self._lock = Lock()
        self._by_kind = {kind: _HTTPActivity() for kind in HTTP_KINDS}
        self._active = {}
        self._recent = OrderedDict()
        self._peak_active_replicas = 0

    def for_endpoint(self, endpoint):
        return partial(self._measure, endpoint)

    @contextmanager
    def _measure(self, endpoint, kind, model=None):
        replica = None
        key = (endpoint, model)
        with self._lock:
            activity = self._by_kind[kind]
            activity.begin()
            if kind == "embedding" and model:
                replica = self._active.get(key)
                if replica is None:
                    replica = self._recent.pop(key, None) or _HTTPActivity()
                    self._active[key] = replica
                replica.begin()
                self._peak_active_replicas = max(
                    self._peak_active_replicas, len(self._active)
                )
        failed = True
        try:
            yield
            failed = False
        finally:
            with self._lock:
                activity.finish(failed)
                if replica is not None:
                    replica.finish(failed)
                    if not replica.active:
                        del self._active[key]
                        self._recent[key] = replica
                        if len(self._recent) > 128:
                            self._recent.popitem(last=False)

    def snapshot(self):
        with self._lock:
            return {
                "by_kind": {
                    kind: activity.snapshot()
                    for kind, activity in self._by_kind.items()
                },
                "replicas": [
                    {"endpoint": endpoint, "model": model, **activity.snapshot()}
                    for entries in (
                        self._active.items(),
                        reversed(self._recent.items()),
                    )
                    for (endpoint, model), activity in entries
                ],
                "active_replicas": len(self._active),
                "peak_active_replicas": self._peak_active_replicas,
            }
