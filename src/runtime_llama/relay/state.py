"""Owned generation leases and content-free request state for the relay.

There is no admission, model-count or memory quota here. The registry retains
active requests and one explicitly labelled latest completion, not a history.
Adapters, iterators, request bodies and exceptions never belong to summaries.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
import json
from threading import Event, Lock, RLock
import time
from typing import Any
from uuid import uuid4

from runtime_llama.relay.config import RelayConfigSnapshot


_SUMMARY_FIELDS = frozenset(
    {
        "status",
        "backend",
        "endpoint",
        "selected_model",
        "terminal_reason",
        "error_code",
        "corrections_used",
        "reasons",
        "usage",
        "observed_usage",
        "total_usage",
        "usage_complete",
        "unknown_usage_fields",
        "coordinator_elapsed_ms",
        "server_timings",
    }
)
_PRIVATE_FIELDS = frozenset(
    {
        "messages",
        "prompt",
        "content",
        "arguments",
        "thinking",
        "thinking_diagnostic",
        "last_proposal",
        "raw",
        "api_key",
        "auth_key",
        "proxy_url",
    }
)


class RelayGenerationClosed(RuntimeError):
    def __init__(self) -> None:
        super().__init__("relay_generation_closed")


class RelayGenerationLease:
    """One caller's ownership; release is idempotent and drops the reference."""

    def __init__(self, generation: RelayGeneration) -> None:
        self._generation: RelayGeneration | None = generation
        self._lock = RLock()

    @property
    def snapshot(self) -> RelayConfigSnapshot:
        with self._lock:
            if self._generation is None:
                raise RelayGenerationClosed()
            return self._generation.snapshot

    def resource(self, key: Any, factory: Callable[[], Any]) -> Any:
        """Create/reuse an owned resource while this lease is alive.

        Factories should only construct resources, not perform discovery or
        inference while the generation construction lock is held.
        """
        with self._lock:
            if self._generation is None:
                raise RelayGenerationClosed()
            return self._generation._resource(key, factory)

    def release(self) -> None:
        with self._lock:
            generation, self._generation = self._generation, None
        if generation is not None:
            generation._release()

    close = release

    def __enter__(self) -> RelayGenerationLease:
        with self._lock:
            if self._generation is None:
                raise RelayGenerationClosed()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.release()


class RelayGeneration:
    """A frozen configuration and resources retained until actual lease drain."""

    def __init__(
        self,
        snapshot: RelayConfigSnapshot,
        *,
        on_drained: Callable[[RelayGeneration], None] | None = None,
    ) -> None:
        self.snapshot = snapshot
        self._lock = RLock()
        self._leases = 0
        self._retired = False
        self._closing = False
        self._resources: dict[Any, Any] = {}
        self._drained = Event()
        self._on_drained = on_drained
        self._close_errors: tuple[str, ...] = ()

    def acquire(self) -> RelayGenerationLease:
        with self._lock:
            if self._retired or self._closing:
                raise RelayGenerationClosed()
            self._leases += 1
        return RelayGenerationLease(self)

    def _resource(self, key: Any, factory: Callable[[], Any]) -> Any:
        with self._lock:
            if self._closing or self._leases == 0:
                raise RelayGenerationClosed()
            if key not in self._resources:
                self._resources[key] = factory()
            return self._resources[key]

    def _release(self) -> None:
        with self._lock:
            if self._leases <= 0:
                raise RuntimeError("relay_generation_lease_underflow")
            self._leases -= 1
        self._close_if_drained()

    def retire(self) -> None:
        """Stop new acquisitions; existing work keeps its resources unchanged."""
        with self._lock:
            self._retired = True
        self._close_if_drained()

    def _close_if_drained(self) -> None:
        with self._lock:
            if not self._retired or self._leases or self._closing:
                return
            self._closing = True
            resources = tuple(
                {id(value): value for value in self._resources.values()}.values()
            )
            self._resources.clear()
            on_drained, self._on_drained = self._on_drained, None
        errors = []
        try:
            for resource in resources:
                close = getattr(resource, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as error:
                        # Do not retain an exception/traceback, provider strings
                        # or a failed resource in long-lived manager state.
                        errors.append(type(error).__name__)
        finally:
            with self._lock:
                self._close_errors = tuple(errors)
            self._drained.set()
            if on_drained is not None:
                on_drained(self)

    def wait_drained(self, timeout: float | None = None) -> bool:
        return self._drained.wait(timeout)

    def diagnostics(self) -> dict:
        with self._lock:
            return {
                "revision": self.snapshot.revision,
                "leases": self._leases,
                "retired": self._retired,
                "drained": self._drained.is_set(),
                "resources": len(self._resources),
                "close_error_types": list(self._close_errors),
            }


class RelayGenerationManager:
    """Own current/retired generations; configuration changes do not cancel work."""

    def __init__(self, snapshot: RelayConfigSnapshot) -> None:
        self._lock = RLock()
        self._closed = False
        self._generations: set[RelayGeneration] = set()
        self._current: RelayGeneration | None = self._new(snapshot)

    def _new(self, snapshot: RelayConfigSnapshot) -> RelayGeneration:
        generation = RelayGeneration(snapshot, on_drained=self._forget)
        self._generations.add(generation)
        return generation

    def _forget(self, generation: RelayGeneration) -> None:
        with self._lock:
            self._generations.discard(generation)

    def acquire(self) -> RelayGenerationLease:
        with self._lock:
            if self._closed or self._current is None:
                raise RelayGenerationClosed()
            return self._current.acquire()

    def replace(self, snapshot: RelayConfigSnapshot) -> None:
        with self._lock:
            if self._closed or self._current is None:
                raise RelayGenerationClosed()
            if snapshot.revision <= self._current.snapshot.revision:
                raise ValueError("relay_generation_revision_must_increase")
            old = self._current
            self._current = self._new(snapshot)
        old.retire()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._current = None
            generations = tuple(self._generations)
        for generation in generations:
            generation.retire()

    def wait_drained(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                generations = tuple(self._generations)
            if not generations:
                return True
            for generation in generations:
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                if not generation.wait_drained(remaining):
                    return False

    def diagnostics(self) -> dict:
        with self._lock:
            return {
                "closed": self._closed,
                "current_revision": self._current.snapshot.revision
                if self._current
                else None,
                "generations": [item.diagnostics() for item in self._generations],
            }


def _safe_updates(fields: Mapping) -> dict:
    if set(fields) - _SUMMARY_FIELDS:
        raise ValueError("relay_request_summary_fields_invalid")

    def check(value: Any) -> None:
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) or key in _PRIVATE_FIELDS for key in value):
                raise ValueError("relay_request_summary_private_field")
            for item in value.values():
                check(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check(item)
        elif value is not None and type(value) not in {str, bool, int, float}:
            raise ValueError("relay_request_summary_must_be_json")

    check(fields)
    try:
        return json.loads(json.dumps(dict(fields), allow_nan=False))
    except (ValueError, TypeError, OverflowError):
        raise ValueError("relay_request_summary_must_be_finite_json") from None


class RelayRequest:
    """An invocation ID and its independent cooperative cancellation event."""

    __slots__ = ("request_id", "cancel_event", "__weakref__")

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.cancel_event = Event()

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()


class RelayRequestRegistry:
    """Active state plus one content-free latest completion, explicitly no history."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._active: dict[str, tuple[RelayRequest, dict, float]] = {}
        self._last_completed: dict | None = None
        self._closed = False

    def start(
        self,
        kind: str,
        model: str,
        generation: int,
        *,
        request_id: str | None = None,
    ) -> RelayRequest:
        if not isinstance(kind, str) or not kind or not isinstance(model, str):
            raise ValueError("relay_request_identity_invalid")
        if type(generation) is not int or generation < 0:
            raise ValueError("relay_request_generation_invalid")
        request_id = request_id if request_id is not None else uuid4().hex
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("relay_request_identity_invalid")
        request = RelayRequest(request_id)
        summary = {
            "request_id": request_id,
            "kind": kind,
            "model": model,
            "generation": generation,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            if self._closed:
                raise RuntimeError("relay_request_registry_closed")
            if request_id in self._active:
                raise ValueError("relay_request_id_duplicate")
            self._active[request_id] = (request, summary, time.monotonic())
        return request

    def update(self, request_id: str, **fields: Any) -> bool:
        fields = _safe_updates(fields)
        with self._lock:
            record = self._active.get(request_id)
            if record is None:
                return False
            record[1].update(fields)
            return True

    def finish(
        self, request_id: str, *, status: str = "completed", **fields: Any
    ) -> dict | None:
        fields = _safe_updates({"status": status, **fields})
        with self._lock:
            record = self._active.pop(request_id, None)
            if record is None:
                return None
            _request, summary, started = record
            summary.update(fields)
            summary.update(
                finished_at=datetime.now(timezone.utc).isoformat(),
                elapsed_ms=max(0.0, (time.monotonic() - started) * 1000.0),
            )
            self._last_completed = deepcopy(summary)
            return deepcopy(summary)

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            record = self._active.get(request_id)
            if record is None:
                return False
            record[0].cancel_event.set()
            record[1]["status"] = "cancelling"
            return True

    def cancel_all(self) -> int:
        with self._lock:
            for request, summary, _started in self._active.values():
                request.cancel_event.set()
                summary["status"] = "cancelling"
            return len(self._active)

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self.cancel_all()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "active": [deepcopy(item[1]) for item in self._active.values()],
                "last_completed": deepcopy(self._last_completed),
            }

    def clear_last_completed(self) -> None:
        with self._lock:
            self._last_completed = None
