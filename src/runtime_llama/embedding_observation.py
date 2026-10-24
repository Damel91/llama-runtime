"""Non-owning, stream-local admission observations, never HTTP permission.

The adapter alone binds and validates a scope. A scope retains immutable
preparation metadata but no pool reference, scheduler permit or transport.
Every actual embedding POST still passes the normal fresh metadata preflight.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class _AdmissionBinding:
    owner: object
    generation: object
    scheduler: object
    pool: object
    observations: tuple[tuple[str, Any], ...]
    conservative: tuple[int | None, int | None]


class _EmbeddingAdmissionScope:
    """An inert capability for one automatic stream's local preparation.

    The lock protects clearing on the async loop against binding on a metadata
    worker. Adapter pool locks may nest this lock; this class never acquires an
    adapter lock or calls external code.
    """

    def __init__(self):
        self._lock = Lock()
        self._binding: _AdmissionBinding | None = None

    def _snapshot(self):
        with self._lock:
            return self._binding

    def _bind(self, binding: _AdmissionBinding):
        with self._lock:
            self._binding = binding

    def clear(self):
        with self._lock:
            self._binding = None
