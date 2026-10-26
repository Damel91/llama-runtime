"""Opt-in contract for a cooperating server, with no assumed HTTP endpoint.

This module does not claim that native llama.cpp implements this protocol.
An application may supply a provider only after explicitly negotiating version
1 and its coverage. A coherent snapshot reduces observations; only a server
that enforces the expected instance/loading/configuration at POST admission
can authorize reuse between POSTs. These are independent capabilities.

Providers own bounded transport and must honor finite timeout, cancellation,
``cancel_wait=True`` and reject stale conditions *before* inference. The SDK
cannot prove that a provider/server implements that promise. No server, model,
process or consumer configuration is changed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from threading import Condition, Event, get_ident
import time
from typing import Any, Callable, Protocol

from runtime_llama.client import CancellationProbe, _cancelled, _safe_failure, _timeout
from runtime_llama.errors import LLMBackendError, LLMTimeoutError

_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]+\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
__all__ = [
    "ServerObservationCapability",
    "ServerObservationScope",
    "ObservedServerReplica",
    "CoherentServerSnapshot",
    "ServerAdmissionExpectation",
    "StaleServerObservationError",
    "ServerObservationProvider",
    "ServerObservationSession",
]


def _identifier(value: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("observation identifiers must be opaque identifiers")


def _hash(value: str) -> None:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError("observation identities must be SHA-256 digests")


def _alias(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or any(
            ord(char) < 32 or 127 <= ord(char) <= 159 or 0xD800 <= ord(char) <= 0xDFFF
            for char in value
        )
    ):
        raise ValueError("observation aliases must be non-control strings")


@dataclass(frozen=True)
class ServerObservationCapability:
    """Explicit negotiation result; no inference from ordinary native props.

    Version 1 snapshots cover the complete selected logical pool, including
    unloaded entries, under one server observation revision. Conditional
    admission additionally binds every POST to the server instance, observation
    revision, replica identity, load/config generations and authority scope.
    Restarts must produce a new instance identifier; load identifiers must not
    be reused within that instance (including unload/reload of identical bytes).
    """

    version: int = 1
    coherent_snapshot: bool = True
    conditional_admission: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported server observation protocol version")
        if self.coherent_snapshot is not True:
            raise ValueError("version 1 requires a complete coherent snapshot")
        if type(self.conditional_admission) is not bool:
            raise ValueError("conditional_admission must be a boolean")


@dataclass(frozen=True)
class ServerObservationScope:
    """Opaque endpoint/authority fingerprints and the selected weight identity."""

    endpoint_id: str
    authority_id: str
    content_sha256: str
    embedding_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.endpoint_id)
        _identifier(self.authority_id)
        _hash(self.content_sha256)
        _hash(self.embedding_sha256)


@dataclass(frozen=True)
class ObservedServerReplica:
    """Complete effective embedding observations for one canonical instance."""

    replica_id: str
    alias: str
    loading_generation: str
    configuration_revision: str
    build_id: str
    tokenizer_id: str
    content_sha256: str
    embedding_sha256: str
    context_length: int
    n_batch: int
    n_ubatch: int
    slots: int
    loaded: bool = True

    def __post_init__(self) -> None:
        for value in (
            self.replica_id,
            self.loading_generation,
            self.configuration_revision,
            self.build_id,
            self.tokenizer_id,
        ):
            _identifier(value)
        _alias(self.alias)
        _hash(self.content_sha256)
        _hash(self.embedding_sha256)
        for value in (self.context_length, self.n_batch, self.n_ubatch, self.slots):
            if type(value) is not int or value <= 0:
                raise ValueError(
                    "effective observation limits must be positive integers"
                )
        if type(self.loaded) is not bool:
            raise ValueError("loaded must be a boolean")


@dataclass(frozen=True)
class CoherentServerSnapshot:
    scope: ServerObservationScope
    server_instance_id: str
    observation_revision: str
    replicas: tuple[ObservedServerReplica, ...]
    valid_for_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ServerObservationScope):
            raise ValueError("snapshot requires a typed observation scope")
        _identifier(self.server_instance_id)
        _identifier(self.observation_revision)
        if (
            not isinstance(self.replicas, tuple)
            or not self.replicas
            or not all(
                isinstance(item, ObservedServerReplica) for item in self.replicas
            )
        ):
            raise ValueError("snapshot requires a nonempty immutable replica tuple")
        if len({r.replica_id for r in self.replicas}) != len(self.replicas) or len(
            {r.alias for r in self.replicas}
        ) != len(self.replicas):
            raise ValueError("snapshot aliases and canonical instances must be unique")
        for replica in self.replicas:
            if (replica.content_sha256, replica.embedding_sha256) != (
                self.scope.content_sha256,
                self.scope.embedding_sha256,
            ):
                raise ValueError(
                    "snapshot contains a different logical embedding model"
                )
        if (
            isinstance(self.valid_for_seconds, bool)
            or not isinstance(self.valid_for_seconds, (float, int))
            or self.valid_for_seconds <= 0
            or not math.isfinite(self.valid_for_seconds)
        ):
            raise ValueError("snapshot validity must be finite and positive")


@dataclass(frozen=True)
class ServerAdmissionExpectation:
    """Server-enforced pre-inference condition, never a local scheduling epoch.

    The provider must send all fields without reducing this to a weight hash
    or a client-only check. Rejecting stale admission must happen before model
    selection/inference, including autoload and queued execution. The server
    owns atomic admission and lease lifetime through actual request drain.
    """

    version: int
    scope: ServerObservationScope
    server_instance_id: str
    observation_revision: str
    replica: ObservedServerReplica

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported server observation protocol version")
        if not isinstance(self.scope, ServerObservationScope) or not isinstance(
            self.replica, ObservedServerReplica
        ):
            raise ValueError("admission requires typed scope and replica")
        _identifier(self.server_instance_id)
        _identifier(self.observation_revision)
        if (self.replica.content_sha256, self.replica.embedding_sha256) != (
            self.scope.content_sha256,
            self.scope.embedding_sha256,
        ):
            raise ValueError("admission contains a different logical embedding model")


class StaleServerObservationError(LLMBackendError):
    """An explicit server rejection before inference; the SDK never retries it."""

    def __init__(self) -> None:
        super().__init__("llama_server_observation_stale")


class ServerObservationProvider(Protocol):
    capability: ServerObservationCapability | None

    def snapshot(
        self,
        scope: ServerObservationScope,
        *,
        timeout: float,
        cancel: CancellationProbe,
        cancel_wait: bool,
        max_response_bytes: int | None,
    ) -> CoherentServerSnapshot: ...

    def embeddings(
        self,
        inputs: tuple[str, ...],
        alias: str,
        *,
        expected: ServerAdmissionExpectation | None,
        timeout: float,
        cancel: CancellationProbe,
        cancel_wait: bool,
        max_response_bytes: int | None,
    ) -> Any: ...


class ServerObservationSession:
    """Bounded optional provider dispatch; native behavior remains explicit.

    This helper supplies no independent embedding scheduler: invoke it under
    the caller's existing shared lease and resource admission. The input tuple
    stays caller-owned; the provider owns bounded serialization and transport.
    One immutable snapshot is retained per session. Complete snapshots are
    fetched on every call unless conditional server admission was negotiated.
    Only that stronger mode permits reuse until expiry. Missing capability
    uses ``native`` before dispatch; errors and ambiguous POST outcomes are
    propagated without fallback/replay. The provider must not call close from
    an active callback; uncooperative providers cannot guarantee bounded close.
    """

    def __init__(
        self,
        scope: ServerObservationScope,
        provider: ServerObservationProvider,
        *,
        max_response_bytes: int | None = None,
    ) -> None:
        if not isinstance(scope, ServerObservationScope):
            raise ValueError("session requires a typed observation scope")
        if max_response_bytes is not None and (
            type(max_response_bytes) is not int or max_response_bytes <= 0
        ):
            raise ValueError("max_response_bytes must be a positive integer or None")
        self.scope = scope
        self.provider = provider
        self.max_response_bytes = max_response_bytes
        self._condition = Condition()
        self._stop = Event()
        self._generation = 0
        self._active: dict[int, int] = {}
        self._cached: (
            tuple[ServerObservationCapability, CoherentServerSnapshot, float] | None
        ) = None

    def revoke(self) -> None:
        """Revoke local reuse; this is not evidence of a remote load epoch."""
        with self._condition:
            self._generation += 1
            self._cached = None

    def dispatch(
        self,
        inputs: tuple[str, ...],
        alias: str,
        *,
        native: Callable[..., Any],
        timeout: float,
        cancel: CancellationProbe | None = None,
    ) -> Any:
        """Dispatch once, under one deadline, with drained callback ownership."""
        deadline = time.monotonic() + _timeout(timeout)
        if (
            not isinstance(inputs, tuple)
            or not inputs
            or not all(isinstance(item, str) for item in inputs)
        ):
            raise ValueError("inputs must be a nonempty immutable string tuple")
        _alias(alias)
        thread = get_ident()
        with self._condition:
            if self._stop.is_set():
                raise LLMBackendError("llama_server_observation_closed")
            self._active[thread] = self._active.get(thread, 0) + 1
            generation = self._generation

        def cancelled() -> bool:
            return self._stop.is_set() or (
                bool(cancel()) if cancel is not None else False
            )

        def remaining() -> float:
            _cancelled(cancelled)
            value = deadline - time.monotonic()
            if value <= 0:
                raise LLMTimeoutError("llama_server_observation_timeout")
            return value

        def call_options() -> dict[str, Any]:
            return {
                "timeout": remaining(),
                "cancel": cancelled,
                "cancel_wait": True,
                "max_response_bytes": self.max_response_bytes,
            }

        try:
            remaining()
            capability = self.provider.capability
            if capability is None:
                self.revoke()
                result = native(inputs, alias, **call_options())
                remaining()
                return result
            if not isinstance(capability, ServerObservationCapability):
                raise LLMBackendError("llama_server_observation_invalid_capability")
            with self._condition:
                cached = self._cached
            if (
                capability.conditional_admission
                and cached is not None
                and cached[0] == capability
                and cached[1].scope == self.scope
                and cached[2] > time.monotonic()
            ):
                snapshot = cached[1]
            else:
                observed_at = time.monotonic()
                snapshot = self.provider.snapshot(self.scope, **call_options())
                remaining()
                if (
                    not isinstance(snapshot, CoherentServerSnapshot)
                    or snapshot.scope != self.scope
                ):
                    raise LLMBackendError("llama_server_observation_scope_mismatch")
                expiry = observed_at + snapshot.valid_for_seconds
                if expiry <= time.monotonic():
                    raise StaleServerObservationError()
                with self._condition:
                    if generation != self._generation or self._stop.is_set():
                        raise StaleServerObservationError()
                    if capability.conditional_admission:
                        self._cached = (capability, snapshot, expiry)
            with self._condition:
                if generation != self._generation or self._stop.is_set():
                    raise StaleServerObservationError()
            replica = next((r for r in snapshot.replicas if r.alias == alias), None)
            if replica is None or not replica.loaded:
                raise LLMBackendError("llama_server_observation_replica_unavailable")
            expected = (
                ServerAdmissionExpectation(
                    capability.version,
                    self.scope,
                    snapshot.server_instance_id,
                    snapshot.observation_revision,
                    replica,
                )
                if capability.conditional_admission
                else None
            )
            result = self.provider.embeddings(
                inputs, alias, expected=expected, **call_options()
            )
            remaining()
            return result
        except StaleServerObservationError:
            self.revoke()
            raise
        except Exception as exc:
            failure = _safe_failure(exc)
            if failure is exc:
                raise
            raise failure
        finally:
            with self._condition:
                self._active[thread] -= 1
                if self._active[thread] == 0:
                    del self._active[thread]
                self._condition.notify_all()

    def close(self) -> None:
        """Signal cancellation and wait for actual synchronous callback drain."""
        with self._condition:
            if get_ident() in self._active:
                raise LLMBackendError("llama_server_observation_close_from_active_call")
            self._stop.set()
            self._generation += 1
            self._cached = None
            while self._active:
                self._condition.wait()

    def __enter__(self) -> ServerObservationSession:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
