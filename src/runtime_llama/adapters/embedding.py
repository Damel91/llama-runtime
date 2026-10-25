"""HTTP embeddings; behavioral baseline: lmstudio-agent-runtime 9300f295."""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import asdict, dataclass, field
import math
import json
import os
import secrets
from threading import BoundedSemaphore, Condition, Event, Lock
import time
from typing import Any, AsyncIterable, Literal, Sequence

from runtime_llama.client import ClientHTTPError, LlamaCppClient
from runtime_llama.embedding_metrics import EmbeddingHTTPMetrics
from runtime_llama._proxy import assert_client_proxy
from runtime_llama.configuration import (
    AdapterProfile,
    EmbeddingPrefixes,
    _finite,
    _nonnegative_int,
    _positive_int,
)
from runtime_llama.contracts import CancellationProbe, EmbeddingAdapter
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError
from runtime_llama.replica_discovery import (
    EmbeddingPool,
    ReplicaIdentity,
    attest_single_model_catalog,
    discover_embedding_pool,
    verify_loaded_replica,
    verify_explicit_pool,
)

from runtime_llama.embedding_scheduler import EmbeddingScheduler
from runtime_llama.embedding_resources import EmbeddingIntake
from runtime_llama.embedding_engine import EngineContext, PersistentEmbeddingEngine
from runtime_llama.embedding_metadata import loaded_execution_metadata
from runtime_llama.embedding_observation import (
    _AdmissionBinding,
    _EmbeddingAdmissionScope,
)
from runtime_llama.embedding_preparation import (
    PreparedEmbeddingInput,
    issue_preparation,
    valid_preparation,
)
from runtime_llama.embedding_preparation_cache import PreparationCache, PreparationKey
from runtime_llama.embedding_halving import _RejectedEmbedding, _rejected_embedding
from runtime_llama.embedding_stream import (
    EmbeddingWindowStream,
    _EmbeddingWindowBridge,
    _close_embedding_adapter,
)

_USE_CACHED_CONTEXT = object()
_NATIVE_CLIENT_CLASS = LlamaCppClient


class _CatalogObservation(dict):
    """One frozen boundary's root attestation; never an across-POST cache."""

    def __init__(self, catalog, props, client, endpoint):
        super().__init__(catalog)
        self.root_props, self.client, self.endpoint = props, client, endpoint


class _TextInputs(Sequence):
    """Freeze source references; create prefixed strings only as items are admitted."""

    def __init__(self, texts, prefix, input_type):
        self.texts = texts if isinstance(texts, tuple) else tuple(str(t) for t in texts)
        self.indices = range(len(self.texts))
        self.prefix, self.input_type = prefix, input_type

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            view = object.__new__(type(self))
            view.texts, view.indices = self.texts, self.indices[index]
            view.prefix, view.input_type = self.prefix, self.input_type
            return view
        return self.prefix + str(self.texts[self.indices[index]]), self.input_type


@dataclass
class _Replica:
    model: str
    max_in_flight: int
    configured_max_input_tokens: int | None
    batch_token_limit: int | None
    adaptive_batch_size: int
    recovery_ceiling: int
    replica_id: int | None = None
    identity: ReplicaIdentity | None = None
    context_length: int | None = None
    total_slots: int | None = None
    loaded_batch_token_limit: int | None = None
    native_microbatch_tokens: int | None = None
    tokenizer_build: str | None = None
    props_checked: bool = False
    props_identity_verified: bool = False
    vocab_size: int | None = None
    accepted_aliases: set[str] = field(default_factory=set)
    recovery_window: int = 32
    recovery_successes: int = 0
    reductions: int = 0
    generation: int = 0
    lock: Any = field(default_factory=Lock)


@dataclass(frozen=True)
class _Observation:
    context_length: int | None
    context_token_limit: int | None
    batch_token_limit: int | None
    tokenizer_build: str | None
    identity_key: tuple[str, str] | None = None
    native_batch_tokens: int | None = None
    native_microbatch_tokens: int | None = None
    total_slots: int | None = None
    vocab_size: int | None = None


@dataclass(frozen=True)
class _CountedInput:
    """Internal exact count; only public preparations incur serialization/HMAC."""

    text: str
    input_type: str
    token_count: int
    context_token_limit: int | None
    batch_token_limit: int | None
    max_input_tokens: int | None
    identity_verified: bool
    content_sha256: str | None
    embedding_sha256: str | None
    _tokenizer_build: str | None
    _policy: str
    _token_ids: tuple[int, ...] | None = None
    _text_wire_bytes: int = 0
    _tokens_wire_bytes: int | None = None


@dataclass(frozen=True)
class _DirectEmbeddingPolicy:
    """Verified representation and task policy for direct string requests."""

    content_sha256: str
    embedding_sha256: str
    tokenizer_build: str
    policy: str
    prefix: str

    def __post_init__(self):
        for name in ("content_sha256", "embedding_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in ("tokenizer_build", "policy"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.prefix, str):
            raise ValueError("prefix must be a string")


@dataclass(frozen=True)
class _DirectInput:
    """Bounded queue reservation, not a measured or estimated input count."""

    text: str
    input_type: str
    reserved_tokens: int
    context_token_limit: int | None
    batch_token_limit: int | None
    max_input_tokens: int | None
    identity_verified: bool
    content_sha256: str
    embedding_sha256: str
    _tokenizer_build: str
    _policy: str
    _direct_policy: _DirectEmbeddingPolicy
    _token_ids: None = None
    _text_wire_bytes: int = 0
    _tokens_wire_bytes: None = None

    @property
    def token_count(self):
        """Use the original engine ledger without claiming exact token usage."""
        return self.reserved_tokens


class _Call:
    def __init__(self, adapter, cancel, timeout=None):
        self.adapter, self.cancel = adapter, cancel
        value = adapter.profile.timeout if timeout is None else timeout
        if isinstance(value, bool):
            raise ValueError("timeout must be a positive finite number")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("timeout must be a positive finite number") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError("timeout must be a positive finite number")
        self.deadline = time.monotonic() + value
        self.stop = Event()
        self.wake = Event()
        self.failure: BaseException | None = None
        self.lock = Lock()
        self.dimension = adapter.embedding_dim or None

    def probe(self) -> bool:
        if self.stop.is_set() or self.adapter._closing.is_set():
            return True
        if time.monotonic() >= self.deadline:
            self.fail(LLMTimeoutError("llamacpp_embedding_call_timeout"))
            return True
        if self.cancel is None:
            return False
        try:
            return bool(self.cancel())
        except Exception as exc:
            raise LLMBackendError(
                "llama_server_cancellation_probe_failed"
            ) from RuntimeError(type(exc).__name__)

    def check(self) -> None:
        if self.adapter._closing.is_set():
            raise LLMBackendError("llamacpp_embedding_adapter_closed")
        if self.probe():
            if self.failure is not None:
                raise self.failure
            raise LLMCancelledError()

    def check_local(self) -> None:
        """Inspect owned state without invoking the consumer's cancellation hook."""
        if self.adapter._closing.is_set():
            raise LLMBackendError("llamacpp_embedding_adapter_closed")
        if time.monotonic() >= self.deadline and self.failure is None:
            self.fail(LLMTimeoutError("llamacpp_embedding_call_timeout"))
        if self.stop.is_set():
            if self.failure is not None:
                raise self.failure
            raise LLMCancelledError()

    def remaining(self) -> float:
        self.check()
        return max(0.000001, self.deadline - time.monotonic())

    def fail(self, exc: BaseException) -> None:
        with self.lock:
            if self.failure is None:
                if self.adapter._closing.is_set() and (
                    isinstance(exc, CancelledError) or type(exc) is RuntimeError
                ):
                    exc = LLMBackendError("llamacpp_embedding_adapter_closed")
                self.failure = exc
            self.stop.set()
        self.wake.set()

    def validate_dimensions(self, vectors: list[list[float]]) -> None:
        with self.lock:
            for vector in vectors:
                if self.dimension is not None and len(vector) != self.dimension:
                    raise LLMBackendError("llamacpp_embedding_dimension_mismatch")
                self.dimension = len(vector)


class _BatchCall:
    """One transport lease with immutable, independently failing members.

    An owner's work deadline is never extended. The shared HTTP deadline is
    fixed at dispatch, and its transport is stopped only after the last live
    member leaves. Owner Futures retain their share of the lease through drain.
    Consumer cancellation hooks are evaluated outside engine/scheduler locks.
    """

    def __init__(self, calls):
        self.calls = tuple(dict.fromkeys(calls))
        if not self.calls:
            raise ValueError("an embedding batch needs at least one owner")
        self.adapter = self.calls[0].adapter
        if any(call.adapter is not self.adapter for call in self.calls):
            raise ValueError("embedding batch owners must share one adapter")
        self.deadline = max(call.deadline for call in self.calls)
        self._validated_records = ()
        self._validated_observation = None
        self._record_owners = ()
        self._input_positions = ()
        self._record_input_indices = ()

    def _live(self, *, local=False):
        live = []
        for call in self.calls:
            if call.failure is not None:
                continue
            try:
                call.check_local() if local else call.check()
            except BaseException as exc:
                call.fail(exc)
            else:
                live.append(call)
        return live

    def _check(self, *, local=False):
        if self.adapter._closing.is_set():
            raise LLMBackendError("llamacpp_embedding_adapter_closed")
        if self._live(local=local):
            return
        for call in self.calls:
            if call.failure is not None:
                raise call.failure
        raise LLMCancelledError()

    def check(self):
        self._check()

    def check_local(self):
        self._check(local=True)

    def probe(self):
        return self.adapter._closing.is_set() or not self._live()

    def remaining(self):
        self.check()
        return max(0.000001, self.deadline - time.monotonic())

    def validate_dimensions(self, _vectors):
        # The shared response is parsed once; each owner's selected subset is
        # dimension-checked separately before it is returned to the engine.
        return None

    def prevalidated(self, records, observation):
        if (
            observation is not self._validated_observation
            or tuple(id(record) for record in records) != self._validated_records
        ):
            return False
        self._validated_records = ()
        self._validated_observation = None
        return True


class LlamaCppEmbeddingAdapter(EmbeddingAdapter):
    """Validate embeddings from a server whose runtime policy remains external."""

    _BATCH_RECOVERY_SUCCESS_WINDOW = 32
    _BATCH_RECOVERY_MAX_WINDOW = 1024

    def __init__(
        self, profile: AdapterProfile, *, client: LlamaCppClient | None = None
    ) -> None:
        if profile.adapter_type != "embedding":
            raise ValueError("Profile must describe an embedding role")
        if getattr(profile.model_config, "instance_count", 1) != 1:
            raise ValueError(
                "instance_count greater than one is not supported; server owns instances"
            )
        self.profile = profile
        self._embedding_prefixes = getattr(
            profile, "embedding_prefixes", EmbeddingPrefixes()
        )
        self.role_id = profile.role_id
        self.model_name = profile.model
        self.base_url = profile.base_url
        self.batch_size = profile.batch_size or 32
        self.embedding_batch_wait_sec = _finite(
            getattr(profile, "embedding_batch_wait_sec", 0.0),
            "embedding_batch_wait_sec",
        )
        if self.embedding_batch_wait_sec < 0:
            raise ValueError("embedding_batch_wait_sec must be nonnegative")
        self.embedding_max_pending_bytes = getattr(
            profile, "embedding_max_pending_bytes", 64 * 1024 * 1024
        )
        self.embedding_max_pending_tokens = getattr(
            profile, "embedding_max_pending_tokens", 1_048_576
        )
        self.embedding_max_request_bytes = getattr(
            profile, "embedding_max_request_bytes", 8 * 1024 * 1024
        )
        self.embedding_max_response_bytes = getattr(
            profile, "embedding_max_response_bytes", 8 * 1024 * 1024
        )
        self.embedding_isolation_max_requests = _positive_int(
            getattr(profile, "embedding_isolation_max_requests", 512),
            "embedding_isolation_max_requests",
        )
        if self.embedding_isolation_max_requests is None:
            raise ValueError(
                "embedding_isolation_max_requests must be a positive integer"
            )
        if (
            _positive_int(
                self.embedding_max_response_bytes, "embedding_max_response_bytes"
            )
            is None
        ):
            raise ValueError("embedding_max_response_bytes must be a positive integer")
        self.embedding_preparation_cache_entries = _nonnegative_int(
            getattr(profile, "embedding_preparation_cache_entries", 1024),
            "embedding_preparation_cache_entries",
        )
        self.embedding_preparation_cache_bytes = _nonnegative_int(
            getattr(profile, "embedding_preparation_cache_bytes", 8 * 1024 * 1024),
            "embedding_preparation_cache_bytes",
        )
        self.embedding_input_format = getattr(profile, "embedding_input_format", "auto")
        if self.embedding_input_format not in ("auto", "text", "tokens"):
            raise ValueError("embedding_input_format must be auto, text or tokens")
        self.batch_token_limit = profile.batch_token_limit
        self.token_count_parallelism = int(profile.token_count_parallelism or 1)
        self.normalize = bool(profile.normalize)
        self.configured_max_input_tokens = profile.max_input_tokens
        self.embedding_dim = int(profile.dimension or 0)
        generation = profile.generation
        proxy_url = getattr(profile.model_config, "proxy_url", None)
        assert_client_proxy(client, proxy_url)
        api_key = generation.api_key or (
            os.environ.get(generation.api_key_env) if generation.api_key_env else None
        )
        self._owns_client = client is None
        self._http_metrics = EmbeddingHTTPMetrics()
        self._client = (
            client
            if client is not None
            else LlamaCppClient(
                self.base_url,
                api_key=api_key,
                timeout=profile.timeout,
                verify_ssl=profile.model_config.verify_ssl,
                proxy_url=proxy_url,
                connection_policy=getattr(
                    profile.model_config, "http_connection_policy", "close"
                ),
            )
        )
        self._loaded_context_length: int | None = None
        self._props_checked = False
        self._closed = False
        self._closing = Event()
        self._close_lock = Lock()
        self._calls_lock = Lock()
        self._capacity_lock = Lock()
        self._calls_condition = Condition(self._calls_lock)
        self._calls: set[_Call] = set()
        self._async_window_bridge = None
        self._tokenizer_permits = BoundedSemaphore(self.token_count_parallelism)
        self._preparation_key = secrets.token_bytes(32)
        self._preparation_cache = PreparationCache(
            max_entries=self.embedding_preparation_cache_entries,
            max_bytes=max(
                0,
                min(
                    self.embedding_preparation_cache_bytes,
                    self.embedding_max_pending_bytes // 4,
                ),
            ),
        )
        self._dimension_lock = Lock()
        self._direct_token_accounting = Event()
        self._requested_embedding_parallelism = getattr(
            profile, "embedding_request_parallelism", None
        )
        self.embedding_queue_size = int(getattr(profile, "embedding_queue_size", 16))
        self.embedding_control_parallelism = max(
            1, min(4, self._requested_embedding_parallelism or 2)
        )
        self.embedding_max_pending_owners = getattr(
            profile, "embedding_max_pending_owners", None
        ) or max(
            1, (self._requested_embedding_parallelism or 2) + self.embedding_queue_size
        )
        self._intake = EmbeddingIntake(
            self.embedding_max_pending_owners, self.embedding_max_pending_bytes
        )
        self._discovery_enabled = bool(
            getattr(profile, "embedding_replica_discovery", False)
        )
        self._discovery_lock = Lock()
        self._pool_condition = Condition()
        self._pool_users = 0
        self._admission_owner = object()
        self._admission_generation = object()
        self._admission_revoked = True
        self._discovered_pool: EmbeddingPool | None = None
        self._pinned_identity: ReplicaIdentity | None = None
        self._fallback_alias: str | None = None
        self._replica_ids: dict[tuple[str, tuple[str, str] | None], int] = {}
        self._next_replica_id = 0
        self._discovery_error: str | None = None
        self._discovery_server_healthy = False
        configured_replicas = getattr(profile, "embedding_replicas", ())
        self._explicit_replicas = tuple(configured_replicas)
        self._uses_verified_pool = (
            self._discovery_enabled or len(configured_replicas) > 1
        )
        self._replicas = {
            replica.model: _Replica(
                replica.model,
                1,
                replica.max_input_tokens,
                replica.batch_token_limit,
                self.batch_size,
                self.batch_size,
            )
            for replica in configured_replicas
        } or {
            self.model_name: _Replica(
                self.model_name,
                1,
                None,
                None,
                self.batch_size,
                self.batch_size,
            )
        }
        self._primary = next(iter(self._replicas.values()))
        self._configured_replica_limits = {
            r.model: r.max_in_flight for r in configured_replicas
        }
        for replica in self._replicas.values():
            replica.accepted_aliases.add(replica.model)
        self._scheduler = EmbeddingScheduler(
            self._requested_embedding_parallelism,
            self.embedding_queue_size,
            {
                model: self._configured_replica_limits.get(model, 1)
                for model in self._replicas
            },
        )
        self.embedding_request_parallelism = self._scheduler.parallelism
        self._engine = PersistentEmbeddingEngine(self)
        self._intake.set_notify(self._engine.wake)

    def _resolve_pool(self, catalog):
        if self._discovery_enabled:
            return discover_embedding_pool(
                catalog,
                self.model_name,
                pinned=self._pinned_identity,
                fallback_alias=self._fallback_alias,
            )
        return verify_explicit_pool(
            catalog,
            [r.model for r in self._explicit_replicas],
            pinned=self._pinned_identity,
        )

    def _http_call(self, method, *args, phase="preflight", **kwargs):
        kind = {"models": "catalog", "embeddings": "embedding"}.get(method, method)
        with self._http_metrics.measure(kind, phase):
            return getattr(self._client, method)(*args, **kwargs)

    def _join_pool(
        self, call: _Call, timeout: float, *, wait_for_drain: bool = True
    ) -> bool | None:
        """Observe before admission; replace topology only after old calls drain."""
        joined, _catalog, _generation = self._join_pool_snapshot(
            call, timeout, wait_for_drain=wait_for_drain
        )
        return joined

    def _join_pool_snapshot(self, call, timeout, *, wait_for_drain=True):
        """Return the catalog and generation that governed this admission."""
        if not self._uses_verified_pool:
            return False, None, None
        deadline = time.monotonic() + timeout
        previous_scheduler = None

        def check_wait():
            call.check()
            if time.monotonic() >= deadline:
                raise LLMTimeoutError("llamacpp_embedding_discovery_wait_timeout")

        if not wait_for_drain:
            if not self._discovery_lock.acquire(blocking=False):
                return None, None, None
        else:
            while not self._discovery_lock.acquire(timeout=0.05):
                check_wait()
        try:
            call.check()
            with self._pool_condition:
                attempt_generation = self._admission_generation
            self._discovery_server_healthy = False
            health = self._http_call(
                "health",
                phase="admission",
                timeout=call.remaining(),
                cancel=call.probe,
                cancel_wait=True,
                **self._engine_response_options(),
            )
            self._discovery_server_healthy = (
                isinstance(health, dict) and health.get("status") == "ok"
            )
            if not self._discovery_server_healthy:
                raise LLMBackendError("llamacpp_embedding_server_not_healthy")
            catalog = self._discovery_catalog(
                timeout=call.remaining(), cancel=call.probe, phase="admission"
            )
            pool = self._resolve_pool(catalog)
            with self._pool_condition:
                if self._discovered_pool is None or (
                    pool.topology != self._discovered_pool.topology
                ):
                    # Fresh dispatch/recovery catalogs also revoke scopes.
                    # Stop local forks before waiting for old owners to drain.
                    if self._discovered_pool is not None:
                        self._revoke_admission_locked()
                    waited = False
                    while self._pool_users:
                        if not wait_for_drain:
                            return None, None, None
                        call.check_local()
                        if time.monotonic() >= deadline:
                            raise LLMTimeoutError(
                                "llamacpp_embedding_discovery_wait_timeout"
                            )
                        self._pool_condition.wait(0.05)
                        waited = True
                    call.check_local()
                    if waited:
                        self._pool_condition.release()
                        try:
                            catalog = self._discovery_catalog(
                                timeout=call.remaining(),
                                cancel=call.probe,
                                phase="admission",
                            )
                            pool = self._resolve_pool(catalog)
                        finally:
                            self._pool_condition.acquire()
                        call.check_local()
                    replicas = {}
                    for observed in pool.replicas:
                        key = (
                            observed.model,
                            observed.identity.key if observed.identity else None,
                        )
                        if key not in self._replica_ids:
                            self._replica_ids[key] = self._next_replica_id
                            self._next_replica_id += 1
                        configured = [
                            r
                            for r in self._explicit_replicas
                            if r.model in observed.aliases
                        ]
                        replicas[observed.model] = _Replica(
                            observed.model,
                            1,
                            self._minimum(*(r.max_input_tokens for r in configured)),
                            self._minimum(*(r.batch_token_limit for r in configured)),
                            self.batch_size,
                            self.batch_size,
                            replica_id=self._replica_ids[key],
                            identity=pool.identity,
                            accepted_aliases=set(observed.aliases),
                        )
                    active_keys = {
                        (
                            observed.model,
                            observed.identity.key if observed.identity else None,
                        )
                        for observed in pool.replicas
                    }
                    retired = [
                        key for key in self._replica_ids if key not in active_keys
                    ]
                    for key in retired[:-1024]:
                        del self._replica_ids[key]
                    scheduler = EmbeddingScheduler(
                        self._requested_embedding_parallelism,
                        self.embedding_queue_size,
                        {
                            r.model: min(
                                (
                                    c.max_in_flight
                                    for c in self._explicit_replicas
                                    if c.model in r.aliases
                                ),
                                default=1,
                            )
                            for r in pool.replicas
                        },
                    )
                    previous_scheduler = self._scheduler
                    self._scheduler = scheduler
                    self.embedding_request_parallelism = scheduler.parallelism
                    self._replicas = replicas
                    self._primary = next(iter(replicas.values()))
                    self._loaded_context_length = None
                    self._discovered_pool = pool
                    if pool.identity is not None:
                        self._pinned_identity = pool.identity
                    if pool.fallback_alias is not None:
                        self._fallback_alias = pool.fallback_alias
                # Only a serialized fresh admission may enable local reuse.
                # A newer topology observation during this attempt wins, even
                # when it came from a dispatch or recovery worker. In that
                # case a subsequent stable fresh join can enable the scope.
                if attempt_generation is self._admission_generation:
                    if self._admission_revoked:
                        self._admission_generation = object()
                    self._admission_revoked = False
                self._pool_users += 1
                self._discovery_error = None
                generation = (
                    self._admission_generation if not self._admission_revoked else None
                )
            return True, catalog, generation
        except (LLMBackendError, LLMTimeoutError, LLMCancelledError) as exc:
            self._discovery_error = type(exc).__name__
            with self._pool_condition:
                self._revoke_admission_locked()
            raise
        finally:
            if previous_scheduler is not None:
                previous_scheduler.close()
            self._discovery_lock.release()
            # Public readiness/discovery calls do not produce engine events.
            # A failed internal nonblocking attempt must not wake itself into
            # a repeated health/catalog loop while the same pool stays busy.
            if wait_for_drain:
                self._engine.metadata_available()

    def _leave_pool(self, joined: bool) -> None:
        if joined:
            with self._pool_condition:
                self._pool_users -= 1
                drained = self._pool_users == 0
                self._pool_condition.notify_all()
            if drained:
                self._engine.metadata_available()

    def _verify_discovered_replica(
        self, replica: _Replica, *, timeout, cancel, catalog=None, phase="preflight"
    ):
        models = getattr(self._client, "models", None)
        if not callable(models):
            if self._uses_verified_pool:
                raise LLMBackendError("llamacpp_embedding_invalid_model_catalog")
            return None
        if catalog is None:
            catalog = (
                self._discovery_catalog(timeout=timeout, cancel=cancel, phase=phase)
                if self._uses_verified_pool
                else self._http_call(
                    "models",
                    phase=phase,
                    timeout=timeout,
                    cancel=cancel,
                    cancel_wait=True,
                    **self._engine_response_options(),
                )
            )
        if self._uses_verified_pool:
            verify_loaded_replica(catalog, replica.model, replica.identity)
        entries = catalog.get("data") if isinstance(catalog, dict) else None
        matches = self._catalog_matches(replica.model, entries)
        if len(matches) == 1:
            entry = matches[0]
            if "status" in entry:
                status = entry["status"]
                state = status.get("value") if isinstance(status, dict) else status
                if state != "loaded":
                    raise LLMBackendError("llamacpp_embedding_model_not_loaded")
            return entry
        if self._uses_verified_pool:
            raise LLMBackendError("llamacpp_embedding_invalid_model_catalog")
        return None

    def _discovery_catalog(self, *, timeout, cancel, phase="preflight"):
        try:
            catalog = self._http_call(
                "models",
                phase=phase,
                timeout=timeout,
                cancel=cancel,
                cancel_wait=True,
                **self._engine_response_options(),
            )
            entries = catalog.get("data") if isinstance(catalog, dict) else None
            if (
                isinstance(entries, list)
                and len(entries) == 1
                and isinstance(entries[0], dict)
                and "status" not in entries[0]
            ):
                # Native single-model servers omit status. Root /props does
                # not select a router model or load an unloaded alias.
                props = self._http_call(
                    "props",
                    phase=phase,
                    timeout=timeout,
                    cancel=cancel,
                    cancel_wait=True,
                    **self._engine_response_options(),
                )
                catalog = attest_single_model_catalog(catalog, props)
                if catalog.get("data", [{}])[0].get("status") == {"value": "loaded"}:
                    catalog = _CatalogObservation(
                        catalog, props, self._client, self.base_url
                    )
            self._observe_admission_topology(catalog)
            return catalog
        except BaseException:
            with self._pool_condition:
                self._revoke_admission_locked()
            raise

    def _revoke_admission_locked(self):
        """Invalidate every non-owning scope; caller owns the pool condition."""
        self._admission_generation = object()
        self._admission_revoked = True

    def _observe_admission_topology(self, catalog):
        """Fresh catalogs revoke scopes for changes beyond the selected alias."""
        with self._pool_condition:
            current = self._discovered_pool
        if current is None:
            return
        observed = self._resolve_pool(catalog)
        with self._pool_condition:
            if (
                self._discovered_pool is not current
                or observed.topology != current.topology
            ):
                self._revoke_admission_locked()

    def _engine_bind_admission_scope(self, scope, context):
        """Capture preparation metadata without retaining an admission ref."""
        if not isinstance(scope, _EmbeddingAdmissionScope):
            raise ValueError("invalid embedding admission scope")
        with self._pool_condition:
            if (
                not self._discovery_enabled
                or not context.joined
                or self._admission_revoked
                or self._discovered_pool is None
                or context.admission_generation is not self._admission_generation
                or self._closing.is_set()
            ):
                scope.clear()
                return False
            scope._bind(
                _AdmissionBinding(
                    self._admission_owner,
                    self._admission_generation,
                    self._scheduler,
                    self._discovered_pool,
                    tuple(context.observations.items()),
                    context.conservative,
                )
            )
            return True

    def _engine_fork_admission_scope(self, calls, active):
        """Fork only actual direct owners; fresh HTTP preflight stays mandatory."""
        with self._pool_condition:
            if (
                self._admission_revoked
                or self._discovered_pool is None
                or self._closing.is_set()
            ):
                return None
            bindings = {}
            for call in active:
                scope = getattr(call, "_admission_scope", None)
                if (
                    not isinstance(scope, _EmbeddingAdmissionScope)
                    or getattr(call, "_direct_policy", None) is None
                ):
                    return None
                binding = scope._snapshot()
                if (
                    binding is None
                    or binding.owner is not self._admission_owner
                    or binding.generation is not self._admission_generation
                    or binding.scheduler is not self._scheduler
                    or binding.pool is not self._discovered_pool
                ):
                    return None
                bindings[call] = binding
            contexts = tuple(
                None
                if call.failure is not None
                else EngineContext(
                    True,
                    dict(bindings[call].observations),
                    bindings[call].conservative,
                    self._admission_generation,
                )
                for call in calls
            )
            self._pool_users += sum(context is not None for context in contexts)
            return contexts

    @classmethod
    def from_profile(
        cls, profile: AdapterProfile, *, client: LlamaCppClient | None = None
    ):
        return cls(profile, client=client)

    @staticmethod
    def _positive_int(value: Any) -> int | None:
        return (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
            else None
        )

    def _refresh_context(
        self,
        *,
        replica: _Replica | None = None,
        timeout=None,
        cancel=None,
        catalog=None,
    ) -> int | None:
        return self._observe_replica(
            replica=replica, timeout=timeout, cancel=cancel, catalog=catalog
        ).context_length

    def _observe_replica(
        self,
        *,
        replica: _Replica | None = None,
        timeout=None,
        cancel=None,
        call=None,
        catalog=None,
        phase="preflight",
    ) -> _Observation:
        replica = replica or self._primary
        remaining = call.remaining if call is not None else lambda: timeout
        entry = self._verify_discovered_replica(
            replica, timeout=remaining(), cancel=cancel, catalog=catalog, phase=phase
        )
        loaded_context_length = None
        replica.props_checked = True
        self._props_checked = True
        try:
            if (
                isinstance(catalog, _CatalogObservation)
                and catalog.client is self._client
                and catalog.endpoint == self.base_url
                and len(catalog["data"]) == 1
                and replica.model in self._catalog_aliases(catalog["data"][0])
            ):
                props = catalog.root_props
            else:
                props = self._http_call(
                    "props",
                    phase=phase,
                    model=replica.model,
                    timeout=remaining(),
                    cancel=cancel,
                    cancel_wait=True,
                    **self._engine_response_options(),
                )
        except Exception as exc:
            self._invalidate_replica(replica)
            if not isinstance(exc, ClientHTTPError) or exc.status_code not in {
                404,
                405,
            }:
                raise
            props = None
        if isinstance(props, dict):
            alias = props.get("model_alias")
            if props.get("is_sleeping") is True:
                self._invalidate_replica(replica)
                raise LLMBackendError("llamacpp_embedding_model_sleeping")
            if entry is not None:
                replica.accepted_aliases.update(self._catalog_aliases(entry))
            if alias is not None and (
                not isinstance(alias, str) or alias not in replica.accepted_aliases
            ):
                self._invalidate_replica(replica)
                raise LLMBackendError("llamacpp_embedding_model_metadata_mismatch")
            settings = props.get("default_generation_settings")
            if isinstance(settings, dict):
                loaded_context_length = self._positive_int(settings.get("n_ctx"))
            total_slots = self._positive_int(props.get("total_slots"))
            build = props.get("build_info")
            build = build if isinstance(build, str) and build else None
        else:
            alias, total_slots, build = None, None, None
        try:
            native = loaded_execution_metadata(props, entry)
        except LLMBackendError:
            self._invalidate_replica(replica)
            raise
        context_limit = self._input_limit(loaded_context_length, replica=replica)
        batch_limit = self._minimum(self.batch_token_limit, replica.batch_token_limit)
        vocab_size = None
        if replica.identity is not None and isinstance(entry, dict):
            metadata = entry.get("meta")
            if isinstance(metadata, dict) and "n_vocab" in metadata:
                vocab_size = self._positive_int(metadata["n_vocab"])
                if vocab_size is None:
                    raise LLMBackendError("llamacpp_embedding_invalid_tokenizer_vocab")
        with replica.lock:
            replica.context_length = loaded_context_length
            replica.total_slots = total_slots
            replica.props_identity_verified = isinstance(alias, str)
            replica.loaded_batch_token_limit = native.n_batch
            replica.native_microbatch_tokens = native.n_ubatch
            replica.tokenizer_build = build
            replica.vocab_size = vocab_size
            if replica is self._primary:
                self._loaded_context_length = loaded_context_length
        # Native sequences do not multiply original POST leases on a replica.
        # Observations publish native sequence counts independently. HTTP
        # admission remains one lease even when concurrent props arrive late.
        # No network request is made while holding this lock.
        with self._capacity_lock:
            replica.max_in_flight = 1
        return _Observation(
            loaded_context_length,
            context_limit,
            batch_limit,
            build,
            replica.identity.key if replica.identity else None,
            native.n_batch,
            native.n_ubatch,
            native.total_slots,
            vocab_size,
        )

    @staticmethod
    def _minimum(*values):
        return min((value for value in values if value is not None), default=None)

    def _invalidate_replica(self, replica: _Replica) -> None:
        replica.context_length = None
        replica.total_slots = None
        replica.loaded_batch_token_limit = None
        replica.native_microbatch_tokens = None
        replica.tokenizer_build = None
        replica.vocab_size = None
        replica.props_identity_verified = False
        if replica is self._primary:
            self._loaded_context_length = None

    @staticmethod
    def _catalog_aliases(entry: dict[str, Any]) -> set[str]:
        values = [entry.get("id")]
        if isinstance(entry.get("aliases"), list):
            values.extend(entry["aliases"])
        return {value for value in values if isinstance(value, str)}

    @classmethod
    def _catalog_matches(cls, model: str, entries: Any) -> list[dict[str, Any]]:
        if not isinstance(entries, list):
            return []
        return [
            entry
            for entry in entries
            if isinstance(entry, dict) and model in cls._catalog_aliases(entry)
        ]

    def _input_limit(
        self,
        loaded_context_length: Any = _USE_CACHED_CONTEXT,
        *,
        replica: _Replica | None = None,
    ) -> int | None:
        replica = replica or self._primary
        observed = (
            replica.context_length
            if loaded_context_length is _USE_CACHED_CONTEXT
            else loaded_context_length
        )
        limits = [
            self._positive_int(value)
            for value in (
                self.configured_max_input_tokens,
                replica.configured_max_input_tokens,
                self.profile.model_config.context_length,
                observed,
            )
        ]
        return (
            min(value for value in limits if value is not None)
            if any(value is not None for value in limits)
            else None
        )

    @property
    def max_input_tokens(self) -> int | None:
        joined = self._join_pool(_Call(self, None), self.profile.timeout)
        try:
            return self._observed_input_limit()
        finally:
            self._leave_pool(joined)

    def _observed_input_limit(self) -> int | None:
        for replica in self._replicas.values():
            if not replica.props_checked and not self._closed:
                try:
                    self._refresh_context(replica=replica, timeout=self.profile.timeout)
                except LLMTimeoutError:
                    pass
        limits = [
            self._input_limit(replica=replica) for replica in self._replicas.values()
        ]
        return min((value for value in limits if value is not None), default=None)

    def _policy(self, input_type: str) -> str:
        prefix = (
            "" if input_type == "raw" else getattr(self._embedding_prefixes, input_type)
        )
        return "add_special=true;parse_special=true;prefix=" + prefix

    def _tokenize_input(
        self, text: str, replica: _Replica, call: _Call
    ) -> tuple[int, ...]:
        payload = {
            "content": text,
            "model": replica.model,
            "add_special": True,
            "parse_special": True,
            "with_pieces": False,
        }
        # JSON escaping and the tokenizer envelope also consume wire bytes.
        if (
            len(json.dumps(payload, ensure_ascii=True))
            > self.embedding_max_request_bytes
        ):
            raise LLMBackendError("llamacpp_embedding_request_bytes_limit")
        acquired = False
        try:
            while not acquired:
                call.check()
                acquired = self._tokenizer_permits.acquire(
                    timeout=min(0.05, call.remaining())
                )
            result = self._http_call(
                "tokenize",
                text,
                phase="preparation",
                model=replica.model,
                add_special=True,
                parse_special=True,
                with_pieces=False,
                timeout=call.remaining(),
                cancel=call.probe,
                cancel_wait=True,
                **self._engine_response_options(),
            )
            call.check()
            tokens = result.get("tokens") if isinstance(result, dict) else None
            if not isinstance(tokens, list) or any(
                not isinstance(token, int)
                or isinstance(token, bool)
                or not 0 <= token <= 2**31 - 1
                for token in tokens
            ):
                raise LLMBackendError("llamacpp_embedding_invalid_tokenization")
            return tuple(tokens)
        finally:
            if acquired:
                self._tokenizer_permits.release()

    def _token_count(self, text: str, replica: _Replica, call: _Call) -> int:
        return len(self._tokenize_input(text, replica, call))

    def _engine_response_options(self, call=None):
        # Custom clients own decoding and may not expose the native bound.
        return (
            {
                "max_response_bytes": min(
                    self.embedding_max_response_bytes,
                    getattr(
                        call,
                        "_embedding_response_limit",
                        self.embedding_max_response_bytes,
                    ),
                )
            }
            if isinstance(self._client, _NATIVE_CLIENT_CLASS)
            or getattr(self._client, "supports_bounded_json", False) is True
            else {}
        )

    def _embedding_input(self, texts, records, replica, observation):
        if records is not None and any(
            isinstance(record, _DirectInput) for record in records
        ):
            return texts
        if records is not None and observation is not None:
            for record in records:
                self._validate_vocab(record._token_ids, observation)
        if self.embedding_input_format == "text":
            return texts
        usable = (
            getattr(self._client, "supports_embedding_token_ids", None) is True
            and observation is not None
            and records is not None
            and len(records) == len(texts)
            and all(
                self._can_reuse(record, replica, observation)
                and isinstance(record._token_ids, tuple)
                and len(record._token_ids) == record.token_count
                and bool(record._token_ids)
                for record in records
            )
        )
        if usable:
            return [list(record._token_ids) for record in records]
        if self.embedding_input_format == "tokens":
            raise LLMBackendError("llamacpp_embedding_token_input_unavailable")
        return texts

    @staticmethod
    def _validate_vocab(token_ids, observation):
        if (
            token_ids is not None
            and observation.vocab_size is not None
            and any(token >= observation.vocab_size for token in token_ids)
        ):
            raise LLMBackendError("llamacpp_embedding_invalid_tokenization")

    def _can_reuse(self, record, replica, observation):
        identity = replica.identity
        return (
            record.identity_verified
            and identity is not None
            and record.content_sha256 == identity.content_sha256
            and record.embedding_sha256 == identity.embedding_sha256
            and record._policy == self._policy(record.input_type)
            and observation.tokenizer_build is not None
            and record._tokenizer_build == observation.tokenizer_build
        )

    def _engine_prepare(self, item, model, call, observation, conservative):
        """Prepare one record on a bounded tokenizer worker, never the coordinator."""
        replica = self._replicas[model]
        context_limit = self._minimum(observation.context_token_limit, conservative[0])
        batch_limit = self._minimum(observation.batch_token_limit, conservative[1])
        call.check()
        direct = getattr(call, "_direct_policy", None)
        if direct is not None:
            if isinstance(item, PreparedEmbeddingInput):
                raise LLMBackendError("llamacpp_embedding_invalid_direct_input")
            text, input_type = item
            self._validate_direct_observation(direct, input_type, replica, observation)
            context_limit = self._minimum(
                context_limit, observation.native_microbatch_tokens
            )
            if not text.startswith(direct.prefix):
                raise LLMBackendError("llamacpp_embedding_direct_prefix_mismatch")
            # The direct path never measures or estimates this input. Reserve
            # a complete observed input allowance solely for bounded intake.
            # The native server remains authoritative for actual token usage.
            reservation = context_limit
            call.check()
            return _DirectInput(
                text=text,
                input_type=input_type,
                reserved_tokens=reservation,
                context_token_limit=context_limit,
                batch_token_limit=batch_limit,
                max_input_tokens=self._minimum(context_limit, batch_limit),
                identity_verified=True,
                content_sha256=direct.content_sha256,
                embedding_sha256=direct.embedding_sha256,
                _tokenizer_build=direct.tokenizer_build,
                _policy=direct.policy,
                _direct_policy=direct,
                _text_wire_bytes=len(json.dumps(text, ensure_ascii=True)),
            )
        if isinstance(item, PreparedEmbeddingInput):
            text, input_type = item.text, item.input_type
        else:
            text, input_type = item
        identity = replica.identity
        computed = False
        cache_key = (
            PreparationKey(
                text=text,
                input_type=input_type,
                policy=self._policy(input_type),
                content_sha256=identity.content_sha256,
                embedding_sha256=identity.embedding_sha256,
                tokenizer_build=observation.tokenizer_build,
                vocab_size=observation.vocab_size,
            )
            if identity is not None
            and observation.identity_key == identity.key
            and type(text) is str
            and type(input_type) is str
            and type(observation.tokenizer_build) is str
            and bool(observation.tokenizer_build)
            and (observation.vocab_size is None or type(observation.vocab_size) is int)
            else None
        )
        if isinstance(item, PreparedEmbeddingInput) and self._can_reuse(
            item, replica, observation
        ):
            count, token_ids = item.token_count, item._token_ids
            if token_ids is None and cache_key is not None:
                token_ids = self._preparation_cache.get(cache_key)
                if token_ids is not None and len(token_ids) != count:
                    raise LLMBackendError(
                        "llamacpp_embedding_preparation_cache_mismatch"
                    )
        else:
            token_ids = (
                self._preparation_cache.get(cache_key)
                if cache_key is not None
                else None
            )
            if token_ids is None:
                token_ids = self._tokenize_input(text, replica, call)
                computed = True
            count = len(token_ids)
        call.check()
        self._validate_vocab(token_ids, observation)
        if computed and cache_key is not None:
            self._preparation_cache.put(cache_key, token_ids)
        retained = (
            token_ids
            if identity is not None
            and observation.tokenizer_build is not None
            and (context_limit is None or count <= context_limit)
            and (batch_limit is None or count <= batch_limit)
            else None
        )
        return _CountedInput(
            text=text,
            input_type=input_type,
            token_count=count,
            context_token_limit=context_limit,
            batch_token_limit=batch_limit,
            max_input_tokens=self._minimum(context_limit, batch_limit),
            identity_verified=identity is not None,
            content_sha256=identity.content_sha256 if identity else None,
            embedding_sha256=identity.embedding_sha256 if identity else None,
            _tokenizer_build=observation.tokenizer_build,
            _policy=self._policy(input_type),
            _token_ids=retained,
            _text_wire_bytes=len(json.dumps(text, ensure_ascii=True)),
            _tokens_wire_bytes=(
                2
                + sum(len(str(token)) for token in retained)
                + 2 * max(0, len(retained) - 1)
                if retained is not None
                else None
            ),
        )

    @staticmethod
    def _admit(record, observation):
        count = record.token_count
        limit = observation.context_token_limit
        if limit is not None and count > limit:
            raise LLMBackendError(
                f"llamacpp_embedding_input_exceeds_context:tokens={count}:limit={limit}"
            )
        limit = observation.batch_token_limit
        if limit is not None and count > limit:
            raise LLMBackendError(
                f"llamacpp_embedding_input_exceeds_batch_budget:tokens={count}:limit={limit}"
            )

    @staticmethod
    def _normalize(vector: list[float], scale: float | None = None) -> list[float]:
        scale = max(abs(value) for value in vector) if scale is None else scale
        if scale == 0:
            return vector
        for index, value in enumerate(vector):
            vector[index] = value / scale
        norm = math.sqrt(math.fsum(value * value for value in vector))
        for index, value in enumerate(vector):
            vector[index] = value / norm
        return vector

    def _parse_batch_response(
        self, payload: Any, *, expected_count: int
    ) -> list[list[float]]:
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != expected_count:
            raise LLMBackendError("llamacpp_embedding_count_or_response_mismatch")
        indexed: dict[int, list[float]] = {}
        expected_dimension = None
        for position, item in enumerate(data):
            if not isinstance(item, dict):
                raise LLMBackendError("llamacpp_embedding_invalid_response")
            index = item.get("index", position)
            raw = item.get("embedding")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < expected_count
                or index in indexed
                or not isinstance(raw, list)
                or not raw
            ):
                raise LLMBackendError("llamacpp_embedding_invalid_response")
            vector, scale = [], 0.0
            try:
                for value in raw:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError("invalid vector member")
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError("nonfinite vector member")
                    vector.append(number)
                    scale = max(scale, abs(number))
            except (ValueError, TypeError, OverflowError) as exc:
                raise LLMBackendError("llamacpp_embedding_invalid_response") from exc
            if expected_dimension is not None and len(vector) != expected_dimension:
                raise LLMBackendError("llamacpp_embedding_dimension_mismatch")
            expected_dimension = len(vector)
            indexed[index] = (
                self._normalize(vector, scale) if self.normalize else vector
            )
        return [indexed[index] for index in range(expected_count)]

    def _tighten_adaptive_batch_size(self, value: int, *, replica: _Replica) -> None:
        with replica.lock:
            current = replica.adaptive_batch_size
            if value < current:
                replica.adaptive_batch_size = max(1, value)
                if replica.reductions:
                    replica.recovery_window = min(
                        self._BATCH_RECOVERY_MAX_WINDOW, replica.recovery_window * 2
                    )
                replica.reductions += 1
                replica.recovery_successes = 0
                replica.generation += 1

    def _record_batch_success(
        self, size: int, generation: int, replica: _Replica
    ) -> None:
        with replica.lock:
            current, ceiling = replica.adaptive_batch_size, replica.recovery_ceiling
            if generation != replica.generation or current >= ceiling or size < current:
                return
            replica.recovery_successes += 1
            if replica.recovery_successes >= replica.recovery_window:
                replica.adaptive_batch_size = min(ceiling, current * 2)
                replica.recovery_successes = 0
                replica.generation += 1

    def _current_batch_size(self, remaining: int, replica: _Replica) -> int:
        with replica.lock:
            return min(remaining, replica.adaptive_batch_size)

    def _request_batch_adaptive(
        self,
        texts: list[str],
        *,
        replica: _Replica,
        timeout=None,
        cancel=None,
        recovery_child: bool = False,
        call: _Call | None = None,
        observation: _Observation | None = None,
        records: Sequence[_CountedInput] | None = None,
        wire_input: list | None = None,
    ) -> list[list[float] | _RejectedEmbedding]:
        # Recovery shares one pending set. A fresh observation is authoritative
        # for every unsent record, while historical limits remain record-local.
        pending = set(range(len(texts)))
        admitted = None
        results = [None] * len(texts)
        owners = call._record_owners if isinstance(call, _BatchCall) else ()

        def live(index):
            return not owners or owners[index].failure is None

        def send(indices, recovering, observed=None, isolation=None):
            nonlocal admitted
            current_timeout = timeout
            http_limit = None
            if call is not None:
                call.check()
                indices = tuple(index for index in indices if live(index))
                if not indices:
                    return
                if observed is None:
                    observed = self._observe_replica(
                        replica=replica,
                        cancel=call.probe,
                        call=call,
                        phase="recovery" if recovering else "preflight",
                    )
                current_timeout = call.remaining()
                if records is not None:
                    # The complete immutable snapshot includes vocabulary and
                    # identity. A sibling's smaller historical limit must not
                    # hide a server-side reduction or tokenizer change.
                    if observed != admitted:
                        prevalidated = isinstance(
                            call, _BatchCall
                        ) and call.prevalidated(records, observed)
                        for index in () if prevalidated else pending:
                            if not live(index):
                                continue
                            record = records[index]
                            try:
                                if (
                                    record.identity_verified
                                    and observed.identity_key
                                    != (
                                        record.content_sha256,
                                        record.embedding_sha256,
                                    )
                                ):
                                    raise LLMBackendError(
                                        "llamacpp_embedding_content_identity_changed"
                                    )
                                self._engine_validate(record, replica.model, observed)
                            except BaseException as exc:
                                if not owners:
                                    raise
                                owners[index].fail(exc)
                        admitted = observed
                    call.check()
                    indices = tuple(index for index in indices if live(index))
                    if not indices:
                        return
                    http_limit = self._minimum(
                        observed.batch_token_limit,
                        *(records[i].batch_token_limit for i in indices),
                    )
                    if (
                        http_limit is not None
                        and sum(records[index].token_count for index in indices)
                        > http_limit
                    ):
                        cursor = 0
                        while cursor < len(indices):
                            end, tokens = cursor, 0
                            child_limit = observed.batch_token_limit
                            while end < len(indices):
                                record = records[indices[end]]
                                limit = self._minimum(
                                    child_limit, record.batch_token_limit
                                )
                                count = record.token_count
                                if (
                                    end > cursor
                                    and limit is not None
                                    and tokens + count > limit
                                ):
                                    break
                                tokens += count
                                child_limit = limit
                                end += 1
                            send(indices[cursor:end], True, isolation=isolation)
                            cursor = end
                        return
            if cancel is not None and cancel():
                raise LLMCancelledError()
            with replica.lock:
                generation = replica.generation
            post_call = (
                _BatchCall(owners[index] for index in indices)
                if isolation is not None and owners
                else call
            )
            post_cancel = cancel
            try:
                if post_call is not call:
                    # A descendant may contain only one of a shared POST's
                    # owners. Drain it when those participants withdraw,
                    # without waiting for owners of later sibling subsets.
                    current_timeout = post_call.remaining()
                    post_cancel = post_call.probe
                if isolation is not None:
                    if isolation[0] >= self.embedding_isolation_max_requests:
                        raise LLMBackendError(
                            "llamacpp_embedding_isolation_request_limit"
                        ) from isolation[1]
                    isolation[0] += 1
                payload = self._http_call(
                    "embeddings",
                    [wire_input[index] for index in indices]
                    if wire_input is not None
                    else self._embedding_input(
                        [texts[index] for index in indices],
                        None
                        if records is None
                        else [records[index] for index in indices],
                        replica,
                        observed,
                    ),
                    phase="recovery" if recovering else "preflight",
                    model=replica.model,
                    timeout=current_timeout,
                    cancel=post_cancel,
                    cancel_wait=True,
                    **self._engine_response_options(call),
                )
            except (LLMCancelledError, LLMTimeoutError):
                if post_call is not call and not post_call._live():
                    pending.difference_update(indices)
                    call.check()
                    return
                raise
            except ClientHTTPError as exc:
                if exc.status_code == 500 and records is not None:
                    direct = tuple(
                        index
                        for index in indices
                        if live(index) and isinstance(records[index], _DirectInput)
                    )
                    if direct:
                        direct_set = set(direct)
                        exact = tuple(
                            index for index in indices if index not in direct_set
                        )
                        if exact:
                            if not owners:
                                # Unowned mixed internal calls cannot safely
                                # retire only the exact participant.
                                raise
                            for index in exact:
                                owners[index].fail(exc)
                        direct = tuple(index for index in direct if live(index))
                        if not direct:
                            return
                        if len(indices) == len(direct) == 1:
                            index = direct[0]
                            record = records[index]
                            rejection = _rejected_embedding(
                                record.text[len(record._direct_policy.prefix) :],
                                record.input_type,
                                exc,
                            )
                            if rejection.char_count < 2:
                                if not owners:
                                    raise
                                owners[index].fail(exc)
                            else:
                                results[index] = rejection
                            pending.discard(index)
                            return
                        # Only failed native POSTs are bisected. Successful
                        # descendants keep their vectors; no tokenization,
                        # character-budget update or replay of exact rows.
                        budget = [0, exc] if isolation is None else isolation
                        if exact:
                            send(direct, True, isolation=budget)
                            return
                        midpoint = max(1, len(direct) // 2)
                        send(direct[:midpoint], True, isolation=budget)
                        if midpoint < len(direct):
                            send(direct[midpoint:], True, isolation=budget)
                        return
                if records is not None and any(
                    isinstance(records[index], _DirectInput) for index in indices
                ):
                    # Direct requests leave actual token admission to the
                    # native server. Never replay or silently truncate a reject.
                    raise
                size = len(indices)
                if (
                    getattr(self._client, "supports_embedding_token_ids", None) is True
                    or exc.status_code != 400
                    or getattr(exc, "error_type", None) != "exceed_context_size_error"
                    or size <= 1
                ):
                    raise
                midpoint = size // 2
                self._tighten_adaptive_batch_size(
                    max(midpoint, size - midpoint), replica=replica
                )
                send(indices[:midpoint], True)
                send(indices[midpoint:], True)
                return
            vectors = self._parse_batch_response(payload, expected_count=len(indices))
            for index, vector in zip(indices, vectors):
                results[index] = vector
            pending.difference_update(indices)
            if not recovering:
                self._record_batch_success(len(indices), generation, replica)

        send(tuple(range(len(texts))), recovery_child, observation)
        # A withdrawn shared owner has no result. Its empty internal row is
        # discarded by _engine_run_shared and never published to a consumer.
        return [vector if vector is not None else [] for vector in results]

    def _batch_now(self):
        """Clock isolated from cancellation/deadline clocks for deterministic checks."""
        return time.monotonic()

    @staticmethod
    def _validate_task(input_type):
        if not isinstance(input_type, str) or input_type not in (
            "document",
            "query",
            "raw",
        ):
            raise ValueError("input_type must be document, query or raw")

    def _engine_start(self, call):
        contexts = self._engine_start_shared((call,))
        if contexts is None:
            return None
        if contexts[0] is None:
            call.check()
        return contexts[0]

    def _engine_start_shared(self, calls):
        """Observe one frozen admission cohort; each returned context owns a ref."""
        calls = tuple(calls)
        if len(set(calls)) != len(calls):
            raise LLMBackendError("llamacpp_embedding_duplicate_admission_owner")
        for call in calls:
            if call.failure is not None:
                continue
            try:
                call.check()
                for item in getattr(call, "prepared_inputs", ()):
                    call.check()
                    if not valid_preparation(self._preparation_key, item):
                        raise LLMBackendError("llamacpp_embedding_invalid_preparation")
            except BaseException as exc:
                call.fail(exc)
        active = tuple(call for call in calls if call.failure is None)
        if not active:
            return (None,) * len(calls)
        scoped = self._engine_fork_admission_scope(calls, active)
        if scoped is not None:
            return scoped
        group = self._engine_batch_call(active)
        joined, catalog, admission_generation = self._join_pool_snapshot(
            group, group.remaining(), wait_for_drain=False
        )
        if joined is None:
            return None
        staging = joined
        try:
            if catalog is None:
                catalog = self._engine_catalog_shared(active, phase="admission")
            observations = {
                model: self._observe_replica(
                    replica=replica,
                    cancel=group.probe,
                    call=group,
                    catalog=catalog,
                    phase="admission",
                )
                for model, replica in tuple(self._replicas.items())
            }
            conservative = (
                self._minimum(
                    *(item.context_token_limit for item in observations.values())
                ),
                self._minimum(
                    *(item.batch_token_limit for item in observations.values())
                ),
            )
            contexts = []
            for call in calls:
                if call.failure is None:
                    try:
                        call.check()
                    except BaseException as exc:
                        call.fail(exc)
                contexts.append(
                    None
                    if call.failure is not None
                    else EngineContext(
                        joined,
                        dict(observations),
                        conservative,
                        admission_generation,
                    )
                )
            contexts = tuple(contexts)
            accepted = sum(context is not None for context in contexts)
            if joined:
                # The staging ref kept this generation alive during HTTP.
                # Transfer it atomically to the independently drained owners.
                with self._pool_condition:
                    self._pool_users += accepted - 1
                    self._pool_condition.notify_all()
                staging = False
            for call, context in zip(calls, contexts):
                scope = getattr(call, "_admission_scope", None)
                if (
                    context is not None
                    and getattr(call, "_direct_policy", None) is not None
                    and isinstance(scope, _EmbeddingAdmissionScope)
                ):
                    self._engine_bind_admission_scope(scope, context)
            return contexts
        finally:
            if staging:
                self._leave_pool(True)

    def _engine_catalog_shared(self, calls, *, phase="preflight"):
        """One fresh catalog for members fixed before the request starts."""
        group = self._engine_batch_call(calls)
        models = getattr(self._client, "models", None)
        if not callable(models):
            return None
        group.check()
        catalog = (
            self._discovery_catalog(
                timeout=group.remaining(), cancel=group.probe, phase=phase
            )
            if self._uses_verified_pool
            else self._http_call(
                "models",
                phase=phase,
                timeout=group.remaining(),
                cancel=group.probe,
                cancel_wait=True,
                **self._engine_response_options(),
            )
        )
        group.check()
        return catalog

    def _engine_observe(self, model, call, *, catalog=None):
        return self._observe_replica(
            replica=self._replicas[model], cancel=call.probe, call=call, catalog=catalog
        )

    def _engine_validate(self, record, model, observation):
        if isinstance(record, _DirectInput):
            self._validate_direct_observation(
                record._direct_policy,
                record.input_type,
                self._replicas[model],
                observation,
            )
            if (
                record._policy != record._direct_policy.policy
                or not record.text.startswith(record._direct_policy.prefix)
            ):
                raise LLMBackendError("llamacpp_embedding_direct_policy_mismatch")
            # A queue reservation is not evidence that the actual input is too
            # long. Fresh smaller contexts remain authoritative at inference;
            # retaining the older reservation only makes intake conservative.
            return
        if record._tokenizer_build != observation.tokenizer_build:
            raise LLMBackendError("llamacpp_embedding_tokenizer_policy_changed")
        effective = _Observation(
            observation.context_length,
            self._minimum(
                observation.context_token_limit,
                record.context_token_limit,
            ),
            self._minimum(observation.batch_token_limit, record.batch_token_limit),
            observation.tokenizer_build,
            observation.identity_key,
            observation.native_batch_tokens,
            observation.native_microbatch_tokens,
            observation.total_slots,
            observation.vocab_size,
        )
        self._admit(record, effective)
        self._validate_vocab(record._token_ids, effective)
        if self.embedding_input_format == "tokens":
            self._embedding_input(
                [record.text], [record], self._replicas[model], effective
            )

    @staticmethod
    def _engine_payload_overhead(model):
        return len(json.dumps({"input": [], "model": model}, ensure_ascii=True))

    def _engine_run(self, model, records, call, observation):
        call.check()
        replica = self._replicas[model]
        wire = self._embedding_input(
            [record.text for record in records], records, replica, observation
        )
        wire_bytes = len(json.dumps({"input": wire, "model": model}, ensure_ascii=True))
        if wire_bytes > self.embedding_max_request_bytes:
            raise LLMBackendError("llamacpp_embedding_request_bytes_exceeded")
        vectors = self._request_batch_adaptive(
            [record.text for record in records],
            replica=replica,
            cancel=call.probe,
            call=call,
            observation=observation,
            records=records,
            wire_input=wire,
        )
        call.validate_dimensions(
            [vector for vector in vectors if not isinstance(vector, _RejectedEmbedding)]
        )
        return vectors

    @staticmethod
    def _engine_batch_call(calls):
        return _BatchCall(calls)

    def _engine_run_shared(self, model, records, calls_per_record, group, observation):
        """Prune owner-local failures, send once, and retain original row indexes."""
        if len(records) != len(calls_per_record):
            raise LLMBackendError("llamacpp_embedding_batch_owner_mismatch")
        owned = {}
        for record, call in zip(records, calls_per_record):
            owned.setdefault(call, []).append(record)
        if any(call not in group.calls for call in owned):
            raise LLMBackendError("llamacpp_embedding_batch_owner_mismatch")
        for call, owner_records in owned.items():
            if call.failure is not None:
                continue
            try:
                call.check()
                for record in owner_records:
                    self._engine_validate(record, model, observation)
                call.check()
            except BaseException as exc:
                call.fail(exc)
        sent = [
            (index, record, call)
            for index, (record, call) in enumerate(zip(records, calls_per_record))
            if call.failure is None
        ]
        if not sent:
            return []
        active_records = [record for _, record, _ in sent]
        group._validated_records = tuple(id(record) for record in active_records)
        group._validated_observation = observation
        group._record_owners = tuple(call for _, _, call in sent)
        positions = group._input_positions
        group._record_input_indices = (
            tuple(positions[index] for index, _, _ in sent)
            if len(positions) == len(records)
            and all(type(index) is int and index >= 0 for index in positions)
            else ()
        )
        vectors = self._engine_run(model, active_records, group, observation)
        if len(vectors) != len(sent):
            raise LLMBackendError("llamacpp_embedding_count_or_response_mismatch")
        by_owner = {}
        for (_, _, call), vector in zip(sent, vectors):
            if not isinstance(vector, _RejectedEmbedding):
                by_owner.setdefault(call, []).append(vector)
        for call, owner_vectors in by_owner.items():
            if call.failure is not None:
                continue
            try:
                call.check()
                call.validate_dimensions(owner_vectors)
            except BaseException as exc:
                call.fail(exc)
        return [
            (index, vector) for (index, _, _), vector in zip(sent, vectors) if vector
        ]

    def _engine_result(self, call, operation, outputs):
        call.check()
        if not outputs:
            return []
        if operation == "prepare":
            result = []
            for record in outputs:
                call.check()
                if isinstance(record, _DirectInput):
                    raise LLMBackendError(
                        "llamacpp_embedding_direct_preparation_not_exact"
                    )
                values = asdict(record)
                values.pop("_text_wire_bytes", None)
                values.pop("_tokens_wire_bytes", None)
                result.append(issue_preparation(self._preparation_key, **values))
            call.check()
            return result
        with self._dimension_lock:
            if self.embedding_dim and call.dimension != self.embedding_dim:
                raise LLMBackendError("llamacpp_embedding_dimension_mismatch")
            if call.dimension is not None:
                self.embedding_dim = call.dimension
        return outputs

    def _engine_finish(self, call, joined):
        self._leave_pool(joined)
        with self._calls_condition:
            self._calls.discard(call)
            self._calls_condition.notify_all()

    def _submit_execute(self, inputs, *, operation, call, retain_result=False):
        self._intake.acquire_owner(call)
        with self._calls_condition:
            if self._closed:
                raise LLMBackendError("llamacpp_embedding_adapter_closed")
            self._calls.add(call)
        try:
            return self._engine.submit(
                inputs, operation=operation, call=call, retain_result=retain_result
            )
        except BaseException:
            self._engine_finish(call, False)
            raise

    def _execute(self, inputs, *, operation, timeout, cancel, call=None):
        call = _Call(self, cancel, timeout) if call is None else call
        future = None
        exception_pending = False
        try:
            future = self._submit_execute(
                inputs, operation=operation, call=call, retain_result=True
            )
            return future.result()
        except BaseException:
            exception_pending = True
            # An interrupted waiter still owns its call until the actual
            # worker/HTTP lease drains. Retire only this owner; peers in a
            # shared POST retain their independent cancellation and deadline.
            while future is not None:
                try:
                    if future.done():
                        break
                    call.stop.set()
                    self._engine.wake()
                    future.result()
                except BaseException:
                    # Repeated interrupts cannot abandon cleanup or replace
                    # the original exception. Worker failures imply done().
                    continue
            raise
        finally:
            try:
                if future is not None:
                    try:
                        self._engine.release_result(call, wait=True)
                    except BaseException:
                        if not exception_pending:
                            raise
            finally:
                lease = getattr(call, "_intake_lease", None)
                if lease is not None:
                    lease.release()

    def _snapshot_inputs(self, values, call, *, prepared=False, coerce_text=False):
        lease = self._intake.acquire_owner(call)
        try:
            return lease.snapshot(values, prepared=prepared, coerce_text=coerce_text)
        except BaseException:
            lease.release()
            raise

    def embed_texts(
        self,
        texts: list[str],
        *,
        input_type: Literal["document", "query", "raw"] = "document",
        timeout=None,
        cancel=None,
    ) -> list[list[float]]:
        """Embed total-text windows distributed over usable replica quotas."""
        self._validate_task(input_type)
        call = _Call(self, cancel, timeout)
        if not texts:
            return []
        prefix = (
            "" if input_type == "raw" else getattr(self._embedding_prefixes, input_type)
        )
        inputs = _TextInputs(
            self._snapshot_inputs(texts, call, coerce_text=True), prefix, input_type
        )
        return self._execute(
            inputs, operation="embed", timeout=timeout, cancel=cancel, call=call
        )

    def _validate_direct_policy(self, policy, input_type):
        self._validate_task(input_type)
        if type(policy) is not _DirectEmbeddingPolicy:
            raise ValueError("policy must be a verified direct embedding policy")
        prefix = (
            "" if input_type == "raw" else getattr(self._embedding_prefixes, input_type)
        )
        if policy.policy != self._policy(input_type) or policy.prefix != prefix:
            raise LLMBackendError("llamacpp_embedding_direct_policy_mismatch")
        if self.embedding_input_format == "tokens":
            raise LLMBackendError("llamacpp_embedding_direct_text_input_required")

    def _validate_direct_observation(self, policy, input_type, replica, observation):
        self._validate_direct_policy(policy, input_type)
        identity = replica.identity
        expected = (policy.content_sha256, policy.embedding_sha256)
        if identity is None or observation.identity_key is None:
            raise LLMBackendError("llamacpp_embedding_direct_identity_unverified")
        if identity.key != expected or observation.identity_key != expected:
            raise LLMBackendError("llamacpp_embedding_direct_identity_changed")
        if observation.tokenizer_build != policy.tokenizer_build:
            raise LLMBackendError("llamacpp_embedding_direct_tokenizer_changed")
        if observation.context_token_limit is None:
            raise LLMBackendError("llamacpp_embedding_direct_context_unknown")

    def _embed_texts_direct(
        self,
        texts,
        *,
        policy: _DirectEmbeddingPolicy,
        input_type="document",
        timeout=None,
        cancel=None,
    ) -> list[list[float]]:
        """Submit strings through the shared engine without local tokenization."""
        self._validate_direct_policy(policy, input_type)
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise ValueError("texts must be a sequence of strings")
        call = _Call(self, cancel, timeout)
        call._direct_policy = policy
        if not texts:
            return []
        self._direct_token_accounting.set()
        return self._execute(
            _TextInputs(self._snapshot_inputs(texts, call), policy.prefix, input_type),
            operation="embed",
            timeout=timeout,
            cancel=cancel,
            call=call,
        )

    def _astream_texts_direct(
        self,
        windows,
        *,
        policy: _DirectEmbeddingPolicy,
        input_type="document",
        max_pending_windows=2,
        timeout=None,
        cancel=None,
        close_source=False,
        admission_scope=None,
    ) -> EmbeddingWindowStream:
        """Create inert direct intake over the same async bridge and engine."""
        self._validate_direct_policy(policy, input_type)
        return EmbeddingWindowStream(
            self,
            windows,
            input_type=input_type,
            max_pending_windows=max_pending_windows,
            timeout=timeout,
            cancel=cancel,
            close_source=close_source,
            _direct_policy=policy,
            _admission_scope=admission_scope,
        )

    def _stream_bridge(self):
        with self._calls_lock:
            if self._closed or self._closing.is_set():
                raise LLMBackendError("llamacpp_embedding_adapter_closed")
            if self._async_window_bridge is None:
                self._async_window_bridge = _EmbeddingWindowBridge(self)
            return self._async_window_bridge

    def _stream_call(self, cancel, timeout):
        return _Call(self, cancel, timeout)

    def _stream_submit(self, window, *, prepared, input_type, call):
        call.check()
        direct = getattr(call, "_direct_policy", None)
        if direct is not None:
            if prepared:
                raise LLMBackendError("llamacpp_embedding_invalid_direct_input")
            self._validate_direct_policy(direct, input_type)
            self._direct_token_accounting.set()
        if prepared:
            inputs = self._snapshot_inputs(window, call, prepared=True)
            call.prepared_inputs = inputs
        else:
            prefix = (
                ""
                if input_type == "raw"
                else getattr(self._embedding_prefixes, input_type)
            )
            inputs = _TextInputs(
                self._snapshot_inputs(window, call), prefix, input_type
            )
        return self._submit_execute(
            inputs, operation="embed", call=call, retain_result=True
        )

    def _stream_release_result(self, call):
        self._engine.release_result(call)

    def astream_texts(
        self,
        windows: AsyncIterable[Sequence[str]],
        *,
        input_type: Literal["document", "query", "raw"] = "document",
        max_pending_windows: int = 2,
        timeout: float | None = None,
        cancel: CancellationProbe | None = None,
        close_source: bool = False,
    ) -> EmbeddingWindowStream:
        """Return inert async intake over the same shared embedding scheduler."""
        self._validate_task(input_type)
        return EmbeddingWindowStream(
            self,
            windows,
            input_type=input_type,
            max_pending_windows=max_pending_windows,
            timeout=timeout,
            cancel=cancel,
            close_source=close_source,
        )

    def astream_prepared(
        self,
        windows: AsyncIterable[Sequence[PreparedEmbeddingInput]],
        *,
        max_pending_windows: int = 2,
        timeout: float | None = None,
        cancel: CancellationProbe | None = None,
        close_source: bool = False,
    ) -> EmbeddingWindowStream:
        """Stream same-adapter preparations without a separate HTTP engine."""
        return EmbeddingWindowStream(
            self,
            windows,
            prepared=True,
            max_pending_windows=max_pending_windows,
            timeout=timeout,
            cancel=cancel,
            close_source=close_source,
        )

    async def aclose(self) -> None:
        """Drain async intake/source ownership, then shared HTTP work off-loop."""
        await _close_embedding_adapter(self)

    def prepare_texts(
        self,
        texts: Sequence[str],
        *,
        input_type: Literal["document", "query", "raw"] = "document",
        timeout=None,
        cancel=None,
    ) -> tuple[PreparedEmbeddingInput, ...]:
        """Measure exact once-prefixed inputs, including overlimit inputs."""
        self._validate_task(input_type)
        call = _Call(self, cancel, timeout)
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise ValueError("texts must be a sequence of strings")
        if not texts:
            return ()
        prefix = (
            "" if input_type == "raw" else getattr(self._embedding_prefixes, input_type)
        )
        inputs = _TextInputs(self._snapshot_inputs(texts, call), prefix, input_type)
        return tuple(
            self._execute(
                inputs, operation="prepare", timeout=timeout, cancel=cancel, call=call
            )
        )

    def prepare_query(
        self, text: str, *, timeout=None, cancel=None
    ) -> PreparedEmbeddingInput:
        """Measure a query, including its prefix and native special tokens."""
        return self.prepare_texts(
            [text], input_type="query", timeout=timeout, cancel=cancel
        )[0]

    def embed_prepared(
        self,
        inputs: Sequence[PreparedEmbeddingInput],
        *,
        timeout=None,
        cancel=None,
    ) -> list[list[float]]:
        """Embed same-adapter preparations with fresh identity/limit admission."""
        call = _Call(self, cancel, timeout)
        if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
            raise ValueError("inputs must be a sequence of prepared embedding inputs")
        if not inputs:
            return []
        inputs = self._snapshot_inputs(inputs, call, prepared=True)
        call.prepared_inputs = inputs
        return self._execute(
            inputs, operation="embed", timeout=timeout, cancel=cancel, call=call
        )

    def embed_query(self, text: str, *, timeout=None, cancel=None) -> list[float]:
        """Embed a query using the model's configured query prefix."""
        return (
            self.embed_texts(
                [str(text)], input_type="query", timeout=timeout, cancel=cancel
            )[0]
            if text
            else []
        )

    def load(self) -> bool:
        """Compatibility readiness method; on-demand loading follows server policy."""
        return self.is_ready()

    def is_ready(self) -> bool:
        if self._closed:
            return False
        return self._readiness()["model_ready"]

    def _readiness(self) -> dict[str, Any]:
        joined = False
        call = _Call(self, None)
        try:
            joined = self._join_pool(call, call.remaining())
            return self._readiness_current(call)
        except (LLMBackendError, LLMTimeoutError, LLMCancelledError):
            return {
                "server_healthy": self._discovery_server_healthy,
                "model_ready": False,
                "models": {model: "unknown" for model in self._replicas},
            }
        finally:
            self._leave_pool(joined)

    def _readiness_current(self, call=None) -> dict[str, Any]:
        call = _Call(self, None) if call is None else call
        states = {model: "unknown" for model in self._replicas}
        result = {"server_healthy": False, "model_ready": False, "models": states}
        if self._closed:
            for replica in self._replicas.values():
                self._invalidate_replica(replica)
            return result
        try:
            health = self._http_call(
                "health",
                phase="readiness",
                timeout=call.remaining(),
                cancel=call.probe,
                cancel_wait=True,
                **self._engine_response_options(),
            )
            result["server_healthy"] = (
                isinstance(health, dict) and health.get("status") == "ok"
            )
            if not result["server_healthy"]:
                for replica in self._replicas.values():
                    self._invalidate_replica(replica)
                return result
            catalog = self._discovery_catalog(
                timeout=call.remaining(), cancel=call.probe, phase="readiness"
            )
            entries = catalog.get("data") if isinstance(catalog, dict) else None
            if not isinstance(entries, list):
                for replica in self._replicas.values():
                    self._invalidate_replica(replica)
                return result
            for model, replica in self._replicas.items():
                matches = self._catalog_matches(model, entries)
                if len(matches) != 1:
                    states[model] = "missing" if not matches else "ambiguous"
                    self._invalidate_replica(replica)
                    continue
                entry = matches[0]
                replica.accepted_aliases = self._catalog_aliases(entry)
                status = entry.get("status")
                if "status" in entry:
                    state = status.get("value") if isinstance(status, dict) else status
                    states[model] = "loaded" if state == "loaded" else "not_loaded"
                    if states[model] != "loaded":
                        # Do not invoke router /props for unloaded models: it can
                        # otherwise trigger the external auto-load policy.
                        self._invalidate_replica(replica)
                        continue
                try:
                    self._observe_replica(
                        replica=replica,
                        call=call,
                        cancel=call.probe,
                        catalog=catalog,
                        phase="readiness",
                    )
                    if "status" not in entry:
                        # Context size alone does not establish model identity.
                        states[model] = (
                            "loaded" if replica.props_identity_verified else "unknown"
                        )
                except (LLMBackendError, LLMTimeoutError, LLMCancelledError):
                    self._invalidate_replica(replica)
                    states[model] = "unknown"
            result["model_ready"] = result["server_healthy"] and all(
                state == "loaded" for state in states.values()
            )
        except (LLMBackendError, LLMTimeoutError, LLMCancelledError):
            for replica in self._replicas.values():
                self._invalidate_replica(replica)
        return result

    def _engine_health_snapshot(self):
        snapshot = self._engine.snapshot()
        snapshot["intake"] = self._intake.snapshot()
        snapshot["http_metrics"] = self._http_metrics.snapshot()
        snapshot["control_json_bounds"] = (
            "owned_reader"
            if self._owns_client
            else "declared_bounded_reader"
            if self._engine_response_options()
            else "borrowed_decoder"
        )
        if self._direct_token_accounting.is_set():
            snapshot = dict(snapshot)
            snapshot["token_accounting"] = "may_include_context_reservations"
            snapshot["preparation_work_includes_direct_reservations"] = True
        return snapshot

    def health_status(self) -> dict[str, Any]:
        joined = False
        call = _Call(self, None)
        try:
            joined = self._join_pool(call, call.remaining())
            return self._health_status_current(call)
        except (LLMBackendError, LLMTimeoutError, LLMCancelledError):
            return {
                "backend": "llamacpp",
                "healthy": False,
                "server_healthy": self._discovery_server_healthy,
                "model_ready": False,
                "model": self.model_name,
                "embedding_replica_discovery": self._discovery_status(),
                "replicas": {},
                "scheduler": self._scheduler.snapshot(),
                "engine": self._engine_health_snapshot(),
            }
        finally:
            self._leave_pool(joined)

    def _discovery_status(self) -> dict[str, Any]:
        return {
            "enabled": self._discovery_enabled,
            "identity_verified": self._pinned_identity is not None,
            "content_sha256": (
                self._pinned_identity.content_sha256 if self._pinned_identity else None
            ),
            "embedding_sha256": (
                self._pinned_identity.embedding_sha256
                if self._pinned_identity
                else None
            ),
            "fallback_alias": self._fallback_alias,
            "error": self._discovery_error,
        }

    def _health_status_current(self, call=None) -> dict[str, Any]:
        readiness = self._readiness_current(call)
        replicas = {}
        for model, replica in self._replicas.items():
            with replica.lock:
                replicas[model] = {
                    "replica_id": replica.replica_id,
                    "readiness": readiness["models"][model],
                    "max_in_flight": replica.max_in_flight,
                    "effective_batch_size": replica.adaptive_batch_size,
                    "batch_recovery": {
                        "ceiling": replica.recovery_ceiling,
                        "successes": replica.recovery_successes,
                        "success_window": replica.recovery_window,
                        "reductions": replica.reductions,
                    },
                    "configured_max_input_tokens": replica.configured_max_input_tokens,
                    "batch_token_limit": self._minimum(
                        self.batch_token_limit,
                        replica.batch_token_limit,
                    ),
                    "loaded_batch_token_limit": replica.loaded_batch_token_limit,
                    "observed_native_batch_tokens": replica.loaded_batch_token_limit,
                    "observed_native_microbatch_tokens": replica.native_microbatch_tokens,
                    "tokenizer_build": replica.tokenizer_build,
                    "observed_vocab_size": replica.vocab_size,
                    "max_input_tokens": self._input_limit(replica=replica),
                    "loaded_context_length": replica.context_length,
                    "observed_total_slots": replica.total_slots,
                }
        primary = replicas[self._primary.model]
        return {
            "backend": "llamacpp",
            "transport": "llama-server-http",
            "embedding_data_plane": "openai-compatible-http",
            "runtime_ownership": "external",
            "role_id": self.role_id,
            "model": self.model_name,
            "healthy": readiness["model_ready"],
            "server_healthy": readiness["server_healthy"],
            "model_ready": readiness["model_ready"],
            "batch_size": self.batch_size,
            "embedding_batch_wait_sec": self.embedding_batch_wait_sec,
            "embedding_input_format": self.embedding_input_format,
            "embedding_token_ids_supported": (
                getattr(self._client, "supports_embedding_token_ids", None) is True
            ),
            "effective_batch_size": primary["effective_batch_size"],
            "batch_recovery": primary["batch_recovery"],
            "batch_token_limit": self.batch_token_limit,
            "token_count_parallelism": self.token_count_parallelism,
            "dimension": self.embedding_dim or None,
            "max_input_tokens": min(
                (
                    limits["max_input_tokens"]
                    for limits in replicas.values()
                    if limits["max_input_tokens"] is not None
                ),
                default=None,
            ),
            "loaded_context_length": self._loaded_context_length,
            "embedding_request_parallelism": self.embedding_request_parallelism,
            "embedding_queue_size": self.embedding_queue_size,
            "embedding_replica_discovery": self._discovery_status(),
            "scheduler": self._scheduler.snapshot(),
            "engine": self._engine_health_snapshot(),
            "preparation_cache": self._preparation_cache.snapshot(),
            "replicas": replicas,
        }

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._intake.assert_can_drain_sync()
            with self._calls_lock:
                self._closed = True
                self._closing.set()
                bridge = self._async_window_bridge
                for call in self._calls:
                    call.stop.set()
            self._preparation_cache.close()
            self._intake.close()
            if bridge is not None:
                bridge.signal_close()
            self._engine.close()
            self._scheduler.close()
            self._intake.drain_sync()
            if self._owns_client:
                self._client.close()
            with self._calls_condition:
                while self._calls:
                    self._calls_condition.wait(0.05)
