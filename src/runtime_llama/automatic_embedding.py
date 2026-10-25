"""Optional direct embedding service with an explicit character-budget helper.

Ordinary embedding never requires a calibration cache or calls the tokenizer.
Consumers can explicitly obtain a calibrated character budget, persist it in
their own configuration, and continue sending strings on later starts.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import sys
from threading import Condition, Event, Lock, local
import time

from runtime_llama._proxy import normalize_proxy_url
from runtime_llama.adapters.embedding import (
    LlamaCppEmbeddingAdapter,
    _Call,
    _DirectEmbeddingPolicy,
)
from runtime_llama.client import LlamaCppClient
from runtime_llama.embedding_metrics import EmbeddingHTTPMetrics

from runtime_llama.configuration import (
    AdapterProfile,
    EmbeddingPrefixes,
    _finite,
    _nonnegative_int,
    _positive_int,
    normalize_rest_root,
)
from runtime_llama.embedding_calibration import (
    CALIBRATION_CATEGORIES,
    load_existing_calibration,
    load_or_calibrate,
)
from runtime_llama.embedding_budget import (
    EMBEDDING_BUDGET_POLICY_ID,
    budget_context_scope_id,
)
from runtime_llama.embedding_recovery import (
    EmbeddingBudgetSnapshot,
)
from runtime_llama.embedding_halving import (
    EmbeddingChunkSplitRequiredError,
    _HalvingResult,
    _halving_error,
)
from runtime_llama.embedding_observation import _EmbeddingAdmissionScope
from runtime_llama.embedding_stream import _drain
from runtime_llama.errors import (
    LLMBackendError,
    LLMCancelledError,
    LLMTimeoutError,
    sanitize_diagnostic,
)
from runtime_llama.replica_discovery import (
    attest_single_model_catalog,
    discover_embedding_pool,
    is_loaded,
    parse_replica,
)

_NATIVE_CLIENT_CLASS = LlamaCppClient


def _category(value):
    if type(value) is not str or value not in (*CALIBRATION_CATEGORIES, "mixed"):
        raise ValueError("unknown embedding calibration category")
    return value


def _task(value):
    if value not in ("document", "query", "raw"):
        raise ValueError("input_type must be document, query or raw")
    return value


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _default_cache_dir():
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "runtime-llama" / "embedding-calibration"


@dataclass(frozen=True, slots=True)
class AutomaticEmbeddingConfig:
    """Consumer-neutral connection and optional character-budget policy.

    ``timeout`` bounds ordinary synchronous discovery and inference together.
    Only an explicit ``character_budget`` request can spend up to
    ``calibration_timeout`` calibrating the tokenizer. Async streams have a
    bounded readiness stage followed by independent per-window deadlines.
    """

    base_url: str
    category: str = "mixed"
    cache_dir: str | os.PathLike | None = None
    model: str | None = None
    embedding_prefixes: EmbeddingPrefixes | Mapping[str, str] | None = None
    timeout: float = 120.0
    calibration_timeout: float = 600.0
    calibration_workers: int = 2
    batch_size: int = 256
    request_parallelism: int | None = None
    embedding_max_pending_owners: int | None = None
    queue_size: int = 16
    normalize: bool = True
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str | None = field(default=None, repr=False)
    verify_ssl: bool = True
    proxy_url: str | None = field(default=None, repr=False)
    trust_env: bool = False
    embedding_max_pending_bytes: int = 64 * 1024 * 1024
    embedding_max_pending_tokens: int = 1024 * 1024
    embedding_max_request_bytes: int = 8 * 1024 * 1024
    embedding_max_response_bytes: int = 8 * 1024 * 1024
    embedding_isolation_max_requests: int = 512
    http_connection_policy: str = "close"

    def __post_init__(self):
        if not isinstance(self.base_url, str) or any(
            char.isspace() or ord(char) < 32 or char == "\\" for char in self.base_url
        ):
            raise ValueError("base_url must identify an HTTP(S) server")
        object.__setattr__(self, "base_url", normalize_rest_root(self.base_url))
        _category(self.category)
        if self.cache_dir is not None and not isinstance(
            self.cache_dir, (str, os.PathLike)
        ):
            raise ValueError("cache_dir must be a user-owned filesystem path")
        if self.model is not None:
            if not isinstance(self.model, str) or not self.model.strip():
                raise ValueError("model must be a nonempty selector or None")
            object.__setattr__(self, "model", self.model.strip())
        if self.embedding_prefixes is not None:
            object.__setattr__(
                self,
                "embedding_prefixes",
                EmbeddingPrefixes.from_value(self.embedding_prefixes),
            )
        for name in ("timeout", "calibration_timeout"):
            _finite(getattr(self, name), name, positive=True)
        for name in ("batch_size", "calibration_workers"):
            if _positive_int(getattr(self, name), name) is None:
                raise ValueError(f"{name} must be a positive integer")
        _positive_int(self.request_parallelism, "request_parallelism")
        _positive_int(self.embedding_max_pending_owners, "embedding_max_pending_owners")
        if self.calibration_workers > 32:
            raise ValueError("calibration_workers must not exceed 32")
        _nonnegative_int(self.queue_size, "queue_size")
        for name in (
            "embedding_max_pending_bytes",
            "embedding_max_pending_tokens",
            "embedding_max_request_bytes",
            "embedding_max_response_bytes",
            "embedding_isolation_max_requests",
        ):
            if _positive_int(getattr(self, name), name) is None:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(
            self.http_connection_policy, str
        ) or self.http_connection_policy not in ("close", "reuse"):
            raise ValueError("http_connection_policy must be close or reuse")
        for name in ("normalize", "verify_ssl", "trust_env"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("api_key", "api_key_env"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty string or None")
        object.__setattr__(self, "proxy_url", normalize_proxy_url(self.proxy_url))


@dataclass(frozen=True, slots=True)
class AutomaticEmbeddingStatus:
    """Availability and a recommended Unicode-character budget, never exact."""

    available: bool
    max_input_chars: int | None
    category: str | None
    input_type: str
    reason: str | None
    representation_id: str | None
    calibration_id: str | None
    budget: EmbeddingBudgetSnapshot | None = None


class _ReentrantCloseError(LLMBackendError):
    """A consumer callback cannot synchronously drain its own operation."""

    def __init__(self):
        super().__init__("llamacpp_automatic_embedding_close_from_cancel_callback")


class _Operation:
    def __init__(self, service, timeout, cancel):
        _finite(timeout, "timeout", positive=True)
        if cancel is not None and not callable(cancel):
            raise ValueError("cancel must be a callable or None")
        self.service, self.cancel = service, cancel
        self.deadline = time.monotonic() + timeout
        self.stop = Event()
        self.failure = None

    def check(self):
        if self.service._closing.is_set():
            raise LLMBackendError("llamacpp_automatic_embedding_closed")
        if self.failure is not None:
            raise self.failure
        if self.stop.is_set():
            raise LLMCancelledError()
        if time.monotonic() >= self.deadline:
            raise LLMTimeoutError("llamacpp_automatic_embedding_timeout")
        if self.cancel is not None:
            try:
                cancelled = self.service._invoke_cancel(self.cancel)
            except _ReentrantCloseError:
                raise
            except Exception as exc:
                raise LLMBackendError(
                    "llamacpp_automatic_embedding_cancel_probe_failed"
                ) from RuntimeError(type(exc).__name__)
            if cancelled:
                raise LLMCancelledError()

    def probe(self):
        try:
            self.check()
            return False
        except (LLMBackendError, LLMTimeoutError, LLMCancelledError) as exc:
            self.failure = exc
            return True

    def remaining(self):
        self.check()
        return max(0.001, self.deadline - time.monotonic())


class AutomaticEmbeddingService:
    """An optional automatic facade; it never loads or administers models."""

    def __init__(self, config: AutomaticEmbeddingConfig | Mapping, *, client=None):
        if isinstance(config, Mapping):
            config = AutomaticEmbeddingConfig(**dict(config))
        if not isinstance(config, AutomaticEmbeddingConfig):
            raise ValueError("config must describe the automatic embedding service")
        self.config = config
        self._client = client
        self._http_metrics = EmbeddingHTTPMetrics()
        self._owns_client = client is None
        self._adapter = None
        self._identity = self._prefixes = None
        self._profiles = {}
        self._profile_flights = set()
        self._initialization_lock = Lock()
        self._close_lock = Lock()
        self._condition = Condition()
        self._closing = Event()
        self._closed = False
        self._operations = set()
        self._streams = set()
        self._callback_state = local()

    def _invoke_cancel(self, callback):
        depth = getattr(self._callback_state, "depth", 0)
        self._callback_state.depth = depth + 1
        try:
            return bool(callback())
        finally:
            self._callback_state.depth = depth

    def _check_close_reentrancy(self):
        if getattr(self._callback_state, "depth", 0):
            raise _ReentrantCloseError()

    @contextmanager
    def _operation(self, timeout, cancel):
        operation = _Operation(self, timeout, cancel)
        with self._condition:
            operation.check()
            self._operations.add(operation)
        try:
            yield operation
        finally:
            with self._condition:
                self._operations.discard(operation)
                self._condition.notify_all()

    def _ensure_client(self, operation):
        with self._condition:
            operation.check()
            if self._client is None:
                api_key = self.config.api_key or (
                    os.environ.get(self.config.api_key_env)
                    if self.config.api_key_env
                    else None
                )
                self._client = LlamaCppClient(
                    self.config.base_url,
                    api_key=api_key,
                    timeout=self.config.timeout,
                    verify_ssl=self.config.verify_ssl,
                    proxy_url=self.config.proxy_url,
                    trust_env=self.config.trust_env,
                    connection_policy=self.config.http_connection_policy,
                )
            return self._client

    def _response_options(self, client):
        # Injected clients own decoding unless they explicitly support bounds.
        return (
            {"max_response_bytes": self.config.embedding_max_response_bytes}
            if isinstance(client, _NATIVE_CLIENT_CLASS)
            or getattr(client, "supports_bounded_json", False) is True
            else {}
        )

    def _http_call(self, client, method, *args, **kwargs):
        kind = {"models": "catalog"}.get(method, method)
        with self._http_metrics.measure(kind, "admission"):
            return getattr(client, method)(*args, **kwargs)

    def diagnostics(self):
        """Read bounded local counters without health/catalog network traffic."""
        with self._condition:
            adapter = self._adapter
        return {
            "initialization_http": self._http_metrics.snapshot(),
            "engine": adapter._engine_health_snapshot()
            if adapter is not None
            else None,
        }

    def _select_pool(self, catalog):
        if self.config.model is not None:
            pool = discover_embedding_pool(catalog, self.config.model)
            if pool.identity is None:
                raise LLMBackendError(
                    "llamacpp_automatic_embedding_identity_unverified"
                )
            return pool
        entries = catalog.get("data") if isinstance(catalog, dict) else None
        if not isinstance(entries, list) or any(
            not isinstance(item, dict) for item in entries
        ):
            raise LLMBackendError("llamacpp_embedding_invalid_model_catalog")
        groups = {}
        for entry in entries:
            if not is_loaded(entry):
                continue
            tags = entry.get("tags", ())
            if not isinstance(tags, list) or not any(
                isinstance(tag, str) and tag.startswith("noxi-embedding-sha256-v1:")
                for tag in tags
            ):
                continue
            replica = parse_replica(entry)
            if replica.identity is not None:
                groups.setdefault(replica.identity.key, []).append(replica)
        if not groups:
            raise LLMBackendError("llamacpp_automatic_embedding_model_not_loaded")
        if len(groups) != 1:
            raise LLMBackendError("llamacpp_automatic_embedding_ambiguous_model")
        group = next(iter(groups.values()))
        return discover_embedding_pool(
            catalog, "sha256:" + group[0].identity.content_sha256
        )

    def _select_prefixes(self, pool):
        if self.config.embedding_prefixes is not None:
            return self.config.embedding_prefixes
        profiles = set()
        for replica in pool.replicas:
            name = (replica.model_name or "").lower()
            if "nomic-embed-text-v2" in name:
                profiles.add(("search_query: ", "search_document: "))
            elif "embeddinggemma" in name:
                profiles.add(("task: search result | query: ", "title: none | text: "))
            else:
                raise LLMBackendError("llamacpp_automatic_embedding_prefixes_required")
        if len(profiles) != 1:
            raise LLMBackendError("llamacpp_automatic_embedding_ambiguous_prefixes")
        query, document = next(iter(profiles))
        return EmbeddingPrefixes(query=query, document=document)

    def _ensure_adapter(self, operation):
        if self._adapter is not None:
            return self._adapter
        client = self._ensure_client(operation)
        healthy = self._http_call(
            client,
            "health",
            timeout=min(self.config.timeout, operation.remaining()),
            cancel=operation.probe,
            cancel_wait=True,
            **self._response_options(client),
        )
        operation.check()
        if not isinstance(healthy, dict) or healthy.get("status") != "ok":
            raise LLMBackendError("llamacpp_embedding_server_not_healthy")
        catalog = self._http_call(
            client,
            "models",
            timeout=min(self.config.timeout, operation.remaining()),
            cancel=operation.probe,
            cancel_wait=True,
            **self._response_options(client),
        )
        operation.check()
        entries = catalog.get("data") if isinstance(catalog, dict) else None
        if (
            isinstance(entries, list)
            and len(entries) == 1
            and isinstance(entries[0], dict)
            and "status" not in entries[0]
        ):
            # Require proven content before a statusless compatibility probe.
            # Root props cannot select or auto-load a router model. The same
            # loaded-state authority is used by the existing exact adapter.
            if parse_replica(entries[0]).identity is None:
                raise LLMBackendError(
                    "llamacpp_automatic_embedding_identity_unverified"
                )
            props = self._http_call(
                client,
                "props",
                timeout=min(self.config.timeout, operation.remaining()),
                cancel=operation.probe,
                cancel_wait=True,
                **self._response_options(client),
            )
            operation.check()
            catalog = attest_single_model_catalog(catalog, props)
        pool = self._select_pool(catalog)
        prefixes = self._select_prefixes(pool)
        profile = AdapterProfile.from_config(
            "automatic_embedding",
            {
                "adapter_type": "embedding",
                "model": "sha256:" + pool.identity.content_sha256,
                "base_url": self.config.base_url,
                "timeout": self.config.timeout,
                "model_context_length": None,
                "max_input_tokens": None,
                "batch_token_limit": None,
                "embedding_prefixes": asdict(prefixes),
                "embedding_replica_discovery": True,
                "embedding_input_format": "text",
                "embedding_request_parallelism": self.config.request_parallelism,
                "embedding_max_pending_owners": self.config.embedding_max_pending_owners,
                "embedding_queue_size": self.config.queue_size,
                "token_count_parallelism": self.config.calibration_workers,
                "embedding_batch_wait_sec": 0,
                "batch_size": self.config.batch_size,
                "normalize": self.config.normalize,
                "verify_ssl": self.config.verify_ssl,
                "proxy_url": self.config.proxy_url,
                "api_key": self.config.api_key,
                "api_key_env": self.config.api_key_env,
                "http_connection_policy": self.config.http_connection_policy,
                "embedding_max_pending_bytes": self.config.embedding_max_pending_bytes,
                "embedding_max_pending_tokens": self.config.embedding_max_pending_tokens,
                "embedding_max_request_bytes": self.config.embedding_max_request_bytes,
                "embedding_max_response_bytes": self.config.embedding_max_response_bytes,
                "embedding_isolation_max_requests": self.config.embedding_isolation_max_requests,
            },
        )
        with self._condition:
            operation.check()
            adapter = LlamaCppEmbeddingAdapter(profile, client=client)
            # The selector can denote weights loaded with multiple embedding
            # semantics. Carry the already verified selected group into its
            # first admission rather than resolving the weight digest again.
            adapter._pinned_identity = pool.identity
            self._adapter, self._identity, self._prefixes = (
                adapter,
                pool.identity,
                prefixes,
            )
            return adapter

    def _binding(self, adapter, context, input_type):
        observations = context.observations
        if not observations or any(
            observation.identity_key != self._identity.key
            for observation in observations.values()
        ):
            raise LLMBackendError("llamacpp_automatic_embedding_identity_changed")
        builds = {observation.tokenizer_build for observation in observations.values()}
        vocabs = {observation.vocab_size for observation in observations.values()}
        if len(builds) != 1 or None in builds or len(vocabs) != 1:
            raise LLMBackendError("llamacpp_automatic_embedding_inconsistent_tokenizer")
        if any(
            observation.context_token_limit is None
            for observation in observations.values()
        ):
            raise LLMBackendError("llamacpp_automatic_embedding_context_unknown")
        binding = {
            "content_sha256": self._identity.content_sha256,
            "embedding_sha256": self._identity.embedding_sha256,
            "tokenizer_build": next(iter(builds)),
            "vocab_size": next(iter(vocabs)),
            "policy": adapter._policy(input_type),
            "input_type": input_type,
            "prefix": ""
            if input_type == "raw"
            else getattr(self._prefixes, input_type),
        }
        budget = min(
            adapter._minimum(
                observation.context_token_limit, observation.native_microbatch_tokens
            )
            for observation in observations.values()
        )
        return binding, budget

    @contextmanager
    def _observed(self, operation, *, input_type, calibration=False):
        """Own fresh metadata and a stable pool generation until the user exits."""
        _task(input_type)
        acquired = False
        call = None
        try:
            while not acquired:
                operation.check()
                acquired = self._initialization_lock.acquire(
                    timeout=min(0.05, operation.remaining())
                )
            adapter = self._ensure_adapter(operation)
            call = _Call(
                adapter,
                operation.probe,
                min(
                    self.config.calibration_timeout
                    if calibration
                    else self.config.timeout,
                    operation.remaining(),
                ),
            )

            def metadata_call():
                return _Call(
                    adapter,
                    call.probe,
                    min(self.config.timeout, call.remaining(), operation.remaining()),
                )

            context = None
            while context is None:
                operation.check()
                call.check()
                context = adapter._engine_start(metadata_call())
                if context is None:
                    with adapter._pool_condition:
                        adapter._pool_condition.wait(
                            min(0.05, operation.remaining(), call.remaining())
                        )
            try:
                call.check()
                binding, budget = self._binding(adapter, context, input_type)
                # The pool reference, rather than this initialization mutex,
                # protects the observed generation through calibration/use.
                # Ordinary direct requests can progress during a budget probe.
                self._initialization_lock.release()
                acquired = False
                yield adapter, call, context, binding, budget, metadata_call
            finally:
                adapter._leave_pool(context.joined)
        except LLMCancelledError:
            # Boolean transport cancellation preserves this owner's cause.
            operation.check()
            if call is not None:
                call.check()
            raise
        finally:
            if acquired:
                self._initialization_lock.release()

    @contextmanager
    def _profile_slot(self, key, operation):
        """Deduplicate explicit budget work without blocking ordinary requests."""
        leader = False
        waited = False
        try:
            while True:
                operation.check()
                timeout = min(0.05, operation.remaining())
                with self._condition:
                    profile = self._profiles.get(key)
                    if profile is not None:
                        break
                    if key not in self._profile_flights:
                        self._profile_flights.add(key)
                        leader = True
                        break
                    waited = True
                    self._condition.wait(timeout)
            yield profile, waited
        finally:
            if leader:
                with self._condition:
                    self._profile_flights.discard(key)
                    self._condition.notify_all()

    def _representation(self, binding):
        return _digest(
            {
                "content_sha256": self._identity.content_sha256,
                "embedding_sha256": self._identity.embedding_sha256,
                "tokenizer_build": binding["tokenizer_build"],
                "vocab_size": binding["vocab_size"],
                "prefixes": asdict(self._prefixes),
                "normalize": self.config.normalize,
            }
        )

    def _cached_profile(self, binding, operation):
        """Read completed measurements only; never start calibration here."""
        key = _digest(binding)
        with self._condition:
            profile = self._profiles.get(key)
        if profile is None:
            profile = load_existing_calibration(
                self.config.cache_dir or _default_cache_dir(),
                binding,
                operation.check,
            )
            if profile is not None:
                with self._condition:
                    operation.check()
                    self._profiles[key] = profile
        return profile

    def _budget_snapshot(self, binding, context_tokens, category, profile, operation):
        operation.check()
        initial = (
            None
            if profile is None
            else profile.max_input_chars(context_tokens, category)
        )
        if initial is not None and initial <= 0:
            raise LLMBackendError(
                "llamacpp_automatic_embedding_character_budget_unavailable"
            )
        maximum = initial
        if maximum is None:
            return None
        return EmbeddingBudgetSnapshot(
            policy_id=EMBEDDING_BUDGET_POLICY_ID,
            representation_id=self._representation(binding),
            context_scope_id=budget_context_scope_id(binding, context_tokens),
            input_type=binding["input_type"],
            category=category,
            calibration_id=None if profile is None else profile.profile_id,
            max_input_chars=maximum,
            initial_max_input_chars=initial,
            budget_revision=0,
            effective_context_tokens=context_tokens,
            source="calibrated_mean",
        )

    def _ready(self, operation, *, input_type, admission_scope=None):
        # No corpus, calibration profiles or cache paths are consulted here.
        with self._observed(operation, input_type=input_type) as observed:
            adapter, _, context, binding, _, _ = observed
            policy = _DirectEmbeddingPolicy(
                content_sha256=binding["content_sha256"],
                embedding_sha256=binding["embedding_sha256"],
                tokenizer_build=binding["tokenizer_build"],
                policy=binding["policy"],
                prefix=binding["prefix"],
            )
            if admission_scope is not None:
                adapter._engine_bind_admission_scope(admission_scope, context)
            return (
                AutomaticEmbeddingStatus(
                    True,
                    None,
                    None,
                    input_type,
                    None,
                    self._representation(binding),
                    None,
                ),
                policy,
                adapter,
            )

    def initialize(
        self, *, input_type="document", timeout=None, cancel=None
    ) -> AutomaticEmbeddingStatus:
        """Verify loaded embedding availability without requesting calibration."""
        input_type = _task(input_type)
        bound = self.config.timeout if timeout is None else timeout
        with self._operation(bound, cancel) as operation:
            status, _, _ = self._ready(operation, input_type=input_type)
            return status

    def status(
        self, *, input_type="document", timeout=None, cancel=None
    ) -> AutomaticEmbeddingStatus:
        """Observe availability only; no calibration or cache access occurs."""
        input_type = _task(input_type)
        try:
            return self.initialize(
                input_type=input_type, timeout=timeout, cancel=cancel
            )
        except (LLMBackendError, LLMTimeoutError) as exc:
            return AutomaticEmbeddingStatus(
                False, None, None, input_type, sanitize_diagnostic(exc), None, None
            )

    def character_budget(
        self, *, category=None, input_type="document", timeout=None, cancel=None
    ) -> AutomaticEmbeddingStatus:
        """Explicitly calibrate/load a character recommendation for the consumer.

        All 10,000 native counts are collected once per verified task/prefix/
        tokenizer binding. All categories reuse those completed statistics.
        Consumers own persistence and source splitting. Ordinary embedding does
        not call this method, require its cache, or enforce its recommendation.
        """
        input_type = _task(input_type)
        category = _category(self.config.category if category is None else category)
        bound = self.config.calibration_timeout if timeout is None else timeout
        with self._operation(bound, cancel) as operation:
            with self._observed(
                operation, input_type=input_type, calibration=True
            ) as observed:
                adapter, call, context, binding, context_budget, metadata_call = (
                    observed
                )
                key = _digest(binding)
                models = tuple(context.observations)

                def validate():
                    nonlocal context_budget
                    operation.check()
                    metadata = metadata_call()
                    catalog = adapter._engine_catalog_shared((metadata,))
                    observations = {
                        model: adapter._engine_observe(model, metadata, catalog=catalog)
                        for model in models
                    }
                    fresh_context = type(context)(
                        context.joined, observations, context.conservative
                    )
                    fresh_binding, fresh_budget = self._binding(
                        adapter, fresh_context, input_type
                    )
                    if fresh_binding != binding:
                        raise LLMBackendError(
                            "llamacpp_automatic_embedding_calibration_identity_changed"
                        )
                    operation.check()
                    context_budget = fresh_budget

                with self._profile_slot(key, operation) as (profile, waited):
                    if waited:
                        # A completed profile can be reused after waiting, but
                        # its character recommendation uses current limits.
                        validate()
                    if profile is None:
                        rotation = 0
                        rotation_lock = Lock()

                        def tokenize(source):
                            nonlocal rotation
                            operation.check()
                            call.check()
                            with rotation_lock:
                                model = models[rotation % len(models)]
                                rotation += 1
                            token_call = _Call(
                                adapter,
                                call.probe,
                                min(
                                    self.config.timeout,
                                    call.remaining(),
                                    operation.remaining(),
                                ),
                            )
                            try:
                                count = len(
                                    adapter._tokenize_input(
                                        binding["prefix"] + source,
                                        adapter._replicas[model],
                                        token_call,
                                    )
                                )
                            except LLMCancelledError:
                                operation.check()
                                call.check()
                                raise
                            operation.check()
                            return count

                        profile = load_or_calibrate(
                            self.config.cache_dir or _default_cache_dir(),
                            binding,
                            tokenize,
                            call.check,
                            workers=self.config.calibration_workers,
                            validate=validate,
                        )
                        operation.check()
                        with self._condition:
                            self._profiles[key] = profile
                    snapshot = self._budget_snapshot(
                        binding, context_budget, category, profile, operation
                    )
                    return AutomaticEmbeddingStatus(
                        True,
                        snapshot.max_input_chars,
                        category,
                        input_type,
                        None,
                        self._representation(binding),
                        profile.profile_id,
                        snapshot,
                    )

    def refresh_character_budget(
        self, *, category=None, input_type="document", timeout=None, cancel=None
    ) -> AutomaticEmbeddingStatus:
        """Observe current limits and reuse completed state, never calibrate."""
        input_type = _task(input_type)
        category = _category(self.config.category if category is None else category)
        bound = self.config.timeout if timeout is None else timeout
        with self._operation(bound, cancel) as operation:
            with self._observed(operation, input_type=input_type) as observed:
                _, _, _, binding, context_budget, _ = observed
                profile = self._cached_profile(binding, operation)
                snapshot = self._budget_snapshot(
                    binding, context_budget, category, profile, operation
                )
                return AutomaticEmbeddingStatus(
                    True,
                    None if snapshot is None else snapshot.max_input_chars,
                    category,
                    input_type,
                    "character_budget_not_calibrated" if snapshot is None else None,
                    self._representation(binding),
                    None if profile is None else profile.profile_id,
                    snapshot,
                )

    def embed_texts(self, texts, *, input_type="document", timeout=None, cancel=None):
        """Submit strings directly; no calibration or token-count preflight."""
        input_type = _task(input_type)
        if isinstance(texts, (str, bytes)) or not isinstance(texts, (list, tuple)):
            raise ValueError("texts must be a finite sequence of strings")
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("texts must contain only strings")
        bound = self.config.timeout if timeout is None else timeout
        with self._operation(bound, cancel) as operation:
            if not texts:
                return []
            status, policy, adapter = self._ready(operation, input_type=input_type)
            try:
                result = adapter._embed_texts_direct(
                    texts,
                    policy=policy,
                    input_type=input_type,
                    timeout=operation.remaining(),
                    cancel=operation.probe,
                )
            except LLMCancelledError:
                operation.check()
                raise
            operation.check()
            if isinstance(result, _HalvingResult):
                error = _halving_error(result, status.representation_id, input_type)
                operation.check()
                raise error from next(iter(result.rejections.values())).original
            return result

    def embed_query(self, text, *, timeout=None, cancel=None):
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        return self.embed_texts(
            [text], input_type="query", timeout=timeout, cancel=cancel
        )[0]

    def astream_texts(
        self,
        windows,
        *,
        input_type="document",
        max_pending_windows=2,
        timeout=None,
        cancel=None,
        close_source=False,
    ):
        """Return inert direct intake, with bounded readiness off the event loop.

        ``timeout`` bounds readiness and each later window independently; the
        default is ``config.timeout``. It is not an entire-stream deadline.
        """
        return _AutomaticEmbeddingStream(
            self,
            windows,
            input_type=_task(input_type),
            max_pending_windows=max_pending_windows,
            timeout=timeout,
            cancel=cancel,
            close_source=close_source,
        )

    def close(self):
        """Stop admission and drain calibration and shared inference ownership."""
        self._check_close_reentrancy()
        with self._close_lock:
            with self._condition:
                if self._closed:
                    return
                self._closing.set()
                for operation in self._operations:
                    operation.stop.set()
                adapter = self._adapter
            if adapter is not None:
                adapter.close()
            with self._condition:
                while self._operations:
                    self._condition.wait(0.05)
                if self._owns_client and self._client is not None:
                    self._client.close()
                self._closed = True
                self._condition.notify_all()

    async def aclose(self):
        self._check_close_reentrancy()

        async def cleanup():
            with self._condition:
                self._closing.set()
                streams = tuple(self._streams)
                for operation in self._operations:
                    operation.stop.set()
            await asyncio.gather(
                *(stream.aclose() for stream in streams), return_exceptions=True
            )
            adapter = self._adapter
            if adapter is not None:
                await adapter.aclose()
            await asyncio.to_thread(self.close)

        await _drain(cleanup())

    def __enter__(self):
        if self._closing.is_set():
            raise LLMBackendError("llamacpp_automatic_embedding_closed")
        return self

    def __exit__(self, *_):
        self.close()

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *_):
        await self.aclose()


class _AutomaticEmbeddingStream:
    """Lazy readiness ownership before delegating to EmbeddingWindowStream."""

    def __init__(
        self,
        service,
        windows,
        *,
        input_type,
        max_pending_windows,
        timeout,
        cancel,
        close_source,
    ):
        if not callable(getattr(windows, "__aiter__", None)):
            raise ValueError("windows must be an async iterable")
        if _positive_int(max_pending_windows, "max_pending_windows") is None:
            raise ValueError("max_pending_windows must be a positive integer")
        if timeout is not None:
            _finite(timeout, "timeout", positive=True)
        if cancel is not None and not callable(cancel):
            raise ValueError("cancel must be a callable or None")
        if not isinstance(close_source, bool):
            raise ValueError("close_source must be a boolean")
        self.service, self.windows = service, windows
        self.input_type = input_type
        self.maximum, self.timeout = max_pending_windows, timeout
        self.cancel, self.close_source = cancel, close_source
        self._stop = Event()
        self._loop = self._initialization = self._delegate = None
        self._conversion = None
        self._closed = self._next_active = False
        self._callback_failure = None
        self._admission_scope = _EmbeddingAdmissionScope()

    def __aiter__(self):
        return self

    def _check_loop(self):
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise LLMBackendError("llamacpp_embedding_async_event_loop_mismatch")
        return loop

    def _cancelled(self):
        if self._stop.is_set() or self._callback_failure is not None:
            return True
        if self.cancel is None:
            return False
        try:
            return self.service._invoke_cancel(self.cancel)
        except _ReentrantCloseError as exc:
            # Native adapter probes are boolean and deliberately sanitize
            # arbitrary callback failures. Preserve this facade policy error
            # until the stream's owned request has actually drained.
            self._callback_failure = exc
            return True

    def _primary_failure(self, exc):
        return (
            exc
            if isinstance(exc, asyncio.CancelledError)
            else self._callback_failure or exc
        )

    def _initialize(self):
        timeout = self.service.config.timeout if self.timeout is None else self.timeout
        with self.service._operation(timeout, self._cancelled) as operation:
            return self.service._ready(
                operation,
                input_type=self.input_type,
                admission_scope=self._admission_scope,
            )

    async def _start(self):
        loop = self._check_loop()
        if self._closed or self._delegate is not None:
            return
        self._loop = loop
        with self.service._condition:
            if self.service._closing.is_set():
                raise LLMBackendError("llamacpp_automatic_embedding_closed")
            self.service._streams.add(self)
        if self._initialization is None:
            self._initialization = loop.create_task(asyncio.to_thread(self._initialize))
        status, policy, adapter = await asyncio.shield(self._initialization)
        self._status = status
        if self._closed or self._stop.is_set():
            return
        if self._delegate is not None:
            return
        self._delegate = adapter._astream_texts_direct(
            self.windows,
            policy=policy,
            input_type=self.input_type,
            max_pending_windows=self.maximum,
            timeout=self.service.config.timeout
            if self.timeout is None
            else self.timeout,
            cancel=self._cancelled,
            close_source=self.close_source,
            admission_scope=self._admission_scope,
        )
        await self._delegate.__aenter__()

    async def __anext__(self):
        self._check_loop()
        if self._next_active:
            raise RuntimeError("concurrent iteration of one embedding stream")
        self._next_active = True
        try:
            await self._start()
            if self._closed or self._delegate is None:
                raise StopAsyncIteration
            result = await anext(self._delegate)
            if isinstance(result, _HalvingResult):
                self._conversion = self._loop.create_task(
                    asyncio.to_thread(
                        _halving_error,
                        result,
                        self._status.representation_id,
                        self.input_type,
                    )
                )
                try:
                    error = await _drain(self._conversion)
                finally:
                    self._conversion = None
                if self.service._closing.is_set():
                    raise LLMBackendError("llamacpp_automatic_embedding_closed")
                if self._cancelled():
                    raise self._callback_failure or LLMCancelledError()
                if result.deadline is not None and time.monotonic() >= result.deadline:
                    raise LLMTimeoutError("llamacpp_embedding_timeout")
                raise error from next(iter(result.rejections.values())).original
            return result
        except EmbeddingChunkSplitRequiredError:
            # A recoverable window is delivered in source order. The iterator
            # stays usable so later buffered windows are neither lost nor
            # silently replayed while the consumer handles the local halves.
            raise
        except BaseException as exc:
            await _drain(self.aclose(), primary=self._primary_failure(exc))
        finally:
            self._next_active = False

    async def __aenter__(self):
        try:
            await self._start()
            return self
        except BaseException as exc:
            await _drain(self.aclose(), primary=self._primary_failure(exc))

    async def __aexit__(self, exc_type, exc, traceback):
        await _drain(self.aclose(), primary=exc)

    async def aclose(self):
        self._check_loop()
        self._closed = True
        self._stop.set()

        async def cleanup():
            if self._initialization is not None:
                await asyncio.gather(self._initialization, return_exceptions=True)
            if self._delegate is not None:
                await self._delegate.aclose()
            if self._conversion is not None:
                await asyncio.gather(self._conversion, return_exceptions=True)
            self._admission_scope.clear()
            with self.service._condition:
                self.service._streams.discard(self)

        await _drain(cleanup())
