"""HTTP embeddings; behavioral baseline: lmstudio-agent-runtime 9300f295."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
import math
import os
import secrets
from threading import BoundedSemaphore, Condition, Event, Lock
import time
from typing import Any, Literal, Sequence

from runtime_llama.client import ClientHTTPError, LlamaCppClient
from runtime_llama.configuration import AdapterProfile, EmbeddingPrefixes
from runtime_llama.contracts import EmbeddingAdapter
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError
from runtime_llama.replica_discovery import (
    EmbeddingPool,
    ReplicaIdentity,
    catalog_aliases,
    discover_embedding_pool,
    verify_loaded_replica,
)

from runtime_llama.embedding_scheduler import EmbeddingScheduler
from runtime_llama.embedding_metadata import loaded_batch_limit
from runtime_llama.embedding_preparation import (
    PreparedEmbeddingInput,
    issue_preparation,
    valid_preparation,
)

_USE_CACHED_CONTEXT = object()


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
    tokenizer_build: str | None = None
    props_checked: bool = False
    props_identity_verified: bool = False
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

    def remaining(self) -> float:
        self.check()
        return max(0.000001, self.deadline - time.monotonic())

    def fail(self, exc: BaseException) -> None:
        with self.lock:
            if self.failure is None:
                self.failure = exc
            self.stop.set()

    def validate_dimensions(self, vectors: list[list[float]]) -> None:
        with self.lock:
            for vector in vectors:
                if self.dimension is not None and len(vector) != self.dimension:
                    raise LLMBackendError("llamacpp_embedding_dimension_mismatch")
                self.dimension = len(vector)


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
        self.batch_token_limit = profile.batch_token_limit
        self.token_count_parallelism = int(profile.token_count_parallelism or 1)
        self.normalize = bool(profile.normalize)
        self.configured_max_input_tokens = profile.max_input_tokens
        self.embedding_dim = int(profile.dimension or 0)
        generation = profile.generation
        api_key = generation.api_key or (
            os.environ.get(generation.api_key_env) if generation.api_key_env else None
        )
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else LlamaCppClient(
                self.base_url,
                api_key=api_key,
                timeout=profile.timeout,
                verify_ssl=profile.model_config.verify_ssl,
            )
        )
        self._loaded_context_length: int | None = None
        self._props_checked = False
        self._closed = False
        self._closing = Event()
        self._close_lock = Lock()
        self._calls_lock = Lock()
        self._calls_condition = Condition(self._calls_lock)
        self._calls: set[_Call] = set()
        self._tokenizer_permits = BoundedSemaphore(self.token_count_parallelism)
        self._preparation_key = secrets.token_bytes(32)
        self._dimension_lock = Lock()
        self.embedding_request_parallelism = int(
            getattr(profile, "embedding_request_parallelism", 1)
        )
        self.embedding_queue_size = int(getattr(profile, "embedding_queue_size", 16))
        self._discovery_enabled = bool(
            getattr(profile, "embedding_replica_discovery", False)
        )
        self._discovery_lock = Lock()
        self._pool_condition = Condition()
        self._pool_users = 0
        self._discovered_pool: EmbeddingPool | None = None
        self._pinned_identity: ReplicaIdentity | None = None
        self._fallback_alias: str | None = None
        self._replica_ids: dict[tuple[str, tuple[str, str] | None], int] = {}
        self._next_replica_id = 0
        self._discovery_error: str | None = None
        self._discovery_server_healthy = False
        configured_replicas = getattr(profile, "embedding_replicas", ())
        self._replicas = {
            replica.model: _Replica(
                replica.model,
                replica.max_in_flight,
                replica.max_input_tokens,
                replica.batch_token_limit,
                self.batch_size,
                self.batch_size,
            )
            for replica in configured_replicas
        } or {
            self.model_name: _Replica(
                self.model_name,
                self.embedding_request_parallelism,
                None,
                None,
                self.batch_size,
                self.batch_size,
            )
        }
        self._primary = next(iter(self._replicas.values()))
        for replica in self._replicas.values():
            replica.accepted_aliases.add(replica.model)
        self._scheduler = EmbeddingScheduler(
            self.embedding_request_parallelism,
            self.embedding_queue_size,
            {model: replica.max_in_flight for model, replica in self._replicas.items()},
        )

    def _join_pool(self, call: _Call, timeout: float) -> bool:
        """Observe before admission; replace topology only after old calls drain."""
        if not self._discovery_enabled:
            return False
        deadline = time.monotonic() + timeout

        def check_wait():
            call.check()
            if time.monotonic() >= deadline:
                raise LLMTimeoutError("llamacpp_embedding_discovery_wait_timeout")

        while not self._discovery_lock.acquire(timeout=0.05):
            check_wait()
        try:
            call.check()
            self._discovery_server_healthy = False
            health = self._client.health(
                timeout=call.remaining(), cancel=call.probe, cancel_wait=True
            )
            self._discovery_server_healthy = (
                isinstance(health, dict) and health.get("status") == "ok"
            )
            if not self._discovery_server_healthy:
                raise LLMBackendError("llamacpp_embedding_server_not_healthy")
            catalog = self._discovery_catalog(
                timeout=call.remaining(), cancel=call.probe
            )
            pool = discover_embedding_pool(
                catalog,
                self.model_name,
                pinned=self._pinned_identity,
                fallback_alias=self._fallback_alias,
            )
            with self._pool_condition:
                if self._discovered_pool is None or (
                    pool.topology != self._discovered_pool.topology
                ):
                    while self._pool_users:
                        check_wait()
                        self._pool_condition.wait(0.05)
                    call.check()
                    replicas = {}
                    for observed in pool.replicas:
                        key = (
                            observed.model,
                            observed.identity.key if observed.identity else None,
                        )
                        if key not in self._replica_ids:
                            self._replica_ids[key] = self._next_replica_id
                            self._next_replica_id += 1
                        replicas[observed.model] = _Replica(
                            observed.model,
                            1,
                            None,
                            None,
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
                        self.embedding_request_parallelism,
                        self.embedding_queue_size,
                        {model: 1 for model in replicas},
                    )
                    self._scheduler.close()
                    self._scheduler = scheduler
                    self._replicas = replicas
                    self._primary = next(iter(replicas.values()))
                    self._loaded_context_length = None
                    self._discovered_pool = pool
                    if pool.identity is not None:
                        self._pinned_identity = pool.identity
                    if pool.fallback_alias is not None:
                        self._fallback_alias = pool.fallback_alias
                self._pool_users += 1
                self._discovery_error = None
            return True
        except (LLMBackendError, LLMTimeoutError, LLMCancelledError) as exc:
            self._discovery_error = type(exc).__name__
            raise
        finally:
            self._discovery_lock.release()

    def _leave_pool(self, joined: bool) -> None:
        if joined:
            with self._pool_condition:
                self._pool_users -= 1
                self._pool_condition.notify_all()

    def _verify_discovered_replica(self, replica: _Replica, *, timeout, cancel):
        models = getattr(self._client, "models", None)
        if not callable(models):
            if self._discovery_enabled:
                raise LLMBackendError("llamacpp_embedding_invalid_model_catalog")
            return None
        catalog = (
            self._discovery_catalog(timeout=timeout, cancel=cancel)
            if self._discovery_enabled
            else models(timeout=timeout, cancel=cancel, cancel_wait=True)
        )
        if self._discovery_enabled:
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
        if self._discovery_enabled:
            raise LLMBackendError("llamacpp_embedding_invalid_model_catalog")
        return None

    def _discovery_catalog(self, *, timeout, cancel):
        catalog = self._client.models(timeout=timeout, cancel=cancel, cancel_wait=True)
        entries = catalog.get("data") if isinstance(catalog, dict) else None
        if (
            isinstance(entries, list)
            and len(entries) == 1
            and isinstance(entries[0], dict)
            and "status" not in entries[0]
        ):
            # Native single-model servers omit status. Root /props does not
            # select a router model and cannot request an unloaded alias.
            props = self._client.props(timeout=timeout, cancel=cancel, cancel_wait=True)
            if (
                isinstance(props, dict)
                and props.get("role") != "router"
                and props.get("is_sleeping") is not True
                and isinstance(props.get("model_alias"), str)
                and props["model_alias"] in catalog_aliases(entries[0])
            ):
                return {
                    **catalog,
                    "data": [{**entries[0], "status": {"value": "loaded"}}],
                }
        return catalog

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
        self, *, replica: _Replica | None = None, timeout=None, cancel=None
    ) -> int | None:
        return self._observe_replica(
            replica=replica, timeout=timeout, cancel=cancel
        ).context_length

    def _observe_replica(
        self, *, replica: _Replica | None = None, timeout=None, cancel=None, call=None
    ) -> _Observation:
        replica = replica or self._primary
        remaining = call.remaining if call is not None else lambda: timeout
        entry = self._verify_discovered_replica(
            replica, timeout=remaining(), cancel=cancel
        )
        loaded_context_length = None
        replica.props_checked = True
        self._props_checked = True
        try:
            props = self._client.props(
                model=replica.model,
                timeout=remaining(),
                cancel=cancel,
                cancel_wait=True,
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
            if alias is not None and (
                not isinstance(alias, str) or alias not in replica.accepted_aliases
            ):
                try:
                    catalog = self._client.models(
                        timeout=remaining(), cancel=cancel, cancel_wait=True
                    )
                    entries = catalog.get("data") if isinstance(catalog, dict) else None
                    matches = self._catalog_matches(replica.model, entries)
                    if len(matches) == 1:
                        replica.accepted_aliases.update(
                            self._catalog_aliases(matches[0])
                        )
                except (LLMBackendError, LLMTimeoutError, LLMCancelledError):
                    self._invalidate_replica(replica)
                    raise
                if not isinstance(alias, str) or alias not in replica.accepted_aliases:
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
            native_batch = loaded_batch_limit(props, entry)
        except LLMBackendError:
            self._invalidate_replica(replica)
            raise
        context_limit = self._input_limit(loaded_context_length, replica=replica)
        batch_limit = self._minimum(
            self.batch_token_limit, replica.batch_token_limit, native_batch
        )
        with replica.lock:
            replica.context_length = loaded_context_length
            replica.total_slots = total_slots
            replica.props_identity_verified = isinstance(alias, str)
            replica.loaded_batch_token_limit = native_batch
            replica.tokenizer_build = build
            if replica is self._primary:
                self._loaded_context_length = loaded_context_length
        return _Observation(loaded_context_length, context_limit, batch_limit, build)

    @staticmethod
    def _minimum(*values):
        return min((value for value in values if value is not None), default=None)

    def _invalidate_replica(self, replica: _Replica) -> None:
        replica.context_length = None
        replica.total_slots = None
        replica.loaded_batch_token_limit = None
        replica.tokenizer_build = None
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

    def _token_count(self, text: str, replica: _Replica, call: _Call) -> int:
        acquired = False
        try:
            while not acquired:
                call.check()
                acquired = self._tokenizer_permits.acquire(
                    timeout=min(0.05, call.remaining())
                )
            result = self._client.tokenize(
                text,
                model=replica.model,
                add_special=True,
                parse_special=True,
                with_pieces=False,
                timeout=call.remaining(),
                cancel=call.probe,
                cancel_wait=True,
            )
            call.check()
            tokens = result.get("tokens") if isinstance(result, dict) else None
            if not isinstance(tokens, list) or any(
                not isinstance(token, int) or isinstance(token, bool) or token < 0
                for token in tokens
            ):
                raise LLMBackendError("llamacpp_embedding_invalid_tokenization")
            return len(tokens)
        finally:
            if acquired:
                self._tokenizer_permits.release()

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

    def _prepare_quota(self, inputs, replica, call, observation, conservative):
        context_limit = self._minimum(observation.context_token_limit, conservative[0])
        batch_limit = self._minimum(observation.batch_token_limit, conservative[1])

        def prepare(item):
            try:
                call.check()
                if isinstance(item, PreparedEmbeddingInput):
                    text, input_type = item.text, item.input_type
                    count = (
                        item.token_count
                        if self._can_reuse(item, replica, observation)
                        else self._token_count(text, replica, call)
                    )
                else:
                    text, input_type = item
                    count = self._token_count(text, replica, call)
                identity = replica.identity
                return issue_preparation(
                    self._preparation_key,
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
                )
            except BaseException as exc:
                call.fail(exc)
                raise

        workers = min(self.token_count_parallelism, len(inputs))
        if workers <= 1:
            return [prepare(item) for item in inputs]
        results = [None] * len(inputs)
        pending = {}
        cursor = 0
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="runtime_llama_tokenizer"
        ) as executor:
            try:
                while cursor < len(inputs) or pending:
                    call.check()
                    while cursor < len(inputs) and len(pending) < workers:
                        call.check()
                        pending[executor.submit(prepare, inputs[cursor])] = cursor
                        cursor += 1
                    finished, _ = wait(
                        pending, timeout=0.05, return_when=FIRST_COMPLETED
                    )
                    for future in finished:
                        results[pending.pop(future)] = future.result()
            except BaseException as exc:
                call.fail(exc)
                for future in pending:
                    try:
                        future.result()
                    except BaseException:
                        pass
                raise
        return results

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
    def _normalize(vector: list[float]) -> list[float]:
        scale = max(abs(value) for value in vector)
        if scale == 0:
            return vector
        scaled = [value / scale for value in vector]
        norm = math.sqrt(math.fsum(value * value for value in scaled))
        return [value / norm for value in scaled]

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
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in raw
                )
            ):
                raise LLMBackendError("llamacpp_embedding_invalid_response")
            try:
                vector = [float(value) for value in raw]
            except (ValueError, TypeError, OverflowError) as exc:
                raise LLMBackendError("llamacpp_embedding_invalid_response") from exc
            if not all(math.isfinite(value) for value in vector):
                raise LLMBackendError("llamacpp_embedding_invalid_response")
            if expected_dimension is not None and len(vector) != expected_dimension:
                raise LLMBackendError("llamacpp_embedding_dimension_mismatch")
            expected_dimension = len(vector)
            indexed[index] = self._normalize(vector) if self.normalize else vector
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
    ) -> list[list[float]]:
        if call is not None:
            self._verify_discovered_replica(
                replica, timeout=call.remaining(), cancel=call.probe
            )
            timeout = call.remaining()
        if cancel is not None and cancel():
            raise LLMCancelledError()
        with replica.lock:
            generation = replica.generation
        try:
            result = self._client.embeddings(
                texts,
                model=replica.model,
                timeout=timeout,
                cancel=cancel,
                cancel_wait=True,
            )
        except ClientHTTPError as exc:
            if (
                exc.status_code != 400
                or getattr(exc, "error_type", None) != "exceed_context_size_error"
                or len(texts) <= 1
            ):
                raise
            midpoint = len(texts) // 2
            self._tighten_adaptive_batch_size(
                max(midpoint, len(texts) - midpoint), replica=replica
            )
            return self._request_batch_adaptive(
                texts[:midpoint],
                replica=replica,
                timeout=timeout,
                cancel=cancel,
                recovery_child=True,
                call=call,
            ) + self._request_batch_adaptive(
                texts[midpoint:],
                replica=replica,
                timeout=timeout,
                cancel=cancel,
                recovery_child=True,
                call=call,
            )
        vectors = self._parse_batch_response(result, expected_count=len(texts))
        if not recovery_child:
            self._record_batch_success(len(texts), generation, replica)
        return vectors

    def _run_job(self, inputs, model, call, operation, conservative):
        replica = self._replicas[model]
        try:
            call.check()
            observation = self._observe_replica(
                replica=replica, cancel=call.probe, call=call
            )
            prepared = self._prepare_quota(
                inputs, replica, call, observation, conservative
            )
            if operation == "prepare":
                call.check()
                return prepared
            vectors: list[list[float]] = []
            start = 0
            while start < len(prepared):
                call.check()
                observation = self._observe_replica(
                    replica=replica, cancel=call.probe, call=call
                )
                if observation.tokenizer_build != prepared[0]._tokenizer_build:
                    raise LLMBackendError("llamacpp_embedding_tokenizer_policy_changed")
                # A later incomplete observation cannot erase a known bound
                # under which this quota started preparing.
                observation = _Observation(
                    observation.context_length,
                    self._minimum(
                        observation.context_token_limit, prepared[0].context_token_limit
                    ),
                    self._minimum(
                        observation.batch_token_limit, prepared[0].batch_token_limit
                    ),
                    observation.tokenizer_build,
                )
                for record in prepared[start:]:
                    self._admit(record, observation)
                maximum = self._current_batch_size(len(prepared) - start, replica)
                stop, tokens = start, 0
                while stop < start + maximum:
                    count = prepared[stop].token_count
                    if (
                        stop > start
                        and observation.batch_token_limit is not None
                        and tokens + count > observation.batch_token_limit
                    ):
                        break
                    tokens += count
                    stop += 1
                provider_batch = [record.text for record in prepared[start:stop]]
                result = self._request_batch_adaptive(
                    provider_batch,
                    replica=replica,
                    cancel=call.probe,
                    call=call,
                )
                call.validate_dimensions(result)
                vectors.extend(result)
                start = stop
            call.check()
            return vectors
        except BaseException as exc:
            call.fail(exc)
            raise

    @staticmethod
    def _validate_task(input_type):
        if not isinstance(input_type, str) or input_type not in (
            "document",
            "query",
            "raw",
        ):
            raise ValueError("input_type must be document, query or raw")

    def _execute(self, inputs, *, operation, timeout, cancel, call=None):
        call = _Call(self, cancel, timeout) if call is None else call
        with self._calls_condition:
            if self._closed:
                raise LLMBackendError("llamacpp_embedding_adapter_closed")
            self._calls.add(call)
        outputs = [None] * len(inputs)
        pending = {}
        joined = False
        try:
            call.check()
            joined = self._join_pool(call, call.remaining())
            observations = [
                self._observe_replica(replica=replica, cancel=call.probe, call=call)
                for replica in self._replicas.values()
            ]
            conservative = (
                self._minimum(*(item.context_token_limit for item in observations)),
                self._minimum(*(item.batch_token_limit for item in observations)),
            )
            window_start = 0
            descriptors = []
            pending_limit = max(
                1, self.embedding_request_parallelism + self.embedding_queue_size
            )
            while window_start < len(inputs) or descriptors or pending:
                call.check()
                while len(pending) < pending_limit and (
                    descriptors or window_start < len(inputs)
                ):
                    call.check()
                    if not descriptors:
                        window_end = min(len(inputs), window_start + self.batch_size)
                        offset = window_start
                        for model, size in self._scheduler.plan_window(
                            window_end - window_start
                        ):
                            descriptors.append((model, offset, size))
                            offset += size
                        window_start = window_end
                    model, offset, size = descriptors.pop(0)
                    quota = inputs[offset : offset + size]
                    future = self._scheduler.submit(
                        lambda alias, quota=quota: self._run_job(
                            quota, alias, call, operation, conservative
                        ),
                        call.check,
                        model=model,
                    )
                    pending[future] = (offset, size)
                if pending:
                    finished, _ = wait(
                        pending, timeout=0.05, return_when=FIRST_COMPLETED
                    )
                    for future in finished:
                        offset, expected = pending.pop(future)
                        result = future.result()
                        if len(result) != expected:
                            raise LLMBackendError(
                                "llamacpp_embedding_count_or_response_mismatch"
                            )
                        outputs[offset : offset + expected] = result
            call.check()
            if any(output is None for output in outputs):
                raise LLMBackendError("llamacpp_embedding_count_or_response_mismatch")
            if operation != "prepare":
                with self._dimension_lock:
                    if self.embedding_dim and call.dimension != self.embedding_dim:
                        raise LLMBackendError("llamacpp_embedding_dimension_mismatch")
                    if call.dimension is not None:
                        self.embedding_dim = call.dimension
            return outputs
        except BaseException as exc:
            call.fail(exc)
            for future in pending:
                try:
                    future.result()
                except BaseException:
                    pass
            raise call.failure if call.failure is not None else exc
        finally:
            self._leave_pool(joined)
            with self._calls_condition:
                self._calls.discard(call)
                self._calls_condition.notify_all()

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
        inputs = [(prefix + str(text), input_type) for text in texts]
        return self._execute(
            inputs, operation="embed", timeout=timeout, cancel=cancel, call=call
        )

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
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("texts must contain only strings")
        if not texts:
            return ()
        prefix = (
            "" if input_type == "raw" else getattr(self._embedding_prefixes, input_type)
        )
        inputs = [(prefix + text, input_type) for text in texts]
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
        if any(not valid_preparation(self._preparation_key, item) for item in inputs):
            raise LLMBackendError("llamacpp_embedding_invalid_preparation")
        if not inputs:
            return []
        return self._execute(
            list(inputs), operation="embed", timeout=timeout, cancel=cancel, call=call
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
        try:
            joined = self._join_pool(_Call(self, None), self.profile.timeout)
            return self._readiness_current()
        except (LLMBackendError, LLMTimeoutError, LLMCancelledError):
            return {
                "server_healthy": self._discovery_server_healthy,
                "model_ready": False,
                "models": {model: "unknown" for model in self._replicas},
            }
        finally:
            self._leave_pool(joined)

    def _readiness_current(self) -> dict[str, Any]:
        states = {model: "unknown" for model in self._replicas}
        result = {"server_healthy": False, "model_ready": False, "models": states}
        if self._closed:
            for replica in self._replicas.values():
                self._invalidate_replica(replica)
            return result
        try:
            health = self._client.health(timeout=self.profile.timeout, cancel_wait=True)
            result["server_healthy"] = (
                isinstance(health, dict) and health.get("status") == "ok"
            )
            if not result["server_healthy"]:
                for replica in self._replicas.values():
                    self._invalidate_replica(replica)
                return result
            catalog = self._client.models(
                timeout=self.profile.timeout, cancel_wait=True
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
                    self._refresh_context(replica=replica, timeout=self.profile.timeout)
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

    def health_status(self) -> dict[str, Any]:
        joined = False
        try:
            joined = self._join_pool(_Call(self, None), self.profile.timeout)
            return self._health_status_current()
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

    def _health_status_current(self) -> dict[str, Any]:
        readiness = self._readiness_current()
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
                        replica.loaded_batch_token_limit,
                    ),
                    "loaded_batch_token_limit": replica.loaded_batch_token_limit,
                    "tokenizer_build": replica.tokenizer_build,
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
            "replicas": replicas,
        }

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            with self._calls_lock:
                self._closed = True
                self._closing.set()
                for call in self._calls:
                    call.stop.set()
            with self._discovery_lock:
                self._scheduler.close()
                if self._owns_client:
                    self._client.close()
            with self._calls_condition:
                while self._calls:
                    self._calls_condition.wait(0.05)
