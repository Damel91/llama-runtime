"""Read-only runtime detection and loaded-model routing across known endpoints.

Discovery never scans a network, loads a model or submits inference. Endpoints
come from application/deployment configuration; the requested model determines
which of the supported, positively identified runtimes the library selects.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import math
import os
import re
from threading import Event
import time
from typing import Any, Callable, Literal, Sequence

from runtime_llama.client import (
    CancellationProbe,
    ClientHTTPError,
    LlamaCppClient,
    _cancelled,
)
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError
from runtime_llama.model_discovery import (
    ModelCatalog,
    ModelObservation,
    _ENV_NAME,
    _observe,
    _root_proof,
)

RuntimeBackend = Literal["llamacpp", "strata"]
_BACKENDS = {"auto", "llamacpp", "strata"}
_NATIVE_BUILD = re.compile(
    r"(?:b\d+(?:-[0-9a-f]+)?(?:$|[ (])|\d+ \([0-9a-f]{7,40}\)(?:$| )|llama[. _-]?(?:cpp|server))",
    re.I,
)


def _valid_name(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not any(
            ord(char) < 32 or 127 <= ord(char) <= 159 or 0xD800 <= ord(char) <= 0xDFFF
            for char in value
        )
    )


@dataclass(frozen=True)
class RuntimeEndpoint:
    """Deployment-owned network access, with automatic runtime identification.

    ``name`` is a stable local identifier, never a backend choice made by the
    inference consumer. An explicit backend is available for older servers
    without identifying metadata, but contradicting server markers fail closed.
    Credentials and proxy routing are omitted from representations.
    """

    name: str
    base_url: str
    backend: Literal["auto", "llamacpp", "strata"] = "auto"
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str | None = field(default=None, repr=False)
    verify_ssl: bool = True
    proxy_url: str | None = field(default=None, repr=False)
    http_connection_policy: Literal["close", "reuse"] = "close"

    @classmethod
    def from_value(cls, value: RuntimeEndpoint | Mapping[str, Any]) -> RuntimeEndpoint:
        """Normalize an endpoint or a portable JSON deployment configuration."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("runtime endpoint must be a RuntimeEndpoint or mapping")
        allowed = {
            "name",
            "base_url",
            "backend",
            "api_key",
            "api_key_env",
            "verify_ssl",
            "proxy_url",
            "http_connection_policy",
        }
        if set(value) - allowed or "base_url" not in value:
            raise ValueError("runtime endpoint configuration fields are invalid")
        configured = dict(value)
        configured.setdefault("name", configured["base_url"])
        return cls(**configured)

    def __post_init__(self) -> None:
        if not _valid_name(self.name):
            raise ValueError("runtime endpoint name must be nonempty text")
        aliases = {
            "llama.cpp": "llamacpp",
            "llama_cpp": "llamacpp",
            "llama-server": "llamacpp",
        }
        normalized = (
            aliases.get(self.backend, self.backend)
            if isinstance(self.backend, str)
            else self.backend
        )
        if not isinstance(normalized, str) or normalized not in _BACKENDS:
            raise ValueError(
                "runtime endpoint backend must be auto, llamacpp or strata"
            )
        object.__setattr__(self, "backend", normalized)
        if not isinstance(self.verify_ssl, bool):
            raise ValueError("runtime endpoint verify_ssl must be boolean")
        if self.api_key is not None and self.api_key_env is not None:
            raise ValueError("runtime endpoint accepts one authentication source")
        if self.api_key is not None and (
            not _valid_name(self.api_key)
            or any(ord(char) == 127 for char in self.api_key)
        ):
            raise ValueError("runtime endpoint authentication value is invalid")
        if self.api_key_env is not None and (
            not isinstance(self.api_key_env, str)
            or not _ENV_NAME.fullmatch(self.api_key_env)
        ):
            raise ValueError(
                "runtime endpoint authentication environment name is invalid"
            )
        # Validate URL/proxy/connection semantics without opening a connection.
        with LlamaCppClient(
            self.base_url,
            timeout=1.0,
            verify_ssl=self.verify_ssl,
            trust_env=False,
            proxy_url=self.proxy_url,
            connection_policy=self.http_connection_policy,
        ) as client:
            root = client.base_url
            if root.endswith("/v1"):
                root = root[:-3]
            object.__setattr__(self, "base_url", root)
            object.__setattr__(self, "proxy_url", client.proxy_url)

    def resolved_api_key(self) -> str | None:
        if self.api_key_env is None:
            return self.api_key
        value = os.environ.get(self.api_key_env)
        if not _valid_name(value):
            raise ValueError(
                "runtime endpoint authentication environment value is unavailable"
            )
        return value


@dataclass(frozen=True)
class RuntimeObservation:
    """One immutable probe result, including unavailable/unsupported runtimes."""

    endpoint: RuntimeEndpoint
    backend: RuntimeBackend | None
    catalog: ModelCatalog | None
    diagnostics: tuple[str, ...]
    http_requests: int
    elapsed_seconds: float
    available: bool = False
    llm_model_id: str | None = None

    @property
    def loaded_models(self) -> tuple[ModelObservation, ...]:
        return self.catalog.loaded_models if self.catalog is not None else ()


@dataclass(frozen=True)
class RuntimeModelSelection:
    """A selected observed model; not a reservation or proof of future state."""

    runtime: RuntimeObservation
    model: ModelObservation

    @property
    def backend(self) -> RuntimeBackend:
        # Selections are created only from positively identified observations.
        if self.runtime.backend is None:
            raise LLMBackendError("runtime_backend_unsupported")
        return self.runtime.backend


@dataclass(frozen=True)
class RuntimeCatalog:
    runtimes: tuple[RuntimeObservation, ...]
    elapsed_seconds: float

    @property
    def http_requests(self) -> int:
        return sum(runtime.http_requests for runtime in self.runtimes)

    def select(
        self, model: str, *, purpose: Literal["llm", "embedding"] = "llm"
    ) -> RuntimeModelSelection:
        """Select a loaded exact ID, otherwise an unambiguous exact alias.

        Repeated canonical IDs with compatible observed identities use stable
        endpoint configuration order. Different known weight hashes or aliases
        referring to different canonical IDs are an error, never a guess.
        Names and paths are not evidence that distinct weights are equivalent.
        """
        if not _valid_name(model):
            raise ValueError("runtime requested model must be nonempty text")
        if purpose not in {"llm", "embedding"}:
            raise ValueError("runtime selection purpose must be llm or embedding")
        candidates = []
        exact_known = False
        for runtime in self.runtimes:
            if not runtime.available or runtime.backend is None:
                continue
            if purpose == "embedding" and runtime.backend != "llamacpp":
                continue
            exact_known |= (
                any(observed.id == model for observed in runtime.catalog.models)
                if runtime.catalog is not None
                else False
            )
            for observed in runtime.loaded_models:
                if (
                    purpose == "llm"
                    and runtime.backend == "strata"
                    and observed.id != runtime.llm_model_id
                ):
                    continue
                capability = (
                    observed.capabilities.chat
                    if purpose == "llm"
                    else observed.capabilities.embeddings
                )
                if capability is False or observed.failed is True:
                    continue
                if observed.id == model or model in observed.aliases:
                    candidates.append(RuntimeModelSelection(runtime, observed))
        if exact_known:
            candidates = [item for item in candidates if item.model.id == model]
        if not candidates:
            raise LLMBackendError("runtime_model_unavailable")
        if len({item.model.id for item in candidates}) != 1:
            raise LLMBackendError("runtime_model_ambiguous")
        for attr in ("content_sha256", "embedding_sha256"):
            known = {
                value
                for item in candidates
                if (value := getattr(item.model.identity, attr)) is not None
            }
            if len(known) > 1:
                raise LLMBackendError("runtime_model_ambiguous_identity")
        return candidates[0]


def _runtime_marker(value: Any) -> RuntimeBackend | None:
    if not isinstance(value, str):
        return None
    lower = value.lower().strip()
    if lower == "strata" or lower.startswith("strata "):
        return "strata"
    if lower in {"llama.cpp", "llama-server", "llamacpp"} or _NATIVE_BUILD.match(lower):
        return "llamacpp"
    return None


def _backend(
    endpoint: RuntimeEndpoint, health: dict[str, Any], props: dict[str, Any]
) -> RuntimeBackend | None:
    markers = {
        marker
        for value in (health.get("service"), props.get("build_info"))
        if (marker := _runtime_marker(value)) is not None
    }
    if len(markers) > 1:
        raise LLMBackendError("runtime_identity_conflict")
    observed = next(iter(markers), None)
    if endpoint.backend != "auto":
        if observed is not None and observed != endpoint.backend:
            raise LLMBackendError("runtime_identity_conflict")
        return endpoint.backend
    return observed


def _normalize_strata_entries(
    entries: Any, *, check: Callable[[], Any] | None = None
) -> list[dict[str, Any]]:
    """Validate Strata's explicit alias rows and keep canonical declarations.

    Shared by read-only discovery and the Strata adapter's readiness check. This
    does not merge merely similar names, paths, metadata or independent models.
    """
    if not isinstance(entries, list):
        raise LLMBackendError("runtime_discovery_invalid_catalog")
    for entry in entries:
        if check is not None:
            check()
        if not isinstance(entry, dict):
            raise LLMBackendError("runtime_discovery_invalid_catalog")
        _observe(entry, set())
    canonical: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if check is not None:
            check()
        if "alias_of" not in entry:
            if entry["id"] in canonical:
                raise LLMBackendError("runtime_discovery_ambiguous_alias")
            canonical[entry["id"]] = entry
    filtered = []
    for entry in entries:
        if check is not None:
            check()
        if "alias_of" not in entry:
            filtered.append(entry)
            continue
        parent_id = entry.get("alias_of")
        parent = canonical.get(parent_id) if isinstance(parent_id, str) else None
        if (
            parent is None
            or entry.get("id") not in parent.get("aliases", [])
            or {
                key: value
                for key, value in entry.items()
                if key not in {"id", "alias_of"}
            }
            != {key: value for key, value in parent.items() if key != "id"}
        ):
            raise LLMBackendError("runtime_discovery_invalid_alias")
    return filtered


def _catalog(
    payload: Any,
    props: dict[str, Any],
    backend: RuntimeBackend | None,
    diagnostics: set[str],
    *,
    check: Callable[[], Any] | None = None,
) -> tuple[ModelObservation, ...]:
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise LLMBackendError("runtime_discovery_invalid_catalog")
    # Strata intentionally publishes one extra row per alias. Only its explicit
    # alias_of relation permits collapsing these duplicate declarations.
    if backend == "strata":
        entries = _normalize_strata_entries(entries, check=check)
    observations = []
    claimed: set[str] = set()
    for entry in entries:
        if check is not None:
            check()
        if not isinstance(entry, dict):
            raise LLMBackendError("runtime_discovery_invalid_catalog")
        observation = _observe(entry, diagnostics)
        if claimed.intersection(observation.aliases):
            raise LLMBackendError("runtime_discovery_ambiguous_alias")
        claimed.update(observation.aliases)
        # Explicit non-text output modalities cannot serve an LLM chat request.
        architecture = entry.get("architecture")
        modalities = (
            architecture.get("output_modalities")
            if isinstance(architecture, dict)
            else None
        )
        if isinstance(modalities, list) and modalities and "text" not in modalities:
            observation = replace(
                observation, capabilities=replace(observation.capabilities, chat=False)
            )
        observations.append(observation)
    if any(item.state == "unknown" for item in observations) and props:
        status = entries[0].get("status") if len(entries) == 1 else None
        eligible = status is None or (
            isinstance(status, dict) and status.get("value") is None
        )
        _root_proof(observations, props, diagnostics, eligible=eligible)
    return tuple(observations)


def _strata_model(
    health: dict[str, Any], props: dict[str, Any], models: tuple[ModelObservation, ...]
) -> str | None:
    """Accept one canonical LLM only when all independent markers agree."""
    canonical = health.get("model")
    selected = [item for item in models if item.id == canonical]
    if (
        health.get("loaded") is not True
        or not _valid_name(canonical)
        or len(selected) != 1
        or selected[0].state != "loaded"
        or selected[0].failed is True
        or props.get("model_alias") not in selected[0].aliases
        or props.get("is_sleeping") is not False
    ):
        return None
    observed = selected[0]
    contexts = []
    settings = props.get("default_generation_settings")
    if settings is not None and not isinstance(settings, dict):
        return None
    for value in (
        health.get("max_context"),
        settings.get("n_ctx") if settings else None,
    ):
        if value is not None:
            if type(value) is not int or value <= 0:
                return None
            contexts.append(value)
    if observed.loaded_context is not None:
        contexts.append(observed.loaded_context.value)
    if len(set(contexts)) > 1:
        return None
    return canonical


def _probe_endpoint(
    endpoint: RuntimeEndpoint, deadline: float, cancel: CancellationProbe
) -> RuntimeObservation:
    started = time.monotonic()
    requests = 0
    diagnostics: set[str] = set()
    backend: RuntimeBackend | None = None
    catalog = None
    available = False
    llm_model_id = None

    def remaining() -> float:
        _cancelled(cancel)
        value = deadline - time.monotonic()
        if value <= 0:
            raise LLMTimeoutError("runtime_discovery_timeout")
        return value

    try:
        with LlamaCppClient(
            endpoint.base_url,
            api_key=endpoint.resolved_api_key(),
            timeout=remaining(),
            verify_ssl=endpoint.verify_ssl,
            trust_env=False,
            proxy_url=endpoint.proxy_url,
            connection_policy=endpoint.http_connection_policy,
        ) as client:
            values = []
            for method in (client.health, client.props):
                call_timeout = remaining()
                requests += 1
                try:
                    value = method(
                        timeout=call_timeout, cancel=cancel, cancel_wait=True
                    )
                except ClientHTTPError as exc:
                    if exc.status_code not in {404, 405, 501}:
                        raise
                    value = {}
                if not isinstance(value, dict):
                    raise LLMBackendError("runtime_discovery_invalid_metadata")
                values.append(value)
            health, props = values
            backend = _backend(endpoint, health, props)
            if backend is None:
                diagnostics.add("runtime_backend_unsupported")
            available = health.get("status") == "ok" and backend is not None
            if health.get("status") != "ok":
                diagnostics.add("runtime_health_unverified")
            if backend == "strata" and health.get("loaded") is not True:
                available = False
                diagnostics.add("runtime_loaded_state_unverified")
            call_timeout = remaining()
            requests += 1
            payload = client.models(
                timeout=call_timeout, cancel=cancel, cancel_wait=True
            )
            observations = _catalog(
                payload, props, backend, diagnostics, check=remaining
            )
            if backend == "strata":
                llm_model_id = _strata_model(health, props, observations)
                if llm_model_id is None:
                    available = False
                    diagnostics.add("runtime_model_state_incoherent")
            remaining()
            catalog = ModelCatalog(
                observations,
                tuple(sorted(diagnostics)),
                requests,
                time.monotonic() - started,
            )
    except LLMCancelledError:
        raise
    except LLMTimeoutError:
        diagnostics.add("runtime_probe_timeout")
        available = False
    except ClientHTTPError as exc:
        diagnostics.add(f"runtime_probe_http_{exc.status_code}")
        available = False
    except LLMBackendError as exc:
        # Client diagnostics are sanitized, but expose a fixed code vocabulary
        # here instead of persisting an arbitrary server/client exception text.
        code = str(exc)
        code = {
            "llama_server_discovery_invalid_catalog": "runtime_discovery_invalid_catalog",
            "llama_server_discovery_ambiguous_identity": "runtime_discovery_ambiguous_identity",
        }.get(code, code)
        diagnostics.add(
            code
            if code
            in {
                "runtime_identity_conflict",
                "runtime_discovery_invalid_catalog",
                "runtime_discovery_ambiguous_identity",
                "runtime_discovery_invalid_alias",
                "runtime_discovery_ambiguous_alias",
                "runtime_discovery_invalid_metadata",
            }
            else "runtime_probe_failed"
        )
        available = False
    except (ValueError, OSError):
        diagnostics.add("runtime_probe_failed")
        available = False
    return RuntimeObservation(
        endpoint,
        backend,
        catalog,
        tuple(sorted(diagnostics)),
        requests,
        time.monotonic() - started,
        available,
        llm_model_id,
    )


def discover_runtimes(
    endpoints: Sequence[RuntimeEndpoint],
    *,
    timeout: float = 5.0,
    cancellation_probe: CancellationProbe | None = None,
) -> RuntimeCatalog:
    """Probe every configured endpoint concurrently under one finite deadline.

    Each endpoint performs at most three GETs: health, root props, catalog.
    Root props are reused for standalone load/context proof. Unsupported and
    unavailable endpoints remain visible as typed observations, while successful
    endpoints stay usable. There is no network scanning, fixed endpoint/model
    count limit, cache, administrative call or inference request. Workers and
    owned HTTP clients are drained on completion, cancellation and interruption.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("runtime discovery timeout must be finite and positive")
    try:
        timeout = float(timeout)
    except (ValueError, OverflowError):
        raise ValueError(
            "runtime discovery timeout must be finite and positive"
        ) from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("runtime discovery timeout must be finite and positive")
    if cancellation_probe is not None and not callable(cancellation_probe):
        raise ValueError("runtime discovery cancellation probe must be callable")
    if isinstance(endpoints, (str, bytes)) or not isinstance(endpoints, Sequence):
        raise ValueError("runtime endpoints must be a sequence")
    endpoints = tuple(endpoints)
    if any(not isinstance(item, RuntimeEndpoint) for item in endpoints):
        raise ValueError("runtime endpoints must contain RuntimeEndpoint values")
    if len({item.name for item in endpoints}) != len(endpoints):
        raise ValueError("runtime endpoint names must be unique")
    _cancelled(cancellation_probe)
    started = time.monotonic()
    if not endpoints:
        return RuntimeCatalog((), time.monotonic() - started)
    deadline = started + timeout
    stop = Event()

    def cancelled() -> bool:
        return stop.is_set() or bool(cancellation_probe and cancellation_probe())

    executor = ThreadPoolExecutor(
        max_workers=len(endpoints), thread_name_prefix="runtime-probe"
    )
    futures = {}
    results: list[RuntimeObservation | None] = [None] * len(endpoints)
    try:
        for index, endpoint in enumerate(endpoints):
            futures[executor.submit(_probe_endpoint, endpoint, deadline, cancelled)] = (
                index
            )
        for future in as_completed(futures):
            _cancelled(cancellation_probe)
            results[futures[future]] = future.result()
        _cancelled(cancellation_probe)
    except BaseException:
        stop.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return RuntimeCatalog(
        tuple(item for item in results if item is not None), time.monotonic() - started
    )
