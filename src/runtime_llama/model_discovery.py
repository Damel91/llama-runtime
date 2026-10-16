"""Bounded, read-only observations of an explicitly selected llama-server.

Catalog membership is not proof that a model is loaded. No inference, model
selection, loading or administrative request is performed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
import re
import time
from typing import Any, Literal
from urllib.parse import unquote_to_bytes

from runtime_llama.client import (
    CancellationProbe,
    ClientHTTPError,
    LlamaCppClient,
    _cancelled,
)
from runtime_llama.errors import LLMBackendError, LLMTimeoutError
from runtime_llama.replica_discovery import (
    _CONTENT,
    _EMBEDDING,
    _NAME,
    catalog_aliases,
    is_loaded,
)

ModelState = Literal[
    "loaded", "loading", "unloaded", "sleeping", "downloading", "downloaded", "unknown"
]
ObservationSource = Literal[
    "catalog.meta.n_ctx_train",
    "catalog.context_length",
    "catalog.meta.n_embd",
    "catalog.meta.n_ctx",
    "root_props.default_generation_settings.n_ctx",
    "root_props.total_slots",
]
_STATES = frozenset(
    {"loaded", "loading", "unloaded", "sleeping", "downloading", "downloaded"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_BAD_ESCAPE = re.compile(r"%(?![0-9a-fA-F]{2})")


@dataclass(frozen=True)
class ObservedInt:
    """A positive integer and its exact supported observation location."""

    value: int
    source: ObservationSource


@dataclass(frozen=True)
class ModelCapabilities:
    """Explicit catalog booleans only; ``None`` means unobserved support."""

    chat: bool | None = None
    embeddings: bool | None = None


@dataclass(frozen=True)
class ModelIdentity:
    """Independent server assertions, not locally verified weight equivalence.

    Missing hashes remain missing. Names, aliases and architecture dimensions
    never establish that two entries contain the same weights.
    """

    content_sha256: str | None = None
    embedding_sha256: str | None = None
    model_name: str | None = None


@dataclass(frozen=True)
class ModelObservation:
    id: str
    aliases: tuple[str, ...]
    state: ModelState
    failed: bool | None
    identity: ModelIdentity
    capabilities: ModelCapabilities
    training_context: ObservedInt | None = None
    model_embedding_width: ObservedInt | None = None
    loaded_context: ObservedInt | None = None
    slots: ObservedInt | None = None


@dataclass(frozen=True)
class ModelCatalog:
    """Immutable observation, not a reservation or a future readiness promise.

    ``elapsed_seconds`` includes HTTP, parsing and cleanup, not GPU execution.
    ``models`` is the available catalog; only ``loaded_models`` proves a loaded
    observation. Diagnostic codes contain no URLs, credentials or raw metadata.
    """

    models: tuple[ModelObservation, ...]
    diagnostics: tuple[str, ...]
    http_requests: int
    elapsed_seconds: float

    @property
    def loaded_models(self) -> tuple[ModelObservation, ...]:
        return tuple(model for model in self.models if model.state == "loaded")


def _invalid() -> None:
    raise LLMBackendError("llama_server_discovery_invalid_catalog") from None


def _text(value: Any, *, limit: int = 1024) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > limit
        or any(
            ord(char) < 32 or 127 <= ord(char) <= 159 or 0xD800 <= ord(char) <= 0xDFFF
            for char in value
        )
    ):
        _invalid()
    return value


def _integer(value: Any, source: ObservationSource) -> ObservedInt | None:
    if value is None:
        return None
    if type(value) is not int or not 1 <= value <= 2**31 - 1:
        _invalid()
    return ObservedInt(value, source)


def _identity(entry: dict[str, Any]) -> ModelIdentity:
    tags = entry.get("tags", [])
    if not isinstance(tags, list) or len(tags) > 128:
        _invalid()
    values: dict[str, str] = {}
    for tag in tags:
        _text(tag, limit=4096)
        for prefix in (_CONTENT, _EMBEDDING, _NAME):
            if tag.startswith(prefix):
                value = tag[len(prefix) :]
                if prefix in values and values[prefix] != value:
                    raise LLMBackendError("llama_server_discovery_ambiguous_identity")
                values[prefix] = value
    content, embedding, name = (
        values.get(_CONTENT),
        values.get(_EMBEDDING),
        values.get(_NAME),
    )
    for digest in (content, embedding):
        if digest is not None and not _SHA256.fullmatch(digest):
            _invalid()
    if name is not None:
        if _BAD_ESCAPE.search(name):
            _invalid()
        try:
            name = unquote_to_bytes(name).decode("utf-8", errors="strict")
        except UnicodeError:
            _invalid()
        _text(name)
        if "/" in name or "\\" in name:
            _invalid()
    return ModelIdentity(content, embedding, name)


def _observe(entry: dict[str, Any], diagnostics: set[str]) -> ModelObservation:
    model = _text(entry.get("id"))
    aliases = entry.get("aliases", [])
    if not isinstance(aliases, list) or len(aliases) > 64:
        _invalid()
    for alias in aliases:
        _text(alias)
    accepted_aliases = tuple(sorted(catalog_aliases(entry)))
    status = entry.get("status")
    if status is not None and not isinstance(status, (str, dict)):
        _invalid()
    value = status.get("value") if isinstance(status, dict) else status
    failed = status.get("failed") if isinstance(status, dict) else None
    if failed is not None and not isinstance(failed, bool):
        _invalid()
    if value is not None:
        _text(value, limit=128)
    state: ModelState = value if value in _STATES else "unknown"
    if state == "loaded" and failed is True:
        _invalid()
    if value is not None and state == "unknown":
        diagnostics.add("unknown_model_state")
    caps = entry.get("capabilities")
    if caps is None:
        capabilities = ModelCapabilities()
    elif not isinstance(caps, dict):
        diagnostics.add("unsupported_capabilities_shape")
        capabilities = ModelCapabilities()
    else:
        selected = {key: caps.get(key) for key in ("chat", "embeddings")}
        if any(
            value is not None and not isinstance(value, bool)
            for value in selected.values()
        ):
            _invalid()
        capabilities = ModelCapabilities(**selected)
    meta = entry.get("meta")
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        diagnostics.add("unsupported_metadata_shape")
        meta = {}
    training = _integer(meta.get("n_ctx_train"), "catalog.meta.n_ctx_train")
    if training is None:
        training = _integer(entry.get("context_length"), "catalog.context_length")
    width = _integer(meta.get("n_embd"), "catalog.meta.n_embd")
    loaded = (
        _integer(meta.get("n_ctx"), "catalog.meta.n_ctx") if is_loaded(entry) else None
    )
    return ModelObservation(
        model,
        accepted_aliases,
        state,
        failed,
        _identity(entry),
        capabilities,
        training,
        width,
        loaded,
    )


def _root_proof(
    observations: list[ModelObservation], props: Any, diagnostics: set[str]
) -> None:
    if not isinstance(props, dict):
        _invalid()
    if len(observations) != 1:
        diagnostics.add("root_props_load_state_unverified")
        return
    observed = observations[0]
    alias = props.get("model_alias")
    sleeping = props.get("is_sleeping")
    if (
        props.get("role") != "model"
        or not isinstance(alias, str)
        or alias not in observed.aliases
        or not isinstance(sleeping, bool)
    ):
        diagnostics.add("root_props_load_state_unverified")
        return
    settings = props.get("default_generation_settings")
    if not sleeping and observed.failed is True:
        _invalid()
    if settings is not None and not isinstance(settings, dict):
        _invalid()
    context = (
        _integer(
            settings.get("n_ctx"),
            "root_props.default_generation_settings.n_ctx",
        )
        if isinstance(settings, dict) and not sleeping
        else None
    )
    slots = _integer(props.get("total_slots"), "root_props.total_slots")
    observations[0] = replace(
        observed,
        state="sleeping" if sleeping else "loaded",
        loaded_context=context,
        slots=slots,
    )


def discover_models(
    base_url: str,
    *,
    api_key: str | None = None,
    api_key_env: str | None = None,
    timeout: float = 5.0,
    verify_ssl: bool = True,
    cancellation_probe: CancellationProbe | None = None,
    max_response_bytes: int = 1048576,
    max_models: int = 256,
) -> ModelCatalog:
    """Observe a remote catalog through at most two GETs and one deadline.

    `/v1/models` is always queried. If a load state is unknown, root `/props`
    may prove a single standalone model; `/props?model=...` is never requested.
    Unsupported root props (404/405/501) leave unknown states with a fixed
    diagnostic. All other HTTP, timeout, cancellation and malformed-response
    errors propagate as sanitized typed errors. The owned client drains on
    every outcome, ignores ambient proxy/netrc credentials, verifies TLS by
    default and never follows redirects. Byte limits apply per response.

    Capabilities are accepted only from explicit `capabilities.chat` and
    `capabilities.embeddings` catalog booleans. They are not inference probes.
    Training context and model embedding width are not inference limits or
    proof of an embedding output dimension. An observed load state can change
    immediately after this call; no capacity or model lifecycle is managed.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("discovery timeout must be finite and positive")
    try:
        timeout = float(timeout)
    except (ValueError, OverflowError):
        raise ValueError("discovery timeout must be finite and positive") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("discovery timeout must be finite and positive")
    if not isinstance(verify_ssl, bool):
        raise ValueError("discovery verify_ssl must be a boolean")
    if cancellation_probe is not None and not callable(cancellation_probe):
        raise ValueError("discovery cancellation_probe must be callable")
    for value, upper in ((max_response_bytes, 16 * 1024 * 1024), (max_models, 4096)):
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError("discovery limit is outside its supported bounds")
    if api_key is not None and api_key_env is not None:
        raise ValueError("discovery accepts one explicit authentication source")
    if api_key_env is not None:
        if not isinstance(api_key_env, str) or not _ENV_NAME.fullmatch(api_key_env):
            raise ValueError("discovery authentication environment name is invalid")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                "discovery authentication environment value is unavailable"
            )
    if api_key is not None and (
        not isinstance(api_key, str)
        or not 1 <= len(api_key) <= 8192
        or any(ord(char) < 32 or ord(char) == 127 for char in api_key)
    ):
        raise ValueError("discovery authentication value is invalid")
    started = time.monotonic()
    deadline = started + timeout

    def remaining() -> float:
        _cancelled(cancellation_probe)
        value = deadline - time.monotonic()
        if value <= 0:
            raise LLMTimeoutError("llama_server_discovery_timeout")
        return value

    diagnostics: set[str] = set()
    http_requests = 0
    with LlamaCppClient(
        base_url,
        api_key=api_key,
        timeout=timeout,
        verify_ssl=verify_ssl,
        trust_env=False,
    ) as client:
        http_requests += 1
        catalog = client.models(
            timeout=remaining(),
            cancel=cancellation_probe,
            cancel_wait=True,
            max_response_bytes=max_response_bytes,
        )
        entries = catalog.get("data") if isinstance(catalog, dict) else None
        if not isinstance(entries, list):
            _invalid()
        if len(entries) > max_models:
            raise LLMBackendError("llama_server_discovery_model_limit_exceeded")
        observations: list[ModelObservation] = []
        claimed: set[str] = set()
        for entry in entries:
            remaining()
            if not isinstance(entry, dict):
                _invalid()
            observation = _observe(entry, diagnostics)
            if claimed.intersection(observation.aliases):
                raise LLMBackendError("llama_server_discovery_ambiguous_alias")
            claimed.update(observation.aliases)
            observations.append(observation)
        if any(model.state == "unknown" for model in observations):
            try:
                http_requests += 1
                props = client.props(
                    timeout=remaining(),
                    cancel=cancellation_probe,
                    cancel_wait=True,
                    max_response_bytes=max_response_bytes,
                )
            except ClientHTTPError as exc:
                if exc.status_code not in {404, 405, 501}:
                    raise
                diagnostics.add("root_props_unavailable")
            else:
                _root_proof(observations, props, diagnostics)
        remaining()
    return ModelCatalog(
        tuple(observations),
        tuple(sorted(diagnostics)),
        http_requests,
        time.monotonic() - started,
    )
