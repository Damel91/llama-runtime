"""Versioned, secret-safe configuration for the optional local relay.

This module uses the SDK's configuration types but imports no web framework,
starts no processes and performs no network discovery. Persistence is an
explicit store operation at the caller's chosen path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import json
import os
from pathlib import Path
import re
import tempfile
from threading import RLock
from types import MappingProxyType
from typing import Any

from runtime_llama.automatic_embedding import AutomaticEmbeddingConfig
from runtime_llama.configuration import AdapterProfile
from runtime_llama.runtime_discovery import RuntimeEndpoint
from runtime_llama.tailnet import TailnetConfig


CONFIG_SCHEMA_VERSION = 1
_VALIDATION_URL = "http://127.0.0.1:8080"
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "listener",
        "access",
        "runtime_endpoints",
        "llm_defaults",
        "embedding_defaults",
        "model_overrides",
        "tailnet",
    }
)
LLM_SETTING_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "output_mode",
        "stream_idle_timeout_sec",
        "extra_body",
        "json_schema",
        "chat_template_kwargs",
        "thinking_diagnostics",
        "reasoning",
        "context_ceiling",
        "num_ctx",
        "num_predict",
        "model_context_length",
        "runtime_discovery_timeout",
        "runaway_min_chars",
        "runaway_window_chars",
        "runaway_block_chars",
        "runaway_repeat_threshold",
        "runaway_ngram_chars",
        "runtime_char_token_ratio_seed",
    }
)
_EMBEDDING_CONNECTION_FIELDS = frozenset(
    {
        "base_url",
        "model",
        "category",
        "cache_dir",
        "calibration_timeout",
        "calibration_workers",
        "api_key",
        "api_key_env",
        "verify_ssl",
        "proxy_url",
        "trust_env",
        "http_connection_policy",
    }
)
EMBEDDING_SETTING_FIELDS = frozenset(
    {item.name for item in fields(AutomaticEmbeddingConfig)}
    - _EMBEDDING_CONNECTION_FIELDS
    | {"input_type"}
)
_ENDPOINT_FIELDS = frozenset(
    {item.name for item in fields(RuntimeEndpoint)} | {"use_tailnet"}
)
_TAILNET_FIELDS = frozenset(item.name for item in fields(TailnetConfig))


class RelayConfigError(ValueError):
    """Invalid configuration, with a content-free diagnostic code."""


class RelayConfigConflict(RelayConfigError):
    """The writer did not observe the current configuration revision."""

    def __init__(self, expected_revision: int, actual_revision: int) -> None:
        super().__init__("relay_config_revision_conflict")
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision


def default_config() -> dict[str, Any]:
    """Return independent defaults without changing SDK inference defaults."""
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "listener": {
            "host": "127.0.0.1",
            "port": 8088,
            "allowed_hosts": ["localhost", "127.0.0.1", "::1"],
        },
        "access": {"api_key": None, "api_key_env": None},
        "runtime_endpoints": [],
        "llm_defaults": {},
        "embedding_defaults": {
            "input_type": "document",
            "normalize": True,
            "timeout": 120.0,
        },
        "model_overrides": {},
        "tailnet": None,
    }


def _json_copy(value: Any) -> Any:
    def check(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise RelayConfigError("relay_config_object_keys_must_be_strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is not None and type(item) not in (str, bool, int, float):
            raise RelayConfigError("relay_config_must_be_json")

    check(value)
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError, OverflowError):
        raise RelayConfigError("relay_config_must_be_finite_json") from None


def _object(value: Any, code: str) -> dict:
    if not isinstance(value, dict):
        raise RelayConfigError(code)
    return value


def _text(value: Any, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise RelayConfigError(code)
    return value


def _auth(value: dict) -> None:
    if set(value) - {"api_key", "api_key_env"}:
        raise RelayConfigError("relay_access_fields_invalid")
    try:
        RuntimeEndpoint("validation", _VALIDATION_URL, **value)
    except (ValueError, TypeError):
        raise RelayConfigError("relay_access_authentication_invalid") from None


def _llm_settings(value: Any) -> dict:
    value = _object(value, "relay_llm_settings_must_be_object")
    if set(value) - LLM_SETTING_FIELDS:
        raise RelayConfigError("relay_llm_settings_fields_invalid")
    try:
        AdapterProfile.from_config("relay-validation", {"model": "validation", **value})
    except (ValueError, TypeError, KeyError, OverflowError):
        raise RelayConfigError("relay_llm_settings_invalid") from None
    return value


def _embedding_settings(value: Any) -> dict:
    value = _object(value, "relay_embedding_settings_must_be_object")
    if set(value) - EMBEDDING_SETTING_FIELDS:
        raise RelayConfigError("relay_embedding_settings_fields_invalid")
    task = value.get("input_type", "document")
    if not isinstance(task, str) or task not in {"query", "document", "raw"}:
        raise RelayConfigError("relay_embedding_input_type_invalid")
    try:
        AutomaticEmbeddingConfig(
            base_url=_VALIDATION_URL,
            **{key: item for key, item in value.items() if key != "input_type"},
        )
    except (ValueError, TypeError, KeyError, OverflowError):
        raise RelayConfigError("relay_embedding_settings_invalid") from None
    return value


def _preserve_secrets(config: dict, previous: Mapping | None) -> dict:
    if previous is None:
        return config
    previous = _thaw(previous)
    old_endpoints = {
        endpoint.get("name", endpoint.get("base_url")): endpoint
        for endpoint in previous.get("runtime_endpoints", [])
        if isinstance(endpoint, dict)
    }
    if isinstance(config.get("runtime_endpoints"), list):
        for endpoint in config["runtime_endpoints"]:
            if not isinstance(endpoint, dict):
                continue
            old = old_endpoints.get(endpoint.get("name", endpoint.get("base_url")), {})
            for key in ("api_key", "proxy_url"):
                if key not in endpoint and key in old:
                    endpoint[key] = old[key]
    for section, keys in (("access", ("api_key",)), ("tailnet", ("auth_key",))):
        current = config.get(section)
        old = previous.get(section)
        if isinstance(current, dict) and isinstance(old, dict):
            for key in keys:
                if key not in current and key in old:
                    current[key] = old[key]
    return config


def validate_config(value: Mapping, *, previous: Mapping | None = None) -> dict:
    """Validate portable JSON through existing SDK types; perform no probes."""
    if not isinstance(value, Mapping):
        raise RelayConfigError("relay_config_must_be_object")
    raw = _json_copy(dict(value))
    if set(raw) - _CONFIG_FIELDS:
        raise RelayConfigError("relay_config_fields_invalid")
    raw = _preserve_secrets(raw, previous)
    config = {**default_config(), **raw}
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise RelayConfigError("relay_config_schema_version_unsupported")
    listener = _object(config["listener"], "relay_listener_must_be_object")
    if set(listener) - {"host", "port", "allowed_hosts"}:
        raise RelayConfigError("relay_listener_fields_invalid")
    listener = {**default_config()["listener"], **listener}
    host = _text(listener["host"], "relay_listener_host_invalid")
    if any(char.isspace() for char in host) or any(char in host for char in "/\\?#@"):
        raise RelayConfigError("relay_listener_host_invalid")
    if type(listener["port"]) is not int or not 1 <= listener["port"] <= 65535:
        raise RelayConfigError("relay_listener_port_invalid")
    hosts = listener["allowed_hosts"]
    if not isinstance(hosts, list):
        raise RelayConfigError("relay_listener_allowed_hosts_invalid")
    for allowed in hosts:
        _text(allowed, "relay_listener_allowed_hosts_invalid")
        if (
            allowed == "*"
            or any(char.isspace() for char in allowed)
            or any(char in allowed for char in "/\\?#@")
        ):
            raise RelayConfigError("relay_listener_allowed_hosts_invalid")
    listener["allowed_hosts"] = list(dict.fromkeys(hosts))
    config["listener"] = listener
    access = _object(config["access"], "relay_access_must_be_object")
    _auth(access)
    endpoints = config["runtime_endpoints"]
    if not isinstance(endpoints, list):
        raise RelayConfigError("relay_runtime_endpoints_must_be_array")
    names = set()
    for position, endpoint in enumerate(endpoints):
        endpoint = _object(endpoint, "relay_runtime_endpoint_must_be_object")
        if set(endpoint) - _ENDPOINT_FIELDS:
            raise RelayConfigError("relay_runtime_endpoint_fields_invalid")
        use_tailnet = endpoint.get("use_tailnet", False)
        if type(use_tailnet) is not bool:
            raise RelayConfigError("relay_runtime_endpoint_use_tailnet_invalid")
        try:
            observed = RuntimeEndpoint.from_value(
                {key: item for key, item in endpoint.items() if key != "use_tailnet"}
            )
        except (ValueError, TypeError, KeyError, OverflowError):
            raise RelayConfigError("relay_runtime_endpoint_invalid") from None
        if observed.name in names:
            raise RelayConfigError("relay_runtime_endpoint_name_duplicate")
        names.add(observed.name)
        endpoints[position] = {
            **endpoint,
            "name": observed.name,
            "base_url": observed.base_url,
            "backend": observed.backend,
            "verify_ssl": observed.verify_ssl,
            "http_connection_policy": observed.http_connection_policy,
            "use_tailnet": use_tailnet,
        }
    _llm_settings(config["llm_defaults"])
    _embedding_settings(config["embedding_defaults"])
    overrides = _object(
        config["model_overrides"], "relay_model_overrides_must_be_object"
    )
    for model, override in overrides.items():
        _text(model, "relay_model_override_name_invalid")
        override = _object(override, "relay_model_override_must_be_object")
        if set(override) - {"llm", "embedding"}:
            raise RelayConfigError("relay_model_override_fields_invalid")
        if "llm" in override:
            _object(override["llm"], "relay_llm_settings_must_be_object")
            _llm_settings({**config["llm_defaults"], **override["llm"]})
        if "embedding" in override:
            _object(override["embedding"], "relay_embedding_settings_must_be_object")
            _embedding_settings(
                {**config["embedding_defaults"], **override["embedding"]}
            )
    tailnet = config["tailnet"]
    if tailnet is not None:
        tailnet = _object(tailnet, "relay_tailnet_must_be_object")
        if set(tailnet) - _TAILNET_FIELDS:
            raise RelayConfigError("relay_tailnet_fields_invalid")
        try:
            TailnetConfig(**tailnet)
        except (ValueError, TypeError, KeyError, OverflowError):
            raise RelayConfigError("relay_tailnet_config_invalid") from None
    return config


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _redact(config: dict) -> tuple[dict, list[str]]:
    secrets = []

    def redact(value: dict, keys: tuple[str, ...], pointer: str) -> None:
        for key in keys:
            if key in value:
                if value[key] is not None:
                    secrets.append(pointer + "/" + key)
                del value[key]

    for position, endpoint in enumerate(config["runtime_endpoints"]):
        redact(endpoint, ("api_key", "proxy_url"), f"/runtime_endpoints/{position}")
    redact(config["access"], ("api_key",), "/access")
    if config["tailnet"] is not None:
        redact(config["tailnet"], ("auth_key",), "/tailnet")
    return config, secrets


@dataclass(frozen=True, slots=True, weakref_slot=True)
class RelayConfigSnapshot:
    revision: int
    _data: Mapping = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.revision) is not int or self.revision < 0:
            raise RelayConfigError("relay_config_revision_invalid")
        object.__setattr__(self, "_data", _freeze(_thaw(self._data)))

    @property
    def config(self) -> dict:
        """Return an independent private copy, including configured secrets."""
        return _thaw(self._data)

    def public(self) -> dict:
        config, secret_fields = _redact(self.config)
        return {
            "revision": self.revision,
            "config": config,
            "secret_fields": secret_fields,
        }

    def settings(self, model: str, kind: str) -> dict:
        if kind not in {"llm", "embedding"}:
            raise ValueError("relay_settings_kind_invalid")
        config = self.config
        return {
            **config[kind + "_defaults"],
            **config["model_overrides"].get(model, {}).get(kind, {}),
        }


class RelayConfigStore:
    """Thread-safe configuration revisions with optional atomic persistence."""

    def __init__(
        self,
        path: str | os.PathLike | None = None,
        initial_config: Mapping | None = None,
    ) -> None:
        self.path = Path(path).expanduser() if path is not None else None
        self._lock = RLock()
        revision = 0
        config = initial_config if initial_config is not None else default_config()
        if self.path is not None and self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or set(payload) != {
                    "revision",
                    "config",
                }:
                    raise ValueError
                revision = payload["revision"]
                if type(revision) is not int or revision < 0:
                    raise ValueError
                config = payload["config"]
            except (ValueError, TypeError, KeyError):
                raise RelayConfigError("relay_persisted_config_invalid") from None
        self._snapshot = RelayConfigSnapshot(revision, _freeze(validate_config(config)))

    def snapshot(self) -> RelayConfigSnapshot:
        with self._lock:
            return self._snapshot

    def replace(self, expected_revision: int, config: Mapping) -> RelayConfigSnapshot:
        if type(expected_revision) is not int or expected_revision < 0:
            raise RelayConfigError("relay_expected_revision_invalid")
        with self._lock:
            current = self._snapshot
            if expected_revision != current.revision:
                raise RelayConfigConflict(expected_revision, current.revision)
            normalized = validate_config(config, previous=current.config)
            replacement = RelayConfigSnapshot(current.revision + 1, _freeze(normalized))
            if self.path is not None:
                self._persist(replacement)
            self._snapshot = replacement
            return replacement

    def _persist(self, snapshot: RelayConfigSnapshot) -> None:
        assert self.path is not None
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".relay-config-", suffix=".json", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                if hasattr(os, "fchmod"):
                    os.fchmod(stream.fileno(), 0o600)
                json.dump(
                    {"revision": snapshot.revision, "config": snapshot.config},
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def environment_key_name(value: str) -> bool:
    """Expose the SDK-compatible environment-name shape to admin consumers."""
    return (
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is not None
    )
