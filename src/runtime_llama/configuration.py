"""External server profiles, adapted from lmstudio-agent-runtime@9300f295.

These profiles describe a server chosen by the caller. They never provision,
compile, start or replace that server. On-demand loading remains server policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any
from urllib.parse import urlsplit

from runtime_llama.diagnostics import ThinkingDiagnosticsConfig
from runtime_llama._proxy import normalize_proxy_url


DEFAULT_BASE_URL = "http://127.0.0.1:8080"
REASONING_POLICIES = {"auto", "off", "on", "low", "medium", "high"}


def _positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(
    value: Any, name: str, *, minimum: float = 0.0, positive: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric")
    try:
        parsed = float(value)
    except (ValueError, OverflowError):
        raise ValueError(f"{name} is out of range") from None
    if (
        not math.isfinite(parsed)
        or parsed < minimum
        or (positive and parsed == minimum)
    ):
        raise ValueError(f"{name} is out of range")
    return parsed


def normalize_rest_root(value: str) -> str:
    raw = str(value or DEFAULT_BASE_URL).strip().rstrip("/")
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base_url must identify an HTTP(S) server")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url cannot contain credentials, query or fragment")
    try:
        parsed.port
    except ValueError:
        raise ValueError("base_url port is invalid") from None
    if raw.endswith("/v1"):
        raw = raw[:-3]
    return raw


def normalize_reasoning(value: Any) -> str:
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "on" if value else "off"
    normalized = str(value).strip().lower()
    if normalized not in REASONING_POLICIES:
        raise ValueError("reasoning policy is unsupported")
    return normalized


@dataclass(frozen=True)
class LlamaServerConfig:
    model: str
    base_url: str = DEFAULT_BASE_URL
    context_length: int | None = None
    verify_ssl: bool = True
    runtime_identity: str | None = None
    proxy_url: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        model = str(self.model or "").strip()
        if not model:
            raise ValueError("model is required")
        _positive_int(self.context_length, "context_length")
        if not isinstance(self.verify_ssl, bool):
            raise ValueError("verify_ssl must be boolean")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", normalize_rest_root(self.base_url))
        object.__setattr__(self, "proxy_url", normalize_proxy_url(self.proxy_url))

    @property
    def physical_identity(self) -> str:
        description = {
            "base_url": self.base_url,
            "model": self.model,
            "context_length": self.context_length,
            "runtime_identity": self.runtime_identity,
        }
        return hashlib.sha256(
            json.dumps(description, sort_keys=True).encode()
        ).hexdigest()


@dataclass(frozen=True)
class LlamaGenerationConfig:
    temperature: float = 0.2
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    output_mode: str = "text"
    stream_idle_timeout_sec: float | None = 30.0
    api_key: str | None = field(default=None, repr=False, compare=False)
    api_key_env: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    json_schema: dict[str, Any] | None = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    thinking_diagnostics: ThinkingDiagnosticsConfig = field(
        default_factory=ThinkingDiagnosticsConfig
    )

    def __post_init__(self) -> None:
        _finite(self.temperature, "temperature")
        if self.top_p is not None and not 0 < _finite(self.top_p, "top_p") <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.min_p is not None and not 0 <= _finite(self.min_p, "min_p") <= 1:
            raise ValueError("min_p must be in [0, 1]")
        if self.top_k is not None and (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k < 0
        ):
            raise ValueError("top_k must be a nonnegative integer")
        if self.repeat_penalty is not None:
            _finite(self.repeat_penalty, "repeat_penalty", positive=True)
        if self.stream_idle_timeout_sec is not None:
            _finite(
                self.stream_idle_timeout_sec, "stream_idle_timeout_sec", positive=True
            )
        if self.output_mode not in {"text", "json"}:
            raise ValueError("output_mode must be text or json")
        if self.json_schema is not None and not isinstance(self.json_schema, dict):
            raise ValueError("json_schema must be a JSON object")
        for name in ("extra_body", "chat_template_kwargs"):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a mapping")
            object.__setattr__(self, name, deepcopy(value))
        object.__setattr__(self, "json_schema", deepcopy(self.json_schema))
        object.__setattr__(
            self,
            "thinking_diagnostics",
            ThinkingDiagnosticsConfig.from_value(self.thinking_diagnostics),
        )


@dataclass(frozen=True)
class LlamaRoleConfig:
    role_id: str
    reasoning: str = "auto"
    context_ceiling: int | None = None
    visible_output_tokens: int | None = None
    runaway_min_chars: int = 512
    runaway_window_chars: int = 4096
    runaway_block_chars: int = 128
    runaway_repeat_threshold: int = 4
    runaway_ngram_chars: int = 24
    runtime_char_token_ratio_seed: float = 0.33

    def __post_init__(self) -> None:
        if not str(self.role_id or "").strip():
            raise ValueError("role_id is required")
        object.__setattr__(self, "reasoning", normalize_reasoning(self.reasoning))
        for name in (
            "context_ceiling",
            "visible_output_tokens",
            "runaway_min_chars",
            "runaway_window_chars",
            "runaway_block_chars",
            "runaway_repeat_threshold",
            "runaway_ngram_chars",
        ):
            _positive_int(getattr(self, name), name)
        _finite(
            self.runtime_char_token_ratio_seed,
            "runtime_char_token_ratio_seed",
            positive=True,
        )


@dataclass(frozen=True)
class EmbeddingPrefixes:
    """Literal task prefixes for one model and its equivalent replicas."""

    query: str = ""
    document: str = ""

    def __post_init__(self) -> None:
        for name in ("query", "document"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"embedding prefix {name} must be a string")

    @classmethod
    def from_value(cls, value: Any) -> EmbeddingPrefixes:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or set(value) - {"query", "document"}:
            raise ValueError("embedding_prefixes must contain only query/document")
        return cls(**dict(value))


@dataclass(frozen=True)
class EmbeddingReplicaConfig:
    """An externally provisioned alias; no process or GPU lifecycle settings."""

    model: str
    max_in_flight: int = 1
    max_input_tokens: int | None = None
    batch_token_limit: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("replica model must be a nonempty string")
        object.__setattr__(self, "model", self.model.strip())
        if self.max_in_flight is None:
            raise ValueError("max_in_flight must be a positive integer")
        for name in ("max_in_flight", "max_input_tokens", "batch_token_limit"):
            _positive_int(getattr(self, name), name)

    @classmethod
    def from_value(cls, value: Any) -> EmbeddingReplicaConfig:
        if isinstance(value, cls):
            return value
        if (
            not isinstance(value, dict)
            or set(value)
            - {"model", "max_in_flight", "max_input_tokens", "batch_token_limit"}
            or "model" not in value
        ):
            raise ValueError(
                "embedding replica must declare model and supported limits"
            )
        return cls(**value)


@dataclass(frozen=True)
class AdapterProfile:
    adapter_type: str
    model_config: LlamaServerConfig
    role: LlamaRoleConfig
    generation: LlamaGenerationConfig = field(default_factory=LlamaGenerationConfig)
    batch_size: int | None = None
    batch_token_limit: int | None = None
    token_count_parallelism: int | None = None
    normalize: bool | None = None
    dimension: int | None = None
    max_input_tokens: int | None = None
    embedding_request_parallelism: int | None = None
    embedding_queue_size: int = 16
    embedding_replicas: tuple[EmbeddingReplicaConfig, ...] = ()
    embedding_prefixes: EmbeddingPrefixes = field(default_factory=EmbeddingPrefixes)
    embedding_replica_discovery: bool = False

    def __post_init__(self) -> None:
        if self.adapter_type not in {"llm", "embedding"}:
            raise ValueError("adapter_type must be llm or embedding")
        if (
            self.adapter_type == "embedding"
            and self.generation.stream_idle_timeout_sec is None
        ):
            raise ValueError("embedding timeout must remain finite and positive")
        for name in (
            "batch_size",
            "batch_token_limit",
            "token_count_parallelism",
            "dimension",
            "max_input_tokens",
        ):
            _positive_int(getattr(self, name), name)
        if self.normalize is not None and not isinstance(self.normalize, bool):
            raise ValueError("normalize must be boolean")
        if not isinstance(self.embedding_replica_discovery, bool):
            raise ValueError("embedding_replica_discovery must be boolean")
        if self.adapter_type != "embedding" and self.embedding_replica_discovery:
            raise ValueError(
                "embedding_replica_discovery requires an embedding profile"
            )
        if self.embedding_request_parallelism is None:
            object.__setattr__(
                self,
                "embedding_request_parallelism",
                4 if self.embedding_replica_discovery else 1,
            )
        _positive_int(
            self.embedding_request_parallelism, "embedding_request_parallelism"
        )
        if (
            isinstance(self.embedding_queue_size, bool)
            or not isinstance(self.embedding_queue_size, int)
            or self.embedding_queue_size < 0
        ):
            raise ValueError("embedding_queue_size must be a nonnegative integer")
        if not isinstance(self.embedding_replicas, (tuple, list)):
            raise ValueError("embedding_replicas must be a sequence")
        replicas = tuple(
            EmbeddingReplicaConfig.from_value(r) for r in self.embedding_replicas
        )
        if len({r.model for r in replicas}) != len(replicas):
            raise ValueError("embedding replica models must be unique")
        if self.adapter_type != "embedding" and replicas:
            raise ValueError("embedding_replicas requires an embedding profile")
        if self.embedding_replica_discovery and replicas:
            raise ValueError(
                "automatic and explicit embedding replicas cannot be combined"
            )
        object.__setattr__(self, "embedding_replicas", replicas)
        prefixes = EmbeddingPrefixes.from_value(self.embedding_prefixes)
        if self.adapter_type != "embedding" and (prefixes.query or prefixes.document):
            raise ValueError("embedding_prefixes requires an embedding profile")
        object.__setattr__(self, "embedding_prefixes", prefixes)

    @classmethod
    def from_config(
        cls,
        role_id: str,
        role_cfg: dict[str, Any],
        defaults: dict[str, Any] | None = None,
    ) -> AdapterProfile:
        merged = {**(defaults or {}), **(role_cfg or {})}
        if merged.get("backend", "llamacpp") not in {
            "llamacpp",
            "llama.cpp",
            "llama_cpp",
        }:
            raise ValueError("runtime-llama supports only llama.cpp HTTP")
        # Historical LM Studio lifecycle knobs are never silently accepted.
        forbidden = {
            "lmstudio_load_config",
            "lmstudio_auto_load",
            "load_config",
            "auto_load",
            "ttl",
            "sdk_host",
            "sdk_idle_timeout_sec",
        }
        if forbidden.intersection(merged):
            raise ValueError("runtime lifecycle settings belong to the external server")
        if merged.get("instance_count", 1) != 1 or isinstance(
            merged.get("instance_count", 1), bool
        ):
            raise ValueError("multiple managed instances are not supported")
        timeout = merged.get(
            "stream_idle_timeout_sec",
            merged.get("timeout", merged.get("invoke_timeout_sec", 30.0)),
        )
        model = LlamaServerConfig(
            model=merged["model"],
            base_url=merged.get("base_url", DEFAULT_BASE_URL),
            context_length=merged.get("model_context_length"),
            verify_ssl=merged.get("verify_ssl", True),
            runtime_identity=merged.get("runtime_identity"),
            proxy_url=merged.get("proxy_url"),
        )
        role_fields = {
            name: merged[name]
            for name in (
                "runaway_min_chars",
                "runaway_window_chars",
                "runaway_block_chars",
                "runaway_repeat_threshold",
                "runaway_ngram_chars",
                "runtime_char_token_ratio_seed",
            )
            if name in merged
        }
        role = LlamaRoleConfig(
            role_id=role_id,
            reasoning=merged.get("reasoning", "auto"),
            context_ceiling=merged.get("context_ceiling", merged.get("num_ctx")),
            visible_output_tokens=merged.get("num_predict"),
            **role_fields,
        )
        fields = {
            name: merged[name]
            for name in (
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "repeat_penalty",
                "output_mode",
                "api_key",
                "api_key_env",
                "extra_body",
                "json_schema",
                "chat_template_kwargs",
                "thinking_diagnostics",
            )
            if name in merged
        }
        generation = LlamaGenerationConfig(stream_idle_timeout_sec=timeout, **fields)
        emb = {
            name: merged[name]
            for name in (
                "batch_size",
                "batch_token_limit",
                "token_count_parallelism",
                "normalize",
                "dimension",
                "max_input_tokens",
                "embedding_request_parallelism",
                "embedding_queue_size",
                "embedding_replicas",
                "embedding_prefixes",
                "embedding_replica_discovery",
            )
            if name in merged
        }
        return cls(merged.get("adapter_type", "llm"), model, role, generation, **emb)

    @property
    def role_id(self) -> str:
        return self.role.role_id

    @property
    def model(self) -> str:
        return self.model_config.model

    @property
    def base_url(self) -> str:
        return self.model_config.base_url

    @property
    def backend(self) -> str:
        return "llamacpp"

    raw_backend = backend

    @property
    def timeout(self) -> float:
        idle = self.generation.stream_idle_timeout_sec
        return 30.0 if idle is None else idle

    @property
    def num_ctx(self) -> int | None:
        return self.role.context_ceiling

    @property
    def num_predict(self) -> int | None:
        return self.role.visible_output_tokens

    @property
    def reasoning(self) -> str:
        return self.role.reasoning

    def __getattr__(self, name: str) -> Any:
        if name.startswith("runaway_") or name == "runtime_char_token_ratio_seed":
            return getattr(self.role, name)
        raise AttributeError(name)

    def to_llm_config(self) -> dict[str, Any]:
        if self.adapter_type != "llm":
            raise ValueError("profile is not an LLM role")
        return {"profile": self}

    def to_embedding_config(self) -> dict[str, Any]:
        if self.adapter_type != "embedding":
            raise ValueError("profile is not an embedding role")
        return {"profile": self}


ModelRoleProfile = AdapterProfile
