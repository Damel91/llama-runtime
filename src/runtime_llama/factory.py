"""Consumer factories for the dedicated llama.cpp HTTP adapters."""

from __future__ import annotations

from typing import Any

from runtime_llama.adapters.embedding import LlamaCppEmbeddingAdapter
from runtime_llama.adapters.llama import LlamaCppAdapter
from runtime_llama.client import LlamaCppClient
from runtime_llama.configuration import AdapterProfile


def _profile(
    config: dict[str, Any] | AdapterProfile, adapter_type: str
) -> AdapterProfile:
    if isinstance(config, AdapterProfile):
        profile = config
    else:
        raw = dict(config)
        if raw.get("backend", "llamacpp") not in {"llamacpp", "llama.cpp", "llama_cpp"}:
            raise ValueError(
                "Unsupported backend; runtime-llama supports only llama.cpp HTTP"
            )
        existing = raw.get("profile", raw.get("role_profile"))
        profile = (
            existing
            if isinstance(existing, AdapterProfile)
            else AdapterProfile.from_config(
                raw.get(
                    "role_id", "embedding" if adapter_type == "embedding" else "unknown"
                ),
                {**raw, "adapter_type": adapter_type},
            )
        )
    if profile.adapter_type != adapter_type:
        raise ValueError(f"Profile is not an {adapter_type} role")
    return profile


def build_adapter(
    config: dict[str, Any] | AdapterProfile, *, client: LlamaCppClient | None = None
) -> LlamaCppAdapter:
    return LlamaCppAdapter.from_profile(_profile(config, "llm"), client=client)


def build_embedding_adapter(
    config: dict[str, Any] | AdapterProfile, *, client: LlamaCppClient | None = None
) -> LlamaCppEmbeddingAdapter:
    return LlamaCppEmbeddingAdapter.from_profile(
        _profile(config, "embedding"), client=client
    )
