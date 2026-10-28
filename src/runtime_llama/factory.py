"""Consumer factories for externally managed inference runtimes."""

from __future__ import annotations

from typing import Any

from runtime_llama.adapters.embedding import LlamaCppEmbeddingAdapter
from runtime_llama.adapters.llama import LlamaCppAdapter
from runtime_llama.adapters.strata import StrataAdapter
from runtime_llama.adapters.automatic import AutoRuntimeAdapter
from runtime_llama.client import LlamaCppClient
from runtime_llama.configuration import AdapterProfile


def _profile(
    config: dict[str, Any] | AdapterProfile, adapter_type: str
) -> AdapterProfile:
    if isinstance(config, AdapterProfile):
        profile = config
    else:
        raw = dict(config)
        if "backend" in raw:
            backend = raw["backend"]
            if not isinstance(backend, str) or backend not in {
                "llamacpp",
                "llama.cpp",
                "llama_cpp",
                "strata",
                "auto",
            }:
                raise ValueError("unsupported runtime backend")
        existing = raw.get("profile", raw.get("role_profile"))
        if isinstance(existing, AdapterProfile) and "backend" in raw:
            backend = {"llama.cpp": "llamacpp", "llama_cpp": "llamacpp"}.get(
                raw["backend"], raw["backend"]
            )
            if existing.backend != backend:
                raise ValueError("backend conflicts with the supplied profile")
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
    profile = _profile(config, "llm")
    if profile.backend == "auto":
        if client is not None:
            raise ValueError("automatic routing owns its endpoint-specific clients")
        return AutoRuntimeAdapter(profile)
    adapter = StrataAdapter if profile.backend == "strata" else LlamaCppAdapter
    return adapter.from_profile(profile, client=client)


def build_embedding_adapter(
    config: dict[str, Any] | AdapterProfile, *, client: LlamaCppClient | None = None
) -> LlamaCppEmbeddingAdapter:
    return LlamaCppEmbeddingAdapter.from_profile(
        _profile(config, "embedding"), client=client
    )
