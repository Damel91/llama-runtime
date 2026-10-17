from dataclasses import asdict
import json

import pytest

from runtime_llama import build_adapter
from runtime_llama.configuration import (
    AdapterProfile,
    LlamaGenerationConfig,
    LlamaRoleConfig,
    LlamaServerConfig,
)
from runtime_llama.usage import LLMTransportAttempt


def test_omission_preserves_finite_generation_and_profile_defaults():
    generation = LlamaGenerationConfig()
    profile = AdapterProfile.from_config("chat", {"model": "chosen"})
    assert generation.stream_idle_timeout_sec == 30.0
    assert profile.generation.stream_idle_timeout_sec == 30.0
    assert profile.timeout == 30.0


@pytest.mark.parametrize("value", [0.001, 12, 60.0])
def test_positive_generation_idle_remains_the_finite_profile_timeout(value):
    profile = AdapterProfile.from_config(
        "chat", {"model": "chosen", "stream_idle_timeout_sec": value}
    )
    assert profile.generation.stream_idle_timeout_sec == value
    assert profile.timeout == value


def test_explicit_none_survives_configuration_serialization_and_factory_roundtrip():
    raw = {"model": "chosen", "stream_idle_timeout_sec": None}
    profile = AdapterProfile.from_config("chat", json.loads(json.dumps(raw)))
    assert profile.generation.stream_idle_timeout_sec is None
    assert profile.timeout == 30.0
    assert asdict(profile.generation)["stream_idle_timeout_sec"] is None
    borrowed_client = object()
    adapter = build_adapter(profile.to_llm_config(), client=borrowed_client)
    try:
        assert adapter.role_profile is profile
        assert adapter._generation_config.stream_idle_timeout_sec is None
        assert adapter._client is borrowed_client
    finally:
        adapter.close()


def test_role_none_overrides_numeric_default_and_explicit_number_restores_idle():
    defaults = {"model": "chosen", "stream_idle_timeout_sec": 60.0}
    disabled = AdapterProfile.from_config(
        "chat", {"stream_idle_timeout_sec": None}, defaults
    )
    inherited = AdapterProfile.from_config("chat", {}, defaults)
    enabled = AdapterProfile.from_config(
        "chat",
        {"stream_idle_timeout_sec": 12},
        {"model": "chosen", "stream_idle_timeout_sec": None},
    )
    assert disabled.generation.stream_idle_timeout_sec is None
    assert disabled.timeout == 30.0
    assert inherited.generation.stream_idle_timeout_sec == 60.0
    assert enabled.generation.stream_idle_timeout_sec == 12


@pytest.mark.parametrize("alias", ["timeout", "invoke_timeout_sec"])
def test_existing_numeric_timeout_alias_remains_finite(alias):
    profile = AdapterProfile.from_config("chat", {"model": "chosen", alias: 10})
    assert profile.generation.stream_idle_timeout_sec == profile.timeout == 10


@pytest.mark.parametrize(
    "value",
    [0, -1, True, False, float("nan"), float("inf"), -float("inf"), 10**1000, "12"],
)
def test_invalid_idle_is_rejected_by_direct_and_mapping_configuration(value):
    for construct in (
        lambda: LlamaGenerationConfig(stream_idle_timeout_sec=value),
        lambda: AdapterProfile.from_config(
            "chat", {"model": "chosen", "stream_idle_timeout_sec": value}
        ),
    ):
        with pytest.raises(ValueError, match="stream_idle_timeout_sec"):
            construct()


def test_nullable_generation_does_not_disable_embedding_deadlines():
    generation = LlamaGenerationConfig(stream_idle_timeout_sec=None)
    with pytest.raises(ValueError, match="embedding timeout must remain finite"):
        AdapterProfile(
            "embedding",
            LlamaServerConfig("chosen"),
            LlamaRoleConfig("embedding"),
            generation,
        )
    with pytest.raises(ValueError, match="embedding timeout must remain finite"):
        AdapterProfile.from_config(
            "embedding",
            {
                "adapter_type": "embedding",
                "model": "chosen",
                "stream_idle_timeout_sec": None,
            },
        )
    finite = AdapterProfile.from_config(
        "embedding", {"adapter_type": "embedding", "model": "chosen", "timeout": 12}
    )
    assert finite.timeout == 12


def test_disabled_idle_evidence_serializes_to_null_without_changing_unknown_default():
    disabled = LLMTransportAttempt(idle_timeout_sec=None, terminal_reason="completed")
    encoded = json.loads(json.dumps(disabled.to_dict()))
    assert encoded["idle_timeout_sec"] is None
    assert encoded["terminal_reason"] == "completed"
    assert LLMTransportAttempt().to_dict()["idle_timeout_sec"] == 0.0
    assert (
        LLMTransportAttempt(idle_timeout_sec=12).to_dict()["idle_timeout_sec"] == 12.0
    )
