"""Public profile bounds and immutable configuration at the integration boundary."""

from dataclasses import FrozenInstanceError

import pytest

from runtime_llama import AdapterProfile, EmbeddingReplicaConfig


def profile(**kwargs):
    return AdapterProfile.from_config(
        "ingest", {"adapter_type": "embedding", "model": "embed_0", **kwargs}
    )


def test_explicit_replica_config_is_copied_and_sampling_keeps_runtime_identity():
    settings = [
        {"model": "embed_0", "max_in_flight": 2},
        {"model": "embed_1", "max_in_flight": 2, "max_input_tokens": 512},
    ]
    first = profile(embedding_request_parallelism=4, embedding_replicas=settings)
    settings[0]["model"] = "changed-after-construction"
    second = profile(temperature=0.9)
    assert first.embedding_replicas[0].model == "embed_0"
    assert first.model_config.physical_identity == second.model_config.physical_identity
    with pytest.raises(FrozenInstanceError):
        first.embedding_replicas[0].max_in_flight = 10


def test_sequential_defaults_and_zero_queue_are_supported():
    parsed = profile(embedding_queue_size=0)
    assert parsed.embedding_request_parallelism == 1
    assert parsed.embedding_replicas == ()
    assert parsed.embedding_queue_size == 0


@pytest.mark.parametrize("discovery, expected", [(False, 1), (True, 4)])
def test_none_parallelism_materializes_mode_default_without_http(
    discovery, expected, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("profile construction must not send HTTP")

    monkeypatch.setattr("requests.Session.request", forbidden)
    parsed = profile(
        embedding_replica_discovery=discovery, embedding_request_parallelism=None
    )
    assert parsed.embedding_request_parallelism == expected


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4"])
def test_invalid_parallelism_rejected_before_http(value):
    with pytest.raises(ValueError):
        profile(embedding_request_parallelism=value)


@pytest.mark.parametrize("value", [None, -1, True, 1.5, "2"])
def test_invalid_queue_capacity_rejected_before_http(value):
    with pytest.raises(ValueError):
        profile(embedding_queue_size=value)


@pytest.mark.parametrize(
    "value",
    [
        [{"model": "same"}, {"model": "same"}],
        [{"model": " "}],
        [{"model": "embed", "max_in_flight": 0}],
        [{"model": "embed", "max_in_flight": None}],
        [{"model": "embed", "batch_token_limit": True}],
        [{"model": "embed", "device": "ROCm0"}],
        [{"model": "embed", "api_key": "private"}],
        ["embed"],
        "embed",
    ],
)
def test_invalid_or_runtime_managed_replica_options_rejected(value):
    with pytest.raises(ValueError):
        profile(embedding_replicas=value)


def test_explicit_replicas_cannot_be_used_as_llm_profiles():
    with pytest.raises(ValueError):
        AdapterProfile.from_config(
            "coder", {"model": "llm", "embedding_replicas": [{"model": "embed"}]}
        )


def test_replica_objects_are_accepted():
    replica = EmbeddingReplicaConfig("embed_0", max_in_flight=2)
    assert profile(embedding_replicas=(replica,)).embedding_replicas == (replica,)
