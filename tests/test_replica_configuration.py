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


def test_persistent_queue_defaults_are_finite_without_a_required_timer():
    parsed = profile()
    assert parsed.embedding_batch_wait_sec == 0.0
    assert parsed.embedding_max_pending_bytes == 64 * 1024 * 1024
    assert parsed.embedding_max_pending_tokens == 1024 * 1024
    assert parsed.embedding_max_request_bytes == 8 * 1024 * 1024


@pytest.mark.parametrize(
    "name",
    [
        "embedding_max_pending_bytes",
        "embedding_max_pending_tokens",
        "embedding_max_request_bytes",
    ],
)
@pytest.mark.parametrize("value", [None, 0, -1, True, 1.5, "1024"])
def test_queue_resource_limits_require_positive_integer_budgets(name, value):
    with pytest.raises(ValueError, match=name):
        profile(**{name: value})


@pytest.mark.parametrize("wait", [0, 0.0, 0.25, 1.0])
def test_nonnegative_coalescing_policy_survives_profile_parsing(wait):
    assert profile(embedding_batch_wait_sec=wait).embedding_batch_wait_sec == wait


@pytest.mark.parametrize("wait", [None, -1, True, float("inf"), float("nan"), "slow"])
def test_invalid_coalescing_policy_is_rejected_by_profile(wait):
    with pytest.raises(ValueError, match="embedding_batch_wait_sec"):
        profile(embedding_batch_wait_sec=wait)


def test_queue_budgets_are_configurable_without_changing_model_identity(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("profile construction must not send HTTP")

    monkeypatch.setattr("requests.Session.request", forbidden)
    settings = {
        "embedding_max_pending_bytes": 2 * 1024 * 1024,
        "embedding_max_pending_tokens": 32768,
        "embedding_max_request_bytes": 256 * 1024,
        "embedding_batch_wait_sec": 0.75,
    }
    parsed = profile(**settings)
    for name, value in settings.items():
        assert getattr(parsed, name) == value
    assert (
        parsed.model_config.physical_identity
        == profile().model_config.physical_identity
    )
