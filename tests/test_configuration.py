import pytest

from runtime_llama.configuration import AdapterProfile, LlamaServerConfig


def test_remote_https_default_port_and_proxy_path():
    server = LlamaServerConfig("model", "https://inference.internal/llama/v1/")
    assert server.base_url == "https://inference.internal/llama"


def test_sampling_and_roles_share_physical_identity():
    first = AdapterProfile.from_config("coder", {"model": "qwen", "temperature": 0.1})
    second = AdapterProfile.from_config(
        "reviewer", {"model": "qwen", "temperature": 1.2, "num_predict": 400}
    )
    assert first.model_config.physical_identity == second.model_config.physical_identity
    assert first.role != second.role


@pytest.mark.parametrize(
    "settings",
    [
        {"auto_load": True},
        {"lmstudio_auto_load": False},
        {"load_config": {}},
        {"instance_count": 2},
    ],
)
def test_local_runtime_controls_are_explicitly_rejected(settings):
    with pytest.raises(ValueError):
        AdapterProfile.from_config("test", {"model": "qwen", **settings})


@pytest.mark.parametrize(
    "settings",
    [
        {"temperature": float("nan")},
        {"top_k": True},
        {"batch_size": 0},
        {"timeout": float("inf")},
        {"num_predict": -1},
    ],
)
def test_invalid_bounds_are_rejected(settings):
    with pytest.raises(ValueError):
        AdapterProfile.from_config("test", {"model": "qwen", **settings})


def test_api_key_not_in_profile_repr():
    profile = AdapterProfile.from_config(
        "test", {"model": "qwen", "api_key": "do-not-log-me"}
    )
    assert "do-not-log-me" not in repr(profile)


def test_thinking_log_does_not_include_content(caplog):
    import logging
    from runtime_llama.contracts import BaseLLMAdapter

    class Adapter(BaseLLMAdapter):
        def is_ready(self):
            return True

        def health_status(self):
            return {"healthy": True}

    with caplog.at_level(logging.INFO):
        Adapter()._log_thinking("sensitive-model-thought")
    assert "sensitive-model-thought" not in caplog.text
    assert "chars=23" in caplog.text
