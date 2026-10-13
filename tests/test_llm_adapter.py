from runtime_llama import LLMCallResult, build_adapter


def test_factory_attaches_role_profile_without_loading_model():
    adapter = build_adapter(
        {
            "backend": "llamacpp",
            "model": "qwen",
            "model_context_length": 4096,
            "role_id": "routing",
            "num_ctx": 4096,
            "num_predict": 512,
            "output_mode": "json",
        }
    )
    assert adapter.role_profile.role_id == "routing"
    assert adapter.role_profile.num_ctx == 4096
    assert adapter.role_profile.num_predict == 512
    assert LLMCallResult.__name__ == "LLMCallResult"
