from dataclasses import replace

import pytest

from runtime_llama import (
    AdapterProfile,
    AutoRuntimeAdapter,
    LlamaCppAdapter,
    RuntimeEndpoint,
    StrataAdapter,
    build_adapter,
    build_embedding_adapter,
)


@pytest.mark.parametrize(
    "backend", ["llamacpp", "llama.cpp", "llama_cpp", "strata", "auto"]
)
def test_factory_provider_selection_is_lazy(monkeypatch, backend):
    def no_http(*args, **kwargs):
        raise AssertionError("construction performed HTTP")

    monkeypatch.setattr("requests.Session.request", no_http)
    expected = {"strata": StrataAdapter, "auto": AutoRuntimeAdapter}.get(
        backend, LlamaCppAdapter
    )
    adapter = build_adapter({"model": "requested", "backend": backend})
    assert type(adapter) is expected
    adapter.close()


def test_endpoint_list_implies_auto_and_survives_profile_roundtrip(monkeypatch):
    def no_http(*args, **kwargs):
        raise AssertionError("construction performed HTTP")

    monkeypatch.setattr("requests.Session.request", no_http)
    profile = AdapterProfile.from_config(
        "coder",
        {
            "model": "requested",
            "runtime_endpoints": [
                {"name": "one", "base_url": "https://inference.example/v1"}
            ],
            "runtime_discovery_timeout": 7,
        },
    )
    assert profile.backend == "auto"
    assert isinstance(profile.runtime_endpoints[0], RuntimeEndpoint)
    assert profile.runtime_endpoints[0].base_url == "https://inference.example"
    adapter = build_adapter(profile.to_llm_config())
    assert isinstance(adapter, AutoRuntimeAdapter)
    assert adapter.role_profile is profile
    assert profile.runtime_discovery_timeout == 7
    adapter.close()


@pytest.mark.parametrize("backend", [[], {}, True, None, "openai"])
def test_invalid_backend_is_value_error(backend):
    with pytest.raises(ValueError):
        build_adapter({"model": "requested", "backend": backend})
    with pytest.raises(ValueError):
        AdapterProfile.from_config("coder", {"model": "requested", "backend": backend})


@pytest.mark.parametrize("backend", ["strata", "auto"])
def test_embedding_factory_does_not_claim_unimplemented_backend(backend):
    with pytest.raises(ValueError, match="LLM-only"):
        build_embedding_adapter({"model": "embedding", "backend": backend})


def test_explicit_profile_backend_conflict_is_not_ignored():
    profile = AdapterProfile.from_config("coder", {"model": "requested"})
    with pytest.raises(ValueError, match="conflicts"):
        build_adapter({"profile": profile, "backend": "strata"})
    with pytest.raises(ValueError, match="unsupported"):
        replace(profile, backend_id=[])


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan"), None])
def test_discovery_timeout_stays_finite_with_unbounded_stream_idle(timeout):
    with pytest.raises(ValueError):
        build_adapter(
            {
                "model": "requested",
                "backend": "auto",
                "runtime_discovery_timeout": timeout,
                "stream_idle_timeout_sec": None,
            }
        )


def test_automatic_factory_rejects_borrowed_client_without_closing_it():
    class Borrowed:
        def close(self):
            raise AssertionError("borrowed client was closed")

    with pytest.raises(ValueError, match="owns"):
        build_adapter({"model": "requested", "backend": "auto"}, client=Borrowed())
