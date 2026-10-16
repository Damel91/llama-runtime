from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import json
from threading import Event
import time
import traceback

import pytest
import requests

from runtime_llama.client import ClientHTTPError
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError
from runtime_llama.model_discovery import (
    ModelCapabilities,
    ModelIdentity,
    ObservedInt,
    discover_models,
)
from test_model_discovery_transport import endpoint as endpoint, send


def fixture_entry(model="chosen", state="loaded", **extra):
    return {"id": model, "status": {"value": state}, **extra}


def fixture_server(endpoint, entries, props=None, *, props_status=200):
    def respond(handler):
        if handler.path == "/v1/models":
            send(handler, json.dumps({"data": entries}).encode())
        elif handler.path == "/props":
            send(handler, json.dumps(props or {}).encode(), status=props_status)
        else:
            pytest.fail("discovery selected a model or invoked another endpoint")

    return endpoint(respond)


def test_catalog_loaded_separation_metadata_and_frozen_projection(endpoint):
    states = ("loaded", "loading", "unloaded", "sleeping", "downloading", "downloaded")
    entries = [
        fixture_entry(f"entry-{index}", state) for index, state in enumerate(states)
    ]
    entries[0].update(
        aliases=["friendly"],
        capabilities={"chat": True, "embeddings": False},
        meta={"n_ctx": 2048, "n_ctx_train": 8192, "n_embd": 512},
        tags=[
            "noxi-content-sha256-v1:" + "a" * 64,
            "noxi-model-name-v1:Weights%20%C3%A9.gguf",
        ],
    )
    entries[1]["meta"] = {"n_ctx": 2048}
    entries[2]["status"]["failed"] = True
    entries[3]["context_length"] = 4096
    base_url, observed = fixture_server(endpoint, entries)
    catalog = discover_models(base_url)
    assert [model.state for model in catalog.models] == list(states)
    assert catalog.loaded_models == (catalog.models[0],)
    loaded = catalog.models[0]
    assert loaded.aliases == ("entry-0", "friendly")
    assert loaded.capabilities == ModelCapabilities(True, False)
    assert loaded.identity == ModelIdentity("a" * 64, None, "Weights é.gguf")
    assert loaded.loaded_context == ObservedInt(2048, "catalog.meta.n_ctx")
    assert loaded.training_context == ObservedInt(8192, "catalog.meta.n_ctx_train")
    assert loaded.model_embedding_width == ObservedInt(512, "catalog.meta.n_embd")
    assert catalog.models[1].loaded_context is None
    assert catalog.models[1].capabilities == ModelCapabilities()
    assert catalog.models[2].failed is True
    assert catalog.models[3].training_context == ObservedInt(
        4096, "catalog.context_length"
    )
    assert catalog.http_requests == 1 and catalog.elapsed_seconds >= 0
    assert catalog.diagnostics == ()
    assert observed == [("/v1/models", None)]
    with pytest.raises(FrozenInstanceError):
        loaded.state = "unloaded"
    with pytest.raises(FrozenInstanceError):
        loaded.capabilities.chat = False


def test_empty_catalog_uses_one_get_without_root_props(endpoint):
    base_url, observed = fixture_server(endpoint, [])
    catalog = discover_models(base_url)
    assert catalog.models == catalog.loaded_models == catalog.diagnostics == ()
    assert catalog.http_requests == 1
    assert observed == [("/v1/models", None)]


def test_string_loaded_status_uses_existing_helper_without_root_props(endpoint):
    base_url, observed = fixture_server(
        endpoint, [{"id": "chosen", "status": "loaded", "meta": {"n_ctx": 2048}}]
    )
    catalog = discover_models(base_url)
    assert catalog.loaded_models == catalog.models
    assert catalog.models[0].loaded_context == ObservedInt(2048, "catalog.meta.n_ctx")
    assert observed == [("/v1/models", None)]


@pytest.mark.parametrize("sleeping", [False, True])
def test_single_unknown_model_proved_only_by_explicit_root_metadata(endpoint, sleeping):
    entry = {"id": "chosen", "aliases": ["exact-alias"], "meta": {"n_ctx": 100}}
    props = {
        "role": "model",
        "model_alias": "exact-alias",
        "is_sleeping": sleeping,
        "default_generation_settings": {"n_ctx": 2048},
        "total_slots": 4,
    }
    base_url, observed = fixture_server(endpoint, [entry], props)
    catalog = discover_models(base_url)
    model = catalog.models[0]
    assert model.state == ("sleeping" if sleeping else "loaded")
    assert model.loaded_context == (
        None
        if sleeping
        else ObservedInt(2048, "root_props.default_generation_settings.n_ctx")
    )
    assert model.slots == ObservedInt(4, "root_props.total_slots")
    assert catalog.loaded_models == (() if sleeping else (model,))
    assert catalog.http_requests == 2
    assert observed == [("/v1/models", None), ("/props", None)]


@pytest.mark.parametrize(
    "props",
    [
        {},
        {"role": "router", "model_alias": "chosen", "is_sleeping": False},
        {"model_alias": "chosen", "is_sleeping": False},
        {"role": "model", "model_alias": "different", "is_sleeping": False},
        {"role": "model", "model_alias": "chosen"},
        {"role": "model", "model_alias": "chosen", "is_sleeping": 0},
    ],
)
def test_unknown_stays_unknown_without_complete_standalone_proof(endpoint, props):
    base_url, observed = fixture_server(endpoint, [{"id": "chosen"}], props)
    catalog = discover_models(base_url)
    assert catalog.models[0].state == "unknown"
    assert catalog.models[0].loaded_context is None
    assert catalog.loaded_models == ()
    assert catalog.diagnostics == ("root_props_load_state_unverified",)
    assert all("?" not in path for path, _ in observed)


def test_multi_model_catalog_never_promoted_by_root_alias(endpoint):
    props = {"role": "model", "model_alias": "unknown", "is_sleeping": False}
    base_url, observed = fixture_server(
        endpoint, [fixture_entry("loaded"), {"id": "unknown"}], props
    )
    catalog = discover_models(base_url)
    assert [model.state for model in catalog.models] == ["loaded", "unknown"]
    assert [model.id for model in catalog.loaded_models] == ["loaded"]
    assert len(observed) == 2


@pytest.mark.parametrize("status", [404, 405, 501])
def test_optional_root_props_unsupported_keeps_catalog(endpoint, status):
    base_url, observed = fixture_server(
        endpoint, [{"id": "chosen"}], props_status=status
    )
    catalog = discover_models(base_url)
    assert catalog.models[0].state == "unknown"
    assert catalog.diagnostics == ("root_props_unavailable",)
    assert len(observed) == 2


@pytest.mark.parametrize("status", [301, 400, 401, 403, 500, 503])
def test_other_props_errors_propagate_without_body_or_credentials(endpoint, status):
    secret = "PRIVATE-TOKEN-AND-SERVER-MESSAGE"
    base_url, observed = fixture_server(
        endpoint,
        [{"id": "chosen"}],
        {"error": {"message": secret}},
        props_status=status,
    )
    with pytest.raises(ClientHTTPError) as caught:
        discover_models(base_url, api_key=secret)
    assert caught.value.status_code == status
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert len(observed) == 2


def test_future_state_and_unstructured_capabilities_are_unknown_not_guessed(endpoint):
    entry = fixture_entry(
        "embedding-chat.gguf",
        "future-native-state",
        capabilities=["completion", "embedding"],
        meta={"n_embd": 768, "n_ctx": 9999},
        path="/private/model.gguf",
        chat_template="template-present",
    )
    base_url, _ = fixture_server(endpoint, [entry], props_status=404)
    catalog = discover_models(base_url)
    model = catalog.models[0]
    assert model.state == "unknown" and model.loaded_context is None
    assert model.capabilities == ModelCapabilities()
    assert model.identity == ModelIdentity()
    assert model.model_embedding_width.value == 768
    assert catalog.diagnostics == (
        "root_props_unavailable",
        "unknown_model_state",
        "unsupported_capabilities_shape",
    )
    assert "/private/" not in repr(catalog)
    assert "template-present" not in repr(catalog)


@pytest.mark.parametrize(
    "entries",
    [
        [fixture_entry(), fixture_entry()],
        [
            fixture_entry("one", aliases=["same"]),
            fixture_entry("two", aliases=["same"]),
        ],
        [fixture_entry("one", aliases=["two"]), fixture_entry("two")],
    ],
)
def test_duplicate_ids_and_cross_entry_alias_collisions_rejected(endpoint, entries):
    base_url, observed = fixture_server(endpoint, entries)
    with pytest.raises(LLMBackendError, match="ambiguous_alias"):
        discover_models(base_url)
    assert len(observed) == 1


@pytest.mark.parametrize(
    "entry",
    [
        {"id": ""},
        {"id": True},
        {"id": "x" * 1025},
        {"id": "bad\ud800"},
        fixture_entry(aliases=["x"] * 65),
        fixture_entry(aliases=[1]),
        fixture_entry(aliases=["bad\udfff"]),
        fixture_entry(status=True),
        fixture_entry(status={"value": []}),
        fixture_entry(status={"value": "loaded", "failed": "yes"}),
        fixture_entry(status={"value": "loaded", "failed": True}),
        fixture_entry(capabilities={"chat": 1}),
        fixture_entry(capabilities={"embeddings": "true"}),
        fixture_entry(meta={"n_embd": True}),
        fixture_entry(meta={"n_ctx": -1}),
        fixture_entry(meta={"n_ctx_train": 2**31}),
        fixture_entry(context_length="8192"),
        fixture_entry(tags=["noxi-content-sha256-v1:bad"]),
        fixture_entry(tags=["noxi-model-name-v1:%ZZ"]),
        fixture_entry(tags=["noxi-model-name-v1:%FF-SECRET"]),
        fixture_entry(tags=["noxi-model-name-v1:%2Fhome%2Fsecret"]),
        fixture_entry(tags=[False]),
    ],
)
def test_malformed_or_unbounded_selected_fields_are_sanitized(endpoint, entry):
    base_url, observed = fixture_server(endpoint, [entry])
    with pytest.raises(LLMBackendError, match="invalid_catalog") as caught:
        discover_models(base_url)
    assert "SECRET" not in "".join(traceback.format_exception(caught.value))
    assert len(observed) == 1


def test_identity_hashes_independent_and_conflicts_not_silently_overwritten(endpoint):
    entry = fixture_entry(tags=["noxi-embedding-sha256-v1:" + "b" * 64])
    base_url, _ = fixture_server(endpoint, [entry])
    assert discover_models(base_url).models[0].identity == ModelIdentity(None, "b" * 64)
    entry["tags"] = ["noxi-content-sha256-v1:" + digest * 64 for digest in ("a", "b")]
    with pytest.raises(LLMBackendError, match="ambiguous_identity"):
        discover_models(base_url)


def test_failed_unknown_model_cannot_be_promoted_by_root_proof(endpoint):
    entry = {"id": "chosen", "status": {"failed": True}}
    props = {"role": "model", "model_alias": "chosen", "is_sleeping": False}
    base_url, observed = fixture_server(endpoint, [entry], props)
    with pytest.raises(LLMBackendError, match="invalid_catalog"):
        discover_models(base_url)
    assert len(observed) == 2


@pytest.mark.parametrize("payload", [[], {}, {"data": None}, {"data": [None]}])
def test_catalog_shape_must_be_supported(endpoint, payload):
    base_url, observed = endpoint(lambda h: send(h, json.dumps(payload).encode()))
    with pytest.raises(LLMBackendError, match="invalid_catalog"):
        discover_models(base_url)
    assert len(observed) == 1


def test_model_count_and_response_bytes_bound_before_optional_requests(endpoint):
    base_url, observed = fixture_server(
        endpoint, [fixture_entry("one"), fixture_entry("two")]
    )
    with pytest.raises(LLMBackendError, match="model_limit"):
        discover_models(base_url, max_models=1)
    with pytest.raises(LLMBackendError, match="response_too_large"):
        discover_models(base_url, max_response_bytes=16)
    assert observed == [("/v1/models", None), ("/v1/models", None)]


@pytest.mark.parametrize(
    "options",
    [
        {"timeout": 0},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"timeout": True},
        {"timeout": 10**1000},
        {"verify_ssl": 1},
        {"cancellation_probe": True},
        {"max_models": 0},
        {"max_models": 4097},
        {"max_response_bytes": True},
        {"max_response_bytes": 16 * 1024 * 1024 + 1},
        {"api_key": "one", "api_key_env": "OTHER"},
        {"api_key": ""},
        {"api_key": "bad\nheader"},
        {"api_key_env": "NOT AN ENV NAME"},
    ],
)
def test_invalid_parameters_never_dispatch(endpoint, options):
    base_url, observed = fixture_server(endpoint, [])
    with pytest.raises(ValueError):
        discover_models(base_url, **options)
    assert observed == []


def test_explicit_env_auth_and_ambient_auth_ignored(endpoint, monkeypatch, tmp_path):
    netrc = tmp_path / "netrc"
    netrc.write_text("machine 127.0.0.1 login wrong password wrong\n")
    monkeypatch.setenv("NETRC", str(netrc))
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("DISCOVERY_TEST_KEY", "explicit-key")
    base_url, observed = fixture_server(endpoint, [fixture_entry()])
    assert discover_models(base_url, api_key_env="DISCOVERY_TEST_KEY").loaded_models
    assert observed == [("/v1/models", "Bearer explicit-key")]
    monkeypatch.delenv("DISCOVERY_TEST_KEY")
    with pytest.raises(ValueError, match="unavailable"):
        discover_models(base_url, api_key_env="DISCOVERY_TEST_KEY")
    assert len(observed) == 1


def test_tls_and_redirect_policy_forwarded_to_owned_client(monkeypatch):
    seen = []

    class Session:
        def mount(self, *_args):
            pass

        def request(self, **kwargs):
            seen.append((kwargs, self.trust_env))
            response = requests.Response()
            response.status_code = 200
            response._content = b'{"data":[]}'
            return response

        def close(self):
            pass

    monkeypatch.setattr(requests, "Session", Session)
    assert discover_models("https://chosen.example").models == ()
    call, trust_env = seen[0]
    assert call["method"] == "GET" and call["json"] is None
    assert call["verify"] is True and call["allow_redirects"] is False
    assert trust_env is False


def test_redirect_is_not_followed(endpoint):
    target_url, target_calls = fixture_server(endpoint, [])
    base_url, observed = endpoint(
        lambda h: send(
            h, b"{}", status=302, extra=(("Location", target_url + "/v1/models"),)
        )
    )
    with pytest.raises(ClientHTTPError) as caught:
        discover_models(base_url, api_key="private-key")
    assert caught.value.status_code == 302
    assert len(observed) == 1 and target_calls == []


def test_aggregate_deadline_not_reset_for_root_request(endpoint):
    def respond(handler):
        time.sleep(0.08)
        body = {"data": [{"id": "chosen"}]} if handler.path == "/v1/models" else {}
        send(handler, json.dumps(body).encode())

    base_url, observed = endpoint(respond)
    started = time.monotonic()
    with pytest.raises(LLMTimeoutError):
        discover_models(base_url, timeout=0.13)
    assert time.monotonic() - started < 0.3
    assert len(observed) == 2


def test_cancellation_before_request_and_while_headers_pending(endpoint):
    started = Event()

    def respond(handler):
        started.set()
        time.sleep(0.12)
        send(handler, b'{"data":[]}')

    base_url, observed = endpoint(respond)
    with pytest.raises(LLMCancelledError):
        discover_models(base_url, cancellation_probe=lambda: True)
    assert observed == []
    cancel = Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            discover_models, base_url, timeout=0.3, cancellation_probe=cancel.is_set
        )
        assert started.wait(1)
        cancel.set()
        with pytest.raises(LLMCancelledError):
            pending.result(timeout=1)
    assert len(observed) == 1


def test_cancellation_probe_failure_has_no_raw_message(endpoint):
    base_url, observed = fixture_server(endpoint, [])

    def failed():
        raise RuntimeError("PRIVATE-CANCELLATION-CREDENTIAL")

    with pytest.raises(LLMBackendError, match="cancellation_probe_failed") as caught:
        discover_models(base_url, cancellation_probe=failed)
    assert "PRIVATE-CANCELLATION" not in "".join(
        traceback.format_exception(caught.value)
    )
    assert observed == []
