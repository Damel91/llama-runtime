from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import json
from threading import Barrier, Event, enumerate as threads
import time
import traceback

import pytest

from runtime_llama.errors import LLMBackendError, LLMCancelledError
from runtime_llama.runtime_discovery import RuntimeEndpoint, discover_runtimes
import runtime_llama.runtime_discovery as discovery
from test_model_discovery_transport import endpoint as endpoint, send


def server(endpoint, *, backend="llamacpp", entries=None, health=None, props=None):
    if entries is None:
        entries = [{"id": "chosen", "status": {"value": "loaded"}}]
    canonical = next(
        (item["id"] for item in entries if "alias_of" not in item), "chosen"
    )
    health = (
        {
            "status": "ok",
            **(
                {"service": "strata", "loaded": True, "model": canonical}
                if backend == "strata"
                else {}
            ),
        }
        if health is None
        else health
    )
    props = (
        {
            "build_info": "Strata 0.1.52"
            if backend == "strata"
            else "b11548-781dbc5ac",
            **(
                {"model_alias": canonical, "is_sleeping": False}
                if backend == "strata"
                else {}
            ),
        }
        if props is None
        else props
    )

    def respond(handler):
        value = {
            "/health": health,
            "/props": props,
            "/v1/models": {"data": entries},
        }.get(handler.path)
        if value is None:
            pytest.fail("runtime probe invoked a non-read-only endpoint")
        send(handler, json.dumps(value).encode())

    return endpoint(respond)


def test_catalog_lists_each_runtime_and_auto_selects_requested_loaded_model(endpoint):
    llama, llama_calls = server(endpoint)
    strata, strata_calls = server(
        endpoint,
        backend="strata",
        entries=[{"id": "next", "status": {"value": "loaded"}}],
    )
    catalog = discover_runtimes(
        [RuntimeEndpoint("native", llama), RuntimeEndpoint("second", strata)]
    )
    assert [item.backend for item in catalog.runtimes] == ["llamacpp", "strata"]
    assert [item.loaded_models[0].id for item in catalog.runtimes] == ["chosen", "next"]
    assert catalog.select("chosen").backend == "llamacpp"
    assert catalog.select("next").runtime.endpoint.name == "second"
    assert catalog.http_requests == 6
    assert (
        llama_calls
        == strata_calls
        == [("/health", None), ("/props", None), ("/v1/models", None)]
    )
    with pytest.raises(FrozenInstanceError):
        catalog.runtimes[0].available = False


def test_native_standalone_props_reused_without_repeated_catalog(endpoint):
    base_url, calls = server(
        endpoint,
        entries=[{"id": "chosen"}],
        props={
            "build_info": "11548 (781dbc5ac)",
            "model_alias": "chosen",
            "is_sleeping": False,
            "default_generation_settings": {"n_ctx": 8192},
            "total_slots": 4,
        },
    )
    selected = discover_runtimes([RuntimeEndpoint("local", base_url)]).select("chosen")
    assert selected.model.loaded_context.value == 8192
    assert selected.model.slots.value == 4
    assert len(calls) == 3


def test_strata_explicit_alias_rows_collapsed_and_selected_as_canonical(endpoint):
    canonical = {
        "id": "chosen",
        "aliases": ["friendly"],
        "status": {"value": "loaded"},
        "meta": {"n_ctx": 4096},
    }
    alias = {**canonical, "id": "friendly", "alias_of": "chosen"}
    base_url, calls = server(endpoint, backend="strata", entries=[alias, canonical])
    catalog = discover_runtimes([RuntimeEndpoint("local", base_url)])
    assert len(catalog.runtimes[0].loaded_models) == 1
    assert catalog.select("friendly").model.id == "chosen"
    assert len(calls) == 3


@pytest.mark.parametrize(
    "patch", [{"alias_of": "absent"}, {"meta": {"n_ctx": 1024}}, {"alias_of": []}]
)
def test_strata_alias_rows_never_hide_contradictions(endpoint, patch):
    canonical = {"id": "chosen", "aliases": ["friendly"], "status": {"value": "loaded"}}
    alias = {**canonical, "id": "friendly", "alias_of": "chosen", **patch}
    base_url, _ = server(endpoint, backend="strata", entries=[canonical, alias])
    catalog = discover_runtimes([RuntimeEndpoint("local", base_url)])
    assert not catalog.runtimes[0].available
    assert catalog.runtimes[0].diagnostics == ("runtime_discovery_invalid_alias",)
    with pytest.raises(LLMBackendError, match="runtime_model_unavailable"):
        catalog.select("chosen")


def test_unknown_compatible_server_is_listed_but_not_guessed(endpoint):
    base_url, calls = server(endpoint, props={"build_info": "some-other-runtime 1.2.3"})
    catalog = discover_runtimes([RuntimeEndpoint("unknown", base_url)])
    observed = catalog.runtimes[0]
    assert observed.backend is None and not observed.available
    assert observed.loaded_models[0].id == "chosen"
    assert "runtime_backend_unsupported" in observed.diagnostics
    assert len(calls) == 3
    with pytest.raises(LLMBackendError, match="unavailable"):
        catalog.select("chosen")


@pytest.mark.parametrize("endpoint_backend", ["auto", "llamacpp", "strata"])
def test_conflicting_server_markers_are_never_silently_overridden(
    endpoint, endpoint_backend
):
    base_url, calls = server(
        endpoint, health={"status": "ok", "service": "strata", "loaded": True}
    )
    catalog = discover_runtimes(
        [RuntimeEndpoint("local", base_url, backend=endpoint_backend)]
    )
    assert catalog.runtimes[0].diagnostics == ("runtime_identity_conflict",)
    assert not catalog.runtimes[0].available and len(calls) == 2


def test_explicit_legacy_deployment_backend_is_usable_without_version_marker(endpoint):
    base_url, calls = server(endpoint, props={})
    catalog = discover_runtimes(
        [RuntimeEndpoint("legacy", base_url, backend="llamacpp")]
    )
    assert catalog.select("chosen").backend == "llamacpp"
    assert len(calls) == 3


def test_no_loading_or_failed_unknown_models_are_selected(endpoint):
    base_url, calls = server(
        endpoint,
        entries=[
            {"id": "loading", "status": {"value": "loading"}},
            {"id": "unloaded", "status": {"value": "unloaded"}},
            {"id": "unknown"},
        ],
    )
    catalog = discover_runtimes([RuntimeEndpoint("local", base_url)])
    assert catalog.runtimes[0].loaded_models == ()
    for name in ("loading", "unloaded", "unknown", "absent"):
        with pytest.raises(LLMBackendError, match="unavailable"):
            catalog.select(name)
    assert len(calls) == 3


def test_strata_health_does_not_promote_stale_catalog_loaded_model(endpoint):
    base_url, _ = server(
        endpoint,
        backend="strata",
        health={"status": "ok", "service": "strata", "loaded": False},
    )
    catalog = discover_runtimes([RuntimeEndpoint("local", base_url)])
    assert catalog.runtimes[0].diagnostics == (
        "runtime_loaded_state_unverified",
        "runtime_model_state_incoherent",
    )
    with pytest.raises(LLMBackendError, match="unavailable"):
        catalog.select("chosen")


@pytest.mark.parametrize(
    "health_patch,props_patch",
    [
        ({"model": "different"}, {}),
        ({"model": None}, {}),
        ({}, {"model_alias": "different"}),
        ({}, {"is_sleeping": True}),
        ({"max_context": 1024}, {"default_generation_settings": {"n_ctx": 2048}}),
        ({}, {"default_generation_settings": {"n_ctx": True}}),
    ],
)
def test_strata_model_state_must_agree_across_independent_metadata(
    endpoint, health_patch, props_patch
):
    health = {
        "status": "ok",
        "service": "strata",
        "loaded": True,
        "model": "chosen",
        **health_patch,
    }
    props = {
        "build_info": "Strata 0.1.52",
        "model_alias": "chosen",
        "is_sleeping": False,
        **props_patch,
    }
    url, calls = server(endpoint, backend="strata", health=health, props=props)
    catalog = discover_runtimes([RuntimeEndpoint("configured", url)])
    runtime = catalog.runtimes[0]
    assert not runtime.available and runtime.llm_model_id is None
    assert runtime.loaded_models[0].id == "chosen"
    assert "runtime_model_state_incoherent" in runtime.diagnostics
    with pytest.raises(LLMBackendError, match="unavailable"):
        catalog.select("chosen")
    assert len(calls) == 3


def test_strata_other_loaded_rows_cannot_route_to_unrelated_llm(endpoint):
    entries = [{"id": name, "status": "loaded"} for name in ("chosen", "other")]
    url, _ = server(endpoint, backend="strata", entries=entries)
    catalog = discover_runtimes([RuntimeEndpoint("configured", url)])
    assert [model.id for model in catalog.runtimes[0].loaded_models] == [
        "chosen",
        "other",
    ]
    assert catalog.select("chosen").model.id == "chosen"
    with pytest.raises(LLMBackendError, match="unavailable"):
        catalog.select("other")


def test_bad_endpoint_does_not_hide_a_good_runtime(endpoint):
    bad_url, bad_calls = endpoint(
        lambda handler: send(handler, b'{"error":{"message":"PRIVATE"}}', status=401)
    )
    good_url, _ = server(endpoint, backend="strata")
    catalog = discover_runtimes(
        [
            RuntimeEndpoint("bad", bad_url, api_key="PRIVATE"),
            RuntimeEndpoint("good", good_url),
        ]
    )
    assert catalog.select("chosen").runtime.endpoint.name == "good"
    assert catalog.runtimes[0].diagnostics == ("runtime_probe_http_401",)
    assert len(bad_calls) == 1
    assert "PRIVATE" not in repr(catalog)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"data": None},
        {"data": [{"id": []}]},
        {"data": [{"id": "chosen", "status": "loaded", "meta": {"n_ctx": True}}]},
    ],
)
def test_malformed_catalog_is_distinct_from_unreachable_runtime(endpoint, payload):
    def bad(handler):
        value = {
            "/health": {"status": "ok"},
            "/props": {"build_info": "b11548-781dbc5ac"},
            "/v1/models": payload,
        }[handler.path]
        send(handler, json.dumps(value).encode())

    bad_url, calls = endpoint(bad)
    good_url, _ = server(endpoint)
    catalog = discover_runtimes(
        [RuntimeEndpoint("bad", bad_url), RuntimeEndpoint("good", good_url)]
    )
    assert catalog.runtimes[0].diagnostics == ("runtime_discovery_invalid_catalog",)
    assert catalog.select("chosen").runtime.endpoint.name == "good"
    assert len(calls) == 3


def test_unreachable_endpoint_stays_visible_without_hiding_good_runtime(endpoint):
    good, _ = server(endpoint)
    catalog = discover_runtimes(
        [
            RuntimeEndpoint("offline", "http://127.0.0.1:1"),
            RuntimeEndpoint("good", good),
        ],
        timeout=0.5,
    )
    assert catalog.runtimes[0].catalog is None
    assert not catalog.runtimes[0].available
    assert catalog.runtimes[0].diagnostics == ("runtime_probe_failed",)
    assert catalog.select("chosen").runtime.endpoint.name == "good"


def test_strata_missing_root_props_is_identified_but_not_used_for_inference(endpoint):
    def respond(handler):
        if handler.path == "/props":
            send(handler, b"{}", status=404)
        else:
            value = (
                {"status": "ok", "service": "strata", "model": "chosen", "loaded": True}
                if handler.path == "/health"
                else {"data": [{"id": "chosen", "status": "loaded"}]}
            )
            send(handler, json.dumps(value).encode())

    url, calls = endpoint(respond)
    catalog = discover_runtimes([RuntimeEndpoint("configured", url)])
    assert catalog.runtimes[0].backend == "strata"
    assert catalog.runtimes[0].loaded_models[0].id == "chosen"
    assert not catalog.runtimes[0].available
    assert catalog.runtimes[0].diagnostics == ("runtime_model_state_incoherent",)
    with pytest.raises(LLMBackendError, match="unavailable"):
        catalog.select("chosen")
    assert len(calls) == 3


def test_discovery_is_fresh_and_does_not_keep_selected_runtime(endpoint):
    native_entries = [{"id": "chosen", "status": {"value": "loaded"}}]
    strata_entries = []
    native, native_calls = server(endpoint, entries=native_entries)
    strata, strata_calls = server(endpoint, backend="strata", entries=strata_entries)
    endpoints = [RuntimeEndpoint("first", native), RuntimeEndpoint("second", strata)]
    first = discover_runtimes(endpoints)
    assert first.select("chosen").backend == "llamacpp"
    native_entries.clear()
    strata_entries.append({"id": "chosen", "status": {"value": "loaded"}})
    assert discover_runtimes(endpoints).select("chosen").backend == "strata"
    assert first.select("chosen").backend == "llamacpp"
    assert len(native_calls) == len(strata_calls) == 6


def test_same_canonical_id_uses_configuration_order_not_probe_completion_order(
    endpoint,
):
    barrier = Barrier(2)

    def reply(handler):
        if handler.path == "/health":
            barrier.wait(timeout=1)
            time.sleep(0.05)
        send(
            handler,
            json.dumps(
                {
                    "/health": {"status": "ok"},
                    "/props": {"build_info": "b11548-781dbc5ac"},
                    "/v1/models": {"data": [{"id": "chosen", "status": "loaded"}]},
                }[handler.path]
            ).encode(),
        )

    def reply_fast(handler):
        if handler.path == "/health":
            barrier.wait(timeout=1)
        send(
            handler,
            json.dumps(
                {
                    "/health": {
                        "status": "ok",
                        "service": "strata",
                        "loaded": True,
                        "model": "chosen",
                    },
                    "/props": {
                        "build_info": "Strata 0.1.52",
                        "model_alias": "chosen",
                        "is_sleeping": False,
                    },
                    "/v1/models": {"data": [{"id": "chosen", "status": "loaded"}]},
                }[handler.path]
            ).encode(),
        )

    first, _ = endpoint(reply)
    second, _ = endpoint(reply_fast)
    catalog = discover_runtimes(
        [RuntimeEndpoint("first", first), RuntimeEndpoint("second", second)]
    )
    assert catalog.select("chosen").runtime.endpoint.name == "first"


def test_exact_id_wins_over_same_alias_on_another_runtime(endpoint):
    first, _ = server(
        endpoint,
        entries=[{"id": "different", "aliases": ["chosen"], "status": "loaded"}],
    )
    second, _ = server(endpoint, backend="strata")
    selected = discover_runtimes(
        [RuntimeEndpoint("first", first), RuntimeEndpoint("second", second)]
    ).select("chosen")
    assert selected.runtime.endpoint.name == "second"


@pytest.mark.parametrize("conflict", ["alias", "hash"])
def test_ambiguous_model_identity_is_not_guessed(endpoint, conflict):
    one = {
        "id": "chosen" if conflict == "hash" else "one",
        "aliases": ["friendly"],
        "status": "loaded",
        "tags": ["noxi-content-sha256-v1:" + "a" * 64],
    }
    two = {
        **one,
        "id": "chosen" if conflict == "hash" else "two",
        "tags": ["noxi-content-sha256-v1:" + "b" * 64],
    }
    first, _ = server(endpoint, entries=[one])
    second, _ = server(endpoint, backend="strata", entries=[two])
    catalog = discover_runtimes(
        [RuntimeEndpoint("first", first), RuntimeEndpoint("second", second)]
    )
    with pytest.raises(LLMBackendError, match="ambiguous"):
        catalog.select("chosen" if conflict == "hash" else "friendly")


def test_llm_and_embedding_purpose_do_not_route_to_explicitly_unsupported_models(
    endpoint,
):
    native, _ = server(
        endpoint,
        entries=[
            {
                "id": "embed",
                "status": "loaded",
                "capabilities": {"chat": False, "embeddings": True},
            }
        ],
    )
    strata, _ = server(
        endpoint,
        backend="strata",
        entries=[
            {
                "id": "image",
                "status": "loaded",
                "architecture": {"output_modalities": ["image"]},
            }
        ],
    )
    catalog = discover_runtimes(
        [RuntimeEndpoint("first", native), RuntimeEndpoint("second", strata)]
    )
    assert catalog.select("embed", purpose="embedding").backend == "llamacpp"
    for model in ("embed", "image"):
        with pytest.raises(LLMBackendError, match="unavailable"):
            catalog.select(model)
    with pytest.raises(LLMBackendError, match="unavailable"):
        catalog.select("image", purpose="embedding")


def test_one_timeout_does_not_hide_already_observed_runtime_and_workers_are_drained(
    endpoint,
):
    started = Event()

    def slow(handler):
        started.set()
        time.sleep(0.08)
        send(handler, b'{"status":"ok"}')

    slow_url, _ = endpoint(slow)
    good_url, _ = server(endpoint)
    catalog = discover_runtimes(
        [RuntimeEndpoint("slow", slow_url), RuntimeEndpoint("good", good_url)],
        timeout=0.06,
    )
    assert started.is_set()
    assert catalog.select("chosen").runtime.endpoint.name == "good"
    assert catalog.runtimes[0].diagnostics == ("runtime_probe_timeout",)
    assert not any(thread.name.startswith("runtime-probe") for thread in threads())


def test_cancellation_before_requests_and_during_probes_drains_workers(endpoint):
    began = Event()
    cancel = Event()

    def reply(handler):
        began.set()
        time.sleep(0.12)
        send(handler, b'{"status":"ok"}')

    url, calls = endpoint(reply)
    endpoints = [RuntimeEndpoint("slow", url)]
    with pytest.raises(LLMCancelledError):
        discover_runtimes(endpoints, cancellation_probe=lambda: True)
    assert calls == []
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            discover_runtimes, endpoints, timeout=0.3, cancellation_probe=cancel.is_set
        )
        assert began.wait(1)
        cancel.set()
        with pytest.raises(LLMCancelledError):
            pending.result(timeout=1)
    assert len(calls) == 1
    assert not any(thread.name.startswith("runtime-probe") for thread in threads())


def test_environment_auth_is_explicit_and_not_visible_in_projection(
    endpoint, monkeypatch
):
    monkeypatch.setenv("RUNTIME_TEST_KEY", "PRIVATE-RUNTIME-KEY")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    url, calls = server(endpoint)
    catalog = discover_runtimes(
        [RuntimeEndpoint("known", url, api_key_env="RUNTIME_TEST_KEY")]
    )
    assert catalog.select("chosen").backend == "llamacpp"
    assert {auth for _, auth in calls} == {"Bearer PRIVATE-RUNTIME-KEY"}
    assert "PRIVATE-RUNTIME-KEY" not in repr(catalog)


def test_cancellation_callback_failure_is_sanitized(endpoint):
    url, calls = server(endpoint)

    def failed():
        raise RuntimeError("PRIVATE-CALLBACK-MESSAGE")

    with pytest.raises(LLMBackendError, match="cancellation_probe_failed") as caught:
        discover_runtimes([RuntimeEndpoint("known", url)], cancellation_probe=failed)
    assert "PRIVATE" not in "".join(traceback.format_exception(caught.value))
    assert calls == []


@pytest.mark.parametrize("timeout", [0, True, float("nan"), float("inf"), 10**1000])
def test_invalid_timeout_is_rejected_before_requests(endpoint, timeout):
    url, calls = server(endpoint)
    with pytest.raises(ValueError):
        discover_runtimes([RuntimeEndpoint("known", url)], timeout=timeout)
    assert calls == []


def test_empty_runtime_configuration_returns_empty_catalog():
    catalog = discover_runtimes([])
    assert catalog.runtimes == () and catalog.http_requests == 0
    with pytest.raises(LLMBackendError, match="unavailable"):
        catalog.select("chosen")


def test_portable_endpoint_mapping_defaults_to_automatic_detection():
    configured = RuntimeEndpoint.from_value(
        {"base_url": "https://inference.example.net"}
    )
    assert configured.name == "https://inference.example.net"
    assert configured.backend == "auto"
    assert RuntimeEndpoint.from_value(configured) is configured
    native = RuntimeEndpoint.from_value(
        {
            "name": "native",
            "base_url": "https://inference.example.net",
            "backend": "llama.cpp",
        }
    )
    assert native.backend == "llamacpp"


@pytest.mark.parametrize(
    "value",
    [
        None,
        "https://inference.example.net",
        {},
        {"base_url": "https://inference.example.net", "unexpected": "PRIVATE"},
    ],
)
def test_invalid_portable_endpoint_configuration_is_sanitized(value):
    with pytest.raises(ValueError) as caught:
        RuntimeEndpoint.from_value(value)
    assert "PRIVATE" not in str(caught.value)


@pytest.mark.parametrize(
    "options",
    [
        {"backend": "other"},
        {"verify_ssl": 1},
        {"api_key": "one", "api_key_env": "KEY"},
        {"api_key": "bad\nheader"},
        {"api_key_env": "BAD ENV"},
        {"name": "bad\ud800"},
        {"base_url": "https://user:PRIVATE@example.net"},
    ],
)
def test_endpoint_configuration_never_echoes_credentials(options):
    arguments = {
        "name": "configured",
        "base_url": "https://inference.example.net",
        **options,
    }
    with pytest.raises(ValueError) as caught:
        RuntimeEndpoint(**arguments)
    assert "PRIVATE" not in str(caught.value)


def test_no_fixed_model_count_limit_is_introduced(endpoint):
    url, _ = server(
        endpoint,
        entries=[{"id": f"model-{index}", "status": "loaded"} for index in range(1001)],
    )
    catalog = discover_runtimes([RuntimeEndpoint("large", url)])
    assert len(catalog.runtimes[0].loaded_models) == 1001
    assert catalog.select("model-1000").model.id == "model-1000"


@pytest.mark.parametrize("backend", ["llamacpp", "strata"])
def test_cancellation_during_large_catalog_normalization_is_prompt(
    endpoint, monkeypatch, backend
):
    entries = [{"id": f"model-{index}", "status": "loaded"} for index in range(1001)]
    url, calls = server(endpoint, backend=backend, entries=entries)
    cancel = Event()
    visits = []
    observe = discovery._observe

    def observed(entry, diagnostics):
        visits.append(entry["id"])
        if len(visits) == 5:
            cancel.set()
        return observe(entry, diagnostics)

    monkeypatch.setattr(discovery, "_observe", observed)
    with pytest.raises(LLMCancelledError):
        discover_runtimes(
            [RuntimeEndpoint("many", url)], cancellation_probe=cancel.is_set
        )
    assert len(visits) == 5
    assert len(calls) == 3
    assert not any(thread.name.startswith("runtime-probe") for thread in threads())


def test_exact_but_unloaded_model_does_not_fall_back_to_different_alias(endpoint):
    one, _ = server(endpoint, entries=[{"id": "chosen", "status": "unloaded"}])
    two, _ = server(
        endpoint,
        entries=[{"id": "different", "aliases": ["chosen"], "status": "loaded"}],
    )
    catalog = discover_runtimes(
        [RuntimeEndpoint("one", one), RuntimeEndpoint("two", two)]
    )
    with pytest.raises(LLMBackendError, match="runtime_model_unavailable"):
        catalog.select("chosen")
