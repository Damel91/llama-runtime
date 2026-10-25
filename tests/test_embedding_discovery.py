from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
import time
from urllib.parse import quote

import pytest

from runtime_llama.adapters.embedding import LlamaCppEmbeddingAdapter
from runtime_llama.configuration import AdapterProfile
from runtime_llama.replica_discovery import (
    ReplicaIdentity,
    discover_embedding_pool,
)
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError

CONTENT = "a" * 64
SEMANTIC = "b" * 64
OTHER = "c" * 64
NAME = "embedding model BF16.gguf"


def entry(alias, *, content=CONTENT, semantic=SEMANTIC, name=NAME, loaded=True):
    tags = []
    if content is not None:
        tags.append(f"noxi-content-sha256-v1:{content}")
    if semantic is not None:
        tags.append(f"noxi-embedding-sha256-v1:{semantic}")
    if name is not None:
        tags.append(f"noxi-model-name-v1:{quote(name, safe='')}")
    return {
        "id": alias,
        "aliases": [],
        "tags": tags,
        "status": {"value": "loaded" if loaded else "unloaded"},
    }


def pool(entries, selector="embed_0", **kwargs):
    return discover_embedding_pool({"data": entries}, selector, **kwargs)


def test_content_identity_groups_paths_without_relying_on_names_or_dimensions():
    first, second = entry("embed_0"), entry("embed_1", name="different.gguf")
    first["path"], second["path"] = "/first/model.gguf", "/second/copy.gguf"
    first["meta"], second["meta"] = {"n_embd": 512}, {"n_embd": 768}
    result = pool([second, first])
    assert [r.model for r in result.replicas] == ["embed_0", "embed_1"]
    assert result.identity == ReplicaIdentity(CONTENT, SEMANTIC)


def test_same_path_or_dimension_does_not_group_different_content():
    first, second = entry("embed_0"), entry("embed_1", content=OTHER)
    for item in (first, second):
        item["path"] = "/same/model.gguf"
        item["meta"] = {"n_embd": 768}
    assert [r.model for r in pool([first, second]).replicas] == ["embed_0"]


def test_embedding_semantics_must_match_even_with_identical_weights():
    assert len(pool([entry("embed_0"), entry("embed_1", semantic=OTHER)]).replicas) == 1


@pytest.mark.parametrize("selector", [NAME, f"sha256:{CONTENT}"])
def test_logical_name_and_digest_resolve_all_copies(selector):
    assert len(pool([entry("embed_0"), entry("embed_1")], selector).replicas) == 2


@pytest.mark.parametrize("selector", [NAME, f"sha256:{CONTENT}"])
def test_logical_or_digest_selector_does_not_choose_ambiguous_semantics(selector):
    with pytest.raises(LLMBackendError, match="ambiguous_model_identity"):
        pool([entry("embed_0"), entry("embed_1", semantic=OTHER)], selector)


def test_logical_name_with_different_weights_is_ambiguous():
    with pytest.raises(LLMBackendError, match="ambiguous_model_identity"):
        pool([entry("embed_0"), entry("embed_1", content=OTHER)], NAME)


@pytest.mark.parametrize("selector", [NAME, f"sha256:{CONTENT}"])
def test_malformed_matching_identity_cannot_silently_resolve_logical_selector(selector):
    with pytest.raises(LLMBackendError):
        pool([entry("embed_0"), entry("embed_1", semantic=None)], selector)


def test_unloaded_replicas_are_excluded():
    assert len(pool([entry("embed_0"), entry("embed_1", loaded=False)]).replicas) == 1


def test_missing_hash_allows_only_one_exact_alias_and_remains_sticky():
    missing = entry("embed_0", content=None, semantic=None)
    result = pool([missing, entry("embed_1")])
    assert result.identity is None and result.fallback_alias == "embed_0"
    assert len(result.replicas) == 1
    upgraded = pool(
        [entry("embed_0"), entry("embed_1")], fallback_alias=result.fallback_alias
    )
    assert upgraded.identity is None and len(upgraded.replicas) == 1


def test_logical_name_cannot_use_unverified_fallback():
    with pytest.raises(LLMBackendError, match="ambiguous_model_identity"):
        pool([entry("embed_0", content=None, semantic=None)], NAME)


@pytest.mark.parametrize(
    "changes",
    [
        {"content": "path-hash"},
        {"content": CONTENT.upper()},
        {"content": None},
        {"semantic": None},
        {"name": "directory/model.gguf"},
    ],
)
def test_invalid_identity_on_exact_anchor_never_degrades_to_fallback(changes):
    with pytest.raises(LLMBackendError):
        pool([entry("embed_0", **changes)])


def test_conflicting_identity_tags_are_rejected():
    target = entry("embed_0")
    target["tags"].append(f"noxi-content-sha256-v1:{OTHER}")
    with pytest.raises(LLMBackendError, match="ambiguous_identity_tags"):
        pool([target])


def test_alias_collision_never_routes_to_arbitrary_entry():
    other = entry("other", content=OTHER)
    other["aliases"].append("embed_0")
    with pytest.raises(LLMBackendError, match="ambiguous_model"):
        pool([entry("embed_0"), other])


def test_pinned_identity_survives_anchor_removal_but_rejects_alias_reuse():
    pinned = ReplicaIdentity(CONTENT, SEMANTIC)
    assert pool([entry("embed_1")], pinned=pinned).identity == pinned
    with pytest.raises(LLMBackendError, match="content_identity_changed"):
        pool([entry("embed_0", content=OTHER), entry("embed_1")], pinned=pinned)


@pytest.mark.parametrize("catalog", [None, {}, {"data": None}, {"data": [3]}])
def test_invalid_catalog_is_typed_failure(catalog):
    with pytest.raises(LLMBackendError, match="invalid_model_catalog"):
        discover_embedding_pool(catalog, "embed_0")


@pytest.mark.parametrize(
    "settings",
    [{"embedding_replica_discovery": 1}, {"embedding_replica_discovery": "true"}],
)
def test_discovery_configuration_is_strict_boolean(settings):
    with pytest.raises(ValueError):
        AdapterProfile.from_config(
            "embedding", {"model": "x", "adapter_type": "embedding", **settings}
        )


def test_defaults_keep_automatic_policy_and_preserve_positive_guards():
    legacy = AdapterProfile.from_config(
        "embedding", {"model": "x", "adapter_type": "embedding"}
    )
    auto = AdapterProfile.from_config(
        "embedding",
        {
            "model": "x",
            "adapter_type": "embedding",
            "embedding_replica_discovery": True,
        },
    )
    explicit = AdapterProfile.from_config(
        "embedding",
        {
            "model": "x",
            "adapter_type": "embedding",
            "embedding_replica_discovery": True,
            "embedding_request_parallelism": 2,
        },
    )
    assert legacy.embedding_request_parallelism is None
    assert auto.embedding_request_parallelism is None
    assert explicit.embedding_request_parallelism == 2


def test_discovery_rejects_explicit_replica_list_and_llm_profile():
    for values in (
        {"adapter_type": "llm"},
        {"adapter_type": "embedding", "embedding_replicas": [{"model": "x"}]},
    ):
        with pytest.raises(ValueError):
            AdapterProfile.from_config(
                "test", {"model": "x", "embedding_replica_discovery": True, **values}
            )


class Client:
    def __init__(self):
        self.catalog = [entry("embed_0"), entry("embed_1")]
        self.contexts = {"embed_0": 4096, "embed_1": 4096, "embed_2": 4096}
        self.lock = Lock()
        self.active = {}
        self.peak = {}
        self.requests = []
        self.props_calls = []
        self.tokens = []
        self.release = Event()
        self.release.set()
        self.started = Event()
        self.failure = None
        self.closed = False

    def models(self, **kwargs):
        if self.failure:
            raise self.failure
        return {"data": deepcopy(self.catalog)}

    def health(self, **kwargs):
        return {"status": "ok"}

    def props(self, model=None, **kwargs):
        self.props_calls.append(model)
        model = model or self.catalog[0]["id"]
        return {
            "model_alias": model,
            "total_slots": 1,
            "default_generation_settings": {"n_ctx": self.contexts[model]},
        }

    def tokenize(self, content, **kwargs):
        self.tokens.append(content)
        return {"tokens": list(range(len(content) + 1))}

    def embeddings(self, input, model, *, cancel=None, **kwargs):
        with self.lock:
            self.active[model] = self.active.get(model, 0) + 1
            self.peak[model] = max(self.peak.get(model, 0), self.active[model])
            self.requests.append((model, tuple(input)))
            self.started.set()
        try:
            while not self.release.wait(0.01):
                if cancel and cancel():
                    raise LLMCancelledError()
            return {
                "data": [
                    {"index": i, "embedding": [1.0, float(len(text))]}
                    for i, text in enumerate(input)
                ]
            }
        finally:
            with self.lock:
                self.active[model] -= 1

    def close(self):
        self.closed = True


def adapter(client, **settings):
    profile = AdapterProfile.from_config(
        "documents",
        {
            "adapter_type": "embedding",
            "model": "embed_0",
            "embedding_replica_discovery": True,
            "batch_size": 1,
            "timeout": 2,
            **settings,
        },
    )
    return LlamaCppEmbeddingAdapter(profile, client=client)


def eventually(check):
    deadline = time.monotonic() + 3
    while not check():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_shared_callers_use_both_discovered_copies_once_per_alias():
    client = Client()
    client.release.clear()
    runtime = adapter(client, embedding_queue_size=1)
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(runtime.embed_texts, ["a", "bb", "ccc"])
            second = callers.submit(runtime.embed_query, "query")
            eventually(lambda: sum(client.active.values()) == 2)
            assert client.peak == {"embed_0": 1, "embed_1": 1}
            assert runtime._scheduler.snapshot()["queued"] <= 1
            client.release.set()
            assert first.result(timeout=3) == [[1, 1], [1, 2], [1, 3]]
            assert second.result(timeout=3) == [1, 5]
        health = runtime.health_status()
        assert health["embedding_replica_discovery"]["identity_verified"]
        assert {r["replica_id"] for r in health["replicas"].values()} == {0, 1}
        assert health["scheduler"]["in_flight"] == 0
    finally:
        client.release.set()
        runtime.close()
    assert not client.closed  # The injected transport is caller-owned.


def test_discovery_uses_logical_name_without_consumer_alias_list():
    client = Client()
    runtime = adapter(client, model=NAME)
    try:
        assert runtime.embed_query("test") == [1, 4]
        assert set(runtime.health_status()["replicas"]) == {"embed_0", "embed_1"}
    finally:
        runtime.close()


def test_fallback_is_serial_even_with_global_parallelism_four():
    client = Client()
    client.catalog = [entry("embed_0", content=None, semantic=None), entry("embed_1")]
    client.release.clear()
    runtime = adapter(client)
    try:
        with ThreadPoolExecutor(max_workers=1) as callers:
            pending = callers.submit(runtime.embed_texts, ["a", "bb", "ccc"])
            assert client.started.wait(2)
            time.sleep(0.03)
            assert client.requests == [("embed_0", ("a",))]
            client.release.set()
            assert len(pending.result(timeout=3)) == 3
        client.catalog = [entry("embed_0"), entry("embed_1")]
        assert len(runtime.health_status()["replicas"]) == 1
        assert not runtime.health_status()["embedding_replica_discovery"][
            "identity_verified"
        ]
    finally:
        client.release.set()
        runtime.close()


def test_outage_does_not_use_cached_pool_and_next_call_recovers():
    client = Client()
    runtime = adapter(client)
    try:
        runtime.embed_query("a")
        count = len(client.requests)
        client.failure = LLMBackendError("llama_server_transport_failure")
        with pytest.raises(LLMBackendError, match="transport_failure"):
            runtime.embed_query("bb")
        assert len(client.requests) == count
        assert not runtime.is_ready()
        client.failure = None
        assert runtime.embed_query("ccc") == [1, 3]
    finally:
        runtime.close()


def test_context_change_is_observed_and_prefix_tokens_are_counted():
    client = Client()
    client.contexts = {"embed_0": 4, "embed_1": 4}
    runtime = adapter(client, embedding_prefixes={"query": "Q: ", "document": ""})
    try:
        assert runtime.max_input_tokens == 4
        with pytest.raises(LLMBackendError, match="input_exceeds_context"):
            runtime.embed_query("aa")
        assert not client.requests
        assert runtime.embed_texts(["aa"], input_type="raw") == [[1, 2]]
        client.contexts = {"embed_0": 4096, "embed_1": 4096}
        assert runtime.embed_query("aa") == [1, 5]
    finally:
        runtime.close()


def test_unloaded_anchor_never_causes_props_or_embedding_requests():
    client = Client()
    client.catalog = [entry("embed_0", loaded=False), entry("embed_1")]
    runtime = adapter(client)
    try:
        with pytest.raises(LLMBackendError, match="model_not_loaded"):
            runtime.embed_query("test")
        assert not client.props_calls and not client.tokens and not client.requests
        health = runtime.health_status()
        assert health["server_healthy"] and not health["model_ready"]
    finally:
        runtime.close()


@pytest.mark.parametrize("verified", [True, False])
def test_native_single_model_catalog_without_status_uses_root_props(verified):
    client = Client()
    target = entry(
        "embed_0",
        content=CONTENT if verified else None,
        semantic=SEMANTIC if verified else None,
    )
    del target["status"]
    client.catalog = [target]
    runtime = adapter(client)
    try:
        assert runtime.embed_query("test") == [1, 4]
        assert None in client.props_calls
        assert runtime.health_status()["model_ready"]
        assert (
            runtime.health_status()["embedding_replica_discovery"]["identity_verified"]
            == verified
        )
    finally:
        runtime.close()


def test_router_catalog_without_status_never_selects_an_unloaded_alias():
    client = Client()
    target = entry("embed_0")
    del target["status"]
    client.catalog = [target]
    client.props = lambda model=None, **kwargs: {
        "role": "router",
        "model_alias": "embed_0",
    }
    runtime = adapter(client)
    try:
        with pytest.raises(LLMBackendError, match="model_not_loaded"):
            runtime.embed_query("test")
        assert not client.tokens and not client.requests
    finally:
        runtime.close()


def test_ids_survive_catalog_order_and_topology_replacement():
    client = Client()
    runtime = adapter(client)
    try:
        original = {
            alias: r["replica_id"]
            for alias, r in runtime.health_status()["replicas"].items()
        }
        client.catalog.reverse()
        assert original == {
            alias: r["replica_id"]
            for alias, r in runtime.health_status()["replicas"].items()
        }
        client.catalog = [entry("embed_1"), entry("embed_2")]
        changed = runtime.health_status()["replicas"]
        assert changed["embed_1"]["replica_id"] == original["embed_1"]
        assert changed["embed_2"]["replica_id"] == 2
        client.catalog = [entry("embed_0"), entry("embed_1")]
        assert (
            runtime.health_status()["replicas"]["embed_0"]["replica_id"]
            == original["embed_0"]
        )
    finally:
        runtime.close()


def test_alias_reused_for_different_weights_never_changes_pinned_identity():
    client = Client()
    runtime = adapter(client)
    try:
        runtime.embed_query("first")
        client.catalog = [entry("embed_0", content=OTHER), entry("embed_1")]
        with pytest.raises(LLMBackendError, match="identity_changed"):
            runtime.embed_query("second")
        assert len(client.requests) == 1
    finally:
        runtime.close()


def test_identity_change_at_dispatch_is_rejected_before_selected_props_and_post():
    client = Client()
    original_models = client.models
    calls = 0
    at_change = None

    def changing_catalog(**kwargs):
        nonlocal calls, at_change
        calls += 1
        if calls == 2:
            at_change = (len(client.props_calls), len(client.tokens))
            client.catalog = [
                entry("embed_0", content=OTHER),
                entry("embed_1", content=OTHER),
            ]
        return original_models(**kwargs)

    client.models = changing_catalog
    runtime = adapter(client)
    try:
        with pytest.raises(LLMBackendError, match="identity_changed"):
            runtime.embed_query("test")
        # The admission catalog is shared with its properties, so the second
        # catalog now observes dispatch. Nothing contacts the changed model
        # after that observation: neither selected properties nor inference.
        assert at_change is not None
        assert (len(client.props_calls), len(client.tokens)) == at_change
        assert not client.requests
    finally:
        runtime.close()


def test_replica_id_history_is_bounded_and_evicted_ids_are_never_reused():
    client = Client()
    runtime = adapter(client)
    try:
        for index in range(1030):
            name = f"copy_{index}"
            client.catalog = [entry("embed_0"), entry(name)]
            client.contexts[name] = 4096
            runtime.health_status()
        assert len(runtime._replica_ids) <= 1026
        previous = runtime._next_replica_id
        client.catalog = [entry("embed_0"), entry("embed_1")]
        assert runtime.health_status()["replicas"]["embed_1"]["replica_id"] >= previous
    finally:
        runtime.close()


def test_topology_transition_waits_for_existing_call_then_uses_remaining_copy():
    client = Client()
    client.release.clear()
    runtime = adapter(client)
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(runtime.embed_query, "first")
            assert client.started.wait(2)
            client.catalog = [entry("embed_1")]
            second = callers.submit(runtime.embed_query, "second")
            time.sleep(0.03)
            assert not second.done()
            client.release.set()
            assert first.result(timeout=3) == [1, 5]
            assert second.result(timeout=3) == [1, 6]
        assert [model for model, _ in client.requests] == ["embed_0", "embed_1"]
    finally:
        client.release.set()
        runtime.close()


@pytest.mark.parametrize("cancel_wait", [False, True])
def test_topology_wait_is_bounded_and_cancellable(cancel_wait):
    client = Client()
    client.release.clear()
    runtime = adapter(client)
    cancelled = Event()
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(runtime.embed_query, "first")
            assert client.started.wait(2)
            client.catalog = [entry("embed_1")]
            second = callers.submit(
                runtime.embed_query, "second", timeout=0.15, cancel=cancelled.is_set
            )
            if cancel_wait:
                time.sleep(0.03)
                cancelled.set()
            with pytest.raises(LLMCancelledError if cancel_wait else LLMTimeoutError):
                second.result(timeout=2)
            client.release.set()
            assert first.result(timeout=3) == [1, 5]
    finally:
        client.release.set()
        runtime.close()


def test_close_wakes_topology_waiters_and_drains_active_call():
    client = Client()
    client.release.clear()
    runtime = adapter(client)
    with ThreadPoolExecutor(max_workers=3) as callers:
        first = callers.submit(runtime.embed_query, "first")
        assert client.started.wait(2)
        client.catalog = [entry("embed_1")]
        second = callers.submit(runtime.embed_query, "second")
        time.sleep(0.03)
        closing = callers.submit(runtime.close)
        closing.result(timeout=3)
        for pending in (first, second):
            with pytest.raises((LLMBackendError, LLMCancelledError)):
                pending.result(timeout=3)
    assert runtime._pool_users == 0
    assert runtime._scheduler.snapshot()["in_flight"] == 0


@pytest.mark.parametrize("replica_count", [2, 4])
def test_real_http_discovers_copies_with_one_request_per_alias(replica_count):
    from test_embedding_concurrency import response, server, vectors

    with server() as (state, url):
        aliases = [f"embed_{index}" for index in range(replica_count)]
        state.contexts = dict.fromkeys(aliases, 4096)
        state.catalog = [entry(alias) for alias in aliases]
        state.props_overrides = {alias: {"total_slots": 1} for alias in aliases}

        def action(body):
            assert state.release.wait(3)
            return 200, response(body["input"])

        state.action = action
        runtime = LlamaCppEmbeddingAdapter(
            AdapterProfile.from_config(
                "documents",
                {
                    "adapter_type": "embedding",
                    "model": NAME,
                    "base_url": url,
                    "embedding_replica_discovery": True,
                    "batch_size": 2,
                    "embedding_queue_size": 2,
                    "timeout": 2,
                },
            )
        )
        first = [f"item-{i}" for i in range(12)]
        second = [f"item-{i}" for i in range(100, 112)]
        try:
            with ThreadPoolExecutor(max_workers=2) as callers:
                a = callers.submit(runtime.embed_texts, first)
                b = callers.submit(runtime.embed_texts, second)
                eventually(lambda: state.active == replica_count)
                assert state.peak_by_model == dict.fromkeys(aliases, 1)
                assert runtime._scheduler.snapshot()["queued"] <= 2
                state.release.set()
                assert a.result(timeout=4) == vectors(first)
                assert b.result(timeout=4) == vectors(second)
            status = runtime.health_status()
            assert status["model_ready"]
            assert status["scheduler"]["peak_in_flight"] == replica_count
            assert status["scheduler"]["in_flight"] == 0
        finally:
            state.release.set()
            runtime.close()
