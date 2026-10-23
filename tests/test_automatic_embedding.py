"""Optional automatic facade: discovery, calibration, ownership and direct input."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json
import math
from threading import Event
from urllib.parse import quote

import pytest

import runtime_llama.automatic_embedding as module
from runtime_llama.automatic_embedding import (
    AutomaticEmbeddingConfig,
    AutomaticEmbeddingService,
    AutomaticEmbeddingStatus,
)
from runtime_llama.embedding_calibration import (
    CALIBRATION_CATEGORIES,
    CalibrationProfile,
    CalibrationStats,
)
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError

from test_embedding_async_stream import WindowSource, run, wait_thread
from test_embedding_contracts import (
    CONTENT,
    SEMANTIC,
    ContractClient,
    eventually,
    vector,
)


NAME = "nomic-embed-text-v2-moe.Q8_0.gguf"


def remote(*, replicas=2, context=512, name=NAME, verified=True):
    client = ContractClient(replicas, context=context, verified=verified)
    for entry in client.catalog:
        entry["tags"] = [
            tag for tag in entry["tags"] if not tag.startswith("noxi-model-name-v1:")
        ]
        if verified:
            entry["tags"].append("noxi-model-name-v1:" + quote(name, safe=""))
    return client


@pytest.fixture
def calibration(monkeypatch):
    calls = []

    def calibrate(cache_dir, binding, tokenize, check, workers=2, validate=None):
        check()
        calls.append((cache_dir, deepcopy(binding), workers))
        for source in ("calibration α", "second <tag> item"):
            tokenize(source)
        check()
        if validate is not None:
            validate()
        stats = {
            category: CalibrationStats(2, 80, 20, 4.0, 4.0, 4.0, 4.0)
            for category in (*CALIBRATION_CATEGORIES, "mixed")
        }
        key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        return CalibrationProfile(key, binding, "c" * 64, stats)

    monkeypatch.setattr(module, "load_or_calibrate", calibrate)
    return calls


@pytest.fixture
def make_service(tmp_path):
    services = []

    def create(client=None, **changes):
        client = client or remote()
        values = {
            "base_url": "http://runtime.example:8080",
            "cache_dir": tmp_path / "user-cache",
            "normalize": False,
        }
        values.update(changes)
        service = AutomaticEmbeddingService(values, client=client)
        services.append(service)
        return service, client

    yield create
    for service in services:
        service.close()


def test_constructor_and_async_factory_are_inert(make_service, monkeypatch):
    client = remote()

    def forbidden(*args, **kwargs):
        raise AssertionError("constructor performed remote work")

    for method in ("health", "models", "props", "tokenize", "embeddings"):
        monkeypatch.setattr(client, method, forbidden)
    service, _ = make_service(client)
    source = WindowSource([["one"]])
    service.astream_texts(source)
    assert service._adapter is None
    assert source.pulls == 0


def test_initialize_returns_frozen_availability_without_budget_or_calibration(
    make_service, calibration
):
    service, client = make_service()
    status = service.initialize()
    assert type(status) is AutomaticEmbeddingStatus
    assert status.available and status.reason is None
    assert status.max_input_chars is None and status.category is None
    assert status.calibration_id is None and status.input_type == "document"
    assert len(status.representation_id) == 64
    assert not hasattr(status, "replicas") and not hasattr(status, "model")
    with pytest.raises(FrozenInstanceError):
        status.available = False
    assert not client.token_calls and not calibration and not client.embedding_calls
    assert service.status().available
    profile = service._adapter.profile
    assert profile.max_input_tokens is None
    assert profile.model_config.context_length is None
    assert profile.batch_token_limit is None
    assert profile.embedding_replica_discovery is True


def test_explicit_character_budget_is_the_only_calibrating_entry_point(
    make_service, calibration
):
    service, client = make_service()
    status = service.character_budget()
    assert status.max_input_chars == math.floor(512 * 4 * 0.90)
    assert status.category == "mixed" and len(status.calibration_id) == 64
    assert [text for text, _ in client.token_calls] == [
        "search_document: calibration α",
        "search_document: second <tag> item",
    ]
    assert len(calibration) == 1
    assert calibration[0][1]["content_sha256"] == CONTENT
    assert calibration[0][1]["embedding_sha256"] == SEMANTIC
    assert calibration[0][1]["policy"].endswith("prefix=search_document: ")
    assert not client.embedding_calls


def test_categories_reuse_one_complete_profile_and_preserve_representation(
    make_service, calibration
):
    service, client = make_service()
    mixed = service.character_budget()
    code = service.character_budget(category="code")
    docs = service.character_budget(category="documentation")
    assert len(calibration) == 1 and len(client.token_calls) == 2
    assert mixed.max_input_chars == math.floor(512 * 4 * 0.90)
    assert code.max_input_chars == docs.max_input_chars == math.floor(512 * 4 * 0.80)
    assert mixed.representation_id == code.representation_id == docs.representation_id
    assert mixed.calibration_id == code.calibration_id == docs.calibration_id


def test_task_prefixes_are_calibrated_separately_and_applied_once(
    make_service, calibration
):
    service, client = make_service()
    document = service.character_budget()
    query = service.character_budget(input_type="query")
    raw = service.character_budget(input_type="raw")
    assert len(calibration) == 3
    assert len({document.calibration_id, query.calibration_id, raw.calibration_id}) == 3
    assert (
        document.representation_id == query.representation_id == raw.representation_id
    )
    assert service.embed_query("question") == vector("search_query: question")
    assert service.embed_texts(["body", "body", "search_document: source"]) == [
        vector("search_document: " + text)
        for text in ("body", "body", "search_document: source")
    ]
    assert service.embed_texts(["raw"], input_type="raw") == [vector("raw")]
    assert len(client.token_calls) == 6
    assert len(service._adapter._replicas) == 2


def test_loaded_context_and_microbatch_update_budget_without_recalibration(
    make_service, calibration
):
    service, client = make_service()
    first = service.character_budget()
    for model in client.contexts:
        client.contexts[model] = 4096
    larger = service.character_budget()
    assert larger.max_input_chars == math.floor(4096 * 4 * 0.90)
    for entry in client.catalog:
        entry["status"]["args"] += ["--ubatch-size", "256"]
    bounded = service.character_budget()
    assert bounded.max_input_chars == math.floor(256 * 4 * 0.90)
    assert first.calibration_id == larger.calibration_id == bounded.calibration_id
    assert len(calibration) == 1


def test_final_calibration_validation_refreshes_loaded_budget(
    make_service, monkeypatch
):
    service, client = make_service()

    def changing(cache_dir, binding, tokenize, check, workers=2, validate=None):
        for model in client.contexts:
            client.contexts[model] = 128
        validate()
        stats = CalibrationStats(2, 80, 20, 4.0, 4.0, 4.0, 4.0)
        return CalibrationProfile("calibration", binding, "c" * 64, {"mixed": stats})

    monkeypatch.setattr(module, "load_or_calibrate", changing)
    assert service.character_budget().max_input_chars == math.floor(128 * 4 * 0.90)


def test_build_change_recalibrates_and_inconsistent_builds_fail(
    make_service, calibration
):
    service, client = make_service()
    first = service.character_budget()
    for model in client.builds:
        client.builds[model] = "test-build-2"
    changed = service.character_budget()
    assert changed.calibration_id != first.calibration_id
    assert changed.representation_id != first.representation_id
    assert len(calibration) == 2
    client.builds["embed_1"] = "another-build"
    status = service.status()
    assert not status.available and "inconsistent_tokenizer" in status.reason
    assert len(calibration) == 2


@pytest.mark.parametrize(
    "name", ["embeddinggemma-2-BF16.gguf", "embeddinggemma-300M.gguf"]
)
def test_embeddinggemma_known_prefix_profile(make_service, calibration, name):
    service, client = make_service(remote(name=name))
    assert service.embed_query("question") == vector(
        "task: search result | query: question"
    )
    assert not client.token_calls and not calibration


def test_unknown_model_requires_explicit_prefix_override(make_service, calibration):
    service, client = make_service(remote(name="unknown-embedding.gguf"))
    status = service.status()
    assert not status.available and "prefixes_required" in status.reason
    assert not calibration and not client.token_calls
    explicit, _ = make_service(
        remote(name="unknown-embedding.gguf"), embedding_prefixes={}
    )
    assert explicit.embed_texts(["body"]) == [vector("body")]


def test_automatic_discovery_ignores_unrelated_llm_and_requires_one_identity_group(
    make_service, calibration
):
    client = remote()
    client.catalog.append(
        {"id": "orchestrator", "status": {"value": "loaded"}, "tags": []}
    )
    service, _ = make_service(client)
    assert service.initialize().available
    assert "orchestrator" not in client.props_calls
    other = deepcopy(client.catalog[0])
    other["id"] = "another-embedding"
    other["tags"] = [tag.replace(CONTENT, "d" * 64) for tag in other["tags"]]
    client.catalog.append(other)
    ambiguous, _ = make_service(client)
    assert not ambiguous.status().available
    assert "ambiguous_model" in ambiguous.status().reason
    selected, _ = make_service(client, model=NAME)
    assert not selected.status().available  # same logical name now resolves two weights
    selected_by_alias, _ = make_service(client, model="embed_0")
    assert selected_by_alias.initialize().available


def test_unverified_and_unloaded_models_never_trigger_props_or_calibration(
    make_service, calibration
):
    service, client = make_service(remote(verified=False))
    assert not service.status().available
    assert not client.props_calls and not client.token_calls and not calibration
    for entry in client.catalog:
        entry["status"]["value"] = "unloaded"
    explicit, _ = make_service(client, model="embed_0", embedding_prefixes={})
    assert not explicit.status().available
    assert not client.props_calls and not client.token_calls


def test_service_pins_model_identity_and_does_not_switch_weights(
    make_service, calibration
):
    service, client = make_service()
    assert service.initialize().available
    for entry in client.catalog:
        entry["tags"] = [tag.replace(CONTENT, "d" * 64) for tag in entry["tags"]]
    with pytest.raises(LLMBackendError):
        service.embed_texts(["source"])
    assert not client.embedding_calls
    assert not calibration and not client.token_calls


def test_calibration_detects_midflight_identity_change_before_profile_publish(
    make_service, calibration, monkeypatch
):
    service, client = make_service()
    tokenize = client.tokenize

    def changing(*args, **kwargs):
        result = tokenize(*args, **kwargs)
        for entry in client.catalog:
            entry["tags"] = [tag.replace(SEMANTIC, "d" * 64) for tag in entry["tags"]]
        return result

    monkeypatch.setattr(client, "tokenize", changing)
    with pytest.raises(LLMBackendError):
        service.character_budget()
    assert not service._profiles and not client.embedding_calls


def test_tokenizer_http_timeout_is_finite_and_capped_to_regular_timeout(
    make_service, calibration, monkeypatch
):
    service, client = make_service(timeout=0.5, calibration_timeout=10)
    observed = []
    original = client.tokenize

    def inspect(*args, **kwargs):
        observed.append(kwargs["timeout"])
        return original(*args, **kwargs)

    monkeypatch.setattr(client, "tokenize", inspect)
    assert service.character_budget().available
    assert observed and all(0 < value <= 0.5 for value in observed)


def test_explicit_timeout_covers_cold_calibration_and_stops_inference(
    make_service, calibration
):
    service, client = make_service()
    client.token_gate.clear()
    with pytest.raises(LLMTimeoutError):
        service.character_budget(timeout=0.05)
    assert client.token_active == 0
    assert not client.embedding_calls and not service._operations


def test_cancellation_and_close_drain_initial_calibration(make_service, calibration):
    service, client = make_service()
    client.token_gate.clear()
    cancel = Event()
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(service.character_budget, cancel=cancel.is_set)
        assert client.token_started.wait(2)
        cancel.set()
        with pytest.raises(LLMCancelledError):
            future.result(3)
    assert client.token_active == 0 and not service._operations
    client.token_started.clear()
    with ThreadPoolExecutor(2) as executor:
        pending = executor.submit(service.character_budget)
        assert client.token_started.wait(2)
        closing = executor.submit(service.close)
        closing.result(3)
        with pytest.raises(LLMBackendError, match="closed"):
            pending.result(3)
    assert not service._operations and client.token_active == 0
    assert client.closed is False  # borrowed transport remains caller-owned


def test_close_during_http_drains_and_does_not_close_borrowed_client(
    make_service, calibration
):
    service, client = make_service()
    service.initialize()
    client.embed_gate.clear()
    with ThreadPoolExecutor(2) as executor:
        pending = executor.submit(service.embed_texts, ["source"])
        assert client.embed_started.wait(2)
        closing = executor.submit(service.close)
        closing.result(3)
        with pytest.raises(LLMBackendError, match="closed"):
            pending.result(3)
    assert not service._operations and client.embed_active == 0
    assert not client.closed
    assert not service.status().available


def test_invalid_inputs_do_not_initialize_or_infer(make_service, calibration):
    service, client = make_service()
    for incoming in ("text", [None], [1], iter(["text"])):
        with pytest.raises(ValueError):
            service.embed_texts(incoming)
    with pytest.raises(ValueError):
        service.character_budget(category="other")
    with pytest.raises(ValueError):
        service.status(input_type="other")
    assert service.embed_texts([]) == []
    assert service._adapter is None and not calibration and not client.token_calls


@pytest.mark.parametrize(
    "changes",
    [
        {"category": "unknown"},
        {"timeout": 0},
        {"calibration_timeout": float("inf")},
        {"calibration_workers": True},
        {"calibration_workers": 33},
        {"batch_size": 0},
        {"request_parallelism": None},
        {"queue_size": -1},
        {"normalize": 1},
        {"trust_env": "yes"},
        {"model": ""},
        {"embedding_prefixes": {"other": "x"}},
        {"base_url": "http://user:secret@runtime.example:8080"},
        {"base_url": "http://runtime.example /"},
    ],
)
def test_config_validation(changes):
    values = {"base_url": "http://runtime.example:8080"}
    values.update(changes)
    with pytest.raises(ValueError):
        AutomaticEmbeddingConfig(**values)


def test_config_secrets_are_not_in_repr():
    config = AutomaticEmbeddingConfig(
        "http://runtime.example:8080", api_key="private-key", api_key_env="SECRET_NAME"
    )
    assert "private-key" not in repr(config) and "SECRET_NAME" not in repr(config)
    assert config.trust_env is False


def test_lazy_async_stream_is_direct_without_calibration_and_reuses_shared_engine(
    calibration,
):
    async def scenario():
        client = remote()
        service = AutomaticEmbeddingService(
            {"base_url": "http://runtime.example:8080", "normalize": False},
            client=client,
        )
        source = WindowSource([["one", "one"], [], ["東京🙂"]])
        stream = service.astream_texts(source)
        assert service._adapter is None and source.pulls == 0
        try:
            async with stream:
                result = [window async for window in stream]
            assert result == [
                [vector("search_document: one")] * 2,
                [],
                [vector("search_document: 東京🙂")],
            ]
            assert not client.token_calls and not calibration
            assert not service._streams and not service._operations
            assert stream._delegate._adapter is service._adapter
        finally:
            await service.aclose()
        assert not client.closed

    run(scenario)


def test_async_cancel_during_readiness_does_not_pull_source_or_leak_work(
    calibration, monkeypatch
):
    async def scenario():
        client = remote()
        started, gate = Event(), Event()
        original = client.props

        def props(*args, **kwargs):
            started.set()
            client._wait(
                gate, cancel=kwargs.get("cancel"), timeout=kwargs.get("timeout")
            )
            return original(*args, **kwargs)

        monkeypatch.setattr(client, "props", props)
        service = AutomaticEmbeddingService(
            {"base_url": "http://runtime.example:8080"}, client=client
        )
        source = WindowSource([["one"]])
        stream = service.astream_texts(source)
        pending = asyncio.create_task(anext(stream))
        try:
            await wait_thread(started)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert source.pulls == 0
            assert not service._streams and not service._operations
            assert not client.token_calls and not calibration
        finally:
            gate.set()
            await service.aclose()

    run(scenario)


def test_aclose_during_direct_stream_drains_transport_and_source(calibration):
    async def scenario():
        client = remote()
        client.embed_gate.clear()
        service = AutomaticEmbeddingService(
            {"base_url": "http://runtime.example:8080"}, client=client
        )
        source = WindowSource([["one"], ["pending"]])
        stream = service.astream_texts(source, close_source=True)
        pending = asyncio.create_task(anext(stream))
        await wait_thread(client.embed_started)
        await service.aclose()
        with pytest.raises((LLMBackendError, LLMCancelledError, StopAsyncIteration)):
            await pending
        assert client.embed_active == 0 and not service._streams
        assert source.close_calls == 1 and not client.closed

    run(scenario)


def test_concurrent_initialize_reuses_one_calibration(make_service, calibration):
    service, client = make_service()
    client.token_gate.clear()
    with ThreadPoolExecutor(3) as executor:
        first = executor.submit(service.character_budget)
        assert client.token_started.wait(2)
        second = executor.submit(service.character_budget, category="code")
        third = executor.submit(service.character_budget, input_type="document")
        eventually(lambda: len(service._operations) == 3)
        client.token_gate.set()
        statuses = [future.result(3) for future in (first, second, third)]
    assert all(status.available for status in statuses)
    assert len(calibration) == 1 and len(client.token_calls) == 2


def test_initialization_waits_boundedly_for_a_draining_topology(
    make_service, calibration, monkeypatch
):
    service, client = make_service()
    service.initialize()
    original = service._adapter._engine_start
    attempts = []

    def unavailable_once(call):
        attempts.append(True)
        return None if len(attempts) == 1 else original(call)

    monkeypatch.setattr(service._adapter, "_engine_start", unavailable_once)
    assert service.initialize().available
    assert len(attempts) == 2
    monkeypatch.setattr(service._adapter, "_engine_start", lambda _: None)
    with pytest.raises(LLMTimeoutError):
        service.initialize(timeout=0.02)
    assert not service._operations and not client.token_calls


def test_all_metadata_http_calls_are_capped_during_cold_calibration(
    make_service, calibration, monkeypatch
):
    service, client = make_service(timeout=0.5, calibration_timeout=10)
    seen = []
    for method in ("health", "models", "props"):
        original = getattr(client, method)

        def capped(*args, _original=original, **kwargs):
            seen.append(kwargs["timeout"])
            return _original(*args, **kwargs)

        monkeypatch.setattr(client, method, capped)
    assert service.character_budget().available
    assert len(seen) > 4 and all(0 < timeout <= 0.5 for timeout in seen)


def test_coordinator_calibration_check_preserves_owner_timeout(
    make_service, monkeypatch
):
    service, client = make_service()

    def expire_owner(cache_dir, binding, tokenize, check, workers=2, validate=None):
        operation = next(iter(service._operations))
        operation.deadline = 0
        check()
        raise AssertionError("expired calibration was allowed")

    monkeypatch.setattr(module, "load_or_calibrate", expire_owner)
    with pytest.raises(LLMTimeoutError, match="automatic_embedding_timeout"):
        service.character_budget()
    assert not service._operations and not client.token_calls
    status = service.status()
    assert status.available and status.max_input_chars is None


def test_coordinator_calibration_check_preserves_private_probe_failure(
    make_service, monkeypatch
):
    service, client = make_service()
    fail = Event()
    private = "PRIVATE /Users/private-user/credentials.txt"

    def probe():
        if fail.is_set():
            raise ValueError(private)
        return False

    def break_probe(cache_dir, binding, tokenize, check, workers=2, validate=None):
        fail.set()
        check()
        raise AssertionError("broken cancellation probe was allowed")

    monkeypatch.setattr(module, "load_or_calibrate", break_probe)
    with pytest.raises(LLMBackendError, match="cancel_probe_failed") as raised:
        service.character_budget(cancel=probe)
    assert private not in str(raised.value)
    assert private not in str(raised.value.__cause__)
    assert not service._operations and not client.token_calls


def test_concurrent_async_start_reuses_one_delegate_and_source(
    calibration, monkeypatch
):
    async def scenario():
        client = remote()
        metadata_started, metadata_gate = Event(), Event()
        props = client.props

        def blocked_props(*args, **kwargs):
            metadata_started.set()
            client._wait(
                metadata_gate,
                cancel=kwargs.get("cancel"),
                timeout=kwargs.get("timeout"),
            )
            return props(*args, **kwargs)

        monkeypatch.setattr(client, "props", blocked_props)
        service = AutomaticEmbeddingService(
            {"base_url": "http://runtime.example:8080", "normalize": False},
            client=client,
        )
        source = WindowSource([["one", "two"], ["last"]])
        stream = service.astream_texts(source)
        created = []
        original = module.LlamaCppEmbeddingAdapter._astream_texts_direct

        def factory(adapter, *args, **kwargs):
            delegate = original(adapter, *args, **kwargs)
            created.append(delegate)
            return delegate

        monkeypatch.setattr(
            module.LlamaCppEmbeddingAdapter, "_astream_texts_direct", factory
        )
        starts = [asyncio.create_task(stream.__aenter__()) for _ in range(2)]
        try:
            await wait_thread(metadata_started)
            metadata_gate.set()
            assert await asyncio.gather(*starts) == [stream, stream]
            result = [window async for window in stream]
            assert result == [
                [vector("search_document: one"), vector("search_document: two")],
                [vector("search_document: last")],
            ]
            assert len(created) == 1 and source.position == 2
            assert sorted(
                text for _, batch in client.embedding_calls for text in batch
            ) == [
                "search_document: last",
                "search_document: one",
                "search_document: two",
            ]
        finally:
            metadata_gate.set()
            await stream.aclose()
            await service.aclose()

    run(scenario)


def test_explicit_alias_pins_selected_semantics_before_digest_admission(
    make_service, calibration
):
    client = remote()
    client.catalog[1]["tags"] = [
        tag.replace(SEMANTIC, "d" * 64) for tag in client.catalog[1]["tags"]
    ]
    ambiguous, _ = make_service(client)
    assert not ambiguous.status().available
    selected, _ = make_service(client, model="embed_0")
    assert selected.initialize().available
    assert selected._adapter._pinned_identity.key == (CONTENT, SEMANTIC)
    assert tuple(selected._adapter._replicas) == ("embed_0",)
    assert selected.embed_texts(["source"]) == [vector("search_document: source")]


def test_cold_ordinary_paths_never_touch_calibration_profiles_or_cache(
    make_service, monkeypatch, tmp_path
):
    cache_path = tmp_path / "not-a-cache-directory"
    cache_path.write_text("ordinary embedding must not open this", encoding="utf-8")
    service, client = make_service(cache_dir=cache_path, category="code")

    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary embedding accessed calibration")

    class UnavailableProfiles:
        get = forbidden

    monkeypatch.setattr(module, "load_or_calibrate", forbidden)
    monkeypatch.setattr(module, "_default_cache_dir", forbidden)
    monkeypatch.setattr(client, "tokenize", forbidden)
    service._profiles = UnavailableProfiles()
    assert service.initialize().available
    assert service.status().max_input_chars is None
    assert service.embed_texts(["one", "one", "東京🙂"]) == [
        vector("search_document: " + text) for text in ("one", "one", "東京🙂")
    ]
    assert service.embed_query("question") == vector("search_query: question")
    assert (
        cache_path.read_text(encoding="utf-8")
        == "ordinary embedding must not open this"
    )


def test_explicit_full_corpus_calibration_then_cached_categories_and_consumer_restart(
    make_service, tmp_path, monkeypatch
):
    service, client = make_service()
    assert service.initialize().available and not client.token_calls
    budget = service.character_budget()
    assert budget.available and budget.max_input_chars > 0
    assert len(client.token_calls) == 2500
    assert all(text.startswith("search_document: ") for text, _ in client.token_calls)
    assert {options["model"] for _, options in client.token_calls} == {
        "embed_0",
        "embed_1",
    }
    code = service.character_budget(category="code")
    assert len(client.token_calls) == 2500
    assert code.calibration_id == budget.calibration_id
    assert service.status().max_input_chars is None
    persisted = json.loads(
        json.dumps(
            {
                "max_input_chars": budget.max_input_chars,
                "representation_id": budget.representation_id,
            }
        )
    )
    service.close()

    cached, cached_client = make_service()
    cached_budget = cached.character_budget(category="documentation")
    assert cached_budget.calibration_id == budget.calibration_id
    assert not cached_client.token_calls
    cached.close()

    # Consumers retain the budget; the next ordinary service requires no cache.
    restarted, restarted_client = make_service(cache_dir=tmp_path / "absent-cache")

    def forbidden(*args, **kwargs):
        raise AssertionError("consumer restart recalibrated")

    monkeypatch.setattr(module, "load_or_calibrate", forbidden)
    monkeypatch.setattr(restarted_client, "tokenize", forbidden)
    current = restarted.initialize()
    assert current.representation_id == persisted["representation_id"]
    body = "consumer source " * 10
    width = persisted["max_input_chars"]
    chunks = [body[index : index + width] for index in range(0, len(body), width)]
    assert restarted.embed_texts(chunks) == [
        vector("search_document: " + chunk) for chunk in chunks
    ]
    assert not (tmp_path / "absent-cache").exists()


def test_learned_character_recommendation_is_never_direct_admission_authority(
    make_service, monkeypatch
):
    service, client = make_service()

    def tiny_budget(cache_dir, binding, tokenize, check, workers=2, validate=None):
        validate()
        stats = CalibrationStats(2, 10, 100, 0.1, 0.1, 0.1, 0.1)
        return CalibrationProfile(
            "tiny-recommendation", binding, "c" * 64, {"mixed": stats}
        )

    monkeypatch.setattr(module, "load_or_calibrate", tiny_budget)
    suggested = service.character_budget()
    assert suggested.max_input_chars == math.floor(512 * 0.1 * 0.90)
    source = "x" * 100
    assert len(source) > suggested.max_input_chars
    assert service.embed_texts([source]) == [vector("search_document: " + source)]
    assert not client.token_calls
    assert service.status().max_input_chars is None


def test_direct_embed_and_status_progress_while_explicit_calibration_is_blocked(
    make_service, calibration
):
    service, client = make_service()
    client.token_gate.clear()
    with ThreadPoolExecutor(3) as executor:
        budget = executor.submit(service.character_budget)
        assert client.token_started.wait(2)
        token_requests = len(client.token_calls)
        try:
            direct = executor.submit(service.embed_texts, ["ordinary"], timeout=0.5)
            available = executor.submit(service.status, timeout=0.5)
            assert direct.result(2) == [vector("search_document: ordinary")]
            assert available.result(2).available
            assert not budget.done()
            assert len(client.token_calls) == token_requests
            assert len(calibration) == 1
        finally:
            client.token_gate.set()
        assert budget.result(3).available
    assert not service._operations and not service._profile_flights


def test_waiting_budget_owner_refreshes_context_after_shared_calibration(
    make_service, calibration
):
    service, client = make_service()
    client.token_gate.clear()
    with ThreadPoolExecutor(2) as executor:
        first = executor.submit(service.character_budget)
        assert client.token_started.wait(2)
        second = executor.submit(service.character_budget, category="code")
        eventually(lambda: service._adapter._pool_users == 2)
        try:
            for model in client.contexts:
                client.contexts[model] = 128
        finally:
            client.token_gate.set()
        assert first.result(3).max_input_chars == math.floor(128 * 4 * 0.90)
        assert second.result(3).max_input_chars == math.floor(128 * 4 * 0.80)
    assert len(calibration) == 1 and len(client.token_calls) == 2
    assert not service._profile_flights and not service._operations


@pytest.mark.parametrize(
    "name",
    [
        "embedding_max_pending_bytes",
        "embedding_max_pending_tokens",
        "embedding_max_request_bytes",
        "embedding_max_response_bytes",
    ],
)
@pytest.mark.parametrize("value", [0, -1, None, True, 1.5, "1024"])
def test_automatic_resource_budgets_reject_nonpositive_or_noninteger(name, value):
    with pytest.raises(ValueError, match=name):
        AutomaticEmbeddingConfig("http://runtime.example:8080", **{name: value})


def test_automatic_resource_defaults_and_overrides_reach_existing_engine(make_service):
    defaults = AutomaticEmbeddingConfig("http://runtime.example:8080")
    assert defaults.embedding_max_pending_bytes == 64 * 1024 * 1024
    assert defaults.embedding_max_pending_tokens == 1024 * 1024
    assert defaults.embedding_max_request_bytes == 8 * 1024 * 1024
    assert defaults.embedding_max_response_bytes == 8 * 1024 * 1024
    configured = {
        "embedding_max_pending_bytes": 2 * 1024 * 1024,
        "embedding_max_pending_tokens": 8192,
        "embedding_max_request_bytes": 4096,
        "embedding_max_response_bytes": 16384,
    }
    service, _ = make_service(**configured)
    assert service.initialize().available
    for name, value in configured.items():
        assert getattr(service._adapter.profile, name) == value
        assert getattr(service._adapter, name) == value
    assert (
        service._adapter._engine.max_bytes == configured["embedding_max_pending_bytes"]
    )
    assert (
        service._adapter._engine.max_tokens
        == configured["embedding_max_pending_tokens"]
    )
    assert (
        service._adapter._engine.max_request_bytes
        == configured["embedding_max_request_bytes"]
    )


@pytest.mark.parametrize("value", [None, False, 0, [], "auto", "REUSE"])
def test_automatic_http_connection_policy_validation(value):
    with pytest.raises(ValueError, match="http_connection_policy"):
        AutomaticEmbeddingConfig(
            "http://runtime.example:8080", http_connection_policy=value
        )


@pytest.mark.parametrize("selected", ["close", "reuse"])
def test_owned_client_receives_selected_http_policy(monkeypatch, selected):
    client = remote()
    constructed = []

    def create(base_url, **kwargs):
        constructed.append((base_url, kwargs))
        return client

    monkeypatch.setattr(module, "LlamaCppClient", create)
    service = AutomaticEmbeddingService(
        {
            "base_url": "http://runtime.example:8080",
            "http_connection_policy": selected,
        }
    )
    try:
        assert service.initialize().available
        assert constructed[0][1]["connection_policy"] == selected
        assert service._adapter.profile.model_config.http_connection_policy == selected
        assert client.closed is False
    finally:
        service.close()
    assert client.closed is True


def test_injected_client_keeps_its_own_policy_and_ownership(make_service):
    client = remote()
    client.connection_policy = "reuse"
    service, _ = make_service(client, http_connection_policy="close")
    assert service.embed_texts(["source"]) == [vector("search_document: source")]
    assert client.connection_policy == "reuse"
    service.close()
    assert client.closed is False


@pytest.mark.parametrize("selector", [None, "embed_0", NAME, "sha256:" + CONTENT])
def test_verified_statusless_single_model_uses_root_props_attestation(
    make_service, selector
):
    client = remote(replicas=1)
    del client.catalog[0]["status"]
    service, _ = make_service(client, model=selector)
    assert service.initialize().available
    assert None in client.props_calls
    assert service.embed_texts(["body"]) == [vector("search_document: body")]
    assert not client.token_calls
    # The shared authority normalizes its observation, never the server fixture.
    assert "status" not in client.catalog[0]


@pytest.mark.parametrize(
    "changes",
    [
        {"role": "router"},
        {"is_sleeping": True},
        {"model_alias": "different"},
        {"model_alias": None},
    ],
)
def test_statusless_router_sleeping_or_wrong_alias_never_selects_model(
    make_service, monkeypatch, changes
):
    client = remote(replicas=1)
    del client.catalog[0]["status"]
    native = client.props

    def props(*args, **kwargs):
        result = native(*args, **kwargs)
        result.update(changes)
        return result

    monkeypatch.setattr(client, "props", props)
    service, _ = make_service(client)
    assert not service.status().available
    assert client.props_calls == [None]
    assert not client.embedding_calls and not client.token_calls


def test_statusless_ambiguous_catalog_never_probes_any_model(make_service):
    client = remote()
    for item in client.catalog:
        del item["status"]
    service, _ = make_service(client)
    assert not service.status().available
    assert (
        not client.props_calls and not client.token_calls and not client.embedding_calls
    )


@pytest.mark.parametrize("malformed", [False, True])
def test_statusless_unverified_or_malformed_identity_never_probes_props(
    make_service, malformed
):
    client = remote(replicas=1, verified=malformed)
    del client.catalog[0]["status"]
    if malformed:
        client.catalog[0]["tags"] = [
            tag
            for tag in client.catalog[0]["tags"]
            if not tag.startswith("noxi-embedding-sha256-v1:")
        ]
    service, _ = make_service(client, model="embed_0")
    assert not service.status().available
    assert not client.props_calls and not client.embedding_calls


@pytest.mark.parametrize("old", [CONTENT, SEMANTIC])
def test_statusless_identity_drift_after_root_props_fails_before_inference(
    make_service, monkeypatch, old
):
    client = remote(replicas=1)
    del client.catalog[0]["status"]
    native = client.props

    def props(*args, **kwargs):
        result = native(*args, **kwargs)
        if kwargs.get("model") is None:
            client.catalog[0]["tags"] = [
                tag.replace(old, "d" * 64) for tag in client.catalog[0]["tags"]
            ]
        return result

    monkeypatch.setattr(client, "props", props)
    service, _ = make_service(client)
    with pytest.raises(LLMBackendError):
        service.embed_texts(["source"])
    assert not client.embedding_calls and not client.token_calls
