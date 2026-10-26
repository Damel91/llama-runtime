"""No implicit capacity policies; explicit policies retain their actual values."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from runtime_llama.automatic_embedding import (
    AutomaticEmbeddingConfig,
    AutomaticEmbeddingService,
)
from runtime_llama.client import ClientHTTPError, LlamaCppClient, _sse
from runtime_llama.configuration import AdapterProfile, EmbeddingPrefixes
from runtime_llama.errors import LLMBackendError
from runtime_llama.model_discovery import discover_models
from runtime_llama.server_observation import CoherentServerSnapshot
from test_client import Response, Session, wait_for
from test_model_discovery import fixture_entry, fixture_server
from test_model_discovery_transport import endpoint as endpoint, send
from test_server_observation import replica, snapshot


CAPACITIES = (
    "embedding_max_pending_bytes",
    "embedding_max_pending_tokens",
    "embedding_max_request_bytes",
    "embedding_max_response_bytes",
    "embedding_max_handoff_bytes",
    "embedding_isolation_max_requests",
    "embedding_max_pending_owners",
)


def test_absent_profile_and_service_policies_are_not_implicit_budgets():
    profile = AdapterProfile.from_config(
        "embedding", {"adapter_type": "embedding", "model": "chosen"}
    )
    config = AutomaticEmbeddingConfig("http://chosen.example")
    for value in (profile, config):
        assert all(getattr(value, name) is None for name in CAPACITIES)
        assert value.batch_size is None
        assert value.embedding_capacity_limits is False
        assert value.embedding_preparation_cache_enabled is False
    assert profile.embedding_queue_size is None and config.queue_size is None
    assert profile.embedding_preparation_cache_entries is None
    assert profile.embedding_preparation_cache_bytes is None
    assert config.calibration_workers is None


def test_opt_in_values_are_independent_and_have_no_hidden_upper_ceiling():
    configured = {name: 10**12 for name in CAPACITIES}
    config = AutomaticEmbeddingConfig(
        "http://chosen.example",
        embedding_capacity_limits=True,
        embedding_preparation_cache_enabled=True,
        calibration_workers=100,
        batch_size=10000,
        queue_size=10000,
        **configured,
    )
    assert all(getattr(config, name) == 10**12 for name in CAPACITIES)
    assert config.calibration_workers == 100 and config.batch_size == 10000


def test_direct_binding_does_not_require_unused_tokenizer_or_microbatch_ceiling():
    with AutomaticEmbeddingService({"base_url": "http://chosen.example"}) as service:
        service._identity = SimpleNamespace(
            key="identity", content_sha256="a" * 64, embedding_sha256="b" * 64
        )
        service._prefixes = EmbeddingPrefixes()
        observation = SimpleNamespace(
            identity_key="identity",
            tokenizer_build=None,
            vocab_size=None,
            context_token_limit=8192,
            native_microbatch_tokens=512,
        )
        context = SimpleNamespace(observations={"replica": observation})
        adapter = SimpleNamespace(_policy=lambda _task: "policy")
        binding, context_limit = service._binding(adapter, context, "document")
        assert context_limit == 8192 and binding["tokenizer_build"] is None
        with pytest.raises(LLMBackendError, match="inconsistent_tokenizer"):
            service._binding(adapter, context, "document", require_tokenizer=True)


def test_discovery_accepts_catalog_and_fields_beyond_previous_caps(endpoint):
    entries = [fixture_entry(f"model-{i}") for i in range(4200)]
    entries[0].update(
        id="a" * 2000,
        aliases=[f"alias-{i}" for i in range(70)],
        tags=[f"annotation-{i}" for i in range(140)],
        meta={"n_ctx": 2**40},
        ignored="x" * (1024 * 1024),
    )
    base_url, calls = fixture_server(endpoint, entries)
    observed = discover_models(base_url)
    assert len(observed.models) == 4200 and len(calls) == 1
    assert observed.models[0].loaded_context.value == 2**40
    assert len(observed.models[0].aliases) == 71
    # An explicitly selected limit is still enforced without reducing it to
    # the old 4096 maximum accepted configuration value.
    with pytest.raises(LLMBackendError, match="model_limit_exceeded"):
        discover_models(base_url, max_models=4199, max_response_bytes=32 * 1024 * 1024)


def test_optional_snapshot_does_not_impose_model_count_or_identifier_ceiling():
    replicas = tuple(replica(f"model-{i}", f"replica-{i}") for i in range(4100))
    observed = replace(snapshot(), replicas=replicas, valid_for_seconds=7200)
    assert isinstance(observed, CoherentServerSnapshot)
    assert len(observed.replicas) == 4100
    assert replace(observed.replicas[0], alias="a" * 2048, slots=2**40).slots == 2**40


def test_sse_frame_limit_is_only_an_explicit_policy():
    value = "x" * (8 * 1024 * 1024 + 1)
    chunks = [b"data: " + json.dumps({"text": value}).encode() + b"\n\n"]
    assert list(_sse(iter(chunks)))[0]["text"] == value
    with pytest.raises(LLMBackendError, match="frame_too_large"):
        list(_sse(iter(chunks), max_frame_chars=1024))


@pytest.mark.parametrize("bound", [None, 200000])
def test_error_codes_survive_bodies_above_old_hidden_error_limit(endpoint, bound):
    body = json.dumps(
        {"error": {"code": "context_length_exceeded", "message": "x" * 100000}}
    ).encode()
    base_url, _ = endpoint(lambda handler: send(handler, body, status=400))
    with LlamaCppClient(base_url, timeout=1, trust_env=False) as api:
        with pytest.raises(ClientHTTPError) as caught:
            api.health(max_response_bytes=bound, cancel_wait=True)
        assert caught.value.error_code == "context_length_exceeded"
        assert not api._streams


def test_closed_sse_handle_releases_all_buffered_payloads_and_transport():
    chunks = [b'data: {"index": ' + str(i).encode() + b"}\n\n" for i in range(100)]
    response = Response(chunks=chunks)
    api = LlamaCppClient("http://chosen.example", session=Session(response))
    stream = api.complete("source", stream=True, cancel_wait=True)
    wait_for(lambda: stream._drained.is_set())
    assert stream._queue.qsize() == 101 and stream._queue.maxsize == 0
    stream.close()
    assert stream._queue.empty()
    assert stream._body is stream._response is stream._session is None
    assert response.closed.is_set() and not api._streams
    api.close()


@pytest.mark.parametrize("option", ["max_sse_frame_chars", "max_sse_pending_events"])
def test_explicit_sse_policy_validation(option):
    with pytest.raises(ValueError):
        LlamaCppClient("http://chosen.example", **{option: 0})
    with LlamaCppClient("http://chosen.example", **{option: 10000}) as api:
        assert getattr(api, option) == 10000
