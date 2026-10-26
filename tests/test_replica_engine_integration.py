"""End-to-end deterministic integration of replica-width scheduling.

The fake server advertises four native slots per destination. Those slots
cannot multiply original HTTP leases or reduce the verified quota width.
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event

import pytest

from runtime_llama.adapters.embedding import LlamaCppEmbeddingAdapter
from runtime_llama.automatic_embedding import AutomaticEmbeddingService
from runtime_llama.configuration import AdapterProfile
from runtime_llama.errors import LLMBackendError

from test_embedding_contracts import ContractClient, eventually, texts, vector


class FourSlotClient(ContractClient):
    """Loaded, verified copies with independent HTTP execution gates."""

    def __init__(self, replicas=4):
        super().__init__(replicas, context=2048, native_batch=2048)
        self.growth_observed = Event()
        self.initial_replicas = replicas

    def models(self, **kwargs):
        with self.lock:
            catalog = {"data": deepcopy(self.catalog)}
        if len(catalog["data"]) > self.initial_replicas:
            self.growth_observed.set()
        return catalog

    def props(self, model=None, **kwargs):
        return {
            **super().props(model=model, **kwargs),
            "total_slots": 4,
            "n_batch": 2048,
            "n_ubatch": 2048,
        }

    def grow_to(self, count):
        with self.lock:
            self.set_aliases([f"embed_{index}" for index in range(count)])


def exact_adapter(client, **changes):
    config = {
        "adapter_type": "embedding",
        "model": "embed_0",
        "embedding_replica_discovery": True,
        "batch_size": 256,
        "dimension": 2,
        "normalize": False,
        "timeout": 5,
        "token_count_parallelism": 4,
        "embedding_queue_size": 16,
        **changes,
    }
    return LlamaCppEmbeddingAdapter(
        AdapterProfile.from_config("replica-engine-integration", config), client=client
    )


def assert_exactly_once(client, inputs):
    assert Counter(
        text for _, batch in client.embedding_calls for text in batch
    ) == Counter(inputs)
    assert all(peak == 1 for peak in client.alias_peak.values())


@pytest.mark.parametrize("facade", ["exact", "automatic"])
def test_absent_guard_uses_all_four_verified_copies_with_64_text_quotas(facade):
    client = FourSlotClient()
    incoming = texts(256)
    if facade == "exact":
        service = exact_adapter(client)
        adapter = service
        assert adapter.profile.embedding_request_parallelism is None
    else:
        service = AutomaticEmbeddingService(
            {
                "base_url": "http://runtime.example:8080",
                "model": "embed_0",
                "embedding_prefixes": {"query": "", "document": ""},
                "batch_size": 256,
                "normalize": False,
                "timeout": 5,
            },
            client=client,
        )
        assert service.config.request_parallelism is None
        # Discover vector size before the measurement; unknown-size bootstrap
        # is a separate safety path and is not a quota fragmentation defect.
        assert service.embed_query("prime") == vector("prime")
        adapter = service._adapter
        client.embedding_calls.clear()
        client.embed_peak = 0
        client.alias_peak.clear()
    client.embed_gate.clear()
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            pending = caller.submit(service.embed_texts, incoming)
            try:
                eventually(lambda: client.embed_active == 4)
                status = adapter._scheduler.snapshot()
                assert status["requested_parallelism"] is None
                assert status["effective_parallelism"] == 4
                assert status["replica_count"] == 4
                assert adapter._engine.parallelism == 4
                assert adapter._scheduler.text_cap(256) == 64
                assert (
                    sorted(len(batch) for _, batch in client.embedding_calls)
                    == [64] * 4
                )
                assert all(
                    entry["max_in_flight"] == 1 for entry in status["replicas"].values()
                )
            finally:
                client.embed_gate.set()
            assert pending.result(timeout=3) == [vector(text) for text in incoming]
        assert_exactly_once(client, incoming)
        assert client.embed_peak == 4
        if facade == "automatic":
            assert not client.token_calls
    finally:
        client.embed_gate.set()
        service.close()


@pytest.mark.parametrize("guard", [1, 2])
def test_positive_guard_limits_http_but_keeps_four_destination_quota_width(guard):
    client = FourSlotClient()
    incoming = texts(256)
    adapter = exact_adapter(client, embedding_request_parallelism=guard)
    client.embed_gate.clear()
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            pending = caller.submit(adapter.embed_texts, incoming)
            try:
                eventually(lambda: client.embed_active == guard)
                assert adapter._scheduler.parallelism == guard
                assert adapter._scheduler.text_cap(256) == 64
            finally:
                client.embed_gate.set()
            assert pending.result(timeout=3) == [vector(text) for text in incoming]
        assert sorted(len(batch) for _, batch in client.embedding_calls) == [64] * 4
        assert set(model for model, _ in client.embedding_calls) == {
            f"embed_{index}" for index in range(4)
        }
        assert client.embed_peak == guard
        assert_exactly_once(client, incoming)
    finally:
        client.embed_gate.set()
        adapter.close()


def test_guard_one_observes_and_dispatches_ready_peer_before_first_alias_exhaustion():
    client = FourSlotClient(2)
    incoming = texts(256)
    adapter = exact_adapter(client, embedding_request_parallelism=1, batch_size=64)
    client.embed_gate.clear()
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            pending = caller.submit(adapter.embed_texts, incoming)
            try:
                assert client.embed_started.wait(2)
                eventually(
                    lambda: all(
                        adapter._engine.snapshot()["ready_by_replica"].get(model, 0)
                        >= 64
                        for model in ("embed_0", "embed_1")
                    )
                )
            finally:
                client.embed_gate.set()
            assert pending.result(timeout=3) == [vector(text) for text in incoming]
        dispatched = [model for model, _ in client.embedding_calls]
        assert len(dispatched) == 8
        assert all(
            left != right for left, right in zip(dispatched[:3], dispatched[1:4])
        )
        assert Counter(dispatched) == {"embed_0": 4, "embed_1": 4}
        assert client.embed_peak == 1
        assert_exactly_once(client, incoming)
    finally:
        client.embed_gate.set()
        adapter.close()


def test_topology_growth_preserves_old_transport_then_replaces_scheduler_after_drain():
    client = FourSlotClient(1)
    adapter = exact_adapter(client)
    client.embed_gate.clear()
    incoming = texts(256)
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(adapter.embed_query, "first")
            assert client.embed_started.wait(2)
            old = adapter._scheduler
            assert old.parallelism == 1
            client.grow_to(4)
            following = callers.submit(adapter.embed_texts, incoming)
            try:
                assert client.growth_observed.wait(2)
                assert adapter._scheduler is old
                assert not first.done() and not following.done()
            finally:
                client.embed_gate.set()
            assert first.result(timeout=3) == vector("first")
            assert following.result(timeout=3) == [vector(text) for text in incoming]
        assert old._closed and old.snapshot()["in_flight"] == 0
        assert adapter._scheduler is not old
        assert adapter._scheduler.parallelism == adapter._engine.parallelism == 4
        assert adapter._scheduler.text_cap(256) == 64
        assert_exactly_once(client, ["first", *incoming])
    finally:
        client.embed_gate.set()
        adapter.close()


def test_explicit_synonyms_deduplicate_without_discovering_unconfigured_copies():
    client = FourSlotClient(3)
    client.catalog[0]["aliases"] = ["first-copy"]
    adapter = exact_adapter(
        client,
        embedding_replica_discovery=False,
        embedding_replicas=[
            {"model": "embed_0", "max_in_flight": 4},
            {"model": "first-copy", "max_in_flight": 4},
            {"model": "embed_1", "max_in_flight": 4},
        ],
    )
    incoming = texts(256)
    try:
        assert adapter.embed_texts(incoming) == [vector(text) for text in incoming]
        assert adapter._scheduler.parallelism == 2
        assert adapter._scheduler.text_cap(256) == 128
        assert sorted(len(batch) for _, batch in client.embedding_calls) == [128, 128]
        assert set(model for model, _ in client.embedding_calls) == {
            "embed_0",
            "embed_1",
        }
        assert_exactly_once(client, incoming)
    finally:
        adapter.close()


@pytest.mark.parametrize("unverified", [False, True])
def test_explicit_multi_rejects_unverified_or_different_weights_before_post(unverified):
    client = FourSlotClient(2)
    client.catalog[1]["tags"] = (
        []
        if unverified
        else [
            "noxi-content-sha256-v1:" + "c" * 64,
            "noxi-embedding-sha256-v1:" + "b" * 64,
        ]
    )
    adapter = exact_adapter(
        client,
        embedding_replica_discovery=False,
        embedding_replicas=[{"model": "embed_0"}, {"model": "embed_1"}],
    )
    try:
        with pytest.raises(LLMBackendError, match="ambiguous_model_identity"):
            adapter.embed_texts(["first", "second"])
        assert not client.embedding_calls and not client.token_calls
    finally:
        adapter.close()
