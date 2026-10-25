"""Bounded ready-batch pipeline and native execution/HTTP contract regressions."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event

import pytest

import runtime_llama.adapters.embedding as embedding_module
import runtime_llama.embedding_engine as engine_module
from runtime_llama.client import ClientHTTPError
from runtime_llama.errors import LLMBackendError, LLMCancelledError

from test_embedding_contracts import (
    ContractClient,
    configured,
    eventually,
    texts,
    vector,
)


def test_base_case_observes_start_and_dispatch_without_per_quota_catalog():
    client = ContractClient(1)
    model_calls = []
    original = client.models

    def models(**kwargs):
        model_calls.append(kwargs)
        return original(**kwargs)

    client.models = models
    adapter = configured(
        client,
        model="embed_0",
        embedding_replica_discovery=False,
        embedding_request_parallelism=1,
    )
    try:
        assert adapter.embed_query("one") == vector("one")
        assert len(model_calls) == 2
        assert len(client.props_calls) == 2
        assert len(client.token_calls) == 1
        assert len(client.embedding_calls) == 1
    finally:
        adapter.close()


def test_only_public_preparation_seals_and_reusable_counts_are_not_duplicated(
    monkeypatch,
):
    client = ContractClient(1)
    adapter = configured(client)
    sealed = []
    original = embedding_module.issue_preparation

    def issue(*args, **kwargs):
        sealed.append(kwargs["text"])
        return original(*args, **kwargs)

    monkeypatch.setattr(embedding_module, "issue_preparation", issue)
    try:
        assert adapter.embed_texts(["one", "two"]) == [vector("one"), vector("two")]
        assert sealed == []
        prepared = adapter.prepare_query("query")
        assert sealed == ["query"]
        counted = len(client.token_calls)
        assert adapter.embed_prepared([prepared]) == [vector("query")]
        assert len(client.token_calls) == counted
        assert sealed == ["query"]
    finally:
        adapter.close()


def test_control_and_tokenizer_executors_are_reused_across_windows_and_calls(
    monkeypatch,
):
    client = ContractClient(1)
    created = []
    original = engine_module.ThreadPoolExecutor

    def create(*args, **kwargs):
        result = original(*args, **kwargs)
        created.append(result)
        return result

    monkeypatch.setattr(engine_module, "ThreadPoolExecutor", create)
    adapter = configured(client, batch_size=2, token_count_parallelism=2)
    try:
        assert not created
        first = texts(8)
        assert adapter.embed_texts(first) == [vector(text) for text in first]
        owned = tuple(created)
        assert len(owned) == 2
        assert adapter._engine._control in owned
        assert adapter._engine._tokenizer in owned
        coordinator = adapter._engine._thread
        assert coordinator.is_alive()
        incoming = texts(8, start=100)
        assert adapter.embed_texts(incoming) == [vector(text) for text in incoming]
        assert tuple(created) == owned
        assert adapter._engine._thread is coordinator
        assert client.token_peak <= 2
    finally:
        adapter.close()


def test_two_native_slots_do_not_multiply_original_http_leases():
    client = ContractClient(1, context=64, native_batch=1)
    original = client.props
    client.props = lambda **kwargs: {**original(**kwargs), "total_slots": 2}
    incoming = ["first", "second", "third", "fourth"]
    client.counts.update(dict.fromkeys(incoming, 2))
    client.embed_gate.clear()
    adapter = configured(
        client,
        batch_size=4,
        batch_token_limit=4,
        embedding_request_parallelism=2,
        embedding_queue_size=0,
        timeout=5,
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as callers:
            pending = callers.submit(adapter.embed_texts, incoming)
            try:
                eventually(lambda: client.embed_active == 1)
                assert client.alias_peak == {"embed_0": 1}
                assert Counter(client.embedding_calls) == {
                    ("embed_0", tuple(incoming[:2])): 1,
                }
                assert adapter._replicas["embed_0"].total_slots == 2
            finally:
                client.embed_gate.set()
            assert pending.result(timeout=3) == [vector(text) for text in incoming]
        assert Counter(client.embedding_calls) == {
            ("embed_0", tuple(incoming[:2])): 1,
            ("embed_0", tuple(incoming[2:])): 1,
        }
        assert adapter._scheduler.snapshot()["peak_in_flight"] == 1
    finally:
        client.embed_gate.set()
        adapter.close()


def test_native_batch_and_microbatch_do_not_split_aggregate_http_array():
    client = ContractClient(1, context=64, native_batch=2)
    original = client.props
    client.props = lambda **kwargs: {**original(**kwargs), "n_ubatch": 1}
    incoming = ["one", "two", "three", "four"]
    client.counts.update(dict.fromkeys(incoming, 2))
    adapter = configured(client, batch_size=4)
    try:
        assert adapter.embed_texts(incoming) == [vector(text) for text in incoming]
        assert client.embedding_calls == [("embed_0", tuple(incoming))]
    finally:
        adapter.close()


def test_preparation_of_next_window_overlaps_current_http_with_bounded_lookahead():
    client = ContractClient(1)
    original_tokenize = client.tokenize
    original_embeddings = client.embeddings
    first_http_entered, second_prepared, release = Event(), Event(), Event()

    def tokenize(content, **kwargs):
        if content == "second":
            assert first_http_entered.wait(2)
            second_prepared.set()
        return original_tokenize(content, **kwargs)

    def embeddings(input, **kwargs):
        if input == ["first"]:
            first_http_entered.set()
            assert release.wait(3)
        return original_embeddings(input, **kwargs)

    client.tokenize = tokenize
    client.embeddings = embeddings
    adapter = configured(
        client,
        batch_size=1,
        embedding_request_parallelism=1,
        embedding_queue_size=1,
        timeout=5,
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as callers:
            pending = callers.submit(adapter.embed_texts, ["first", "second"])
            try:
                assert first_http_entered.wait(2)
                assert second_prepared.wait(2)
                assert not pending.done()
            finally:
                release.set()
            assert pending.result(timeout=3) == [vector("first"), vector("second")]
    finally:
        release.set()
        adapter.close()


def test_preparation_credit_releases_while_charged_http_rows_bound_lookahead():
    client = ContractClient(1)
    second_tokenized, cancel = Event(), Event()
    original = client.tokenize

    def tokenize(content, **kwargs):
        if client.token_calls:
            second_tokenized.set()
        return original(content, **kwargs)

    client.tokenize = tokenize
    client.embed_gate.clear()
    adapter = configured(
        client,
        batch_size=1,
        embedding_request_parallelism=1,
        embedding_queue_size=0,
        embedding_max_pending_bytes=1024,
        # This test isolates working-record pressure across eight admitted
        # owners; admission outside H has separate coverage below.
        embedding_max_pending_owners=8,
        timeout=5,
    )
    try:
        with ThreadPoolExecutor(max_workers=8) as callers:
            pending = [
                callers.submit(adapter.embed_query, f"text-{i}", cancel=cancel.is_set)
                for i in range(8)
            ]
            try:
                assert client.embed_started.wait(2)
                eventually(lambda: len(adapter._calls) == 8)
                eventually(lambda: adapter._engine.snapshot()["retained_plans"] >= 2)
                pressure = adapter._engine.snapshot()
                # The completed preparation plan released its phase credit;
                # its HTTP rows still consume bytes until real transport drain.
                assert pressure["active_preparation_plans"] <= 1
                assert pressure["retained_plans"] > pressure["active_preparation_plans"]
                assert 0 < pressure["pending_bytes"] <= 1024
                assert pressure["peak_pending_bytes"] <= 1024
                assert not second_tokenized.is_set()
                assert len(client.token_calls) == 1
            finally:
                cancel.set()
            for future in pending:
                with pytest.raises(LLMCancelledError):
                    future.result(timeout=3)
        assert adapter._scheduler.snapshot()["in_flight"] == 0
        assert len(client.embedding_calls) == 1
        assert adapter._engine.snapshot()["pending_bytes"] == 0
    finally:
        cancel.set()
        client.embed_gate.set()
        adapter.close()


@pytest.mark.parametrize("change", ["build", "context", "identity"])
def test_adaptive_children_revalidate_counts_and_identity_after_parent_failure(change):
    client = ContractClient(1)
    incoming = ["first", "second", "third", "fourth"]
    client.counts.update(dict.fromkeys(incoming, 4))

    def fail(model, batch):
        if len(batch) == 4:
            if change == "build":
                client.builds[model] = "changed-tokenizer-build"
            elif change == "context":
                client.contexts[model] = 3
            else:
                client.catalog[0]["tags"][0] = "noxi-content-sha256-v1:" + "c" * 64
            return ClientHTTPError(400, error_type="exceed_context_size_error")
        return None

    client.failure = fail
    adapter = configured(
        client,
        batch_size=4,
        embedding_request_parallelism=1,
        embedding_queue_size=0,
    )
    try:
        with pytest.raises(LLMBackendError):
            adapter.embed_texts(incoming)
        assert client.embedding_calls == [("embed_0", tuple(incoming))]
    finally:
        adapter.close()


def test_queued_batch_refreshes_loaded_context_after_scheduler_wait():
    client = ContractClient(1, context=64)
    client.counts.update({"first": 2, "second": 2})
    observed_contexts = []
    original_props = client.props

    def props(**kwargs):
        result = original_props(**kwargs)
        observed_contexts.append(result["default_generation_settings"]["n_ctx"])
        return result

    client.props = props
    client.embed_gate.clear()
    adapter = configured(
        client,
        batch_size=1,
        embedding_request_parallelism=1,
        embedding_queue_size=1,
        timeout=5,
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(adapter.embed_query, "first")
            assert client.embed_started.wait(2)
            second = callers.submit(adapter.embed_query, "second")
            try:
                eventually(
                    lambda: (
                        adapter._engine.snapshot()["ready_by_replica"].get("embed_0", 0)
                        == 1
                    )
                )
                assert adapter._scheduler.snapshot()["queued"] == 0
                before_release = len(observed_contexts)
                assert observed_contexts and set(observed_contexts) == {64}
                client.contexts["embed_0"] = 1
            finally:
                client.embed_gate.set()
            assert first.result(timeout=3) == vector("first")
            with pytest.raises(LLMBackendError, match="exceeds_context"):
                second.result(timeout=3)
            assert 1 in observed_contexts[before_release:]
        assert client.embedding_calls == [("embed_0", ("first",))]
    finally:
        client.embed_gate.set()
        adapter.close()


def test_successful_adaptive_child_is_not_replayed_after_later_child_failure():
    client = ContractClient(1)
    incoming = ["first", "second", "third", "fourth"]

    def fail(_model, batch):
        if len(batch) == 4:
            return ClientHTTPError(400, error_type="exceed_context_size_error")
        if batch == tuple(incoming[2:]):
            return ClientHTTPError(500)
        return None

    client.failure = fail
    adapter = configured(client, batch_size=4, embedding_request_parallelism=1)
    try:
        with pytest.raises(ClientHTTPError):
            adapter.embed_texts(incoming)
        assert Counter(client.embedding_calls) == {
            ("embed_0", tuple(incoming)): 1,
            ("embed_0", tuple(incoming[:2])): 1,
            ("embed_0", tuple(incoming[2:])): 1,
        }
        assert adapter._scheduler.snapshot()["in_flight"] == 0
    finally:
        adapter.close()


def test_stable_limit_admission_is_linear_across_many_http_batches():
    client = ContractClient(1, context=64)
    incoming = texts(32)
    client.counts.update(dict.fromkeys(incoming, 1))
    adapter = configured(
        client,
        batch_size=32,
        batch_token_limit=1,
        embedding_request_parallelism=1,
        embedding_queue_size=0,
    )
    checks = []
    original = adapter._admit

    def admit(record, observation):
        checks.append(record.text)
        return original(record, observation)

    adapter._admit = admit
    try:
        assert adapter.embed_texts(incoming) == [vector(text) for text in incoming]
        assert set(checks) == set(incoming)
        # Preparation and pre-send are separate observation boundaries. Their
        # linear passes must not become N + (N - 1) + ... suffix validation.
        assert len(incoming) <= len(checks) <= 2 * len(incoming)
        assert len(client.embedding_calls) == len(incoming)
    finally:
        adapter.close()


def test_topology_change_waits_for_active_call_and_does_not_replay_its_outputs():
    client = ContractClient(1)
    client.embed_gate.clear()
    adapter = configured(client, batch_size=1, timeout=5)
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(adapter.embed_query, "first")
            assert client.embed_started.wait(2)
            old = deepcopy(client.catalog[0])
            client.set_aliases(["embed_new"])
            second = callers.submit(adapter.embed_query, "second")
            try:
                eventually(lambda: len(adapter._calls) == 2)
                assert not second.done()
                assert client.embedding_calls == [(old["id"], ("first",))]
            finally:
                client.embed_gate.set()
            assert first.result(timeout=3) == vector("first")
            assert second.result(timeout=3) == vector("second")
        assert Counter(client.embedding_calls) == {
            ("embed_0", ("first",)): 1,
            ("embed_new", ("second",)): 1,
        }
    finally:
        client.embed_gate.set()
        adapter.close()


def test_reduced_context_revalidates_pending_inputs_without_rejecting_completed_ones():
    client = ContractClient(1, context=4)
    client.counts.update({"large": 4, "small": 1})

    def after_first(_model, batch):
        if batch == ("large",):
            client.contexts["embed_0"] = 1
        return None

    client.failure = after_first
    adapter = configured(
        client,
        batch_size=2,
        batch_token_limit=4,
        embedding_request_parallelism=1,
        embedding_queue_size=0,
    )
    try:
        assert adapter.embed_texts(["large", "small"]) == [
            vector("large"),
            vector("small"),
        ]
        assert client.embedding_calls == [
            ("embed_0", ("large",)),
            ("embed_0", ("small",)),
        ]
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "identity_verified,known_build,expected_counts",
    [(True, True, 1), (False, True, 4), (True, False, 4)],
)
def test_duplicate_count_reuse_is_local_to_verified_identity_and_observed_tokenizer(
    identity_verified, known_build, expected_counts
):
    client = ContractClient(1, verified=identity_verified)
    if not known_build:
        client.builds["embed_0"] = None
    adapter = configured(client, model="embed_0", batch_size=4)
    try:
        incoming = ["duplicate"] * 4
        assert adapter.embed_texts(incoming) == [vector("duplicate")] * 4
        assert len(client.token_calls) == expected_counts
        assert client.embedding_calls == [("embed_0", tuple(incoming))]
    finally:
        adapter.close()


def test_close_drains_pending_preparation_and_all_global_admission_waiters():
    client = ContractClient(1)
    client.token_gate.clear()
    adapter = configured(
        client,
        batch_size=1,
        embedding_request_parallelism=1,
        embedding_queue_size=0,
        timeout=5,
    )
    try:
        with ThreadPoolExecutor(max_workers=6) as callers:
            pending = [
                callers.submit(adapter.embed_query, f"waiting-{i}") for i in range(6)
            ]
            assert client.token_started.wait(2)
            eventually(lambda: len(adapter._calls) == 1)
            eventually(lambda: adapter._intake.snapshot()["owner_waiters"] == 5)
            assert adapter._intake.snapshot()["owners"] == 1
            adapter.close()
            for future in pending:
                with pytest.raises((LLMBackendError, LLMCancelledError)):
                    future.result(timeout=2)
        assert not adapter._calls
        assert client.token_active == client.embed_active == 0
        assert adapter._scheduler.snapshot()["in_flight"] == 0
        assert not adapter._engine._thread.is_alive()
        assert all(not t.is_alive() for t in adapter._engine._control._threads)
        assert all(not t.is_alive() for t in adapter._engine._tokenizer._threads)
        assert all(not t.is_alive() for t in adapter._scheduler._executor._threads)
        assert not client.embedding_calls
    finally:
        client.token_gate.set()
        adapter.close()


def test_adaptive_first_child_does_not_precede_already_known_oversized_sibling():
    client = ContractClient(1, context=8)
    incoming = ["small-first", "small-second", "large-third", "large-fourth"]
    client.counts.update(dict(zip(incoming, [1, 1, 4, 4])))

    def fail(model, batch):
        if len(batch) == 4:
            client.contexts[model] = 2
            return ClientHTTPError(400, error_type="exceed_context_size_error")
        return None

    client.failure = fail
    adapter = configured(
        client,
        batch_size=4,
        embedding_request_parallelism=1,
        embedding_queue_size=0,
    )
    try:
        with pytest.raises(LLMBackendError, match="exceeds_context"):
            adapter.embed_texts(incoming)
        assert client.embedding_calls == [("embed_0", tuple(incoming))]
    finally:
        adapter.close()
