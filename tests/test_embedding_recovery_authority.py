"""Recognized custom-client recovery must use fresh per-record authority."""

from dataclasses import replace
from collections import Counter
from threading import Event
import time

import pytest

from runtime_llama.adapters.embedding import (
    LlamaCppEmbeddingAdapter,
    _Call,
    _CountedInput,
    _Observation,
)
from runtime_llama.client import ClientHTTPError
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError

from test_embedding import Client, payload, profile
from test_embedding_async_stream import assert_drained, run, wait_thread
from test_embedding_contracts import vector
from test_embedding_shared_lifecycle import SharedScenario


def counted(text, count, limit=2048, **changes):
    return replace(
        _CountedInput(
            text=text,
            input_type="raw",
            token_count=count,
            context_token_limit=limit,
            batch_token_limit=None,
            max_input_tokens=limit,
            identity_verified=False,
            content_sha256=None,
            embedding_sha256=None,
            _tokenizer_build="build-1",
            _policy="test",
            _token_ids=tuple(range(count)),
        ),
        **changes,
    )


def rejection():
    return ClientHTTPError(400, error_type="exceed_context_size_error")


def scenario(monkeypatch, records, observations, *, responses=None):
    client = Client(responses=[rejection()] if responses is None else responses)
    client.supports_embedding_token_ids = False
    adapter = LlamaCppEmbeddingAdapter(profile(), client=client)
    initial = _Observation(2048, 2048, None, "build-1", vocab_size=2048)
    observed = iter(observations)
    monkeypatch.setattr(adapter, "_observe_replica", lambda **_kwargs: next(observed))
    calls = [_Call(adapter, None) for _ in records]
    return adapter, client, initial, calls


def execute(adapter, records, calls, observation):
    return adapter._engine_run_shared(
        "embed-model", records, calls, adapter._engine_batch_call(calls), observation
    )


def test_context_reduction_hidden_by_another_owners_historical_limit(monkeypatch):
    initial = _Observation(2048, 2048, None, "build-1", vocab_size=2048)
    current = replace(initial, context_length=100, context_token_limit=100)
    records = [counted("short", 1, 2), counted("long", 400)]
    adapter, client, initial, calls = scenario(monkeypatch, records, [current])
    try:
        assert execute(adapter, records, calls, initial) == [(0, [5.0, 1.0])]
        assert calls[0].failure is None
        assert isinstance(calls[1].failure, LLMBackendError)
        assert "input_exceeds_context" in str(calls[1].failure)
        assert [call[0] for call in client.embedding_calls] == [
            ["short", "long"],
            ["short"],
        ]
    finally:
        adapter.close()


def test_vocabulary_change_revalidates_unsent_records_even_with_same_limits(
    monkeypatch,
):
    current = _Observation(2048, 2048, None, "build-1", vocab_size=100)
    records = [counted("short", 1, 2), counted("long", 400)]
    adapter, client, initial, calls = scenario(monkeypatch, records, [current])
    try:
        assert execute(adapter, records, calls, initial) == [(0, [5.0, 1.0])]
        assert calls[0].failure is None
        assert isinstance(calls[1].failure, LLMBackendError)
        assert "invalid_tokenization" in str(calls[1].failure)
        assert [call[0] for call in client.embedding_calls] == [
            ["short", "long"],
            ["short"],
        ]
    finally:
        adapter.close()


def test_identity_change_cannot_reuse_old_recovery_authority(monkeypatch):
    old_identity, new_identity = ("a" * 64, "b" * 64), ("c" * 64, "d" * 64)
    current = _Observation(
        2048, 2048, None, "build-1", identity_key=new_identity, vocab_size=2048
    )
    records = [
        counted("unknown", 1),
        counted(
            "verified",
            1,
            identity_verified=True,
            content_sha256=old_identity[0],
            embedding_sha256=old_identity[1],
        ),
    ]
    adapter, client, initial, calls = scenario(monkeypatch, records, [current])
    initial = replace(initial, identity_key=old_identity)
    try:
        assert execute(adapter, records, calls, initial) == [(0, [7.0, 1.0])]
        assert calls[0].failure is None
        assert "content_identity_changed" in str(calls[1].failure)
        assert [call[0] for call in client.embedding_calls] == [
            ["unknown", "verified"],
            ["unknown"],
        ]
    finally:
        adapter.close()


def test_new_tokenizer_build_rejects_all_pending_without_child_posts(monkeypatch):
    current = _Observation(2048, 2048, None, "build-2", vocab_size=2048)
    records = [counted("first", 1), counted("second", 1)]
    adapter, client, initial, calls = scenario(monkeypatch, records, [current])
    try:
        with pytest.raises(LLMBackendError, match="tokenizer_policy_changed"):
            execute(adapter, records, calls, initial)
        assert all("tokenizer_policy_changed" in str(call.failure) for call in calls)
        assert len(client.embedding_calls) == 1
    finally:
        adapter.close()


@pytest.mark.parametrize("retired", ["failed", "cancelled", "timeout"])
def test_retired_owner_is_not_revalidated_or_sent_in_recovery(monkeypatch, retired):
    current = _Observation(100, 100, None, "build-1", vocab_size=100)
    records = [counted("withdrawn", 400), counted("survivor", 1)]
    adapter, client, initial, calls = scenario(monkeypatch, records, [current])
    first, second = calls
    original = client.embeddings
    expected = {
        "failed": LLMBackendError,
        "cancelled": LLMCancelledError,
        "timeout": LLMTimeoutError,
    }[retired]

    def request(input, **kwargs):
        if not client.embedding_calls:
            if retired == "failed":
                first.fail(LLMBackendError("owner_already_failed"))
            elif retired == "cancelled":
                first.stop.set()
            else:
                first.deadline = time.monotonic() - 1
        return original(input, **kwargs)

    monkeypatch.setattr(client, "embeddings", request)
    try:
        assert execute(adapter, records, calls, initial) == [(1, [8.0, 1.0])]
        assert isinstance(first.failure, expected)
        assert second.failure is None
        assert [call[0] for call in client.embedding_calls] == [
            ["withdrawn", "survivor"],
            ["survivor"],
        ]
        assert all(0 < call[2]["timeout"] <= 7 for call in client.embedding_calls)
    finally:
        adapter.close()


def test_aggregate_request_budget_does_not_become_another_records_input_limit(
    monkeypatch,
):
    current = _Observation(2048, 2048, 5, "build-1", vocab_size=2048)
    records = [counted("one", 1, 2), counted("four", 4)]
    adapter, client, initial, calls = scenario(monkeypatch, records, [current, current])
    try:
        assert execute(adapter, records, calls, initial) == [
            (0, [3.0, 1.0]),
            (1, [4.0, 1.0]),
        ]
        assert all(call.failure is None for call in calls)
        assert [call[0] for call in client.embedding_calls] == [
            ["one", "four"],
            ["one"],
            ["four"],
        ]
    finally:
        adapter.close()


def test_historical_http_budget_is_scoped_to_each_child_and_always_progresses(
    monkeypatch,
):
    initial = _Observation(2048, 2048, 2048, "build-1", vocab_size=2048)
    records = [
        counted("large", 400, batch_token_limit=1000),
        counted("small", 1, batch_token_limit=2),
    ]
    adapter, client, _, calls = scenario(
        monkeypatch, records, [initial, initial], responses=[]
    )
    try:
        assert execute(adapter, records, calls, initial) == [
            (0, [5.0, 1.0]),
            (1, [5.0, 1.0]),
        ]
        assert all(call.failure is None for call in calls)
        assert [call[0] for call in client.embedding_calls] == [["large"], ["small"]]
    finally:
        adapter.close()


@pytest.mark.parametrize("second_child_failure", [False, True])
def test_successful_child_is_never_replayed(monkeypatch, second_child_failure):
    initial = _Observation(2048, 2048, None, "build-1", vocab_size=2048)
    records = [counted(text, 1) for text in ["a", "b", "c", "d"]]
    responses = [
        rejection(),
        payload([[1, 0], [2, 0]]),
        ClientHTTPError(500) if second_child_failure else rejection(),
    ]
    adapter, client, _, calls = scenario(
        monkeypatch, records, [initial] * 4, responses=responses
    )
    try:
        if second_child_failure:
            with pytest.raises(ClientHTTPError) as failure:
                execute(adapter, records, calls, initial)
            assert failure.value.status_code == 500
            assert [call[0] for call in client.embedding_calls] == [
                ["a", "b", "c", "d"],
                ["a", "b"],
                ["c", "d"],
            ]
        else:
            assert execute(adapter, records, calls, initial) == [
                (0, [1.0, 0.0]),
                (1, [2.0, 0.0]),
                (2, [1.0, 1.0]),
                (3, [1.0, 1.0]),
            ]
            assert [call[0] for call in client.embedding_calls] == [
                ["a", "b", "c", "d"],
                ["a", "b"],
                ["c", "d"],
                ["c"],
                ["d"],
            ]
    finally:
        adapter.close()


def test_native_context_failure_preserves_nonreplay_policy(monkeypatch):
    records = [counted("a", 1), counted("b", 1)]
    adapter, client, initial, calls = scenario(monkeypatch, records, [])
    client.supports_embedding_token_ids = True
    original_error = client.responses[0]
    try:
        with pytest.raises(ClientHTTPError) as failure:
            execute(adapter, records, calls, initial)
        assert failure.value is original_error
        assert len(client.embedding_calls) == 1
    finally:
        adapter.close()


def test_shared_recovery_preserves_survivor_scatter_and_actual_scheduler_drain(
    monkeypatch,
):
    async def scenario():
        shared = SharedScenario()
        client = shared.client
        client.supports_embedding_token_ids = False
        client.counts.update({"a-0": 400, "a-1": 400, "b-0": 1, "b-1": 1})
        original = client.embeddings
        rejected = False
        child_started, release_child = Event(), Event()

        def request(input, **kwargs):
            nonlocal rejected
            shared_request = any(text.startswith(("a-", "b-")) for text in input)
            if rejected and shared_request:
                child_started.set()
                assert release_child.wait(3), "recovery child was not released"
            response = original(input, **kwargs)
            if shared_request and not rejected:
                rejected = True
                client.contexts["embed_0"] = 100
                raise rejection()
            return response

        monkeypatch.setattr(client, "embeddings", request)
        try:
            async with shared:
                client.shared_gate.set()
                await wait_thread(child_started)
                assert isinstance(shared.calls["a"].failure, LLMBackendError)
                assert not shared.tasks[0].done()
                assert not shared.tasks[1].done()
                assert shared.adapter._scheduler.snapshot()["in_flight"] == 1
                release_child.set()
                with pytest.raises(LLMBackendError, match="input_exceeds_context"):
                    await shared.tasks[0]
                assert await shared.tasks[1] == [vector("b-0"), vector("b-1")]
                assert shared.calls["b"].failure is None
                assert not client.transport_cancelled.is_set()
                assert Counter(
                    text for batch in client.shared_posts[1:] for text in batch
                ) == Counter(["b-0", "b-1"])
                await shared.streams[1].aclose()
                assert_drained(shared.adapter, client)
        finally:
            release_child.set()
            await shared.close()

    run(scenario)
