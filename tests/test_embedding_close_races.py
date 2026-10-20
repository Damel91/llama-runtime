"""Deterministic executor-close races and busy-replica admission regressions."""

from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event

import pytest

from runtime_llama.errors import LLMBackendError

from test_embedding_contracts import ContractClient, configured, vector


@pytest.mark.parametrize("boundary", ["closed_submit", "cancelled_future"])
@pytest.mark.parametrize("prior_failure", [False, True])
def test_close_at_preparation_submission_preserves_typed_first_failure(
    monkeypatch, boundary, prior_failure
):
    client = ContractClient(1)
    adapter = configured(client, timeout=5)
    executor = adapter._preparation_executor
    original_submit, original_shutdown = executor.submit, executor.shutdown
    submission_entered, executor_shutdown = Event(), Event()
    cancelled = Future()
    expected = LLMBackendError("test_existing_backend_failure")

    def submit(run):
        if prior_failure:
            # A concurrent child can fail before close cancels another future.
            with adapter._calls_lock:
                call = next(iter(adapter._calls))
            call.fail(expected)
        submission_entered.set()
        assert executor_shutdown.wait(3), "close did not shut down preparation"
        if boundary == "closed_submit":
            # Exercise the real executor's post-shutdown RuntimeError.
            return original_submit(run)
        assert cancelled.cancelled()
        return cancelled

    def shutdown(*args, **kwargs):
        original_shutdown(*args, **kwargs)
        # Model a queued executor future cancelled by cancel_futures=True,
        # at the boundary after check() and before the coordinator reads it.
        cancelled.cancel()
        executor_shutdown.set()

    monkeypatch.setattr(executor, "submit", submit)
    monkeypatch.setattr(executor, "shutdown", shutdown)
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            pending = callers.submit(adapter.embed_query, "one")
            assert submission_entered.wait(2)
            closing = callers.submit(adapter.close)
            with pytest.raises(LLMBackendError) as failure:
                pending.result(timeout=4)
            if prior_failure:
                assert failure.value is expected
            else:
                assert str(failure.value) == "llamacpp_embedding_adapter_closed"
            closing.result(timeout=4)
        assert not adapter._calls
        assert adapter._scheduler.snapshot()["in_flight"] == 0
    finally:
        executor_shutdown.set()
        adapter.close()


def test_busy_replica_does_not_block_collection_of_other_prepared_quota(monkeypatch):
    client = ContractClient(2)
    incoming = ["a0", "a1", "b0", "b1"]
    client.counts.update(dict.fromkeys(incoming, 2))
    adapter = configured(
        client,
        batch_size=4,
        batch_token_limit=2,
        embedding_request_parallelism=2,
        embedding_queue_size=0,
        timeout=5,
    )
    a_first_entered, b_second_entered, release_a = Event(), Event(), Event()
    prepare_job, embeddings = adapter._prepare_job, client.embeddings

    def prepare(inputs, model, offset, call, conservative):
        if model == "embed_1":
            # Ensure B is still in preparations when A becomes ready, so the
            # coordinator must collect it while A's next request cannot start.
            assert a_first_entered.wait(3)
        return prepare_job(inputs, model, offset, call, conservative)

    def request(input, *, model, **kwargs):
        result = embeddings(input, model=model, **kwargs)
        if model == "embed_0" and input == ["a0"]:
            a_first_entered.set()
            client._wait(
                release_a,
                cancel=kwargs.get("cancel"),
                timeout=kwargs.get("timeout"),
            )
        if model == "embed_1" and input == ["b1"]:
            b_second_entered.set()
        return result

    monkeypatch.setattr(adapter, "_prepare_job", prepare)
    monkeypatch.setattr(client, "embeddings", request)
    try:
        with ThreadPoolExecutor(max_workers=1) as callers:
            pending = callers.submit(adapter.embed_texts, incoming)
            try:
                assert a_first_entered.wait(2)
                assert b_second_entered.wait(2)
                assert not pending.done()
                assert not release_a.is_set()
            finally:
                release_a.set()
            assert pending.result(timeout=3) == [vector(text) for text in incoming]
        assert Counter(client.embedding_calls) == Counter(
            (f"embed_{index // 2}", (text,)) for index, text in enumerate(incoming)
        )
        assert all(value == 1 for value in client.alias_peak.values())
        assert adapter._scheduler.snapshot()["peak_in_flight"] == 2
    finally:
        release_a.set()
        adapter.close()
