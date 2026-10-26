"""Synchronous results stay credited through caller transfer, as async results do."""

from concurrent.futures import ThreadPoolExecutor
import gc
from threading import Event, Lock
import time
import weakref

import pytest

from runtime_llama.embedding_resources import EmbeddingResourceLimitError
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError
from runtime_llama.automatic_embedding import AutomaticEmbeddingService
from runtime_llama.client import ClientHTTPError
from runtime_llama.embedding_halving import EmbeddingChunkSplitRequiredError
import runtime_llama.embedding_engine as engine_module
import runtime_llama.adapters.embedding as adapter_module
from test_automatic_embedding import remote
from test_embedding_contracts import (
    ContractClient,
    configured,
    eventually,
    texts,
    vector,
)


def adapter_for(client, **changes):
    values = {
        "embedding_capacity_limits": True,
        "embedding_max_pending_bytes": 65536,
        "embedding_batch_wait_sec": 0,
        "dimension": 2,
        "timeout": 3,
    }
    values.update(changes)
    return configured(client, **values)


def hold_first_transfer(adapter, monkeypatch):
    original = adapter._submit_execute
    completed, release = Event(), Event()
    lock = Lock()
    first = True

    def submit(*args, **kwargs):
        nonlocal first
        future = original(*args, **kwargs)
        with lock:
            selected, first = first, False
        if selected:
            result = future.result

            def wait(*args, **kwargs):
                value = result(*args, **kwargs)
                completed.set()
                assert release.wait(2)
                return value

            future.result = wait
        return future

    monkeypatch.setattr(adapter, "_submit_execute", submit)
    return completed, release


@pytest.mark.parametrize("operation", ["embed_texts", "prepare_texts"])
def test_completed_sync_future_still_owns_result_bytes_until_caller_transfer(
    monkeypatch, operation
):
    client = ContractClient(1)
    adapter = adapter_for(client)
    completed, release = hold_first_transfer(adapter, monkeypatch)
    try:
        with ThreadPoolExecutor(1) as callers:
            result = callers.submit(getattr(adapter, operation), ["one", "two"])
            try:
                assert completed.wait(2)
                view = adapter._engine.snapshot()
                assert (
                    0
                    < view["retained_result_bytes"]
                    <= view["completed_result_bytes_limit"]
                )
                assert view["result_reserved_bytes"] >= view["retained_result_bytes"]
                assert adapter._intake.snapshot()["owners"] == 1
                assert not result.done()
            finally:
                release.set()
            value = result.result(2)
        if operation == "embed_texts":
            assert value == [vector("one"), vector("two")]
        else:
            assert type(value) is tuple
            assert [row.text for row in value] == ["one", "two"]
            assert adapter.embed_prepared(value) == [vector("one"), vector("two")]
        assert adapter._engine.snapshot()["result_reserved_bytes"] == 0
        assert adapter._intake.snapshot()["owners"] == 0
    finally:
        release.set()
        adapter.close()


def test_oversized_whole_sync_result_fails_before_post_with_typed_budget():
    client = ContractClient(1)
    adapter = adapter_for(client, embedding_max_pending_bytes=8192, dimension=128)
    try:
        with pytest.raises(EmbeddingResourceLimitError) as caught:
            adapter.embed_texts(["one", "two"])
        assert caught.value.resource == "result_bytes"
        assert caught.value.requested > caught.value.limit == 8192
        assert client.embedding_calls == []
        assert adapter._engine.snapshot()["result_reserved_bytes"] == 0
    finally:
        adapter.close()


def test_aggregate_sync_pressure_waits_for_real_caller_transfer(monkeypatch):
    client = ContractClient(1)
    adapter = adapter_for(client, embedding_max_pending_bytes=8192)
    completed, release = hold_first_transfer(adapter, monkeypatch)
    first_texts, second_texts = texts(30), texts(30, 1000)
    try:
        with ThreadPoolExecutor(2) as callers:
            first = callers.submit(adapter.embed_texts, first_texts)
            try:
                assert completed.wait(2)
                second = callers.submit(adapter.embed_texts, second_texts)
                eventually(lambda: adapter._intake.snapshot()["owners"] == 2)
                time.sleep(0.02)
                assert not second.done()
                assert not any(
                    text in second_texts
                    for _, batch in client.embedding_calls
                    for text in batch
                )
                assert adapter._engine.snapshot()["result_reserved_bytes"] <= 8192
            finally:
                release.set()
            assert first.result(2) == [vector(text) for text in first_texts]
            assert second.result(2) == [vector(text) for text in second_texts]
        assert adapter._engine.snapshot()["result_reserved_bytes"] == 0
    finally:
        release.set()
        adapter.close()


def test_bulk_exact_preparation_is_bounded_and_overcontext_inputs_remain_measurable():
    client = ContractClient(1, context=2)
    client.counts = {"long": 20}
    adapter = adapter_for(client, embedding_max_pending_bytes=16384)
    try:
        measured = adapter.prepare_texts(["long"])
        assert measured[0].token_count == 20
        assert measured[0].fits_context is False
        with pytest.raises(LLMBackendError, match="exceeds_context"):
            adapter.embed_prepared(measured)
        with pytest.raises(EmbeddingResourceLimitError, match="result_bytes"):
            adapter.prepare_texts(texts(20))
        assert client.embedding_calls == []
        assert adapter._engine.snapshot()["result_reserved_bytes"] == 0
        assert adapter.prepare_texts(["ok"])[0].text == "ok"
    finally:
        adapter.close()


def test_two_prepare_owners_do_not_deadlock_partial_output_credits_and_cancel_drains():
    client = ContractClient(1)
    client.token_gate.clear()
    adapter = adapter_for(client)
    cancel = Event()
    first_inputs, second_inputs = texts(4), texts(4, 1000)
    try:
        with ThreadPoolExecutor(2) as callers:
            first = callers.submit(adapter.prepare_texts, first_inputs)
            assert client.token_started.wait(2)
            eventually(lambda: client.token_active == 2)
            second = callers.submit(
                adapter.prepare_texts, second_inputs, cancel=cancel.is_set
            )
            eventually(lambda: adapter._intake.snapshot()["owners"] == 2)
            try:
                assert not second.done()
                assert not any(text in second_inputs for text, _ in client.token_calls)
                cancel.set()
                with pytest.raises(LLMCancelledError):
                    second.result(2)
            finally:
                client.token_gate.set()
            measured = first.result(2)
        assert [row.text for row in measured] == first_inputs
        assert client.token_peak == 2
        assert client.token_active == 0
        assert adapter._engine.snapshot()["result_reserved_bytes"] == 0
        assert adapter._intake.snapshot()["owners"] == 0
    finally:
        cancel.set()
        client.token_gate.set()
        adapter.close()


def test_rejection_growth_waits_for_missing_delta_before_publication(monkeypatch):
    client = remote(replicas=1)
    service = AutomaticEmbeddingService(
        {
            "base_url": "http://runtime.example",
            "embedding_capacity_limits": True,
            "embedding_max_pending_bytes": 8192,
            "normalize": False,
            "timeout": 3,
        },
        client=client,
    )
    release = None
    try:
        service.embed_texts(["warmup"])
        adapter = service._adapter
        assert adapter._engine.snapshot()["result_reserved_bytes"] == 0
        completed, release = hold_first_transfer(adapter, monkeypatch)
        first_inputs = texts(30)
        second_inputs = ["rejected"] + texts(14, 1000)
        with ThreadPoolExecutor(2) as callers:
            first = callers.submit(service.embed_texts, first_inputs)
            try:
                assert completed.wait(2)
                client.failure = lambda _model, batch: (
                    ClientHTTPError(500)
                    if "search_document: rejected" in batch
                    else None
                )
                second = callers.submit(service.embed_texts, second_inputs)
                eventually(
                    lambda: any(
                        state.rejection_slots
                        and state.result_required > state.result_credit > 0
                        for state in tuple(adapter._engine._states)
                    )
                )
                state = next(
                    state
                    for state in tuple(adapter._engine._states)
                    if state.rejection_slots
                )
                assert not state.result_ready
                assert 0 not in state.rejections
                assert state.outputs[0] is None
                assert adapter._engine.snapshot()["http_results_waiting"] > 0
                assert not second.done()
                assert adapter._engine.snapshot()["result_reserved_bytes"] <= 8192
            finally:
                release.set()
            assert len(first.result(2)) == len(first_inputs)
            with pytest.raises(EmbeddingChunkSplitRequiredError) as caught:
                second.result(2)
        assert len(caught.value.inputs) == 1
        assert len(caught.value.completed) == len(second_inputs) - 1
        assert adapter._engine.snapshot()["result_reserved_bytes"] == 0
        assert adapter._engine.snapshot()["handoff"]["tickets"] == 0
    finally:
        if release is not None:
            release.set()
        service.close()


@pytest.mark.parametrize("original_failure", [False, True])
def test_interrupted_release_ack_drains_and_preserves_primary_exception(
    monkeypatch, original_failure
):
    class InterruptedAcknowledgement(Event):
        def __init__(self):
            super().__init__()
            self.interrupted = False

        def is_set(self):
            # Exercise an interrupted wait even when coordinator cleanup is
            # very fast; its real event still determines the eventual drain.
            return self.interrupted and super().is_set()

        def wait(self, timeout=None):
            if not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt()
            return super().wait(timeout)

    client = ContractClient(1)
    if original_failure:
        client.failure = lambda _alias, _texts: LLMBackendError(
            "original-owner-failure"
        )
    adapter = adapter_for(client)
    monkeypatch.setattr(engine_module, "Event", InterruptedAcknowledgement)
    try:
        expected = LLMBackendError if original_failure else KeyboardInterrupt
        with pytest.raises(expected) as caught:
            adapter.embed_query("text")
        if original_failure:
            assert str(caught.value) == "original-owner-failure"
        snapshot = adapter._engine.snapshot()
        assert (
            snapshot["retained_result_bytes"] == snapshot["result_reserved_bytes"] == 0
        )
        assert adapter._intake.snapshot()["owners"] == 0
        assert client.embed_active == client.token_active == 0
    finally:
        adapter.close()


def test_idle_coordinator_releases_completed_future_after_ack_without_new_work(
    monkeypatch,
):
    client = ContractClient(1)
    adapter = adapter_for(client)
    original = adapter._submit_execute
    references = []

    def submit(*args, **kwargs):
        future = original(*args, **kwargs)
        references.append(weakref.ref(future))
        return future

    monkeypatch.setattr(adapter, "_submit_execute", submit)
    try:
        result = adapter.embed_texts(["last-result"])
        assert result == [vector("last-result")]
        assert adapter._engine.snapshot()["result_reserved_bytes"] == 0
        gc.collect()
        assert references and all(reference() is None for reference in references)
        assert result == [vector("last-result")]
    finally:
        adapter.close()


@pytest.mark.parametrize("status", [500, 503])
def test_failed_transport_frames_do_not_outlive_released_handoff(status):
    class DecodedBody:
        pass

    class BodyClient(ContractClient):
        references = []

        def embeddings(self, input, **kwargs):
            decoded_body = DecodedBody()
            self.references.append(weakref.ref(decoded_body))
            if "search_document: rejected" in input:
                raise ClientHTTPError(status, error_type="bounded-provider-status")
            return super().embeddings(input, **kwargs)

    client = BodyClient(1)
    # Use the automatic facade's explicit known-model prefix configuration.
    service = AutomaticEmbeddingService(
        {
            "base_url": "http://runtime.example",
            "model": "sha256:" + "a" * 64,
            "embedding_prefixes": {"query": "", "document": "search_document: "},
            "normalize": False,
        },
        client=client,
    )
    try:
        expected = (
            EmbeddingChunkSplitRequiredError if status == 500 else ClientHTTPError
        )
        with pytest.raises(expected) as caught:
            service.embed_texts(["rejected"])
        assert caught.value is not None
        assert service._adapter._engine.snapshot()["handoff"]["tickets"] == 0
        gc.collect()
        assert client.references and all(
            reference() is None for reference in client.references
        )
        if status == 503:
            assert caught.value.status_code == 503
            assert caught.value.error_type == "bounded-provider-status"
    finally:
        service.close()


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("termination", ["cancel", "deadline"])
def test_exact_sealing_honors_original_cancellation_and_deadline(
    monkeypatch, count, termination
):
    client = ContractClient(1)
    adapter = adapter_for(client)
    cancel = Event()
    original_finish = adapter._engine_result
    original_issue = adapter_module.issue_preparation
    current, sealed = {}, []
    armed = True

    def finish(call, operation, outputs):
        current["call"] = call
        return original_finish(call, operation, outputs)

    def issue(*args, **kwargs):
        nonlocal armed
        result = original_issue(*args, **kwargs)
        sealed.append(result)
        if armed:
            armed = False
            if termination == "cancel":
                cancel.set()
            else:
                current["call"].deadline = time.monotonic() - 1
        return result

    monkeypatch.setattr(adapter, "_engine_result", finish)
    monkeypatch.setattr(adapter_module, "issue_preparation", issue)
    try:
        error = LLMCancelledError if termination == "cancel" else LLMTimeoutError
        with pytest.raises(error):
            adapter.prepare_texts(texts(count), cancel=cancel.is_set)
        assert len(sealed) == 1
        assert adapter._engine.snapshot()["result_reserved_bytes"] == 0
        assert adapter._intake.snapshot()["owners"] == 0
        assert client.token_active == 0
        assert not adapter._calls
        cancel.clear()
        assert adapter.prepare_texts(["recovery"])[0].text == "recovery"
    finally:
        cancel.set()
        adapter.close()
