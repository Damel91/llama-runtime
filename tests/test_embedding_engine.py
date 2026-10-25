"""Persistent queue ownership and actual dispatch boundaries, entirely offline."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
from threading import Event, Lock, current_thread, get_ident

import pytest

from runtime_llama.adapters.embedding import _TextInputs
from runtime_llama.embedding_scheduler import EmbeddingScheduler
from runtime_llama.errors import LLMBackendError

from test_embedding_contracts import (
    ContractClient,
    configured,
    eventually,
    texts,
    vector,
)
from test_embedding_progressive import TokenClient
from test_embedding_async_stream import WindowSource, run, wait_thread


def test_batch_is_selected_after_capacity_not_when_its_first_record_arrives():
    client = TokenClient(1)
    incoming = texts(15)
    client.embed_gate.clear()
    adapter = configured(
        client,
        batch_size=8,
        embedding_batch_wait_sec=0,
        embedding_request_parallelism=1,
        embedding_queue_size=1,
        token_count_parallelism=1,
        timeout=5,
    )
    try:
        with ThreadPoolExecutor(2) as callers:
            first = callers.submit(adapter.embed_texts, incoming[:8])
            assert client.embed_started.wait(2)
            assert client.embedding_calls == [("embed_0", tuple(incoming[:8]))]
            pending = callers.submit(adapter.embed_texts, incoming[8:])
            eventually(
                lambda: all(
                    client.token_done.get(text, Event()).is_set() for text in incoming
                )
            )
            eventually(
                lambda: (
                    adapter._engine.snapshot()["ready_by_replica"].get("embed_0") == 7
                )
            )
            assert adapter._scheduler.snapshot()["queued"] == 0
            assert len(client.embedding_calls) == 1
            client.embed_gate.set()
            assert first.result(3) == [vector(text) for text in incoming[:8]]
            assert pending.result(3) == [vector(text) for text in incoming[8:]]
        assert [len(batch) for _, batch in client.embedding_calls] == [8, 7]
        assert Counter(
            text for _, batch in client.embedding_calls for text in batch
        ) == Counter(incoming)
    finally:
        client.embed_gate.set()
        adapter.close()


def test_two_replicas_really_run_on_distinct_threads_while_preparation_continues():
    client = TokenClient(2)
    client.embed_gate.clear()
    observed, lock = {}, Lock()
    request = client.embeddings

    def embeddings(input, *, model, **kwargs):
        with lock:
            observed.setdefault(model, set()).add(get_ident())
        return request(input, model=model, **kwargs)

    client.embeddings = embeddings
    adapter = configured(
        client,
        embedding_batch_wait_sec=0,
        embedding_request_parallelism=2,
        embedding_queue_size=0,
        token_count_parallelism=2,
        timeout=5,
    )
    incoming = texts(32)
    try:
        with ThreadPoolExecutor(1) as callers:
            pending = callers.submit(adapter.embed_texts, incoming)
            eventually(lambda: client.embed_active == 2)
            eventually(lambda: len(client.token_calls) == len(incoming))
            assert set(observed) == {"embed_0", "embed_1"}
            assert len(set.union(*observed.values())) == 2
            assert client.embed_peak == 2 and client.token_peak <= 2
            assert not pending.done()
            client.embed_gate.set()
            assert pending.result(3) == [vector(text) for text in incoming]
        assert adapter._engine.snapshot()["peak_embedding_in_flight"] == 2
        assert not adapter._engine.snapshot()["ready_by_replica"].get("embed_0", 0)
    finally:
        client.embed_gate.set()
        adapter.close()


def test_slow_preflight_for_one_replica_does_not_stop_the_other_replica():
    class SlowProps(ContractClient):
        def __init__(self):
            super().__init__(2)
            self.observations = Counter()
            self.blocked, self.release, self.other = Event(), Event(), Event()

        def props(self, model=None, **kwargs):
            self.observations[model] += 1
            if model == "embed_0" and self.observations[model] == 2:
                self.blocked.set()
                self._wait(
                    self.release,
                    cancel=kwargs.get("cancel"),
                    timeout=kwargs.get("timeout"),
                )
            return super().props(model, **kwargs)

        def embeddings(self, input, *, model, **kwargs):
            if model == "embed_1":
                self.other.set()
            return super().embeddings(input, model=model, **kwargs)

    client = SlowProps()
    adapter = configured(
        client,
        embedding_batch_wait_sec=0,
        embedding_request_parallelism=2,
        token_count_parallelism=2,
        timeout=5,
    )
    incoming = texts(8)
    try:
        with ThreadPoolExecutor(1) as callers:
            pending = callers.submit(adapter.embed_texts, incoming)
            assert client.blocked.wait(2)
            assert client.other.wait(2)
            assert not pending.done()
            client.release.set()
            assert pending.result(3) == [vector(text) for text in incoming]
    finally:
        client.release.set()
        adapter.close()


def test_slow_inputs_on_first_replica_do_not_take_every_tokenizer_worker():
    client = TokenClient(2)
    incoming = texts(8)
    first = client.gate(incoming[0])
    second = client.gate(incoming[1])
    adapter = configured(
        client,
        embedding_batch_wait_sec=0,
        embedding_request_parallelism=2,
        embedding_queue_size=0,
        token_count_parallelism=2,
        timeout=5,
    )
    try:
        with ThreadPoolExecutor(1) as callers:
            pending = callers.submit(adapter.embed_texts, incoming)
            eventually(
                lambda: any(model == "embed_1" for model, _ in client.embedding_calls)
            )
            assert not first.is_set() and not second.is_set()
            assert not pending.done()
            first.set()
            second.set()
            assert pending.result(3) == [vector(text) for text in incoming]
    finally:
        first.set()
        second.set()
        adapter.close()


def test_prefix_is_not_materialized_before_its_actual_byte_admission(monkeypatch):
    client = ContractClient(1)
    adapter = configured(
        client,
        embedding_max_pending_bytes=1024,
        embedding_prefixes={"document": "p" * 1024},
    )
    materialized = []
    get_item = _TextInputs.__getitem__

    def item(inputs, index):
        materialized.append(index)
        return get_item(inputs, index)

    monkeypatch.setattr(_TextInputs, "__getitem__", item)
    try:
        with pytest.raises(LLMBackendError, match="pending_bytes_limit"):
            adapter.embed_texts(["one", "two"])
        assert materialized == []
        assert not client.token_calls and not client.embedding_calls
        assert adapter._engine.snapshot()["peak_pending_bytes"] <= 1024
    finally:
        adapter.close()


def test_exact_json_request_budget_packs_unicode_without_changing_outputs():
    client = ContractClient(1)
    request = client.embeddings
    sizes = []

    def embeddings(input, *, model, **kwargs):
        sizes.append(len(json.dumps({"input": input, "model": model}).encode("utf-8")))
        return request(input, model=model, **kwargs)

    client.embeddings = embeddings
    adapter = configured(
        client,
        embedding_input_format="text",
        embedding_max_request_bytes=200,
        embedding_batch_wait_sec=0.5,
    )
    incoming = ["a" * 60, "🧪" * 5, "中文" * 5, "b" * 60]
    try:
        assert adapter.embed_texts(incoming) == [vector(text) for text in incoming]
        assert len(sizes) >= 2 and max(sizes) <= 200
        assert Counter(
            text for _, batch in client.embedding_calls for text in batch
        ) == Counter(incoming)
    finally:
        adapter.close()


def test_cancellation_callbacks_never_run_on_the_coordinator_thread():
    client = ContractClient(2)
    adapter = configured(client, embedding_batch_wait_sec=0)
    sampled = []

    def probe():
        sampled.append(current_thread().name)
        assert "coordinator" not in current_thread().name
        return False

    try:
        incoming = texts(12)
        assert adapter.embed_texts(incoming, cancel=probe) == [
            vector(text) for text in incoming
        ]
        assert sampled
        assert any(name.startswith("runtime_llama_control") for name in sampled)
    finally:
        adapter.close()


@pytest.mark.parametrize("text", ['"' * 70, "🧪" * 10])
def test_tokenizer_json_envelope_is_admitted_before_http(text):
    client = ContractClient(1)
    adapter = configured(client, embedding_max_request_bytes=200)
    try:
        assert len(text.encode("utf-8")) < 200
        with pytest.raises(LLMBackendError, match="request_bytes_limit"):
            adapter.prepare_texts([text])
        assert not client.token_calls and not client.embedding_calls
    finally:
        adapter.close()


def test_nonqueueing_scheduler_dispatch_never_freezes_waiting_work():
    scheduler = EmbeddingScheduler(1, 8, {"a": 1})
    started, release = Event(), Event()

    def first(_alias):
        started.set()
        assert release.wait(3)

    try:
        active = scheduler.try_dispatch(first, lambda: None, model="a")
        assert started.wait(2)
        assert (
            scheduler.try_dispatch(lambda alias: alias, lambda: None, model="a") is None
        )
        assert not scheduler.has_capacity("a")
        assert scheduler.snapshot()["queued"] == 0
        release.set()
        assert active.result(2) is None
        assert scheduler.has_capacity("a")
        assert (
            scheduler.try_dispatch(lambda alias: alias, lambda: None, model="a").result(
                2
            )
            == "a"
        )
    finally:
        release.set()
        scheduler.close()


def test_result_credits_follow_source_order_when_earlier_discovery_is_slow(monkeypatch):
    class SlowFirstStart(ContractClient):
        def __init__(self):
            super().__init__(1)
            self.first_props, self.release = Event(), Event()
            self.count = 0

        def props(self, model=None, **kwargs):
            with self.lock:
                self.count += 1
                first = self.count == 1
            if first:
                self.first_props.set()
                self._wait(
                    self.release,
                    cancel=kwargs.get("cancel"),
                    timeout=kwargs.get("timeout"),
                )
            return super().props(model, **kwargs)

    async def scenario():
        client = SlowFirstStart()
        adapter = configured(
            client,
            embedding_request_parallelism=2,
            embedding_queue_size=2,
            embedding_batch_wait_sec=0,
            embedding_max_pending_bytes=1024,
            token_count_parallelism=1,
            timeout=5,
            # This case needs an unknown-dimension seed from the later owner
            # while the earlier owner's metadata is still delayed.
            dimension=None,
        )
        windows = [["a0", "a1"], ["b0", "b1", "b2", "b3"]]
        validated, return_response, reserve_checked, completed_checked = (
            Event() for _ in range(4)
        )
        run_shared = adapter._engine_run_shared
        reserve_result = adapter._engine._reserve_result

        def held_response(model, records, owners, group, observed):
            result = run_shared(model, records, owners, group, observed)
            if any(record.text.startswith("b") for record in records):
                # Width is already visible in call.dimension, but the HTTP
                # Future/lease has not completed. Force a coordinator pass in
                # precisely the race that used to grant the younger credits.
                validated.set()
                adapter._engine.wake()
                client._wait(
                    return_response, cancel=group.probe, timeout=group.remaining()
                )
            return result

        def checked_reservation(state):
            result = reserve_result(state)
            if getattr(state.call, "_window_index", None) == 1:
                if validated.is_set() and not return_response.is_set():
                    reserve_checked.set()
                if state.response_dimension is not None:
                    completed_checked.set()
            return result

        monkeypatch.setattr(adapter, "_engine_run_shared", held_response)
        monkeypatch.setattr(adapter._engine, "_reserve_result", checked_reservation)
        stream = adapter.astream_texts(WindowSource(windows), max_pending_windows=2)
        first = asyncio.create_task(anext(stream))
        try:
            await wait_thread(client.first_props)
            await wait_thread(client.embed_started)
            assert client.embedding_calls[0][1][0].startswith("b")
            await wait_thread(validated)
            await wait_thread(reserve_checked)
            states = sorted(
                tuple(adapter._engine._states),
                key=lambda state: state.call._window_index,
            )
            assert states[1].call.dimension == 2
            assert states[1].response_dimension is None
            assert states[1].result_credit == 0
            return_response.set()
            await wait_thread(completed_checked)
            assert states[0].context is None
            assert states[0].result_credit == 416
            assert states[1].result_credit == 0
            client.release.set()
            assert await asyncio.wait_for(first, 3) == [
                vector(text) for text in windows[0]
            ]
            assert await asyncio.wait_for(anext(stream), 3) == [
                vector(text) for text in windows[1]
            ]
            assert adapter._engine.snapshot()["peak_result_reserved_bytes"] <= 1024
        finally:
            return_response.set()
            client.release.set()
            await asyncio.gather(first, return_exceptions=True)
            await stream.aclose()
            await adapter.aclose()

    run(scenario)
