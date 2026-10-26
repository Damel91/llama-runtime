"""Large accepted work and object lifetime, independently of process RSS.

The 10,000-row fixtures cross the old whole-result reservation. The count is a
sample, not a supported maximum. Sentinel collection and drained ownership are
the assertions; an allocator retaining its high-water RSS is not a leak test.
"""

import asyncio
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gc
from threading import Event
import tracemalloc
from types import SimpleNamespace
import weakref

import pytest

import runtime_llama.embedding_engine as engine_module
from runtime_llama.automatic_embedding import AutomaticEmbeddingService
from runtime_llama.client import ClientHTTPError
from runtime_llama.embedding_halving import EmbeddingChunkSplitRequiredError
from runtime_llama.embedding_resources import EmbeddingResourceLimitError
from runtime_llama.errors import LLMBackendError, LLMCancelledError

from test_embedding_async_stream import WindowSource
from test_embedding_contracts import ContractClient, configured, eventually


class DecodedBody(dict):
    """A real response-shaped object whose lifetime can be observed."""


class DeliveredResults(list):
    """Observe delivery ownership without requiring a weakref-able public list."""


class WideClient(ContractClient):
    """Offline external server, deliberately not retaining source payloads."""

    def __init__(self, *, dimension=768, replicas=1):
        super().__init__(replicas, context=8192, native_batch=8192)
        self.dimension = dimension
        self.body_refs = []
        self.calls = []
        self.failure_mode = None
        self.initial_replicas = replicas
        self.topology_seen = Event()
        self.warm_gate = None

    def models(self, **kwargs):
        result = super().models(**kwargs)
        if len(result["data"]) > self.initial_replicas:
            self.topology_seen.set()
        return result

    @staticmethod
    def key(text):
        return int(text.rsplit("-", 1)[-1])

    def embeddings(self, input, *, model, timeout=None, cancel=None, **kwargs):
        # Keep scalar occurrence evidence only: the fixture itself must not
        # retain strings or decoded bodies and confuse the SDK lifetime audit.
        indices = tuple(self.key(text) for text in input)
        body = DecodedBody(
            data=[
                {
                    "index": position,
                    "embedding": [float(value), 1.0] + [0.0] * (self.dimension - 2),
                }
                for position, value in reversed(list(enumerate(indices)))
            ]
        )
        self.body_refs.append(weakref.ref(body))
        with self.lock:
            self.calls.append((model, indices))
            self.embed_active += 1
            self.embed_peak = max(self.embed_peak, self.embed_active)
            self.by_alias[model] += 1
            self.alias_peak[model] = max(self.alias_peak[model], self.by_alias[model])
            self.embed_started.set()
        try:
            gate = (
                self.warm_gate
                if self.warm_gate is not None and indices == (99999,)
                else self.embed_gate
            )
            self._wait(gate, timeout=timeout, cancel=cancel)
            if self.failure_mode == "transport":
                raise ClientHTTPError(503, error_type="fixture-native-error")
            if self.failure_mode == "malformed":
                body["data"] = []
            if self.failure_mode == "singleton" and 13 in indices:
                raise ClientHTTPError(500, error_type="fixture-rejected-input")
            return body
        finally:
            with self.lock:
                self.embed_active -= 1
                self.by_alias[model] -= 1


def incoming(count, *, start=0):
    return [f"source-{index}" for index in range(start, start + count)]


def service_for(client, **changes):
    values = {
        "base_url": "http://runtime.example",
        "model": "sha256:" + "a" * 64,
        "embedding_prefixes": {"query": "", "document": ""},
        "normalize": False,
        "batch_size": 10000,
        "timeout": 30,
        # These were commonly serialized by older consumers. Their presence
        # must not turn implicit capacity controls back on in the new contract.
        "embedding_max_pending_bytes": 64 * 1024 * 1024,
        "embedding_max_pending_tokens": 1024 * 1024,
        "embedding_max_request_bytes": 8 * 1024 * 1024,
        "embedding_max_response_bytes": 8 * 1024 * 1024,
        "embedding_max_pending_owners": 18,
        "queue_size": 16,
        "embedding_capacity_limits": False,
    }
    values.update(changes)
    return AutomaticEmbeddingService(values, client=client)


class OwnershipObserver:
    """Keep weak references, never the engine objects under investigation."""

    def __init__(self, adapter, monkeypatch):
        self.refs = defaultdict(list)
        self.state_sites = set()
        original_state = engine_module._State
        original_prepare = adapter._engine_prepare
        original_submit = adapter._submit_execute
        original_result = adapter._engine_result

        def state(*args, **kwargs):
            value = original_state(*args, **kwargs)
            self.refs["state"].append(weakref.ref(value))
            self.refs["call"].append(weakref.ref(value.call))
            if tracemalloc.is_tracing():
                # CPython does not expose an allocation trace for every
                # ordinary Python instance. Its real attribute dictionary is
                # still state-owned and traceable; materializing it here also
                # gives the audit an attributable site rather than an RSS cap.
                trace = tracemalloc.get_object_traceback(value)
                if trace is None:
                    attributes = vars(value)
                    trace = tracemalloc.get_object_traceback(attributes)
                if trace:
                    self.state_sites.add((trace[-1].filename, trace[-1].lineno))
            return value

        def prepare(*args, **kwargs):
            value = original_prepare(*args, **kwargs)
            self.refs["record"].append(weakref.ref(value))
            return value

        def submit(*args, **kwargs):
            future = original_submit(*args, **kwargs)
            self.refs["future"].append(weakref.ref(future))
            return future

        def result(call, operation, outputs):
            value = original_result(call, operation, outputs)
            if operation == "embed" and isinstance(value, list):
                value = DeliveredResults(value)
                self.refs["delivered"].append(weakref.ref(value))
            return value

        monkeypatch.setattr(engine_module, "_State", state)
        monkeypatch.setattr(adapter, "_engine_prepare", prepare)
        monkeypatch.setattr(adapter, "_submit_execute", submit)
        monkeypatch.setattr(adapter, "_engine_result", result)

    def assert_released(self, *, delivered=True):
        gc.collect()
        for kind, references in self.refs.items():
            if kind != "delivered" or delivered:
                assert all(ref() is None for ref in references), kind


def assert_idle(service, client):
    adapter = service._adapter

    def drained():
        view = adapter._engine.snapshot()
        return not adapter._calls and all(
            view[name] == 0
            for name in (
                "pending_bytes",
                "pending_tokens",
                "retained_result_bytes",
                "result_reserved_bytes",
                "embedding_in_flight",
                "tokenization_in_flight",
                "control_in_flight",
                "http_results_waiting",
                "preparation_results_waiting",
            )
        )

    eventually(drained)
    intake = adapter._intake.snapshot()
    assert intake["owners"] == intake["snapshot_bytes"] == 0
    scheduler = adapter._scheduler.snapshot()
    assert scheduler["in_flight"] == scheduler["queued"] == 0
    handoff = adapter._engine.snapshot()["handoff"]
    assert handoff["tickets"] == handoff["used"] == 0
    assert client.embed_active == client.token_active == 0


def assert_rows(rows, expected):
    assert len(rows) == len(expected)
    assert [row[:2] for row in rows] == [
        [float(WideClient.key(text)), 1.0] for text in expected
    ]
    assert all(len(row) == 768 and not any(row[2:]) for row in rows)


def test_ten_thousand_wide_results_are_accepted_and_transferred_without_old_pools(
    monkeypatch,
):
    client = WideClient()
    service = service_for(client)
    try:
        # Actual output width, not catalog training metadata, seeds packing.
        service.embed_query("source-99999")
        client.calls.clear()
        observer = OwnershipObserver(service._adapter, monkeypatch)
        texts = incoming(10000)
        texts[5000] = texts[3]  # Every occurrence must survive, including repeats.
        result = service.embed_texts(texts)
        assert_rows(result, texts)
        assert Counter(value for _, values in client.calls for value in values) == (
            Counter(WideClient.key(text) for text in texts)
        )
        assert [len(values) for _, values in client.calls] == [10000]
        assert not client.token_calls
        assert_idle(service, client)
        observer.assert_released(delivered=False)
        # The caller is allowed to retain its completed output. It is not an
        # SDK leak; only dropping this final owner should collect the container.
        assert observer.refs["delivered"][-1]() is result
        assert all(ref() is None for ref in client.body_refs)
        del result
        observer.assert_released()
    finally:
        service.close()


def test_ten_thousand_text_async_window_is_not_limited_by_whole_result_reservation(
    monkeypatch,
):
    async def scenario():
        client = WideClient()
        service = service_for(client)
        stream = None
        try:
            await asyncio.to_thread(service.embed_query, "source-99999")
            observer = OwnershipObserver(service._adapter, monkeypatch)
            texts = incoming(10000)
            source = WindowSource([texts])
            stream = service.astream_texts(source, close_source=True)
            async with stream:
                result = await anext(stream)
                assert_rows(result, texts)
                with pytest.raises(StopAsyncIteration):
                    await anext(stream)
            assert source.close_calls == 1
            assert not client.token_calls
            assert_idle(service, client)
            del result
            observer.assert_released()
            assert all(ref() is None for ref in client.body_refs)
        finally:
            if stream is not None:
                await stream.aclose()
            await service.aclose()

    asyncio.run(scenario())


def test_consumer_can_still_opt_into_a_real_result_budget():
    client = WideClient()
    adapter = configured(
        client,
        dimension=768,
        embedding_capacity_limits=True,
        embedding_max_pending_bytes=65536,
        embedding_max_pending_owners=1,
        embedding_batch_wait_sec=0,
    )
    try:
        with pytest.raises(EmbeddingResourceLimitError) as caught:
            adapter.embed_texts(incoming(100))
        assert caught.value.resource == "result_bytes"
        assert caught.value.limit == 65536
        assert caught.value.requested > caught.value.limit
        assert client.calls == []
        assert adapter._intake.snapshot()["owners"] == 0
        assert adapter._engine.snapshot()["result_reserved_bytes"] == 0
    finally:
        adapter.close()


def test_repeated_completed_calls_leave_no_state_allocation_or_internal_owner(
    monkeypatch,
):
    client = WideClient(dimension=8)
    service = service_for(client, batch_size=33)
    started = not tracemalloc.is_tracing()
    try:
        service.embed_query("source-99999")
        if started:
            tracemalloc.start()
        gc.collect()
        before = tracemalloc.take_snapshot()
        observer = OwnershipObserver(service._adapter, monkeypatch)
        for cycle in range(8):
            value = service.embed_texts(incoming(33, start=cycle * 100))
            assert len(value) == 33
            del value
            assert_idle(service, client)
            observer.assert_released()
        gc.collect()
        after = tracemalloc.take_snapshot()
        assert observer.state_sites

        def retained_state_allocations(snapshot):
            return {
                (stat.traceback[0].filename, stat.traceback[0].lineno): stat.size
                for stat in snapshot.statistics("lineno")
                if (stat.traceback[0].filename, stat.traceback[0].lineno)
                in observer.state_sites
            }

        initial = retained_state_allocations(before)
        final = retained_state_allocations(after)
        assert all(size <= initial.get(site, 0) for site, size in final.items())
        assert all(ref() is None for ref in client.body_refs)
    finally:
        if started:
            tracemalloc.stop()
        service.close()


@pytest.mark.parametrize("failure_mode", ["transport", "malformed"])
def test_failed_response_and_future_do_not_stay_alive_while_error_is_retained(
    monkeypatch, failure_mode
):
    client = WideClient(dimension=8)
    service = service_for(client, batch_size=100)
    try:
        service.embed_query("source-99999")
        observer = OwnershipObserver(service._adapter, monkeypatch)
        client.failure_mode = failure_mode
        with pytest.raises((ClientHTTPError, LLMBackendError)) as caught:
            service.embed_texts(incoming(100))
        assert caught.value is not None  # Retain the useful public error.
        assert_idle(service, client)
        observer.assert_released()
        assert all(ref() is None for ref in client.body_refs)
        client.failure_mode = None
        assert len(service.embed_texts(incoming(3))) == 3
        assert_idle(service, client)
    finally:
        service.close()


def test_cancelled_large_owner_drains_real_body_and_releases_all_records(monkeypatch):
    client = WideClient(dimension=8)
    service = service_for(client)
    cancel = Event()
    try:
        service.embed_query("source-99999")
        observer = OwnershipObserver(service._adapter, monkeypatch)
        client.embed_started.clear()
        client.embed_gate.clear()
        with ThreadPoolExecutor(1) as callers:
            work = callers.submit(
                service.embed_texts, incoming(10000), cancel=cancel.is_set
            )
            assert client.embed_started.wait(20)
            cancel.set()
            with pytest.raises(LLMCancelledError):
                work.result(10)
        assert_idle(service, client)
        del work
        observer.assert_released()
        assert all(ref() is None for ref in client.body_refs)
    finally:
        cancel.set()
        client.embed_gate.set()
        service.close()


def test_isolated_native_rejection_releases_internal_state_but_preserves_public_rows(
    monkeypatch,
):
    client = WideClient(dimension=8)
    service = service_for(client, batch_size=40)
    try:
        service.embed_query("source-99999")
        observer = OwnershipObserver(service._adapter, monkeypatch)
        client.failure_mode = "singleton"
        with pytest.raises(EmbeddingChunkSplitRequiredError) as caught:
            service.embed_texts(incoming(40))
        error = caught.value
        assert error.failed_input_indices == (13,)
        assert [row.input_index for row in error.completed] == [
            index for index in range(40) if index != 13
        ]
        assert all(row.embedding[0] == row.input_index for row in error.completed)
        assert_idle(service, client)
        observer.assert_released()
        assert all(ref() is None for ref in client.body_refs)
        assert not client.token_calls
    finally:
        service.close()


def test_retired_topology_objects_are_collectable_after_real_transport_drain():
    client = WideClient(dimension=8)
    service = service_for(client, batch_size=16)
    try:
        service.embed_query("source-99999")
        adapter = service._adapter
        old_scheduler = weakref.ref(adapter._scheduler)
        old_replica = weakref.ref(adapter._replicas["embed_0"])
        client.embed_started.clear()
        client.embed_gate.clear()
        with ThreadPoolExecutor(2) as callers:
            first = callers.submit(service.embed_texts, incoming(8, start=100))
            assert client.embed_started.wait(10)
            with client.lock:
                client.set_aliases(["embed_0", "embed_1"], context=8192)
            second = callers.submit(service.embed_texts, incoming(16, start=200))
            try:
                assert client.topology_seen.wait(10)
                assert adapter._scheduler is old_scheduler()
                assert not first.done()
            finally:
                client.embed_gate.set()
            assert len(first.result(10)) == 8
            assert len(second.result(10)) == 16
        del first, second
        assert adapter._scheduler.parallelism == 2
        assert_idle(service, client)
        gc.collect()
        assert old_scheduler() is None
        assert old_replica() is None
        assert all(ref() is None for ref in client.body_refs)
    finally:
        client.embed_gate.set()
        service.close()


def test_shared_post_withdraws_one_owner_and_releases_both_after_survivor_drain(
    monkeypatch,
):
    client = WideClient(dimension=8)
    adapter = configured(
        client,
        dimension=8,
        batch_size=4,
        embedding_request_parallelism=1,
        token_count_parallelism=2,
        embedding_capacity_limits=False,
        embedding_batch_wait_sec=0,
        timeout=20,
    )
    observer = OwnershipObserver(adapter, monkeypatch)
    cancel = Event()
    client.warm_gate = Event()
    client.embed_gate.clear()
    try:
        with ThreadPoolExecutor(3) as callers:
            warm = callers.submit(adapter.embed_query, "source-99999")
            assert client.embed_started.wait(10)
            client.embed_started.clear()
            client.token_gate.clear()
            first = callers.submit(
                adapter.embed_texts, ["source-100"], cancel=cancel.is_set
            )
            second = callers.submit(adapter.embed_texts, ["source-200"])
            eventually(lambda: client.token_active == 2)
            client.token_gate.set()
            eventually(
                lambda: (
                    adapter._engine.snapshot()["ready_by_replica"].get("embed_0") == 2
                )
            )
            # The occupied alias, not a timer, makes both owners available for
            # the same dispatch. The following transport stays held for Stop.
            client.warm_gate.set()
            assert warm.result(10) == [99999.0, 1.0] + [0.0] * 6
            assert client.embed_started.wait(10)
            assert len(client.calls) == 2
            assert Counter(client.calls[1][1]) == Counter((100, 200))
            cancel.set()
            eventually(
                lambda: any(
                    state.call.failure is not None
                    for state in tuple(adapter._engine._states)
                )
            )
            assert not first.done()
            assert not second.done()
            client.embed_gate.set()
            with pytest.raises(LLMCancelledError):
                first.result(10)
            assert second.result(10) == [[200.0, 1.0] + [0.0] * 6]
        del warm, first, second
        assert_idle(SimpleNamespace(_adapter=adapter), client)
        observer.assert_released()
        assert all(ref() is None for ref in client.body_refs)
    finally:
        cancel.set()
        client.token_gate.set()
        client.embed_gate.set()
        client.warm_gate.set()
        adapter.close()


def test_explicit_result_and_handoff_budgets_do_not_form_cold_ordering_cycle():
    class HeldFirstProps(WideClient):
        def __init__(self):
            super().__init__(dimension=2, replicas=2)
            self.first_props = Event()
            self.release_props = Event()
            self.props_count = 0

        def props(self, model=None, **kwargs):
            with self.lock:
                self.props_count += 1
                first = self.props_count == 1
            if first:
                self.first_props.set()
                self._wait(
                    self.release_props,
                    cancel=kwargs.get("cancel"),
                    timeout=kwargs.get("timeout"),
                )
            return super().props(model, **kwargs)

    async def scenario():
        client = HeldFirstProps()
        older = ["x" * 70 + f"-{index}" for index in range(24)]
        younger = ["x" * 70 + f"-{index}" for index in range(1000, 1024)]
        client.counts = dict.fromkeys([*older, *younger], 1)
        adapter = configured(
            client,
            embedding_replica_discovery=False,
            embedding_replicas=[{"model": "embed_0"}, {"model": "embed_1"}],
            batch_size=2,
            dimension=None,
            embedding_capacity_limits=True,
            embedding_max_pending_bytes=8192,
            embedding_max_handoff_bytes=400,
            embedding_batch_wait_sec=0,
            timeout=15,
        )

        async def windows():
            yield older
            assert await asyncio.to_thread(client.first_props.wait, 5)
            yield younger

        stream = adapter.astream_texts(windows())
        awaiting_first = asyncio.create_task(anext(stream))
        try:
            assert await asyncio.to_thread(client.first_props.wait, 5)

            def younger_observed():
                return any(
                    getattr(state.call, "_window_index", None) == 1
                    and state.context is not None
                    for state in tuple(adapter._engine._states)
                )

            await asyncio.to_thread(eventually, younger_observed)
            # A younger cold body must not consume the handoff while it waits
            # for credits owned by the still-unobserved earlier window.
            assert client.calls == []
            client.release_props.set()
            first = await asyncio.wait_for(awaiting_first, 10)
            second = await asyncio.wait_for(anext(stream), 10)
            assert [row[0] for row in first] == list(map(float, range(24)))
            assert [row[0] for row in second] == list(map(float, range(1000, 1024)))
            assert client.calls[0][1][0] < 1000
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
            assert_idle(SimpleNamespace(_adapter=adapter), client)
        finally:
            client.release_props.set()
            if not awaiting_first.done():
                awaiting_first.cancel()
            await asyncio.gather(awaiting_first, return_exceptions=True)
            await stream.aclose()
            await adapter.aclose()

    asyncio.run(scenario())


def test_discovered_worker_topology_grows_past_four_and_reaps_retired_generations(
    monkeypatch,
):
    client = WideClient(dimension=8, replicas=5)
    adapter = configured(
        client,
        dimension=8,
        token_count_parallelism=None,
        embedding_request_parallelism=None,
        batch_size=None,
        embedding_batch_wait_sec=0,
        timeout=20,
    )
    engine = adapter._engine
    expected_topology = [None]
    topology_observed = Event()
    catalog = client.models

    def models(**kwargs):
        result = catalog(**kwargs)
        if len(result["data"]) == expected_topology[0]:
            topology_observed.set()
        return result

    monkeypatch.setattr(client, "models", models)
    retired_threads = []
    try:
        # Make all five native tokenization calls remain active together: this
        # proves the discovered topology, not a fixed four-worker ceiling,
        # controls actual execution as well as diagnostic configuration.
        client.token_gate.clear()
        with ThreadPoolExecutor(1) as callers:
            preparation = callers.submit(adapter.prepare_texts, incoming(10))
            try:
                eventually(lambda: client.token_active == 5)
                assert engine.tokenizers == engine.control_workers == 5
                assert engine._tokenizer._max_workers == 5
                assert engine._control._max_workers == 5
            finally:
                client.token_gate.set()
            assert len(preparation.result(10)) == 10

        def change_topology(count, offset):
            previous_count = len(adapter._replicas)
            old_control, old_tokenizer = engine._control, engine._tokenizer
            old_threads = tuple(old_control._threads | old_tokenizer._threads)
            retired_threads.extend(old_threads)
            client.embed_started.clear()
            client.embed_gate.clear()
            expected_topology[0] = count
            topology_observed.clear()
            with ThreadPoolExecutor(2) as callers:
                draining = callers.submit(
                    adapter.embed_texts,
                    incoming(previous_count * 2, start=offset),
                )
                try:
                    eventually(lambda: client.embed_active == previous_count)
                    with client.lock:
                        client.set_aliases(
                            [f"embed_{index}" for index in range(count)],
                            context=8192,
                            native_batch=8192,
                        )
                    incoming_generation = callers.submit(
                        adapter.embed_texts, incoming(count * 2, start=offset + 100)
                    )
                    assert topology_observed.wait(5)
                    # A fresh observation cannot resize workers while the old
                    # generation still owns the in-flight original requests.
                    assert engine._control is old_control
                    assert engine._tokenizer is old_tokenizer
                    assert engine.control_workers == engine.tokenizers == previous_count
                    assert not draining.done()
                finally:
                    client.embed_gate.set()
                assert len(draining.result(10)) == previous_count * 2
                assert len(incoming_generation.result(10)) == count * 2
            eventually(lambda: engine.control_workers == engine.tokenizers == count)
            assert engine._control is not old_control
            assert engine._tokenizer is not old_tokenizer
            assert engine._control._max_workers == count
            assert engine._tokenizer._max_workers == count
            eventually(lambda: all(not thread.is_alive() for thread in old_threads))
            assert old_control._shutdown and old_tokenizer._shutdown
            # A further real operation also exercises reaping after late
            # executor worker exit, without retaining the retired generation.
            assert (
                len(adapter.prepare_texts(incoming(count, start=offset + 200))) == count
            )
            eventually(lambda: not engine._retired_executors)

        change_topology(7, 1000)
        change_topology(2, 2000)
    finally:
        client.token_gate.set()
        client.embed_gate.set()
        current = (engine._control, engine._tokenizer)
        adapter.close()
    assert all(not thread.is_alive() for thread in retired_threads)
    assert all(executor._shutdown for executor in current if executor is not None)
    assert all(
        not thread.is_alive()
        for executor in current
        if executor is not None
        for thread in executor._threads
    )
    assert not engine._retired_executors
