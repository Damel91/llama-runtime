"""Fresh metadata cohorts, independent owners, and bounded dispatch rounds."""

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Event

import pytest

from runtime_llama.adapters.embedding import _Call, _TextInputs
from runtime_llama.embedding_engine import _MetadataRound, _Preflight, _State
from runtime_llama.errors import LLMBackendError, LLMCancelledError

from test_embedding_contracts import (
    ContractClient,
    configured,
    eventually,
    texts,
    vector,
)


class MetadataClient(ContractClient):
    def __init__(self, replicas=2):
        super().__init__(replicas)
        self.metadata_calls = Counter()
        self.props_entered, self.props_release = Event(), Event()
        self.props_release.set()

    def health(self, **kwargs):
        self.metadata_calls["health"] += 1
        return super().health(**kwargs)

    def models(self, **kwargs):
        self.metadata_calls["models"] += 1
        return super().models(**kwargs)

    def props(self, model=None, **kwargs):
        self.metadata_calls["props"] += 1
        self.props_entered.set()
        self._wait(
            self.props_release,
            cancel=kwargs.get("cancel"),
            timeout=kwargs.get("timeout"),
        )
        return super().props(model, **kwargs)


def test_admission_cohort_uses_one_catalog_and_individual_pool_references():
    client = MetadataClient()
    adapter = configured(client)
    calls = tuple(_Call(adapter, None) for _ in range(4))
    contexts = ()
    try:
        contexts = adapter._engine_start_shared(calls)
        assert all(context.joined for context in contexts)
        assert client.metadata_calls == {"health": 1, "models": 1, "props": 2}
        assert adapter._pool_users == 4
        assert len({id(context.observations) for context in contexts}) == 4
        assert (
            contexts[0].observations["embed_0"] is contexts[1].observations["embed_0"]
        )
    finally:
        for call, context in zip(calls, contexts):
            if context is not None:
                adapter._engine_finish(call, context.joined)
        assert adapter._pool_users == 0
        adapter.close()


def test_cancellation_during_shared_admission_preserves_survivor_and_staging_ref():
    client = MetadataClient()
    client.props_release.clear()
    adapter = configured(client)
    cancelled = Event()
    calls = (_Call(adapter, cancelled.is_set), _Call(adapter, None))
    contexts = ()
    try:
        with ThreadPoolExecutor(1) as workers:
            pending = workers.submit(adapter._engine_start_shared, calls)
            assert client.props_entered.wait(2)
            assert adapter._pool_users == 1
            cancelled.set()
            eventually(lambda: calls[0].failure is not None)
            assert not pending.done()
            client.props_release.set()
            contexts = pending.result(3)
        assert contexts[0] is None and contexts[1].joined
        assert isinstance(calls[0].failure, LLMCancelledError)
        assert adapter._pool_users == 1
    finally:
        client.props_release.set()
        for call, context in zip(calls, contexts):
            if context is not None:
                adapter._engine_finish(call, context.joined)
        assert adapter._pool_users == 0
        adapter.close()


def test_all_cancelled_admission_drains_and_releases_staging_reference():
    client = MetadataClient()
    client.props_release.clear()
    adapter = configured(client)
    cancelled = Event()
    calls = tuple(_Call(adapter, cancelled.is_set) for _ in range(2))
    try:
        with ThreadPoolExecutor(1) as workers:
            pending = workers.submit(adapter._engine_start_shared, calls)
            assert client.props_entered.wait(2)
            assert adapter._pool_users == 1
            cancelled.set()
            with pytest.raises(LLMCancelledError):
                pending.result(3)
        assert adapter._pool_users == 0
        assert adapter._discovery_lock.acquire(blocking=False)
        adapter._discovery_lock.release()
    finally:
        client.props_release.set()
        adapter.close()


def test_registered_windows_share_admission_and_simultaneous_dispatch_catalog():
    client = MetadataClient()
    client.embed_gate.clear()
    adapter = configured(
        client,
        batch_size=8,
        dimension=2,
        embedding_request_parallelism=2,
        embedding_queue_size=2,
        # One priming owner plus the four owners injected as one cohort.
        embedding_max_pending_owners=5,
        embedding_batch_wait_sec=0,
        timeout=5,
    )
    pending, calls = [], []
    try:
        with ThreadPoolExecutor(1) as workers:
            prime = workers.submit(adapter.embed_texts, ["prime0", "prime1"])
            eventually(lambda: client.embed_active == 2)
            client.metadata_calls.clear()
            windows = [texts(2, index * 10) for index in range(4)]
            with adapter._engine._condition:
                for window in windows:
                    call = _Call(adapter, None)
                    calls.append(call)
                    pending.append(
                        adapter._submit_execute(
                            _TextInputs(window, "", "raw"),
                            operation="embed",
                            call=call,
                        )
                    )
            eventually(
                lambda: (
                    sum(adapter._engine.snapshot()["ready_by_replica"].values()) == 8
                )
            )
            assert client.metadata_calls == {"health": 1, "models": 1, "props": 2}
            # Both scheduler permits become free before the coordinator sees
            # either completion. One round can therefore observe both aliases.
            with adapter._engine._condition:
                client.embed_gate.set()
                eventually(
                    lambda: (
                        client.embed_active == 0
                        and adapter._scheduler.snapshot()["in_flight"] == 0
                    )
                )
            assert prime.result(3) == [vector("prime0"), vector("prime1")]
            assert [future.result(3) for future in pending] == [
                [vector(text) for text in window] for window in windows
            ]
        assert client.metadata_calls == {"health": 1, "models": 2, "props": 4}
        assert sorted(len(batch) for _, batch in client.embedding_calls[2:]) == [4, 4]
        assert adapter._pool_users == 0
        assert adapter._engine.snapshot()["metadata_targets"] == 0
    finally:
        client.embed_gate.set()
        for call in calls:
            call.stop.set()
        adapter._engine.wake()
        for future in pending:
            try:
                future.result(3)
            except (LLMBackendError, LLMCancelledError):
                pass
        for call in calls:
            lease = getattr(call, "_intake_lease", None)
            if lease is not None:
                lease.release()
        adapter.close()


@pytest.mark.parametrize("retired", ["finishing", "done", "failed", "no_work"])
def test_queued_round_drops_retired_or_exhausted_owners_before_props(retired):
    client = MetadataClient(1)
    adapter = configured(client)
    engine = adapter._engine
    call = _Call(adapter, None)
    state = _State([], "embed", call, Future(), False)
    contexts = adapter._engine_start_shared((call,))
    state.context = contexts[0]
    state.ready["embed_0"] = deque()
    if retired == "finishing":
        state.finishing = True
    elif retired == "done":
        state.future.set_result([])
    elif retired == "failed":
        call.fail(LLMCancelledError())
    metadata = _MetadataRound({"embed_0": (state,)}, {}, deque(["embed_0"]))
    engine._metadata_rounds.append(metadata)
    engine._observing.add("embed_0")
    before = client.metadata_calls.copy()
    try:
        engine._advance_metadata()
        assert not engine._metadata_rounds and not engine._observing
        assert client.metadata_calls == before
        assert state.pending == 0
        engine._preflights["embed_0"] = _Preflight(object(), (state,))
        engine.close()
        assert not engine._preflights
        assert engine.snapshot()["metadata_targets"] == 0
    finally:
        adapter._engine_finish(call, True)
        adapter.close()
