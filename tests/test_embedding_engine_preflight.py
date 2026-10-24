"""Fresh observation arbitration between concurrent window owners."""

from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from runtime_llama.embedding_engine import PersistentEmbeddingEngine, _Preflight, _State


class ImmediateScheduler:
    def __init__(self):
        self.available = True
        self.dispatched = []

    def has_capacity(self, model):
        return self.available

    def snapshot(self):
        return {"in_flight": int(not self.available)}

    def try_dispatch(self, run, check, *, model):
        check()
        self.available = False
        self.dispatched.append(model)
        return Future()


@pytest.mark.parametrize(
    "owner_status", ["ready", "failed", "contextless", "uncredited", "empty"]
)
def test_completed_preflight_wins_rotation_before_another_owner_probes(owner_status):
    scheduler = ImmediateScheduler()
    adapter = SimpleNamespace(
        embedding_request_parallelism=1,
        token_count_parallelism=1,
        embedding_queue_size=0,
        embedding_dim=2,
        embedding_batch_wait_sec=0,
        _scheduler=scheduler,
        _engine_batch_call=lambda calls: SimpleNamespace(check_local=lambda: None),
    )
    engine = PersistentEmbeddingEngine(adapter)
    observations = []
    engine._shared_task = lambda kind, owners, metadata, run: observations.append(
        (kind, owners, metadata)
    )
    engine._pack = lambda model, observed, states: (
        [
            entry
            for owner in states
            if engine._owner_eligible(owner, model)
            for entry in owner.ready[model]
        ],
        True,
    )
    engine._eligible = lambda model, selected, bounded, states: bool(selected)
    engine._ready = lambda model, observed, states: any(
        engine._owner_eligible(owner, model) and owner.ready.get(model)
        for owner in states
    )

    def state():
        call = SimpleNamespace(failure=None, check_local=lambda: None)
        result = _State(
            ["text"],
            "embed",
            call,
            Future(),
            False,
            context=SimpleNamespace(observations={"replica": object()}),
        )
        result.ready["replica"] = deque(
            [
                SimpleNamespace(
                    plan=SimpleNamespace(http_active=0, state=result), record=object()
                )
            ]
        )
        result.cursor = 1
        return result

    requesting, observed = state(), state()
    # This test bypasses normal readiness/packing. Initialize their fair
    # consumer roster so it still isolates the completed-preflight boundary.
    engine._packing_turns.prune([requesting, observed], ("replica",))
    engine._preflights["replica"] = _Preflight(object(), (observed,))
    if owner_status == "failed":
        observed.call.failure = RuntimeError("already failed")
    elif owner_status == "contextless":
        observed.context = None
    elif owner_status == "uncredited":
        observed.retain_result = True
    elif owner_status == "empty":
        observed.ready["replica"].clear()

    # The rotation visits the owner without metadata first immediately after
    # the other owner's observation completion cleared the in-flight marker.
    engine._dispatch([requesting, observed])

    if owner_status == "ready":
        assert observations == []
        assert scheduler.dispatched == ["replica"]
        assert not observed.ready["replica"]
        # A newly registered owner cannot join the already observed round.
        assert len(requesting.ready["replica"]) == 1
    else:
        assert len(observations) == 1
        kind, members, metadata = observations[0]
        assert kind == "catalog_shared"
        assert members == (requesting,)
        assert metadata.members == {"replica": (requesting,)}
        assert scheduler.dispatched == []
