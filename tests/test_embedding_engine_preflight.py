"""Fresh observation arbitration between concurrent window owners."""

from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from runtime_llama.embedding_engine import PersistentEmbeddingEngine, _State


class ImmediateScheduler:
    def __init__(self):
        self.available = True
        self.dispatched = []

    def has_capacity(self, model):
        return self.available

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
    engine._task = lambda executor, kind, state, model, run: observations.append(
        (kind, state, model)
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
    engine._eligible = lambda model, selected, bounded, states: True

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
        return result

    requesting, observed = state(), state()
    observed.preflight["replica"] = object()
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
    else:
        assert observations == [("observe", requesting, "replica")]
        assert scheduler.dispatched == []
