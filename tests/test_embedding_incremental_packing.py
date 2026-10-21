"""Packing work follows newly ready rows rather than repeatedly walking prefixes."""

from concurrent.futures import Future
import random
from types import SimpleNamespace

import pytest

from runtime_llama.embedding_engine import (
    PersistentEmbeddingEngine,
    _Entry,
    _Preflight,
    _ReadyQueue,
    _State,
)


class PackingScheduler:
    def __init__(self):
        self.available = True
        self.dispatched = []

    def text_cap(self, budget):
        return budget

    def has_capacity(self, model):
        return self.available

    def snapshot(self):
        return {"in_flight": int(not self.available)}

    def try_dispatch(self, run, check, *, model):
        check()
        self.available = False
        self.dispatched.append(model)
        return Future()


def engine_fixture(*, cap=128, request_bytes=1_000_000, native=True):
    scheduler = PackingScheduler()
    adapter = SimpleNamespace(
        embedding_request_parallelism=1,
        token_count_parallelism=1,
        embedding_queue_size=0,
        embedding_dim=2,
        embedding_batch_wait_sec=0,
        embedding_input_format="auto" if native else "text",
        embedding_max_request_bytes=request_bytes,
        batch_size=cap,
        _scheduler=scheduler,
        _client=SimpleNamespace(supports_embedding_token_ids=native),
        _replicas={"model": object()},
        _current_batch_size=lambda size, replica: size,
        _engine_payload_overhead=lambda model: 31,
        _batch_now=lambda: 0.0,
        _engine_batch_call=lambda calls: SimpleNamespace(check_local=lambda: None),
    )
    engine = PersistentEmbeddingEngine(adapter)
    engine._can_produce = lambda model, states: True
    failures = []

    def fail(state, error):
        failures.append(state)
        state.call.failure = error

    engine._fail = fail
    return engine, failures


def owner(observation=None):
    observation = observation or SimpleNamespace(batch_token_limit=None)
    return _State(
        ["borrowed"],
        "embed",
        SimpleNamespace(failure=None, check_local=lambda: None),
        Future(),
        False,
        context=SimpleNamespace(observations={"model": observation}),
    )


def entry(state, *, index=0, tokens=1, limit=None, text=8, ids=4):
    return _Entry(
        SimpleNamespace(state=state, http_active=0),
        index,
        SimpleNamespace(
            token_count=tokens,
            batch_token_limit=limit,
            _text_wire_bytes=text,
            _tokens_wire_bytes=ids,
            _token_ids=(1,) if ids is not None else None,
        ),
        800,
        0.0,
    )


def previous_pack(engine, observation, states):
    """Independent reference to the previous cursor-based fair selection."""
    cap = engine.adapter.batch_size
    owners = [
        state
        for state in states
        if engine._owner_eligible(state, "model") and state.ready.get("model")
    ]
    cursors = dict.fromkeys(owners, 0)
    selected, tokens = [], 0
    budget = observation.batch_token_limit
    text_bytes = token_bytes = 31
    native = engine.adapter.embedding_input_format != "text"
    while owners:
        state = owners.pop(0)
        item = state.ready["model"][cursors[state]]
        cursors[state] += 1
        if cursors[state] < len(state.ready["model"]):
            owners.append(state)
        record = item.record
        limit = record.batch_token_limit
        effective = (
            budget if limit is None else limit if budget is None else min(budget, limit)
        )
        if (
            selected
            and effective is not None
            and tokens + record.token_count > effective
        ):
            return selected, True, []
        use_ids = (
            native and record._tokens_wire_bytes is not None and bool(record._token_ids)
        )
        next_text = text_bytes + record._text_wire_bytes + (2 if selected else 0)
        next_tokens = (
            token_bytes + (record._tokens_wire_bytes or 0) + (2 if selected else 0)
        )
        size = next_tokens if use_ids else next_text
        if size > engine.max_request_bytes:
            return (selected, True, []) if selected else ([], False, [state])
        selected.append(item)
        tokens += record.token_count
        text_bytes, token_bytes, native = next_text, next_tokens, use_ids
        budget = effective
        if (
            len(selected) >= cap
            or (budget is not None and tokens >= budget)
            or size >= engine.max_request_bytes
        ):
            return selected, True, []
    return selected, False, []


def test_iterator_selection_and_summary_readiness_match_1000_fair_packing_cases():
    rng = random.Random(253_190)
    for case in range(1000):
        engine, failures = engine_fixture(
            cap=rng.randint(1, 64),
            request_bytes=rng.randint(35, 900),
            native=bool(case % 2),
        )
        observation = SimpleNamespace(batch_token_limit=rng.choice([None, 40, 70, 200]))
        states = [owner(observation) for _ in range(rng.randint(1, 6))]
        for state in states:
            queue = _ReadyQueue()
            for index in range(rng.randint(0, 20)):
                item = entry(
                    state,
                    index=index,
                    tokens=rng.randint(1, 50),
                    limit=rng.choice([None, 30, 90, 150]),
                    text=rng.randint(2, 100),
                    ids=rng.choice([None, 2, 40, 150]),
                )
                if rng.randrange(8) == 0:
                    item.record._token_ids = ()
                queue.append(item)
            state.ready["model"] = queue
            if rng.randrange(12) == 0:
                state.call.failure = RuntimeError("already failed")
        expected, bounded, rejected = previous_pack(engine, observation, states)
        assert engine._ready("model", observation, states) == bool(
            bounded or rejected
        ), case
        actual, actual_bounded = engine._pack("model", observation, states)
        assert actual == expected and actual_bounded == bounded, case
        assert failures == rejected, case


def test_128_individual_arrivals_do_not_revisit_the_growing_candidate_prefix():
    engine, _ = engine_fixture()
    observation = SimpleNamespace(batch_token_limit=None)
    state = owner(observation)
    state.ready["model"] = _ReadyQueue()
    metadata = []
    engine._shared_task = lambda kind, owners, value, run: metadata.append(value)
    for index in range(128):
        state.ready["model"].append(entry(state, index=index))
        engine._dispatch([state])
        if index < 127:
            assert metadata == []
            assert engine._packing_candidate_visits == 0
    assert len(metadata) == 1
    assert engine._packing_candidate_visits == 128
    engine._observing.clear()
    engine._preflights["model"] = _Preflight(observation, (state,))
    engine._dispatch([state])
    engine._update_view()
    assert engine.adapter._scheduler.dispatched == ["model"]
    assert not state.ready["model"]
    assert engine.snapshot()["packing_candidate_visits"] == 256


def test_ready_ledger_tracks_append_pop_clear_and_bounds_consumed_prefix_storage():
    state = owner()
    queue = _ReadyQueue()
    for index in range(256):
        queue.append(
            entry(
                state,
                index=index,
                tokens=index + 1,
                limit=index + 3,
                ids=None if index % 3 == 0 else index + 2,
            )
        )
    assert queue.tokens == sum(range(1, 257))
    assert queue.text_bytes == 256 * 8
    assert queue.minimum_limit == 3
    for index in range(256):
        assert queue.popleft().index == index
        assert len(queue._id_prefix) <= 2 * len(queue) + 1
        assert len(queue._missing_ids) <= len(queue)
        assert len(queue.limits) <= len(queue)
        assert queue.minimum_limit == (index + 4 if queue else None)
        assert queue.id_prefix_bytes(len(queue)) == sum(
            item.record._tokens_wire_bytes or 0 for item in queue
        )
    assert queue.tokens == queue.text_bytes == 0
    assert queue._id_prefix == [0] and not queue._missing_ids
    queue.append(entry(state, tokens=10, limit=11, ids=None))
    queue.clear()
    assert not queue and not queue.limits and queue.minimum_limit is None
    assert queue.tokens == queue.text_bytes == 0
    assert queue._id_prefix == [0] and queue.first_missing_ids is None


@pytest.mark.parametrize(
    "change", ["owner", "scope", "token_limit", "byte_limit", "capacity", "text_cap"]
)
def test_dispatch_reads_current_scope_limits_capacity_and_cancelled_owners(change):
    engine, _ = engine_fixture(cap=4)
    observation = SimpleNamespace(batch_token_limit=None)
    earlier, later = owner(observation), owner(observation)
    for state in (earlier, later):
        state.ready["model"] = _ReadyQueue(
            [entry(state, index=index, tokens=2, text=10) for index in range(2)]
        )
    engine._can_produce = lambda model, states: False
    # This observation belongs to the owners captured before the catalog GET.
    members = (earlier,) if change == "scope" else (earlier, later)
    engine._preflights["model"] = _Preflight(observation, members)
    metadata = []
    engine._shared_task = lambda kind, owners, value, run: metadata.append(value)
    if change == "owner":
        earlier.call.failure = RuntimeError("cancelled owner")
    elif change == "token_limit":
        observation.batch_token_limit = 3
    elif change == "byte_limit":
        engine.max_request_bytes = 36
    elif change == "capacity":
        engine.adapter._scheduler.available = False
    elif change == "text_cap":
        engine.adapter._current_batch_size = lambda size, replica: 1
    engine._dispatch([later, earlier])
    expected = {
        "owner": (2, 0),
        "scope": (2, 2),
        "token_limit": (2, 1),
        "byte_limit": (2, 1),
        "capacity": (2, 2),
        "text_cap": (2, 1),
    }[change]
    assert (len(earlier.ready["model"]), len(later.ready["model"])) == expected
    assert engine.adapter._scheduler.dispatched == (
        [] if change in {"capacity", "scope"} else ["model"]
    )
    if change == "scope":
        assert len(metadata) == 1
        assert set(metadata[0].members["model"]) == {earlier, later}
        engine._observing.clear()
        engine._preflights["model"] = _Preflight(observation, (earlier, later))
        engine._dispatch([later, earlier])
        assert engine.adapter._scheduler.dispatched == ["model"]
        assert not earlier.ready["model"] and not later.ready["model"]
    else:
        assert metadata == []


def test_native_prefix_boundary_is_detected_before_later_text_fallback():
    engine, _ = engine_fixture(cap=128, request_bytes=100)
    first, second = owner(), owner()
    first.ready["model"] = _ReadyQueue([entry(first, ids=80, text=2)])
    second.ready["model"] = _ReadyQueue([entry(second, ids=None, text=2)])
    # The final text payload fits, but the first native-ID row alone does not.
    assert engine._ready(
        "model", SimpleNamespace(batch_token_limit=None), [first, second]
    )
    selected, bounded = engine._pack(
        "model", SimpleNamespace(batch_token_limit=None), [first, second]
    )
    assert selected == [] and not bounded


def late_owner_round(*, late_status="ready", forced=None):
    engine, _ = engine_fixture()
    engine._can_produce = PersistentEmbeddingEngine._can_produce.__get__(engine)
    observation = SimpleNamespace(batch_token_limit=None)
    members = [owner(observation) for _ in range(3)]
    for state in members:
        state.cursor = len(state.inputs)
        state.ready["model"] = _ReadyQueue(
            entry(state, index=index) for index in range(32)
        )
    rounds = []
    engine._shared_task = lambda kind, owners, value, run: rounds.append(value)
    engine._dispatch(members)
    assert len(rounds) == 1 and set(rounds[0].members["model"]) == set(members)
    later = owner(observation)
    later.cursor = len(later.inputs)
    later.ready["model"] = _ReadyQueue(entry(later, index=index) for index in range(31))
    if late_status != "ready":
        later.context = None
        if late_status == "starting":
            later.starting = True
        elif late_status == "blocked":
            later.blocked_at = (engine._progress, engine._metadata_availability)
        elif late_status == "uncredited":
            later.retain_result = True
        elif late_status == "failed":
            later.call.failure = RuntimeError("cancelled late owner")
    if forced == "text_cap":
        engine.adapter._current_batch_size = lambda size, replica: 96
    elif forced == "pressure":
        engine._preparation_blocked = True
    elif forced == "elapsed_wait":
        engine.adapter.embedding_batch_wait_sec = 0.5
        engine.adapter._batch_now = lambda: 1.0
    engine._observing.clear()
    engine._preflights["model"] = _Preflight(observation, tuple(members))
    return engine, observation, members, later, rounds


def test_owner_arriving_between_metadata_and_post_renews_partial_snapshot_then_sends_127():
    engine, observation, members, later, rounds = late_owner_round()
    engine._dispatch([*members, later])
    # The late owner cannot use the old snapshot, and its 31 rows must not
    # create a separate POST merely because the old membership ended at 96.
    assert engine.adapter._scheduler.dispatched == []
    assert sum(len(state.ready["model"]) for state in members) == 96
    assert len(rounds) == 2
    assert set(rounds[-1].members["model"]) == {*members, later}
    engine._observing.clear()
    engine._preflights["model"] = _Preflight(observation, tuple([*members, later]))
    engine._dispatch([*members, later])
    assert engine.adapter._scheduler.dispatched == ["model"]
    assert not any(state.ready["model"] for state in [*members, later])


@pytest.mark.parametrize("late_status", ["not_started", "starting"])
def test_useful_pending_admission_prevents_a_false_model_tail(late_status):
    engine, observation, members, later, rounds = late_owner_round(
        late_status=late_status
    )
    engine._dispatch([*members, later])
    assert engine.adapter._scheduler.dispatched == []
    assert len(rounds) == 1
    later.context = SimpleNamespace(observations={"model": observation})
    later.starting = False
    engine._dispatch([*members, later])
    assert len(rounds) == 2
    assert set(rounds[-1].members["model"]) == {*members, later}


@pytest.mark.parametrize("late_status", ["blocked", "uncredited", "failed"])
def test_admission_that_needs_existing_owner_drain_does_not_hold_its_tail(late_status):
    engine, _, members, later, rounds = late_owner_round(late_status=late_status)
    engine._dispatch([*members, later])
    assert engine.adapter._scheduler.dispatched == ["model"]
    assert len(rounds) == 1
    assert not any(state.ready["model"] for state in members)
    assert len(later.ready["model"]) == 31


@pytest.mark.parametrize("forced", ["text_cap", "pressure", "elapsed_wait"])
def test_real_boundary_pressure_or_elapsed_optional_wait_wins_over_late_membership(
    forced,
):
    engine, _, members, later, rounds = late_owner_round(forced=forced)
    engine._dispatch([*members, later])
    assert engine.adapter._scheduler.dispatched == ["model"]
    assert len(rounds) == 1
    assert not any(state.ready["model"] for state in members)
    assert len(later.ready["model"]) == 31
