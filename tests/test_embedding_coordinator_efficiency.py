"""Event/deadline coordinator liveness and equivalent local plan accounting."""

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread
from types import SimpleNamespace
import time

import pytest

from runtime_llama.errors import LLMCancelledError, LLMTimeoutError

from test_embedding_contracts import (
    ContractClient,
    configured,
    eventually,
    vector,
)
from test_embedding_progressive import BatchClock


def owner(*, deadline, cancel=None, checking=False, checked_at=0):
    return SimpleNamespace(
        finishing=False,
        checking=checking,
        checked_at=checked_at,
        call=SimpleNamespace(deadline=deadline, cancel=cancel, failure=None),
    )


def test_next_wait_uses_deadline_without_periodic_ten_millisecond_poll():
    adapter = configured(ContractClient(1), embedding_batch_wait_sec=0)
    engine = adapter._engine
    try:
        engine._states.append(owner(deadline=time.monotonic() + 10))
        assert 9 < engine._next_wait() <= 10
        engine._states.clear()
        assert engine._next_wait() is None
    finally:
        engine._states.clear()
        adapter.close()


def test_callback_timer_is_finite_and_not_polled_when_already_checking():
    adapter = configured(ContractClient(1), embedding_batch_wait_sec=0)
    engine = adapter._engine
    state = owner(
        deadline=time.monotonic() + 10,
        cancel=lambda: False,
        checked_at=time.monotonic(),
    )
    try:
        engine._states.append(state)
        assert 0 < engine._next_wait() <= 0.05
        state.checking = True
        assert engine._next_wait() > 9
        state.checking = False
        engine._control_active = engine.control_workers
        assert engine._next_wait() > 9
        engine._control_active = 0
        state.checked_at = 0
        assert engine._next_wait() == 0
    finally:
        engine._states.clear()
        adapter.close()


@pytest.mark.parametrize("custom_clock", [False, True])
def test_coalescing_uses_actual_deadline_or_finite_custom_clock_probe(custom_clock):
    adapter = configured(ContractClient(1), embedding_batch_wait_sec=1)
    engine = adapter._engine
    clock = BatchClock()
    if custom_clock:
        adapter._batch_now = clock
    try:
        engine._lanes["embed_0"] = {
            "owner": deque([SimpleNamespace(ready_at=adapter._batch_now())])
        }
        engine.adapter._scheduler.has_capacity = lambda _model: True
        delay = engine._next_wait()
        assert 0 < delay <= (0.05 if custom_clock else 1)
        if not custom_clock:
            assert delay > 0.9
        else:
            clock.advance(1)
            # Expired timers cannot spin while waiting for an external resource.
            assert engine._next_wait() is None
        engine._observing.add("embed_0")
        assert engine._next_wait() is None
    finally:
        engine._lanes.clear()
        adapter.close()


def test_metadata_notification_immediately_before_wait_is_not_lost(monkeypatch):
    adapter = configured(ContractClient(1), embedding_batch_wait_sec=0, timeout=5)
    engine = adapter._engine
    start, update = adapter._engine_start_shared, engine._update_view
    attempts = []
    notified = Event()

    def first_busy(calls):
        attempts.append(1)
        return None if len(attempts) == 1 else start(calls)

    def notify_before_wait():
        if not notified.is_set() and any(
            state.blocked_at is not None for state in engine._states
        ):
            notified.set()
            engine.metadata_available()
        update()

    monkeypatch.setattr(adapter, "_engine_start_shared", first_busy)
    monkeypatch.setattr(engine, "_update_view", notify_before_wait)
    try:
        with ThreadPoolExecutor(1) as callers:
            pending = callers.submit(adapter.embed_texts, ["one"])
            assert notified.wait(1)
            assert pending.result(1) == [vector("one")]
        assert len(attempts) == 2
    finally:
        adapter.close()


def test_active_http_without_callbacks_waits_for_event_instead_of_busy_loop(
    monkeypatch,
):
    client = ContractClient(1)
    client.embed_gate.clear()
    adapter = configured(client, embedding_batch_wait_sec=0, timeout=5)
    engine = adapter._engine
    wait = engine._condition.wait
    observed = []
    sleeping = Event()

    def capture_wait(timeout=None):
        if engine._http_active:
            observed.append(timeout)
            sleeping.set()
        return wait(timeout)

    monkeypatch.setattr(engine._condition, "wait", capture_wait)
    try:
        with ThreadPoolExecutor(1) as callers:
            pending = callers.submit(adapter.embed_texts, ["one"])
            assert client.embed_started.wait(1)
            assert sleeping.wait(1)
            assert observed and observed[-1] > 4
            client.embed_gate.set()
            assert pending.result(2) == [vector("one")]
    finally:
        client.embed_gate.set()
        adapter.close()


@pytest.mark.parametrize("reason", ["cancel", "deadline"])
def test_unnotifiable_cancel_and_deadline_retire_blocked_admission(reason, monkeypatch):
    adapter = configured(ContractClient(1), embedding_batch_wait_sec=0, timeout=5)
    engine = adapter._engine
    blocked, cancelled = Event(), Event()
    probes = []

    def start_busy(_calls):
        blocked.set()
        return None

    def cancel():
        probes.append(current_thread().name)
        return cancelled.is_set()

    monkeypatch.setattr(adapter, "_engine_start_shared", start_busy)
    try:
        with ThreadPoolExecutor(1) as callers:
            pending = callers.submit(
                adapter.embed_texts,
                ["one"],
                cancel=cancel if reason == "cancel" else None,
                timeout=0.12 if reason == "deadline" else 5,
            )
            assert blocked.wait(1)
            cancelled.set()
            error = LLMCancelledError if reason == "cancel" else LLMTimeoutError
            with pytest.raises(error):
                pending.result(1)
        assert not engine._states
        if reason == "cancel":
            assert probes and all("coordinator" not in name for name in probes)
    finally:
        adapter.close()


def test_idle_adapter_resumes_and_concurrent_close_drains_once():
    client = ContractClient(1)
    adapter = configured(client, embedding_batch_wait_sec=0)
    engine = adapter._engine
    try:
        assert adapter.embed_texts(["first"]) == [vector("first")]
        eventually(lambda: not engine._states)
        assert adapter.embed_texts(["second"]) == [vector("second")]
        with ThreadPoolExecutor(2) as callers:
            closes = [callers.submit(adapter.close) for _ in range(2)]
            for pending in closes:
                pending.result(2)
        assert not engine._thread.is_alive()
        assert not engine._plan_counts and not engine._preparing_by_model
    finally:
        adapter.close()


def test_unchanged_diagnostics_preserve_snapshot_and_changed_counters_publish():
    adapter = configured(ContractClient(1))
    engine = adapter._engine
    try:
        before = engine._view
        engine._update_view()
        assert engine._view is before
        engine._packing_candidate_visits += 1
        engine._update_view()
        assert engine._view is not before
        assert engine.snapshot()["packing_candidate_visits"] == 1
    finally:
        adapter.close()


def test_unverified_plan_does_not_compute_or_hash_deduplication_keys(monkeypatch):
    adapter = configured(ContractClient(1))
    engine = adapter._engine
    observed = SimpleNamespace(identity_key=None, tokenizer_build=None)
    state = SimpleNamespace(
        descriptors=deque([("embed_0", 0, 3, 3)]),
        inputs=["repeat"] * 3,
        context=SimpleNamespace(observations={"embed_0": observed}),
        plans=[],
    )

    def forbidden_key(*_args):
        raise AssertionError("unverified rows must not build deduplication keys")

    monkeypatch.setattr(engine, "_borrowed_key", forbidden_key)
    try:
        engine._plan(state)
        assert list(state.plans[0].groups) == [(0, [0]), (1, [1]), (2, [2])]
        assert engine._plan_counts == {"embed_0": 1}
        assert engine._retained_plan_count == 1
    finally:
        adapter.close()


@pytest.mark.parametrize("verified", [False, True])
def test_plan_ledgers_preserve_duplicate_results_and_drain_across_owners(
    verified, monkeypatch
):
    # Explicit aliases permit the unverified client without asserting that
    # unrelated unknown aliases have the same model weights.
    client = ContractClient(1, verified=verified)
    adapter = configured(
        client,
        model="embed_0" if not verified else "contract-embedding.gguf",
        embedding_replica_discovery=verified,
        batch_size=4,
        embedding_request_parallelism=1,
        token_count_parallelism=2,
        embedding_batch_wait_sec=0,
        timeout=5,
    )
    engine = adapter._engine
    update = engine._update_view
    checks = []

    def checked_view():
        expected = Counter(
            plan.model for state in engine._states for plan in state.plans
        )
        preparing = Counter()
        for state in engine._states:
            for plan in state.plans:
                preparing[plan.model] += plan.preparing
        preparing = +preparing
        assert engine._plan_counts == expected
        assert engine._retained_plan_count == sum(expected.values())
        assert engine._preparing_by_model == preparing
        checks.append(1)
        update()

    monkeypatch.setattr(engine, "_update_view", checked_view)
    incoming = [[f"owner{owner}"] * 4 + [f"tail{owner}"] for owner in range(5)]
    try:
        with ThreadPoolExecutor(5) as callers:
            futures = [callers.submit(adapter.embed_texts, row) for row in incoming]
            assert [pending.result(3) for pending in futures] == [
                [vector(text) for text in row] for row in incoming
            ]
        assert checks and not engine._plan_counts and not engine._preparing_by_model
        assert Counter(
            text for _, row in client.embedding_calls for text in row
        ) == Counter(text for row in incoming for text in row)
        if not verified:
            assert len(client.token_calls) == sum(map(len, incoming))
        else:
            assert len(client.token_calls) < sum(map(len, incoming))
        assert engine.snapshot()["active_plans"] == 0
    finally:
        adapter.close()
