"""Replica-width plans, original leases and independent fair dispatch turns."""

from threading import Event
from concurrent.futures import ThreadPoolExecutor

import pytest

from runtime_llama.embedding_scheduler import EmbeddingScheduler
from runtime_llama.errors import LLMBackendError, LLMCancelledError


def unchecked():
    pass


@pytest.mark.parametrize("guard,effective", [(None, 4), (1, 1), (2, 2), (9, 4)])
def test_verified_replica_count_sets_capacity_without_changing_quotas(guard, effective):
    aliases = {f"embedding_{index}": 4 for index in range(4)}
    scheduler = EmbeddingScheduler(guard, 0, aliases)
    try:
        assert scheduler.parallelism == effective
        assert scheduler._executor._max_workers == effective
        assert scheduler.plan_window(256) == tuple((alias, 64) for alias in aliases)
        assert scheduler.text_cap(256) == 64
        status = scheduler.snapshot()
        assert status["requested_parallelism"] == guard
        assert status["effective_parallelism"] == effective
        assert status["replica_count"] == 4
        assert all(
            entry["max_in_flight"] == 1
            and entry["requested_max_in_flight"] == 4
            and entry["capacity_reason"] == "one_original_post_per_replica"
            for entry in status["replicas"].values()
        )
    finally:
        scheduler.close()


def test_auto_uses_four_independent_original_leases():
    scheduler = EmbeddingScheduler(None, 0, dict.fromkeys("abcd", 4))
    started = {alias: Event() for alias in "abcd"}
    release = Event()

    def run(alias):
        started[alias].set()
        assert release.wait(3)
        return alias

    try:
        futures = [
            scheduler.try_dispatch(run, unchecked, model=alias) for alias in "abcd"
        ]
        assert all(future is not None for future in futures)
        assert all(event.wait(2) for event in started.values())
        assert scheduler.snapshot()["in_flight"] == 4
        assert all(
            scheduler.try_dispatch(run, unchecked, model=alias) is None
            for alias in "abcd"
        )
        release.set()
        assert [future.result(timeout=2) for future in futures] == list("abcd")
        assert scheduler.snapshot()["peak_in_flight"] == 4
    finally:
        release.set()
        scheduler.close()


def test_dispatch_reads_failed_attempts_and_quota_plans_do_not_consume_turns():
    scheduler = EmbeddingScheduler(1, 0, dict.fromkeys("abc", 1))
    started, release = Event(), Event()

    def run(alias):
        started.set()
        assert release.wait(3)
        return alias

    try:
        assert scheduler.dispatch_order() == ("a", "b", "c")
        assert scheduler.plan_window(1) == (("a", 1),)
        assert scheduler.plan_window(1) == (("b", 1),)
        assert scheduler.dispatch_order() == ("a", "b", "c")
        future = scheduler.try_dispatch(run, unchecked, model="a")
        assert future is not None and started.wait(2)
        assert scheduler.dispatch_order() == ("b", "c", "a")
        assert scheduler.dispatch_order(["a", "stale", "c", "a"]) == ("c", "a")
        assert not scheduler.has_capacity("b")
        assert scheduler.try_dispatch(run, unchecked, model="b") is None
        assert scheduler.dispatch_order() == ("b", "c", "a")
        release.set()
        assert future.result(timeout=2) == "a"

        def cancelled():
            raise LLMCancelledError()

        with pytest.raises(LLMCancelledError):
            scheduler.try_dispatch(run, cancelled, model="b")
        assert scheduler.dispatch_order() == ("b", "c", "a")
        next_future = scheduler.try_dispatch(lambda alias: alias, unchecked, model="b")
        assert next_future is not None and next_future.result(timeout=2) == "b"
        assert scheduler.dispatch_order() == ("c", "a", "b")
    finally:
        release.set()
        scheduler.close()


def test_guard_one_does_not_starve_queued_peer_and_keeps_fifo_within_alias():
    scheduler = EmbeddingScheduler(1, 3, {"a": 1, "b": 1})
    started, release = Event(), Event()
    order = []

    def first(alias):
        started.set()
        assert release.wait(3)
        order.append((alias, 0))

    def following(number):
        def run(alias):
            order.append((alias, number))

        return run

    try:
        active = scheduler.submit(first, unchecked, model="a")
        assert started.wait(2)
        a1 = scheduler.submit(following(1), unchecked, model="a")
        a2 = scheduler.submit(following(2), unchecked, model="a")
        b1 = scheduler.submit(following(1), unchecked, model="b")
        release.set()
        for future in (active, a1, a2, b1):
            future.result(timeout=2)
        assert order == [("a", 0), ("b", 1), ("a", 1), ("a", 2)]
        assert scheduler.snapshot()["peak_in_flight"] == 1
    finally:
        release.set()
        scheduler.close()


def test_no_verified_replicas_is_unavailable_without_a_zero_worker_executor():
    scheduler = EmbeddingScheduler(None, 0, {})
    try:
        assert scheduler.parallelism == 0
        assert scheduler.dispatch_order() == ()
        assert scheduler.plan_window(0) == ()
        assert scheduler.snapshot()["capacity_reason"] == "unavailable"
        with pytest.raises(LLMBackendError, match="no_usable_replica"):
            scheduler.plan_window(1)
        with pytest.raises(LLMBackendError, match="no_usable_replica"):
            scheduler.text_cap(256)
        assert not scheduler.has_capacity("a")
    finally:
        scheduler.close()


def test_closed_topology_epoch_reports_no_dispatch_turn_instead_of_raising():
    scheduler = EmbeddingScheduler(None, 0, {"a": 1, "b": 1})
    captured_order = scheduler.dispatch_order
    scheduler.close()
    assert captured_order() == captured_order(["a", "stale"]) == ()
    assert not scheduler.has_capacity("a")


def test_topology_replacement_waits_for_old_original_transport_drain():
    old = EmbeddingScheduler(None, 0, {"a": 1})
    started, release, closing_started = Event(), Event(), Event()

    def run(alias):
        started.set()
        assert release.wait(3)
        return alias

    try:
        future = old.submit(run, unchecked, model="a")
        assert started.wait(2)
        with pytest.raises(LLMBackendError, match="scheduler_topology_changed"):
            old.update_limits({"a": 1, "b": 1})
        assert old.parallelism == 1 and not future.done()

        def close_old():
            closing_started.set()
            old.close()

        with ThreadPoolExecutor(max_workers=1) as caller:
            closing = caller.submit(close_old)
            assert closing_started.wait(2)
            assert not closing.done()
            release.set()
            assert future.result(timeout=2) == "a"
            closing.result(timeout=2)
        replacement = EmbeddingScheduler(None, 0, {"a": 1, "b": 1})
        try:
            assert replacement.parallelism == 2
            assert replacement.plan_window(256) == (("a", 128), ("b", 128))
        finally:
            replacement.close()
    finally:
        release.set()
        old.close()


@pytest.mark.parametrize("value", [0, -1, True, False, "2", 1.5])
def test_guard_rejects_nonpositive_or_noninteger_values(value):
    with pytest.raises(ValueError, match="parallelism"):
        EmbeddingScheduler(value, 0, {"a": 1})


def test_submission_failure_rolls_back_lease_peaks_and_alias_turn(monkeypatch):
    scheduler = EmbeddingScheduler(None, 0, {"a": 1, "b": 1})
    original_submit = scheduler._executor.submit
    failure = RuntimeError("fixture cannot start worker")
    orphans, calls = [], []

    def fail_after_enqueue(run, job):
        # The real executor queues work before trying to create a thread. A
        # later worker may still encounter that queued callable after failure.
        orphans.append((run, job))
        raise failure

    try:
        monkeypatch.setattr(scheduler._executor, "submit", fail_after_enqueue)
        with pytest.raises(
            LLMBackendError, match="scheduler_submission_failed"
        ) as caught:
            scheduler.try_dispatch(
                lambda alias: calls.append(alias), unchecked, model="a"
            )
        assert caught.value.__cause__ is failure
        status = scheduler.snapshot()
        assert status["in_flight"] == status["peak_in_flight"] == 0
        assert status["replicas"]["a"]["in_flight"] == 0
        assert status["replicas"]["a"]["peak_in_flight"] == 0
        assert scheduler.dispatch_order() == ("a", "b")
        run, job = orphans.pop()
        run(job)
        assert not calls and scheduler.snapshot()["in_flight"] == 0
        monkeypatch.setattr(scheduler._executor, "submit", original_submit)
        following = scheduler.try_dispatch(lambda alias: alias, unchecked, model="a")
        assert following is not None and following.result(timeout=2) == "a"
        assert scheduler.dispatch_order() == ("b", "a")
    finally:
        scheduler.close()
    assert scheduler.snapshot()["in_flight"] == 0


@pytest.mark.parametrize("previous_failed", [False, True])
def test_pump_submission_failure_preserves_previous_future_and_drains_queue(
    monkeypatch, previous_failed
):
    scheduler = EmbeddingScheduler(1, 2, {"a": 1, "b": 1})
    started, release = Event(), Event()
    previous_error = LLMBackendError("fixture_previous_job_failed")
    submission_error = RuntimeError("fixture executor is broken")
    orphans, calls = [], []

    def active(alias):
        started.set()
        assert release.wait(3)
        if previous_failed:
            raise previous_error
        return alias

    def fail_after_enqueue(run, job):
        orphans.append((run, job))
        raise submission_error

    try:
        previous = scheduler.submit(active, unchecked, model="a")
        assert started.wait(2)
        first = scheduler.submit(
            lambda alias: calls.append(alias), unchecked, model="a"
        )
        second = scheduler.submit(
            lambda alias: calls.append(alias), unchecked, model="b"
        )
        monkeypatch.setattr(scheduler._executor, "submit", fail_after_enqueue)
        release.set()
        if previous_failed:
            with pytest.raises(LLMBackendError) as caught:
                previous.result(timeout=2)
            assert caught.value is previous_error
        else:
            assert previous.result(timeout=2) == "a"
        for future in (first, second):
            with pytest.raises(
                LLMBackendError, match="scheduler_submission_failed"
            ) as caught:
                future.result(timeout=2)
            assert caught.value.__cause__ is submission_error
        assert scheduler.snapshot()["in_flight"] == scheduler.snapshot()["queued"] == 0
        assert scheduler.dispatch_order() == ("b", "a")
        for run, job in orphans:
            run(job)
        assert not calls and scheduler.snapshot()["in_flight"] == 0
    finally:
        release.set()
        scheduler.close()
