"""Deterministic admission and total-budget tests for the sole scheduler."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from threading import Event, Lock, enumerate as threads
import time

import pytest

from runtime_llama.embedding_scheduler import EmbeddingScheduler
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError


def eventually(check, timeout=3):
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() >= deadline:
            raise AssertionError("scheduler fixture condition not reached")
        time.sleep(0.005)


def unchecked():
    pass


@pytest.fixture
def schedulers():
    created = []

    def make(parallelism=4, queue_size=2, replicas=None):
        scheduler = EmbeddingScheduler(
            parallelism,
            queue_size,
            {"a": 1, "b": 1} if replicas is None else replicas,
        )
        created.append(scheduler)
        return scheduler

    yield make
    for scheduler in created:
        scheduler.close()


@pytest.mark.parametrize(
    "replicas,parallelism,expected",
    [
        ({"a": 1, "b": 1}, 4, (("a", 128), ("b", 128))),
        (
            {"a": 1, "b": 1, "c": 1, "d": 1},
            4,
            (("a", 64), ("b", 64), ("c", 64), ("d", 64)),
        ),
        ({"a": 1, "b": 1, "c": 1, "d": 1}, 2, (("a", 128), ("b", 128))),
        ({"a": 2, "b": 2}, 1, (("a", 256),)),
    ],
)
def test_total_budget_is_balanced_over_distinct_effective_aliases(
    schedulers, replicas, parallelism, expected
):
    scheduler = schedulers(parallelism=parallelism, replicas=replicas)
    planned = scheduler.plan_window(256)
    assert planned == expected
    assert sum(size for _, size in planned) == 256
    assert len({alias for alias, _ in planned}) == len(planned)


def test_global_cap_rotates_distinct_alias_pairs_across_windows(schedulers):
    scheduler = schedulers(parallelism=2, replicas={"a": 1, "b": 1, "c": 1, "d": 1})
    assert scheduler.plan_window(256) == (("a", 128), ("b", 128))
    assert scheduler.plan_window(256) == (("c", 128), ("d", 128))
    assert scheduler.plan_window(256) == (("a", 128), ("b", 128))


@pytest.mark.parametrize(
    "count,expected",
    [
        (255, (("a", 64), ("b", 64), ("c", 64), ("d", 63))),
        (257, (("a", 65), ("b", 64), ("c", 64), ("d", 64))),
        (1, (("a", 1),)),
        (2, (("a", 1), ("b", 1))),
        (3, (("a", 1), ("b", 1), ("c", 1))),
    ],
)
def test_odd_and_small_windows_have_no_empty_or_lost_quota(schedulers, count, expected):
    scheduler = schedulers(replicas={"a": 1, "b": 1, "c": 1, "d": 1})
    assert scheduler.plan_window(count) == expected
    sizes = [size for _, size in expected]
    assert sum(sizes) == count
    assert max(sizes) - min(sizes) <= 1


def test_empty_window_neither_dispatches_nor_consumes_rotation(schedulers):
    scheduler = schedulers()
    assert scheduler.plan_window(1) == (("a", 1),)
    assert scheduler.plan_window(0) == ()
    assert scheduler.plan_window(1) == (("b", 1),)
    assert scheduler.snapshot()["peak_in_flight"] == 0


@pytest.mark.parametrize("value", [-1, True, 1.2, "2", None])
def test_window_count_is_a_nonnegative_integer(schedulers, value):
    with pytest.raises(ValueError, match="text_count"):
        schedulers().plan_window(value)


def test_busy_alias_does_not_change_window_capacity_or_assignment(schedulers):
    scheduler = schedulers()
    started, release = Event(), Event()

    def blocked(alias):
        started.set()
        assert release.wait(3)
        return alias

    active = scheduler.submit(blocked, unchecked, model="a")
    try:
        assert started.wait(2)
        assert scheduler.plan_window(256) == (("a", 128), ("b", 128))
        assert scheduler.snapshot()["replicas"]["a"]["in_flight"] == 1
    finally:
        release.set()
    assert active.result(timeout=2) == "a"


def test_oldest_eligible_job_bypasses_busy_alias_without_breaking_alias_fifo(
    schedulers,
):
    scheduler = schedulers(parallelism=2, queue_size=3)
    release_a, release_b = Event(), Event()
    started_a, started_b, b1_started = Event(), Event(), Event()
    order = []
    lock = Lock()

    def work(label, release=None, started=None):
        def run(alias):
            with lock:
                order.append((label, alias))
            if started is not None:
                started.set()
            if release is not None:
                assert release.wait(3)
            return label

        return run

    a0 = scheduler.submit(work("a0", release_a, started_a), unchecked, model="a")
    b0 = scheduler.submit(work("b0", release_b, started_b), unchecked, model="b")
    try:
        assert started_a.wait(2) and started_b.wait(2)
        a1 = scheduler.submit(work("a1"), unchecked, model="a")
        b1 = scheduler.submit(work("b1", started=b1_started), unchecked, model="b")
        a2 = scheduler.submit(work("a2"), unchecked, model="a")
        assert scheduler.snapshot()["queued"] == 3
        release_b.set()
        assert b1_started.wait(2)
        assert b0.result(timeout=2) == "b0"
        assert b1.result(timeout=2) == "b1"
        assert not a1.done() and not a2.done()
        release_a.set()
        assert [future.result(timeout=2) for future in (a0, a1, a2)] == [
            "a0",
            "a1",
            "a2",
        ]
        assert [label for label, alias in order if alias == "a"] == ["a0", "a1", "a2"]
        assert [label for label, alias in order if alias == "b"] == ["b0", "b1"]
        eventually(lambda: scheduler.snapshot()["in_flight"] == 0)
        assert scheduler.snapshot()["queued"] == 0
        assert scheduler.snapshot()["peak_queued"] == 3
    finally:
        release_a.set()
        release_b.set()


def test_new_free_alias_job_can_bypass_a_full_queue_blocked_on_another_alias(
    schedulers,
):
    scheduler = schedulers(parallelism=2, queue_size=1)
    started, release = Event(), Event()

    def blocked(alias):
        started.set()
        assert release.wait(3)
        return alias

    active = scheduler.submit(blocked, unchecked, model="a")
    try:
        assert started.wait(2)
        waiting = scheduler.submit(lambda alias: alias, unchecked, model="a")
        assert scheduler.snapshot()["queued"] == 1
        free = scheduler.submit(lambda alias: alias, unchecked, model="b")
        assert free.result(timeout=2) == "b"
        assert not waiting.done()
        assert scheduler.snapshot()["peak_queued"] == 1
    finally:
        release.set()
    assert active.result(timeout=2) == "a"
    assert waiting.result(timeout=2) == "a"


def test_zero_queue_backpressures_one_alias_without_blocking_another(schedulers):
    scheduler = schedulers(parallelism=2, queue_size=0)
    started, release, entering, cancelled = Event(), Event(), Event(), Event()

    def blocked(alias):
        started.set()
        assert release.wait(3)
        return alias

    def check_waiter():
        entering.set()
        if cancelled.is_set():
            raise LLMCancelledError()

    active = scheduler.submit(blocked, unchecked, model="a")
    with ThreadPoolExecutor(max_workers=1) as callers:
        try:
            assert started.wait(2)
            waiting = callers.submit(
                scheduler.submit, lambda alias: alias, check_waiter, model="a"
            )
            assert entering.wait(2)
            assert not waiting.done()
            other = scheduler.submit(lambda alias: alias, unchecked, model="b")
            assert other.result(timeout=2) == "b"
            assert scheduler.snapshot()["queued"] == 0
            cancelled.set()
            with pytest.raises(LLMCancelledError):
                waiting.result(timeout=2)
            assert scheduler.snapshot()["peak_queued"] == 0
        finally:
            cancelled.set()
            release.set()
    assert active.result(timeout=2) == "a"


@pytest.mark.parametrize("queue_size", [0, 1])
def test_admission_deadline_expires_while_capacity_is_blocked(schedulers, queue_size):
    scheduler = schedulers(parallelism=1, queue_size=queue_size)
    started, release = Event(), Event()

    def blocked(alias):
        started.set()
        assert release.wait(3)
        return alias

    active = scheduler.submit(blocked, unchecked, model="a")
    queued = None
    try:
        assert started.wait(2)
        if queue_size:
            queued = scheduler.submit(lambda alias: alias, unchecked, model="b")
        deadline = time.monotonic() + 0.08

        def check_deadline():
            if time.monotonic() >= deadline:
                raise LLMTimeoutError("fixture_admission_timeout")

        with ThreadPoolExecutor(max_workers=1) as callers:
            waiting = callers.submit(
                scheduler.submit, lambda alias: alias, check_deadline, model="a"
            )
            with pytest.raises(LLMTimeoutError, match="fixture_admission_timeout"):
                waiting.result(timeout=2)
        assert scheduler.snapshot()["queued"] == queue_size
    finally:
        release.set()
    assert active.result(timeout=2) == "a"
    if queued is not None:
        assert queued.result(timeout=2) == "b"


def test_cancelled_admitted_job_does_not_run_or_block_its_successor(schedulers):
    scheduler = schedulers(parallelism=1, queue_size=2)
    started, release, cancel = Event(), Event(), Event()
    ran = []

    def blocked(alias):
        started.set()
        assert release.wait(3)
        return alias

    def check_cancel():
        if cancel.is_set():
            raise LLMCancelledError()

    active = scheduler.submit(blocked, unchecked, model="a")
    try:
        assert started.wait(2)
        first = scheduler.submit(
            lambda alias: ran.append("cancelled"), check_cancel, model="a"
        )
        second = scheduler.submit(
            lambda alias: ran.append("second"), unchecked, model="a"
        )
        cancel.set()
        release.set()
        assert active.result(timeout=2) == "a"
        with pytest.raises(LLMCancelledError):
            first.result(timeout=2)
        assert second.result(timeout=2) is None
        assert ran == ["second"]
        eventually(lambda: scheduler.snapshot()["in_flight"] == 0)
        assert scheduler.snapshot()["queued"] == 0
    finally:
        release.set()


def test_future_cancellation_is_safe_during_pump_and_close(schedulers):
    scheduler = schedulers(parallelism=1, queue_size=2)
    started, release = Event(), Event()
    ran = []

    def blocked(alias):
        started.set()
        assert release.wait(3)
        return alias

    active = scheduler.submit(blocked, unchecked, model="a")
    try:
        assert started.wait(2)
        cancelled = scheduler.submit(
            lambda alias: ran.append(alias), unchecked, model="a"
        )
        assert cancelled.cancel()
        release.set()
        assert active.result(timeout=2) == "a"
        with pytest.raises(CancelledError):
            cancelled.result(timeout=2)
        eventually(lambda: scheduler.snapshot()["queued"] == 0)
        assert ran == []
    finally:
        release.set()
    scheduler.close()


def test_configured_alias_capacity_and_global_limit_are_shared(schedulers):
    scheduler = schedulers(parallelism=3, queue_size=1, replicas={"a": 2, "b": 1})
    release = Event()

    def blocked(alias):
        assert release.wait(3)
        return alias

    first = scheduler.submit(blocked, unchecked, model="a")
    second = scheduler.submit(blocked, unchecked, model="a")
    queued = scheduler.submit(blocked, unchecked, model="a")
    third = scheduler.submit(blocked, unchecked, model="b")
    try:
        snapshot = scheduler.snapshot()
        assert snapshot["in_flight"] == 3 and snapshot["queued"] == 1
        assert snapshot["replicas"]["a"]["in_flight"] == 2
        assert snapshot["replicas"]["b"]["in_flight"] == 1
        release.set()
        assert [
            future.result(timeout=2) for future in (first, second, queued, third)
        ] == [
            "a",
            "a",
            "a",
            "b",
        ]
        eventually(lambda: scheduler.snapshot()["in_flight"] == 0)
        snapshot = scheduler.snapshot()
        assert snapshot["peak_in_flight"] == 3
        assert snapshot["replicas"]["a"]["peak_in_flight"] == 2
        assert snapshot["replicas"]["b"]["peak_in_flight"] == 1
    finally:
        release.set()


def test_job_failure_does_not_contaminate_independent_work(schedulers):
    scheduler = schedulers(parallelism=2)
    failure = LLMBackendError("fixture_failure")

    def broken(alias):
        raise failure

    first = scheduler.submit(broken, unchecked, model="a")
    second = scheduler.submit(lambda alias: alias, unchecked, model="b")
    with pytest.raises(LLMBackendError) as caught:
        first.result(timeout=2)
    assert caught.value is failure
    assert second.result(timeout=2) == "b"
    assert (
        scheduler.submit(lambda alias: alias, unchecked, model="a").result(timeout=2)
        == "a"
    )


def test_close_wakes_backpressured_submitters_fails_queue_and_drains_active(schedulers):
    scheduler = schedulers(parallelism=1, queue_size=1)
    before_threads = {thread.ident for thread in threads()}
    started, release, entering = Event(), Event(), Event()

    def blocked(alias):
        started.set()
        assert release.wait(3)
        return alias

    def entering_check():
        entering.set()

    active = scheduler.submit(blocked, unchecked, model="a")
    with ThreadPoolExecutor(max_workers=2) as callers:
        try:
            assert started.wait(2)
            queued = scheduler.submit(lambda alias: alias, unchecked, model="b")
            waiting = callers.submit(
                scheduler.submit, lambda alias: alias, entering_check, model="a"
            )
            assert entering.wait(2)
            closing = callers.submit(scheduler.close)
            eventually(lambda: scheduler._closed)
            with pytest.raises(LLMBackendError, match="closed"):
                waiting.result(timeout=2)
            with pytest.raises(LLMBackendError, match="closed"):
                queued.result(timeout=2)
            assert not closing.done()
            release.set()
            assert active.result(timeout=2) == "a"
            closing.result(timeout=2)
        finally:
            release.set()
    assert scheduler.snapshot()["in_flight"] == scheduler.snapshot()["queued"] == 0
    assert not [
        thread
        for thread in threads()
        if thread.ident not in before_threads
        and thread.name.startswith("runtime_llama_embedding")
    ]
    scheduler.close()
    with pytest.raises(LLMBackendError, match="closed"):
        scheduler.plan_window(1)
    with pytest.raises(LLMBackendError, match="closed"):
        scheduler.submit(lambda alias: alias, unchecked, model="a")


def test_unknown_affine_alias_is_a_typed_error(schedulers):
    scheduler = schedulers()
    with pytest.raises(LLMBackendError, match="unknown_replica"):
        scheduler.submit(lambda alias: alias, unchecked, model="missing")
    assert scheduler.snapshot()["in_flight"] == scheduler.snapshot()["queued"] == 0


@pytest.mark.parametrize("failed", [False, True])
def test_future_completion_and_callbacks_observe_released_quota_permits(
    schedulers, failed
):
    scheduler = schedulers(parallelism=1)
    started, release, observed = Event(), Event(), Event()
    completion_states = []
    failure = LLMBackendError("fixture_quota_failure")

    def work(alias):
        started.set()
        assert release.wait(3)
        if failed:
            raise failure
        return alias

    def capture_completion(future):
        completion_states.append(scheduler.snapshot())
        observed.set()

    future = scheduler.submit(work, unchecked, model="a")
    try:
        assert started.wait(2)
        future.add_done_callback(capture_completion)
        release.set()
        if failed:
            with pytest.raises(LLMBackendError) as caught:
                future.result(timeout=2)
            assert caught.value is failure
        else:
            assert future.result(timeout=2) == "a"
        assert observed.wait(2)
        assert len(completion_states) == 1
        assert completion_states[0]["in_flight"] == 0
        assert completion_states[0]["replicas"]["a"]["in_flight"] == 0
        assert completion_states[0]["queued"] == 0
    finally:
        release.set()
