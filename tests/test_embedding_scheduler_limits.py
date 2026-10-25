"""Capacity refresh for the one ready-job scheduler preserves drain semantics."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from runtime_llama.embedding_scheduler import EmbeddingScheduler
from runtime_llama.errors import LLMBackendError


def unchecked():
    pass


@pytest.fixture
def scheduler():
    instance = EmbeddingScheduler(3, 4, {"a": 2, "b": 1})
    yield instance
    instance.close()


def blocked(started, release, value):
    def run(model):
        started.set()
        assert release.wait(3), "fixture failed to release native-ready job"
        return model, value

    return run


def test_cap_refresh_does_not_overlap_original_jobs_or_replay_them(scheduler):
    first_started, next_started = Event(), Event()
    release_first = Event()
    first = scheduler.submit(
        blocked(first_started, release_first, 0), unchecked, model="a"
    )
    try:
        assert first_started.wait(2)
        following = scheduler.submit(
            lambda model: (next_started.set(), model)[1], unchecked, model="a"
        )
        scheduler.update_limits({"a": 100, "b": 4})
        status = scheduler.snapshot()
        assert status["replicas"]["a"]["in_flight"] == 1
        assert status["replicas"]["a"]["max_in_flight"] == 1
        assert status["replicas"]["a"]["requested_max_in_flight"] == 100
        assert not next_started.is_set()
        release_first.set()
        assert first.result(timeout=2) == ("a", 0)
        assert following.result(timeout=2) == "a"
        assert next_started.is_set()
    finally:
        release_first.set()


def test_increased_observed_slot_count_does_not_admit_second_original_post():
    scheduler = EmbeddingScheduler(2, 2, {"a": 1})
    first_started, second_started, release = Event(), Event(), Event()
    calls = []

    def work(started, number):
        def run(model):
            calls.append(number)
            started.set()
            assert release.wait(3)
            return model, number

        return run

    first = scheduler.submit(work(first_started, 0), unchecked, model="a")
    second = scheduler.submit(work(second_started, 1), unchecked, model="a")
    try:
        assert first_started.wait(2)
        assert not second_started.is_set()
        scheduler.update_limits({"a": 2})
        assert not second_started.is_set()
        assert scheduler.snapshot()["in_flight"] == 1
        release.set()
        assert first.result(timeout=2) == ("a", 0)
        assert second.result(timeout=2) == ("a", 1)
        assert sorted(calls) == [0, 1]
    finally:
        release.set()
        scheduler.close()


def test_capacity_refresh_retains_global_bound_and_affinity():
    scheduler = EmbeddingScheduler(2, 2, {"a": 1, "b": 1})
    release = Event()
    started = [Event() for _ in range(3)]
    try:
        scheduler.update_limits({"a": 100, "b": 100})
        pending = [
            scheduler.submit(blocked(started[i], release, i), unchecked, model=model)
            for i, model in enumerate(("a", "a", "b"))
        ]
        assert started[0].wait(2) and started[2].wait(2)
        assert not started[1].is_set()
        assert scheduler.snapshot()["in_flight"] == 2
        release.set()
        assert [p.result(timeout=2) for p in pending] == [("a", 0), ("a", 1), ("b", 2)]
        assert scheduler.snapshot()["peak_in_flight"] == 2
    finally:
        release.set()
        scheduler.close()


def test_zero_queue_backpressure_survives_a_slot_limit_increase():
    scheduler = EmbeddingScheduler(2, 0, {"a": 1})
    started, release, waiting = Event(), Event(), Event()
    first = scheduler.submit(blocked(started, release, 0), unchecked, model="a")
    try:
        assert started.wait(2)
        with ThreadPoolExecutor(max_workers=1) as caller:
            admitted = caller.submit(
                scheduler.submit,
                lambda model: model,
                waiting.set,
                model="a",
            )
            assert waiting.wait(2)
            scheduler.update_limits({"a": 2})
            assert not admitted.done()
            release.set()
            assert first.result(timeout=2) == ("a", 0)
            assert admitted.result(timeout=2).result(timeout=2) == "a"
        assert scheduler.snapshot()["peak_queued"] == 0
        release.set()
        assert first.result(timeout=2) == ("a", 0)
    finally:
        release.set()
        scheduler.close()


@pytest.mark.parametrize("limits", [{"a": 1}, {"a": 1, "b": 1, "c": 1}, {}])
def test_topology_refresh_requires_adapter_drained_replacement(scheduler, limits):
    previous = scheduler.snapshot()
    with pytest.raises(LLMBackendError, match="scheduler_topology_changed"):
        scheduler.update_limits(limits)
    assert scheduler.snapshot() == previous


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.5, "2", None])
def test_invalid_capacity_does_not_change_existing_limits(scheduler, limit):
    previous = scheduler.snapshot()
    with pytest.raises(ValueError, match="positive integers"):
        scheduler.update_limits({"a": limit, "b": 1})
    assert scheduler.snapshot() == previous


def test_closed_scheduler_rejects_capacity_refresh():
    scheduler = EmbeddingScheduler(1, 0, {"a": 1})
    scheduler.close()
    with pytest.raises(LLMBackendError, match="closed"):
        scheduler.update_limits({"a": 2})


def test_refresh_changes_only_admission_not_total_text_window_distribution(scheduler):
    scheduler.update_limits({"a": 3, "b": 3})
    assert scheduler.plan_window(255) == (("a", 128), ("b", 127))


def test_nonblocking_admission_skips_busy_alias_without_head_of_line_blocking():
    scheduler = EmbeddingScheduler(2, 0, {"a": 1, "b": 1})
    first_started, other_started = Event(), Event()
    release_first, release_other = Event(), Event()
    first = scheduler.submit(
        blocked(first_started, release_first, 0), unchecked, model="a"
    )
    other = scheduler.submit(
        blocked(other_started, release_other, 0), unchecked, model="b"
    )
    try:
        assert first_started.wait(2) and other_started.wait(2)
        assert scheduler.try_submit(lambda model: model, unchecked, model="a") is None
        release_other.set()
        assert other.result(timeout=2) == ("b", 0)
        assert scheduler.try_submit(lambda model: model, unchecked, model="a") is None
        free = scheduler.try_submit(lambda model: model, unchecked, model="b")
        assert free is not None and free.result(timeout=2) == "b"
        assert scheduler.snapshot()["peak_queued"] == 0
        release_first.set()
        assert first.result(timeout=2) == ("a", 0)
    finally:
        release_first.set()
        release_other.set()
        scheduler.close()


def test_rejected_nonblocking_jobs_are_not_queued_or_replayed_later():
    scheduler = EmbeddingScheduler(1, 0, {"a": 1})
    started, release = Event(), Event()
    calls = []
    first = scheduler.submit(blocked(started, release, 0), unchecked, model="a")
    try:
        assert started.wait(2)
        for _ in range(20):
            assert (
                scheduler.try_submit(
                    lambda model: calls.append(model), unchecked, model="a"
                )
                is None
            )
        assert calls == []
        release.set()
        assert first.result(timeout=2) == ("a", 0)
        admitted = scheduler.try_submit(
            lambda model: calls.append(model), unchecked, model="a"
        )
        assert admitted is not None
        admitted.result(timeout=2)
        assert calls == ["a"]
        assert scheduler.snapshot()["peak_queued"] == 0
    finally:
        release.set()
        scheduler.close()


def test_nonblocking_admission_preserves_fifo_for_already_queued_same_alias():
    scheduler = EmbeddingScheduler(2, 2, {"a": 1, "b": 1})
    started, release = Event(), Event()
    calls = []
    first = scheduler.submit(blocked(started, release, 0), unchecked, model="a")
    try:
        assert started.wait(2)
        older = scheduler.submit(
            lambda model: calls.append((model, "older")), unchecked, model="a"
        )
        newer = scheduler.try_submit(
            lambda model: calls.append((model, "newer")), unchecked, model="a"
        )
        assert newer is not None
        free = scheduler.try_submit(
            lambda model: calls.append((model, "free")), unchecked, model="b"
        )
        assert free is not None
        free.result(timeout=2)
        assert calls == [("b", "free")]
        release.set()
        first.result(timeout=2)
        older.result(timeout=2)
        newer.result(timeout=2)
        assert calls == [("b", "free"), ("a", "older"), ("a", "newer")]
        assert scheduler.snapshot()["peak_queued"] == 2
    finally:
        release.set()
        scheduler.close()


def test_nonblocking_admission_keeps_typed_unknown_alias_and_closed_errors():
    scheduler = EmbeddingScheduler(1, 0, {"a": 1})
    try:
        with pytest.raises(LLMBackendError, match="unknown_replica"):
            scheduler.try_submit(lambda model: model, unchecked, model="missing")
    finally:
        scheduler.close()
    with pytest.raises(LLMBackendError, match="closed"):
        scheduler.try_submit(lambda model: model, unchecked, model="a")
