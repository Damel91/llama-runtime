from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event

import pytest

import runtime_llama.embedding_metrics as metrics_module
from runtime_llama.embedding_metrics import (
    EmbeddingHTTPMetrics,
    HTTP_KINDS,
    HTTP_PHASES,
)


def measured_clock(metrics):
    clock = {"wall": 0.0, "cpu": 0.0}
    metrics._clock = lambda: clock["wall"]
    metrics._cpu_clock = lambda: clock["cpu"]
    metrics._cpu_supported = True
    return clock


def test_wall_union_distinguishes_overlap_sum_idle_and_inclusive_thread_cpu():
    metrics = EmbeddingHTTPMetrics()
    clock = measured_clock(metrics)
    with metrics.measure("embedding", "preflight"):
        clock.update(wall=1.0, cpu=0.1)
        with metrics.measure("tokenize", "preparation"):
            clock.update(wall=2.0, cpu=0.2)
            active = metrics.snapshot()
            assert active["total"]["active"] == active["total"]["peak"] == 2
            assert active["total"]["wall_seconds"] == 3.0
            assert active["total"]["wall_union_seconds"] == 2.0
            assert active["total"]["inclusive_thread_cpu_seconds"] is None
            clock.update(wall=3.0, cpu=0.3)
        clock.update(wall=4.0, cpu=0.5)
    clock.update(wall=10.0, cpu=0.6)
    with metrics.measure("catalog", "readiness"):
        clock.update(wall=12.0, cpu=0.7)
    result = metrics.snapshot()
    assert result["total"]["count"] == 3
    assert result["total"]["failed"] == result["total"]["active"] == 0
    assert result["total"]["wall_seconds"] == 8.0
    assert result["total"]["wall_union_seconds"] == 6.0
    assert result["total"]["inclusive_thread_cpu_seconds"] == pytest.approx(0.8)
    assert result["total"]["thread_cpu_samples"] == 3
    assert result["by_kind"]["embedding"]["wall_seconds"] == 4.0
    assert result["by_phase"]["preparation"]["wall_union_seconds"] == 2.0
    assert result["by_kind_phase"]["catalog:readiness"]["count"] == 1


def test_concurrent_workers_share_exact_union_and_peak_without_event_history():
    metrics = EmbeddingHTTPMetrics()
    clock = measured_clock(metrics)
    first, second, release_first, release_second, first_done = (
        Event() for _ in range(5)
    )

    def work(entered, release, finished=None):
        with metrics.measure("embedding", "preflight"):
            entered.set()
            assert release.wait(1)
        if finished is not None:
            finished.set()

    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(work, first, release_first, first_done)
        assert first.wait(1)
        clock["wall"] = 1.0
        b = pool.submit(work, second, release_second)
        assert second.wait(1)
        clock["wall"] = 3.0
        release_first.set()
        assert first_done.wait(1)
        clock["wall"] = 4.0
        release_second.set()
        a.result(1)
        b.result(1)
    total = metrics.snapshot()["total"]
    assert total["count"] == 2
    assert total["active"] == 0
    assert total["peak"] == 2
    assert total["wall_seconds"] == 6.0
    assert total["wall_union_seconds"] == 4.0


@pytest.mark.parametrize(
    "exception", [RuntimeError("private payload"), KeyboardInterrupt()]
)
def test_failure_counters_include_interruption_without_retaining_exception(exception):
    metrics = EmbeddingHTTPMetrics()
    clock = measured_clock(metrics)
    with pytest.raises(type(exception)):
        with metrics.measure("props", "recovery"):
            clock.update(wall=2.0, cpu=0.2)
            raise exception
    result = metrics.snapshot()
    assert result["total"]["failed"] == result["total"]["count"] == 1
    assert result["total"]["active"] == 0
    assert result["total"]["wall_union_seconds"] == 2.0
    assert "private payload" not in json.dumps(result)


@pytest.mark.parametrize(
    "kind,phase",
    [
        ("private-model", "preflight"),
        ("props", "private-path"),
        (None, "preflight"),
        ("props", []),
    ],
)
def test_unknown_labels_fail_without_adding_cardinality(kind, phase):
    metrics = EmbeddingHTTPMetrics()
    with pytest.raises(ValueError):
        with metrics.measure(kind, phase):
            pytest.fail("invalid metric scope must not execute")
    result = metrics.snapshot()
    assert result["total"]["count"] == 0
    assert set(result["by_kind"]) == set(HTTP_KINDS)
    assert set(result["by_phase"]) == set(HTTP_PHASES)
    assert len(result["by_kind_phase"]) == len(HTTP_KINDS) * len(HTTP_PHASES)
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("unsupported", [None, "raises"])
def test_missing_thread_cpu_clock_is_explicit_and_does_not_break_measurement(
    monkeypatch, unsupported
):
    def unavailable():
        raise NotImplementedError

    monkeypatch.setattr(
        metrics_module.time, "thread_time", unavailable if unsupported else None
    )
    metrics = EmbeddingHTTPMetrics()
    with metrics.measure("health", "admission"):
        pass
    result = metrics.snapshot()
    assert result["thread_cpu_supported"] is False
    assert result["total"]["count"] == 1
    assert result["total"]["thread_cpu_samples"] == 0
    assert result["total"]["inclusive_thread_cpu_seconds"] is None


def test_repeated_calls_keep_fixed_buckets_and_snapshots_are_detached():
    metrics = EmbeddingHTTPMetrics()
    for _ in range(1000):
        with metrics.measure("props", "preflight"):
            pass
    first = metrics.snapshot()
    first["by_kind"]["props"]["count"] = -1
    current = metrics.snapshot()
    assert current["total"]["count"] == current["by_kind"]["props"]["count"] == 1000
    assert current["total"]["active"] == 0
    assert len(current["by_kind_phase"]) == 25
    assert not any("events" in key for key in current)
