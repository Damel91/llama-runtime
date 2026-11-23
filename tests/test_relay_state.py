"""Generation/request lifetimes independent of sockets and ASGI frameworks."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gc
import json
from threading import Event, Thread
import time
import weakref

import pytest

from runtime_llama.relay.config import RelayConfigStore
from runtime_llama.relay.state import (
    RelayGeneration,
    RelayGenerationClosed,
    RelayGenerationManager,
    RelayRequestRegistry,
)
from runtime_llama.relay.telemetry import RelayEmbeddingTelemetry


class Resource:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def test_configuration_change_keeps_old_leases_and_resources_until_drain():
    store = RelayConfigStore(initial_config={"llm_defaults": {"temperature": 0.1}})
    manager = RelayGenerationManager(store.snapshot())
    old = manager.acquire()
    sibling = manager.acquire()
    resource = old.resource("adapter", Resource)
    assert (
        sibling.resource("adapter", lambda: pytest.fail("duplicate construction"))
        is resource
    )
    manager.replace(store.replace(0, {"llm_defaults": {"temperature": 0.9}}))
    with manager.acquire() as new:
        assert new.snapshot.revision == 1
        assert new.snapshot.config["llm_defaults"]["temperature"] == 0.9
        assert old.snapshot.config["llm_defaults"]["temperature"] == 0.1
    assert resource.closed == 0
    old.release()
    assert resource.closed == 0
    sibling.release()
    sibling.release()
    assert resource.closed == 1
    assert len(manager.diagnostics()["generations"]) == 1
    manager.close()
    assert manager.wait_drained(0)


def test_released_lease_drops_generation_and_disallows_late_resource():
    generation = RelayGeneration(RelayConfigStore().snapshot())
    reference = weakref.ref(generation)
    lease = generation.acquire()
    generation.retire()
    del generation
    assert reference() is not None
    lease.release()
    gc.collect()
    assert reference() is None
    with pytest.raises(RelayGenerationClosed):
        lease.resource("late", Resource)
    with pytest.raises(RelayGenerationClosed):
        _ = lease.snapshot


def test_retired_live_lease_can_finish_lazy_construction():
    generation = RelayGeneration(RelayConfigStore().snapshot())
    lease = generation.acquire()
    generation.retire()
    resource = lease.resource("late-in-existing-call", Resource)
    assert resource.closed == 0
    with pytest.raises(RelayGenerationClosed):
        generation.acquire()
    lease.close()
    assert resource.closed == 1 and generation.wait_drained(0)


def test_current_resource_is_not_retired_merely_when_a_request_finishes():
    manager = RelayGenerationManager(RelayConfigStore().snapshot())
    with manager.acquire() as lease:
        resource = lease.resource("shared-engine", Resource)
    assert resource.closed == 0
    with manager.acquire() as lease:
        assert lease.resource("shared-engine", lambda: None) is resource
    manager.close()
    manager.close()
    assert resource.closed == 1
    with pytest.raises(RelayGenerationClosed):
        manager.acquire()


def test_resource_factory_once_under_concurrent_leases():
    manager = RelayGenerationManager(RelayConfigStore().snapshot())
    built = []

    def get(_index):
        with manager.acquire() as lease:

            def factory():
                result = Resource()
                built.append(result)
                return result

            return lease.resource("adapter", factory)

    with ThreadPoolExecutor(12) as pool:
        resources = list(pool.map(get, range(40)))
    assert len(built) == 1 and all(resource is built[0] for resource in resources)
    manager.close()
    assert built[0].closed == 1


def test_failed_factory_is_not_retained_or_mistaken_for_success():
    manager = RelayGenerationManager(RelayConfigStore().snapshot())
    with manager.acquire() as lease:
        with pytest.raises(RuntimeError):
            lease.resource(
                "adapter", lambda: (_ for _ in ()).throw(RuntimeError("fixture"))
            )
        resource = lease.resource("adapter", Resource)
    manager.close()
    assert resource.closed == 1


def test_shutdown_waits_for_actual_resource_close_and_callback_is_outside_lock():
    started, finish = Event(), Event()
    manager = RelayGenerationManager(RelayConfigStore().snapshot())

    class BlockingResource:
        def close(self):
            started.set()
            assert finish.wait(2)
            assert manager.diagnostics()["closed"]

    lease = manager.acquire()
    lease.resource("blocking", BlockingResource)
    manager.close()
    assert not manager.wait_drained(0)
    thread = Thread(target=lease.release)
    thread.start()
    assert started.wait(2)
    assert not manager.wait_drained(0)
    finish.set()
    thread.join(2)
    assert not thread.is_alive()
    assert manager.wait_drained(0)


def test_failed_close_retains_live_worker_until_explicit_retry_without_traceback():
    manager = RelayGenerationManager(RelayConfigStore().snapshot())
    attempts, traceback_references = [], []

    class WorkerResource:
        def __init__(self):
            self.stop = Event()
            self.thread = Thread(target=self.stop.wait, daemon=True)
            self.thread.start()

        def close(self):
            attempts.append(1)
            if len(attempts) == 1:
                traceback_sentinel = Resource()
                traceback_references.append(weakref.ref(traceback_sentinel))
                raise RuntimeError("a-private-fixture-value")
            self.stop.set()
            self.thread.join(2)
            assert not self.thread.is_alive()

    lease = manager.acquire()
    failing = lease.resource("failing", WorkerResource)
    reference = weakref.ref(failing)
    stop, worker = failing.stop, failing.thread
    resource = lease.resource("second", Resource)
    lease.resource("alias", lambda: resource)
    lease.release()
    del failing
    try:
        manager.close()
        gc.collect()
        assert worker.is_alive()
        assert reference() is not None
        assert traceback_references[0]() is None
        assert not manager.wait_drained(0)
        assert resource.closed == 1
        diagnostics = manager.diagnostics()
        assert len(diagnostics["generations"]) == 1
        generation = diagnostics["generations"][0]
        assert generation["close_error_types"] == ["RuntimeError"]
        assert generation["resources"] == 1
        assert generation["retired"] and not generation["drained"]
        assert "private-fixture" not in json.dumps(diagnostics)

        manager.close()
        manager.close()
        gc.collect()
        assert not worker.is_alive()
        assert reference() is None
        assert resource.closed == 1
        assert len(attempts) == 2
        assert manager.wait_drained(0)
        assert manager.diagnostics()["generations"] == []
    finally:
        stop.set()
        worker.join(2)


def test_concurrent_closes_coalesce_without_automatic_retry_after_failure():
    manager = RelayGenerationManager(RelayConfigStore().snapshot())
    started, finish = Event(), Event()
    attempts = []

    class BlockingResource:
        def close(self):
            attempts.append(1)
            if len(attempts) == 1:
                started.set()
                assert finish.wait(2)
                raise RuntimeError("a-private-fixture-value")

    with manager.acquire() as lease:
        lease.resource("blocking", BlockingResource)
    thread = Thread(target=manager.close)
    thread.start()
    try:
        assert started.wait(2)
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda _index: manager.close(), range(16)))
        assert len(attempts) == 1
        assert not manager.wait_drained(0)
        finish.set()
        thread.join(2)
        assert not thread.is_alive()
        assert len(attempts) == 1
        assert not manager.wait_drained(0)

        manager.close()
        manager.close()
        assert len(attempts) == 2
        assert manager.wait_drained(0)
    finally:
        finish.set()
        thread.join(2)


def test_failed_retired_generation_remains_owned_until_manager_close_retries():
    store = RelayConfigStore()
    manager = RelayGenerationManager(store.snapshot())
    attempts = []

    class FailingResource:
        def close(self):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("fixture")

    with manager.acquire() as lease:
        resource = lease.resource("failing", FailingResource)
        reference = weakref.ref(resource)
    del resource
    manager.replace(store.replace(0, {}))
    gc.collect()
    assert reference() is not None
    diagnostics = manager.diagnostics()
    assert diagnostics["current_revision"] == 1
    assert len(diagnostics["generations"]) == 2
    failed = next(item for item in diagnostics["generations"] if item["revision"] == 0)
    assert failed["resources"] == 1 and not failed["drained"]
    assert failed["close_error_types"] == ["RuntimeError"]
    with manager.acquire() as current:
        assert current.snapshot.revision == 1
    manager.close()
    gc.collect()
    assert reference() is None
    assert len(attempts) == 2
    assert manager.wait_drained(0)


def test_interrupted_close_preserves_failed_and_unattempted_resources_for_retry():
    callbacks, attempts = [], []
    generation = RelayGeneration(
        RelayConfigStore().snapshot(), on_drained=lambda _item: callbacks.append(1)
    )

    class InterruptedResource:
        def close(self):
            attempts.append(1)
            if len(attempts) == 1:
                raise KeyboardInterrupt("a-private-fixture-value")

    with generation.acquire() as lease:
        successful = lease.resource("first", Resource)
        lease.resource("interrupted", InterruptedResource)
        unattempted = lease.resource("last", Resource)
    with pytest.raises(KeyboardInterrupt):
        generation.retire()
    diagnostics = generation.diagnostics()
    assert diagnostics["resources"] == 2 and not diagnostics["drained"]
    assert diagnostics["close_error_types"] == ["KeyboardInterrupt"]
    assert "private-fixture" not in json.dumps(diagnostics)
    assert successful.closed == 1 and unattempted.closed == 0
    assert not generation.wait_drained(0)
    assert callbacks == []

    generation.retire()
    generation.retire()
    assert len(attempts) == 2
    assert successful.closed == 1 and unattempted.closed == 1
    assert callbacks == [1]
    assert generation.wait_drained(0)
    assert generation.diagnostics()["resources"] == 0
    assert generation.diagnostics()["close_error_types"] == []


def test_many_reconfigurations_do_not_retain_old_generations():
    store = RelayConfigStore()
    manager = RelayGenerationManager(store.snapshot())
    references = []
    for index in range(80):
        with manager.acquire() as lease:
            resource = lease.resource("resource", Resource)
            references.append(weakref.ref(resource))
        manager.replace(
            store.replace(index, {"llm_defaults": {"temperature": index / 80}})
        )
    del resource
    gc.collect()
    assert all(reference() is None for reference in references)
    assert len(manager.diagnostics()["generations"]) == 1
    manager.close()
    assert not manager.diagnostics()["generations"]


def test_request_cancel_is_independent_and_completion_retains_no_handle():
    registry = RelayRequestRegistry()
    first = registry.start("chat", "chat-a", 1)
    second = registry.start("chat", "chat-b", 1)
    reference = weakref.ref(first)
    assert registry.cancel(first.request_id)
    assert first.cancelled() and not second.cancelled()
    assert not registry.cancel("not-present")
    request_id = first.request_id
    registry.update(
        request_id,
        backend="strata",
        corrections_used=1,
        reasons=("native_rejected_tool_call",),
    )
    registry.finish(request_id, status="cancelled", terminal_reason="cancelled")
    del first
    gc.collect()
    assert reference() is None
    snapshot = registry.snapshot()
    assert len(snapshot["active"]) == 1
    assert snapshot["last_completed"]["request_id"] == request_id
    assert snapshot["last_completed"]["status"] == "cancelled"
    assert snapshot["last_completed"]["corrections_used"] == 1
    assert "cancel_event" not in json.dumps(snapshot)
    assert registry.finish(request_id) is None
    registry.finish(second.request_id)
    assert registry.snapshot()["last_completed"]["request_id"] == second.request_id
    registry.clear_last_completed()
    assert registry.snapshot() == {"active": [], "last_completed": None}


def test_snapshot_and_update_do_not_share_mutable_counters():
    registry = RelayRequestRegistry()
    request = registry.start("chat", "model", 0)
    usage = {"total_tokens": 5}
    registry.update(request.request_id, usage=usage)
    usage["total_tokens"] = 999
    first = registry.snapshot()
    assert first["active"][0]["usage"]["total_tokens"] == 5
    first["active"][0]["usage"]["total_tokens"] = 123
    assert registry.snapshot()["active"][0]["usage"]["total_tokens"] == 5
    finished = registry.finish(request.request_id)
    finished["usage"]["total_tokens"] = 9
    assert registry.snapshot()["last_completed"]["usage"]["total_tokens"] == 5


def _complete_chat(registry, model="model"):
    request = registry.start("chat", model, 1)
    registry.finish(request.request_id)
    return request.request_id


def test_retention_limit_keeps_only_newest_observations():
    registry = RelayRequestRegistry(retained_observations=3)
    identifiers = [_complete_chat(registry) for _ in range(6)]
    snapshot = registry.telemetry_snapshot()
    assert [row["request_id"] for row in snapshot["recent"]] == list(
        reversed(identifiers[-3:])
    )
    assert snapshot["counters"]["completed"] == 6
    registry.apply_retention(1, None)
    assert [row["request_id"] for row in registry.telemetry_snapshot()["recent"]] == [
        identifiers[-1]
    ]
    assert registry.telemetry_snapshot()["counters"]["completed"] == 6


def test_retention_age_expiry_drops_older_observations():
    registry = RelayRequestRegistry()
    _complete_chat(registry)
    assert len(registry.telemetry_snapshot()["recent"]) == 1
    registry.apply_retention(128, 0.02)
    time.sleep(0.05)
    fresh = _complete_chat(registry)
    snapshot = registry.telemetry_snapshot()
    assert [row["request_id"] for row in snapshot["recent"]] == [fresh]
    assert snapshot["counters"]["completed"] == 2
    registry.apply_retention(128, None)
    assert len(registry.telemetry_snapshot()["recent"]) == 1


def test_clear_recent_empties_only_the_retained_list():
    registry = RelayRequestRegistry()
    finished = _complete_chat(registry)
    active = registry.start("chat", "model", 1)
    before = registry.telemetry_snapshot()
    assert before["recent"] and before["counters"]["completed"] == 1
    registry.clear_recent()
    after = registry.telemetry_snapshot()
    assert after["recent"] == []
    assert after["counters"] == before["counters"]
    assert after["active"]["total"] == 1
    assert not active.cancelled()
    assert registry.snapshot()["last_completed"]["request_id"] == finished


def test_retention_values_are_validated_at_construction_and_apply():
    with pytest.raises(ValueError):
        RelayRequestRegistry(retained_observations=0)
    with pytest.raises(ValueError):
        RelayRequestRegistry(retained_max_age_sec=-1)
    registry = RelayRequestRegistry()
    _complete_chat(registry)
    for limit, age in [
        (0, None),
        (-2, None),
        (12.5, None),
        (True, None),
        (128, 0),
        (128, -1),
        (128, float("inf")),
        (128, "60"),
        (128, True),
    ]:
        with pytest.raises(ValueError):
            registry.apply_retention(limit, age)
    assert len(registry.telemetry_snapshot()["recent"]) == 1


@pytest.mark.parametrize(
    "updates",
    [
        {"messages": []},
        {"last_proposal": {}},
        {"usage": {"raw": {"prompt": "private"}}},
        {"server_timings": {"object": Resource()}},
        {"corrections_used": float("nan")},
    ],
)
def test_registry_rejects_private_or_non_json_metadata_without_retention(updates):
    registry = RelayRequestRegistry()
    request = registry.start("chat", "model", 0)
    original = deepcopy(registry.snapshot())
    with pytest.raises(ValueError):
        registry.update(request.request_id, **updates)
    assert registry.snapshot() == original
    registry.finish(request.request_id)


def test_registry_close_cancels_all_but_keeps_work_registered_until_finish():
    registry = RelayRequestRegistry()
    requests = [registry.start("embedding", f"model-{i}", 0) for i in range(3)]
    registry.close()
    registry.close()
    assert all(request.cancelled() for request in requests)
    assert len(registry.snapshot()["active"]) == 3
    with pytest.raises(RuntimeError, match="closed"):
        registry.start("chat", "another-model", 0)
    for request in requests:
        registry.finish(request.request_id, status="cancelled")
    assert not registry.snapshot()["active"]


def test_terminal_registry_publication_observes_an_accepted_bulk_stop():
    registry = RelayRequestRegistry()
    request = registry.start("chat", "model", 0)
    assert registry.cancel_all() == 1
    result = registry.finish(request.request_id, terminal_reason="completed")
    assert result["status"] == "cancelled"
    assert result["terminal_reason"] == "cancelled"
    assert registry.telemetry_snapshot()["counters"]["cancelled"] == 1


def test_duplicate_ids_are_rejected_without_mixing_request_state():
    registry = RelayRequestRegistry()
    request = registry.start("chat", "one", 0, request_id="same-id")
    with pytest.raises(ValueError, match="duplicate"):
        registry.start("embedding", "another", 2, request_id="same-id")
    assert registry.snapshot()["active"][0]["model"] == "one"
    registry.finish(request.request_id)


def test_ten_thousand_requests_have_no_admission_quota_and_release_handles():
    registry = RelayRequestRegistry()
    requests = [registry.start("embedding", "model", 0) for _ in range(10000)]
    references = [weakref.ref(request) for request in requests]
    assert len(registry.snapshot()["active"]) == 10000
    for request in requests:
        registry.finish(request.request_id)
    del request, requests
    gc.collect()
    assert all(reference() is None for reference in references)
    assert registry.snapshot()["active"] == []
    assert isinstance(registry.snapshot()["last_completed"], dict)
    registry.clear_last_completed()
    assert registry.snapshot() == {"active": [], "last_completed": None}
    telemetry = registry.telemetry_snapshot()
    assert telemetry["counters"]["completed"] == 10000
    assert len(telemetry["recent"]) == 128
    assert telemetry["models"][0]["samples"] == 128


def test_telemetry_retains_concurrent_inference_completions_and_ignores_control():
    registry = RelayRequestRegistry()
    requests = [registry.start("chat", f"chat-{i}", 0) for i in range(4)]
    active = registry.telemetry_snapshot()["active"]
    assert active["chat"] == active["total"] == 4
    assert active["embedding"] == 0
    with ThreadPoolExecutor() as pool:
        list(
            pool.map(
                lambda request: registry.finish(
                    request.request_id, status="failed", error_code="test_failure"
                ),
                requests,
            )
        )
    catalog = registry.start("catalog", "", 0)
    registry.finish(catalog.request_id)
    assert registry.snapshot()["last_completed"]["kind"] == "catalog"
    snapshot = registry.telemetry_snapshot()
    assert snapshot["counters"] == {"completed": 0, "failed": 4, "cancelled": 0}
    assert {item["request_id"] for item in snapshot["recent"]} == {
        request.request_id for request in requests
    }
    snapshot["recent"][0]["error_code"] = "modified"
    assert registry.telemetry_snapshot()["recent"][0]["error_code"] == "test_failure"


def test_embedding_http_telemetry_counts_shared_calls_and_distinct_live_replicas():
    telemetry = RelayEmbeddingTelemetry()
    observe = telemetry.for_endpoint("llama")
    with observe("embedding", "replica-0"), observe("embedding", "replica-1"):
        with observe("embedding", "replica-0"), observe("props"):
            snapshot = telemetry.snapshot()
            assert snapshot["by_kind"]["embedding"] == {
                "count": 3,
                "failed": 0,
                "active": 3,
                "peak": 3,
            }
            assert snapshot["active_replicas"] == 2
    with pytest.raises(RuntimeError), observe("embedding", "replica-1"):
        raise RuntimeError("simulated HTTP failure")
    snapshot = telemetry.snapshot()
    assert snapshot["active_replicas"] == 0
    assert snapshot["peak_active_replicas"] == 2
    assert snapshot["by_kind"]["props"]["count"] == 1
    replicas = {row["model"]: row for row in snapshot["replicas"]}
    assert replicas["replica-0"]["peak"] == 2
    assert replicas["replica-1"]["count"] == 2
    assert replicas["replica-1"]["failed"] == 1
