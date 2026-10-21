"""External metadata owners release blocked admission even without engine work."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from threading import Event, current_thread
from types import SimpleNamespace

import pytest

from runtime_llama.errors import LLMBackendError

from test_embedding_contracts import ContractClient, configured, eventually, vector


@contextmanager
def blocked_by_external_health(phase, *, hold_none_completion=False):
    client = ContractClient(1)
    adapter = configured(client, embedding_batch_wait_sec=0, timeout=5)
    entered, release = Event(), Event()
    returned_none, release_none = Event(), Event()
    stop = Event()
    health, props = client.health, client.props
    start_shared = adapter._engine_start_shared

    def external_health(**kwargs):
        if phase == "discovery_error" and current_thread().name.startswith(
            "external_metadata"
        ):
            entered.set()
            assert release.wait(3)
            raise LLMBackendError("external_metadata_failure")
        return health(**kwargs)

    def external_props(*args, **kwargs):
        if phase == "pool" and current_thread().name.startswith("external_metadata"):
            entered.set()
            assert release.wait(3)
        return props(*args, **kwargs)

    def delayed_none(calls):
        result = start_shared(calls)
        if result is None:
            returned_none.set()
            if hold_none_completion:
                assert release_none.wait(3)
        return result

    client.health, client.props = external_health, external_props
    adapter._engine_start_shared = delayed_none
    with (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="external_metadata") as hp,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="external_consumer") as qp,
    ):
        external = hp.submit(adapter.health_status)
        query = None
        try:
            assert entered.wait(2)
            if phase == "pool":
                assert adapter._pool_users == 1
                client.set_aliases(["renamed"])
            else:
                assert adapter._pool_users == 0
            query = qp.submit(adapter.embed_query, "request", cancel=stop.is_set)
            assert returned_none.wait(2)
            if not hold_none_completion:
                eventually(
                    lambda: any(
                        state.context is None and not state.starting
                        for state in adapter._engine._states
                    )
                )
            assert not query.done()
            assert not client.token_calls and not client.embedding_calls
            yield SimpleNamespace(
                client=client,
                adapter=adapter,
                external=external,
                query=query,
                release=release,
                release_none=release_none,
            )
        finally:
            release.set()
            release_none.set()
            stop.set()
            for future in (external, query):
                if future is not None:
                    with suppress(BaseException):
                        future.result(3)
            adapter.close()


def assert_completed_and_drained(blocked, alias):
    assert blocked.query.result(2) == vector("request")
    assert blocked.client.embedding_calls == [(alias, ("request",))]
    assert len(blocked.client.token_calls) == 1
    assert not blocked.adapter._calls
    assert blocked.adapter._pool_users == 0
    engine = blocked.adapter._engine.snapshot()
    assert engine["metadata_targets"] == engine["metadata_rounds_waiting"] == 0
    assert engine["active_plans"] == engine["pending_bytes"] == 0
    scheduler = blocked.adapter._scheduler.snapshot()
    assert scheduler["in_flight"] == scheduler["queued"] == 0


def test_external_health_pool_reference_drain_retries_changed_topology_cohort():
    with blocked_by_external_health("pool") as blocked:
        blocked.release.set()
        blocked.external.result(2)
        assert_completed_and_drained(blocked, "renamed")


def test_external_discovery_error_unlock_retries_cohort_without_a_pool_reference():
    with blocked_by_external_health("discovery_error") as blocked:
        blocked.release.set()
        assert blocked.external.result(2)["healthy"] is False
        assert_completed_and_drained(blocked, "embed_0")


@pytest.mark.parametrize("phase", ["pool", "discovery_error"])
def test_external_availability_change_before_none_completion_is_not_lost(phase):
    with blocked_by_external_health(phase, hold_none_completion=True) as blocked:
        # The worker has already observed unavailability. Complete the external
        # operation before allowing its None outcome to reach the coordinator.
        blocked.release.set()
        blocked.external.result(2)
        assert blocked.adapter._pool_users == 0
        blocked.release_none.set()
        assert_completed_and_drained(
            blocked, "renamed" if phase == "pool" else "embed_0"
        )
