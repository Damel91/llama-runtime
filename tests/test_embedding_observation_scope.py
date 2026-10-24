"""Stream-local preparation snapshots never replace fresh HTTP admission."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError
from threading import Event, current_thread

import pytest

from runtime_llama.adapters.embedding import _Call, _DirectEmbeddingPolicy
from runtime_llama.embedding_observation import _EmbeddingAdmissionScope
from runtime_llama.embedding_scheduler import EmbeddingScheduler
from runtime_llama.errors import LLMBackendError, LLMCancelledError

from test_embedding_contracts import CONTENT, SEMANTIC, configured
from test_embedding_metadata_cohorts import MetadataClient


def direct_call(adapter, scope, cancel=None):
    call = _Call(adapter, cancel)
    call._admission_scope = scope
    call._direct_policy = _DirectEmbeddingPolicy(
        CONTENT, SEMANTIC, "test-build-1", adapter._policy("raw"), ""
    )
    return call


def seeded(adapter):
    scope = _EmbeddingAdmissionScope()
    call = _Call(adapter, None)
    context = adapter._engine_start(call)
    assert adapter._engine_bind_admission_scope(scope, context)
    adapter._engine_finish(call, context.joined)
    assert adapter._pool_users == 0
    return scope


def finish(adapter, calls, contexts):
    if contexts is not None:
        for call, context in zip(calls, contexts):
            if context is not None:
                adapter._engine_finish(call, context.joined)


def test_scope_is_inert_nonowning_and_copies_only_immutable_observations():
    client = MetadataClient()
    adapter = configured(client)
    try:
        scope = seeded(adapter)
        before = client.metadata_calls.copy()
        calls = tuple(direct_call(adapter, scope) for _ in range(4))
        contexts = adapter._engine_start_shared(calls)
        try:
            assert client.metadata_calls == before
            assert adapter._pool_users == 4
            assert len({id(context.observations) for context in contexts}) == 4
            observation = contexts[0].observations["embed_0"]
            assert contexts[1].observations["embed_0"] is observation
            with pytest.raises(FrozenInstanceError):
                observation.total_slots = 12
            contexts[0].observations.clear()
            assert len(contexts[1].observations) == 2
            assert len(scope._snapshot().observations) == 2
        finally:
            finish(adapter, calls, contexts)
        assert adapter._pool_users == 0
        scope.clear()
        assert scope._snapshot() is None
    finally:
        adapter.close()


@pytest.mark.parametrize("kind", ["unscoped", "exact", "cleared", "scheduler"])
def test_invalid_or_mixed_scope_falls_back_to_fresh_admission(kind):
    client = MetadataClient()
    adapter = configured(client)
    try:
        scope = seeded(adapter)
        before = client.metadata_calls.copy()
        calls = [direct_call(adapter, scope)]
        if kind == "unscoped":
            calls.append(direct_call(adapter, None))
        elif kind == "exact":
            calls.append(_Call(adapter, None))
        elif kind == "cleared":
            scope.clear()
        else:
            previous = adapter._scheduler
            adapter._scheduler = EmbeddingScheduler(
                adapter.embedding_request_parallelism,
                adapter.embedding_queue_size,
                {model: 1 for model in adapter._replicas},
            )
            previous.close()
        contexts = adapter._engine_start_shared(calls)
        try:
            assert client.metadata_calls - before == {
                "health": 1,
                "models": 1,
                "props": 2,
            }
            assert adapter._pool_users == len(calls)
            assert scope._snapshot().scheduler is adapter._scheduler
        finally:
            finish(adapter, calls, contexts)
    finally:
        adapter.close()


def test_a_scope_cannot_authorize_another_adapter():
    clients = MetadataClient(), MetadataClient()
    adapters = tuple(configured(client) for client in clients)
    try:
        scope = seeded(adapters[0])
        call = direct_call(adapters[1], scope)
        contexts = adapters[1]._engine_start_shared((call,))
        try:
            assert clients[1].metadata_calls == {
                "health": 1,
                "models": 1,
                "props": 2,
            }
            assert adapters[0]._pool_users == 0
            assert adapters[1]._pool_users == 1
        finally:
            finish(adapters[1], (call,), contexts)
    finally:
        for adapter in adapters:
            adapter.close()


@pytest.mark.parametrize(
    "aliases", [["embed_0"], ["embed_0", "embed_1", "embed_2"], ["renamed"]]
)
def test_any_full_pool_topology_change_revokes_late_forks_and_drains(aliases):
    client = MetadataClient()
    adapter = configured(client)
    try:
        scope = seeded(adapter)
        old = direct_call(adapter, scope)
        old_context = adapter._engine_start(old)
        assert adapter._pool_users == 1
        client.set_aliases(aliases)
        metadata_owner = _Call(adapter, None)
        catalog = adapter._engine_catalog_shared((metadata_owner,))
        assert len(catalog["data"]) == len(aliases)
        assert adapter._admission_revoked
        late = direct_call(adapter, scope)
        assert adapter._engine_start_shared((late,)) is None
        assert adapter._pool_users == 1
        adapter._engine_finish(old, old_context.joined)
        contexts = adapter._engine_start_shared((late,))
        try:
            assert set(contexts[0].observations) == set(aliases)
            assert adapter._pool_users == 1
        finally:
            finish(adapter, (late,), contexts)
        # A stable authoritative join can now enable a new generation. No
        # stream anchor kept the old pool alive while the topology changed.
        renewed = direct_call(adapter, scope)
        contexts = adapter._engine_start_shared((renewed,))
        finish(adapter, (renewed,), contexts)
        assert not adapter._admission_revoked
        before = client.metadata_calls.copy()
        final = direct_call(adapter, scope)
        contexts = adapter._engine_start_shared((final,))
        try:
            assert client.metadata_calls == before
        finally:
            finish(adapter, (final,), contexts)
        assert adapter._pool_users == 0
    finally:
        adapter.close()


def test_malformed_fresh_catalog_revokes_scope_without_a_post():
    client = MetadataClient()
    adapter = configured(client)
    try:
        scope = seeded(adapter)
        client.catalog[0] = "invalid catalogue entry"
        call = direct_call(adapter, scope)
        with pytest.raises(LLMBackendError):
            adapter._engine_catalog_shared((call,))
        assert adapter._admission_revoked
        assert adapter._engine_fork_admission_scope((call,), (call,)) is None
        assert not client.embedding_calls and adapter._pool_users == 0
    finally:
        adapter.close()


def test_cancelled_scope_owner_does_not_retire_surviving_owner():
    client = MetadataClient()
    adapter = configured(client)
    try:
        scope = seeded(adapter)
        cancelled = Event()
        cancelled.set()
        calls = (
            direct_call(adapter, scope, cancelled.is_set),
            direct_call(adapter, scope),
        )
        before = client.metadata_calls.copy()
        contexts = adapter._engine_start_shared(calls)
        try:
            assert contexts[0] is None and contexts[1] is not None
            assert isinstance(calls[0].failure, LLMCancelledError)
            assert adapter._pool_users == 1
            assert client.metadata_calls == before
        finally:
            finish(adapter, calls, contexts)
        assert adapter._pool_users == 0
    finally:
        adapter.close()


def test_old_snapshot_never_restores_capacity_and_fresh_props_reduce_slots():
    client = MetadataClient(1)
    props = client.props
    slots = [4]

    def current_props(*args, **kwargs):
        result = props(*args, **kwargs)
        result["total_slots"] = slots[0]
        return result

    client.props = current_props
    adapter = configured(client, embedding_request_parallelism=4)
    try:
        scope = seeded(adapter)
        assert adapter._replicas["embed_0"].max_in_flight == 4
        slots[0] = 1
        call = direct_call(adapter, scope)
        catalog = adapter._engine_catalog_shared((call,))
        current = adapter._engine_observe("embed_0", call, catalog=catalog)
        assert current.total_slots == 1
        assert adapter._replicas["embed_0"].max_in_flight == 1
        before = client.metadata_calls.copy()
        contexts = adapter._engine_start_shared((call,))
        try:
            assert contexts[0].observations["embed_0"].total_slots == 4
            assert client.metadata_calls == before
            assert adapter._replicas["embed_0"].max_in_flight == 1
        finally:
            finish(adapter, (call,), contexts)
    finally:
        adapter.close()


def test_newer_dispatch_revocation_wins_over_an_older_join_catalog():
    client = MetadataClient()
    adapter = configured(client)
    entered, release = Event(), Event()
    models = client.models

    def ordered_models(**kwargs):
        result = models(**kwargs)
        if current_thread().name.startswith("older_join"):
            entered.set()
            assert release.wait(3)
        return result

    try:
        scope = seeded(adapter)
        scope.clear()
        client.models = ordered_models
        late = direct_call(adapter, scope)
        with ThreadPoolExecutor(1, thread_name_prefix="older_join") as workers:
            pending = workers.submit(adapter._engine_start_shared, (late,))
            try:
                assert entered.wait(2)
                client.set_aliases(["embed_0", "embed_1", "embed_2"])
                adapter._engine_catalog_shared((_Call(adapter, None),))
                revoked = adapter._admission_generation
                assert adapter._admission_revoked
            finally:
                release.set()
            contexts = pending.result(3)
        try:
            assert adapter._admission_revoked
            assert adapter._admission_generation is revoked
            assert contexts[0].admission_generation is None
            assert scope._snapshot() is None
        finally:
            finish(adapter, (late,), contexts)
        assert adapter._pool_users == 0
    finally:
        release.set()
        adapter.close()


def test_join_generation_is_captured_before_props_and_cannot_be_relabelled():
    client = MetadataClient()
    adapter = configured(client)
    entered, release = Event(), Event()
    props = client.props

    def ordered_props(*args, **kwargs):
        result = props(*args, **kwargs)
        if current_thread().name.startswith("older_props"):
            entered.set()
            assert release.wait(3)
        return result

    try:
        scope = seeded(adapter)
        scope.clear()
        client.props = ordered_props
        late = direct_call(adapter, scope)
        with ThreadPoolExecutor(1, thread_name_prefix="older_props") as workers:
            pending = workers.submit(adapter._engine_start_shared, (late,))
            try:
                assert entered.wait(2)
                first_generation = adapter._admission_generation
                original_catalog = deepcopy(client.catalog)
                client.catalog[0] = "malformed"
                with pytest.raises(LLMBackendError):
                    adapter._engine_catalog_shared((_Call(adapter, None),))
                client.catalog = original_catalog
                fresh = _Call(adapter, None)
                fresh_contexts = adapter._engine_start_shared((fresh,))
                finish(adapter, (fresh,), fresh_contexts)
                assert not adapter._admission_revoked
                assert adapter._admission_generation is not first_generation
            finally:
                release.set()
            contexts = pending.result(3)
        try:
            assert contexts[0].admission_generation is first_generation
            assert scope._snapshot() is None
            assert adapter._pool_users == 1
        finally:
            finish(adapter, (late,), contexts)
        assert adapter._pool_users == 0
    finally:
        release.set()
        adapter.close()


def test_fresh_recovery_observation_also_revokes_changed_full_pool():
    client = MetadataClient()
    adapter = configured(client)
    try:
        scope = seeded(adapter)
        client.set_aliases(["embed_0", "embed_1", "embed_2"])
        call = direct_call(adapter, scope)
        observed = adapter._observe_replica(
            replica=adapter._replicas["embed_0"], call=call, cancel=call.probe
        )
        assert observed.identity_key == (CONTENT, SEMANTIC)
        assert adapter._admission_revoked
        assert adapter._engine_fork_admission_scope((call,), (call,)) is None
        assert not client.embedding_calls and adapter._pool_users == 0
    finally:
        adapter.close()
