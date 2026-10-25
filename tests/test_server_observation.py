"""Optional SDK protocol tests; these do not qualify any real server."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, Thread
import time

import pytest

from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError
from runtime_llama.server_observation import (
    CoherentServerSnapshot,
    ObservedServerReplica,
    ServerAdmissionExpectation,
    ServerObservationCapability,
    ServerObservationScope,
    ServerObservationSession,
    StaleServerObservationError,
)


def scope():
    return ServerObservationScope("endpoint-v1", "authority-v1", "a" * 64, "b" * 64)


def replica(alias="embedding-0", replica_id="replica-0"):
    return ObservedServerReplica(
        replica_id=replica_id,
        alias=alias,
        loading_generation="load-1",
        configuration_revision="config-1",
        build_id="native-build-1",
        tokenizer_id="vocab-policy-1",
        content_sha256="a" * 64,
        embedding_sha256="b" * 64,
        context_length=2048,
        n_batch=2048,
        n_ubatch=512,
        slots=4,
    )


def snapshot(**kwargs):
    return CoherentServerSnapshot(
        scope=scope(),
        server_instance_id=kwargs.pop("server_instance_id", "server-start-1"),
        observation_revision=kwargs.pop("observation_revision", "snapshot-1"),
        replicas=kwargs.pop("replicas", (replica(),)),
        valid_for_seconds=kwargs.pop("valid_for_seconds", 30),
        **kwargs,
    )


class Provider:
    def __init__(self, *, conditional=False):
        self.capability = ServerObservationCapability(conditional_admission=conditional)
        self.observation = snapshot()
        self.snapshots = []
        self.posts = []
        self.fail = None

    def snapshot(self, selected, **options):
        assert selected == scope()
        assert 0 < options["timeout"] <= 1
        assert options["max_response_bytes"] == 1024
        assert options["cancel_wait"] is True
        self.snapshots.append(options)
        return self.observation

    def embeddings(self, inputs, alias, *, expected, **options):
        self.posts.append((inputs, alias, expected, options))
        if self.fail is not None:
            raise self.fail
        return [[1.0]]


def session(provider):
    return ServerObservationSession(scope(), provider, max_response_bytes=1024)


def no_native(*_args, **_kwargs):
    pytest.fail("native must not be used after negotiated provider selection")


@pytest.mark.parametrize("conditional,reads", [(False, 2), (True, 1)])
def test_complete_snapshot_and_conditional_admission_are_independent(
    conditional, reads
):
    provider = Provider(conditional=conditional)
    with session(provider) as selected:
        for _ in range(2):
            assert selected.dispatch(
                ("text",), "embedding-0", native=no_native, timeout=1
            ) == [[1.0]]
    assert len(provider.snapshots) == reads
    assert len(provider.posts) == 2
    expectation = provider.posts[0][2]
    if conditional:
        assert isinstance(expectation, ServerAdmissionExpectation)
        assert expectation.scope == scope()
        assert expectation.replica.loading_generation == "load-1"
        assert expectation.replica.configuration_revision == "config-1"
        assert expectation.server_instance_id == "server-start-1"
    else:
        assert expectation is None


def test_missing_capability_falls_back_before_any_optional_server_use():
    provider = Provider()
    provider.capability = None
    native_calls = []

    def native(inputs, alias, **options):
        native_calls.append((inputs, alias, options))
        assert options["cancel_wait"] is True
        return [[2.0]]

    with session(provider) as selected:
        assert selected.dispatch(
            ("text",), "embedding-0", native=native, timeout=1
        ) == [[2.0]]
    assert len(native_calls) == 1
    assert provider.snapshots == provider.posts == []


@pytest.mark.parametrize(
    "failure", [RuntimeError("ambiguous POST"), StaleServerObservationError()]
)
def test_post_errors_never_retry_or_fall_back(failure):
    provider = Provider(conditional=True)
    provider.fail = failure
    with session(provider) as selected:
        with pytest.raises(LLMBackendError) as caught:
            selected.dispatch(("text",), "embedding-0", native=no_native, timeout=1)
        assert "ambiguous POST" not in str(caught.value)
    assert len(provider.posts) == len(provider.snapshots) == 1


@pytest.mark.parametrize(
    "change", ["restart", "reload", "config", "alias", "vocab", "build"]
)
def test_server_enforcement_rejects_cached_snapshot_across_state_changes(change):
    class EnforcingProvider(Provider):
        def embeddings(self, inputs, alias, *, expected, **options):
            self.posts.append((inputs, alias, expected, options))
            current = self.observation
            if (
                expected.server_instance_id != current.server_instance_id
                or expected.observation_revision != current.observation_revision
                or expected.replica not in current.replicas
            ):
                raise StaleServerObservationError()
            return [[1.0]]

    provider = EnforcingProvider(conditional=True)
    with session(provider) as selected:
        selected.dispatch(("first",), "embedding-0", native=no_native, timeout=1)
        if change == "restart":
            provider.observation = replace(
                provider.observation, server_instance_id="server-start-2"
            )
        else:
            changes = {
                "reload": {"loading_generation": "load-2"},
                "config": {"configuration_revision": "config-2", "context_length": 512},
                "alias": {"alias": "other-alias"},
                "vocab": {"tokenizer_id": "vocab-policy-2"},
                "build": {"build_id": "native-build-2"},
            }
            provider.observation = replace(
                provider.observation,
                observation_revision="snapshot-2",
                replicas=(replace(replica(), **changes[change]),),
            )
        with pytest.raises(StaleServerObservationError):
            selected.dispatch(("second",), "embedding-0", native=no_native, timeout=1)
        assert len(provider.snapshots) == 1
        assert len(provider.posts) == 2
        assert selected._cached is None


def test_late_snapshot_cannot_resurrect_revoked_local_generation():
    entered, release = Event(), Event()

    class DelayedProvider(Provider):
        def snapshot(self, selected, **options):
            entered.set()
            assert release.wait(1)
            return super().snapshot(selected, **options)

    provider = DelayedProvider(conditional=True)
    with session(provider) as selected, ThreadPoolExecutor(1) as workers:
        future = workers.submit(
            selected.dispatch, ("text",), "embedding-0", native=no_native, timeout=1
        )
        assert entered.wait(1)
        selected.revoke()
        release.set()
        with pytest.raises(StaleServerObservationError):
            future.result(1)
        assert selected._cached is None
    assert provider.posts == []


def test_scope_and_full_pool_identity_mismatch_fail_before_post():
    provider = Provider()
    provider.observation = replace(
        snapshot(), scope=replace(scope(), authority_id="another-authority")
    )
    with session(provider) as selected:
        with pytest.raises(LLMBackendError, match="scope_mismatch"):
            selected.dispatch(("text",), "embedding-0", native=no_native, timeout=1)
    assert provider.posts == []
    with pytest.raises(ValueError, match="different logical"):
        snapshot(
            replicas=(
                replica(),
                replace(replica("embedding-1", "replica-1"), embedding_sha256="c" * 64),
            )
        )


def test_expiry_uses_one_original_deadline_and_never_sends_after_late_observation():
    class SlowProvider(Provider):
        def snapshot(self, selected, **options):
            time.sleep(0.02)
            return super().snapshot(selected, **options)

    provider = SlowProvider(conditional=True)
    with session(provider) as selected:
        with pytest.raises(LLMTimeoutError):
            selected.dispatch(("text",), "embedding-0", native=no_native, timeout=0.01)
    assert provider.posts == []


def test_close_cancels_and_waits_for_actual_provider_cleanup():
    entered, cleanup, release = Event(), Event(), Event()

    class BlockingProvider(Provider):
        def embeddings(self, inputs, alias, *, expected, **options):
            entered.set()
            while not options["cancel"]():
                time.sleep(0.001)
            cleanup.set()
            assert release.wait(1)
            raise LLMCancelledError()

    provider = BlockingProvider(conditional=True)
    selected = session(provider)
    with ThreadPoolExecutor(1) as workers:
        future = workers.submit(
            selected.dispatch, ("text",), "embedding-0", native=no_native, timeout=1
        )
        assert entered.wait(1)
        closed = Event()
        closer = Thread(target=lambda: (selected.close(), closed.set()))
        closer.start()
        assert cleanup.wait(1)
        assert not closed.is_set()
        release.set()
        closer.join(1)
        assert closed.is_set()
        with pytest.raises(LLMCancelledError):
            future.result(1)
    with pytest.raises(LLMBackendError, match="closed"):
        selected.dispatch(("text",), "embedding-0", native=no_native, timeout=1)


def test_external_cancellation_precedes_any_provider_call():
    provider = Provider()
    with session(provider) as selected:
        with pytest.raises(LLMCancelledError):
            selected.dispatch(
                ("text",),
                "embedding-0",
                native=no_native,
                timeout=1,
                cancel=lambda: True,
            )
    assert provider.snapshots == provider.posts == []


@pytest.mark.parametrize("value", [0, 2, True, "1"])
def test_unknown_or_ambiguous_protocol_version_is_not_inferred(value):
    with pytest.raises(ValueError, match="protocol version"):
        ServerObservationCapability(version=value)


def test_duplicate_aliases_or_instances_and_mutable_snapshot_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        snapshot(replicas=(replica(), replica()))
    with pytest.raises(ValueError, match="immutable"):
        snapshot(replicas=[replica()])


def test_aliases_remain_native_strings_instead_of_becoming_opaque_generation_ids():
    provider = Provider(conditional=True)
    alias = "organization/embedding modèle-1"
    provider.observation = snapshot(replicas=(replace(replica(), alias=alias),))
    with session(provider) as selected:
        selected.dispatch(("text",), alias, native=no_native, timeout=1)
    assert provider.posts[0][1] == alias
