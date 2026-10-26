"""Isolated direct rejections retain only indexed source metadata and vectors."""

from contextlib import contextmanager
import hashlib
from threading import Event

import pytest

from runtime_llama.adapters.embedding import _Call
from runtime_llama.client import ClientHTTPError
from runtime_llama.embedding_halving import _HalvingResult, _RejectedEmbedding
from runtime_llama.errors import LLMCancelledError

from test_embedding_async_stream import WindowSource, assert_drained, run, wait_thread
from test_embedding_contracts import ContractClient, configured, vector
from test_embedding_direct import policy
from test_embedding_shared_lifecycle import SharedScenario


def assert_sanitized(marker, original):
    assert marker.original is not original
    assert marker.original.status_code == original.status_code
    assert marker.original.error_type == original.error_type
    assert marker.original.error_code == original.error_code
    assert marker.original.__traceback__ is None
    assert marker.original.__context__ is None
    assert marker.original.__cause__ is None


@contextmanager
def direct_batch(*, prefix="doc: ", context=100, batch_token_limit=None):
    client = ContractClient(1, context=context, native_batch=context)
    adapter = configured(
        client,
        embedding_prefixes={"document": prefix},
        batch_token_limit=batch_token_limit,
    )
    left, right = _Call(adapter, None), _Call(adapter, None)
    for call in (left, right):
        call._direct_policy = policy(adapter)
    admitted = adapter._engine_start(left)
    observation = admitted.observations["embed_0"]
    try:
        yield adapter, client, left, right, admitted, observation
    finally:
        adapter._engine_finish(left, admitted.joined)
        adapter.close()


def prepare(adapter, admitted, observation, occurrences):
    return [
        adapter._engine_prepare(
            (call._direct_policy.prefix + text, "document"),
            "embed_0",
            call,
            observation,
            admitted.conservative,
        )
        for call, text in occurrences
    ]


def test_failed_direct_post_preserves_occurrence_indices_and_strips_prefix_once():
    with direct_batch() as (adapter, client, left, right, admitted, observation):
        occurrences = [(left, "same"), (right, "doc: source"), (left, "same")]
        records = prepare(adapter, admitted, observation, occurrences)
        # Preparation reuse may share the exact same record for equal source;
        # positions must still describe separate occurrences in the window.
        records[2] = records[0]
        group = adapter._engine_batch_call([left, right])
        group._input_positions = (8, 3, 11)
        original = ClientHTTPError(500, error_type="server_error")
        client.failure = lambda *_: original
        result = dict(
            adapter._engine_run_shared(
                "embed_0", records, [left, right, left], group, observation
            )
        )
        assert tuple(result) == (0, 1, 2)
        for index, source in enumerate(("same", "doc: source", "same")):
            marker = result[index]
            assert isinstance(marker, _RejectedEmbedding)
            assert_sanitized(marker, original)
            assert marker.text_sha256 == hashlib.sha256(source.encode()).hexdigest()
            assert marker.char_count == len(source)
            assert not hasattr(marker, "text") and not hasattr(marker, "owner")
            assert "doc: source" not in repr(marker)
        assert result[0] is not result[2]
        assert len(client.embedding_calls) == 5
        assert not client.token_calls


def test_failed_post_receipt_excludes_owner_pruned_before_native_post():
    with direct_batch() as (adapter, client, left, right, admitted, observation):
        records = prepare(
            adapter, admitted, observation, [(left, "withdrawn"), (right, "survivor")]
        )
        group = adapter._engine_batch_call([left, right])
        group._input_positions = (21, 34)
        left.stop.set()
        original = ClientHTTPError(500)
        client.failure = lambda *_: original
        result = dict(
            adapter._engine_run_shared(
                "embed_0", records, [left, right], group, observation
            )
        )
        assert isinstance(left.failure, LLMCancelledError)
        assert tuple(result) == (1,)
        assert_sanitized(result[1], original)
        assert result[1].text_sha256 == hashlib.sha256(b"survivor").hexdigest()
        assert client.embedding_calls == [("embed_0", ("doc: survivor",))]
        assert not client.token_calls


def test_rejection_isolates_failed_rows_without_replaying_successful_descendants():
    with direct_batch(context=10, batch_token_limit=20) as (
        adapter,
        client,
        left,
        _,
        admitted,
        observation,
    ):
        source = ["first", "second", "third", "fourth", "fifth"]
        records = prepare(
            adapter, admitted, observation, [(left, text) for text in source]
        )
        group = adapter._engine_batch_call([left])
        group._input_positions = (10, 11, 12, 13, 14)
        original = ClientHTTPError(500)
        client.failure = lambda _model, rows: original if "doc: third" in rows else None
        result = dict(
            adapter._engine_run_shared(
                "embed_0", records, [left] * 5, group, observation
            )
        )
        assert tuple(result) == tuple(range(5))
        assert_sanitized(result[2], original)
        assert result[2].text_sha256 == hashlib.sha256(b"third").hexdigest()
        for index in (0, 1, 3, 4):
            assert result[index] == vector("doc: " + source[index])
        assert client.embedding_calls == [
            ("embed_0", tuple("doc: " + text for text in source)),
            ("embed_0", ("doc: first", "doc: second")),
            ("embed_0", ("doc: third", "doc: fourth", "doc: fifth")),
            ("embed_0", ("doc: third",)),
            ("embed_0", ("doc: fourth", "doc: fifth")),
        ]
        assert not client.token_calls


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 502, 503])
def test_other_http_statuses_keep_original_error_without_provenance(status):
    with direct_batch() as (adapter, client, left, _, admitted, observation):
        records = prepare(adapter, admitted, observation, [(left, "source")])
        group = adapter._engine_batch_call([left])
        group._input_positions = (7,)
        original = ClientHTTPError(status)
        client.failure = lambda *_: original
        with pytest.raises(ClientHTTPError) as raised:
            adapter._engine_run_shared("embed_0", records, [left], group, observation)
        assert raised.value is original
        assert len(client.embedding_calls) == 1 and not client.token_calls


@pytest.mark.parametrize("exact", [False, True])
def test_direct_metadata_needs_no_owner_provenance_and_exact_keeps_original_500(exact):
    with direct_batch() as (adapter, client, left, _, admitted, observation):
        if exact:
            del left._direct_policy
            record = adapter._engine_prepare(
                ("doc: source", "document"),
                "embed_0",
                left,
                observation,
                admitted.conservative,
            )
        else:
            record = prepare(adapter, admitted, observation, [(left, "source")])[0]
        group = adapter._engine_batch_call([left])
        if exact:
            group._input_positions = (8,)
        original = ClientHTTPError(500)
        client.failure = lambda *_: original
        if exact:
            with pytest.raises(ClientHTTPError) as raised:
                adapter._engine_run_shared(
                    "embed_0", [record], [left], group, observation
                )
            assert raised.value is original
        else:
            result = dict(
                adapter._engine_run_shared(
                    "embed_0", [record], [left], group, observation
                )
            )
            assert_sanitized(result[0], original)
        assert len(client.embedding_calls) == 1
        assert len(client.token_calls) == int(exact)


@pytest.mark.parametrize("exact_left", [False, True])
def test_engine_projects_failed_rows_to_owning_windows_without_peer_or_owner_refs(
    exact_left,
):
    async def scenario():
        shared = SharedScenario()
        shared.sources = [
            WindowSource([["a-0", "same"]]),
            WindowSource([["b-0", "same"]]),
        ]
        shared.expected_shared = ["a-0", "b-0", "same", "same"]
        shared.streams = [
            shared.adapter.astream_texts(shared.sources[0])
            if exact_left
            else shared.adapter._astream_texts_direct(
                shared.sources[0], policy=policy(shared.adapter)
            ),
            shared.adapter._astream_texts_direct(
                shared.sources[1], policy=policy(shared.adapter)
            ),
        ]
        original = ClientHTTPError(500)
        shared.client.shared_error = original
        async with shared:
            shared.client.shared_gate.set()
            if exact_left:
                with pytest.raises(ClientHTTPError) as left:
                    await shared.tasks[0]
                assert left.value is original
            else:
                left_result = await shared.tasks[0]
                assert isinstance(left_result, _HalvingResult)
                assert left_result.outputs == (None, vector("same"))
                assert tuple(left_result.rejections) == (0,)
                assert_sanitized(left_result.rejections[0], original)
                assert not hasattr(left_result.rejections[0], "owner")
                assert left_result.window_index == 0
            right_result = await shared.tasks[1]
            assert isinstance(right_result, _HalvingResult)
            assert right_result.outputs == (None, vector("same"))
            assert tuple(right_result.rejections) == (0,)
            assert_sanitized(right_result.rejections[0], original)
            assert not hasattr(right_result.rejections[0], "owner")
            assert right_result.window_index == 0
            assert len(shared.client.shared_posts) > 1
            assert_drained(shared.adapter, shared.client)

    run(scenario)


def test_owner_retired_during_failed_post_keeps_cancellation_and_no_receipt():
    async def scenario():
        stop = Event()
        shared = SharedScenario(cancel_a=stop.is_set)
        shared.streams = [
            shared.adapter._astream_texts_direct(
                source,
                policy=policy(shared.adapter),
                cancel=stop.is_set if index == 0 else None,
            )
            for index, source in enumerate(shared.sources)
        ]
        original = ClientHTTPError(500)
        shared.client.shared_error = original
        async with shared:
            shared.client.retired_owner = shared.calls["a"].stop.is_set
            stop.set()
            await wait_thread(shared.client.retired_owner_polled)
            shared.client.shared_gate.set()
            with pytest.raises(LLMCancelledError):
                await shared.tasks[0]
            right = await shared.tasks[1]
            assert isinstance(right, _HalvingResult)
            assert right.outputs == (None, None)
            assert tuple(right.rejections) == (0, 1)
            for marker in right.rejections.values():
                assert_sanitized(marker, original)
            assert not shared.client.transport_cancelled.is_set()
            assert all(
                not hasattr(marker, "owner") for marker in right.rejections.values()
            )
            assert_drained(shared.adapter, shared.client)

    run(scenario)
