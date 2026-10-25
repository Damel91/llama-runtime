"""Native 500 isolation retains successful descendants under one HTTP lease."""

from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from contextlib import contextmanager
import hashlib
from threading import Barrier, Event, get_ident
import time

import pytest

from runtime_llama.adapters.embedding import _Call
from runtime_llama.client import ClientHTTPError, LlamaCppClient
from runtime_llama.embedding_halving import _RejectedEmbedding
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError

from test_embedding_contracts import ContractClient, configured, eventually, vector
from test_embedding_direct import policy


@contextmanager
def batch(*, count=1, prefix="doc: ", limit=512, dimension=2):
    client = ContractClient(count, context=100, native_batch=100)
    adapter = configured(
        client, embedding_prefixes={"document": prefix}, dimension=dimension
    )
    adapter.embedding_isolation_max_requests = limit
    calls = (_Call(adapter, None), _Call(adapter, None))
    for call in calls:
        call._direct_policy = policy(adapter)
    admitted = adapter._engine_start(calls[0])
    try:
        yield adapter, client, calls, admitted
    finally:
        adapter._engine_finish(calls[0], admitted.joined)
        adapter.close()


def execute(adapter, model, admitted, occurrences):
    observed = admitted.observations[model]
    records = [
        adapter._engine_prepare(
            (call._direct_policy.prefix + text, "document"),
            model,
            call,
            observed,
            admitted.conservative,
        )
        for call, text in occurrences
    ]
    group = adapter._engine_batch_call([call for call, _ in occurrences])
    group._input_positions = tuple(range(len(occurrences)))
    return adapter._engine_run_shared(
        model, records, [call for call, _ in occurrences], group, observed
    )


def test_isolation_retains_successes_and_duplicate_occurrences_without_tokenizer():
    with batch() as (adapter, client, (owner, _), admitted):
        incoming = ["good-0", "bad", "good-1", "good-2", "bad", "good-3"]
        original = ClientHTTPError(500, error_type="server_error")
        successful = set()

        def fail(_model, rows):
            # A row from any successful POST must never be sent again, even
            # while other branches are being isolated.
            assert not successful.intersection(rows)
            if any(row == "doc: bad" for row in rows):
                return original
            successful.update(rows)
            return None

        client.failure = fail
        result = dict(
            execute(adapter, "embed_0", admitted, [(owner, text) for text in incoming])
        )
        assert tuple(result) == tuple(range(len(incoming)))
        for index, text in enumerate(incoming):
            if text == "bad":
                marker = result[index]
                assert isinstance(marker, _RejectedEmbedding)
                assert marker.original is not original
                assert marker.original.status_code == original.status_code
                assert marker.original.error_type == original.error_type
                assert marker.original.__traceback__ is None
                assert marker.original.__context__ is None
                assert marker.original.__cause__ is None
                assert marker.char_count == len(text)
                assert marker.input_type == "document"
                assert not hasattr(marker, "source_text")
            else:
                assert result[index] == vector("doc: " + text)
        assert result[1].text_sha256 == result[4].text_sha256
        assert not client.token_calls
        assert len(client.embedding_calls) <= 2 * len(incoming) - 1
        assert owner.failure is None


def test_failed_array_can_recover_completely_when_singletons_are_valid():
    with batch() as (adapter, client, (owner, _), admitted):
        original = ClientHTTPError(500)
        client.failure = lambda _model, rows: original if len(rows) > 1 else None
        result = execute(
            adapter, "embed_0", admitted, [(owner, text) for text in ["a", "b", "c"]]
        )
        assert result == [
            (index, vector("doc: " + text))
            for index, text in enumerate(["a", "b", "c"])
        ]
        assert len(client.embedding_calls) == 5
        assert not client.token_calls


@pytest.mark.parametrize("status", [400, 401, 429, 502, 503])
def test_non500_is_not_isolated_and_preserves_original(status):
    with batch() as (adapter, client, (owner, _), admitted):
        original = ClientHTTPError(status)
        client.failure = lambda *_: original
        with pytest.raises(ClientHTTPError) as raised:
            execute(adapter, "embed_0", admitted, [(owner, "first"), (owner, "second")])
        assert raised.value is original
        assert len(client.embedding_calls) == 1
        assert not client.token_calls


def test_withdrawn_owner_is_pruned_before_child_but_peer_survives():
    with batch() as (adapter, client, (left, right), admitted):
        original = ClientHTTPError(500)

        def fail(_model, _rows):
            if len(client.embedding_calls) == 1:
                left.stop.set()
                return original
            return None

        client.failure = fail
        result = execute(
            adapter, "embed_0", admitted, [(left, "withdrawn"), (right, "survivor")]
        )
        assert result == [(1, vector("doc: survivor"))]
        assert isinstance(left.failure, LLMCancelledError)
        assert right.failure is None
        assert client.embedding_calls == [
            ("embed_0", ("doc: withdrawn", "doc: survivor")),
            ("embed_0", ("doc: survivor",)),
        ]
        assert not client.token_calls


def test_identity_drift_after_parent_500_prevents_any_child_post():
    with batch() as (adapter, client, (owner, _), admitted):
        original = ClientHTTPError(500)

        def fail(_model, _rows):
            client.builds["embed_0"] = "replaced-tokenizer"
            return original

        client.failure = fail
        with pytest.raises(LLMBackendError, match="direct_tokenizer_changed"):
            execute(adapter, "embed_0", admitted, [(owner, "first"), (owner, "second")])
        assert len(client.embedding_calls) == 1
        assert len(client.props_calls) >= 2
        assert not client.token_calls


def test_finite_descendant_post_budget_stops_full_failure_tree():
    with batch(limit=2) as (adapter, client, (owner, _), admitted):
        original = ClientHTTPError(500)
        client.failure = lambda *_: original
        with pytest.raises(LLMBackendError, match="isolation_request_limit") as raised:
            execute(
                adapter,
                "embed_0",
                admitted,
                [(owner, text) for text in ["aa", "bb", "cc", "dd"]],
            )
        assert raised.value.__cause__ is original
        assert len(client.embedding_calls) == 3
        assert not client.token_calls


def test_mixed_failed_post_preserves_exact_owner_error_without_exact_replay():
    with batch() as (adapter, client, (direct, exact), admitted):
        del exact._direct_policy
        observation = admitted.observations["embed_0"]
        records = [
            adapter._engine_prepare(
                ("doc: direct", "document"),
                "embed_0",
                direct,
                observation,
                admitted.conservative,
            ),
            adapter._engine_prepare(
                ("doc: exact", "document"),
                "embed_0",
                exact,
                observation,
                admitted.conservative,
            ),
        ]
        original = ClientHTTPError(500)
        client.failure = lambda _model, rows: original if "doc: exact" in rows else None
        group = adapter._engine_batch_call([direct, exact])
        group._input_positions = (3, 8)
        result = adapter._engine_run_shared(
            "embed_0", records, [direct, exact], group, observation
        )
        assert result == [(0, vector("doc: direct"))]
        assert exact.failure is original and direct.failure is None
        assert client.embedding_calls == [
            ("embed_0", ("doc: direct", "doc: exact")),
            ("embed_0", ("doc: direct",)),
        ]
        assert len(client.token_calls) == 1


def test_existing_scheduler_runs_peer_alias_while_failed_lane_is_isolating():
    class ParallelClient(ContractClient):
        def __init__(self):
            super().__init__(2)
            self.child_started, self.child_release, self.peer_completed = (
                Event(),
                Event(),
                Event(),
            )
            self.original = ClientHTTPError(500)

        def embeddings(self, input, *, model, **kwargs):
            rows = tuple(input)
            if any("bad" in row for row in rows):
                if len(rows) == 1:
                    self.child_started.set()
                    self._wait(
                        self.child_release,
                        cancel=kwargs.get("cancel"),
                        timeout=kwargs.get("timeout"),
                    )
                with self.lock:
                    self.embedding_calls.append((model, rows))
                raise self.original
            result = super().embeddings(input, model=model, **kwargs)
            self.peer_completed.set()
            return result

    client = ParallelClient()
    adapter = configured(client, embedding_batch_wait_sec=0.5, batch_size=4)
    incoming = ["bad", "first", "second", "third"]
    executor = ThreadPoolExecutor(1)
    try:
        # Full-engine entry proves a blocked isolation descendant retains its
        # lane without serializing an independent alias through a new queue.
        future = executor.submit(
            adapter._embed_texts_direct, incoming, policy=policy(adapter)
        )
        assert client.child_started.wait(3)
        assert client.peer_completed.wait(3)
        assert not future.done()
        scheduler = adapter._scheduler.snapshot()
        assert scheduler["in_flight"] >= 1
        assert len({alias for alias, _ in client.embedding_calls}) == 2
        client.child_release.set()
        future.result(3)
        eventually(lambda: adapter._scheduler.snapshot()["in_flight"] == 0)
        assert not client.token_calls
    finally:
        client.child_release.set()
        executor.shutdown(wait=True)
        adapter.close()


def test_unsplittable_singleton_retires_only_its_owner():
    with batch() as (adapter, client, (left, right), admitted):
        original = ClientHTTPError(500)
        client.failure = lambda _model, rows: original if "doc: x" in rows else None
        result = execute(
            adapter, "embed_0", admitted, [(left, "x"), (right, "healthy")]
        )
        assert result == [(1, vector("doc: healthy"))]
        assert left.failure is original and right.failure is None
        assert len(client.embedding_calls) == 3
        assert not client.token_calls


def test_parent_500_does_not_extend_owner_deadline_for_isolation():
    with batch() as (adapter, client, (owner, _), admitted):
        original = ClientHTTPError(500)

        def fail(_model, _rows):
            owner.deadline = time.monotonic() - 1
            return original

        client.failure = fail
        with pytest.raises(LLMTimeoutError):
            execute(adapter, "embed_0", admitted, [(owner, "first"), (owner, "second")])
        assert len(client.embedding_calls) == 1
        assert not client.token_calls


def test_non500_descendant_preserves_its_error_without_more_replay():
    with batch() as (adapter, client, (owner, _), admitted):
        parent, child = ClientHTTPError(500), ClientHTTPError(503)
        client.failure = lambda _model, rows: parent if len(rows) > 1 else child
        with pytest.raises(ClientHTTPError) as caught:
            execute(adapter, "embed_0", admitted, [(owner, "first"), (owner, "second")])
        assert caught.value is child
        assert client.embedding_calls == [
            ("embed_0", ("doc: first", "doc: second")),
            ("embed_0", ("doc: first",)),
        ]
        assert not client.token_calls


@pytest.mark.parametrize("size,bad_indices", [(1, [0]), (3, [1]), (7, [0, 3, 6])])
def test_odd_subsets_preserve_original_indices_and_successes(size, bad_indices):
    with batch() as (adapter, client, (owner, _), admitted):
        incoming = [f"source-{index}" for index in range(size)]
        bad_rows = {"doc: " + incoming[index] for index in bad_indices}
        successful = Counter()

        def fail(_model, rows):
            if bad_rows.intersection(rows):
                return ClientHTTPError(500)
            successful.update(rows)
            return None

        client.failure = fail
        result = dict(
            execute(adapter, "embed_0", admitted, [(owner, text) for text in incoming])
        )
        assert set(result) == set(range(size))
        for index, text in enumerate(incoming):
            if index in bad_indices:
                assert isinstance(result[index], _RejectedEmbedding)
            else:
                assert result[index] == vector("doc: " + text)
        assert successful == Counter(
            "doc: " + text
            for index, text in enumerate(incoming)
            if index not in bad_indices
        )
        assert len(client.embedding_calls) <= 2 * size - 1
        assert not client.token_calls


def test_successful_duplicate_occurrences_are_not_deduplicated_or_replayed():
    with batch() as (adapter, client, (owner, _), admitted):
        incoming = ["same", "bad-row", "same", "last"]
        successful = Counter()

        def fail(_model, rows):
            if "doc: bad-row" in rows:
                return ClientHTTPError(500)
            successful.update(rows)
            return None

        client.failure = fail
        result = dict(
            execute(adapter, "embed_0", admitted, [(owner, text) for text in incoming])
        )
        assert result[0] == result[2] == vector("doc: same")
        assert successful == Counter({"doc: same": 2, "doc: last": 1})
        assert isinstance(result[1], _RejectedEmbedding)
        assert not client.token_calls


def test_singleton_source_metadata_excludes_prefix_and_preserves_utf8_halves():
    with batch(prefix="special: ") as (adapter, client, (owner, _), admitted):
        source = "é🍰中a?"
        original = ClientHTTPError(500)
        client.failure = lambda *_: original
        result = dict(execute(adapter, "embed_0", admitted, [(owner, source)]))
        marker = result[0]
        assert marker.text_sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
        assert marker.char_count == len(source)
        assert marker.utf8_count == len(source.encode("utf-8"))
        assert marker.halves[0].char_end == len(source) // 2
        assert marker.halves[0].utf8_end == len(source[:2].encode("utf-8"))
        assert marker.halves[1].utf8_start == marker.halves[0].utf8_end
        assert marker.halves[1].utf8_end == len(source.encode("utf-8"))
        assert len(client.embedding_calls) == 1
        assert not client.token_calls


@pytest.mark.parametrize("source", ["", "x", "🍰"])
def test_unsplittable_source_preserves_original_error(source):
    with batch() as (adapter, client, (owner, _), admitted):
        original = ClientHTTPError(500)
        client.failure = lambda *_: original
        assert execute(adapter, "embed_0", admitted, [(owner, source)]) == []
        assert owner.failure is original
        assert len(client.embedding_calls) == 1
        assert not client.token_calls


def test_cancelled_isolation_owner_sends_no_later_sibling():
    with batch() as (adapter, client, (owner, _), admitted):

        def fail(_model, _rows):
            if len(client.embedding_calls) == 2:
                owner.stop.set()
            return ClientHTTPError(500)

        client.failure = fail
        with pytest.raises(LLMCancelledError):
            execute(
                adapter,
                "embed_0",
                admitted,
                [(owner, text) for text in ["first", "second", "third"]],
            )
        assert len(client.embedding_calls) == 2
        assert not client.token_calls


def test_removed_alias_after_parent_failure_prevents_child_post():
    with batch() as (adapter, client, (owner, _), admitted):

        def fail(_model, _rows):
            client.catalog = []
            return ClientHTTPError(500)

        client.failure = fail
        with pytest.raises(LLMBackendError):
            execute(adapter, "embed_0", admitted, [(owner, "first"), (owner, "second")])
        assert len(client.embedding_calls) == 1
        assert not client.token_calls


def test_isolated_descendants_validate_dimensions_per_owner_without_markers():
    with batch(dimension=None) as (adapter, client, (left, right), admitted):
        embed = client.embeddings

        def embeddings(input, **kwargs):
            payload = embed(input, **kwargs)
            for row in payload["data"]:
                if input[row["index"]] in ("doc: wrong-dimension", "doc: peer"):
                    row["embedding"].append(7.0)
            return payload

        client.embeddings = embeddings
        client.failure = lambda _model, rows: (
            ClientHTTPError(500) if "doc: rejected" in rows else None
        )
        result = execute(
            adapter,
            "embed_0",
            admitted,
            [
                (left, "ordinary"),
                (left, "rejected"),
                (left, "wrong-dimension"),
                (right, "peer"),
            ],
        )
        assert tuple(index for index, _ in result) == (0, 1, 2, 3)
        assert result[-1] == (3, vector("doc: peer") + [7.0])
        assert isinstance(left.failure, LLMBackendError)
        assert str(left.failure) == "llamacpp_embedding_dimension_mismatch"
        assert right.failure is None
        assert not client.token_calls


def test_true_http_two_alias_connections_overlap_while_one_lane_isolates():
    from test_embedding_concurrency import server

    with server() as (state, url):
        metadata = ContractClient(2)
        state.catalog = metadata.catalog
        state.contexts = dict(metadata.contexts)
        state.props_overrides = {
            alias: {"build_info": "test-build-1", "total_slots": 1}
            for alias in state.contexts
        }
        roots_overlap = Barrier(2)
        child_started, child_release, peer_completed = Event(), Event(), Event()
        root_handlers, successful = {}, Counter()

        def action(body):
            rows = tuple(body["input"])
            if len(rows) == 2:
                with state.lock:
                    root_handlers[body["model"]] = get_ident()
                roots_overlap.wait(3)
            if "bad-0" in rows:
                if len(rows) == 1:
                    child_started.set()
                    assert child_release.wait(3)
                return 500, {"error": {"type": "server_error"}}
            if len(rows) == 2:
                assert child_started.wait(3)
                peer_completed.set()
            with state.lock:
                successful.update(rows)
            return 200, {
                "data": [
                    {"index": index, "embedding": vector(text)}
                    for index, text in enumerate(rows)
                ]
            }

        state.action = action
        client = LlamaCppClient(url, timeout=5, trust_env=False)
        adapter = configured(client, batch_size=4, timeout=5)
        executor = ThreadPoolExecutor(1)
        try:
            future = executor.submit(
                adapter._embed_texts_direct,
                ["bad-0", "good-1", "good-2", "good-3"],
                policy=policy(adapter),
            )
            assert child_started.wait(3)
            assert peer_completed.wait(3)
            assert not future.done()
            assert state.peak == 2
            assert len(root_handlers) == len(set(root_handlers.values())) == 2
            assert all(peak == 1 for peak in state.peak_by_model.values())
            child_release.set()
            result = future.result(3)
            assert result.outputs == (
                None,
                vector("good-1"),
                vector("good-2"),
                vector("good-3"),
            )
            assert tuple(result.rejections) == (0,)
            assert successful == Counter({"good-1": 1, "good-2": 1, "good-3": 1})
            assert not state.tokenize_calls
            assert adapter._scheduler.snapshot()["in_flight"] == 0
        finally:
            child_release.set()
            executor.shutdown(wait=True)
            adapter.close()
            client.close()
