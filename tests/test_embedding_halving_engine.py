"""Persistent engine owner, occurrence and retained-result recovery contracts."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from runtime_llama.adapters.embedding import _Call, _TextInputs
from runtime_llama.client import ClientHTTPError
from runtime_llama.embedding_halving import _HalvingResult, _RejectedEmbedding
from runtime_llama.errors import LLMBackendError, LLMCancelledError

from test_embedding_contracts import ContractClient, configured, eventually, vector
from test_embedding_direct import policy


def submit(adapter, rows, *, retained=False, cancel=None, window_index=None):
    call = _Call(adapter, cancel)
    call._direct_policy = chosen = policy(adapter)
    call._window_index = window_index
    future = adapter._submit_execute(
        _TextInputs(rows, chosen.prefix, "document"),
        operation="embed",
        call=call,
        retain_result=retained,
    )
    # These tests deliberately exercise the private engine boundary. Keep its
    # caller-owned transfer protocol explicit instead of releasing on done().
    adapter.__dict__.setdefault("_test_private_calls", []).append((call, future))
    return call, future


def release_private_calls(adapter):
    """Transfer/discard only after the original engine work really drains."""
    calls = adapter.__dict__.pop("_test_private_calls", [])
    for call, future in calls:
        if not future.done():
            call.stop.set()
    adapter._engine.wake()
    for call, future in calls:
        while True:
            try:
                future.result()
            except BaseException:
                # Interruption of the wait cannot abandon live engine work.
                # The tests assert their typed worker outcomes separately.
                if not future.done():
                    continue
            break
        try:
            adapter._engine.release_result(call, wait=True)
        finally:
            call._intake_lease.release()


def reject(client, bad):
    original = ClientHTTPError(500, error_type="server_error")
    client.failure = lambda _model, rows: (
        original if any(row in bad for row in rows) else None
    )
    return original


def assert_clean(adapter, client):
    eventually(lambda: not adapter._calls)
    assert adapter._scheduler.snapshot()["in_flight"] == 0
    assert not client.token_calls
    snapshot = adapter._engine.snapshot()
    assert snapshot["pending_bytes"] == snapshot["pending_tokens"] == 0
    assert snapshot["active_plans"] == 0


@pytest.mark.parametrize("retained", [False, True])
def test_first_rejection_rearms_unknown_dimension_and_preserves_every_occurrence(
    retained,
):
    client = ContractClient(1)
    adapter = configured(
        client, dimension=None, batch_size=16, embedding_batch_wait_sec=0, timeout=5
    )
    reject(client, {"bad"})
    rows = ["bad", "bad", "good-one", "good-two", "bad", "good-three"]
    try:
        call, future = submit(adapter, rows, retained=retained, window_index=7)
        result = future.result(4)
        assert isinstance(result, _HalvingResult)
        assert result.window_index == 7
        assert sorted(result.rejections) == [0, 1, 4]
        assert result.outputs == tuple(
            None if row == "bad" else vector(row) for row in rows
        )
        assert all(
            isinstance(marker, _RejectedEmbedding)
            for marker in result.rejections.values()
        )
        assert adapter.embedding_dim == 2
        assert call.failure is None and not call.stop.is_set()
        assert_clean(adapter, client)
        if retained:
            expected = adapter._engine._output_bytes(result)
            assert adapter._engine.snapshot()["retained_result_bytes"] == expected
            adapter._engine.release_result(call)
            eventually(lambda: adapter._engine.snapshot()["result_reserved_bytes"] == 0)
    finally:
        release_private_calls(adapter)
        adapter.close()


def test_all_rejected_unknown_dimension_windows_complete_and_release_scalar_credits():
    client = ContractClient(1)
    adapter = configured(client, dimension=None, embedding_batch_wait_sec=0, timeout=5)
    reject(client, {"bad-one", "bad-two", "bad-three"})
    try:
        call, future = submit(
            adapter, ["bad-one", "bad-two", "bad-three"], retained=True
        )
        result = future.result(4)
        assert isinstance(result, _HalvingResult)
        assert result.outputs == (None, None, None)
        assert tuple(result.rejections) == (0, 1, 2)
        assert adapter.embedding_dim == 0
        assert adapter._engine._unknown_seeds == 0
        expected = 64 + 3 * (96 + 640)
        assert adapter._engine.snapshot()["retained_result_bytes"] == expected
        assert adapter._engine.snapshot()["result_reserved_bytes"] == expected
        adapter._engine.release_result(call)
        eventually(lambda: adapter._engine.snapshot()["result_reserved_bytes"] == 0)
        assert_clean(adapter, client)
        assert adapter._embed_texts_direct(["next-good"], policy=policy(adapter)) == [
            vector("next-good")
        ]
    finally:
        release_private_calls(adapter)
        adapter.close()


def test_rejection_metadata_obeys_retained_result_limit_and_drains():
    client = ContractClient(1)
    adapter = configured(
        client,
        embedding_max_pending_bytes=2048,
        embedding_batch_wait_sec=0,
        embedding_request_parallelism=1,
        timeout=5,
    )
    reject(client, {"bad-one", "bad-two", "bad-three"})
    try:
        _, future = submit(adapter, ["bad-one", "bad-two", "bad-three"], retained=True)
        with pytest.raises(LLMBackendError, match="result_bytes_limit"):
            future.result(4)
        eventually(lambda: adapter._engine.snapshot()["result_reserved_bytes"] == 0)
        assert adapter._engine.snapshot()["peak_result_reserved_bytes"] <= 2048
        assert_clean(adapter, client)
    finally:
        release_private_calls(adapter)
        adapter.close()


def test_marker_credit_is_reserved_before_retained_publication(monkeypatch):
    client = ContractClient(1)
    adapter = configured(client, embedding_batch_wait_sec=0, timeout=5)
    adapter._embed_texts_direct(["learn-dimension"], policy=policy(adapter))
    reject(client, {"bad"})
    publish = adapter._engine._publish_http_result
    observed = []

    def credited(state, destinations, values):
        if state.retain_result:
            wanted = (
                64
                + 96 * len(state.inputs)
                + 40
                * adapter._engine._vector_dim
                * (len(state.inputs) - state.rejection_slots)
                + 640 * state.rejection_slots
            )
            observed.append((state.result_credit, wanted))
        return publish(state, destinations, values)

    monkeypatch.setattr(adapter._engine, "_publish_http_result", credited)
    try:
        call, future = submit(adapter, ["bad", "good"], retained=True)
        result = future.result(4)
        assert isinstance(result, _HalvingResult)
        assert observed and all(credit >= wanted for credit, wanted in observed)
        adapter._engine.release_result(call)
        eventually(lambda: adapter._engine.snapshot()["result_reserved_bytes"] == 0)
    finally:
        release_private_calls(adapter)
        adapter.close()


def test_shared_post_has_distinct_owner_local_indices_and_never_replays_success():
    client = ContractClient(1)
    client.embed_gate.clear()
    adapter = configured(
        client,
        batch_size=8,
        embedding_batch_wait_sec=0.5,
        embedding_request_parallelism=1,
        timeout=5,
    )
    successes = Counter()
    original = ClientHTTPError(500)

    def failure(_model, rows):
        if "bad" in rows:
            return original
        successes.update(rows)
        return None

    client.failure = failure
    try:
        warm_call, warm = submit(adapter, ["warm"])
        assert client.embed_started.wait(2)
        left_call, left = submit(adapter, ["bad", "left-good", "bad"])
        right_call, right = submit(adapter, ["right-good", "bad"])
        eventually(
            lambda: adapter._engine.snapshot()["ready_by_replica"].get("embed_0") == 5
        )
        client.embed_gate.set()
        assert warm.result(3) == [vector("warm")]
        left_result, right_result = left.result(3), right.result(3)
        assert isinstance(left_result, _HalvingResult)
        assert isinstance(right_result, _HalvingResult)
        assert left_result.outputs == (None, vector("left-good"), None)
        assert right_result.outputs == (vector("right-good"), None)
        assert sorted(left_result.rejections) == [0, 2]
        assert sorted(right_result.rejections) == [1]
        assert any(
            "left-good" in rows and "right-good" in rows
            for _, rows in client.embedding_calls
        )
        assert successes == Counter({"warm": 1, "left-good": 1, "right-good": 1})
        assert all(call.failure is None for call in (warm_call, left_call, right_call))
        assert_clean(adapter, client)
    finally:
        client.embed_gate.set()
        release_private_calls(adapter)
        adapter.close()


def test_paused_all_rejected_windows_keep_bounded_credits_and_resume_on_release():
    client = ContractClient(1)
    adapter = configured(
        client,
        # No successful vector ever establishes a width in this scalar-credit
        # regression; keep the cold contract explicit after the shared default.
        dimension=None,
        batch_size=1,
        embedding_batch_wait_sec=0,
        embedding_request_parallelism=1,
        embedding_queue_size=7,
        embedding_max_pending_bytes=4096,
        timeout=5,
    )
    reject(client, {f"bad-{index}" for index in range(8)})
    calls = []
    try:
        calls = [submit(adapter, [f"bad-{index}"], retained=True) for index in range(8)]
        eventually(lambda: sum(future.done() for _, future in calls) == 5)
        assert all(
            isinstance(future.result(), _HalvingResult)
            for _, future in calls
            if future.done()
        )
        assert adapter._engine.snapshot()["retained_result_bytes"] == 4000
        assert adapter._engine.snapshot()["peak_result_reserved_bytes"] <= 4096
        completed = next(call for call, future in calls if future.done())
        adapter._engine.release_result(completed)
        eventually(lambda: sum(future.done() for _, future in calls) >= 6)
        released = {completed}
        for _ in range(8):
            for call, future in calls:
                if future.done() and call not in released:
                    assert isinstance(future.result(), _HalvingResult)
                    adapter._engine.release_result(call)
                    released.add(call)
            if len(released) == len(calls):
                break
            eventually(
                lambda: any(
                    future.done() and call not in released for call, future in calls
                )
            )
        assert len(released) == 8
        eventually(lambda: adapter._engine.snapshot()["result_reserved_bytes"] == 0)
        assert adapter._engine.snapshot()["peak_retained_result_bytes"] <= 4096
        assert_clean(adapter, client)
    finally:
        for call, _ in calls:
            adapter._engine.release_result(call)
        release_private_calls(adapter)
        adapter.close()


def test_unsplittable_singleton_fails_only_its_owner_in_shared_http_batch():
    client = ContractClient(1)
    client.embed_gate.clear()
    adapter = configured(
        client,
        batch_size=8,
        embedding_batch_wait_sec=0.5,
        embedding_request_parallelism=1,
        timeout=5,
    )
    original = reject(client, {"x"})
    try:
        _, warm = submit(adapter, ["warm"])
        assert client.embed_started.wait(2)
        left_call, left = submit(adapter, ["x", "left-good"])
        right_call, right = submit(adapter, ["right-good", "right-two"])
        eventually(
            lambda: adapter._engine.snapshot()["ready_by_replica"].get("embed_0") == 4
        )
        client.embed_gate.set()
        assert warm.result(3) == [vector("warm")]
        with pytest.raises(ClientHTTPError) as caught:
            left.result(3)
        assert caught.value is original and left_call.failure is original
        assert right.result(3) == [vector("right-good"), vector("right-two")]
        assert right_call.failure is None
        assert any(
            "x" in rows and "right-good" in rows for _, rows in client.embedding_calls
        )
        assert_clean(adapter, client)
    finally:
        client.embed_gate.set()
        release_private_calls(adapter)
        adapter.close()


def test_cancelling_owner_during_isolation_drains_it_and_keeps_shared_peer():
    class GatedIsolation(ContractClient):
        def __init__(self):
            super().__init__(1)
            self.child_started = Event()
            self.child_release = Event()

        def embeddings(self, input, *, model, **kwargs):
            if tuple(input) == ("bad",):
                self.child_started.set()
                self._wait(
                    self.child_release,
                    cancel=kwargs.get("cancel"),
                    timeout=kwargs.get("timeout"),
                )
            return super().embeddings(input, model=model, **kwargs)

    client = GatedIsolation()
    client.embed_gate.clear()
    adapter = configured(
        client,
        batch_size=8,
        embedding_batch_wait_sec=0.5,
        embedding_request_parallelism=1,
        timeout=5,
    )
    reject(client, {"bad"})
    try:
        _, warm = submit(adapter, ["warm"])
        assert client.embed_started.wait(2)
        left_call, left = submit(adapter, ["bad", "left-good"])
        right_call, right = submit(adapter, ["right-good", "right-two"])
        eventually(
            lambda: adapter._engine.snapshot()["ready_by_replica"].get("embed_0") == 4
        )
        client.embed_gate.set()
        assert warm.result(3) == [vector("warm")]
        assert client.child_started.wait(2)
        left_call.stop.set()
        adapter._engine.wake()
        with pytest.raises(LLMCancelledError):
            left.result(3)
        assert right.result(3) == [vector("right-good"), vector("right-two")]
        assert right_call.failure is None
        assert_clean(adapter, client)
    finally:
        client.child_release.set()
        client.embed_gate.set()
        release_private_calls(adapter)
        adapter.close()


def test_close_while_isolation_child_is_running_drains_scheduler_and_owner():
    client = ContractClient(1)
    child_started, child_release = Event(), Event()
    original = ClientHTTPError(500)
    request = client.embeddings

    def embeddings(input, *, model, **kwargs):
        if tuple(input) == ("bad",):
            child_started.set()
            client._wait(
                child_release,
                cancel=kwargs.get("cancel"),
                timeout=kwargs.get("timeout"),
            )
        return request(input, model=model, **kwargs)

    client.embeddings = embeddings
    client.failure = lambda _model, rows: original if "bad" in rows else None
    adapter = configured(client, embedding_batch_wait_sec=0, timeout=5)
    executor = ThreadPoolExecutor(1)
    try:
        _, future = submit(adapter, ["bad", "good"])
        assert child_started.wait(2)
        closing = executor.submit(adapter.close)
        with pytest.raises((LLMCancelledError, LLMBackendError)) as stopped:
            future.result(3)
        if not isinstance(stopped.value, LLMCancelledError):
            assert str(stopped.value) == "llamacpp_embedding_adapter_closed"
        # Synchronous close correctly waits for the caller to release its
        # private owner after observing the drained engine Future.
        release_private_calls(adapter)
        closing.result(3)
        assert_clean(adapter, client)
        assert not adapter._engine._thread.is_alive()
        assert adapter._engine.snapshot()["result_reserved_bytes"] == 0
    finally:
        child_release.set()
        executor.shutdown(wait=True)
        release_private_calls(adapter)
        adapter.close()


def test_rejected_rows_release_their_absent_vector_reservation(monkeypatch):
    client = ContractClient(1)
    adapter = configured(client, dimension=64, embedding_batch_wait_sec=0, timeout=5)
    original = ClientHTTPError(500)

    def embeddings(input, *, model, **_kwargs):
        rows = tuple(input)
        client.embedding_calls.append((model, rows))
        if "bad" in rows:
            raise original
        return {
            "data": [
                {"index": i, "embedding": [float(i)] * 64} for i, _ in enumerate(rows)
            ]
        }

    monkeypatch.setattr(client, "embeddings", embeddings)
    credits = []
    publish = adapter._engine._publish_http_result

    def capture(state, destinations, values):
        credits.append(state.result_credit)
        return publish(state, destinations, values)

    monkeypatch.setattr(adapter._engine, "_publish_http_result", capture)
    try:
        call, future = submit(adapter, ["bad", "good"], retained=True)
        result = future.result(4)
        assert isinstance(result, _HalvingResult)
        expected = 64 + 2 * 96 + 64 * 40 + 640
        assert credits == [expected]
        assert adapter._engine.snapshot()["retained_result_bytes"] == expected
        adapter._engine.release_result(call)
        eventually(lambda: adapter._engine.snapshot()["result_reserved_bytes"] == 0)
        assert_clean(adapter, client)
    finally:
        release_private_calls(adapter)
        adapter.close()
