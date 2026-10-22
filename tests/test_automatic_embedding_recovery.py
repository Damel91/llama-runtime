"""Public failed-POST recovery, without touching consumers or native models."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
from threading import Event

import pytest

from runtime_llama import (
    EmbeddingInputOverflowError,
    LLMBackendError,
    LLMCancelledError,
    LLMTimeoutError,
)
from runtime_llama.client import ClientHTTPError

from test_automatic_embedding import (
    calibration as calibration,
    make_service as make_service,
    remote,
)
from test_embedding_async_stream import WindowSource, run, wait_thread


def failing_native(client):
    error = ClientHTTPError(500, error_type="server_error")
    client.failure = lambda model, rows: (
        error if any(len(row) + 1 > client.contexts[model] for row in rows) else None
    )
    return error


def assert_partition(item, source, client):
    assert item.text_sha256 == hashlib.sha256(source.encode()).hexdigest()
    assert item.char_count == len(source)
    assert item.utf8_count == len(source.encode())
    pieces = []
    for span in item.fragments:
        piece = source[span.char_start : span.char_end]
        assert source.encode()[span.utf8_start : span.utf8_end].decode() == piece
        assert span.token_count == len("search_document: " + piece) + 1
        assert span.token_count <= item.token_limit
        pieces.append(piece)
    assert "".join(pieces) == source
    assert all(len(piece) < len(source) for piece in pieces)


def test_proven_overflow_returns_exact_source_spans_and_learned_only_snapshot(
    make_service,
):
    client = remote(replicas=1, context=64)
    original = failing_native(client)
    service, _ = make_service(client, batch_size=3)
    source = "α🙂漢字" * 25
    with pytest.raises(EmbeddingInputOverflowError) as caught:
        service.embed_texts(["good", source, "also good"])
    error = caught.value
    assert error.__cause__ is original
    assert error.failed_input_indices == (0, 1, 2)
    assert error.window_index is None
    assert len(error.inputs) == 1 and error.inputs[0].input_index == 1
    assert_partition(error.inputs[0], source, client)
    budget = error.budget
    assert budget.calibration_id is None and budget.initial_max_input_chars is None
    assert budget.source == "learned_reduction" and budget.budget_revision == 1
    assert budget.effective_context_tokens == 64
    assert budget.max_input_chars == max(
        span.char_end - span.char_start for span in error.inputs[0].fragments
    )
    assert len(client.embedding_calls) == 1
    assert client.token_calls[0][0] == "search_document: good"
    assert client.token_calls[1][0] == "search_document: " + source
    assert all(
        options["add_special"] and options["parse_special"]
        for _, options in client.token_calls
    )
    assert not service._operations
    # An explicit consumer retry follows exact spans; cardinality is its choice.
    pieces = [
        source[span.char_start : span.char_end] for span in error.inputs[0].fragments
    ]
    count_before = len(client.token_calls)
    assert len(service.embed_texts(["good", *pieces, "also good"])) == len(pieces) + 2
    assert len(client.token_calls) == count_before


def test_unproven_500_preserves_same_error_and_creates_no_cache(make_service):
    client = remote(replicas=1, context=64)
    original = ClientHTTPError(500, error_type="server_error")
    client.failure = lambda *_: original
    service, _ = make_service(client)
    with pytest.raises(ClientHTTPError) as caught:
        service.embed_texts(["short"])
    assert caught.value is original
    assert len(client.token_calls) == 1 and len(client.embedding_calls) == 1
    assert not service.config.cache_dir.exists()


@pytest.mark.parametrize("status", [400, 429, 502, 503])
def test_non500_never_diagnoses_or_persists(make_service, status):
    client = remote(replicas=1, context=64)
    original = ClientHTTPError(status)
    client.failure = lambda *_: original
    service, _ = make_service(client)
    with pytest.raises(ClientHTTPError) as caught:
        service.embed_texts(["x" * 100])
    assert caught.value is original
    assert not client.token_calls
    assert not service.config.cache_dir.exists()


def test_refresh_without_any_state_never_calibrates(make_service):
    service, client = make_service()
    result = service.refresh_character_budget()
    assert result.available and result.max_input_chars is None
    assert result.reason == "character_budget_not_calibrated"
    assert result.budget is None and result.calibration_id is None
    assert not client.token_calls and not service.config.cache_dir.exists()


def test_learned_ceiling_survives_restart_and_is_task_scoped(make_service):
    client = remote(replicas=1, context=64)
    failing_native(client)
    service, _ = make_service(client)
    with pytest.raises(EmbeddingInputOverflowError) as caught:
        service.embed_texts(["x" * 120])
    snapshot = caught.value.budget
    service.close()
    restarted, next_client = make_service(remote(replicas=1, context=64))
    status = restarted.refresh_character_budget(category="code")
    assert status.max_input_chars == snapshot.max_input_chars
    assert status.budget.context_scope_id == snapshot.context_scope_id
    assert status.budget.category == "code"
    assert status.budget.budget_revision == snapshot.budget_revision
    assert not next_client.token_calls
    assert restarted.refresh_character_budget(input_type="query").budget is None
    assert restarted.initialize().budget is None


def test_changed_tokenizer_after_count_prevents_persistence(make_service):
    client = remote(replicas=1, context=64)
    failing_native(client)
    tokenize = client.tokenize

    def drift(*args, **kwargs):
        result = tokenize(*args, **kwargs)
        client.builds["embed_0"] = "changed-build"
        return result

    client.tokenize = drift
    service, _ = make_service(client)
    with pytest.raises(LLMBackendError, match="identity_changed|snapshot_changed"):
        service.embed_texts(["x" * 120])
    assert not service.config.cache_dir.exists()
    assert len(client.embedding_calls) == 1


def test_cancel_during_diagnosis_drains_and_does_not_commit(make_service):
    client = remote(replicas=1, context=64)
    failing_native(client)
    client.token_gate.clear()
    cancel = Event()
    service, _ = make_service(client)
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(service.embed_texts, ["x" * 120], cancel=cancel.is_set)
        assert client.token_started.wait(2)
        cancel.set()
        with pytest.raises(LLMCancelledError):
            future.result(2)
    assert client.embed_active == client.token_active == 0
    assert not service._operations and not service.config.cache_dir.exists()


def test_diagnosis_uses_original_sync_deadline(make_service):
    client = remote(replicas=1, context=64)
    failing_native(client)
    client.token_gate.clear()
    service, _ = make_service(client)
    with pytest.raises(LLMTimeoutError):
        service.embed_texts(["x" * 120], timeout=0.12)
    assert client.token_active == client.embed_active == 0
    assert not service._operations and not service.config.cache_dir.exists()


def test_async_overflow_keeps_window_provenance_and_own_deadline(make_service):
    client = remote(replicas=1, context=64)
    original = failing_native(client)
    service, _ = make_service(client)

    async def scenario():
        async with service.astream_texts(
            WindowSource([["x" * 100]]), max_pending_windows=1
        ) as stream:
            with pytest.raises(EmbeddingInputOverflowError) as caught:
                await anext(stream)
            assert caught.value.window_index == 0
            assert caught.value.failed_input_indices == (0,)
            assert caught.value.__cause__ is original
            assert_partition(caught.value.inputs[0], "x" * 100, client)
        await service.aclose()

    run(scenario)
    assert not service._operations and not service._streams


def test_async_task_cancel_interrupts_offloop_diagnosis_and_drains(make_service):
    client = remote(replicas=1, context=64)
    failing_native(client)
    client.token_gate.clear()
    service, _ = make_service(client)

    async def scenario():
        stream = service.astream_texts(WindowSource([["x" * 120]]))
        waiter = asyncio.create_task(anext(stream))
        await wait_thread(client.token_started)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await stream.aclose()
        await service.aclose()

    run(scenario)
    assert client.token_active == client.embed_active == 0
    assert not service._operations and not service._streams
    assert not service.config.cache_dir.exists()


def test_incomplete_partition_preserves_original500_without_persist(make_service):
    client = remote(replicas=1, context=8)
    original = failing_native(client)
    service, _ = make_service(client)
    # The task prefix alone exceeds this loaded context: no nonempty source
    # fragment can be proven to fit, so advice must never be fabricated.
    with pytest.raises(ClientHTTPError) as caught:
        service.embed_texts(["xx"])
    assert caught.value is original
    assert not service.config.cache_dir.exists()


def test_consumer_source_prefix_is_not_stripped_twice(make_service):
    client = remote(replicas=1, context=64)
    failing_native(client)
    service, _ = make_service(client)
    text = "search_document: " + "x" * 100
    with pytest.raises(EmbeddingInputOverflowError) as caught:
        service.embed_texts([text])
    assert_partition(caught.value.inputs[0], text, client)
    assert client.token_calls[0][0] == "search_document: " + text


def test_initial_factor_is_not_applied_again_to_exact_learned_ceiling(
    make_service, calibration
):
    client = remote(replicas=1, context=64)
    service, _ = make_service(client)
    initial = service.character_budget()
    assert initial.max_input_chars == int(64 * 4 * 0.9)
    failing_native(client)
    with pytest.raises(EmbeddingInputOverflowError) as caught:
        service.embed_texts(["x" * 120])
    error = caught.value
    measured_maximum = max(
        span.char_end - span.char_start for span in error.inputs[0].fragments
    )
    assert error.budget.max_input_chars == measured_maximum
    assert error.budget.initial_max_input_chars == initial.max_input_chars
    assert error.budget.calibration_id == initial.calibration_id
    before = len(client.token_calls)
    warm = service.refresh_character_budget(category="code")
    assert warm.max_input_chars == measured_maximum
    assert warm.budget.budget_revision == error.budget.budget_revision
    assert len(client.token_calls) == before and len(calibration) == 1


def test_already_lower_initial_recommendation_still_returns_strict_partition(
    make_service, calibration, monkeypatch
):
    from runtime_llama.embedding_calibration import CalibrationProfile

    monkeypatch.setattr(CalibrationProfile, "max_input_chars", lambda *_: 20)
    client = remote(replicas=1, context=64)
    service, _ = make_service(client)
    initial = service.character_budget()
    failing_native(client)
    with pytest.raises(EmbeddingInputOverflowError) as caught:
        service.embed_texts(["x" * 120])
    error = caught.value
    assert error.budget.max_input_chars == initial.max_input_chars == 20
    assert len(error.inputs[0].fragments) >= 2
    assert_partition(error.inputs[0], "x" * 120, client)
