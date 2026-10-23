"""Tokenizer-free failed-POST isolation, fixed budgets and drained ownership."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
from threading import Event

import pytest

from runtime_llama import (
    EmbeddingChunkSplitRequiredError,
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
from test_embedding_contracts import vector


def failing_native(client):
    error = ClientHTTPError(500, error_type="server_error")
    client.failure = lambda model, rows: (
        error if any(len(row) + 1 > client.contexts[model] for row in rows) else None
    )
    return error


def assert_halves(item, source):
    assert item.text_sha256 == hashlib.sha256(source.encode()).hexdigest()
    assert item.char_count == len(source)
    assert item.utf8_count == len(source.encode())
    assert not hasattr(item, "token_count") and not hasattr(item, "token_limit")
    assert len(item.halves) == 2
    assert item.halves[0].char_end == len(source) // 2
    pieces = []
    for span in item.halves:
        piece = source[span.char_start : span.char_end]
        assert source.encode()[span.utf8_start : span.utf8_end].decode() == piece
        assert not hasattr(span, "token_count")
        pieces.append(piece)
    assert "".join(pieces) == source
    assert all(0 < len(piece) < len(source) for piece in pieces)
    return pieces


def block_isolation(client):
    """Only a child POST blocks, so cancellation exercises native isolation."""
    original = client.embeddings
    child_started = Event()

    def embeddings(*args, **kwargs):
        if client.embedding_calls:
            client.embed_gate.clear()
            child_started.set()
        return original(*args, **kwargs)

    client.embeddings = embeddings
    return child_started


def test_failed_post_isolates_one_source_and_returns_already_completed_rows(
    make_service,
):
    client = remote(replicas=1, context=64)
    original = failing_native(client)
    service, _ = make_service(client, batch_size=3)
    source = "α🙂漢字" * 20
    with pytest.raises(EmbeddingChunkSplitRequiredError) as caught:
        service.embed_texts(["good", source, "also good"])
    error = caught.value
    assert error.__cause__.status_code == original.status_code
    assert error.__cause__.error_type == original.error_type
    assert error.input_count == 3 and error.failed_input_indices == (1,)
    assert error.window_index is None and error.input_type == "document"
    assert len(error.inputs) == 1 and error.inputs[0].input_index == 1
    pieces = assert_halves(error.inputs[0], source)
    assert [(row.input_index, row.embedding) for row in error.completed] == [
        (0, tuple(vector("search_document: good"))),
        (2, tuple(vector("search_document: also good"))),
    ]
    assert not hasattr(error, "budget")
    assert error.representation_id == service.initialize().representation_id
    assert not client.token_calls and not service.config.cache_dir.exists()
    assert not service._operations
    # A consumer retries the rejected source halves only; successful indexed
    # embeddings are already available and are not calculated again.
    before = len(client.embedding_calls)
    assert service.embed_texts(pieces) == [
        vector("search_document: " + piece) for piece in pieces
    ]
    retried = [text for _, rows in client.embedding_calls[before:] for text in rows]
    assert retried == ["search_document: " + piece for piece in pieces]
    assert not client.token_calls


def test_generic_singleton500_returns_halving_advice_without_claiming_overflow(
    make_service,
):
    client = remote(replicas=1, context=64)
    original = ClientHTTPError(500, error_type="server_error")
    client.failure = lambda *_: original
    service, _ = make_service(client)
    with pytest.raises(EmbeddingChunkSplitRequiredError) as caught:
        service.embed_texts(["short"])
    assert caught.value.__cause__.status_code == original.status_code
    assert caught.value.__cause__.error_type == original.error_type
    assert caught.value.original_status_code == 500
    assert_halves(caught.value.inputs[0], "short")
    assert len(client.embedding_calls) == 1 and not client.token_calls
    assert not service.config.cache_dir.exists()


@pytest.mark.parametrize("status", [400, 429, 502, 503])
def test_non500_never_isolates_tokenizes_or_persists(make_service, status):
    client = remote(replicas=1, context=64)
    original = ClientHTTPError(status)
    client.failure = lambda *_: original
    service, _ = make_service(client)
    with pytest.raises(ClientHTTPError) as caught:
        service.embed_texts(["x" * 100, "okay"])
    assert caught.value is original
    assert not client.token_calls and len(client.embedding_calls) == 1
    assert not service.config.cache_dir.exists()


def test_refresh_without_any_state_never_calibrates(make_service):
    service, client = make_service()
    result = service.refresh_character_budget()
    assert result.available and result.max_input_chars is None
    assert result.reason == "character_budget_not_calibrated"
    assert result.budget is None and result.calibration_id is None
    assert not client.token_calls and not service.config.cache_dir.exists()


def test_failed_post_does_not_create_learned_state_for_later_restarts(make_service):
    client = remote(replicas=1, context=64)
    failing_native(client)
    service, _ = make_service(client)
    with pytest.raises(EmbeddingChunkSplitRequiredError):
        service.embed_texts(["x" * 80])
    service.close()
    restarted, next_client = make_service(remote(replicas=1, context=64))
    for task in ("document", "query", "raw"):
        status = restarted.refresh_character_budget(category="code", input_type=task)
        assert status.max_input_chars is None and status.budget is None
    assert not next_client.token_calls and not service.config.cache_dir.exists()


def test_cancel_during_native_subbatch_isolation_drains_without_tokenization(
    make_service,
):
    client = remote(replicas=1, context=64)
    failing_native(client)
    child_started = block_isolation(client)
    cancel = Event()
    service, _ = make_service(client)
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(
            service.embed_texts, ["okay", "x" * 80], cancel=cancel.is_set
        )
        assert child_started.wait(2)
        cancel.set()
        with pytest.raises(LLMCancelledError):
            future.result(2)
    assert client.embed_active == client.token_active == 0
    assert not service._operations and not service.config.cache_dir.exists()
    assert not client.token_calls


def test_native_subbatch_isolation_retains_original_sync_deadline(make_service):
    client = remote(replicas=1, context=64)
    failing_native(client)
    child_started = block_isolation(client)
    service, _ = make_service(client)
    with pytest.raises(LLMTimeoutError):
        service.embed_texts(["okay", "x" * 80], timeout=0.12)
    assert child_started.is_set()
    assert client.token_active == client.embed_active == 0
    assert not service._operations and not service.config.cache_dir.exists()
    assert not client.token_calls


def test_async_rejection_keeps_window_provenance_and_no_token_count_promise(
    make_service,
):
    client = remote(replicas=1, context=64)
    original = failing_native(client)
    service, _ = make_service(client)

    async def scenario():
        async with service.astream_texts(
            WindowSource([["x" * 80]]), max_pending_windows=1
        ) as stream:
            with pytest.raises(EmbeddingChunkSplitRequiredError) as caught:
                await anext(stream)
            assert caught.value.window_index == 0
            assert caught.value.failed_input_indices == (0,)
            assert caught.value.__cause__.status_code == original.status_code
            assert caught.value.__cause__.error_type == original.error_type
            assert_halves(caught.value.inputs[0], "x" * 80)
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
        await service.aclose()

    run(scenario)
    assert not service._operations and not service._streams
    assert not client.token_calls


def test_async_task_cancel_interrupts_native_isolation_and_drains(make_service):
    client = remote(replicas=1, context=64)
    failing_native(client)
    child_started = block_isolation(client)
    service, _ = make_service(client)

    async def scenario():
        stream = service.astream_texts(WindowSource([["okay", "x" * 80]]))
        waiter = asyncio.create_task(anext(stream))
        await wait_thread(child_started)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await stream.aclose()
        await service.aclose()

    run(scenario)
    assert client.token_active == client.embed_active == 0
    assert not service._operations and not service._streams
    assert not service.config.cache_dir.exists() and not client.token_calls


@pytest.mark.parametrize("text", ["", "x", "🙂"])
def test_source_without_two_nonempty_halves_preserves_original500(make_service, text):
    client = remote(replicas=1, context=8)
    original = failing_native(client)
    service, _ = make_service(client)
    with pytest.raises(ClientHTTPError) as caught:
        service.embed_texts([text])
    assert caught.value is original
    assert not service.config.cache_dir.exists() and not client.token_calls


def test_source_containing_task_prefix_is_halved_without_stripping_source(
    make_service,
):
    client = remote(replicas=1, context=64)
    failing_native(client)
    service, _ = make_service(client)
    source = "search_document: " + "x" * 80
    with pytest.raises(EmbeddingChunkSplitRequiredError) as caught:
        service.embed_texts([source])
    assert_halves(caught.value.inputs[0], source)
    assert client.embedding_calls[0][1] == ("search_document: " + source,)
    assert not client.token_calls


def test_calibrated_mean_and_factors_remain_fixed_after_native_rejection(
    make_service, calibration
):
    client = remote(replicas=1, context=64)
    service, _ = make_service(client)
    initial = service.character_budget()
    assert initial.max_input_chars == int(64 * 4 * 0.9)
    before = len(client.token_calls)
    failing_native(client)
    with pytest.raises(EmbeddingChunkSplitRequiredError):
        service.embed_texts(["x" * 120])
    mixed = service.refresh_character_budget()
    code = service.refresh_character_budget(category="code")
    assert mixed == initial
    assert code.max_input_chars == int(64 * 4 * 0.8)
    assert code.budget.budget_revision == mixed.budget.budget_revision == 0
    assert code.budget.source == mixed.budget.source == "calibrated_mean"
    assert len(client.token_calls) == before and len(calibration) == 1
    assert not service.config.cache_dir.exists()


def test_existing_lower_recommendation_is_not_reduced_again(make_service, calibration):
    client = remote(replicas=1, context=64)
    service, _ = make_service(client)
    initial = service.character_budget()
    failing_native(client)
    source = "x" * 120
    with pytest.raises(EmbeddingChunkSplitRequiredError) as caught:
        service.embed_texts([source])
    pieces = assert_halves(caught.value.inputs[0], source)
    assert service.refresh_character_budget() == initial
    # Advice is a midpoint split, not a claim that each half fits the server.
    assert all(len("search_document: " + piece) + 1 > 64 for piece in pieces)
