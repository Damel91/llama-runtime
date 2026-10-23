"""Public tokenizer-free halving advice keeps completed source occurrences."""

from dataclasses import FrozenInstanceError
import hashlib

import pytest

from runtime_llama import (
    EmbeddingChunkSplitRequiredError,
    EmbeddingCompletedInput,
    EmbeddingHalvingInput,
    EmbeddingSourceSpan,
)
from runtime_llama.client import ClientHTTPError
from runtime_llama.embedding_halving import _rejected_embedding

from test_automatic_embedding import calibration as calibration  # noqa: F401
from test_automatic_embedding import make_service as make_service
from test_automatic_embedding import remote
from test_embedding_contracts import vector


def native_context_limit(client):
    original = ClientHTTPError(500)
    client.failure = lambda alias, rows: (
        original if any(len(row) + 1 > client.contexts[alias] for row in rows) else None
    )
    return original


def test_public_partial_results_keep_duplicates_indices_and_unicode_halves(
    make_service,
):
    client = remote(replicas=1, context=64)
    original = native_context_limit(client)
    service, _ = make_service(client, batch_size=4)
    long = "é🦊文" * 19
    texts = ["short", long, "short", "other"]
    with pytest.raises(EmbeddingChunkSplitRequiredError) as caught:
        service.embed_texts(texts)
    error = caught.value
    assert isinstance(error.__cause__, ClientHTTPError)
    assert error.__cause__ is not original
    assert error.__cause__.status_code == original.status_code
    assert error.__cause__.__traceback__ is None
    assert error.__cause__.__context__ is None
    assert error.input_count == 4 and error.window_index is None
    assert [item.input_index for item in error.completed] == [0, 2, 3]
    assert [item.embedding for item in error.completed] == [
        tuple(vector("search_document: " + texts[index])) for index in (0, 2, 3)
    ]
    assert error.failed_input_indices == (1,)
    item = error.inputs[0]
    assert item.text_sha256 == hashlib.sha256(long.encode()).hexdigest()
    assert not hasattr(item, "token_count") and not hasattr(error, "budget")
    left, right = item.halves
    assert left.char_end == len(long) // 2 == right.char_start
    assert left.utf8_end == right.utf8_start == len(long[: len(long) // 2].encode())
    assert (
        b"".join(long.encode()[s.utf8_start : s.utf8_end] for s in item.halves)
        == long.encode()
    )
    assert "".join(long[s.char_start : s.char_end] for s in item.halves) == long
    with pytest.raises(FrozenInstanceError):
        item.char_count = 1
    assert not client.token_calls


def test_consumer_retries_only_requested_halves_without_reembedding_good_rows(
    make_service,
):
    client = remote(replicas=1, context=64)
    native_context_limit(client)
    service, _ = make_service(client, batch_size=4)
    bad = "é🦊文" * 19
    with pytest.raises(EmbeddingChunkSplitRequiredError) as caught:
        service.embed_texts(["ok", bad, "fine"])
    error = caught.value
    before = list(client.embedding_calls)
    halves = [bad[s.char_start : s.char_end] for s in error.inputs[0].halves]
    retry = service.embed_texts(halves)
    assert retry == [vector("search_document: " + text) for text in halves]
    sent = [text for _, rows in client.embedding_calls[len(before) :] for text in rows]
    assert sorted(sent) == sorted("search_document: " + text for text in halves)
    assert error.completed[0].input_index == 0 and error.completed[1].input_index == 2
    assert not client.token_calls


def test_calibrated_budget_and_cache_are_unchanged_after_isolated_rejection(
    make_service, calibration
):
    client = remote(replicas=1, context=64)
    service, _ = make_service(client, batch_size=3)
    before = service.character_budget(category="code")
    token_calls = list(client.token_calls)
    native_context_limit(client)
    with pytest.raises(EmbeddingChunkSplitRequiredError):
        service.embed_texts(["ok", "x" * 90, "other"])
    after = service.refresh_character_budget(category="code")
    assert before.budget == after.budget
    assert before.max_input_chars == after.max_input_chars
    assert before.budget.budget_revision == 0
    assert before.budget.source == "calibrated_mean"
    assert client.token_calls == token_calls


def test_cold_rejection_needs_no_cache_and_generic_500_is_not_overflow_proof(
    make_service, tmp_path
):
    client = remote(replicas=1)
    original = ClientHTTPError(500)
    client.failure = lambda *_: original
    cache = tmp_path / "no-cache"
    service, _ = make_service(client, cache_dir=cache)
    with pytest.raises(EmbeddingChunkSplitRequiredError) as caught:
        service.embed_texts(["ab", "cd"])
    assert caught.value.failed_input_indices == (0, 1)
    assert caught.value.completed == ()
    assert not cache.exists() and not client.token_calls
    assert not hasattr(caught.value.inputs[0], "token_count")


@pytest.mark.parametrize("source", ["", "x"])
def test_impossible_utf8_or_indivisible_source_preserves_native_failure(
    make_service, source
):
    client = remote(replicas=1)
    original = ClientHTTPError(500)
    client.failure = lambda *_: original
    service, _ = make_service(client)
    with pytest.raises(ClientHTTPError) as caught:
        service.embed_texts([source])
    assert caught.value is original and not client.token_calls


def test_impossible_utf8_rejection_metadata_preserves_native_failure():
    original = ClientHTTPError(500)
    with pytest.raises(ClientHTTPError) as caught:
        _rejected_embedding("a\ud800", "document", original)
    assert caught.value is original


def test_public_contract_validates_complete_partition_and_freezes_vectors():
    spans = (EmbeddingSourceSpan(0, 1, 0, 2), EmbeddingSourceSpan(1, 3, 2, 6))
    item = EmbeddingHalvingInput(1, "a" * 64, 3, 6, spans)
    vector_input = [1.0, 2.0]
    completed = EmbeddingCompletedInput(0, vector_input)
    vector_input[0] = 42
    assert completed.embedding == (1.0, 2.0)
    error = EmbeddingChunkSplitRequiredError(
        input_count=2,
        completed=[completed],
        inputs=[item],
        representation_id="identity",
        input_type="document",
        window_index=2,
    )
    assert error.window_index == 2 and error.original_status_code == 500
    with pytest.raises(ValueError, match="partition"):
        EmbeddingChunkSplitRequiredError(
            input_count=3,
            completed=[completed],
            inputs=[item],
            representation_id="identity",
            input_type="document",
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "x", True])
def test_completed_vectors_reject_invalid_values(value):
    with pytest.raises(ValueError, match="finite"):
        EmbeddingCompletedInput(0, [value])


@pytest.mark.parametrize("changes", [(0, 0, 0, 1), (0, 2, 0, 1), (False, 1, 0, 1)])
def test_invalid_source_spans_are_rejected(changes):
    with pytest.raises(ValueError):
        EmbeddingSourceSpan(*changes)


def test_private_rejection_retains_only_source_metadata():
    marker = _rejected_embedding("é🦊x", "document", ClientHTTPError(500))
    assert not hasattr(marker, "text") and not hasattr(marker, "source_text")
    assert marker.char_count == 3 and marker.utf8_count == 7
