"""Exact overflow diagnostics expose only complete lossless recovery advice."""

from dataclasses import FrozenInstanceError, fields, replace
import hashlib

import pytest

from runtime_llama.client import ClientHTTPError
from runtime_llama.embedding_recovery import (
    EmbeddingBudgetSnapshot,
    EmbeddingInputOverflowError,
    EmbeddingOverflowInput,
    EmbeddingVerifiedSpan,
    _FailedDirectPost,
    _FailedOwnerPost,
    _FailedPostRow,
    diagnose_overflow_inputs,
)
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError


def snapshot(**changes):
    values = {
        "policy_id": "minimum-observed-v2",
        "representation_id": "representation",
        "context_scope_id": "context",
        "input_type": "document",
        "category": "mixed",
        "calibration_id": None,
        "max_input_chars": 10,
        "initial_max_input_chars": None,
        "budget_revision": 1,
        "effective_context_tokens": 12,
        "source": "learned",
    }
    values.update(changes)
    return EmbeddingBudgetSnapshot(**values)


def check():
    pass


def verified(result, originals, count_tokens, limit):
    assert result.complete
    for item in result.inputs:
        source = originals[item.input_index]
        encoded = source.encode("utf-8")
        assert item.text_sha256 == hashlib.sha256(encoded).hexdigest()
        assert item.char_count == len(source) and item.utf8_count == len(encoded)
        assert item.token_count == count_tokens(source) > limit
        assert len(item.fragments) >= 2
        chunks, byte_chunks = [], []
        for span in item.fragments:
            chunk = source[span.char_start : span.char_end]
            byte_chunk = encoded[span.utf8_start : span.utf8_end]
            assert chunk and len(chunk) < len(source)
            assert byte_chunk.decode("utf-8") == chunk
            assert span.token_count == count_tokens(chunk) <= limit
            chunks.append(chunk)
            byte_chunks.append(byte_chunk)
        assert "".join(chunks) == source
        assert b"".join(byte_chunks) == encoded


def test_fitting_failed_batch_only_counts_full_inputs_and_proposes_nothing():
    calls = []

    def count(text):
        calls.append(text)
        return len(text) + 2

    result = diagnose_overflow_inputs(
        [(3, "fit"), (10, "also fits"), (15, "fit")],
        token_limit=20,
        count_tokens=count,
        check=check,
    )
    assert result.complete and result.reason == "no_overflow"
    assert result.inputs == () and result.suggested_max_input_chars is None
    assert calls == ["fit", "also fits"]
    assert result.tokenize_requests == 2


def test_all_full_failed_inputs_measured_before_any_fragment():
    calls = []
    originals = {4: "0123456789" * 2, 11: "short", 19: "abcdefghijklmnopqrstuv"}

    def count(text):
        calls.append(text)
        return len(text)

    result = diagnose_overflow_inputs(
        list(originals.items()), token_limit=8, count_tokens=count, check=check
    )
    assert calls[:3] == list(originals.values())
    assert [item.input_index for item in result.inputs] == [4, 19]
    verified(result, originals, len, 8)
    assert result.tokenize_requests == len(calls)
    assert result.suggested_max_input_chars == min(
        max(span.char_end - span.char_start for span in item.fragments)
        for item in result.inputs
    )


def test_unicode_crlf_combining_marks_and_special_literals_remain_lossless():
    source = "α🙂\n東京 e\u0301\r\n\t<special>\x00" * 7

    def count(text):
        return len(("doc: " + text).encode("utf-8")) + 2

    result = diagnose_overflow_inputs(
        [(9, source)], token_limit=28, count_tokens=count, check=check
    )
    verified(result, {9: source}, count, 28)
    assert result.inputs[0].char_count < result.inputs[0].utf8_count
    assert any(
        span.utf8_end - span.utf8_start > span.char_end - span.char_start
        for span in result.inputs[0].fragments
    )


def test_prefix_overhead_is_counted_by_callback_for_every_fragment():
    source = "doc: source prefix belongs to the raw text " * 3
    calls = []

    def count(raw):
        calls.append(raw)
        return len("doc: " + raw) + 3

    result = diagnose_overflow_inputs(
        [(0, source)], token_limit=25, count_tokens=count, check=check
    )
    verified(result, {0: source}, lambda text: len("doc: " + text) + 3, 25)
    assert calls[0] == source
    assert result.inputs[0].fragments[0].char_start == 0
    assert source.startswith("doc: ")


def test_refinement_approaches_verified_boundary_without_another_safety_factor():
    source = "x" * 100

    def count(text):
        return len(text) + 10

    result = diagnose_overflow_inputs(
        [(0, source)], token_limit=50, count_tokens=count, check=check
    )
    verified(result, {0: source}, count, 50)
    # The ratio's first proposal (45 chars) fails. Halving only searches; the
    # measured refinement recovers 40 chars instead of a new 80/90% multiplier.
    assert result.inputs[0].fragments[0].char_end == 40
    assert result.suggested_max_input_chars == 40


def test_nonmonotonic_counts_never_authorize_an_unmeasured_cut():
    source = "abcdefghij"
    special = {source: 13, "abcd": 9, "ab": 3, "abc": 7, "abcde": 2}
    calls = []

    def count(text):
        calls.append(text)
        return special.get(text, len(text))

    assert count("abcde") < count("abcd")
    calls.clear()
    result = diagnose_overflow_inputs(
        [(0, source)], token_limit=6, count_tokens=count, check=check
    )
    verified(result, {0: source}, lambda text: special.get(text, len(text)), 6)
    assert result.inputs[0].fragments[0].char_end == 2
    assert "abcde" not in calls  # No globally maximal boundary is claimed.


def test_nonmonotonic_failed_search_is_incomplete_not_a_claim_of_impossibility():
    source = "abcd"
    table = {source: 20, "a": 7, "ab": 1}
    result = diagnose_overflow_inputs(
        [(0, source)],
        token_limit=3,
        count_tokens=lambda text: table.get(text, len(text)),
        check=check,
    )
    assert not result.complete and result.reason == "no_verified_partition"
    assert result.inputs == () and result.suggested_max_input_chars is None


def test_duplicate_texts_keep_distinct_occurrences_but_reuse_exact_measurements():
    source = "abcdefghijklmnopqrstuvwxyz"
    one_calls, two_calls = [], []

    def count_into(target):
        def count(text):
            target.append(text)
            return len(text)

        return count

    one = diagnose_overflow_inputs(
        [(5, source)], token_limit=10, count_tokens=count_into(one_calls), check=check
    )
    two = diagnose_overflow_inputs(
        [(5, source), (8, source)],
        token_limit=10,
        count_tokens=count_into(two_calls),
        check=check,
    )
    assert [item.input_index for item in two.inputs] == [5, 8]
    assert one.tokenize_requests == two.tokenize_requests
    assert one_calls == two_calls
    verified(two, {5: source, 8: source}, len, 10)


@pytest.mark.parametrize("source", ["", "🙂"])
def test_unpartitionable_input_never_returns_partial_advice(source):
    result = diagnose_overflow_inputs(
        [(0, source)], token_limit=1, count_tokens=lambda _: 2, check=check
    )
    assert not result.complete and result.reason == "unpartitionable_input"
    assert result.inputs == () and result.suggested_max_input_chars is None


def test_request_budget_is_hard_and_never_returns_partial_advice():
    calls = []

    def count(text):
        calls.append(text)
        return len(text)

    result = diagnose_overflow_inputs(
        [(0, "abcdefghijklmnopqrst")],
        token_limit=5,
        count_tokens=count,
        check=check,
        max_tokenize_requests=1,
    )
    assert not result.complete and result.reason == "tokenize_request_limit"
    assert result.inputs == () and result.suggested_max_input_chars is None
    assert result.tokenize_requests == len(calls) == 1


def test_fragment_budget_drops_all_advice_if_a_later_input_cannot_complete():
    result = diagnose_overflow_inputs(
        [(0, "abcdefghij"), (8, "ABCDEFGHIJ")],
        token_limit=5,
        count_tokens=len,
        check=check,
        max_fragments=2,
    )
    assert not result.complete and result.reason == "fragment_limit"
    assert result.inputs == () and result.suggested_max_input_chars is None


@pytest.mark.parametrize("error", [LLMCancelledError(), LLMTimeoutError("deadline")])
def test_cancellation_and_timeout_propagate_without_changed_error_identity(error):
    calls = []

    def count(text):
        calls.append(text)
        return len(text)

    def cancel():
        if calls:
            raise error

    with pytest.raises(type(error)) as raised:
        diagnose_overflow_inputs(
            [(0, "long source")], token_limit=2, count_tokens=count, check=cancel
        )
    assert raised.value is error and calls == ["long source"]


def test_exact_callback_exception_is_not_replaced_or_swallowed():
    error = ClientHTTPError(500)

    def count(_):
        raise error

    with pytest.raises(ClientHTTPError) as raised:
        diagnose_overflow_inputs(
            [(0, "text")], token_limit=2, count_tokens=count, check=check
        )
    assert raised.value is error


@pytest.mark.parametrize("bad", [True, -1, None, 1.5, "2"])
def test_invalid_native_counts_are_typed_failures(bad):
    with pytest.raises(LLMBackendError, match="overflow_invalid_token_count"):
        diagnose_overflow_inputs(
            [(0, "text")], token_limit=2, count_tokens=lambda _: bad, check=check
        )


@pytest.mark.parametrize(
    "field", ["token_limit", "max_tokenize_requests", "max_fragments"]
)
@pytest.mark.parametrize("bad", [True, 0, -1, None, 1.5])
def test_bounds_must_be_positive_integers(field, bad):
    values = dict(token_limit=2, count_tokens=len, check=check)
    values[field] = bad
    with pytest.raises(ValueError):
        diagnose_overflow_inputs([(0, "text")], **values)


@pytest.mark.parametrize(
    "bad", ["text", [(-1, "a")], [(True, "a")], [(0, 3)], [(0, "a"), (0, "b")]]
)
def test_inputs_require_distinct_occurrence_indices_and_finite_texts(bad):
    with pytest.raises(ValueError):
        diagnose_overflow_inputs(bad, token_limit=2, count_tokens=len, check=check)


def test_public_schema_is_immutable_and_does_not_contain_source_text_or_ids():
    result = diagnose_overflow_inputs(
        [(4, "abcdefghij")], token_limit=5, count_tokens=len, check=check
    )
    budget = snapshot()
    error = EmbeddingInputOverflowError(
        budget=budget, inputs=result.inputs, failed_input_indices=(4, 7), window_index=3
    )
    assert isinstance(error, LLMBackendError)
    assert error.budget is budget
    assert error.inputs == result.inputs and error.failed_input_indices == (4, 7)
    assert error.window_index == 3
    assert "abcdefghij" not in str(error) and "abcdefghij" not in repr(error.inputs)
    assert {item.name for item in fields(EmbeddingOverflowInput)} == {
        "input_index",
        "text_sha256",
        "char_count",
        "utf8_count",
        "token_count",
        "token_limit",
        "fragments",
    }
    with pytest.raises(FrozenInstanceError):
        budget.max_input_chars = 1
    with pytest.raises(FrozenInstanceError):
        error.inputs[0].fragments[0].char_end = 1
    sync = EmbeddingInputOverflowError(
        budget=budget, inputs=result.inputs, failed_input_indices=(4,)
    )
    assert sync.window_index is None


@pytest.mark.parametrize(
    "changes",
    [
        {"max_input_chars": 0},
        {"budget_revision": -1},
        {"effective_context_tokens": None},
        {"max_input_chars": True},
        {"initial_max_input_chars": 5},
        {"policy_id": ""},
    ],
)
def test_snapshot_rejects_incoherent_advice(changes):
    with pytest.raises(ValueError):
        snapshot(**changes)


def test_verified_spans_enforce_exact_fit_contiguous_coverage_and_strict_progress():
    result = diagnose_overflow_inputs(
        [(4, "abcdefghij")], token_limit=5, count_tokens=len, check=check
    )
    item = result.inputs[0]
    with pytest.raises(ValueError, match="contiguous"):
        replace(
            item,
            fragments=(replace(item.fragments[0], char_start=1), item.fragments[1]),
        )
    with pytest.raises(ValueError, match="exact token limit"):
        replace(
            item,
            fragments=(replace(item.fragments[0], token_count=6), item.fragments[1]),
        )
    with pytest.raises(ValueError, match="whole original"):
        replace(item, char_count=11)
    with pytest.raises(ValueError, match="proper partition"):
        replace(item, fragments=(EmbeddingVerifiedSpan(0, 10, 0, 10, 5),))


def test_overflow_error_cannot_misattribute_inputs_to_other_failed_occurrences():
    result = diagnose_overflow_inputs(
        [(4, "abcdefghij")], token_limit=5, count_tokens=len, check=check
    )
    with pytest.raises(ValueError, match="distinct failed"):
        EmbeddingInputOverflowError(
            budget=snapshot(), inputs=result.inputs, failed_input_indices=(7,)
        )
    with pytest.raises(ValueError, match="distinct occurrences"):
        EmbeddingInputOverflowError(
            budget=snapshot(), inputs=result.inputs, failed_input_indices=(4, 4)
        )


def test_private_receipts_hide_raw_source_and_release_old_owner_references():
    original = ClientHTTPError(500)
    owner = object()
    row = _FailedPostRow(owner, 4, "private source", "document", "private prefix")
    assert "private" not in repr(row)
    full = _FailedDirectPost(original, "alias", object(), [row])
    projected = replace(row, owner=None)
    single = _FailedOwnerPost(
        original,
        full.model,
        full.observation,
        [projected],
        deadline=12.5,
        window_index=3,
    )
    assert single.rows == (projected,) and single.rows[0].owner is None
    assert full.original is single.original is original
    assert single.deadline == 12.5 and single.window_index == 3
    assert "private" not in str(full) and "private" not in repr(single)
