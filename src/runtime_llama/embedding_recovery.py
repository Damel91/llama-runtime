"""Bounded exact diagnostics for an already drained direct embedding POST.

No function in this module sends or retries an embedding request. Exact-count
callbacks receive unprefixed source text; their owner applies the verified task
prefix and native special-token policy. Only fully measured, lossless partitions
can become public overflow advice. A character ceiling remains a recommendation
for future input preparation, not token admission authority.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import hashlib
from typing import Any

from runtime_llama.errors import LLMBackendError


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _identifier(value, name, *, optional=False):
    if optional and value is None:
        return
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class EmbeddingBudgetSnapshot:
    """A versioned character recommendation bound to one verified context."""

    policy_id: str
    representation_id: str
    context_scope_id: str
    input_type: str
    category: str
    calibration_id: str | None
    max_input_chars: int
    initial_max_input_chars: int | None
    budget_revision: int
    effective_context_tokens: int
    source: str

    def __post_init__(self):
        for name in (
            "policy_id",
            "representation_id",
            "context_scope_id",
            "category",
            "source",
        ):
            _identifier(getattr(self, name), name)
        _identifier(self.calibration_id, "calibration_id", optional=True)
        if self.input_type not in ("document", "query", "raw"):
            raise ValueError("input_type must be document, query or raw")
        _integer(self.max_input_chars, "max_input_chars", 1)
        if self.initial_max_input_chars is not None:
            _integer(self.initial_max_input_chars, "initial_max_input_chars", 1)
            if self.max_input_chars > self.initial_max_input_chars:
                raise ValueError("max_input_chars exceeds the initial recommendation")
        _integer(self.budget_revision, "budget_revision")
        _integer(self.effective_context_tokens, "effective_context_tokens", 1)


@dataclass(frozen=True, slots=True)
class EmbeddingVerifiedSpan:
    """Relative source boundaries and the exact prefixed embedding token count."""

    char_start: int
    char_end: int
    utf8_start: int
    utf8_end: int
    token_count: int

    def __post_init__(self):
        for name in ("char_start", "char_end", "utf8_start", "utf8_end", "token_count"):
            _integer(getattr(self, name), name)
        if self.char_start >= self.char_end or self.utf8_start >= self.utf8_end:
            raise ValueError("verified spans must be nonempty")
        if self.utf8_end - self.utf8_start < self.char_end - self.char_start:
            raise ValueError("UTF-8 spans cannot be shorter than their code points")


@dataclass(frozen=True, slots=True)
class EmbeddingOverflowInput:
    """One oversized occurrence and its complete, independently verified split."""

    input_index: int
    text_sha256: str
    char_count: int
    utf8_count: int
    token_count: int
    token_limit: int
    fragments: tuple[EmbeddingVerifiedSpan, ...]

    def __post_init__(self):
        _integer(self.input_index, "input_index")
        if (
            type(self.text_sha256) is not str
            or len(self.text_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.text_sha256)
        ):
            raise ValueError("text_sha256 must be a lowercase SHA-256 digest")
        _integer(self.char_count, "char_count", 1)
        _integer(self.utf8_count, "utf8_count", 1)
        _integer(self.token_limit, "token_limit", 1)
        _integer(self.token_count, "token_count", self.token_limit + 1)
        fragments = tuple(self.fragments)
        object.__setattr__(self, "fragments", fragments)
        if len(fragments) < 2:
            raise ValueError("overflow recovery requires a proper partition")
        chars = utf8 = 0
        for fragment in fragments:
            if type(fragment) is not EmbeddingVerifiedSpan:
                raise ValueError("fragments must contain verified spans")
            if fragment.char_start != chars or fragment.utf8_start != utf8:
                raise ValueError("fragment coverage must be contiguous and ordered")
            if fragment.token_count > self.token_limit:
                raise ValueError("fragment exceeds the exact token limit")
            if fragment.char_end - fragment.char_start >= self.char_count:
                raise ValueError("fragmentation must make strict progress")
            chars, utf8 = fragment.char_end, fragment.utf8_end
        if chars != self.char_count or utf8 != self.utf8_count:
            raise ValueError("fragments must cover the whole original input")


class EmbeddingInputOverflowError(LLMBackendError):
    """Verified advice only; the consumer owns splitting and any explicit retry."""

    def __init__(
        self,
        *,
        budget: EmbeddingBudgetSnapshot,
        inputs: Sequence[EmbeddingOverflowInput],
        failed_input_indices: Sequence[int],
        window_index: int | None = None,
    ):
        if type(budget) is not EmbeddingBudgetSnapshot:
            raise ValueError("budget must be an embedding budget snapshot")
        inputs, failed_input_indices = tuple(inputs), tuple(failed_input_indices)
        if not inputs or any(
            type(item) is not EmbeddingOverflowInput for item in inputs
        ):
            raise ValueError("inputs must contain verified overflow partitions")
        for index in failed_input_indices:
            _integer(index, "failed_input_indices")
        if not failed_input_indices or len(set(failed_input_indices)) != len(
            failed_input_indices
        ):
            raise ValueError("failed_input_indices must identify distinct occurrences")
        indices = [item.input_index for item in inputs]
        if len(set(indices)) != len(indices) or not set(indices).issubset(
            failed_input_indices
        ):
            raise ValueError("overflow inputs must be distinct failed occurrences")
        if window_index is not None:
            _integer(window_index, "window_index")
        super().__init__("llamacpp_embedding_input_overflow")
        self.budget = budget
        self.inputs = inputs
        self.failed_input_indices = failed_input_indices
        self.window_index = window_index


@dataclass(frozen=True)
class _FailedPostRow:
    """One actual native POST row; never serialize its source or call owner."""

    owner: Any = field(repr=False)
    input_index: int
    text: str = field(repr=False)
    input_type: str
    prefix: str = field(repr=False)


class _FailedPostReceipt(Exception):
    def __init__(self, original, model, observation, rows):
        super().__init__("llamacpp_embedding_failed_direct_post")
        self.original = original
        self.model = model
        self.observation = observation
        self.rows = tuple(rows)


class _FailedDirectPost(_FailedPostReceipt):
    """Private evidence for engine projection, not an external API error."""


class _FailedOwnerPost(_FailedPostReceipt):
    """One drained owner's actual failed rows, without retaining old call owners."""

    def __init__(
        self, original, model, observation, rows, *, deadline=None, window_index=None
    ):
        super().__init__(original, model, observation, rows)
        self.deadline = deadline
        self.window_index = window_index


@dataclass(frozen=True, slots=True)
class _OverflowDiagnosis:
    inputs: tuple[EmbeddingOverflowInput, ...]
    suggested_max_input_chars: int | None
    tokenize_requests: int
    complete: bool
    reason: str


class _PartitionFailure(Exception):
    pass


def diagnose_overflow_inputs(
    inputs: Sequence[tuple[int, str]],
    *,
    token_limit: int,
    count_tokens: Callable[[str], int],
    check: Callable[[], None],
    max_tokenize_requests: int = 512,
    max_fragments: int = 256,
) -> _OverflowDiagnosis:
    """Measure only failed source inputs and construct bounded exact partitions.

    ``count_tokens`` counts the source with the actual prefix/special policy.
    Its integer result is authority; proportional and binary cuts only select
    probes. Token counts need not be monotonic and no globally maximal cut is
    claimed. Every returned fragment was measured and fits independently.

    At most ``max_tokenize_requests`` count callbacks run. Equal source strings
    and equal slices reuse a count within this one verified diagnostic scope.
    ``max_fragments`` bounds the complete public partition across occurrences.
    Incomplete diagnostics contain no partial advice or candidate ceiling.
    Cancellation, timeout and callback exceptions propagate unchanged.
    """
    _integer(token_limit, "token_limit", 1)
    _integer(max_tokenize_requests, "max_tokenize_requests", 1)
    _integer(max_fragments, "max_fragments", 1)
    if not callable(count_tokens) or not callable(check):
        raise ValueError("diagnostics require exact count and cancellation callbacks")
    if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
        raise ValueError("inputs must be a finite sequence of indexed source strings")
    sources, seen_indices, canonical, originals = [], set(), {}, []
    for item in inputs:
        check()
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("each input must contain an index and source string")
        index, text = item
        _integer(index, "input_index")
        if index in seen_indices or type(text) is not str:
            raise ValueError("inputs must identify distinct indexed source strings")
        seen_indices.add(index)
        source_id = canonical.get(text)
        if source_id is None:
            source_id = len(originals)
            canonical[text] = source_id
            originals.append(text)
        sources.append((index, source_id, text))
    requests = fragment_count = 0
    counts = {}

    def count(source_id, start, end):
        nonlocal requests
        check()
        key = source_id, start, end
        if key not in counts:
            if requests >= max_tokenize_requests:
                raise _PartitionFailure("tokenize_request_limit")
            requests += 1
            value = count_tokens(originals[source_id][start:end])
            check()
            if type(value) is not int or value < 0:
                raise LLMBackendError("llamacpp_embedding_overflow_invalid_token_count")
            counts[key] = value
        return counts[key]

    def partition(index, source_id, text, full_count):
        nonlocal fragment_count
        end = len(text)
        start = utf8_start = 0
        remaining_count = full_count
        fragments = []
        while start < end:
            check()
            remaining = end - start
            if remaining_count <= token_limit:
                width, measured = remaining, remaining_count
            else:
                if remaining <= 1:
                    raise _PartitionFailure("unpartitionable_input")
                # An observed ratio chooses the first probe, not admission.
                width = min(
                    remaining - 1,
                    max(1, remaining * token_limit // remaining_count),
                )
                rejected = None
                while True:
                    measured = count(source_id, start, start + width)
                    if measured <= token_limit:
                        break
                    rejected = width
                    if width == 1:
                        # Non-monotonic tokenizers may admit a different cut.
                        # No such unmeasured partition is claimed impossible.
                        raise _PartitionFailure("no_verified_partition")
                    width = max(1, width // 2)
                if rejected is not None:
                    best, best_count = width, measured
                    low, high = width + 1, rejected - 1
                    while low <= high:
                        candidate = (low + high) // 2
                        candidate_count = count(source_id, start, start + candidate)
                        if candidate_count <= token_limit:
                            best, best_count = candidate, candidate_count
                            low = candidate + 1
                        else:
                            high = candidate - 1
                    width, measured = best, best_count
            if fragment_count >= max_fragments:
                raise _PartitionFailure("fragment_limit")
            utf8_end = utf8_start + len(text[start : start + width].encode("utf-8"))
            fragments.append(
                EmbeddingVerifiedSpan(
                    start, start + width, utf8_start, utf8_end, measured
                )
            )
            fragment_count += 1
            start, utf8_start = start + width, utf8_end
            if start < end:
                remaining_count = count(source_id, start, end)
        if not fragments or len(fragments) < 2:
            raise _PartitionFailure("unpartitionable_input")
        encoded = text.encode("utf-8")
        return EmbeddingOverflowInput(
            index,
            hashlib.sha256(encoded).hexdigest(),
            len(text),
            len(encoded),
            full_count,
            token_limit,
            tuple(fragments),
        )

    try:
        measured = [
            (index, source_id, text, count(source_id, 0, len(text)))
            for index, source_id, text in sources
        ]
        overflow = []
        for index, source_id, text, full_count in measured:
            check()
            if full_count > token_limit:
                overflow.append(partition(index, source_id, text, full_count))
        check()
        candidate = min(
            (
                max(span.char_end - span.char_start for span in item.fragments)
                for item in overflow
            ),
            default=None,
        )
        return _OverflowDiagnosis(
            tuple(overflow),
            candidate,
            requests,
            True,
            "verified" if overflow else "no_overflow",
        )
    except _PartitionFailure as failure:
        check()
        return _OverflowDiagnosis((), None, requests, False, str(failure))
