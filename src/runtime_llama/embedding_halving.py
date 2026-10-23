"""Tokenizer-free recovery advice and indexed partial embedding results.

A native singleton rejection does not prove token overflow. These source halves
are lossless retry instructions, never exact token-admission statements. The
consumer owns fragment identities and any subsequent request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from types import MappingProxyType
from typing import Mapping

from runtime_llama.client import ClientHTTPError
from runtime_llama.errors import LLMBackendError


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


@dataclass(frozen=True, slots=True)
class EmbeddingSourceSpan:
    """Relative Unicode/UTF-8 source interval, without a token-count promise."""

    char_start: int
    char_end: int
    utf8_start: int
    utf8_end: int

    def __post_init__(self):
        for name in ("char_start", "char_end", "utf8_start", "utf8_end"):
            _integer(getattr(self, name), name)
        if self.char_end <= self.char_start or self.utf8_end <= self.utf8_start:
            raise ValueError("source spans must be nonempty")
        if self.utf8_end - self.utf8_start < self.char_end - self.char_start:
            raise ValueError("UTF-8 spans cannot be shorter than their code points")


@dataclass(frozen=True, slots=True)
class EmbeddingHalvingInput:
    """One rejected source occurrence and its two proper midpoint halves."""

    input_index: int
    text_sha256: str
    char_count: int
    utf8_count: int
    halves: tuple[EmbeddingSourceSpan, EmbeddingSourceSpan]

    def __post_init__(self):
        _integer(self.input_index, "input_index")
        _integer(self.char_count, "char_count", 2)
        _integer(self.utf8_count, "utf8_count", self.char_count)
        if (
            type(self.text_sha256) is not str
            or len(self.text_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.text_sha256)
        ):
            raise ValueError("text_sha256 must be a lowercase SHA-256 digest")
        halves = tuple(self.halves)
        object.__setattr__(self, "halves", halves)
        if len(halves) != 2 or any(
            type(span) is not EmbeddingSourceSpan for span in halves
        ):
            raise ValueError("halving advice requires two source spans")
        left, right = halves
        if (
            left.char_start != 0
            or left.utf8_start != 0
            or left.char_end != self.char_count // 2
            or right.char_start != left.char_end
            or right.utf8_start != left.utf8_end
            or right.char_end != self.char_count
            or right.utf8_end != self.utf8_count
        ):
            raise ValueError("halves must cover the original source at its midpoint")


@dataclass(frozen=True, slots=True)
class EmbeddingCompletedInput:
    """A successful original occurrence; consumers must not embed it again."""

    input_index: int
    embedding: tuple[float, ...]

    def __post_init__(self):
        _integer(self.input_index, "input_index")
        embedding = tuple(self.embedding)
        if not embedding or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in embedding
        ):
            raise ValueError("embedding must be a nonempty finite vector")
        object.__setattr__(self, "embedding", embedding)


class EmbeddingChunkSplitRequiredError(LLMBackendError):
    """Completed rows plus local halving advice, with no changed budget."""

    def __init__(
        self,
        *,
        input_count,
        completed,
        inputs,
        representation_id,
        input_type,
        window_index=None,
    ):
        _integer(input_count, "input_count", 1)
        completed, inputs = tuple(completed), tuple(inputs)
        if any(type(item) is not EmbeddingCompletedInput for item in completed):
            raise ValueError("completed must contain indexed embedding results")
        if not inputs or any(
            type(item) is not EmbeddingHalvingInput for item in inputs
        ):
            raise ValueError("inputs must contain isolated halving instructions")
        indices = [item.input_index for item in (*completed, *inputs)]
        if sorted(indices) != list(range(input_count)):
            raise ValueError(
                "completed and rejected inputs must partition the full window"
            )
        if input_type not in ("document", "query", "raw"):
            raise ValueError("input_type must be document, query or raw")
        if type(representation_id) is not str or not representation_id:
            raise ValueError("representation_id must be defined")
        if window_index is not None:
            _integer(window_index, "window_index")
        super().__init__("llamacpp_embedding_chunk_split_required")
        self.input_count = input_count
        self.completed = completed
        self.inputs = inputs
        self.failed_input_indices = tuple(item.input_index for item in inputs)
        self.representation_id = representation_id
        self.input_type = input_type
        self.window_index = window_index
        self.original_status_code = 500


@dataclass(frozen=True, slots=True)
class _RejectedEmbedding:
    original: Exception = field(repr=False, compare=False)
    input_type: str
    text_sha256: str
    char_count: int
    utf8_count: int
    halves: tuple[EmbeddingSourceSpan, ...]

    def for_input(self, index):
        if self.char_count < 2:
            raise self.original
        return EmbeddingHalvingInput(
            index, self.text_sha256, self.char_count, self.utf8_count, self.halves
        )


def _rejected_embedding(source_text, input_type, original):
    """Compute bounded source metadata on the HTTP worker; retain no text."""
    try:
        encoded = source_text.encode("utf-8")
    except UnicodeEncodeError:
        # No valid UTF-8 interval can be offered for an unpaired surrogate.
        # Keep the inference failure rather than inventing source offsets.
        raise original
    midpoint = len(source_text) // 2
    if len(source_text) < 2:
        halves = ()
    else:
        byte_midpoint = len(source_text[:midpoint].encode("utf-8"))
        halves = (
            EmbeddingSourceSpan(0, midpoint, 0, byte_midpoint),
            EmbeddingSourceSpan(
                midpoint, len(source_text), byte_midpoint, len(encoded)
            ),
        )
    return _RejectedEmbedding(
        ClientHTTPError(
            original.status_code,
            error_code=original.error_code,
            error_type=original.error_type,
        ),
        input_type,
        hashlib.sha256(encoded).hexdigest(),
        len(source_text),
        len(encoded),
        halves,
    )


@dataclass(frozen=True, slots=True)
class _HalvingResult:
    """A drained internal window, delivered in stream order under result credit."""

    outputs: tuple[list[float] | None, ...]
    rejections: Mapping[int, _RejectedEmbedding]
    window_index: int | None = None
    deadline: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "rejections", MappingProxyType(dict(self.rejections)))


def _halving_error(result, representation_id, input_type):
    error = EmbeddingChunkSplitRequiredError(
        input_count=len(result.outputs),
        completed=tuple(
            EmbeddingCompletedInput(index, vector)
            for index, vector in enumerate(result.outputs)
            if vector is not None
        ),
        inputs=tuple(
            marker.for_input(index)
            for index, marker in sorted(result.rejections.items())
        ),
        representation_id=representation_id,
        input_type=input_type,
        window_index=result.window_index,
    )
    return error
