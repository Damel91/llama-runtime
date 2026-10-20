"""Immutable, adapter-issued exact embedding preparations."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import json
from typing import Literal

EmbeddingInputType = Literal["document", "query", "raw"]


@dataclass(frozen=True)
class PreparedEmbeddingInput:
    """An exact native count and limit snapshot, not an inference permit.

    ``text`` already includes its task prefix. Instances returned by an adapter
    can be submitted to that same adapter; altered or foreign records cannot.
    """

    text: str
    input_type: EmbeddingInputType
    token_count: int
    context_token_limit: int | None
    batch_token_limit: int | None
    max_input_tokens: int | None
    identity_verified: bool
    content_sha256: str | None
    embedding_sha256: str | None
    _tokenizer_build: str | None = field(default=None, repr=False)
    _policy: str = field(default="", repr=False)
    _seal: bytes = field(default=b"", repr=False, compare=False)
    _token_ids: tuple[int, ...] | None = field(default=None, repr=False)

    @staticmethod
    def _fits(count: int, limit: int | None) -> bool | None:
        return None if limit is None else count <= limit

    @property
    def fits_context(self) -> bool | None:
        return self._fits(self.token_count, self.context_token_limit)

    @property
    def fits_request(self) -> bool | None:
        return self._fits(self.token_count, self.batch_token_limit)

    @property
    def fits_input(self) -> bool | None:
        return self._fits(self.token_count, self.max_input_tokens)


def _message(record: PreparedEmbeddingInput) -> bytes:
    return json.dumps(
        [
            "runtime-llama-preparation-v2",
            record.text,
            record.input_type,
            record.token_count,
            record.context_token_limit,
            record.batch_token_limit,
            record.max_input_tokens,
            record.identity_verified,
            record.content_sha256,
            record.embedding_sha256,
            record._tokenizer_build,
            record._policy,
            record._token_ids,
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")


def issue_preparation(key: bytes, **fields) -> PreparedEmbeddingInput:
    record = PreparedEmbeddingInput(**fields)
    object.__setattr__(
        record, "_seal", hmac.digest(key, _message(record), hashlib.sha256)
    )
    return record


def valid_preparation(key: bytes, record: object) -> bool:
    if type(record) is not PreparedEmbeddingInput:
        return False
    try:
        if record._token_ids is not None and (
            not isinstance(record._token_ids, tuple)
            or len(record._token_ids) != record.token_count
            or any(
                not isinstance(token, int)
                or isinstance(token, bool)
                or not 0 <= token <= 2**31 - 1
                for token in record._token_ids
            )
        ):
            return False
        expected = hmac.digest(key, _message(record), hashlib.sha256)
        return isinstance(record._seal, bytes) and hmac.compare_digest(
            expected, record._seal
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
