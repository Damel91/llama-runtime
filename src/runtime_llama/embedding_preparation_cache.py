"""Bounded adapter-local reuse of completed, exact native tokenizations.

These values carry no context, batch, topology or inference permission. The
adapter verifies identity and tokenizer metadata before lookup and rebuilds
current admission limits when it uses a hit. Concurrent cold misses remain
independent requests in their original cancellation and deadline scopes.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import re
import sys
from threading import Lock

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class PreparationKey:
    """Full tokenization identity, independent of a physical replica alias."""

    text: str
    input_type: str
    policy: str
    content_sha256: str
    embedding_sha256: str
    tokenizer_build: str
    vocab_size: int | None = None

    def __post_init__(self):
        # Exact builtin values keep equality/hash operations inside the LRU
        # free from consumer-defined callbacks. Full text is compared after
        # hashing; a hash collision never establishes tokenization identity.
        for value in (
            self.text,
            self.input_type,
            self.policy,
            self.content_sha256,
            self.embedding_sha256,
            self.tokenizer_build,
        ):
            if type(value) is not str:
                raise ValueError("preparation cache key fields must be strings")
        if self.input_type not in ("document", "query", "raw"):
            raise ValueError("preparation cache input_type is invalid")
        if not self.policy or not self.tokenizer_build:
            raise ValueError("preparation cache requires policy and tokenizer build")
        if not _SHA256.fullmatch(self.content_sha256) or not _SHA256.fullmatch(
            self.embedding_sha256
        ):
            raise ValueError("preparation cache requires verified SHA-256 identities")
        if self.vocab_size is not None and (
            type(self.vocab_size) is not int or self.vocab_size <= 0
        ):
            raise ValueError("preparation cache vocabulary must be positive")


def _key(key):
    if type(key) is not PreparationKey:
        raise TypeError("preparation cache requires a PreparationKey")
    # Populate builtin string hashes before taking the bookkeeping lock.
    hash(key)
    return key


def _entry_bytes(key, token_ids):
    if type(token_ids) is not tuple or any(
        type(token) is not int or not 0 <= token <= 2**31 - 1 for token in token_ids
    ):
        raise ValueError("preparation cache token IDs must be an immutable valid tuple")
    # Charge shared strings and small integers independently for every entry.
    # The fixed allowance covers LRU/dictionary nodes and entry bookkeeping;
    # this is a conservative retained-object budget, not an RSS measurement.
    return (
        512
        + sys.getsizeof(key)
        + sum(
            sys.getsizeof(value)
            for value in (
                key.text,
                key.input_type,
                key.policy,
                key.content_sha256,
                key.embedding_sha256,
                key.tokenizer_build,
                key.vocab_size,
            )
        )
        + sys.getsizeof(token_ids)
        + sum(sys.getsizeof(token) for token in token_ids)
    )


class PreparationCache:
    """Completed-only LRU bounded by both entry count and retained bytes.

    A zero budget disables retention. There are no in-flight Futures, timers,
    callbacks or network operations, so closing cannot strand a waiting owner.
    """

    def __init__(self, max_entries: int, max_bytes: int):
        for name, value in (("max_entries", max_entries), ("max_bytes", max_bytes)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._lock = Lock()
        self._entries = OrderedDict()
        self._bytes = self._peak_entries = self._peak_bytes = 0
        self._hits = self._misses = self._puts = self._evictions = 0
        self._oversized_skips = 0
        self._closed = False

    def get(self, key: PreparationKey) -> tuple[int, ...] | None:
        key = _key(key)
        with self._lock:
            entry = None if self._closed else self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._hits += 1
            self._entries.move_to_end(key)
            return entry[0]

    def put(self, key: PreparationKey, token_ids: tuple[int, ...]) -> bool:
        key = _key(key)
        cost = _entry_bytes(key, token_ids)
        with self._lock:
            if self._closed:
                return False
            if not self.max_entries or cost > self.max_bytes:
                self._oversized_skips += 1
                return False
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
            while self._entries and (
                len(self._entries) >= self.max_entries
                or self._bytes + cost > self.max_bytes
            ):
                _, (_, removed) = self._entries.popitem(last=False)
                self._bytes -= removed
                self._evictions += 1
            self._entries[key] = (token_ids, cost)
            self._bytes += cost
            self._puts += 1
            self._peak_entries = max(self._peak_entries, len(self._entries))
            self._peak_bytes = max(self._peak_bytes, self._bytes)
            return True

    def close(self):
        with self._lock:
            self._closed = True
            self._entries.clear()
            self._bytes = 0

    def snapshot(self):
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._bytes,
                "peak_entries": self._peak_entries,
                "peak_bytes": self._peak_bytes,
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "puts": self._puts,
                "evictions": self._evictions,
                "oversized_skips": self._oversized_skips,
                "closed": self._closed,
            }
