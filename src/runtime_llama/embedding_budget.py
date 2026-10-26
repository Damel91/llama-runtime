"""Monotone absolute character recommendations learned from verified overflow.

This module performs local scalar-state I/O only. Its caller must finish exact
failed-batch diagnosis and fresh remote identity/context verification before an
update. No text, token array, endpoint, credential or fragment plan is retained.
Ordinary direct embedding does not consult this store.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from threading import Lock
import time
from typing import Callable
from weakref import WeakValueDictionary

from runtime_llama.embedding_calibration import (
    CATEGORY_SAFETY_FACTOR,
    MIXED_SAFETY_FACTOR,
    _binding_payload,
    _canonical,
)
from runtime_llama.errors import LLMBackendError, LLMTimeoutError

_STATE_SCHEMA = "runtime-llama-learned-character-budget-v1"
_POLICY = {
    "schema": _STATE_SCHEMA,
    "basis": "weighted-mean-source-characters-per-fully-prepared-token-selected-hard-2500",
    "character_unit": "unicode-code-points",
    "category_factor": CATEGORY_SAFETY_FACTOR,
    "mixed_factor": MIXED_SAFETY_FACTOR,
    "learned_limit": "fixed-initial-recommendation-no-inference-updates",
    "context_scope": "exact-effective-input-token-limit",
}
EMBEDDING_BUDGET_POLICY_ID = hashlib.sha256(_canonical(_POLICY)).hexdigest()
_GATES_LOCK = Lock()
_GATES = WeakValueDictionary()


@dataclass(frozen=True, slots=True)
class LearnedBudgetState:
    """One committed recommendation, global across categories within its scope."""

    context_scope_id: str
    budget_revision: int
    max_input_chars: int
    effective_context_tokens: int


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _binding_digest(binding):
    normalized = _binding_payload(binding)
    required = {
        "content_sha256",
        "embedding_sha256",
        "tokenizer_build",
        "vocab_size",
        "input_type",
        "policy",
        "prefix",
    }
    if not required <= set(normalized):
        raise ValueError("learned budget requires a verified tokenizer binding")
    # Only tokenizer-relevant fields participate. Physical aliases, category,
    # address and native scheduling topology never create a new learned space.
    normalized = {name: normalized[name] for name in required}
    for name in ("content_sha256", "embedding_sha256"):
        value = normalized[name]
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("learned budget requires verified SHA-256 identities")
    for name in ("tokenizer_build", "policy"):
        if type(normalized[name]) is not str or not normalized[name]:
            raise ValueError("learned budget requires tokenizer build and policy")
    if normalized["input_type"] not in ("document", "query", "raw"):
        raise ValueError("learned budget input_type is invalid")
    if type(normalized["prefix"]) is not str or (
        normalized["input_type"] == "raw" and normalized["prefix"]
    ):
        raise ValueError("learned budget prefix is invalid")
    if normalized["vocab_size"] is not None:
        _positive(normalized["vocab_size"], "vocab_size")
    return hashlib.sha256(_canonical(normalized)).hexdigest()


def budget_context_scope_id(binding: dict, effective_context_tokens: int) -> str:
    """Bind a recommendation to verified tokenizer semantics and exact limits.

    A different effective context deliberately creates a new scope. Existing
    lessons remain intact; they are never implicitly rescaled or carried over.
    """
    _positive(effective_context_tokens, "effective_context_tokens")
    return hashlib.sha256(
        _canonical(
            {
                "policy_id": EMBEDDING_BUDGET_POLICY_ID,
                "binding_sha256": _binding_digest(binding),
                "effective_context_tokens": effective_context_tokens,
            }
        )
    ).hexdigest()


def _location(cache_dir, binding, effective_context_tokens):
    scope = budget_context_scope_id(binding, effective_context_tokens)
    try:
        path = Path(cache_dir).expanduser().resolve() / ("budget-" + scope + ".json")
    except OSError as exc:
        raise LLMBackendError(
            "llamacpp_embedding_learned_budget_storage_failed"
        ) from exc
    return path, scope, _binding_digest(binding)


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate learned budget key")
        result[name] = value
    return result


def _read(path, scope, binding_digest, effective_context_tokens):
    payload = value = None
    try:
        with path.open("rb") as stream:
            payload = stream.read()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise LLMBackendError("llamacpp_embedding_learned_budget_read_failed") from exc
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object)
        if (
            type(value) is not dict
            or set(value)
            != {
                "schema",
                "policy_id",
                "binding_sha256",
                "context_scope_id",
                "budget_revision",
                "max_input_chars",
                "effective_context_tokens",
            }
            or value["schema"] != _STATE_SCHEMA
            or value["policy_id"] != EMBEDDING_BUDGET_POLICY_ID
            or value["binding_sha256"] != binding_digest
            or value["context_scope_id"] != scope
            or type(value["effective_context_tokens"]) is not int
            or value["effective_context_tokens"] != effective_context_tokens
        ):
            raise ValueError("learned budget scope mismatch")
        return LearnedBudgetState(
            scope,
            _positive(value["budget_revision"], "budget_revision"),
            _positive(value["max_input_chars"], "max_input_chars"),
            effective_context_tokens,
        )
    except (UnicodeError, TypeError, ValueError, RecursionError):
        # A corrupt learned ceiling must not silently disappear and restore a
        # larger initial recommendation. Recovery requires explicit repair.
        # Raise after the decoder's exception scope ends: JSONDecodeError.doc
        # and its traceback can otherwise retain the complete failed payload.
        pass
    finally:
        payload = value = None
    raise LLMBackendError("llamacpp_embedding_learned_budget_invalid") from None


def load_learned_budget(
    cache_dir: str | os.PathLike,
    binding: dict,
    effective_context_tokens: int,
    check: Callable[[], None],
) -> LearnedBudgetState | None:
    """Read one atomic snapshot without creating directories or lock files."""
    if not callable(check):
        raise TypeError("learned budget requires a check callback")
    check()
    path, scope, digest = _location(cache_dir, binding, effective_context_tokens)
    result = _read(path, scope, digest, effective_context_tokens)
    check()
    return result


def _try_file_lock(descriptor):
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(descriptor):
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextmanager
def _locked(path, check):
    """Serialize writers using a stable lock inode, never the replaced JSON."""
    with _GATES_LOCK:
        gate = _GATES.get(str(path))
        if gate is None:
            gate = Lock()
            _GATES[str(path)] = gate
    acquired = locked = False
    descriptor = None
    try:
        while not acquired:
            check()
            acquired = gate.acquire(timeout=0.05)
        check()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        os.chmod(path, 0o600)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("learned budget lock is not a regular file")
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\x00")
        while not locked:
            check()
            try:
                _try_file_lock(descriptor)
                locked = True
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                time.sleep(0.02)
        check()
        yield
    except LLMTimeoutError:
        raise
    except OSError as exc:
        raise LLMBackendError(
            "llamacpp_embedding_learned_budget_storage_failed"
        ) from exc
    finally:
        try:
            if locked:
                _unlock_file(descriptor)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if acquired:
                gate.release()
        # Keep the small empty lock file: deleting it could let another process
        # lock a different inode while an older writer still owns the old one.


def _save(path, state, binding_digest, check):
    payload = _canonical(
        {
            "schema": _STATE_SCHEMA,
            "policy_id": EMBEDDING_BUDGET_POLICY_ID,
            "binding_sha256": binding_digest,
            **asdict(state),
        }
    )
    temporary = None
    descriptor = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=".budget-tmp-", dir=path.parent)
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        check()
        # This replacement is the publication boundary. A cancellation after
        # it cannot undo a complete, independently valid monotone reduction.
        os.replace(temporary, path)
        temporary = None
        if os.name != "nt":
            directory = os.open(
                path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except LLMTimeoutError:
        raise
    except OSError as exc:
        raise LLMBackendError(
            "llamacpp_embedding_learned_budget_storage_failed"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def lower_learned_budget(
    cache_dir: str | os.PathLike,
    binding: dict,
    effective_context_tokens: int,
    max_input_chars: int,
    check: Callable[[], None],
) -> LearnedBudgetState:
    """Commit ``min(current, candidate)`` without applying any safety factor.

    The caller has already verified a complete exact lossless partition and
    current remote identity/limits. The local critical section contains no
    HTTP. Revisions increase only when the absolute character ceiling decreases.
    """
    if not callable(check):
        raise TypeError("learned budget requires a check callback")
    _positive(max_input_chars, "max_input_chars")
    check()
    path, scope, digest = _location(cache_dir, binding, effective_context_tokens)
    with _locked(path.with_suffix(".lock"), check):
        current = _read(path, scope, digest, effective_context_tokens)
        if current is not None and current.max_input_chars <= max_input_chars:
            return current
        revision = 1 if current is None else current.budget_revision + 1
        _positive(revision, "budget_revision")
        state = LearnedBudgetState(
            scope, revision, max_input_chars, effective_context_tokens
        )
        check()
        _save(path, state, digest, check)
        return state
