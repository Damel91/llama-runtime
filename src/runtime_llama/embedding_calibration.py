"""Persistent, explicitly requested statistical embedding-input calibration.

The numerator is the number of Unicode code points in the unprefixed source.
The supplied tokenizer must count the *actual* prefixed embedding input,
including its special tokens. The resulting coefficient already accounts for
that overhead; callers must not subtract the prefix a second time.

A character budget is an empirical operating recommendation, never an exact
token limit or permission to truncate a server input.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
import hashlib
from importlib import resources
import json
import math
import os
from pathlib import Path
import tempfile
from threading import Event, Lock
from types import MappingProxyType
from typing import Callable, Mapping

from runtime_llama.errors import LLMBackendError, LLMTimeoutError

CALIBRATION_CATEGORIES = (
    "code",
    "documentation",
    "prose",
    "structured",
    "multilingual",
)
CATEGORY_SAFETY_FACTOR = 0.80
MIXED_SAFETY_FACTOR = 0.90
_CORPUS_SCHEMA = "runtime-llama-calibration-corpus-v2"
_PROFILE_SCHEMA = "runtime-llama-embedding-calibration-v1"
_MAX_CACHE_BYTES = 128 * 1024
_MAX_CORPUS_FILE_BYTES = 16 * 1024 * 1024
_MAX_TEXT_CHARACTERS = 32000
_FLIGHT_LOCK = Lock()
_FLIGHTS: dict[tuple[str, str], Event] = {}


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _freeze(value):
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class CalibrationStats:
    """Scalar measurements; ``chars_per_token`` is the weighted aggregate."""

    sample_count: int
    total_characters: int
    total_tokens: int
    chars_per_token: float
    arithmetic_mean_chars_per_token: float
    min_chars_per_token: float
    max_chars_per_token: float


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    """Immutable measurements bound to a verified tokenizer and prefix policy."""

    profile_id: str
    binding: Mapping[str, object]
    corpus_sha256: str
    statistics: Mapping[str, CalibrationStats]

    def __post_init__(self):
        object.__setattr__(self, "binding", _freeze(dict(self.binding)))
        object.__setattr__(self, "statistics", MappingProxyType(dict(self.statistics)))

    def stats_for(self, category: str = "mixed") -> CalibrationStats:
        if type(category) is not str or category not in self.statistics:
            raise ValueError("unknown embedding calibration category")
        return self.statistics[category]

    def safe_chars_per_token(self, category: str = "mixed") -> float:
        # The selected hard texts are measured with this model/task policy.
        # This fixed recommendation is never revised by inference failures.
        mean = self.stats_for(category).chars_per_token
        factor = MIXED_SAFETY_FACTOR if category == "mixed" else CATEGORY_SAFETY_FACTOR
        return mean * factor

    def max_input_chars(self, context_tokens: int, category: str = "mixed") -> int:
        if type(context_tokens) is not int or not 0 < context_tokens <= 2**31 - 1:
            raise ValueError("context_tokens must be a positive bounded integer")
        return math.floor(context_tokens * self.safe_chars_per_token(category))


@dataclass(frozen=True, slots=True)
class _CorpusRecord:
    category: str
    text: str


def _corpus_error():
    return LLMBackendError("llamacpp_embedding_calibration_invalid_corpus")


def _read_bounded(resource, limit):
    with resource.open("rb") as stream:
        result = stream.read(limit + 1)
    if len(result) > limit:
        raise _corpus_error()
    return result


def _load_manifest(check: Callable[[], None]):
    """Read the packaged identity of the original calibration observations."""
    check()
    directory = resources.files("runtime_llama").joinpath("calibration_corpus")
    try:
        manifest = json.loads(_read_bounded(directory.joinpath("manifest.json"), 32768))
        categories = manifest["categories"]
        if (
            manifest["schema"] != _CORPUS_SCHEMA
            or type(manifest["sample_count"]) is not int
            or manifest["sample_count"] != 2500
            or type(categories) is not dict
            or set(categories) != set(CALIBRATION_CATEGORIES)
            or manifest["corpus_sha256"]
            != hashlib.sha256(
                json.dumps(
                    categories,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        ):
            raise _corpus_error()
        for category in CALIBRATION_CATEGORIES:
            check()
            description = categories[category]
            if (
                type(description) is not dict
                or set(description)
                != {"file", "count", "total_characters", "bytes", "sha256"}
                or description["file"] != category + ".jsonl"
                or type(description["count"]) is not int
                or description["count"] != 500
                or type(description["bytes"]) is not int
                or not 0 < description["bytes"] <= _MAX_CORPUS_FILE_BYTES
                or type(description["total_characters"]) is not int
                or description["total_characters"] <= 0
            ):
                raise _corpus_error()
        return directory, manifest
    except LLMTimeoutError:
        raise
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as exc:
        raise _corpus_error() from exc


def _load_corpus(check: Callable[[], None]):
    """Verify every shipped byte and source record before remote work starts."""
    directory, manifest = _load_manifest(check)
    try:
        records, seen_ids, seen_texts = [], set(), set()
        for category in CALIBRATION_CATEGORIES:
            check()
            description = manifest["categories"][category]
            payload = _read_bounded(
                directory.joinpath(description["file"]), _MAX_CORPUS_FILE_BYTES
            )
            if (
                len(payload) != description["bytes"]
                or hashlib.sha256(payload).hexdigest() != description["sha256"]
            ):
                raise _corpus_error()
            lines = payload.splitlines()
            if len(lines) != description["count"]:
                raise _corpus_error()
            characters = 0
            for index, line in enumerate(lines):
                if index % 64 == 0:
                    check()
                record = json.loads(line)
                if (
                    type(record) is not dict
                    or set(record) != {"id", "category", "language", "text"}
                    or record["id"] != f"{category}-{index + 1:04d}"
                    or record["category"] != category
                    or type(record["language"]) is not str
                    or not record["language"]
                    or type(record["text"]) is not str
                    or not 0 < len(record["text"]) <= _MAX_TEXT_CHARACTERS
                    or record["id"] in seen_ids
                    or record["text"] in seen_texts
                ):
                    raise _corpus_error()
                seen_ids.add(record["id"])
                seen_texts.add(record["text"])
                characters += len(record["text"])
                records.append(_CorpusRecord(category, record["text"]))
            if characters != description["total_characters"]:
                raise _corpus_error()
        return manifest, tuple(records)
    except LLMTimeoutError:
        raise
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as exc:
        raise _corpus_error() from exc


def _binding_payload(binding: dict) -> dict:
    """Accept JSON builtins only, avoiding consumer callbacks during cache I/O."""

    def validate(value, depth=0):
        if depth > 8:
            raise ValueError("embedding calibration binding is too deeply nested")
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("embedding calibration binding keys must be strings")
            for item in value.values():
                validate(item, depth + 1)
        elif type(value) is list:
            for item in value:
                validate(item, depth + 1)
        elif value is not None and type(value) not in (str, int, float, bool):
            raise ValueError("embedding calibration binding must contain JSON builtins")

    if type(binding) is not dict or not binding:
        raise ValueError("embedding calibration binding must be a nonempty dictionary")
    validate(binding)
    encoded = _canonical(binding)
    if len(encoded) > 32768:
        raise ValueError("embedding calibration binding is too large")
    return json.loads(encoded)


def _make_stats(rows):
    count = len(rows)
    characters = sum(row[0] for row in rows)
    tokens = sum(row[1] for row in rows)
    ratios = [chars / token_count for chars, token_count in rows]
    return CalibrationStats(
        count,
        characters,
        tokens,
        characters / tokens,
        math.fsum(ratios) / count,
        min(ratios),
        max(ratios),
    )


def _decode_stats(value, expected_count, expected_characters):
    fields = set(CalibrationStats.__dataclass_fields__)
    if type(value) is not dict or set(value) != fields:
        raise ValueError("invalid calibration statistics")
    for field in ("sample_count", "total_characters", "total_tokens"):
        if type(value[field]) is not int or not 0 < value[field] <= 2**53:
            raise ValueError("invalid calibration integer")
    if (
        value["sample_count"] != expected_count
        or value["total_characters"] != expected_characters
        or value["total_tokens"] > expected_count * (2**31 - 1)
    ):
        raise ValueError("calibration statistics do not match the corpus")
    for field in fields - {"sample_count", "total_characters", "total_tokens"}:
        if type(value[field]) not in (int, float) or not (
            math.isfinite(value[field]) and 0 < value[field] <= _MAX_TEXT_CHARACTERS
        ):
            raise ValueError("invalid calibration ratio")
    minimum, maximum = value["min_chars_per_token"], value["max_chars_per_token"]
    if not (
        minimum <= value["arithmetic_mean_chars_per_token"] <= maximum
        and minimum <= value["chars_per_token"] <= maximum
        and math.isclose(
            value["chars_per_token"],
            value["total_characters"] / value["total_tokens"],
            rel_tol=1e-12,
        )
    ):
        raise ValueError("inconsistent calibration ratio")
    return CalibrationStats(**value)


def _load_cached(path, manifest, binding, profile_id):
    try:
        with path.open("rb") as stream:
            data = stream.read(_MAX_CACHE_BYTES + 1)
        if len(data) > _MAX_CACHE_BYTES:
            return None
        value = json.loads(data)
        if (
            type(value) is not dict
            or set(value)
            != {"schema", "profile_id", "binding", "corpus_sha256", "statistics"}
            or value["schema"] != _PROFILE_SCHEMA
            or value["profile_id"] != profile_id
            or _canonical(value["binding"]) != _canonical(binding)
            or value["corpus_sha256"] != manifest["corpus_sha256"]
            or type(value["statistics"]) is not dict
            or set(value["statistics"]) != {*CALIBRATION_CATEGORIES, "mixed"}
        ):
            return None
        statistics = {}
        for category in CALIBRATION_CATEGORIES:
            entry = manifest["categories"][category]
            statistics[category] = _decode_stats(
                value["statistics"][category], entry["count"], entry["total_characters"]
            )
        mixed = statistics["mixed"] = _decode_stats(
            value["statistics"]["mixed"],
            manifest["sample_count"],
            sum(item.total_characters for item in statistics.values()),
        )
        categories = [statistics[name] for name in CALIBRATION_CATEGORIES]
        if (
            mixed.total_tokens != sum(item.total_tokens for item in categories)
            or mixed.min_chars_per_token
            != min(item.min_chars_per_token for item in categories)
            or mixed.max_chars_per_token
            != max(item.max_chars_per_token for item in categories)
            or not math.isclose(
                mixed.arithmetic_mean_chars_per_token,
                math.fsum(
                    item.arithmetic_mean_chars_per_token * item.sample_count
                    for item in categories
                )
                / mixed.sample_count,
                rel_tol=1e-12,
            )
        ):
            return None
        return CalibrationProfile(
            profile_id, binding, manifest["corpus_sha256"], statistics
        )
    except (OSError, UnicodeError, TypeError, ValueError, KeyError, OverflowError):
        return None


def _save_cached(path, profile, binding, check):
    payload = _canonical(
        {
            "schema": _PROFILE_SCHEMA,
            "profile_id": profile.profile_id,
            "binding": binding,
            "corpus_sha256": profile.corpus_sha256,
            "statistics": {
                category: asdict(value)
                for category, value in profile.statistics.items()
            },
        }
    )
    check()
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".calibration-", dir=path.parent
        )
        try:
            os.chmod(temporary, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            check()
            os.replace(temporary, path)
            temporary = None
        finally:
            if descriptor is not None:
                os.close(descriptor)
    except LLMTimeoutError:
        raise
    except OSError as exc:
        raise LLMBackendError(
            "llamacpp_embedding_calibration_cache_write_failed"
        ) from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _preflight_cache(path, check):
    """Reject predictable cold-cache failures before remote tokenization.

    This tests real create/write/flush permission, not ``os.access``. It is
    deliberately separate from cache reads: complete readable profiles remain
    usable without write access. Filesystem state can still change afterwards,
    so the final validated atomic write remains authoritative.
    """
    check()
    temporary = None
    descriptor = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.is_dir():
            raise IsADirectoryError("calibration profile destination is a directory")
        descriptor, temporary = tempfile.mkstemp(
            prefix=".calibration-preflight-", dir=path.parent
        )
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            check()
            stream.write(b"runtime-llama-calibration-write-probe\n")
            stream.flush()
            os.fsync(stream.fileno())
        check()
    except LLMTimeoutError:
        raise
    except OSError as exc:
        raise LLMBackendError(
            "llamacpp_embedding_calibration_cache_write_failed"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _calibrate(records, tokenize, check, workers):
    stop = Event()

    def measure(record):
        if stop.is_set():
            return None
        check()
        count = tokenize(record.text)
        check()
        if type(count) is not int or not 0 < count <= 2**31 - 1:
            raise LLMBackendError("llamacpp_embedding_calibration_invalid_token_count")
        return record.category, len(record.text), count

    rows = {category: [] for category in CALIBRATION_CATEGORIES}
    iterator = iter(records)
    executor = ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="llama-calibration"
    )
    pending = set()
    try:
        for _ in range(workers):
            record = next(iterator, None)
            if record is not None:
                pending.add(executor.submit(measure, record))
        while pending:
            check()
            completed, pending = wait(
                pending, timeout=0.05, return_when=FIRST_COMPLETED
            )
            for future in completed:
                category, characters, count = future.result()
                rows[category].append((characters, count))
            for _ in completed:
                check()
                record = next(iterator, None)
                if record is not None:
                    pending.add(executor.submit(measure, record))
        check()
        statistics = {name: _make_stats(values) for name, values in rows.items()}
        statistics["mixed"] = _make_stats(
            [row for name in CALIBRATION_CATEGORIES for row in rows[name]]
        )
        return statistics
    finally:
        stop.set()
        for future in pending:
            future.cancel()
        # Tokenizer callbacks have finite transport timeouts. Running work is
        # drained before this owner exits; no background calibration is leaked.
        executor.shutdown(wait=True, cancel_futures=True)


def _profile_location(cache_dir, manifest, binding):
    profile_id = hashlib.sha256(
        _canonical(
            {
                "schema": _PROFILE_SCHEMA,
                "corpus_sha256": manifest["corpus_sha256"],
                "binding": binding,
            }
        )
    ).hexdigest()
    try:
        path = Path(cache_dir).expanduser().resolve() / (profile_id + ".json")
    except OSError as exc:
        raise LLMBackendError(
            "llamacpp_embedding_calibration_cache_write_failed"
        ) from exc
    return profile_id, path


def load_existing_calibration(
    cache_dir: str | os.PathLike,
    binding: dict,
    check: Callable[[], None],
) -> CalibrationProfile | None:
    """Read complete existing raw statistics; never tokenize or create files.

    The packaged manifest identifies the observations, including each corpus
    file hash. Loading those historical scalar statistics does not materialize
    or retokenize the 2,500 source records. Missing/invalid profiles return
    ``None``. The caller supplies a freshly verified tokenizer binding.
    """
    if not callable(check):
        raise TypeError("calibration requires a check callback")
    check()
    normalized_binding = _binding_payload(binding)
    _, manifest = _load_manifest(check)
    profile_id, path = _profile_location(cache_dir, manifest, normalized_binding)
    profile = _load_cached(path, manifest, normalized_binding, profile_id)
    check()
    return profile


def load_or_calibrate(
    cache_dir: str | os.PathLike,
    binding: dict,
    tokenize: Callable[[str], int],
    check: Callable[[], None],
    workers: int = 2,
    validate: Callable[[], None] | None = None,
) -> CalibrationProfile:
    """Load verified complete measurements, or tokenize all 2,500 selected texts.

    ``tokenize`` owns prefix application, exact native special-token policy,
    finite HTTP bounds and model verification. ``check`` enforces this caller's
    cancellation/deadline and must be safe to call from worker threads. There
    is an optional fresh identity ``validate`` after all counts complete and
    before any cache is published. Cached results rely on the caller's fresh
    binding observation and do not invoke that final validation callback.
    There are at most ``workers`` submitted requests, and at most one active cold
    calibration per cache path/key in this process. Waiting callers retain
    their own cancellation/deadline scopes. Separate processes may repeat a
    cold calibration; atomic replacement prevents partial cache publication.
    Cold persistence is checked before any remote tokenization; a complete
    readable cache does not require write permission.
    """
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("calibration workers must be between 1 and 32")
    if not callable(tokenize) or not callable(check):
        raise TypeError("calibration requires tokenize and check callbacks")
    if validate is not None and not callable(validate):
        raise TypeError("calibration validate must be a callable or None")
    normalized_binding = _binding_payload(binding)
    manifest, records = _load_corpus(check)
    profile_id, path = _profile_location(cache_dir, manifest, normalized_binding)
    flight_key = (str(path.parent), profile_id)
    while True:
        check()
        cached = _load_cached(path, manifest, normalized_binding, profile_id)
        if cached is not None:
            check()
            return cached
        with _FLIGHT_LOCK:
            event = _FLIGHTS.get(flight_key)
            leader = event is None
            if leader:
                event = _FLIGHTS[flight_key] = Event()
        if not leader:
            while not event.wait(0.05):
                check()
            continue
        try:
            # Another owner may have finished between the first read and this
            # flight claim, or a different process may have published a result.
            cached = _load_cached(path, manifest, normalized_binding, profile_id)
            if cached is not None:
                check()
                return cached
            _preflight_cache(path, check)
            statistics = _calibrate(records, tokenize, check, workers)
            profile = CalibrationProfile(
                profile_id, normalized_binding, manifest["corpus_sha256"], statistics
            )
            check()
            if validate is not None:
                validate()
                check()
            _save_cached(path, profile, normalized_binding, check)
            return profile
        finally:
            with _FLIGHT_LOCK:
                _FLIGHTS.pop(flight_key, None)
                event.set()
