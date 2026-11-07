"""Apply the relay's OpenCode V2 export to a local user configuration."""

from contextlib import suppress
import errno
import json
import os
from pathlib import Path
import re
import stat
from tempfile import mkstemp
from threading import Lock

from runtime_llama.relay.protocol import RelayError


_LOCK = Lock()
_STRING = r'"(?:[^"\\]|\\.)*"'
_COMMENTS = re.compile(_STRING + r"|//[^\r\n]*|/\*[\s\S]*?\*/")
_TRAILING_COMMAS = re.compile(_STRING + r"|,(?=[ \t\r\n]*[}\]])")
_SPACE = " \t\r\n"
_WHITESPACE = re.compile(r"[ \t\r\n]*")
_DECODER = json.JSONDecoder()


def _conflict() -> RelayError:
    return RelayError(409, "relay_opencode_config_conflict")


def _config_directory() -> Path:
    override = os.environ.get("OPENCODE_CONFIG_DIR")
    if override:
        directory = Path(override)
        if not directory.is_absolute():
            raise _conflict()
    else:
        xdg = Path(os.environ.get("XDG_CONFIG_HOME", ""))
        directory = (xdg if xdg.is_absolute() else Path.home() / ".config") / "opencode"
    return Path(os.path.abspath(directory))


def _snapshot(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _identity(info: os.stat_result | None) -> tuple | None:
    if info is None:
        return None
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _target(directory: Path) -> tuple[Path, os.stat_result | None]:
    for name in ("opencode.jsonc", "opencode.json"):
        path = directory / name
        info = _snapshot(path)
        if info is not None:
            if not stat.S_ISREG(info.st_mode):
                raise _conflict()
            return path, info
    return directory / "opencode.json", None


def _read(path: Path, expected: os.stat_result) -> bytes:
    flags = os.O_RDONLY
    for name in ("O_NOFOLLOW", "O_NONBLOCK", "O_BINARY"):
        flags |= getattr(os, name, 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno in (errno.ENOENT, errno.ELOOP):
            raise _conflict() from None
        raise
    with os.fdopen(descriptor, "rb") as stream:
        if _identity(os.fstat(stream.fileno())) != _identity(expected):
            raise _conflict()
        data = stream.read()
        if _identity(os.fstat(stream.fileno())) != _identity(expected):
            raise _conflict()
    return data


def _mask_jsonc(source: str) -> str:
    """Make a parseable JSON view without changing source offsets."""

    def comment(match: re.Match) -> str:
        value = match[0]
        if value.startswith('"'):
            return value
        return re.sub(r"[^\r\n]", " ", value)

    masked = _COMMENTS.sub(comment, source)

    def comma(match: re.Match) -> str:
        if match[0] != ",":
            return match[0]
        previous = match.start() - 1
        while previous >= 0 and masked[previous] in _SPACE:
            previous -= 1
        return " " if previous >= 0 and masked[previous] not in "[{,:" else ","

    if masked.startswith("\ufeff"):
        masked = " " + masked[1:]
    return _TRAILING_COMMAS.sub(comma, masked)


def _unique_object(pairs: list[tuple]) -> dict:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("duplicate JSON property")
    return value


def _invalid_constant(_value: str):
    raise ValueError("invalid JSON constant")


def _skip_space(masked: str, offset: int) -> int:
    return _WHITESPACE.match(masked, offset).end()


def _member_spans(masked: str, opening: int) -> dict[str, tuple[int, int]]:
    """Find the value spans of a previously validated JSON object."""
    spans = {}
    offset = _skip_space(masked, opening + 1)
    while masked[offset] != "}":
        name, key_end = _DECODER.raw_decode(masked, offset)
        start = _skip_space(masked, _skip_space(masked, key_end) + 1)
        _, end = _DECODER.raw_decode(masked, start)
        spans[name] = start, end
        offset = _skip_space(masked, end)
        if masked[offset] == ",":
            offset = _skip_space(masked, offset + 1)
    return spans


def _indent(source: str, offset: int) -> str:
    line = source[source.rfind("\n", 0, offset) + 1 : offset]
    return line[: len(line) - len(line.lstrip(" \t"))]


def _dump(value: object, indent: str = "") -> str:
    return json.dumps(value, indent=2, allow_nan=False).replace("\n", "\n" + indent)


def _object_edits(
    source: str, opening: int, spans: dict, previous: dict, updates: dict
) -> list[tuple[int, int, str]]:
    base_indent = _indent(source, opening)
    insert_indent = base_indent + "  "
    inserts = []
    edits = []
    for name, value in updates.items():
        if name not in spans:
            inserts.append(
                f"{insert_indent}{json.dumps(name)}: {_dump(value, insert_indent)}"
            )
        elif previous[name] != value:
            start, end = spans[name]
            edits.append((start, end, _dump(value, _indent(source, start))))
    if inserts:
        ending = "," if spans else "\n" + base_indent
        edits.append((opening + 1, opening + 1, "\n" + ",\n".join(inserts) + ending))
    return edits


def _merge(original: bytes | None, provider: dict, model: str) -> bytes:
    if original is None:
        return (
            _dump({"providers": {"llama-relay": provider}, "model": model}) + "\n"
        ).encode()

    try:
        source = original.decode("utf-8")
        masked = _mask_jsonc(source)
        current = json.loads(
            masked, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
        )
        if not isinstance(current, dict):
            raise ValueError("expected JSON object")
        opening = _skip_space(masked, 0)
        spans = _member_spans(masked, opening)
        updates = {"model": model}
        edits = []
        if "providers" in current:
            providers = current["providers"]
            if not isinstance(providers, dict):
                raise ValueError("expected providers object")
            providers_start, _ = spans["providers"]
            edits = _object_edits(
                source,
                providers_start,
                _member_spans(masked, providers_start),
                providers,
                {"llama-relay": provider},
            )
        else:
            updates["providers"] = {"llama-relay": provider}
        edits.extend(_object_edits(source, opening, spans, current, updates))
        for start, end, replacement in sorted(edits, reverse=True):
            source = source[:start] + replacement + source[end:]
        return source.encode("utf-8")
    except (ValueError, RecursionError):
        raise RelayError(422, "relay_opencode_config_invalid") from None


def _private_file(directory: Path, prefix: str, suffix: str, content: bytes) -> Path:
    descriptor, name = mkstemp(dir=directory, prefix=prefix, suffix=suffix)
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with suppress(OSError):
            path.unlink()
        raise
    return path


def _directory_identity(directory: Path) -> tuple[int, int] | None:
    if directory.resolve() != directory:
        raise _conflict()
    info = _snapshot(directory)
    if info is None:
        return None
    if not stat.S_ISDIR(info.st_mode):
        raise _conflict()
    return info.st_dev, info.st_ino


def _apply(directory: Path, provider: dict, model: str) -> dict:
    expected_directory = _directory_identity(directory)
    path, expected = _target(directory)
    original = _read(path, expected) if expected is not None else None
    replacement = _merge(original, provider, model)
    if replacement == original:
        return {"filename": path.name, "changed": False}

    if expected_directory is None:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory_identity = _directory_identity(directory)
    if directory_identity is None or (
        expected_directory is not None and directory_identity != expected_directory
    ):
        raise _conflict()
    candidate = _private_file(directory, ".opencode-relay-", ".tmp", replacement)
    backup = None
    try:
        if original is not None:
            backup = _private_file(directory, path.name + ".", ".bak", original)
        current_path, current = _target(directory)
        if (
            _directory_identity(directory) != directory_identity
            or current_path != path
            or _identity(current) != _identity(expected)
        ):
            raise _conflict()
        os.replace(candidate, path)
    except BaseException:
        if backup is not None:
            with suppress(OSError):
                backup.unlink()
        raise
    finally:
        with suppress(OSError):
            candidate.unlink()

    result = {"filename": path.name, "changed": True}
    if backup is not None:
        result["backup"] = backup.name
    return result


def apply_opencode_v2(config: dict) -> dict:
    """Apply a trusted V2 export. The caller never selects the destination path."""
    try:
        provider = config["providers"]["llama-relay"]
        model = config["model"]
        if not isinstance(provider, dict) or not isinstance(model, str):
            raise TypeError
    except (KeyError, TypeError):
        raise RelayError(400, "relay_opencode_apply_invalid_export") from None

    with _LOCK:
        try:
            return _apply(_config_directory(), provider, model)
        except (OSError, RuntimeError):
            raise RelayError(500, "relay_opencode_config_write_failed") from None
