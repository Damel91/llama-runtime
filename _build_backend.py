"""Setuptools hooks with a host-independent, validated source archive.

The source distribution is rewritten before it is returned to the build
frontend. Wheel and editable hooks are the original setuptools hooks.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path
import re
import stat
import tarfile
import tempfile
from typing import Any, BinaryIO

from setuptools import build_meta as _setuptools

_DEFAULT_EPOCH = 315532800  # 1980-01-01 UTC, also representable in wheel ZIPs.
_MAX_EPOCH = 0xFFFFFFFF  # gzip stores an unsigned 32-bit timestamp.
_MAX_MEMBERS = 20_000
_MAX_MEMBER_BYTES = 128 * 1024 * 1024
_MAX_CONTENT_BYTES = 512 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_MAX_OUTPUT_BYTES = _MAX_CONTENT_BYTES + _MAX_MEMBERS * 2048 + 1024 * 1024
_MAX_HEADER_BYTES = 64 * 1024


def __getattr__(name: str) -> Any:
    """Preserve every other present or future setuptools build hook."""
    return getattr(_setuptools, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_setuptools)))


def _source_epoch() -> int:
    value = os.environ.get("SOURCE_DATE_EPOCH")
    if value is None:
        return _DEFAULT_EPOCH
    if re.fullmatch(r"[0-9]{1,20}", value) is None:
        raise ValueError("SOURCE_DATE_EPOCH must be a nonnegative integer")
    epoch = int(value)
    if epoch > _MAX_EPOCH:
        raise ValueError("SOURCE_DATE_EPOCH exceeds the gzip timestamp range")
    return epoch


class _BoundedReader:
    def __init__(self, source: BinaryIO, limit: int) -> None:
        self._source = source
        self._remaining = limit

    def read(self, size: int = -1) -> bytes:
        # Never follow a growing input beyond the original descriptor size.
        count = min(self._remaining, 1024 * 1024)
        if size >= 0:
            count = min(count, size)
        if not count and size != 0:
            if self._source.read(1):
                raise ValueError("Source archive exceeds its stream size limit")
            return b""
        data = self._source.read(count)
        self._remaining -= len(data)
        return data


class _BoundedWriter:
    def __init__(self, target: BinaryIO, limit: int) -> None:
        self._target = target
        self._remaining = limit

    def write(self, data: bytes) -> int:
        if len(data) > self._remaining:
            raise ValueError("Normalized source archive exceeds its size limit")
        written = self._target.write(data)
        self._remaining -= written
        return written

    def flush(self) -> None:
        self._target.flush()


class _InputTarInfo(tarfile.TarInfo):
    @classmethod
    def frombuf(cls, buf: bytes, encoding: str, errors: str) -> tarfile.TarInfo:
        member = super().frombuf(buf, encoding, errors)
        extended = {
            tarfile.XHDTYPE,
            tarfile.XGLTYPE,
            tarfile.SOLARIS_XHDTYPE,
            tarfile.GNUTYPE_LONGNAME,
            tarfile.GNUTYPE_LONGLINK,
        }
        if member.size < 0 or member.size > _MAX_MEMBER_BYTES:
            raise ValueError("Source archive member exceeds its size limit")
        if member.type in extended and member.size > _MAX_HEADER_BYTES:
            raise ValueError("Source archive extended header exceeds its size limit")
        if member.type == tarfile.GNUTYPE_SPARSE:
            raise ValueError("Sparse source archive members are forbidden")
        return member


def _member_name(member: tarfile.TarInfo) -> str:
    name = member.name
    if member.isdir() and name.endswith("/"):
        name = name[:-1]
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or re.match(r"^[A-Za-z]:", name)
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError("Unsafe source archive member path")
    return name


def _same_file(before: os.stat_result, after: os.stat_result) -> bool:
    return all(
        getattr(before, field) == getattr(after, field)
        for field in (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mode",
            "st_nlink",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    )


def _rewrite_members(
    source: tarfile.TarFile, target: tarfile.TarFile, epoch: int
) -> None:
    names: set[str] = set()
    total_size = 0
    for member in source:
        if len(names) >= _MAX_MEMBERS:
            raise ValueError("Source archive contains too many members")
        if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}:
            raise ValueError("Source archive links and special members are forbidden")
        if member.issparse():
            raise ValueError("Sparse source archive members are forbidden")
        name = _member_name(member)
        if name in names:
            raise ValueError("Duplicate source archive member path")
        names.add(name)
        if member.size < 0 or member.size > _MAX_MEMBER_BYTES:
            raise ValueError("Source archive member exceeds its size limit")
        if member.isdir() and member.size:
            raise ValueError("Source archive directory contains unexpected data")
        total_size += member.size
        if total_size > _MAX_CONTENT_BYTES:
            raise ValueError("Source archive exceeds its content size limit")

        # A fresh USTAR header cannot retain host PAX metadata, owner names,
        # unused link targets, device numbers, or extended timestamp fields.
        normalized = tarfile.TarInfo(name)
        normalized.type = tarfile.DIRTYPE if member.isdir() else tarfile.REGTYPE
        normalized.mode = 0o755 if member.isdir() or member.mode & 0o111 else 0o644
        normalized.size = member.size
        normalized.uid = normalized.gid = 0
        normalized.uname = normalized.gname = ""
        normalized.mtime = epoch
        normalized.pax_headers = {}
        if member.isdir():
            target.addfile(normalized)
        else:
            payload = source.extractfile(member)
            if payload is None:
                raise ValueError("Source archive member content is unavailable")
            with payload:
                target.addfile(normalized, payload)
    if not names:
        raise ValueError("Source archive is empty")


def _normalize_sdist(archive: Path, epoch: int) -> None:
    temporary: Path | None = None
    try:
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(archive, flags)
        with os.fdopen(descriptor, "rb") as original:
            before = os.fstat(original.fileno())
            current = archive.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size > _MAX_ARCHIVE_BYTES
                or not _same_file(before, current)
            ):
                raise ValueError("Source archive must be one bounded regular file")
            fd, filename = tempfile.mkstemp(
                prefix=".sdist-normalize-", dir=archive.parent
            )
            temporary = Path(filename)
            with os.fdopen(fd, "wb") as raw:
                output = _BoundedWriter(raw, _MAX_OUTPUT_BYTES)
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=output, mtime=epoch
                ) as gz:
                    with tarfile.open(
                        fileobj=gz, mode="w|", format=tarfile.USTAR_FORMAT
                    ) as target:
                        compressed = _BoundedReader(original, before.st_size)
                        with gzip.GzipFile(fileobj=compressed, mode="rb") as expanded:
                            bounded = _BoundedReader(expanded, _MAX_OUTPUT_BYTES)
                            with tarfile.open(
                                fileobj=bounded, mode="r|", tarinfo=_InputTarInfo
                            ) as source:
                                _rewrite_members(source, target, epoch)
                            # Verify gzip trailer/CRC and bound all expanded data,
                            # including padding or metadata beyond the tar EOF.
                            while bounded.read(65536):
                                pass
                raw.flush()
                os.fsync(raw.fileno())
            if not _same_file(before, os.fstat(original.fileno())) or not _same_file(
                before, archive.lstat()
            ):
                raise ValueError("Source archive changed during normalization")
        os.chmod(temporary, 0o644)
        os.replace(temporary, archive)
        temporary = None
    except BaseException:
        # A frontend must never receive the original host-identifying archive
        # after a failed rewrite, including disk/replace errors or cancellation.
        archive.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_sdist(
    sdist_directory: str | os.PathLike[str], config_settings: Any = None
) -> str:
    epoch = _source_epoch()
    filename = _setuptools.build_sdist(sdist_directory, config_settings=config_settings)
    if (
        not isinstance(filename, str)
        or not filename.endswith(".tar.gz")
        or any(char in filename for char in "/\\\0\r\n:")
    ):
        raise ValueError("Setuptools returned an unsafe source archive filename")
    _normalize_sdist(Path(sdist_directory) / filename, epoch)
    return filename
