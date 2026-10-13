"""Deterministic source-release privacy checks; no inference or live servers."""

from __future__ import annotations

import gzip
import importlib.util
import io
from pathlib import Path
import struct
import tarfile

import pytest
from setuptools import build_meta

_SPEC = importlib.util.spec_from_file_location(
    "release_build_backend", Path(__file__).resolve().parents[1] / "_build_backend.py"
)
assert _SPEC is not None and _SPEC.loader is not None
backend = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(backend)


def _archive(path: Path, members: list[tarfile.TarInfo] | None = None) -> bytes:
    if members is None:
        directory = tarfile.TarInfo("runtime_llama-0.4.1/")
        directory.type = tarfile.DIRTYPE
        module = tarfile.TarInfo("runtime_llama-0.4.1/src/module.py")
        module.size = len(b"payload\x00\xff\n")
        executable = tarfile.TarInfo("runtime_llama-0.4.1/probes/check")
        executable.mode = 0o4755
        executable.size = module.size
        members = [directory, module, executable]
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="private-host-source.tar", mode="wb", fileobj=raw
        ) as gz:
            with tarfile.open(
                fileobj=gz, mode="w", format=tarfile.PAX_FORMAT
            ) as archive:
                for member in members:
                    member.uid, member.gid = 12345, 54321
                    member.uname, member.gname = "private-owner", "private-group"
                    member.mtime = 1723456789.125
                    member.pax_headers = {
                        "comment": "/private/home/operator@example.invalid",
                        "atime": "1723456790.456",
                    }
                    archive.addfile(
                        member,
                        io.BytesIO(b"payload\x00\xff\n") if member.size else None,
                    )
    return path.read_bytes()


def test_build_sdist_normalizes_all_headers_and_preserves_payload(
    tmp_path, monkeypatch
):
    path = tmp_path / "runtime_llama-0.4.1.tar.gz"
    original = _archive(path)
    settings = {"--global-option": ["test-setting"]}
    calls = []

    def setuptools_sdist(directory, config_settings=None):
        calls.append((directory, config_settings))
        return path.name

    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    monkeypatch.setattr(backend._setuptools, "build_sdist", setuptools_sdist)
    assert backend.build_sdist(tmp_path, settings) == path.name
    assert calls == [(tmp_path, settings)]
    normalized = path.read_bytes()
    assert normalized != original
    assert (
        normalized[:4] == b"\x1f\x8b\x08\x00"
    )  # No gzip filename, comment or extra fields.
    assert struct.unpack("<I", normalized[4:8])[0] == backend._DEFAULT_EPOCH
    expanded = gzip.decompress(normalized)
    for private in [
        b"private-owner",
        b"private-group",
        b"operator@example",
        b"private-host",
    ]:
        assert private not in normalized
        assert private not in expanded
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == [
            "runtime_llama-0.4.1",
            "runtime_llama-0.4.1/src/module.py",
            "runtime_llama-0.4.1/probes/check",
        ]
        for member in members:
            assert (member.uid, member.gid, member.uname, member.gname) == (
                0,
                0,
                "",
                "",
            )
            assert member.mtime == backend._DEFAULT_EPOCH
            assert member.pax_headers == {}
            assert member.linkname == ""
            if member.isfile():
                assert archive.extractfile(member).read() == b"payload\x00\xff\n"
        assert [member.mode for member in members] == [0o755, 0o644, 0o755]
    assert not list(tmp_path.glob(".sdist-normalize-*"))


def test_source_epoch_controls_gzip_and_tar_and_repeated_normalization_is_identical(
    tmp_path, monkeypatch
):
    path = tmp_path / "release.tar.gz"
    _archive(path)
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "946684800")
    epoch = backend._source_epoch()
    backend._normalize_sdist(path, epoch)
    once = path.read_bytes()
    backend._normalize_sdist(path, epoch)
    assert path.read_bytes() == once
    assert struct.unpack("<I", once[4:8])[0] == epoch
    with tarfile.open(path, "r:gz") as archive:
        assert all(member.mtime == epoch for member in archive)


@pytest.mark.parametrize(
    "value", ["", "-1", "1.5", "NaN", " 12", "4294967296", "9" * 30]
)
def test_invalid_epoch_fails_before_setuptools_produces_an_archive(
    tmp_path, monkeypatch, value
):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", value)
    monkeypatch.setattr(
        backend._setuptools,
        "build_sdist",
        lambda *args, **kwargs: pytest.fail("not called"),
    )
    with pytest.raises(ValueError, match="SOURCE_DATE_EPOCH"):
        backend.build_sdist(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "root/../../escape",
        "root//file",
        "root/./file",
        "C:/host",
        "root\\host",
        "root/bad\nname",
    ],
)
def test_unsafe_member_paths_remove_the_unpublishable_archive(tmp_path, name):
    path = tmp_path / "release.tar.gz"
    member = tarfile.TarInfo(name)
    _archive(path, [member])
    with pytest.raises(ValueError, match="Unsafe"):
        backend._normalize_sdist(path, backend._DEFAULT_EPOCH)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "kind",
    [
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.FIFOTYPE,
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
    ],
)
def test_links_and_special_members_are_never_republished(tmp_path, kind):
    path = tmp_path / "release.tar.gz"
    member = tarfile.TarInfo("root/member")
    member.type, member.linkname = kind, "/outside/private"
    _archive(path, [member])
    with pytest.raises(ValueError, match="links and special"):
        backend._normalize_sdist(path, backend._DEFAULT_EPOCH)
    assert list(tmp_path.iterdir()) == []


def test_member_count_content_and_compressed_input_quotas_fail_closed(
    tmp_path, monkeypatch
):
    for limit in [
        "_MAX_MEMBERS",
        "_MAX_MEMBER_BYTES",
        "_MAX_CONTENT_BYTES",
        "_MAX_ARCHIVE_BYTES",
        "_MAX_OUTPUT_BYTES",
    ]:
        path = tmp_path / "release.tar.gz"
        _archive(path)
        with monkeypatch.context() as patch:
            patch.setattr(backend, limit, 1)
            with pytest.raises(ValueError):
                backend._normalize_sdist(path, backend._DEFAULT_EPOCH)
        assert list(tmp_path.iterdir()) == []


def test_duplicate_names_and_unrepresentable_ustar_paths_fail_closed(tmp_path):
    cases = [
        [tarfile.TarInfo("root/duplicate"), tarfile.TarInfo("root/duplicate")],
        [tarfile.TarInfo("root/" + "x" * 300)],
    ]
    for members in cases:
        path = tmp_path / "release.tar.gz"
        _archive(path, members)
        with pytest.raises(ValueError):
            backend._normalize_sdist(path, backend._DEFAULT_EPOCH)
        assert list(tmp_path.iterdir()) == []


def test_oversized_extended_headers_are_rejected_before_metadata_allocation(tmp_path):
    path = tmp_path / "release.tar.gz"
    member = tarfile.TarInfo("root/file")
    member.pax_headers = {"comment": "x" * (backend._MAX_HEADER_BYTES + 1)}
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        archive.addfile(member)
    with pytest.raises(ValueError, match="extended header"):
        backend._normalize_sdist(path, backend._DEFAULT_EPOCH)
    assert list(tmp_path.iterdir()) == []


def test_corrupt_gzip_trailer_is_not_hidden_by_tar_end_markers(tmp_path):
    path = tmp_path / "release.tar.gz"
    _archive(path)
    data = bytearray(path.read_bytes())
    data[-8] ^= 1
    path.write_bytes(data)
    with pytest.raises(gzip.BadGzipFile):
        backend._normalize_sdist(path, backend._DEFAULT_EPOCH)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["replace", "rewrite"])
def test_failed_normalization_leaves_neither_original_nor_partial_output(
    tmp_path, monkeypatch, failure
):
    path = tmp_path / "release.tar.gz"
    _archive(path)

    def fail(*args, **kwargs):
        raise OSError("injected publication failure")

    monkeypatch.setattr(
        backend.os if failure == "replace" else backend,
        "replace" if failure == "replace" else "_rewrite_members",
        fail,
    )
    with pytest.raises(OSError, match="injected"):
        backend._normalize_sdist(path, backend._DEFAULT_EPOCH)
    assert list(tmp_path.iterdir()) == []


def test_other_pep517_and_pep660_hooks_are_the_original_setuptools_hooks():
    for name in [
        "build_wheel",
        "get_requires_for_build_wheel",
        "get_requires_for_build_sdist",
        "prepare_metadata_for_build_wheel",
        "build_editable",
        "get_requires_for_build_editable",
        "prepare_metadata_for_build_editable",
    ]:
        assert getattr(backend, name) is getattr(build_meta, name)
