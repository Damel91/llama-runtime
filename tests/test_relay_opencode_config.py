from concurrent.futures import ThreadPoolExecutor
import json
import os
import stat

import pytest

from runtime_llama.relay import opencode_config as writer
from runtime_llama.relay.protocol import RelayError


_PROVIDER = {
    "package": "@opencode/ai/providers/openai-compatible",
    "settings": {"baseURL": "http://127.0.0.1:8088/v1", "apiKey": "{env:RELAY_KEY}"},
    "models": {"chat": {"name": "chat", "limit": {"context": 8192, "output": 512}}},
}
_EXPORT = {"providers": {"llama-relay": _PROVIDER}, "model": "llama-relay/chat"}


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    directory = tmp_path.resolve() / "opencode"
    monkeypatch.delenv("OPENCODE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(directory.parent))
    return directory


def test_create_and_retry_strict_json(config_dir):
    assert writer.apply_opencode_v2(_EXPORT) == {
        "filename": "opencode.json",
        "changed": True,
    }
    path = config_dir / "opencode.json"
    assert json.loads(path.read_bytes()) == _EXPORT
    info = path.stat()
    if os.name == "posix":
        assert stat.S_IMODE(info.st_mode) == 0o600
    assert writer.apply_opencode_v2(_EXPORT) == {
        "filename": "opencode.json",
        "changed": False,
    }
    assert path.stat().st_mtime_ns == info.st_mtime_ns
    assert list(config_dir.iterdir()) == [path]


def test_update_preserves_strict_json_and_copies_private_backup(config_dir):
    config_dir.mkdir()
    path = config_dir / "opencode.json"
    old = (
        b'{"$schema":"https://opencode.ai/config.json",'
        b' "provider":{"legacy":{"name":"retain"}},'
        b' "providers":{"other":{"settings":{"apiKey":"unrelated-key"}}},'
        b' "model":"other/model", "plugins":["keep-me"]}\r\n'
    )
    path.write_bytes(old)
    result = writer.apply_opencode_v2(_EXPORT)
    assert result["filename"] == path.name and result["changed"] is True
    backup = config_dir / result["backup"]
    assert backup.parent == config_dir and backup.read_bytes() == old
    if os.name == "posix":
        assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    expected = json.loads(old)
    expected["model"] = _EXPORT["model"]
    expected["providers"]["llama-relay"] = _PROVIDER
    assert json.loads(path.read_bytes()) == expected
    assert b'"other":{"settings":{"apiKey":"unrelated-key"}}' in path.read_bytes()
    assert path.read_bytes().endswith(b"}\r\n")


def test_highest_precedence_jsonc_preserves_comments_and_other_source(
    config_dir, tmp_path, monkeypatch
):
    directory = tmp_path.resolve() / "override"
    directory.mkdir()
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(directory))
    json_path = directory / "opencode.json"
    json_path.write_text('{"model":"lower/config"}\n')
    path = directory / "opencode.jsonc"
    original = (
        b"// keep header\n"
        b"{\n"
        b'  "providers": { /* keep provider comment */\n'
        b'    "custom": {"settings":{"baseURL":"http://local/v1"}},\n'
        b'    "llama-relay": {"name":"old"}, // keep suffix\n'
        b"  },\n"
        b'  "model": "custom/old", // chosen model\n'
        b'  "tools": ["search // slash /* literal */",],\n'
        b"}\n"
    )
    path.write_bytes(original)
    result = writer.apply_opencode_v2(_EXPORT)
    assert result["filename"] == path.name and result["changed"] is True
    assert (directory / result["backup"]).read_bytes() == original
    assert json_path.read_text() == '{"model":"lower/config"}\n'
    assert not config_dir.exists()
    updated = path.read_bytes()
    assert b"// keep header\n" in updated
    assert b'"providers": { /* keep provider comment */\n' in updated
    assert b'"custom": {"settings":{"baseURL":"http://local/v1"}},\n' in updated
    assert b"}, // keep suffix\n" in updated
    assert b'"model": "llama-relay/chat", // chosen model\n' in updated
    assert b'"tools": ["search // slash /* literal */",],\n' in updated
    assert json.loads(writer._mask_jsonc(updated.decode()))["providers"][
        "llama-relay"
    ] == (_PROVIDER)
    assert writer.apply_opencode_v2(_EXPORT)["changed"] is False


@pytest.mark.parametrize(
    "data",
    [
        b'{"providers":null}',
        b'{"model":"one","model":"two"}',
        b'{"providers":{"x":{"private":1,"private":2}}}',
        b'{"items":[,]}',
        b'{"items":[1,,]}',
        b'{"value":NaN}',
        b'{"private": invalid secret value}',
        b"{\xff}",
        b"[]",
    ],
)
def test_refuses_malformed_config_without_modifying_it(config_dir, data):
    config_dir.mkdir()
    path = config_dir / "opencode.jsonc"
    path.write_bytes(data)
    with pytest.raises(RelayError) as caught:
        writer.apply_opencode_v2(_EXPORT)
    assert (caught.value.status, caught.value.code) == (
        422,
        "relay_opencode_config_invalid",
    )
    assert path.read_bytes() == data
    assert list(config_dir.iterdir()) == [path]


@pytest.mark.skipif(
    os.name != "posix", reason="symlink creation may require privileges"
)
@pytest.mark.parametrize("symlink_directory", [False, True])
def test_refuses_symlink_destination(config_dir, tmp_path, symlink_directory):
    other_dir = tmp_path.resolve() / "other"
    other_dir.mkdir()
    original = other_dir / "opencode.json"
    original.write_bytes(b'{"model":"other/original"}\n')
    if symlink_directory:
        config_dir.symlink_to(other_dir, target_is_directory=True)
    else:
        config_dir.mkdir()
        (config_dir / "opencode.json").symlink_to(original)
    with pytest.raises(RelayError) as caught:
        writer.apply_opencode_v2(_EXPORT)
    assert (caught.value.status, caught.value.code) == (
        409,
        "relay_opencode_config_conflict",
    )
    assert original.read_bytes() == b'{"model":"other/original"}\n'
    assert list(other_dir.iterdir()) == [original]


def test_conflict_does_not_overwrite_a_concurrent_edit(config_dir, monkeypatch):
    config_dir.mkdir()
    path = config_dir / "opencode.json"
    path.write_bytes(b'{"model":"old/initial"}\n')
    other_write = b'{"model":"changed/outside-relay"}\n'
    save = writer._private_file

    def write_then_conflict(directory, prefix, suffix, content):
        candidate = save(directory, prefix, suffix, content)
        if suffix == ".tmp":
            path.write_bytes(other_write)
        return candidate

    monkeypatch.setattr(writer, "_private_file", write_then_conflict)
    with pytest.raises(RelayError) as caught:
        writer.apply_opencode_v2(_EXPORT)
    assert (caught.value.status, caught.value.code) == (
        409,
        "relay_opencode_config_conflict",
    )
    assert path.read_bytes() == other_write
    assert list(config_dir.iterdir()) == [path]


def test_failed_replace_leaves_original_and_hides_storage_error(
    config_dir, monkeypatch
):
    config_dir.mkdir()
    path = config_dir / "opencode.json"
    path.write_bytes(b'{"model":"old/initial"}\n')

    def fail(*_args, **_kwargs):
        raise PermissionError("private storage path")

    monkeypatch.setattr(writer.os, "replace", fail)
    with pytest.raises(RelayError) as caught:
        writer.apply_opencode_v2(_EXPORT)
    assert (caught.value.status, caught.value.code) == (
        500,
        "relay_opencode_config_write_failed",
    )
    assert path.read_bytes() == b'{"model":"old/initial"}\n'
    assert list(config_dir.iterdir()) == [path]


def test_concurrent_duplicate_apply_is_idempotent(config_dir):
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(writer.apply_opencode_v2, [_EXPORT] * 8))
    assert sum(result["changed"] for result in outcomes) == 1
    assert list(config_dir.iterdir()) == [config_dir / "opencode.json"]
