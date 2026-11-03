"""Optional entry points and process-level graceful shutdown over real TCP."""

from pathlib import Path
import os
import signal
import socket
import subprocess
import sys

import httpx
import pytest
import uvicorn

from runtime_llama import __version__
from runtime_llama.relay import cli
from runtime_llama.relay.config import RelayConfigStore, default_config
from test_relay_qualification import configuration, eventually, native_server


_WITHOUT_WEB = """
import runpy
import sys

class BlockWeb:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'uvicorn', 'starlette'}:
            raise ModuleNotFoundError(fullname, name=fullname)

sys.meta_path.insert(0, BlockWeb())
runpy.run_module('runtime_llama.relay', run_name='__main__')
"""


def source_environment():
    return {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }


@pytest.mark.parametrize("flag", ["--help", "--version"])
def test_module_help_and_version_do_not_require_web_dependencies(flag, tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", _WITHOUT_WEB, flag],
        cwd=tmp_path,
        env=source_environment(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0 and not result.stderr
    assert "llama-runtime-relay" in result.stdout
    assert (__version__ if flag == "--version" else "--config") in result.stdout
    assert not list(tmp_path.iterdir())


def test_missing_web_dependencies_have_a_friendly_install_message(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _WITHOUT_WEB,
            "--config",
            str(tmp_path / "missing.json"),
        ],
        cwd=tmp_path,
        env=source_environment(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "pip install 'runtime-llama[relay]'" in result.stderr
    assert "Traceback" not in result.stderr and not list(tmp_path.iterdir())


def test_per_user_configuration_path_uses_absolute_platform_config_dir(
    monkeypatch, tmp_path
):
    variable = "APPDATA" if sys.platform == "win32" else "XDG_CONFIG_HOME"
    fallback = (
        Path("AppData") / "Roaming" if sys.platform == "win32" else Path(".config")
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv(variable, "relative")
    assert (
        cli.default_config_path()
        == tmp_path / fallback / "llama-runtime" / "relay.json"
    )
    monkeypatch.setenv(variable, str(tmp_path / "custom"))
    assert (
        cli.default_config_path()
        == tmp_path / "custom" / "llama-runtime" / "relay.json"
    )
    assert not list(tmp_path.iterdir())


def test_listener_overrides_are_active_only_and_do_not_add_server_quotas(
    monkeypatch, tmp_path
):
    path = tmp_path / "relay.json"
    store = RelayConfigStore(path)
    saved = store.replace(0, default_config())
    before = path.read_bytes()
    seen = []

    def run(server):
        config = server.config
        assert (config.host, config.port) == ("127.0.0.2", 19081)
        assert config.ws == "none" and not config.access_log
        assert config.limit_concurrency is None and config.limit_max_requests is None
        assert config.timeout_graceful_shutdown is None
        service = config.app.state.relay_service
        assert service.listener["host"] == config.host
        assert service.listener["port"] == config.port
        assert service.config_store.snapshot().public() == saved.public()
        assert service.catalog_snapshot()["observed_at"] is None
        assert service.status()["tailnet"]["state"] == "stopped"
        service.close()
        seen.append(True)

    monkeypatch.setattr(uvicorn.Server, "run", run)
    assert (
        cli.main(["--config", str(path), "--host", "127.0.0.2", "--port", "19081"]) == 0
    )
    assert seen == [True] and path.read_bytes() == before


def test_invalid_configuration_fails_without_printing_private_content(
    monkeypatch, tmp_path, capsys
):
    path = tmp_path / "bad.json"
    path.write_text('{"private-configuration-contents":', encoding="utf-8")

    def forbidden(_server):
        pytest.fail("invalid configuration must not open a listener")

    monkeypatch.setattr(uvicorn.Server, "run", forbidden)
    assert cli.main(["--config", str(path)]) == 2
    error = capsys.readouterr().err
    assert "relay_persisted_config_invalid" in error
    assert "private-configuration-contents" not in error
    assert "Traceback" not in error


@pytest.mark.skipif(
    os.name == "nt", reason="this process signal test uses POSIX signals"
)
@pytest.mark.parametrize(
    "sig", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"]
)
def test_module_signal_drains_quiet_native_stream_and_exits_zero(tmp_path, sig):
    with native_server("strata", lambda *_args: "quiet") as (native_url, model, native):
        path = tmp_path / "relay.json"
        config = configuration(native_url)
        config["tailnet"] = {
            "executable": str(tmp_path / "not-a-running-helper"),
            "state_dir": str(tmp_path / "unused-state"),
            "hostname": "fixture-relay",
        }
        RelayConfigStore(path).replace(0, config)
        before = path.read_bytes()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "runtime_llama.relay",
                "--config",
                str(path),
                "--port",
                str(port),
            ],
            cwd=tmp_path,
            env=source_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", timeout=5, trust_env=False
            ) as api:

                def ready():
                    if process.poll() is not None:
                        stdout, stderr = process.communicate()
                        pytest.fail(f"relay stopped before serving: {stdout}\n{stderr}")
                    try:
                        return api.get("/health").status_code == 200
                    except httpx.TransportError:
                        return False

                eventually(ready, timeout=10)
                assert not native["requests"]
                assert not (tmp_path / "unused-state").exists()
                with api.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "quiet"}],
                        "stream": True,
                    },
                ) as response:
                    assert response.status_code == 200 and native["quiet_started"].wait(
                        5
                    )
                    process.send_signal(sig)
                    assert native["peer_closed"].wait(5), (
                        "native quiet socket was not closed"
                    )
                    stdout, stderr = process.communicate(timeout=10)
                    assert process.returncode == 0, (stdout, stderr)
                    assert "Traceback" not in stderr
            assert native["requests"]["POST /v1/chat/completions"] == 1
            assert path.read_bytes() == before
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
