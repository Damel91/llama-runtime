from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread, enumerate as threads
import time
import traceback

import pytest

from runtime_llama import TailnetConfig, TailnetError, TailnetHelper
from runtime_llama.errors import LLMCancelledError


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate()


def tailnet_threads():
    return [t for t in threads() if t.name.startswith("runtime_llama_tailnet_")]


@pytest.fixture
def helper_config(tmp_path):
    created = []

    def make(mode="normal", *, ready=None):
        index = len(created)
        script = tmp_path / f"helper-{index}"
        record = tmp_path / f"record-{index}.json"
        stopped = tmp_path / f"stopped-{index}"
        event = ready or {
            "protocol": 1,
            "status": "ready",
            "proxy_url": "http://127.0.0.1:12345",
            "backend_version": "1.104.1",
        }
        body = f"""#!{sys.executable}
import json,os,signal,sys,time
from pathlib import Path
mode={mode!r}
if mode=='blocked-stdin':
    time.sleep(20)
    sys.exit(1)
config=json.loads(sys.stdin.buffer.readline(65537))
Path({str(record)!r}).write_text(json.dumps({{
    'config':config,'argv':sys.argv,
    'env':{{k:v for k,v in os.environ.items() if k.upper().startswith(('TS_','TSNET_')) or k.lower() in ('http_proxy','https_proxy','all_proxy','no_proxy')}},
}}))
if mode=='timeout':
    time.sleep(20)
    sys.exit(1)
if mode=='early-exit':
    print('SECRET_PROVIDER_MESSAGE',file=sys.stderr,flush=True)
    sys.exit(1)
if mode=='oversize':
    sys.stdout.write('x'*9000)
    sys.stdout.flush()
elif mode=='malformed':
    print('SECRET_PROVIDER_MESSAGE',flush=True)
elif mode=='duplicate':
    print('{{"protocol":1,"protocol":1,"status":"ready","proxy_url":"http://127.0.0.1:12345","backend_version":"1.104.1"}}',flush=True)
else:
    print(json.dumps({event!r}),flush=True)
if mode=='late-malformed':
    time.sleep(0.15)
    print('SECRET_PROVIDER_MESSAGE',flush=True)
if mode=='ignore-stop':
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    time.sleep(20)
    sys.exit(1)
for line in sys.stdin.buffer:
    if json.loads(line).get('command')=='shutdown':
        break
Path({str(stopped)!r}).write_text('closed')
if mode=='shutdown-error':
    print(json.dumps({{'protocol':1,'status':'error','code':'shutdown_timeout'}}),flush=True)
    sys.exit(2)
print(json.dumps({{'protocol':1,'status':'closed'}}),flush=True)
"""
        script.write_text(body)
        script.chmod(0o700)
        config = TailnetConfig(
            executable=script,
            state_dir=tmp_path / f"state-{index}",
            hostname="sdk-consumer",
            auth_key="synthetic-auth-key-only",
            control_url="https://headscale.example.test",
            startup_timeout_sec=1,
            dial_timeout_sec=0.2,
            shutdown_timeout_sec=0.1,
        )
        helper = TailnetHelper(config)
        created.append(helper)
        return helper, record, stopped

    yield make
    for helper in created:
        helper.close()
    wait_for(lambda: not tailnet_threads())


def test_construct_and_import_do_not_launch_or_touch_state(helper_config, monkeypatch):
    helper, record, _ = helper_config()

    def forbidden(*args, **kwargs):
        raise AssertionError("process launch during construction")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    extra = TailnetHelper(helper.config)
    assert not extra.running and not record.exists()
    assert not Path(helper.config.state_dir).exists()
    assert helper.config.auth_key not in repr(helper.config)
    assert helper.config.auth_key not in repr(extra)
    with pytest.raises(TailnetError, match="unavailable"):
        _ = extra.proxy_url
    extra.close()


@pytest.mark.parametrize(
    "value",
    [0, -1, True, "2", float("nan"), float("inf"), 1e308, 10**1000, 86401, 1e-12],
)
@pytest.mark.parametrize(
    "field", ["startup_timeout_sec", "dial_timeout_sec", "shutdown_timeout_sec"]
)
def test_lifecycle_budgets_fail_before_launch(helper_config, field, value):
    helper, record, _ = helper_config()
    with pytest.raises(ValueError, match="at most 86400"):
        replace(helper.config, **{field: value})
    assert not record.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("hostname", "bad host"),
        ("hostname", ""),
        ("ephemeral", 1),
        ("auth_key", "secret\nline"),
        ("auth_key", ""),
        ("executable", "relative"),
        ("state_dir", "relative"),
        ("control_url", "http://headscale.example.test"),
        ("control_url", "https://user:secret@example.test"),
        ("control_url", "https://example.test/path"),
        ("control_url", "https://example.test?secret=key"),
    ],
)
def test_configuration_rejects_invalid_values_without_echoing(
    helper_config, field, value
):
    helper, _, _ = helper_config()
    with pytest.raises(ValueError) as caught:
        replace(helper.config, **{field: value})
    assert "secret" not in str(caught.value)


def test_consumer_configuration_only_stdin_and_child_environment_isolated(
    helper_config, monkeypatch
):
    for key in [
        "TS_AUTHKEY",
        "TS_CONTROL_URL",
        "TSNET_FORCE_LOGIN",
        "TS_CLIENT_SECRET",
        "HTTP_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "NO_PROXY",
    ]:
        monkeypatch.setenv(key, "hostile-ambient-value")
    helper, record, stopped = helper_config()
    with helper:
        data = json.loads(record.read_text())
        cfg = data["config"]
        assert data["env"] == {}
        assert data["argv"] == [helper.config.executable]
        assert cfg["control_url"] == "https://headscale.example.test"
        assert cfg["auth_key"] == helper.config.auth_key
        assert cfg["hostname"] == "sdk-consumer" and cfg["protocol"] == 1
        assert len(cfg["proxy_token"]) >= 32
        assert (
            helper.proxy_url == f"http://runtime:{cfg['proxy_token']}@127.0.0.1:12345"
        )
        assert cfg["proxy_token"] not in repr(helper)
        assert helper.start() is helper and helper.running
        assert os.environ["TS_AUTHKEY"] == "hostile-ambient-value"
    assert stopped.exists() and helper._process.poll() == 0
    assert not helper.running and not tailnet_threads()
    helper.close()
    with pytest.raises(TailnetError, match="closed"):
        helper.start()


def test_tailscale_default_is_explicit_empty_control_not_ambient(helper_config):
    helper, record, _ = helper_config()
    helper.config = replace(
        helper.config, control_url=None, auth_key=None, ephemeral=True
    )
    with helper:
        cfg = json.loads(record.read_text())["config"]
        assert (
            cfg["control_url"] == ""
            and cfg["auth_key"] == ""
            and cfg["ephemeral"] is True
        )


@pytest.mark.parametrize(
    "patch",
    [
        {"protocol": True},
        {"protocol": 2},
        {"backend_version": "other"},
        {"proxy_url": "http://0.0.0.0:12345"},
        {"proxy_url": "http://localhost:12345"},
        {"proxy_url": "http://127.0.0.1:0"},
        {"proxy_url": "http://127.0.0.1:12345/"},
        {"proxy_url": "http://secret@127.0.0.1:12345"},
        {"proxy_url": "http://127.0.0.1:12345?secret=key"},
        {"extra": "SECRET_PROVIDER_MESSAGE"},
    ],
)
def test_invalid_readiness_fails_closed_and_drains(helper_config, patch):
    ready = {
        "protocol": 1,
        "status": "ready",
        "proxy_url": "http://127.0.0.1:12345",
        "backend_version": "1.104.1",
        **patch,
    }
    helper, _, _ = helper_config(ready=ready)
    with pytest.raises(TailnetError) as caught:
        helper.start()
    assert "secret" not in str(caught.value).lower()
    assert helper._process.poll() is not None and not tailnet_threads()


@pytest.mark.parametrize("mode", ["oversize", "malformed", "duplicate", "early-exit"])
def test_invalid_status_or_early_exit_does_not_leak(helper_config, mode):
    helper, _, _ = helper_config(mode)
    with pytest.raises(TailnetError) as caught:
        helper.start()
    assert "SECRET_PROVIDER_MESSAGE" not in "".join(
        traceback.format_exception(caught.value)
    )
    assert helper._process.poll() is not None and not tailnet_threads()


def test_structured_error_is_sanitized_and_no_secret_code_echo(helper_config):
    helper, _, _ = helper_config(
        ready={"protocol": 1, "status": "error", "code": "state_in_use"}
    )
    with pytest.raises(TailnetError, match="state_in_use"):
        helper.start()
    other, _, _ = helper_config(
        ready={"protocol": 1, "status": "error", "code": "SECRET_PROVIDER_MESSAGE"}
    )
    with pytest.raises(TailnetError) as caught:
        other.start()
    assert "SECRET_PROVIDER_MESSAGE" not in str(caught.value)


@pytest.mark.parametrize("mode", ["timeout", "blocked-stdin"])
def test_outer_startup_budget_covers_silent_child_and_blocked_config_writer(
    helper_config, mode
):
    helper, _, _ = helper_config(mode)
    helper.config = replace(
        helper.config,
        startup_timeout_sec=0.1,
        state_dir="/" + "x" * 40000,
        auth_key="a" * 8192,
    )
    start = time.monotonic()
    with pytest.raises(TailnetError, match="start_timeout"):
        helper.start()
    assert time.monotonic() - start < 2
    assert helper._process.poll() is not None and not tailnet_threads()


def test_shutdown_kills_child_that_ignores_eof_and_terminate(helper_config):
    helper, _, _ = helper_config("ignore-stop")
    helper.start()
    start = time.monotonic()
    helper.close()
    assert time.monotonic() - start < 2.5
    assert helper._process.poll() is not None and not tailnet_threads()


def test_helper_shutdown_error_is_reported_after_drain(helper_config):
    helper, _, _ = helper_config("shutdown-error")
    helper.start()
    with pytest.raises(TailnetError, match="tailnet_helper_shutdown_failed"):
        helper.close()
    assert helper._process.poll() == 2 and not tailnet_threads()
    assert not helper.running
    helper.close()


def test_startup_cancellation_and_concurrent_close_are_bounded(helper_config):
    for close_from_other_thread in (False, True):
        helper, record, _ = helper_config("timeout")
        requested = Event()
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = pool.submit(helper.start, cancellation_probe=requested.is_set)
            wait_for(record.exists)
            if close_from_other_thread:
                cleanup = pool.submit(helper.close)
                with pytest.raises(TailnetError, match="closed"):
                    pending.result(timeout=2)
                cleanup.result(timeout=2)
            else:
                requested.set()
                with pytest.raises(LLMCancelledError):
                    pending.result(timeout=2)
        assert helper._process.poll() is not None and not tailnet_threads()


def test_precancel_does_not_launch_and_probe_exception_is_safe(helper_config):
    helper, record, _ = helper_config()
    with pytest.raises(LLMCancelledError):
        helper.start(cancellation_probe=lambda: True)
    assert not record.exists() and helper._process is None

    def broken():
        raise RuntimeError("SECRET_PROVIDER_MESSAGE")

    with pytest.raises(TailnetError) as caught:
        helper.start(cancellation_probe=broken)
    assert "SECRET_PROVIDER_MESSAGE" not in "".join(
        traceback.format_exception(caught.value)
    )
    assert helper._process is None


def test_parallel_start_is_single_process_and_helper_closes_owned_clients(
    helper_config,
):
    helper, _, _ = helper_config()
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(
            value is helper for value in pool.map(lambda _: helper.start(), range(4))
        )
    client = helper.client("http://peer.invalid:8080", timeout=0.2)
    assert client.proxy_url == helper.proxy_url and client.trust_env is False
    for kwargs in [
        {"proxy_url": helper.proxy_url},
        {"trust_env": True},
        {"session": object()},
    ]:
        with pytest.raises(ValueError):
            helper.client("http://peer.invalid:8080", **kwargs)
    helper.close()
    assert client._closed and not tailnet_threads()


def test_post_ready_protocol_failure_closes_process_and_owned_clients(helper_config):
    helper, _, _ = helper_config("late-malformed")
    helper.start()
    client = helper.client("http://peer.invalid:8080")
    wait_for(lambda: not helper.running)
    helper.close()
    assert client._closed and helper._process.poll() is not None
    assert not tailnet_threads()


def test_discovery_route_and_close_cancel_and_drain_active_call(helper_config):
    entered, release = Event(), Event()
    observed = []
    record_holder = []

    class Proxy(BaseHTTPRequestHandler):
        def do_GET(self):
            cfg = json.loads(record_holder[0].read_text())["config"]
            expected = (
                "Basic "
                + base64.b64encode(f"runtime:{cfg['proxy_token']}".encode()).decode()
            )
            observed.append(
                (
                    self.path,
                    self.headers.get("Authorization"),
                    self.headers.get("Proxy-Authorization"),
                )
            )
            assert self.headers.get("Proxy-Authorization") == expected
            entered.set()
            release.wait(timeout=3)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    server.daemon_threads = True
    serving = Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    serving.start()
    helper, record, _ = helper_config(
        ready={
            "protocol": 1,
            "status": "ready",
            "proxy_url": f"http://127.0.0.1:{server.server_port}",
            "backend_version": "1.104.1",
        }
    )
    record_holder.append(record)
    try:
        helper.start()
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = pool.submit(
                helper.discover_models,
                "http://peer.invalid:8080",
                api_key="origin-token",
                timeout=20,
            )
            assert entered.wait(timeout=1)
            pool.submit(helper.close).result(timeout=2)
            with pytest.raises(LLMCancelledError):
                pending.result(timeout=1)
        assert len(observed) == 1 and observed[0][1] == "Bearer origin-token"
        assert observed[0][0] == "http://peer.invalid:8080/v1/models"
        assert not tailnet_threads()
    finally:
        release.set()
        helper.close()
        server.shutdown()
        server.server_close()
        serving.join(timeout=1)
