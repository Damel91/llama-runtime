import json
import subprocess
import sys

import pytest


_CHILD = r"""
import json
import sys
from threading import current_thread
from unittest.mock import patch

sys.path[:0] = json.loads(sys.argv[2])

from runtime_llama import TailnetConfig, TailnetError, TailnetHelper, build_adapter
from runtime_llama.errors import LLMBackendError

mode, fixture_path = sys.argv[1], sys.argv[3]
fixture_url = "http://fixture.invalid:8080"

if mode.startswith("automatic"):
    adapter = build_adapter({
        "model": "fixture",
        "runtime_endpoints": [{"base_url": fixture_url}],
        "runtime_discovery_timeout": 0.1,
    })
    seen = []

    def cancel():
        thread = current_thread().name
        if mode == "automatic_caller" or thread.startswith("runtime-probe"):
            seen.append(thread)
            adapter.close()
        return False

    try:
        adapter.discover_models(cancellation_probe=cancel)
    except LLMBackendError as exc:
        assert str(exc) == "runtime_adapter_close_from_discovery_callback", str(exc)
    else:
        raise AssertionError("reentrant close did not reject")

    assert len(seen) == 1
    assert seen[0].startswith("runtime-probe") == (mode == "automatic_worker")
    assert not adapter._runtime_closed and not adapter._runtime_operations
    adapter.close()
    assert adapter._runtime_closed
else:
    helper = TailnetHelper(TailnetConfig(fixture_path, fixture_path, "fixture"))
    calls = 0

    def cancel():
        global calls
        calls += 1
        if mode == "tailnet_discovery" or calls == 2:
            helper.close()
        return False

    try:
        if mode == "tailnet_discovery":
            with patch.object(helper, "_assert_running"):
                helper.discover_models(fixture_url, cancellation_probe=cancel)
        else:
            with (
                patch("runtime_llama.tailnet.subprocess.Popen"),
                patch.object(helper, "_read_status"),
                patch.object(helper, "_drain_stderr"),
                patch.object(helper, "_write_control"),
            ):
                helper.start(cancellation_probe=cancel)
    except TailnetError as exc:
        assert str(exc) == "tailnet_helper_close_from_cancel_callback", str(exc)
    else:
        raise AssertionError("reentrant close did not reject")

    assert not helper._discovery_calls
    assert all(not worker.is_alive() for worker in helper._threads)
    if mode == "tailnet_discovery":
        assert calls == 1 and not helper._closing.is_set() and not helper._closed
    else:
        assert calls == 2 and helper._closed
    helper.close()
    assert helper._closed
"""


@pytest.mark.parametrize(
    "mode",
    ["automatic_caller", "automatic_worker", "tailnet_start", "tailnet_discovery"],
)
def test_sdk_close_from_cancellation_callback_is_rejected_without_hanging(
    tmp_path, mode
):
    # Supervise the child so a regression cannot deadlock the test runner.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD,
            mode,
            json.dumps(sys.path),
            str(tmp_path / "fixture"),
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
