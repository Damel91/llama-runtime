"""The live probe is qualified entirely against local HTTP fixtures here."""

from __future__ import annotations

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
import time

import pytest


PROBE_PATH = Path(__file__).resolve().parents[1] / "probes/qualify_tool_recovery.py"
spec = importlib.util.spec_from_file_location("tool_recovery_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def cli_args(tmp_path, base_url="http://fixture.invalid", *extra):
    return [
        "--base-url",
        base_url,
        "--model",
        "fixture-model",
        "--output",
        str(tmp_path / "report.json"),
        *extra,
    ]


def test_help_needs_only_standard_library_and_writes_nothing(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-I", str(PROBE_PATH), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--allow-live" in completed.stdout
    assert not list(tmp_path.iterdir())


def test_missing_optin_never_imports_runtime_or_writes_files(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-I", str(PROBE_PATH), *cli_args(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "explicit --allow-live" in completed.stderr
    assert "ModuleNotFoundError" not in completed.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/runtime",
        "http://user:credential@host",
        "http://host/path",
        "http://host?",
        "http://host#",
        "http://host:bad",
        "http://host:0",
        "http://host\\bad",
        "http://host/ bad",
    ],
)
def test_invalid_origin_refused_before_import_or_file_creation(tmp_path, url):
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(PROBE_PATH),
            "--allow-live",
            *cli_args(tmp_path, url),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "ModuleNotFoundError" not in completed.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "extra",
    [["--timeout", "nan"], ["--timeout", "0"], ["--max-tokens", "257"]],
)
def test_invalid_budget_never_reaches_run(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(probe, "run", lambda _: pytest.fail("must not start requests"))
    with pytest.raises(SystemExit) as caught:
        probe.main(["--allow-live", *cli_args(tmp_path), *extra])
    assert caught.value.code == 2
    assert not list(tmp_path.iterdir())


def valid_frames(marker=None):
    frames = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "fixture-call",
                                "type": "function",
                                "function": {
                                    "name": probe.TOOL_NAME,
                                    "arguments": '{"value":"ok"}',
                                },
                            }
                        ]
                    },
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 4}},
    ]
    if marker is not None:
        frames.append({"choices": [], probe.MARKER: marker})
    return frames


@pytest.fixture
def native_server():
    servers = []

    def start(
        *, frames=None, loaded=True, redirect=False, keepalive=False, quiet=False
    ):
        state = {
            "posts": [],
            "gets": [],
            "closed": False,
            "headers": [],
            "release": Event(),
        }

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                state["gets"].append(self.path)
                payloads = {
                    "/health": {
                        "status": "ok",
                        "service": "strata",
                        "model": "fixture-model",
                        "loaded": loaded,
                    },
                    "/props": {
                        "build_info": "Strata fixture engine build",
                        "model_alias": "fixture-model",
                        "is_sleeping": False,
                        "model_path": "/fixture/private/model.gguf",
                    },
                    "/v1/models": {
                        "data": [{"id": "fixture-model", "status": {"value": "loaded"}}]
                    },
                }
                wire = json.dumps(payloads[self.path]).encode()
                self.send_response(302 if redirect else 200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(wire)))
                self.send_header("Connection", "close")
                self.send_header("Location", "/models/load")
                self.end_headers()
                self.wfile.write(wire)
                self.close_connection = True

            def do_POST(self):
                assert self.path == "/v1/chat/completions"
                state["posts"].append(
                    json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                )
                state["headers"].append(self.headers.get("Authorization"))
                selected = (
                    valid_frames()
                    if frames is None
                    else frames(len(state["posts"]))
                    if callable(frames)
                    else frames
                )
                wire = (
                    b"".join(
                        b"data: " + json.dumps(frame).encode() + b"\n\n"
                        for frame in selected
                    )
                    + b"data: [DONE]\n\n"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                if not keepalive and not quiet:
                    self.send_header("Content-Length", str(len(wire)))
                    self.send_header("Connection", "close")
                self.end_headers()
                if not quiet:
                    self.wfile.write(wire)
                    self.wfile.flush()
                if keepalive or quiet:
                    state["release"].wait(timeout=3)
                self.close_connection = True

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01))
        thread.start()
        servers.append((server, thread, state))
        return f"http://127.0.0.1:{server.server_port}", state

    yield start
    for server, thread, state in servers:
        state["release"].set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        state["closed"] = not thread.is_alive()


def test_two_native_posts_prove_automatic_recovery_without_executing_tools(
    native_server, tmp_path
):
    origin, state = native_server(frames=valid_frames(marker=[]))
    args = probe.parse_args(["--allow-live", *cli_args(tmp_path, origin)])
    status, report = probe.run(args)
    assert status == 0, report
    native, injected = report["cases"]
    assert native["recovery"]["corrections_used"] == 0
    assert injected["recovery"]["corrections_used"] == 1
    assert [attempt["index"] for attempt in injected["recovery"]["attempts"]] == [0, 1]
    assert [attempt["outcome"] for attempt in injected["recovery"]["attempts"]] == [
        "rejected",
        "accepted",
    ]
    assert injected["sdk_posts"] == 2
    assert (
        native["sdk_posts"] == native["native_posts"] == injected["native_posts"] == 1
    )
    assert injected["event_kinds"][0] == "tool_recovery"
    assert native["rejection_source"] == "native_if_observed"
    assert injected["rejection_source"] == "loopback_injected"
    assert all(
        case["accepted_generation_source"] == "native" for case in report["cases"]
    )
    assert all(case["tool_execution_count"] == 0 for case in report["cases"])
    assert all(
        case["relay_closed"] and case["adapter_closed"] for case in report["cases"]
    )
    assert report["active_native_responses"] == 0
    assert report["closed"] is True
    assert report["native_http_counts"]["POST"] == len(state["posts"]) == 2
    assert set(state["gets"]) == probe.GET_PATHS
    first, corrected = state["posts"]
    assert {key: value for key, value in first.items() if key != "messages"} == {
        key: value for key, value in corrected.items() if key != "messages"
    }
    assert corrected["messages"][: len(first["messages"])] == first["messages"]
    assert len(corrected["messages"]) == len(first["messages"]) + 2
    assert corrected["tools"] == first["tools"] == probe.TOOLS
    assert corrected["strata_mcp"] is first["strata_mcp"] is False
    assert report["native_rejection_emission_proven"] is False
    assert report["before"]["frontend_version"] is None
    text = args.output.read_text()
    assert json.loads(text) == report
    for private in [origin, "fixture-model", "/fixture/private", probe.PROMPT]:
        assert private not in text
    assert '"arguments"' not in text


def test_native_marker_summary_never_exports_arbitrary_provider_strings():
    raw = [{"name": "/home/operator/private credential", "reason": "undeclared"}]
    summary = probe.marker_summary(raw)
    assert summary["reported_count"] == 1
    assert summary["contract_shape_valid"] is True
    assert summary["raw_value_sha256"] == probe.digest(raw)
    assert "private" not in json.dumps(summary)
    assert (
        probe.marker_summary("arbitrary provider body")["contract_shape_valid"] is False
    )


def test_native_done_finishes_without_waiting_for_http_eof(native_server, tmp_path):
    origin, state = native_server(frames=valid_frames(marker=[]), keepalive=True)
    args = probe.parse_args(
        ["--allow-live", *cli_args(tmp_path, origin), "--timeout", "1"]
    )
    started = time.monotonic()
    status, report = probe.run(args)
    elapsed = time.monotonic() - started
    assert status == 0, report
    assert elapsed < 2
    assert not state["release"].is_set()
    assert len(state["posts"]) == 2
    assert report["active_native_responses"] == 0
    assert report["closed"] is True
    assert all(case["active_relay_handlers"] == 0 for case in report["cases"])
    assert all(case["relay_closed"] for case in report["cases"])


def test_silent_native_timeout_drains_handler_before_export(native_server, tmp_path):
    origin, state = native_server(quiet=True)
    args = probe.parse_args(
        ["--allow-live", *cli_args(tmp_path, origin), "--timeout", "0.15"]
    )
    status, report = probe.run(args)
    assert status == 1
    assert len(state["posts"]) == 1
    assert report["active_native_responses"] == 0
    assert report["closed"] is True
    case = report["cases"][0]
    assert case["result"] == "failed"
    assert case["active_relay_handlers"] == 0
    assert case["relay_closed"] and case["adapter_closed"]
    assert case["recovery"]["corrections_used"] == 0


def test_unsuccessful_native_correction_cannot_forward_a_third_post(
    native_server, tmp_path
):
    marker = [{"name": "undeclared_fixture", "reason": "undeclared"}]
    origin, state = native_server(
        frames=lambda number: valid_frames(marker=marker if number == 2 else None)
    )
    args = probe.parse_args(["--allow-live", *cli_args(tmp_path, origin)])
    status, report = probe.run(args)
    assert status == 1
    assert len(state["posts"]) == report["native_http_counts"]["POST"] == 2
    assert report["cases"][0]["result"] == "passed"
    injected = report["cases"][1]
    assert injected["result"] == "failed"
    assert injected["sdk_posts"] == 3
    assert injected["native_posts"] == 1
    assert injected["recovery"]["corrections_used"] == 2
    assert injected["recovery"]["exhausted"] is False
    assert injected["native_diagnostics"] == [probe.marker_summary(marker)]
    assert injected["relay_closed"] and injected["adapter_closed"]
    assert report["active_native_responses"] == 0


def test_native_rejection_cannot_spend_more_than_one_post_in_direct_case(
    native_server, tmp_path, monkeypatch
):
    secret = "fixture-credential-long-enough"
    monkeypatch.setenv("TOOL_PROBE_FIXTURE_KEY", secret)
    marker = [{"name": "/home/operator/private " + secret, "reason": "undeclared"}]
    origin, state = native_server(frames=valid_frames(marker=marker))
    args = probe.parse_args(
        [
            "--allow-live",
            *cli_args(tmp_path, origin),
            "--api-key-env",
            "TOOL_PROBE_FIXTURE_KEY",
        ]
    )
    status, report = probe.run(args)
    assert status == 1
    assert report["result"] == "failed"
    assert len(state["posts"]) == 1
    assert report["native_http_counts"]["POST"] == 1
    assert report["cases"][0]["sdk_posts"] == 2
    assert report["cases"][0]["recovery"]["corrections_used"] == 1
    assert report["native_rejection_emission_proven"] is True
    assert report["cases"][0]["native_diagnostics"] == [probe.marker_summary(marker)]
    assert state["headers"] == ["Bearer " + secret]
    text = args.output.read_text()
    assert secret not in text
    assert "/home/operator" not in text
    assert report["active_native_responses"] == 0
    assert report["cases"][0]["relay_closed"] is True


@pytest.mark.parametrize("mode", ["unloaded", "redirect"])
def test_metadata_failure_never_submits_inference(native_server, tmp_path, mode):
    origin, state = native_server(
        loaded=mode != "unloaded", redirect=mode == "redirect"
    )
    args = probe.parse_args(["--allow-live", *cli_args(tmp_path, origin)])
    status, report = probe.run(args)
    assert status == 1
    assert report["cases"] == []
    assert state["posts"] == []
    assert not report["native_http_counts"].get("POST")
    assert set(state["gets"]) <= probe.GET_PATHS
    assert report["closed"] is True


def test_method_allowlist_and_global_native_budget_refuse_before_transport(tmp_path):
    args = probe.parse_args(["--allow-live", *cli_args(tmp_path)])
    upstream = probe.Upstream(args)
    try:
        for method, path in [
            ("GET", "/load"),
            ("POST", "/models/load"),
            ("DELETE", "/health"),
        ]:
            with pytest.raises(ValueError, match="endpoint_not_permitted"):
                with upstream.request(method, path):
                    pytest.fail("must not reach transport")
        upstream.counts["POST"] = probe.MAX_NATIVE_POSTS
        with pytest.raises(ValueError, match="request_budget_exhausted"):
            with upstream.request("POST", "/v1/chat/completions"):
                pytest.fail("must not reach transport")
        assert upstream.active == 0
    finally:
        upstream.close()


def test_relay_rejects_changed_tool_authority_before_forwarding(tmp_path):
    args = probe.parse_args(["--allow-live", *cli_args(tmp_path)])
    upstream = probe.Upstream(args)
    relay = probe.Relay(upstream, inject=True)
    body = {
        "model": args.model,
        "tools": deepcopy(probe.TOOLS),
        "tool_choice": "required",
        "strata_mcp": False,
        "stream": True,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "messages": [{"role": "user", "content": probe.PROMPT}],
    }
    try:
        assert relay.check_body(body) == 1
        changed = deepcopy(body)
        changed["strata_mcp"] = True
        with pytest.raises(ValueError, match="server_mcp_not_disabled"):
            relay.check_body(changed)
        changed = deepcopy(body)
        changed["tools"][0]["function"]["name"] = "different"
        with pytest.raises(ValueError, match="tools_changed"):
            relay.check_body(changed)
        assert relay.forwarded == 0
        assert upstream.counts["POST"] == 0
    finally:
        upstream.close()
