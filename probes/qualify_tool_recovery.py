#!/usr/bin/env python3
"""Opt-in, two-native-POST qualification of SDK tool recovery on loaded Strata.

One benign native proposal is followed by a separate invocation whose initial
rejection is injected over loopback HTTP. Only its SDK corrective POST reaches
the native server. Tools are data: this probe has no tool executor. It neither
loads models nor changes consumers, server configuration or runtime processes.
Use an installed candidate; no checkout path is added to Python's import path.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
import math
import os
from pathlib import Path
from threading import Condition, Lock, Thread
import time
from urllib.parse import urlsplit


TOOL_NAME = "qualification_echo"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Return harmless qualification data; never executed.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string", "const": "ok"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }
]
PROMPT = 'Propose one qualification_echo call with arguments {"value":"ok"}.'
REJECTED_CONTENT = "SYNTHETIC_REJECTED_PROPOSAL_NOT_DISPATCHED"
MARKER = "strata_rejected_tool_calls"
GET_PATHS = frozenset({"/health", "/props", "/v1/models"})
MAX_BYTES = 1_048_576
MAX_NATIVE_POSTS = 2  # Probe scope only; never changes the SDK recovery budget.


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def marker_summary(value):
    """Retain wire evidence without exporting native names or arbitrary strings."""
    return {
        "present": True,
        "json_type": type(value).__name__,
        "reported_count": len(value) if isinstance(value, list) else None,
        "contract_shape_valid": isinstance(value, list)
        and all(
            isinstance(item, dict)
            and set(item) == {"name", "reason"}
            and isinstance(item["name"], str)
            and item["reason"] == "undeclared"
            for item in value
        ),
        "raw_value_sha256": digest(value),
    }


def validate_url(value):
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.port != 0
            and parsed.username is None
            and parsed.password is None
            and parsed.path in {"", "/"}
            and "?" not in value
            and "#" not in value
            and "\\" not in value
            and not any(char.isspace() or ord(char) < 32 for char in value)
        )
    except ValueError:
        valid = False
    if not valid:
        raise argparse.ArgumentTypeError("base URL must be an HTTP(S) origin")
    return value.rstrip("/")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--base-url", required=True, type=validate_url)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api-key-env")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    if not args.allow_live:
        parser.error("live requests require explicit --allow-live")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 120:
        parser.error("--timeout must be finite and in (0, 120]")
    if not 1 <= args.max_tokens <= 256:
        parser.error("--max-tokens must be in [1, 256]")
    if not args.model.strip() or any(ord(char) < 32 for char in args.model):
        parser.error("--model must be nonempty and contain no control characters")
    if args.api_key_env and not os.environ.get(args.api_key_env):
        parser.error("--api-key-env must name a nonempty environment variable")
    return args


class Upstream:
    """One explicit origin, no redirects/proxies/retries, finite probe resources."""

    def __init__(self, args):
        import requests

        self.args = args
        self.session = requests.Session()
        self.session.trust_env = False
        if args.api_key_env:
            self.session.headers["Authorization"] = (
                "Bearer " + os.environ[args.api_key_env]
            )
        self.deadline = time.monotonic() + args.timeout * 4
        self.lock = Lock()
        self.drained = Condition(self.lock)
        self.counts = Counter()
        self.active = 0
        self.closed = False
        self.closing = False
        self.responses = set()
        self.canonical_model = None

    def remaining(self):
        remaining = min(self.args.timeout, self.deadline - time.monotonic())
        require(remaining > 0, "probe_wall_budget_exhausted")
        return remaining

    @contextmanager
    def request(self, method, path, *, body=None):
        require(
            (method == "GET" and path in GET_PATHS)
            or (method == "POST" and path == "/v1/chat/completions"),
            "probe_endpoint_not_permitted",
        )
        with self.lock:
            require(not self.closing, "probe_upstream_closed")
            limit = MAX_NATIVE_POSTS if method == "POST" else 18
            require(
                self.counts[method] < limit, "probe_native_request_budget_exhausted"
            )
            self.counts[method] += 1
            self.active += 1
        response = None
        try:
            response = self.session.request(
                method,
                self.args.base_url + path,
                json=body,
                timeout=self.remaining(),
                allow_redirects=False,
                stream=True,
            )
            with self.lock:
                self.responses.add(response)
            require(response.status_code == 200, "native_http_status_not_200")
            yield response
        finally:
            if response is not None:
                response.close()
            with self.drained:
                self.responses.discard(response)
                self.active -= 1
                self.drained.notify_all()

    def get(self, path):
        with self.request("GET", path) as response:
            data = bytearray()
            for part in response.iter_content(8192):
                self.remaining()
                data.extend(part)
                require(len(data) <= MAX_BYTES, "native_metadata_too_large")
            return json.loads(data)

    def close(self):
        with self.lock:
            self.closing = True
            responses = tuple(self.responses)
        for response in responses:
            response.close()
        self.session.close()
        with self.drained:
            self.closed = self.drained.wait_for(
                lambda: self.active == 0, timeout=self.args.timeout + 1
            )


def snapshot(upstream):
    health, props, models = [upstream.get(path) for path in GET_PATHS_ORDERED]
    require(isinstance(health, dict), "native_health_not_object")
    require(isinstance(props, dict), "native_props_not_object")
    require(isinstance(models, dict), "native_models_not_object")
    require(
        health.get("status") == "ok"
        and health.get("service") == "strata"
        and health.get("loaded") is True,
        "loaded_strata_not_verified",
    )
    rows = models.get("data")
    require(isinstance(rows, list), "native_model_catalog_invalid")
    canonical_model = health.get("model")
    selected = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("id") == canonical_model
    ]
    require(len(selected) == 1, "native_loaded_model_not_unique")
    row = selected[0]
    aliases = row.get("aliases", [])
    require(isinstance(aliases, list), "native_aliases_invalid")
    require(
        upstream.args.model in [canonical_model, *aliases],
        "requested_model_not_loaded",
    )
    state = row.get("status")
    require(
        (state.get("value") == "loaded" and state.get("failed") is not True)
        if isinstance(state, dict)
        else state == "loaded",
        "native_catalog_model_not_loaded",
    )
    require(
        props.get("model_alias") == canonical_model
        and props.get("is_sleeping") is not True,
        "native_properties_model_mismatch",
    )
    upstream.canonical_model = canonical_model
    return {
        "runtime": "strata",
        "loaded": True,
        "model_sha256": digest(canonical_model),
        "build_info_sha256": digest(props.get("build_info")),
        "frontend_version": None,
        "frontend_version_source": "not_independently_identified",
    }


GET_PATHS_ORDERED = ("/health", "/props", "/v1/models")


class Relay:
    """Real loopback HTTP/SSE boundary; the SDK alone decides to regenerate."""

    def __init__(self, upstream, *, inject):
        self.upstream, self.inject = upstream, inject
        self.posts = 0
        self.forwarded = 0
        self.first_body = None
        self.diagnostics = []
        self.errors = []
        self.closed = False
        self.active_handlers = 0
        self.lock = Lock()

    def check_body(self, body):
        require(isinstance(body, dict), "probe_request_not_object")
        require(
            body.get("model")
            == (self.upstream.canonical_model or self.upstream.args.model),
            "probe_model_changed",
        )
        require(body.get("tools") == TOOLS, "probe_tools_changed")
        require(body.get("tool_choice") == "required", "probe_tool_choice_changed")
        require(body.get("strata_mcp") is False, "probe_server_mcp_not_disabled")
        require(body.get("stream") is True, "probe_stream_required")
        require(body.get("temperature") == 0, "probe_sampling_changed")
        require(
            body.get("max_tokens") == self.upstream.args.max_tokens,
            "probe_output_budget_changed",
        )
        require(isinstance(body.get("messages"), list), "probe_messages_invalid")
        with self.lock:
            self.posts += 1
            number = self.posts
        require(number <= (2 if self.inject else 1), "probe_case_post_budget_exhausted")
        if number == 1:
            self.first_body = deepcopy(body)
        else:
            original = self.first_body
            require(
                {key: value for key, value in body.items() if key != "messages"}
                == {key: value for key, value in original.items() if key != "messages"},
                "probe_corrective_options_changed",
            )
            history = original["messages"]
            require(
                body["messages"][: len(history)] == history
                and len(body["messages"]) == len(history) + 2
                and [item.get("role") for item in body["messages"][-2:]]
                == ["assistant", "user"],
                "probe_corrective_history_changed",
            )
        return number

    def observe_line(self, line):
        if not line.startswith(b"data:"):
            return
        raw = line[5:].strip()
        if raw == b"[DONE]":
            return
        payload = json.loads(raw)
        if isinstance(payload, dict) and MARKER in payload:
            self.diagnostics.append(marker_summary(payload[MARKER]))

    def __enter__(self):
        relay = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def handle(self):
                with relay.lock:
                    relay.active_handlers += 1
                try:
                    super().handle()
                finally:
                    with relay.lock:
                        relay.active_handlers -= 1

            def setup(self):
                super().setup()
                self.connection.settimeout(relay.upstream.remaining())

            def send_json(self, value, status=200):
                wire = canonical(value)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(wire)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(wire)
                self.close_connection = True

            def do_GET(self):
                try:
                    require(self.path in GET_PATHS, "probe_endpoint_not_permitted")
                    self.send_json(relay.upstream.get(self.path))
                except Exception as exc:
                    relay.errors.append(type(exc).__name__)
                    self.send_json({"error": "probe_metadata_failed"}, 502)

            def do_POST(self):
                started = False
                try:
                    require(
                        self.path == "/v1/chat/completions",
                        "probe_endpoint_not_permitted",
                    )
                    length = int(self.headers.get("Content-Length", "0"))
                    require(0 < length <= MAX_BYTES, "probe_request_size_invalid")
                    body = json.loads(self.rfile.read(length))
                    number = relay.check_body(body)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    started = True
                    if relay.inject and number == 1:
                        chunks = [
                            {
                                "choices": [
                                    {"index": 0, "delta": {"content": REJECTED_CONTENT}}
                                ]
                            },
                            {
                                "choices": [
                                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                                ]
                            },
                            {
                                "choices": [],
                                MARKER: [
                                    {"name": "undeclared_probe", "reason": "undeclared"}
                                ],
                            },
                        ]
                        for chunk in chunks:
                            self.wfile.write(b"data: " + canonical(chunk) + b"\n\n")
                        self.wfile.write(b"data: [DONE]\n\n")
                        return
                    with relay.upstream.request(
                        "POST", self.path, body=body
                    ) as response:
                        relay.forwarded += 1
                        total = 0
                        pending = bytearray()
                        for part in response.iter_content(chunk_size=1):
                            relay.upstream.remaining()
                            total += len(part)
                            require(total <= MAX_BYTES, "native_stream_too_large")
                            pending.extend(part)
                            if pending.endswith(b"\n"):
                                line = bytes(pending[:-1]).rstrip(b"\r")
                                relay.observe_line(line)
                                self.wfile.write(pending)
                                self.wfile.flush()
                                pending.clear()
                                if (
                                    line.startswith(b"data:")
                                    and line[5:].strip() == b"[DONE]"
                                ):
                                    # Native SSE completion is authoritative even
                                    # when its HTTP connection remains open.
                                    self.wfile.write(b"\n")
                                    self.wfile.flush()
                                    break
                        if pending:
                            relay.observe_line(bytes(pending).rstrip(b"\r"))
                except Exception as exc:
                    relay.errors.append(type(exc).__name__)
                    try:
                        if started:
                            self.wfile.write(
                                b'data: {"error":"probe_forward_failed"}\n\n'
                            )
                        else:
                            self.send_json({"error": "probe_request_refused"}, 502)
                    except OSError:
                        pass
                finally:
                    self.close_connection = True

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # ThreadingHTTPServer defaults to daemon request threads. Explicitly
        # join owned handlers before declaring cleanup or exporting counters.
        self.server.daemon_threads = False
        self.thread = Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.05),
            name="tool-recovery-relay",
            daemon=True,
        )
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.closed = not self.thread.is_alive() and self.active_handlers == 0


def report_output(report):
    value = report.to_dict()
    # Transport model labels belong to the operator; hashes above identify them.
    for attempt in value["attempts"]:
        if attempt.get("transport"):
            attempt["transport"].pop("model", None)
    return value


def run_case(args, upstream, *, inject, records):
    from runtime_llama import build_adapter

    record = {
        "case": "injected_rejection_native_correction"
        if inject
        else "native_benign_call",
        "rejection_source": "loopback_injected" if inject else "native_if_observed",
        "accepted_generation_source": "native",
        "tool_execution_count": 0,
    }
    records.append(record)
    relay = Relay(upstream, inject=inject)
    adapter = None
    try:
        with relay as origin:
            adapter = build_adapter(
                {
                    "model": args.model,
                    "runtime_endpoints": [
                        {
                            "name": "qualification",
                            "base_url": origin,
                            **(
                                {"api_key_env": args.api_key_env}
                                if args.api_key_env
                                else {}
                            ),
                        }
                    ],
                    "temperature": 0,
                    "reasoning": "off",
                    "num_predict": args.max_tokens,
                    "stream_idle_timeout_sec": args.timeout,
                    "extra_body": {"strata_mcp": False},
                    **({"api_key_env": args.api_key_env} if args.api_key_env else {}),
                }
            )
            events = list(
                adapter.stream_with_metadata(
                    PROMPT, tools=TOOLS, tool_choice="required"
                )
            )
            require(
                events and events[-1].kind == "terminal", "accepted_terminal_missing"
            )
            terminal = events[-1]
            recovery = terminal.tool_recovery
            record["recovery"] = report_output(recovery) if recovery else None
            record["event_kinds"] = [event.kind for event in events]
            require(
                terminal.runtime_selection.backend == "strata",
                "strata_route_not_selected",
            )
            require(
                not any(REJECTED_CONTENT in event.text for event in events),
                "rejected_content_escaped",
            )
            calls = (terminal.data or {}).get("tool_calls", [])
            require(len(calls) == 1, "accepted_tool_cardinality_invalid")
            function = calls[0].get("function", {})
            require(
                function.get("name") == TOOL_NAME
                and json.loads(function.get("arguments", "null")) == {"value": "ok"},
                "accepted_tool_data_invalid",
            )
            expected = ["rejected", "accepted"] if inject else ["accepted"]
            require(
                recovery is not None
                and [attempt.outcome for attempt in recovery.attempts] == expected
                and recovery.corrections_used == int(inject),
                "recovery_attempts_invalid",
            )
            if inject:
                require(
                    events[0].kind == "tool_recovery"
                    and events[0].data.get("next_attempt") == 1,
                    "automatic_correction_event_missing",
                )
            require(relay.forwarded == 1, "native_generation_count_invalid")
            record.update(result="passed", accepted_tool_calls=1)
    except Exception as exc:
        record.update(result="failed", error_type=type(exc).__name__)
        recovery = getattr(exc, "tool_recovery", None)
        if recovery is not None:
            record["recovery"] = report_output(recovery)
        raise
    finally:
        if adapter is not None:
            adapter.close()
        record.update(
            sdk_posts=relay.posts,
            native_posts=relay.forwarded,
            native_diagnostics=relay.diagnostics,
            relay_error_types=relay.errors,
            relay_closed=relay.closed,
            active_relay_handlers=relay.active_handlers,
            adapter_closed=adapter is not None,
        )


def run(args):
    import runtime_llama

    result = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "distribution_version": importlib.metadata.version("runtime-llama"),
        "runtime_version": runtime_llama.__version__,
        "endpoint_sha256": digest(args.base_url),
        "model_selector_sha256": digest(args.model),
        "scope": "one native benign call and one injected rejection with native correction",
        "native_post_limit": MAX_NATIVE_POSTS,
        "tools_executed": 0,
        "server_administration": False,
        "consumer_changes": False,
        "native_rejection_emission_proven": False,
        "cases": [],
    }
    upstream = Upstream(args)
    status = 1
    try:
        result["before"] = snapshot(upstream)
        for inject in (False, True):
            run_case(args, upstream, inject=inject, records=result["cases"])
        result["result"] = "passed"
        status = 0
    except Exception as exc:
        result.update(result="failed", error_type=type(exc).__name__)
    finally:
        if "before" in result:
            try:
                result["after"] = snapshot(upstream)
                require(result["before"] == result["after"], "native_identity_changed")
            except Exception as exc:
                result.update(result="failed", snapshot_error_type=type(exc).__name__)
                status = 1
        upstream.close()
        if not upstream.closed:
            result.update(result="failed", native_drain_failed=True)
            status = 1
        result.update(
            native_http_counts=dict(upstream.counts),
            active_native_responses=upstream.active,
            closed=upstream.closed,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        result["native_rejection_emission_proven"] = any(
            marker["contract_shape_valid"] and marker["reported_count"] > 0
            for case in result["cases"]
            for marker in case["native_diagnostics"]
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return status, result


def main(argv=None):
    status, result = run(parse_args(argv))
    print(
        json.dumps(
            {
                "result": result["result"],
                "native_posts": result["native_http_counts"].get("POST", 0),
                "closed": result["closed"],
            }
        )
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
