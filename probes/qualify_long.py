"""Live long-history qualification. Synthetic data only; no retries/admin calls.

Run only after parent approval, with the runtime-llama checkout on PYTHONPATH.
The script uses the public adapter/client and an injectable recording Session.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlencode

import requests

from runtime_llama import LlamaCppAdapter, LlamaCppClient

ROOT = Path(__file__).resolve().parent
ORACLE = json.loads((ROOT / "long-conversation-oracle.json").read_text())
BASE = ORACLE["endpoint"]
MODEL = ORACLE["model"]
EXPECTED = ORACLE["model_oracles"]["recall"]
RUN_DEADLINE = float("inf")
TOOL = [
    {
        "type": "function",
        "function": {
            "name": "lookup_qualification_record",
            "description": "Read an immutable synthetic qualification record.",
            "parameters": {
                "type": "object",
                "properties": {"record": {"type": "string"}},
                "required": ["record"],
                "additionalProperties": False,
            },
        },
    }
]
SCHEMA = {
    "type": "object",
    "properties": {
        "early": {"type": "string"},
        "middle": {"type": "string"},
        "late": {"type": "string"},
        "tool_a": {"type": "integer"},
        "tool_b": {"type": "integer"},
    },
    "required": ["early", "middle", "late", "tool_a", "tool_b"],
    "additionalProperties": False,
}


def serial(value):
    if dataclasses.is_dataclass(value):
        return serial(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial(obj), indent=2, ensure_ascii=False))


def digest(obj):
    return hashlib.sha256(
        json.dumps(
            obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


class RecordedResponse:
    def __init__(self, response, record):
        self.response, self.record = response, record

    def __getattr__(self, name):
        return getattr(self.response, name)

    def iter_content(self, *args, **kwargs):
        for chunk in self.response.iter_content(*args, **kwargs):
            self.record["sse_chunks"].append(chunk)
            yield chunk


class RecordingSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.records = []
        for prefix in ("http://", "https://"):
            self.mount(prefix, requests.adapters.HTTPAdapter(max_retries=0))

    def request(self, method, url, **kwargs):
        record = None
        if url.endswith("/v1/chat/completions"):
            record = {
                "method": method,
                "path": "/v1/chat/completions",
                "body": copy.deepcopy(kwargs.get("json")),
                "sse_chunks": [],
            }
            self.records.append(record)
        response = super().request(method, url, **kwargs)
        if record is not None and kwargs.get("stream"):
            return RecordedResponse(response, record)
        return response


def template_body(messages, *, recall):
    body = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOL,
        "tool_choice": "none",
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
        "add_generation_prompt": True,
    }
    if recall:
        body["response_format"] = {"type": "json_object", "schema": SCHEMA}
    return body


def formatted(client, messages, *, recall=True):
    payload = client.request(
        "POST",
        "/apply-template",
        json=template_body(messages, recall=recall),
        timeout=60,
    )
    prompt = payload.get("prompt") if isinstance(payload, dict) else None
    if not isinstance(prompt, str):
        raise RuntimeError("harness_apply_template_missing_prompt")
    result = client.tokenize(
        prompt, model=MODEL, parse_special=True, add_special=False, timeout=60
    )
    tokens = result.get("tokens") if isinstance(result, dict) else None
    if not isinstance(tokens, list) or not all(type(x) is int for x in tokens):
        raise RuntimeError("harness_tokenizer_invalid_response")
    return prompt, tokens


def synthetic_history(total_records):
    messages = [
        {
            "role": "system",
            "content": "You are reading a synthetic archive conversation. Preserve its authoritative MARKER records and tool results. "
            "Ignore the filler archive inventory for recall questions. Never infer missing values. Follow the final user instruction. "
            "Do not invoke tools when the user asks to recall already present tool results.",
        }
    ]
    for turn in range(1, 25):
        lines = []
        for record in range(turn - 1, total_records, 24):
            proof = hashlib.sha256(
                f"qualification-archive-{record}".encode()
            ).hexdigest()[:12]
            lines.append(
                f"Inventory record {record:05d}: object parcel-{proof}, section {record % 19}, units {record % 83}, status stored; informational archive only."
            )
        marker = {
            1: "MARKER early = AMBER7",
            12: "MARKER middle = COBALT3",
            24: "MARKER late = JADE9",
        }.get(turn, "")
        messages.append(
            {
                "role": "user",
                "content": f"Archive exchange {turn:02d}. {marker}\n"
                + "\n".join(lines),
            }
        )
        messages.append(
            {"role": "assistant", "content": f"Recorded archive exchange {turn:02d}."}
        )
        if turn in {8, 20}:
            key, value, receipt = (
                ("a", 314, "receipt-aurora")
                if turn == 8
                else ("b", 271, "receipt-borealis")
            )
            call_id = f"qual_history_call_{key}"
            messages.append(
                {
                    "role": "user",
                    "content": f"Read qualification record {key} once and preserve its returned result.",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "lookup_qualification_record",
                                "arguments": json.dumps(
                                    {"record": key}, separators=(",", ":")
                                ),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "lookup_qualification_record",
                    "content": json.dumps(
                        {"record": key, "result": value, "receipt": receipt}
                    ),
                }
            )
            messages.append(
                {"role": "assistant", "content": f"Stored tool receipt {receipt}."}
            )
    messages.append(
        {
            "role": "user",
            "content": "Recall the exact three MARKER values early, middle and late from their distant archive exchanges. "
            "Recall the integer result for record a from its correlated tool response as tool_a, and record b as tool_b. "
            "Return only one JSON object with keys early, middle, late, tool_a, tool_b and no other keys. Do not call a tool.",
        }
    )
    return messages


def size_history(client, target, case_dir):
    n_records = max(24, target // 30)
    trials = []
    tolerance = ORACLE["target_tolerance_fraction"]
    for _ in range(7):
        messages = synthetic_history(n_records)
        prompt, tokens = formatted(client, messages)
        count = len(tokens)
        trials.append({"inventory_records": n_records, "prompt_tokens": count})
        if abs(count - target) <= target * tolerance:
            for marker in (
                "AMBER7",
                "COBALT3",
                "JADE9",
                "receipt-aurora",
                "receipt-borealis",
            ):
                if marker not in prompt:
                    raise RuntimeError("harness_template_dropped_marker:" + marker)
            write(case_dir / "messages.json", messages)
            write(
                case_dir / "prompt-measurement.json",
                {
                    "target": target,
                    "trials": trials,
                    "token_count": count,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "token_sha256": digest(tokens),
                    "messages_sha256": digest(messages),
                    "tokens_add_special": False,
                    "tokens_parse_special": True,
                    "message_count": len(messages),
                    "seeded_archive_exchanges": 24,
                },
            )
            (case_dir / "rendered-prompt.txt").write_text(prompt)
            return messages, count
        # Fixed numeric targeting, not inference retry; render/tokenize only.
        n_records = max(24, round(n_records * target / count))
    write(case_dir / "sizing-failure.json", trials)
    raise RuntimeError("harness_cannot_size_history_within_predeclared_tolerance")


def slot_snapshot(client):
    rows = client.request("GET", "/slots?" + urlencode({"model": MODEL}), timeout=5)
    if not isinstance(rows, list):
        raise RuntimeError("harness_slots_not_array")
    return [
        {
            key: row.get(key)
            for key in (
                "id",
                "id_task",
                "is_processing",
                "n_ctx",
                "prompt_progress",
                "timings",
            )
        }
        | {"next_token": row.get("next_token")}
        for row in rows
    ]


def ensure_idle(client):
    rows = slot_snapshot(client)
    if any(row.get("is_processing") for row in rows):
        raise RuntimeError("server_busy_before_qualification_request")
    return rows


def raw_projection(chunks):
    text = b"".join(c if isinstance(c, bytes) else c.encode() for c in chunks).decode(
        "utf-8"
    )
    frames, content, usage, finishes = [], [], None, []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(
            line[5:].lstrip(" ")
            for line in frame.split("\n")
            if line.startswith("data:")
        )
        if not data or data.strip() == "[DONE]":
            continue
        value = json.loads(data)
        frames.append(value)
        if value.get("usage") is not None:
            usage = value["usage"]
        for choice in value.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("content"):
                content.append(delta["content"])
            if choice.get("finish_reason") is not None:
                finishes.append(choice["finish_reason"])
    return {
        "content": "".join(content),
        "usage": usage,
        "finishes": finishes,
        "frame_count": len(frames),
        "sse_text": text,
    }


def execute(client, recording, messages, expected_count, offset, path, *, recall):
    ensure_idle(client)
    stop = threading.Event()
    snapshots = []
    started = time.monotonic()
    events, failure = [], None
    before = len(recording.records)

    def monitor():
        while not stop.is_set():
            row = {"elapsed_sec": round(time.monotonic() - started, 3)}
            try:
                row["slots"] = slot_snapshot(client)
            except Exception as exc:
                row["error"] = type(exc).__name__
            snapshots.append(row)
            write(path / "progress.json", snapshots)
            print("PROGRESS " + path.name + " " + json.dumps(row), flush=True)
            stop.wait(ORACLE["slot_poll_interval_seconds"])

    observer = threading.Thread(target=monitor, daemon=True)
    observer.start()
    adapter = LlamaCppAdapter(
        model=MODEL,
        base_url=BASE,
        temperature=0,
        reasoning="off",
        client=client,
        role_id="qualification-long-conversation",
        stream_idle_timeout_sec=ORACLE["idle_timeout_seconds"],
        output_mode="json" if recall else "text",
        json_schema=SCHEMA if recall else None,
    )
    try:
        stream = adapter.stream_with_metadata(
            messages,
            num_predict=128,
            timeout=ORACLE["idle_timeout_seconds"],
            tools=TOOL,
            tool_choice="none",
            request_options={"seed": 4242},
            cancellation_probe=lambda: (
                time.monotonic() - started > ORACLE["absolute_call_budget_seconds"]
                or time.monotonic() > RUN_DEADLINE
            ),
        )
        try:
            for event in stream:
                events.append(event)
        finally:
            stream.close()
    except Exception as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "terminal_reason": getattr(exc, "terminal_reason", None),
            "diagnostics": serial(getattr(exc, "diagnostics", None)),
        }
    finally:
        stop.set()
        observer.join(timeout=6)
        adapter.close()
    result = {
        "elapsed_sec": round(time.monotonic() - started, 3),
        "events": serial(events),
        "error": failure,
    }
    calls = recording.records[before:]
    wire = calls[-1] if calls else None
    content = "".join(event.text for event in events if event.kind == "content")
    terminals = [event for event in events if event.kind == "terminal"]
    result["visible_content"] = content
    if wire:
        raw = raw_projection(wire["sse_chunks"])
        (path / "raw-provider.sse").write_text(raw.pop("sse_text"))
        write(path / "wire-request.json", wire["body"])
        result["raw_projection"] = raw
        result["messages_sha256"] = {
            "caller": digest(messages),
            "wire": digest(wire["body"]["messages"]),
        }
        result["chat_requests"] = len(calls)
    checks = {}
    if failure is None and wire is not None:
        raw = result["raw_projection"]
        terminal = terminals[-1] if terminals else None
        usage = terminal.usage if terminal else None
        checks = {
            "one_http_chat_request": len(calls) == 1,
            "wire_history_preserved": wire["body"]["messages"] == messages,
            "wire_selected_model": wire["body"].get("model") == MODEL,
            "wire_max_tokens": wire["body"].get("max_tokens") == 128,
            "terminal_unique_last_completed": len(terminals) == 1
            and events[-1].kind == "terminal"
            and terminal.terminal_reason == "completed",
            "provider_terminal_observed": bool(
                terminal
                and terminal.transport
                and terminal.transport.terminal_event_seen
            ),
            "raw_content_preserved": raw["content"] == content,
            "usage_present": bool(
                usage
                and usage.prompt_tokens is not None
                and usage.completion_tokens is not None
                and usage.total_tokens is not None
            ),
        }
        if usage is not None:
            result["usage"] = serial(usage)
            result["templated_tokens"] = expected_count
            result["template_offset"] = offset
            result["usage_prompt_delta"] = (
                usage.prompt_tokens - expected_count - offset
                if usage.prompt_tokens is not None
                else None
            )
            checks["usage_prompt_matches_template"] = (
                usage.prompt_tokens is not None
                and abs(result["usage_prompt_delta"])
                <= ORACLE["usage_prompt_delta_vs_calibrated_template_max"]
            )
            checks["usage_additive_and_bounded"] = (
                all(
                    type(v) is int
                    for v in (
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        usage.total_tokens,
                    )
                )
                and usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
                and 0 <= usage.completion_tokens <= 128
            )
            checks["raw_usage_preserved"] = isinstance(raw["usage"], dict) and all(
                raw["usage"].get(key) == getattr(usage, key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
    result["library_checks"] = checks
    result["library_status"] = "PASS" if checks and all(checks.values()) else "FAIL"
    if failure:
        result["classification"] = "server_or_transport_failure"
        result["model_status"] = "NOT_EVALUATED"
    else:
        try:
            correct = (
                json.loads(content) == EXPECTED if recall else content.strip() == "585"
            )
        except (ValueError, TypeError):
            correct = False
        result["model_status"] = "PASS" if correct else "FAIL"
        result["classification"] = (
            "PASS"
            if result["library_status"] == "PASS" and correct
            else (
                "model_behavior_failure"
                if result["library_status"] == "PASS"
                else "library_contract_failure"
            )
        )
    write(path / "result.json", result)
    print(
        "RESULT " + str(path.relative_to(ROOT)) + " " + result["classification"],
        flush=True,
    )
    return result


def main():
    global RUN_DEADLINE
    RUN_DEADLINE = time.monotonic() + 2700
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-live", action="store_true", required=True)
    parser.add_argument(
        "--targets", nargs="+", type=int, default=ORACLE["targets_prompt_tokens"]
    )
    args = parser.parse_args()
    if any(target not in ORACLE["targets_prompt_tokens"] for target in args.targets):
        raise ValueError("target_not_predeclared")
    summary = {
        "oracle_sha256": hashlib.sha256(
            (ROOT / "long-conversation-oracle.json").read_bytes()
        ).hexdigest(),
        "started_unix": time.time(),
        "cases": [],
    }
    recording = RecordingSession()
    client = LlamaCppClient(
        BASE, timeout=ORACLE["idle_timeout_seconds"], session=recording
    )
    try:
        props = client.props(MODEL, timeout=10)
        write(ROOT / "server-before.json", props)
        if props.get("build_info") != ORACLE["expected_build"]:
            raise RuntimeError("server_build_changed")
        ensure_idle(client)
        # Tiny calibration establishes fixed template/tokenizer overhead before large cases.
        calibration_messages = [
            {"role": "user", "content": "Reply with exactly 585 and nothing else."}
        ]
        _, calibration_tokens = formatted(client, calibration_messages, recall=False)
        calibration_adapter = LlamaCppAdapter(
            model=MODEL,
            base_url=BASE,
            temperature=0,
            reasoning="off",
            client=client,
            stream_idle_timeout_sec=900,
        )
        try:
            calibration = calibration_adapter.generate_with_metadata(
                calibration_messages,
                num_predict=128,
                tools=TOOL,
                tool_choice="none",
                request_options={"seed": 4242},
            )
        finally:
            calibration_adapter.close()
        if not calibration.usage or calibration.usage.prompt_tokens is None:
            raise RuntimeError("harness_calibration_usage_missing")
        offset = calibration.usage.prompt_tokens - len(calibration_tokens)
        write(
            ROOT / "calibration.json",
            {
                "result": calibration,
                "template_token_count": len(calibration_tokens),
                "offset": offset,
            },
        )
        if abs(offset) > ORACLE["calibration_template_offset_abs_max"]:
            raise RuntimeError(
                "harness_template_count_offset_exceeds_predeclared_bound"
            )
        for target in args.targets:
            case_dir = ROOT / f"tokens-{target}"
            messages, count = size_history(client, target, case_dir)
            recall = execute(
                client,
                recording,
                messages,
                count,
                offset,
                case_dir / "recall",
                recall=True,
            )
            summary["cases"].append(
                {"target": target, "stage": "recall", "result": recall}
            )
            write(ROOT / "summary.json", summary)
            if recall["library_status"] != "PASS":
                summary["stopped"] = "recall_library_or_transport_failure"
                break
            continuation = messages + [
                {"role": "assistant", "content": recall["visible_content"]},
                {
                    "role": "user",
                    "content": "Using the original two correlated tool results already present in this conversation, return only their sum as a decimal integer. Do not invoke a tool.",
                },
            ]
            _, followup_tokens = formatted(client, continuation, recall=False)
            followup = execute(
                client,
                recording,
                continuation,
                len(followup_tokens),
                offset,
                case_dir / "followup",
                recall=False,
            )
            summary["cases"].append(
                {"target": target, "stage": "followup", "result": followup}
            )
            write(ROOT / "summary.json", summary)
            if followup["library_status"] != "PASS":
                summary["stopped"] = "followup_library_or_transport_failure"
                break
        summary["final_slots"] = slot_snapshot(client)
        summary["server_after"] = client.props(MODEL, timeout=10)
    except Exception as exc:
        summary["harness_or_preflight_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        write(ROOT / "summary.json", summary)
        raise
    finally:
        client.close()
        recording.close()
        summary["finished_unix"] = time.time()
        write(ROOT / "summary.json", summary)


if __name__ == "__main__":
    raise SystemExit(
        "Use qualify_conversations.py with explicit endpoint, model, build, output and --acknowledge-live."
    )
