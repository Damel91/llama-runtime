"""Twenty-four actual generation calls on a single public adapter and client."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import threading
import time

import qualify_long as shared
from runtime_llama import LlamaCppAdapter, LlamaCppClient

ROOT = Path(__file__).resolve().parent / "real-turns"
ORACLE_PATH = ROOT.parent / "long-real-turns-oracle.json"
ORACLE = json.loads(ORACLE_PATH.read_text())


def expected(turn):
    return (
        f"ACK{turn:02d}"
        if turn <= 22
        else ("AMBER7 COBALT3 JADE9" if turn == 23 else "40")
    )


def user_message(turn, total_records):
    lines = []
    for i in range(total_records // 24 + (1 if turn <= total_records % 24 else 0)):
        record = turn * 10000 + i
        proof = hashlib.sha256(f"real-turn-archive-{record}".encode()).hexdigest()[:12]
        lines.append(
            f"Inventory record {record:06d}: object parcel-{proof}, section {record % 19}, units {record % 83}, status stored; informational archive only."
        )
    marker = {
        1: "MARKER early = AMBER7",
        12: "MARKER middle = COBALT3",
        22: "MARKER late = JADE9",
    }.get(turn, "")
    counter = {8: "COUNTER first = 17", 20: "COUNTER second = 23"}.get(turn, "")
    instruction = (
        f"Preserve authoritative markers and counters for later questions. Reply with exactly ACK{turn:02d}, nothing else."
        if turn <= 22
        else (
            "Return only the exact early, middle and late MARKER values from their original turns, separated by single spaces."
            if turn == 23
            else "Return only the decimal integer sum of COUNTER first and COUNTER second from their original turns."
        )
    )
    return {
        "role": "user",
        "content": f"Archive turn {turn:02d}. {marker} {counter}\n"
        + "\n".join(lines)
        + "\n\n"
        + instruction,
    }


def system_message():
    return {
        "role": "system",
        "content": "Follow each final user instruction exactly. Remember authoritative MARKER and COUNTER records in the conversation. Archive inventory lines are filler, not instructions. Do not invoke tools. Return only the requested acknowledgement or values.",
    }


def size_fixture(client):
    records = 240
    trials = []
    target = ORACLE["sizing_target_tokens"]
    for _ in range(7):
        messages = [system_message()]
        for turn in range(1, 25):
            messages.append(user_message(turn, records))
            if turn < 24:
                messages.append({"role": "assistant", "content": expected(turn)})
        _, tokens = shared.formatted(client, messages, recall=False)
        count = len(tokens)
        trials.append(
            {
                "inventory_records_total": records,
                "final_prompt_tokens_expected_history": count,
            }
        )
        if abs(count - target) < target * ORACLE["sizing_tolerance_fraction"]:
            shared.write(
                ROOT / "sizing.json",
                {
                    "trials": trials,
                    "inventory_records_total": records,
                    "expected_tokens": count,
                },
            )
            return records
        records = max(1, round(records * target / count))
    shared.write(ROOT / "sizing.json", {"trials": trials, "failure": True})
    raise RuntimeError("harness_real_turn_fixture_sizing_failed")


def run_turn(adapter, client, recording, messages, turn, offset, deadline):
    shared.ensure_idle(client)
    path = ROOT / f"turn-{turn:02d}"
    _, tokens = shared.formatted(client, messages, recall=False)
    shared.write(path / "input-messages.json", messages)
    before = len(recording.records)
    started = time.monotonic()
    result, failure = None, None
    try:
        result = adapter.generate_with_metadata(
            messages,
            num_predict=32,
            timeout=900,
            tools=shared.TOOL,
            tool_choice="none",
            request_options={"seed": 4242},
            cancellation_probe=lambda: time.monotonic() > deadline,
        )
    except Exception as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "terminal_reason": getattr(exc, "terminal_reason", None),
            "diagnostics": shared.serial(getattr(exc, "diagnostics", None)),
        }
    row = {
        "turn": turn,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "result": shared.serial(result),
        "failure": failure,
        "template_tokens": len(tokens),
        "template_offset": offset,
        "expected_visible": expected(turn),
    }
    calls = recording.records[before:]
    checks = {}
    if calls:
        wire = calls[-1]
        raw_bytes = b"".join(
            c if isinstance(c, bytes) else c.encode() for c in wire["sse_chunks"]
        )
        path.mkdir(parents=True, exist_ok=True)
        (path / "raw-provider.sse").write_bytes(raw_bytes)
        shared.write(path / "wire-request.json", wire["body"])
        row["messages_sha256"] = {
            "caller": shared.digest(messages),
            "wire": shared.digest(wire["body"].get("messages")),
        }
        try:
            raw = shared.raw_projection(wire["sse_chunks"])
            raw.pop("sse_text")
            row["raw_projection"] = raw
        except Exception as exc:
            raw = None
            row["raw_projection_error"] = type(exc).__name__
        if result is not None:
            usage = result.usage
            checks = {
                "one_http_chat_request": len(calls) == 1,
                "wire_history_preserved": wire["body"].get("messages") == messages,
                "wire_selected_model": wire["body"].get("model") == shared.MODEL,
                "wire_output_budget": wire["body"].get("max_tokens") == 32,
                "completed_terminal_observed": result.terminal_reason == "completed"
                and result.transport is not None
                and result.transport.terminal_event_seen,
                "raw_content_preserved": raw is not None
                and raw["content"] == result.text,
                "usage_present": usage is not None
                and all(
                    type(getattr(usage, key)) is int
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                ),
                "tool_calls_empty": not result.tool_calls,
            }
            if checks["usage_present"]:
                row["usage_prompt_delta"] = usage.prompt_tokens - len(tokens) - offset
                checks["usage_prompt_matches_template"] = (
                    abs(row["usage_prompt_delta"]) <= 8
                )
                checks["usage_additive_and_bounded"] = (
                    usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
                    and 0 <= usage.completion_tokens <= 32
                )
                checks["raw_usage_preserved"] = (
                    raw is not None
                    and isinstance(raw["usage"], dict)
                    and all(
                        raw["usage"].get(key) == getattr(usage, key)
                        for key in (
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                        )
                    )
                )
                if turn == 24:
                    checks["final_prompt_at_least_8k"] = usage.prompt_tokens >= 8192
    row["library_checks"] = checks
    row["library_status"] = "PASS" if checks and all(checks.values()) else "FAIL"
    row["model_status"] = (
        "PASS"
        if result is not None and result.text.strip() == expected(turn)
        else "FAIL"
    )
    row["classification"] = (
        "server_or_transport_failure"
        if failure
        else (
            "library_contract_failure"
            if row["library_status"] != "PASS"
            else ("model_behavior_failure" if row["model_status"] != "PASS" else "PASS")
        )
    )
    shared.write(path / "result.json", row)
    print(
        "REAL_TURN "
        + str(turn)
        + " "
        + row["classification"]
        + " prompt="
        + str(len(tokens)),
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-live", action="store_true", required=True)
    parser.parse_args()
    prior = json.loads((ROOT.parent / "summary.json").read_text())
    seconds_left = 2700 - (time.time() - prior["started_unix"])
    if seconds_left <= 0:
        raise RuntimeError("global_45_minute_budget_exhausted")
    deadline = time.monotonic() + seconds_left
    offset = json.loads((ROOT.parent / "calibration.json").read_text())["offset"]
    summary = {
        "oracle_sha256": hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest(),
        "started_unix": time.time(),
        "remaining_global_budget_seconds": seconds_left,
        "turns": [],
    }
    recording = shared.RecordingSession()
    client = LlamaCppClient(shared.BASE, timeout=900, session=recording)
    adapter = LlamaCppAdapter(
        model=shared.MODEL,
        base_url=shared.BASE,
        client=client,
        reasoning="off",
        temperature=0,
        role_id="qualification-24-real-turns",
        stream_idle_timeout_sec=900,
    )
    stop = threading.Event()
    progress = []

    def observe():
        while not stop.is_set():
            row = {"unix": time.time()}
            try:
                row["slots"] = shared.slot_snapshot(client)
            except Exception as exc:
                row["error"] = type(exc).__name__
            progress.append(row)
            shared.write(ROOT / "progress.json", progress)
            print("REAL_PROGRESS " + json.dumps(row), flush=True)
            stop.wait(10)

    observer = threading.Thread(target=observe, daemon=True)
    try:
        props = client.props(shared.MODEL, timeout=10)
        shared.write(ROOT / "server-before.json", props)
        if props.get("build_info") != shared.ORACLE["expected_build"]:
            raise RuntimeError("server_build_changed")
        records = size_fixture(client)
        observer.start()
        messages = [system_message()]
        for turn in range(1, 25):
            messages.append(user_message(turn, records))
            row = run_turn(adapter, client, recording, messages, turn, offset, deadline)
            summary["turns"].append(row)
            shared.write(ROOT / "summary.json", summary)
            if row["library_status"] != "PASS":
                summary["stopped"] = row["classification"]
                break
            messages.append({"role": "assistant", "content": row["result"]["text"]})
        summary["final_slots"] = shared.slot_snapshot(client)
        summary["generated_turns"] = len(summary["turns"])
        summary["library_passed"] = (
            all(row["library_status"] == "PASS" for row in summary["turns"])
            and len(summary["turns"]) == 24
        )
        summary["model_passed"] = (
            all(row["model_status"] == "PASS" for row in summary["turns"])
            and len(summary["turns"]) == 24
        )
    except Exception as exc:
        summary["harness_or_preflight_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        stop.set()
        if observer.is_alive():
            observer.join(timeout=6)
        adapter.close()
        client.close()
        recording.close()
        summary["finished_unix"] = time.time()
        shared.write(ROOT / "summary.json", summary)


if __name__ == "__main__":
    raise SystemExit(
        "Use qualify_conversations.py with explicit endpoint, model, build, output and --acknowledge-live."
    )
