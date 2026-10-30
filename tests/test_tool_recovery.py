"""SDK recovery through public adapters, without executing any proposed tool."""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Barrier, Event, Lock, Thread
from types import SimpleNamespace

import pytest

from runtime_llama import AdapterProfile
from runtime_llama.adapters.automatic import AutoRuntimeAdapter
from runtime_llama.adapters.llama import LlamaCppAdapter
from runtime_llama.adapters.strata import StrataAdapter
from runtime_llama.client import LlamaCppClient
from runtime_llama.errors import (
    LLMBackendError,
    LLMCancelledError,
    LLMRunawayError,
    LLMTimeoutError,
    LLMToolRecoveryError,
)
from test_generation import _Client, _chunk


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pi_write",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }
]


def rejected(*, reasoning=None, marker=None):
    return [
        *([_chunk(reasoning=reasoning)] if reasoning else []),
        _chunk(content="<tool_call><function=write>not executed</tool_call>"),
        _chunk(finish="stop"),
        {
            "choices": [],
            "strata_rejected_tool_calls": (
                [{"name": "write", "reason": "undeclared"}]
                if marker is None
                else marker
            ),
        },
    ]


def valid(
    *,
    name="pi_write",
    arguments='{"path":"probe"}',
    call_id="call_valid",
    reasoning=None,
):
    return [
        *([_chunk(reasoning=reasoning)] if reasoning else []),
        _chunk(
            tools=[
                {
                    "index": 0,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ]
        ),
        _chunk(finish="tool_calls"),
    ]


class SequenceClient(_Client):
    def __init__(self, attempts):
        super().__init__()
        self.attempts = deque(attempts)
        self.lock = Lock()

    def chat_completion(
        self, messages, model=None, stream=False, cancel=None, *, timeout=None, **kwargs
    ):
        with self.lock:
            payloads = self.attempts.popleft()
            self.calls.append(
                {
                    "messages": deepcopy(messages),
                    "model": model,
                    "stream": stream,
                    "cancel": cancel,
                    "timeout": timeout,
                    **deepcopy(kwargs),
                }
            )

        def events():
            try:
                for payload in payloads:
                    if callable(payload):
                        payload = payload()
                    if isinstance(payload, BaseException):
                        raise payload
                    if payload is not None:
                        yield deepcopy(payload)
            finally:
                with self.lock:
                    self.streams_closed += 1

        return events()


@pytest.mark.parametrize("adapter_type", [LlamaCppAdapter, StrataAdapter])
def test_native_rejection_corrects_without_host_and_preserves_evidence(adapter_type):
    client = SequenceClient([rejected(), valid()])
    adapter = adapter_type(model="shared", client=client)
    result = adapter.generate_with_metadata("write the probe", tools=TOOLS)
    assert result.text == ""
    assert result.tool_calls[0]["function"]["name"] == "pi_write"
    assert result.tool_recovery.corrections_used == 1
    assert [a.outcome for a in result.tool_recovery.attempts] == [
        "rejected",
        "accepted",
    ]
    assert result.tool_recovery.attempts[0].native_rejections[0].name == "write"
    assert len(client.calls) == client.streams_closed == 2
    assert client.calls[1]["messages"][0] == client.calls[0]["messages"][0]
    assert client.calls[1]["tools"] == client.calls[0]["tools"] == TOOLS


def test_budget_is_three_corrections_and_exhaustion_does_not_poison_next_call():
    client = SequenceClient([*([rejected()] * 4), valid()])
    adapter = LlamaCppAdapter(model="shared", client=client)
    with pytest.raises(LLMToolRecoveryError) as caught:
        adapter.generate_with_metadata("first", tools=TOOLS)
    error = caught.value
    assert error.terminal_reason == "tool_recovery_exhausted"
    assert error.tool_recovery.exhausted
    assert [a.index for a in error.tool_recovery.attempts] == [0, 1, 2, 3]
    assert len(client.calls) == client.streams_closed == 4
    assert "<function=write>" in error.last_proposal["content"]
    assert "<function=write>" not in repr(error)
    result = adapter.generate_with_metadata("next external call", tools=TOOLS)
    assert result.tool_recovery.corrections_used == 0
    assert len(result.tool_recovery.attempts) == 1
    assert len(client.calls) == 5


def test_success_restores_full_budget_for_later_invocations():
    client = SequenceClient(
        [rejected(), valid(), rejected(), rejected(), rejected(), valid()]
    )
    adapter = LlamaCppAdapter(model="shared", client=client)
    first = adapter.generate_with_metadata("one", tools=TOOLS)
    second = adapter.generate_with_metadata("two", tools=TOOLS)
    assert first.tool_recovery.corrections_used == 1
    assert second.tool_recovery.corrections_used == 3
    assert [a.index for a in second.tool_recovery.attempts] == [0, 1, 2, 3]
    assert len(client.calls) == 6


def test_mixed_proposals_never_escape_before_atomic_correction():
    first = [*valid(call_id="not_dispatched")[:-1], *rejected()]
    client = SequenceClient([first, valid(call_id="accepted_once")])
    adapter = LlamaCppAdapter(model="shared", client=client)
    events = list(adapter.stream_with_metadata("write", tools=TOOLS))
    deltas = [e.data["tool_calls"] for e in events if e.kind == "tool_call_delta"]
    assert len(deltas) == 1 and deltas[0][0]["id"] == "accepted_once"
    assert not any(e.text for e in events if e.kind == "content")
    assert events[-1].data["tool_calls"][0]["id"] == "accepted_once"
    assert events[-1].tool_recovery.corrections_used == 1
    assert [e.kind for e in events].count("terminal") == 1


@pytest.mark.parametrize(
    "text",
    [
        "Use <tool_call> to document the format.",
        "```xml\n<tool_call><function=write>example</tool_call>\n```",
        "`<function=write>` is quoted code, not a request.",
    ],
)
def test_without_proof_normal_xml_and_code_are_unchanged(text):
    client = SequenceClient([[_chunk(content=text), _chunk(finish="stop")]])
    result = LlamaCppAdapter(model="shared", client=client).generate_with_metadata(
        "explain", tools=TOOLS
    )
    assert result.text == text
    assert result.tool_recovery.corrections_used == 0
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "marker", ["rejected", [{"name": "write", "reason": "unknown"}], [5]]
)
def test_malformed_native_marker_fails_without_regeneration(marker):
    client = SequenceClient([rejected(marker=marker), valid()])
    with pytest.raises(LLMToolRecoveryError) as caught:
        LlamaCppAdapter(model="shared", client=client).generate_with_metadata(
            "write", tools=TOOLS
        )
    assert caught.value.terminal_reason == "tool_protocol_error"
    assert not caught.value.tool_recovery.exhausted
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "marker", [[{"name": "write", "reason": "undeclared"}], "malformed"]
)
def test_explicit_native_rejection_without_declared_tools_is_not_success(marker):
    client = SequenceClient([rejected(marker=marker), valid()])
    adapter = LlamaCppAdapter(model="shared", client=client)
    with pytest.raises(LLMToolRecoveryError) as caught:
        adapter.generate_with_metadata("write")
    assert caught.value.terminal_reason == "tool_protocol_error"
    assert (
        caught.value.tool_recovery.attempts[0].reasons[0]
        == "no_declared_tool_authority"
    )
    assert not caught.value.tool_recovery.exhausted
    assert len(client.calls) == client.streams_closed == 1


def test_no_tools_empty_native_marker_keeps_progressive_streaming():
    client = SequenceClient(
        [
            [
                _chunk(content="ordinary prose"),
                {"choices": [], "strata_rejected_tool_calls": []},
                _chunk(finish="stop"),
            ]
        ]
    )
    adapter = LlamaCppAdapter(model="shared", client=client)
    stream = adapter.stream_with_metadata("explain")
    assert next(stream).text == "ordinary prose"
    assert client.streams_closed == 0
    rest = list(stream)
    assert rest[-1].terminal_reason == "completed"
    assert rest[-1].data["strata_rejected_tool_calls"] == []
    assert rest[-1].tool_recovery is None


@pytest.mark.parametrize(
    "bad",
    [
        valid(name="write"),
        valid(arguments="not JSON"),
        valid(arguments='{"path":7}'),
        [_chunk(content="done"), _chunk(finish="tool_calls")],
    ],
)
def test_complete_structured_protocol_errors_are_corrected(bad):
    client = SequenceClient([bad, valid()])
    result = LlamaCppAdapter(model="shared", client=client).generate_with_metadata(
        "write", tools=TOOLS
    )
    assert result.tool_recovery.corrections_used == 1
    assert result.tool_calls[0]["function"]["arguments"] == '{"path":"probe"}'


def test_required_choice_is_a_proven_obligation_without_native_marker():
    client = SequenceClient(
        [[_chunk(content="I will do it"), _chunk(finish="stop")], valid()]
    )
    result = LlamaCppAdapter(model="shared", client=client).generate_with_metadata(
        "write", tools=TOOLS, tool_choice="required"
    )
    assert result.tool_recovery.attempts[0].reasons == ("required_tool_call_missing",)
    assert result.tool_recovery.corrections_used == 1


@pytest.mark.parametrize("where", ["extra_body", "request_options", "activity"])
def test_server_effects_never_trigger_corrective_post(where):
    first = rejected()
    if where == "activity":
        first.insert(0, {"choices": [], "strata_mcp": {"event": "call"}})
    client = SequenceClient([first, valid()])
    adapter = LlamaCppAdapter(
        model="shared",
        client=client,
        **({"extra_body": {"strata_mcp": True}} if where == "extra_body" else {}),
    )
    with pytest.raises(LLMToolRecoveryError) as caught:
        adapter.generate_with_metadata(
            "write",
            tools=TOOLS,
            **(
                {"request_options": {"strata_mcp": True}}
                if where == "request_options"
                else {}
            ),
        )
    assert (
        "server_side_tool_effects_possible"
        in caught.value.tool_recovery.attempts[0].reasons
    )
    assert len(client.calls) == 1


def test_effective_body_and_caller_inputs_are_frozen_through_correction():
    history = [{"role": "user", "content": "write"}]
    tools = deepcopy(TOOLS)
    options = {"seed": 17, "tools": tools}
    client = SequenceClient([])
    adapter = LlamaCppAdapter(model="shared", client=client, extra_body={"top_k": 13})

    def mutate():
        history[0]["content"] = "mutated externally"
        tools[0]["function"]["name"] = "other"
        options["seed"] = 99
        adapter._generation_config.extra_body["top_k"] = 71

    client.attempts.extend([[mutate, *rejected()], valid()])
    result = adapter.generate_with_metadata(history, request_options=options)
    assert result.tool_recovery.corrections_used == 1
    for request in client.calls:
        assert request["messages"][0]["content"] == "write"
        assert request["tools"][0]["function"]["name"] == "pi_write"
        assert request["seed"] == 17 and request["top_k"] == 13


def test_rejected_and_accepted_usage_are_not_conflated():
    first = rejected() + [
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    ]
    second = valid() + [
        {"choices": [], "usage": {"prompt_tokens": 14, "completion_tokens": 3}}
    ]
    result = LlamaCppAdapter(
        model="shared", client=SequenceClient([first, second])
    ).generate_with_metadata("write", tools=TOOLS)
    assert result.usage.prompt_tokens == 14
    assert result.tool_recovery.total_usage.prompt_tokens == 24
    assert result.tool_recovery.total_usage.completion_tokens == 5
    assert result.tool_recovery.elapsed_ms >= 0


def test_thinking_capture_finalizes_each_rejected_attempt_and_accepted_once():
    sink = []
    client = SequenceClient(
        [rejected(reasoning="failed plan"), valid(reasoning="accepted plan")]
    )
    result = LlamaCppAdapter(model="shared", client=client).generate_with_metadata(
        "write",
        tools=TOOLS,
        thinking_diagnostics={"enabled": True},
        thinking_sink=sink.append,
    )
    assert [d.text for d in sink] == ["failed plan", "accepted plan"]
    assert result.tool_recovery.attempts[0].thinking_diagnostic.text == "failed plan"
    assert result.reasoning_text == "accepted plan"


def test_exhaustion_does_not_replace_thinking_with_empty_outer_capture():
    sink = []
    client = SequenceClient([rejected(reasoning="failed plan")] * 4)
    with pytest.raises(LLMToolRecoveryError) as caught:
        LlamaCppAdapter(model="shared", client=client).generate_with_metadata(
            "write",
            tools=TOOLS,
            thinking_diagnostics={"enabled": True},
            thinking_sink=sink.append,
        )
    assert len(sink) == 4
    assert caught.value.thinking_diagnostic.text == "failed plan"


def test_output_cap_is_terminal_partial_text_without_candidate_tools():
    client = SequenceClient(
        [[*valid()[:-1], _chunk(content="abcdefgh"), _chunk(finish="stop")], valid()]
    )
    result = LlamaCppAdapter(model="shared", client=client).generate_with_metadata(
        "write", tools=TOOLS, num_predict=1
    )
    assert result.text == "abcd" and result.terminal_reason == "visible_output_cap"
    assert result.tool_calls == ()
    assert result.tool_recovery.attempts[0].outcome == "visible_output_cap"
    assert len(client.calls) == client.streams_closed == 1


@pytest.mark.parametrize("local_cap", [True, False])
def test_incomplete_attempt_keeps_sanitized_native_proof_without_raw_diagnostic(
    local_cap,
):
    secret = "test-secret-key"
    client = SequenceClient(
        [
            [
                {
                    "choices": [],
                    "strata_rejected_tool_calls": [
                        {"name": secret, "reason": "undeclared"}
                    ],
                },
                _chunk(content="abcdefgh"),
                _chunk(finish="length"),
            ]
        ]
    )
    adapter = LlamaCppAdapter(model="shared", client=client, api_key=secret)
    events = list(
        adapter.stream_with_metadata(
            "write", tools=TOOLS, num_predict=1 if local_cap else None
        )
    )
    assert not any(event.kind == "tool_diagnostic" for event in events)
    attempt = events[-1].tool_recovery.attempts[0]
    assert attempt.native_rejected_tool_calls == 1
    assert attempt.native_rejections[0].name == "<credential_redacted>"
    assert "native_rejected_tool_call" in attempt.reasons
    assert events[-1].data["tool_calls"] == []
    assert events[-1].terminal_reason == (
        "visible_output_cap" if local_cap else "length_cap"
    )
    assert len(client.calls) == client.streams_closed == 1


@pytest.mark.parametrize(
    "failure",
    [
        LLMBackendError("http_failure"),
        LLMTimeoutError("timed out"),
        LLMCancelledError(),
    ],
)
def test_transport_errors_are_not_blindly_retried(failure):
    client = SequenceClient([[_chunk(reasoning="partial"), failure], valid()])
    with pytest.raises(type(failure)) as caught:
        LlamaCppAdapter(model="shared", client=client).generate_with_metadata(
            "write", tools=TOOLS
        )
    assert len(client.calls) == client.streams_closed == 1
    assert len(caught.value.tool_recovery.attempts) == 1


def test_early_eof_preserves_typed_transport_failure_without_retry():
    client = SequenceClient([[_chunk(content="unfinished proposal")], valid()])
    with pytest.raises(LLMBackendError) as caught:
        LlamaCppAdapter(model="shared", client=client).generate_with_metadata(
            "write", tools=TOOLS
        )
    assert caught.value.diagnostics.terminal_reason == "early_eof"
    assert len(caught.value.tool_recovery.attempts) == 1
    assert len(client.calls) == client.streams_closed == 1


@pytest.mark.parametrize(
    "finish,reason",
    [
        ("length", "length_cap"),
        ("max_tokens", "length_cap"),
        ("content_filter", "content_filter"),
    ],
)
def test_native_incomplete_terminal_does_not_release_or_retry_candidate_tools(
    finish, reason
):
    client = SequenceClient(
        [
            [_chunk(content="partial text"), *valid()[:-1], _chunk(finish=finish)],
            valid(),
        ]
    )
    result = LlamaCppAdapter(model="shared", client=client).generate_with_metadata(
        "write", tools=TOOLS
    )
    assert result.text == "partial text" and result.terminal_reason == reason
    assert result.tool_calls == ()
    assert result.tool_recovery.corrections_used == 0
    assert len(client.calls) == client.streams_closed == 1


def test_reentrant_sink_invocation_has_its_own_budget_and_history():
    client = SequenceClient(
        [
            rejected(reasoning="failed plan"),
            valid(call_id="nested"),
            valid(call_id="outer"),
        ]
    )
    adapter = LlamaCppAdapter(model="shared", client=client)
    nested = []

    def sink(_diagnostic):
        if not nested:
            nested.append(adapter.generate_with_metadata("nested", tools=TOOLS))

    result = adapter.generate_with_metadata(
        "outer", tools=TOOLS, thinking_diagnostics={"enabled": True}, thinking_sink=sink
    )
    assert result.tool_recovery.corrections_used == 1
    assert nested[0].tool_recovery.corrections_used == 0
    assert result.tool_calls[0]["id"] == "outer"
    assert nested[0].tool_calls[0]["id"] == "nested"
    assert [request["messages"][0]["content"] for request in client.calls] == [
        "outer",
        "nested",
        "outer",
    ]


def test_cancel_at_recovery_boundary_prevents_another_post():
    stop = Event()
    client = SequenceClient([rejected(), valid()])
    stream = LlamaCppAdapter(model="shared", client=client).stream_with_metadata(
        "write", tools=TOOLS, cancellation_probe=stop.is_set
    )
    assert next(stream).kind == "tool_recovery"
    stop.set()
    with pytest.raises(LLMCancelledError):
        next(stream)
    assert len(client.calls) == client.streams_closed == 1


def test_abandonment_at_recovery_boundary_prevents_another_post():
    client = SequenceClient([rejected(), valid()])
    stream = LlamaCppAdapter(model="shared", client=client).stream_with_metadata(
        "write", tools=TOOLS
    )
    assert next(stream).kind == "tool_recovery"
    stream.close()
    assert len(client.calls) == client.streams_closed == 1


def test_close_after_first_accepted_staged_event_prevents_late_tool_output():
    client = SequenceClient([valid(reasoning="accepted reasoning")])
    adapter = LlamaCppAdapter(model="shared", client=client)
    stream = adapter.stream_with_metadata("write", tools=TOOLS)
    assert next(stream).kind == "reasoning"
    assert client.streams_closed == 1
    adapter.close()
    with pytest.raises(LLMBackendError, match="adapter_closed"):
        next(stream)
    assert len(client.calls) == 1


@pytest.mark.parametrize("projection", ["metadata", "blocking", "text"])
@pytest.mark.parametrize("close_after", [4, 5])
def test_reentrant_false_probe_close_retires_staged_public_projection(
    projection, close_after
):
    client = SequenceClient([valid(reasoning="accepted reasoning")])
    adapter = LlamaCppAdapter(model="shared", client=client)
    replay_checks = 0
    escaped = []

    def close_during_replay():
        nonlocal replay_checks
        if client.streams_closed:
            replay_checks += 1
            if replay_checks == close_after:
                adapter.close()
        return False

    with pytest.raises(LLMBackendError, match="adapter_closed"):
        if projection == "metadata":
            escaped.extend(
                adapter.stream_with_metadata(
                    "write", tools=TOOLS, cancellation_probe=close_during_replay
                )
            )
        elif projection == "blocking":
            adapter.invoke_with_ctx(
                "write",
                request_options={"tools": TOOLS},
                cancellation_probe=close_during_replay,
            )
        else:
            escaped.extend(
                adapter.stream_with_ctx(
                    "write",
                    request_options={"tools": TOOLS},
                    cancellation_probe=close_during_replay,
                )
            )
    assert replay_checks == close_after
    assert not any(getattr(event, "kind", "") == "tool_call_delta" for event in escaped)
    assert len(client.calls) == client.streams_closed == 1


@pytest.mark.parametrize("projection", ["metadata", "blocking", "text"])
def test_accepted_thinking_sink_close_cannot_return_staged_success(projection):
    client = SequenceClient([valid(reasoning="accepted reasoning")])
    adapter = LlamaCppAdapter(model="shared", client=client)
    sink = []

    def close_after_capture(diagnostic):
        sink.append(diagnostic)
        adapter.close()

    options = {
        "thinking_diagnostics": {"enabled": True},
        "thinking_sink": close_after_capture,
    }
    with pytest.raises(LLMBackendError, match="adapter_closed"):
        if projection == "metadata":
            list(adapter.stream_with_metadata("write", tools=TOOLS, **options))
        elif projection == "blocking":
            adapter.invoke_with_ctx(
                "write", request_options={"tools": TOOLS}, **options
            )
        else:
            list(
                adapter.stream_with_ctx(
                    "write", request_options={"tools": TOOLS}, **options
                )
            )
    assert len(sink) == 1
    assert len(client.calls) == client.streams_closed == 1


def test_thinking_sink_reentrant_close_prevents_corrective_post():
    client = SequenceClient([rejected(reasoning="rejected thinking"), valid()])
    adapter = LlamaCppAdapter(model="shared", client=client)
    sink = []

    def close_after_capture(diagnostic):
        sink.append(diagnostic)
        adapter.close()

    with pytest.raises(LLMBackendError, match="adapter_closed"):
        adapter.generate_with_metadata(
            "write",
            tools=TOOLS,
            thinking_diagnostics={"enabled": True},
            thinking_sink=close_after_capture,
        )
    assert len(client.calls) == client.streams_closed == 1
    assert len(sink) == 1


def test_async_bridge_uses_the_same_recovery_without_async_host_retry():
    client = SequenceClient([rejected(), valid()])
    adapter = LlamaCppAdapter(model="shared", client=client)

    async def run():
        return await asyncio.to_thread(
            adapter.generate_with_metadata, "write", tools=TOOLS
        )

    assert asyncio.run(run()).tool_recovery.corrections_used == 1


def test_runaway_guard_remains_active_before_staged_content_is_exposed():
    sink = []
    client = SequenceClient(
        [
            [_chunk(content="<think>hidden</think>abababab"), _chunk(finish="stop")],
            valid(),
        ]
    )
    adapter = LlamaCppAdapter(model="shared", client=client)
    adapter._role_profile = SimpleNamespace(
        runaway_min_chars=4,
        runaway_window_chars=16,
        runaway_block_chars=2,
        runaway_repeat_threshold=2,
        runaway_ngram_chars=2,
    )
    with pytest.raises(LLMRunawayError) as caught:
        list(
            adapter.stream_with_metadata(
                "write",
                tools=TOOLS,
                thinking_diagnostics={"enabled": True},
                thinking_sink=sink.append,
            )
        )
    assert len(client.calls) == client.streams_closed == 1
    assert caught.value.thinking_diagnostic.text == "hidden"
    assert len(sink) == 1


def test_concurrent_invocations_do_not_share_or_poison_budgets():
    barrier = Barrier(2)

    class PerCallerClient(_Client):
        def __init__(self):
            super().__init__()
            self.counts = {}
            self.lock = Lock()

        def chat_completion(self, messages, **kwargs):
            label = messages[0]["content"]
            with self.lock:
                index = self.counts.get(label, 0)
                self.counts[label] = index + 1
                self.calls.append({"messages": deepcopy(messages), **kwargs})
            if index == 0:
                barrier.wait(timeout=2)
            payloads = rejected() if label == "exhaust" or index < 2 else valid()

            def events():
                try:
                    yield from deepcopy(payloads)
                finally:
                    with self.lock:
                        self.streams_closed += 1

            return events()

    client = PerCallerClient()
    adapter = LlamaCppAdapter(model="shared", client=client)
    with ThreadPoolExecutor(2) as workers:
        failed = workers.submit(adapter.generate_with_metadata, "exhaust", tools=TOOLS)
        successful = workers.submit(
            adapter.generate_with_metadata, "succeed", tools=TOOLS
        )
        with pytest.raises(LLMToolRecoveryError) as caught:
            failed.result(timeout=3)
        result = successful.result(timeout=3)
    assert caught.value.tool_recovery.corrections_used == 3
    assert result.tool_recovery.corrections_used == 2
    assert client.counts == {"exhaust": 4, "succeed": 3}
    assert client.streams_closed == 7


@pytest.fixture
def recovery_server():
    servers = []

    def start(attempts, *, hold_second=False):
        state = {"posts": [], "gets": [], "entered": Event(), "release": Event()}
        lock = Lock()
        pending = deque(attempts)

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                state["gets"].append(self.path)
                payloads = {
                    "/health": {
                        "status": "ok",
                        "service": "strata",
                        "model": "shared",
                        "loaded": True,
                    },
                    "/props": {
                        "build_info": "Strata 0.1.54",
                        "model_alias": "shared",
                        "is_sleeping": False,
                        "default_generation_settings": {"n_ctx": 4096},
                        "total_slots": 1,
                    },
                    "/v1/models": {
                        "data": [
                            {
                                "id": "shared",
                                "status": {"value": "loaded"},
                                "meta": {"n_ctx": 4096},
                            }
                        ]
                    },
                }
                wire = json.dumps(payloads[self.path]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)
                self.close_connection = True

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with lock:
                    state["posts"].append(body)
                    payloads = pending.popleft()
                    number = len(state["posts"])
                if hold_second and number == 2:
                    state["entered"].set()
                    state["release"].wait(3)
                wire = (
                    "".join("data: " + json.dumps(item) + "\n\n" for item in payloads)
                    + "data: [DONE]\n\n"
                ).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(wire)))
                    self.end_headers()
                    self.wfile.write(wire)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
        )
        thread.start()
        servers.append((server, thread, state))
        return f"http://127.0.0.1:{server.server_port}", state

    yield start
    for server, thread, state in servers:
        state["release"].set()
        server.shutdown()
        server.server_close()
        thread.join(1)


def test_actual_http_200_rejection_is_recovered_and_route_stays_pinned(recovery_server):
    endpoint, state = recovery_server([rejected(), valid()])
    profile = AdapterProfile.from_config(
        "test", {"model": "shared", "runtime_endpoints": [{"base_url": endpoint}]}
    )
    adapter = AutoRuntimeAdapter(profile)
    try:
        result = adapter.generate_with_metadata("write", tools=TOOLS)
        assert result.runtime_selection.backend == "strata"
        assert result.tool_recovery.corrections_used == 1
        assert len(state["gets"]) == 3 and len(state["posts"]) == 2
        assert {r["model"] for r in state["posts"]} == {"shared"}
        assert not adapter._runtime_operations
    finally:
        adapter.close()


@pytest.mark.parametrize("projection", ["metadata", "blocking", "text"])
@pytest.mark.parametrize("close_after", [5, 6, 7])
def test_automatic_false_probe_close_cannot_replay_tool_calls(
    recovery_server, projection, close_after
):
    endpoint, state = recovery_server([valid(reasoning="accepted reasoning")])
    profile = AdapterProfile.from_config(
        "test", {"model": "shared", "runtime_endpoints": [{"base_url": endpoint}]}
    )
    adapter = AutoRuntimeAdapter(profile)
    replay_checks = 0
    escaped = []

    def close_during_replay():
        nonlocal replay_checks
        with adapter._runtime_lock:
            clients = [
                operation.client
                for operation in adapter._runtime_operations
                if operation.client is not None
            ]
        if (
            state["posts"]
            and clients
            and not any(client._streams for client in clients)
        ):
            replay_checks += 1
            if replay_checks == close_after:
                adapter.close()
        return False

    try:
        with pytest.raises(LLMCancelledError):
            if projection == "metadata":
                escaped.extend(
                    adapter.stream_with_metadata(
                        "write", tools=TOOLS, cancellation_probe=close_during_replay
                    )
                )
            elif projection == "blocking":
                adapter.invoke_with_ctx(
                    "write",
                    request_options={"tools": TOOLS},
                    cancellation_probe=close_during_replay,
                )
            else:
                escaped.extend(
                    adapter.stream_with_ctx(
                        "write",
                        request_options={"tools": TOOLS},
                        cancellation_probe=close_during_replay,
                    )
                )
        assert replay_checks == close_after
        assert not any(
            getattr(event, "kind", "") == "tool_call_delta" for event in escaped
        )
        assert len(state["gets"]) == 3 and len(state["posts"]) == 1
        assert not adapter._runtime_operations
    finally:
        adapter.close()


@pytest.mark.parametrize("action", ["cancel", "close", "timeout"])
def test_prefill_interruption_during_correction_drains_without_third_post(
    recovery_server, action
):
    endpoint, state = recovery_server([rejected(), valid(), valid()], hold_second=True)
    stop = Event()
    adapter = StrataAdapter(
        model="shared",
        base_url=endpoint,
        stream_idle_timeout_sec=0.1 if action == "timeout" else None,
    )
    try:
        with ThreadPoolExecutor(2) as workers:
            pending = workers.submit(
                adapter.generate_with_metadata,
                "write",
                tools=TOOLS,
                cancellation_probe=stop.is_set,
            )
            assert state["entered"].wait(2)
            if action == "cancel":
                stop.set()
            elif action == "close":
                adapter.close()
            expected = (
                LLMTimeoutError
                if action == "timeout"
                else (LLMBackendError, LLMCancelledError)
                if action == "close"
                else LLMCancelledError
            )
            with pytest.raises(expected) as caught:
                pending.result(timeout=2)
            assert caught.value.tool_recovery.corrections_used == 1
        assert len(state["posts"]) == 2
        assert not adapter._client._streams
    finally:
        state["release"].set()
        adapter.close()


def test_borrowed_client_close_interrupts_only_its_silent_recovery(recovery_server):
    endpoint, state = recovery_server([rejected(), valid(), valid()], hold_second=True)
    client = LlamaCppClient(endpoint, timeout=1, trust_env=False)
    first = StrataAdapter(
        model="shared", base_url=endpoint, client=client, stream_idle_timeout_sec=None
    )
    sibling = StrataAdapter(
        model="shared", base_url=endpoint, client=client, stream_idle_timeout_sec=None
    )
    try:
        with ThreadPoolExecutor(2) as workers:
            pending = workers.submit(first.generate_with_metadata, "first", tools=TOOLS)
            assert state["entered"].wait(2)
            first.close()
            with pytest.raises(LLMCancelledError):
                pending.result(timeout=1)
            result = workers.submit(
                sibling.generate_with_metadata, "second", tools=TOOLS
            ).result(timeout=1)
        assert result.tool_recovery.corrections_used == 0
        assert result.tool_calls[0]["function"]["name"] == "pi_write"
        assert len(state["posts"]) == 3
        assert not client._closed and not client._streams
    finally:
        state["release"].set()
        first.close()
        sibling.close()
        client.close()
