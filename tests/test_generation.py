from __future__ import annotations

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from types import SimpleNamespace

import pytest

from runtime_llama.adapters.llama import LlamaCppAdapter
from runtime_llama.configuration import LlamaServerConfig
from runtime_llama.errors import (
    LLMBackendError,
    LLMCancelledError,
    LLMRunawayError,
    LLMTimeoutError,
)


def _chunk(*, content=None, reasoning=None, tools=None, finish=None, usage=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    if tools is not None:
        delta["tool_calls"] = tools
    payload = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


class _Client:
    def __init__(self, payloads=None):
        self.payloads = list(
            payloads
            if payloads is not None
            else [
                _chunk(content="done"),
                _chunk(finish="stop"),
            ]
        )
        self.calls = []
        self.streams_closed = 0
        self.client_closed = False
        self.health_calls = 0
        self.props_calls = []
        self.models_calls = 0
        self.health_result = {"status": "ok"}
        self.props_result = {"default_generation_settings": {"n_ctx": 4096}}
        self.models_result = {"data": [{"id": "qwen", "status": {"value": "loaded"}}]}

    def chat_completion(
        self, messages, model=None, stream=False, cancel=None, *, timeout=None, **kwargs
    ):
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
                for payload in self.payloads:
                    if isinstance(payload, Exception):
                        raise payload
                    yield deepcopy(payload)
            finally:
                self.streams_closed += 1

        return events()

    def health(self):
        self.health_calls += 1
        if isinstance(self.health_result, Exception):
            raise self.health_result
        return deepcopy(self.health_result)

    def props(self, model=None):
        self.props_calls.append(model)
        if isinstance(self.props_result, Exception):
            raise self.props_result
        return deepcopy(self.props_result)

    def models(self):
        self.models_calls += 1
        if isinstance(self.models_result, Exception):
            raise self.models_result
        return deepcopy(self.models_result)

    def close(self):
        self.client_closed = True


def _adapter(client=None, **kwargs):
    return LlamaCppAdapter(model="qwen", client=client or _Client(), **kwargs)


def test_string_generation_uses_selected_server_without_control_plane():
    client = _Client()
    adapter = _adapter(client, base_url="https://vpn.example/inference")
    result = adapter.generate_with_metadata("hello", timeout=12)
    assert result.text == "done"
    assert result.terminal_reason == "completed"
    assert result.usage is None
    assert client.calls[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert client.calls[0]["model"] == "qwen"
    assert client.calls[0]["stream"] is True
    assert client.calls[0]["timeout"] == 12
    assert client.calls[0]["stream_options"] == {"include_usage": True}
    assert "autoload" not in client.calls[0]
    assert client.health_calls == 0
    assert client.props_calls == []
    assert client.streams_closed == 1


def test_full_history_retains_tool_identifiers_reasoning_and_multimodal_parts():
    client = _Client()
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "prior reasoning",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"id":1}'},
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"value":2}',
            "tool_call_id": "call_a",
            "name": "lookup",
        },
    ]
    original = deepcopy(messages)
    assert _adapter(client).chat(messages) == "done"
    assert client.calls[0]["messages"] == original
    assert messages == original


def test_legacy_message_objects_still_project_basic_fields():
    client = _Client()
    _adapter(client).invoke([SimpleNamespace(type="human", content="hello")])
    assert client.calls[0]["messages"] == [{"role": "user", "content": "hello"}]


def test_sampling_and_budget_are_request_local_without_context_load_fields():
    client = _Client()
    adapter = _adapter(
        client,
        num_ctx=2048,
        num_predict=5,
        temperature=0.3,
        top_k=40,
        top_p=0.9,
        min_p=0.05,
        repeat_penalty=1.1,
    )
    adapter.generate(
        "hello", num_predict=20, request_options={"temperature": 0.8, "mirostat": 2}
    )
    call = client.calls[0]
    assert call["max_tokens"] == 20
    assert call["temperature"] == 0.8
    assert call["top_k"] == 40
    assert call["top_p"] == 0.9
    assert call["min_p"] == 0.05
    assert call["repeat_penalty"] == 1.1
    assert call["mirostat"] == 2
    assert not {"context_length", "num_ctx", "n_ctx", "n_predict"}.intersection(call)
    adapter.generate("again")
    assert client.calls[1]["temperature"] == 0.3
    assert client.calls[1]["max_tokens"] == 5


def test_request_context_is_checked_against_external_metadata_without_loading():
    client = _Client()
    adapter = _adapter(
        client, model_config=LlamaServerConfig(model="qwen", context_length=4096)
    )
    with pytest.raises(LLMBackendError, match="exceeds_loaded_context"):
        adapter.invoke_with_ctx("hello", num_ctx=8192)
    assert client.calls == []
    assert client.health_calls == 0


@pytest.mark.parametrize(
    "field",
    [
        "messages",
        "model",
        "stream",
        "stream_options",
        "max_tokens",
        "n_predict",
        "context_length",
        "n_gpu_layers",
    ],
)
def test_managed_request_fields_fail_before_http(field):
    client = _Client()
    with pytest.raises(ValueError, match="managed_fields"):
        _adapter(client).generate("hello", request_options={field: 1})
    assert client.calls == []


def test_extra_body_managed_fields_and_multiple_choices_fail_closed():
    client = _Client()
    with pytest.raises(ValueError, match="managed_fields"):
        _adapter(client, extra_body={"temperature": 1}).invoke("hello")
    with pytest.raises(ValueError, match="multiple_choices"):
        _adapter(client).generate("hello", request_options={"n": 2})
    assert client.calls == []


def test_structured_output_and_template_options_reach_native_chat_api():
    client = _Client()
    schema = {"type": "object", "properties": {"value": {"type": "integer"}}}
    _adapter(
        client,
        output_mode="json",
        json_schema=schema,
        chat_template_kwargs={"custom": "value"},
        reasoning="off",
    ).invoke("hello")
    call = client.calls[0]
    assert call["response_format"] == {"type": "json_object", "schema": schema}
    assert call["chat_template_kwargs"] == {"custom": "value", "enable_thinking": False}
    assert call["reasoning_effort"] == "none"


def test_conflicting_reasoning_template_request_fails_before_http():
    client = _Client()
    with pytest.raises(ValueError, match="reasoning_template_conflict"):
        _adapter(client, reasoning="off").generate(
            "hello",
            request_options={"chat_template_kwargs": {"enable_thinking": True}},
        )
    assert client.calls == []
    with pytest.raises(ValueError, match="reasoning_effort_conflict"):
        _adapter(client, reasoning="on").generate(
            "hello",
            request_options={"reasoning_effort": "none"},
        )
    assert client.calls == []


def test_separated_reasoning_and_inline_thinking_do_not_escape_visible_stream():
    client = _Client(
        [
            _chunk(reasoning="separate thinking "),
            _chunk(content="<thi"),
            _chunk(content="nk>inline secret</think>answer"),
            _chunk(finish="stop"),
        ]
    )
    result = _adapter(client).generate_with_metadata("hello")
    assert result.text == "answer"
    assert result.reasoning_text == "separate thinking "
    assert client.streams_closed == 1


def test_usage_missing_values_stay_unknown_and_zero_is_observed():
    client = _Client(
        [_chunk(finish="stop", usage={"prompt_tokens": 0, "completion_tokens": 4})]
    )
    usage = _adapter(client).generate_with_metadata("hello").usage
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 4
    assert usage.total_tokens is None
    assert usage.reasoning_tokens is None
    assert usage.visible_output_tokens is None
    assert usage.derived_fields == ()


def test_final_usage_after_finish_is_retained_and_derivation_is_explicit():
    client = _Client(
        [
            _chunk(content="answer"),
            _chunk(finish="stop"),
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 9,
                    "total_tokens": 16,
                    "completion_tokens_details": {"reasoning_tokens": 4},
                },
            },
        ]
    )
    result = _adapter(client).generate_with_metadata("hello")
    assert result.usage.reasoning_tokens == 4
    assert result.usage.visible_output_tokens == 5
    assert result.usage.derived_fields == ("visible_output_tokens",)
    assert result.transport.event_count == 3
    assert result.transport.terminal_event_seen is True


def test_tool_deltas_are_assembled_losslessly_without_execution():
    client = _Client(
        [
            _chunk(
                tools=[
                    {
                        "index": 0,
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "look", "arguments": '{"id":'},
                    }
                ]
            ),
            _chunk(tools=[{"index": 0, "function": {"name": "up", "arguments": "1}"}}]),
            _chunk(finish="tool_calls"),
        ]
    )
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]
    result = _adapter(client).generate_with_metadata(
        "hello", tools=tools, tool_choice="auto"
    )
    assert result.text == ""
    assert result.terminal_reason == "tool_calls"
    assert result.tool_calls == (
        {
            "id": "call_a",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"id":1}'},
        },
    )
    assert client.calls[0]["tools"] == tools
    assert client.calls[0]["tool_choice"] == "auto"
    assert len(client.calls) == 1


def test_metadata_stream_exposes_reasoning_and_tools_and_terminal_result():
    client = _Client(
        [
            _chunk(reasoning="plan"),
            _chunk(
                tools=[
                    {
                        "index": 0,
                        "id": "call_a",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "invalid-json-preserved",
                        },
                    }
                ]
            ),
            _chunk(finish="tool_calls"),
        ]
    )
    events = list(_adapter(client).stream_with_metadata("hello"))
    assert [event.kind for event in events] == [
        "reasoning",
        "tool_call_delta",
        "terminal",
    ]
    assert events[0].text == "plan"
    assert (
        events[-1].data["tool_calls"][0]["function"]["arguments"]
        == "invalid-json-preserved"
    )
    assert events[-1].terminal_reason == "tool_calls"


def test_tool_tokens_are_not_mislabeled_as_observed_visible_output_tokens():
    client = _Client(
        [
            _chunk(
                tools=[
                    {
                        "index": 0,
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ]
            ),
            _chunk(finish="tool_calls"),
            {
                "choices": [],
                "usage": {
                    "completion_tokens": 9,
                    "completion_tokens_details": {"reasoning_tokens": 4},
                },
            },
        ]
    )
    usage = _adapter(client).generate_with_metadata("hello").usage
    assert usage.completion_tokens == 9
    assert usage.reasoning_tokens == 4
    assert usage.visible_output_tokens is None
    assert usage.derived_fields == ()


@pytest.mark.parametrize(
    "payloads,match",
    [
        ([_chunk(content="partial")], "ended_before_finish_reason"),
        (
            [{"choices": [{"delta": {"content": {"secret": "not text"}}}]}],
            "malformed_chat_text",
        ),
        ([_chunk(finish="unknown secret")], "unsupported_finish_reason"),
        ([_chunk(finish="tool_calls")], "tool_finish_without_calls"),
        (
            [
                _chunk(
                    tools=[
                        {"index": 0, "function": {"name": "lookup", "arguments": "{}"}}
                    ]
                ),
                _chunk(finish="tool_calls"),
            ],
            "missing_tool_call_identity",
        ),
        ([_chunk(finish="stop"), _chunk(content="late")], "after_finish_reason"),
        (
            [_chunk(finish="stop", usage={"completion_tokens": -1})],
            "malformed_usage_counter",
        ),
    ],
)
def test_invalid_or_incomplete_streams_fail_closed_and_close_transport(payloads, match):
    client = _Client(payloads)
    with pytest.raises(LLMBackendError, match=match) as caught:
        _adapter(client).invoke("private prompt")
    assert client.streams_closed == 1
    assert caught.value.diagnostics is not None
    assert "private prompt" not in str(caught.value.diagnostics.to_dict())
    assert "secret" not in str(caught.value)


def test_transport_error_metadata_remains_bounded_and_redacts_model_paths():
    client = _Client(
        [
            *[_chunk(content=str(index)) for index in range(40)],
            LLMTimeoutError("timeout"),
        ]
    )
    adapter = LlamaCppAdapter(
        model="/Users/alice/private/model.gguf", role_id="role", client=client
    )
    with pytest.raises(LLMTimeoutError) as caught:
        adapter.invoke("private prompt")
    evidence = caught.value.diagnostics
    assert evidence.event_count == 40
    assert len(evidence.event_kinds) == 24
    assert evidence.model == "<abs_path_redacted>"
    assert evidence.terminal_reason == "idle_timeout"
    assert adapter.get_last_terminal_reason() == "idle_timeout"
    assert "private prompt" not in str(evidence.to_dict())


def test_cancel_before_request_prevents_any_http_call():
    client = _Client()
    adapter = _adapter(client)
    with pytest.raises(LLMCancelledError):
        adapter.generate("hello", cancellation_probe=lambda: True)
    assert client.calls == []
    assert adapter.get_last_usage() is None


def test_close_visible_and_metadata_iterators_closes_underlying_transport():
    for method in ("stream", "stream_with_metadata"):
        client = _Client(
            [_chunk(content="one"), _chunk(content="two"), _chunk(finish="stop")]
        )
        stream = getattr(_adapter(client), method)("hello")
        next(stream)
        stream.close()
        assert client.streams_closed == 1


def test_runaway_and_visible_output_cap_reuse_baseline_protections():
    profile = SimpleNamespace(
        runaway_min_chars=8,
        runaway_window_chars=64,
        runaway_block_chars=4,
        runaway_repeat_threshold=3,
        runaway_ngram_chars=3,
    )
    client = _Client([_chunk(reasoning="plan") for _ in range(5)])
    adapter = _adapter(client, role_profile=profile)
    with pytest.raises(LLMRunawayError):
        adapter.generate("hello")
    assert client.streams_closed == 1
    assert adapter.get_last_terminal_reason() == "runaway"
    client = _Client([_chunk(content="abcdefghij"), _chunk(finish="stop")])
    result = _adapter(client, num_predict=1).generate_with_metadata("hello")
    assert result.text == "abcd"
    assert result.terminal_reason == "visible_output_cap"
    assert client.streams_closed == 1


def test_health_and_compatibility_load_are_read_only():
    client = _Client()
    adapter = _adapter(client)
    assert adapter.load() is True
    status = adapter.health_status()
    assert status["loaded_context_length"] == 4096
    assert status["configured_context_length"] is None
    assert status["model_loading"] == "server_owned"
    assert status["process_lifecycle"] == "external"
    assert status["reasoning_capability_verified"] is False
    assert status["server_healthy"] is True
    assert status["model_ready"] is True
    assert status["model_available"] is True
    assert adapter.warmup_context(num_ctx=8192) is False
    assert client.calls == []
    assert set(client.props_calls) == {"qwen"}
    adapter.close()
    assert client.client_closed is False


def test_missing_props_does_not_invent_context_or_model_capabilities():
    client = _Client()
    client.props_result = LLMBackendError("unsupported")
    status = _adapter(client).health_status()
    assert status["healthy"] is True
    assert status["loaded_context_length"] is None
    assert status["reasoning_capability_verified"] is False
    client.props_result = []
    assert _adapter(client).health_status()["loaded_context_length"] is None


@pytest.mark.parametrize(
    "catalog,reason,state",
    [
        (
            {"data": [{"id": "different", "status": {"value": "loaded"}}]},
            "model_not_found",
            None,
        ),
        (
            {"data": [{"id": "qwen", "status": {"value": "unloaded"}}]},
            "model_not_loaded",
            "unloaded",
        ),
        (
            {"data": [{"id": "qwen", "status": {"value": "loading"}}]},
            "model_not_loaded",
            "loading",
        ),
        ({"data": [{"id": "qwen", "status": {}}]}, "model_not_loaded", "unknown"),
        (
            {"data": [{"id": "qwen", "status": {"value": []}}]},
            "model_not_loaded",
            "unknown",
        ),
        ({"data": [{"id": "qwen"}, {"id": "qwen"}]}, "model_alias_ambiguous", None),
    ],
)
def test_server_health_does_not_imply_selected_model_ready_or_auto_load(
    catalog, reason, state
):
    client = _Client()
    client.models_result = catalog
    runtime = _adapter(client)
    status = runtime.health_status()
    assert status["healthy"] is status["server_healthy"] is True
    assert status["model_ready"] is status["backend_ready"] is False
    assert status["reason"] == reason
    assert status["model_state"] == state
    assert status["loaded_context_length"] is None
    assert status["observed_slots"] is None
    assert runtime.is_ready() is False
    assert client.props_calls == []
    assert client.calls == []


def test_direct_server_model_readiness_requires_catalog_and_coherent_props():
    client = _Client()
    client.models_result = {"data": [{"id": "canonical", "aliases": ["qwen"]}]}
    client.props_result = {
        "model_alias": "canonical",
        "is_sleeping": False,
        "default_generation_settings": {"n_ctx": 2048},
        "total_slots": 4,
    }
    runtime = _adapter(client)
    status = runtime.health_status()
    assert status["model_ready"] is True
    assert status["loaded_context_length"] == 2048
    assert status["observed_slots"] == 4
    client.props_result["model_alias"] = "wrong-model"
    status = runtime.health_status()
    assert status["model_ready"] is False
    assert status["reason"] == "model_metadata_mismatch"
    assert status["loaded_context_length"] is None
    assert status["observed_slots"] is None
    client.props_result["model_alias"] = "qwen"
    client.props_result["is_sleeping"] = True
    assert runtime.health_status()["reason"] == "model_sleeping"
    assert runtime.is_ready() is False


def test_catalog_unavailable_leaves_model_readiness_unverified_even_with_props():
    client = _Client()
    client.models_result = LLMBackendError("catalog unavailable")
    client.props_result = {
        "model_alias": "qwen",
        "default_generation_settings": {"n_ctx": 2048},
    }
    status = _adapter(client).health_status()
    assert status["server_healthy"] is True
    assert status["model_ready"] is False
    assert status["model_available"] is False
    assert status["reason"] == "catalog_unavailable"
    assert status["loaded_context_length"] is None
    assert client.props_calls == []


def test_server_unavailability_never_launches_or_substitutes_runtime():
    client = _Client()
    client.health_result = LLMBackendError("unreachable")
    adapter = _adapter(client)
    assert adapter.is_ready() is False
    assert adapter.health_status()["reason"] == "unreachable"
    assert client.props_calls == []
    assert client.calls == []


def test_adapter_and_real_http_client_preserve_chat_usage_and_tools_end_to_end():
    """Qualify HTTP wiring without requiring a live inference model."""
    captured = []
    payloads = [
        _chunk(reasoning="plan €"),
        _chunk(
            tools=[
                {
                    "index": 0,
                    "id": "call_lookup",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"id":1}'},
                }
            ]
        ),
        _chunk(finish="tool_calls"),
        {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7}},
    ]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            captured.append((self.path, json.loads(body)))
            response = (
                "".join(
                    "data: " + json.dumps(item, ensure_ascii=False) + "\r\n\r\n"
                    for item in payloads
                )
                + "data: [DONE]\r\n\r\n"
            )
            encoded = response.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    adapter = LlamaCppAdapter(
        model="already-selected",
        base_url=f"http://127.0.0.1:{server.server_port}/gateway",
    )
    try:
        result = adapter.generate_with_metadata("hello", num_predict=128)
    finally:
        adapter.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)
    assert captured[0][0] == "/gateway/v1/chat/completions"
    assert captured[0][1]["max_tokens"] == 128
    assert captured[0][1]["model"] == "already-selected"
    assert captured[0][1]["messages"] == [{"role": "user", "content": "hello"}]
    assert result.reasoning_text == "plan €"
    assert result.tool_calls[0]["id"] == "call_lookup"
    assert result.usage.prompt_tokens == 11
    assert result.usage.reasoning_tokens is None
    assert result.usage.visible_output_tokens is None
    assert result.terminal_reason == "tool_calls"


@pytest.mark.parametrize("finish", [[], {}])
def test_non_string_finish_reason_is_a_classified_backend_error(finish):
    client = _Client([_chunk(finish=finish)])
    adapter = _adapter(client)
    with pytest.raises(LLMBackendError, match="unsupported_finish_reason") as caught:
        adapter.invoke("hello")
    assert adapter.get_last_terminal_reason() == "backend_error"
    assert caught.value.diagnostics.event_count == 1
    assert client.streams_closed == 1


@pytest.mark.parametrize("details", [[], "", 0, False])
def test_non_object_usage_details_are_not_silently_treated_as_missing(details):
    client = _Client(
        [
            _chunk(
                finish="stop",
                usage={
                    "completion_tokens": 7,
                    "completion_tokens_details": details,
                },
            )
        ]
    )
    with pytest.raises(LLMBackendError, match="malformed_usage_details"):
        _adapter(client).invoke("hello")
    assert client.streams_closed == 1


def test_chat_choice_index_requires_integer_zero():
    chunk = _chunk(finish="stop")
    chunk["choices"][0]["index"] = 0.0
    client = _Client([chunk])
    with pytest.raises(LLMBackendError, match="unsupported_chat_choice"):
        _adapter(client).invoke("hello")
    assert client.streams_closed == 1


def test_visible_cap_closes_transport_before_emitting_public_terminal_event():
    client = _Client(
        [
            _chunk(content="abcdefghijk"),
            _chunk(content="still running"),
            _chunk(finish="stop"),
        ]
    )
    events = _adapter(client, num_predict=1).stream_with_metadata("hello")
    try:
        assert next(events).text == "abcd"
        terminal = next(events)
        assert terminal.kind == "terminal"
        assert terminal.terminal_reason == "visible_output_cap"
        assert client.streams_closed == 1
    finally:
        events.close()
    assert client.streams_closed == 1


@pytest.mark.parametrize("key", ["timeout", "cancel"])
def test_transport_options_cannot_collide_with_explicit_helper_arguments(key):
    client = _Client()
    with pytest.raises(ValueError, match="managed_fields"):
        _adapter(client).generate("hello", request_options={key: 1})
    with pytest.raises(ValueError, match="managed_fields"):
        _adapter(client, extra_body={key: 1}).invoke("hello")
    assert client.calls == []


@pytest.mark.parametrize(
    "endpoint,replacement",
    [
        ("health", LLMTimeoutError("timeout")),
        ("health", LLMBackendError("unreachable")),
        ("health", {"status": "loading"}),
        ("props", LLMTimeoutError("timeout")),
        ("props", LLMBackendError("unavailable")),
        ("props", {}),
        ("props", {"default_generation_settings": {}}),
        ("props", []),
    ],
)
def test_failed_context_refresh_discards_stale_observation(endpoint, replacement):
    client = _Client()
    client.props_result = {"default_generation_settings": {"n_ctx": 4}}
    adapter = _adapter(client)
    assert adapter.health_status()["loaded_context_length"] == 4
    with pytest.raises(LLMBackendError, match="exceeds_loaded_context"):
        adapter.invoke_with_ctx("hello", num_ctx=8)
    assert client.calls == []

    setattr(client, f"{endpoint}_result", replacement)
    status = adapter.health_status()
    assert status["loaded_context_length"] is None
    assert status["configured_context_length"] is None
    assert adapter.invoke_with_ctx("hello", num_ctx=8).text == "done"
    assert len(client.calls) == 1


def test_failed_context_refresh_keeps_declared_configuration_separate():
    client = _Client()
    client.props_result = {"default_generation_settings": {"n_ctx": 4}}
    adapter = _adapter(
        client,
        model_config=LlamaServerConfig(model="qwen", context_length=4),
    )
    assert adapter.health_status()["loaded_context_length"] == 4
    client.props_result = LLMTimeoutError("timeout")
    status = adapter.health_status()
    assert status["loaded_context_length"] is None
    assert status["configured_context_length"] == 4
    with pytest.raises(LLMBackendError, match="exceeds_loaded_context"):
        adapter.invoke_with_ctx("hello", num_ctx=8)
    assert client.calls == []
