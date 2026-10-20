from types import SimpleNamespace

import pytest

from runtime_llama.adapters import llama as generation
from runtime_llama.contracts import _StreamAccumulator
from runtime_llama.errors import LLMBackendError, LLMRunawayError
from runtime_llama.usage import LLMServerTimings, LLMStreamEvent
from test_generation import _Client, _adapter, _chunk


def test_server_timing_rate_is_not_http_wall_or_usage(monkeypatch):
    clock = iter((0.0, 12.0))
    monkeypatch.setattr(generation.time, "monotonic", lambda: next(clock))
    payload = _chunk(finish="stop", usage={"completion_tokens": 999})
    payload["timings"] = {"predicted_n": 20, "predicted_ms": 500}
    runtime = _adapter(_Client([payload]))
    result = runtime.generate_with_metadata("hello")
    assert result.transport.elapsed_ms == 12_000
    assert result.usage.completion_tokens == 999
    assert result.server_timings.predicted_per_second == 40
    assert result.server_timings.derived_fields == ("predicted_per_second",)
    assert runtime.get_last_server_timings() is result.server_timings


def test_timing_only_event_after_finish_is_retained_without_mixing_usage():
    client = _Client(
        [
            _chunk(content="answer"),
            _chunk(finish="stop"),
            {"choices": [], "usage": {"completion_tokens": 3}},
            {
                "choices": [],
                "timings": {
                    "prompt_n": 10,
                    "prompt_ms": 25,
                    "predicted_n": 3,
                    "predicted_ms": 50,
                    "predicted_per_second": 59.75,
                    "cache_n": 7,
                    "unrelated_payload": {"prompt": "never retain this"},
                },
            },
        ]
    )
    runtime = _adapter(client)
    events = list(runtime.stream_with_metadata("hello"))
    assert [event.kind for event in events] == [
        "content",
        "usage",
        "timings",
        "terminal",
    ]
    timings = events[-1].server_timings
    assert timings is events[-2].server_timings
    assert timings.predicted_per_second == 59.75
    assert timings.prompt_per_second == 400
    assert timings.derived_fields == ("prompt_per_second",)
    assert timings.cache_n == 7
    assert "unrelated_payload" not in timings.to_dict()
    assert runtime.get_last_server_timings() is timings


@pytest.mark.parametrize("invalid", [None, True, -1, float("inf"), float("nan"), "12"])
def test_invalid_native_durations_never_fall_back_to_usage_or_wall(invalid):
    payload = _chunk(finish="stop", usage={"completion_tokens": 100})
    payload["timings"] = {"predicted_n": 3, "predicted_ms": invalid}
    timings = _adapter(_Client([payload])).invoke_with_ctx("hello").server_timings
    assert timings.predicted_n == 3
    assert timings.predicted_ms is None
    assert timings.predicted_per_second is None
    assert timings.derived_fields == ()


@pytest.mark.parametrize(
    "value", [None, [], "bad", {"predicted_n": True}, {"predicted_ms": -1}]
)
def test_absent_or_unusable_timing_snapshot_stays_unknown(value):
    assert LLMServerTimings.from_native(value) is None


def test_zero_native_counters_and_rate_are_observed_not_missing():
    timings = LLMServerTimings.from_native(
        {"predicted_n": 0, "predicted_ms": 0, "predicted_per_second": 0, "cache_n": 0}
    )
    assert timings.predicted_per_second == 0
    assert timings.predicted_ms == 0
    assert timings.cache_n == 0
    assert timings.derived_fields == ()
    assert (
        LLMServerTimings.from_native(
            {"predicted_n": 2, "predicted_ms": 0}
        ).predicted_per_second
        is None
    )


def test_timing_snapshots_do_not_join_count_and_duration_across_events():
    runtime = _adapter(
        _Client(
            [
                _chunk(finish="stop"),
                {"choices": [], "timings": {"predicted_n": 7}},
                {"choices": [], "timings": {"predicted_ms": 100}},
            ]
        )
    )
    timings = runtime.generate_with_metadata("hello").server_timings
    assert timings.predicted_n is None
    assert timings.predicted_ms == 100
    assert timings.predicted_per_second is None


def test_timing_only_metadata_is_not_a_terminal():
    client = _Client(
        [{"choices": [], "timings": {"predicted_n": 3, "predicted_ms": 10}}]
    )
    with pytest.raises(LLMBackendError, match="ended_before_finish_reason"):
        _adapter(client).generate("hello")
    assert client.streams_closed == 1


def test_native_timings_reset_between_calls():
    payload = _chunk(finish="stop")
    payload["timings"] = {"predicted_n": 2, "predicted_ms": 10}
    client = _Client([payload])
    runtime = _adapter(client)
    assert runtime.invoke_with_ctx("first").server_timings is not None
    client.payloads = [_chunk(finish="stop")]
    assert runtime.invoke_with_ctx("second").server_timings is None
    assert runtime.get_last_server_timings() is None


@pytest.mark.parametrize("method", ["stream", "stream_with_metadata"])
def test_streams_do_not_retain_full_reasoning_but_keep_diagnostics(method, monkeypatch):
    captures = []

    class RecordingAccumulator(_StreamAccumulator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captures.append(self)

    monkeypatch.setattr(generation, "_StreamAccumulator", RecordingAccumulator)
    import runtime_llama.contracts as contracts

    monkeypatch.setattr(contracts, "_StreamAccumulator", RecordingAccumulator)
    runtime = _adapter(
        _Client(
            [
                _chunk(reasoning="reasoning"),
                _chunk(content="visible"),
                _chunk(finish="stop"),
            ]
        )
    )
    diagnostics = []
    output = list(
        getattr(runtime, method)(
            "hello",
            **(
                {
                    "thinking_diagnostics": {"enabled": True},
                    "thinking_sink": diagnostics.append,
                }
                if method == "stream_with_metadata"
                else {}
            ),
        )
    )
    assert captures[0]._reasoning_parts == []
    assert captures[0]._reasoning_chars == len("reasoning")
    if method == "stream_with_metadata":
        assert [e.text for e in output if e.kind == "reasoning"] == ["reasoning"]
        assert diagnostics[0].text == "reasoning"
    else:
        assert output == ["visible"]
    blocking = runtime.generate_with_metadata("hello")
    assert blocking.reasoning_text == "reasoning"


def test_disabled_capture_does_not_install_or_call_capture_callbacks(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("disabled thinking capture called")

    monkeypatch.setattr(
        "runtime_llama.diagnostics._ThinkingCapture.observe", unexpected
    )
    result = _adapter(
        _Client(
            [
                _chunk(reasoning="separated"),
                _chunk(content="<think>inline</think>answer"),
                _chunk(finish="stop"),
            ]
        )
    ).invoke_with_ctx("hello")
    assert result.text == "answer"
    assert result.reasoning_text == "separated"


def test_unchanged_metadata_content_reuses_event_and_runaway_precedes_yield(
    monkeypatch,
):
    runtime = _adapter(_Client())
    event = LLMStreamEvent(kind="content", text="plain")
    monkeypatch.setattr(
        runtime, "_iter_stream_events_with_ctx", lambda *args, **kwargs: iter([event])
    )
    output = list(runtime.stream_with_metadata("hello"))
    assert output[0] is event

    runtime._role_profile = SimpleNamespace(
        runaway_min_chars=4,
        runaway_window_chars=16,
        runaway_block_chars=2,
        runaway_repeat_threshold=2,
        runaway_ngram_chars=2,
    )
    rejected = LLMStreamEvent(kind="content", text="<think>hidden</think>abababab")
    monkeypatch.setattr(
        runtime,
        "_iter_stream_events_with_ctx",
        lambda *args, **kwargs: iter([rejected]),
    )
    stream = runtime.stream_with_metadata(
        "hello", thinking_diagnostics={"enabled": True}
    )
    with pytest.raises(LLMRunawayError) as caught:
        next(stream)
    assert caught.value.thinking_diagnostic.text == "hidden"


def test_fragmented_tool_arguments_keep_exact_bytes_and_ignore_consumer_mutation():
    text = '{\n  "code": "' + "x €\\n" * 1000 + '"\n}'
    payloads = [
        _chunk(
            tools=[
                {
                    "index": 0,
                    "id": "call",
                    "function": {
                        "name": "look",
                        "arguments": text[:1],
                        "metadata": {"source": [1]},
                    },
                }
            ]
        ),
        *[
            _chunk(tools=[{"index": 0, "function": {"arguments": part}}])
            for part in text[1:]
        ],
        _chunk(tools=[{"index": 0, "function": {"name": "up"}}], finish="tool_calls"),
    ]
    runtime = _adapter(_Client(payloads))
    terminal = None
    for event in runtime.stream_with_metadata("hello"):
        if event.kind == "tool_call_delta":
            for call in event.data["tool_calls"]:
                call.setdefault("function", {})["arguments"] = "corrupted"
                call["id"] = "changed"
                if "metadata" in call["function"]:
                    call["function"]["metadata"]["source"].append(2)
        if event.kind == "terminal":
            terminal = event
    call = terminal.data["tool_calls"][0]
    assert call["id"] == "call"
    assert call["function"] == {
        "name": "lookup",
        "arguments": text,
        "metadata": {"source": [1]},
    }


def test_tool_usage_same_payload_does_not_derive_visible_tokens():
    runtime = _adapter(
        _Client(
            [
                _chunk(
                    tools=[
                        {
                            "index": 0,
                            "id": "call",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                    finish="tool_calls",
                    usage={"completion_tokens": 5, "reasoning_tokens": 2},
                )
            ]
        )
    )
    result = runtime.generate_with_metadata("hello")
    assert result.usage.visible_output_tokens is None
    assert result.usage.derived_fields == ()


def test_null_choice_cannot_be_hidden_by_usage():
    with pytest.raises(LLMBackendError, match="unsupported_chat_choice"):
        _adapter(
            _Client([{"choices": [None], "usage": {"completion_tokens": 0}}])
        ).invoke("hello")


def test_explicit_connection_policy_is_transport_only():
    runtime = _adapter(http_connection_policy="reuse")
    assert runtime._model_config.http_connection_policy == "reuse"
    runtime.generate("hello")
    assert "http_connection_policy" not in runtime._client.calls[0]
    with pytest.raises(ValueError, match="managed_fields"):
        runtime.generate("hello", request_options={"connection_policy": "reuse"})
    with pytest.raises(ValueError, match="conflicts"):
        generation.LlamaCppAdapter(
            model_config=runtime._model_config,
            http_connection_policy="close",
            client=_Client(),
        )
