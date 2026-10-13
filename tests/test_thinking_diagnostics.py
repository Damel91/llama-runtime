from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import logging
from threading import Barrier, Event
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from runtime_llama.adapters.llama import LlamaCppAdapter
from runtime_llama.diagnostics import ThinkingDiagnosticsConfig
from runtime_llama.errors import (
    LLMBackendError,
    LLMCancelledError,
    LLMRunawayError,
    LLMTimeoutError,
)
from runtime_llama.logging import log_thinking_diagnostic
from runtime_llama.usage import LLMCallResult, LLMStreamEvent


def chunk(*, content=None, reasoning=None, finish=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


class Client:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []
        self.closed_streams = 0

    def chat_completion(self, messages, **kwargs):
        self.calls.append({"messages": deepcopy(messages), **kwargs})

        def events():
            try:
                for payload in self.payloads:
                    if isinstance(payload, Exception):
                        raise payload
                    yield deepcopy(payload)
            finally:
                self.closed_streams += 1

        return events()


def adapter(payloads, **kwargs):
    return LlamaCppAdapter(
        model="qwen", role_id="coder", client=Client(payloads), **kwargs
    )


ENABLED = ThinkingDiagnosticsConfig(enabled=True)


def test_disabled_default_does_not_capture_or_deliver_thinking(caplog):
    sink = []
    runtime = adapter(
        [
            chunk(reasoning="separated secret"),
            chunk(content="<think>inline secret</think>ok"),
            chunk(finish="stop"),
        ]
    )
    with caplog.at_level(logging.INFO):
        result = runtime.generate_with_metadata(
            "private prompt", thinking_sink=sink.append
        )
    assert result.text == "ok"
    assert result.reasoning_text == "separated secret"
    assert result.thinking_diagnostic is None
    assert sink == []
    assert "secret" not in caplog.text
    assert "private prompt" not in caplog.text


def test_blocking_captures_both_channels_and_keeps_legacy_reasoning_contract():
    sink = []
    runtime = adapter(
        [
            chunk(reasoning="plan "),
            chunk(content="<thi"),
            chunk(content="nk>inline "),
            chunk(content="work</think>answer"),
            chunk(reasoning="done"),
            chunk(finish="stop"),
        ],
        thinking_diagnostics=ENABLED,
    )
    result = runtime.generate_with_metadata("hello", thinking_sink=sink.append)
    diagnostic = result.thinking_diagnostic
    assert result.text == "answer"
    assert result.reasoning_text == "plan done"
    assert diagnostic.text == "plan inline workdone"
    assert diagnostic.sources == ("separated", "inline")
    assert diagnostic.status == diagnostic.terminal_reason == "completed"
    assert diagnostic.role_id == "coder"
    assert diagnostic.model == "qwen"
    assert (
        diagnostic.observed_chars == diagnostic.captured_chars == len(diagnostic.text)
    )
    assert diagnostic.truncated is False
    assert sink == [diagnostic]
    assert "plan inline workdone" not in repr(diagnostic)
    assert not {"thinking_diagnostics", "thinking_sink"}.intersection(
        runtime._client.calls[0]
    )


def test_capture_limit_changes_only_diagnostic_and_can_override_profile():
    payloads = [
        chunk(reasoning="abcdefghij"),
        chunk(content="<think>klmnop</think>answer"),
        chunk(finish="stop"),
    ]
    runtime = adapter(payloads, thinking_diagnostics=ENABLED)
    result = runtime.generate_with_metadata(
        "hello", thinking_diagnostics={"enabled": True, "max_chars": 7}
    )
    assert result.text == "answer"
    assert result.reasoning_text == "abcdefghij"
    diagnostic = result.thinking_diagnostic
    assert diagnostic.text == "abcdefg"
    assert diagnostic.observed_chars == 16
    assert diagnostic.captured_chars == diagnostic.max_chars == 7
    assert diagnostic.truncated is True
    assert (
        runtime.generate_with_metadata(
            "hello", thinking_diagnostics=ThinkingDiagnosticsConfig()
        ).thinking_diagnostic
        is None
    )


@pytest.mark.parametrize("method", ["stream_with_ctx", "stream_with_metadata"])
def test_stream_final_snapshot_is_call_local_and_sink_fires_once(method):
    sink = []
    runtime = adapter(
        [
            chunk(reasoning="plan"),
            chunk(content="<think>detail</think>answer"),
            chunk(finish="stop"),
        ]
    )
    stream = getattr(runtime, method)(
        "hello", thinking_diagnostics=ENABLED, thinking_sink=sink.append
    )
    output = list(stream)
    stream.close()
    assert len(sink) == 1
    assert sink[0].text == "plandetail"
    assert sink[0].status == "completed"
    if method == "stream_with_metadata":
        assert output[-1].kind == "terminal"
        assert output[-1].thinking_diagnostic is sink[0]
        assert (
            "".join(event.text for event in output if event.kind == "content")
            == "answer"
        )
    else:
        assert "".join(output) == "answer"
    assert runtime._client.closed_streams == 1


def test_early_close_finalizes_partial_inline_and_preserves_no_success_terminal():
    sink = []
    runtime = adapter(
        [
            chunk(content="answer<think>partial</thi"),
            chunk(content="nk>later"),
            chunk(finish="stop"),
        ]
    )
    stream = runtime.stream_with_ctx(
        "hello", thinking_diagnostics=ENABLED, thinking_sink=sink.append
    )
    assert next(stream) == "answer"
    assert sink == []
    stream.close()
    assert len(sink) == 1
    assert sink[0].text == "partial</thi"
    assert sink[0].status == sink[0].terminal_reason == "interrupted"
    assert runtime.get_last_terminal_reason() == ""
    assert runtime._client.closed_streams == 1


@pytest.mark.parametrize(
    "method", ["generate_with_metadata", "stream_with_ctx", "stream_with_metadata"]
)
def test_precancel_delivers_cancelled_snapshot_without_http(method):
    sink = []
    runtime = adapter([])
    with pytest.raises(LLMCancelledError) as caught:
        result = getattr(runtime, method)(
            "private prompt",
            cancellation_probe=lambda: True,
            thinking_diagnostics=ENABLED,
            thinking_sink=sink.append,
        )
        if method != "generate_with_metadata":
            list(result)
    assert runtime._client.calls == []
    assert len(sink) == 1
    assert sink[0].status == "cancelled"
    assert sink[0].text == ""
    assert caught.value.thinking_diagnostic is sink[0]
    assert "private prompt" not in str(sink[0].to_dict())


def test_cancellation_retains_thinking_received_before_cancellation():
    sink = []
    cancelled = Event()
    runtime = adapter(
        [chunk(reasoning="plan"), chunk(content="answer"), chunk(finish="stop")]
    )
    stream = runtime.stream_with_metadata(
        "hello",
        cancellation_probe=cancelled.is_set,
        thinking_diagnostics=ENABLED,
        thinking_sink=sink.append,
    )
    assert next(stream).text == "plan"
    cancelled.set()
    with pytest.raises(LLMCancelledError) as caught:
        next(stream)
    assert sink[0].text == "plan"
    assert sink[0].status == "cancelled"
    assert caught.value.thinking_diagnostic is sink[0]
    assert runtime._client.closed_streams == 1


@pytest.mark.parametrize(
    "failure,reason,status",
    [
        (LLMTimeoutError("timeout"), "idle_timeout", "error"),
        (LLMBackendError("invalid response"), "backend_error", "error"),
        (None, "early_eof", "interrupted"),
    ],
)
def test_typed_failure_retains_capture_and_existing_transport_evidence(
    failure, reason, status
):
    sink = []
    payloads = [chunk(reasoning="plan"), chunk(content="<think>partial")]
    if failure is not None:
        payloads.append(failure)
    runtime = adapter(payloads)
    expected = type(failure) if failure is not None else LLMBackendError
    with pytest.raises(expected) as caught:
        runtime.generate_with_metadata(
            "private prompt", thinking_diagnostics=ENABLED, thinking_sink=sink.append
        )
    assert caught.value.diagnostics is not None
    assert caught.value.thinking_diagnostic is sink[0]
    assert sink[0].text == "planpartial"
    assert sink[0].terminal_reason == reason
    assert sink[0].status == status
    assert runtime._client.closed_streams == 1


def test_capture_does_not_change_runaway_detection_and_includes_triggering_delta():
    profile = SimpleNamespace(
        runaway_min_chars=8,
        runaway_window_chars=64,
        runaway_block_chars=4,
        runaway_repeat_threshold=3,
        runaway_ngram_chars=3,
    )
    payloads = [chunk(reasoning="plan") for _ in range(5)]
    diagnostics = []
    for enabled in (False, True):
        runtime = adapter(payloads, role_profile=profile)
        with pytest.raises(LLMRunawayError) as caught:
            runtime.generate_with_metadata(
                "hello", thinking_diagnostics={"enabled": enabled, "max_chars": 5}
            )
        diagnostics.append(caught.value.diagnostics)
        if enabled:
            captured = caught.value.thinking_diagnostic
            assert captured.status == "runaway"
            assert captured.text == "planp"
            assert captured.observed_chars >= 12
            assert captured.truncated is True
        assert runtime._client.closed_streams == 1
    assert diagnostics[0] == diagnostics[1]


def test_concurrent_calls_on_same_adapter_never_mix_snapshots():
    barrier = Barrier(2)

    class ConcurrentClient:
        def chat_completion(self, messages, **kwargs):
            name = messages[0]["content"]
            yield chunk(reasoning=f"separated-{name}|")
            barrier.wait(timeout=2)
            yield chunk(content=f"<think>inline-{name}</think>answer-{name}")
            yield chunk(finish="stop")

    runtime = LlamaCppAdapter(model="qwen", role_id="coder", client=ConcurrentClient())

    def invoke(name):
        sink = []
        result = runtime.generate_with_metadata(
            name, thinking_diagnostics=ENABLED, thinking_sink=sink.append
        )
        return result, sink

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(invoke, name) for name in ("one", "two")]
        completed = [future.result(timeout=3) for future in futures]
    for name, (result, sink) in zip(("one", "two"), completed, strict=True):
        assert result.text == f"answer-{name}"
        assert result.reasoning_text == f"separated-{name}|"
        assert result.thinking_diagnostic.text == f"separated-{name}|inline-{name}"
        assert sink == [result.thinking_diagnostic]
    assert (
        completed[0][0].thinking_diagnostic.call_id
        != completed[1][0].thinking_diagnostic.call_id
    )


@pytest.mark.parametrize("max_chars", [8, 100])
def test_known_credentials_are_redacted_across_deltas_and_at_capture_boundary(
    max_chars, caplog
):
    key = "api-secret-value"
    runtime = adapter(
        [
            chunk(reasoning="api-se"),
            chunk(reasoning="cret-value detail"),
            chunk(finish="stop"),
        ],
        api_key=key,
    )
    with caplog.at_level(logging.INFO):
        result = runtime.generate_with_metadata(
            "private prompt",
            thinking_diagnostics={"enabled": True, "max_chars": max_chars},
            thinking_sink=log_thinking_diagnostic,
        )
    text = result.thinking_diagnostic.text
    assert key not in text
    assert "api-secr" not in text
    assert key not in caplog.text
    assert "private prompt" not in caplog.text
    assert "credential" in text or text == "<credent"
    assert len(text) <= max_chars


def test_logging_full_content_is_explicit_and_redacts_bearer_tokens(caplog):
    runtime = adapter(
        [chunk(reasoning="detail Bearer opaque-token private"), chunk(finish="stop")]
    )
    with caplog.at_level(logging.INFO):
        result = runtime.generate_with_metadata(
            "hello", thinking_diagnostics=ENABLED, thinking_sink=log_thinking_diagnostic
        )
    assert "opaque-token" not in result.thinking_diagnostic.text
    assert "opaque-token" not in caplog.text
    assert "detail" in caplog.text
    assert "thinking diagnostic |" in caplog.text


def test_sink_failure_never_changes_answer_or_masks_model_failure(caplog):
    def broken_sink(_diagnostic):
        raise RuntimeError("private sink detail")

    runtime = adapter(
        [chunk(reasoning="plan"), chunk(content="answer"), chunk(finish="stop")]
    )
    result = runtime.generate_with_metadata(
        "hello", thinking_diagnostics=ENABLED, thinking_sink=broken_sink
    )
    assert result.text == "answer"
    runtime._client.payloads = [chunk(reasoning="plan"), LLMTimeoutError("timeout")]
    with pytest.raises(LLMTimeoutError) as caught:
        runtime.generate_with_metadata(
            "hello", thinking_diagnostics=ENABLED, thinking_sink=broken_sink
        )
    assert caught.value.thinking_diagnostic.text == "plan"
    assert "private sink detail" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_visible_cap_closes_transport_before_terminal_and_reports_interruption():
    sink = []
    runtime = adapter(
        [chunk(reasoning="plan"), chunk(content="abcdefghij"), chunk(finish="stop")]
    )
    events = list(
        runtime.stream_with_metadata(
            "hello",
            num_predict=1,
            thinking_diagnostics=ENABLED,
            thinking_sink=sink.append,
        )
    )
    assert "".join(event.text for event in events if event.kind == "content") == "abcd"
    assert events[-1].terminal_reason == "visible_output_cap"
    assert events[-1].thinking_diagnostic.status == "interrupted"
    assert sink == [events[-1].thinking_diagnostic]
    assert runtime._client.closed_streams == 1


@pytest.mark.parametrize(
    "value",
    [
        {"enabled": 1},
        {"max_chars": 0},
        {"max_chars": True},
        {"max_chars": 1.5},
        {"enabled": True, "unknown": True},
        "enabled",
    ],
)
def test_capture_configuration_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        ThinkingDiagnosticsConfig.from_value(value)


def test_public_diagnostic_annotations_resolve_without_consumer_import_workarounds():
    assert "thinking_diagnostic" in get_type_hints(LLMCallResult)
    assert "thinking_diagnostic" in get_type_hints(LLMStreamEvent)
    assert "diagnostic" in get_type_hints(log_thinking_diagnostic)


def test_invalid_sink_is_rejected_before_http():
    runtime = adapter([])
    with pytest.raises(ValueError, match="sink must be callable"):
        runtime.generate_with_metadata(
            "hello", thinking_diagnostics=ENABLED, thinking_sink=42
        )
    assert runtime._client.calls == []


@pytest.mark.parametrize(
    "method", ["generate_with_metadata", "stream_with_ctx", "stream_with_metadata"]
)
def test_visible_cap_cleanup_failure_precedes_final_snapshot(method):
    class BrokenCloseClient:
        def chat_completion(self, _messages, **_kwargs):
            try:
                yield chunk(reasoning="plan")
                yield chunk(content="abcdefghij")
                yield chunk(finish="stop")
            finally:
                raise LLMBackendError("llama_transport_cleanup_failure")

    runtime = LlamaCppAdapter(model="qwen", client=BrokenCloseClient())
    sink = []
    with pytest.raises(LLMBackendError) as caught:
        result = getattr(runtime, method)(
            "hello",
            num_predict=1,
            thinking_diagnostics=ENABLED,
            thinking_sink=sink.append,
        )
        if method != "generate_with_metadata":
            list(result)
    assert len(sink) == 1
    assert sink[0].text == "plan"
    assert sink[0].status == "error"
    assert sink[0].terminal_reason == "backend_error"
    assert caught.value.thinking_diagnostic is sink[0]
