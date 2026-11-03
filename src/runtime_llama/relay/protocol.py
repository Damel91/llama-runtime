"""The declared OpenAI Chat/embedding wire projection over SDK results."""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import struct

from runtime_llama.embedding_halving import EmbeddingChunkSplitRequiredError
from runtime_llama.errors import (
    LLMBackendError,
    LLMCancelledError,
    LLMRunawayError,
    LLMTimeoutError,
    LLMToolRecoveryError,
)


class RelayError(Exception):
    """A safe client-visible error, separate from raw native exceptions."""

    def __init__(
        self, status: int, code: str, *, param: str | None = None, details=None
    ):
        super().__init__(code)
        self.status, self.code, self.param = status, code, param
        self.details = deepcopy(details)

    def payload(self) -> dict:
        error = {
            "message": self.code,
            "type": "invalid_request_error" if self.status < 500 else "server_error",
            "code": self.code,
        }
        if self.param is not None:
            error["param"] = self.param
        if self.details is not None:
            error["details"] = deepcopy(self.details)
        return {"error": error}


def _invalid(code, param=None):
    raise RelayError(400, code, param=param)


def _body(body):
    if not isinstance(body, dict):
        _invalid("relay_request_must_be_object")
    try:
        return json.loads(json.dumps(body, allow_nan=False))
    except (ValueError, TypeError, OverflowError):
        _invalid("relay_request_must_be_finite_json")


def _model(value):
    if not isinstance(value, str) or not value.strip():
        _invalid("relay_model_required", "model")
    return value


_CHAT_FIELDS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "n",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "stop",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "reasoning_effort",
        "chat_template_kwargs",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "user",
        "strata_mcp",
    }
)


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: list
    stream: bool
    include_usage: bool
    options: dict
    num_predict: int | None = None


def parse_chat(body) -> ChatRequest:
    body = _body(body)
    if set(body) - _CHAT_FIELDS:
        _invalid(
            "relay_unsupported_chat_parameter", sorted(set(body) - _CHAT_FIELDS)[0]
        )
    model = _model(body.get("model"))
    messages = body.get("messages")
    if (
        not isinstance(messages, list)
        or not messages
        or any(not isinstance(item, dict) for item in messages)
    ):
        _invalid("relay_messages_required", "messages")
    for item in messages:
        if item.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
            _invalid("relay_message_role_unsupported", "messages")
    stream = body.get("stream", False)
    if type(stream) is not bool:
        _invalid("relay_stream_must_be_boolean", "stream")
    if type(body.get("n", 1)) is not int or body.get("n", 1) != 1:
        _invalid("relay_multiple_choices_unsupported", "n")
    stream_options = body.get("stream_options", {})
    if not isinstance(stream_options, dict) or set(stream_options) - {"include_usage"}:
        _invalid("relay_stream_options_unsupported", "stream_options")
    include_usage = stream_options.get("include_usage", False)
    if type(include_usage) is not bool:
        _invalid("relay_include_usage_must_be_boolean", "stream_options")
    if "max_tokens" in body and "max_completion_tokens" in body:
        _invalid("relay_duplicate_output_budget", "max_tokens")
    budget = body.get("max_tokens", body.get("max_completion_tokens"))
    if budget is not None and (type(budget) is not int or budget <= 0):
        _invalid("relay_output_budget_invalid", "max_tokens")
    options = {
        key: value
        for key, value in body.items()
        if key
        not in {
            "model",
            "messages",
            "stream",
            "stream_options",
            "max_tokens",
            "max_completion_tokens",
        }
    }
    return ChatRequest(model, messages, stream, include_usage, options, budget)


@dataclass(frozen=True)
class EmbeddingRequest:
    model: str
    inputs: list[str]
    input_type: str | None
    encoding_format: str


def parse_embedding(body) -> EmbeddingRequest:
    body = _body(body)
    if set(body) - {"model", "input", "input_type", "encoding_format"}:
        _invalid(
            "relay_unsupported_embedding_parameter",
            sorted(set(body) - {"model", "input", "input_type", "encoding_format"})[0],
        )
    model = _model(body.get("model"))
    inputs = body.get("input")
    if isinstance(inputs, str):
        inputs = [inputs]
    if not isinstance(inputs, list) or any(
        not isinstance(item, str) for item in inputs
    ):
        _invalid("relay_embedding_text_inputs_required", "input")
    task = body.get("input_type")
    if task is not None and (
        not isinstance(task, str) or task not in {"query", "document", "raw"}
    ):
        _invalid("relay_embedding_input_type_invalid", "input_type")
    encoding = body.get("encoding_format", "float")
    if not isinstance(encoding, str) or encoding not in {"float", "base64"}:
        _invalid("relay_embedding_encoding_unsupported", "encoding_format")
    return EmbeddingRequest(model, inputs, task, encoding)


def usage_payload(usage) -> dict | None:
    if usage is None:
        return None
    value = {
        name: getattr(usage, name)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        if getattr(usage, name) is not None
    }
    reasoning = getattr(usage, "reasoning_tokens", None)
    if reasoning is not None:
        value["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return value or None


def finish_reason(event) -> str:
    if event.terminal_reason == "length_cap":
        return "length"
    reason = (event.data or {}).get("finish_reason")
    if reason in {"stop", "length", "tool_calls", "content_filter"}:
        return reason
    if reason == "max_tokens":
        return "length"
    mapped = {
        "completed": "stop",
        "tool_calls": "tool_calls",
        "content_filter": "content_filter",
    }.get(event.terminal_reason)
    if mapped is None:
        raise RelayError(502, "relay_incomplete_generation")
    return mapped


def chat_payload(result, *, request_id: str, created: int, model: str) -> dict:
    message = {
        "role": "assistant",
        "content": result.text if result.text or not result.tool_calls else None,
    }
    if result.reasoning_text:
        message["reasoning_content"] = result.reasoning_text
    if result.tool_calls:
        message["tool_calls"] = deepcopy(list(result.tool_calls))
    native = type(
        "Terminal", (), {"terminal_reason": result.terminal_reason, "data": {}}
    )()
    payload = {
        "id": "chatcmpl-" + request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason(native)}
        ],
    }
    usage = usage_payload(result.usage)
    if usage is not None:
        payload["usage"] = usage
    return payload


def sse(payload) -> bytes:
    return (
        b"data: "
        + json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        + b"\n\n"
    )


class ChatStreamProjection:
    """Project accepted SDK events; private recovery/timing events stay private."""

    def __init__(self, *, request_id, created, model, include_usage=False):
        self.base = {
            "id": "chatcmpl-" + request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
        }
        self.include_usage = include_usage
        self.started = self.finished = False
        self.usage = None

    def _chunk(self, delta, finish=None):
        payload = {
            **self.base,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if self.include_usage:
            payload["usage"] = None
        return sse(payload)

    def consume(self, event) -> list[bytes]:
        if self.finished:
            raise RelayError(502, "relay_event_after_terminal")
        if event.usage is not None:
            self.usage = event.usage
        chunks = []
        if (
            event.kind in {"content", "reasoning", "tool_call_delta", "terminal"}
            and not self.started
        ):
            self.started = True
            chunks.append(self._chunk({"role": "assistant"}))
        if event.kind == "content":
            chunks.append(self._chunk({"content": event.text}))
        elif event.kind == "reasoning":
            chunks.append(self._chunk({"reasoning_content": event.text}))
        elif event.kind == "tool_call_delta":
            calls = (event.data or {}).get("tool_calls")
            if not isinstance(calls, list):
                raise RelayError(502, "relay_malformed_sdk_tool_delta")
            chunks.append(self._chunk({"tool_calls": deepcopy(calls)}))
        elif event.kind == "terminal":
            chunks.append(self._chunk({}, finish_reason(event)))
            if self.include_usage:
                usage = usage_payload(self.usage)
                if usage is not None:
                    chunks.append(sse({**self.base, "choices": [], "usage": usage}))
            chunks.append(b"data: [DONE]\n\n")
            self.finished = True
        return chunks


def embedding_payload(vectors, request: EmbeddingRequest) -> dict:
    if len(vectors) != len(request.inputs):
        raise RelayError(502, "relay_embedding_cardinality_mismatch")
    rows = []
    for index, vector in enumerate(vectors):
        if (
            not isinstance(vector, (tuple, list))
            or not vector
            or any(
                type(value) not in {float, int} or not math.isfinite(value)
                for value in vector
            )
        ):
            raise RelayError(502, "relay_invalid_embedding")
        value = list(vector)
        if request.encoding_format == "base64":
            try:
                value = base64.b64encode(
                    struct.pack("<" + "f" * len(vector), *vector)
                ).decode("ascii")
            except (OverflowError, struct.error):
                raise RelayError(502, "relay_embedding_not_float32") from None
        rows.append({"object": "embedding", "index": index, "embedding": value})
    # The SDK does not observe ordinary direct-string token usage. Omit it;
    # never tokenize merely to fill these optional response counters.
    return {"object": "list", "model": request.model, "data": rows}


def map_error(error: BaseException) -> RelayError:
    if isinstance(error, RelayError):
        return error
    if isinstance(error, EmbeddingChunkSplitRequiredError):
        return RelayError(
            422,
            "embedding_chunk_split_required",
            details={
                "input_count": error.input_count,
                "input_type": error.input_type,
                "representation_id": error.representation_id,
                "completed": [asdict(item) for item in error.completed],
                "inputs": [asdict(item) for item in error.inputs],
            },
        )
    if isinstance(error, LLMCancelledError):
        return RelayError(499, "relay_request_cancelled")
    if isinstance(error, LLMTimeoutError):
        return RelayError(504, "relay_upstream_timeout")
    if isinstance(error, LLMToolRecoveryError):
        return RelayError(502, error.terminal_reason)
    if isinstance(error, LLMRunawayError):
        return RelayError(502, "relay_runaway_generation")
    if isinstance(error, (ValueError, TypeError)):
        return RelayError(400, "relay_invalid_request")
    if isinstance(error, LLMBackendError):
        code = str(error)
        if "model_unavailable" in code or "model_not_loaded" in code:
            return RelayError(404, "relay_model_unavailable")
        if "ambiguous" in code:
            return RelayError(409, "relay_model_ambiguous")
        return RelayError(502, "relay_upstream_error")
    return RelayError(500, "relay_internal_error")
