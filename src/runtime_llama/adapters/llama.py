"""Generation adapter for an externally managed llama-server.

The normalized generation protections derive from lmstudio-agent-runtime. This
adapter only maps HTTP requests and responses; it never manages an inference
process or executes returned tool calls.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, replace
import os
import math
import time
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence

from runtime_llama.client import LlamaCppClient
from runtime_llama._proxy import assert_client_proxy, normalize_proxy_url, proxy_secrets
from runtime_llama.configuration import (
    AdapterProfile,
    LlamaGenerationConfig,
    LlamaRoleConfig,
    LlamaServerConfig,
)
from runtime_llama.contracts import (
    CancellationProbe,
    LLMAdapter,
    _StreamAccumulator,
    _ThinkingStreamStripper,
)
from runtime_llama.diagnostics import (
    ThinkingDiagnosticsConfig,
    ThinkingDiagnosticSink,
    _ThinkingCapture,
)
from runtime_llama.errors import (
    LLMBackendError,
    LLMCancelledError,
    LLMRunawayError,
    LLMTimeoutError,
    LLMToolRecoveryError,
    sanitize_diagnostic,
)
from runtime_llama.usage import (
    LLMCallResult,
    LLMStreamEvent,
    LLMServerTimings,
    LLMTransportAttempt,
    LLMUsageStats,
    LLMToolRecoveryAttempt,
    LLMToolRecoveryReport,
)
from runtime_llama.tool_recovery import ToolRecoveryPolicy


_RESERVED_REQUEST_FIELDS = {
    "messages",
    "model",
    "stream",
    "stream_options",
    "max_tokens",
    "n_predict",
    "timeout",
    "http_connection_policy",
    "connection_policy",
    "idle_timeout",
    "stream_idle_timeout_sec",
    "cancel_wait",
    "max_response_bytes",
    "trust_env",
    "verify_ssl",
    "api_key",
    "api_key_env",
    "proxy_url",
    "cancel",
    "num_predict",
    "num_ctx",
    "n_ctx",
    "context_length",
    "load_config",
    "auto_load",
    "instance_count",
    "n_gpu_layers",
    "gpu_layers",
    "main_gpu",
    "tensor_split",
    "model_path",
    "model_file",
}
_MESSAGE_FIELDS = (
    "name",
    "tool_calls",
    "tool_call_id",
    "function_call",
    "reasoning_content",
)


@dataclass
class _ToolCallBuilder:
    """Retain fragments once; materialize the public call only at completion."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    function_metadata: Dict[str, Any] = field(default_factory=dict)
    name_parts: List[str] = field(default_factory=list)
    argument_parts: List[str] = field(default_factory=list)

    def materialize(self) -> Dict[str, Any]:
        return {
            "function": {
                "name": "".join(self.name_parts),
                "arguments": "".join(self.argument_parts),
                **deepcopy(self.function_metadata),
            },
            **deepcopy(self.metadata),
        }


class LlamaCppAdapter(LLMAdapter):
    """Use chat completions on a local, remote or VPN-accessible llama-server.

    ``num_ctx`` is a logical ceiling checked against known external context
    metadata. It does not resize context or load a model. Sampling and tools are
    request-local; the selected server owns any model auto-load policy.
    """

    _client_trust_env = True

    def __init__(
        self,
        model: str = "",
        temperature: float = 0.2,
        base_url: str = "http://127.0.0.1:8080",
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        output_mode: str = "text",
        role_id: str = "unknown",
        role_profile: Any = None,
        stream_idle_timeout_sec: float | None = 30.0,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        reasoning: Any = "auto",
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        model_config: Optional[LlamaServerConfig] = None,
        role_config: Optional[LlamaRoleConfig] = None,
        generation_config: Optional[LlamaGenerationConfig] = None,
        client: Optional[LlamaCppClient] = None,
        thinking_diagnostics: Optional[ThinkingDiagnosticsConfig] = None,
        proxy_url: str | None = None,
        http_connection_policy: str | None = None,
    ) -> None:
        if (
            model_config is not None
            and proxy_url is not None
            and getattr(model_config, "proxy_url", None)
            != normalize_proxy_url(proxy_url)
        ):
            raise ValueError(
                "explicit proxy conflicts with the model_config proxy route"
            )
        if (
            model_config is not None
            and http_connection_policy is not None
            and getattr(model_config, "http_connection_policy", "close")
            != http_connection_policy
        ):
            raise ValueError("explicit connection policy conflicts with model_config")
        self._model_config = model_config or LlamaServerConfig(
            model=model,
            base_url=base_url,
            proxy_url=proxy_url,
            http_connection_policy=(
                "close" if http_connection_policy is None else http_connection_policy
            ),
        )
        self._role_config = role_config or LlamaRoleConfig(
            role_id=role_id,
            reasoning=reasoning,
            context_ceiling=num_ctx,
            visible_output_tokens=num_predict,
        )
        self._generation_config = generation_config or LlamaGenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            output_mode=output_mode,
            stream_idle_timeout_sec=stream_idle_timeout_sec,
            api_key=api_key,
            api_key_env=api_key_env,
            extra_body=dict(extra_body or {}),
            json_schema=json_schema,
            chat_template_kwargs=chat_template_kwargs or {},
            thinking_diagnostics=ThinkingDiagnosticsConfig.from_value(
                thinking_diagnostics
            ),
        )
        self._role_profile = role_profile or self._role_config
        self._role_id = self._role_config.role_id
        self._model_name = self._model_config.model
        self._num_ctx = self._role_config.context_ceiling
        self._num_predict = self._role_config.visible_output_tokens
        self._requested_context_ceiling: Optional[int] = None
        self._observed_context_length: Optional[int] = None
        self._last_usage: Optional[LLMUsageStats] = None
        self._adapter_closed = False
        token = self._generation_config.api_key or (
            os.environ.get(self._generation_config.api_key_env)
            if self._generation_config.api_key_env
            else None
        )
        assert_client_proxy(client, self._model_config.proxy_url)
        self._diagnostic_secrets = (
            ((token,) if token else ())
            + proxy_secrets(self._model_config.proxy_url)
            + proxy_secrets(getattr(client, "proxy_url", None))
        )
        self._owns_client = client is None
        self._client = client or LlamaCppClient(
            self._model_config.base_url,
            api_key=token,
            timeout=(
                30.0
                if self._generation_config.stream_idle_timeout_sec is None
                else self._generation_config.stream_idle_timeout_sec
            ),
            verify_ssl=self._model_config.verify_ssl,
            trust_env=self._client_trust_env,
            proxy_url=self._model_config.proxy_url,
            connection_policy=getattr(
                self._model_config, "http_connection_policy", "close"
            ),
        )

    @classmethod
    def from_profile(
        cls,
        profile: AdapterProfile,
        *,
        client: Optional[LlamaCppClient] = None,
    ) -> "LlamaCppAdapter":
        if profile.adapter_type != "llm":
            raise ValueError(f"Role '{profile.role_id}' is not an LLM role")
        return cls(
            model=profile.model,
            role_id=profile.role_id,
            role_profile=profile,
            model_config=profile.model_config,
            role_config=profile.role,
            generation_config=profile.generation,
            client=client,
        )

    @property
    def role_id(self) -> str:
        return self._role_id

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def role_profile(self) -> Any:
        return self._role_profile

    def _stream_idle(self, timeout: Optional[float]) -> float | None:
        value = (
            self._generation_config.stream_idle_timeout_sec
            if timeout is None
            else timeout
        )
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("timeout must be positive and finite")
        try:
            value = float(value)
        except (ValueError, OverflowError):
            raise ValueError("timeout must be positive and finite") from None
        if not math.isfinite(value) or value <= 0:
            raise ValueError("timeout must be positive and finite")
        return value

    def _raise_if_cancelled(
        self,
        cancellation_probe: Optional[CancellationProbe],
        *,
        timeout: Optional[float] = None,
        diagnostics: Optional[LLMTransportAttempt] = None,
    ) -> None:
        if cancellation_probe is None:
            return
        idle = self._stream_idle(timeout)
        try:
            super()._raise_if_cancelled(
                cancellation_probe, timeout=timeout, diagnostics=diagnostics
            )
        except LLMCancelledError as exc:
            if diagnostics is None and idle is None and exc.diagnostics is not None:
                exc.diagnostics = replace(exc.diagnostics, idle_timeout_sec=None)
            raise

    def _raise_if_recovery_retired(self) -> None:
        if self._adapter_closed:
            raise LLMBackendError("llama_adapter_closed")

    def _chat_messages(self, messages: Any) -> List[Dict[str, Any]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if not isinstance(messages, Sequence) or isinstance(
            messages, (bytes, bytearray)
        ):
            raise ValueError("llama_messages_must_be_text_or_sequence")
        projected: List[Dict[str, Any]] = []
        for message in messages:
            if isinstance(message, Mapping):
                item = deepcopy(dict(message))
                role = str(item.get("role") or "").lower()
                role = {"human": "user", "ai": "assistant"}.get(role, role)
                item["role"] = role
                item.setdefault("content", "")
            else:
                item = {
                    "role": self._message_role(message),
                    "content": deepcopy(self._message_content(message)),
                }
                additional = getattr(message, "additional_kwargs", None)
                if isinstance(additional, Mapping):
                    for key in _MESSAGE_FIELDS:
                        if key in additional:
                            item[key] = deepcopy(additional[key])
                for key in _MESSAGE_FIELDS:
                    value = getattr(message, key, None)
                    if value is not None:
                        item[key] = deepcopy(value)
            if item["role"] not in {"system", "developer", "user", "assistant", "tool"}:
                raise ValueError("llama_unsupported_message_role")
            projected.append(item)
        if not projected:
            raise ValueError("llama_messages_required")
        return projected

    def _body(
        self,
        messages: Any,
        *,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        output_format: Optional[str] = None,
        request_options: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        config = self._generation_config
        output_mode = output_format or config.output_mode
        if output_mode not in {"text", "json"}:
            raise ValueError("output_format must be 'text', 'json', or None")
        context = num_ctx if num_ctx is not None else self._num_ctx
        if context is None:
            context = self._requested_context_ceiling
        if context is not None:
            context = self._positive_integer(context, "num_ctx")
            loaded = self._observed_context_length or self._model_config.context_length
            if loaded is not None and context > loaded:
                raise LLMBackendError("llama_requested_context_exceeds_loaded_context")
        budget = num_predict if num_predict is not None else self._num_predict
        if budget is not None:
            budget = self._positive_integer(budget, "num_predict")
        body: Dict[str, Any] = {"temperature": config.temperature}
        for key in ("top_p", "top_k", "min_p", "repeat_penalty"):
            value = getattr(config, key, None)
            if value is not None:
                body[key] = value
        extras = deepcopy(dict(config.extra_body or {}))
        collisions = _RESERVED_REQUEST_FIELDS.intersection(extras)
        if collisions or set(body).intersection(extras):
            raise ValueError("llama_extra_body_overrides_managed_fields")
        body.update(extras)
        options = deepcopy(dict(request_options or {}))
        if _RESERVED_REQUEST_FIELDS.intersection(options):
            raise ValueError("llama_request_options_overrides_managed_fields")
        body.update(options)
        if body.get("n", 1) != 1:
            raise ValueError("llama_multiple_choices_unsupported")
        if budget is not None:
            body["max_tokens"] = budget
        template = deepcopy(dict(config.chat_template_kwargs or {}))
        requested_template = body.pop("chat_template_kwargs", None)
        if requested_template is not None:
            if not isinstance(requested_template, Mapping):
                raise ValueError("chat_template_kwargs must be an object")
            template.update(deepcopy(dict(requested_template)))
        reasoning = self._role_config.reasoning
        if reasoning != "auto":
            enabled = reasoning != "off"
            if (
                "enable_thinking" in template
                and template["enable_thinking"] is not enabled
            ):
                raise ValueError("llama_reasoning_template_conflict")
            template["enable_thinking"] = enabled
            if enabled and body.get("reasoning_effort") == "none":
                raise ValueError("llama_reasoning_effort_conflict")
            effort = (
                "none"
                if reasoning == "off"
                else (reasoning if reasoning in {"low", "medium", "high"} else None)
            )
            if effort is not None:
                if "reasoning_effort" in body and body["reasoning_effort"] != effort:
                    raise ValueError("llama_reasoning_effort_conflict")
                body["reasoning_effort"] = effort
        if template:
            body["chat_template_kwargs"] = template
        if output_mode == "json":
            if "response_format" in body:
                raise ValueError("llama_response_format_overrides_managed_fields")
            body["response_format"] = self._json_response_format(config.json_schema)
        body["messages"] = self._chat_messages(messages)
        return body

    def _json_response_format(self, schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        response: Dict[str, Any] = {"type": "json_object"}
        if schema is not None:
            response["schema"] = deepcopy(schema)
        return response

    def _response_validator(
        self, body: Mapping[str, Any]
    ) -> Optional[Callable[[str], None]]:
        """Optional provider validation before a completed success terminal.

        Ordinary text streams retain no extra response copy. A provider may
        require complete structured output to check its declared contract.
        """
        return None

    @staticmethod
    def _positive_integer(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _usage_from_payload(
        payload: Mapping[str, Any],
        *,
        has_tool_calls: bool = False,
    ) -> Optional[LLMUsageStats]:
        stats = payload.get("usage")
        if stats is None:
            return None
        if not isinstance(stats, Mapping):
            raise LLMBackendError("llama_malformed_usage")

        def count(source: Mapping[str, Any], key: str) -> Optional[int]:
            value = source.get(key)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LLMBackendError("llama_malformed_usage_counter")
            return value

        prompt = count(stats, "prompt_tokens")
        completion = count(stats, "completion_tokens")
        total = count(stats, "total_tokens")
        details = stats.get("completion_tokens_details")
        if details is None:
            details = {}
        if not isinstance(details, Mapping):
            raise LLMBackendError("llama_malformed_usage_details")
        reasoning = count(details, "reasoning_tokens")
        if reasoning is None:
            reasoning = count(stats, "reasoning_tokens")
        if all(value is None for value in (prompt, completion, total, reasoning)):
            return None
        visible = None
        if completion is not None and reasoning is not None:
            if reasoning > completion:
                raise LLMBackendError("llama_inconsistent_reasoning_usage")
            # Tool arguments also consume completion tokens. Their contribution
            # cannot be separated from visible prose using these counters.
            if not has_tool_calls:
                visible = completion - reasoning
        raw = {
            key: deepcopy(stats[key])
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "reasoning_tokens",
                "completion_tokens_details",
                "prompt_tokens_details",
            )
            if key in stats
        }
        return LLMUsageStats(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            reasoning_tokens=reasoning,
            visible_output_tokens=visible,
            raw=raw,
            derived_fields=("visible_output_tokens",) if visible is not None else (),
        )

    @staticmethod
    def _merge_tool_deltas(
        accumulated: Dict[int, _ToolCallBuilder],
        deltas: Any,
    ) -> None:
        if not isinstance(deltas, list):
            raise LLMBackendError("llama_malformed_tool_call_delta")
        for delta in deltas:
            if not isinstance(delta, Mapping):
                raise LLMBackendError("llama_malformed_tool_call_delta")
            index = delta.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise LLMBackendError("llama_missing_tool_call_index")
            call = accumulated.get(index)
            if call is None:
                call = _ToolCallBuilder()
                accumulated[index] = call
            for key, value in delta.items():
                if key in {"index", "function"}:
                    continue
                if (
                    key in {"id", "type"}
                    and key in call.metadata
                    and call.metadata[key] != value
                ):
                    raise LLMBackendError("llama_conflicting_tool_call_identity")
                call.metadata[key] = deepcopy(value)
            function = delta.get("function", {})
            if not isinstance(function, Mapping):
                raise LLMBackendError("llama_malformed_tool_call_function")
            for key, value in function.items():
                if key in {"name", "arguments"}:
                    if not isinstance(value, str):
                        raise LLMBackendError("llama_malformed_tool_call_function")
                    parts = call.name_parts if key == "name" else call.argument_parts
                    parts.append(value)
                else:
                    call.function_metadata[key] = deepcopy(value)

    @staticmethod
    def _completed_tools(
        accumulated: Dict[int, _ToolCallBuilder],
    ) -> List[Dict[str, Any]]:
        calls = [accumulated[index].materialize() for index in sorted(accumulated)]
        for call in calls:
            if not isinstance(call.get("id"), str) or not call["id"]:
                raise LLMBackendError("llama_missing_tool_call_identity")
            if (
                call.get("type", "function") != "function"
                or not call["function"]["name"]
            ):
                raise LLMBackendError("llama_malformed_completed_tool_call")
            call.setdefault("type", "function")
        if len({call["id"] for call in calls}) != len(calls):
            raise LLMBackendError("llama_duplicate_tool_call_identity")
        return calls

    def _iter_stream_events_with_ctx(
        self,
        messages: Any,
        *,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        output_format: Optional[str] = None,
        timeout: Optional[float] = None,
        cancellation_probe: Optional[CancellationProbe] = None,
        request_options: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[LLMStreamEvent]:
        idle = self._stream_idle(timeout)
        self._raise_if_cancelled(cancellation_probe, timeout=idle)
        body = self._body(
            messages,
            num_ctx=num_ctx,
            num_predict=num_predict,
            output_format=output_format,
            request_options=request_options,
        )
        yield from self._iter_chat_attempt(
            body, idle=idle, cancellation_probe=cancellation_probe
        )

    def _iter_chat_attempt(
        self,
        body: Mapping[str, Any],
        *,
        idle: float | None,
        cancellation_probe: Optional[CancellationProbe],
        tool_policy_enabled: bool = False,
    ) -> Iterator[LLMStreamEvent]:
        """Perform exactly one native POST against a frozen effective body."""
        body = deepcopy(dict(body))
        validator = self._response_validator(body)
        response_parts: Optional[List[str]] = [] if validator is not None else None
        request_messages = body.pop("messages")
        started = time.monotonic()
        event_count = 0
        event_kinds: deque[str] = deque(maxlen=24)
        finish_reason: Optional[str] = None
        usage: Optional[LLMUsageStats] = None
        server_timings: Optional[LLMServerTimings] = None
        tool_calls: Dict[int, _ToolCallBuilder] = {}
        native_tool_diagnostics: Dict[str, Any] = {}
        stream = None

        def evidence(reason: str) -> LLMTransportAttempt:
            return LLMTransportAttempt(
                elapsed_ms=(time.monotonic() - started) * 1000,
                idle_timeout_sec=idle,
                terminal_reason=reason,
                terminal_event_seen=finish_reason is not None,
                event_count=event_count,
                event_kinds=tuple(event_kinds),
                role_id=sanitize_diagnostic(self.role_id, limit=80),
                model=sanitize_diagnostic(self.model_name, limit=80),
            )

        try:
            stream = self._client.chat_completion(
                request_messages,
                model=self.model_name,
                stream=True,
                cancel=cancellation_probe,
                timeout=idle,
                **({"idle_timeout": None} if idle is None else {}),
                stream_options={"include_usage": True},
                **body,
            )
            if isinstance(stream, (Mapping, list)):
                raise LLMBackendError("llama_stream_response_required")
            for payload in stream:
                self._raise_if_cancelled(cancellation_probe, timeout=idle)
                event_count += 1
                event_kinds.append("chat_chunk")
                if not isinstance(payload, Mapping):
                    raise LLMBackendError("llama_malformed_chat_chunk")
                if payload.get("error") is not None:
                    raise LLMBackendError("llama_server_error")
                observed_diagnostics = {
                    key: deepcopy(payload[key])
                    for key in ("strata_rejected_tool_calls", "strata_mcp")
                    if key in payload
                }
                if observed_diagnostics:
                    native_tool_diagnostics.update(observed_diagnostics)
                    yield LLMStreamEvent(
                        kind="tool_diagnostic", data=observed_diagnostics
                    )
                choices = payload.get("choices", [] if observed_diagnostics else None)
                if not isinstance(choices, list) or len(choices) > 1:
                    raise LLMBackendError("llama_malformed_chat_choices")
                choice = choices[0] if choices else None
                delta = None
                if choices:
                    if (
                        not isinstance(choice, Mapping)
                        or type(choice.get("index", 0)) is not int
                        or choice.get("index", 0) != 0
                    ):
                        raise LLMBackendError("llama_unsupported_chat_choice")
                    delta = choice.get("delta")
                    if not isinstance(delta, Mapping):
                        raise LLMBackendError("llama_malformed_chat_delta")
                observed_usage = self._usage_from_payload(
                    payload,
                    has_tool_calls=bool(tool_calls)
                    or bool(delta and delta.get("tool_calls")),
                )
                if observed_usage is not None:
                    usage = observed_usage
                    yield LLMStreamEvent(kind="usage", usage=usage)
                observed_timings = LLMServerTimings.from_native(payload.get("timings"))
                if observed_timings is not None:
                    server_timings = observed_timings
                    yield LLMStreamEvent(kind="timings", server_timings=server_timings)
                if not choices:
                    if (
                        "usage" not in payload
                        and "timings" not in payload
                        and not observed_diagnostics
                    ):
                        raise LLMBackendError("llama_empty_chat_chunk")
                    continue
                if finish_reason is not None and delta:
                    raise LLMBackendError("llama_delta_after_finish_reason")
                if "function_call" in delta:
                    raise LLMBackendError("llama_legacy_function_call_unsupported")
                for key, kind in (
                    ("reasoning_content", "reasoning"),
                    ("content", "content"),
                ):
                    value = delta.get(key)
                    if value is not None and not isinstance(value, str):
                        raise LLMBackendError("llama_malformed_chat_text")
                    if value:
                        if kind == "content" and response_parts is not None:
                            response_parts.append(value)
                        yield LLMStreamEvent(kind=kind, text=value)
                if "tool_calls" in delta:
                    self._merge_tool_deltas(tool_calls, delta["tool_calls"])
                    yield LLMStreamEvent(
                        kind="tool_call_delta",
                        data={"tool_calls": deepcopy(delta["tool_calls"])},
                    )
                observed_finish = choice.get("finish_reason")
                if observed_finish is not None:
                    if not isinstance(observed_finish, str) or observed_finish not in {
                        "stop",
                        "length",
                        "max_tokens",
                        "tool_calls",
                        "content_filter",
                    }:
                        raise LLMBackendError("llama_unsupported_finish_reason")
                    if finish_reason is not None and finish_reason != observed_finish:
                        raise LLMBackendError("llama_conflicting_finish_reason")
                    finish_reason = observed_finish
            self._raise_if_cancelled(cancellation_probe, timeout=idle)
            if finish_reason is None:
                error = LLMBackendError("llama_stream_ended_before_finish_reason")
                error.diagnostics = evidence("early_eof")
                raise error
            calls = self._completed_tools(tool_calls)
            if finish_reason == "tool_calls" and not calls and not tool_policy_enabled:
                raise LLMBackendError("llama_tool_finish_without_calls")
            if (
                validator is not None
                and finish_reason == "stop"
                and not calls
                and not (
                    tool_policy_enabled
                    and native_tool_diagnostics.get("strata_rejected_tool_calls")
                )
            ):
                validator("".join(response_parts or ()))
            terminal = {
                "stop": "completed",
                "length": "length_cap",
                "max_tokens": "length_cap",
            }.get(
                finish_reason,
                finish_reason,
            )
            yield LLMStreamEvent(
                kind="terminal",
                usage=usage,
                terminal_reason=terminal,
                transport=evidence(terminal),
                data={
                    "tool_calls": calls,
                    "finish_reason": finish_reason,
                    **deepcopy(native_tool_diagnostics),
                },
                server_timings=server_timings,
            )
        except (LLMBackendError, LLMCancelledError, LLMTimeoutError) as exc:
            if exc.diagnostics is None:
                exc.diagnostics = evidence(exc.terminal_reason)
            raise
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

    def _capture_rejected_thinking(
        self,
        events: Sequence[LLMStreamEvent],
        reason: str,
        thinking_diagnostics: Optional[ThinkingDiagnosticsConfig],
        thinking_sink: Optional[ThinkingDiagnosticSink],
    ):
        configured = getattr(self._generation_config, "thinking_diagnostics", None)
        config = ThinkingDiagnosticsConfig.from_value(
            configured if thinking_diagnostics is None else thinking_diagnostics
        )
        if not config.enabled:
            return None
        capture = _ThinkingCapture(
            config,
            role_id=self.role_id,
            model=self.model_name,
            sink=thinking_sink,
            secrets=self._diagnostic_secrets,
        )
        stripper = _ThinkingStreamStripper(
            self,
            lambda text: capture.observe("inline", text),
            log_thinking=False,
        )
        for event in events:
            if event.kind == "reasoning":
                capture.observe("separated", event.text)
            elif event.kind == "content":
                for _ in stripper.feed(event.text):
                    pass
        for _ in stripper.flush():
            pass
        return capture.finish(reason)

    @staticmethod
    def _recovery_proposal(events: Sequence[LLMStreamEvent]) -> Dict[str, Any]:
        terminal = next(
            (event for event in reversed(events) if event.kind == "terminal"), None
        )
        return {
            "content": "".join(
                event.text for event in events if event.kind == "content"
            ),
            "tool_calls": deepcopy(
                (terminal.data or {}).get("tool_calls", []) if terminal else []
            ),
        }

    def _iter_invocation_events(
        self,
        messages: Any,
        *,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        output_format: Optional[str] = None,
        timeout: Optional[float] = None,
        cancellation_probe: Optional[CancellationProbe] = None,
        request_options: Optional[Mapping[str, Any]] = None,
        thinking_diagnostics: Optional[ThinkingDiagnosticsConfig] = None,
        thinking_sink: Optional[ThinkingDiagnosticSink] = None,
    ) -> Iterator[LLMStreamEvent]:
        idle = self._stream_idle(timeout)
        self._raise_if_cancelled(cancellation_probe, timeout=idle)
        body = self._body(
            messages,
            num_ctx=num_ctx,
            num_predict=num_predict,
            output_format=output_format,
            request_options=request_options,
        )
        policy = ToolRecoveryPolicy.from_request(body)
        if not policy.enabled:
            # Keep the legacy native seam observable to third-party adapters
            # and test transports when no tool recovery is applicable.
            events = self._iter_stream_events_with_ctx(
                messages,
                num_ctx=num_ctx,
                num_predict=num_predict,
                output_format=output_format,
                timeout=timeout,
                cancellation_probe=cancellation_probe,
                request_options=request_options,
            )
            started = time.perf_counter()
            usage = None
            server_timings = None
            try:
                for event in events:
                    usage = event.usage if event.usage is not None else usage
                    server_timings = (
                        event.server_timings
                        if event.server_timings is not None
                        else server_timings
                    )
                    if event.data and "strata_rejected_tool_calls" in event.data:
                        decision = policy.classify(
                            [event],
                            event if event.kind == "terminal" else None,
                            secrets=self._diagnostic_secrets,
                        )
                        if not decision.accepted:
                            close = getattr(events, "close", None)
                            if callable(close):
                                close()
                            attempt = LLMToolRecoveryAttempt(
                                index=0,
                                outcome="unsafe",
                                reasons=decision.reasons,
                                native_rejected_tool_calls=decision.native_rejected_tool_calls,
                                native_rejections=decision.native_rejections,
                                provenance=decision.provenance,
                                usage=usage,
                                transport=event.transport,
                                server_timings=server_timings,
                            )
                            raise LLMToolRecoveryError(
                                "tool_recovery_unsafe",
                                tool_recovery=LLMToolRecoveryReport(
                                    (attempt,),
                                    elapsed_ms=(time.perf_counter() - started) * 1000,
                                ),
                                last_proposal={"content_already_streamed": True},
                                diagnostics=event.transport,
                            )
                    yield event
            finally:
                close = getattr(events, "close", None)
                if callable(close):
                    close()
            return

        # Everything below belongs to this external invocation, never the
        # adapter/session. New calls therefore start with a fresh budget.
        started = time.monotonic()
        history = deepcopy(body["messages"])
        attempts: List[LLMToolRecoveryAttempt] = []
        staged: List[LLMStreamEvent] = []

        def native_cancelled() -> bool:
            # The native request owns its interruption even with a borrowed
            # client. Retiring this adapter must not close sibling requests.
            if self._adapter_closed:
                return True
            cancelled = bool(cancellation_probe and cancellation_probe())
            return cancelled or self._adapter_closed

        def check_cancelled() -> None:
            # Closing a borrowed adapter must not close its caller-owned
            # transport. It still retires private staged proposals and cannot
            # allow another correction or late executable event to escape.
            self._raise_if_projection_cancelled(
                cancellation_probe, timeout=idle, recovery_active=True
            )

        def report(*, exhausted: bool = False) -> LLMToolRecoveryReport:
            return LLMToolRecoveryReport(
                tuple(attempts), exhausted, (time.monotonic() - started) * 1000
            )

        def snapshot(index, outcome, decision=None, diagnostic=None, error=None):
            def latest(attribute):
                return next(
                    (
                        getattr(event, attribute)
                        for event in reversed(staged)
                        if getattr(event, attribute) is not None
                    ),
                    None,
                )

            transport = latest("transport")
            if transport is None and isinstance(
                getattr(error, "diagnostics", None), LLMTransportAttempt
            ):
                transport = error.diagnostics
            return LLMToolRecoveryAttempt(
                index=index,
                outcome=outcome,
                reasons=(decision.reasons if decision else (outcome,)),
                native_rejected_tool_calls=(
                    decision.native_rejected_tool_calls if decision else None
                ),
                native_rejections=(decision.native_rejections if decision else ()),
                provenance=decision.provenance if decision else (),
                usage=latest("usage"),
                transport=transport,
                server_timings=latest("server_timings"),
                thinking_diagnostic=diagnostic,
            )

        try:
            for index in range(4):
                check_cancelled()
                staged.clear()
                terminal = None
                local_cap = False
                guard = _StreamAccumulator(
                    self,
                    visible_output_token_limit=num_predict,
                    collect_reasoning=False,
                    inspection_only=True,
                )
                attempt_body = deepcopy(body)
                attempt_body["messages"] = deepcopy(history)
                stream = self._iter_chat_attempt(
                    attempt_body,
                    idle=idle,
                    cancellation_probe=native_cancelled,
                    tool_policy_enabled=True,
                )
                try:
                    try:
                        for event in stream:
                            check_cancelled()
                            staged.append(event)
                            if event.kind == "terminal":
                                terminal = event
                            for _ in guard.consume(event):
                                pass
                            if guard.visible_limit_reached:
                                local_cap = True
                                break
                        for _ in guard.finish():
                            pass
                        local_cap = local_cap or guard.visible_limit_reached
                    finally:
                        stream.close()
                except BaseException as error:
                    if isinstance(error, GeneratorExit):
                        raise
                    reason = str(getattr(error, "terminal_reason", "invocation_error"))
                    diagnostic = self._capture_rejected_thinking(
                        staged, reason, thinking_diagnostics, thinking_sink
                    )
                    attempts.append(
                        snapshot(index, reason, diagnostic=diagnostic, error=error)
                    )
                    error.tool_recovery = report()
                    if diagnostic is not None:
                        error.thinking_diagnostic = diagnostic
                    raise

                check_cancelled()
                decision = policy.classify(
                    staged, terminal, secrets=self._diagnostic_secrets
                )
                incomplete = terminal is not None and terminal.terminal_reason in {
                    "length_cap",
                    "visible_output_cap",
                    "content_filter",
                }
                if local_cap or incomplete:
                    reason = (
                        "visible_output_cap" if local_cap else terminal.terminal_reason
                    )
                    attempts.append(snapshot(index, reason, decision))
                    current_report = report()
                    # Existing partial-text behavior remains, but no candidate
                    # tool fragment can escape an incomplete generation.
                    yield LLMStreamEvent(
                        kind="tool_recovery", tool_recovery=current_report
                    )
                    for event in staged:
                        if event.kind not in {
                            "tool_call_delta",
                            "tool_diagnostic",
                            "terminal",
                        }:
                            check_cancelled()
                            yield replace(event, tool_recovery=current_report)
                    yield LLMStreamEvent(
                        kind="terminal",
                        terminal_reason=reason,
                        usage=attempts[-1].usage,
                        transport=attempts[-1].transport,
                        server_timings=attempts[-1].server_timings,
                        data={
                            "tool_calls": [],
                            **(
                                {"finish_reason": terminal.data.get("finish_reason")}
                                if terminal is not None and terminal.data
                                else {}
                            ),
                        },
                        tool_recovery=current_report,
                    )
                    return

                if decision.accepted:
                    attempts.append(snapshot(index, "accepted", decision))
                    current_report = report()
                    for event in staged:
                        check_cancelled()
                        yield replace(event, tool_recovery=current_report)
                    return

                diagnostic = self._capture_rejected_thinking(
                    staged,
                    "tool_protocol_rejected",
                    thinking_diagnostics,
                    thinking_sink,
                )
                attempts.append(
                    snapshot(
                        index,
                        "rejected" if decision.recoverable else "unsafe",
                        decision,
                        diagnostic,
                    )
                )
                exhausted = decision.recoverable and index == 3
                current_report = report(exhausted=exhausted)
                if not decision.recoverable or exhausted:
                    error = LLMToolRecoveryError(
                        "tool_recovery_exhausted"
                        if exhausted
                        else "tool_recovery_unsafe",
                        tool_recovery=current_report,
                        last_proposal=self._recovery_proposal(staged),
                        diagnostics=attempts[-1].transport,
                    )
                    if diagnostic is not None:
                        error.thinking_diagnostic = diagnostic
                    raise error
                check_cancelled()
                history.extend(policy.correction_messages(staged, decision))
                staged.clear()
                yield LLMStreamEvent(
                    kind="tool_recovery",
                    data={"next_attempt": index + 1},
                    tool_recovery=current_report,
                )
        except BaseException as error:
            if (
                not isinstance(error, GeneratorExit)
                and attempts
                and not hasattr(error, "tool_recovery")
            ):
                error.tool_recovery = report()
            if (
                not isinstance(error, GeneratorExit)
                and attempts
                and attempts[-1].thinking_diagnostic is not None
                and not hasattr(error, "thinking_diagnostic")
            ):
                error.thinking_diagnostic = attempts[-1].thinking_diagnostic
            raise
        finally:
            staged.clear()
            history.clear()

    @staticmethod
    def _tool_options(
        request_options: Optional[Mapping[str, Any]],
        tools: Optional[Sequence[Mapping[str, Any]]],
        tool_choice: Any,
    ) -> Optional[Dict[str, Any]]:
        options = deepcopy(dict(request_options or {}))
        for name, value in (("tools", tools), ("tool_choice", tool_choice)):
            if value is not None:
                if name in options:
                    raise ValueError(f"llama_duplicate_request_option:{name}")
                options[name] = deepcopy(list(value) if name == "tools" else value)
        return options or None

    def generate_with_metadata(
        self,
        messages: Any,
        *,
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        tool_choice: Any = None,
        request_options: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> LLMCallResult:
        return self.invoke_with_ctx(
            messages,
            request_options=self._tool_options(request_options, tools, tool_choice),
            **kwargs,
        )

    def generate(self, messages: Any, **kwargs: Any) -> str:
        return self.generate_with_metadata(messages, **kwargs).text

    def stream_with_metadata(
        self,
        messages: Any,
        *,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        output_format: Optional[str] = None,
        timeout: Optional[float] = None,
        cancellation_probe: Optional[CancellationProbe] = None,
        request_options: Optional[Mapping[str, Any]] = None,
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        tool_choice: Any = None,
        thinking_diagnostics: Optional[ThinkingDiagnosticsConfig] = None,
        thinking_sink: Optional[ThinkingDiagnosticSink] = None,
    ) -> Iterator[LLMStreamEvent]:
        self._set_last_usage(None)
        self._set_last_server_timings(None)
        self._set_last_terminal_reason("")
        events = self._iter_invocation_events(
            messages,
            num_ctx=num_ctx,
            num_predict=num_predict,
            output_format=output_format,
            timeout=timeout,
            cancellation_probe=cancellation_probe,
            request_options=self._tool_options(request_options, tools, tool_choice),
            thinking_diagnostics=thinking_diagnostics,
            thinking_sink=thinking_sink,
        )
        accumulator = _StreamAccumulator(
            self,
            visible_output_token_limit=num_predict,
            thinking_diagnostics=thinking_diagnostics,
            thinking_sink=thinking_sink,
            collect_reasoning=False,
        )
        terminal: Optional[LLMStreamEvent] = None
        recovery_active = False
        try:
            for event in events:
                recovery_active = recovery_active or event.tool_recovery is not None
                self._raise_if_projection_cancelled(
                    cancellation_probe, timeout=timeout, recovery_active=recovery_active
                )
                if event.kind == "content":
                    for chunk in accumulator.consume(event):
                        yield (
                            event if chunk == event.text else replace(event, text=chunk)
                        )
                else:
                    for _ in accumulator.consume(event):
                        pass
                    if event.kind == "terminal":
                        terminal = event
                    else:
                        yield event
                if accumulator.visible_limit_reached:
                    break
            self._raise_if_projection_cancelled(
                cancellation_probe, timeout=timeout, recovery_active=recovery_active
            )
            for chunk in accumulator.finish():
                self._raise_if_projection_cancelled(
                    cancellation_probe, timeout=timeout, recovery_active=recovery_active
                )
                yield LLMStreamEvent(kind="content", text=chunk)
            self._raise_if_projection_cancelled(
                cancellation_probe, timeout=timeout, recovery_active=recovery_active
            )
            # A terminal event tells a harness that this request has ended. In
            # particular, a visible cap must close the active transport before
            # the consumer stops reading on that event.
            close = getattr(events, "close", None)
            events = None
            if callable(close):
                close()
            result = accumulator.result()
            if recovery_active:
                self._raise_if_recovery_retired()
            yield LLMStreamEvent(
                kind="terminal",
                usage=result.usage,
                terminal_reason=result.terminal_reason,
                transport=result.transport,
                data={
                    **(
                        deepcopy(terminal.data)
                        if terminal is not None and terminal.data
                        else {}
                    ),
                    "tool_calls": list(result.tool_calls),
                    **(
                        {"finish_reason": terminal.data["finish_reason"]}
                        if terminal is not None and terminal.data is not None
                        else {}
                    ),
                },
                thinking_diagnostic=result.thinking_diagnostic,
                server_timings=result.server_timings,
                runtime_selection=result.runtime_selection,
                tool_recovery=result.tool_recovery,
            )
        except (
            LLMBackendError,
            LLMCancelledError,
            LLMTimeoutError,
            LLMRunawayError,
        ) as exc:
            self._set_last_terminal_reason(exc.terminal_reason)
            accumulator.fail_diagnostic(exc)
            raise
        except Exception as exc:
            accumulator.fail_diagnostic(exc)
            raise
        finally:
            try:
                close = getattr(events, "close", None)
                if callable(close):
                    close()
            finally:
                accumulator.finish_diagnostic()

    def configure_context_ceiling(self, num_ctx: Optional[int]) -> None:
        self._requested_context_ceiling = (
            self._positive_integer(num_ctx, "num_ctx") if num_ctx is not None else None
        )

    def warmup_context(
        self,
        *,
        num_ctx: Optional[int] = None,
        num_predict: int = 1,
    ) -> bool:
        del num_predict
        requested = num_ctx if num_ctx is not None else self._num_ctx
        self.configure_context_ceiling(requested)
        ready = self.is_ready()
        known = self._observed_context_length or self._model_config.context_length
        return bool(
            ready and (requested is None or known is None or requested <= known)
        )

    def is_ready(self) -> bool:
        return bool(self.health_status()["model_ready"])

    def health_status(self) -> Dict[str, Any]:
        # This observation describes the current refresh, not the last server
        # that happened to answer. Failed or missing metadata leaves it unknown.
        self._observed_context_length = None
        healthy = False
        model_ready = False
        model_available = False
        model_state: Optional[str] = None
        observed_slots: Optional[int] = None
        reason = "unreachable"
        try:
            payload = self._client.health()
            healthy = isinstance(payload, Mapping) and payload.get("status") == "ok"
            reason = "model_readiness_unverified" if healthy else "not_ready"
        except (LLMBackendError, LLMTimeoutError):
            pass
        if healthy:
            selected: Optional[Mapping[str, Any]] = None
            models = getattr(self._client, "models", None)
            if callable(models):
                try:
                    catalog = models()
                    entries = (
                        catalog.get("data") if isinstance(catalog, Mapping) else None
                    )
                    if isinstance(entries, list):
                        matches = [
                            entry
                            for entry in entries
                            if isinstance(entry, Mapping)
                            and (
                                entry.get("id") == self.model_name
                                or (
                                    isinstance(entry.get("aliases"), list)
                                    and self.model_name in entry["aliases"]
                                )
                            )
                        ]
                        if len(matches) == 1:
                            selected = matches[0]
                            model_available = True
                            status = selected.get("status")
                            if "status" in selected:
                                value = (
                                    status.get("value")
                                    if isinstance(status, Mapping)
                                    else None
                                )
                                model_state = (
                                    value
                                    if isinstance(value, str)
                                    and value
                                    in {
                                        "loaded",
                                        "loading",
                                        "unloaded",
                                        "failed",
                                        "sleeping",
                                    }
                                    else "unknown"
                                )
                            model_ready = model_state == "loaded"
                            reason = "ready" if model_ready else "model_not_loaded"
                        elif not matches:
                            reason = "model_not_found"
                        else:
                            reason = "model_alias_ambiguous"
                except (LLMBackendError, LLMTimeoutError):
                    reason = "catalog_unavailable"
            # A router /props request for an unloaded alias can invoke its
            # auto-load policy. Readiness must not initiate that transition.
            query_props = selected is not None and (model_state is None or model_ready)
            try:
                props = (
                    self._client.props(model=self.model_name) if query_props else None
                )
                alias = props.get("model_alias") if isinstance(props, Mapping) else None
                sleeping = (
                    props.get("is_sleeping") is True
                    if isinstance(props, Mapping)
                    else False
                )
                accepted_aliases = {self.model_name}
                if selected is not None:
                    if isinstance(selected.get("id"), str):
                        accepted_aliases.add(selected["id"])
                    aliases = selected.get("aliases")
                    if isinstance(aliases, list):
                        accepted_aliases.update(
                            item for item in aliases if isinstance(item, str)
                        )
                coherent_props = alias is None or (
                    isinstance(alias, str) and alias in accepted_aliases
                )
                if alias is not None and not coherent_props:
                    model_ready = False
                    reason = "model_metadata_mismatch"
                elif sleeping:
                    model_ready = False
                    reason = "model_sleeping"
                elif selected is not None and model_state is None:
                    model_ready = isinstance(alias, str) and alias in accepted_aliases
                    reason = "ready" if model_ready else "model_readiness_unverified"
                settings = (
                    props.get("default_generation_settings", {})
                    if isinstance(props, Mapping) and coherent_props and not sleeping
                    else {}
                )
                context = (
                    settings.get("n_ctx") if isinstance(settings, Mapping) else None
                )
                if (
                    isinstance(context, int)
                    and not isinstance(context, bool)
                    and context > 0
                ):
                    self._observed_context_length = context
                slots = props.get("total_slots") if isinstance(props, Mapping) else None
                if coherent_props and not sleeping and type(slots) is int and slots > 0:
                    observed_slots = slots
            except (LLMBackendError, LLMTimeoutError):
                pass
        return {
            "backend": "llamacpp",
            "role_id": self.role_id,
            "model": self.model_name,
            "healthy": healthy,
            "server_healthy": healthy,
            "backend_ready": model_ready,
            "model_available": model_available,
            "model_ready": model_ready,
            "model_state": model_state,
            "reason": reason,
            "loaded_context_length": self._observed_context_length,
            "configured_context_length": self._model_config.context_length,
            "observed_slots": observed_slots,
            "requested_context_ceiling": self._requested_context_ceiling
            or self._num_ctx,
            "num_predict": self._num_predict,
            "reasoning_requested": self._role_config.reasoning,
            "reasoning_capability_verified": False,
            "runtime_identity": self._model_config.runtime_identity,
            "process_lifecycle": "external",
            "model_loading": "server_owned",
        }

    def close(self) -> None:
        self._adapter_closed = True
        if self._owns_client:
            self._client.close()


LlamaAdapter = LlamaCppAdapter
