from abc import ABC, abstractmethod
import math
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    TypeAlias,
)

from runtime_llama.diagnostics import (
    ThinkingDiagnostic,
    ThinkingDiagnosticsConfig,
    ThinkingDiagnosticSink,
    _ThinkingCapture,
)
from runtime_llama.errors import (
    LLMBackendError,
    LLMCancelledError,
    LLMRunawayError,
    LLMTimeoutError,
    sanitize_diagnostic,
)
from runtime_llama.runaway import RunawayDetector
from runtime_llama.usage import (
    LLMCallResult,
    LLMStreamEvent,
    LLMServerTimings,
    LLMTransportAttempt,
    LLMUsageStats,
)
from runtime_llama.embedding_preparation import PreparedEmbeddingInput
from runtime_llama.embedding_stream import EmbeddingWindowStream
from runtime_llama.logging import get_logger
from runtime_llama.mcp import (
    MCPCallResult,
    MCPIntegration,
    MCPStreamEvent,
)


CancellationProbe: TypeAlias = Callable[[], bool]


class _ThinkingStreamStripper:
    def __init__(
        self,
        adapter: "BaseLLMAdapter",
        on_thinking: Optional[Callable[[str], None]] = None,
    ):
        self._adapter = adapter
        self._on_thinking = on_thinking
        self._buffer = ""
        self._thinking_chars = 0
        self._in_think = False

    def _observe_thinking(self, text: str) -> None:
        self._thinking_chars += len(text)
        if self._on_thinking is not None:
            self._on_thinking(text)

    def _finish_thinking(self) -> None:
        self._adapter._log_thinking_chars(self._thinking_chars)
        self._thinking_chars = 0

    @staticmethod
    def _safe_suffix(text: str, marker: str) -> int:
        max_len = min(len(text), len(marker) - 1)
        for size in range(max_len, 0, -1):
            if marker.startswith(text[-size:]):
                return size
        return 0

    def feed(self, token: str) -> Iterator[str]:
        self._buffer += token
        while self._buffer:
            if self._in_think:
                end = self._buffer.find("</think>")
                if end >= 0:
                    self._observe_thinking(self._buffer[:end])
                    self._finish_thinking()
                    self._buffer = self._buffer[end + len("</think>") :]
                    self._in_think = False
                    continue
                keep = self._safe_suffix(self._buffer, "</think>")
                if keep:
                    self._observe_thinking(self._buffer[:-keep])
                    self._buffer = self._buffer[-keep:]
                    break
                self._observe_thinking(self._buffer)
                self._buffer = ""
                break

            start = self._buffer.find("<think>")
            if start >= 0:
                visible = self._buffer[:start]
                self._buffer = self._buffer[start + len("<think>") :]
                self._in_think = True
                if visible:
                    yield visible
                continue
            keep = self._safe_suffix(self._buffer, "<think>")
            if keep:
                visible = self._buffer[:-keep]
                self._buffer = self._buffer[-keep:]
                if visible:
                    yield visible
                break
            visible = self._buffer
            self._buffer = ""
            if visible:
                yield visible

    def flush(self) -> Iterator[str]:
        if self._in_think:
            if self._buffer:
                self._observe_thinking(self._buffer)
                self._buffer = ""
            self._finish_thinking()
            self._in_think = False
            return
        if self._buffer:
            yield self._buffer
            self._buffer = ""


class _StreamAccumulator:
    """Project normalized transport events while retaining bounded metadata."""

    def __init__(
        self,
        adapter: "BaseLLMAdapter",
        *,
        visible_output_token_limit: Optional[int] = None,
        thinking_diagnostics: Optional[ThinkingDiagnosticsConfig] = None,
        thinking_sink: Optional[ThinkingDiagnosticSink] = None,
        collect_reasoning: bool = True,
    ) -> None:
        self._adapter = adapter
        configured = getattr(
            getattr(adapter, "_generation_config", None), "thinking_diagnostics", None
        )
        capture_config = ThinkingDiagnosticsConfig.from_value(
            configured if thinking_diagnostics is None else thinking_diagnostics
        )
        self._capture_enabled = capture_config.enabled
        self._capture = _ThinkingCapture(
            capture_config,
            role_id=adapter.role_id,
            model=adapter.model_name,
            sink=thinking_sink,
            secrets=getattr(adapter, "_diagnostic_secrets", ()),
        )
        self._stripper = adapter._new_thinking_stream_stripper(
            (lambda text: self._capture.observe("inline", text))
            if self._capture_enabled
            else None
        )
        self._detector = adapter._new_runaway_detector()
        self._visible_char_limit = adapter._visible_output_char_limit(
            visible_output_token_limit
        )
        self._visible_chars = 0
        self._visible_limit_reached = False
        self._reasoning_parts: List[str] = []
        self._collect_reasoning = collect_reasoning
        self._reasoning_chars = 0
        self._usage: Optional[LLMUsageStats] = None
        self._transport = None
        self._server_timings: Optional[LLMServerTimings] = None
        self._terminal_reason = "completed"
        self._tool_calls: tuple[Dict[str, Any], ...] = ()

    def consume(self, event: LLMStreamEvent) -> Iterator[str]:
        kind = str(event.kind or "")
        if event.usage is not None:
            self._usage = event.usage
        if event.transport is not None:
            self._transport = event.transport
        if event.server_timings is not None:
            self._server_timings = event.server_timings

        if kind in {"content", "reasoning"}:
            text = str(event.text or "")
            # Capture the received delta before runaway rejection, without
            # projecting rejected content into the visible response.
            if kind == "reasoning":
                if self._capture_enabled:
                    self._capture.observe("separated", text)
                projected = []
            else:
                # Resolve the complete delta before the guard: opted-in inline
                # diagnostics also observe rejected text, but none of its
                # visible fragments may escape a runaway decision.
                projected = list(self._stripper.feed(text))
            diagnostics = self._detector.observe(kind, text)
            if diagnostics is not None:
                raise LLMRunawayError(diagnostics=diagnostics.to_dict())
            if kind == "reasoning":
                if text:
                    self._reasoning_chars += len(text)
                    if self._collect_reasoning:
                        self._reasoning_parts.append(text)
                return
            if text:
                yield from self._apply_visible_limit(iter(projected))
            return

        if kind == "terminal":
            self._terminal_reason = str(event.terminal_reason or "completed")
            if event.data and "tool_calls" in event.data:
                from copy import deepcopy

                self._tool_calls = tuple(deepcopy(event.data["tool_calls"]))

    def finish(self) -> Iterator[str]:
        if not self._visible_limit_reached:
            yield from self._apply_visible_limit(self._stripper.flush())
        if self._reasoning_chars:
            self._adapter._log_thinking_chars(self._reasoning_chars)

    def finish_diagnostic(
        self, reason: str = "interrupted"
    ) -> Optional[ThinkingDiagnostic]:
        # Flush dangling inline thinking even when a consumer closes the stream
        # or an exception prevents normal visible-output finalization.
        for _ in self._stripper.flush():
            pass
        return self._capture.finish(reason)

    def fail_diagnostic(self, exc: BaseException) -> None:
        reason = str(getattr(exc, "terminal_reason", "invocation_error"))
        transport = getattr(exc, "diagnostics", None)
        if getattr(transport, "terminal_reason", "") == "early_eof":
            reason = "early_eof"
        diagnostic = self.finish_diagnostic(reason)
        if diagnostic is not None:
            exc.thinking_diagnostic = diagnostic

    def _apply_visible_limit(self, chunks: Iterator[str]) -> Iterator[str]:
        for chunk in chunks:
            if not chunk:
                continue
            if self._visible_char_limit is None:
                self._visible_chars += len(chunk)
                yield chunk
                continue
            remaining = max(0, self._visible_char_limit - self._visible_chars)
            if remaining <= 0:
                self._visible_limit_reached = True
                return
            if len(chunk) <= remaining:
                self._visible_chars += len(chunk)
                yield chunk
                continue
            self._visible_chars = self._visible_char_limit
            self._visible_limit_reached = True
            yield chunk[:remaining]
            return

    @property
    def visible_limit_reached(self) -> bool:
        return self._visible_limit_reached

    def result(self) -> LLMCallResult:
        if self._usage is not None:
            self._adapter._set_last_usage(self._usage)
        self._adapter._set_last_server_timings(self._server_timings)
        terminal_reason = (
            "visible_output_cap"
            if self._visible_limit_reached
            else self._terminal_reason
        )
        self._adapter._set_last_terminal_reason(terminal_reason)
        return LLMCallResult(
            text="",  # populated by BaseLLMAdapter.invoke_with_ctx
            usage=self._usage,
            role_id=self._adapter.role_id,
            model=self._adapter.model_name,
            reasoning_text="".join(self._reasoning_parts),
            terminal_reason=terminal_reason,
            transport=self._transport,
            tool_calls=self._tool_calls,
            thinking_diagnostic=self.finish_diagnostic(terminal_reason),
            server_timings=self._server_timings,
        )


class BaseLLMAdapter(ABC):
    """
    Normalized interface for external llama.cpp role adapters.

    Consumers can depend on this contract without owning transport details.
    All LLM interaction goes through this interface.
    """

    def invoke(self, messages: List) -> str:
        """Blocking call realized by consuming the normalized stream once."""
        return self.invoke_with_ctx(messages).text

    def stream(self, messages: List) -> Iterator[str]:
        """Streaming call. Yields visible response tokens one at a time."""
        return self.stream_with_ctx(messages)

    def stream_with_ctx(
        self,
        messages: List,
        *,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        output_format: Optional[str] = None,
        timeout: Optional[float] = None,
        cancellation_probe: Optional[CancellationProbe] = None,
        request_options: Optional[Dict[str, Any]] = None,
        thinking_diagnostics: Optional[ThinkingDiagnosticsConfig] = None,
        thinking_sink: Optional[ThinkingDiagnosticSink] = None,
    ) -> Iterator[str]:
        self._set_last_usage(None)
        self._set_last_server_timings(None)
        self._set_last_terminal_reason("")
        accumulator = _StreamAccumulator(
            self,
            visible_output_token_limit=num_predict,
            thinking_diagnostics=thinking_diagnostics,
            thinking_sink=thinking_sink,
            collect_reasoning=False,
        )
        events = None
        try:
            self._raise_if_cancelled(cancellation_probe, timeout=timeout)
            events = self._iter_stream_events_with_ctx(
                messages,
                num_ctx=num_ctx,
                num_predict=num_predict,
                output_format=output_format,
                timeout=timeout,
                cancellation_probe=cancellation_probe,
                **(
                    {"request_options": request_options}
                    if request_options is not None
                    else {}
                ),
            )
            for event in events:
                self._raise_if_cancelled(cancellation_probe, timeout=timeout)
                for chunk in accumulator.consume(event):
                    self._raise_if_cancelled(cancellation_probe, timeout=timeout)
                    yield chunk
                if accumulator.visible_limit_reached:
                    break
            self._raise_if_cancelled(cancellation_probe, timeout=timeout)
            for chunk in accumulator.finish():
                self._raise_if_cancelled(cancellation_probe, timeout=timeout)
                yield chunk
            self._raise_if_cancelled(cancellation_probe, timeout=timeout)
            close = getattr(events, "close", None)
            events = None
            if callable(close):
                close()
            accumulator.result()
        except (
            LLMBackendError,
            LLMCancelledError,
            LLMTimeoutError,
            LLMRunawayError,
        ) as exc:
            self._set_last_terminal_reason(str(getattr(exc, "terminal_reason", "")))
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

    def chat(self, messages: List) -> str:
        """Structured-message call. Default implementation delegates to invoke."""
        return self.invoke(messages)

    def invoke_with_ctx(
        self,
        messages: List,
        *,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        output_format: Optional[str] = None,
        timeout: Optional[float] = None,
        cancellation_probe: Optional[CancellationProbe] = None,
        request_options: Optional[Dict[str, Any]] = None,
        thinking_diagnostics: Optional[ThinkingDiagnosticsConfig] = None,
        thinking_sink: Optional[ThinkingDiagnosticSink] = None,
    ) -> LLMCallResult:
        self._set_last_usage(None)
        self._set_last_server_timings(None)
        self._set_last_terminal_reason("")
        accumulator = _StreamAccumulator(
            self,
            visible_output_token_limit=num_predict,
            thinking_diagnostics=thinking_diagnostics,
            thinking_sink=thinking_sink,
        )
        visible_parts: List[str] = []
        events = None
        try:
            self._raise_if_cancelled(cancellation_probe, timeout=timeout)
            events = self._iter_stream_events_with_ctx(
                messages,
                num_ctx=num_ctx,
                num_predict=num_predict,
                output_format=output_format,
                timeout=timeout,
                cancellation_probe=cancellation_probe,
                **(
                    {"request_options": request_options}
                    if request_options is not None
                    else {}
                ),
            )
            for event in events:
                self._raise_if_cancelled(cancellation_probe, timeout=timeout)
                visible_parts.extend(accumulator.consume(event))
                if accumulator.visible_limit_reached:
                    break
            # The blocking contract still strips inline thinking from visible
            # text, exactly as the old per-adapter implementations did.
            self._raise_if_cancelled(cancellation_probe, timeout=timeout)
            visible_parts.extend(accumulator.finish())
            self._raise_if_cancelled(cancellation_probe, timeout=timeout)
            close = getattr(events, "close", None)
            events = None
            if callable(close):
                close()
            result = accumulator.result()
            return LLMCallResult(
                text="".join(visible_parts),
                usage=result.usage,
                role_id=result.role_id,
                model=result.model,
                reasoning_text=result.reasoning_text,
                terminal_reason=result.terminal_reason,
                transport=result.transport,
                tool_calls=result.tool_calls,
                thinking_diagnostic=result.thinking_diagnostic,
                server_timings=result.server_timings,
            )
        except (
            LLMBackendError,
            LLMCancelledError,
            LLMTimeoutError,
            LLMRunawayError,
        ) as exc:
            self._set_last_terminal_reason(str(getattr(exc, "terminal_reason", "")))
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

    def stream_mcp(
        self,
        messages: List,
        *,
        integrations: Sequence[MCPIntegration],
        store: bool = False,
        previous_response_id: Optional[str] = None,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        output_format: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Iterator[MCPStreamEvent]:
        del (
            messages,
            integrations,
            store,
            previous_response_id,
            num_ctx,
            num_predict,
            output_format,
            timeout,
        )
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support MCP invocation"
        )

    def invoke_mcp(
        self,
        messages: List,
        *,
        integrations: Sequence[MCPIntegration],
        store: bool = False,
        previous_response_id: Optional[str] = None,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        output_format: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> MCPCallResult:
        del (
            messages,
            integrations,
            store,
            previous_response_id,
            num_ctx,
            num_predict,
            output_format,
            timeout,
        )
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support MCP invocation"
        )

    def invoke_result(self, messages: List) -> LLMCallResult:
        return self.invoke_with_ctx(messages)

    def get_last_usage(self) -> Optional[LLMUsageStats]:
        return getattr(self, "_last_usage", None)

    def get_last_server_timings(self) -> Optional[LLMServerTimings]:
        return getattr(self, "_last_server_timings", None)

    def get_last_terminal_reason(self) -> str:
        return str(getattr(self, "_last_terminal_reason", "") or "")

    def _set_last_usage(self, usage: Optional[LLMUsageStats]) -> None:
        self._last_usage = usage

    def _set_last_server_timings(self, timings: Optional[LLMServerTimings]) -> None:
        self._last_server_timings = timings

    def _set_last_terminal_reason(self, reason: str) -> None:
        self._last_terminal_reason = str(reason or "")

    def _raise_if_cancelled(
        self,
        cancellation_probe: Optional[CancellationProbe],
        *,
        timeout: Optional[float] = None,
        diagnostics: Optional[LLMTransportAttempt] = None,
    ) -> None:
        if cancellation_probe is None:
            return
        try:
            requested = bool(cancellation_probe())
        except Exception as exc:
            raise LLMBackendError(
                "llm_cancellation_probe_failed:"
                f"{sanitize_diagnostic(type(exc).__name__, limit=80)}"
            ) from RuntimeError(type(exc).__name__)
        if not requested:
            return
        evidence = diagnostics or LLMTransportAttempt(
            idle_timeout_sec=max(0.0, float(timeout or 0.0)),
            terminal_reason="cancelled",
            role_id=sanitize_diagnostic(self.role_id, limit=80),
            model=sanitize_diagnostic(self.model_name, limit=80),
        )
        raise LLMCancelledError(diagnostics=evidence)

    def _new_runaway_detector(self) -> RunawayDetector:
        profile = self.role_profile
        return RunawayDetector(
            min_chars=int(getattr(profile, "runaway_min_chars", 512) or 512),
            window_chars=int(getattr(profile, "runaway_window_chars", 4096) or 4096),
            block_chars=int(getattr(profile, "runaway_block_chars", 128) or 128),
            repeat_threshold=int(getattr(profile, "runaway_repeat_threshold", 4) or 4),
            ngram_chars=int(getattr(profile, "runaway_ngram_chars", 24) or 24),
        )

    def _visible_output_char_limit(self, token_limit: Optional[int]) -> Optional[int]:
        profile = self.role_profile
        if token_limit is not None:
            effective_limit = token_limit
        else:
            profile_limit = getattr(profile, "num_predict", None)
            effective_limit = (
                profile_limit
                if profile_limit is not None
                else getattr(self, "_num_predict", None)
            )
        if effective_limit is None:
            return None
        parsed = int(effective_limit)
        if parsed <= 0:
            return None
        ratio = float(getattr(profile, "runtime_char_token_ratio_seed", None) or 0.33)
        if ratio <= 0:
            ratio = 0.33
        return max(1, int(math.ceil(parsed / ratio)))

    def _iter_stream_events_with_ctx(
        self,
        messages: List,
        *,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        output_format: Optional[str] = None,
        timeout: Optional[float] = None,
        cancellation_probe: Optional[CancellationProbe] = None,
        request_options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[LLMStreamEvent]:
        """Private transport seam implemented by built-in adapters.

        Legacy third-party/test adapters that override ``stream`` continue to
        work through a visible-text compatibility projection. They simply do
        not gain separated reasoning or terminal metadata until migrated.
        """
        del num_ctx, num_predict, output_format, request_options
        self._raise_if_cancelled(cancellation_probe, timeout=timeout)
        if type(self).stream is not BaseLLMAdapter.stream:
            stream = self.stream(messages)
            try:
                for token in stream:
                    self._raise_if_cancelled(cancellation_probe, timeout=timeout)
                    yield LLMStreamEvent(kind="content", text=str(token or ""))
                self._raise_if_cancelled(cancellation_probe, timeout=timeout)
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            yield LLMStreamEvent(kind="terminal", terminal_reason="completed")
            return
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _iter_stream_events_with_ctx"
        )

    @property
    def role_id(self) -> str:
        return ""

    @property
    def model_name(self) -> str:
        return ""

    @property
    def role_profile(self):
        return None

    def load(self) -> bool:
        return self.is_ready()

    def warmup_context(
        self,
        *,
        num_ctx: Optional[int] = None,
        num_predict: int = 1,
    ) -> bool:
        return self.load()

    def configure_context_ceiling(self, num_ctx: Optional[int]) -> None:
        return None

    @abstractmethod
    def is_ready(self) -> bool:
        """Health check. Returns True if the model backend is reachable."""

    @abstractmethod
    def health_status(self) -> Dict[str, Any]:
        """Detailed health for backend reachability and bounded generation."""

    def _message_role(self, message: Any) -> str:
        role = ""
        if isinstance(message, dict):
            role = str(message.get("role") or "")
        else:
            role = str(
                getattr(message, "role", "") or getattr(message, "type", "") or ""
            )
        role = role.lower()
        if role in {"human", "user"}:
            return "user"
        if role in {"ai", "assistant"}:
            return "assistant"
        if role == "system":
            return "system"
        if role == "tool":
            return "tool"
        return "user"

    def _message_content(self, message: Any) -> Any:
        if isinstance(message, dict):
            return message.get("content", "")
        return getattr(message, "content", str(message))

    def _chat_messages(self, messages: List) -> List[Dict[str, Any]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return [
            {
                "role": self._message_role(message),
                "content": self._message_content(message),
            }
            for message in (messages or [])
        ]

    def _log_thinking(self, body: str) -> None:
        self._log_thinking_chars(len(body))

    def _log_thinking_chars(self, count: int) -> None:
        if count:
            get_logger(self.__class__.__name__).info("thinking | chars=%d", count)

    def _strip_thinking_text(self, raw_text: str) -> str:
        if "<think>" not in raw_text:
            return raw_text
        import re

        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        for match in pattern.finditer(raw_text):
            self._log_thinking(match.group(1).strip())
        cleaned = pattern.sub("", raw_text)
        if "<think>" in cleaned:
            before, dangling = cleaned.split("<think>", 1)
            self._log_thinking(dangling.strip())
            cleaned = before
        return cleaned.strip()

    def _new_thinking_stream_stripper(
        self, on_thinking: Optional[Callable[[str], None]] = None
    ) -> _ThinkingStreamStripper:
        return _ThinkingStreamStripper(self, on_thinking)

    @staticmethod
    def _extract_content_and_reasoning(payload: Any) -> tuple[str, str]:
        """Return ``(content, reasoning_content)`` from a structured message.

        Thinking-aware transports may split
        reasoning tokens into ``reasoning_content`` while leaving ``content``
        empty until the visible answer begins. When the model exhausts its
        budget mid-reasoning, ``content`` stays empty and downstream parsers
        see nothing. Surfacing ``reasoning_content`` as a fallback keeps the
        contract aligned with adapters that strip an inline ``<think>`` block
        and yield the remaining body.

        Defined on the normalized contract so transport revisions retain the
        same projection.
        """
        if payload is None:
            return "", ""
        if isinstance(payload, dict):
            content = str(payload.get("content") or "")
            reasoning = str(payload.get("reasoning_content") or "")
            return content, reasoning
        content = str(getattr(payload, "content", "") or "")
        reasoning = str(getattr(payload, "reasoning_content", "") or "")
        return content, reasoning


class LLMAdapter(BaseLLMAdapter):
    """Backward-compatible name for the shared adapter contract."""

    pass


class EmbeddingAdapter(ABC):
    """Abstract interface for embedding backends."""

    role_id: str
    model_name: str
    embedding_dim: int
    max_input_tokens: Optional[int]

    @abstractmethod
    def embed_texts(
        self,
        texts: List[str],
        *,
        input_type: Literal["document", "query", "raw"] = "document",
    ) -> List[List[float]]:
        """Embed documents or queries using model prefixes; raw skips preparation."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed one query using the model's configured query prefix."""

    def prepare_texts(
        self,
        texts: Sequence[str],
        *,
        input_type: Literal["document", "query", "raw"] = "document",
        timeout: Optional[float] = None,
        cancel: Optional[CancellationProbe] = None,
    ) -> tuple[PreparedEmbeddingInput, ...]:
        """Return exact native counts and current limits, including oversize inputs."""
        raise NotImplementedError

    def prepare_query(
        self,
        text: str,
        *,
        timeout: Optional[float] = None,
        cancel: Optional[CancellationProbe] = None,
    ) -> PreparedEmbeddingInput:
        return self.prepare_texts(
            [text], input_type="query", timeout=timeout, cancel=cancel
        )[0]

    def embed_prepared(
        self,
        inputs: Sequence[PreparedEmbeddingInput],
        *,
        timeout: Optional[float] = None,
        cancel: Optional[CancellationProbe] = None,
    ) -> List[List[float]]:
        """Embed issued preparations after fresh identity and limit admission."""
        raise NotImplementedError

    def load(self) -> bool:
        return self.is_ready()

    def astream_texts(
        self,
        windows: AsyncIterable[Sequence[str]],
        *,
        input_type: Literal["document", "query", "raw"] = "document",
        max_pending_windows: int = 2,
        timeout: Optional[float] = None,
        cancel: Optional[CancellationProbe] = None,
        close_source: bool = False,
    ) -> EmbeddingWindowStream:
        """Optional bounded asynchronous window API; sync implementations remain valid."""
        raise NotImplementedError

    def astream_prepared(
        self,
        windows: AsyncIterable[Sequence[PreparedEmbeddingInput]],
        *,
        max_pending_windows: int = 2,
        timeout: Optional[float] = None,
        cancel: Optional[CancellationProbe] = None,
        close_source: bool = False,
    ) -> EmbeddingWindowStream:
        """Optional asynchronous intake of already prepared windows."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Optional async lifecycle without adding abstract requirements."""
        import asyncio

        close = getattr(self, "close", None)
        if callable(close):
            await asyncio.to_thread(close)

    @abstractmethod
    def is_ready(self) -> bool:
        """Health check. Returns True if embedding backend is reachable."""

    @abstractmethod
    def health_status(self) -> Dict[str, Any]:
        """Detailed embedding backend health."""
