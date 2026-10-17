from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from runtime_llama.diagnostics import ThinkingDiagnostic


@dataclass(frozen=True)
class LLMTransportAttempt:
    """Bounded, content-free evidence for one transport attempt."""

    elapsed_ms: float = 0.0
    idle_timeout_sec: float | None = 0.0
    terminal_reason: str = ""
    terminal_event_seen: bool = False
    event_count: int = 0
    event_kinds: tuple[str, ...] = ()
    role_id: str = ""
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elapsed_ms": round(max(0.0, float(self.elapsed_ms)), 3),
            "idle_timeout_sec": (
                None
                if self.idle_timeout_sec is None
                else max(0.0, float(self.idle_timeout_sec))
            ),
            "terminal_reason": str(self.terminal_reason or ""),
            "terminal_event_seen": bool(self.terminal_event_seen),
            "event_count": max(0, int(self.event_count)),
            "event_kinds": list(self.event_kinds),
            "role_id": str(self.role_id or ""),
            "model": str(self.model or ""),
        }


@dataclass(frozen=True)
class LLMUsageStats:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    visible_output_tokens: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None
    derived_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMStreamEvent:
    """Normalized event emitted by the HTTP LLM transport.

    ``text`` is a bounded transport delta, never a persisted diagnostic.  The
    text-only stream API projects visible text; metadata streams retain events.
    """

    kind: str
    text: str = ""
    usage: Optional[LLMUsageStats] = None
    terminal_reason: str = ""
    transport: Optional[LLMTransportAttempt] = None
    data: Optional[Dict[str, Any]] = None
    thinking_diagnostic: Optional[ThinkingDiagnostic] = None


@dataclass(frozen=True)
class LLMCallResult:
    text: str
    usage: Optional[LLMUsageStats] = None
    role_id: str = ""
    model: str = ""
    reasoning_text: str = ""
    terminal_reason: str = "completed"
    transport: Optional[LLMTransportAttempt] = None
    tool_calls: tuple[Dict[str, Any], ...] = ()
    thinking_diagnostic: Optional[ThinkingDiagnostic] = None
