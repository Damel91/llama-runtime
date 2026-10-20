from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional
from collections.abc import Mapping

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
class LLMServerTimings:
    """One native server timing snapshot, separate from usage and HTTP wall time.

    Rates are in tokens/second and durations in milliseconds. Missing or invalid
    native fields stay unknown. A derived rate uses only counts and durations
    from this same snapshot, never usage counters or another transport event.
    """

    prompt_n: Optional[int] = None
    prompt_ms: Optional[float] = None
    prompt_per_second: Optional[float] = None
    predicted_n: Optional[int] = None
    predicted_ms: Optional[float] = None
    predicted_per_second: Optional[float] = None
    cache_n: Optional[int] = None
    derived_fields: tuple[str, ...] = ()

    @classmethod
    def from_native(cls, value: Any) -> Optional["LLMServerTimings"]:
        if not isinstance(value, Mapping):
            return None

        def count(name: str) -> Optional[int]:
            item = value.get(name)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                return None
            return item

        def number(name: str) -> Optional[float]:
            item = value.get(name)
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return None
            try:
                result = float(item)
            except (OverflowError, ValueError):
                return None
            return result if math.isfinite(result) and result >= 0 else None

        fields = {
            "prompt_n": count("prompt_n"),
            "prompt_ms": number("prompt_ms"),
            "prompt_per_second": number("prompt_per_second"),
            "predicted_n": count("predicted_n"),
            "predicted_ms": number("predicted_ms"),
            "predicted_per_second": number("predicted_per_second"),
            "cache_n": count("cache_n"),
        }
        derived = []
        # Keep the reported rate when available. A missing rate may be derived
        # from one coherent native count/duration pair; HTTP elapsed is unrelated.
        for prefix in ("prompt", "predicted"):
            rate_key = f"{prefix}_per_second"
            tokens, duration = fields[f"{prefix}_n"], fields[f"{prefix}_ms"]
            if fields[rate_key] is None and tokens is not None and duration:
                try:
                    rate = tokens * 1000 / duration
                except OverflowError:
                    continue
                if math.isfinite(rate):
                    fields[rate_key] = rate
                    derived.append(rate_key)
        if all(item is None for item in fields.values()):
            return None
        return cls(**fields, derived_fields=tuple(derived))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_n": self.prompt_n,
            "prompt_ms": self.prompt_ms,
            "prompt_per_second": self.prompt_per_second,
            "predicted_n": self.predicted_n,
            "predicted_ms": self.predicted_ms,
            "predicted_per_second": self.predicted_per_second,
            "cache_n": self.cache_n,
            "derived_fields": list(self.derived_fields),
        }


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
    server_timings: Optional[LLMServerTimings] = None


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
    server_timings: Optional[LLMServerTimings] = None
