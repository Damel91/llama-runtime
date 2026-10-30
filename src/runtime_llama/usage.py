from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Dict, Optional, Protocol
from collections.abc import Mapping

from runtime_llama.diagnostics import ThinkingDiagnostic


class RuntimeSelection(Protocol):
    """Request-local routing metadata, without importing HTTP dependencies.

    Automatic adapters return the immutable ``RuntimeModelSelection`` catalog
    object. This structural contract keeps ordinary result annotations usable
    independently of discovery, including ``typing.get_type_hints``.
    """

    runtime: Any
    model: Any

    @property
    def backend(self) -> str: ...


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
class LLMToolRecoveryRejection:
    """Non-executable proof of a rejected proposal, without its arguments.

    Native names are intentionally absent from ``repr``. Consumers explicitly
    inspecting or persisting them still own the resulting diagnostic data.
    """

    name: str = field(repr=False)
    reason: str
    source: str = "strata_rejected_tool_calls"

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) for value in (self.name, self.reason, self.source)
        ):
            raise ValueError("tool_recovery_rejection_fields_must_be_strings")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "reason": self.reason, "source": self.source}


_RECOVERY_USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "visible_output_tokens",
)


@dataclass(frozen=True)
class LLMToolRecoveryAttempt:
    """One final native-generation snapshot: initial0, corrections1 through3.

    Usage is a content-free copy, not the mutable ``raw`` provider payload.
    Server timings are per generation, never an aggregate GPU measurement.
    """

    index: int
    outcome: str
    reasons: tuple[str, ...] = ()
    native_rejected_tool_calls: int | None = None
    usage: LLMUsageStats | None = None
    transport: LLMTransportAttempt | None = None
    server_timings: LLMServerTimings | None = None
    native_rejections: tuple[LLMToolRecoveryRejection, ...] = ()
    thinking_diagnostic: ThinkingDiagnostic | None = field(default=None, repr=False)
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.index) is not int or not 0 <= self.index <= 3:
            raise ValueError("tool_recovery_attempt_index_must_be_0_through_3")
        if not isinstance(self.outcome, str) or not self.outcome:
            raise ValueError("tool_recovery_attempt_outcome_required")
        if self.native_rejected_tool_calls is not None and (
            type(self.native_rejected_tool_calls) is not int
            or self.native_rejected_tool_calls < 0
        ):
            raise ValueError("tool_recovery_native_rejection_count_invalid")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        if any(
            not isinstance(value, str) for value in (*self.reasons, *self.provenance)
        ):
            raise ValueError("tool_recovery_reason_and_provenance_must_be_strings")
        rejected = tuple(self.native_rejections)
        if any(not isinstance(item, LLMToolRecoveryRejection) for item in rejected):
            raise ValueError("tool_recovery_native_rejections_invalid")
        object.__setattr__(self, "native_rejections", rejected)
        if self.usage is not None:
            values = {
                name: (
                    value
                    if type(value := getattr(self.usage, name)) is int and value >= 0
                    else None
                )
                for name in _RECOVERY_USAGE_FIELDS
            }
            object.__setattr__(
                self,
                "usage",
                LLMUsageStats(
                    **values, derived_fields=tuple(self.usage.derived_fields)
                ),
            )
        if self.server_timings is not None:
            object.__setattr__(
                self,
                "server_timings",
                replace(
                    self.server_timings,
                    derived_fields=tuple(self.server_timings.derived_fields),
                ),
            )
        if self.transport is not None:
            object.__setattr__(
                self,
                "transport",
                replace(self.transport, event_kinds=tuple(self.transport.event_kinds)),
            )
        if self.thinking_diagnostic is not None:
            object.__setattr__(
                self,
                "thinking_diagnostic",
                replace(
                    self.thinking_diagnostic,
                    sources=tuple(self.thinking_diagnostic.sources),
                ),
            )

    def to_dict(self, *, include_thinking: bool = False) -> Dict[str, Any]:
        value = {
            "index": self.index,
            "outcome": self.outcome,
            "reasons": list(self.reasons),
            "native_rejected_tool_calls": self.native_rejected_tool_calls,
            "native_rejections": [item.to_dict() for item in self.native_rejections],
            "provenance": list(self.provenance),
            "usage": (
                {name: getattr(self.usage, name) for name in _RECOVERY_USAGE_FIELDS}
                if self.usage is not None
                else None
            ),
            "transport": self.transport.to_dict() if self.transport else None,
            "server_timings": (
                self.server_timings.to_dict() if self.server_timings else None
            ),
        }
        if include_thinking and self.thinking_diagnostic is not None:
            value["thinking_diagnostic"] = self.thinking_diagnostic.to_dict()
        return value


@dataclass(frozen=True)
class LLMToolRecoveryReport:
    """Immutable invocation-local evidence, separate from accepted usage.

    ``total_usage`` sums final snapshots only when that field was observed on
    every attempt. ``observed_usage`` is a lower-bound sum of known snapshots;
    ``unknown_usage_fields`` explicitly identifies incomplete coverage.
    """

    attempts: tuple[LLMToolRecoveryAttempt, ...] = ()
    exhausted: bool = False
    elapsed_ms: float = 0.0

    def __post_init__(self) -> None:
        attempts = tuple(self.attempts)
        if any(not isinstance(item, LLMToolRecoveryAttempt) for item in attempts):
            raise ValueError("tool_recovery_attempts_invalid")
        if tuple(item.index for item in attempts) != tuple(range(len(attempts))):
            raise ValueError(
                "tool_recovery_attempts_must_start_at_0_and_be_consecutive"
            )
        if type(self.exhausted) is not bool:
            raise ValueError("tool_recovery_exhausted_must_be_boolean")
        if self.exhausted and (
            not attempts
            or attempts[-1].index != 3
            or attempts[-1].outcome == "accepted"
        ):
            raise ValueError(
                "tool_recovery_exhaustion_requires_failed_third_correction"
            )
        if (
            isinstance(self.elapsed_ms, bool)
            or not isinstance(self.elapsed_ms, (int, float))
            or not math.isfinite(self.elapsed_ms)
            or self.elapsed_ms < 0
        ):
            raise ValueError("tool_recovery_elapsed_ms_must_be_finite_nonnegative")
        object.__setattr__(self, "attempts", attempts)

    @property
    def corrections_used(self) -> int:
        return max(0, len(self.attempts) - 1)

    @property
    def max_corrections(self) -> int:
        return 3

    @property
    def unknown_usage_fields(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in _RECOVERY_USAGE_FIELDS
            if not self.attempts
            or any(
                attempt.usage is None or getattr(attempt.usage, name) is None
                for attempt in self.attempts
            )
        )

    @property
    def usage_complete(self) -> bool:
        return not self.unknown_usage_fields

    def _sum_usage(self, *, require_complete: bool) -> LLMUsageStats:
        fields = {}
        for name in _RECOVERY_USAGE_FIELDS:
            observed = [
                getattr(attempt.usage, name)
                for attempt in self.attempts
                if attempt.usage is not None
                and getattr(attempt.usage, name) is not None
            ]
            fields[name] = (
                sum(observed)
                if observed
                and (not require_complete or len(observed) == len(self.attempts))
                else None
            )
        return LLMUsageStats(**fields)

    @property
    def total_usage(self) -> LLMUsageStats:
        return self._sum_usage(require_complete=True)

    @property
    def observed_usage(self) -> LLMUsageStats:
        return self._sum_usage(require_complete=False)

    def to_dict(self, *, include_thinking: bool = False) -> Dict[str, Any]:
        total, observed = self.total_usage, self.observed_usage
        return {
            "attempts": [
                item.to_dict(include_thinking=include_thinking)
                for item in self.attempts
            ],
            "corrections_used": self.corrections_used,
            "max_corrections": self.max_corrections,
            "exhausted": self.exhausted,
            "elapsed_ms": self.elapsed_ms,
            "usage_complete": self.usage_complete,
            "unknown_usage_fields": list(self.unknown_usage_fields),
            "total_usage": {
                name: getattr(total, name) for name in _RECOVERY_USAGE_FIELDS
            },
            "observed_usage": {
                name: getattr(observed, name) for name in _RECOVERY_USAGE_FIELDS
            },
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
    runtime_selection: RuntimeSelection | None = None
    tool_recovery: LLMToolRecoveryReport | None = None


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
    runtime_selection: RuntimeSelection | None = None
    tool_recovery: LLMToolRecoveryReport | None = None
