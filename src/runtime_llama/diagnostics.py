"""Opt-in, bounded thinking diagnostics owned by a single invocation.

No prompt, request, credential header or file is collected here. Consumers own
the lifetime and persistence of snapshots delivered to an explicit sink.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Mapping, TypeAlias
from uuid import uuid4


@dataclass(frozen=True)
class ThinkingDiagnosticsConfig:
    enabled: bool = False
    max_chars: int = 65536

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("thinking diagnostics enabled must be boolean")
        if (
            isinstance(self.max_chars, bool)
            or not isinstance(self.max_chars, int)
            or self.max_chars <= 0
        ):
            raise ValueError(
                "thinking diagnostics max_chars must be a positive integer"
            )

    @classmethod
    def from_value(
        cls,
        value: ThinkingDiagnosticsConfig | Mapping[str, Any] | None,
    ) -> ThinkingDiagnosticsConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or set(value) - {"enabled", "max_chars"}:
            raise ValueError("thinking diagnostics must contain enabled/max_chars")
        return cls(**dict(value))


@dataclass(frozen=True)
class ThinkingDiagnostic:
    """Final diagnostic snapshot; text is deliberately omitted from repr.

    ``text`` combines separated and inline thinking in arrival order, without
    inline tags. ``reasoning_text`` on the call result keeps its older meaning
    of separated reasoning only. Character limits affect this snapshot alone.
    """

    call_id: str
    role_id: str
    model: str
    text: str = field(repr=False)
    sources: tuple[str, ...]
    observed_chars: int
    captured_chars: int
    max_chars: int
    truncated: bool
    status: str
    terminal_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Explicit content projection for an opted-in consumer sink."""
        return {
            "call_id": self.call_id,
            "role_id": self.role_id,
            "model": self.model,
            "text": self.text,
            "sources": list(self.sources),
            "observed_chars": self.observed_chars,
            "captured_chars": self.captured_chars,
            "max_chars": self.max_chars,
            "truncated": self.truncated,
            "status": self.status,
            "terminal_reason": self.terminal_reason,
        }


ThinkingDiagnosticSink: TypeAlias = Callable[[ThinkingDiagnostic], None]

_CREDENTIAL = re.compile(
    r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]+"
    r"|\b(?:api[_-]?key|access[_-]?token|password)\b[\"']?\s*[:=]\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)"
)


def _diagnostic_status(reason: str) -> str:
    if reason in {"completed", "tool_calls"}:
        return "completed"
    if reason == "cancelled":
        return "cancelled"
    if reason == "runaway":
        return "runaway"
    if reason in {
        "interrupted",
        "early_eof",
        "length_cap",
        "visible_output_cap",
        "content_filter",
    }:
        return "interrupted"
    return "error"


class _ThinkingCapture:
    """Private mutable capture, never shared between calls on an adapter."""

    def __init__(
        self,
        config: ThinkingDiagnosticsConfig,
        *,
        role_id: str,
        model: str,
        sink: ThinkingDiagnosticSink | None = None,
        secrets: tuple[str, ...] = (),
    ) -> None:
        from runtime_llama.errors import sanitize_diagnostic

        if sink is not None and not callable(sink):
            raise ValueError("thinking sink must be callable")
        self._config = config
        self._call_id = str(uuid4())
        self._secrets = tuple(sorted(set(filter(None, secrets)), key=len, reverse=True))
        self._role_id = sanitize_diagnostic(self._redact(role_id), limit=80)
        self._model = sanitize_diagnostic(self._redact(model), limit=80)
        self._sink = sink
        self._parts: list[str] = []
        self._sources: list[str] = []
        self._observed = 0
        self._stored = 0
        self._final: ThinkingDiagnostic | None = None

    def observe(self, source: str, text: str) -> None:
        if not self._config.enabled or not text or self._final is not None:
            return
        if source not in self._sources:
            self._sources.append(source)
        self._observed += len(text)
        remaining = max(0, self._config.max_chars - self._stored)
        if remaining:
            part = text[:remaining]
            self._parts.append(part)
            self._stored += len(part)

    def _redact(self, text: str, *, truncated: bool = False) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "<credential_redacted>")
            if truncated:
                # The capture quota can split a credential across its boundary.
                for size in range(min(len(secret) - 1, len(text)), 0, -1):
                    if text.endswith(secret[:size]):
                        text = text[:-size] + "<credential_redacted>"
                        break
        return _CREDENTIAL.sub("<credential_redacted>", text)

    def finish(self, reason: str) -> ThinkingDiagnostic | None:
        if not self._config.enabled:
            return None
        if self._final is not None:
            return self._final
        truncated = self._observed > self._stored
        text = self._redact("".join(self._parts), truncated=truncated)
        truncated = truncated or len(text) > self._config.max_chars
        text = text[: self._config.max_chars]
        self._parts.clear()
        self._final = ThinkingDiagnostic(
            call_id=self._call_id,
            role_id=self._role_id,
            model=self._model,
            text=text,
            sources=tuple(self._sources),
            observed_chars=self._observed,
            captured_chars=len(text),
            max_chars=self._config.max_chars,
            truncated=truncated,
            status=_diagnostic_status(reason),
            terminal_reason=reason,
        )
        if self._sink is not None:
            try:
                self._sink(self._final)
            except Exception as exc:
                from runtime_llama.logging import get_logger

                # Diagnostics must never replace a successful answer or the
                # original typed model/transport error with a sink failure.
                get_logger("diagnostics").warning(
                    "thinking diagnostic sink failed | type=%s", type(exc).__name__
                )
        return self._final
