from pathlib import Path
from copy import deepcopy
import re
import tempfile

from runtime_llama.usage import LLMToolRecoveryReport, LLMTransportAttempt


_HOME_PATH = re.compile(
    r"(?<![A-Za-z0-9])(?:/Users/[^/\s'\"]+|/home/[^/\s'\"]+)"
    r"(?:/[^\s'\"),;]*)*"
)


def sanitize_diagnostic(value: object, *, limit: int = 500) -> str:
    """Bound provider diagnostics and remove local environment paths."""
    text = str(value or "")
    prefixes = {str(Path.home()), str(Path.cwd()), tempfile.gettempdir()}
    for prefix in sorted((item for item in prefixes if item), key=len, reverse=True):
        text = text.replace(prefix, "<abs_path_redacted>")
    text = _HOME_PATH.sub("<abs_path_redacted>", text)
    return text[: max(1, int(limit))]


class LLMTimeoutError(TimeoutError):
    """Raised when a model call exceeds the configured runtime bound."""

    terminal_reason = "idle_timeout"

    def __init__(
        self,
        message: str,
        *,
        diagnostics: LLMTransportAttempt | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class LLMBackendError(RuntimeError):
    """Raised when a backend returns an error payload or invalid response."""

    terminal_reason = "backend_error"

    def __init__(
        self,
        message: str,
        *,
        diagnostics: LLMTransportAttempt | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class LLMToolRecoveryError(LLMBackendError):
    """A tool proposal could not be safely recovered in this invocation.

    ``last_proposal`` is retained solely for explicit, non-executable inspection.
    It never appears in exception text, repr or the ordinary diagnostic export.
    A failure does not modify the adapter or a later invocation's budget.
    """

    terminal_reason = "tool_protocol_error"

    def __init__(
        self,
        message: str = "tool_recovery_exhausted",
        *,
        tool_recovery: LLMToolRecoveryReport,
        last_proposal: object = None,
        diagnostics: LLMTransportAttempt | None = None,
    ) -> None:
        # Only a protocol reason belongs in automatic exception logging.
        safe_message = (
            message
            if message
            in {
                "tool_recovery_exhausted",
                "tool_protocol_error",
                "tool_recovery_unsafe",
                "tool_recovery_incomplete",
            }
            else "tool_protocol_error"
        )
        super().__init__(safe_message, diagnostics=diagnostics)
        self.tool_recovery = tool_recovery
        self._last_proposal = deepcopy(last_proposal)
        self.terminal_reason = (
            "tool_recovery_exhausted"
            if tool_recovery.exhausted
            else "tool_protocol_error"
        )

    @property
    def last_proposal(self) -> object:
        return deepcopy(self._last_proposal)

    def to_dict(self) -> dict:
        return {
            "terminal_reason": self.terminal_reason,
            "tool_recovery": self.tool_recovery.to_dict(),
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
        }


class LLMCancelledError(RuntimeError):
    """Raised after a consumer cancellation is acknowledged by the transport."""

    terminal_reason = "cancelled"

    def __init__(
        self,
        message: str = "Model generation was cancelled",
        *,
        diagnostics: LLMTransportAttempt | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class LLMRunawayError(RuntimeError):
    """Raised when bounded stream analysis detects a non-converging response."""

    terminal_reason = "runaway"

    def __init__(
        self,
        message: str = "Model generation entered a repeated-output loop",
        *,
        diagnostics: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})
