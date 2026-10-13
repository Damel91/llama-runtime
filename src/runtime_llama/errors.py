from pathlib import Path
import re
import tempfile

from runtime_llama.usage import LLMTransportAttempt


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
