import logging

from runtime_llama.diagnostics import ThinkingDiagnostic


LOGGER_NAMESPACE = "runtime_llama"


def get_logger(component: str | None = None) -> logging.Logger:
    """Return a standard logger without configuring consumer handlers."""
    if not component:
        return logging.getLogger(LOGGER_NAMESPACE)
    return logging.getLogger(f"{LOGGER_NAMESPACE}.{component}")


def log_thinking_diagnostic(diagnostic: ThinkingDiagnostic) -> None:
    """Explicit full-content sink; consumer handlers own any persistence.

    Passing this function as ``thinking_sink`` is an additional conscious choice
    beyond enabling capture. Ordinary adapter logging only emits character counts.
    """
    get_logger("diagnostics").info("thinking diagnostic | %s", diagnostic.to_dict())
