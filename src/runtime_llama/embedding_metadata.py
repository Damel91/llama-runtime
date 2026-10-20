"""Loaded native execution parameters, distinct from HTTP request budgets."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from runtime_llama.errors import LLMBackendError

_DECIMAL = re.compile(r"[0-9]+\Z")
# These native options consume one argv value even when that value resembles an
# option. In particular a model basename is never a batch-size declaration.
_VALUE_OPTIONS = frozenset(
    {
        "--model",
        "-m",
        "--alias",
        "-a",
        "--mmproj",
        "--tags",
        "--device",
        "--host",
        "--port",
        "--log-colors",
        "--ctx-size",
        "-c",
        "--parallel",
        "-np",
        "--cache-ram",
        "--cache-type-k",
        "--cache-type-v",
        "--flash-attn",
        "-fa",
        "--fit",
        "--load-mode",
        "--n-gpu-layers",
        "-ngl",
        "--split-mode",
        "--tensor-split",
        "-ts",
        "--model-draft",
        "--n-gpu-layers-draft",
        "--spec-draft-n-max",
        "--spec-type",
        "--lora",
        "--override-kv",
        "--chat-template",
        "--chat-template-file",
        "--api-key",
        "--api-key-file",
        "--threads",
        "-t",
        "--threads-batch",
        "-tb",
        "--pooling",
        "--rope-scaling",
        "--rope-freq-base",
        "--rope-freq-scale",
    }
)

_BATCH_OPTIONS = {
    "--batch-size": "n_batch",
    "-b": "n_batch",
    "--ubatch-size": "n_ubatch",
    "-ub": "n_ubatch",
}


@dataclass(frozen=True)
class LoadedExecutionMetadata:
    """Native execution observations; none is an aggregate HTTP token limit.

    A slot count describes loaded capacity, not currently idle slots. Native
    batch and microbatch sizes do not establish whether a model can split an
    individual input, and no such capability is inferred here.
    """

    n_batch: int | None = None
    n_ubatch: int | None = None
    total_slots: int | None = None


def _positive(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LLMBackendError("llamacpp_embedding_invalid_loaded_batch_limit")
    return value


def loaded_execution_metadata(props: Any, entry: Any = None) -> LoadedExecutionMetadata:
    """Read native batch parameters from exact supported loaded observations.

    The caller verifies the selected entry's identity and loaded state. Presets,
    paths and training context are not execution observations. Missing values
    stay unknown; conflicting or malformed supported batch values are errors.
    A missing or malformed slot count cannot authorize additional concurrency.
    """
    observations: dict[str, list[int]] = {"n_batch": [], "n_ubatch": []}
    total_slots = None
    if isinstance(props, dict):
        for name in observations:
            if name in props:
                observations[name].append(_positive(props[name]))
        settings = props.get("default_generation_settings")
        if isinstance(settings, dict):
            for name in observations:
                if name in settings:
                    observations[name].append(_positive(settings[name]))
        candidate = props.get("total_slots")
        if (
            isinstance(candidate, int)
            and not isinstance(candidate, bool)
            and candidate > 0
        ):
            total_slots = candidate
    status = entry.get("status") if isinstance(entry, dict) else None
    if isinstance(status, dict) and "args" in status:
        args = status["args"]
        if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
            raise LLMBackendError("llamacpp_embedding_invalid_loaded_batch_limit")
        found: dict[str, list[int]] = {"n_batch": [], "n_ubatch": []}
        index = 0
        while index < len(args):
            arg = args[index]
            if arg == "--":
                break
            if arg in _VALUE_OPTIONS:
                index += 2
                continue
            value, name = None, _BATCH_OPTIONS.get(arg)
            if name is not None:
                index += 1
                if index >= len(args):
                    raise LLMBackendError(
                        "llamacpp_embedding_invalid_loaded_batch_limit"
                    )
                value = args[index]
            else:
                for option in ("--batch-size", "--ubatch-size"):
                    if arg.startswith(option + "="):
                        name, value = _BATCH_OPTIONS[option], arg.split("=", 1)[1]
                        break
            if value is not None:
                try:
                    parsed = int(value) if _DECIMAL.fullmatch(value) else 0
                except (ValueError, OverflowError) as exc:
                    raise LLMBackendError(
                        "llamacpp_embedding_invalid_loaded_batch_limit"
                    ) from exc
                if parsed <= 0:
                    raise LLMBackendError(
                        "llamacpp_embedding_invalid_loaded_batch_limit"
                    )
                found[name].append(parsed)
            index += 1
        for name, values in found.items():
            if len(values) > 1:
                raise LLMBackendError("llamacpp_embedding_ambiguous_loaded_batch_limit")
            observations[name].extend(values)
    result = {}
    for name, values in observations.items():
        if len(set(values)) > 1:
            raise LLMBackendError("llamacpp_embedding_ambiguous_loaded_batch_limit")
        result[name] = values[0] if values else None
    return LoadedExecutionMetadata(**result, total_slots=total_slots)


def loaded_batch_limit(props: Any, entry: Any = None) -> int | None:
    """Compatibility observation of native ``n_batch``, never an HTTP limit."""
    return loaded_execution_metadata(props, entry).n_batch
