"""Strict loaded-runtime embedding budget observations, never model identity."""

from __future__ import annotations

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
        "--ubatch-size",
        "-ub",
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


def _positive(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LLMBackendError("llamacpp_embedding_invalid_loaded_batch_limit")
    return value


def loaded_batch_limit(props: Any, entry: Any = None) -> int | None:
    """Read supported integer n_batch or the exact loaded entry's argv.

    Free-form presets, model paths, training context and ubatch are not limits.
    Repeated options and disagreement between supported observations are errors.
    """
    observations: list[int] = []
    if isinstance(props, dict):
        if "n_batch" in props:
            observations.append(_positive(props["n_batch"]))
        settings = props.get("default_generation_settings")
        if isinstance(settings, dict) and "n_batch" in settings:
            observations.append(_positive(settings["n_batch"]))
    status = entry.get("status") if isinstance(entry, dict) else None
    if isinstance(status, dict) and "args" in status:
        args = status["args"]
        if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
            raise LLMBackendError("llamacpp_embedding_invalid_loaded_batch_limit")
        found = []
        index = 0
        while index < len(args):
            arg = args[index]
            if arg == "--":
                break
            if arg in _VALUE_OPTIONS:
                index += 2
                continue
            value = None
            if arg in {"--batch-size", "-b"}:
                index += 1
                if index >= len(args):
                    raise LLMBackendError(
                        "llamacpp_embedding_invalid_loaded_batch_limit"
                    )
                value = args[index]
            elif arg.startswith("--batch-size="):
                value = arg.split("=", 1)[1]
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
                found.append(parsed)
            index += 1
        if len(found) > 1:
            raise LLMBackendError("llamacpp_embedding_ambiguous_loaded_batch_limit")
        observations.extend(found)
    if len(set(observations)) > 1:
        raise LLMBackendError("llamacpp_embedding_ambiguous_loaded_batch_limit")
    return observations[0] if observations else None
