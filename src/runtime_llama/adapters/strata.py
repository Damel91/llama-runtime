"""LLM requests to an externally managed Strata HTTP server.

The transport, stream parser, tools, cancellation and drain ownership are shared
with the llama.cpp adapter. Strata-specific request and observation contracts
live here; no inference process, model or MCP server is managed by this client.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Callable, Dict, Mapping, Optional

from runtime_llama.adapters.llama import LlamaCppAdapter
from runtime_llama.errors import LLMBackendError, LLMTimeoutError
from runtime_llama.runtime_discovery import _normalize_strata_entries


class StrataAdapter(LlamaCppAdapter):
    """The common LLM adapter API specialized for the Strata chat endpoint.

    Constructor and public generation signatures are inherited unchanged.
    Structured streams are provisional until the success terminal: a complete
    response is checked locally even when the server lacks schema validation.
    Native or local output caps, cancellation and abandonment remain incomplete
    outcomes and never promise a schema-valid complete response.
    """

    _client_trust_env = False

    def _json_response_format(self, schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if schema is None:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "runtime_llama_response",
                "schema": deepcopy(schema),
                "strict": True,
            },
        }

    def _response_validator(
        self, body: Mapping[str, Any]
    ) -> Optional[Callable[[str], None]]:
        response = body.get("response_format")
        if response is None:
            return None
        if not isinstance(response, Mapping):
            raise ValueError("strata_response_format_must_be_an_object")
        kind = response.get("type")
        if kind == "text":
            return None
        if kind not in {"json_object", "json_schema"}:
            raise ValueError("strata_unsupported_response_format")
        validator = None
        if kind == "json_schema":
            spec = response.get("json_schema")
            if (
                not isinstance(spec, Mapping)
                or not isinstance(spec.get("schema"), dict)
                or not isinstance(spec.get("name"), str)
                or not spec["name"]
                or ("strict" in spec and type(spec["strict"]) is not bool)
            ):
                raise ValueError("strata_invalid_json_schema_format")
            from jsonschema import validators
            from jsonschema.exceptions import SchemaError
            from referencing import Registry

            schema = deepcopy(spec["schema"])
            # Strata accepts local references only. Validate the same boundary
            # before sending; never fetch a URL supplied by a schema.
            pending = [schema]
            visited: set[int] = set()
            while pending:
                node = pending.pop()
                if not isinstance(node, (dict, list)) or id(node) in visited:
                    continue
                visited.add(id(node))
                if isinstance(node, dict):
                    for key, value in node.items():
                        if (
                            key in {"$ref", "$dynamicRef"}
                            and isinstance(value, str)
                            and not value.startswith("#")
                        ):
                            raise ValueError("strata_schema_requires_local_references")
                        if isinstance(value, (dict, list)):
                            pending.append(value)
                elif isinstance(node, list):
                    pending.extend(node)
            try:
                dialect = schema.get("$schema")
                if dialect is not None and not isinstance(dialect, str):
                    raise ValueError("strata_invalid_json_schema")
                selected = validators.validator_for(schema, default=None)
                if selected is None:
                    # An absent dialect uses the current JSON Schema draft;
                    # an explicit unknown dialect is not silently guessed.
                    if dialect is not None:
                        raise ValueError("strata_unknown_json_schema_dialect")
                    selected = validators.validator_for(schema)
                selected.check_schema(schema)
            except (SchemaError, RecursionError):
                raise ValueError("strata_invalid_json_schema") from None
            validator = selected(schema, registry=Registry())

        def validate(text: str) -> None:
            def invalid_constant(value: str) -> None:
                raise ValueError("non_json_number")

            try:
                value = json.loads(text, parse_constant=invalid_constant)
            except (ValueError, RecursionError):
                raise LLMBackendError("strata_invalid_json_output") from None
            if not isinstance(value, dict):
                raise LLMBackendError("strata_json_output_requires_object")
            if validator is not None:
                # Do not expose model text, schema content or reference URLs
                # in a diagnostic. Registry has no remote resource retriever.
                from referencing.exceptions import Unresolvable

                try:
                    invalid = next(validator.iter_errors(value), None)
                except (Unresolvable, RecursionError):
                    raise LLMBackendError("strata_json_schema_unresolvable") from None
                if invalid is not None:
                    raise LLMBackendError("strata_json_schema_mismatch") from None

        return validate

    def health_status(self) -> Dict[str, Any]:
        """Observe Strata using root GETs only, without auto-loading models."""
        self._observed_context_length = None
        healthy = False
        ready = False
        available = False
        model_state = None
        slots = None
        reason = "unreachable"
        try:
            health = self._client.health()
            healthy = (
                isinstance(health, Mapping)
                and health.get("status") == "ok"
                and health.get("service") == "strata"
            )
            if not healthy:
                reason = "runtime_identity_unverified"
            else:
                reason = "model_readiness_unverified"
                catalog = self._client.models()
                entries = catalog.get("data") if isinstance(catalog, Mapping) else None
                if not isinstance(entries, list):
                    reason = "catalog_unavailable"
                else:
                    entries = _normalize_strata_entries(entries)
                    canonical = health.get("model")
                    matches = [
                        row
                        for row in entries
                        if isinstance(row, Mapping) and row.get("id") == canonical
                    ]
                    selected = matches[0] if len(matches) == 1 else None
                    aliases = selected.get("aliases", []) if selected else []
                    names = (
                        {canonical, *aliases}
                        if isinstance(canonical, str)
                        and isinstance(aliases, list)
                        and all(isinstance(item, str) for item in aliases)
                        else set()
                    )
                    available = self.model_name in names
                    if len(matches) > 1:
                        reason = "model_alias_ambiguous"
                    elif not available:
                        reason = "model_not_found"
                    else:
                        status = selected.get("status")
                        state = (
                            status.get("value")
                            if isinstance(status, Mapping)
                            else status
                        )
                        failed = (
                            status.get("failed")
                            if isinstance(status, Mapping)
                            else None
                        )
                        loaded = health.get("loaded")
                        ready = (
                            loaded is True and state == "loaded" and failed is not True
                        )
                        model_state = state if isinstance(state, str) else "unknown"
                        reason = "ready" if ready else "model_not_loaded"
                        # Root props are a reporting endpoint. Never send
                        # ?model=...: discovery is not model administration.
                        if ready:
                            props = self._client.props()
                            alias = (
                                props.get("model_alias")
                                if isinstance(props, Mapping)
                                else None
                            )
                            if alias not in names or props.get("is_sleeping") is True:
                                ready = False
                                reason = "model_metadata_mismatch"
                            else:
                                settings = props.get("default_generation_settings")
                                context = (
                                    settings.get("n_ctx")
                                    if isinstance(settings, Mapping)
                                    else None
                                )
                                if type(context) is int and context > 0:
                                    self._observed_context_length = context
                                observed = props.get("total_slots")
                                if type(observed) is int and observed > 0:
                                    slots = observed
        except (LLMBackendError, LLMTimeoutError):
            ready = False
            reason = "metadata_unavailable" if healthy else "unreachable"
        return {
            "backend": "strata",
            "role_id": self.role_id,
            "model": self.model_name,
            "healthy": healthy,
            "server_healthy": healthy,
            "backend_ready": ready,
            "model_available": available,
            "model_ready": ready,
            "model_state": model_state,
            "reason": reason,
            "loaded_context_length": self._observed_context_length,
            "configured_context_length": self._model_config.context_length,
            "observed_slots": slots,
            "requested_context_ceiling": self._requested_context_ceiling
            or self._num_ctx,
            "num_predict": self._num_predict,
            "reasoning_requested": self._role_config.reasoning,
            "reasoning_capability_verified": False,
            "runtime_identity": self._model_config.runtime_identity,
            "process_lifecycle": "external",
            "model_loading": "server_owned",
        }
