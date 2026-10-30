"""Pure request-local proof and feedback for automatic tool-call recovery.

This module neither sends requests nor executes tools. Native diagnostic fields
and completed structured calls are authority; markup in ordinary text is not.
The native Strata marker contract is list[{name: str, reason: 'undeclared'}].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import re
from typing import Any, Literal

from runtime_llama.errors import sanitize_diagnostic
from runtime_llama.usage import LLMStreamEvent, LLMToolRecoveryRejection


_NATIVE_MARKER = "strata_rejected_tool_calls"
_CREDENTIAL = re.compile(
    r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]+"
    r"|\b(?:api[_-]?key|access[_-]?token|password)\b[\"']?\s*[:=]\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)"
)


def _safe_name(value: str, secrets: Sequence[str]) -> str:
    # Strata truncates names to80 characters before transmitting this field.
    # Redact a registered secret split at that boundary before our own export.
    native_boundary = len(value) >= 80
    for secret in sorted(set(filter(None, secrets)), key=len, reverse=True):
        value = value.replace(secret, "<credential_redacted>")
        if native_boundary:
            for size in range(min(len(secret) - 1, len(value)), 0, -1):
                if value.endswith(secret[:size]):
                    value = value[:-size] + "<credential_redacted>"
                    break
    return sanitize_diagnostic(
        _CREDENTIAL.sub("<credential_redacted>", value), limit=80
    )


def _json_snapshot(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        raise ValueError("tool_recovery_request_requires_json_values") from None


def _schema_validator(schema: Any):
    from jsonschema import validators
    from jsonschema.exceptions import SchemaError
    from referencing import Registry, Resource
    from referencing.exceptions import Unresolvable
    from referencing.jsonschema import specification_with

    if not isinstance(schema, (dict, bool)):
        raise ValueError("tool_parameters_require_json_schema")
    dialect = schema.get("$schema") if isinstance(schema, dict) else None
    if dialect is not None and not isinstance(dialect, str):
        raise ValueError("tool_schema_dialect_invalid")
    selected = validators.validator_for(schema, default=None)
    if selected is None:
        if dialect is not None:
            raise ValueError("tool_schema_dialect_unknown")
        selected = validators.validator_for(schema)
    try:
        selected.check_schema(schema)
    except (SchemaError, RecursionError):
        raise ValueError("tool_schema_invalid") from None
    registry = Registry()
    # Resolve local references against the actual resource scope before HTTP,
    # without following them recursively or registering a remote retriever.
    specification = specification_with(
        selected.META_SCHEMA.get("$id") or selected.META_SCHEMA["id"]
    )
    root = Resource.from_contents(schema, default_specification=specification)
    resources = [(root, registry.resolver_with_root(root), selected)]
    while resources:
        resource, resolver, parent_validator = resources.pop()
        value = resource.contents
        node_validator = validators.validator_for(value, default=parent_validator)
        if isinstance(value, dict):
            for key in ("$ref", "$dynamicRef"):
                if key in value and key in node_validator.VALIDATORS:
                    if not isinstance(value[key], str) or not value[key].startswith(
                        "#"
                    ):
                        raise ValueError("tool_schema_requires_local_references")
                    try:
                        resolver.lookup(value[key])
                    except Unresolvable:
                        raise ValueError("tool_schema_unresolvable") from None
        for child in resource.subresources():
            resources.append((child, resolver.in_subresource(child), node_validator))
    return selected(schema, registry=registry)


@dataclass(frozen=True)
class ToolRecoveryDecision:
    """Content-free classification, not an executable proposal."""

    status: Literal["accepted", "recoverable", "unsafe"]
    reasons: tuple[str, ...] = ()
    native_rejected_tool_calls: int | None = None
    server_activity: bool = False
    native_rejections: tuple[LLMToolRecoveryRejection, ...] = ()
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "native_rejections", tuple(self.native_rejections))
        object.__setattr__(self, "provenance", tuple(self.provenance))

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    @property
    def recoverable(self) -> bool:
        return self.status == "recoverable"

    @property
    def reason(self) -> str:
        return self.reasons[0] if self.reasons else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "native_rejected_tool_calls": self.native_rejected_tool_calls,
            "server_activity": self.server_activity,
            "native_rejections": [item.to_dict() for item in self.native_rejections],
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class ToolRecoveryPolicy:
    """Frozen declared-tool authority for one external invocation.

    ``from_request`` validates schemas before HTTP. Returned feedback contains
    only ordinary messages and must be appended to a private history snapshot;
    the caller's history and native tools/schema/options are never changed.
    """

    declared_names: tuple[str, ...]
    _schemas: tuple[tuple[str, str], ...] = field(repr=False)
    _tools_json: str = field(repr=False)
    _choice: str = field(default="auto", repr=False)
    _specific_name: str | None = field(default=None, repr=False)
    _server_mcp: bool = field(default=False, repr=False)

    @property
    def enabled(self) -> bool:
        return bool(self.declared_names)

    @property
    def tools(self) -> list[dict[str, Any]]:
        return json.loads(self._tools_json)

    @classmethod
    def from_request(cls, body: Mapping[str, Any]) -> ToolRecoveryPolicy:
        if not isinstance(body, Mapping):
            raise ValueError("tool_recovery_request_must_be_mapping")
        supplied = body.get("tools")
        if supplied is None:
            supplied = []
        if not isinstance(supplied, (list, tuple)):
            raise ValueError("tool_recovery_tools_must_be_list")
        tools_json = _json_snapshot(supplied)
        tools = json.loads(tools_json)
        names: list[str] = []
        schemas: list[tuple[str, str]] = []
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type", "function") != "function":
                raise ValueError("tool_recovery_requires_function_tools")
            function = tool.get("function")
            name = function.get("name") if isinstance(function, dict) else None
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("tool_recovery_declared_name_invalid")
            schema = function.get("parameters", {"type": "object"})
            _schema_validator(schema)
            names.append(name)
            schemas.append((name, _json_snapshot(schema)))
        choice = body.get("tool_choice", "auto")
        specific = None
        if isinstance(choice, Mapping):
            function = choice.get("function")
            specific = function.get("name") if isinstance(function, Mapping) else None
            if choice.get("type") != "function" or specific not in names:
                raise ValueError("tool_recovery_specific_choice_invalid")
            choice = "specific"
        elif not isinstance(choice, str) or choice not in {"auto", "required", "none"}:
            raise ValueError("tool_recovery_tool_choice_invalid")
        if choice in {"required", "specific"} and not names:
            raise ValueError("tool_recovery_tool_choice_requires_tools")
        return cls(
            tuple(names),
            tuple(schemas),
            tools_json,
            choice,
            specific,
            body.get("strata_mcp") is True,
        )

    def classify(
        self,
        events: Sequence[LLMStreamEvent],
        terminal: LLMStreamEvent | None,
        *,
        secrets: Sequence[str] = (),
    ) -> ToolRecoveryDecision:
        if not self.enabled:
            return ToolRecoveryDecision("accepted")
        reasons: list[str] = []
        fatal: list[str] = []
        provenance: list[str] = []
        rejected: list[LLMToolRecoveryRejection] = []
        marker_seen = False
        server_activity = self._server_mcp
        marker_snapshots: set[str] = set()
        observed = list(events)
        if terminal is not None and all(item is not terminal for item in observed):
            observed.append(terminal)
        for event in observed:
            data = event.data
            if not isinstance(data, Mapping):
                continue
            if (
                "strata_mcp" in data
                and data["strata_mcp"] is not None
                and (data["strata_mcp"] is not False)
            ):
                server_activity = True
            if data.get("error") is not None:
                fatal.append("server_error")
            if _NATIVE_MARKER not in data:
                continue
            marker = data[_NATIVE_MARKER]
            if not isinstance(marker, list) or any(
                not isinstance(item, Mapping)
                or set(item) != {"name", "reason"}
                or not isinstance(item.get("name"), str)
                or item.get("reason") != "undeclared"
                for item in marker
            ):
                fatal.append("malformed_native_tool_rejection")
                provenance.append("sdk_protocol")
                continue
            marker_seen = True
            # The parser may expose the same final native marker in both an
            # observation event and terminal. Keep that snapshot only once.
            identity = json.dumps(marker, ensure_ascii=False, sort_keys=True)
            if identity in marker_snapshots:
                continue
            marker_snapshots.add(identity)
            rejected.extend(
                LLMToolRecoveryRejection(
                    _safe_name(item["name"], secrets), "undeclared", _NATIVE_MARKER
                )
                for item in marker
            )
        if rejected:
            reasons.append("native_rejected_tool_call")
            provenance.append(_NATIVE_MARKER)
        if terminal is None or terminal.kind != "terminal":
            fatal.append("incomplete_tool_response")
        elif terminal.terminal_reason not in {"completed", "tool_calls"}:
            fatal.append("incomplete_tool_response")
        data = terminal.data if terminal is not None else None
        calls = data.get("tool_calls", []) if isinstance(data, Mapping) else []
        if not isinstance(calls, (list, tuple)):
            fatal.append("malformed_completed_tool_calls")
            calls = []
        validators = {
            name: _schema_validator(json.loads(raw)) for name, raw in self._schemas
        }
        names: list[str] = []
        identities: set[str] = set()
        for call in calls:
            if not isinstance(call, Mapping):
                fatal.append("malformed_completed_tool_call")
                continue
            function = call.get("function")
            name = function.get("name") if isinstance(function, Mapping) else None
            identity = call.get("id")
            if (
                call.get("type", "function") != "function"
                or not isinstance(name, str)
                or not name
                or not isinstance(identity, str)
                or not identity
                or identity in identities
            ):
                fatal.append("malformed_completed_tool_call")
                continue
            identities.add(identity)
            names.append(name)
            provenance.append("sdk_completed_tool_call")
            if name not in validators:
                reasons.append("undeclared_tool_name")
                continue
            arguments = function.get("arguments")
            try:
                if not isinstance(arguments, str):
                    raise ValueError("nonstring")

                def invalid_constant(_value: str) -> None:
                    raise ValueError("nonfinite")

                value = json.loads(arguments, parse_constant=invalid_constant)
            except (ValueError, TypeError, RecursionError):
                reasons.append("invalid_tool_arguments_json")
                continue
            if not isinstance(value, dict):
                reasons.append("tool_arguments_not_object")
                continue
            from referencing.exceptions import Unresolvable

            try:
                invalid = next(validators[name].iter_errors(value), None)
            except (Unresolvable, RecursionError):
                # A broken schema reference is a request contract failure,
                # not evidence that the model can repair its arguments.
                fatal.append("tool_schema_unresolvable")
                continue
            if invalid is not None:
                reasons.append("tool_arguments_schema_mismatch")
        if self._choice == "required" and not calls:
            reasons.append("required_tool_call_missing")
            provenance.append("sdk_tool_choice")
        if self._choice == "specific" and (
            not names or any(name != self._specific_name for name in names)
        ):
            reasons.append("specific_tool_choice_violated")
            provenance.append("sdk_tool_choice")
        if self._choice == "none" and calls:
            reasons.append("tool_choice_none_violated")
            provenance.append("sdk_tool_choice")
        if (
            terminal is not None
            and terminal.terminal_reason == "tool_calls"
            and not calls
        ):
            reasons.append("tool_finish_without_calls")
            provenance.append("sdk_completed_tool_call")
        if server_activity and reasons:
            fatal.append("server_side_tool_effects_possible")
        all_reasons = tuple(dict.fromkeys([*fatal, *reasons]))
        return ToolRecoveryDecision(
            "unsafe" if fatal else "recoverable" if reasons else "accepted",
            all_reasons,
            len(rejected) if marker_seen else None,
            server_activity,
            tuple(rejected),
            tuple(dict.fromkeys(provenance)),
        )

    def correction_messages(
        self,
        events: Sequence[LLMStreamEvent],
        decision: ToolRecoveryDecision,
    ) -> list[dict[str, str]]:
        """Produce SDK feedback, never executable or fabricated tool results.

        The rejected proposal is retained by the invocation owner for explicit
        inspection. It is not promoted to an assistant tool call, and none of
        its calls is said to have executed. Only fixed reason codes and the
        trusted declared names are included in the feedback.
        """
        if not self.enabled or not decision.recoverable or decision.server_activity:
            raise ValueError(
                "tool_recovery_feedback_requires_safe_recoverable_decision"
            )
        # Keep the signature event-aware for integrations, without parsing or
        # echoing untrusted raw XML, reasoning, arguments or diagnostic content.
        del events
        required = (
            f"Call only {json.dumps(self._specific_name)}."
            if self._choice == "specific"
            else "At least one declared tool call is required."
            if self._choice == "required"
            else "Do not call any tool."
            if self._choice == "none"
            else "A normal response or valid declared tool calls are allowed."
        )
        feedback = (
            "The preceding generated proposal was rejected before dispatch; no tool "
            "from it has executed. Regenerate only that proposal under the unchanged "
            "current tools and schemas. Use the exact declared names, never aliases. "
            "Each tool's arguments must be one valid JSON object matching its schema. "
            "Do not repeat tools already completed in the original conversation. "
            "Return native structured tool calls rather than quoted markup. "
            f"Declared names: {json.dumps(self.declared_names, ensure_ascii=False)}. "
            f"Protocol failures: {json.dumps(decision.reasons)}. {required}"
        )
        return [
            {
                "role": "assistant",
                "content": "My preceding generated tool proposal was not accepted or executed.",
            },
            {"role": "user", "content": feedback},
        ]


__all__ = ["ToolRecoveryDecision", "ToolRecoveryPolicy"]
