from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, TypeAlias
from urllib.parse import urlsplit

from runtime_llama.usage import LLMTransportAttempt, LLMUsageStats


def _required_text(value: Any, *, field_name: str, limit: int = 256) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    if len(text) > limit or any(ord(char) < 32 for char in text):
        raise ValueError(f"{field_name} is invalid")
    return text


def _normalized_tools(
    values: Sequence[str],
    *,
    allow_all_tools: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("allowed_tools must be a sequence of tool names")
    tools = tuple(
        dict.fromkeys(
            _required_text(value, field_name="allowed tool") for value in values
        )
    )
    if tools and allow_all_tools:
        raise ValueError("allowed_tools and allow_all_tools are mutually exclusive")
    if not tools and not allow_all_tools:
        raise ValueError(
            "an MCP integration requires allowed_tools or explicit allow_all_tools"
        )
    return tools


def _integration_payload(
    payload: dict[str, Any],
    *,
    allowed_tools: tuple[str, ...],
) -> dict[str, Any]:
    if allowed_tools:
        payload["allowed_tools"] = list(allowed_tools)
    return payload


@dataclass(frozen=True)
class MCPPluginIntegration:
    plugin_id: str
    allowed_tools: tuple[str, ...] = ()
    allow_all_tools: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.allow_all_tools, bool):
            raise TypeError("allow_all_tools must be a boolean")
        plugin_id = _required_text(self.plugin_id, field_name="plugin_id")
        if not plugin_id.startswith("mcp/"):
            raise ValueError("plugin_id must identify an mcp/<server_label> plugin")
        object.__setattr__(self, "plugin_id", plugin_id)
        object.__setattr__(
            self,
            "allowed_tools",
            _normalized_tools(
                self.allowed_tools,
                allow_all_tools=bool(self.allow_all_tools),
            ),
        )

    @property
    def identity(self) -> tuple[str, str]:
        return ("plugin", self.plugin_id)

    def to_payload(self) -> dict[str, Any]:
        return _integration_payload(
            {"type": "plugin", "id": self.plugin_id},
            allowed_tools=self.allowed_tools,
        )


@dataclass(frozen=True)
class MCPEphemeralIntegration:
    server_label: str
    server_url: str = field(repr=False)
    allowed_tools: tuple[str, ...] = ()
    allow_all_tools: bool = False
    headers: Mapping[str, str] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.allow_all_tools, bool):
            raise TypeError("allow_all_tools must be a boolean")
        label = _required_text(self.server_label, field_name="server_label")
        url = _required_text(self.server_url, field_name="server_url", limit=2048)
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("server_url must be an absolute HTTP or HTTPS URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("server_url must not contain embedded credentials")

        if not isinstance(self.headers, Mapping):
            raise TypeError("headers must be a mapping")
        normalized_headers: dict[str, str] = {}
        header_names: set[str] = set()
        for raw_name, raw_value in dict(self.headers or {}).items():
            name = _required_text(raw_name, field_name="header name")
            value = str(raw_value)
            if not value or any(char in value for char in ("\r", "\n")):
                raise ValueError("MCP integration header value is invalid")
            folded = name.casefold()
            if folded in header_names:
                raise ValueError("duplicate MCP integration header name")
            header_names.add(folded)
            normalized_headers[name] = value

        object.__setattr__(self, "server_label", label)
        object.__setattr__(self, "server_url", url)
        object.__setattr__(
            self,
            "allowed_tools",
            _normalized_tools(
                self.allowed_tools,
                allow_all_tools=bool(self.allow_all_tools),
            ),
        )
        object.__setattr__(
            self,
            "headers",
            MappingProxyType(normalized_headers),
        )

    @property
    def identity(self) -> tuple[str, str]:
        return ("ephemeral_mcp", self.server_label)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "ephemeral_mcp",
            "server_label": self.server_label,
            "server_url": self.server_url,
        }
        if self.headers:
            payload["headers"] = dict(self.headers)
        return _integration_payload(payload, allowed_tools=self.allowed_tools)


MCPIntegration: TypeAlias = MCPPluginIntegration | MCPEphemeralIntegration


@dataclass(frozen=True)
class MCPInvocationOptions:
    integrations: tuple[MCPIntegration, ...]
    store: bool = False
    previous_response_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.store, bool):
            raise TypeError("store must be a boolean")
        integrations = tuple(self.integrations or ())
        if not integrations:
            raise ValueError("at least one MCP integration is required")
        identities: set[tuple[str, str]] = set()
        for integration in integrations:
            if not isinstance(
                integration,
                (MCPPluginIntegration, MCPEphemeralIntegration),
            ):
                raise TypeError("unsupported MCP integration type")
            if integration.identity in identities:
                raise ValueError("duplicate MCP integration identity")
            identities.add(integration.identity)

        response_id = self.previous_response_id
        if response_id is not None:
            response_id = _required_text(
                response_id,
                field_name="previous_response_id",
            )
            if not response_id.startswith("resp_"):
                raise ValueError("previous_response_id must start with 'resp_'")
            if not self.store:
                raise ValueError("previous_response_id requires explicit store=True")

        object.__setattr__(self, "integrations", integrations)
        object.__setattr__(self, "previous_response_id", response_id)

    def integration_payloads(self) -> list[dict[str, Any]]:
        return [integration.to_payload() for integration in self.integrations]


@dataclass(frozen=True)
class MCPProviderInfo:
    provider_type: str
    plugin_id: str = ""
    server_label: str = ""

    @classmethod
    def from_payload(cls, value: Any) -> Optional["MCPProviderInfo"]:
        if not isinstance(value, Mapping):
            return None
        provider_type = str(value.get("type") or "unknown").strip() or "unknown"
        return cls(
            provider_type=provider_type,
            plugin_id=str(value.get("plugin_id") or ""),
            server_label=str(value.get("server_label") or ""),
        )


@dataclass(frozen=True)
class MCPToolCall:
    tool: str
    arguments: Mapping[str, Any]
    output: Any
    provider: Optional[MCPProviderInfo] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool", _required_text(self.tool, field_name="tool"))
        if not isinstance(self.arguments, Mapping):
            raise ValueError("tool call arguments must be an object")
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(dict(self.arguments)),
        )


@dataclass(frozen=True)
class MCPInvalidToolCall:
    reason: str
    metadata_type: str
    tool_name: str
    arguments: Optional[Mapping[str, Any]] = None
    provider: Optional[MCPProviderInfo] = None

    def __post_init__(self) -> None:
        if self.arguments is not None:
            if not isinstance(self.arguments, Mapping):
                raise ValueError("invalid tool call arguments must be an object")
            object.__setattr__(
                self,
                "arguments",
                MappingProxyType(dict(self.arguments)),
            )


@dataclass(frozen=True)
class MCPStreamEvent:
    kind: str
    text: str = ""
    tool: str = ""
    arguments: Any = None
    output: Any = None
    provider: Optional[MCPProviderInfo] = None
    failure_reason: str = ""
    usage: Optional[LLMUsageStats] = None
    terminal_reason: str = ""
    response_id: str = ""
    tool_calls: tuple[MCPToolCall, ...] = ()
    invalid_tool_calls: tuple[MCPInvalidToolCall, ...] = ()
    transport: Optional[LLMTransportAttempt] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _required_text(self.kind, field_name="kind"))
        if isinstance(self.arguments, Mapping):
            object.__setattr__(
                self,
                "arguments",
                MappingProxyType(dict(self.arguments)),
            )
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls or ()))
        object.__setattr__(
            self,
            "invalid_tool_calls",
            tuple(self.invalid_tool_calls or ()),
        )


@dataclass(frozen=True)
class MCPCallResult:
    text: str
    usage: Optional[LLMUsageStats]
    role_id: str
    model: str
    reasoning_text: str = ""
    terminal_reason: str = "completed"
    response_id: str = ""
    tool_calls: tuple[MCPToolCall, ...] = ()
    invalid_tool_calls: tuple[MCPInvalidToolCall, ...] = ()
    transport: Optional[LLMTransportAttempt] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls or ()))
        object.__setattr__(
            self,
            "invalid_tool_calls",
            tuple(self.invalid_tool_calls or ()),
        )
