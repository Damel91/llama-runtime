from dataclasses import FrozenInstanceError

import pytest

from runtime_llama import (
    MCPEphemeralIntegration,
    MCPInvocationOptions,
    MCPPluginIntegration,
    MCPToolCall,
)


def test_plugin_requires_bounded_tool_access_and_serializes_deterministically():
    integration = MCPPluginIntegration(
        "mcp/codingcastle",
        allowed_tools=("review", "jobs", "review"),
    )

    assert integration.allowed_tools == ("review", "jobs")
    assert integration.identity == ("plugin", "mcp/codingcastle")
    assert integration.to_payload() == {
        "type": "plugin",
        "id": "mcp/codingcastle",
        "allowed_tools": ["review", "jobs"],
    }

    with pytest.raises(ValueError, match="allowed_tools"):
        MCPPluginIntegration("mcp/codingcastle")
    with pytest.raises(ValueError, match="mutually exclusive"):
        MCPPluginIntegration(
            "mcp/codingcastle",
            allowed_tools=("review",),
            allow_all_tools=True,
        )
    with pytest.raises(TypeError, match="sequence"):
        MCPPluginIntegration("mcp/codingcastle", allowed_tools="review")


def test_explicit_allow_all_omits_allowlist():
    integration = MCPPluginIntegration(
        "mcp/flow",
        allow_all_tools=True,
    )

    assert integration.to_payload() == {"type": "plugin", "id": "mcp/flow"}


def test_ephemeral_headers_are_secret_but_reach_transport_payload():
    first = MCPEphemeralIntegration(
        "codingcastle",
        "https://mcp.example.test/mcp",
        allowed_tools=("review",),
        headers={"Authorization": "Bearer first-secret"},
    )
    second = MCPEphemeralIntegration(
        "codingcastle",
        "https://mcp.example.test/mcp",
        allowed_tools=("review",),
        headers={"Authorization": "Bearer second-secret"},
    )

    assert "first-secret" not in repr(first)
    assert "mcp.example.test" not in repr(first)
    assert first == second
    assert first.to_payload()["headers"] == {"Authorization": "Bearer first-secret"}
    with pytest.raises(TypeError):
        first.headers["Authorization"] = "changed"


@pytest.mark.parametrize(
    "url",
    (
        "stdio://server",
        "localhost:8000/mcp",
        "https://user:secret@example.test/mcp",
    ),
)
def test_ephemeral_url_fails_closed(url):
    with pytest.raises(ValueError, match="server_url"):
        MCPEphemeralIntegration(
            "server",
            url,
            allowed_tools=("tool",),
        )


def test_invocation_rejects_duplicate_identity_and_implicit_state():
    integration = MCPPluginIntegration("mcp/flow", allowed_tools=("next",))
    with pytest.raises(ValueError, match="duplicate"):
        MCPInvocationOptions((integration, integration))
    with pytest.raises(ValueError, match="store=True"):
        MCPInvocationOptions(
            (integration,),
            previous_response_id="resp_123",
        )
    with pytest.raises(ValueError, match="resp_"):
        MCPInvocationOptions(
            (integration,),
            store=True,
            previous_response_id="invalid",
        )


def test_public_result_values_are_frozen_and_copy_arguments():
    arguments = {"surface": "repo"}
    call = MCPToolCall(tool="review", arguments=arguments, output="ok")
    arguments["surface"] = "workspace"

    assert call.arguments == {"surface": "repo"}
    with pytest.raises(TypeError):
        call.arguments["surface"] = "changed"
    with pytest.raises(FrozenInstanceError):
        call.tool = "changed"
