from copy import deepcopy
from dataclasses import FrozenInstanceError
import json

import pytest

from runtime_llama import (
    LLMCallResult,
    LLMStreamEvent,
    LLMToolRecoveryAttempt,
    LLMToolRecoveryError,
    LLMToolRecoveryRejection,
    LLMToolRecoveryReport,
    LLMUsageStats,
)
from runtime_llama.tool_recovery import ToolRecoveryPolicy


def request(*, choice="auto", schema=None, **options):
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "pi_write",
                    "parameters": schema
                    if schema is not None
                    else {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": choice,
        **options,
    }


def call(name="pi_write", arguments='{"path":"fixture.txt"}', identity="call_a"):
    return {
        "id": identity,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def terminal(calls=(), **data):
    return LLMStreamEvent(
        "terminal",
        terminal_reason="tool_calls" if calls else "completed",
        data={
            "tool_calls": list(calls),
            "finish_reason": "tool_calls" if calls else "stop",
            **data,
        },
    )


def native(name="write"):
    return [{"name": name, "reason": "undeclared"}]


def test_authoritative_native_rejection_is_not_success_and_keeps_proof():
    policy = ToolRecoveryPolicy.from_request(request())
    content = LLMStreamEvent("content", text="<tool_call><function=write>secret args")
    end = terminal(strata_rejected_tool_calls=native())
    result = policy.classify([content], end)
    assert result.recoverable and not result.accepted
    assert result.native_rejected_tool_calls == 1
    assert result.reasons == ("native_rejected_tool_call",)
    assert result.native_rejections == (
        LLMToolRecoveryRejection("write", "undeclared", "strata_rejected_tool_calls"),
    )
    assert result.to_dict()["native_rejections"][0] == {
        "name": "write",
        "reason": "undeclared",
        "source": "strata_rejected_tool_calls",
    }


@pytest.mark.parametrize(
    "text",
    [
        "normal answer",
        "Here is an example: <tool_call><function=write>...</tool_call>",
        "```xml\n<tool_call><function=write><parameter=path>x</tool_call>\n```",
        '{"tool_calls": "a documented field"}',
    ],
)
def test_text_and_quoted_xml_without_proof_are_ordinary_content(text):
    event = LLMStreamEvent("content", text=text)
    result = ToolRecoveryPolicy.from_request(request()).classify([event], terminal())
    assert result.accepted and result.native_rejected_tool_calls is None
    assert event.text == text


@pytest.mark.parametrize(
    "marker",
    [
        None,
        1,
        True,
        {},
        "undeclared",
        ["write"],
        [{"name": "write"}],
        [{"name": 5, "reason": "undeclared"}],
        [{"name": "write", "reason": "invalid_arguments"}],
        [{"name": "write", "reason": "undeclared", "arguments": "secret"}],
    ],
)
def test_malformed_native_marker_is_unsafe_not_recoverable(marker):
    decision = ToolRecoveryPolicy.from_request(request()).classify(
        [], terminal(strata_rejected_tool_calls=marker)
    )
    assert decision.status == "unsafe"
    assert "malformed_native_tool_rejection" in decision.reasons
    assert not decision.recoverable


def test_empty_marker_is_success_and_duplicate_snapshot_is_not_counted_twice():
    policy = ToolRecoveryPolicy.from_request(request())
    assert policy.classify([], terminal(strata_rejected_tool_calls=[])).accepted
    diagnostic = LLMStreamEvent(
        "native_diagnostic", data={"strata_rejected_tool_calls": native()}
    )
    end = terminal(strata_rejected_tool_calls=native())
    result = policy.classify([diagnostic, end], end)
    assert result.native_rejected_tool_calls == 1


@pytest.mark.parametrize(
    ("proposal", "reason"),
    [
        (call(name="write"), "undeclared_tool_name"),
        (call(arguments="{"), "invalid_tool_arguments_json"),
        (call(arguments={"path": "x"}), "invalid_tool_arguments_json"),
        (call(arguments='{"path":NaN}'), "invalid_tool_arguments_json"),
        (call(arguments='["x"]'), "tool_arguments_not_object"),
        (call(arguments="null"), "tool_arguments_not_object"),
        (call(arguments='{"path":9}'), "tool_arguments_schema_mismatch"),
        (call(arguments='{"other":"x"}'), "tool_arguments_schema_mismatch"),
    ],
)
def test_completed_structured_call_has_proven_recoverable_contract_failure(
    proposal, reason
):
    policy = ToolRecoveryPolicy.from_request(request())
    decision = policy.classify([], terminal([proposal]))
    assert decision.recoverable and reason in decision.reasons
    assert "sdk_completed_tool_call" in decision.provenance


def test_valid_call_and_mixed_native_rejection_classify_without_mutation_or_execution():
    policy = ToolRecoveryPolicy.from_request(request())
    end = terminal([call()])
    snapshot = deepcopy(end.data)
    assert policy.classify([], end).accepted
    mixed = terminal([call()], strata_rejected_tool_calls=native())
    assert policy.classify([], mixed).recoverable
    assert end.data == snapshot


@pytest.mark.parametrize(
    "calls",
    [
        [call(identity="")],
        [call(), call()],
        [{"function": {"name": "write", "arguments": "{}"}}],
        ["write"],
    ],
)
def test_uncorrelatable_or_malformed_calls_are_unsafe(calls):
    decision = ToolRecoveryPolicy.from_request(request()).classify([], terminal(calls))
    assert decision.status == "unsafe"
    assert "malformed_completed_tool_call" in decision.reasons


def test_required_specific_and_none_choices_are_obligations():
    required = ToolRecoveryPolicy.from_request(request(choice="required"))
    assert "required_tool_call_missing" in required.classify([], terminal()).reasons
    specific = ToolRecoveryPolicy.from_request(
        request(choice={"type": "function", "function": {"name": "pi_write"}})
    )
    assert "specific_tool_choice_violated" in specific.classify([], terminal()).reasons
    assert specific.classify([], terminal([call()])).accepted
    prohibited = ToolRecoveryPolicy.from_request(request(choice="none"))
    assert (
        "tool_choice_none_violated"
        in prohibited.classify([], terminal([call()])).reasons
    )


@pytest.mark.parametrize(
    "reason",
    [
        "length_cap",
        "visible_output_cap",
        "content_filter",
        "idle_timeout",
        "cancelled",
        "backend_error",
    ],
)
def test_incomplete_or_nonprotocol_terminal_never_admits_recovery(reason):
    end = LLMStreamEvent(
        "terminal",
        terminal_reason=reason,
        data={"strata_rejected_tool_calls": native()},
    )
    result = ToolRecoveryPolicy.from_request(request()).classify([], end)
    assert result.status == "unsafe"
    assert "incomplete_tool_response" in result.reasons


def test_missing_terminal_is_not_a_recoverable_generation():
    assert (
        ToolRecoveryPolicy.from_request(request()).classify([], None).status == "unsafe"
    )


@pytest.mark.parametrize("source", ["request", "event", "empty_event"])
def test_possible_server_mcp_effects_block_regeneration(source):
    policy = ToolRecoveryPolicy.from_request(request(strata_mcp=source == "request"))
    events = (
        []
        if source == "request"
        else [
            LLMStreamEvent(
                "native_diagnostic",
                data={"strata_mcp": {"event": "call"} if source == "event" else {}},
            )
        ]
    )
    decision = policy.classify(events, terminal(strata_rejected_tool_calls=native()))
    assert decision.server_activity and decision.status == "unsafe"
    assert "server_side_tool_effects_possible" in decision.reasons


def test_false_mcp_does_not_imply_server_side_execution():
    policy = ToolRecoveryPolicy.from_request(request(strata_mcp=False))
    event = LLMStreamEvent("native_diagnostic", data={"strata_mcp": False})
    decision = policy.classify([event], terminal(strata_rejected_tool_calls=native()))
    assert decision.recoverable and not decision.server_activity


def test_absent_tools_preserve_legacy_path_even_if_text_has_tags():
    policy = ToolRecoveryPolicy.from_request({})
    assert not policy.enabled and policy.tools == []
    assert policy.classify(
        [LLMStreamEvent("content", text="<tool_call>")], terminal()
    ).accepted


def test_effective_tools_are_snapshotted_and_feedback_does_not_echo_bad_arguments():
    body = request(choice="required")
    original = deepcopy(body)
    policy = ToolRecoveryPolicy.from_request(body)
    body["tools"][0]["function"]["name"] = "changed"
    policy.tools[0]["function"]["parameters"]["required"].clear()
    end = terminal(strata_rejected_tool_calls=native())
    bad = LLMStreamEvent("content", text="<tool_call>TOP_SECRET_ARGUMENTS</tool_call>")
    decision = policy.classify([bad], end)
    messages = policy.correction_messages([bad], decision)
    assert policy.declared_names == ("pi_write",)
    assert policy.tools == original["tools"]
    assert [message["role"] for message in messages] == ["assistant", "user"]
    text = json.dumps(messages)
    assert "pi_write" in text and "At least one" in text
    assert "TOP_SECRET_ARGUMENTS" not in text and "<tool_call>" not in text
    assert all(
        "tool_calls" not in message and "tool_call_id" not in message
        for message in messages
    )
    with pytest.raises(ValueError, match="safe_recoverable"):
        policy.correction_messages([], policy.classify([], terminal([call()])))


def test_native_proof_redacts_secrets_and_paths_without_parsing_markup():
    policy = ToolRecoveryPolicy.from_request(request())
    decision = policy.classify(
        [],
        terminal(strata_rejected_tool_calls=native("secret-token")),
        secrets=("secret-token",),
    )
    assert decision.native_rejections[0].name == "<credential_redacted>"
    path = policy.classify(
        [], terminal(strata_rejected_tool_calls=native("/home/alice/private/file"))
    )
    assert "alice" not in path.native_rejections[0].name


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "not-a-type"},
        {"$ref": "https://invalid.example/schema"},
        {"$dynamicRef": "file:///private/schema"},
        {"$schema": "https://invalid.example/draft"},
        {"$schema": []},
    ],
)
def test_invalid_or_remote_request_schema_is_input_error(schema):
    with pytest.raises(ValueError, match="tool_schema"):
        ToolRecoveryPolicy.from_request(request(schema=schema))


def test_local_schema_reference_and_boolean_schema_are_validated():
    schema = {
        "$defs": {"path": {"type": "string"}},
        "type": "object",
        "properties": {"path": {"$ref": "#/$defs/path"}},
        "required": ["path"],
    }
    policy = ToolRecoveryPolicy.from_request(request(schema=schema))
    assert policy.classify([], terminal([call()])).accepted
    assert policy.classify([], terminal([call(arguments='{"path":1}')])).recoverable
    false_schema = ToolRecoveryPolicy.from_request(request(schema=False))
    assert false_schema.classify([], terminal([call()])).recoverable


def test_cyclic_python_schema_is_input_error_without_loop():
    schema = {"type": "object"}
    schema["properties"] = {"self": schema}
    with pytest.raises(ValueError, match="requires_json_values"):
        ToolRecoveryPolicy.from_request(request(schema=schema))


def test_missing_local_schema_reference_is_input_error_before_http():
    with pytest.raises(ValueError, match="tool_schema_unresolvable"):
        ToolRecoveryPolicy.from_request(request(schema={"$ref": "#/$defs/missing"}))


@pytest.mark.parametrize("name", ["$ref", "$dynamicRef"])
@pytest.mark.parametrize(
    "dialect",
    [
        None,
        "http://json-schema.org/draft-07/schema#",
        "https://json-schema.org/draft/2020-12/schema",
    ],
)
def test_property_names_that_look_like_reference_keywords_are_instance_fields(
    name, dialect
):
    schema = {
        "type": "object",
        "properties": {name: {"type": "string"}},
        "required": [name],
    }
    if dialect:
        schema["$schema"] = dialect
    policy = ToolRecoveryPolicy.from_request(request(schema=schema))
    proposal = call(
        arguments=json.dumps({name: "https://example.invalid/instance-data"})
    )
    assert policy.classify([], terminal([proposal])).accepted


@pytest.mark.parametrize("keyword", ["const", "default", "examples"])
def test_reference_keys_inside_instance_annotations_are_not_schema_references(keyword):
    instance = {
        "$ref": "https://example.invalid/ordinary-data",
        "$dynamicRef": "file:///ordinary-data",
    }
    schema = {
        "type": "object",
        keyword: [instance] if keyword == "examples" else instance,
    }
    policy = ToolRecoveryPolicy.from_request(request(schema=schema))
    assert policy.classify(
        [], terminal([call(arguments=json.dumps(instance))])
    ).accepted


@pytest.mark.parametrize("keyword", ["$ref", "$dynamicRef"])
@pytest.mark.parametrize("container", ["properties", "$defs"])
def test_true_nested_remote_schema_references_remain_blocked(keyword, container):
    schema = {
        "type": "object",
        container: {"nested": {keyword: "https://example.invalid/schema"}},
    }
    with pytest.raises(ValueError, match="requires_local_references"):
        ToolRecoveryPolicy.from_request(request(schema=schema))


def test_inactive_dynamic_reference_keyword_respects_draft7_dialect():
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "$dynamicRef": "https://example.invalid/unknown-annotation-in-draft7",
    }
    assert (
        ToolRecoveryPolicy.from_request(request(schema=schema))
        .classify([], terminal([call()]))
        .accepted
    )


@pytest.mark.parametrize("native_name", ["full", "native80", "normal"])
def test_native_truncation_cannot_leak_registered_secret_prefix(native_name):
    secret = "sk-proj-" + "X" * 100
    name = (
        secret
        if native_name == "full"
        else secret[:80]
        if native_name == "native80"
        else "write"
    )
    result = ToolRecoveryPolicy.from_request(request()).classify(
        [], terminal(strata_rejected_tool_calls=native(name)), secrets=(secret,)
    )
    assert result.recoverable
    if native_name == "normal":
        assert result.native_rejections[0].name == "write"
    else:
        exported = json.dumps(result.to_dict())
        assert "sk-proj-" not in exported and "X" * 20 not in exported
        assert result.native_rejections[0].name == "<credential_redacted>"


def test_report_uses_final_attempt_snapshots_and_keeps_unknown_fields_unknown():
    first = LLMToolRecoveryAttempt(
        0,
        "recoverable",
        usage=LLMUsageStats(
            prompt_tokens=20,
            completion_tokens=4,
            total_tokens=24,
            raw={"secret": "raw"},
        ),
    )
    last = LLMToolRecoveryAttempt(
        1,
        "accepted",
        usage=LLMUsageStats(
            prompt_tokens=30, completion_tokens=2, total_tokens=32, reasoning_tokens=0
        ),
    )
    report = LLMToolRecoveryReport((first, last), elapsed_ms=123.5)
    assert report.corrections_used == 1 and report.max_corrections == 3
    assert report.total_usage.prompt_tokens == 50
    assert report.total_usage.total_tokens == 56
    assert report.total_usage.reasoning_tokens is None
    assert report.observed_usage.reasoning_tokens == 0
    assert not report.usage_complete
    assert report.unknown_usage_fields == ("reasoning_tokens", "visible_output_tokens")
    assert report.attempts[0].usage.raw is None
    assert report.to_dict()["elapsed_ms"] == 123.5
    assert "secret" not in repr(report)


def test_missing_usage_is_explicit_partial_not_zero():
    report = LLMToolRecoveryReport(
        (
            LLMToolRecoveryAttempt(0, "recoverable"),
            LLMToolRecoveryAttempt(1, "accepted", usage=LLMUsageStats(prompt_tokens=9)),
        )
    )
    assert report.total_usage.prompt_tokens is None
    assert report.observed_usage.prompt_tokens == 9
    assert not report.usage_complete
    assert LLMToolRecoveryReport().total_usage.total_tokens is None


def test_zero_counters_are_observed_and_all_fields_can_be_complete():
    usage = LLMUsageStats(0, 0, 0, 0, 0)
    report = LLMToolRecoveryReport(
        (LLMToolRecoveryAttempt(0, "accepted", usage=usage),)
    )
    assert report.usage_complete and report.total_usage == usage


def test_immutable_attempts_and_report_enforce_three_correction_indices():
    attempts = [LLMToolRecoveryAttempt(index, "recoverable") for index in range(4)]
    report = LLMToolRecoveryReport(attempts, exhausted=True)
    attempts.clear()
    assert len(report.attempts) == 4 and report.corrections_used == 3
    with pytest.raises(FrozenInstanceError):
        report.exhausted = False
    with pytest.raises(FrozenInstanceError):
        report.attempts[0].index = 2
    with pytest.raises(ValueError, match="0_through_3"):
        LLMToolRecoveryAttempt(4, "recoverable")
    with pytest.raises(ValueError, match="consecutive"):
        LLMToolRecoveryReport((LLMToolRecoveryAttempt(1, "recoverable"),))
    with pytest.raises(ValueError, match="failed_third"):
        LLMToolRecoveryReport((LLMToolRecoveryAttempt(0, "accepted"),), exhausted=True)


def test_error_keeps_inspect_only_proposal_without_logging_or_mutable_alias():
    attempts = tuple(LLMToolRecoveryAttempt(index, "recoverable") for index in range(4))
    report = LLMToolRecoveryReport(attempts, exhausted=True)
    proposal = {
        "content": "PRIVATE TEXT",
        "tool_calls": [call(arguments='{"path":"SECRET_PATH"}')],
    }
    error = LLMToolRecoveryError(tool_recovery=report, last_proposal=proposal)
    proposal["content"] = "changed"
    snapshot = error.last_proposal
    snapshot["content"] = "consumer mutation"
    assert error.last_proposal["content"] == "PRIVATE TEXT"
    assert error.terminal_reason == "tool_recovery_exhausted"
    assert "PRIVATE TEXT" not in repr(error) and "SECRET_PATH" not in str(error)
    assert "PRIVATE TEXT" not in json.dumps(error.to_dict())
    refused = LLMToolRecoveryError(
        tool_recovery=LLMToolRecoveryReport((LLMToolRecoveryAttempt(0, "unsafe"),))
    )
    assert refused.terminal_reason == "tool_protocol_error"


def test_append_only_public_fields_preserve_existing_result_and_event_construction():
    result = LLMCallResult("answer")
    event = LLMStreamEvent("content", "answer")
    assert result.tool_recovery is None and event.tool_recovery is None
    report = LLMToolRecoveryReport(
        (LLMToolRecoveryAttempt(0, "incomplete_tool_response"),)
    )
    assert not report.exhausted
    assert LLMCallResult("partial", tool_recovery=report).tool_recovery is report
