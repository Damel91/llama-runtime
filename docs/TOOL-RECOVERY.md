# SDK tool-call recovery in 0.17.0

The LLM adapters can correct a recoverable tool proposal inside one public
generation call. This applies to `LlamaCppAdapter`, `StrataAdapter` and
`AutoRuntimeAdapter`, including adapters returned by `build_adapter()`.
The SDK validates proposals and requests another generation; the consumer
continues to own tool execution and subsequent conversation messages.

## Calling the adapter

Declare function tools with their argument schemas. Recovery is automatic when
the effective request declares tools; there is no host retry loop to implement.

```python
from runtime_llama import build_adapter, LLMToolRecoveryError

tools = [{
    "type": "function",
    "function": {
        "name": "lookup_item",
        "description": "Look up an item by its identifier.",
        "parameters": {
            "type": "object",
            "properties": {"item_id": {"type": "string"}},
            "required": ["item_id"],
            "additionalProperties": False,
        },
    },
}]

llm = build_adapter({
    "model": "chat-model",
    "runtime_endpoints": [
        {"name": "native", "base_url": "http://localhost:8080"},
        {"name": "alternate", "base_url": "http://localhost:8082"},
    ],
    "temperature": 0.2,
})
try:
    result = llm.generate_with_metadata(
        "Look up item example-42.", tools=tools, tool_choice="required"
    )
    # Structured data only: this example executes no proposed function.
    print(result.tool_calls)
    if result.tool_recovery is not None:
        print(result.tool_recovery.to_dict())
except LLMToolRecoveryError as exc:
    print(exc.terminal_reason, exc.tool_recovery.to_dict())
finally:
    llm.close()
```

`generate_with_metadata()` returns `LLMCallResult`.
`stream_with_metadata()` accepts the same tool arguments and yields
`LLMStreamEvent`. The common `invoke_with_ctx()` and `stream_with_ctx()` methods
accept tools through `request_options={"tools": tools, "tool_choice": "required"}`.
The text-only methods do not expose tool-call or recovery metadata.

## Attempts and publication of output

One external call has an initial generation, indexed **0**, and at most **three
corrective generations**, indexed **1–3**. If the third correction remains
recoverably invalid, the call raises an exhausted error. A new external call
starts with the full budget, including after exhaustion. Concurrent calls have
independent budgets and private corrective histories.

For requests declaring tools, the SDK stages native events until the complete
proposal can be classified. Content and executable tool fragments from a
rejected attempt never escape through the public stream. An accepted attempt
is projected once, in its original event order. This can delay the first
visible event until that native generation finishes. A mixed proposal is
regenerated as a whole only because none of its proposed calls was released.

Ordinary requests without declared tools retain progressive streaming. An
explicit nonempty or malformed native rejection marker is an exception: it
raises a protocol error without regeneration, because there is no declared
client-tool authority. An absent or empty marker does not change that path.

The effective tools, schemas, inference settings and server options are frozen
for the invocation. Corrections add SDK-authored assistant/user feedback to its
private history; they do not mutate the caller's messages or fabricate tool
results. Automatic routing discovers and selects once for the call, then keeps
every correction on that selected runtime, model and client. The next external
call performs its normal fresh discovery.

## Evidence and conditions for correction

| Evidence | SDK behavior |
| --- | --- |
| Explicit native `strata_rejected_tool_calls` entries with reason `undeclared` | Retains the observed evidence and can request a correction. |
| Completed structured call with an undeclared name, invalid argument JSON or arguments that fail its declared schema | Classifies the completed proposal locally and can request a correction. |
| Completed response violating `tool_choice`, or a completed tool-call finish without calls | Can request a correction under the same safety rules. |
| XML-like text, quoted examples or code containing tool syntax | No rejection is inferred from the text alone. |
| Malformed protocol evidence, incomplete responses or unresolvable schema contracts | Does not treat the problem as a safe corrective generation. |

Native diagnostics are optional. SDK validation evidence is recorded with its
own provenance; it is not presented as a native server rejection. An HTTP 200
response can still contain a rejected or invalid tool proposal.

The SDK never executes tools, resolves tool-name aliases or turns XML into an
executable call. If effective `strata_mcp=True` or observed server-side MCP
activity means tool effects may already have occurred, a rejected proposal is
not replayed. The SDK cannot promise to undo server-side effects.

Transport failures, early EOF, timeout, Stop, close and runaway detection are
terminal for the invocation, without a generic generation retry. Output caps
and native incomplete finishes retain existing partial-text behavior while
suppressing candidate tool calls; they do not start a correction.

## Reports, usage and errors

The additive public types are exported from `runtime_llama`:

```text
LLMToolRecoveryRejection(name: str, reason: str,
                        source: str = "strata_rejected_tool_calls")
LLMToolRecoveryAttempt(index: int, outcome: str, reasons: tuple[str, ...] = (),
    native_rejected_tool_calls: int | None = None,
    usage: LLMUsageStats | None = None,
    transport: LLMTransportAttempt | None = None,
    server_timings: LLMServerTimings | None = None,
    native_rejections: tuple[LLMToolRecoveryRejection, ...] = (),
    thinking_diagnostic: ThinkingDiagnostic | None = None,
    provenance: tuple[str, ...] = ())
LLMToolRecoveryReport(attempts: tuple[LLMToolRecoveryAttempt, ...] = (),
                      exhausted: bool = False, elapsed_ms: float = 0.0)
LLMToolRecoveryError(message: str = "tool_recovery_exhausted", *,
    tool_recovery: LLMToolRecoveryReport, last_proposal: object = None,
    diagnostics: LLMTransportAttempt | None = None)
LLMToolRecoveryReport.to_dict(*, include_thinking: bool = False) -> dict
LLMToolRecoveryAttempt.to_dict(*, include_thinking: bool = False) -> dict
LLMToolRecoveryError.to_dict() -> dict
```

`LLMCallResult.tool_recovery` and `LLMStreamEvent.tool_recovery` are optional
`LLMToolRecoveryReport` snapshots. Metadata streams can also emit
`kind="tool_recovery"`; a corrective boundary includes
`event.data["next_attempt"]`. This is recovery progress, not ordinary usage or
an executable tool event. Consumers must allow recovery events without that
optional next-attempt field, such as a partial-output terminal path.

A report exposes `attempts`, `corrections_used`, `max_corrections`, `exhausted`
and `elapsed_ms`. Each `LLMToolRecoveryAttempt` includes its `index`, `outcome`,
`reasons`, `provenance`, optional `native_rejected_tool_calls` count and
`native_rejections`, plus available usage, transport and server timings.
A missing native observation remains unknown, rather than becoming zero.

Successful ordinary result usage describes the accepted generation. The report
keeps final per-generation snapshots separately, so discarded attempts remain
visible without adding cumulative streaming counters repeatedly:

- `total_usage` sums each counter only when every attempt supplied that counter;
  otherwise that counter is `None`.
- `observed_usage` sums known counters; `unknown_usage_fields` and
  `usage_complete` describe missing coverage.
- `elapsed_ms` is selected-backend recovery-coordinator wall time. It excludes
  automatic discovery and the later replay of the final staged result. It is
  neither pure inference time nor GPU execution time. Server timings remain
  separate, optional per-generation observations.

`LLMToolRecoveryError` derives from `LLMBackendError` and exposes the report as
`exc.tool_recovery`. Its `terminal_reason` is `tool_recovery_exhausted` only
after the failed third correction; otherwise it is `tool_protocol_error`.
Terminal transport/cancellation errors retain their existing types and can
carry the report when an attempt has been recorded.

`exc.last_proposal` is a copy for explicit inspection only. It must never be
executed as a fallback. It can contain model output and is omitted from the
exception text, representation and `to_dict()` export. Report exports omit
thinking diagnostics unless `to_dict(include_thinking=True)` is requested;
actual capture still requires the ordinary thinking-diagnostics opt-in.

## Cancellation and iterator ownership

Existing cancellation probes, stream-idle settings and runaway guards remain
active while events are staged. Each native attempt is closed before a
correction begins. Closing an adapter also retires its staged proposals; a
probe that closes the adapter and returns `False` cannot permit late output.

Close a stream iterator when abandoning it, including after a recovery-progress
event. This releases retained staged output and prevents another correction.
With an open adapter configured as above, before calling `llm.close()`:

```python
events = llm.stream_with_metadata("Look up item example-42.", tools=tools)
try:
    for event in events:
        if event.kind == "terminal":
            print(event.data.get("tool_calls", []))
finally:
    events.close()
```

This release adds no native asynchronous LLM method. The public generation and
stream methods remain synchronous. An application using a thread bridge must
still arrange cooperative cancellation and iterator cleanup; cancelling an
async task alone does not stop work running in another thread.
