# Tool-calling recovery: planned 0.17.0 contract

Flower MCP is the canonical lifecycle and architecture authority. This is an
explanatory snapshot of architecture revision 7 and the accepted work plan for
`CHANGE-000044` / `PACKET-000045`, specification 8, plan 6. It does not claim
implementation, qualification or application acceptance.

The SDK will preserve authoritative native tool-rejection diagnostics and
validate completed structured calls against the current declared tool names
and schemas. Ordinary prose, quoted XML and code remain text when no verified
protocol obligation is violated. The SDK never executes tool markup or invents
tool aliases.

Each external invocation owns its budget: initial generation 0 followed by at
most three corrective generations, numbered 1–3. Internal corrections do not
reset that budget. Success ends the recovery sequence; every new external
invocation starts fresh, even on the same client after an exhausted call.
There is no persistent circuit breaker or shared concurrent-call counter.

Tool-enabled attempts are staged until their final protocol disposition so
rejected or mixed proposals cannot be executed before correction. Existing
thinking, runaway, output, cancellation and transport guards remain active
while receiving. This changes tool-enabled streaming latency; the ordinary
no-tools stream remains progressive. Only an accepted proposal reaches the
consumer, once. The route, model, tool definitions and sampling settings stay
fixed through corrective generations.

Ambiguous transport failures and requests that may have executed server-side
tools are not automatically replayed. Stop, close, timeout, early EOF, output
caps and runaway detection remain terminal according to their existing
contracts. Recovery never changes models, consumer configuration or embedding
scheduling.

Typed per-attempt evidence will accompany events, results and terminal errors.
Ordinary usage retains accepted-generation semantics; recovery reports account
for the final usage snapshot of each attempt, with missing counters explicit.
The last unaccepted proposal is inspection evidence, not permission to execute
its calls. Prompt text, tool arguments and credentials are not diagnostic logs.

Implementation order: pure policy and outcome contracts; guarded adapter and
automatic-route integration; deterministic/HTTP/concurrency qualification;
clean package installation, public documentation and release. Tests include
three failed corrections without a fourth, a fresh invocation after exhaustion,
mixed calls without replay, quoted XML, late/missing/malformed diagnostics,
prefill cancellation, abandonment and concurrent route/budget isolation.

The native diagnostic contract is grounded in Strata v0.2.8 source
[`870aedd9`](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/serve/server.py).
That source is not evidence that a particular installed model emitted the field
in an earlier incident. Native live checks and injected protocol fixtures will
be reported separately.
