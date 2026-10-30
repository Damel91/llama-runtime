# Public release 0.17.0 verification

Date: 2026-10-30, Europe/Rome. Distribution `runtime-llama`, import
`runtime_llama`, Python 3.11 or later. This release adds automatic recovery
of safely rejected LLM tool proposals inside one SDK invocation.

Flower MCP owns requirements `REQ-000073`–`REQ-000079`, milestone
`MILE-000042`, change `CHANGE-000044`, packet `PACKET-000045` and adopted
architecture revision 7. The repository explains that named architecture;
it does not replace the canonical ledger. Implementation, review and test
evidence are host-reported. Release publication does not imply application
acceptance or close unrelated historical verification obligations.

## Delivered contract

The initial generation is attempt 0. At most three corrective generations,
attempts 1–3, can follow inside that invocation. Exhaustion ends the current
call with `LLMToolRecoveryError`; concurrent and subsequent external calls
have independent budgets. Tool-enabled events are staged until classification,
so rejected text and tool fragments cannot be dispatched by a consumer.

The SDK preserves explicit native rejection evidence and distinguishes it
from local name, JSON, schema and tool-choice validation. XML-like output
alone is not proof of rejection. No tool is executed, aliased or reconstructed
from XML. Corrections preserve request options and the selected runtime/model;
they append SDK feedback to private history without mutating caller messages.
Uncertain server-side MCP effects prevent regeneration. HTTP failures, EOF,
timeout, cancellation and runaway detection are not generic retry triggers.

Results, stream events and errors expose typed recovery evidence. Ordinary
usage describes the accepted generation; the report accounts for discarded
attempts separately and keeps missing counters unknown. Tool-enabled streams
may deliver their first visible event only after native generation completes.
No native async LLM method is added. See [API signatures and consumer examples](../TOOL-RECOVERY.md).

## Deterministic and installed-package checks

The final suite passed **2,592 tests in 109.92 seconds**. Four warnings are
from deliberately unverified HTTPS loopback fixtures. Ruff lint, formatting
and whitespace checks passed. The final implementation/version commit is
`e9d9d32873b889c27b7402fc5d5e011530ef9d3b`; documentation is committed separately.

Tests cover late, absent and malformed native diagnostics; mixed proposals;
schema-resource traversal without external retrieval; counter reset and
exhaustion; concurrent calls; fixed automatic routing; negative controls for
transport errors and server MCP; actual HTTP/SSE cancellation and drained
close; asynchronous thread-bridge abandonment; and reentrant cancellation
callbacks that close an adapter while returning `False`.

The final wheel was installed into a clean environment outside the checkout,
with pinned tested dependencies and a successful `pip check`. Installed
runtime bytes match the wheel. The smoke exercises automatic routing,
structured output and tool recovery over actual loopback HTTP, including
four rejected generations, a fresh successful next invocation and abandoned
iterator cleanup. It also verifies 10,000 synthetic progressive embedding
records in order, with released owner/result/handoff/HTTP reservations.
Those fixtures execute no native inference and no tools.

The Python wheel remains `py3-none-any`. Existing dependencies, licenses,
embedding scheduler, 2,500-text corpus and category 80% / mixed 90% margins
are unchanged. There is no new text, model or memory quota.

## Limited native qualification

Two native POSTs to the already loaded Strata model passed: one benign
tool proposal, then a separate invocation whose initial rejection was injected
over a loopback HTTP boundary and whose corrective generation was native.
The SDK emitted the correction without host intervention. There were twelve
metadata GETs, no tokenizer or administration requests, and no tool execution.
Model/build observations matched before and after. Owned native responses and
relay handlers were zero after close.

The model was `qwen3.8-flash-next-iq3_s`; properties reported engine build
`Strata 0.1.52`. The remote frontend revision was not independently identified.
**No spontaneous native rejection marker was observed.** The injected marker
proves client recovery across HTTP; it does not prove that this remote
frontend emits that marker. The documented marker contract is grounded in
[Strata source at v0.2.8](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/serve/server.py).
The original conversation's HTTP response was not captured, so its native
marker presence cannot be established retrospectively.

Live execution used a private candidate whose metadata still declared
0.16.0. Final packaging directly compares every runtime file with that retained
candidate: only the `__version__` assignment differs. The final 0.17.0 wheel
has its own clean installed-package smoke. The manifest records both facts;
it does not claim that final-version metadata was executed natively.

The initial probe timed out and showed incomplete relay drain. Its relay had
buffered short SSE data and waited beyond `[DONE]`. Corrected framing stops
at native completion and joins owned handlers; actual HTTP fixtures cover
persistent connections and silent timeouts. A separate intermittent fixture
failure was corrected by advertising `Connection: close`, without adding
transport retries or weakening assertions. Ten consecutive focused runs then
passed. Original failed experiments remain in private verification history.

These are client wall-time and request-lifecycle checks, not pure GPU timing,
a throughput benchmark or a model-quality result. Native llama.cpp inference,
long-context load, spontaneous native rejection and downstream application
acceptance are not newly certified here.

## Distribution and handoff

Wheel: `runtime_llama-0.17.0-py3-none-any.whl`.

SHA-256: `5a3933eebe73928abed4bfe54c4f92fa51f0beb6e1a94e5fc0a611825bbb9bbb`.

The release contains the wheel, normalized source archive, manifest, SHA-256
checksums and three optional tsnet helper archives. Audits compare committed
source and package payloads, verify examples, licenses, corpus hashes and
normalized metadata, and exclude private endpoints, credentials and paths.
Published main/tag and anonymously downloaded asset hashes are checked against
the staged bytes before delivery is reported complete.

Apache-2.0, copyright Davide Mele, NOTICE and upstream attribution remain.
Optional helpers reuse byte-identical verified native binaries and complete
notices; no new OS execution or VPN enrollment qualification is claimed.
The accepted historical embedding-profile deviation remains `FIND-000007`,
with unchanged thresholds and no new numerical pass.

Consumers can keep existing generation methods and inspect the additive
`tool_recovery` fields and typed error. They still execute tools, retain tool
identifiers and send subsequent tool-result messages. Exhaustion should end
the invocation rather than start an unbounded outer retry loop. Consumer
code, pins, installations, jobs, services, databases and model profiles were
not updated. Publication is a GitHub release, without a PyPI upload.
