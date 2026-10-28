# Public release 0.16.0 verification

Date: 2026-10-28, Europe/Rome. Distribution `runtime-llama`, import
`runtime_llama`, Python 3.11 or later. This release adds Strata for LLM
inference and automatic selection between configured llama.cpp and Strata
endpoints. It preserves the existing llama.cpp embedding engine, scheduler,
2,500-text calibration corpus and category 80% / mixed 90% margins.

Flower MCP records requirements `REQ-000070`–`REQ-000072`, milestone
`MILE-000041`, change `CHANGE-000043` and packet `PACKET-000044`.
Implementation and verification evidence are host-reported. Publishing a
package does not establish application acceptance, close historical acceptance
obligations or certify every server build and model profile.

## Delivered behavior

- `StrataAdapter` shares the existing message, HTTP/SSE, tool-call assembly,
  reasoning, usage, timing, cancellation and transport cleanup contracts.
- `runtime_endpoints` selects automatic mode without requiring a consumer
  backend choice. The deployment declares host/port, TLS, authentication and
  optional proxy configuration for each endpoint.
- `discover_runtimes()` and `AutoRuntimeAdapter.discover_models()` return
  immutable observations and loaded-model catalogs per runtime. Three GETs per
  endpoint identify health, root properties and catalog under a finite deadline;
  the catalog is acquired once. There is no network scan or model administration.
- Generation refreshes the configured catalogs, resolves the exact requested
  model or unambiguous alias and pins a client/backend/canonical model for that
  request. Concurrent streams retain their own routes. Results and events carry
  optional `runtime_selection`; snapshot properties are observational state.
- Unknown or unavailable runtimes remain visible as diagnostics. Conflicting
  runtime markers, model identities or aliases cannot silently select another
  model. A submitted POST is not replayed on another endpoint.
- Strata JSON Schema uses the provider's nested `json_schema` format and local
  validation before a successful complete response. An unsatisfiable schema is
  rejected even if the server returns ordinary JSON successfully. External
  reference retrieval is disabled; invalid application schemas fail before HTTP.
  Partial streams and tool responses do not promise complete schema validity.
- Stop, timeout and close retain endpoint-specific ownership and drain active
  clients and discovery workers. Close also interrupts sockets for paused stream
  iterators. No new text, model, endpoint or memory quota is introduced.

See [public APIs and examples](../AUTOMATIC-RUNTIMES.md).

## Deterministic and package verification

The final repository suite passed **2,418 tests in 107.65 seconds**. Four
warnings come from deliberately unverified HTTPS loopback fixtures. Ruff
checks, formatting and the source whitespace check passed. Package audit
results are recorded in the release manifest.
The implementation commit is `1816384e0a26a26f8593f38248812342b4352843`;
documentation is committed separately.
Focused regressions cover both backends over real loopback HTTP, exact/alias
selection, conflicting and unavailable endpoints, model movement between
overlapping streams, endpoint credential isolation, schema failures/recovery,
finite discovery with an idle-free stream, cancellation during catalog parsing,
paused-stream close, concurrent drain, coherent snapshots and no state
resurrection after close. Existing generation and embedding regressions remain.

A pre-existing asynchronous embedding fixture could block its sole control
worker while waiting for a sibling completion queued on that same worker.
The fixture synchronization was corrected without changing the production
embedding engine, scheduling policy or timeout values. The corrected test
passed 20 out of 20 fresh-process repetitions; its 55-test module passed, and
a mutation check still failed when the real empty-output guard was removed.
The isolated failure and
its diagnosis remain in private verification history instead of being relabeled
as a model or runtime regression.

The wheel is installed outside the source checkout with resolved dependencies
and a successful dependency check. Installed runtime payloads are compared with
the final wheel. The smoke uses actual local HTTP for automatic catalogs,
Strata routing, usage/terminal metadata, schema rejection and recovery, plus
10,000 synthetic progressive embedding records with ordered output and no
remaining owner/result/handoff/HTTP reservations. No consumer installation is
updated by these checks.

The Python wheel remains `py3-none-any`; JSON Schema validation adds
`jsonschema>=4.23,<5`. No model runtime, GPU framework, OpenAI/LM Studio SDK or
Flower package is a core dependency. Local Python execution and remote Linux
service compatibility are recorded separately; this is not a claim of new
execution tests on every supported OS or Python version.

## Limited live qualification

The operator's two configured endpoints were reachable during the probe. The
native llama.cpp catalog reported one loaded EmbeddingGemma replica, with an
effective observed context of 2,048. Strata reported the loaded canonical model
`qwen3.8-flash-next-iq3_s`, context 262,144, and engine build
`Strata 0.1.52`. The remote frontend commit was not independently identified.

**Ten live cases passed:** catalog recognition of both runtimes, automatic
blocking chat, structured history, stream metadata, a single forced tool call
and correlated result, schema-conforming JSON, rejection of an unsatisfiable
schema, cancellation before discovery, Stop during active streaming and recovery.
The tool test performs only a synthetic local addition; it does not execute
MCP tools or consumer jobs.

There were **71 HTTP requests**: 60 discovery GETs across both endpoints, nine
chat-completion POSTs to Strata, and two final read-only properties/slot checks.
The initial catalog required six GETs, three per runtime. There were no inference
POSTs to llama.cpp, tokenizer requests or administration calls. The final Strata
slot was idle. The same ten cases also passed from the clean-installed wheel. These are request counts for this short qualification, not an
embedding benchmark or a pure GPU timing measurement.

In the initial live run, automatic blocking chat took approximately 0.47 seconds, the schema case 0.67
seconds, and the tool roundtrip 1.96 seconds in this run. Those client wall times
include discovery, networking, inference, decoding and cleanup. They do not
establish sustained throughput, a universal speedup, full-context behavior or
long silent-prefill qualification. Multiple tool calls are covered by
deterministic transport tests; remote model instruction adherence for multiple
calls is not asserted here.

## Release artifacts and boundaries

The release contains a wheel, normalized source archive, manifest, SHA-256
checksums and three optional tsnet helper archives. Package checks compare the
wheel/source payloads to committed source, verify archive metadata, public API
signatures, examples, corpus hashes, licenses and exclusions of private paths,
credentials and deployment addresses. Anonymous download verification checks
the published bytes, main and tag against the staged release.

Wheel: `runtime_llama-0.16.0-py3-none-any.whl`.

SHA-256: `b02637c724b942d249515415a3a73deadca48fa2d6192a0ef3ed77ef4e77f0e6`.

Apache-2.0, copyright Davide Mele, NOTICE and upstream MIT attribution remain.
Optional helper executables reuse verified v0.6.0 binaries with their complete
notices; no new native-platform, VPN enrollment or VPN inference qualification
is implied. No consumer code, configuration, installation, job, service, model
profile or database is changed. This is a GitHub release, without a PyPI upload.

The earlier operator-accepted embedding-profile numerical deviation remains
`FIND-000007`, with unchanged strict thresholds. This LLM integration does not
relabel that finding as a numerical pass or claim new embedding quality results.
Strata embeddings, other inference providers and a Rust port are outside this
release.
