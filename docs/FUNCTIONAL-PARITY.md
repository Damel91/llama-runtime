# Functional contract coverage

`llama-runtime` provides a Python HTTP client, contracts, profiles and adapters
for LLM generation with llama.cpp or Strata, and embeddings with llama.cpp. The functional comparison
below concerns behavior, not identical server protocols or implementations.
It does not certify live compatibility or consumer acceptance.

The transport derives from the client documented in
[PROVENANCE.md](PROVENANCE.md). This path does not depend on the OpenAI SDK or
the upstream package's bootstrap code. The Python distribution remains
`runtime-llama`, with import namespace `runtime_llama`.

## Behavior map

| Area | Library contract | Adaptation or limit |
| --- | --- | --- |
| Contracts and roles | Generation/embedding interfaces, results, events, errors, profiles and Python adapter factories. | LLM backends are llama.cpp and Strata; deployment endpoints can be probed and selected automatically from the requested model. Embeddings remain llama.cpp-only. |
| Identity and context | Endpoint, model, declared runtime identity and external context are separate from role/request parameters. Logical limits are compared with known context. | The library does not inspect remote binaries or reallocate context/KV. Missing metadata stays unknown. |
| Generation | Blocking and streaming calls, visible text, separate reasoning, removal of inline thinking and structured output. | Chat uses `/v1/chat/completions`. Templates, schemas and reasoning options need qualification with the intended build and model. |
| History and tools | Structured messages, tool results with `tool_call_id`, tool schemas and tool-call deltas preserve names, identifiers and arguments. | The library returns tool calls; it does not execute them. |
| Protections | Cooperative cancellation, including stream waits, timeouts, repetition detection and a visible-output limit derived from the budget. | The visible limit is a character estimate, not exact tokenization. Generation timeouts are not an overall deadline, and client closure does not prove remote inference has stopped. |
| Usage and outcomes | Observed counters, unknown absent fields, terminal reasons and bounded diagnostics. Reading continues after `finish_reason` to collect final usage. | `[DONE]` ends transport only. A stream without `finish_reason` fails. Derived counters are marked, and tool tokens are not attributed to visible prose. |
| Embedding | Optional exact preparation, automatic direct strings, optional text targets, ordered complete/progressive results, finite vectors, consistent dimensions and normalization. | Exact counts use `/tokenize`; ordinary automatic inputs do not. Context, native execution sizes and explicit HTTP policies stay distinct. Inputs are never silently truncated. |
| Batch recovery | A recognized custom-client 400 splits only its failed POST; automatic 500 isolation retains successful vectors and returns lossless halves for rejected singleton inputs. | No persistent reduced target, ordinary per-input tokenization, saved-budget recalculation or ambiguous transport replay. A generic server error does not prove context overflow. |
| MCP data | Data types and validation for integrations, calls, providers, events and results. | There is no MCP executor. Unsupported invocation methods are explicit; provider-specific legacy payloads are not declared llama.cpp APIs. |

## Operational boundaries

The configured server can be local or remote. Its operator owns builds,
binaries, processes and loading resources. Automatic mode selects a currently
loaded matching model among configured endpoints before submitting a request.
It never launches a runtime, substitutes another model or replays a submitted
POST on a fallback. See [automatic runtime routing](AUTOMATIC-RUNTIMES.md).

The selected server's auto-load policy is allowed. The client does not force
`autoload=false` or independently run a load/unload protocol. Adapter `load()`
compatibility methods verify availability; they are not loading controllers.
Requests remain subject to the configured server policy.

Application prompts, conversation context, tool authorization and orchestration
belong to consumers. A consumer can supply its own tool/MCP executor. The
library preserves the data needed for that handoff without invoking tools or
MCP servers for the model.

## Verification boundary

The client, generation, embedding and shared configuration/contract tests use
deterministic fixtures, injected transports and loopback HTTP servers. Coverage
includes stream closure, typed/redacted errors, final usage and batch integrity.
See the [README verification section](../README.md#verification) and
[verification requirements](REQUIREMENTS.md) for reproducible checks.

Software tests do not certify every server build or consumer. Before claiming
operational compatibility, identify the build, model, template and server
configuration and exercise the capabilities the application actually uses.
Transport properties do not promise universal cancellation latency, remote
termination or improved token throughput. A complete model → authorized tool
→ tool result → final response cycle requires separate consumer verification.

## Additional contracts

Embedding concurrency is independent of tokenization and shared across callers
of an adapter. Queue, owner and byte quotas are optional explicit policies,
not inferred from host RAM. Default worker capacity follows verified replicas;
one original POST lease per replica and explicit B/G policies remain. Optional thinking
diagnostics are bounded and correlated per invocation without changing
`reasoning_text`, budgets or protections. Concurrent JSON/SSE transport has
explicit ownership. Model availability remains distinct from server
reachability. See [integration](INTEGRATION.md) and
[concurrency and diagnostics](CONCURRENCY-AND-DIAGNOSTICS.md).

Automatic embedding discovery uses content and semantic identities published
in Noxi's native catalog. The consumer selects one logical model without listing
or counting instances. Each automatic replica permits one active request.
Catalog/replica cardinality has no fixed application maximum.
Context, GPU assignment and Flash Attention remain separate execution
properties. Explicit replicas remain supported; see
[replica discovery](AUTOMATIC-EMBEDDING-REPLICAS.md).

Version 0.4.0 introduced balanced distribution of a total text budget across
verified aliases, one shared scheduler, packing within observed native batch
limits and public task-aware preparation through `prepare_texts`,
`prepare_query` and `embed_prepared`. Preparation measures oversized inputs;
count reuse requires verifiable identity/tokenizer policy. Embedding calls have
an overall deadline and tokenizer admission is global to the adapter.
Splitting/AST processing, model administration and consumer orchestration stay
outside this contract. See [embedding contracts](EMBEDDING-CONTRACTS.md).

Version 0.4.1 changes public distribution metadata and documentation while
preserving the inference and embedding behavior of 0.4.0. It does not extend a
historical live qualification to a new deployment or consumer.

Version 0.5.0 adds the [public read-only discovery API](MODEL-DISCOVERY.md):
immutable catalog and loaded observations, explicit capability unknowns,
independent identity assertions and context provenance. The shared HTTP client
adds optional decompressed JSON limits and explicit ambient-session policy.
Existing generation and embedding APIs remain compatible; new deterministic
checks do not establish live or consumer acceptance.

Version 0.15.0 adds flat `aiter_texts()` delivery and removes implicit text,
model and memory quotas. Legacy scalar values only apply behind
`embedding_capacity_limits=True`; cache retention requires explicit enabling.
Existing complete-list/window APIs remain. Finished inputs, results, Futures,
tasks and topology generations release at ownership transfer or real drain,
including abandoned iteration and retained public failures. This does not
promise constant RSS for caller-held output or introduce automatic disk spill.
See [current resource contracts](CONCURRENCY-AND-DIAGNOSTICS.md).

## Licensing and attribution

The project uses [Apache-2.0](../LICENSE), copyright 2026 Davide Mele. The derived
llamatelemetry client retains its upstream MIT license and attribution.
[PROVENANCE.md](PROVENANCE.md) and [NOTICE](../NOTICE) describe those origins.
