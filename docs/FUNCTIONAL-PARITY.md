# Functional contract coverage

`llama-runtime` provides a Python HTTP client, contracts, profiles and adapters
for generation and embedding with llama.cpp servers. The functional comparison
below concerns behavior, not identical server protocols or implementations.
It does not certify live compatibility or consumer acceptance.

The transport derives from the client documented in
[PROVENANCE.md](PROVENANCE.md). This path does not depend on the OpenAI SDK or
the upstream package's bootstrap code. The Python distribution remains
`runtime-llama`, with import namespace `runtime_llama`.

## Behavior map

| Area | Library contract | Adaptation or limit |
| --- | --- | --- |
| Contracts and roles | Generation/embedding interfaces, results, events, errors, profiles and Python adapter factories. | The backend is llama.cpp HTTP; there is no alternative-provider registry. |
| Identity and context | Endpoint, model, declared runtime identity and external context are separate from role/request parameters. Logical limits are compared with known context. | The library does not inspect remote binaries or reallocate context/KV. Missing metadata stays unknown. |
| Generation | Blocking and streaming calls, visible text, separate reasoning, removal of inline thinking and structured output. | Chat uses `/v1/chat/completions`. Templates, schemas and reasoning options need qualification with the intended build and model. |
| History and tools | Structured messages, tool results with `tool_call_id`, tool schemas and tool-call deltas preserve names, identifiers and arguments. | The library returns tool calls; it does not execute them. |
| Protections | Cooperative cancellation, including stream waits, timeouts, repetition detection and a visible-output limit derived from the budget. | The visible limit is a character estimate, not exact tokenization. Generation timeouts are not an overall deadline, and client closure does not prove remote inference has stopped. |
| Usage and outcomes | Observed counters, unknown absent fields, terminal reasons and bounded diagnostics. Reading continues after `finish_reason` to collect final usage. | `[DONE]` ends transport only. A stream without `finish_reason` fails. Derived counters are marked, and tool tokens are not attributed to visible prose. |
| Embedding | Exact pre-tokenization, text/token batch budgets, stable ordering, finite vectors, consistent dimensions and optional normalization. | `/tokenize`, `/props` and `/v1/embeddings` provide server capabilities. Configured/observed limits stay distinct. Inputs are never automatically truncated. |
| Batch recovery | Adaptive splitting after structured HTTP 400 `exceed_context_size_error`, with gradual capacity recovery. | A generic 400, authentication failure or server error does not trigger splitting. An oversized single input remains an error. Transport has no hidden retries. |
| MCP data | Data types and validation for integrations, calls, providers, events and results. | There is no MCP executor. Unsupported invocation methods are explicit; provider-specific legacy payloads are not declared llama.cpp APIs. |

## Operational boundaries

The configured server can be local or remote. Its operator owns builds,
binaries, processes and loading resources. The library neither launches a
local runtime nor selects a fallback when an endpoint is unavailable.

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
of an adapter, with a bounded queue and per-replica limits. Optional thinking
diagnostics are bounded and correlated per invocation without changing
`reasoning_text`, budgets or protections. Concurrent JSON/SSE transport has
explicit ownership. Model availability remains distinct from server
reachability. See [integration](INTEGRATION.md) and
[concurrency and diagnostics](CONCURRENCY-AND-DIAGNOSTICS.md).

Automatic embedding discovery uses content and semantic identities published
in Noxi's native catalog. The consumer selects one logical model without listing
or counting instances. Each automatic replica permits one active request.
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

## Licensing and attribution

The project uses [Apache-2.0](../LICENSE), copyright 2026 Davide Mele. The derived
llamatelemetry client retains its upstream MIT license and attribution.
[PROVENANCE.md](PROVENANCE.md) and [NOTICE](../NOTICE) describe those origins.
