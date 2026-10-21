# Functional requirements

The public contract covers Python adapters for an externally managed
llama.cpp HTTP server. This guide maps the requirements to the technical
documents and deterministic tests. It does not describe a private lifecycle
ledger or certify a particular deployment.

| Requirement | Contract and verification |
| --- | --- |
| Connect to a configured server without starting or compiling inference processes | [Architecture](ARCHITECTURE.md); factory and package boundary tests |
| Preserve blocking and streamed generation, conversation fields, reasoning, usage and terminal states | [Functional parity](FUNCTIONAL-PARITY.md); generation, client and adapter tests |
| Preserve tool calls and their correlation while leaving execution to the consumer | [Integration](INTEGRATION.md); generation and MCP data contract tests |
| Separate server reachability from model readiness | [Integration](INTEGRATION.md); adapter and discovery tests |
| Observe a bounded typed catalog without loading models; preserve unknown capabilities and metadata provenance | [Model discovery](MODEL-DISCOVERY.md); public discovery and transport tests |
| Keep physical configuration distinct from per-call sampling and logical consumer limits | [Architecture](ARCHITECTURE.md); configuration tests |
| Prepare model-specific query/document prefixes exactly once and count server special tokens | [Embedding contracts](EMBEDDING-CONTRACTS.md); prefix and preparation tests |
| Distribute a total text window across replicas and respect known aggregate HTTP token budgets | [Embedding contracts](EMBEDDING-CONTRACTS.md); embedding contract and scheduler tests |
| Expose immutable prepared inputs, measure oversize inputs and reject them before inference | [Embedding contracts](EMBEDDING-CONTRACTS.md); prepared-input contract tests |
| Preserve vector order, cardinality, dimensions and finite values | [Functional parity](FUNCTIONAL-PARITY.md); embedding and numeric integrity tests |
| Discover equivalent loaded replicas only from supported content and semantic identity attestations | [Replica discovery](AUTOMATIC-EMBEDDING-REPLICAS.md); discovery and replica configuration tests |
| Bound admission, queues and tokenization across callers sharing an adapter | [Concurrency](CONCURRENCY-AND-DIAGNOSTICS.md); concurrency and scheduler tests |
| Keep persistent replica queues and compose batches at dispatch, with shared sync/async execution and finite byte/token/request budgets | [Queue engine contract](PROGRESSIVE-EMBEDDING.md); persistent-engine, configuration and async lifecycle tests |
| Preserve typed failures, cancellation, cleanup and resource ownership without automatic retries | [Integration](INTEGRATION.md); transport, cancellation and cleanup tests |
| Allow explicit idle-free generation while preserving finite non-SSE budgets and exact-call Stop/cleanup | [Stream timeouts](STREAM-TIMEOUTS.md); nullable config, optional idle transport and adapter tests |
| Capture bounded thinking diagnostics only when enabled | [Concurrency](CONCURRENCY-AND-DIAGNOSTICS.md); thinking diagnostic tests |

Use the commands in the [README](../README.md#verification) to run local
verification. Tests use synthetic inputs and local HTTP fixtures. Live probes
are optional, require explicit opt-in and must record the server build, model,
template and configuration actually exercised.

Generation template behavior, schema conformance, numerical equivalence across
replicas, server-side cancellation and a consuming application's full tool
workflow require qualification in that environment. The public `0.5.0`
release adds read-only discovery with deterministic verification; it makes no
new live qualification or consumer acceptance claim.
