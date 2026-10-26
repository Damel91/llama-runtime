# Functional requirements

The current 0.15.0 lifecycle follows Flower's adopted architecture revision 6,
requirements `REQ-000061`–`REQ-000069`, milestones `MILE-000036`–`MILE-000040`
and packets `PACKET-000039`–`PACKET-000043`. The
[earlier optimization plan](EMBEDDING-OPTIMIZATION-PLAN-20261025.md) remains
historical context. Requirement registration does not establish implementation,
verification or application acceptance.

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
| Observe a typed catalog with optional explicit policies without loading models; preserve unknown capabilities and metadata provenance | [Model discovery](MODEL-DISCOVERY.md); public discovery and transport tests |
| Keep physical configuration distinct from per-call sampling and logical consumer limits | [Architecture](ARCHITECTURE.md); configuration tests |
| Prepare model-specific query/document prefixes exactly once and count server special tokens | [Embedding contracts](EMBEDDING-CONTRACTS.md); prefix and preparation tests |
| Distribute an optional total-text target across verified replicas, independently of submission size, and respect explicit aggregate HTTP token budgets | [Embedding contracts](EMBEDDING-CONTRACTS.md); embedding contract and scheduler tests |
| Expose immutable prepared inputs, measure oversize inputs and reject them before inference | [Embedding contracts](EMBEDDING-CONTRACTS.md); prepared-input contract tests |
| Reuse completed exact IDs only through an explicitly enabled session-owned cache without bypassing seals, current loaded identity or inference limits | [Embedding contracts](EMBEDDING-CONTRACTS.md); preparation cache, metadata and optional reuse-probe tests |
| Retire and drain an owner when its blocking result wait is abandoned; keep ready-queue summaries incremental without frozen dispatch candidates | [Queue engine contract](PROGRESSIVE-EMBEDDING.md); blocking interruption and packing-accounting tests |
| Preserve vector order, cardinality, dimensions and finite values | [Functional parity](FUNCTIONAL-PARITY.md); embedding and numeric integrity tests |
| Discover equivalent loaded replicas only from supported content and semantic identity attestations | [Replica discovery](AUTOMATIC-EMBEDDING-REPLICAS.md); discovery and replica configuration tests |
| Coordinate admission, queues and tokenization across callers; use only explicit optional capacity policies | [Concurrency](CONCURRENCY-AND-DIAGNOSTICS.md); concurrency and scheduler tests |
| Keep persistent replica queues and compose batches at dispatch, with shared sync/async execution and no implicit owner/text/model/byte quotas | [Queue engine contract](PROGRESSIVE-EMBEDDING.md); persistent-engine, configuration and async lifecycle tests |
| Fill compatible HTTP batches across source windows without a required timer; preserve per-owner cancellation and shared transport drain | [Embedding contracts](EMBEDDING-CONTRACTS.md); shared-batch, async ownership and queue probe tests |
| Yield source-ordered flat progressive records without caller windowing; release transferred results, completed tasks and abandoned iteration | [Async interfaces](ASYNC-EMBEDDING-WINDOWS.md); record, lifecycle and 10,000-text ownership tests |
| Treat legacy scalar limits as inactive unless capacity policy is explicitly enabled; retain independent handoff/cache ownership | [Concurrency](CONCURRENCY-AND-DIAGNOSTICS.md); optional-capacity and ownership regressions |
| Preserve typed failures, cancellation, cleanup and resource ownership without automatic retries | [Integration](INTEGRATION.md); transport, cancellation and cleanup tests |
| Allow explicit idle-free generation while preserving finite non-SSE budgets and exact-call Stop/cleanup | [Stream timeouts](STREAM-TIMEOUTS.md); nullable config, optional idle transport and adapter tests |
| Capture bounded thinking diagnostics only when enabled | [Concurrency](CONCURRENCY-AND-DIAGNOSTICS.md); thinking diagnostic tests |

Use the commands in the [README](../README.md#verification) to run local
verification. Tests use synthetic inputs and local HTTP fixtures. Live probes
are optional, require explicit opt-in and must record the server build, model,
template and configuration actually exercised.

Generation template behavior, schema conformance, numerical equivalence across
replicas, server-side cancellation and a consuming application's full tool
workflow require qualification in that environment.
[Release verification](verification/PUBLIC-RELEASE-0.15.0.md) records actual
0.15.0 checks and artifact identities separately from live model qualification
and consumer acceptance. The earlier accepted numerical profile limitation is
not rewritten as a passed numerical gate.
