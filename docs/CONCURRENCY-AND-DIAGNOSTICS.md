# Concurrency, replicas and invocation diagnostics

This document describes the library's concurrency and diagnostic contracts.
Tests of these contracts do not qualify a consumer integration or a particular
server, model or device configuration.

## Embedding and external replicas

`embedding_request_parallelism` limits active embedding jobs across an adapter
and defaults to 1. `embedding_queue_size` bounds waiting jobs and defaults to
16; zero permits only active capacity. Admission applies backpressure to
callers. These budgets are shared by all callers of the adapter.

`batch_size` limits the total texts in one distribution window, with balanced
quotas assigned to distinct aliases. `batch_token_limit`, together with the
observed native batch limit, bounds aggregate tokens in each HTTP request.
Context limits apply to individual inputs. A quota can require several HTTP
requests. `token_count_parallelism` is a global adapter limit, including public
preparation. Results preserve input order and cardinality exactly.

`embedding_replicas` explicitly lists destinations on the same endpoint. Each
`EmbeddingReplicaConfig` provides `model`, `max_in_flight` and, when needed,
`max_input_tokens` and `batch_token_limit`. An empty list uses the profile's
model. Available replicas are selected round-robin under global and
per-replica limits. Alias suffixes do not identify devices; the server operator
controls processes, loading and device assignment. Automatic discovery has
its own attestation rules in [replica discovery](AUTOMATIC-EMBEDDING-REPLICAS.md).

A job selects a replica, observes its properties and tokenizes/packs inputs for
that destination. Adaptive splitting after a recognized capacity error stays
on the same replica and does not replay successful batches. Configured and
observed limits remain distinguishable; an absent property remains unknown.
Dimensions and normalization are validated before publishing the complete
result. Failure stops new submissions for the affected call and drains its
already started work without cancelling independent calls.

Server reachability and model availability are distinct observations. Router
aliases must be loaded. A direct server may omit router state, but must expose
model identity.

## Transport ownership and closure

Concurrent client requests use owned sessions with exclusive leases. Keep-alive
reuse occurs after consuming and closing the response. Dispatch through an
injected, borrowed session is protected. Active JSON and SSE requests are
tracked. Embedding cancellation retains its admission permit until HTTP cleanup
finishes so that active work is not undercounted. Adapter `close()` drains its
scheduler and owned resources.

Embedding and preparation calls apply a finite overall deadline to queueing,
discovery and HTTP work. Closure can wait for transport cleanup; it does not
promise immediate termination of remote computation. The transport introduces
no automatic retries. An injected client remains owned by its caller; the
library closes only resources it creates.

## Thinking diagnostics

`thinking_diagnostics` accepts `enabled` (default `False`) and `max_chars`
(default 65536). The character limit applies only to diagnostic capture.
Visible output, budgets, runaway protection and `reasoning_text` retain their
contracts: `reasoning_text` contains separately supplied reasoning, while
inline thinking is removed from visible text. Diagnostics can capture both
sources.

Each invocation has its own identifier and collector, correlated with role and
model. A result, final event or error can expose optional diagnostics containing
text, sources, counts, truncation and outcome. An explicit final sink allows
the caller to retain diagnostics on interruption, cancellation, runaway or
failure and receives at most one record per invocation.

Enabling capture and persisting it are explicit caller choices. Ordinary logs
record lengths without automatically dumping prompt contents. Known
credentials are excluded from diagnostics.

## Verification boundary and licensing

Deterministic tests use barriers and loopback HTTP servers to exercise overlap,
global/per-replica budgets, ordering, partial failures, cancellation and cleanup.
A deployment-specific live test should identify its build and aliases and
compare the same corpus on each replica and the pool with declared numerical
tolerances. Ingestion and mixed generation/embedding loads can measure the
effects of a configuration; the concurrency contract does not promise a speedup.
See [verification requirements](REQUIREMENTS.md).

The project is licensed under [Apache-2.0](../LICENSE), copyright 2026 Davide
Mele. The derived llamatelemetry client retains its upstream MIT license and
attribution; see [provenance](PROVENANCE.md) and [NOTICE](../NOTICE).
