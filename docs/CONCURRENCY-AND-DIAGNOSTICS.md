# Concurrency, replicas and invocation diagnostics

This document describes the library's concurrency and diagnostic contracts.
Tests of these contracts do not qualify a consumer integration or a particular
server, model or device configuration.

## Embedding and external replicas

The development candidate uses one persistent model engine and one embedding
HTTP scheduler for synchronous calls and async windows. `N` is the number of
independently addressable, loaded canonical replicas in the verified local
topology. Both weight-content and embedding-semantic identities must agree;
synonyms of one catalog entry do not add capacity. Explicit multi-alias lists
also require that verification. A single unattested legacy alias remains an
isolated capacity-one fallback; it cannot authorize pooling.

`embedding_request_parallelism=None`, including an absent setting, selects
automatic capacity `C=N`. A positive integer `G` is an upper guard:
`C=min(N,G)`. Persisted positive settings retain their meaning as guards; they
are not silently promoted to automatic. Each replica admits **one original
embedding POST**. Positive `EmbeddingReplicaConfig.max_in_flight` values remain
parsable, but their effective value is one. This is an intentional behavioral
change, not a claim that old concurrency behavior is unchanged.

`total_slots`, `n_batch` and `n_ubatch` remain native execution observations.
They do not multiply HTTP permits. An array POST can already supply several
native sequences. Replica aliases do not prove GPU/process placement, and
HTTP concurrency does not guarantee simultaneous GPU execution or a speedup.
The server operator owns loading, slots, builds and hardware.

`batch_size=B` is a total-text budget. Quotas and HTTP text targets use `N`,
independently of `G`: the target is `ceil(B/min(B,N))`. A window with fewer
texts uses at most `min(texts,N)` nonempty destinations. Quotient/remainder
allocation rotates the uneven shares. For 256 texts, two replicas receive up
to 128 each and four receive up to 64 each. With `N=2,G=1`, 2,048 texts still
have sixteen full 128-text POSTs, admitted one at a time. Byte and explicit
HTTP token limits can require additional POSTs. `N=0` is unavailable.

Queues accumulate compatible records across source windows and callers.
`embedding_batch_wait_sec=0.0` fills from preparation/capacity events, without
a required timer; sparse work or resource pressure may flush partial batches.
Positive finite values opt into bounded coalescing in this same engine.
Original occurrence indices preserve ordered scatter, cardinality and owner
boundaries. Replica observation and dispatch take fair turns even when `G<N`;
consumer and source-window fairness remain separate. A dispatch turn advances
only when work is actually admitted.

A root POST and its bounded isolation descendants retain one original alias
lease through real transport drain. Descendants run serially on that alias;
other replicas may advance. Cumulative POST count therefore differs from the
number of simultaneous original leases. No ambiguous native POST is retried.

Topology and HTTP executors change only after the previous local generation's
work and references drain. Existing row affinity/results are not silently
redistributed or replayed. HTTP capacity follows the current verified topology;
tokenizer and control pools remain separately finite. A local generation is
not a remote loading-generation attestation. Discovery does not resize the
worker pool on which it is itself running.

## Admission and resource ownership

`embedding_max_pending_owners=H` is a positive optional global owner ceiling.
When absent, it resolves once as `max(1,(G if configured else 2)+queue_size)`.
The seed two is an admission default, not a replica count. The same H survives
topology changes. A synchronous call acquires it before snapshot/registration;
an async window acquires it before pulling its source. A stream's positive
`max_pending_windows` is a sublimit consuming H, not another global allowance.
A completed but unconsumed async window still holds its owner lease.
Preparation lookahead is separately bounded by `min(H,max(1,C+queue_size))`.

| Resource family | Ownership limit and release |
| --- | --- |
| Input snapshots | A separate `embedding_max_pending_bytes` pool covers immutable input references/text envelopes before copying. A shared snapshot builder prevents partial-copy credit deadlocks. Release follows caller transfer or real cleanup. |
| Raw/counted/ready records | Existing `embedding_max_pending_bytes` working pool; record credits remain through queued and active HTTP until drain. |
| Prepared tokens / direct reservations | `embedding_max_pending_tokens`, default 1,048,576; direct-path reservations are not measured token consumption. |
| Complete results | A separate `embedding_max_pending_bytes` pool covers **sync and async** outputs, including prepare results, before accumulation/transfer. Sync release is acknowledged by the coordinator before returning. |
| Exact preparation cache | Separate LRU, at most configured entries and `min(embedding_preparation_cache_bytes, embedding_max_pending_bytes//4)` bytes. |
| Embedding HTTP handoff | Separate global envelope `L=max(65536,embedding_max_pending_bytes)`, never multiplied by N; retained until result-credit transfer or discard after drain. |
| Control/tokenizer responses | Separate finite workers and body bounds, independent of embedding handoff credits so observations needed for progress cannot be starved by held responses. |

Defaults remain 64 MiB pending bytes, 8 MiB request bytes, 8 MiB response bytes,
1,024 preparation-cache entries and 8 MiB configured cache bytes. These are
separate ownership ledgers, not a combined process RSS ceiling. Caller/source
storage, arbitrary custom `__str__` behavior, Python/transport overhead and
uncooperative source/provider code are not covered by an exact RSS promise.
An indivisible impossible input fails with a typed resource/limit/requested
error; aggregate pressure waits under the original cancellation/deadline.
Snapshots freeze once and are not reread from a mutable source for dispatch.

Bulk exact preparation has unknown final token/ID sizes. One accumulator
reserves the global result pool, checks each prepared output before retaining
it, then shrinks the reservation at transfer. Tokenizer workers remain parallel
inside that owner; other bulk-preparation accumulators wait. This prevents
partial result owners from each retaining credits needed by the other to finish.
An oversized input can still be measured only within these finite resources.

### Response handoff and progress

Three quarters of L fund normal admission; one quarter is reserved for the
oldest eligible owner's progress. Normal capacity is tried first, including
parallel POSTs belonging to the same owner on distinct aliases. Every candidate
must fit the progress partition; the scheduler trims oversized candidates,
and a single impossible row fails explicitly. Later ready membership of the
oldest owner invalidates a previously resource-blocked dispatch observation.

Known-width handoff reserves actual request-envelope bytes, a finite body,
and a conservative materialization envelope. Cold width uses a per-candidate
body limit `min(default_body,max(1,(L//4-request_wire)//65))`: one body plus a
64-times-body materialization allowance must fit progress before the POST.
The known-width default body cap is `min(embedding_max_response_bytes,L//16)`.
A body bound is checked before JSON decode, also for compressed responses;
it is not a vector-size guarantee or a promise about total RSS. A transport
Future completing does not release handoff bytes while an owner still lacks
result credits. Cleanup does not acquire new credits merely to destroy data.

## Metadata, worker work and diagnostics

All embedding-adapter health/catalog/props, tokenizer, embedding and recovery
JSON paths receive a finite pre-decode body limit. Wire and inflated bodies
are bounded before materialization; error bodies are bounded too. Injected
clients must support the bounded-call contract; a `TypeError` does not cause
an unbounded compatibility retry.

Fresh full-pool identity, loaded-limit and build checks remain after waits and
at POST/recovery boundaries. Same-boundary catalog sharing already exists.
An eligible singleton statusless observation can also reuse its attested root
props as selected props, saving one GET at that boundary. Router observations
retain their necessary checks. Frozen membership and endpoint/alias/authority
provenance prevent late owners from inheriting an earlier snapshot. Revoked
local generations cannot be revived by late responses. There is no TTL cache
of inference authorization or weight digest used as proof of loaded context.
A partial preflight can renew for useful arrivals without renewing indefinitely
and starving dispatch.

Ready summaries, iterator packing and incremental plan counts avoid repeated
whole-backlog work. The coordinator uses remembered notifications, deadlines
and necessary cancellation/custom-clock checks rather than a fixed global poll.
`packing_candidate_visits` counts local record visits, not token or GPU work.

`health_status()` exposes bounded scheduler/engine/intake/cache snapshots:
requested/effective capacity and reason, replica/native observations, current
and peak working/result/handoff/snapshot credits, owner counts, active original
leases and resource wait reasons. Resource counters exclude text/token arrays,
credentials and private source paths. The engine's HTTP bound describes its
embedding, tokenizer and control workers; it is not a count of unrelated
consumer-owned HTTP work. Reports must count metadata/tokenization/POSTs per
phase and distinguish client wall-time unions, CPU and replica-seconds. None
is automatically pure GPU time or a stable throughput guarantee.

An optional [server-observation SDK](SERVER-OBSERVATION.md) defines complete
snapshots and server-enforced conditional admission. It does not make an
unpatched server expose that capability, add a new scheduler, or replace the
native fresh-control path by default. Its real-server deployment qualification
remains a separate obligation.

Independent owners retain deadlines, cancellation, indices and typed outcomes
inside a shared POST. Stop/expiry withdraws only that owner's pending work and
response portion. The shared transport deadline is fixed at dispatch from its
initial participants; survivors do not acquire a reset or extended deadline.
The stopped owner waits for real shared drain even beyond its own work budget.
All-member Stop or adapter close cancels and drains the transport. Synchronous
interruption retains the engine Future, stops only that owner and preserves
the primary exception through repeated cleanup interruptions.

## Transport ownership and closure

Concurrent client requests use owned sessions with exclusive leases and
separate active HTTP connections. The 0.7.0 candidate defaults owned requests to
`http_connection_policy="close"`; explicit `"reuse"` permits keep-alive after
the response has been consumed and closed. An injected, borrowed session stays
under its caller's ownership and its dispatch is protected. Active JSON and
SSE requests are tracked. Embedding cancellation retains its HTTP admission permit
until local HTTP cleanup finishes so that active work is not undercounted.
Adapter `close()` drains its scheduler, preparation and owned resources.
See [native HTTP transport](NATIVE-HTTP.md) for connection policy, upstream
compatibility and the native transport verification boundary.

Embedding and preparation calls apply a finite overall deadline to queueing,
discovery and HTTP work. Closure can wait for transport cleanup; it does not
promise immediate termination of remote computation. The transport introduces
no automatic retries. An injected client remains owned by its caller; the
library closes only resources it creates.

Generation idle can be disabled explicitly with `stream_idle_timeout_sec=None`.
Connection/JSON budgets remain finite. Owned SSE cancellation interrupts only
the active leased connection, then drains and detaches its hook before pool
reuse; an old stream cannot close a later call. A borrowed Requests Session
rejects unbounded SSE before dispatch. An injected raw client still belongs to
its caller. See [stream timeouts](STREAM-TIMEOUTS.md) for the public bridge
pattern, nullable evidence and DNS/connect limitations.

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
