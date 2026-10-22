# Concurrency, replicas and invocation diagnostics

This document describes the library's concurrency and diagnostic contracts.
Tests of these contracts do not qualify a consumer integration or a particular
server, model or device configuration.

## Embedding and external replicas

In development version 0.10.3, `embedding_request_parallelism` limits active
embedding HTTP work across an adapter. Its default is 4 with automatic replica
discovery and 1 otherwise. `embedding_queue_size` defaults to 16 and adds
bounded preparation lookahead to the request-parallelism budget; zero retains
only that base preparation capacity. These budgets are shared across callers
and complement the byte, token, result and async-intake limits below.

`batch_size` limits the total texts in one distribution window, with balanced
quotas assigned to distinct aliases. Explicit adapter and per-replica
`batch_token_limit` settings bound aggregate tokens in each HTTP request.
Native `n_batch`/`n_ubatch` are execution observations, not aggregate HTTP limits.
Context limits apply to individual inputs; an unpublished backend splitting
capability stays unknown. A quota can require several HTTP requests.
`token_count_parallelism` is a global adapter limit, including public
preparation. Results preserve input order and cardinality exactly.

`embedding_replicas` explicitly lists destinations on the same endpoint. Each
`EmbeddingReplicaConfig` provides `model`, `max_in_flight` and, when needed,
`max_input_tokens` and `batch_token_limit`. An empty list uses the profile's
model. Available replicas are selected round-robin under global and
per-replica limits. Alias suffixes do not identify devices; the server operator
controls processes, loading and device assignment. Automatic discovery has
its own attestation rules in [replica discovery](AUTOMATIC-EMBEDDING-REPLICAS.md).

Loaded `total_slots` caps per-replica HTTP concurrency together with global and
configured per-replica ceilings. Unknown slot capacity falls back to one;
observed slots are not currently idle slots or a GPU speed guarantee.
Reductions drain already admitted work, and increases can admit additional
ready batches. Topology replacement remains a separate drained boundary.

One persistent engine owns input admission and ready queues for each verified
replica. Its fixed worker pools overlap exact remote tokenization, metadata and
embedding HTTP. Batch contents are selected when execution capacity is
available, so records prepared during a previous POST can join the next batch.
The engine can skip a busy alias and rotates among independent ready callers.
The synchronous and async APIs submit to that same engine; there is no
blocking coordinator thread per async window or second embedding scheduler.

Readiness summaries update incrementally when ready entries enter or leave a
queue. Packing uses iterators for the needed prefix and constant-time owner
rotation, avoiding whole-backlog copies on every preparation event. It does not
retain a frozen candidate batch across capacity or metadata waits. This changes
local scheduling work, not native tokenization, inference or numerical limits.
Diagnostic `packing_candidate_visits` is a cumulative integer counting records
actually examined by `_pack`; it is not elapsed time or GPU work. An adapter
snapshot can include preliminary checks as well as the main pipeline.

A partial preflight is renewed when another useful owner arrives during its
metadata checks. New rows require that fresh membership observation; they do
not inherit the earlier snapshot. Admission still in progress can help fill a
batch, while topology-busy or result-credit-blocked owners cannot prevent old
work from draining. Full batches, resource pressure and an explicitly elapsed
coalescing wait retain progress.

A preparation plan holds its lookahead lease only while it has input groups
left to prepare or preparation workers still running. Once those groups are
exhausted and its workers have finished, that phase credit can admit more
preparation even if the earlier plan retains ready records or active HTTP.
Its records, token/byte credits, results and shared HTTP ownership remain held
until their own publication or drain boundaries. Releasing preparation credit
does not release an HTTP permit or discard retained work. Diagnostic
`active_plans` counts preparation leases; `retained_plans` also includes plans
whose preparation phase ended but whose records or HTTP still remain.

`embedding_batch_wait_sec=0.0` remains the default and fills batches from
preparation and capacity events without a required timer. One ready input does
not force dispatch while compatible admitted work can still fill the batch.
Positive finite values retain optional bounded coalescing in the same engine.
Compatible records from different windows or callers can share an HTTP request.
The target derives from the adapter's text budget and usable replicas;
small source windows neither shrink nor multiply it. Sparse input and resource
pressure can still cause partial batches.

Finite queue resource budgets complement work/window counts:
`embedding_max_pending_bytes=67108864`,
`embedding_max_pending_tokens=1048576`, and
`embedding_max_request_bytes=8388608`. They bound raw/counted-record queue
credits, prepared tokens queued or active for inference, and individual
request payloads. A separate completed-async-output credit pool has the same
`embedding_max_pending_bytes` ceiling; it is not shared with the raw-record
pool. Result reservations follow submission order, including windows still
awaiting metadata. Cold-start response handoffs are bounded by active HTTP
workers until the first result establishes dimensions. Tokenizer and embedding
response handoffs also have native JSON response limits. Original caller inputs
and a synchronous call's complete output still occupy proportional memory;
these separate pools and transient handoffs do not claim a total process RSS
ceiling. Async completed windows retain their admission until consumed.

Completed exact IDs use an additional adapter-owned LRU pool. Its configurable
entry/byte limits default to 1,024 and 8 MiB; effective retained bytes are
`min(embedding_preparation_cache_bytes, embedding_max_pending_bytes // 4)`.
Zero entries or bytes disables retention. Entries require
verified identity, tokenizer build and task/prefix/special policy; raw text
must match exactly. Unknown identity/build bypasses the cache. It has no TTL
or singleflight, and concurrent cold misses can perform separate tokenizations.
Eviction and adapter close release entries. `health_status` exposes cache
occupancy, hits, misses and eviction counters; these bytes are separate from
raw/ready and result reservations and are not a process RSS limit.
`embedding_max_response_bytes` separately bounds each decoded native JSON body
before materialization (default 8 MiB), independently of request and queue
budgets. In-flight response bodies and materialized objects remain transient
resources bounded by worker concurrency and per-response limits.
Ready-record byte accounting also reserves a conservative extra 128 bytes per
record for the finite-limit summary index. Dispatch removes ready references,
but the record's byte/token credits stay held until actual HTTP drain. Fixed
container overhead and operating-system RSS are not an exact accounting claim.

Tokenizer and metadata HTTP use their own bounded capacity. The total HTTP
ceiling derives from embedding, tokenizer and control limits; the embedding
limit alone is not a count of all HTTP requests. Metadata waits and HTTP run
outside scheduler locks and do not block the coordinator thread.
Coherent admission observations avoid duplicate catalog reads without removing
fresh loaded identity, build and limit checks at actual dispatch boundaries.

The coordinator records wake generations under its condition lock, including
notifications received before entering a wait. It waits until a worker/resource
event, owner deadline, optional coalescing deadline or the next required
cancellation-probe check. Non-notifiable cancellation probes and custom clocks
retain finite checks; there is no fixed global 10 ms poll. Diagnostic snapshots
are republished only when their observable values change. Plan counts and
preparing counts are maintained by that same coordinator, preserving the
existing owner/model ordering and distinct preparation/result/HTTP lifetimes.

Independent owners retain their cancellation, deadline, output indices and typed
outcomes in a shared batch. Cancelling or expiring one owner removes its pending
work and discards its portion of a response without aborting survivors. Shared
HTTP has a finite deadline fixed from the latest initial member deadline.
That owner's Future waits for actual shared transport drain, even beyond its
own work deadline; the active lease remains held until cleanup. All-member
cancellation or adapter close stops and drains the transport. Cancelling an
awaiting Future alone does not release its lease.

A synchronous embedding/preparation call retains its engine Future while
waiting. If an external `BaseException` abandons that wait, it stops its owner,
wakes the coordinator and drains the actual Future before reraising the original
exception. Repeated interruptions during this cleanup do not replace that
exception or release resources early. Completed typed worker errors propagate
unchanged. Surviving shared owners and the shared adapter/client remain open;
the abandoned owner can wait past its work deadline for shared HTTP cleanup.

Identity, tokenizer build and limits are observed again after waiting and at
actual batch boundaries. Stable limits require linear admission passes; a
changed limit revalidates pending inputs. Adaptive splitting after a recognized
capacity error stays on the same replica, revalidates pending children and
does not replay completed inputs. Configured and observed limits remain
distinguishable; an absent property remains unknown.
The owned native client never replays a failed or ambiguous embedding POST.
Recognized capacity recovery remains available for injected clients with the
existing explicit recovery contract; its children retain the same HTTP lease.
Dimensions and normalization are validated before publishing the complete
result. Owner-local validation, deadline and cancellation failures stop that
owner's submissions and drain its started work without cancelling survivors.
A transport or backend failure of a shared POST fails all participating
owners; unrelated calls outside that POST remain independent. Failed or
ambiguous shared requests are not replayed.

Server reachability and model availability are distinct observations. Router
aliases must be loaded. A direct server may omit router state, but must expose
model identity.

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
