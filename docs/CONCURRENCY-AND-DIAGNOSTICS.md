# Concurrency, replicas and invocation diagnostics

This document describes the library's concurrency and diagnostic contracts.
Tests of these contracts do not qualify a consumer integration or a particular
server, model or device configuration.

## Embedding and external replicas

In the local 0.10.0 candidate, `embedding_request_parallelism` limits active
embedding HTTP work across an adapter. Its default is 4 with automatic replica
discovery and 1 otherwise. `embedding_queue_size` supplies the waiting-work
budget and defaults to 16; zero retains only active admission capacity.
Admission applies backpressure to callers. These budgets are shared by all
callers of the adapter.

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

`embedding_batch_wait_sec=0.0` is the new default and dispatches available work
without waiting to fill a batch. Positive finite values retain optional
coalescing in the same engine. More ready data can accumulate while a replica
is busy; sparse input can still produce small batches. Accumulating multiple
windows does not multiply the text cap derived from a distribution quota.

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

Tokenizer and metadata HTTP use their own bounded capacity. The total HTTP
ceiling derives from embedding, tokenizer and control limits; the embedding
limit alone is not a count of all HTTP requests. Metadata waits and HTTP run
outside scheduler locks and do not block the coordinator thread.

Independent cancellation and deadline domains stay separate in HTTP batches.
Cancelling one owner cannot close a request containing another owner's work.
The engine retains an active lease until actual transport cleanup, not merely
until an awaiting Future is cancelled.

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
result. Failure stops new submissions for the affected call and drains its
already started work without cancelling independent calls.

Server reachability and model availability are distinct observations. Router
aliases must be loaded. A direct server may omit router state, but must expose
model identity.

## Transport ownership and closure

Concurrent client requests use owned sessions with exclusive leases and
separate active HTTP connections. The 0.7.0 candidate defaults owned requests to
`http_connection_policy="close"`; explicit `"reuse"` permits keep-alive after
the response has been consumed and closed. An injected, borrowed session stays
under its caller's ownership and its dispatch is protected. Active JSON and
SSE requests are tracked. Embedding cancellation retains its admission permit
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
