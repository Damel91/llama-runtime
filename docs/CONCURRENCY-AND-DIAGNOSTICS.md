# Concurrency, replicas and resource ownership

Version 0.15.0 separates the amount of work a consumer submits from the work
currently executing. There is no implicit application quota on text count,
loaded-model count, pending owners or total result size. Real allocations still
consume memory; this contract does not promise constant memory or a process RSS
ceiling while a consumer accumulates work or retains results.

The library owns correct resource acquisition, transfer and release. The server
operator owns model loading, llama.cpp builds, processes and hardware. Tests of
these contracts do not accept a consumer integration or a model profile.

## Embedding and external replicas

One persistent model engine and one embedding HTTP scheduler serve synchronous
calls, asynchronous windows and flat progressive text streams. Verified weight
content and embedding-semantic identities define a homogeneous replica pool;
aliases for one canonical entry do not add capacity. Different model identities
remain separate and are never mixed to increase throughput.

Missing/`None` `embedding_request_parallelism` selects N usable replicas. A
positive explicit guard G selects `min(N,G)`. Each canonical replica owns one
original embedding POST lease through actual transport drain, including serial
isolation descendants. Native slots, batch and microbatch describe server
execution; they do not multiply HTTP permits. There is no fixed catalog or
replica-count ceiling. Worker defaults follow the actual pool rather than the
former preparation/control constants of one, two or four.

`batch_size=B` is an optional, consumer-selected total text distribution target.
It is not a maximum input sequence or output size. With B=256, two verified
replicas receive quotas up to 128 texts and four up to 64; these are consequences
of B/N, not native maximum POST sizes. A global guard changes overlap, not quota
width. Missing B imposes no fixed text target. Compatible records may accumulate
across source windows and callers without a mandatory batching timer.

Occurrence indices preserve order and duplicates. Topology replacement retains
old work and its references until real drain; no ambiguous native POST is
replayed. HTTP overlap is not proof of simultaneous GPU execution or a speedup.

## Admission and resource ownership

Capacity fields default to `None`. `embedding_capacity_limits=False` is the
standard policy: legacy scalar byte/token/owner/queue/isolation values are not
silently treated as consent to enable the old limits. A consumer choosing those
policies sets `embedding_capacity_limits=True` and supplies each desired value.
Omitted fields remain unset; enabling the flag does not manufacture defaults.
Explicit batch targets, global HTTP guards and per-stream
`max_pending_windows` remain independent choices.

| Resource | Ownership and release |
| --- | --- |
| Input | An acquired text or admitted window is owned until its work transfers or drains. No compulsory 64 MiB snapshot ceiling is installed. |
| Queued and active records | Records remain associated with their owner and occurrence. Completed/cancelled records are removed when no operation needs them. |
| Complete results | Existing list/window APIs retain the output they promise, without requiring it to fit an artificial whole-window pool. Delivery transfers it to the consumer and removes library-only references. |
| Progressive results | `aiter_texts()` transfers indexed vectors individually through the same engine. The consumer need not construct HTTP-sized input windows. |
| HTTP responses | Body, decoded values and transport ownership are retained only until transfer or destruction after real drain. |
| Preparation cache | Disabled unless `embedding_preparation_cache_enabled=True`. An enabled session cache has explicit ownership and clears at close; optional entry/byte policies require the capacity opt-in. |
| Tasks, Futures and generations | Finished registrations, callbacks and retired topology references are released. A cancelled waiter does not masquerade as transport drain. |

The capacity opt-in can select pending-byte, pending-token, request-byte,
response-byte, handoff-byte, owner, queue and isolation policies. Request,
response and handoff settings are independent: none is divided or multiplied
implicitly to derive another setting. Cache entry/byte options are available on
the adapter profile. Zero in a cache option disables its retention.

No new budget is derived automatically from available RAM. No implicit spill to
disk, truncation or change to consumer-owned source splitting is introduced.
A retained result, legitimate pending work or an intentionally enabled cache is
not automatically a leak. References to finished operations must still go away.

## Response handoff and progress

The former compulsory 75/25 handoff partition, response `pool/16` cap,
materialization multipliers used as admission, and memory-driven batch
halving are removed. An optional `embedding_max_handoff_bytes` is independent
of pending bytes and applies only when the consumer enables capacity limits.

Without an explicit policy, a POST is not reduced to satisfy those former
local formulas. With a chosen policy, waits and typed resource failures describe
that specific setting. Real server constraints remain separate. Cleanup never
needs fresh admission credits to destroy or release owned data.

## Metadata, worker work and diagnostics

Loaded identity and inference limits are checked at the relevant boundaries.
Same-boundary catalog/root-properties reuse avoids duplicate reads without
turning an old observation into a new inference permit. Unknown capabilities
remain unknown. Direct text inference does not require tokenizer-only metadata
unless the requested path actually uses exact preparation, caching or
calibration.

Ordinary direct inputs do not reserve the whole model context as fake token
consumption. The character recommendation uses effective context; native
microbatch is not substituted for that context merely because it is smaller.
The category 80% and mixed 90% calibration policy is unchanged.

`AutomaticEmbeddingService.diagnostics()` is local and performs no HTTP.
Diagnostics distinguish requested/effective HTTP capacity, owned resources,
queues, metadata/tokenization/embedding calls and their phases. These counters
observe work; they do not enable another capacity policy. Reported wall time,
CPU, wire bytes and Python allocations are different measurements, not pure GPU
time. The optional [server-observation SDK](SERVER-OBSERVATION.md) is not a
required server extension.

## Transport ownership and closure

Owned concurrent requests use independent leased sessions/connections. The
connection policy remains `close` by default, with explicit `reuse` available.
Borrowed sessions and injected clients retain their declared ownership.
Finite work/connection/JSON deadlines and cancellation remain supported.

Stopping an owner withdraws only its work in a shared POST; surviving owners
continue. The stopped operation holds ownership until the real shared drain,
which can outlast its work deadline. All-member Stop or adapter close cancels
and drains the transport. Synchronous interruption preserves the primary
exception while draining, including repeated interruptions.

`close()` releases owned workers, sessions, caches and records. Async consumers
use `await aclose()` to include source cleanup. Prefer `async with` or explicit
`aclose()` for deterministic early exit. A dropped progressive facade signals
Stop and schedules cleanup on its live owning loop; it cannot run asynchronous
cleanup on an already closed loop or forcibly kill uncooperative user code.
See [async interfaces](ASYNC-EMBEDDING-WINDOWS.md) and
[stream timeouts](STREAM-TIMEOUTS.md).

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

Resource tests use repeated success/error/cancellation/close cycles, observable
ownership and weak references/finalizers to distinguish finished-operation
retention from live work. Large-input tests include 10,000 texts as a test size,
not an API maximum. `tracemalloc` and RSS are observations: allocator high-water
marks are not interchangeable with library-owned live objects.

Deterministic verification, installed-wheel checks, live qualification and
consumer acceptance remain separate. The known numerical limitation recorded
for the earlier model profile is not relabeled as a numerical pass. See the
[0.15.0 release verification](verification/PUBLIC-RELEASE-0.15.0.md).

Apache-2.0, copyright Davide Mele, and upstream MIT attribution are preserved;
see [provenance](PROVENANCE.md) and [NOTICE](../NOTICE).
