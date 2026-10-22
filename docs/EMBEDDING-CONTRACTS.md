# Embedding contracts and public preparation

This guide describes embedding semantics in the development 0.10.3 candidate. The
[progressive queue and native token contract](PROGRESSIVE-EMBEDDING.md) extends
the scheduler shared by synchronous calls and async window sources. Local
qualification, publication and consumer acceptance remain distinct; see
[verification requirements](REQUIREMENTS.md) for the public checks.

## Total text budget, distribution and integrity

`batch_size` limits the total texts in a distribution window. Only the shared
adapter scheduler chooses distinct aliases in rotation. The number selected is
the minimum of texts in the window, usable aliases and the global active-work
limit. Balanced contiguous quotas use quotient/remainder allocation: 256 texts
over two aliases gives 128 + 128; over four gives 64 + 64 + 64 + 64. These values
are examples, not hardcoded limits. `batch_size=8` means eight texts total;
a consumer wanting windows of 256 must configure 256.

Each quota remains affine to its selected alias and can require several HTTP
requests. Text count, individual input context and aggregate HTTP token budget
are distinct limits. The effective HTTP token budget is the minimum of the
explicitly configured adapter and per-replica `batch_token_limit` values. No
request exceeds that budget, including a single indivisible input. If neither
value is configured, the aggregate client HTTP token limit is unknown.

Native `n_batch` and `n_ubatch` describe execution batch and microbatch sizes.
They are retained as observations, not converted into aggregate HTTP limits.
Loaded context and configured input ceilings bound individual inputs. Whether
a backend/model can split one input internally is not inferred from these
sizes; an unpublished capability remains unknown. A server can still reject
an input that fits the limits the client can verify, and its typed error is
preserved. No text is truncated or split by the library.

The path preserves order/cardinality, cancellation, finite overall deadlines,
typed errors, content and semantic identity, frozen active topology, draining
and all-or-error result publication. Successful quotas are not replayed. The
engine owns persistent queues for the verified replicas and selects ready
records when global and local execution capacity are available. Inputs that
become ready while a replica is busy can join its next batch. Ready inputs
retain their original indices and can combine across owners. Public windows
remain ordered result boundaries, without
their own blocking coordinator threads.

`embedding_batch_wait_sec` defaults to zero: filling follows preparation,
capacity and resource events without requiring a timer. The first ready input
does not force a partial POST while compatible admitted work can still fill it.
Compatible records from different calls and source windows can share a batch.
Explicit positive values retain bounded coalescing in the same engine: the
timer starts at the first unsent ready input; a filled cap or exhausted eligible
preparation can flush earlier.
Later arrivals do not restart it. The per-replica request text target derives
from the adapter's total-text budget and usable replicas, independent of source
window size. Accumulated windows neither shrink nor multiply that target.
A quota can have multiple batches active when verified capacity permits. The shared scheduler
rotates among eligible replicas and independent callers; a busy alias does
not prevent another alias's ready work from advancing.

Effective per-replica concurrency is bounded by positive observed
`total_slots`, the adapter's global ceiling and an explicit per-replica ceiling
when configured. Unknown or malformed slot observations fall back to one;
additional capacity is not assumed. Slot counts describe loaded capacity,
not currently idle slots or simultaneous GPU execution.

## Exact public preparation

The adapter exposes `prepare_texts`, `prepare_query`, `embed_prepared` and the
immutable `PreparedEmbeddingInput` type. Exact token counts include the task
prefix and embedding special tokens; `raw` adds no prefix. Oversized inputs
can be measured but cannot be admitted for inference. Character/token ratios
are not authoritative. Lossless splitting, UTF-8 fragment identity, AST and
graph processing belong to the consumer.

A prepared record exposes prepared text, task, count, context and HTTP limits,
the effective per-input limit, verified identity and digests. Its limits are
conservative snapshots; admission observes identity and limits again. A private
adapter seal prevents accidental use of altered records or records from
another adapter. Retained immutable token IDs are covered by this seal too.
Count/ID reuse requires verified identity, tokenizer build and tokenizer policy;
an alias/topology change alone does not change the count. An unverified fallback
or unknown tokenizer build requires recounting. There is no unbounded record
registry or cache.

Version 0.10.2 adds a bounded adapter-owned LRU cache for completed
exact token tuples. Prepared and raw inputs can reuse a tuple only for the
same verified weights and semantic identity, tokenizer build, task, text and
prefix/special policy. A changed context limit alone does not invalidate an
exact count, but fresh admission must still reject an input over that limit.
Unknown identity/build bypasses this cache. Public records retain same-adapter
seal validation; a cached count is not an inference permit.

The cache uses `embedding_preparation_cache_entries` (default 1,024) and
`embedding_preparation_cache_bytes` (default 8 MiB), both nonnegative integers.
Its effective byte budget is
`min(embedding_preparation_cache_bytes, embedding_max_pending_bytes // 4)` in a
separate retained pool. Zero in either configured limit disables retention.
No TTL or singleflight is promised. Concurrent cold misses may each tokenize;
eviction and close release entries. Errors and incomplete tokenizations do
not become hits. Cache diagnostics include occupancy and hit/miss counters.

Preparation retains the same distribution windows, quota affinity and exact
token policy. Persistent tokenizer workers share `token_count_parallelism`
across callers and publish ready records while inference is running. One
adapter engine owns scheduling for sync and async APIs. No record is frozen
into a waiting HTTP batch before execution capacity is available.

Resource ceilings complement the existing queue and async-window admission
bounds. `embedding_max_pending_bytes` defaults to 64 MiB,
`embedding_max_pending_tokens` to 1,048,576, and
`embedding_max_request_bytes` to 8 MiB; all require positive integers.
These budgets bound raw/counted-record queue credits, prepared tokens queued
or active for inference, and individual request payloads. Transient tokenizer
response handoffs are separately bounded by worker count and native JSON
response limits. Oversized inputs are measurable within those resource
ceilings but remain invalid for inference. The caller's original input and
complete synchronous result are outside any constant-memory claim. Async
completed outputs remain bounded with their unconsumed windows.

`embedding_max_response_bytes` is an independent positive integer, default
8 MiB, bounding each decoded native JSON response for tokenization and
embedding. It is enforced before JSON materialization, including decompressed
bodies. It does not inherit the request or pending-record ceiling. Overflow or
incomplete framing fails with a typed error and does not trigger replay or
partial result publication. Concurrent responses can each reach their own
bound; decoded JSON objects, worker handoffs and result reservations are
separate allocations rather than an overall RSS guarantee.

Tokenization and metadata have their own concurrency limits; total HTTP
capacity is derived from embedding, tokenizer and control capacity. The
embedding ceiling does not by itself count every tokenizer or metadata GET.
Native exact tokenization still requires calls for distinct new texts; this
engine does not add a bulk tokenizer endpoint.

Initial metadata observations can be reused coherently across admission work,
including the catalog already read for pool membership. Loaded identity,
tokenizer build and limits are still observed at actual dispatch boundaries.
This reuse does not add a time-based authorization window or remove topology
and current-limit checks.

The normal embedding path uses internal counted records without serializing
or signing public preparation envelopes. Only `prepare_*` emits sealed public
records; `embed_prepared` verifies them without signing them again. Exact
duplicates can share counts inside a verified identity/build/task observation;
unverified identity or an unknown tokenizer build does not authorize reuse.

Admission observes loaded identity and limits after waits and before each
actual batch, including adaptive children. Local validation shares each
boundary's observation rather than performing hidden duplicate catalog reads.
Recovery compares the complete immutable observation, including vocabulary and
identity. Each pending record keeps its own historical context/batch limits;
another owner's smaller limit cannot mask a fresh change. Aggregate HTTP token
packing remains separate. Retired or invalid owners are excluded from children
without shifting surviving result indices. Successful children are not replayed.
After a limit changes, remaining inputs are revalidated without invalidating
already completed inputs. A changed tokenizer build rejects stale counts.
These observations are not a server-side lease or atomic GET-and-POST
transaction. Bulk work and waits are cancellable under a finite overall
deadline; `close()` drains preparation and HTTP cleanup as well as inference.

`embedding_input_format="auto"` uses native nested token arrays when the client
explicitly declares support and retained IDs match the verified identity,
build and policy. This avoids string tokenization inside the embedding handler;
new inputs still require exact server tokenization first. `text` selects string
input; `tokens` requires safe IDs or fails before sending. Unknown custom clients
use text in auto mode. There is no fallback/replay after an ambiguous POST.
The standard native client also propagates structured context errors without
adaptive bisection: the server may already have evaluated sibling inputs.
Recognized-context recovery remains for custom clients outside that declared
native contract. Empty token arrays remain measurable; auto uses text for them
and forced token mode rejects them before inference.

## Public API

```python
prepare_texts(texts: Sequence[str], *, input_type="document",
              timeout=None, cancel=None) -> tuple[PreparedEmbeddingInput, ...]
prepare_query(text: str, *, timeout=None, cancel=None) -> PreparedEmbeddingInput
embed_prepared(inputs: Sequence[PreparedEmbeddingInput], *,
               timeout=None, cancel=None) -> list[list[float]]
```

Principal fields are `text`, `input_type`, `token_count`, `context_token_limit`,
`batch_token_limit`, `max_input_tokens`, `identity_verified`, `content_sha256`
and `embedding_sha256`. The `fits_context`, `fits_request` and `fits_input`
properties report fit against the available limits. Unknown limits remain
`None`. Preparation accepts strings; `embed_texts` retains its existing string
coercion. `prepare_query("")` measures prefix and special tokens, while
`embed_query("")` retains its historical `[]` result.

Native execution sizes are observed from supported metadata and structured
arguments on the verified loaded entry (`--batch-size`, `--batch-size=`, `-b`,
`--ubatch-size`, `--ubatch-size=`, `-ub`). Malformed, duplicated or conflicting
supported values are rejected. Diagnostic `observed_native_batch_tokens` and
`observed_native_microbatch_tokens` expose these values; the compatibility key
`loaded_batch_token_limit` also refers to native `n_batch`, not an HTTP budget.
Sizes are not inferred from paths, free-form presets or training context.
Loaded context and training context are distinct. The client performs no server
administration. See [native HTTP transport](NATIVE-HTTP.md) for connection
policy and its separate verification boundary.

Each owner retains its cancellation, finite work deadline and result boundary,
including when compatible records share an embedding POST. Stop or deadline
expiry removes that owner's pending inputs and discards its response portion;
it does not abort surviving owners. Shared transport uses one finite deadline
fixed at dispatch from the latest deadline among its initial members. An owner
waits for actual shared HTTP drain before completing, so cleanup can exceed its
own work budget. All-member cancellation and adapter close cancel and drain
the shared request. The overall work deadline starts at call entry and covers
queueing, discovery, preparation and HTTP. Immediate remote termination is not
promised; completed inputs are not replayed and owners keep typed outcomes.

When an external exception interrupts a synchronous call's wait for its engine
Future, the caller stops its owner and waits for real drain before propagating
that original exception. This also applies to preparation waits. Repeated
interruptions during cleanup do not release the lease early or close surviving
owners' transport. A completed typed worker failure propagates unchanged; the
adapter and borrowed client remain available to independent calls.

The additive [asynchronous window API](ASYNC-EMBEDDING-WINDOWS.md) accepts async
sources of text or prepared-input windows. It uses this same execution engine
and HTTP scheduler, preserves each result boundary and bounds source pulls and
unconsumed outputs. Existing synchronous methods retain their signatures.

## Verification criteria

- Dynamic distribution of 256 texts over two/four replicas, smaller global
  limits, uneven remainders, fewer texts than replicas and multiple windows
  without omissions or duplicates.
- Exact packing within known budgets; rejection of an indivisible input over
  the HTTP budget before embedding; no native replay after a failed POST.
- Event-driven filling across source windows, optional coalescing deadlines, sealed exact token
  inputs, special/prefix parity and safe pre-dispatch input-mode selection.
- Concurrent callers, bounded/zero queues, isolated cancellation, finite
  deadlines, partial failure, drained closure and topology transitions.
- Query/document/raw tasks, Unicode, whitespace, special markers and empty
  text; boundary and over-boundary inputs that remain measurable, with distinct
  configured and loaded limits.
- Reusable preparation without duplicate tokenization, rejection of altered or
  foreign records, recounting for unverified fallback and fresh admission limits.
- Typed errors for malformed tokenizer results, ambiguous batch metadata and
  changed identities.
- Deterministic tests with loopback HTTP, lint/format checks, distribution build
  and wheel installation outside the checkout; see
  [README verification](../README.md#verification).

A deployment-specific live comparison should declare vector dimensions and
tolerances for maximum absolute error, cosine similarity and normalization.
It should separately report preparation, tokenizer/embedding requests,
payloads, concurrency and elapsed time. Wall time is not pure GPU time, and
numerical equivalence is not a retrieval-quality evaluation. Build, model and
loading configuration must accompany such results; no historical measurements
are attributed to this public release.

## Consumer example

```python
from runtime_llama import build_embedding_adapter

adapter = build_embedding_adapter({
    "model": "example-embedding.gguf",
    "base_url": "http://localhost:8080",
    "embedding_replica_discovery": True,
    "batch_size": 256,
    "embedding_request_parallelism": 4,
    "token_count_parallelism": 2,
    "timeout": 120,
    "embedding_prefixes": {
        "document": "title: none | text: ",
        "query": "task: search result | query: ",
    },
})
try:
    prepared = adapter.prepare_texts(["first text", "second text"])
    # The consumer can split oversized texts losslessly, then prepare only
    # the new fragments before submitting them.
    if all(item.fits_input is True for item in prepared):
        vectors = adapter.embed_prepared(prepared)
    query = adapter.prepare_query("How does it work?")
    query_vector = adapter.embed_prepared([query])[0]
finally:
    adapter.close()
```

The configured model must already be available under the server's policy.
Automatic verified pooling requires the
[Noxi identity tags](AUTOMATIC-EMBEDDING-REPLICAS.md). Prefixes must match the
chosen model's task contract; the example's strings are not universal defaults.
