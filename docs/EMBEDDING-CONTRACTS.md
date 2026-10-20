# Embedding contracts and public preparation

This guide describes embedding semantics in the local 0.7.0 candidate and their
verification boundary. It does not declare a published 0.7.0 release, a new live
qualification or consumer acceptance. See
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
scheduler admits ready HTTP batch jobs, rather than holding an inference slot
through tokenization of an entire quota. A quota can have multiple ready
batches in flight when its verified slot capacity permits. The same scheduler
can pass a job blocked on another alias while preserving FIFO within an alias
for admitted jobs. Its nonblocking admission lets the coordinator try other
ready aliases, including with `embedding_queue_size=0`, without introducing a
second embedding scheduler.

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
another adapter. Count reuse requires verified identity and tokenizer policy;
an alias/topology change alone does not change the count. An unverified fallback
or unknown tokenizer build requires recounting. There is no unbounded record
registry or cache.

Preparation retains the same windows, quota affinity and exact token policy.
The adapter reuses separate preparation and tokenizer executors;
`token_count_parallelism` and pending tokenizer admission are global across its
callers. Lookahead is bounded globally to
`embedding_request_parallelism + embedding_queue_size` quota leases, held
through preparation, ready work and the quota's last HTTP cleanup. Preparation
of subsequent quotas can overlap active inference. This limits intermediate
lookahead, not the memory occupied by the caller's full input and result.

The normal embedding path uses internal counted records without serializing
or signing public preparation envelopes. Only `prepare_*` emits sealed public
records; `embed_prepared` verifies them without signing them again. Exact
duplicates can share counts inside a verified identity/build/task observation;
unverified identity or an unknown tokenizer build does not authorize reuse.

Admission observes loaded identity and limits after waits and before each
actual batch, including adaptive children. Local validation shares each
boundary's observation rather than performing hidden duplicate catalog reads.
After a limit changes, remaining inputs are revalidated without invalidating
already completed inputs. A changed tokenizer build rejects stale counts.
These observations are not a server-side lease or atomic GET-and-POST
transaction. Bulk work and waits are cancellable under a finite overall
deadline; `close()` drains preparation and HTTP cleanup as well as inference.

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

The overall deadline starts at call entry and covers queueing, discovery,
preparation and HTTP. Cleanup can require transport timeouts to complete;
immediate termination of remote computation is not promised. Failure preserves
the first typed error without contaminating independent calls.

## Verification criteria

- Dynamic distribution of 256 texts over two/four replicas, smaller global
  limits, uneven remainders, fewer texts than replicas and multiple windows
  without omissions or duplicates.
- Exact packing within known budgets; rejection of an indivisible input over
  the HTTP budget before embedding; adaptive splitting without replaying
  successful batches.
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
