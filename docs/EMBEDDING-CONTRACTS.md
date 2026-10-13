# Embedding contracts and public preparation

This guide describes implemented embedding semantics and their verification
boundary. Version 0.4.1 preserves the inference and embedding behavior of
0.4.0. It does not introduce a new live qualification or establish consumer
acceptance. See [verification requirements](REQUIREMENTS.md) for the public
checks.

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
are distinct limits. The effective HTTP budget is the minimum of known
configured and native limits. No request exceeds it, including a single
indivisible input. Native microbatch size does not replace batch size. Missing
limits remain unknown.

The path preserves order/cardinality, cancellation, finite overall deadlines,
typed errors, content and semantic identity, frozen active topology, draining
and all-or-error result publication. Successful quotas are not replayed. The
scheduler can pass a job blocked on another alias while preserving FIFO within
an alias for admitted jobs. `embedding_queue_size=0` applies backpressure
without a second job queue.

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

Preparation and inference share the scheduler, windows and quotas.
`token_count_parallelism` is global to the adapter. Bulk admission, waits and
HTTP calls are bounded, cancellable and covered by an overall deadline.
`close()` drains preparation and HTTP cleanup as well as inference. Preparation
does not create a second embedding scheduler.

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

Native batch capacity is observed from supported metadata and structured
arguments on the verified loaded entry (`--batch-size`, `--batch-size=`, `-b`).
Malformed, duplicated or conflicting values are rejected. Capacity is not
inferred from paths, free-form presets, microbatch size or training context.
Loaded context and training context are distinct. The client performs no
server administration.

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
