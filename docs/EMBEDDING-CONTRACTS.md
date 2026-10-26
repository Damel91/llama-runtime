# Embedding contracts and public preparation

Version 0.15.0 removes implicit application quotas on text count, model/replica
count and retained input/result size. Resource correctness follows ownership
and actual release. Consumer applications still own source splitting, UTF-8
intervals, fragment identity, graph construction and persistence.

## Total text target, distribution and integrity

`batch_size=B` is an optional target selected by the consumer for distribution
across N verified usable replicas. It does not restrict how many texts a caller
may submit. With B=256, N=2 gives quotas up to 128 and N=4 up to 64. Quotient and
remainder handling preserve all original occurrences. These are consequences
of the chosen target, not hardcoded maximum POST sizes.

When B is omitted there is no replacement target of 32 or 256. The engine packs
compatible available work at actual dispatch capacity. Configured aggregate
HTTP token policies, explicit byte policies and verified server constraints
remain distinct from per-input context. Native `n_batch`, `n_ubatch` and slots
are observations; they do not establish an aggregate HTTP token budget.

Missing/`None` `embedding_request_parallelism` follows N. A positive G is an
explicit global guard `min(N,G)` without reducing N-wide quotas. Each canonical
replica owns one original POST lease, including serial isolation descendants,
until actual drain. Aliases of one entry do not add capacity; different
embedding identities are never pooled together. Catalog and executor handling
have no fixed maximum model count.

One persistent engine serves blocking calls, exact preparations, asynchronous
windows and flat progressive results. Compatible records may share a POST
across windows/callers, preserving independent cancellation and occurrence
indices. No second scheduler, compulsory batching timer, implicit truncation
or ambiguous native POST replay is introduced.

## Exact public preparation

```text
prepare_texts(texts: Sequence[str], *, input_type="document",
              timeout=None, cancel=None) -> tuple[PreparedEmbeddingInput, ...]
prepare_query(text: str, *, timeout=None, cancel=None) -> PreparedEmbeddingInput
embed_prepared(inputs: Sequence[PreparedEmbeddingInput], *,
               timeout=None, cancel=None) -> list[list[float]]
```

Preparation measures the actual task-prefixed input through the server
tokenizer, including the same special-token policy used by embedding. Prefixes
are applied once. An oversized input remains measurable; measurement does not
authorize inference beyond its actual configured/loaded limits.

`PreparedEmbeddingInput` exposes `text`, `input_type`, `token_count`,
`context_token_limit`, `batch_token_limit`, `max_input_tokens`,
`identity_verified`, content and semantic digests. `fits_context`,
`fits_request` and `fits_input` report fit; unknown limits remain `None`.
Prepared records are sealed for their issuing adapter. Altered or foreign
records are rejected, and retained token IDs participate in that seal.

Safe count/ID reuse requires matching weights, embedding semantics, task,
prefix/special policy and tokenizer build. A changed context alone does not
change an exact count, but fresh admission must apply the current actual limit.
Unknown identity/build cannot authorize reuse. Preparation does not install an
input-character ratio as exact token authority.

The ordinary automatic path is separate: it sends strings directly without
per-input `/tokenize`, implicit calibration or fake whole-context token
reservations. An explicit pending-token policy applies to counted/prepared
work; direct strings have unknown token usage and are not charged fabricated
counts. Explicit `character_budget()` calibration retains the 2,500-text
corpus and category 80% / mixed 90% factors; its recommendation uses effective
context rather than substituting native microbatch as a context ceiling.

## Optional cache and capacity policies

`embedding_preparation_cache_enabled=False` is the default. Enabling the cache
requests session-owned retention of completed exact counts/IDs, still subject
to the identity/policy checks above. The cache clears on close and does not
admit late insertion into a closed adapter.

Cache entry/byte policies are independent optional profile fields. With
`embedding_capacity_limits=True`, supplied `embedding_preparation_cache_entries`
and `embedding_preparation_cache_bytes` apply as chosen; zero disables
retention. Missing values impose no corresponding quota. Without the capacity
opt-in, legacy numeric fields do not activate limits. An intentionally enabled
cache with no quota can grow during its session; its contents are not silently
bounded to 1,024 entries, 8 MiB or a fraction of pending memory.

The same capacity opt-in governs pending-byte/token, request/response/handoff,
owner, queue and isolation policies. The fields default to `None`; no 64 MiB,
8 MiB, 1,048,576-token or derived-H preset is installed. `embedding_max_handoff_bytes`
is independent from pending and response settings. The old 75/25 partition,
response `pool/16`, materialization-based admission and automatic memory
halving of POST candidates are removed.

A consumer may explicitly opt into a resource policy. Its wait/failure names
that setting, not a presumed server limit. Enabling capacity controls does not
invent values for omitted fields. Existing list-returning calls naturally
allocate their complete output, without a compulsory whole-result pool or
exclusive bulk-preparation accumulator. This does not promise constant RAM or
prevent the physical environment from exhausting memory.

## Input and result ownership

Every acquired input, record, HTTP body, result, Future, callback and topology
generation has an owner. Ownership ends at explicit transfer or actual cleanup.
An object retained by the consumer is not retained on the library's behalf.
Closed iterators and public partial errors do not retain unrelated private
input/result structures through completed task frames or exception chains.

The flat progressive interface accepts a finite text sequence or asynchronous
text source and yields `EmbeddingRecordResult(index, embedding)` for individual
occurrences. Input cardinality is independent of the chosen POST target.
Source order and duplicates are preserved. Use it to transfer results while
later source work is still pending:

```text
aiter_texts(texts, *, input_type="document", timeout=None,
             cancel=None, close_source=False) -> EmbeddingRecordStream
```

The existing `astream_texts` and `astream_prepared` interfaces remain available
for consumers that choose complete-window delivery. Their default
`max_pending_windows=None` installs no per-stream count quota; a positive value
is an explicit stream policy. A large window is not rejected by the former
whole-output estimate. See [async interfaces](ASYNC-EMBEDDING-WINDOWS.md).

## Current identity, native tokens and errors

Relevant loaded identity and inference limits are observed after waits and
before actual dispatch/recovery. Coherent same-boundary catalog/props reuse
avoids duplicate GETs without creating a stale inference permit. Local topology
counters are not a remote load lease. The server operator retains all model,
process, context, batch and hardware administration.

`embedding_input_format="auto"` sends retained native token arrays only when
the client declares support and identity/build/policy prove them reusable.
`text` forces strings; `tokens` requires safe IDs or fails before dispatch.
Unknown custom clients use text in auto mode. There is no replay after an
ambiguous native POST. Standard native context errors remain typed;
recognized adaptive 400 recovery is confined to the separate custom-client
contract and the failed POST. Its subdivisions do not install a smaller target
for later work. Native execution sizes remain diagnostics, not inferred HTTP
limits.

In the automatic direct path a 500 response can trigger the requested
subbatch isolation on the same original alias lease. Successful descendants
are retained and rejected singletons produce lossless halving advice for the
consumer. This does not prove token overflow, tokenize ordinary inputs or
change the saved character recommendation. Optional isolation request limits
apply only through the capacity opt-in; otherwise finite input subdivision and
the existing deadline/cancellation govern that work.

An owner keeps its original work deadline. Stop/expiry withdraws its pending
work and response portion without cancelling surviving owners in a shared
POST. The group transport deadline is fixed at dispatch; it is not renewed as
members leave. Actual cleanup may outlast a shorter owner's work deadline.
All-member Stop or close cancels/drains the transport. Synchronous wait
abandonment preserves the primary exception through repeated cleanup
interruptions. Async callers use `await aclose()` to include source cleanup.

Use `async with` or `aclose()` for deterministic early exit. Dropping a
progressive facade signals Stop and schedules inner cleanup on a live owning
loop; it cannot execute cleanup after that loop has already closed or forcibly
terminate uncooperative source/provider code.

## Consumer example

```python
from runtime_llama import build_embedding_adapter

adapter = build_embedding_adapter({
    "model": "example-embedding.gguf",
    "base_url": "http://inference.example.invalid:8080",
    "embedding_replica_discovery": True,
    "batch_size": 256,
    "embedding_request_parallelism": None,
    "timeout": 120,
    "embedding_prefixes": {
        "document": "title: none | text: ",
        "query": "task: search result | query: ",
    },
})
try:
    prepared = adapter.prepare_texts(["first text", "second text"])
    if all(item.fits_input is True for item in prepared):
        vectors = adapter.embed_prepared(prepared)
finally:
    adapter.close()
```

The configured model must be available under the server's policy. Prefix
strings in the example belong to one task contract, not every model. Verified
replica pooling uses supported content/semantic identity attestations; direct
selection and tokenizer/cache/calibration prerequisites remain distinct.

## Verification criteria

- Large sequence/window/progressive input, including 10,000 texts and vector
  dimensions beyond the old whole-output boundary; no omissions or duplicates.
- Distribution across verified replicas, explicit global guards, uneven tails,
  topology changes and catalog sizes beyond the former client ceilings.
- Opt-in controls apply exactly; omitted/legacy scalar settings do not activate
  the former limits or hidden fractions.
- Source-ordered progressive transfer, consumer pauses, partial errors,
  cancellation, interruption, iterator abandonment and repeated close.
- Ownership counters, weak references/finalizers and repeated allocation cycles
  show that finished work is no longer retained. RSS and allocator high-water
  marks are not substituted for this evidence.
- Repository tests, lint/format, build and external installed-wheel smoke;
  [release verification](verification/PUBLIC-RELEASE-0.15.0.md) records actual
  outcomes separately from live model qualification and consumer acceptance.

Historical numerical failures remain recorded as the accepted model-profile
limitation. These changes do not invent a numerical pass, a retrieval-quality
result, a general throughput improvement or pure GPU timings.
