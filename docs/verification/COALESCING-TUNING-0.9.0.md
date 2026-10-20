# Coalescing tuning on the loaded Nomic profile

21 October 2026, Europe/Rome. Flower project `llama-runtime`,
`CHANGE-000013`, `PACKET-000014`, `RUN-000008`.

## Diagnosis and bounded plan

The progressive queue publishes exactly tokenized records before their full
replica quota finishes. The finite `embedding_batch_wait_sec` timer starts at
the oldest unsent ready record and is not reset by later arrivals. At the
default 0.5 seconds, a quota can therefore produce several timed partial
requests followed by a final tail. These splits are not required by a native
computation batch size or an aggregate HTTP token limit.

For the preceding 511-document Nomic comparison, baseline 0.8.0 made four
embedding POSTs and candidate 0.9.0 made twelve. Each additional POST correctly
required a fresh catalog and properties check, explaining the metadata increase
from 26 to 42 within the document phase. There is no duplicate observation
inside the same send. Both exact tokenizers remained concurrent, with a global
peak of two; 511 unique inputs still required 511 exact tokenization calls.
Some long request intervals occurred, but their remote/client cause was not
isolated. Request counts establish extra work; they do not assign the entire
wall-time difference to that work.

The minimum next experiment was to compare existing wait settings 0.5, 1.0 and
1.5 seconds before introducing another policy, caching metadata or changing
the scheduler. All three arms used the same clean-installed 0.9.0 wheel,
511 unique document bodies in ordered windows 256 + 255, Nomic task prefixes,
batch budget 256, global embedding/tokenizer parallelism two and verified
dynamic replica discovery. The loaded Q8_0 replicas had matching content and
semantic identity tags, one slot each, context/native batch/microbatch 512,
mean pooling and Flash Attention off, on separate ROCm devices, using build
`b11716-797de63b2`. Before/after checks pinned each loaded profile and idle slots.

## Observations

| Wait setting | 0.5 s | 1.0 s | 1.5 s |
| --- | ---: | ---: | ---: |
| Client wall for 511 documents | 3.540 s | 3.494 s | 3.584 s |
| Embedding POSTs | 14 | 8 | 4 |
| Exact tokenizer POSTs | 511 | 511 | 511 |
| Metadata GETs within the document phase | 46 | 34 | 26 |
| First tokenize start to first embedding start | 0.532 s | 1.036 s | 1.201 s |
| Preparation interval union | 3.077 s | 2.873 s | 2.657 s |
| Requests starting before their quota finished preparing | 10 | 4 | 0 |
| Global/per-alias embedding HTTP peak | 2 / 1 | 2 / 1 | 2 / 1 |

Each arm returned all 511 vectors once, in order, and distributed 256 and 255
texts across the two replicas. All 1022 comparisons of the latter two arms
against the first were identical, with maximum component difference zero.
Thresholds were unchanged: 768 dimensions, maximum absolute error `1e-5`,
cosine at least `0.999999`, unit-norm error at most `1e-6`, same-response
normalization error at most `1e-12`. All document requests sent actual exact
token-ID matrices, and every input fit its measured 512-token loaded context.
Native computation batch/microbatch sizes were not treated as aggregate HTTP
token ceilings.

The three arms together used 26 embedding POSTs, 1533 tokenizations and 184
metadata GETs in 13.864 seconds, including controller/arm overhead, under global
ceilings of 120 embedding, 1600 tokenization, 300 metadata requests and 240
seconds. All work drained. No automatic retries, additional canary inference,
model administration, consumer changes or shared installations were performed.
The preceding text/ID/singleton/cross-replica canary was retained by hash;
fresh profile checks guarded each arm.

## Selected configuration and limits

For this measured Nomic profile, use **`embedding_batch_wait_sec=1.0`**. It cut
embedding requests from 14 to 8, about 43%, and preserved four observed partial
dispatches while preparation continued. The first request started about half
a second later. The 46 ms total-wall difference is small and is not evidence
of a reproducible throughput gain. At 1.5 seconds the complete quotas were
ready before their timer expired, so that arm did not exercise partial
progressive dispatch and did not improve total time.

The setting is already public. It applies equally to synchronous calls and
async window sources using the existing scheduler. Example, with the consumer
providing its endpoint:

```python
from runtime_llama import build_embedding_adapter

adapter = build_embedding_adapter({
    "base_url": server_url,
    "model": "nomic-embed-text-v2-moe.Q8_0.gguf",
    "embedding_replica_discovery": True,
    "batch_size": 256,
    "embedding_request_parallelism": 2,
    "token_count_parallelism": 2,
    "embedding_batch_wait_sec": 1.0,
    "embedding_input_format": "auto",
    "embedding_prefixes": {
        "document": "search_document: ",
        "query": "search_query: ",
    },
})
```

The global default remains 0.5 seconds: this one model/profile/corpus does not
justify changing latency for every consumer. No runtime source, API, scheduler,
package version or wheel changed, so no code regression suite, rebuild or
release was needed. Documentation changes passed `git diff --check`. Applying
the measured setting to a consumer is separate work; none was modified here.

A fixed-order adjacent sweep cannot control remote caches, thermal state or
other GPU load. Timings include Python preparation, HTTP, capture overhead and
server work; overlapping intervals cannot be summed as elapsed time or called
pure GPU time. These are model-specific observations, not universal request
counts, retrieval-quality measurements or a statistical speedup claim. All
per-POST identity, limits and topology checks remain intact.

## Evidence binding

Unchanged tested 0.9.0 wheel SHA-256:
`704e32891fdf8ecf6579825dd84a7d7039a367cd4814776c719b7d29939f4b06`.
Private sweep controller result SHA-256:
`7d61d13c6b36f0f05d99fef4682101ac8b7eb9ca9d6592610f276dfa384cf202`.
Flower retains the current-profile checks, semantic request/response payloads,
full token IDs, ordered public vectors and preparation/HTTP intervals. Private
endpoint and model paths remain outside this document.

An independent standard-library-only offline audit passed all 1022 A/B
comparisons, 1533 same-response public normalization projections and 1533
full token-ID row associations. It also checked actual installed-wheel files,
source/capture hashes, profiles, ordering, concurrency and complete drainage.
Its maximum recomputed projection error was `4.164e-17`. Retained audit report
SHA-256:
`cbf3ece0e0616ce7739a6cd55e3e4bf23f2f12033b6906e1b727e670c7672116`.
