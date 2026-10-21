# Model-wide embedding batching qualification 0.10.1

Date: 2026-10-21. Distribution `runtime-llama`, import `runtime_llama`.
Flower MCP governs this correction through `BOOT-000015`, `REQ-000028` through
`REQ-000030`, `MILE-000016`, `CHANGE-000016`, `PACKET-000017` and `RUN-000011`.
The adopted architecture is revision 2. Results below are host-reported
verification, not consumer acceptance.

## Corrected contract

The persistent engine gathers compatible prepared records across source
windows and concurrent callers in each verified replica's queue. One existing
scheduler still owns global and per-replica HTTP admission. Source windows
remain ordered delivery boundaries and no longer shrink HTTP batch targets.

`batch_size` remains the total-text distribution budget. The request text cap
is derived from that budget, usable aliases and the global concurrency limit.
For example, 256 texts over two usable replicas permits 128 per replica;
four permits 64. Neither value is hardcoded. Input context, explicit aggregate
HTTP token budgets, payload bytes and native computation batch sizes remain
distinct limits. A quota may need multiple requests.

Default `embedding_batch_wait_sec=0.0` fills through preparation and capacity
events. Finite admitted tails, resource pressure and unknown-dimension
bootstrap ensure progress without a required fill timer. Explicit positive
coalescing remains optional on the same engine. Preparation admission can
retire before HTTP drain; ready records, byte/token charges, output credits
and actual HTTP ownership remain charged for their respective lifetimes.

A shared POST retains immutable row membership and independent logical
owners. Pre-send validation, deadline or cancellation failure removes the
affected owner's work; surviving rows retain their original scatter indices.
Stopping one owner does not abort surviving owners. The shared transport's
finite deadline is fixed to the latest initial member deadline, and every
participant waits for actual drain, possibly beyond its own work deadline.
All-owner cancellation or close interrupts the transport. Backend/transport
failure of a shared POST fails all its participating owners; unrelated calls
remain independent. Native failed or ambiguous POSTs are never replayed.

## Deterministic and package checks

The final source passed **1,401 tests in 55.77 seconds** on macOS ARM64,
Python 3.11.15, plus Ruff check and format, wheel/source builds, clean wheel
installation, import/version checks, inert adapter construction/close and
dependency checks outside the checkout. Four warnings come from intentional
self-signed local TLS fixtures. No new Linux/Windows execution is claimed.

Controlled tests freeze the coalescing clock: four windows of 64 produce two
128-text POSTs; seven windows of 64 plus 63 produce 128/128/128/127. Coverage
also includes other replica counts, fewer texts than replicas, odd remainders,
slow replica preparation, concurrent callers, token/byte pressure, bounded
result delivery, topology changes, partial failures, shared cancellation and
deadlines, no replay and concurrent close. Worker-side owner failure cleans
unsent backlog and credits exactly once; an unaffected owner and subsequent
healthy call still succeed.

Tests reside in `test_embedding_model_batching.py`,
`test_embedding_shared_lifecycle.py`, `test_embedding_shared_validation.py`
and the existing embedding, async, topology and transport suites.

The installed candidate wheel SHA-256 is
`99d89d1f3e96c0f0f88032988269802a0c1db0ac83be531fdac0cc9369133780`.
All 27 runtime payload files matched source. The public wheel may have a
different hash because finalized README metadata and archive timestamps differ;
its runtime files must match this qualified candidate byte-for-byte.

## Limited live functional qualification

One installed-wheel run used two already loaded, content-verified
EmbeddingGemma 2 BF16 aliases, build `b11716-797de63b2`, context and native
batch/microbatch 2048, one slot each, Flash Attention off. Their reported
content and semantic identities were pinned. Before/after catalog, loaded
arguments and properties matched. No server administration, model reload,
consumer modification or consumer database/job access occurred.

The async source supplied **511 distinct texts in eight windows**:
64/64/64/64/64/64/64/63. Configuration used `batch_size=256`, global embedding
parallelism 2, queue size 2, tokenizer parallelism 2, four pending windows and
the default zero coalescing delay.

| Observation | Result |
| --- | --- |
| Pipeline embedding POST sizes, observed dispatch order | 128, 128, 127, 128 |
| POSTs mixing source windows | 4 of 4; four source windows per POST |
| Ordered output correlations | 511 of 511, native token-ID inputs, no duplicates |
| Pipeline embedding / tokenizer requests | 4 / 511 |
| Full probe requests including canaries and snapshots | 8 embedding, 528 tokenizer, 72 metadata |
| Peak active embedding / per alias | 2 / 1 |
| Peak tokenizer / all HTTP | 2 / 4 |
| Exact preparation worker calls | 511 |
| Pipeline elapsed wall time | 4.648 seconds |
| Sum of preparation worker wall times | 5.188 seconds, overlapping workers |
| Full functional probe wall time | 5.391 seconds |
| Pending bytes/tokens, plans, HTTP, handoffs, output credits at finish | All zero |

Each full pipeline POST contained 4096 tokens across 128 inputs of 32 tokens;
the final 127-input POST contained 4064. These aggregate request totals are
distinct from the reported 2048 per-input context and native batch size.
No explicit aggregate token budget was configured for this probe.

Thresholds were unchanged: dimension 768, maximum coordinate error `1e-5`,
minimum cosine `0.999999`, norm error `1e-6`, and raw-response projection
error `1e-12`. String/native-token canaries on both aliases passed with zero
coordinate difference. An offline comparison of those same current-run
eight-input canaries to the corresponding outputs from mixed 128-input POSTs
passed 32 comparisons: maximum coordinate error `5.56e-17`, minimum cosine
`0.9999999999999999`. All 511 raw-response projections passed, maximum error
`1.12e-16`, maximum norm error `2.23e-16`.

An oversized input measured exactly 3002 tokens against the effective limit
2048 and was rejected without embedding HTTP. The probe had a 180-second
wall budget and finite per-request limits, with no automatic retries.

The corpus digest is
`a723a493887bf14096265c52bb1f09f2171e1a13efb198b8e10f572cad4d7252`;
retained vector evidence SHA-256 is
`cfd1398c2bc16a8f67189eac0b2812a90a718e78e09e726eecfe6857dec64664`.
Private endpoint configuration and raw evidence remain outside the public
package. The opt-in probe is `probes/qualify_queue_engine.py`.

The LLM workload was allowed to remain active. **No comparison with 0.9.0 or
throughput improvement claim was made.** Times include client preparation,
network, runtime work, scheduling and instrumentation; they are not pure GPU
times. These synthetic checks do not establish retrieval quality or universal
equivalence across models, builds and hardware. Live model interruption and
consumer acceptance remain outside this qualification.
