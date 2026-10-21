# Metadata, preparation and packing qualification 0.10.2

Date: 2026-10-22. Distribution `runtime-llama`, import `runtime_llama`.
This is a verified local candidate, without consumer migration or publication.
The implementation commit is `c021bf4`.

Flower MCP retains lifecycle authority through `BOOT-000016`, requirements
`REQ-000031` through `REQ-000035`, milestones `MILE-000017`/`MILE-000018`,
changes `CHANGE-000017`/`CHANGE-000018`, packets `PACKET-000018`/`PACKET-000019`
and runs `RUN-000012`/`RUN-000013`. Architecture revision 4 adds the synchronous
owner-drain and incremental-readiness decisions while retaining revision 3's
metadata/preparation decisions. These results are host-reported verification;
human and consumer acceptance are not inferred.

## Corrected contracts

Admission returns the catalog that actually governed joining the pool.
`_engine_start` reuses it instead of fetching the same catalog again. Frozen
admission cohorts share coherent metadata while retaining each owner's pool
reference, failure, cancellation and deadline. Dispatch rounds share a fresh
catalog among their already selected aliases and obtain properties separately.
There is no TTL-based permission to use stale model metadata.

A completed-token LRU is local to one adapter and bounded by both 1,024 entries
and `min(8 MiB, embedding_max_pending_bytes // 4)` retained bytes. Its key
compares the full prepared text, task, prefix/special-token policy, verified
content and semantic hashes, tokenizer build and vocabulary observation.
Unknown identity/build bypasses reuse. Hits do not retain inference permission:
current limits, prepared-record seals and fresh pre-POST checks still apply.
Incomplete or failed tokenizations are not retained. Concurrent cold misses
remain independent, and close clears the cache and rejects late insertion.

An abandoned synchronous result wait retains its real Future, sets only that
call's stop flag, wakes the coordinator and waits for actual drain before
reraising the original exception. Repeated interruptions cannot abandon cleanup
or replace that exception. Other participants in a shared POST continue.
Completed typed worker errors retain their existing behavior. This covers the
result-wait boundary, without changing submission-registration ownership.

Ready queues update exact token, text/ID wire-byte and finite-limit summaries
on append, removal and clear. Numeric ID prefixes account for a switch from
native IDs to text without rescanning all prior rows. Consumed storage is
compacted geometrically. Actual packing uses deque iterators and constant-time
owner rotation only when readiness warrants constructing a candidate. No
selected record prefix persists across metadata or capacity waits; membership,
observations and capacity are reconsidered. An additional conservative 128-byte
charge per ready record covers the summary index; queue credits are not a
process RSS guarantee.

A partial preflight renews its metadata membership when another useful owner
arrives. Newly admitted rows never inherit the older snapshot. Pending admission
can fill a batch, while topology-busy, failed or result-credit-blocked owners
cannot prevent existing work from draining. Full batches, resource pressure,
bootstrap and an explicitly elapsed coalescing wait preserve progress.

`batch_size` remains a configurable total-text budget. Usable replicas and the
existing scheduler determine per-alias targets; 128 is not hardcoded. Input
context, aggregate HTTP token budgets and native batch/microbatch remain
separate. Sync and async callers still use the same scheduler, cancellation,
backpressure, ordered scatter, typed failures, identity checks and drained close.

## Deterministic and package checks

The final source passed **1,506 tests in 55.90 seconds** on macOS ARM64 with
Python 3.11.15, plus Ruff check/format, wheel and source builds and an isolated
installed-wheel smoke test outside the checkout. Four warnings come from
intentional self-signed loopback HTTPS fixtures. No new Linux/Windows execution
is claimed.

Relevant additional coverage includes:

- Four bounded child processes sending repeated real SIGINT during metadata,
  preparation, embedding HTTP and a shared POST. The genuine Future remains
  active until drain, the first exception is preserved and the survivor succeeds.
- Portable active-Future `KeyboardInterrupt`/`SystemExit` injections and unchanged
  typed preparation/HTTP worker failures.
- 1,000 deterministic comparisons of selected records, fair order, saturation
  and rejection with varying text caps, token limits, wire budgets and mixed IDs.
- 128 individual arrivals: zero record visits while filling, then 128 preview
  and 128 dispatch visits, **256 total**, replacing the 8,256 growing-prefix
  visits in the reported example. These are local record visits, not HTTP calls.
- Late membership, pending admission, cancelled owners, topology-busy admission,
  result-credit pressure, full-batch progress and explicit coalescing deadlines.
- Cache bounds, collisions, task/prefix separation, changed build/vocabulary,
  unknown identity, altered/foreign seals, oversized counts, current reduced
  context, cancellation/timeouts, eviction, concurrent access and close.
- External health/discovery owners releasing pool or discovery gates, including
  availability changes before a nonblocking admission completion is processed.

An existing concurrent test was synchronized on the coordinator's published
`embedding_in_flight=1` state. A worker can start before that snapshot is
updated; its unchanged ready-count and token-budget assertions now observe the
intended post-dispatch state rather than an earlier snapshot.

The live-qualified candidate wheel SHA-256 is
`5e07f27fc977d81872da9214e8a2b243afd0aaec2ba096b757ee75dc6c564f56`.
All **28 runtime payload files** matched source and the isolated installation.
The final delivery archive can differ because README metadata and archive
timestamps are finalized separately; its runtime payload must match this
qualified candidate byte-for-byte. Delivery checksums accompany the artifacts.

## Limited live qualification

The installed wheel used the same two content-verified EmbeddingGemma 2 BF16
replicas as the prior 0.10.1 qualification, build `b11716-797de63b2`, context
and native batch/microbatch 2048, one slot each and Flash Attention off. Before
and after profiles matched, and they also matched the retained baseline. No
models, processes, server configuration, consumers, databases or jobs changed.

The unchanged corpus contains **511 previously unseen texts** in eight windows
of 64/64/64/64/64/64/64/63. Settings remain `batch_size=256`, embedding parallelism
2, queue size 2, tokenizer parallelism 2, four pending windows and default zero
coalescing wait. Canaries and an unrelated oversized input precede this phase;
they do not warm these 511 cache keys.

| Pipeline observation | 0.10.1 retained baseline | 0.10.2 qualified candidate |
| --- | --- | --- |
| Exact tokenizer requests | 511 | 511 |
| Embedding POSTs | 4 | 4 |
| Metadata requests | 50 | 24 |
| Health / catalog / properties | 8 / 21 / 21 | 4 / 8 / 12 |
| All pipeline HTTP requests | 565 | 539 |
| POST sizes in observed dispatch order | 128, 128, 127, 128 | 128, 128, 127, 128 |
| Pipeline wall time | 4.5116 s | 4.6436 s |

Metadata requests fell by **52%**. This single run does **not** demonstrate a
wall-time improvement. Timing includes client work, exact preparation, HTTP,
server scheduling and result handling; it is not pure GPU time. Preparation
ran 511 hooks with a sum of overlapping worker wall times of 5.1568 seconds.
Peak active embedding requests were 2, one per alias; tokenizer peak was 2 and
the observed all-HTTP peak was 3. All four POSTs mixed source windows, and all
511 ordered correlations passed. The cumulative packing counter was 1,022;
pending work, results, metadata, HTTP and queue credits drained to zero.

Thresholds remain dimension 768, maximum coordinate error `1e-5`, minimum cosine
`0.999999`, norm error `1e-6` and same-response projection error `1e-12`.
All 511 final outputs matched the retained baseline with **zero coordinate
difference**; minimum cosine was `0.9999999999999998`. Current raw projection
error was at most `1.12e-16`, and norm error at most `2.23e-16`. String/native-ID
canaries, cross-batch checks and both replicas passed. A 3,002-token input was
measured against limit 2048 and rejected without embedding HTTP.

An earlier candidate passed numerical checks but produced 128/128/96/31/128
POSTs. Its retained evidence exposed the late-membership tail issue; the
correction and deterministic regression above preceded the final qualification.

## Separate exact-reuse exercise

A fresh adapter prepared eight additional Unicode/whitespace texts in 0.0648 s,
using eight tokenizer requests and four metadata reads. It then performed two
`embed_prepared` calls and two repeated raw `embed_texts` calls. **Each used zero
additional tokenization**, two embedding POSTs and seven metadata reads. Their
wall times were respectively 0.1064, 0.0915, 0.0924 and 0.0942 s. All exact-ID
correlations, projection and numerical comparisons passed.

Preparation worker sums for the four reuse phases were 0.152, 0.131, 0.204 and
0.164 ms; prepared-seal checks added overlapping measured sums of 0.081 and
0.117 ms for the prepared phases. These small-sample hook measurements are not
performance guarantees. Cache occupancy remained eight entries/16,312 charged
bytes, with 16 raw-input hits; close cleared every entry.

The full probe, including initial/final snapshots, canaries, oversize validation
and the extra reuse phases, used **628 requests**: 536 tokenizer, 16 embedding
and 76 metadata, in 5.8345 s. It is not comparable to the older full probe
without those extra phases; the pipeline rows above are the common scope.

Unique cold inputs still require one exact native `/tokenize` request each.
Reuse benefits previously measured or repeated compatible inputs; it does not
replace exact tokenization with character ratios or pretend native `/tokenize`
supports independent bulk sequences. Concurrent cold duplicates may compute
independently rather than sharing a leader-owned cancellation/deadline.

Corpus SHA-256:
`a723a493887bf14096265c52bb1f09f2171e1a13efb198b8e10f572cad4d7252`.
Final vector evidence SHA-256:
`75279fe8fad2513d0470ad8176325971dfe3c7111c13d5522fa0b67c53ba7ccd`.
Private endpoint configuration and raw evidence remain outside the repository.
Reproduction uses the opt-in `probes/qualify_queue_engine.py --qualify-reuse`.
