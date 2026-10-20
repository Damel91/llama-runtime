# Progressive embedding and native token inputs — local 0.9.0

Qualification on 20 October 2026 for Flower project `llama-runtime`,
`CHANGE-000012`, `PACKET-000013`, `RUN-000007`. This is a local wheel candidate;
publication, shared installation and consumer acceptance are separate. No
CodingCastle, Flow, Flower or Noxi consumer/configuration was changed.

## Delivered behavior

Ready records retain their original indices and feed the existing HTTP scheduler
before their full quota finishes tokenizing. Sync calls and async window sources
share this path, global admission, per-replica slot limits and drained cleanup.
Complete public windows remain ordered and all-or-error. A late invalid input
is never inferenced; earlier valid work is drained without replay or a partial
window result.

`batch_size` remains the total text budget: 256 texts over two usable aliases
gives quotas of 128 each, derived dynamically. A quota may generate several
HTTP requests. `embedding_batch_wait_sec=0.5` starts at its first unsent ready
record, is not reset by new arrivals and coalesces partial batches. Filled
derived text caps, explicit HTTP token budgets and producer completion can
flush earlier. Context per input, native computation batch/microbatch and
aggregate HTTP token budgets remain distinct.

`embedding_input_format="auto"` uses immutable exact token IDs when the native
client contract, verified content/semantic identity, tokenizer build and
prefix/special policy authorize reuse. `text` sends strings; `tokens` requires
safe IDs before dispatch. Unknown custom clients use text in auto mode. IDs
are covered by the same-adapter preparation seal. No hidden inference probe or
text fallback occurs after an ambiguous POST. The native client propagates even
structured context errors without bisection, since sibling inputs may have run.

See [configuration and consumer example](../PROGRESSIVE-EMBEDDING.md),
[preparation signatures](../EMBEDDING-CONTRACTS.md#public-api) and the unchanged
[async window interface](../ASYNC-EMBEDDING-WINDOWS.md#public-interface).

## Native contract and deployment

Source inspection pinned upstream commit
`797de63b2fa7396dcc8caca76834c699ac84e1c3` and checked the relevant implementation
against `839a572a4cdecdb80d211297b8e7d7323a6e99d2`. The inspected input dispatch
and output formatter agree: nested integer arrays enter the native token path,
which copies IDs without calling string tokenization or adding BOS/EOS.
References: [input implementation](https://github.com/ggml-org/llama.cpp/blob/797de63b2fa7396dcc8caca76834c699ac84e1c3/tools/server/server-common.cpp#L1029-L1061)
and [official embedding tests](https://github.com/ggml-org/llama.cpp/blob/797de63b2fa7396dcc8caca76834c699ac84e1c3/tools/server/tests/unit/test_embedding.py).
The source observation is complemented by the live text/ID canary; a version
string does not audit a remote binary or every compatible proxy.

The existing loaded deployment exposed two content- and semantics-verified
EmbeddingGemma 2 BF16 replicas, on separate ROCm devices, build
`b11716-797de63b2`. Both had context/native batch/microbatch 2048, one slot and
Flash Attention disabled. The configured dimension was 768. Profile, identities
and idle slots were checked before and after each arm; no loading, reconfiguration
or server administration was performed. HTTP used the standard connection-close
policy; no keep-alive patch is required by this client path.

## Deterministic and package verification

- Full suite: **1,303 passed** in 50.40 seconds; four expected warnings from
  deliberate unverified-TLS loopback fixtures.
- Ruff check passed; format check passed for 71 files.
- Wheel and source distribution built successfully. Clean installed-wheel
  smoke resolved all 51 public exports, checked API signatures and verified
  inert factories, unstarted-stream closure and idempotent adapter close.
- An independent initial-wheel audit passed 27 checks, including all 26 runtime
  files matching the tested source, package metadata, RECORD/CRC, typing marker,
  unchanged Apache/NOTICE/vendored MIT attribution and absence of concrete private
  endpoints, user paths, email addresses or credentials.

The new deterministic cases exercise dispatch before the last count, coalescing
without timer resets, odd/small windows, duplicates and out-of-order readiness,
token budgets, sealed/recounted IDs, once-only prefixes and special tokens,
malformed IDs, text/auto/tokens selection, stale identity/build, mixed sync/async
calls, cancellation, timeout, topology, backpressure and late-failure drainage.
Native HTTP context-failure tests cover completed and still-draining siblings
without replay. A concurrent capacity-observation regression prevents an older
snapshot from restoring a replica's reduced admission limit. Earlier failing
test and audit evidence was retained; the corrected source passed the full suite.

## Bounded live workload and numerical checks

One opt-in execution ran canary, baseline 0.8.0 and candidate 0.9.0 sequentially
from distinct clean wheel installations. Hard global ceilings were 50 embedding
POSTs, 1,200 tokenization POSTs, 600 metadata GETs and 240 seconds. Actual totals
were **27 embedding POSTs, 1,046 tokenizations and 256 metadata GETs**, within
22.625 seconds including controller overhead. No retry, model reload, consumer job
or forced live cancellation was used. All three stages drained completely.

| Check | Result |
| --- | --- |
| Native canary: 31- and 919-token inputs, text versus exact IDs on both aliases | 4 comparisons passed; raw maximum absolute difference 0 |
| Same 511 ordered documents through baseline text and candidate IDs | 511 comparisons passed; public maximum absolute difference 0 |
| Eleven frozen discriminants, SDK versus native text references on both replicas | 22 comparisons passed |
| Eleven raw native cross-device comparisons | Passed; maximum absolute difference 0 |
| Same-response native-to-public normalization projection | Maximum absolute difference 1.11e-16 |
| Three prepared queries reused through async API | Zero extra tokenizations during reuse |
| Exact input boundary and oversize | 2048 admitted; 2049 measured and rejected with zero embedding POSTs; following query passed |
| Raw/query/document modes | Once-only prefix and special-token policy preserved |

Thresholds stayed unchanged: 768 dimensions, component maximum absolute error
`1e-5`, cosine similarity at least `0.999999`, unit-norm error at most `1e-6`,
and same-response normalized projection error at most `1e-12`. The 33
discriminant comparisons had maximum component difference `5.56e-17` and
maximum norm error `4.13e-8`. Numerical agreement is not a retrieval-quality
evaluation.

Candidate SDK requests all used actual nested ID matrices: 17 POSTs for 530
public vectors. The two additional textual POSTs produced 22 direct native
references after SDK work drained. The baseline produced 511 public vectors;
the canary produced eight. Token/text association used captured full ID arrays,
once-prefixed UTF-8 text hashes and input ordinals, rather than sequence length.

## Descriptive timing for the same 511 documents

Both arms used windows 256 + 255, total batch budget 256, global HTTP parallelism
two, tokenizer parallelism two and one observed slot per replica. The first
256 texts were byte-identical to the historical corpus; the following 255 were
distinct continuations. This was not a replay of every older qualification input.

| Observation | Baseline 0.8.0 | Candidate 0.9.0 |
| --- | ---: | ---: |
| Document phase wall time | 8.303 s | 7.844 s |
| Exact tokenization POSTs | 511 | 511 |
| Embedding POSTs | 4 | 8 |
| Metadata GETs within phase | 26 | 34 |
| Preparation interval union | 2.363 s | 2.619 s |
| Sum of overlapping preparation job times | 4.717 s | 5.074 s |
| First tokenize start to first embedding start | 1.195 s | 0.532 s |
| Captured semantic embedding JSON bytes | 72,718 | 92,803 |
| Global/per-alias embedding HTTP peak | 2 / 1 | 2 / 1 |

Baseline request sizes were 128, 128, 128, 127. Candidate sizes were 43, 23,
58, 58, 47, 27, 128, 127; three candidate requests started before their quota
producer finished. The aliases processed 256 and 255 texts respectively, with
all 511 returned once in order.
During a deliberate consumer pause, only two source windows were pulled and
their leases remained held until ordered delivery resumed. The event loop
continued ticking; the largest observed candidate heartbeat gap was 0.045 s.

The candidate was 0.459 seconds shorter in this adjacent observation. This is
not a controlled speedup claim: the baseline ran first, remote caches and other
system activity were not controlled, and timing includes Python, exact
tokenization, network, capture files and server work. Overlapping intervals
cannot be added as elapsed time or interpreted as pure GPU time. Earlier
4.21-second observations came from a different run and are not this baseline.

The ID path removes the second internal string tokenization, not the initial
511 exact remote tokenizations. Progressive readiness can increase request
count, and ID JSON was larger for this corpus. Captured byte totals describe
canonical semantic JSON, excluding headers; they are not wire-byte measurements.
The coalescing budget is configurable rather than a promise of maximum bandwidth
or universal throughput improvement.

## Artifact and evidence binding

### Repeat with the orchestrator reported inactive

A subsequent opt-in execution on the same date repeated the unchanged controller,
corpus, settings and numerical thresholds after the operator reported stopping
the orchestrator. The final clean-installed 0.9.0 wheel was used. Loaded model
profiles remained unchanged; no server, model, consumer or library code was
modified for this repeat.

| Same 511-document phase | Baseline 0.8.0 | Candidate 0.9.0 |
| --- | ---: | ---: |
| Wall time | 3.813 s | 3.988 s |
| Embedding / tokenize / metadata requests | 4 / 511 / 26 | 14 / 511 / 46 |
| First tokenize start to first embedding start | 1.275 s | 0.537 s |
| Preparation interval union | 2.725 s | 3.584 s |
| Global/per-alias embedding HTTP peak | 2 / 1 | 2 / 1 |

The candidate dispatched ten partial batches before their quota producers
finished. Its requests contained 17 to 51 texts, demonstrating the configurable
coalescing tradeoff rather than a changed total text budget. It was 0.175 seconds
(4.58%) longer in this single adjacent pair, while starting embedding earlier.
This repeat demonstrates no total-throughput improvement at the default
coalescing setting on this workload.

All 511 ordered A/B outputs were identical; canary and 33 discriminant checks
passed at unchanged thresholds. The independent offline audit also verified
the full token sequences, 1,041 same-response projections, bounded admission and
complete drainage. The whole qualification used 33 embedding POSTs, 1,046
tokenizations and 268 metadata GETs within 12.626 seconds.

Both document times returned to approximately four seconds, versus the previous
8.303/7.844-second pair and the historical 4.209-second 0.8.0 observation.
This is consistent with the reported concurrent orchestrator having affected
the earlier measurements. Its inactive state is an operator report, not an
independent GPU utilization/clock/power measurement; remote caches and ordering
remain uncontrolled. It does not establish an exact causal contribution or
GPU-only time.

The repeat's private result SHA-256 is
`8b8010e0c13e9b62343428e02db0e5b79e499f300c3894c0bf0f99c5346ec39a`;
its ordered A/B comparison is
`cf9a47a145f37fe434f9b3531b0d0a7a6fe6963c4dc3fc79b9a0d7f7420dbcbd`.
The original local artifacts and their checksums remain unchanged; this is
additional verification evidence.

### Original artifact binding

Tested implementation/test revision:
`e0333dd410c83ce39fc1827d0c2a738c62ef7ecd`.
The live-qualified initial 0.9.0 wheel SHA-256 was
`f4cdb420c653df49707ddd88cce870171b4231fda18789b9e43fae55979897f9`;
the baseline 0.8.0 wheel was
`64f0d891e37efb8bc8459f6e72fa499280c45e7ff577dc5fee51a7bc4de37b74`.
Final delivery hashes are supplied with the local artifacts. Documentation-only
wheel metadata changes are bound by exact equality of all 26 runtime files to
the installed live-qualified wheel and tested source.

The private top-level live result SHA-256 is
`fe52cf8b00984f567b4cae2e24682766643ea1bf462eccacdbbf80c7dca7a181`;
the ordered A/B comparison is
`733e7787e7cb53de6e4f097563926ed5dcd77d4b8c200f6219b4c3fc3f4905bc`.
Flower retains references to checks, pinned source evidence, full token/request
and response captures, preparation/HTTP intervals, ordered public windows,
independent audits and delivery manifests. Private endpoint/model paths stay
outside public artifacts. Host-reported verification does not imply human
acceptance, production workload endurance or every server/model configuration.

## Bounded repeat with Nomic replicas

On 21 October 2026 (Europe/Rome; execution started at 22:00 UTC on 20 October),
the operator replaced the loaded embedding models with two
`nomic-embed-text-v2-moe.Q8_0.gguf` replicas and requested another bounded test.
Both exposed matching content and embedding-semantic identity tags, dimension
768 and vocabulary 250048. Each ran on a separate ROCm device with context,
native batch and microbatch 512, mean pooling, one slot and Flash Attention off,
using build `b11716-797de63b2`. Profiles and idle slots were checked before and
after each arm. No model, runtime, consumer or shared installation was changed.

The same 511 document bodies were submitted as ordered windows of 256 and 255,
with total batch budget 256, global HTTP and tokenizer parallelism two and one
active embedding request per alias. This model used `search_document: ` and
`search_query: ` prefixes. A fresh model-specific probe replaced the
Gemma-specific tokenizer, prefix and boundary fixtures; the older full Nomic
campaign and Gemma's 2048/2049-token boundary tests were not repeated.

| Nomic 511-document phase | Baseline 0.8.0 | Candidate 0.9.0 |
| --- | ---: | ---: |
| Client wall time | 3.504 s | 3.798 s |
| Embedding / tokenize / metadata requests | 4 / 511 / 26 | 12 / 511 / 42 |
| First tokenize start to first embedding start | 1.164 s | 0.537 s |
| Preparation interval union | 2.684 s | 3.471 s |
| Global/per-alias embedding HTTP peak | 2 / 1 | 2 / 1 |

Each arm distributed 256 and 255 texts across the two verified replicas. All
511 ordered public A/B vectors were identical. A three-input native canary
also passed all 21 comparisons between text and exact token IDs, singleton and
batch shapes, and matching requests across replicas, with maximum absolute
difference zero. Its maximum raw unit-norm error was `3.713e-8`. The probe's
public same-response normalization error was at most `2.776e-17`; an independent
offline recomputation using separate arithmetic measured at most `4.164e-17`.
Thresholds remained
`1e-5` maximum absolute error, cosine at least `0.999999`, unit-norm error at
most `1e-6` and same-response projection error at most `1e-12`.

The candidate sent native ID matrices in all 12 document requests. All inputs
were exactly measured and within the loaded 512-token per-input context.
Native batch/microbatch 512 are computation sizes, not an asserted aggregate
HTTP token ceiling; no explicit aggregate request token limit was configured.
All work drained, and no retry or automatic resubmission was performed. The
whole canary/A/B execution used 26 embedding POSTs, 1028 tokenizations and 130
metadata GETs within 11.519 seconds, under finite global and request budgets.

This adjacent pair showed earlier candidate dispatch but a total time 0.295
seconds longer (8.4%). It demonstrates no throughput improvement at default
coalescing 0.5 seconds on this workload. Timings include preparation, HTTP,
capture overhead and server work; caches, arm ordering and global GPU load
were uncontrolled. They are not pure GPU times or a controlled comparison
between Nomic and Gemma, and numerical agreement is not retrieval-quality
evaluation.

The retained private controller result SHA-256 is
`ae55aa4524020d58fe443dc6d0f32529a510b7321510c2a16c223e5d74049566`;
the ordered A/B comparison is
`a4fe3d8dd1e248f1c8a4e56d21065177178aac7d98be8b35541eb24a36916738`.
The previously delivered wheels and runtime sources remain unchanged.

A separate standard-library-only offline audit recomputed the 21 canary and
511 A/B comparisons, all 1022 public normalization projections and 517 full
token-ID rows from the retained responses. It verified wheel/installed payloads,
source and capture hashes, identities, ordering, concurrency and complete
drainage. Eight candidate requests started before their preparation quota
finished. Its retained report SHA-256 is
`79f6fd5c69001a52429d51df562804fc23201ae032a716738d7d8fba7404b49b`.
