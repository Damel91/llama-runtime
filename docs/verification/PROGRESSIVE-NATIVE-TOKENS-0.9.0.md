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
