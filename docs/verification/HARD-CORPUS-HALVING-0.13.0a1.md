# Hard corpus and isolated source halving — local 0.13.0a1

Date: 2026-10-23. Flower project `llama-runtime`, requirement `REQ-000042`,
change `CHANGE-000027`, packet `PACKET-000028` (specification 3, plan 1).
Code commit: `e59d24e16eaa3c8356125589bdda54a1b3e7053a`.

This is a tested local candidate, not a publication or consumer deployment.
The earlier numerical finding `FIND-000007` remains open. No consumer,
model profile, server build or numerical threshold was changed.

## Delivered behavior

- Package 2,500 hard original synthetic texts: 500 per category.
- Use each category's weighted characters/prepared-token mean at 80%; use the
  complete selected corpus mean at 90% for `mixed`.
- Calibrate only on explicit character-budget requests; cold calibration counts
  2,500 texts. Ordinary embedding and failure isolation never tokenize.
- Keep the calibrated mean and recommendation unchanged on inference failure.
- Isolate direct HTTP 500 failures using bounded subbatches on the existing
  scheduled alias lease. Preserve successful vectors and occurrence indices.
- Return `EmbeddingChunkSplitRequiredError` containing successful original
  results and proper Unicode/UTF-8 midpoint halves for rejected singletons.
- Preserve ordered asynchronous delivery, owner cancellation, finite deadlines,
  identity/topology validation, backpressure and drained close.

The public error is recovery advice, not proof of token overflow or a promise
that one halving fits. Source splitting and repeated retry bounds remain with
the consumer. The [API](../AUTOMATIC-EMBEDDING.md) and
[consumer handoff](../HALVING-CONSUMER.md) provide signatures and examples.

## Corpus provenance and reproducibility

One-off curation measured the original 10,000 synthetic inputs on one verified
Nomic model using its document prefix and native special-token policy. It
selected the 500 lowest exact source-character/prepared-token ratios in each
category; original IDs break ties. This curation took 54.7243 seconds with two
bounded tokenizer workers and made no embedding requests. It is a source-build
selection step, not a repeated library startup operation.

The checked-in scalar receipt and generator reproduce the selected corpus
without a network. Reproduction matched the packaged manifest:

```text
corpus SHA256: 490a53bfba6120c246f9b4c4ec605681cd5ce02ed3d4c3111445be0a58b3cf31
receipt SHA256: 0f9370fb4dc6ec3306aa2563cabb44ead27ea02280c0950c73523ba15cbedd2f
```

The receipt is included in source distributions, outside the runtime wheel.
It contains source hashes and scalar counts, not consumer text or private
connection information. Difficulty ranking is specific to the measured Nomic
policy; it does not establish universal worst cases for other models.

## Deterministic verification

Full suite: **2,031 passed in 71.70 seconds**. Four warnings belong to explicit
unverified-TLS loopback fixtures. Ruff checking and formatting passed for all
111 checked Python files. Source and wheel builds passed; a clean environment
outside the checkout imported the wheel and verified version, exports and its
2,500 packaged records. All 38 runtime source/data files in the wheel match the
checkout bytes. The selection receipt is absent from the wheel.

Coverage includes partial results across shared owners, duplicate occurrences,
odd splits, one or many rejected rows, exact/direct shared requests, unknown
vector dimension with the first or all rows rejected, finite isolation limits,
identity/topology drift, cancellation during descendants, retained-result credit
and paused-stream backpressure. Additional regressions cover cancellation,
close and deadline expiration during off-loop public result conversion.
Sanitized marker causes retain no HTTP tracebacks or source-frame references.

`test_true_http_two_alias_connections_overlap_while_one_lane_isolates` uses real
local HTTP connections and a barrier. It proves two aliases overlap, one slot
per alias, and a healthy alias finishes while the other alias's failed singleton
is blocked. Successful child rows are not replayed. This is a scheduler and
transport proof, separate from a two-GPU live performance comparison.

## Installed-wheel live qualification

The operator had one Nomic v2 MoE Q8 model loaded: context 512, native batch and
microbatch 512, one slot, mean pooling and Flash Attention enabled. Verified
content identity:
`0292bc5c5aa4c434d195a9637e3bf19fa538177ef768b8c4d7154c49e0a63e98`.
Loaded identity, arguments, context and capacity were unchanged before/after the
completed continuation. No administrative or generation requests were sent.

Cold explicit calibration succeeded with exactly 2,500 tokenizer requests in
12.5469 seconds. Its HTTP intervals sum to 24.2999 seconds because two requests
can overlap; that sum is not elapsed time or GPU time. The recommendations were:

| Category | Weighted characters/token | Factor | Characters at loaded 512 |
| --- | ---: | ---: | ---: |
| code | 2.546552 | 80% | 1,043 |
| documentation | 3.485449 | 80% | 1,427 |
| prose | 3.946284 | 80% | 1,616 |
| structured | 1.810510 | 80% | 741 |
| multilingual | 1.922294 | 80% | 787 |
| mixed | 2.389943 | 90% | 1,101 |

The first probe reached successful isolation but stopped because its initial
harness assumed the first halves would fit. They were still oversized. The
library correctly returned further halving advice. The harness was corrected
to retain completed results and retry only rejected halves for at most eight
rounds. A separate warm continuation reused the existing profile with zero
additional corpus tokenizations; the cold evidence remains retained separately.

The historical 511-text queue workload was used in both runs (corpus digest
`a723a493887bf14096265c52bb1f09f2171e1a13efb198b8e10f572cad4d7252`).
The first run took 3.2882 seconds; the completed warm continuation took 2.5971
seconds. Both used two embedding POSTs, ten metadata GETs and zero tokenizer
requests. These are observations, not evidence of an optimization between runs.

Completed warm-continuation phases:

| Phase | Client wall seconds | Embedding POSTs | Tokenizer POSTs | Metadata GETs |
| --- | ---: | ---: | ---: | ---: |
| 511 ordinary inputs | 2.5971 | 2 | 0 | 10 |
| Isolate one rejected input among two valid inputs | 0.1927 | 5 | 0 | 16 |
| Consumer retries only halves, including a second halving round | 0.2698 | 4 | 0 | 20 |
| Independent good-vector references | 0.1297 | 1 | 0 | 8 |
| Explicit text-versus-token-ID numerical check | 0.1082 | 2 | 4 | 0 |

The two good original vectors were preserved. The first two halves were both
rejected; only their four halves were submitted next and completed. Within-owner
identical text may share preparation while retaining distinct result indices.
The saved mixed budget remained exactly 1,101, revision zero, before and after.
No fallback tokenizer call or learned-budget write occurred.

For 511 inputs, local preparation worker intervals summed to 0.00504 seconds;
HTTP intervals summed to 2.43695 seconds inside the 2.59707-second client wall.
These intervals include client processing, networking, server work and
instrumentation. Peak live embedding HTTP concurrency was one, consistent with
one single-slot replica. No conclusion about two-GPU speedup follows from this
profile.

Retained good vectors versus fresh good references, and four same-alias text
versus exact-token-ID checks, all had maximum absolute error zero. Their cosine
was at least 1.0 within floating-point rounding, and maximum norm error was
1.11e-16. Existing thresholds remain max absolute error `1e-5`, minimum cosine
`0.999999`, norm error `1e-6`. This single-alias result does not close the
historical cross-replica numerical finding.

## Local package

```text
runtime_llama-0.13.0a1-py3-none-any.whl
SHA256 c1cb36802579be6bc58624f657f87ad64d7d7305218b3d8bd022b9635c56ce3f
```

The wheel is the same installed artifact used in the live checks. Detailed
request receipts and clean-install records are retained in the private
qualification evidence referenced by Flower. Publication, consumer migration
and a one-versus-two-replica live performance comparison remain separate actions.
