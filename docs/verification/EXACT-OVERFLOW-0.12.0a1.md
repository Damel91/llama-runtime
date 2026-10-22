# Exact overflow recovery: local 0.12.0a1 candidate

Status: **deterministic and functional checks passed; full live numerical
qualification remains blocked**. No new stable release or consumer deployment
is authorized by this record. Published 0.11.1 is unchanged.

Flower authority: `REQ-000040`, `GOAL-000022`, `MILE-000023`, `BOOT-000023`,
`CHANGE-000025`, `PACKET-000026` specification 4 / plan 2. Open finding
`FIND-000007` records the live cross-replica numerical difference. Implementation,
host-reported verification and stakeholder acceptance remain separate.

## Contract and implementation

- Initial recommendation uses the global worst observed source-character/token
  ratio in the existing 10,000-sample raw profile. Category 80% and mixed 90%
  factors, corpus bytes and raw cache identity remain unchanged.
- Successful direct requests use the existing engine/scheduler without tokenizer
  calls. Actual failed-POST provenance preserves owner/window/input occurrences
  through duplicate records, subbatches and owner withdrawal.
- After a direct HTTP 500 drains, bounded exact diagnostics inspect only that
  owner's actual failed POST rows. No proven overflow preserves the original
  error and store. Confirmed overflow returns proper lossless Unicode/UTF-8
  spans with exact native counts and a typed public correction.
- Absolute learned ceilings use locked atomic min/revision persistence with no
  second safety factor. No texts, token arrays, endpoints or fragment plans are
  retained in the learned state. Changed effective context uses a distinct scope.
- `refresh_character_budget()` reads valid completed state after fresh model
  observation and never starts calibration. Learned-only snapshots can have
  nullable calibration identity/initial budget.
- No automatic embedding replay, truncation, splitting or changed vector
  cardinality. Consumer reprojection/publication remains a separate lifecycle.

## Deterministic and packaging checks

The complete suite passed **1,963 tests in 68.59 seconds**. The four warnings
are the existing deliberately unverified HTTPS loopback fixtures. Ruff check,
format check and source diff checks passed. Source/wheel builds, clean
out-of-checkout installation, import/version/API smoke and dependency checks
passed. The candidate has no dependency on a consumer or model SDK.

Coverage includes concurrent thread/process scalar reductions, real held-lock
cancellation, corrupt-state refusal, warm/read-only cache reuse, private atomic
cleanup, prefix/special-token counts, nonmonotonic token-count probes, duplicate
occurrences, exact source coverage, actual subbatch ownership, sync/async
deadlines, cancellation/drain and second-window error provenance. Replica limit
changes hidden by an unchanged sibling minimum are refused before publication.
Unavailable/malformed diagnostic tokenizers preserve the original inference 500.

## Bounded installed-wheel live checks

The observed native Nomic pair had context/native batch/microbatch 512, one
slot per replica and build `b11716-797de63b2`. No configuration, process, model,
consumer, database or job was changed. All reported times are client wall time,
including the applicable metadata, preparation, HTTP and inference, **not pure
GPU time**. Existing thresholds were retained: max absolute error `1e-5`,
minimum cosine `0.999999`, unit-norm error `1e-6`.

| Phase | Wall seconds | Embedding POSTs | Tokenizer POSTs | Metadata GETs |
| --- | ---: | ---: | ---: | ---: |
| Direct 511 synthetic texts | 1.5471 | 4 | 0 | 16 |
| Exact path, same 511 | 3.8400 | 4 | 511 | 15 |
| Failed original 256-input window and diagnosis | 1.8203 | 2 | 130 | 19 |
| Explicit retry with 257 source fragments | 0.8226 | 3 | 0 | 14 |
| Exact comparison for 257 fragments | 1.9700 | 3 | 256 | 14 |

Both replicas were used with peak embedding HTTP concurrency 2. The direct and
exact 511 vectors were identical: max absolute difference 0, cosine 1 and norm
error at floating-point roundoff. Warm budget observation used no tokenizer
requests and reused the preceding raw calibration unchanged: the global minimum
1.2115384615384615 gives mixed 558 and single-category 496 characters.

The original failing source occurrence was recovered from a hash-verified spool:
1176 source characters, 531 prepared tokens against loaded context 512. The
failed POST contained original indices 128 through 255 of the consumer window;
only those 128 rows and two new fragments were counted, for 130 tokenizer POSTs.
The correction identified original window index 189 and returned exact spans
`[0,1133)` / `[1133,1176)`, with respective counts 512 and 25. These ASCII
source boundaries are identical in characters and UTF-8 bytes.

The persisted mixed ceiling remained 558 because it was already smaller than
the largest verified fragment; revision 1 records the absolute learned bound.
The proper partition provides retry progress without an artificial decrement.
An explicit consumer-shaped retry produced 257 vectors without tokenizer calls.
Native single-input controls rejected the original input with HTTP 500 on both
replicas; removing only that occurrence yielded 255 vectors. A new service
reloaded the same learned snapshot successfully without calibration.

## Open numerical result

The comparison on the real 257-input retry **did not pass** the unchanged
cross-replica thresholds: maximum absolute difference
`0.0024030989764955157`, minimum cosine `0.9998033728657948`.
Five inputs of 391, 512, 385, 446 and 467 prepared tokens failed that comparison.
This result prevents declaring full live qualification.

Bounded single-input isolation repeated the five affected inputs on each
replica. On **each individual replica**, strings and their exact native token IDs
gave identical vectors (max absolute difference 0). Single-input vectors agreed
with the corresponding same-replica batch result within `2.8e-17`. The observed
difference follows the replica, rather than text-versus-token transport or the
library's fragment reconstruction. This identifies an external numerical
condition; it does not by itself identify its hardware/kernel root cause.

The scheduler, pool size and thresholds were not altered to hide the difference.
An operator/stakeholder must resolve the runtime condition or explicitly decide
the intended numerical contract before complete qualification and consumer
deployment can proceed. No broader performance or retrieval-quality claim is
made from these bounded checks.

Raw source payloads, internal endpoints, vectors and detailed HTTP events remain
in private qualification artifacts. The local wheel/manifest carries hashes,
source provenance and separate functional/numerical statuses; it is not a new
public release or a claim of application acceptance.
