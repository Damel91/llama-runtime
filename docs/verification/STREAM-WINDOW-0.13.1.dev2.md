# Consumer-fair packing: isolated 0.13.1.dev2 candidate

Date: 2026-10-24. Flower: `REQ-000045`, `GOAL-000027`, `MILE-000028`,
`BOOT-000028`, `CHANGE-000030`, `PACKET-000031` spec 3 / plan 1.
This separate correction builds on the [admission candidate](STREAM-CONTROL-0.13.1.dev1.md);
its existing wheel remains immutable.

## Causally reproduced window head-of-line

A private deterministic barrier probe used eight source windows of 256 texts,
two replicas, total batch budget 256, request parallelism 4 and two pending
windows. The initial two windows were fully ready. Each alias could send 128
texts per POST. The first slow-alias POST was held until the fast alias had
completed two batches; then that slow POST was released and its second POST held.

The existing per-window round-robin packer mixed 64 records from each of the two
windows in every 128-row POST. At the second barrier both windows still needed
64 slow-alias results. Neither could be delivered, so the producer had pulled
only two windows and the fast alias had no further input.

A proof-only oldest-window packing order put all 128 records of window zero in
the first POST. At the same barrier window zero was already delivered, the source
had pulled window two and the fast alias had sent its third batch. Both variants
delivered all eight windows correctly in sixteen full POSTs. Peak plans (4),
ready bytes (499,200) and result reservation (90,240) were identical. Five repeats
per variant preserved these results. These are synthetic data and deterministic
barriers, not a timing benchmark.

The binding limit was `max_pending_windows`, rather than exhaustion of the
result-byte budget. A global oldest-call preference was used only to prove
causality; it is not a sufficient production fairness policy.

## Intended correction and invariants

Each async stream has a private packing identity independent of admission
metadata and per-window result ownership. Packing rotates between independent
consumers; within one stream it drains the oldest ready window first. Empty
queues do not prevent later ready windows from filling a POST. Small windows can
still share a model-wide batch; there is no mandatory window fragmentation.
Synchronous calls remain independent consumers. The per-alias consumer turn
advances only after actual dispatch, not during candidate inspection or metadata.
Retired consumer/alias bookkeeping must remain bounded.

Readiness and real packing use the same grouped ordering, including native-ID
prefix and fallback-to-text accounting. One engine and scheduler remain in use.
No timer, increased prefetch, larger budget, changed public signature or numerical
threshold is introduced. Fresh observations, content identity, topology handling,
per-window cancellation/failures, finite deadlines, ordered delivery, result
credits and drained close remain mandatory.

The fixed 80% category / 90% mixed character recommendations are unchanged.
Models, consumers, services and the shared installed SDK remain untouched.
The candidate is for isolated verification; no publication is authorized.

## Verification boundary

| Check | Result |
| --- | --- |
| Complete suite | 2,070 passed in 70.95 s; four intentional loopback TLS warnings |
| New regressions | 9 packing/causal/fairness cases and 8 stream-identity cases |
| Focused engine/stream run | 112 passed |
| Independent peer review | No actionable findings; additional focused cases passed |
| Ruff lint/format | Passed; 117 Python files formatted |
| Build | Source archive and pure Python wheel built without isolation |
| Clean final-wheel installation | Eight-window public stream smoke and dependency check passed |
| Preservation | 41 wheel runtime payloads match source; 39 shared reference payloads and corpus file unchanged |

The existing explicit-token-budget test now uses unequal window lengths
`1, 3, 1, 3`, keeping its assertions for four full two-row POSTs, correct vectors
and cross-window fill. Equal two-row windows would already fit each two-row POST
and no longer require mixing to exercise that contract. A preflight unit fixture
that replaces readiness and packing explicitly initializes their consumer roster.
Its first full run failed only that missing fixture setup; all 2,070 tests passed
after the setup correction, without a production fallback or removed assertion.

Code commit: `4ad8c22111946bfb8236b474a74704323e1e79ca`.
Wheel: `runtime_llama-0.13.1.dev2-py3-none-any.whl`.
SHA-256: `c889feda2649ce8b9206f589923c702676d9b7c048538df7fe3a46baceb3b378`.
The exact final wheel was handed to the authorized benchmark owner for a new
isolated overlay. The dev1 wheel hash remains unchanged.

Removing this avoidable fast-lane idle does not guarantee a lower total wall
time when the slow replica still dominates the work. Full deterministic tests,
package checks and an exact wheel/hash handoff precede any live performance
claim. The consumer benchmark owner keeps the same corpus and exact native
replays; physical layout changes remain the operator's responsibility.
Performance output summaries must not be called complete numerical qualification
unless both existing tolerances were actually checked. Warmup cosine, exact
native replay and cross-physical loading are distinct comparisons.

## Bounded live qualification: two replicas on one GPU

The initial three-repetition candidate run made sixteen full POSTs of 128 texts
and 35–36 metadata GETs over the same 2,048 texts. The SDK median was 15.487 s
and exact native replay 10.703 s. Its per-alias input grouping returned to the
reference plan. That separated comparison did not pass the latency gate.

A separate diagnostic profile confirmed the local scheduling correction:
inter-request gaps totalled 2.247 replica-seconds, versus 4.135 for dev1 on the
same layout. All fourteen gaps were between 0.152 and 0.211 seconds; the larger
0.5–0.9 second extra gaps disappeared. Full-ready waiting on fresh metadata still
totalled 1.953 replica-seconds. Instrumented timing is not acceptance timing.

An additional sequential AB/BA/AB check alternated the installed reference and
the exact candidate on the unchanged layout. Each child performed one SDK run
and its exact native replay after warmup:

| Pair | Reference SDK | Candidate SDK | Reference native | Candidate native |
| --- | ---: | ---: | ---: | ---: |
| AB | 11.292 s | 12.004 s | 12.057 s | 12.671 s |
| BA | 14.549 s | 12.312 s | 10.163 s | 9.660 s |
| AB | 11.865 s | 10.631 s | 9.322 s | 9.713 s |

SDK medians were 11.865 s and 12.004 s (+1.17%). The ratio of SDK/native medians
and the median of individual SDK/native ratios point in different directions.
There is no robust end-to-end speedup direction in these few variable runs;
the earlier separated comparison also cannot establish that its approximately
30% raw difference was caused entirely by the patch. Metadata reduction and
removal of the locally reproduced head-of-line are the verified improvements.

All twelve paired SDK/native measurements checked both existing tolerances
outside the timing interval: maximum absolute difference against the within-run
warmup was at most `1.4901161193847656e-8` (limit `1e-5`), and minimum cosine
was `0.9999999403953552` (minimum `0.999999`). They retained 2,048 finite
768-dimensional vectors, sixteen full POSTs and no HTTP errors. The reference
used 67 GETs and the candidate 36 in this alternating check. These checks do
not close the cross-physical loading finding.

Consumer process/database/configuration and corpus preservation checks passed.
The split-GPU and single-replica repetitions were subsequently completed below.
No additional performance fix, publication, shared installation or consumer
migration follows from these data.

## Bounded live qualification: one replica per GPU

The operator then supplied the split-GPU layout. Three repetitions of the same
2,048-text corpus covered SDK, exact native replay, fixed native plan and SDK
transport-only paths. All twelve measurements passed the existing `1e-5`
maximum absolute error and `0.999999` minimum cosine checks against their
within-run warmup. The measured extremes were `1.4901161193847656e-8` and
`0.9999999403953552`, respectively. All results retained 2,048 finite
768-dimensional vectors, with no HTTP errors.

| Median client wall time | Reference 0.13.0a1 | Candidate 0.13.1.dev2 |
| --- | ---: | ---: |
| SDK | 11.899 s | 12.768 s |
| Exact native replay | 9.386 s | 9.148 s |

Candidate SDK repetitions took 13.185, 12.768 and 10.395 seconds. SDK time was
7.30% higher in this separated comparison; the SDK/native ratio was 10.10%
higher. These few non-interleaved repetitions do not establish a stable speedup
or causal regression. The fixed native median was 9.005 s and SDK transport-only
median 9.687 s; these are distinct paths, not additive component timings.

Input grouping matched the reference exactly: eight POSTs of 128 texts per alias
with the same token totals. Metadata GETs fell from 67 to 36, 35 and 35. The
separate diagnostic profile retained 35 GETs and sixteen POSTs, with fourteen
inter-request gaps of approximately 0.152–0.178 s and total gaps of 2.215
replica-seconds. The large window head-of-line gaps remained absent; full-ready
waiting on fresh metadata still totalled 1.883 replica-seconds.

Configuration, database version, corpus and stopped-consumer checks passed;
the installed reference and immutable candidate artifacts remain intact.
The final single-replica result follows below. End-to-end latency qualification
and the separate cross-physical numerical finding remain open. No runtime code
or numerical threshold changed for these measurements.

## Final single-replica result and campaign completion

The final layout used one loaded replica on GPU0. Three repetitions across the
same four routes produced twelve valid measurements. Every SDK run sent eight
full POSTs of 256 texts, retained the original input plan and all 262,482 prompt
tokens, and used 19 metadata GETs instead of the reference's 43. There were no
HTTP errors. Both existing numerical thresholds passed for all twelve records;
the maximum absolute difference was `1.4901161193847656e-8` and minimum cosine
`0.9999999403953552`, against within-run warmup.

| Median client wall time | Fresh reference 0.13.0a1 | Candidate 0.13.1.dev2 |
| --- | ---: | ---: |
| SDK | 18.400 s | 15.932 s |
| Exact native replay | 13.999 s | 14.145 s |

Candidate SDK repetitions took 15.932, 16.510 and 15.712 seconds. The observed
SDK change was −13.41%, with exact native +1.04% and SDK/native ratio −14.30%.
This does not establish general acceleration: the earlier historical single
SDK median was already 15.929 s. Fixed native and SDK transport-only medians
were 15.810 s and 14.678 s. All timings remain client wall time.

All three candidate layouts and the separate alternating shared-GPU control are
now complete. The consolidated receipt covers 120 measured routes across
reference, dev1, dev2 and alternation, with hashes of the original evidence
files. Exactly 36 records retain maximum absolute error plus both numerical
assertions; all 36 passed. The older 84 records did not collect that metric and
are not retrospectively qualified against it.

The verified producer improvements are reduced admission/control I/O and the
removal of the causally reproduced same-stream window head-of-line. Stable
end-to-end acceleration remains unproved, and the cross-physical finding stays
open. Evidence collection for this campaign is complete with these residuals;
it does not grant milestone acceptance, release promotion or consumer migration.
Corpus, configuration and database preservation checks passed, CC remained
stopped and the shared SDK remains `0.13.0a1`. No further model-layout change,
inference, code change, publication or shared installation is part of this
campaign.

Flower retains deterministic evidence 218, live evidence 220–222 and final
consolidation evidence 223 for `REQ-000045`. All four units of `PACKET-000031`
are complete with no deferred unit verification. Its final gate remains
`agent_work_complete_acceptance_pending`; human acceptance is not inferred.
