# Automatic stream admission: local 0.13.1.dev1 candidate

Date: 2026-10-24. Flower authority: `REQ-000044`, `GOAL-000026`,
`MILE-000027`, `BOOT-000027`, `CHANGE-000029`, `PACKET-000030` spec 3 / plan 1.
Baseline: `95b1443bab86b3174f1ef1cbeb0d060f0379ec2a` (`0.13.0a1`).

## Diagnosis and scope

A bounded diagnostic profile of 2,048 frozen production-source texts found
67 metadata GETs and 16 full embedding POSTs of 128 texts across two loaded
replicas. Stream readiness and eight window admissions accounted for 36 GETs:
nine health requests, nine catalogs and eighteen per-replica properties.
Dispatch accounted for another fifteen catalogs and sixteen properties.
Metadata HTTP durations summed to 4.115 seconds across overlapping requests;
this is not an additive wall-time cost.

The profile observed 2.013 replica-seconds with an idle alias and at least 128
ready records. Of these, 2.001 had metadata outstanding. Instrumented wall time
was 11.473 seconds and is diagnostic evidence, not the acceptance benchmark.
CPU measurements did not support treating these particular gaps as an expensive
packing loop or a GIL bottleneck. Output validation and normalization remain.

A fresh uninstrumented baseline used the same corpus and settings, with three
repetitions: SDK median 11.899 seconds and exact native replay median 9.386
seconds. Timing includes network, router, serialization and local work. Raw
production texts and deployment configuration stay outside this repository.

## Implementation contract

The automatic stream captures a private, non-owning admission scope during
verified readiness. Subsequent direct windows can fork independent call
references only for the same adapter, scheduler, full discovered pool and
admission generation. The scope does not hold a pool reference while a stream
is idle and is cleared after initialization, delegate and conversion work drain.
Exact, synchronous and unscoped calls retain ordinary admission.

Every original POST still performs a fresh catalog and selected-alias properties
preflight. Recovery retains fresh observations. Full matching-pool additions,
removals, alias changes, malformed catalogs and identity drift revoke reuse;
old owners drain before the scheduler/pool can be replaced. Only an authoritative
fresh join can enable a generation. Saved observations never restore slot
capacities and do not authorize HTTP requests.

There is no TTL, speculative preflight, second scheduler, hidden retry, larger
batch or public API change. Total batch budget 256, numerical thresholds,
per-owner cancellation, finite deadlines, ordering, result credits, backpressure
and drained close are preserved. The explicit 80% category / 90% mixed
recommendations and fixed inference-recovery statistics are unchanged.

## Verification and handoff

Deterministic checks cover request counts across eight windows, independently
advancing aliases, late loaded-state rejection, topology/generation races,
mixed scoped/unscoped owners, reference accounting, cancellation and idle drain.

| Check | Result |
| --- | --- |
| Full deterministic/loopback suite | 2,053 passed in 70.87 s; four intentional loopback TLS warnings |
| New regressions | 15 adapter/generation checks and 7 automatic stream checks |
| Ruff lint and format | Passed; 114 Python files formatted |
| Build without isolation | Source archive and pure Python wheel built |
| Clean installed wheel outside checkout | Public eight-window stream passed; 16 full POSTs, one readiness health, fresh properties retained |
| Dependency check | Passed |
| Wheel/source content | All 40 runtime payload files match source |
| Preservation | All 39 installed reference payload files and frozen corpus file hash unchanged |

Code commit: `d6223a8cf547881a9a2a063848df6d0b92c1a5ad`.
Candidate: `runtime_llama-0.13.1.dev1-py3-none-any.whl`.
SHA-256: `8eacd0631bddd27503d435a4a5b0b3c6c9aaa66f35efc7dc44d084644a31253a`.

The exact wheel and hash were handed to the authorized consumer-side benchmark
owner. Live measurements are separate from the deterministic checks above.

## Bounded live comparison: one replica per GPU

Three uninstrumented repetitions per version used the same 2,048 texts,
batch budget 256, request parallelism 4, eight source windows and two buffered
windows. Both versions made sixteen full POSTs of 128 texts. The reference
made 67 metadata GETs; the candidate made 32 in every repetition, a 52.2%
reduction. Both aliases received 1,024 texts and the same aggregate token totals.

| Median client wall time | Installed 0.13.0a1 | Candidate 0.13.1.dev1 |
| --- | ---: | ---: |
| SDK | 11.899 s | 13.099 s |
| Exact native replay | 9.386 s | 10.330 s |
| SDK / native ratio | 1.267731 | 1.268063 |

The normalized ratio is effectively unchanged (+0.026%); **no speedup is
demonstrated**. Before/after were separate runs, and the native control also
slowed by approximately 10%. Concurrent window admission changes which window
records share a batch. Each exact native replay follows its own SDK round's
payloads and alias grouping; cross-version SDK batch plans are not identical.
The alternative fixed native plan remained identical. These exploratory ratios
are not confidence intervals or a causal decomposition of elapsed time.

Within-run vectors retained minimum cosine 0.9999999403953552; all ordering,
cardinality and finite-value checks passed. This does not close the separate
cross-physical-replica finding. Consumer process, database/configuration and
frozen corpus preservation checks passed.

A separate instrumented candidate profile measured one full admission join
instead of nine, with 33 GETs (one additional dispatch catalog from timing-
dependent coalescing). Free aliases with a full ready batch still waited 2.009
replica-seconds on metadata, compared with 2.013 in the reference. Another
2.412 sampled replica-seconds showed an empty alias while the other was busy.
Window completion/backpressure skew is a hypothesis requiring deterministic
reproduction; these samples alone do not establish its cause.

Interleaved read-only control measurements put both native and SDK metadata
requests near 60 ms. This is a common network/router/transport cost, not evidence
of an SDK polling delay. Packing CPU and post-response validation were not the
dominant measured full-ready wait. The candidate therefore reduces server
control load while leaving an important latency bottleneck unresolved.

Single-replica and shared-GPU repetitions are handled separately with the same
immutable candidate after operator placement changes. No further performance
claim, publication or shared installation follows from this stage.

The candidate is installed only in a disposable verification environment and
an isolated benchmark overlay. The shared installed `0.13.0a1`, consumers,
consumer databases/jobs, model placement and server configuration remain intact.
No publication is authorized by this cycle.

Fresh preflight remains after alias capacity becomes available, so some observed
dispatch gaps can remain. Removing redundant window admission does not prove
that all scheduling loss is eliminated. Subsequent benchmarks must use the same
corpus, payload grouping, settings and numerical checks, with profiling measured
separately. Additional physical layouts require the operator to change placement.

SDK correctness/stability and model-profile numerical qualification are separate
judgments. The existing cross-physical numerical finding remains open; it does
not by itself determine whether this library must be classified as a prerelease.
This `.dev1` artifact is an isolated development candidate, not a public release.
