# Automatic embedding hardening: 0.11.1

Qualification cycle: 2026-10-22/23, Europe/Rome. Flower authority:
`REQ-000039`, `GOAL-000021`, `MILE-000022`, `CHANGE-000024`, `PACKET-000025`
spec 5 / plan 2. Implementation commit:
`82b61a44248b3c8999e6a9451dbe90c9ac633457`.

The independently supplied v0.11.0 review was checked against the tagged source
and bounded offline reproductions. This cycle implements the confirmed boundary
fixes without changing consumers, model profiles, services, shared Python or
database contents. The persistent engine and scheduler are unchanged.

## Findings and resulting behavior

| Finding | Change and verification |
| --- | --- |
| F01: automatic transport/memory limits | Exposes existing finite request, decoded-response, pending-byte and pending-token limits with unchanged defaults. A real loopback HTTP fixture emits 128 normalized vectors of 4,096 dimensions: default 8 MiB rejects the valid large JSON; explicit 16 MiB returns ordered vectors. |
| F02: mixed recommendation | Keeps category 80% / mixed 90%. A mathematical regression and API documentation show that the aggregate recommendation is not conservative for every constituent category. No new statistical policy or tokenizer fallback. |
| F03: owned connection policy | Exposes `http_connection_policy`, default `close`, optional `reuse`. Tests verify connection headers and actual loopback socket reuse. Borrowed clients retain policy and ownership. |
| F04: callback reentrancy | Per-service callback ownership guards synchronous `close()` before locks or shutdown mutation. A dedicated `LLMBackendError` diagnostic survives sync/calibration/stream cancellation plumbing. External close and owner drain remain intact. |
| F05: high-context capacity | Exposes the existing pending-token budget. Tests demonstrate configurable cardinality while retaining full-allowance resource reservations, the existing scheduler, bounded counters and concurrent-owner progress. |
| F06: cold persistence feasibility | After complete-cache lookups, a private create/write/flush/fsync/delete preflight precedes remote tokenization. Valid readable caches require no write. Final validation and atomic replacement remain authoritative. |
| F07: statusless single-model catalog | Shares the existing root-props loaded-state attestation between the declared adapter and automatic facade. Automatic selection requires verified identity before probing; unsafe or inconsistent observations remain rejected. |

The v0.11.0 callback contract already required side-effect-free, thread-safe
cancellation probes. The reentrancy fix hardens misuse of that contract; it does
not make a callback an approved place to synchronously shut down the service.
Callers should set a cancellation event and drain externally.

## Deterministic and real HTTP evidence

The full suite passed **1,811 tests in 67.28 seconds**. Ruff check and formatting
passed. Four expected HTTPS warnings came from deliberately unverified loopback
transport fixtures. Normalized builds passed and the candidate was installed in
a fresh isolated environment outside the source checkout; dependency checks pass.

Meaningful additional regressions cover:

- Large valid responses above 8 MiB and explicit 16 MiB success, decoded gzip
  limits, incomplete plain/gzip bodies, typed failure, no POST replay, reusable
  healthy service after failure and complete resource drain.
- Owned `close`/`reuse`, actual socket reuse and borrowed client preservation.
- Isolated subprocesses for callback close before/after registration, during
  synchronous embedding HTTP, calibration workers, async readiness and active
  async HTTP; guard caught by the callback and subsequent healthy use. The
  formerly hanging paths terminate with the typed diagnostic; tests have bounded
  parent supervision rather than allowing a deadlocked test runner.
- Exactly one verified statusless model with coherent root props, explicit
  selector, router/sleeping/malformed/ambiguous/unverified cases and changed
  identity before embedding. These are read-only compatibility probes.
- Impossible parent/file/directory destinations, denied creation/write/fsync,
  cancellation/timeout cleanup, complete private cache publication, readable
  read-only warm cache and preservation of a late `LLMTimeoutError` instead of
  misclassifying it as a filesystem backend error.
- Existing corpus/profile identity and persisted-statistics compatibility,
  unchanged 80%/90% arithmetic, unchanged exact preparation contracts and the
  retained lifecycle/topology/fairness/backpressure tests.

The HTTP fixture is a deterministic server, not a GPU model benchmark. Numerical
validation still uses the existing tolerances; tests do not relax them.

## High-context queue observations

With one verified synthetic replica, 256 short strings, no tokenization and
`batch_size=256`:

| Observed context / microbatch | Pending reservation budget | HTTP packing |
| --- | ---: | --- |
| 65,536 / 65,536 | 1,048,576 (default) | 16 POSTs of 16 texts |
| 65,536 / 65,536 | 16,777,216 (explicit) | One POST of 256 texts |

Concurrent-owner tests retain progress, ordering, no duplicates and bounded,
drained byte/token counters. These observations qualify public configurability
without server reconfiguration. They do not claim live high-context throughput
or replace context reservations with statistical counts.

## Calibration compatibility and numerical policy

An impossible cold cache directory that is an existing regular file now fails
with `llamacpp_embedding_calibration_cache_write_failed` at **zero tokenizer
callbacks**; the original file remains intact. The previous offline reproduction
performed all 10,000 counts before that predictable failure. The same real corpus
is used in both observations; no model/server changes are involved.

The retained complete v0.11.0 calibration was loaded with zero new tokenizations
and unchanged file bytes. At its observed 512-token Nomic allowance, code remains
1,127 characters and mixed remains 1,509. Applying the mixed recommendation at
the category's mean code density would imply approximately 548 tokens. This is
illustrative arithmetic, not the measured token count of arbitrary code.

Corpus SHA-256 remains
`6a4733cf1d175b8b5b254f5951d19c5e4e5f5db2da8eaa1e15904474c53d3035`:
10,000 original texts, 2,000 per category. Statistics schema, identity keys,
prefix/special-token semantics and factors are unchanged. Ordinary direct
inference still never reads this cache or starts calibration.

## Installed-wheel live qualification

The isolated 0.11.1 wheel was exercised against the two already loaded verified
Nomic replicas, dimension 768, context/batch/microbatch 512, one slot each, mean
pooling and flash attention enabled. Native build remained `b11716-797de63b2`.
No server configuration, model loading, LLM generation or consumer job was used.
The synthetic 511-text fixture and numerical thresholds are the same as the
[0.11.0 qualification](AUTOMATIC-EMBEDDING-0.11.0a1.md).

| Phase | Wall seconds | Tokenizer POSTs | Embedding POSTs | Metadata GETs | Total HTTP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold initialize and status | 0.0725 | 0 | 0 | 10 | 10 |
| Direct 511, before calibration | 1.5078 | 0 | 4 | 16 | 20 |
| Explicit cold character budget | 53.8941 | 10,000 | 0 | 7 | 10,007 |
| Six category selections | 0.1886 | 0 | 0 | 24 | 24 |
| New service initialize and status | 0.0851 | 0 | 0 | 10 | 10 |
| New service direct three inputs | 0.1184 | 0 | 2 | 11 | 13 |
| Explicit warm character budget | 0.1031 | 0 | 0 | 4 | 4 |
| Exact prepare and embed 511 | 3.7297 | 511 | 4 | 15 | 530 |

The direct pass preceded calibration, left its nonexistent cache unchanged and
made no tokenizer calls. Both 511-input paths sent two embedding POSTs to each
replica and reached two simultaneous HTTP embedding requests. All 511 vectors
were identical: maximum absolute difference 0, minimum cosine 1, normalization
error at most 2.22e-16. Thresholds remain 1e-5 maximum absolute difference,
0.999999 minimum cosine and 1e-6 normalization error. All six character budgets
matched the earlier calibration, including code 1,127 and mixed 1,509.

These are end-to-end client wall times including applicable metadata,
preparation, transport and server work. They are not pure GPU timings or a
controlled speed comparison between releases. High-context and high-dimensional
boundary tests above are deterministic fixtures, not reconfigured live models.
Live VPN/TLS performance is not newly qualified by this plain-HTTP test.

## Package and publication boundary

The candidate is distribution `runtime-llama` 0.11.1, import `runtime_llama`.
The public wheel must match all 36 live-qualified runtime payload files exactly;
release metadata and normalized archive wrapping can change outer hashes.
Source assets, corpus, original generator, licenses and retained optional-helper
provenance are checked separately. See the
[public release record](PUBLIC-RELEASE-0.11.1.md).
