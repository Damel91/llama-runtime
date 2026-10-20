# Local native runtime candidate 0.7.0

Date: 2026-10-20. Source implementation commit:
`33fc556056a84781fdca0071513975b1cac10d54`. This is a local candidate, not a
published release or consumer acceptance. Flower project `llama-runtime`,
BOOT-000008 / CHANGE-000009 / PACKET-000010 governs the work; requirements are
REQ-000011–REQ-000014. Code and documentation have separate commits.

## Implemented behavior

Native `n_batch` and `n_ubatch` are computation observations, not the sum-token
limit of an HTTP array. Only configured `batch_token_limit` values impose that
client limit. `batch_size` remains a total-text window: 256 texts across two
usable replicas gives quotas of 128; four gives quotas of 64. Explicit HTTP
budgets can split a quota further. In a deterministic 32-token/input fixture,
256 texts now require two POSTs of 128 rather than the four requests caused by
the former 2048-token-derived aggregate cap. This is a request-count result,
not a measured speedup.

A bounded preparation pipeline overlaps active requests. The single scheduler
admits ready batches without blocking a free replica behind a busy alias.
Observed slots bound concurrency together with configured ceilings; unknown
slots fall back to one. Exact token counts, once-only prefixes/specials,
identity/build validation, order, partial failure, cancellation, deadlines and
drained shutdown remain enforced. Internal records avoid unnecessary public
seals. Native timing snapshots and generation bookkeeping optimizations retain
usage, reasoning, tool snapshots and repetition decisions.

Owned HTTP requests default to `Connection: close`, with owned transport pools
cleared after cleanup. This guarantees fresh connections even when a peer
ignores the header or delays FIN, without retry. `reuse` is an explicit option.
Borrowed sessions retain owner-controlled pooling. See
[native HTTP](../NATIVE-HTTP.md), [embedding contracts](../EMBEDDING-CONTRACTS.md)
and [consumer examples](../INTEGRATION.md).

## Deterministic and packaging verification

| Check | Actual result |
| --- | --- |
| Full Python suite, macOS/Python 3.11.15 | **1164 passed**, 49.37 s; no skips |
| Minimum Requests 2.31.0 / urllib3 1.26.20 transport matrix | **67 passed**, 5.40 s |
| Ruff check and format | Passed, 68 Python files |
| Repeated real HTTP four-replica regression | 12 runs passed, without retry or fixture changes |
| PEP 517 wheel and source archive | Built with `--no-isolation` and fixed `SOURCE_DATE_EPOCH` |
| Clean installed wheel outside checkout | Version, API, inert factories, default/explicit connection policy and timing validation passed |
| Installed/wheel/source runtime payloads | Every packaged runtime file has the same SHA-256 |

The four suite warnings come from deliberately unverified loopback TLS cases,
not a production TLS default change. Production TLS verification remains on.
The current full-suite HTTP dependencies were Requests 2.33.1 and urllib3 2.6.3.
The clean environment received existing HTTP dependency payloads offline and
installed only the local wheel; shared environments were not upgraded.

Meaningful regressions cover native batch versus explicit HTTP budgets,
multiple observed slots, concurrent callers, odd quotas, fewer texts than
replicas, bounded global lookahead, ready work behind busy aliases, queued
metadata changes, partial failures without replay, adaptive sibling validation,
reduced context without rejecting completed inputs, exact count reuse,
executor-close races, JSON/SSE framing, cancellation and drained close.

## Transport and live boundary

A model-free harness uses the original, unpatched native llama.cpp HTTP handler.
The integrated SDK checks exercise independent connections, concurrent native
handlers, recovery and drain: 10 GET / 10 HTTP 200 / 10 distinct sockets,
native concurrent peak two, retry zero, no active work after close and native
process exit zero. Its results JSON SHA-256 is
`eedb8c77023efafbbc74bc5d2e58358739206f30cdd7bb5bf4142732a13c3894`.
This is HTTP evidence; it is not inference,
TLS/VPN, GPU throughput or proof of the remote binary's patch status. The remote
build string alone cannot establish whether the historical patch is present.

Read-only discovery on the running remote server found two embedding replicas
with verified equal content/semantic identity, effective context 2048,
`n_batch=2048`, `n_ubatch=2048` and one slot each, on separate GPU devices. No
model, profile, router, service, consumer database or consumer job was changed.

Direct native calls accepted two concurrent POSTs, each containing four valid
919-token inputs: **3676 aggregate tokens per request**, above `n_batch=2048`.
Both returned HTTP 200 and 768-dimensional vectors in the expected order. Four
sampled comparisons between an array and a single input on the same replica
were identical (`maxabs=0`). These results support the corrected HTTP contract.

The 0.7.0 SDK source then processed eight inputs with automatic discovery:
two embedding POSTs of four texts each, scheduler peak two, one active request
per alias, and zero active/queued work after drain. Its full call used 23 HTTP
requests and 23 distinct sockets: health 1, catalog 6, props 6, tokenization 8
and embeddings 2. Preparation took 0.090/0.100 s per quota; whole-call wall time
was 0.532 s. Embedding HTTP intervals overlapped. These short measurements are
not a controlled throughput comparison with the baseline.

Projection against independently normalized vectors from each exact same
response passed: maximum error `5.56e-17` against `1e-12`, norm error at most
`1.12e-16`. Raw SDK responses matched direct native responses on the same alias.
An initial probe assessor referenced the scheduler replaced by discovery; its
offline correction used recorded offsets/raw payloads without more inference.
No failing inference or numerical result was retried. The final runtime payload
is byte-identical to the clean-installed wheel; this live invocation imported
the checkout source, not the wheel. Final SDK assessment SHA-256:
`10d7299a4c916339a7620745ccf3ec183eb8c4ac959a2cb1b2a8fd45334a2c37`.

**Cross-replica numerical qualification failed at the existing thresholds.**
Direct native responses differed by up to `3.9824098348617554e-4`; minimum
cosine was `0.9999959660631729`. Required limits remain `maxabs <= 1e-5` and
`cosine >= 0.999999`. Native norms were within `4.88e-8` of one, below the
`1e-6` norm limit. The same-request public projection limit remains `1e-12`.
No threshold was relaxed, no replica was silently disabled and no failing
result was promoted to a pass. This deployment needs a separate runtime
investigation before declaring the two GPU replicas numerically equivalent.

Live wall times include scheduling, preparation, HTTP and server processing;
they are not pure GPU time. This work does not claim full network saturation,
a throughput improvement or a new complete ingest/retrieval qualification.

## Local delivery

The Python distribution remains `runtime-llama`; import remains `runtime_llama`.
Version: **0.7.0**. The local wheel is
`runtime_llama-0.7.0-py3-none-any.whl`, SHA-256:

`c97cb9c5e0ab56cfeb2b959706759a8e7ef9072452d4842e0da24e7f43c5c0b2`

The accompanying local manifest binds the final source/documentation commits,
wheel, source archive, runtime payload and private evidence hashes. Full
endpoint metadata and deployment locators remain outside the public source.
No consumer installation, publication, push or shared-environment upgrade is
part of this delivery. The optional Go helper was unchanged and was not rebuilt
or newly qualified through a real VPN in this cycle.
