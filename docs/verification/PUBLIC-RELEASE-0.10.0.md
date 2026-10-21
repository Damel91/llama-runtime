# Public release 0.10.0 verification

Date: 2026-10-21. Distribution: `runtime-llama`; import: `runtime_llama`.
Flower MCP governs the engineering lifecycle and retained evidence. This
publication follows the explicit release instruction after the completed
[persistent-engine qualification](PERSISTENT-QUEUES-0.10.0.md).

## Included behavior

The release includes asynchronous streams of embedding windows, progressive
exact preparation with native token-ID inputs, and one persistent execution
engine shared by synchronous and asynchronous calls. Verified remote replicas
have persistent ready queues; batches are composed at actual dispatch capacity
over the existing sole scheduler. Fixed worker pools overlap tokenizer,
control and embedding HTTP work. Cancellation, finite deadlines, order,
identity, topology boundaries and drained closure remain supported.

`batch_size` remains a total-text distribution budget. A window of 256 texts
on two usable replicas assigns quotas up to 128 each; four replicas imply
quotas up to 64. These are derived examples. Per-input context, explicit
aggregate HTTP token limits, native computation batches and JSON byte limits
remain separate. Prefixes and special tokens are counted once. Oversized
inputs remain measurable within resource ceilings without reaching inference.

The default coalescing delay is zero. Positive finite coalescing remains an
optional policy on the same engine; independent cancellation domains are not
silently fused into one POST. Separate work and async-output credit pools,
bounded transient responses and finite concurrency provide backpressure.
Their ceilings are not a single process RSS limit. Native HTTP requires no
keep-alive patch; connection reuse remains explicit.

Versions 0.8.0 and 0.9.0 were local qualification candidates. Their features
join the public distribution here; the preceding public release was 0.7.0.
The release preserves generation, structured tool calls, sampling controls,
model discovery and optional tailnet transport already offered by that release.

## Verification actually obtained

The final implementation passed **1,372 deterministic tests** on macOS ARM64
with Python 3.11.15, plus Ruff check/format, wheel/source builds and clean
installed-wheel checks outside the checkout. Four warnings are from deliberate
local self-signed TLS fixtures. No new Linux/Windows execution or GitHub CI
coverage is claimed.

Bounded live qualification used the two already loaded, content-verified
EmbeddingGemma 2 BF16 aliases reporting build `b11716-797de63b2`, context and
native batch/microbatch 2048, one slot each and Flash Attention off. Server,
model and consumer configuration was unchanged. Each run used 511 distinct
synthetic documents in async windows 256+255, string/native-ID canaries on
both aliases and an exactly measured oversized input rejected before inference.

All 511 public vectors in the final candidate runs matched the 0.9.0 baseline
with maximum coordinate difference **zero**. Both aliases were exercised,
embedding HTTP peaked at two overall and one per alias, ordering/cardinality
and raw-response projection passed, and profiles and identity stayed pinned.
The numerical thresholds remained `1e-5` maximum coordinate error,
`0.999999` minimum cosine, `1e-6` norm error and `1e-12` raw projection error.
See the qualification report for protocol amendments, incomplete historical
attempts, exact counts, hashes and limits.

**No throughput improvement is established.** The exploratory ABBA median
was 4.258 seconds for 0.9.0 with coalescing versus 6.114 seconds for 0.10.0
without delay. External LLM activity changed during the experiment; the
candidate also emitted more embedding POSTs. A single explicit-coalescing
control took 4.893 seconds and cannot identify a cause. Times include client,
instrumentation, scheduling, network and runtime work, not pure GPU time.
Numerical correctness and structural concurrency do not demonstrate a speedup
or universal model/hardware equivalence.

Cancellation, failures, metadata arbitration, byte/token limits, fair admission,
slow discovery, out-of-order result credits, topology changes and concurrent
closure are covered by deterministic fixtures. This release does not claim
consumer acceptance, retrieval-quality evaluation or live model interruption.

## Public artifacts and provenance

The GitHub release provides seven assets: Python wheel, normalized source
archive, optional helper archives for macOS ARM64/Linux AMD64/Windows AMD64,
`RELEASE-MANIFEST.json` and `SHA256SUMS`. The manifest binds the published
source commit and exact asset hashes. Asset downloads are checked against
those hashes after publication; consumers can perform the same verification.

Public packaging uses `SOURCE_DATE_EPOCH=315532800` and normalized archive
metadata with project and upstream license notices retained. Final archive
contents are checked against tracked source, and all Python runtime payload
files are byte-compared against the qualified local wheel with SHA-256
`717de0acfdd6f8175dd956f1acb4d2096f072db3ac0afcc64c4729d3b18874f5`.
That hash identifies the historical local qualification artifact. Public wheel
metadata and timestamps differ; use the public manifest for download hashes.
A fresh public-wheel installation verifies import/version consistency and
inert construction/closure without using checkout source.

The optional tsnet executable bytes, Go source and license inventory are
unchanged from the original 0.6.0 helper qualification and were also shipped in
0.7.0. The 0.10.0 helper archives explicitly retain that provenance. Rewrapping
does not claim new native builds, operating-system execution or real
Tailscale/Headscale enrollment. See [helper qualification](PUBLIC-RELEASE-0.6.0.md).

Public assets exclude private deployment configuration, credentials, local
evidence dumps and consumer data. Copyright Davide Mele and Apache-2.0 remain
unchanged, with the derived component's MIT attribution preserved.

Flower publication scope is `REQ-000027`, `MILE-000015`, `CHANGE-000015`,
`PACKET-000016`, `RUN-000010` and `BOOT-000014`. Prior implementation evidence
remains in `HAND-000014`. Publication does not install or migrate consumer
environments, restart services, load models or upload a PyPI distribution.
