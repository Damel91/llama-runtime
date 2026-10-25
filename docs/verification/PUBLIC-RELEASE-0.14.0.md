# Public release 0.14.0 verification

Date: 2026-10-25, Europe/Rome. Distribution `runtime-llama`, import
`runtime_llama`. The operator explicitly authorized stable publication with the
observed numerical deviation recorded as a known limitation of the measured
server/model profile. Existing numerical thresholds and category 80% / mixed
90% calibration margins remain unchanged. Consumer code, pins, configuration,
installations, databases, jobs and services are outside this release operation.
Model loading, runtime settings and server processes were not changed.

Flower MCP remains the canonical lifecycle authority for the `llama-runtime`
project. This document is the public projection of the release evidence and
decision; host-reported execution does not imply application acceptance.

## Native client changes

- A verified canonical loaded replica receives one original embedding HTTP
  lease, retained through bounded serial isolation descendants and real drain.
  Content and semantic identity are verified; aliases do not create duplicate
  capacity. An isolated unverified single alias remains conservative.
- Automatic request parallelism uses the number N of usable replicas. A
  positive global guard G caps active original work at `min(N, G)`. Native
  slots, computation batch and microbatch remain observations, not extra HTTP
  permits. There is one production scheduler for synchronous and async calls.
- `batch_size` remains a total text budget distributed using N, independently
  of G. With budget 256 and four replicas, quotas contain up to 64 texts each;
  input/context, payload or explicit aggregate-token policies may require more
  than one POST per quota. Ordering, duplicate occurrences, cancellation,
  bounded topology replacement and owner isolation are preserved.
- Fixed `embedding_max_pending_owners` admission precedes sync snapshot and
  async source pulls. Snapshots, working inputs, completed results, exact-ID
  cache, HTTP handoff and control bodies have separate finite ownership.
  Results stay credited until acknowledged transfer; the oldest owner has a
  progress reserve. These are not a promise to cap total process RSS.
- Cold response width receives a conservative materialization allowance.
  Native JSON controls and embedding/error bodies are bounded before decode,
  including decompression. Reentrant shutdown and repeated interruption retain
  typed failures and actual drain. Width is published by the coordinator after
  response completion, avoiding an ordered-result credit race.
- Eligible singleton statusless observations reuse root properties within the
  same boundary. Fresh identity/limit checks before native inference remain;
  the default HTTP connection policy remains `close`, with explicit `reuse`.
- `AutomaticEmbeddingService.diagnostics()` exposes local bounded counters,
  capacity/resource observations and wait reasons without issuing HTTP.

The direct automatic path continues to send text without per-input `/tokenize`
or implicit calibration. Exact task-aware preparation remains available,
including one application of the configured prefix and special-token policy.
Consumer applications retain source splitting, AST/graph construction and
fragment identities.

## Public API

Existing embedding method signatures are preserved:

```text
service.embed_texts(texts, *, input_type="document", timeout=None, cancel=None)
service.astream_texts(windows, *, input_type="document",
                      max_pending_windows=2, timeout=None, cancel=None,
                      close_source=False)
service.diagnostics()

adapter.prepare_texts(texts, *, input_type="document", timeout=None, cancel=None)
adapter.embed_prepared(inputs, *, timeout=None, cancel=None)
```

Use `request_parallelism=None` on the automatic service, or
`embedding_request_parallelism=None` on the adapter profile, for automatic
N-wide capacity. Positive values retain a global guard.
`embedding_max_pending_owners` is the optional positive owner ceiling on both
configuration surfaces.

```python
from runtime_llama import AutomaticEmbeddingService

service = AutomaticEmbeddingService({
    "base_url": "http://inference.example.invalid:8080",
    "model": "logical-embedding-name",
    "batch_size": 256,
    "request_parallelism": None,
    "embedding_max_pending_owners": 18,
})
try:
    vectors = service.embed_texts(["first document", "second document"])
    counters = service.diagnostics()
finally:
    service.close()
```

This is a consumer example, not an installation or migration. See
[concurrency and diagnostics](../CONCURRENCY-AND-DIAGNOSTICS.md),
[embedding contracts](../EMBEDDING-CONTRACTS.md) and
[automatic embeddings](../AUTOMATIC-EMBEDDING.md) for effective limits.

## Verification and packaging

The immutable `0.14.0.dev1` candidate completed **2,254 deterministic tests**,
Ruff lint/format, wheel/source builds and a clean external installed-wheel
smoke and dependency check. Its executable implementation is the basis for
the stable release. These historical candidate results are not substituted for
fresh stable-package checks. The detailed
[candidate verification record](REPLICA-OWNERSHIP-CANDIDATE.md) remains unchanged.

The stable source passed **2,254 tests in 73.18 seconds**, Ruff lint and
format verification, normalized builds, an external clean-environment
installed-wheel stream/order/resource smoke and dependency verification.
The four warnings come from deliberately unverified local HTTPS test fixtures.
All 45 packaged runtime payload files match the stable source: 44 are
byte-identical to the qualified candidate and only the version assignment in
`runtime_llama/__init__.py` changes. No inference implementation was modified
during promotion. The stable wheel SHA-256 is
`b110d965fb8d98448b68d2541a9db4abe5782a47ca3d8aa7788359f53c903dcf`.

Seven public assets comprise the Python wheel, normalized source archive,
three optional tsnet helper archives, `RELEASE-MANIFEST.json` and `SHA256SUMS`.
The manifest binds the final source revision, candidate provenance, fresh
stable checks, retained live evidence, numerical exception and artifact
hashes. Source archive files are compared with the final committed source;
host names, owner IDs and timestamps are normalized using
`SOURCE_DATE_EPOCH=315532800`. The separate manifest and checksum files carry
the source archive hash without introducing a self-referential document hash.

Publication verification checks public main/tag/source correspondence,
stable/latest metadata, anonymous release-page access and anonymous downloads
of all seven assets against local SHA-256 values. No PyPI upload or shared
installation is part of this publication.

## Four-replica live evidence

Four already loaded EmbeddingGemma 2 BF16 replicas, two per GPU, shared verified
content and semantic identities. The observed server build was
`b11716-797de63b2`; each replica reported four native slots, total context 2048
and effective per-sequence context 512, computation batch 2048, microbatch 512
and Flash Attention off. Responses had 768 dimensions. The library did not
multiply HTTP concurrency by the four slots per replica.

The candidate campaigns collected 65 SDK/native routes and a separate
23-POST native repeatability audit. SDK scatter/order checks passed, including
duplicate occurrences, with no outstanding local HTTP after drain. Peak
original HTTP overlap reached four, with a per-alias peak of one. This is
client HTTP overlap, not proof of simultaneous GPU utilization.

Automatic candidate samples for 511 inputs took **1.529–1.646 seconds**, using
eight embedding POSTs, 21 metadata GETs and zero tokenizer calls. Samples for
2048 inputs took **6.278–6.476 seconds**, with 32 POSTs, 66 metadata GETs and
zero tokenizer calls. Setup/warmup and final probe guards were separate. These
are historical candidate client wall times including transport, controls,
decode, scheduling and server work, not stable-package reruns or pure GPU time.
The limited comparisons did not establish a stable throughput improvement.
The separate `close`/`reuse` experiment showed some connection reuse but no
speed benefit; no real VPN/proxy performance qualification is claimed.

## Accepted known numerical limitation

The strict repeatability gates remain **max_abs <= 1e-5** and
**cosine >= 0.999999**. The measured server/model profile did not meet them.
Repeated identical native requests also failed with the SDK scheduler absent,
including serial requests to one alias and a singleton input. The singleton
repeat audit observed maximum absolute component difference approximately
`0.000291251` and minimum cosine `0.999997034`. Wider repeated/cross-replica
comparisons are preserved in the candidate report; none is relabeled passed.

On 2026-10-25 the operator accepted this observed profile limitation as
non-blocking for stable library publication. This records an explicit release
decision, not a relaxed threshold, full numerical qualification or evidence of
retrieval-quality degradation. The underlying numerical mechanism was not
isolated. A retrieval assessment would require application-level query/document
results; the vector repeatability audit alone does not establish that outcome.
Flower finding `FIND-000007` records `accepted_exception` for this stable
publication, with the direct operator instruction and the measured profile
scope. The historical finding and failed measurements remain retained;
neither the failed numerical outcome nor historical missing evidence is
rewritten as a pass.

## Optional server-observation boundary

`runtime_llama.server_observation` supplies tested client/provider contracts for
versioned coherent snapshots, conditional admission, revocation, deadlines,
drain and native fallback. It is not an installed server extension and does
not imply an endpoint exists on native llama.cpp. Native fresh observations
remain the default; no ambiguous POST replay is introduced.

Real server-enforced generation/admission support in optional milestone 35
requires a separate server-owner implementation and live qualification. That
server work is outside this library release and is not claimed complete.

## Publication boundaries

The release retains Apache-2.0, copyright Davide Mele, NOTICE and upstream MIT
attribution. The three optional tsnet executables are verified byte-identical
reuse from v0.6.0 with complete provenance and third-party notices in each
archive. Helper sources and the license inventory are unchanged. No new
native-platform execution, Headscale/Tailscale enrollment or VPN inference is
implied by Python checks.
Public assets exclude deployment credentials, private endpoint addresses,
local paths, raw vectors and the private live corpus.

Publishing this producer release does not update consumer installations or
accept CodingCastle/Flower application behavior. Their ingestion, persisted
character-budget provenance, lossless splitting and graph acceptance remain
in their separate lifecycles.
