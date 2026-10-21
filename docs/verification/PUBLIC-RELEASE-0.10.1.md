# Public release 0.10.1 verification

Date: 2026-10-21. Distribution `runtime-llama`, import `runtime_llama`.
This corrective release follows the explicit publication instruction and the
[model-wide batching qualification](MODEL-BATCHING-0.10.1.md).

## Included correction and verification

Compatible records from independent source windows and callers accumulate in
each verified replica's ready queue and can share a POST. The configured
total-text budget determines request targets independently of source window
size. Preparation/capacity events fill batches with no mandatory timer;
finite tails and resource pressure still progress. One scheduler preserves
global/per-replica admission, bounded backpressure, order, identity, topology,
typed outcomes and actual drained closure.

Shared transports retain logical owner membership. Owner-local cancellation
or validation failure preserves survivors; backend/transport failure affects
all owners participating in that POST. Each participant waits for actual
transport drain. Native failed or ambiguous POSTs are not replayed. The
existing synchronous and async public signatures remain unchanged.

The final implementation passed **1,401 deterministic tests**, Ruff check and
format, builds and clean installed-wheel checks outside source. A bounded
installed-wheel live run preserved current numerical thresholds and produced
511 ordered vectors from eight small source windows in four mixed-window
embedding POSTs, using both already loaded replicas and zero coalescing delay.
The models, server and consumers were unchanged. See the qualification record
for counts, numerical checks, timings, limits and evidence hashes.

No new 0.9.0 comparison, throughput claim, consumer acceptance, Linux/Windows
execution or GPU-only timing is asserted. An LLM workload remained permitted
during the functional run. Four local TLS fixture warnings are intentional.

## Artifacts and provenance

The GitHub release contains seven assets: Python wheel, normalized source
archive, three optional tsnet helper archives, `RELEASE-MANIFEST.json` and
`SHA256SUMS`. The manifest binds exact asset hashes and the final source
commit. Download hashes are verified after publication. Use the manifest or
checksums for the public wheel; the qualified candidate hash below identifies
the local verification baseline rather than the public download.

Public packaging uses `SOURCE_DATE_EPOCH=315532800`. The normalized source
archive is checked against tracked source; all 27 public Python runtime files
are byte-compared to the installed/live-qualified candidate with SHA-256
`99d89d1f3e96c0f0f88032988269802a0c1db0ac83be531fdac0cc9369133780`.
Final README metadata and timestamps may change the outer wheel hash. A fresh
installation of the public wheel checks import/version, dependency consistency
and inert construction/closure without checkout source or shared consumers.

The optional tsnet executables, Go source and license inventory are unchanged
from their original 0.6.0 qualification, also shipped in 0.7.0 and 0.10.0.
The 0.10.1 archives retain explicit provenance and identical executable bytes.
Rewrapping adds no new native build, operating-system execution, real VPN
enrollment or VPN inference claim. See
[original helper qualification](PUBLIC-RELEASE-0.6.0.md).

Private deployment configuration, credentials, local evidence dumps and
consumer data are excluded. Apache-2.0, copyright Davide Mele, NOTICE and the
derived client's MIT attribution remain unchanged. No PyPI upload, consumer
installation, service restart or remote model administration is included.

Flower MCP retains the lifecycle and evidence under `REQ-000028` through
`REQ-000030`, `MILE-000016`, `CHANGE-000016`, `PACKET-000017`, `RUN-000011`
and `BOOT-000015`, using adopted architecture revision 2. Verification and
publication do not infer human acceptance.
