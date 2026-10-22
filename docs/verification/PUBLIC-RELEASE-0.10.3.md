# Public release 0.10.3 verification

Date: 2026-10-22. Distribution `runtime-llama`, import `runtime_llama`.
This publication follows explicit authorization after the
[incremental efficiency qualification](REVIEW-EFFICIENCY-0.10.3.md).
Consumer deployments remain unchanged.

## Included changes and retained qualification

Version 0.10.3 corrects recovery by validating each pending record against its
own historical limits and the full fresh observation. An owner-local failure
retires that owner's rows while preserving surviving callers, original indices
and actual transport drain. Native failed or ambiguous POSTs remain
non-replayable.

`embedding_max_response_bytes` independently bounds decoded native JSON bodies.
Exact preparation-cache entry and byte limits are configurable; zero disables
retention, and the pending-byte quarter cap remains. The coordinator waits for
events and relevant deadlines rather than a fixed 10 ms poll. Unnotifiable
cancellation callbacks and custom coalescing clocks retain finite checks.
Incremental counters and direct iteration avoid unnecessary hashing, scans,
rotating copies and index searches without replacing the scheduler.

The implementation passed **1,578 tests**, Ruff check/format, builds and clean
installed-wheel verification. Recovery regressions include the original failure,
fresh context/build/vocabulary/identity, shared-owner isolation, child budgets
and drain. Response-limit tests include decoded gzip overflow, framing,
deadlines and healthy concurrent owners. Coordinator tests include remembered
wakes, cancellation, fairness, topology, budgets and close.

The retained installed-wheel live pair used the same two already loaded,
verified EmbeddingGemma 2 BF16 replicas and unchanged numerical thresholds.
Each 511-text pipeline made **539 HTTP requests: 511 exact tokenizations,
four embedding POSTs and 24 metadata reads**. Both replicas ran concurrently;
each carried at most one active embedding POST. Vectors matched exactly
between `close` and `reuse` and against retained 0.10.2 outputs.

Pipeline wall times were 4.8449 s with `close` and 4.6630 s with opt-in `reuse`.
This single pair qualifies the current plain-HTTP route functionally; it does
not establish a general latency improvement. Historical 0.10.2 profile digests
differ, so it is not a controlled version-speed comparison. The default
connection policy stays `close`. Measurements include client/server/HTTP and
preparation work and are not pure GPU timings. Publication adds no new
inference benchmark, model administration or service restart.

## Public assets and provenance

Seven assets are provided: Python wheel, normalized source archive, three
optional tsnet helper archives, `RELEASE-MANIFEST.json` and `SHA256SUMS`.
The public manifest binds the exact source revision, asset hashes, qualification
summary and helper provenance. Anonymous downloads are checked against those
checksums after publication.

Packaging uses `SOURCE_DATE_EPOCH=315532800`. Source archive contents are
compared with tracked files. All 28 runtime payload files in the public wheel
must match the live-qualified local candidate with SHA-256
`54f28be5c5d51241df267609a42d1cf57de01fa6031d5a422d8ce42990d3b013`.
README metadata and normalized timestamps can change the outer wheel hash;
the release manifest and checksums identify downloadable assets. An isolated
installation checks import/version, dependencies and inert adapter
construction/drained close outside the source checkout.

The three optional tsnet executables and license inventory are byte-identical
reuse from v0.6.0. Rewrapping adds no new native build, Linux/Windows execution,
VPN enrollment or VPN inference qualification. See the
[original helper qualification](PUBLIC-RELEASE-0.6.0.md).

Private deployment configuration, absolute local evidence paths and raw evidence
dumps are excluded from public assets. Apache-2.0, copyright Davide Mele, NOTICE
and upstream MIT attribution remain unchanged. No PyPI upload or installation
into shared consumer environments is included.

Flower MCP retains publication authority and evidence under `CHANGE-000021`
and `PACKET-000022`, linked to implemented `REQ-000037` and `MILE-000020`.
The engineering qualification remains under `BOOT-000018`, `CHANGE-000020`
and `PACKET-000021`. Publication does not infer consumer acceptance.
