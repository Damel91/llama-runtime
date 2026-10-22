# Public release 0.11.1 verification

Date: 2026-10-23, Europe/Rome. Distribution `runtime-llama`, import `runtime_llama`.
This stable patch follows explicit human authorization to apply confirmed review
fixes, retain 80%/90% margins and the current scheduler, qualify, push and publish.
Consumer code, configuration, pins, installations, databases, jobs, services and
model profiles are outside the producer changes.

## Included fixes

- Automatic config exposes existing positive request, decoded-response,
  pending-byte and pending-token budgets, preserving their defaults. Large valid
  vector responses can use an explicit higher finite response limit.
- Owned HTTP clients accept explicit `close` or `reuse`, default `close`;
  injected clients retain connection policy and ownership.
- Calling the same service's draining close from its own cancellation callback
  raises a typed diagnostic before locks or shutdown mutation. Callback purity
  remains required; external cancellation and drained close remain intact.
- Cold explicit calibration checks actual persistence feasibility before remote
  counts. Readable complete caches do not require directory writes; final
  validation and private atomic publication remain authoritative. Late timeout
  checks retain `LLMTimeoutError` instead of being misclassified as filesystem
  errors.
- Verified single-model statusless catalogs can use coherent read-only root
  `/props` proof shared with the declared adapter. Ambiguous, router, sleeping,
  unverified and inconsistent observations are not assumed loaded.
- The empirical category 80% / mixed 90% policy is unchanged. Documentation and
  math regressions clarify that mixed is not conservative for every category.
  High-context cardinality can be adjusted through the existing bounded pending
  budget, without replacing the engine/scheduler or inventing token counts.

See the [API/configuration guide](../AUTOMATIC-EMBEDDING.md) and the
[hardening qualification](REVIEW-HARDENING-0.11.1.md) for exact boundaries.
There is no new implicit tokenizer, statistical input admission, silent
truncation, POST replay or consumer fallback path.

## Qualification

The implementation passed **1,811 tests in 67.28 seconds**, Ruff check/format,
normalized builds, dependency checks and isolated installed-wheel verification.
Four expected warnings originate from deliberately unverified loopback HTTPS
fixtures. Meaningful regressions cover 128 vectors of 4,096 dimensions over real
loopback HTTP, decoded gzip bounds, incomplete plain/gzip bodies, no replay,
typed failures and drain; configurable high-context capacity and concurrent
owners; reentrant callback close in supervised subprocesses; cache feasibility,
read-only reuse, cancellation/timeouts; and verified statusless discovery.

The isolated wheel passed a fresh bounded qualification on both already loaded
Nomic embedding replicas without model or service changes. The direct 511-text
pass ran before calibration in **1.5078 seconds**, with **four embedding POSTs,
16 metadata GETs and zero tokenizer calls**. Exact preparation plus embedding
took **3.7297 seconds**, with 511 tokenizer POSTs, four embedding POSTs and
15 metadata GETs. Both replicas ran concurrently; all vectors matched exactly
under unchanged 1e-5 / 0.999999 / 1e-6 thresholds.

Only an explicit cold budget request performed all 10,000 tokenizations,
in **53.8941 seconds**. Category selection, ordinary new-service embedding and
explicit warm-profile reuse required no additional tokenization. Corpus,
identity/statistics schema and character recommendations remained compatible.
These are client wall times including applicable metadata, preparation,
transport and server work, not pure GPU time or a general throughput guarantee.

## Assets and provenance

Seven public assets are supplied: portable Python wheel, normalized source
archive, three optional tsnet helper archives, `RELEASE-MANIFEST.json` and
`SHA256SUMS`. The public manifest binds exact source revision, qualified candidate
hash, package/corpus contents, qualification summary and helper provenance.
All 36 public runtime payload files must match the qualified 0.11.1 candidate
and stable source byte-for-byte. README metadata and normalized wrapping can
change the outer wheel hash; downloadable asset hashes are in the manifest.

Builds use `SOURCE_DATE_EPOCH=315532800`. Source archive files match tracked
source, including the original corpus generator; owners, timestamps and archive
metadata are normalized. A clean installation outside the source checkout
checks APIs, new configuration fields, corpus and inert construction/close.
Published main/tag/latest, raw tagged source and all seven anonymous asset
downloads are verified against local source and checksums.

The three optional helper executables and license inventory are byte-identical
reuse from v0.6.0. No new native build, Linux/Windows execution, VPN enrollment
or VPN inference is claimed. See the
[original helper qualification](PUBLIC-RELEASE-0.6.0.md).
Apache-2.0, copyright Davide Mele, NOTICE and upstream MIT attribution are
unchanged. Public assets exclude private deployment settings, local absolute
evidence paths, caches and raw receipts. No PyPI upload or shared installation
is included.

Flower lifecycle authority and host-reported evidence are recorded under
`REQ-000039`, `MILE-000022`, `CHANGE-000024` and `PACKET-000025`.
Publication does not implement or accept any consumer integration.
