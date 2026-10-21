# Public release 0.10.2 verification

Date: 2026-10-22. Distribution `runtime-llama`, import `runtime_llama`.
This publication follows explicit authorization after the
[local metadata/preparation qualification](METADATA-PREPARATION-0.10.2.md).
Consumer deployments remain unchanged.

## Included changes and retained qualification

Version 0.10.2 removes duplicate admission catalog reads and shares fresh
metadata within frozen owner groups. Completed exact token IDs can be reused
through a byte/entry-bounded adapter cache keyed by full prepared text, task,
token policy, verified content/semantic identity and tokenizer build/vocabulary.
Current limits, prepared seals and fresh dispatch checks remain authoritative.

An interrupted synchronous wait retires only its logical owner, wakes the
coordinator and drains actual work before reraising the original exception.
Shared POST survivors continue. Incremental queue summaries and iterator-based
packing avoid repeatedly scanning growing prefixes; late owner admission renews
partial metadata membership instead of prematurely fragmenting the batch.
One scheduler preserves cancellation, deadlines, ordering and backpressure.

The implementation passed **1,506 tests**, Ruff check/format, builds and clean
installed-wheel verification. Repeated real SIGINT, shared-owner drain, exact
cache reuse, 1,000 packing comparisons and the 128-arrival visitation bound are
covered. The implementation payload is unchanged by publication.

The retained installed-wheel live run used both already loaded, content-verified
EmbeddingGemma 2 BF16 replicas at unchanged profiles and numerical thresholds.
Its 511 unique inputs produced four embedding POSTs and 24 metadata reads,
versus the prior 50 metadata reads; exact native tokenization still used 511
requests. Pipeline wall time was 4.6436 s versus the retained 4.5116 s baseline:
no elapsed-time improvement is claimed. All 511 vectors matched the baseline
with zero coordinate difference. A separate prepared/raw reuse exercise used
zero additional tokenizations after its eight initial exact measurements.

These are client/HTTP measurements, not pure GPU timings or consumer acceptance.
See the qualification record for request scopes, concurrency, oversized-input
rejection, numerical evidence and timing limits. Publication runs no new model
administration, inference benchmark, consumer job or service restart.

## Public assets and provenance

Seven assets are provided: Python wheel, normalized source archive, three
optional tsnet helper archives, `RELEASE-MANIFEST.json` and `SHA256SUMS`.
The manifest binds the exact source revision, asset hashes, qualification
summary and helper provenance. Download checksums are verified after publication.

Packaging uses `SOURCE_DATE_EPOCH=315532800`. Source archive contents are
compared with tracked files. All 28 runtime payload files in the public wheel
must match the live-qualified local candidate with SHA-256
`5e07f27fc977d81872da9214e8a2b243afd0aaec2ba096b757ee75dc6c564f56`.
README metadata and normalized timestamps can change the outer wheel hash;
use the public manifest or checksums for the downloadable archive. A clean,
isolated install checks import/version, dependency consistency and inert
adapter construction/close outside the source checkout.

The three tsnet executables and license inventory are byte-identical reuse
from v0.6.0, also distributed in v0.10.1. Rewrapping adds no new native build,
Linux/Windows execution, live VPN enrollment or VPN inference qualification.
See [original helper qualification](PUBLIC-RELEASE-0.6.0.md).

Private deployment configuration and evidence dumps are excluded. Apache-2.0,
copyright Davide Mele, NOTICE and upstream MIT attribution remain unchanged.
No PyPI upload or installation into shared consumer environments is included.

Flower MCP retains publication authority and evidence under `BOOT-000017`,
`REQ-000036`, `MILE-000019`, `CHANGE-000019`, `PACKET-000020` and `RUN-000014`.
The completed engineering qualification remains under its original packets,
runs and handovers. Publication does not infer human acceptance.
