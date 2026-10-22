# Public release 0.11.0 verification

Date: 2026-10-22. Distribution `runtime-llama`, import `runtime_llama`.
Stable publication follows explicit human authorization after the
[automatic embedding qualification](AUTOMATIC-EMBEDDING-0.11.0a1.md).
Consumer installations, configuration, pins, services and models remain unchanged.

## Included behavior

`AutomaticEmbeddingService` discovers an unambiguous verified loaded embedding
identity and its usable replicas. Synchronous and asynchronous direct strings
use the existing persistent engine and scheduler. The library owns prefixes,
ordering, bounded concurrency, backpressure, cancellation, finite deadlines and
drained shutdown. Ordinary availability and embedding never require calibration
statistics or perform per-input `/tokenize` calls.

Only an explicit `character_budget()` request may load or perform calibration.
The package contains 10,000 original synthetic samples, 2,000 each in `code`,
`documentation`, `prose`, `structured` and `multilingual`; `mixed` aggregates all
samples. Single categories use 80% of the measured source characters per fully
prepared native token; `mixed` uses 90%. Complete verified raw statistics are
persisted independently of those factors. The consumer can save its character
recommendation and subsequently send strings without another budget request.

Recommendations are empirical, not exact tokenizer limits. Native overflows
remain explicit, without silent truncation or replay. Queue reservations are
resource accounting, not exact token measurements. The declared-model exact
`prepare_*` and `embed_prepared` APIs remain available with their existing
contracts. See the [public API and example](../AUTOMATIC-EMBEDDING.md).

## Qualification retained and repeated

Stable source passed **1,730 tests in 64.59 seconds**, Ruff check/format, normalized
builds and clean isolated installation. Four expected HTTPS warnings originate
from deliberate loopback transport fixtures. Corpus hashes/counts, public APIs,
licenses, source contents and package boundaries are checked.

The stable runtime preserves all 36 package payload files from the qualified
local candidate, with only the expected `__version__` literal changing from
`0.11.0a1` to `0.11.0`. All other package payload and corpus bytes are identical.
The qualification baseline wheel SHA-256 is
`3c06791087caf38e0406fff115643852391415802c11e9599c01174e19178724`.
Its recorded live evidence remains applicable; publication adds no inference
benchmark or model administration.

That bounded live test used two already loaded verified Nomic embedding replicas,
native context/batch/microbatch 512 and one slot each. Before calibration, the
direct 511-text path took **1.5626 seconds**, **four embedding POSTs plus 16 metadata
GETs**, **zero tokenizer requests**, with both replicas used concurrently. The
exact path took **3.9873 seconds**, **511 tokenizer POSTs, four embedding POSTs and
15 metadata GETs**. All 511 vectors matched exactly under unchanged thresholds:
maximum absolute difference 1e-5, minimum cosine 0.999999 and normalization error
1e-6. Actual maximum difference was zero and minimum cosine was one.

Only the explicit budget request then performed the full 10,000 tokenizations,
in **54.3861 seconds**. Category selection, new-service ordinary use and explicit
warm-profile reuse caused no further tokenization. These measurements include
applicable preparation, client/server work and HTTP; they are not pure GPU time
or a general throughput guarantee. The linked engineering report retains all
phases and category recommendations.

## Public assets and provenance

Seven assets are provided: portable Python wheel, normalized source archive,
three optional tsnet helper archives, `RELEASE-MANIFEST.json` and `SHA256SUMS`.
The manifest binds source revision, checksums, qualification summary, corpus and
helper provenance. Anonymous downloads are checked against the staged assets.

Packaging uses `SOURCE_DATE_EPOCH=315532800`. Source archive members have normalized
owners and timestamps and match tracked source, including the original corpus
generator. Wheel contents match stable source and the qualified candidate under
the version-only exception above. The isolated installation is outside the
source tree and does not alter a consumer environment.

Optional helper binaries and the license inventory are reused byte-for-byte from
v0.6.0. Rewrapping adds no new native build, Linux/Windows execution, VPN enrollment
or VPN inference qualification. See the
[original helper qualification](PUBLIC-RELEASE-0.6.0.md).

Private deployment configuration, absolute local evidence paths, cache contents
and raw evidence dumps are excluded. Apache-2.0, copyright Davide Mele, NOTICE and
upstream MIT attribution remain unchanged. No PyPI upload is included.

Flower MCP records publication under `CHANGE-000023` and `PACKET-000024`.
Engineering qualification remains under `REQ-000038`, `MILE-000021`,
`CHANGE-000022` and `PACKET-000023`. Publication and a handoff to the separate
CodingCastle chat do not modify or qualify a consumer integration.
