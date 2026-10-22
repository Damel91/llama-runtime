# Public prerelease 0.12.0a1

The stakeholder explicitly authorized pushing and publishing the current
0.12.0a1 candidate on 2026-10-23. GitHub marks this version as a **prerelease**;
0.11.1 remains the latest stable release. The observed cross-replica numerical
finding remains open. Publication does not assert complete numerical qualification
or stakeholder acceptance of that difference.

Flower delivery authority: `REQ-000041`, `MILE-000024`, `CHANGE-000026`,
`PACKET-000027`. The original implementation/verification remains under
`REQ-000040` / `PACKET-000026`, with open `FIND-000007`.

## Exact artifact and source provenance

The Python wheel is the exact artifact tested and used by the diagnostic
consumer, without rebuilding or changing its metadata:

```text
runtime_llama-0.12.0a1-py3-none-any.whl
SHA256 85a0d6f57a20be595dc818448066cf97694ccfe286b1b8c06d8774ea06518b04
```

Its embedded README retains the historical local-candidate wording from the
qualification build. Current repository documentation, GitHub prerelease notes
and `RELEASE-MANIFEST.json` record the subsequent publication authority and
consumer diagnostic result. The source archive is built from the release tag's
current documentation; its 38 runtime payload files are byte-identical to the
frozen wheel. Manifest provenance distinguishes code commit, qualified build
documentation and publication source revision.

Public assets are the wheel, current source archive, three optional tsnet helper
archives, `RELEASE-MANIFEST.json` and `SHA256SUMS`. Release verification checks the
tag/main source, anonymous release page/raw source, all seven anonymous asset
downloads and their hashes. No PyPI upload is part of this publication.

## Verification boundary

The [qualification record](EXACT-OVERFLOW-0.12.0a1.md) records 1,963 passing tests,
Ruff, builds and isolated installation. Source-code tests and live evidence are
unchanged for this documentation/publication step. The exact wheel passed its
installed smoke and dependency checks; source/runtime byte equivalence is
verified before publication.

The direct 511-text check produced identical vectors to the exact path using
both replicas, four embedding POSTs and no tokenizer calls. The original
531-token input was diagnosed only after its failed POST drained and returned
exact 512/25-token fragments. A diagnostic consumer ingestion separately
published 126,045 vectors with one atomic commit and no corpus errors. These
functional results do not resolve the real-input cross-replica difference:
maximum absolute difference `0.0024030989764955157`, minimum cosine
`0.9998033728657948`, outside unchanged thresholds `1e-5` / `0.999999`.

On each individual replica, the affected strings and exact native token IDs
produce identical vectors; the condition follows the replica. No cause at the
hardware/kernel level is asserted. The 80%/90% factors, scheduler, model pool,
thresholds and server configuration are unchanged. This publication does not
install or update a consumer.

## Optional helper and license inventory

The optional helper executables are reused byte-for-byte from v0.6.0, with
unchanged sources and dependency/license inventory. The archives include the
Apache-2.0 project license, Davide Mele copyright, NOTICE, Tailscale license and
third-party notices. Their `SOURCE.txt` binds the original source commit and
binary hashes. No new helper build, native OS execution, live VPN enrollment or
VPN inference qualification is claimed for this prerelease.

Private endpoints, credentials, calibration caches, consumer data, source payloads
and diagnostic vectors are excluded from public assets. Copyright and upstream
license attribution are preserved.
