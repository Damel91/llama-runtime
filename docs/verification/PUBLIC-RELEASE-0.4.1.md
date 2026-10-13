# Public release 0.4.1 verification

Date: 2026-10-13. Environment: Python 3.11.15 on macOS.

This release changes publication, licensing, package metadata and source archive
construction. The 21 runtime payload files preserve the preceding implementation;
only the package version constant changes from 0.4.0 to 0.4.1. The distribution
remains `runtime-llama` and the import remains `runtime_llama`.

## Deterministic checks

| Check | Result |
| --- | --- |
| Full pytest suite, including HTTP loopback fixtures | 583 passed in 22.07 seconds |
| Packaging metadata and unsafe archive cases within that suite | 29 passed |
| Ruff check and format check, including the build backend | Passed |
| PEP 517 source build followed by wheel construction from the extracted sdist | Passed |
| Wheel installation and import outside the source checkout | Passed |
| Generation and embedding factories, version and close without inference | Passed |
| Runtime source comparison against the preceding implementation | Identical except version constant |
| Relative documentation links and Python snippets in README | Valid |

The isolated import check blocks network inference and imports of unrelated
inference SDKs. No remote model operation is required to build or import this
package. The full suite ran with socket access for its local HTTP fixtures.

## Distribution audit

Wheel metadata declares `License-Expression: Apache-2.0`, author `Davide Mele`
without an email address, and `requests` as the only required runtime dependency.
The wheel and sdist include LICENSE, NOTICE and the retained MIT vendor license.

The source archive normalizes UID/GID to zero, owner/group names to empty
strings, permissions to regular source modes and timestamps to the controlled
epoch. Its default epoch is 1980-01-01 UTC; `SOURCE_DATE_EPOCH` can override it.
The archive uses USTAR without PAX metadata and gzip without an original filename.
The backend rejects unsafe paths, links, special files, duplicate entries,
oversized content and corrupt streams, and removes outputs when normalization
fails. The backend and its tests are included in the source archive.

Public source, Git objects and package contents were reviewed for real private
credentials, personal emails, user paths, private network addresses and private
lifecycle exports. No confirmed private values were found. Synthetic fixtures
and redaction patterns remain intact. Public commits use a GitHub noreply
identity. The public repository starts with a new history.

## Artifacts and scope

The [release](https://github.com/Damel91/llama-runtime/releases/tag/v0.4.1)
provides `runtime_llama-0.4.1-py3-none-any.whl`,
`runtime_llama-0.4.1.tar.gz` and `SHA256SUMS`. Hashes live alongside the artifacts
to avoid a source archive containing its own checksum. Verify downloaded assets
with `shasum -a 256 -c SHA256SUMS` or `sha256sum -c SHA256SUMS`.

These are local software and distribution checks. This release does not claim
new live model qualification, consumer migration or human application acceptance.
The probes remain opt-in tools for qualifying a specific server/model setup.
