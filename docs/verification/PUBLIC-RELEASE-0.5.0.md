# Public release 0.5.0 verification

Date: 2026-10-16. Environment: Python 3.11.15 on macOS.

This additive release provides `discover_models` and immutable typed catalog
observations. The distribution remains `runtime-llama`, the import remains
`runtime_llama` and the existing generation and embedding APIs are preserved.
The shared HTTP client adds optional decompressed JSON limits, explicit
owned-session ambient policy and stricter endpoint validation.

## Deterministic checks

| Check | Result |
| --- | --- |
| Full pytest suite, including HTTP loopback fixtures | 727 passed in 23.22 seconds |
| New public discovery and bounded transport cases within that suite | 144 passed: 80 discovery and 64 transport |
| Ruff check and format check, including the build backend | Passed; 52 files formatted |
| PEP 517 sdist build followed by wheel construction from the extracted sdist | Passed |
| Wheel installation and import in a clean environment outside the checkout | Passed |
| Installed public API against a synthetic native direct server omitting `role` | Passed; only catalog and root properties GETs |
| Installed generation and embedding factories and close, without inference | Passed |
| README and discovery-guide Python examples | Syntax checked |
| Existing dependency, licensing and archive normalization contracts | Preserved |

Discovery cases cover all six known native states, future and missing states,
loaded/catalog separation, sleeping models, exact alias matching, router versus
direct properties, explicit capability booleans and unknowns, independent
identity tags, metadata provenance, conflicting identities, alias collisions,
malformed responses, response/model limits, shared deadlines, cancellation,
authentication environment handling and cleanup.

Transport fixtures exercise real local HTTP responses, including chunked UTF-8,
gzip expansion, exact byte boundaries, oversized error bodies, redirect refusal,
ambient authentication isolation, borrowed session ownership and concurrent
session isolation. The final installed smoke used `requests` 2.34.2 and blocked
unrelated inference SDK imports. No live server or consumer job was invoked.

## Distribution audit

The wheel contains the new module and the existing runtime modules. Metadata
declares Python >=3.11, `License-Expression: Apache-2.0`, author `Davide Mele`
without an email address, and `requests` as the only required runtime dependency.
LICENSE, NOTICE and the retained upstream MIT license are included in both
distributions. Wheel runtime sources were compared with the source checkout.

The sdist retains normalized UID/GID zero, empty owner/group names, controlled
timestamps, regular source permissions, USTAR without PAX metadata and gzip
without an original filename. It includes the build backend, public API guide
and new tests. Final artifacts use `SOURCE_DATE_EPOCH=315532800`.

Source, documentation and package contents were reviewed for private credentials,
personal email addresses, user paths, private network addresses and lifecycle
database/ledger exports. No confirmed private values were found. Synthetic test
fixtures and redaction patterns remain. Public commits use a GitHub noreply
identity. The project copyright remains Davide Mele.

The [release](https://github.com/Damel91/llama-runtime/releases/tag/v0.5.0)
provides `runtime_llama-0.5.0-py3-none-any.whl`,
`runtime_llama-0.5.0.tar.gz` and `SHA256SUMS`. Checksums are separate assets so
the source archive does not contain its own digest. Verify downloads with
`shasum -a 256 -c SHA256SUMS` or `sha256sum -c SHA256SUMS`.

## Scope and remaining qualification

These checks establish local software and distribution behavior. They do not
claim new live model qualification, numerical equivalence between embedding
replicas, consumer migration, Linux/Windows execution or human acceptance.
No real endpoint, database, model loading configuration or service was changed.

The discovery snapshot is a server assertion, not a reservation or an inference
test. Capability absence remains `None`; names, templates, architecture widths
and compatibility defaults do not establish chat or embedding support. Identity
tags do not independently certify remote file contents. Training context remains
distinct from allocated context and consumer limits.

Timeouts are finite and shared across discovery stages; DNS resolution and
drained close retain the existing transport timing limitations. A timeout is not
a process-wide forced termination guarantee. See the
[API guide](../MODEL-DISCOVERY.md) for exact proof rules, limits and error types.
