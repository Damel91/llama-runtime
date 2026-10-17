# Public release 0.5.1 verification

Date: 2026-10-17. Environment: Python 3.11.15 on macOS; isolated package smoke
on Python 3.14.6. This corrective release adds explicit nullable generation idle,
following the [local development qualification](LOCAL-IDLE-0.5.1.dev1.md).

## Behavior

`stream_idle_timeout_sec=None` disables generation SSE idle before response
headers, before the first token and between events. Omitted/default behavior
remains 30 seconds; a positive per-call timeout restores a finite idle budget.
Connection, JSON, discovery, preparation and embedding retain finite budgets.
The raw SSE-only `idle_timeout` keyword distinguishes omitted/inherited from
explicit `None`. Cancellation and close interrupt the exact owned request,
drain cleanup and detach before session reuse, without retry or inference replay.
Borrowed Requests Sessions reject unbounded SSE before dispatch. See the
[public timeout and bridge guide](../STREAM-TIMEOUTS.md) for ownership, nullable
evidence, validation and timing limitations.

## Actual verification

| Check | Result |
| --- | --- |
| Stable full deterministic suite | 841 passed in 27.08 seconds |
| Ruff check and format check | Passed; 56 files formatted |
| PEP 517 normalized source and wheel build | Passed |
| Clean stable wheel install outside checkout, Python 3.14.6 | Passed; Requests 2.34.2 / urllib3 2.8.0; pip dependency check passed |
| Installed HTTP wait before headers | 31.005 seconds, then successful response with idle `None` evidence |
| Installed Stop before headers | Typed cancellation in 0.051 seconds, fixture gate still closed |
| Installed request count | Two HTTP requests for two calls, no replay |
| Runtime payload parity, metadata and retained licenses | Passed |
| Source/package privacy and archive normalization audit | Passed |
| README and timeout-guide executable Python examples and relative links | Passed |

The installed fixture blocks unrelated inference SDK imports, uses a synthetic
bearer key, public `trust_env=False`, a finite 30-second connection budget and a
bridge-owned raw client injected into the adapter. The bridge closes both
objects. Timings include artificial waits, transport and Python work; they are
not GPU timings or throughput claims.

The correction's dependency matrices, recorded in the local qualification,
also pass with Requests 2.31.0 / urllib3 1.26.20 and the host combination, plus
Python 3.14.6. Stable promotion changes version metadata and documentation;
those implementation matrices apply to the unchanged qualified runtime code.
The full regression and installed artifact smoke above were run for `0.5.1`.

HTTP/TLS fixtures retain finite/default/borrowed behavior and cover quiet
headers/body/events, cancellation, close, concurrent isolation, keep-alive and
stale cleanup, environment authentication/proxy handling, malformed/truncated
responses and stalled error bodies. Explicit Content-Length enforcement retains
typed truncation failures across the supported urllib3 branches.

## Distribution

The distribution/import remain `runtime-llama` / `runtime_llama`. Metadata
declares Python >=3.11, Apache-2.0, copyright holder Davide Mele without an email,
and `requests>=2.31,<3` plus `urllib3>=1.26.20,<3`. The explicit urllib3 floor
covers the qualified scoped pool/connection extension hooks. LICENSE, NOTICE
and the vendored upstream MIT license remain included and unchanged.

The source archive has UID/GID zero, empty owners, controlled timestamps,
regular permissions, USTAR without PAX metadata and gzip without an original
filename. Artifacts use `SOURCE_DATE_EPOCH=315532800`. Runtime wheel payloads
match the checkout. Reviewed source, documentation and artifacts contain no
confirmed private credentials, personal email, user paths, private endpoint or
lifecycle database exports. Synthetic fixtures remain identifiable as tests.

The [release](https://github.com/Damel91/llama-runtime/releases/tag/v0.5.1)
supplies `runtime_llama-0.5.1-py3-none-any.whl`,
`runtime_llama-0.5.1.tar.gz` and `SHA256SUMS`. Checksums are separate assets,
avoiding a self-referential archive. Verify with
`shasum -a 256 -c SHA256SUMS` or `sha256sum -c SHA256SUMS`.

## Qualification boundary

These checks establish local SDK and distribution behavior. They do not claim
new live model qualification, Noxi integration or consumer acceptance,
Linux/Windows execution, or remote GPU cancellation. No consumer job,
database, service, model profile or shared Python environment was changed.
System DNS cannot be forcibly cancelled by the socket hook; connect/TLS retains
its finite connection budget. Existing generation limits and embedding
scheduling/preparation contracts are unchanged.
