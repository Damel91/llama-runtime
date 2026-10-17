# Local optional-idle qualification 0.5.1.dev1

Date: 2026-10-17. This development build qualifies explicit nullable generation
idle before a separately governed stable publication. Distribution/import remain
`runtime-llama` / `runtime_llama`. No consumer installation is part of this check.

## Actual checks

| Check | Result |
| --- | --- |
| Full deterministic suite, Python 3.11.15, macOS | 841 passed in 27.42 seconds |
| New cases included above | 19 configuration/evidence, 50 transport, 45 adapter |
| Ruff check and format check | Passed; 56 files formatted |
| Host transport/configuration/client subset | 231 passed in 8.21 seconds |
| Minimum Requests 2.31.0 / urllib3 1.26.20 subset | 231 passed in 7.96 seconds; 45 adapter cases passed in 1.04 seconds |
| Recent Requests 2.34.2 / urllib3 2.8.0 subset | 276 passed in 9.13 seconds |
| Python 3.14.6, recent dependencies | 231 passed in 9.16 seconds; 45 adapter cases passed in 1.07 seconds |
| Normalized PEP 517 source and wheel build | Passed |
| Clean installed wheel, outside the checkout, Python 3.14.6 | Passed; unrelated inference SDK imports blocked |
| Installed silent HTTP wait before response headers | 31.018 seconds, then successful generation with nullable idle evidence |
| Installed Stop before response headers | Typed cancellation in 0.058 seconds while the fixture gate remained closed |
| Installed HTTP request count | Two requests for two calls; no replay |

The installed smoke used the public bridge-owned `LlamaCppClient` with
`trust_env=False`, an explicit synthetic bearer key, finite 30-second connection
budget and an injected generation adapter configured with idle `None`. The
bridge closed both objects. It performed no inference against a real model.
Elapsed times include artificial HTTP waits, Python and transport work; they
are neither GPU timings nor a performance guarantee.

Transport fixtures cover quiet headers, first body and inter-event gaps;
cancellation and close; separate concurrent leases; keep-alive and obsolete
stream cleanup; HTTP/TLS and proxy/authentication policy; finite JSON/discovery
and embedding regressions; malformed/truncated bodies; stalled error responses;
borrowed-session rejection before unbounded dispatch; and finite overrides.

The minimum urllib3 branch initially exposed a truncated-response difference.
The client now explicitly enforces Content-Length for actual Requests/urllib3
responses before reading. The test was retained and passed after the correction.
No observed failure was hidden by relaxing a check.

## Package and evidence boundaries

Source and package reviews cover versions, public signatures, runtime source
parity, declared Requests/urllib3 bounds, Apache-2.0 metadata, retained upstream
MIT attribution, archive normalization and absence of confirmed private values.
The README and optional-idle guide's executable Python examples are syntax
checked; existing illustrative API signatures are not executable examples.
Exact artifact digests and source revision accompany the local handoff rather
than being embedded in their own archive.

These results establish local software behavior only. They do not certify Linux
or Windows execution, a live remote model, Noxi integration, consumer acceptance
or that closing the local socket stops remote GPU work. System DNS and the
finite connect/TLS phase retain the limitations described in the
[timeout guide](../STREAM-TIMEOUTS.md). No runtime process, loaded model, consumer
job, database or shared environment was changed.
