# Public release 0.7.0 verification

Date: 2026-10-20. Distribution: `runtime-llama`; import: `runtime_llama`.
The engineering lifecycle and retained evidence are managed with Flower MCP.
This report supplements the earlier [local candidate report](LOCAL-NATIVE-0.7.0.md);
the earlier failed results remain historical evidence.

## Release behavior

Native llama.cpp works through independent HTTP connections by default, without
requiring the historical keep-alive patch. Connection reuse is explicit. The
client separates the total number of texts, per-input context/microbatch limits,
native computation batches and explicitly configured aggregate HTTP budgets.

`batch_size=256` is a total-text window. With two verified usable replicas the
quotas are 128 each; with four they are 64 each. Neither count is hardcoded.
An explicit HTTP token budget can split a quota further. Bounded exact
preparation overlaps ready requests through one scheduler, using observed slots
and global admission limits. Prefixes and special tokens are counted once;
oversized inputs remain measurable without reaching embedding inference.

The release also exposes optional typed native timings and reduces generation
bookkeeping while preserving reasoning, structured tool calls and runaway
protection. See [native HTTP](../NATIVE-HTTP.md) and
[embedding contracts](../EMBEDDING-CONTRACTS.md).

## Actual qualification

Python deterministic verification ran on macOS with Python 3.11.15. The full
suite passed: **1164 tests**, no skips, 48.78 seconds. Ruff check and format check
passed for all 68 Python files. Four warnings come from deliberately unverified
loopback TLS fixtures; production TLS verification stays enabled.

The live server reported build **b11716-797de63b2**. Both EmbeddingGemma 2 BF16
replicas were already loaded on separate HIP devices, with Flash Attention
**off**, context/batch/microbatch 2048 and one slot each. Content and embedding
identity tags matched the earlier qualification. Profiles, identities and idle
slots were checked before and after; no server, model or consumer was changed.
The build string is an observation, not attestation of the executable bytes.

| Check | Actual result |
| --- | --- |
| Same 11 discriminating texts, 31–2000 exact tokens | Passed; unchanged UTF-8 hashes and token IDs |
| Native numerical checks | 77 comparisons passed: 22 same-replica repeats, 22 array/single, 22 cross-replica single and 11 cross-replica array |
| Native coordinate differences | Maximum absolute error **0**, including all cross-replica comparisons |
| Native unit norms | Maximum deviation `4.124e-8`, below `1e-6` |
| Clean-installed candidate wheel, historical SDK corpus | **551 outputs passed**, with ordered results and unchanged task prefixes |
| Public projection versus independently normalized same-response vectors | Maximum error `1.111e-16`, below `1e-12` |
| Initial window of 256 documents | Two embedding POSTs of 128 texts each, one per replica |
| Each initial HTTP token total | 3968; accepted above native `n_batch=2048` |
| Concurrent 255 documents and three queries | Passed; odd quotas, order and repeat comparisons preserved |
| Exact input boundary and recovery | 2048 accepted; 2049 measured and rejected before any embedding POST; subsequent valid input passed |
| Effective embedding concurrency | Peak two overall, one per alias; independent active HTTP intervals |
| Drained close | Zero active requests, queued work or in-flight work |

Thresholds were unchanged: maximum component error `1e-5`, minimum cosine
`0.999999`, native norm deviation `1e-6` and same-response projection error
`1e-12`. Numerical failures were not retried. All previously failing
discriminating texts now pass. The operator changed both the server build and
Flash Attention; this is not a single-variable experiment identifying which
change removed the divergence. It does not establish universal equivalence for
all models, inputs, hardware or future builds, or measure retrieval quality.

The native experiment used 46 embedding POSTs, 22 tokenization POSTs and 12 GETs
in 4.101 seconds. The SDK run used 13 embedding POSTs, 520 tokenization POSTs and
159 metadata requests in 7.107 seconds. For its first 256 documents, exact
preparation took 1.420 seconds and embedding prepared inputs took 1.142 seconds.
These are client wall times including preparation, scheduling, HTTP and server
work, not pure GPU measurements or a controlled throughput comparison.

Cancellation, timeout, partial failure, topology changes and concurrent drained
shutdown are covered by deterministic fixtures. No live process interruption,
model reload or GPU cancellation experiment was performed in this qualification.

Private evidence SHA-256 references:

- Native results: `202feae987936b039ea6a0971a295708574a8a50cf25c275a55ec6b6c377f586`.
- Deterministic results: `64471bdb8dddba0ada84f09646f22164a5616a29c95390da5bc2ee9534905361`.

## Distribution and boundaries

The release provides the Python wheel, normalized source archive, three optional
helper archives, `RELEASE-MANIFEST.json` and `SHA256SUMS`. The manifest binds the
published source commit and exact artifact hashes. Packaging uses
`SOURCE_DATE_EPOCH=315532800`, normalized owner/timestamp metadata and retained
project and upstream licenses. No deployment configuration, credentials,
private endpoint, local evidence dump or consumer database belongs in the assets.

The live replay used the clean-installed candidate wheel. Final packaging is
checked against that exact qualified Python runtime payload, with a fresh
installed-wheel import/API/inert-factory smoke outside the checkout. Public
documentation metadata can change wheel hashes without changing the qualified
runtime files; use the release manifest for final download hashes.

The tsnet helper source and executable bytes are unchanged from 0.6.0. Its
macOS ARM64, Linux AMD64 and Windows AMD64 executables retain their original
0.6.0 build provenance and attribution inventory; the 0.7.0 archives explicitly
identify this reuse. These are not new Linux/Windows execution or real VPN
qualification claims. Historical native tests and crossbuild limits are recorded
in the [0.6.0 report](PUBLIC-RELEASE-0.6.0.md).

This release does not claim new GitHub CI coverage, live Tailscale/Headscale
enrollment or application acceptance. Consumer installations, jobs, databases
and services were not migrated or restarted. The operator retains responsibility
for the chosen runtime, model loading and deployment qualification.
