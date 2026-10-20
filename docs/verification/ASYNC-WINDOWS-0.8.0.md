# Async embedding windows: local 0.8.0 qualification

Verified on 2026-10-20. This is a local candidate delivery; no public release,
shared installation or consumer migration is performed by this work.
Flower MCP project `llama-runtime`, change `CHANGE-000011`, packet
`PACKET-000012`, run `RUN-000006` govern the work. Requirements are
`REQ-000016`–`REQ-000018`. Implementation, verification and human acceptance
remain separate.

## Delivered contract

`astream_texts()` and `astream_prepared()` return the exported, typed
`EmbeddingWindowStream`, an async iterator and context manager. Each output is
one complete source window in source/input order. The synchronous methods retain
their signatures. See [API, ownership and example](../ASYNC-EMBEDDING-WINDOWS.md).

Intake is bounded per stream and across the adapter before a source pull.
Unconsumed completed results retain capacity. The bounded coordinator bridge
uses the existing execution engine and HTTP scheduler, including discovery,
weight identity, exact tokenization, topology refresh and global/per-alias
limits. Finite per-window deadlines include source and admission waits.
Cancellation and close drain the real source/worker work without blocking the
event loop. No consumer storage, AST or splitting functionality was added.

## Deterministic and package verification

| Check | Result |
| --- | --- |
| Dedicated async contract cases | 49 passed; final focused run 1.66 s |
| Full test suite | 1,213 passed in 53.50 s |
| Ruff check / format check | Passed; 70 files formatted |
| Wheel and source build | Passed with the existing build backend |
| Clean installed wheel | Passed outside the checkout; 51 public exports |
| Initial wheel content audit | 23 checks passed; all 26 runtime files match source |

The full suite's four TLS warnings come from intentional unverified loopback
fixtures. One initial package-boundary assertion still expected 0.7.0; it was
updated before the successful full run. No production change was needed for
that failure.

Deterministic cases cover slow/out-of-order windows, empty and odd remainders,
multiple internal batches, source mutation, prepared reuse, oversized input
rejection, source/HTTP cancellation, repeated cancellation during drain,
finite source/admission deadlines, concurrent streams and synchronous callers,
topology/identity changes and close races. During review, `FIND-000003` caught
premature release of source-pull capacity and `FIND-000004` caught cleanup
masking a primary error. Both were corrected and retained as regressions.

The isolated dependency copy includes complete distribution records, including
the character detector's companion extension. No shared environment is updated.
Runtime dependencies remain `requests` and `urllib3`. Licenses, NOTICE and the
vendored MIT license remain intact. No private configuration or captured live
data is packaged.

## Loaded deployment and bounded workload

Two verified replicas of EmbeddingGemma 2 BF16 were already loaded on separate
`ROCm0`/`ROCm1` devices, with identical content and semantic identity tags.
The observed server build is `b11716-797de63b2`, Flash Attention off,
context/native batch/microbatch 2048, and one slot per replica. No build,
runtime, model, loading profile, consumer, database or job was modified.
The standard connection-close transport policy was retained.

The first attempt stopped at a refused health connection and sent no embedding
or tokenizer requests. After the operator restored the router, a fresh
GET-only snapshot showed unchanged inference settings and identity; child
ports and ephemeral media markers changed. A successful run then checked the
profile before and after inference.

The client used `batch_size=256`, request parallelism two, queue size two and
tokenizer parallelism two. Explicit ceilings were 30 embedding POSTs,
600 tokenizer POSTs and 240 seconds overall; individual windows had a finite
60-second budget. There were no forced live cancellations or automatic
inference retries.

The workload contains 530 SDK outputs plus 22 direct native reference outputs:

- Async document windows of 256 and 255 unique texts. The first 256 are
  byte-identical to the historical corpus; the next 255 use new indices from
  the same generator to make concurrent response association unambiguous.
  This is not an identical replay of the older 551-output workload.
- Three query preparations reused through `astream_prepared()` with zero
  additional tokenizer requests during reuse.
- Eleven frozen historical discriminants, retaining their exact text/token
  IDs, delivered through a raw async window. After SDK work drained, all eleven
  were submitted once to each alias as independent native references.
- Exact 2048-token admission, measurable 2049-token rejection before embedding
  HTTP, recovery, and raw/query/document prefix checks.

## Live results

| Observation | Measured result |
| --- | --- |
| Structural / numerical qualification | Passed / passed |
| SDK outputs / direct native references | 530 / 22; all dimension 768 |
| Embedding / tokenizer / metadata HTTP requests | 15 / 531 / 184 |
| Whole qualification wall time | 6.840 s |
| First 511-document phase wall time | 4.209 s |
| First two windows: HTTP text counts | 128, 128, 127, 128 |
| Corresponding exact aggregate token counts | 3968, 3968, 3937, 3968 |
| Embedding HTTP peak / per-alias peak | 2 / 1 |
| Tokenizer HTTP peak | 2 |
| Preparation intervals / summed duration | 21 / 5.427 s |
| First 511 documents: combined preparation intervals | 2.678 s, overlapping inference |
| Summed embedding / tokenizer HTTP duration | 6.662 s / 4.840 s |
| Event-loop maximum heartbeat gap | 0.082 s; coarse guard 1 s |
| Active HTTP, queued/in-flight scheduler work at finish | 0 |

The consumer paused until responses for both document windows had returned:
only two source pulls occurred and zero public windows had been consumed.
The adversarial completed-result/drain boundaries are additionally tested
deterministically. Concurrent HTTP intervals prove overlapping requests to both
aliases, without establishing simultaneous physical GPU execution.

The first document window retained 128-text quotas for both replicas. Per-input
context, native computation batch sizes, source windows and aggregate HTTP token
budgets remain distinct; no hardcoded quota was introduced. The largest native
reference payload contained 9099 aggregate tokens, each input within 2048.

All numerical thresholds remain unchanged:

| Metric | Threshold | Observed worst result |
| --- | --- | --- |
| Component absolute difference, 33 comparisons | ≤ 1e-5 | 5.56e-17 |
| Cosine similarity, 33 comparisons | ≥ 0.999999 | 0.9999999999999998 |
| Native reference unit-norm deviation | ≤ 1e-6 | 4.13e-8 |
| Same-response SDK projection error | ≤ 1e-12 | 1.12e-16 |

All eleven raw cross-replica reference vectors were component-identical.
The 33 comparisons include 22 SDK/native and eleven raw cross-replica checks.
The exact boundary input was accepted; the oversized input caused zero embedding
POSTs. Prefixes and special tokens were counted once, with preserved output
order. These checks do not constitute retrieval-quality evaluation.

Durations include client preparation, scheduling, network and server work.
Summed overlapping intervals must not be added to wall time or described as
pure GPU time. No controlled speedup or latency guarantee is claimed.
These documents are short (31 tokens each including prefix/special tokens).
Their phase performs 511 exact remote tokenizations and four embedding POSTs.
That preparation overhead matters for short inputs. Consumers that already
hold valid same-adapter preparations can stream those records without counting
again; the three-query reuse check confirms zero additional tokenizer calls.
The size of the timing benefit for 511 prepared documents was not benchmarked.
Live qualification is scoped to this loaded profile; cancellation races are
qualified with deterministic fixtures. Consumer acceptance and other operating
systems/models/deployments are outside this result.

## Evidence binding

The qualified code revision is `170bd88867511d224f71feb8f8e6147236ee163e`.
The preliminary clean-installed wheel SHA-256 is
`09abba5755c79d2845368b22173647ddb43e2a07773c4e651dbbf21bda82be9d`.
The final delivery can update documentation metadata while retaining the exact
same 26 runtime files; the delivery manifest binds the final wheel and confirms
that equality. Public runtime-file hash-map digest:
`6dae3d5fb92efcac63591d1c9f131d262dc9f233cae9007ed8c39a672999c95a`.

Private retained evidence has these SHA-256 values:

- Successful live result: `4c1b68ff4b639940a2036307c5b08ce70f54166db913db873da84f933026eb92`.
- Live evidence manifest: `8460628bf4c5120d0d40e4e5bdd16667837c84200defbbb18bc426d7eefe5dda`.
- Full public-window vector capture: `551cb2809767645d1e397fa9996f890a4af81d1a79e8e10b807969b3ce37696d`.
- Full required-check results: `f6023a45f7eecd446a7d0f5e886f6de9e85a6ec3c90b2b786573d0c7a4cebb8a`.
- Independent offline live audit: `aa65f5595fca87e0aabe1496cf4b97e4989fb8bd3ce906035671ad93a3947b5d`.

The independent offline audit checked 1466 captured files/hashes and recomputed
public-window order, response projection, all 33 comparisons, exact counts,
prefixes, boundary behavior and HTTP concurrency from retained payloads. It
reported no divergence from the qualification result and sent no requests.

Raw request/response JSON, tokenizer IDs, immutable vector captures, preparation
and HTTP intervals, source events, profile snapshots and independent audit
records remain in the private work area and Flower evidence references.
Captured JSON represents semantic request/response data, not original wire
byte encoding. Deployment addresses, local paths and credentials are excluded
from this public report and the delivered wheel.
