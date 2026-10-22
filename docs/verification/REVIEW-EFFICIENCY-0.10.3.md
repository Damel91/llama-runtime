# Incremental efficiency review qualification 0.10.3

Date: 2026-10-22. Local candidate `runtime-llama` 0.10.3; import
`runtime_llama`. Implementation commit: `8e6e638`. Public release 0.10.2
remains unchanged. Flower MCP retains the work through `BOOT-000018`,
`REQ-000037`, `MILE-000020`, `CHANGE-000020` and `PACKET-000021`.
Results are host-reported verification, without consumer acceptance.

## Applied changes

| Review item | Implemented behavior |
| --- | --- |
| R01 | Each pending recovery record is checked against its own historical limits and the full fresh observation, including context, HTTP budget, tokenizer build, vocabulary and identity. Owner-local failure removes only that owner's rows. Children preserve original indices and successful children are never replayed. |
| R02 | `embedding_max_response_bytes` independently caps decoded native JSON responses before materialization, default 8 MiB, positive integer. Request, pending-record and result budgets stay separate. |
| R04 | `embedding_preparation_cache_entries` and `embedding_preparation_cache_bytes` configure exact-token retention, defaults 1,024 / 8 MiB. Effective bytes remain `min(configured_cache_bytes, pending_bytes // 4)`; zero entries or bytes disables retention. Identity, policy, vocabulary, seals and fresh admission checks remain unchanged. |
| R05 | Remembered wake notifications and relevant deadlines replace the fixed global 10 ms poll. Arbitrary cancellation callbacks and custom coalescing clocks retain finite checks. Diagnostic snapshots publish only when observable values change. |
| R06 | Unverified/no-dedup plans create index groups without computing keys or a hash map. Coordinator-owned retained-plan/preparing counters replace repeated scans. Modular iteration retains rotation and tie-break order without rotating list copies or `list.index`. |
| R03 | Existing opt-in HTTP reuse is exercised on the same current endpoint and corpus; the probe now selects/reports `--connection-policy close|reuse`. There is no default change, silent fallback or retry. |

The R01 reproducer failed before the fix: record A had one token and a
historical context limit of two; record B had 400 tokens and historical limit
2048. After the first shared POST returned a recognized capacity error, fresh
context 100 was hidden by the old aggregate minimum of two. The corrected
test rejects only B before a child can send it, while A completes. Further
tests cover vocabulary/build/identity changes, retired owners, child-local
HTTP budgets, a failed child after a successful child, typed errors and actual
shared-lease drain. This recovery applies to eligible injected clients. The
native client still refuses replay/splitting of failed or ambiguous POSTs.

The response-budget tests accept a valid 1,536-dimensional response larger
than a 4 KiB pending-input budget while the outgoing request stays within
256 bytes. Other fixtures cover decoded gzip overflow, incomplete framing,
slow drip, cancellation/deadlines, healthy concurrent owners and no replay or
partial publication. Caller-owned synchronous output memory is not presented
as bounded by the pending-input ceiling.

Coordinator regressions verify a wake immediately before waiting, finite
unnotifiable cancellation/deadline checks, coalescing clocks, saturation,
topology, idle resume and concurrent close. With one blocked HTTP request and
no callback/coalescing timer, the captured next wait exceeds four seconds
rather than polling every 10 ms. This demonstrates removal of the periodic
work; it is not a CPU-percentage or GPU-throughput benchmark. Incremental
plan/preparing counters are compared with actual owned plans during execution.

R07 already has native versus end-to-end timing fields in the library; consumer
presentation remains outside this change. R08's speculative SSE rewrite is
not applied. No second scheduler, generalized retries, global cache or alternate
inference path was introduced.

## Verification and local package

The final source passed **1,578 tests in 64.51 seconds**, Ruff check and format,
wheel/source builds, isolated installed-wheel import/version checks,
dependency consistency and inert adapter construction/drained close outside
the checkout. The four warnings come from intentional loopback TLS fixtures.
No new Linux/Windows execution or real VPN qualification is claimed.

The installed live-qualified wheel SHA-256 is
`54f28be5c5d51241df267609a42d1cf57de01fa6031d5a422d8ce42990d3b013`.
All **28 runtime payload files** match source. Final documentation and archive
metadata may change delivery hashes; the delivery manifest/checksums identify
the final files and require byte-identical runtime payload to this candidate.

## Bounded live HTTP-policy pair

Both runs used already loaded, verified EmbeddingGemma 2 BF16 replicas,
build `b11716-797de63b2`, context/native batch/microbatch 2048, one slot each,
Flash Attention off. Complete profiles matched before/after each run and
across this pair. The retained 0.10.2 run has different opaque argument/property
digests, so these measurements are not a controlled version-speed comparison.
The code did not administer or reload models or modify consumers/services.

Each pipeline used 511 unique texts in eight windows (64 x 7 + 63), total-text
batch budget 256, global embedding concurrency 2, tokenizer concurrency 2,
queue size 2, four pending windows and default zero coalescing delay.

| Pipeline measurement | `close` | `reuse` |
| --- | --- | --- |
| HTTP requests | 539 | 539 |
| Tokenize / embedding / metadata | 511 / 4 / 24 | 511 / 4 / 24 |
| POST sizes in dispatch order | 128 / 128 / 127 / 128 | 128 / 128 / 127 / 128 |
| Wall time | 4.8449 s | 4.6630 s |
| Sum of overlapping preparation-worker wall times | 5.3162 s | 5.1289 s |
| Peak embedding / tokenizer / all HTTP | 2 / 2 / 3 | 2 / 2 / 3 |
| Full probe, including canaries/oversize/reuse phases | 7.9325 s | 6.4185 s |

Each replica carried at most one active embedding request. Both policies used
the same exact corpus and identities with no fallback, and all queues, HTTP,
metadata and retained credits drained. These single measurements qualify the
current plain-HTTP endpoint functionally; they do not establish a general
latency improvement. Actual live TCP connection counts were not instrumented.
Loopback tests verify exclusive drained socket reuse, concurrent distinct
sockets, peer FIN/close handling, framing, TLS, cancellation and deadlines.
Real HTTPS proxies and VPN routes require their own qualification.

The 511 vectors matched between policies and against retained 0.10.2 outputs
with zero coordinate difference. Thresholds are unchanged: dimension 768,
maximum absolute error `1e-5`, minimum cosine `0.999999`, norm error `1e-6`,
and same-response projection error `1e-12`. The oversized input was measured
and rejected without inference. In each separate reuse exercise, eight texts
were tokenized once; two prepared-input calls and two raw repeat calls each
added zero tokenization requests. The full probe used 628 requests per policy:
536 tokenizer, 16 embedding and 76 metadata, including those extra phases.

New unique inputs still need exact native tokenization; these changes do not
invent a bulk tokenizer API or substitute character ratios. All reported
times include client work, transport, preparation, server scheduling and
result handling; they are not pure GPU timings.

Corpus SHA-256:
`a723a493887bf14096265c52bb1f09f2171e1a13efb198b8e10f572cad4d7252`.
Both vector-evidence files SHA-256:
`75279fe8fad2513d0470ad8176325971dfe3c7111c13d5522fa0b67c53ba7ccd`.
Private endpoint configuration and raw evidence remain outside the repository.
