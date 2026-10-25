# Replica admission and ownership — 0.14.0.dev1

The SDK implementation and deterministic/package verification are complete.
The four-replica live scheduling, ordering and drain checks passed. **Full
numerical qualification did not pass:** native repeated payloads also exceed
the existing thresholds on the previous SDK baseline. No numerical threshold,
model configuration or consumer was changed. This is a development candidate,
with no release, push, shared installation or application acceptance performed.

Flower MCP is the canonical lifecycle authority. This report is its readable
projection: architecture revision 5, `BOOT-000029`, requirements
`REQ-000046`–`REQ-000060`, packets `PACKET-000032`–`PACKET-000038` and milestones
29–35. External execution/evidence is host-reported; acceptance is not inferred.
The optional real-server extension is distinct from the native client work.

## Candidate and checks

| Item | Actual result |
| --- | --- |
| Source | `f83ff50fccd9a2b3882f379864a1e17390226c62`, branch `fix/stream-window-fairness` |
| Distribution / import | `runtime-llama` / `runtime_llama`, version `0.14.0.dev1` |
| Wheel | `runtime_llama-0.14.0.dev1-py3-none-any.whl` |
| SHA-256 | `2d4c6b04d53513a4eff8d620774bcd514e59772f577412e259f85b5fce1b85c1` |
| Full deterministic suite | **2,254 passed**, 77.04 s; four expected warnings from local HTTPS fixtures using test certificates |
| Ruff lint / format | Pass; 132 Python files formatted |
| Build | Wheel and source archive built without isolation |
| Clean wheel installation | Import, direct async embedding/order/bounds smoke and dependency check passed outside the repository; no shared environment update |
| Payload audit | All packaged Python modules match source; 52 wheel members; public 2,500-text synthetic corpus and licenses preserved; no private work corpus/configuration included |

Deterministic regressions cover N/G combinations, odd remainders, fewer inputs
than replicas, explicit identity validation and alias deduplication, concurrent
owners, cross-window packing, topology replacement and scheduler submit failure.
Resource cases cover pre-copy intake, cold response materialization, reserved
oldest-owner progress, rejection-credit growth, acknowledged synchronous result
transfer, exact-preparation accumulation and actual HTTP/error-body ownership.
Cancellation includes repeated interrupts, shared-owner withdrawal, close races,
sealing after expiry and a forced worker-dimension publication race. The latter
now publishes width only through the coordinator's completed-response event,
so a later window cannot seize credits required by an earlier delayed window.

These checks do not certify a model's numerical behavior or accept CC ingestion.

## Delivered contract

- One original embedding lease per canonical verified loaded replica. A root
  POST and its serial isolation descendants keep that lease through drain.
- Absent/`None` global guard uses N replicas; a persisted positive G remains
  `min(N,G)`. Native slots, batch and microbatch do not multiply HTTP permits.
- `batch_size` is a total text budget. With B=256 and N=4, quotas cap at 64;
  G=1 changes overlap, not that width. Explicit payload/token policies can
  require additional POSTs. No required fill timer or parallel scheduler.
- Fixed owner H precedes sync snapshot/registration and async source pulls.
  Its default is `(configured positive G else 2)+queue_size`, minimum one;
  discovery never enlarges it. Input, working, results, cache, handoff and
  control resources have separate finite ownership, not one process RSS cap.
- Sync and async results remain credited through transfer. Bulk exact
  preparation reserves one unknown-size result accumulator; tokenizer workers
  remain parallel within it. Oversized inputs remain measurable within finite
  resources and are not thereby admitted to inference.
- Native control JSON is bounded before decode, including decompressed bodies.
  Borrowed decoder ownership is declared explicitly. Eligible singleton
  statusless boundaries reuse root props; fresh router checks remain.
- Query/document prefixes, automatic direct inputs, exact APIs, 80% category /
  90% mixed margins and consumer-owned splitting/AST/graph behavior are preserved.

### Public API and consumer example

The existing embedding signatures are preserved:

```python
service.embed_texts(texts, *, input_type="document", timeout=None, cancel=None)
service.astream_texts(windows, *, input_type="document",
                      max_pending_windows=2, timeout=None, cancel=None,
                      close_source=False)
service.diagnostics()  # new local-only counters; performs no HTTP

adapter.prepare_texts(texts, *, input_type="document", timeout=None, cancel=None)
adapter.embed_prepared(inputs, *, timeout=None, cancel=None)
```

`request_parallelism=None` is the automatic-service spelling;
`embedding_request_parallelism=None` is the adapter-profile spelling. Both
accept positive guards. `embedding_max_pending_owners` is the new optional
positive owner ceiling on either configuration surface.

```python
from runtime_llama import AutomaticEmbeddingService

service = AutomaticEmbeddingService({
    "base_url": "http://inference.example.invalid:8080",
    "model": "logical-embedding-name",
    "batch_size": 256,
    "request_parallelism": None,
    "embedding_max_pending_owners": 18,
})
try:
    vectors = service.embed_texts(["first document", "second document"])
    counters = service.diagnostics()
finally:
    service.close()
```

This example is a contract handoff, not a consumer migration or installation.
See [resource details](../CONCURRENCY-AND-DIAGNOSTICS.md),
[embedding APIs](../EMBEDDING-CONTRACTS.md) and
[the optional SDK](../SERVER-OBSERVATION.md).

## Live scope and measurement policy

Four loaded EmbeddingGemma 2 BF16 aliases had matching content and semantic
identities, two configured on each GPU. Build: `b11716-797de63b2`. Each reported
four native slots, context total 2048 / effective per-sequence 512, native batch
2048, microbatch 512 and flash attention off. The actual response width was 768;
`n_embd=512` in catalog metadata is not used as proof of output width.
The task was document, with `title: none | text: ` applied once.

The frozen private corpus manifest is
`f5a8cea46813fd1c9fbf5823bc0e1a2add68aa6fc040790658915ea6c171f52e`.
Its earlier Nomic token counts were not reused as Gemma fit authority. The new
automatic path made no `/tokenize` requests and did not calibrate implicitly.
Source text, source paths, endpoint addresses and raw vectors remain outside
the public repository and package.

The first strict campaign stopped after two routes at baseline/native numerical
failure; those results remain retained. A declared amendment completed 24
routes: B4/C4/CA/CA/C4/B4 at each of 511 and 2048 texts, each SDK route followed
by native replay of its actual payloads and per-alias sequence. B4 is immutable
`0.13.1.dev2` with explicit G=4; C4 is this candidate with G=4; CA is automatic.
Only completed, drained numerical failures could continue collection; other
failures would stop it. Pre/post properties and catalog stayed unchanged in
all amended routes. Three additional bounded SDK routes checked G=1 and
separate preparation/drain instrumentation, for **29 routes in that first collection**.

Every occurrence was correlated by original window/record index, including
duplicates. All 16 SDK routes in that collection passed exact same-request scatter/projection,
with max_abs=0 and cosine=1. Each route drained local HTTP to zero. Overlap is
client HTTP overlap, not a measurement of GPU execution or utilization.

### Interleaved SDK timings

Ranges are two runs per cell. Wall time includes networking, controls, JSON,
scheduling and ordered delivery; setup/warmup and final guards are separate.

| Texts | Variant | Client wall seconds | Embedding POSTs | Measured GETs | `/tokenize` | Peak POSTs |
| --- | --- | --- | --- | --- | --- | --- |
| 511 | Baseline G=4 | 1.607–1.610 | 8 | 21 | 0 | 4 |
| 511 | Candidate G=4 | 1.715–1.766 | 8 | 21–22 | 0 | 4 |
| 511 | Candidate auto | 1.529–1.646 | 8 | 21 | 0 | 4 |
| 2048 | Baseline G=4 | 6.459–6.544 | 32 | 66–68 | 0 | 4 |
| 2048 | Candidate G=4 | 6.286–6.506 | 32 | 67 | 0 | 4 |
| 2048 | Candidate auto | 6.278–6.476 | 32 | 66 | 0 | 4 |

511 inputs produced seven 64-text POSTs plus one 63-text POST. 2048 inputs
produced 32 POSTs of 64. Candidate per-alias peak was always one. G=1 used all
four aliases with the same eight POSTs, peak one and 3.121 s for 511 inputs.

The automatic 511 runs made **29 measured HTTP calls** each: eight embedding,
one health, eight catalog and twelve props. Setup/warmup added 44/45 calls;
five final probe guards give totals 78/79. Automatic 2048 runs made **98
measured calls**: 32 embedding, one health, 29 catalog, 36 props. With setup
45/48 and five final guards, totals were 148/151. Setup includes the deliberately
separate cold-width warmup and probe observations. These counts are not hidden
inside the timed ingestion result. Router freshness checks were retained;
this profile does not exercise the singleton duplicate-removal optimization.

No stable speedup is demonstrated by these limited comparisons. In particular,
the explicit-G 511 samples are slower than baseline. The combined candidate
cannot isolate a causal effect for every safety change. Alternate connection
policy performance and real VPN/proxy routes are not qualified here; close
remains the default.

### Separate local preparation and drain samples

Two additional automatic SDK runs measured aggregate local record preparation,
not remote tokenization. Their clock/CPU instrumentation was not retroactively
added to the paired samples above.

| Texts | Client wall | Local preparation wall sum/union | Inclusive worker CPU | Local close/drain |
| --- | --- | --- | --- | --- |
| 511 | 1.674 s | 5.91 ms / 5.91 ms | 6.06 ms | 0.42 ms |
| 2048 | 6.084 s | 19.12 ms / 19.12 ms | 19.51 ms | 1.88 ms |

There were 511/2048 local preparations, observed peak one, and zero tokenizer
HTTP calls. Native HTTP still peaked at four. CPU, cumulative HTTP durations,
wall unions and local preparation are different quantities; none is pure GPU
execution time. HTTP kind/phase metrics remain fixed-cardinality; lifecycle
metrics in the private receipt include initialization and warmup explicitly.

## Numerical outcome and remaining authority

Thresholds remain **max_abs <= 1e-5, cosine >= 0.999999**. All thirteen native
replays in the first collection, including the original stopped baseline route,
exceeded them.
In the amended native comparisons, worst max_abs was approximately
`7.76042e-4`, minimum cosine `0.999982310`. Cross-version/repeated-baseline
non-self comparisons also failed; their worst max_abs was `0.001331045`, minimum
cosine `0.999951878`. Native-only replay and repeated previous-version results
show this is not an SDK scatter corruption. The underlying numerical mechanism
was not isolated by this campaign.

`FIND-000007` remains open. Historical missing absolute-error audits remain
missing. The current model profile receives no full numerical pass; no model
or consumer was reconfigured to obtain one. Milestones 29–33 have completed SDK
implementation and deterministic/package evidence. Milestone 34 records the
failed strict live numerical gate together with the passed scheduling/order
checks; it must not be accepted as a full qualification. Runtime library
stability, model-profile reproducibility and application retrieval acceptance
remain separate decisions.

Milestone 35 has a tested optional client/provider contract, with no assumed
native endpoint and no ambiguous POST replay. A real coherent snapshot and
server-enforced load/configuration generation at POST admission still require
separate server-owner implementation/authorization and live evidence. No actual
server support is inferred from the SDK tests, and this optional work does not
block the native implementation.

Publication/push and consumer updates were not performed. CC ingestion,
persisted recommendation provenance, lossless splitting and graph/application
acceptance remain owned by the consumer's separate lifecycle.


## Follow-up completion audit

A requirement-by-requirement audit confirmed the native SDK implementation,
the original full test receipt and the immutable wheel/source module hashes.
The numerical block was rechecked independently of the SDK coordinator before
performing the remaining separate performance measurements. No executable
source, wheel, model, consumer or shared installation was changed.

### Identical native requests, without the SDK scheduler

A fixed 64-input payload was sent directly three times on each of four aliases
serially, twice concurrently on all four aliases, and as a singleton three
times on the first alias: **23 native POSTs**. Observed pre/post properties and
catalog stayed unchanged. Eight out of eight same-alias serial-repeat audits
failed (worst max_abs `0.000356677`, minimum cosine `0.999996399`); two out of
two singleton-repeat audits failed (max_abs `0.000291251`, minimum cosine
`0.999997034`). The existing thresholds were used throughout.

An independent audit also verified all 240 SDK/native payload pairs from the
amended campaign, representing 480 POSTs: aliases, JSON values, input order and
output indices matched. For one identical native-to-native baseline body,
61 of 64 rows matched exactly and three differed. The worst row had raw delta
`0.000367528`; it was also the nearest corresponding row by a wide margin,
excluding a simple row permutation. Wire bodies were reconstructed offline
from captured request dictionaries; no claim of packet capture is made.

This establishes a native repeatability failure without requiring SDK
scheduler, scatter or HTTP concurrency as its cause. It does not identify a
specific GPU, kernel, compiler flag or driver defect. Runtime-owner action is
needed before a full numerical pass is possible under the current constraints.

### Isolated scheduler component and remaining integration

A private experimental copy of immutable dev2 replaced **only**
`embedding_scheduler.py` with the candidate module; 47 other files were
unchanged. This variant (SS4) is not a package release or a production legacy
mode. Its offline G=1/N=4/B=256 check changed a single 256-text quota into four
64-text quotas, preserving order, occurrences, no-tokenizer behavior and drain.

The declared live sequence was B4/SS4/C4/C4/SS4/B4 for both corpus sizes, each
followed by exact native payload replay: 24 additional routes. B→SS isolates
that scheduler component. SS→C is a residual integration delta containing
resource/control work **and** engine fairness, worker and telemetry changes;
it is not a clean causal estimate of resource-bound overhead alone.

| Texts | Variant | Two SDK wall samples, seconds | POSTs | GETs |
| --- | --- | --- | --- | --- |
| 511 | B4 | 1.6094, 1.6998 | 8 | 21, 21 |
| 511 | SS4 | 1.5977, 1.6036 | 8 | 21, 21 |
| 511 | C4 | 1.6304, 1.6275 | 8 | 22, 22 |
| 2048 | B4 | 6.6160, 6.0962 | 32 | 65, 69 |
| 2048 | SS4 | 6.5085, 6.4169 | 32 | 66, 67 |
| 2048 | C4 | 6.2997, 6.4820 | 32 | 69, 68 |

Every SDK scatter passed exactly; all native numerical replays failed the
unchanged gates. Concurrency peaked at four, with zero tokenizer calls. The
small, overlapping timing ranges do not demonstrate a stable improvement.

### Connection policy measured separately

The candidate-auto sequence close/reuse/reuse/close used 511 inputs and an
immediate native replay for each route: eight additional routes. Defaults and
server configuration were unchanged; no failure was retried with another
policy. SDK wall samples were 1.607/1.606 s for close and 1.636/1.709 s for
reuse. Each used eight embedding POSTs plus 21 GETs, with peak four.

Response-header hooks observed 29 distinct socket objects across 29 measured
SDK responses under close, versus 20 under reuse. All eight native embedding
responses used distinct socket objects even in the reuse samples. This is
socket-object observation, not TCP packet capture. Some reuse occurred, but no
speed benefit or real VPN/proxy qualification is established. Numerical replay
failed under both policies. The default remains close.

### Bounds, controls and decoder work measured separately

Two more candidate-auto SDK routes and native replays instrumented the actual
scheduler, snapshot/result/handoff ledgers, metadata methods and JSON-reader
threads: four additional routes. Instrumentation ran only inside the probe
process; it did not rewrite installed SDK source. Wall includes waits; CPU
includes instrumentation/nesting. Overlapping or nested buckets must not be
summed into a total overhead or subtracted as pure GPU time.

| Component | 511 texts: calls / inclusive CPU | 2048 texts: calls / inclusive CPU |
| --- | --- | --- |
| Quota planning | 2 / 0.022 ms | 8 / 0.053 ms |
| Scheduler dispatch | 8 / 0.136 ms | 32 / 0.440 ms |
| Snapshot-credit checks | 513 / 0.964 ms | 2056 / 2.496 ms |
| Result-credit checks | 1225 / 3.524 ms | 4833 / 12.805 ms |
| Result-transfer passes | 628 / 1.018 ms | 2462 / 3.686 ms |
| Handoff reservations | 8 / 0.307 ms | 32 / 1.361 ms |
| Catalog-control method | 7 / 1.310 ms | 30 / 6.429 ms |
| Props-control method | 8 / 1.244 ms | 32 / 7.664 ms |
| Control body read/decode | 21 / 2.239 ms | 68 / 7.607 ms |
| Embedding body read/decode | 8 / 139.943 ms | 32 / 539.378 ms |

The instrumented SDK wall times were 1.649 s and 6.377 s. Catalog-control wall
sums were 104.654/579.761 ms; props-control wall sums 62.451/1036.660 ms;
embedding body read/decode wall sums 542.758/2498.837 ms. These are inclusive
function observations, not a differential proof of each safety change's cost.
Exact output order and drain passed; native repeatability still failed.

The follow-up adds **36 SDK/native routes** to the first 29, plus the separate
23-POST native audit. MILE-000034 remains numerically unqualified; this audit
does not turn the failed gate into acceptance. No further identical runs or
SDK threshold changes would resolve that external failure. The operator has
been asked for a build/profile meeting the existing native repeatability gate.
The optional real-server extension remains outside authorized server changes.

The final source archive is regenerated after this report, so its documentation
matches the recorded repository revision. It has its own delivery/audit receipt;
the already qualified wheel and its hash remain unchanged. No release or push
is implied by this archive refresh.
