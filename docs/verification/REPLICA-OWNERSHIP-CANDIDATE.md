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
separate preparation/drain instrumentation, for **29 actual routes overall**.

Every occurrence was correlated by original window/record index, including
duplicates. All 16 SDK routes passed exact same-request scatter/projection,
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
replays, including the original stopped baseline route, exceeded them.
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
