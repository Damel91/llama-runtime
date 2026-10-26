# Progressive preparation and native token inputs

Version 0.15.0 uses one persistent engine for blocking calls, complete-window
async results and flat progressive records. Flower's adopted architecture
revision 6, requirements `REQ-000061`–`REQ-000069` and milestones 36–40 govern
the resource-lifecycle changes. Earlier queue and metadata designs remain
historical context; the current [embedding contract](EMBEDDING-CONTRACTS.md)
and [capacity and ownership guide](CONCURRENCY-AND-DIAGNOSTICS.md) define use.

## Progressive queues

Each verified canonical replica has a ready queue. Compatible occurrences from
different callers or source windows can share a POST while preserving their
input index, result owner and independent cancellation. Windows are delivery
boundaries; they do not create independent schedulers or force POST boundaries.

`batch_size=B` is an optional total-text target distributed across N verified
usable replicas. A positive global G controls `min(N,G)` original POSTs while
quotas still use N. Each replica holds one original lease through actual drain,
including serial recovery descendants. Native slots do not multiply that
lease. Omitting B imposes no replacement text target; input cardinality is not
limited to B, 128 or a source-window size.

The engine composes a request when capacity is available, including work that
arrived during preceding inference. Incremental summaries avoid repeatedly
materializing the whole ready backlog. Packing uses iterators and constant-time
owner rotation; waiting does not freeze a candidate. Text targets, explicit
aggregate HTTP token policies, per-input context and native execution sizes
remain separate. A recognized custom-client 400 recovery subdivides its failed
POST; it does not establish a reduced cap for subsequent work.

`embedding_batch_wait_sec=0.0` fills using preparation/capacity events without a
required timer. Compatible pending preparation may fill a free replica's batch;
tails flush when further filling cannot progress. A positive finite wait is an
explicit coalescing choice in the same engine. Its clock starts at the first
unsent ready record and is not renewed by arrivals. Neither form extends the
owner's finite work deadline.

## Input, result and cache ownership

There is no default owner, queue-count, pending-byte/token, HTTP-body or handoff
quota. These scalar profile fields default to `None` and require
`embedding_capacity_limits=True` before supplied values activate. Legacy saved
numbers do not silently opt into policies. Explicit fields are independent;
there is no derived H, 64 MiB pool, 75/25 split, response `pool/16`, compulsory
whole-result reservation or memory-driven request halving. B and G retain their
separate batching and dispatch roles.

The complete-list and complete-window APIs naturally retain their complete
output until delivery. `aiter_texts()` instead transfers
`EmbeddingRecordResult(index, embedding)` occurrences progressively from a flat
sequence or async source. Its default window intake has no compulsory count
cap. A caller can select `max_pending_windows` on the window APIs explicitly.
See [async interfaces](ASYNC-EMBEDDING-WINDOWS.md) for delivery, failure and
source-ownership semantics.

Inputs, response objects, Futures, completed tasks and retired generations are
released when their owner transfers or drains them. Source order and duplicate
occurrences remain distinct. Explicit close is deterministic; dropping a flat
facade also signals Stop and schedules cleanup on its live owning loop. No
constant-RSS, automatic RAM-budget selection, disk spill or forced termination
of uncooperative source/provider code is promised.

Exact preparation caching is disabled by default. Explicit
`embedding_preparation_cache_enabled=True` enables adapter-session retention;
close clears it. Optional cache entry/byte quotas require the capacity opt-in,
and omitted quotas impose no replacement ceiling. Reuse still requires verified
content/semantic identity, tokenizer build, task, exact text and prefix/special
policy. It does not authorize inference or bypass public-record seals and
fresh dispatch checks. There is no TTL or singleflight; concurrent cold misses
can each tokenize, and failed tokenization cannot publish an entry.

## Preparation, controls and interruption

The automatic direct-text path performs no per-input `/tokenize` requests or
implicit calibration. It does not pretend that every input consumes its whole
context. Token usage is unknown for direct strings; an explicit pending-token
policy applies to exact counted/prepared work. The consumer may explicitly ask
for the character recommendation derived from the 2,500-text calibration;
category 80% and mixed 90% factors are unchanged.

Exact preparation remains available when a consumer requests real counts or
retained token IDs. Worker defaults follow the actual usable replica count;
explicit positive worker choices remain explicit. Metadata is acquired only
when required by the chosen mode. Coherent admission observations can be
reused within that boundary, while current identity and relevant limits are
rechecked before POST/recovery. A local observation is not a remote load lease.

Stopping one owner removes its pending work and discards its response portion
without aborting surviving owners of a shared POST. Transport has one finite
deadline fixed at dispatch; cancellation does not extend it. The retiring
owner completes only after actual transport drain, which can outlast its own
work deadline. All-member Stop or adapter close cancels and drains transport.
Successful inputs are not replayed.

An external exception abandoning a synchronous Future wait retires that owner,
wakes the engine and waits for real completion/drain before reraising the
original exception, including through repeated cleanup interruptions. Shared
survivors and the borrowed client remain open. Async source ownership, explicit
close and recoverable flat-record errors follow the same engine lifecycle.

## Native token input

The native `/v1/embeddings` route accepts texts or token sequences. Several
prepared inputs use `input: [[id, ...], [id, ...]]`; each inner array represents
one input. Exact prefixed text is tokenized with `add_special=True` and
`parse_special=True`. Retained immutable IDs already contain that policy;
payload construction adds neither another prefix nor synthetic BOS/EOS.

The inspected [integer-input implementation](https://github.com/ggml-org/llama.cpp/blob/797de63b2fa7396dcc8caca76834c699ac84e1c3/tools/server/server-common.cpp#L1029-L1061)
and [upstream embedding tests](https://github.com/ggml-org/llama.cpp/blob/797de63b2fa7396dcc8caca76834c699ac84e1c3/tools/server/tests/unit/test_embedding.py)
are versioned source references. Loaded-deployment equivalence requires its own
qualification; a server version string is not a binary audit.

| `embedding_input_format` | Behavior |
| --- | --- |
| `auto` | Use IDs only with declared native client support and verified identity/build/policy; otherwise select text before sending. |
| `text` | Send strings while preserving the exact preparation/admission contract when requested. |
| `tokens` | Require reusable IDs or raise a typed error before embedding dispatch. |

Unknown custom clients use text in auto mode. Empty tokenizer output remains
measurable as zero; auto uses text and explicit tokens rejects it. No
capability inference POST, ambiguous replay or automatic retry as text is
introduced. JSON ID arrays may exceed string payload size; reduced latency or
bandwidth must be measured. Source splitting and fragment identity remain
consumer responsibilities.

```python
from runtime_llama import build_embedding_adapter

adapter = build_embedding_adapter({
    "model": "embedding-model.gguf",
    "base_url": "http://inference.example.invalid:8080",
    "embedding_replica_discovery": True,
    "batch_size": 256,
    "embedding_request_parallelism": None,
    "embedding_input_format": "auto",
})
try:
    vectors = adapter.embed_texts(["first document", "second document"])
finally:
    adapter.close()
```

[Release verification](verification/PUBLIC-RELEASE-0.15.0.md) records executed
checks separately from live model qualification and consumer acceptance.
