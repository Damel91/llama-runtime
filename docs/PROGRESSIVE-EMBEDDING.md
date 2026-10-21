# Progressive preparation and native token inputs

Contract for the local 0.10.0 candidate. Flower MCP governs requirements,
implementation and verification in project `llama-runtime`, packet
`PACKET-000015`. Qualification and consumer acceptance remain distinct.

The [persistent queue engine design](EMBEDDING-QUEUE-DESIGN.md) records the
architecture and verification requirements for replacing per-window
coordination and early batch selection.

## Progressive queues

The synchronous and asynchronous embedding APIs register work with the same
persistent execution engine and shared HTTP scheduler. Each verified replica
has a queue of prepared records, with separate ownership for independent
cancellation domains. Preparation publishes each record with its original
input index. Public windows are result boundaries; they do not create their
own blocking coordinator threads or determine the lifetime of the queues.

`batch_size` remains the total-text budget for dynamic replica distribution.
Each quota remains affine to its discovered alias. Preparation and ready
records accumulate within the adapter's admission and resource budgets while
earlier inference is running. A batch is selected only when global and
per-replica capacity are available; waiting work does not retain an already
frozen list of batch indices. A slow input cannot block other ready records.
Complete public windows and vectors retain input order.

`embedding_batch_wait_sec` is finite and nonnegative, default `0.0`. At zero,
a free replica dispatches eligible ready work without an artificial wait to
fill the batch. While it is busy, later records continue to accumulate for the
next dispatch. Sparse input or a slower tokenizer can still produce small
batches; zero waiting does not guarantee full batches or maximum throughput.

An explicit positive value remains an optional coalescing policy in the same
engine. Its budget starts at the first unsent ready record and is not reset by
new arrivals. A filled text/token cap, completed producer or expired wait can
flush the batch. The call/window deadline includes this waiting. This setting
does not change finite HTTP and overall deadlines.

Text caps come from a single distribution quota and adaptive limits, never
from a hardcoded 8/64/128 or the number of records that happen to be ready.
Accumulating multiple windows does not add their per-request text caps.
Native computation `n_batch`/`n_ubatch` remain separate from aggregate HTTP
request token limits. Request-size and timing measurements accompany
qualification.

The finite resource budgets are configurable:

| Setting | Default | Purpose |
| --- | --- | --- |
| `embedding_max_pending_bytes` | 67,108,864 bytes | Ceiling for the raw/counted queue pool and, separately, the completed async output pool |
| `embedding_max_pending_tokens` | 1,048,576 tokens | Bound prepared tokens queued or active for inference |
| `embedding_max_request_bytes` | 8,388,608 bytes | Bound individual request payloads |

All three require positive integers. Oversized embedding inputs remain
measurable with `prepare_*` within the configured resource ceilings, but are
not admitted for inference. The caller's original input and the complete
result required by the synchronous API remain proportional to that call.
Transient worker response handoffs have separate bounds from worker count
and native JSON response limits; they do not borrow the output pool's credits.
Async intake also bounds source pulls and
completed windows waiting for the consumer. Resource accounting does not
claim a fixed process RSS.

Tokenizer concurrency is bounded across the adapter; HTTP control work has a
separate bound, and total HTTP capacity is derived from the embedding,
tokenizer and control limits. Retained records and IDs have no unbounded
cross-call registry. Limits and identity are rechecked before inference.

An invalid input discovered after valid inputs were dispatched stops/drains the
window and returns an error. It is never sent for inference, and successful
earlier work is not replayed. No partial public window is returned. Preparation
still measures oversized inputs; known limits/identity are rechecked at
admission. Cancellation, finite deadlines, async backpressure and drained close
retain the existing contracts.

## Native token input

The native `/v1/embeddings` route accepts either texts or token sequences. For
several prepared inputs the client sends `input: [[id, ...], [id, ...]]`; each
inner array is one input, including a single-input request. This keeps output
cardinality explicit. Prefixed text is tokenized once with `add_special=True`
and `parse_special=True`, and the returned token IDs are retained immutably.
The native integer-array path copies the IDs without tokenizing strings or
inserting another BOS/EOS. No prefix or synthetic special token is appended
when constructing that payload.

This behavior is supported by the pinned build's
[integer-input implementation](https://github.com/ggml-org/llama.cpp/blob/797de63b2fa7396dcc8caca76834c699ac84e1c3/tools/server/server-common.cpp#L1029-L1061)
and [official embedding tests](https://github.com/ggml-org/llama.cpp/blob/797de63b2fa7396dcc8caca76834c699ac84e1c3/tools/server/tests/unit/test_embedding.py).
Source inspection is complemented by explicit text/ID equivalence checks on
the loaded deployment; a server version string alone is not a binary audit.

`embedding_input_format` controls pre-dispatch selection:

| Setting | Behavior |
| --- | --- |
| `auto` (default) | Use IDs only with an explicitly declared native client contract and verified record identity, tokenizer build and policy; otherwise choose text before sending. |
| `text` | Send prepared strings; retain exact counts and admission checks. |
| `tokens` | Require safe native token input; fail with a typed error before the POST when its conditions are unavailable. |

The standard `LlamaCppClient` declares its native input contract. Custom clients
with unknown support remain on text in `auto`; the adapter does not infer
support from arbitrary truthy values or send hidden capability inference
probes. A nonstandard server can be configured explicitly for text input.
There is no automatic retry as text after an ambiguous token-input failure:
some sibling inputs may already have been evaluated.

Prepared records retain the existing public text/count/limit fields. Retained
IDs are covered by the same-adapter seal and can only be reused under the same
verified weights, tokenizer build and special/prefix policy. Stale/unverified
records are recounted under the existing policy. Empty tokenizer output remains
measurable as count zero; `auto` uses text for it and explicit `tokens` rejects
it before embedding HTTP. The client never manufactures a BOS token.

This removes the second internal tokenization when the native ID path is used.
New texts still require exact `/tokenize` requests. JSON ID arrays may be larger
than their original strings; bandwidth or latency improvements are measured,
not assumed. Source splitting, UTF-8 intervals and fragment identity remain
the consumer's responsibility.

## Configuration example

```python
from runtime_llama import build_embedding_adapter

adapter = build_embedding_adapter({
    "model": "embedding-model.gguf",
    "base_url": "http://localhost:8080",
    "embedding_replica_discovery": True,
    "batch_size": 256,
    "embedding_request_parallelism": 2,
    "token_count_parallelism": 2,
    "embedding_batch_wait_sec": 0.0,
    "embedding_max_pending_bytes": 64 * 1024 * 1024,
    "embedding_max_pending_tokens": 1024 * 1024,
    "embedding_max_request_bytes": 8 * 1024 * 1024,
    "embedding_input_format": "auto",
})
```

The existing `embed_texts`/`embed_prepared` and
[`astream_texts`/`astream_prepared`](ASYNC-EMBEDDING-WINDOWS.md) interfaces share
these settings. Configure model-specific prefixes and limits as described in
the [embedding contract](EMBEDDING-CONTRACTS.md).

The [0.9.0 coalescing comparison](verification/COALESCING-TUNING-0.9.0.md)
records earlier Nomic measurements. Its timer recommendation applies to that
engine and workload, not automatically to 0.10.0. Compare zero waiting and any
explicit coalescing value using the same model, build, profiles, corpus and
numerical thresholds; distinguish preparation, queue, HTTP and total times.
