# Progressive preparation and native token inputs

Contract for the local 0.9.0 candidate. Flower MCP governs requirements,
implementation and verification in project `llama-runtime`, packet
`PACKET-000013`. Qualification and consumer acceptance remain distinct.

## Progressive queues

The synchronous and asynchronous embedding APIs use the same execution engine
and shared HTTP scheduler. Preparation publishes each ready record with its
original input index. A batch can run before every text in its quota has been
tokenized; a slow earlier text does not prevent other ready records from being
processed. Complete public windows and their vectors retain input order.

`batch_size` remains the total-text budget for dynamic replica distribution.
Each quota remains affine to its discovered alias. Preparation, ready records
and HTTP work share the existing global admission bounds. Quota capacity is
held until both its producer and all its HTTP work drain. Tokenizer concurrency
is bounded across the adapter, and retained records/IDs have no unbounded
cross-call registry.

`embedding_batch_wait_sec` is a positive finite coalescing budget, default
`0.5` seconds, starting when the first unsent record becomes ready. New arrivals
do not restart it. A batch can flush when its derived text cap is filled, an
explicit HTTP token budget prevents another addition, preparation finishes or
the wait expires. While its replica is busy, ready records continue to
accumulate within the bounds. The call/window deadline includes this waiting.
This is a throughput/latency setting, not a guaranteed response time.

Text caps come from the planned quota/adaptive limits, never from a hardcoded
8/64/128 or the number of records that happen to be ready. Native computation
`n_batch`/`n_ubatch` remain separate from aggregate HTTP request token limits.
An undersupplied producer can still produce small batches; request-size and
timing measurements accompany qualification.

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
    "embedding_batch_wait_sec": 0.5,
    "embedding_input_format": "auto",
})
```

The existing `embed_texts`/`embed_prepared` and
[`astream_texts`/`astream_prepared`](ASYNC-EMBEDDING-WINDOWS.md) interfaces share
these settings. Configure model-specific prefixes and limits as described in
the [embedding contract](EMBEDDING-CONTRACTS.md).
