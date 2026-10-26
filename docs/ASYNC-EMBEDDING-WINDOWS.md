# Asynchronous embedding results and windows

Version 0.15.0 adds flat progressive text input while preserving the existing
complete-window APIs. All paths use the same persistent engine and replica
scheduler. Neither input cardinality nor loaded-model count has an implicit
application maximum.

## Progressive text interface

```text
service.aiter_texts(texts, *, input_type="document", timeout=None,
                    cancel=None, close_source=False) -> EmbeddingRecordStream
adapter.aiter_texts(texts, *, input_type="document", timeout=None,
                    cancel=None, close_source=False) -> EmbeddingRecordStream
```

`texts` is a finite `Sequence[str]` or an `AsyncIterable[str]`. The stream yields
`EmbeddingRecordResult(index, embedding)` in source order. `index` identifies
an original occurrence, including duplicates; its vector belongs to the
consumer after transfer. A sequence of 10,000 texts is a supported workload,
not a new upper limit. No consumer-side grouping into POST-sized windows is
required.

```python
from runtime_llama import AutomaticEmbeddingService

async def ingest(texts, store_vector):
    service = AutomaticEmbeddingService({
        "base_url": "http://inference.example.invalid:8080",
        "model": "logical-embedding-name",
        "batch_size": 256,  # Optional target, not a maximum input count.
    })
    try:
        async with service.aiter_texts(texts) as stream:
            async for result in stream:
                await store_vector(result.index, result.embedding)
    finally:
        await service.aclose()
```

Results can arrive before an asynchronous source is exhausted. Internally,
correlation units do not define HTTP fragmentation: compatible records can be
packed together by the existing engine. The stream preserves ordering, so a
later completed result can wait for its predecessor. Waiting work and results
that the consumer deliberately retains require real memory; the API does not
promise constant RSS for unlimited input or accumulation.

The automatic service may deliver `EmbeddingChunkSplitRequiredError` for a
rejected occurrence. Its `window_index` is the global record index in this flat
interface; the nested input index and lossless spans remain relative to its
singleton source occurrence. The error keeps its original type and the next
`anext()` can continue with later records. An exception escaping an `async with`
body closes the stream; use an explicit iteration/catch loop when continuing
recovery. No successful inference is silently replayed.

## Complete-window interface

```text
adapter.astream_texts(
    windows, *, input_type="document", max_pending_windows=None,
    timeout=None, cancel=None, close_source=False,
) -> EmbeddingWindowStream

adapter.astream_prepared(
    windows, *, max_pending_windows=None,
    timeout=None, cancel=None, close_source=False,
) -> EmbeddingWindowStream

await stream.aclose()
await adapter.aclose()
```

The automatic service also exposes `astream_texts`. `windows` is an async
iterable of finite text sequences, or same-adapter preparations for
`astream_prepared`. It yields a complete `list[list[float]]` per source window,
including an empty list for an empty window. Window and vector order are
preserved. Mutation after admission does not alter the admitted snapshot.
A complete output naturally occupies its required memory; it is not rejected
because it exceeds the former 64 MiB whole-window reservation.

`max_pending_windows=None` imposes no per-stream count policy. A positive value
is an explicit consumer choice and covers pull, preparation, queued/active HTTP
and completed output awaiting delivery. This argument is independent of the
profile's capacity opt-in. There is no implicit default of two windows or a
mandatory global H derived from queue size.

Factories remain inert until entering or iterating the stream. A stream owns
at most one active source pull; concurrent consumer `__anext__()` calls on the
same iterator are rejected. The current adapter async bridge belongs to one
active event loop; synchronous callers can still share it from other threads.

## Batching and explicit capacity policies

`batch_size` is optional and distinct from source cardinality. If B is chosen,
its total text target is balanced across N verified replicas independently of
a positive global HTTP guard G. The existing one-original-lease-per-replica
policy remains, with serial recovery descendants on the same lease until drain.
Native slots, computation batch and microbatch do not multiply HTTP permits.

`embedding_capacity_limits=False` leaves legacy scalar memory/owner/queue
settings inactive. To request those policies, set the flag to `True` and supply
the desired settings. Omission does not select replacement values or automatic
budgets based on RAM. Source windows and flat records are delivery contracts,
not fixed HTTP batch boundaries. No batching timer is required.
See [concurrency and resource ownership](CONCURRENCY-AND-DIAGNOSTICS.md).

## Deadlines, cancellation and resource transfer

`timeout` is a finite work budget inherited from the profile unless explicitly
provided. For window sources it covers each window's pull, queueing,
preparation and HTTP; it is not a deadline for the whole indefinitely
productive stream. Readiness has its own finite operation budget. Completed
results do not expire merely because the consumer pauses.

Cancellation stops the affected owner's work, preserving survivors of a shared
POST. Engine Futures settle after real transport drain, not simply after a
waiter is cancelled. Repeated cancellation preserves the primary exception and
waits for owned cleanup. Source, backend and timeout failures keep their typed
contracts. If cleanup also fails, its failure is chained rather than replacing
the primary error.

After transfer or cleanup the library releases source/result/task/Future and
retired-generation references it no longer needs. Closed streams do not retain
borrowed sources solely because the iterator object itself remains reachable.
A cancelled owner does not close the shared adapter or another caller's work.

Use `async with` or explicit `aclose()` after an early break. Dropping a
progressive facade also signals Stop and schedules cleanup on a live owning
loop, including when HTTP is still running. Ownership remains until actual
drain. An already closed event loop cannot execute cleanup, and uncooperative
user code cannot be forcibly terminated safely.

Sources are borrowed by default: cleanup cancels and awaits its own source
pull but does not deliberately call the original source's `aclose()`.
`close_source=True` also transfers that close responsibility. Cancelling a
pending async-generator pull may itself execute its `finally`, so source reuse
is not promised after interruption.

`stream.aclose()` drains that stream; `adapter.aclose()` drains active streams
and the shared adapter. A synchronous `close()` cannot await source cleanup on
its own blocked event loop. Consumers retain source splitting, fragment IDs,
UTF-8 intervals, graph and storage.

The [0.15.0 verification](verification/PUBLIC-RELEASE-0.15.0.md) separates
actual tests and package results from model qualification and consumer acceptance.
