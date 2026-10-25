# Asynchronous embedding windows

Public interface introduced in the local 0.8.0 candidate and retained in
version 0.10.2.
The [progressive queue and native token path](PROGRESSIVE-EMBEDDING.md) now run
inside the same shared execution engine. The synchronous embedding API remains
available. Flower MCP governs requirements, the work
packet and verification; implementation completion is separate from acceptance.

## Public interface

```python
adapter.astream_texts(
    windows, *, input_type="document", max_pending_windows=2,
    timeout=None, cancel=None, close_source=False,
) -> EmbeddingWindowStream

adapter.astream_prepared(
    windows, *, max_pending_windows=2,
    timeout=None, cancel=None, close_source=False,
) -> EmbeddingWindowStream

await stream.aclose()
await adapter.aclose()
```

`windows` is an `AsyncIterable` of finite sequences: strings for `astream_texts`,
or same-adapter `PreparedEmbeddingInput` records for `astream_prepared`.
The stream is an async iterator and async context manager. It yields one
`list[list[float]]` per source window in source order, with vectors in input
order. Empty windows yield an empty list. A window is copied at admission;
mutation by its producer after delivery does not change admitted work.

Factories are inert. Async resources, source iteration, threads and HTTP start
only when entering or iterating the stream. Each stream belongs to its active
event loop and has at most one pending `__anext__` on its source. Async use of
one adapter is confined to one active loop; synchronous callers may still use
the same adapter from other threads. Concurrent consumer `__anext__` calls on
one stream are rejected. Exhaustion and errors close the stream.
Use `async with` or explicit `aclose()` for deterministic cleanup after an early
`break`; garbage collection is not a cleanup guarantee.

## Backpressure and scheduling

`max_pending_windows` is a positive integer, default two. A window occupies its
lease during source pull, preparation, queued/active HTTP and completed output
waiting for the consumer. Delivery transfers the result to the consumer and
releases that window's lease. There is no additional speculative source read
beyond the admitted window budget, including when the first window is slow or
the consumer pauses.

One global H owner budget covers synchronous callers and all async streams,
including in-progress source pulls, before snapshot/registration. Set
`embedding_max_pending_owners` explicitly or let it resolve once from
`(configured positive guard, otherwise seed 2)+queue_size`, minimum one.
It never follows the discovered replica count. `max_pending_windows` consumes
this shared H as a per-stream sublimit. A separate finite input-snapshot byte
pool validates/freeze-copies a finite sequence before engine registration;
a shared builder avoids partial-copy credit deadlocks. Caller/source storage
is external to this ownership envelope. An indivisible impossible window fails
explicitly; aggregate pressure remains cancellable under its original deadline.
This intake lease is distinct from an internal preparation-lookahead lease:
completed preparation can free lookahead for another plan while the original
window, its records, results and HTTP ownership remain retained. It does not
free that window's intake lease before delivery or cleanup.

The adapter's `batch_size` continues to control total-text windows for replica
quotas. Source windows are result boundaries and can contain multiple internal
batch windows. Each registers with the persistent engine, discovery, identity
and token-count validation, preparation budgets and the single HTTP scheduler.
One original POST lease per verified replica is authoritative. An absent or
`None` global guard uses N replicas; a positive guard caps HTTP concurrency
without reducing N-wide text quotas. Native slots/batch/microbatch remain
diagnostics. Fixed H and global working/result/handoff pools can limit actual
admission below that capacity, including across synchronous callers. The async bridge submits work and
awaits the engine's completion Future; it does not allocate a blocking
coordinator thread for every window, route replicas or schedule a second set
of HTTP jobs. A completion Future is settled only after the owned work drains,
rather than merely when its waiter has been cancelled.

Engine-owned byte and prepared-token budgets complement window counts; see
the [resource families](CONCURRENCY-AND-DIAGNOSTICS.md#admission-and-resource-ownership).
HTTP completion alone does not release a response handoff while result credits
are unavailable. Transfer into result credits releases handoff; ordered public
delivery then releases the result and owner leases. These are separate steps. Source windows are delivery
boundaries, not HTTP batch boundaries. Compatible prepared records can share a
POST across windows or callers, retaining each owner's cancellation, deadline
and output indices. The request target derives from the adapter's text budget
and usable replicas; small source windows do not force small request caps.
With the default zero wait, preparation and capacity events drive filling
without a required timer, and remaining tails flush when filling cannot progress.

## Deadlines, failure and ownership

`timeout` is a positive finite per-window budget, defaulting to the profile's
timeout. It starts before global admission and the source pull, and includes
discovery, queue waits, exact preparation and HTTP. Each admitted window gets a
new budget, so an indefinitely productive source does not imply an unlimited
individual wait. Results already completed do not expire merely because the
consumer has not collected them.

One stream failure stops new intake and cancels/drains its outstanding work.
Windows already delivered remain valid; no partially computed window is
reported as complete and no successful inference is replayed. Backend and
timeout errors retain their typed contracts. Source exceptions remain source
exceptions. Task cancellation propagates as `asyncio.CancelledError` after
owned cleanup; explicit cancellation probes retain `LLMCancelledError`.
If a stopped window contributed to shared HTTP, its pending records are removed
and its response portion is discarded. Surviving windows keep the transport.
Its finite deadline was fixed at dispatch from the latest initial member
deadline; it does not grow when membership changes. The stopped window's Future
settles after actual shared transport drain, which may exceed its own work
deadline. When all members stop, or the adapter closes, HTTP is cancelled and
drained. A mixed request failure retains typed outcomes without replaying
already evaluated inputs.
If source cleanup also fails, the primary window error or cancellation is
preserved and the cleanup failure is chained as its cause, including during
async-context exit. Standalone `aclose()` can report a source cleanup failure.

Cancelling an asyncio future alone does not stop a running thread. The bridge
must register calls before dispatch, set the real cooperative stop signal,
retain the engine Future and await its owned/shared drain off-loop, including during
repeated cancellation. Cancelling one stream must not close its shared adapter
or cancel another caller. Close is idempotent.

The source iterator is exclusively used by the stream while active. By default
it is borrowed: the stream cancels and awaits its own pending source pull but
does not deliberately call source `aclose()`. Cancellation of a pending async
generator pull can itself run its `finally` and close it; reuse is not promised.
With `close_source=True`, the stream also awaits the source's `aclose()` when
available, after stopping any pending pull. Producers and their cleanup must be
cooperative async code; the library cannot safely kill code that blocks its
event loop or suppresses cancellation indefinitely.

`stream.aclose()` drains only that stream. `adapter.aclose()` drains active
streams and then performs the shared synchronous close off-loop. The existing
`adapter.close()` continues to drain worker/HTTP resources and signals async
streams to stop; it cannot synchronously await source cleanup on the same
event loop. Async consumers use `await adapter.aclose()` for full async cleanup.

## Consumer example

```python
async def windows():
    for texts in incoming_windows:
        yield texts

try:
    async with adapter.astream_texts(windows(), max_pending_windows=2) as stream:
        async for vectors in stream:
            await store_next_window(vectors)
finally:
    await adapter.aclose()
```

The consumer owns source splitting, UTF-8 intervals, fragment identity and its
graph. It chooses how large each source window is; the library owns batching,
replica scheduling and bounded intake. See the
[0.8.0 qualification](verification/ASYNC-WINDOWS-0.8.0.md) for actual checks,
measured timings and deployment boundaries.
