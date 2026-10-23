# Consumer handoff: fixed budget and localized halving

Applies to the `runtime-llama==0.13.0a1` prerelease. No consumer is modified
or upgraded by this change. The installed package remains `runtime-llama`, with
Python imports from `runtime_llama`.

The [automatic API](AUTOMATIC-EMBEDDING.md) is the source of signatures and
resource limits. The exact declared-model preparation API remains available;
this handoff concerns the automatic direct-text path only.

## Startup and normal ingestion

1. Construct one shared `AutomaticEmbeddingService` using the endpoint and
   bounded queue/HTTP settings. `batch_size=256` is a total text budget across
   currently usable verified replicas, not 256 per replica.
2. Check availability and opaque `representation_id`. The consumer does not
   configure the number of copies, aliases or GPUs.
3. If an initial character recommendation is required and the consumer has no
   compatible saved snapshot, explicitly call `character_budget(category=...)`.
   This counts 2,500 hard corpus texts on a cold profile. Save the returned
   snapshot in the consumer's own configuration. Category factors remain 80%,
   mixed 90%; a failed chunk never changes this value.
4. Split source using that empirical character recommendation, preserving the
   consumer's AST/graph and lossless fragment identities. Send strings through
   `embed_texts()` or `astream_texts()`, without per-chunk `prepare_*` calls.
5. Persist vectors only with their actual source fragment identities and the
   matching representation. A changed loaded limit or representation requires
   the consumer to reassess its saved recommendation/index.

## Handling one failed window

The following example handles one recovery round. Production consumers must
bound the number of subsequent rounds, total work and deadline, and persist
successful rounds safely. An isolated 500 does not establish that overflow was
the cause; splitting is a bounded recovery policy.

```python
import hashlib
from runtime_llama import EmbeddingChunkSplitRequiredError


def embed_one_round(service, fragments, expected_representation):
    # Each fragment is consumer-owned: {"id": ..., "text": ...}.
    try:
        vectors = service.embed_texts([f["text"] for f in fragments])
    except EmbeddingChunkSplitRequiredError as error:
        if error.representation_id != expected_representation:
            raise RuntimeError("embedding representation changed") from error
        if error.input_count != len(fragments):
            raise RuntimeError("unexpected recovery window") from error

        completed = [
            (fragments[item.input_index], item.embedding)
            for item in error.completed
        ]
        pending = []
        for item in error.inputs:
            parent = fragments[item.input_index]
            source = parent["text"]
            if hashlib.sha256(source.encode("utf-8")).hexdigest() != item.text_sha256:
                raise RuntimeError("source changed during embedding") from error
            for half, span in enumerate(item.halves):
                pending.append({
                    "id": f'{parent["id"]}/half-{half}',
                    "parent_id": parent["id"],
                    "text": source[span.char_start:span.char_end],
                    "relative_utf8_start": span.utf8_start,
                    "relative_utf8_end": span.utf8_end,
                })
        # Retain completed vectors. Submit only pending halves next round.
        # Compose relative intervals with parent source offsets in the consumer.
        return completed, pending
    return list(zip(fragments, vectors)), []
```

`completed` and rejected indices partition the entire original window. Duplicate
strings have distinct indices. Never retry the whole window merely because one
chunk needs halves: doing so discards the isolation benefit and recomputes valid
vectors. Returned halves carry source intervals, **not exact token counts**.
Do not overwrite the saved character recommendation in this handler.

If the library raises a terminal transport/identity/deadline error instead, it
has not delivered a complete recoverable window. Do not infer a commit boundary
from server-side partial work. Empty or one-code-point rejected inputs cannot
make progress by halving and preserve their terminal failure.

## Asynchronous windows

A recoverable split event is delivered at its original source-window position.
Earlier successful windows remain delivered. The iterator remains usable after
the caller catches `EmbeddingChunkSplitRequiredError`; subsequent calls return
subsequent original windows, without replay.

```python
async with service.astream_texts(windows, max_pending_windows=2) as stream:
    while True:
        try:
            vectors = await anext(stream)
        except StopAsyncIteration:
            break
        except EmbeddingChunkSplitRequiredError as error:
            # Application-owned operation: keep completed vectors and enqueue
            # only validated halves with child source identities.
            await accept_partial_window(error)
        else:
            await accept_next_complete_window(vectors)
```

The application owns the mapping between source-window ordinal and its source
records. `error.window_index` is the zero-based original ordinal. If it wants to
finish local recovery before later windows, it can do so before the next
`anext()`. Other errors stop and drain the stream. Always close a stream after
an early break. Source windows remain delivery boundaries; compatible queued
records may share scheduled HTTP requests across windows.

## Integration constraints

Keep AST splitting, byte/line offsets, parent/fragment IDs, graph writes and
atomic index publication in the consumer. Keep server/model loading in the
operator. The runtime library owns replica discovery, weights identity,
concurrency, local source/result ordering and bounded HTTP isolation. No new
consumer-side tokenizer loop or model-instance configuration is required.

The older `EmbeddingInputOverflowError` exact-diagnostic type remains importable
for compatibility but is no longer emitted by automatic native-500 recovery.
A consumer must use the new partial-results contract before adopting this
candidate. The existing numerical finding across physical replicas is separate
from this functional handoff and remains open.
