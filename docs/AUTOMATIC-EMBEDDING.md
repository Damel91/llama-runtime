# Automatic embeddings and fixed character recommendations

`AutomaticEmbeddingService` selects the requested remote model or one
unambiguous loaded embedding identity group. It owns task prefixes, replica
discovery and dispatch through the same engine as the declared-model adapter.
Different identities remain separate; loading/build/hardware remain operator
owned. Version 0.15.0 removes implicit text/model/memory quotas without moving
source splitting, fragment identity or graph/storage into the library.

Flower MCP governs this change through requirements `REQ-000061`–`REQ-000069`,
milestones 36–40 and packets 39–43. See the
[release verification](verification/PUBLIC-RELEASE-0.15.0.md).

## Ordinary use

```python
from runtime_llama import AutomaticEmbeddingService

with AutomaticEmbeddingService({
    "base_url": "http://inference.example.invalid:8080",
    "model": "logical-embedding-name",
    "batch_size": 256,  # Optional consumer-selected target.
}) as embeddings:
    status = embeddings.status()
    if status.available:
        vectors = embeddings.embed_texts(["A document to index."])
        query_vector = embeddings.embed_query("What does it describe?")
    # Optional explicit provisioning; the consumer can save the snapshot.
    budget = embeddings.character_budget(category="code")
```

Construction, availability, ordinary embedding and async intake never start
calibration or require a saved character limit. Strings go directly to the
server, which tokenizes and enforces its context. Prefixes are applied once;
cardinality, dimensions, finite values and source order are validated. Ordinary
direct inputs do not reserve the entire model context as invented token use.

Only `character_budget()` can start cold calibration. Its recommendation can
be saved by the consumer and need not be sent back with later strings.
Recommendations use effective context, without substituting native microbatch
as a context ceiling. Direct requests do not require metadata needed solely
for unused tokenizer/cache/calibration paths; pooling equivalent replicas still
requires the corresponding identity proof. Missing capabilities stay explicit.

## Public methods

```text
initialize(*, input_type="document", timeout=None, cancel=None)
status(*, input_type="document", timeout=None, cancel=None)
character_budget(*, category=None, input_type="document", timeout=None, cancel=None)
refresh_character_budget(*, category=None, input_type="document", timeout=None, cancel=None)
embed_texts(texts, *, input_type="document", timeout=None, cancel=None)
embed_query(text, *, timeout=None, cancel=None)
aiter_texts(texts, *, input_type="document", timeout=None, cancel=None,
             close_source=False)
astream_texts(windows, *, input_type="document", max_pending_windows=None,
              timeout=None, cancel=None, close_source=False)
close()
await aclose()
```

`aiter_texts()` takes a finite text sequence or async text source and yields
`EmbeddingRecordResult(index, embedding)` in source order. The consumer need
not divide 10,000 texts into HTTP-sized windows. Individual result ownership
transfers as records are delivered. Existing `embed_texts()` and
`astream_texts()` keep their complete-list/window result forms and naturally
occupy the memory those outputs require. See
[async interfaces](ASYNC-EMBEDDING-WINDOWS.md).

Observation methods return `AutomaticEmbeddingStatus`. Ordinary availability
has `max_input_chars`, `category`, `calibration_id` and `budget` set to `None`.
`refresh_character_budget()` reuses completed matching calibration after fresh
observation and never calibrates. Without a complete matching profile its
reason is `character_budget_not_calibrated`, with no budget/cache creation.

`input_type` is `document`, `query` or `raw`. Query/document prefixes are chosen
from the model contract or explicit `embedding_prefixes`; raw adds no prefix.
A changed opaque `representation_id` requires consumer review before reusing
an existing vector index. Identity verification prevents pooling unrelated
models, not a fixed maximum number of loaded models.

## The 2,500 hard calibration texts

The packaged `runtime_llama/calibration_corpus/` contains exactly 500 original
synthetic texts in each of `code`, `documentation`, `prose`, `structured` and
`multilingual`. `mixed` aggregates all 2,500. One-off curation ranked the original
2,000 texts per category using actual prepared Nomic token counts, choosing the
500 smallest source-character/prepared-token ratios in each category. Stable
original IDs break ties. This is a deliberately difficult sample for that
Nomic tokenizer, not a universal worst case for every embedding model.

`tools/calibration-hard-selection.json` records scalar measurements and source
hashes; `tools/generate_calibration_corpus.py` reproduces the selected corpus
offline. The curation receipt is in the source archive, outside the installed
wheel. No consumer text, token IDs, private endpoint or model path is included.

A cold explicit request counts all 2,500 texts using the selected model's native
tokenizer, task prefix and special-token policy. All categories are calculated
in that pass. Changing category reuses it; changing weights, semantics,
tokenizer build, task/prefix policy or corpus identity requires a matching new
profile. Old 10,000-text profiles cannot supply this corpus's statistics.

The operational ratio is the weighted mean `sum(source characters) /
sum(prepared tokens)` of the selected category. A category uses 80% of that
mean; `mixed` uses 90% of the weighted mean of all 2,500 texts:

```text
max_input_chars = floor(effective_input_token_limit * weighted_mean_chars_per_token * factor)
factor = 0.80 for a category, 0.90 for mixed
```

Source characters are Python Unicode code points. Prefix and special tokens
are already in the denominator; do not subtract them a second time. The
recommendation is empirical and does not prove that an arbitrary input fits.
A failed embedding never changes these statistics or reduces the recommendation.
Fresh observation of changed loaded limits can change a newly requested
snapshot without retokenizing a still-valid profile.

`AutomaticEmbeddingStatus.budget` is an immutable `EmbeddingBudgetSnapshot`:

```text
policy_id, representation_id, context_scope_id, input_type, category,
calibration_id, max_input_chars, initial_max_input_chars, budget_revision,
effective_context_tokens, source
```

For this contract `source="calibrated_mean"`, `budget_revision=0` and
`initial_max_input_chars=max_input_chars`. Old learned reductions are not read
or written. A consumer's stored recommendation remains its responsibility.

## Recover only the rejected source

A direct HTTP 500 starts bounded isolation inside the existing scheduled HTTP
lease. The library bisects the failed index set into smaller HTTP requests,
retains each successful child's vectors and descends only into failed children.
Successful subbatches are never sent again by isolation. Two identical strings
remain separate source occurrences with separate original indices.

An isolated rejected singleton of at least two Unicode code points produces
`EmbeddingChunkSplitRequiredError`. It contains **all** successful original
positions in `completed` and only the rejected positions in `inputs`. A generic
500 is not proof of context overflow: these are recovery instructions, with no
claimed token counts or token-admission guarantee. No `/tokenize` request,
calibration, learned-store access or budget recalculation occurs during recovery.

```text
EmbeddingSourceSpan(char_start, char_end, utf8_start, utf8_end)
EmbeddingHalvingInput(input_index, text_sha256, char_count, utf8_count, halves)
EmbeddingCompletedInput(input_index, embedding)
EmbeddingChunkSplitRequiredError(*, input_count, completed, inputs,
                                 representation_id, input_type, window_index=None)
```

Each rejected source has exactly two nonempty halves meeting at
`len(source) // 2`, covering its original Unicode and UTF-8 intervals without
loss. Its SHA-256 identifies the original unprefixed UTF-8 source. Completed
vectors are immutable tuples. Completed and rejected indices partition the
whole original logical window. `window_index` is `None` for synchronous calls,
otherwise the zero-based source-window ordinal. Source text and token arrays
are not fields of the error.

The consumer retains `completed`, verifies identity and the rejected source
hashes, creates child fragment identities from the returned intervals and sends
only those halves. It must bound its own repeated splitting/retries and retain
successful previous rounds. See the [consumer recovery example](HALVING-CONSUMER.md).
The library does not embed halves automatically or return a changed result
cardinality as ordinary success.

Non-500 responses retain their original error. Indivisible rejected sources
(empty or one code point), invalid UTF-8, exhausted isolation allowance,
identity changes, cancellation and deadlines are terminal errors. Failed exact
adapter owners in a shared POST preserve their original error; only automatic
direct owners take this isolation path. If isolation cannot complete safely,
it does not claim a complete partial-result partition.

## Ordered asynchronous delivery

All async methods share the same engine and HTTP scheduler as synchronous
calls. Complete windows and recoverable errors are delivered in source order;
flat records use their global occurrence index. For a flat
`EmbeddingChunkSplitRequiredError`, `window_index` identifies that global
record while nested indices/spans remain relative to the singleton occurrence.
The error remains its original type and iteration can continue. Nothing is
silently replayed. An exception escaping `async with` closes the stream.

`timeout` bounds readiness and each source window/record's work, not the whole
stream. Sources are borrowed unless `close_source=True`. Use `async with` or
`aclose()` for deterministic early Stop; dropping the flat facade signals Stop
and schedules actual inner drain on its live loop. Already closed event loops
and uncooperative user code cannot be made to perform asynchronous cleanup.

## Batching, concurrency and capacity choices

`batch_size=None` imposes no fixed text target. An explicit B is a total text
distribution target across N verified replicas: B=256/N=2 gives up to 128 texts
per quota, and N=4 up to 64. Neither value is a maximum total input count.
`request_parallelism=None` uses N; a positive G caps original HTTP work to
`min(N,G)` independently of quota width. Each canonical replica owns one
original lease through its serial isolation descendants and actual drain.
Native slots/batch/microbatch do not multiply HTTP permits.

Compatible records accumulate across source windows. No mandatory timer or
second scheduler is added. Worker defaults follow the actual pool; there is
no fixed preparation/control cap of one, two or four models.

| Setting | Default | Meaning |
| --- | --- | --- |
| `batch_size` | `None` | Optional total text target, independent of input size. |
| `request_parallelism` | `None` | Automatic N; positive values are explicit HTTP guards. |
| `calibration_workers` | `None` | Follows the active pool unless explicitly selected; no former 32-worker validation maximum. |
| `embedding_capacity_limits` | `False` | Explicit opt-in required before scalar capacity fields take effect. |
| `embedding_max_pending_owners`, `queue_size` | `None` | Optional owner/lookahead policies; no derived H preset. |
| `embedding_max_pending_bytes`, `embedding_max_pending_tokens` | `None` | Optional pending/output policies; no implicit 64 MiB or 1,048,576 allowance. |
| `embedding_max_request_bytes`, `embedding_max_response_bytes` | `None` | Independently requested payload/response controls. |
| `embedding_max_handoff_bytes` | `None` | Independent optional handoff control; no 75/25 partition or response/16 transformation. |
| `embedding_isolation_max_requests` | `None` | Optional descendant-request policy. |
| `embedding_preparation_cache_enabled` | `False` | Explicitly enables session-owned exact preparation reuse. |
| `http_connection_policy` | `"close"` | Independent owned connections; `"reuse"` enables keep-alive. |
| `timeout`, `calibration_timeout` | `120`, `600` seconds | Finite work budgets, preserving cancellation and cleanup. |

`embedding_capacity_limits=True` enables only the supplied capacity values;
missing fields remain unset. Legacy scalar values alone do not reactivate old
policies. Explicit `max_pending_windows` on a window stream remains a separate
consumer choice. Batch targets and HTTP guards are also independent of this
capacity switch. No automatic RAM sizing, compulsory whole-result reservation,
memory-driven candidate halving, implicit spill to disk or constant-RSS promise
is installed.

The session preparation cache is distinct from persisted calibration profiles.
An explicitly enabled preparation cache clears at close. Its entry/byte controls
are available on the adapter profile through the capacity opt-in; absent quotas
allow deliberate session retention to grow. Ordinary direct embeddings do not
consult this exact-token cache.

Without an explicit isolation-request cap, a finite N-input binary subdivision
has at most `2*N-1` POSTs including its root. The existing deadline and Stop are
still checked. Healthy sibling owners survive another owner's withdrawal;
close waits for real transport drain.

## Persistence and failure handling

Only complete validated scalar calibration profiles persist outside the repo.
No consumer source, credentials, token arrays or model paths are stored. Cold
calibration checks persistence before sending tokenizer requests, then validates
identity before atomic publication. Cancellation does not publish a partial
profile. A matching complete cache can be read without directory write access.
Ordinary use and isolation require no writable cache.

Cancellation callbacks must be thread-safe and side-effect free. Closing the
same service inside its own cancellation callback raises
`llamacpp_automatic_embedding_close_from_cancel_callback`; signal cancellation
from the controlling thread, then close externally. A catalog entry without
loaded status needs verified identity tags and coherent root properties;
missing status is not assumed to mean loaded.

## Qualification boundaries

Deterministic tests, installed-wheel checks, live server qualification and
consumer acceptance remain separate. The release record distinguishes actual
new checks from historical measurements. No fresh 0.15 live result or consumer
migration is implied by these contracts. Client wall times include networking,
preparation, controls and decoding; they are not pure GPU time.

The earlier strict numerical repeatability failures remain retained as the
operator-accepted model-profile limitation; no numerical or retrieval-quality
pass is invented. See [resource ownership](CONCURRENCY-AND-DIAGNOSTICS.md) and
[0.15.0 verification](verification/PUBLIC-RELEASE-0.15.0.md).
