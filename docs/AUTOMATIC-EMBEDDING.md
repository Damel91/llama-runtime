# Automatic embeddings and fixed character recommendations

The automatic service discovers one unambiguous verified loaded embedding
identity group. It owns aliases, replica scheduling, task prefixes and dispatch
through the same persistent engine as the declared-model adapter. The operator
owns model loading and server configuration.

The 0.13.0a1 contract is governed by Flower `REQ-000042`,
`GOAL-000024`, `MILE-000025`, `CHANGE-000027`, `PACKET-000028`. It replaces the
0.12.0a1 minimum-ratio and exact post-failure diagnostics contract. Historical
qualification reports describe their historical versions. Publication of the
exact tested artifact is governed separately by `REQ-000043` / `PACKET-000029`;
see the [prerelease record](verification/PUBLIC-RELEASE-0.13.0a1.md).

## Ordinary use

```python
from runtime_llama import AutomaticEmbeddingConfig, AutomaticEmbeddingService

with AutomaticEmbeddingService(AutomaticEmbeddingConfig(
    base_url="http://inference.example:8080",
    batch_size=256,
)) as embeddings:
    status = embeddings.status()
    if status.available:
        vectors = embeddings.embed_texts(["A document to index."])
        query_vector = embeddings.embed_query("What does it describe?")

    # Optional explicit provisioning. Save the snapshot in consumer storage.
    budget = embeddings.character_budget(category="code")
```

Construction, availability, ordinary embedding and asynchronous intake never
start calibration, consult a calibration cache or require a saved character
limit. Strings go directly to the server. The server tokenizes and enforces its
context; the library preserves source order and validates cardinality, dimensions
and finite values. Inputs are never silently truncated or split.

Only `character_budget()` can start a cold calibration. A consumer may request
its character recommendation once, save it and send later strings without
supplying the recommendation back to the library. The consumer owns source
splitting, fragment IDs, UTF-8 intervals, graph relationships and index storage.

## Public methods

```text
initialize(*, input_type="document", timeout=None, cancel=None)
status(*, input_type="document", timeout=None, cancel=None)
character_budget(*, category=None, input_type="document", timeout=None, cancel=None)
refresh_character_budget(*, category=None, input_type="document", timeout=None, cancel=None)
embed_texts(texts, *, input_type="document", timeout=None, cancel=None)
embed_query(text, *, timeout=None, cancel=None)
astream_texts(windows, *, input_type="document", max_pending_windows=2,
              timeout=None, cancel=None, close_source=False)
close()
await aclose()
```

The local `0.13.1.dev1` candidate reuses verified admission within an active
automatic stream. Readiness observes the pool once; subsequent direct windows
fork independent call references while the same pool generation remains valid.
An idle stream holds no pool reference or scheduler permit. Each original
embedding POST still requires fresh catalog and selected-replica properties;
recovery keeps its own fresh observations. A full matching-pool topology change
revokes local reuse, and pool replacement waits for existing calls to drain.
There is no TTL cache or inference authorization attached to saved observations.
The public signatures, synchronous/exact paths, batch quotas, deadlines,
cancellation and bounded intake remain unchanged. This candidate is for isolated
verification and does not update consumer installations. See the
[control-overhead verification](verification/STREAM-CONTROL-0.13.1.dev1.md).

Observation methods return `AutomaticEmbeddingStatus`. Ordinary availability has
`max_input_chars`, `category`, `calibration_id` and `budget` set to `None`.
`refresh_character_budget()` uses completed matching calibration after fresh
observation; it never calibrates. Without a complete profile it returns reason
`character_budget_not_calibrated`, with no budget and no cache creation.

`input_type` is `document`, `query` or `raw`. Query and document prefixes are
applied once by the selected model policy. Prefix/task policies have separate
calibrations. A changed opaque `representation_id` requires consumer review
before reusing an existing vector index.

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

## Ordered asynchronous windows

The asynchronous API shares the same engine, owner cancellation and HTTP
limits as synchronous calls. Results and recoverable errors are delivered in
source-window order. A later prefetched rejection cannot discard an earlier
successful window. Catching `EmbeddingChunkSplitRequiredError` leaves the
iterator usable; the next `anext()` delivers the next original window. Nothing
is replayed automatically. If recovery should occur before later delivery,
handle it before calling `anext()` again.

Use an explicit `anext()` loop to continue after recoverable errors. An exception
escaping an `async with` body closes and drains the stream normally. Other
errors remain fail-stop. `timeout` bounds readiness and each individual window,
not the whole stream. Source iterators are borrowed unless `close_source=True`.
Paused successful or partial results remain covered by result credits until
ordered delivery; closing drains all owned requests.

## Batching, concurrency and bounds

`batch_size` is the total text distribution budget. With budget 256, two usable
replicas can each receive up to 128 texts; four can receive up to 64. These are
text quotas, separate from per-input context, aggregate HTTP token limits and
native computation batch/microbatch sizes. Quotas can require multiple HTTP
requests. The existing scheduler respects global `request_parallelism`,
per-replica configured limits and observed slots. No compulsory accumulation
timer or second scheduler is introduced.

Compatible admitted records accumulate across callers/windows. Source windows
are delivery boundaries, not forced HTTP fragmentation boundaries. Isolation
holds the original alias lease and serializes its own descendant calls; workers
on other aliases remain able to run. A loaded single-slot replica permits one
embedding HTTP call at a time. Adding an alias does not authorize parallel work
unless identity and loaded capacity are verified.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `embedding_max_request_bytes` | 8 MiB | Serialized embedding request bound. |
| `embedding_max_response_bytes` | 8 MiB | Decoded HTTP response bound, including decompressed gzip. |
| `embedding_max_pending_bytes` | 64 MiB | Pending records and retained-result credit limits. |
| `embedding_max_pending_tokens` | 1,048,576 | Queue reservation units for direct inputs; not measured tokens. |
| `embedding_isolation_max_requests` | 512 | Maximum descendant POSTs per originally failed POST, excluding that root request. |
| `http_connection_policy` | `"close"` | Independent connections for an owned client; `"reuse"` explicitly enables keep-alive. |

These integer limits must be positive. There is also an overall finite original
call/window deadline. Isolation checks cancellation and fresh model/limit
observations before descendant calls. With N rows the full binary isolation
tree has at most `2*N-1` POSTs including the root, before the configured cap.
A descendant cancellation retires its participating owner; healthy sibling
owners are not cancelled. Close waits for actual transport drain.

Direct queue token accounting conservatively reserves the observed allowance
per input without invoking the tokenizer. This is not an input token count or
an admission guarantee. Large allowances may require a larger explicitly
configured pending-token budget. Byte, text, result and HTTP limits remain
independent; they are not a claim about total operating-system RSS.

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
consumer acceptance are separate. The current live server has one loaded Nomic;
two-alias HTTP barrier fixtures prove scheduler overlap separately. Reported
client wall times include networking, preparation and validation, not pure GPU
time. The existing cross-replica numerical finding remains open; this contract
does not relax its tolerances or modify consumers or loaded models.
