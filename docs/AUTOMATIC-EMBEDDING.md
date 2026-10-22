# Automatic category-calibrated embeddings

This optional path is governed by Flower `REQ-000038`, `GOAL-000020`,
`MILE-000021`, `CHANGE-000022`, `PACKET-000023`. The existing declared-model
adapter and its exact `prepare_*` / `embed_prepared` contracts remain available.
Version 0.11.1 hardening is governed by Flower `REQ-000039`, `GOAL-000021`,
`MILE-000022`, `CHANGE-000024` and `PACKET-000025`.
The local 0.12.0a1 minimum-budget and overflow-recovery candidate is governed by
`REQ-000040`, `GOAL-000022`, `MILE-000023`, `CHANGE-000025`, `PACKET-000026`.

The automatic service discovers one unambiguous, verified, loaded embedding
identity group. It owns aliases, replica capacity, prefixes, preparation and
dispatch. It uses the existing persistent embedding engine and scheduler.
It never loads, restarts or administers remote models.

Only the explicit `character_budget(...)` operation may start a cold calibration.
Construction, `initialize()`, `status()`, `embed_texts()`, `embed_query()` and
`astream_texts()` never start the 10,000-text pass. Successful ordinary calls do
not consult calibration or learned state and can use an empty or unwritable
cache. A confirmed failed-POST overflow uses completed state if present and
requires writable storage to commit its learned ceiling. The consumer owns
source splitting and may persist a requested character
budget for subsequent runs. A stored budget is a snapshot of its model, task,
category and observed limits; it is not an inference permit for a changed server.

## Consumer API

```python
from runtime_llama import AutomaticEmbeddingConfig, AutomaticEmbeddingService

with AutomaticEmbeddingService(AutomaticEmbeddingConfig(
    base_url="http://inference.example:8080",
)) as embeddings:
    availability = embeddings.status()  # No calibration or cache access.
    if availability.available:
        vectors = embeddings.embed_texts(["A document to index."])
        query_vector = embeddings.embed_query("What does it describe?")

    # Optional, explicit provisioning step. The consumer can save this snapshot.
    budget = embeddings.character_budget(category="code")
    saved_character_limit = budget.max_input_chars
    saved_representation = budget.representation_id
```

On later starts, a consumer can use its saved limit to split its source and send
the strings directly. It does not need to call `character_budget()` again or
provide the saved budget to the service. It should compare the current opaque
`representation_id` before reusing an existing vector index. A changed context
can also invalidate a stored character recommendation; request a fresh budget
when its deployment profile changes.

The public methods are:

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

The four observation/budget methods return `AutomaticEmbeddingStatus`; ordinary
availability has `max_input_chars`, `category`, `calibration_id` and `budget`
set to `None`. `character_budget()` returns a completed budget snapshot.
`refresh_character_budget()` never calibrates: it reuses valid completed
statistics and/or learned state after fresh discovery. Without either, it
returns available status, `max_input_chars=None`, `budget=None` and reason
`character_budget_not_calibrated`. It does not create the cache.
`input_type` accepts
`document`, `query` or `raw`; query and document policies can have distinct
calibrations. The asynchronous iterator yields an ordered `list[list[float]]`
for each input window and has the same scheduler as synchronous calls.

`batch_size` is the total text budget, not a per-replica constant. A budget of
256 can distribute up to 128 texts to each of two usable replicas or up to 64
to each of four; native limits and hard request-byte bounds can require more
HTTP requests. Global admission also respects `request_parallelism` and observed
slots. Calibration uses the separate bounded `calibration_workers` setting.
No compulsory accumulation timer is introduced.

## Categories and explicit calibration

The packaged `runtime_llama/calibration_corpus/` directory contains exactly
10,000 original synthetic texts: 2,000 each in `code`, `documentation`, `prose`,
`structured` and `multilingual`. `mixed` aggregates the complete corpus. The
generator, category counts and manifest make the corpus reproducible; its
purpose is calibration, not a universal or naturally sampled language benchmark.

An explicit initial budget request tokenizes the entire corpus, with the selected
task prefix and native special-token policy. All category statistics are calculated in one
pass. Changing category reuses that complete profile. A different prefix/task
policy needs its own calibration, as do changed weights, semantic identity,
tokenizer build or corpus revision. Equivalent replica aliases share profiles.

Raw statistics retain weighted/arithmetic averages and each category's minimum
and maximum. Starting in 0.12.0a1, the operational coefficient uses the **global
minimum** of `source Unicode characters / actual fully prepared tokens` across
all 10,000 texts, for every category. Individual categories multiply that common
minimum by 0.80; `mixed` multiplies it by 0.90. Categories and their measurements
remain visible; choosing a category cannot bypass a denser corpus sample.
Existing validated raw profiles contain this minimum and need no retokenization.

The initial ceiling is `floor(effective input token limit * coefficient)`.
Prefix and special tokens are already counted in the denominator and are not
subtracted again. Characters mean Python Unicode code points, not UTF-8 bytes
or grapheme clusters. The qualified 512-token Nomic profile's global minimum is
1.2115384615384615 characters/token: category ceilings are 496 and mixed is 558.
The earlier mean-based 1,509-character mixed value belongs to 0.11.0/0.11.1.

Even the worst corpus sample is empirical, not proof about arbitrary source
text. Only an exact measured fragment is admitted as verified advice. Ordinary
calls do not enforce the character recommendation and never silently split or
truncate inputs.

## Exact correction after a failed POST

Only a direct HTTP 500 enters recovery. After the owner's normal embedding work
has drained, the service counts the source rows actually sent in that failed
POST, with the same verified prefix and native special-token policy. It does
not tokenize successful POSTs, the entire ingestion, other owners' rows or the
10,000-text corpus. Duplicate occurrences retain distinct original indices;
equal diagnostic text can safely reuse a count in that one verified scope.

No proven context overflow means the **same original `ClientHTTPError`** is
raised, with no learned-state update. An unavailable/malformed diagnostic
tokenizer also preserves that inference error. Cancellation, finite deadlines,
callback shutdown guards and changed identity/limits keep their own failures.
A microbatch reservation alone is not overflow proof: native pooling modes can
differ in their ability to split computation.

For proven overflow, the service computes a proper source partition and counts
every proposed fragment exactly, including prefix/specials. Search probes do
not assume monotonic BPE counts or establish a global maximum. Accepted spans
cover the original source contiguously in both Unicode code points and UTF-8
bytes. Every fragment fits the common effective limit of all usable replicas.
Diagnostics allow at most 512 tokenizer measurements and 256 returned fragments,
under the failed call/window's original deadline and cancellation. Incomplete
advice is discarded, preserving the original 500.

Fresh observations validate the actual failed alias's context and microbatch
as well as the pool minimum, identity and tokenizer before persistence. A
completed partition yields `EmbeddingInputOverflowError`, never vectors of
changed cardinality or an automatic embedding retry. Public exports are:

```text
EmbeddingVerifiedSpan(char_start, char_end, utf8_start, utf8_end, token_count)
EmbeddingOverflowInput(input_index, text_sha256, char_count, utf8_count,
                       token_count, token_limit, fragments)
EmbeddingInputOverflowError(*, budget, inputs, failed_input_indices,
                            window_index=None)
```

`input_index` and `failed_input_indices` are zero-based in the owner's original
logical window, not positions inferred from hashes or text identity.
`window_index` is `None` for synchronous calls and a zero-based source ordinal
for automatic async streams. `inputs` contains only confirmed oversized
occurrences; hashes identify their unprefixed UTF-8 source. Texts, token arrays
and other owners' provenance are not public fields.

The consumer buffers its whole logical window until success. On correction it
validates identity, hashes and coverage, applies the **exact returned spans**
to affected inputs, preserves other inputs and source identities, then decides
whether to retry that logical window. Do not heuristically split the verified
fragments again: BPE at a new boundary can change counts. Graphs, parent/fragment
IDs, byte/line intervals, publication and retry bounds belong to the consumer.
Strict progress is a proper source partition; the reported character ceiling
can remain unchanged when its initial value is already lower than the newly
measured candidate.

Shared POSTs still isolate owners: valid peer inputs retain the original 500;
only the owner with proven overflow receives its correction. Async streams
remain fail-stop and drain prefetched work; there is no automatic replay/resume
or partial vector delivery from a failed logical window.

## Budget snapshots and learned state

`AutomaticEmbeddingStatus.budget` and `EmbeddingInputOverflowError.budget` expose
the immutable public `EmbeddingBudgetSnapshot`:

```text
policy_id, representation_id, context_scope_id, input_type, category,
calibration_id, max_input_chars, initial_max_input_chars, budget_revision,
effective_context_tokens, source
```

`calibration_id` and `initial_max_input_chars` are nullable for learned-only state
without a completed 10,000-sample profile. All other snapshot fields are defined;
`max_input_chars` is positive. `source` is `calibrated_minimum` or
`learned_reduction`. Availability-only status has no snapshot. Consumers may
serialize the snapshot using `dataclasses.asdict`; historical calibration
provenance must not be presented as a current non-null SDK calibration ID.

The learned candidate is the smallest, across offending inputs, of each complete
partition's largest verified fragment character length. The absolute stored
ceiling is the minimum of that candidate, the current calibrated recommendation
if available, and any existing learned ceiling. **80%/90% are not applied again
to this exact learned value.** It is a future character recommendation; the
returned exact spans govern the current correction.

Learned state is common across categories of the same verified model/task/
prefix/tokenizer and exact effective-context scope. Different tasks, builds,
weights, semantics or effective contexts have distinct scopes; lessons are not
silently rescaled or moved between them. Revisions increase only on a real
stored reduction. `representation_id` identifies vector semantics; a smaller
character ceiling alone does not imply a new vector space.

Thread and cross-process locks enforce read/min/write convergence. Private
atomic replacement publishes only complete scalar state; no network work runs
under the store lock. Cancellation before commit leaves the preceding state;
after atomic publication a valid reduction cannot be undone. Corrupt learned
state fails closed rather than silently restoring a larger recommendation.

## Persistent authority and resource limits

Only complete, validated raw statistics and learned scalar ceilings are persisted, outside the
repository. No consumer texts, token arrays, credentials, IP addresses or model
paths are stored. Cache identity includes verified weights and embedding
semantics, corpus, tokenizer build and task/prefix/special-token policy.
Atomic writes do not publish partial calibration or learned records after
cancellation or timeout.

Fresh runtime observations remain required. A changed context or microbatch
updates the character budget without retokenizing an unchanged corpus/profile.
Budget calculation conservatively respects the smallest observed input context
and native microbatch; native computation sizes are not aggregate HTTP token
budgets. Multiple unrelated embedding groups require explicit selection.

Successful ordinary calls send strings directly, without per-input `/tokenize`, calibration
lookups or statistical character admission. The server owns actual tokenization
and context admission. Internal queue capacity can conservatively reserve one
observed input context per text; these reservations are not measured token counts
and do not determine whether a text fits. Byte, text-count and response bounds
remain hard bounds. Prefixes are applied once, and ordering, owner cancellation,
finite deadlines, backpressure and drained shutdown use the original engine.
Exact public preparations are never populated with reservations or estimates.

## Configurable transport and memory budgets

The automatic config exposes the existing adapter limits without changing their
defaults:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `embedding_max_request_bytes` | 8 MiB | Serialized embedding request bound. |
| `embedding_max_response_bytes` | 8 MiB | Decoded HTTP JSON response bound, including decompressed gzip. |
| `embedding_max_pending_bytes` | 64 MiB | Existing pending-data and retained-result credit limits. |
| `embedding_max_pending_tokens` | 1,048,576 | Queue token-accounting bound; direct inputs reserve their observed allowance. |
| `http_connection_policy` | `"close"` | Owned client uses independent connections; explicit `"reuse"` enables existing keep-alive policy. |

Limits must be positive integers. A decoded JSON response and retained Python
vector results are different resources. Raising the response bound does not
raise result credit or remove request, pending-data, cardinality or concurrency
bounds. These accounting budgets are not an exact operating-system RSS limit.
High-dimensional vectors can require a response budget above 8 MiB even when
their input strings are short. For example:

```python
config = AutomaticEmbeddingConfig(
    base_url="http://inference.example:8080",
    embedding_max_response_bytes=16 * 1024 * 1024,
    embedding_max_pending_bytes=64 * 1024 * 1024,
    http_connection_policy="reuse",
)
```

For direct inputs, the token ledger reserves the smallest observed context and
microbatch allowance per text. At a large allowance this can fill a pending
token budget before `batch_size` texts accumulate. An operator can explicitly
raise `embedding_max_pending_tokens` to accommodate the desired bounded intake;
the library does not fabricate counts or silently enlarge memory limits. For
an allowance of 65,536, accommodating 256 pending direct texts requires
16,777,216 reservation units. This changes admission capacity, not the scheduler
or the server's actual context checks. With multiple active owners, capacity is
shared and the observed HTTP packing can differ.

`http_connection_policy` applies to the service-owned client. An injected client
retains its connection policy and remains owned by its caller. Reuse adds no
retry, transport fallback or requirement for a patched llama-server.

## Calibration and shutdown failures

A valid complete calibration cache can be read without directory write access.
For a cold explicit budget request, the library checks persistence feasibility
before sending tokenizer requests, using a private temporary file which is
removed after the check. Final validation and private atomic replacement still
apply: permissions or storage can change after a successful preflight. Ordinary
availability and successful embedding never perform this check or consult the
cache. Failed-POST recovery has its separately bounded persistence contract.

Cancellation callbacks must be side-effect free and safe across threads, such as
`threading.Event.is_set`. Calling the same service's synchronous `close()` from
inside its own callback raises a typed `LLMBackendError` with diagnostic
`llamacpp_automatic_embedding_close_from_cancel_callback` before shutdown state
changes. Set a cancellation event in the controlling thread, then call `close()`
or await `aclose()` externally to stop and drain owned work.

A single-model catalog without a loaded status is accepted only when verified
identity tags and coherent root `/props` attest the loaded model. Router props,
sleeping entries, ambiguous catalogs, unverified identity and inconsistent model
proof are not promoted to loaded by assuming that missing status means ready.

An opaque representation identity protects stored vector spaces. Automatic
discovery must not silently replace an index's embedding model. A consumer
still owns source splitting, UTF-8 fragment intervals, graph relationships and
the decision to rebuild an index after a representation change.

## Verification

Implementation, deterministic verification, live qualification and human
acceptance are separate. Initial 10,000-text live calibration is explicitly
authorized for this cycle; it is run only through an explicit budget request.
Stable publication is recorded separately in the
[0.11.1 release record](verification/PUBLIC-RELEASE-0.11.1.md).
Consumer migration remains a separate task.
The local [0.12.0a1 qualification](verification/EXACT-OVERFLOW-0.12.0a1.md)
records the new minimum policy, exact correction and installed-wheel evidence;
this candidate does not authorize a new stable publication or deployment.
The [0.11.0a1 qualification](verification/AUTOMATIC-EMBEDDING-0.11.0a1.md)
records deterministic coverage, live HTTP counts, timings and category budgets.
