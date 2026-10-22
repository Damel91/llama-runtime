# Automatic category-calibrated embeddings

This optional path is governed by Flower `REQ-000038`, `GOAL-000020`,
`MILE-000021`, `CHANGE-000022`, `PACKET-000023`. The existing declared-model
adapter and its exact `prepare_*` / `embed_prepared` contracts remain available.
Version 0.11.1 hardening is governed by Flower `REQ-000039`, `GOAL-000021`,
`MILE-000022`, `CHANGE-000024` and `PACKET-000025`.

The automatic service discovers one unambiguous, verified, loaded embedding
identity group. It owns aliases, replica capacity, prefixes, preparation and
dispatch. It uses the existing persistent embedding engine and scheduler.
It never loads, restarts or administers remote models.

Only the explicit `character_budget(...)` operation may start a cold calibration.
Construction, `initialize()`, `status()`, `embed_texts()`, `embed_query()` and
`astream_texts()` do not read or require calibration statistics and never start
the 10,000-text pass. They can operate with an empty or unwritable calibration
cache. The consumer owns source splitting and may persist a requested character
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
embed_texts(texts, *, input_type="document", timeout=None, cancel=None)
embed_query(text, *, timeout=None, cancel=None)
astream_texts(windows, *, input_type="document", max_pending_windows=2,
              timeout=None, cancel=None, close_source=False)
close()
await aclose()
```

The first three return `AutomaticEmbeddingStatus`; ordinary availability has
`max_input_chars`, `category` and `calibration_id` set to `None`.
`character_budget()` returns those fields populated. `input_type` accepts
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

For each category and for `mixed`, the operational ratio is
`sum(source Unicode characters) / sum(actual fully prepared tokens)`.
The effective coefficient is 80% of this ratio for an individual category and
90% for `mixed`. These factors leave 20% and 10% headroom respectively, and
remain separate from the persisted raw measurements. Prefix and special-token cost
is already represented in the denominator; it is not subtracted a second time.
The character budget is `floor(effective input token limit * coefficient)`.
Characters mean Python Unicode code points, not UTF-8 bytes or grapheme clusters.

Mean-based safety margins are empirical. They do not prove that arbitrary
source text fits a tokenizer's context. Native overflows remain explicit
errors; the service never silently truncates or splits source text.
In particular, `mixed` is an aggregate recommendation, not a conservative bound
for every constituent category. In the qualified Nomic sample, the 512-token
recommendations were 1,509 characters for `mixed` and 1,127 for `code`; applying
the mixed recommendation to code at its category's average density would imply
about 548 tokens. This arithmetic illustrates the limitation, not a measurement
of a particular input. The factors remain 90% and 80% respectively. Select a
category deliberately and retain explicit overflow handling; this release adds
no quantile policy, automatic retry or implicit exact-count fallback.

## Persistent authority and resource limits

Only complete, validated aggregate statistics are persisted, outside the
repository. No consumer texts, token arrays, credentials, IP addresses or model
paths are stored. Cache identity includes verified weights and embedding
semantics, corpus, tokenizer build and task/prefix/special-token policy.
Atomic writes do not publish partial calibration after cancellation or timeout.

Fresh runtime observations remain required. A changed context or microbatch
updates the character budget without retokenizing an unchanged corpus/profile.
Budget calculation conservatively respects the smallest observed input context
and native microbatch; native computation sizes are not aggregate HTTP token
budgets. Multiple unrelated embedding groups require explicit selection.

Ordinary calls send strings directly, without per-input `/tokenize`, calibration
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
availability and embedding never perform this check or consult the cache.

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
The [0.11.0a1 qualification](verification/AUTOMATIC-EMBEDDING-0.11.0a1.md)
records deterministic coverage, live HTTP counts, timings and category budgets.
