# Automatic category-calibrated embeddings

This optional path is governed by Flower `REQ-000038`, `GOAL-000020`,
`MILE-000021`, `CHANGE-000022`, `PACKET-000023`. The existing declared-model
adapter and its exact `prepare_*` / `embed_prepared` contracts remain available.

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

```python
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

An opaque representation identity protects stored vector spaces. Automatic
discovery must not silently replace an index's embedding model. A consumer
still owns source splitting, UTF-8 fragment intervals, graph relationships and
the decision to rebuild an index after a representation change.

## Verification

Implementation, deterministic verification, live qualification and human
acceptance are separate. Initial 10,000-text live calibration is explicitly
authorized for this cycle; it is run only through an explicit budget request.
Stable publication is recorded separately in the
[0.11.0 release record](verification/PUBLIC-RELEASE-0.11.0.md).
Consumer migration remains a separate task.
The [0.11.0a1 qualification](verification/AUTOMATIC-EMBEDDING-0.11.0a1.md)
records deterministic coverage, live HTTP counts, timings and category budgets.
