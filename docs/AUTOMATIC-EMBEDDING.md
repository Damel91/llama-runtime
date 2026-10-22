# Automatic category-calibrated embeddings

This optional path is governed by Flower `REQ-000038`, `GOAL-000019`,
`MILE-000021`, `CHANGE-000022`, `PACKET-000023`. The existing declared-model
adapter and its exact `prepare_*` / `embed_prepared` contracts remain available.

The automatic service discovers one unambiguous, verified, loaded embedding
identity group. It owns aliases, replica capacity, prefixes, preparation and
dispatch. It uses the existing persistent embedding engine and scheduler.
It never loads, restarts or administers remote models.

## Categories and first initialization

The packaged `runtime_llama/calibration_corpus/` directory contains exactly
10,000 original synthetic texts: 2,000 each in `code`, `documentation`, `prose`,
`structured` and `multilingual`. `mixed` aggregates the complete corpus. The
generator, category counts and manifest make the corpus reproducible; its
purpose is calibration, not a universal or naturally sampled language benchmark.

Initial calibration tokenizes the entire corpus, with the selected task prefix
and native special-token policy. All category statistics are calculated in one
pass. Changing category reuses that complete profile. A different prefix/task
policy needs its own calibration, as do changed weights, semantic identity,
tokenizer build or corpus revision. Equivalent replica aliases share profiles.

For each category and for `mixed`, the operational ratio is
`sum(source Unicode characters) / sum(actual fully prepared tokens)`.
The effective coefficient is 85% of this ratio. Prefix and special-token cost
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
Automatic admission conservatively respects the smallest observed input
context and native microbatch; native computation sizes are not aggregate HTTP
token budgets. Multiple unrelated embedding groups require explicit selection.

Ordinary calls send strings directly, without per-input `/tokenize`. Token
ledgers in this path are estimates; byte, text-count and response bounds remain
hard bounds. Prefixes are applied once, and ordering, owner cancellation,
finite deadlines, backpressure and drained shutdown use the original engine.
Exact public preparations are never populated with estimated counts.

An opaque representation identity protects stored vector spaces. Automatic
discovery must not silently replace an index's embedding model. A consumer
still owns source splitting, UTF-8 fragment intervals, graph relationships and
the decision to rebuild an index after a representation change.

## Verification

Implementation, deterministic verification, live qualification and human
acceptance are separate. Initial 10,000-text live calibration is explicitly
authorized for this cycle; it is not run at import or package construction.
Public release and consumer migration are outside this implementation scope.
