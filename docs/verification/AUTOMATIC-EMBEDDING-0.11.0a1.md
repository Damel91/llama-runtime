# Optional direct embeddings: local candidate 0.11.0a1

Qualification date: 2026-10-22. Flower authority: `REQ-000038` revision 3,
`GOAL-000020`, `MILE-000021`, `CHANGE-000022`, `PACKET-000023` spec 14 / plan 9.
This records host-reported implementation and qualification, not human acceptance
or a public release. Consumers, model profiles, services and shared installations
were not changed.

## Contract verified

- Existing declared-model exact preparation and embedding remain available.
- Ordinary automatic initialization, availability, sync embedding, queries and
  async windows do not load, require or create calibration statistics. They send
  strings directly through the existing persistent engine and scheduler.
- Only `character_budget()` may read or create the optional calibration. The
  consumer owns splitting and can persist the returned recommendation; subsequent
  embedding calls need neither a budget argument nor a calibration cache.
- Explicit calibration counts all 10,000 original packaged samples, including the
  task prefix and native special-token policy. Single categories use 80% of their
  measured characters/token ratio; `mixed` uses 90%. Raw statistics are independent
  of these factors.
- Direct queue reservations are resource accounting, not measured token counts
  or input admission. Native overflows remain explicit; no truncation or failed
  POST replay is introduced.

Public signatures, configuration and a consumer example are in
[Automatic embeddings](../AUTOMATIC-EMBEDDING.md).

## Deterministic verification and packaging

The complete suite passed: **1,730 tests in 65.11 seconds**. Four expected
`InsecureRequestWarning` messages came from loopback HTTPS transport tests that
deliberately disable certificate verification. Ruff checks and formatting passed.
The wheel and source distribution built successfully.

Coverage includes corpus counts/hash/uniqueness; complete atomic caches and
corrupt/mismatched/partial cache rejection; task/prefix identity; 80%/90% math;
concurrent explicit probes sharing one calibration; ordinary direct calls and
availability progressing during a blocked calibration; independent cancellation;
finite deadlines and drained close; direct sync/async ordering and backpressure;
unknown or changed identity/build; native overflow without replay; empty or
unwritable cache independence; and unchanged exact preparation behavior.

A clean isolated installation outside the source tree loaded all public APIs
and all 10,000 corpus records. Packaged Python and corpus bytes matched source
and installed contents. The constructor remained inert without a client or
adapter. The wheel is platform-independent Python; the existing optional tsnet
helper retains its separate build and provenance requirements.

Corpus SHA-256:
`6a4733cf1d175b8b5b254f5951d19c5e4e5f5db2da8eaa1e15904474c53d3035`.
Five categories contain 2,000 unique samples each; `mixed` aggregates them.

## Live environment and procedure

The installed wheel used the already loaded two replicas of
`nomic-embed-text-v2-moe`, dimension 768, native context/batch/microbatch 512,
one slot per replica, mean pooling and flash attention enabled. Native build:
`b11716-797de63b2`.

Verified weights SHA-256:
`0292bc5c5aa4c434d195a9637e3bf19fa538177ef768b8c4d7154c49e0a63e98`.
Embedding-semantics SHA-256:
`906eaf94c9685db2d76438ef8adcfa0f66c147381fcf666deba21aa0ef0a3a56`.

The procedure first queried availability and embedded the same 511 synthetic
documents with a nonexistent calibration directory. It verified that the
directory still did not exist. Only then did it explicitly request the character
budget, completing the full calibration. It selected all categories, constructed
a new service, sent more strings before any warm budget request, loaded the
completed profile explicitly, and compared the original 511 outputs with exact
preparation plus embedding. No LLM generation or model administration was used.

The 511-text fixture SHA-256 (JSON, Unicode preserved):
`80a782175c81f6325ee5d2ff6c1ae9a269c8d965bc4fdf0efdec2c0abdbde3fa`.
Each source is 115 Unicode code points; the document prefix is added once.

| Phase | Wall seconds | Tokenize POSTs | Embedding POSTs | Metadata GETs | Total HTTP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold initialize and status | 0.0683 | 0 | 0 | 10 | 10 |
| Direct 511, before calibration | 1.5626 | 0 | 4 | 16 | 20 |
| Explicit cold character budget | 54.3861 | 10,000 | 0 | 7 | 10,007 |
| Six explicit category selections | 0.2162 | 0 | 0 | 24 | 24 |
| New service initialize and status | 0.1242 | 0 | 0 | 10 | 10 |
| New service direct three inputs | 0.1009 | 0 | 2 | 11 | 13 |
| Explicit warm character budget | 0.1097 | 0 | 0 | 4 | 4 |
| Exact prepare and embed 511 | 3.9873 | 511 | 4 | 15 | 530 |

Both 511-input paths sent two embedding requests to each replica and reached
two simultaneous embedding HTTP requests. These are client wall times including
the applicable preparation, discovery, transport and server work, not pure GPU
times. This is one bounded adjacent comparison, not a statistical benchmark or
a throughput guarantee under other loads.

The exact path measured 18,192 tokens in total. All 511 corresponding vectors
were identical: maximum absolute difference **0**, minimum cosine **1**, maximum
normalization error **2.22e-16**. Existing thresholds remained **1e-5** maximum
absolute difference, **0.999999** minimum cosine and **1e-6** normalization error.

## Explicit character recommendations

These recommendations apply to the observed 512-token native input allowance
and document prefix `search_document: `. They are empirical snapshots, not
guaranteed tokenizer bounds for arbitrary source text.

| Category | Samples | Raw characters/token | Factor | Recommended characters |
| --- | ---: | ---: | ---: | ---: |
| code | 2,000 | 2.751983 | 0.80 | 1,127 |
| documentation | 2,000 | 4.118999 | 0.80 | 1,687 |
| prose | 2,000 | 4.271923 | 0.80 | 1,749 |
| structured | 2,000 | 2.846118 | 0.80 | 1,165 |
| multilingual | 2,000 | 3.173459 | 0.80 | 1,299 |
| mixed | 10,000 | 3.274915 | 0.90 | 1,509 |

Changing category reused the same completed calibration and issued no tokenizer
requests. Prefix/task, verified identity, tokenizer build or corpus changes
require a distinct profile; context changes update recommendations without
retokenizing unchanged statistics. In-process concurrent probes are deduplicated;
separate cold processes can duplicate work while atomic cache writes remain
coherent. The ordinary direct path uses none of this cache state.

## Delivery boundary

The local candidate is `runtime-llama` **0.11.0a1**, import `runtime_llama`.
Wheel, source archive, hashes, raw timing receipts and test logs are retained as
local qualification artifacts outside the public source tree. This work does
not install into shared consumer environments or publish a release.
