# llama-runtime

A Python library for **llama.cpp / llama-server HTTP servers**, providing
generation and embedding adapters for local servers, remote hosts and private
networks. The Python distribution is `runtime-llama` and the import namespace is
`runtime_llama`. Python 3.11 or later is required. Runtime dependencies are
`requests` and `urllib3`.

## Why this library exists

`llama-runtime` was created to give agents, coding tools, retrieval pipelines
and other Python applications reusable building blocks for llama.cpp inference.
It brings conversation handling, streaming, tool-call data, exact embedding
preparation and replica scheduling into a common client API.

Applications configure inference requests and consume typed results. The server
operator controls the llama.cpp build, model loading, context allocation,
processes and hardware. This separation lets applications use a manually
started server, a managed service or a remote endpoint through the same adapters.
Sampling settings can be adjusted per request while a model stays loaded.

The project's engineering lifecycle is managed with
[Flower MCP](#development-with-flower-mcp).

## Features

| Capability | What the library provides |
| --- | --- |
| Text generation | Blocking calls, text streaming and streams containing text, reasoning, tool fragments and final metadata. |
| Conversations | Structured message histories, including system, developer, user, assistant and tool messages; preserves tool identifiers and additional message fields. |
| Inference controls | Output budgets, temperature, top-p, top-k, min-p, repetition penalties, template options and per-call server extensions. |
| Structured output | JSON output mode and JSON schema forwarding, subject to the server build and model's support. |
| Reasoning | Separate reasoning output, configurable reasoning policies and removal of inline thinking from visible text. |
| Tool calls | Tool definitions and choice controls, streamed argument assembly, complete structured calls and correlation with tool results. |
| Usage and outcomes | Token counters, terminal reasons, bounded transport metadata and optional typed native server timings; unavailable counters remain unknown. |
| Embeddings | Query, document and raw input modes; optional normalization; validates result order, cardinality, vector dimensions and finite values. |
| Task-aware preparation | Configurable model-specific query/document prefixes and exact server tokenization, including special tokens; measures oversized inputs and safely reuses verified counts. |
| Batching | Configurable total text windows, balanced quotas across usable replicas and HTTP packing within known input/context and explicitly configured aggregate token budgets; native computation batch sizes remain separate. Inputs are never silently truncated. |
| Replica discovery | Explicit aliases or automatic discovery using content and semantic identity tags from a compatible server manager; identity pinning and bounded topology refresh. |
| Shared concurrency | Per-adapter global and per-replica admission, bounded queues, backpressure, separate tokenizer limits and ordered results across concurrent callers. Effective admission is bounded by observed slots and configured global/per-replica limits; unknown capacity falls back to one. |
| Async embedding windows | Async text or prepared-input sources, ordered complete windows, bounded intake and buffered results, with the same scheduler and limits used by synchronous callers. |
| Persistent embedding queues | Prepared records accumulate in adapter-owned queues for each verified replica; batches are formed when execution capacity is available, with the same engine for synchronous and async callers. |
| Native token inputs | Reuses exact retained token IDs with verified weights, tokenizer build and prefix/special policy, avoiding a second string tokenization in the native embedding handler. |
| Optional automatic embeddings | Verified loaded-model discovery and direct text inputs using the existing scheduler; an explicit character-budget helper calibrates 10,000 packaged texts by category when requested. |
| Lifecycle and errors | Typed backend, timeout, cancellation and runaway errors; cooperative cancellation, repetition protection and drained embedding shutdown. Embedding calls have overall deadlines; generation supports bounded HTTP and stream-idle waits. |
| Optional stream idle | Explicit `stream_idle_timeout_sec=None` waits through quiet prefill; connection/JSON budgets remain finite, with per-request socket interruption for Stop and drained close. |
| Diagnostics | Optional bounded thinking capture, invocation correlation and final diagnostic sinks, including interrupted calls. |
| HTTP access | Works with native unpatched llama-server using independent connections by default; explicit keep-alive opt-in, authentication, TLS verification and raw endpoint helpers. |
| Read-only model catalog | Typed observations of catalog entries and proven loaded models, explicit capability unknowns, identity tags and context provenance; bounded GET-only discovery without model loading. |
| Optional embedded VPN client | A separately built Go tsnet helper for Tailscale or Headscale, with consumer-owned configuration, authenticated loopback proxy, verified peer routing, TLS and bounded teardown; no system VPN installation. |
| MCP data contracts | Validated integration, call, event and result types for consumer-managed MCP execution. |

The consumer executes tools and MCP calls and supplies subsequent conversation
messages. Reasoning and structured-output behavior depend on the selected
server build, model and chat template. Responses are not locally validated
against the forwarded JSON Schema. Splitting source documents and tracking their
fragments belong to the consuming application. The server's own auto-loading
policy is supported.

## Prerelease 0.12.0a1

The automatic character recommendation uses the worst observed ratio across
the existing 10,000 calibration samples, with the same 80% category and 90%
mixed factors. Successful direct calls still perform no tokenization.

After a direct HTTP 500 drains, bounded exact diagnostics inspect only the
failed caller's actual POST inputs. A confirmed context overflow returns
`EmbeddingInputOverflowError` with verified lossless source spans and a
persisted absolute character ceiling. The consumer owns reprojection and an
explicit retry; no embedding request is replayed automatically. Unproven
overflow preserves the original 500. The existing scheduler is unchanged.

`refresh_character_budget()` observes current limits and reads completed
calibration/learned state without starting calibration. See the
[automatic API](docs/AUTOMATIC-EMBEDDING.md) and
[prerelease record](docs/verification/PUBLIC-RELEASE-0.12.0a1.md).
This is a public prerelease with an open numerical finding between the observed
replicas. Stable 0.11.1 remains the latest stable release. A separately authorized
consumer diagnostic ingestion completed with 126,045 vectors; that functional
result does not close numerical qualification.

## Release 0.11.1

The automatic embedding config now exposes the existing finite request,
decoded-response, pending-byte and pending-token budgets. Defaults stay at
8 MiB request/response, 64 MiB pending bytes and 1,048,576 pending reservation
units. `http_connection_policy="reuse"` selects existing keep-alive behavior
for an owned client; the default stays `"close"` and injected clients retain
their policy and ownership.

Calling the service's draining close from its own cancellation callback now
fails with a typed error before shutdown mutation, instead of deadlocking.
Cold explicit calibration checks persistence before remote tokenization;
readable valid caches remain usable without writes. Verified single-model
catalogs without status can be attested through coherent root properties.

The existing scheduler, exact/direct API separation and category/mixed factors
of 80%/90% are unchanged. High-context queue capacity is explicitly configurable;
no estimated token admission, automatic retry or silent truncation is added.
See the [hardening qualification](docs/verification/REVIEW-HARDENING-0.11.1.md)
and [public release record](docs/verification/PUBLIC-RELEASE-0.11.1.md).

## Earlier release 0.11.0

The optional [automatic embedding service](docs/AUTOMATIC-EMBEDDING.md)
reports availability and sends strings directly. The library owns replica
discovery, prefixes and dispatch. Only an explicit character-budget request can
start calibration over 10,000 original texts in five categories; `mixed` uses all
categories. Individual categories apply 80% of their
measured characters/token ratio; `mixed` applies 90%. Raw measurements persist
independently of these safety factors.

Ordinary inputs go directly to the server without per-input `/tokenize` or any
calibration prerequisite. The consumer owns chunking and may persist its budget.
These statistical budgets do not guarantee that every input fits;
native overflow remains an explicit error. The declared-model exact API stays
available with its original counting and preparation contracts.

The [qualification](docs/verification/AUTOMATIC-EMBEDDING-0.11.0a1.md)
records 1,730 passing tests and a bounded live comparison: 511 direct inputs in
1.56 seconds, four embedding POSTs plus 16 metadata GETs, no tokenizer calls,
both replicas used and vectors identical to the exact path. These are client
wall times, not pure GPU timings or a general performance guarantee.
See the [public release record](docs/verification/PUBLIC-RELEASE-0.11.0.md)
for stable packaging and provenance. Consumer installations remain unchanged.

## Earlier release 0.10.3

Version 0.10.3 applies the incremental efficiency review. Recovery for
eligible injected clients validates each pending input against its own limits
and the fresh server snapshot, retaining owner isolation and original result
indices. Native failed or ambiguous POSTs remain non-replayable.

`embedding_max_response_bytes` independently bounds decoded native JSON bodies
(default 8 MiB). `embedding_preparation_cache_entries` and
`embedding_preparation_cache_bytes` configure exact-token retention (defaults
1,024 and 8 MiB); zero disables retention. Effective cache bytes remain capped
by `embedding_max_pending_bytes // 4`. Request, response, pending-record and
result budgets describe different resources.

The coordinator waits for events and relevant deadlines instead of waking
every 10 ms. Arbitrary cancellation callbacks and custom coalescing clocks
retain finite checks. Planning avoids unused deduplication keys, repeated
plan-count scans, rotating list copies and index searches. Existing sync/async
APIs, numerical thresholds, fairness, backpressure and drained ownership remain
in force.

See the [0.10.3 review qualification](docs/verification/REVIEW-EFFICIENCY-0.10.3.md)
for regression coverage, package checks and the bounded HTTP-policy comparison.
The [public release record](docs/verification/PUBLIC-RELEASE-0.10.3.md) describes
normalized assets, retained helper provenance and installation checks.

## Earlier release 0.10.2

Version 0.10.2 reduces redundant metadata observations while preserving
fresh identity and loaded-limit checks before inference. It also retains a
bounded adapter-owned cache of completed exact token IDs, keyed by verified
weights, tokenizer build, task and prefix/special-token policy. A matching
prepared or raw input can reuse IDs across equivalent aliases; a cache entry
does not authorize inference or bypass current limits.

The cache retains at most 1,024 entries and `min(8 MiB,
embedding_max_pending_bytes // 4)` bytes, separately from queued work and async
results. It has no TTL or singleflight: concurrent cold misses can each need
tokenization. Unknown identity/build bypasses it; eviction and adapter close
release retained entries. `health_status` includes preparation-cache counters.
These resource pools do not define a total process memory limit.

Ready queues maintain incremental summaries and packing walks only the needed
queue prefix through iterators. Candidate batches are still selected at
dispatch; no frozen prefix is retained while capacity is unavailable. If an
external exception interrupts a blocking call's result wait, the library stops
that owner, drains its real work and reraises the original exception while
preserving surviving callers and the shared adapter.

Qualification separates the 511 previously unseen texts from a later prepared/raw
reuse exercise. The existing oversized-input check runs first with an unrelated
text; it does not populate the cache for those 511 inputs. Worker-time sums and
HTTP intervals are reported separately; neither is pure GPU time. The qualified
511-text run used 24 metadata requests instead of 50, with 511 exact tokenizer
requests and four embedding POSTs. It does not demonstrate a wall-time speedup.
Consumer installation remains a separate action.

See the [0.10.2 qualification report](docs/verification/METADATA-PREPARATION-0.10.2.md)
for interruption, packing, request counts and exact-reuse evidence.
The [public release record](docs/verification/PUBLIC-RELEASE-0.10.2.md) describes
normalized assets, unchanged helper provenance and installation checks.

## Earlier release 0.10.1

Version 0.10.1 corrects model-wide accumulation in the persistent embedding
engine. Tokenizer workers publish ready records into queues owned by
each verified replica. The engine forms a batch when global and replica
capacity are available, including records prepared while earlier requests were
running. Synchronous calls and async window sources share admission, ordering,
deadlines and drained cleanup. Compatible records can share an HTTP batch
across source windows or callers, while preserving each owner's output indices
and cancellation. Stopping one owner does not abort survivors; its completion
waits for the actual shared transport drain.

`embedding_batch_wait_sec` defaults to `0.0`: preparation and capacity events
fill batches without a required timer. A first ready record does not force a
small POST while compatible admitted work can still fill it. Source windows
remain delivery boundaries and do not shrink the per-replica request target.
An explicit positive value enables bounded coalescing
in the same engine. The total-text meaning of `batch_size` is unchanged;
context, aggregate HTTP token budgets and native computation batches remain
separate. Queue resource limits are configurable with
`embedding_max_pending_bytes` (64 MiB), `embedding_max_pending_tokens`
(1,048,576) and `embedding_max_request_bytes` (8 MiB).

`embedding_input_format="auto"` retains the verified native token-ID path.
New texts still need exact server tokenization; the queue rewrite does not
introduce a bulk tokenizer endpoint or eliminate those requests. See the
[queue and token contract](docs/PROGRESSIVE-EMBEDDING.md) and
[current architecture](docs/ARCHITECTURE.md).
The [public release record](docs/verification/PUBLIC-RELEASE-0.10.1.md) describes
normalized assets, retained helper provenance and verification boundaries.
Consumer installation remains a separate action. The
[0.9.0 qualification](docs/verification/PROGRESSIVE-NATIVE-TOKENS-0.9.0.md)
records historical measurements. The [model batching verification record](docs/verification/MODEL-BATCHING-0.10.1.md)
separates deterministic results, installed-wheel checks and bounded functional
live evidence. No comparison with 0.9.0 was run for this correction.

## Earlier release 0.7.0

Version 0.7.0 corrects native embedding batch budgets, overlaps bounded
preparation with ready HTTP work and supports unpatched llama.cpp through the
standard HTTP connection-close policy. It also reduces generation bookkeeping
and exposes optional native timings. No server, model or consumer migration is
performed by the library. See [native HTTP and timing APIs](docs/NATIVE-HTTP.md)
and [embedding contracts](docs/EMBEDDING-CONTRACTS.md). Qualification includes
two GPU embedding replicas with Flash Attention disabled, unchanged numerical
thresholds and the previously failing corpus. See the
[release verification](docs/verification/PUBLIC-RELEASE-0.7.0.md) for measured
results and deployment boundaries.

## Install and generate

With Python 3.11 or later, install the tagged source:

```bash
python -m pip install 'git+https://github.com/Damel91/llama-runtime.git@v0.11.1'
```

Alternatively, install a wheel from the [GitHub release](https://github.com/Damel91/llama-runtime/releases/tag/v0.11.1),
or run the following from a checkout:

```bash
python -m pip install .
```

Start and configure your server separately. Replace the example model alias and
endpoint with values from its catalog; `base_url` should not include `/v1`.

```python
from runtime_llama import build_adapter

llm = build_adapter({
    "role_id": "coder",
    "model": "chat-model",
    "base_url": "http://localhost:8080",
    "temperature": 0.2,
    "num_predict": 512,
    "reasoning": "auto",
    "stream_idle_timeout_sec": 60,
    # "api_key_env": "LLAMA_API_KEY",
})
try:
    result = llm.invoke_with_ctx(
        [{"role": "user", "content": "Explain this algorithm."}],
        request_options={"temperature": 0.4, "top_k": 40},
    )
    print(result.text)
    print(result.reasoning_text, result.usage, result.terminal_reason)
finally:
    llm.close()
```

`stream()` yields visible text. `stream_with_metadata()` yields `LLMStreamEvent`
values containing reasoning, usage, tool fragments and terminal information.
Close a generator when stopping early, using `finally` or `contextlib.closing`.
A `cancellation_probe` supports cooperative cancellation while waiting for a
response.

To wait through long silent prefill, set `stream_idle_timeout_sec` to `None`
(`null` in JSON). The default remains 30 seconds; positive per-call `timeout`
restores a finite idle bound. Connection, JSON, discovery and embedding budgets
remain finite. See [stream timeouts and cancellation](docs/STREAM-TIMEOUTS.md)
for the bridge-owned client pattern, authentication policy and cleanup limits.

`num_predict` maps to the request's `max_tokens`. `request_options` accepts
sampling overrides and server extensions while protecting managed request
fields. `num_ctx` is a logical consumer limit; it does not reallocate server
context. Reasoning, templates and JSON schema support depend on the server
build and model. The library forwards JSON schemas without locally validating
the response against JSON Schema.

## Discover available and loaded models

```python
from runtime_llama import discover_models

catalog = discover_models("https://inference.example.net", timeout=5.0)
for model in catalog.loaded_models:
    print(model.id, model.identity.model_name)
    print(model.capabilities.chat, model.capabilities.embeddings)
```

`catalog.models` contains the observed catalog; `loaded_models` contains only
entries proved loaded. Capabilities remain `None` when the server does not
declare them. The API uses a bounded `GET /v1/models` and optional root
`GET /props`, with one deadline and cooperative cancellation. It neither
selects nor loads a model. Authentication is optional and explicit; TLS is
verified and ambient authentication/proxy settings are ignored. See the
[model discovery API](docs/MODEL-DISCOVERY.md) for signatures, identity tags,
context provenance, limits and failure handling.

## Optional Tailscale and Headscale transport

Version `0.6.0` adds `TailnetConfig`, `TailnetHelper` and `TailnetError`.
The consumer supplies the native helper executable, a dedicated node-state
directory, hostname, enrollment credentials when needed, coordination server
and inference endpoint. The Python wheel remains portable and contains no
native binary. Importing or constructing the library never launches a VPN.

The [release](https://github.com/Damel91/llama-runtime/releases/tag/v0.11.1)
provides separate helper archives for macOS ARM64, Linux AMD64 and Windows
AMD64. Go is needed to build the helper from source, not to run a supplied
binary. Both Tailscale's coordinator and an explicit HTTPS Headscale
coordinator are supported by configuration.

`TailnetHelper` provides managed raw clients and discovery. Its ephemeral
`proxy_url` also routes generation and embedding adapter profiles through the
same authenticated helper. Request Stop stays separate from shared-helper
shutdown. Configuration belongs to the consumer and is never persisted in
the repository. See the [tailnet guide](docs/TAILNET.md) for examples, build
commands, ownership, supported peer addresses and qualification boundaries.

## Tool calls and MCP

`generate_with_metadata(messages, request_options={"tools": [...]})` preserves
complete calls in `result.tool_calls`. Dictionary messages preserve
`tool_calls`, `tool_call_id`, arguments and additional fields. The consumer
executes tools, appends their results with the `tool` role and continues the
conversation. MCP data contracts are included; transport and tool execution
belong to the consuming application.

## Embeddings

Use a server and model configured for embeddings, potentially at a separate
endpoint. The prefixes below illustrate document retrieval with
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2); configure
prefixes and limits for your model.

```python
from runtime_llama import build_embedding_adapter

embedding = build_embedding_adapter({
    "model": "embedding-model",
    "base_url": "http://localhost:8081",
    "max_input_tokens": 2048,
    "batch_size": 256,
    "token_count_parallelism": 2,
    "normalize": True,
    "embedding_prefixes": {
        "query": "task: search result | query: ",
        "document": "title: none | text: ",
    },
})
try:
    prepared = embedding.prepare_texts(["first document", "second document"])
    print([(item.token_count, item.max_input_tokens) for item in prepared])
    vectors = embedding.embed_prepared(prepared)
    query_vector = embedding.embed_query("How is the runtime configured?")
finally:
    embedding.close()
```

`batch_size` is the total number of texts per distribution window. With two
usable replicas, a window of 256 texts receives balanced quotas of 128 texts
per replica; token packing may require several HTTP requests per quota.
`prepare_texts()` and `prepare_query()` measure oversized inputs too.
`embed_prepared()` reuses verified counts and rejects inputs exceeding known
limits. Prefixes and special tokens are counted once. Splitting documents and
tracking fragment identity belong to the consumer.

The adapter uses the server tokenizer, preserves order and checks vector
dimensions and finite values. Native POST errors are propagated without replay,
including context errors whose sibling inputs may already have run. Recognized
context recovery remains for custom clients outside the native contract.
Unknown limits remain unknown. Query and
document prefixes default to empty strings; `input_type="raw"` accepts texts
already prepared by the consumer.

An adapter can share bounded concurrency and queue limits across callers.
Optional `embedding_replica_discovery=True` selects loaded replicas using
content and semantic identity tags published by a compatible server manager.
Each replica is bounded by its observed slots and configured admission limits;
unknown capacity falls back to one. Explicit
replica aliases remain supported. Without discovery or concurrency settings,
embedding requests remain sequential. See the
[integration guide](docs/INTEGRATION.md) and
[replica discovery contract](docs/AUTOMATIC-EMBEDDING-REPLICAS.md).

### Asynchronous window sources

Introduced in 0.8.0, `astream_texts()` and `astream_prepared()` remain available.
The synchronous API remains available. A consumer can supply successive
windows while earlier windows are preparing or running:

```python
async def incoming_windows():
    yield ["first document", "second document"]
    yield ["third document"]

async def ingest(embedding):
    try:
        async with embedding.astream_texts(
            incoming_windows(), max_pending_windows=2, timeout=120.0,
        ) as stream:
            async for vectors in stream:
                await store_next_window(vectors)  # Application-owned storage.
    finally:
        await embedding.aclose()
```

Each yielded vector list corresponds to one source window, in order.
`max_pending_windows` includes source pulls, preparation, inference and results
waiting for the consumer. Global intake is bounded across the adapter's async
streams; synchronous and asynchronous work share HTTP scheduling and replica
limits. `batch_size` still controls total-text distribution windows internally.
The finite timeout applies separately to each source window, including waiting
for capacity and the source itself.

Use `async with` or `await stream.aclose()` when stopping early. Closing a stream
drains its work and leaves the shared adapter available; `await embedding.aclose()`
closes the adapter and its active streams. Source iterators are borrowed unless
`close_source=True` is requested. See the [async window API](docs/ASYNC-EMBEDDING-WINDOWS.md)
for ownership, deadlines, error propagation and event-loop constraints.
The [0.8.0 candidate qualification](docs/verification/ASYNC-WINDOWS-0.8.0.md)
records deterministic and bounded live results, including both embedding replicas.

## Verification

Run the deterministic tests, HTTP loopback fixtures, lint checks and package
build from the checkout:

```bash
python -m pip install '.[dev]'
python -m pytest -q
ruff check src tests probes _build_backend.py
ruff format --check src tests probes _build_backend.py
python -m build --no-isolation
```

Install the built wheel in a clean environment outside the checkout to verify
its imports and distribution metadata. Reusable [live probes](probes/README.txt)
require explicit opt-in and your own endpoint, model and build identifiers.
They describe reproducible checks rather than a certification of your setup.

Historical release reports include bounded live qualification of their stated
profiles, without claiming consumer acceptance. Local tests, a
particular server/model qualification and application
acceptance are separate. Closing a client stream does not by itself prove
that remote inference stopped. Missing usage values remain missing.

See the [0.7.0 release verification](docs/verification/PUBLIC-RELEASE-0.7.0.md)
for the checks performed on that release. Its assets include SHA-256
checksums for the wheel, source archive, native helper archives and source
manifest. Native Linux/Windows assets are crossbuilt; this release does not
claim execution on those systems or live VPN enrollment/inference.

## Development with Flower MCP

This library's development lifecycle is managed with
[Flower MCP](https://github.com/Damel91/flower-mcp). The associated Flower
project is `llama-runtime`. Requirements, milestones, implementation packets,
plan validation, verification evidence and handovers are maintained in its
persistent project ledger.

The Markdown documentation in this repository is intentionally focused on
installation, public APIs, architecture and contributor guidance. The detailed
engineering lifecycle and its history live in Flower, which keeps the repository
documentation compact instead of duplicating the complete project record.

Flower is development tooling. Core library use requires the Python
dependencies declared by this package. The optional tailnet transport also
requires its separately distributed native helper.

## Documentation and license

- [Functional requirements](docs/REQUIREMENTS.md)
- [Architecture and mapping](docs/ARCHITECTURE.md)
- [Functional parity and limits](docs/FUNCTIONAL-PARITY.md)
- [Read-only model discovery](docs/MODEL-DISCOVERY.md)
- [Embedding preparation and batching](docs/EMBEDDING-CONTRACTS.md)
- [Progressive preparation and native token inputs](docs/PROGRESSIVE-EMBEDDING.md)
- [Asynchronous embedding windows](docs/ASYNC-EMBEDDING-WINDOWS.md)
- [Concurrency and diagnostics](docs/CONCURRENCY-AND-DIAGNOSTICS.md)
- [Stream timeouts and cancellation](docs/STREAM-TIMEOUTS.md)
- [Optional Tailscale and Headscale transport](docs/TAILNET.md)
- [Provenance and attribution](docs/PROVENANCE.md)

Copyright 2026 Davide Mele. The project is licensed under
[Apache-2.0](LICENSE). The client adapted from llamatelemetry retains its
[upstream MIT license](src/runtime_llama/_vendor/llamatelemetry.LICENSE).
See [NOTICE](NOTICE) for attribution.
The optional native helper retains Tailscale's BSD-3-Clause license and the
dependency notices shipped with each helper archive.
