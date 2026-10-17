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
| Usage and outcomes | Token counters, terminal reasons and bounded transport metadata; unavailable counters remain unknown. |
| Embeddings | Query, document and raw input modes; optional normalization; validates result order, cardinality, vector dimensions and finite values. |
| Task-aware preparation | Configurable model-specific query/document prefixes and exact server tokenization, including special tokens; measures oversized inputs and safely reuses verified counts. |
| Batching | Configurable total text windows, balanced quotas across usable replicas and HTTP packing within known input/context and aggregate token budgets. Inputs are never silently truncated. |
| Replica discovery | Explicit aliases or automatic discovery using content and semantic identity tags from a compatible server manager; identity pinning and bounded topology refresh. |
| Shared concurrency | Per-adapter global and per-replica admission, bounded queues, backpressure, separate tokenizer limits and ordered results across concurrent callers. Each automatic replica admits one active embedding request. |
| Lifecycle and errors | Typed backend, timeout, cancellation and runaway errors; cooperative cancellation, repetition protection and drained embedding shutdown. Embedding calls have overall deadlines; generation supports bounded HTTP and stream-idle waits. |
| Optional stream idle | Explicit `stream_idle_timeout_sec=None` waits through quiet prefill; connection/JSON budgets remain finite, with per-request socket interruption for Stop and drained close. |
| Diagnostics | Optional bounded thinking capture, invocation correlation and final diagnostic sinks, including interrupted calls. |
| HTTP access | Authentication and TLS verification, raw requests and helpers for chat/completion, embeddings, tokenization/detokenization, health, server properties and model catalogs. |
| Read-only model catalog | Typed observations of catalog entries and proven loaded models, explicit capability unknowns, identity tags and context provenance; bounded GET-only discovery without model loading. |
| MCP data contracts | Validated integration, call, event and result types for consumer-managed MCP execution. |

The consumer executes tools and MCP calls and supplies subsequent conversation
messages. Reasoning and structured-output behavior depend on the selected
server build, model and chat template. Responses are not locally validated
against the forwarded JSON Schema. Splitting source documents and tracking their
fragments belong to the consuming application. The server's own auto-loading
policy is supported.

## Install and generate

With Python 3.11 or later, install the tagged source:

```bash
python -m pip install 'git+https://github.com/Damel91/llama-runtime.git@v0.5.0'
```

Alternatively, install a wheel from the [GitHub release](https://github.com/Damel91/llama-runtime/releases/tag/v0.5.0),
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
dimensions and finite values. Only the recognized structured context error
allows adaptive batch splitting. Unknown limits remain unknown. Query and
document prefixes default to empty strings; `input_type="raw"` accepts texts
already prepared by the consumer.

An adapter can share bounded concurrency and queue limits across callers.
Optional `embedding_replica_discovery=True` selects loaded replicas using
content and semantic identity tags published by a compatible server manager.
Each automatically discovered replica admits one active request. Explicit
replica aliases remain supported. Without discovery or concurrency settings,
embedding requests remain sequential. See the
[integration guide](docs/INTEGRATION.md) and
[replica discovery contract](docs/AUTOMATIC-EMBEDDING-REPLICAS.md).

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

This public release makes no new live qualification or consumer acceptance
claim. Local tests, a particular server/model qualification and application
acceptance are separate. Closing a client stream does not by itself prove
that remote inference stopped. Missing usage values remain missing.

See the [0.5.0 release verification](docs/verification/PUBLIC-RELEASE-0.5.0.md)
for the checks performed on this distribution. Release assets include SHA-256
checksums for the wheel and source archive.

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

Flower is development tooling. Installing and using the library requires only
the Python dependencies declared by this package.

## Documentation and license

- [Functional requirements](docs/REQUIREMENTS.md)
- [Architecture and mapping](docs/ARCHITECTURE.md)
- [Functional parity and limits](docs/FUNCTIONAL-PARITY.md)
- [Read-only model discovery](docs/MODEL-DISCOVERY.md)
- [Embedding preparation and batching](docs/EMBEDDING-CONTRACTS.md)
- [Concurrency and diagnostics](docs/CONCURRENCY-AND-DIAGNOSTICS.md)
- [Stream timeouts and cancellation](docs/STREAM-TIMEOUTS.md)
- [Provenance and attribution](docs/PROVENANCE.md)

Copyright 2026 Davide Mele. The project is licensed under
[Apache-2.0](LICENSE). The client adapted from llamatelemetry retains its
[upstream MIT license](src/runtime_llama/_vendor/llamatelemetry.LICENSE).
See [NOTICE](NOTICE) for attribution.
