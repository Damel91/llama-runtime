# llama-runtime

Python adapters for **llama.cpp / llama-server HTTP servers**: streaming
generation, reasoning, structured tool calls, embeddings and replica discovery.
Connect to a server on your machine, a remote host or a private network while
keeping control of its build, model loading and hardware configuration.

| Capability | Library behavior |
| --- | --- |
| Conversations | Preserves message roles, structured tool calls and tool results. |
| Generation | Supports streaming, separate reasoning, usage and request-time sampling overrides. |
| Embeddings | Applies task-aware prefixes, measures exact tokens and enforces known input limits. |
| Replica scheduling | Discovers verified loaded replicas, balances text quotas and bounds shared concurrency. |
| Client lifecycle | Provides finite timeouts, cancellation, typed errors and drained shutdown. |

The repository is named `llama-runtime`. The Python distribution remains
`runtime-llama` and imports remain `runtime_llama` for compatibility. Version
`0.4.1` updates publication metadata, documentation and licensing; it preserves
the inference and embedding behavior of `0.4.0`.

The only runtime dependency is `requests`. The HTTP client is adapted from
llamatelemetry's `LlamaCppClient` and isolated from its package initialization
and server bootstrap. Importing this library does not require OpenAI, LM Studio,
llamatelemetry or local inference bindings.

The server operator owns builds, processes, model files, loading and hardware.
The library connects to the configured server. Server-managed auto-loading is
allowed; the client does not impose `autoload=false`.

## Install and generate

With Python 3.11 or later, install the tagged source:

```bash
python -m pip install 'git+https://github.com/Damel91/llama-runtime.git@v0.4.1'
```

Alternatively, install a wheel from the [GitHub release](https://github.com/Damel91/llama-runtime/releases/tag/v0.4.1),
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

`num_predict` maps to the request's `max_tokens`. `request_options` accepts
sampling overrides and server extensions while protecting managed request
fields. `num_ctx` is a logical consumer limit; it does not reallocate server
context. Reasoning, templates and JSON schema support depend on the server
build and model. The library forwards JSON schemas without locally validating
the response against JSON Schema.

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

See the [0.4.1 release verification](docs/verification/PUBLIC-RELEASE-0.4.1.md)
for the checks performed on this distribution. Release assets include SHA-256
checksums for the wheel and source archive.

## Documentation and license

- [Functional requirements](docs/REQUIREMENTS.md)
- [Architecture and mapping](docs/ARCHITECTURE.md)
- [Functional parity and limits](docs/FUNCTIONAL-PARITY.md)
- [Embedding preparation and batching](docs/EMBEDDING-CONTRACTS.md)
- [Concurrency and diagnostics](docs/CONCURRENCY-AND-DIAGNOSTICS.md)
- [Provenance and attribution](docs/PROVENANCE.md)

Copyright 2026 Davide Mele. The project is licensed under
[Apache-2.0](LICENSE). The client adapted from llamatelemetry retains its
[upstream MIT license](src/runtime_llama/_vendor/llamatelemetry.LICENSE).
See [NOTICE](NOTICE) for attribution.
