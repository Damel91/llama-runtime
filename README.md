# llama-runtime

**Use llama.cpp and Strata from Python or through a local HTTP relay.**

`llama-runtime` connects coding tools, agents and retrieval pipelines to
inference servers you operate. Give it a model ID and configured endpoints.
It discovers the loaded models and routes each chat request to the matching
runtime. Use the Python SDK in your application, or run the relay to connect
an OpenAI-compatible client such as OpenCode.

Python **3.11+** · [Latest release](https://github.com/Damel91/llama-runtime/releases/latest) · [Apache-2.0](LICENSE)

[Quick start](#quick-start) · [Relay and OpenCode](#use-the-local-relay) · [Documentation](#learn-more) · [Development with Flower MCP](#development-with-flower-mcp)

## What you gain

- **Choose the model in the client.** Discover the loaded catalog on each
  configured endpoint. Route chat to llama.cpp or Strata by model ID and keep
  that route for the duration of the call.
- **Build on structured chat.** Send message histories, stream text and
  reasoning, request structured output and access tool calls, token usage and
  terminal outcomes when the server supplies them. Sampling and output budgets
  can be set per request.
- **Recover from invalid tool proposals.** The SDK validates proposals against
  declared tools and can request a correction for recoverable errors. The
  application executes accepted tool calls and returns their results.
- **Embed documents and queries with llama.cpp.** Configure model-specific
  prefixes, use exact token counts or direct text input, and stream indexed
  results in source order. The shared scheduler distributes batches across
  verified replicas.
- **Connect HTTP clients through the same SDK.** The optional relay exposes
  chat completions, embeddings and a model catalog. Use its browser console to
  configure connections, inspect requests, choose a theme and export OpenCode
  settings with per-model limits.
- **Reach private inference endpoints.** The optional userspace helper connects
  through Tailscale or Headscale using consumer-supplied settings.

## How it works

You operate the inference servers and decide which models to load, how much
context to allocate and which hardware to use. The SDK manages request routing,
chat and embedding contracts, cancellation and cleanup. Your application owns
its conversations, tool execution, document splitting and persistence.

The relay runs the same SDK behind a local HTTP interface. OpenCode connects to
that interface, and the relay selects the remote runtime for the requested
model. The browser console manages the relay and observes the model catalog.

## Quick start

Download the wheel and `SHA256SUMS` from the [latest release](https://github.com/Damel91/llama-runtime/releases/latest),
verify the wheel against `SHA256SUMS`, then install it with `python -m pip
install` and the path to the wheel. The following command installs from a source checkout.

```sh
python -m pip install .
```

Start the inference server and load a chat model. In this llama.cpp example,
replace the model alias and server origin with values from your deployment.
The SDK's `base_url` is the server origin, without `/v1`.

```python
from runtime_llama import build_adapter

llm = build_adapter({
    "model": "chat-model",
    "base_url": "http://127.0.0.1:8080",
    "temperature": 0.2,
    "num_predict": 512,
})
try:
    result = llm.generate_with_metadata("Explain this algorithm.")
    print(result.text)
finally:
    llm.close()
```

To configure multiple servers or use Strata, see
[automatic runtime discovery](docs/AUTOMATIC-RUNTIMES.md).

### Use the local relay

Install the `relay` extra from a source checkout and start the local server.

```sh
python -m pip install '.[relay]'
llama-runtime-relay --config ./relay.json
```

1. Open `http://127.0.0.1:8088` and add your inference servers in **Connections**.
2. Refresh **Models** to see the loaded catalog.
3. In **Export OpenCode settings**, choose a client version, an output budget
   for each selected model and a default model. The relay fills in the loaded
   context when the remote server reports it. OpenCode connects to the relay at
   `http://127.0.0.1:8088/v1`.

For release-wheel installation, authentication and HTTP examples, see the
[relay guide](docs/RELAY.md).

## Learn more

| What you want to do | Where to go |
| --- | --- |
| Discover and route to loaded models | [Automatic routing and Strata](docs/AUTOMATIC-RUNTIMES.md) · [Model discovery API](docs/MODEL-DISCOVERY.md) |
| Stream chat, recover tool calls and stop requests | [Tool-call recovery](docs/TOOL-RECOVERY.md) · [Stream timeouts and cancellation](docs/STREAM-TIMEOUTS.md) |
| Serve a local API and configure OpenCode | [Local relay](docs/RELAY.md) |
| Prepare inputs and distribute embedding work | [Embedding contracts](docs/EMBEDDING-CONTRACTS.md) · [Automatic embeddings](docs/AUTOMATIC-EMBEDDING.md) · [Replica discovery](docs/AUTOMATIC-EMBEDDING-REPLICAS.md) |
| Stream embedding records or complete windows | [Progressive embedding](docs/PROGRESSIVE-EMBEDDING.md) · [Async windows](docs/ASYNC-EMBEDDING-WINDOWS.md) |
| Inspect concurrency and HTTP behavior | [Diagnostics](docs/CONCURRENCY-AND-DIAGNOSTICS.md) · [Native HTTP](docs/NATIVE-HTTP.md) |
| Connect via Tailscale or Headscale | [Userspace tailnet helper](docs/TAILNET.md) |
| Understand integration boundaries | [Architecture](docs/ARCHITECTURE.md) · [Functional coverage](docs/FUNCTIONAL-PARITY.md) |
| See changes and package checks | [Releases](https://github.com/Damel91/llama-runtime/releases) · [Verification records](docs/verification/) |

## Development with Flower MCP

This is a real project managed with [Flower MCP](https://github.com/Damel91/flower-mcp).
The `llama-runtime` project ledger holds requirements, milestones, implementation
plans, verification evidence and handovers. Repository documentation focuses
on installation and public contracts. Flower is used for development. Running
the SDK or relay requires its own Python dependencies.

## Verification

From the checkout, install the development and relay dependencies and run the
local checks.

```sh
python -m pip install '.[dev,relay]'
python -m pytest -q
ruff check src tests probes _build_backend.py
ruff format --check src tests probes _build_backend.py
python -m build --no-isolation
```

Install the built wheel in a clean environment outside the checkout to check
imports and packaging. [Live probes](probes/README.txt) require explicit opt-in
and an operator-supplied endpoint and model. Local test results, qualification
of a server profile and acceptance by a consuming application are separate.

## License and compatibility

The Python distribution is `runtime-llama`, imported as `runtime_llama`. The
optional relay command is `llama-runtime-relay`.

Copyright 2026 Davide Mele. [Apache-2.0](LICENSE), with attribution in
[NOTICE](NOTICE) and [provenance details](docs/PROVENANCE.md). The adapted
llamatelemetry client retains its [MIT license](src/runtime_llama/_vendor/llamatelemetry.LICENSE).
The separate native helper archives include Tailscale's BSD-3-Clause license
and dependency notices.
