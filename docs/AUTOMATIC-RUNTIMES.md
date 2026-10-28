# Automatic runtime discovery and Strata

Version 0.16.0 adds a Strata LLM backend and model-based automatic routing.
The deployment declares reachable endpoints. The inference caller supplies a
model and ordinary generation options; the library chooses the supported
runtime that currently reports that model loaded. The distribution remains
`runtime-llama`, imported as `runtime_llama`, with Python 3.11 or later.

## Deployment configuration

```python
from runtime_llama import build_adapter

llm = build_adapter({
    "role_id": "coder",
    "model": "chat-model",
    "runtime_endpoints": [
        {"name": "local-native", "base_url": "http://localhost:8080"},
        {"name": "local-other", "base_url": "http://localhost:8082"},
    ],
    "runtime_discovery_timeout": 5.0,
    "temperature": 0.2,
    "stream_idle_timeout_sec": None,
})
try:
    catalog = llm.discover_models()
    for runtime in catalog.runtimes:
        print(runtime.endpoint.name, runtime.backend, runtime.available)
        print([model.id for model in runtime.loaded_models])
        print(runtime.diagnostics)

    result = llm.generate_with_metadata("Explain this change.")
    print(result.text)
    print(result.runtime_selection.backend)
    print(result.runtime_selection.runtime.endpoint.name)
    print(result.runtime_selection.model.id)
finally:
    llm.close()
```

The endpoint list implies automatic routing; it contains no required backend
choice. A single endpoint can use `backend="auto"` and `base_url` instead.
Construction is HTTP-free. Different processes sharing one IP need different
listening ports, or an operator-managed reverse proxy with distinct routes.
Discovery does not guess ports or scan a LAN.

Each endpoint accepts `name`, `base_url`, optional `api_key` **or**
`api_key_env`, `verify_ssl`, `proxy_url` and `http_connection_policy`.
TLS verification defaults to true. Authentication and proxy configuration
remain attached to that endpoint, including after the selected runtime changes.
Top-level generation credentials are not copied to a configured endpoint list.
Automatic clients ignore ambient proxy and netrc authentication settings.
Never embed credentials in a URL or commit deployment configuration.

## Public interfaces

```text
RuntimeEndpoint(name, base_url, backend="auto", api_key=None, api_key_env=None,
                verify_ssl=True, proxy_url=None, http_connection_policy="close")
RuntimeEndpoint.from_value(endpoint_or_mapping) -> RuntimeEndpoint

discover_runtimes(endpoints, *, timeout=5.0, cancellation_probe=None)
    -> RuntimeCatalog
RuntimeCatalog.runtimes -> tuple[RuntimeObservation, ...]
RuntimeCatalog.http_requests -> int
RuntimeCatalog.select(model, *, purpose="llm") -> RuntimeModelSelection
RuntimeObservation.loaded_models -> tuple[ModelObservation, ...]

AutoRuntimeAdapter(profile)
adapter.discover_models(*, timeout=None, cancellation_probe=None) -> RuntimeCatalog
adapter.current_catalog -> RuntimeCatalog | None
adapter.selected_runtime -> RuntimeModelSelection | None
adapter.close() -> None
```

`build_adapter()` constructs the automatic adapter for a profile using automatic
routing. An injected borrowed client is rejected for that mode because each
endpoint needs its own access configuration and ownership. Existing direct
llama.cpp construction and `backend="llamacpp"` remain available; the public
`StrataAdapter` or `backend="strata"` provides an explicit provider path.

Generation methods and their request arguments match `LlamaCppAdapter`:
`generate()`, `generate_with_metadata()`, `invoke_with_ctx()`, `stream()`,
`stream_with_metadata()`, tool definitions and correlated tool messages.
Blocking `LLMCallResult` and normalized `LLMStreamEvent` gain an optional
`runtime_selection`. Existing direct adapters leave that field `None`.
Per-event/per-result selection is authoritative for that call; snapshot
properties describe the latest observation and are not a shared active route.

## Probe and selection policy

Every configured endpoint is probed concurrently under one finite deadline.
There are at most three GET attempts per endpoint: `/health`, root `/props`
and `/v1/models`. The catalog is acquired once and root properties are reused.
No inference, tokenization, model-specific props request, load or administration
call is part of discovery. Unavailable and unsupported endpoints stay in the
returned catalog with sanitized diagnostic codes.

Supported automatic runtime identities are **llama.cpp** and **Strata**.
Health service and build markers must be coherent. An OpenAI-shaped catalog
alone does not establish either identity. Older deployments lacking identity
markers may deliberately configure the endpoint's backend; contradictory
observed markers still fail. This is an operator compatibility override.

Catalog membership alone is not proof of loading. Unknown capabilities and
states remain unknown; explicit unsupported chat capability is excluded from
LLM routing. Strata's health model, loaded catalog entry and root properties
must agree. Its explicit alias rows are collapsed only through their verified
`alias_of` relation. Other catalog entries can remain visible without becoming
eligible for that Strata LLM route.

An exact canonical ID takes precedence over an alias, including a known ID
that is currently unloaded. Alias matches referring to different canonical
models fail with `runtime_model_ambiguous`. The same canonical ID on several
available endpoints uses stable endpoint configuration order; conflicting
observed content/semantic hashes fail with `runtime_model_ambiguous_identity`.
Missing hashes remain missing and names never prove equal weights.

Each generation refreshes discovery and fixes one selected backend, canonical
model and client for the whole request. There is no global route mutation
that can redirect an in-flight stream, no fallback to a different model after
a lookup failure and no automatic replay of a submitted POST. Moving a model
between servers takes effect on the next request. An observation cannot reserve
the remote model or prevent an operator from changing it after the probe.

## Cancellation and resource ownership

Discovery keeps a finite timeout even with `stream_idle_timeout_sec=None`.
Connection and JSON waits remain finite. A cancellation probe stops only its
operation; close stops all adapter-owned probes and active HTTP clients.
Clients and probe workers drain before their library-owned work is released.
Closing an adapter also closes active sockets when a caller has paused a
stream iterator; callers should still close abandoned iterators themselves.

There are no new implicit text, endpoint, model or memory quotas. Library state
retains only current observation snapshots and active operations; completed
request clients are released. Results retained by a consumer naturally retain
their own immutable selection metadata. Discovery wall time and request counts
include HTTP, parsing, scheduling and cleanup, not pure GPU execution time.

## Strata structured output

The adapter shares the HTTP/SSE transport, message projection, tool assembly,
usage, timing, reasoning, cancellation and guards with llama.cpp. Strata's
schema wire format is provider-specific:

```python
from runtime_llama import build_adapter

llm = build_adapter({
    "model": "chat-model",
    "runtime_endpoints": [{"base_url": "http://localhost:8082"}],
    "output_mode": "json",
    "json_schema": {
        "type": "object",
        "properties": {"ok": {"const": True}},
        "required": ["ok"],
        "additionalProperties": False,
    },
})
try:
    result = llm.generate_with_metadata('Return {"ok": true}.')
    print(result.text)
finally:
    llm.close()
```

Strata receives `response_format.type="json_schema"` with nested
`json_schema.name`, `schema` and `strict`. The completed response is checked
locally with `jsonschema`; invalid JSON or mismatched schemas raise typed
backend errors before a successful terminal/result. JSON object mode also
checks that the completed value is an object. Schemas are application-owned
configuration; external reference retrieval is disabled and malformed or
unsupported dialects fail before HTTP.

Streamed text is provisional until completion. Tool responses, native length
caps, local visible-output caps, cancellation and abandoned partial streams
do not promise a complete schema-valid object. Local validation detects a
failure; it is not constrained decoding and does not silently repair or retry
model output. Native llama.cpp retains its existing schema-forwarding behavior.

Strata qualification is LLM-only. Embeddings continue using the existing
llama.cpp engine, replica identity checks, scheduling and 80%/90% calibration
margins. Ollama, OpenAI and LM Studio providers and a Rust port are outside this
release. A native Strata build, the loaded model/template, tool instruction
adherence and application acceptance are separate from client compatibility.
