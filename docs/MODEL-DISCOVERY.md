# Read-only model discovery

Available in `runtime-llama` 0.5.0. The Python import remains `runtime_llama`.
This API observes an explicitly configured llama-server endpoint, including a
remote host or VPN address. It neither starts the server nor selects, loads,
downloads or runs a model.

```python
from runtime_llama import discover_models

catalog = discover_models(
    "https://inference.example.net",
    timeout=5.0,
    # api_key_env="LLAMA_API_KEY",  # optional explicit authentication
)
for model in catalog.loaded_models:
    print(model.id, model.aliases, model.identity.model_name)
    print(model.capabilities.chat, model.capabilities.embeddings)
    if model.loaded_context is not None:
        print(model.loaded_context.value, model.loaded_context.source)
```

Supply the complete server origin, including a non-default port where needed,
without appending `/v1`. A reverse-proxy path prefix can be part of `base_url`.
The example endpoint and aliases are illustrative.

## Public signature and observations

```python
def discover_models(
    base_url: str,
    *,
    api_key: str | None = None,
    api_key_env: str | None = None,
    timeout: float = 5.0,
    verify_ssl: bool = True,
    cancellation_probe: CancellationProbe | None = None,
    max_response_bytes: int = 1_048_576,
    max_models: int = 256,
) -> ModelCatalog:
    ...
```

`ModelCatalog`, `ModelObservation`, `ModelCapabilities`, `ModelIdentity` and
`ObservedInt` are exported frozen dataclasses. Nested sequences are tuples.
`ModelState` and `ObservationSource` are exported literal type aliases.

| Field | Meaning |
| --- | --- |
| `catalog.models` | All entries in the observed native catalog, in server order. |
| `catalog.loaded_models` | Entries whose state was proved `loaded`; other states are excluded. |
| `catalog.diagnostics` | Bounded fixed codes; no response dumps, endpoint or credentials. |
| `catalog.http_requests` | Number of HTTP attempts in this successful observation, including an unsupported root properties request. |
| `catalog.elapsed_seconds` | Client elapsed time including HTTP, parsing and cleanup; no inference or GPU timing. |
| `model.id`, `model.aliases` | Observed routing ID and a sorted, deduplicated tuple including that ID. Ambiguous IDs or aliases across entries fail explicitly. |
| `model.state`, `model.failed` | Native load state and optional native failure flag. Unknown future states become `unknown`; failure is a flag, not an invented native state. |
| `model.identity` | Independently observed content digest, embedding-semantic digest and decoded model name from supported `noxi-*` tags. Missing values remain `None`. |
| `model.capabilities` | `chat` and `embeddings`: explicit catalog booleans, or `None` when unobserved. |
| `model.training_context` | `meta.n_ctx_train`, falling back to native `context_length`, with provenance. It is not the allocated context. |
| `model.loaded_context` | Observed loaded `meta.n_ctx`, or matched direct-server properties context. Unknown or sleeping models have no loaded context observation. |
| `model.model_embedding_width` | Architecture width `meta.n_embd`, with provenance. It is neither proof of embedding support nor an output-vector dimension contract. |
| `model.slots` | Slot count from matched direct-server root properties, if observed. |

Each observed integer has `value` and `source`. Missing facts remain `None`.
Positive integer observations are bounded to 2³¹−1. Names and aliases are bounded
to 1,024 characters, aliases to 64 per entry and tags to 128 per entry. Malformed
supported fields, conflicting identity assertions and ambiguous aliases raise
`LLMBackendError`; unsupported optional metadata/capability shapes produce fixed
diagnostics. Raw model paths, arguments, presets and arbitrary properties are
not projected.

Identity tags are server attestations. This API does not hash remote files,
certify equivalent weights or replace the stricter identity and semantic
checks of automatic embedding replica discovery.

## Catalog, loaded state and capabilities

The native router catalog can include unloaded, loading, sleeping, downloading
and downloaded entries. Its presence alone does not prove readiness.
`loaded_models` includes only explicit `status="loaded"` or
`status.value="loaded"`, plus the narrowly verified direct-server case below.
The snapshot does not reserve a model: state can change immediately afterward.

Only per-entry `capabilities.chat` and `capabilities.embeddings` booleans are
accepted. Names, aliases, templates, `n_embd`, text modalities and compatibility
`completion` declarations are not used to guess those capabilities. A native
server can expose a loaded model while leaving both capabilities `None`.
Consumers must preserve that unknown state or ask the operator to assign a role.

The API makes one `GET /v1/models`. If any state is unknown, it may also make one
**root** `GET /props`, without a selected model or query parameters. That second
response can prove a missing native load state only when the entire catalog has
one entry, `role` is absent (native direct server) or explicitly `model`,
`model_alias` matches an observed alias and `is_sleeping` is an explicit boolean.
Explicit future load states remain unknown even if root properties are present.
Router properties, mismatched aliases and missing proof leave the state unknown.
HTTP 404, 405 and 501 for root properties leave a fixed
diagnostic; other HTTP errors propagate.

It never requests per-model properties, health, tokenization, inference,
reload, download or administration. In particular, per-model router properties
can trigger auto-loading, so they are excluded from this API. Existing raw
client helpers and inference adapters retain their separate behavior.

These distinctions follow llama.cpp's official
[model information API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#get-v1models-openai-compatible-model-info-api),
[router state definitions](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-models.h)
and [router request handling](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-models.cpp).

## Authentication, bounds and failures

Use either `api_key` or an explicitly named `api_key_env`, or neither for an
anonymous endpoint. An unset or empty requested environment value fails before
dispatch. The owned discovery client uses `trust_env=False`: ambient `.netrc`
authentication, proxy settings and requests certificate environment overrides
are not applied. TLS verification is enabled by default; redirects and retries
are disabled. Endpoints containing credentials, controls, whitespace, invalid
ports, query strings or fragments are rejected without echoing their contents.

`max_response_bytes` bounds each decompressed JSON response before buffer growth;
its supported range is 1 byte to 16 MiB. `max_models` ranges from 1 to 4,096.
Defaults are 1 MiB and 256 entries. There is no silent truncation. The lower-level
`LlamaCppClient.request()`, `models()` and `props()` also accept an optional
`max_response_bytes`; their default remains `None` for compatibility. The raw
client's new keyword-only `trust_env` defaults to `True`, and borrowed sessions
retain their owner's configuration and decoding responsibility.

One finite deadline covers the discovery requests and projection. Cooperative
cancellation is checked between stages and while waiting for responses. Cleanup
drains the owned client on success or failure. DNS resolution and drained close
retain the transport's existing timing limitations, so `timeout` is not a
strict process-wide wall-clock termination guarantee.

Invalid arguments raise sanitized `ValueError`. Transport and response failures
use `LLMBackendError` (including `ClientHTTPError` with a status code),
`LLMTimeoutError` or `LLMCancelledError`. Do not treat a failed discovery as an
empty catalog. No live qualification or consumer acceptance is implied by a
successful deterministic test or catalog observation.
