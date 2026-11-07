# Local inference relay

The relay gives HTTP clients access to the existing `runtime_llama` SDK. It
discovers models on configured servers, selects a runtime from the requested
model, and applies the same chat validation, tool recovery and embedding
scheduler used by Python callers. The server operator controls loaded models,
runtime builds and hardware.

## Install and connect

The base distribution keeps the existing `runtime_llama` imports. To install
the optional relay, use the extra with the 0.18.0 wheel, or install from a
checkout that contains the relay.

```bash
python -m pip install 'runtime_llama-0.18.0-py3-none-any.whl[relay]'
# From a source checkout:
python -m pip install '.[relay]'
```

Choose a configuration path in a private directory. With a new path the relay
starts with an empty connection list and saves the configuration when you
apply changes in the console.

```bash
llama-runtime-relay --config ./relay.json
# Equivalent module entry point:
python -m runtime_llama.relay --config ./relay.json
```

The process runs headlessly. Open `http://127.0.0.1:8088` yourself. Point
inference clients at `http://127.0.0.1:8088/v1`.
Omitting `--config` uses a per-user configuration path. Explicit `--host` and
`--port` apply to the current process and stay outside the saved configuration.
The console shows when the configured listener differs from the running address.

1. In **Connections**, add each runtime origin without `/v1`. For example,
   use `http://localhost:8080` for llama.cpp and `http://localhost:8082` for
   Strata. Leave runtime selection automatic to detect the provider. Endpoint
   authentication, TLS verification and optional private network routing are
   configured per connection.
2. Save changes. In **Models**, refresh the catalog and pick an exact model ID.
   Discovery reads server metadata. Unknown capabilities and context are
   displayed as unknown.
3. In **Settings**, adjust generation and embedding defaults, per-model
   overrides, listener and inference access. Empty optional settings inherit
   the SDK default. Explicit zero and the separate **No idle limit** choice
   retain their meanings. A successful save applies to new requests. Active
   requests keep their configuration, and listener changes require a restart.
4. In **Requests**, inspect active requests or the last completed summary.
   **Stop request** cancels the selected relay invocation. The summary
   separates accepted usage from observed attempts and records missing
   counters as unknown.

Use **Theme** in the header to switch between **Light** and **Dark**. The
browser remembers your choice for this relay.

The console uses its own local administration session. A relay inference key
protects `/v1` and is separate from upstream server keys. Prefer environment
references where available. Literal secrets belong in the private configuration
and are redacted from configuration readback and exports. For private network
connections, supply the [separate tsnet helper](TAILNET.md), save its settings
and start it explicitly in the console.

The local administration session protects against cross-site browser requests.
It is not a remote user account: a client that reaches the console with an
allowed host can obtain a session. Keep the administration listener on a
trusted network when overriding the loopback default.

## HTTP contract

| Route | Behavior |
| --- | --- |
| `GET /v1/models` | Lists observed model IDs. The console shows availability and readiness separately. |
| `POST /v1/chat/completions` | Accepts structured messages and one choice. Supports JSON and SSE with tool calls and reasoning where the runtime provides them. |
| `POST /v1/embeddings` | Accepts a string or a list of strings and returns indexed vectors in input order. Supports float or base64 output. |

For example, replace the model IDs with values from your catalog. Add a Bearer
header if you configured an inference key.

```bash
curl http://127.0.0.1:8088/v1/models
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"chat-model","messages":[{"role":"user","content":"Explain this change."}],"stream":false}'
curl http://127.0.0.1:8088/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embedding-model","input":["first text","second text"],"input_type":"document"}'
```

For chat, set `stream: true` to receive SSE. Ordinary text streams
progressively. With declared tools the SDK stages each proposal until it can
validate it, then emits the accepted content and tool calls. A safely
recoverable rejection may trigger up to three corrections after the initial
generation on the same route. The client executes tools, retains tool call
IDs and submits the results in subsequent messages. See
[tool-call recovery](TOOL-RECOVERY.md).

Set `stream_options.include_usage` to request stream usage. Available counters
describe the accepted generation. Missing counters remain unknown. Client
elapsed time includes transport and coordination, and native timings are
reported separately when available.

Errors use a structured `error` object with a stable code. Invalid or unsupported
parameters receive a request error. Unavailable models, ambiguous selection,
timeouts and recovery exhaustion are separate failure cases. An embedding
split error identifies the rejected inputs and any completed indexed vectors.
For SSE, an error can arrive after the HTTP status has been sent. Stop or
a downstream disconnect can terminate the transport. A successful finish
or `[DONE]` is emitted only on successful completion. Client
transport closure alone cannot verify when native inference stopped.

Embeddings use llama.cpp. `input_type` selects `document`, `query` or `raw`.
When omitted, the configured task applies. Model-specific prefixes can be set
in the console and are applied once. Raw input bypasses task prefixes. The
relay sends direct texts through the existing scheduler without invoking
calibration. The optional total batch target is distributed across usable
replicas and the server can require more HTTP requests. The target does not
cap the submitted text count, and the relay adds no implicit text, model or
memory quotas. Optional capacity policies remain explicit.

## OpenCode export

In **Models**, refresh the catalog and select the chat models to export. If
none are checked, the dialog offers all eligible loaded chat models.
Open **Export OpenCode settings**, choose the OpenCode major version and
default model, and enter the maximum output tokens per generation for each
model. Model loading settings stay on the remote server. The dialog uses
the observed loaded context where available and asks for the effective
context for one conversation when the runtime omits it. It checks that the
output is positive and below that context.

**Generate settings** checks the models and observed context against the
current remote catalog before creating the JSON. If a model or context
has changed, refresh the catalog and reopen the export. Copy or download
the result and merge the generated provider and `model` into your OpenCode
configuration. The console writes no existing configuration files.

| OpenCode | Provider shape |
| --- | --- |
| V1 | `provider.llama-relay` with `npm: "@ai-sdk/openai-compatible"` and `options.baseURL` / `options.apiKey`. |
| V2 | `providers.llama-relay` with `package: "@opencode/ai/providers/openai-compatible"` and `settings.baseURL` / `settings.apiKey`. |

Both exports use the [OpenCode configuration schema](https://opencode.ai/config.json)
and the relay's `/v1` URL. When relay inference authentication is enabled, the
export carries an environment reference such as `{env:LLAMA_RELAY_API_KEY}`.
Set that variable in OpenCode's environment. An unauthenticated local relay
exports the placeholder `relay-local`. Each exported model includes
`limit.context` and `limit.output`. Review OpenCode's defaults for
capabilities the runtime did not report. OpenCode's compaction settings
control when it summarizes the conversation independently of the output
budget.

Client retries and timeouts are independent of the relay's SDK recovery.
Every client retry starts a new invocation with a fresh correction budget.
An outer timeout can expire during prefill or staged tool validation. The
relay can allow unlimited stream idle time while keeping its connection,
JSON, discovery and embedding deadlines finite.

See the [0.18.0 verification record](verification/PUBLIC-RELEASE-0.18.0.md)
for the OpenCode parser and native qualification boundaries.
