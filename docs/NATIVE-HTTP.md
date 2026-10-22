# Native llama.cpp HTTP transport

The client works with unpatched llama-server. Its default connection policy is
`close`: each request uses an independent HTTP connection and sends
`Connection: close`. Concurrent requests still run concurrently. No keep-alive
patch, automatic transport retry or runtime administration is required.

```python
from runtime_llama import LlamaCppClient, build_embedding_adapter, discover_models

client = LlamaCppClient("http://localhost:8080", connection_policy="close")
# Adapter profiles and read-only discovery use the endpoint-level spelling:
embedding = build_embedding_adapter({
    "model": "embedding-model",
    "base_url": "http://localhost:8080",
    "http_connection_policy": "close",
    "batch_size": 256,
    "embedding_request_parallelism": 2,
})
catalog = discover_models("http://localhost:8080", http_connection_policy="close")
try:
    vectors = embedding.embed_texts(["first document", "second document"])
finally:
    embedding.close()
    client.close()
```

Only `close` and `reuse` are accepted. The optional `reuse` policy explicitly
requests keep-alive and permits sequential reuse on a drained session. Use it
when the endpoint's framing and connection reuse are qualified. An injected
client retains its own policy; adapters do not rewrite that client or mutate a
borrowed session's headers or clear its pools. On borrowed sessions only the
request header expresses the close policy; their owner controls pooling. With
owned sessions, close policy clears transport pools after body cleanup, while
reusing the Session object, so a peer that omits the response header cannot
cause a subsequent socket reuse. The policy is transport configuration, never an
inference parameter. Authentication, TLS verification and explicit proxy routing
remain in effect.

For an already qualified persistent endpoint, configure
`"http_connection_policy": "reuse"` in the adapter profile, or
`connection_policy="reuse"` on a directly owned client. The opt-in queue probe
accepts `--connection-policy close` or `--connection-policy reuse` to qualify
the same corpus and count requests separately. It never retries a failed POST
with the other policy. The default remains `close`.

Local transport tests cover both policies: sequential and concurrent socket
ownership, peer-close behavior, incomplete framing, cancellation, slow bodies
and TLS. Those fixtures do not qualify a real HTTPS proxy or a Tailscale/
Headscale deployment; each such route needs its own same-endpoint validation.

Fresh connections have a TCP/TLS establishment cost. This policy avoids relying
on broken persistence; it does not promise greater throughput than a correctly
working persistent endpoint. Keep-alive is a native llama.cpp facility. The
historical patch addressed an EOF callback in the HTTP handler, not a deliberate
absence of keep-alive. The source investigation is retained in
[architecture](ARCHITECTURE.md).

Owned concurrent requests lease separate sessions and sockets. One streaming
response owns its connection until completion or drained cancellation. An HTTP
array of embedding inputs reduces round trips; llama.cpp still queues individual
inputs and executes them according to its loaded slots and backend. A one-slot
replica does not execute all the array's inputs simultaneously.

JSON and SSE retain finite connection budgets and drained Stop/close. The JSON
legacy reader uses bounded blocks only when the owned socket can be interrupted;
borrowed transports keep the conservative fallback. SSE framing and explicit
`stream_idle_timeout_sec=None` behavior are preserved. Local interruption cannot
guarantee that a remote backend has stopped computing after a disconnect.

## Optional native generation timings

`LLMCallResult.server_timings` and `LLMStreamEvent.server_timings` expose a frozen
`LLMServerTimings` snapshot when the server supplies native `timings` metadata.
`adapter.get_last_server_timings()` returns the latest observation.

Fields are `prompt_n`, `prompt_ms`, `prompt_per_second`, `predicted_n`,
`predicted_ms`, `predicted_per_second`, `cache_n` and `derived_fields`.
`from_native(value)` validates a native mapping; `to_dict()` returns a public
projection. Missing or invalid values stay `None`. A missing rate may be derived
from a valid token count and positive duration in the same native snapshot;
`derived_fields` identifies it. Separate snapshots are never fused to invent a
rate. HTTP elapsed time and preparation time are not GPU timing. Existing usage
counters and terminal-event requirements remain separate.

Streaming retains reasoning diagnostics without a second full reasoning copy.
Tool argument fragments are assembled once, while public raw payloads and tool
snapshots remain isolated from consumer mutation. Repetition detection keeps
its existing decisions with bounded suffix storage; these optimizations do not
change the public token, cancellation or error contracts.
