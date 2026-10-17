# Generation stream timeouts and cancellation

The nullable idle contract is available in `0.5.1`, following the separately
recorded [`0.5.1.dev1` local qualification](verification/LOCAL-IDLE-0.5.1.dev1.md).
Set `stream_idle_timeout_sec=None` (`null` in JSON) to wait through silent
generation before headers, before the first token and between events. This
disables SSE idle explicitly, without a large numeric workaround. The default
remains 30 seconds; output budgets and repetition protection keep their settings.

| Setting | Meaning |
| --- | --- |
| Generation idle omitted | Existing 30-second SSE idle budget. |
| `stream_idle_timeout_sec=None` | No generation idle deadline; the owned factory client keeps a finite 30-second HTTP/connect budget. |
| Positive generation idle | Existing finite idle and HTTP behavior. |
| Per-call `timeout=None` | Inherit configured generation idle. |
| Positive per-call `timeout` | Restore or override a finite idle budget for that call. |
| `AdapterProfile.timeout` | Finite HTTP/non-SSE budget, with 30-second fallback when idle is disabled. |
| Raw client `timeout` | Finite connection/JSON budget; request-level `timeout=None` inherits it. |
| Raw SSE `idle_timeout` omitted | Inherit the effective finite request timeout. |
| Raw SSE `idle_timeout=None` | Disable SSE read/queue idle, preserving finite connection timeout. |
| Raw SSE positive `idle_timeout` | Separate finite stream idle from connection timeout. |

Adapters send disabled idle as raw `timeout=None, idle_timeout=None`: the first
inherits the client's finite budget; the second disables only SSE idle. An
injected client's own finite timeout is preserved. JSON, discovery, embeddings
and preparation remain finite. Embedding profiles reject nullable timeouts.
Explicit idle control on non-stream JSON helpers fails before dispatch. Zero,
negatives, booleans, strings, NaN, infinities and overflowing idle values raise
`ValueError`.

## Public bridge-owned client

The bridge chooses authentication/environment policy while the raw client owns
its interruptible sessions. Endpoint and model below are illustrative.

```python
from threading import Event
from runtime_llama import AdapterProfile, LlamaCppClient, build_adapter

profile = AdapterProfile.from_config("chat", {
    "model": "chat-model",
    "base_url": "https://inference.example.net",
    "stream_idle_timeout_sec": None,
})
cancel = Event()
with LlamaCppClient(
    profile.base_url,
    timeout=profile.timeout,
    verify_ssl=profile.model_config.verify_ssl,
    trust_env=False,
    # api_key=token,  # optional, explicitly resolved by the bridge
) as client:
    adapter = build_adapter(profile.to_llm_config(), client=client)
    try:
        for event in adapter.stream_with_metadata(
            "Explain this algorithm.", cancellation_probe=cancel.is_set
        ):
            if event.kind == "content":
                print(event.text, end="")
    finally:
        adapter.close()
```

Set `cancel` from Stop. An injected raw client belongs to the bridge;
`adapter.close()` preserves it, and the bridge closes it in its own Stop/finally
path. Factory-created clients belong to and are closed by the adapter.
`trust_env=False` is public: netrc/proxy/certificate environment settings are
ignored. Optional keys are headers and TLS verification defaults to enabled.
No private Session access is needed. A borrowed Requests Session is different:
unbounded SSE is rejected before dispatch because its owner controls the
transport. Existing finite borrowed-session calls remain supported.

## Cleanup, evidence and compatibility

Unbounded SSE uses the existing exclusively leased sessions with a call-owned
connection hook. Cancellation shuts down that socket before response closure
and drains cleanup. The hook detaches before pool reuse, so an obsolete stream
cannot close a later request. HTTP/TLS, proxy dispatch, keep-alive and concurrent
calls preserve ownership. There is no retry or inference replay.

Cancellation is cooperative: socket interruption cannot force system DNS to
terminate, and connect/TLS retains its finite connection budget. The API does
not promise process-wide forced termination or certify that remote GPU work
stopped when the local socket closed.

`LLMTransportAttempt.idle_timeout_sec` is `None`/JSON `null` when disabled;
the existing default zero remains unknown/unobserved. Normal cancellation polls
do not allocate diagnostics per event. Existing typed backend, timeout,
cancellation and runaway errors remain. Disabled-idle SSE HTTP errors return
sanitized `ClientHTTPError.status_code` immediately, without waiting for optional
error-body codes; other paths retain structured error behavior.

Actual Requests/urllib3 responses enforce Content-Length while reading,
including the older branch: valid JSON followed by premature EOF is a typed
transport failure. The SSE terminal marker still intentionally ends reading;
bytes after it are not a new validation obligation.

Runtime dependencies are `requests>=2.31,<3` and `urllib3>=1.26.20,<3`. The explicit
urllib3 floor covers scoped pool/connection hooks. The implementation uses
Requests [adapter extension points](https://requests.readthedocs.io/en/latest/api/#requests.adapters.HTTPAdapter.init_poolmanager)
and urllib3 [connection classes](https://urllib3.readthedocs.io/en/1.26.20/reference/urllib3.connectionpool.html),
with narrow request hooks checked by HTTP/TLS fixtures. No global pool mapping
or caller-owned Session is modified. Qualification applies to the dependency
combinations tested, rather than every possible environment.
