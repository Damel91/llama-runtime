# Optional server-observation SDK

This development surface is optional. The ordinary embedding adapter continues
to work with native unpatched llama-server and fresh catalog/props checks.
No new endpoint, server patch, Noxi configuration, consumer migration or VPN
installation is implied. A real server implementing this extension has not
been qualified by this document; see the separate
[candidate verification record](verification/REPLICA-OWNERSHIP-CANDIDATE.md).

The SDK lives in `runtime_llama.server_observation`. It supplies typed contracts
and a bounded dispatch session for a consumer-supplied provider. It does not
install a provider automatically in `LlamaCppEmbeddingAdapter` or add another
embedding scheduler. A host using it must keep dispatch inside its existing
shared replica lease, resource admission and result-ownership lifecycle.

## Negotiated capability

`ServerObservationCapability(version=1, coherent_snapshot=True,
conditional_admission=False)` is an explicit negotiation result. It must not
be inferred from a weight hash, native slots, an ordinary props response or a
client-local generation counter.

- A complete coherent snapshot can replace separate catalog and props reads
  for that boundary. Without conditional admission, the SDK fetches it again
  for every dispatch; one GET does not make the following POST atomic.
- With `conditional_admission=True`, the **server** must reject stale expected
  state before model selection/inference, including autoload and queued work.
  Only this stronger contract permits snapshot reuse until its finite expiry.
- With `provider.capability is None`, the SDK chooses the supplied fresh native
  callback before dispatch. A provider error, stale rejection or ambiguous POST
  is propagated; it never causes fallback/replay of a possibly executed POST.

Version 1 covers the complete selected logical pool, including loaded status,
canonical replica IDs, weight and semantic digests, context, native batch,
microbatch and slots, build/tokenizer IDs and load/config revisions. Server
restarts use a new instance ID. A same-weight unload/reload cannot reuse the
previous loading generation within that server instance. Scope includes opaque
endpoint and authority fingerprints so observations cannot cross endpoints or
credential scopes. This requires real producer enforcement, not only SDK types.

## Public types and calls

| Type | Purpose |
| --- | --- |
| `ServerObservationScope` | Opaque `endpoint_id`, `authority_id`, content and semantic SHA-256 digests. |
| `ObservedServerReplica` | Canonical replica/alias, loaded state, load/config generations, build/tokenizer and effective limits. |
| `CoherentServerSnapshot` | Typed scope, server instance, observation revision, immutable replica tuple and finite validity. |
| `ServerAdmissionExpectation` | Complete server-enforced condition passed to a POST when negotiated. |
| `StaleServerObservationError` | Typed pre-inference stale rejection; no automatic retry. |
| `ServerObservationProvider` | Provider protocol for negotiation, snapshot and embedding transport. |
| `ServerObservationSession` | Finite-deadline dispatch, local revocation and drained provider-callback closure. |

```python
ServerObservationSession(scope, provider, *, max_response_bytes=8 * 1024 * 1024)
session.dispatch(inputs: tuple[str, ...], alias: str, *,
                 native, timeout: float, cancel=None)
session.revoke()
session.close()
```

The provider implements `snapshot(scope, *, timeout, cancel, cancel_wait,
max_response_bytes)` and `embeddings(inputs, alias, *, expected, timeout,
cancel, cancel_wait, max_response_bytes)`. The native callback accepts the same
finite transport options. Provider transport must actually enforce response
bounds and cancellation/drain; accepting a keyword alone does not prove that.
All callbacks consume one original monotonic deadline, not a reset per phase.

The SDK retains at most one immutable snapshot per session. Snapshot tuples
are bounded; identifiers are bounded opaque values and do not contain raw
credentials or private endpoint URLs. Scope mismatch, expiry or local revocation
invalidates reuse. A late response cannot revive a locally revoked generation.
`revoke()` is a local safeguard, not evidence of a remote loading event.

`close()` signals cancellation, revokes reuse and waits for real synchronous
callback drain. Closing from an active callback fails with a typed error instead
of waiting on itself. Arbitrary providers that ignore cancellation cannot be
safely killed by this helper; it does not promise a hard shutdown deadline or
remote GPU termination. Native operation remains the baseline when capability
has not been negotiated.
