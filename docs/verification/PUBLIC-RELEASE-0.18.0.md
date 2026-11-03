# Public release 0.18.0 verification

This release adds the optional local relay and its web console. The
implementation and package metadata are pinned at
`edcdb796a63c1037562953dc7f2aabc2de3d14b3`. The release manifest identifies the
final source tree, tests, clean-install environments and asset SHA-256 values.

Flower MCP tracks this work under milestones `MILE-000043`–`MILE-000045`
and adopted architecture revision 8. Packet `PACKET-000047` closes
`FIND-000008` on retaining ownership after a failed close. Packet `PACKET-000048`
addresses `FIND-000009` on browser input persistence. The historical
embedding-profile deviation `FIND-000007` retains its existing evidence
and thresholds.

## Candidate checks

| Surface | Evidence |
| --- | --- |
| Python suite | 2,737 passed in 128.96 seconds with six fixture or dependency warnings. Ruff lint and formatting passed across 165 files. |
| TCP embedding and cleanup | Six targeted checks exercised 10,000 direct texts and resource release across embedding and request lifecycles. |
| OpenCode parser | `@opencode/ai` 2.2.5 passed seven cases over TCP through the relay and SDK to synthetic upstream HTTP. The checks sent 11 upstream POSTs and executed zero tools. |
| Served browser console | Desktop and mobile review covered connection setup, catalog, OpenCode export and saving exact zero, null, a batch target of 10,000 and literal prefix whitespace. The event regression also covered per-model edits, secret handling and focus preservation. |
| Stop in the served console | Stopping a quiet fixture stream closed its upstream peer and returned the active request and bridge counts to zero. The client observed transport termination after HTTP 200. |
| Headless CLI | Eight checks covered optional imports, external configuration and real SIGINT/SIGTERM shutdown during a quiet fixture stream. |

The OpenCode cases covered text and reasoning, tool-only and multi-tool
responses, tool-result round trips, recovery, exhaustion and a fresh
invocation after exhaustion. The browser export uses the OpenCode V2
provider package and schema documented in the [relay guide](../RELAY.md).
The parser check used the published OpenCode AI package and synthetic
upstream responses.

## Native qualification boundary

A separate existing OpenCode GUI connection reached Strata and answered a
short chat. The relay preflight reached native Strata health with HTTP 200,
while its properties and model discovery returned HTTP 401. The authorized
backend credential required for those calls was unavailable. The remote
frontend revision was unidentified. Native relay inference on that endpoint
and spontaneous Flash Next rejection-marker emission therefore remain
unqualified.

Candidate tests exercise HTTP projection, ownership and cleanup. Their
elapsed times include client and transport work. The fixture results do
not supply GPU performance or numerical embedding evidence, or downstream
application acceptance.

## Distribution record

The wheel is `runtime_llama-0.18.0-py3-none-any.whl`. The release manifest and
`SHA256SUMS` bind the reproducible wheel and source archive, and the optional
tsnet archives reuse the verified binaries and complete notices. Clean
installation checks cover base import without web dependencies and the relay
extra with the packaged assets, HTTP routes and shutdown. Tests use separate
environments and leave consumer installations unchanged.
