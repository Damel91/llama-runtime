# Public release 0.6.0 verification

Date: 2026-10-18. Python qualification ran on macOS with Python 3.11.15,
Requests 2.33.1 and urllib3 2.6.3, with an additional Python 3.14.6 matrix.
The optional native helper uses Go 1.27.1 and Tailscale 1.104.1.

## Behavior and compatibility

Version 0.6.0 adds `TailnetConfig`, `TailnetHelper`, `TailnetError` and explicit
`proxy_url` routing. The configuration, binary, node state, enrollment and
endpoint belong to the consumer. Imports, configuration construction and
ordinary adapter factories do not start a VPN or contact a coordinator.
The distribution/import names remain `runtime-llama` / `runtime_llama`.
Generation, embedding, catalog, default timeouts and explicit nullable SSE
idle behavior retain the existing core contracts.

The separately distributed Go helper owns an authenticated loopback proxy and
uses a verified peer's numeric address through the pinned userspace netstack.
Unknown or ambiguous destinations fail without direct inference fallback.
HTTPS uses CONNECT with end-to-end client TLS verification; plain HTTP opens
fresh upstream connections to prevent automatic replay. Stop affects a
request; helper shutdown drains its clients, discovery, proxy and tunnels.
See the [consumer guide](../TAILNET.md) for configuration and ownership.

## Actual verification

| Check | Result |
| --- | --- |
| Stable full Python suite | **967 passed**, zero skips, 52.09 seconds |
| Python 3.14.6 proxy/supervisor/configuration/package matrix | **121 passed**, 14.99 seconds |
| Ruff check and format, including native notice collector | Passed; 62 Python files |
| Tagged Go deterministic tests | **20 top-level tests + 16 subtests passed**, 1.107 seconds |
| Tagged Go race detector | Passed, 2.217 seconds |
| Tagged Go vet and gofmt | Passed |
| Native Darwin ARM64 build | Passed; executable offline smoke passed |
| Linux AMD64 / Windows AMD64 crossbuilds | Passed; execution on those systems not performed |
| Clean installed Python wheel outside checkout | Version/import/source path, inert factories and public API passed |
| Installed supervisor with the real Go executable | Invalid state rejected before `Up`, typed failure, exit 2 and drained threads |
| Native offline protocol cases | Version, empty/invalid input, unexpected argument and aliased keys handled; sanitized output and empty stderr |
| Native attribution inventory | 42 dependency records, 69 notice files, all hashes verified |
| Native build privacy | No VCS metadata, actual user/cache paths or embedded web module/assets |
| Public source and examples review | No confirmed user credentials, personal email, deployment endpoints or lifecycle exports; new examples parse |

The Python tests include loopback HTTP/TLS fixtures, process/protocol limits,
explicit proxy routing for owned leases, adapter and discovery propagation,
no-idle cancellation, finite budgets, no replay, error/truncation handling,
credential redaction and drained concurrent shutdown. Existing embedding
scheduling, preparation, identity and topology tests remain in the full suite.
One embedding cancellation fixture now waits for its exact blocked input
before cancelling, preserving its borrower-isolation and drain assertions.

Go tests use a fake backend and real local HTTP/TLS connections. They cover
verified peer lookup and topology changes, missing shim failure, node-state
locking/identity, exact protocol keys, proxy authentication, stripping proxy
credentials, CONNECT buffering, SSE flush, truncated responses, startup safety,
finite dial and shutdown/EOF cancellation. They do not enroll a real node.

The installed-wheel smoke uses a fresh same-interpreter environment with only
the existing five HTTP dependency distributions copied offline. It blocks
unrelated inference SDK imports and verifies the imported package belongs to
that environment. Shared Python installations are unchanged.

## Distribution and provenance

The [release](https://github.com/Damel91/llama-runtime/releases/tag/v0.6.0)
provides:

- `runtime_llama-0.6.0-py3-none-any.whl`
- `runtime_llama-0.6.0.tar.gz`
- `runtime-llama-tsnet-0.6.0-darwin-arm64.tar.gz`
- `runtime-llama-tsnet-0.6.0-linux-amd64.tar.gz`
- `runtime-llama-tsnet-0.6.0-windows-amd64.zip`
- `RELEASE-MANIFEST.json` and `SHA256SUMS`

The wheel contains Python code and retained Python license notices; it does
not contain native executables, Go SDK, node state or enrollment credentials.
The source archive includes helper sources, pinned module metadata, public
documentation and native attribution inventory. Every native archive includes
its executable, project LICENSE/NOTICE, Tailscale license, the complete
`third_party/` bundle and source/build information.

Native builds use `-trimpath -buildvcs=false -tags ts_omit_webclient`. Exact
public commit and source-file hashes are supplied externally by the release
manifest, avoiding host metadata and a generated-inventory VCS cycle. The
inventory binds its notices to the actual delivered executable hashes.
Required upstream attribution emails in license/author files are retained;
they are not consumer deployment data.

PEP 517 packages and native archives use `SOURCE_DATE_EPOCH=315532800`.
Tar members have UID/GID zero, empty owner names, controlled timestamps and
regular permissions, without PAX metadata. Gzip omits its original filename;
ZIP timestamps and attributes are controlled. Runtime/source payloads are
compared with the checkout and native archive contents with their inventory.
Checksums live outside the archives they describe. Verify downloaded release
assets with `shasum -a 256 -c SHA256SUMS` or `sha256sum -c SHA256SUMS`.

## Qualification boundary

This release makes no claim of live Tailscale/Headscale enrollment, inference
through a VPN, remote GPU cancellation, Linux/Windows execution or new CI
coverage. The Unix-script Python fixture was executed on macOS; crossbuilding
a Windows helper does not qualify Windows process, state ACL or TLS behavior.
The consumer must qualify its deployment and chosen coordinator version.

Only exact current peer FQDNs or verified peer IPs are accepted. Short names,
LAN/subnet-router destinations and generic Internet forwarding are outside
this helper's contract. The narrow `Server.Sys()` shim is upstream-unstable
and fixed to the recorded pin; changing it requires review and qualification.
Plain HTTP's fresh upstream connections may affect latency; no VPN throughput
or inference-performance claim is made. No consumer job, database, service,
model configuration, OS VPN or shared Python environment was changed.
