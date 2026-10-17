# Optional userspace tailnet transport

Version 0.6.0 adds an optional Go helper based on `tailscale.com/tsnet` and a
Python supervisor. It connects the existing HTTP client, generation adapters,
embedding adapters and model discovery to an explicitly selected tailnet route.
The regular HTTP path and its defaults remain available without the helper.

The consuming application supplies the helper executable, node state directory,
hostname, enrollment credential when needed, coordinator and llama-server URL.
The library does not install a system VPN, require root or TUN, change system
routes or DNS, or manage inference servers and models.

## Install the matching helper

The Python wheel remains `py3-none-any`. It contains the supervisor, not a native
executable or Tailscale node state. Install the Python package normally, then
obtain the separate native archive for your operating system and architecture
from the [0.6.0 release](https://github.com/Damel91/llama-runtime/releases/tag/v0.6.0).
The release targets are Darwin arm64, Linux amd64 and Windows amd64. Use the
`.exe` executable on Windows.

Verify the downloaded archive against the release `SHA256SUMS`, extract it and
retain its accompanying licenses. Every native distribution must include the
project `LICENSE` and `NOTICE`, `TAILSCALE-LICENSE.txt`, and the complete native
dependency bundle under `third_party/`. Its manifest binds the dependency
inventory and notice hashes to the delivered helper binaries. The Python wheel
has its own license files; those do not replace the native bundle.

The library never downloads or builds a helper during import, installation or
use. Go is needed to build from source, not to run a prebuilt helper. Running
`runtime-llama-tsnet --version` reports the protocol and backend version without
starting a node or creating state.

## Build from source

The native module pins Go `1.27.1` and Tailscale `1.104.1` in
[`helpers/tsnet/go.mod`](../helpers/tsnet/go.mod), with dependency checksums in
`go.sum`. The Go command may download the pinned toolchain and dependencies if
they are absent. Callers can select isolated `GOCACHE`, `GOMODCACHE` and `GOPATH`
locations. Building the helper does not enroll a node.

On macOS or Linux, from the repository root:

```sh
cd helpers/tsnet
./build.sh /absolute/output/runtime-llama-tsnet
go test -tags ts_omit_webclient ./...
go test -race -tags ts_omit_webclient ./...
go vet -tags ts_omit_webclient ./...
```

The output directory must already exist. The build script uses
`-tags ts_omit_webclient -trimpath -buildvcs=false`. The first flag excludes the
upstream web interface and its embedded assets; disabling it at runtime would
not remove those assets from a binary. The other flags remove local source
paths and automatic VCS metadata from the build. `GOOS` and `GOARCH` select a
cross-compilation target; a successful cross-build does not establish that a
binary has been executed on that operating system.

Windows PowerShell can invoke Go directly, from the repository root:

```powershell
Set-Location helpers/tsnet
$env:GOOS = 'windows'
$env:GOARCH = 'amd64'
$env:CGO_ENABLED = '0'
go build -tags ts_omit_webclient -trimpath -buildvcs=false -o 'C:\absolute\output\runtime-llama-tsnet.exe' ./cmd/runtime-llama-tsnet
go test -tags ts_omit_webclient ./...
go vet -tags ts_omit_webclient ./...
```

Replace the output path with an existing directory on your machine. Run a race
check only with a toolchain and C compiler supported for that target. The Go
commands above are instructions for local verification, not claims that they
have run on your system.

If distributing your own binaries, regenerate the native license inventory
against those exact artifacts using
[`helpers/tsnet/third_party/README.md`](../helpers/tsnet/third_party/README.md).
Ship the resulting bundle with each native archive. Keep checksums outside the
archives they describe and normalize archive owner information and timestamps.

## Supply consumer-owned configuration

All paths below are placeholders. Replace them with absolute paths appropriate
for the consumer's operating system, and select a server already present in its
tailnet. The node credential and the llama-server API key serve different roles.

```python
import os
from pathlib import Path

from runtime_llama import TailnetConfig, TailnetHelper, build_adapter

if os.name == "nt":
    helper_path = Path(r"C:\absolute\bin\runtime-llama-tsnet.exe")
    state_path = Path(r"C:\absolute\private-state\llama-client")
else:
    helper_path = Path("/absolute/bin/runtime-llama-tsnet")
    state_path = Path("/absolute/private-state/llama-client")

config = TailnetConfig(
    executable=helper_path,
    state_dir=state_path,
    hostname="llama-client",
    control_url=None,
    # control_url="https://headscale.example",
    auth_key=os.environ.get("CONSUMER_TAILNET_AUTH_KEY"),
    ephemeral=False,
    startup_timeout_sec=30,
    dial_timeout_sec=10,
    shutdown_timeout_sec=5,
)

base_url = "https://inference.example-tailnet.ts.net:8080"
with TailnetHelper(config) as network:
    catalog = network.discover_models(base_url)
    llm = build_adapter({
        "role_id": "coder",
        "model": "consumer-selected-model",
        "base_url": base_url,
        "proxy_url": network.proxy_url,
        "stream_idle_timeout_sec": None,
        # "api_key_env": "CONSUMER_LLAMA_API_KEY",
    })
    try:
        result = llm.invoke_with_ctx([
            {"role": "user", "content": "Describe this function."},
        ])
        print(result.text)
    finally:
        llm.close()
```

Reading `CONSUMER_TAILNET_AUTH_KEY` is an application choice in this example.
The library does not implicitly resolve Tailscale credentials from the
environment. `auth_key=None` supports reusing previously enrolled state. A new
node needs an enrollment method prepared by the consumer; this helper protocol
does not provide an interactive browser-login flow.

`control_url=None` selects Tailscale. An explicit HTTPS root URL selects the
consumer's Headscale coordinator. Credentials, query strings, fragments and
non-root paths are rejected. Coordinator TLS remains verified and requires a
trusted certificate. The consumer must qualify its chosen Headscale version
and policies against the pinned Tailscale client; accepting a configuration
does not establish compatibility with every coordinator deployment.

## Own and protect node state

Use a dedicated, absolute state directory. The helper applies private access
permissions, takes an exclusive operating-system lock and writes an identity
marker binding the directory to its coordinator, hostname and ephemeral mode.
It rejects foreign state, symlinks, concurrent use and incompatible identity
changes. It does not delete or migrate another client's state.

The directory contains persistent node credentials. The consumer owns its
storage, backups and retirement. Creating a second helper for another identity
requires a separate state directory. An ephemeral-node setting does not make
the directory safe to share or publish.

## Python API and lifecycle

The public API is:

```text
TailnetConfig(
    executable, state_dir, hostname,
    control_url=None, auth_key=None, ephemeral=False,
    startup_timeout_sec=30.0, dial_timeout_sec=10.0,
    shutdown_timeout_sec=5.0,
)
TailnetHelper(config)
helper.start(*, cancellation_probe=None) -> TailnetHelper
helper.client(base_url, **kwargs) -> LlamaCppClient
helper.discover_models(base_url, **kwargs) -> ModelCatalog
helper.proxy_url -> str
helper.running -> bool
helper.close() -> None
```

`TailnetConfig` is a frozen dataclass and accepts named fields. Paths must be
absolute; hostname must be one DNS label. Lifecycle budgets are finite, from
one nanosecond through 86400 seconds. `TailnetError` is a subclass of
`LLMBackendError`; startup cancellation raises `LLMCancelledError`. HTTP calls
retain the existing client and adapter exception types.

Importing the library and constructing configuration or a supervisor do not
launch processes, create state or contact servers. Call `start()` explicitly or
enter `with TailnetHelper(config)`. A helper has one lifecycle: failure or
closure does not restart it or replay inference. Construct another supervisor
when another lifecycle is needed.

`client()` creates an owned raw HTTP client whose route and session cannot be
replaced. `close()` drains these clients and in-flight helper discovery calls
as part of shutdown. `discover_models()` uses the public read-only catalog
operation, with cancellation tied to the helper lifetime; it does not load or
unload models.

`proxy_url` also routes `build_adapter`, `build_embedding_adapter`, or a direct
`discover_models` call. The consumer owns these externally constructed clients
and adapters and must close them before the helper, as in the example. Every
owned HTTP session, including later concurrent leases, inherits the explicit
proxy. An injected client with a conflicting route, or a borrowed Requests
session combined with an explicit route, is rejected before sending.

Closing one stream or cancelling one HTTP call stops that request without
closing a shared helper or other callers' requests. Close abandoned stream
generators and close adapters when finished. Closing the helper is the shared
transport teardown and affects every request using it.

## Credentials and diagnostics

`proxy_url` is a secret, temporary capability for the local authenticated
proxy. Do not print it, share it or save it in persistent configuration. Profile
representations omit the route; opted-in thinking diagnostics redact its URL
and token. `TailnetConfig` excludes `auth_key` from its representation, and the
supervisor representation reports lifecycle state rather than credentials.
Other configuration fields still describe the consumer's environment.

The proxy capability and enrollment credential travel to the helper through
stdin, not command-line arguments. The explicit HTTP route ignores ambient
proxy and netrc configuration. The supervisor removes `TS_`/`TSNET_` settings
and ambient proxy variables from the child environment without changing the
consumer process's environment.

The helper exposes only an authenticated HTTP proxy on `127.0.0.1` and an
ephemeral port. It does not expose Tailscale's LocalAPI or web interface. Proxy
authentication is removed before a request reaches llama-server; the server's
Host and Authorization headers are preserved.

## Peer routing and HTTP behavior

Destinations must be exact peer FQDNs or peer IP addresses found in the current
tailnet status. Short names, generic Internet destinations and LAN addresses
reached through subnet routers are outside this route. Use the server's
tailnet address rather than its Ethernet address. Unknown, ambiguous or removed
peers fail without an operating-system DNS or direct-dial fallback.

Each dial verifies the current peer map, then passes the selected numeric
address directly to the userspace netstack. This relies on a shim through
upstream `Server.Sys()`, an explicitly unstable interface. Tailscale `1.104.1`
is part of the transport contract; changing the pin requires reviewing and
qualifying that integration again. Encrypted coordination, WireGuard and DERP
traffic still uses the host network. The no-fallback guarantee concerns
inference traffic, not the helper's coordination transport.

HTTP requests do not follow redirects or receive automatic replay. For plain
HTTP, the helper opens a fresh upstream connection for each request to prevent
Go's reused-connection retry behavior. HTTPS uses CONNECT: TLS remains between
the Python client and the server, with the client's certificate and hostname
verification. The helper does not terminate TLS or introduce a local CA.

SSE bytes are forwarded immediately, without an additional helper idle
deadline. `stream_idle_timeout_sec=None` retains the existing optional-idle
contract. Connection, peer dial, JSON, embedding and discovery operations keep
their finite timeout budgets. Stopping a local request is not proof that the
remote server has stopped inference.

## Shutdown, protocol and qualification limits

EOF on stdin or a shutdown command closes the proxy, active HTTP connections,
CONNECT tunnels and backend. Completion of either CONNECT copy closes both
directions; this is an HTTP transport, not a general TCP half-close tunnel.
The Python supervisor enforces its shutdown budget, then terminate/kill and
worker draining. A failed drain produces an explicit error.

Protocol 1 accepts one configuration JSON line on stdin, bounded to 64 KiB.
READY on stdout contains protocol, status, the credential-free local proxy URL
and backend version. A shutdown JSON command or EOF follows. Status messages
are bounded to 8 KiB and errors use fixed codes. Stderr is drained without
echoing potentially sensitive upstream text; upstream application logging and
logtail submission are disabled.

Deterministic fixtures, compilation and offline smoke checks establish local
software behavior. They do not establish live Tailscale or Headscale enrollment,
inference over a VPN, compatibility with a particular consumer deployment, or
execution of cross-built binaries on Linux or Windows. Qualifying those
environments is a separate, explicitly chosen activity. See the release
verification report for the checks actually performed on published artifacts.
