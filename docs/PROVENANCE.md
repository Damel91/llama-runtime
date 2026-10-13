# Provenance and attribution

The public `llama-runtime` repository preserves the Python distribution name
`runtime-llama` and import namespace `runtime_llama`. Version `0.4.1` changes
publication metadata, documentation and licensing while preserving `0.4.0`
inference and embedding behavior. This document identifies reused components;
it does not transfer historical deployment results to the public release.

## Functional baseline

Functional contracts, role configuration, reasoning separation, repetition and
output protections, error classifications, usage and diagnostic types, and MCP
data contracts derive from the project owner's earlier
`lmstudio-agent-runtime` implementation. Generation and embedding adapters and
selected regression cases preserve or adapt those behaviors for llama-server.
The [functional parity guide](FUNCTIONAL-PARITY.md) records the resulting
behavior and its limits.

The baseline was a separate implementation. Its earlier verification results
do not establish compatibility of this backend. The public project is licensed
under [Apache-2.0](../LICENSE), copyright 2026 Davide Mele. The separately
identified llamatelemetry-derived component retains its upstream MIT notice
and license.

## HTTP client adapted from llamatelemetry

- Upstream: [llamatelemetry](https://github.com/llamatelemetry/llamatelemetry).
- Source snapshot: `a0a1df88e3b143d1e38e241b2599e82c56092a0b`.
- Declared upstream package version at that snapshot: `0.1.1`.
- Original source: [`llamatelemetry/api/client.py`](https://github.com/llamatelemetry/llamatelemetry/blob/a0a1df88e3b143d1e38e241b2599e82c56092a0b/llamatelemetry/api/client.py), class `LlamaCppClient`.
- SHA-256 of the inspected source file: `7f9c93f9852012ad08f992ea1ee7b5c1260789556af05410398c24250c5f878c`.
- Upstream copyright: 2024 Waqas Muhammad.
- [Original MIT license](https://github.com/llamatelemetry/llamatelemetry/blob/a0a1df88e3b143d1e38e241b2599e82c56092a0b/LICENSE), reproduced in [`_vendor/llamatelemetry.LICENSE`](../src/runtime_llama/_vendor/llamatelemetry.LICENSE) and included in distributions.

[`client.py`](../src/runtime_llama/client.py) adapts constructor behavior,
headers, HTTP dispatch and endpoint/payload mappings. It does not import the
llamatelemetry package. Upstream package initialization, CUDA bootstrap and
server setup are excluded. The source snapshot identifies code provenance;
it is not a dependency pin to the complete upstream package.

Explicit adaptations include strict incremental SSE parsing, cancellation
while waiting for JSON or SSE, exclusive reusable sessions for concurrent
requests, response closure, classified errors without remote body dumps and
raw JSON responses preserving unknown fields. Missing usage counters are not
converted to zero. These adaptations are described in [NOTICE](../NOTICE) and
the source header while the original MIT license remains reproduced intact.

## Server and consumer boundaries

The configured server manager owns build, compilation, binaries, model files,
processes, devices, loading and unloading. The library uses HTTP APIs of the
selected server. Server-managed auto-loading is allowed. It does not add a
local inference fallback or explicit administrative loading calls.

Optional replica discovery consumes the `noxi-*` content, embedding-semantic
and model-name tags documented in
[AUTOMATIC-EMBEDDING-REPLICAS.md](AUTOMATIC-EMBEDDING-REPLICAS.md). Their names
are part of the supported protocol. The client trusts the configured runtime's
attestation and does not hash model files or certify GPU memory itself.

Deterministic tests and local HTTP fixtures verify library behavior. The
[probe guide](../probes/README.txt) describes optional reproducible live checks.
Each live qualification applies to the build, model and configuration actually
tested; it does not certify other deployments or consumer applications.
Application integration, tool execution and acceptance remain separate work.
