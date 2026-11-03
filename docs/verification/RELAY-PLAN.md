# Confirmed local relay scope

The stakeholder confirmed this scope on 2026-11-03. Flower MCP owns canonical
requirements REQ-000080–REQ-000092, use case GOAL-000033 and milestones 43–45.
This document explains adopted architecture revision 8; it is not an alternate
requirements or architecture store. Repository execution follows the accepted
Flower plan and its dependency-ordered contracts.

The Python import remains available. An optional local service adds OpenAI-
compatible Chat Completions, embeddings and model discovery through the existing
SDK adapters. A self-hosted browser console configures endpoints, defaults,
prefixes, access and optional Tailscale/Headscale transport, shows observed
models and per-request diagnostics, and exports versioned OpenCode settings.

The first producers establish atomic configuration snapshots and the admin
protocol, plus a public single-model direct embedding path for native servers
without content tags. The latter uses the existing engine, performs fresh target
observations, never fabricates weight identity, and does not pool unverified
replicas, reuse prepared token IDs or tokenize ordinary inputs. Verified pools
retain the existing automatic embedding path.

The relay then maps native SDK outcomes to HTTP/SSE. One incoming chat request
enters the SDK once. Rejected proposals remain private; the original generation
and up to three corrections use one selected route. Dedicated request workers,
cooperative cancellation and configuration-generation leases preserve sibling
requests through disconnects and retirement. Internal diagnostic events are not
public Chat Completions chunks. No new implicit text/model/request/memory quota
or second embedding scheduler is introduced.

Dependent browser and integration checks exercise real routes, consumer parser
behavior, resource cleanup, imports and packaging. Native Flash Next findings
are recorded separately from injected rejection tests; XML text without native
evidence is not reclassified as an executable call. Release evidence includes
clean base/relay wheel installation, content and license audits and public
artifact hashes. Application acceptance is not inferred from those checks.

Consumer installations and Noxi Inference Desk remain unchanged during this
increment. The same chat and overall engineering goal subsequently continue
Noxi's separately recorded runtime/UI/background/Debian scope. Remote model
loading integration between products remains a later agreed extension.
