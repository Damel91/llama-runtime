# Automatic embedding replica discovery

The consumer selects an endpoint, a logical model and an inference profile.
The library discovers physical instances already loaded on that endpoint and
manages routing, admission and limits. The server operator owns model files,
builds, processes and device placement. This document describes the protocol
and client behavior; it does not certify a server installation.

## Runtime identity protocol

The native `GET /v1/models` catalog exposes `data[].tags`. Noxi publishes three
versioned tags for embedding models:

- `noxi-content-sha256-v1:<64 hexadecimal digits>`: SHA-256 of a canonical
  manifest containing SHA-256 byte digests and sizes for all GGUF shards, plus
  any adapters and scales. This is neither a path hash nor necessarily the
  direct SHA-256 digest of a single GGUF file.
- `noxi-embedding-sha256-v1:<64 hexadecimal digits>`: a separate identity for
  the configuration that affects embedding semantics, including effective
  pooling, normalization and relevant overrides.
- `noxi-model-name-v1:<percent-encoded name>`: the GGUF basename, used for
  logical model selection. A name alone does not establish identity.

These hashes exclude aliases, paths, timestamps, GPUs, allocated context,
Flash Attention, slot counts and batch/microbatch sizes. The library does not
infer equivalence from alias suffixes, paths or vector dimensions alone.
Observed context and capabilities remain separate admission checks.

The protocol producer hashes files by streaming before launch, deduplicates
reads of the same source and bounds its cache. It checks for mutations around
hashing and immediately before process launch. Generated namespaces are
reserved while user tags are preserved. The attestation concerns files
observed by the runtime; it is not a cryptographic inspection of GPU memory.
The client trusts its configured runtime and does not download weights to
recompute their digests.

## Selection and routing

Discovery is lazy: constructing an adapter performs no network requests.
`embedding_replica_discovery=True` opts into automatic discovery. The existing
explicit `embedding_replicas` mode remains available; the two modes cannot be
combined. Automatic mode requires no replica list or replica count. The
consumer selects an anchor alias, a published logical name or
`sha256:<content digest>`.

The library considers only loaded catalog entries, validates their tags and
groups replicas only when both content and embedding-semantic identities
match. A name resolving to different identities is ambiguous. Malformed,
incomplete or conflicting tags do not authorize pooling. An exact alias
without attestation can use the single-instance fallback, explicitly
unverified, without inferring additional replicas.

A verified identity is pinned for the adapter's lifetime. Reusing an alias for
different weights does not silently migrate the adapter's embedding identity.
Internal replica identifiers are monotonic operational identifiers; consumers
should retain the stable content and semantic identities for indexes and
caches instead.

Each automatic replica permits one active request. The adapter also enforces
a global active-work limit and a bounded queue. Multiple replicas permit
requests to distinct processes without requiring multiple slots within one
process. These limits do not establish simultaneous GPU execution.

Calls share the adapter's scheduler. Pool changes take effect at a boundary
with no active calls, using a bounded wait that responds to cancellation and
closure. Already started work is not replayed. Discovery failure does not
authorize new requests using stale catalog data. The client neither invokes
administrative load/unload operations nor probes properties of unloaded
aliases.

## Capacity and verification

The library combines configured budgets with each replica's observed context,
including task prefixes and special tokens in exact input counts. For example,
changing allocated context to 4096 tokens changes admission capacity, not model
identity. Flash Attention and allocated context remain runtime execution
settings.

Runtime-attested weight identity, measured numerical embedding equivalence
and application retrieval quality are separate properties. A weight digest
does not establish the latter two. A change in loading settings alone does
not establish either an improvement or a regression in retrieval quality.

Verification should cover alias/name/digest selection, equal bytes at different
paths, changed bytes at the same path, invalid tags, unloaded aliases, topology
changes, pinned identities, internal identifiers, concurrent callers,
per-replica limits, prefixes under smaller contexts, outages, cancellation and
closure. A live test of verified pooling requires a server that publishes the
protocol tags. Local protocol tests do not certify that deployment.

## Distribution windows and preparation

`batch_size` is the total text budget for a distribution window. Only the
shared scheduler distributes that window across distinct aliases. A per-alias
quota is separate from HTTP token packing and the context limit of an
individual input.

Public preparation returns exact counts even for oversized inputs. Counts are
reused only when their identity and tokenizer policy are verifiable, and
admission checks fresh limits. See [embedding contracts](EMBEDDING-CONTRACTS.md)
for API details and [verification requirements](REQUIREMENTS.md) for the public
verification boundary. Version 0.4.1 preserves the inference and embedding
behavior of 0.4.0.
