# Automatic embedding replica discovery

The consumer selects an endpoint, a logical model and an inference profile.
The library discovers canonical, independently addressable loaded destinations
on that endpoint and
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
`embedding_replica_discovery=True` opts into automatic discovery. Explicit `embedding_replicas` remains available; the two modes cannot be
combined. An explicit list containing multiple aliases must prove the same
content and semantic identity. Duplicate synonyms of one catalog entry are
not separate replicas, and unverified/heterogeneous multi-alias lists fail
before engine admission. A single unverified legacy alias remains isolated. Automatic mode requires no replica list or replica count. The
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

The 0.15.0 contract admits **one original embedding POST per
verified canonical replica**. Missing/`None` `embedding_request_parallelism`
uses all N replicas; a positive G limits original work to `min(N,G)`. Existing
positive settings stay guards. Positive explicit `max_in_flight` values remain
accepted for configuration compatibility, but their effective limit is one.
Requested/effective values and the limiting reason remain visible.

Native `total_slots`, `n_batch` and `n_ubatch` are diagnostics. They do not
multiply original HTTP permits or prove device placement. The library neither
assigns native slot identifiers nor changes the server's parallel setting.
Recovery/isolation children keep their original alias lease and run serially
inside it; other replicas can advance.

Quotas use N independently of G. With `batch_size=256,N=4,G=2`, the plan still
contains up to four 64-text quotas, with at most two original POSTs active.
Byte/explicit token limits may split a quota further. Fair observation and
dispatch turns prevent a continuously ready first alias from starving peers
when `G<N`. Source order and result identity remain unchanged.

Calls share the adapter's scheduler. Pool/executor changes take effect only
after the previous local topology generation's work and references drain,
under the original finite cancellation/deadline. A shrink blocks new admission
without revoking existing leases or results. Explicit opt-in resource policies
can admit fewer than `min(N,G)` POSTs; no fixed owner or memory pool is
installed by default. Worker defaults follow the actual usable replica count.
Already started work is not replayed. Discovery failure does not
authorize new requests using stale catalog data. The client neither invokes
administrative load/unload operations nor probes properties of unloaded
aliases.

## Capacity and verification

The library combines configured budgets with each replica's observed context,
including task prefixes and special tokens in exact input counts. For example,
changing allocated context to 4096 tokens changes admission capacity, not model
identity. Flash Attention and allocated context remain runtime execution
settings.

Text count, input context, native batch/microbatch sizes and aggregate HTTP
token budgets remain separate. Only explicit adapter/per-replica
`batch_token_limit` settings bound the aggregate HTTP token count. Native
`n_batch` and `n_ubatch` are diagnostic execution observations; they do not
establish a universal input-splitting capability or an HTTP array limit. An
unpublished capability remains unknown and server rejections retain their
typed errors.

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

`batch_size` is an optional total-text target distributed across distinct
aliases by the shared scheduler. It is independent of the number of texts
submitted or the size of public source windows. Omission installs no
replacement target. A per-alias quota is separate from explicit HTTP token
packing and the context limit of an individual input.

Public preparation returns exact counts even for oversized inputs. Counts are
reused only when their identity, tokenizer build and task policy are verifiable,
and admission checks fresh limits after waiting and before adaptive children.
Preparation and inference overlap using reused executors. Ordinary automatic
embedding sends strings without per-input tokenization or public sealed
records; explicit preparation and its tamper checks retain their contract.
Owner/queue and byte policies default to `None` and only supplied values behind
`embedding_capacity_limits=True` activate. This does not change B/G policies
or impose a new maximum catalog size.

See [embedding contracts](EMBEDDING-CONTRACTS.md) for API details and
[verification requirements](REQUIREMENTS.md) for the verification boundary.
Current [release evidence](verification/PUBLIC-RELEASE-0.15.0.md) is separate
from earlier profile measurements and consumer migration.

## Local and remote generations

A local scheduling generation protects client ownership; a weight digest does
not attest loaded context, build, slot count or a same-weight reload. Native
catalog/props checks therefore remain fresh at POST/recovery boundaries.
The optional [server-observation SDK](SERVER-OBSERVATION.md) describes the
stronger, server-enforced contract needed to reuse authorization across POSTs.
It is not inferred from native tags and has no implicit deployment dependency.

Worker capacity, optional resource policies, explicit cache retention and
result/drain ownership are specified in
[concurrency and ownership](CONCURRENCY-AND-DIAGNOSTICS.md). Existing complete
window APIs and the flat progressive `aiter_texts()` interface use this same
scheduler. Historical [candidate evidence](verification/REPLICA-OWNERSHIP-CANDIDATE.md)
is retained under its original scope; it is not a new 0.15.0 live qualification.
