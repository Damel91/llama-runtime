# Public release 0.15.0 verification

Date: 2026-10-26, Europe/Rome. Distribution `runtime-llama`; Python import
`runtime_llama`. The operator authorized implementing milestones 36–40,
removing unrequested text/model/memory limits, testing resource release with
10,000 texts, producing a verified wheel and publishing a release. Consumer
code, configurations, pins, installations, jobs, databases and services are
outside these producer changes. No model loading/build/hardware administration
is authorized or implied by this release.

Flower MCP is the lifecycle authority: adopted architecture revision 6,
requirements `REQ-000061`–`REQ-000069`, milestones `MILE-000036`–`MILE-000040`
and packets `PACKET-000039`–`PACKET-000043`. Recorded implementation and tests
are host-reported; application acceptance remains separate.

## Delivered contract

- Flat `aiter_texts()` accepts a finite sequence or async text source and yields
  source-ordered `EmbeddingRecordResult(index, embedding)` values. Occurrences
  remain distinct, including duplicates. The consumer need not group inputs
  into POST-sized windows. 10,000 is a qualification sample, not a new maximum.
- Existing complete-list and complete-window APIs remain. They occupy the
  memory naturally required by their promised output, without the former
  compulsory whole-result reservation.
- Scalar capacity fields default to `None` and require
  `embedding_capacity_limits=True` to activate. The default `False` does not
  silently reinterpret old automatically saved numbers as opt-in. Supplied
  policies are independent; omission does not select replacement quotas.
- Batch targets and positive global HTTP guards retain their explicit roles.
  B/N determines configured quota width, not maximum submission size. The
  original lease-per-canonical-replica policy and shared scheduler remain.
- Compulsory 64 MiB pools, derived owner H, fake direct token-context reservations,
  handoff 75/25, response `pool/16`, materialization-based admission and
  memory-driven candidate halving are removed. A deliberately selected
  handoff-byte policy is independent of pending and response fields.
- Preparation caching is disabled unless explicitly enabled. Enabled session
  retention has ownership and clears at close; optional entry/byte policies
  require the capacity opt-in. Catalog handling has no fixed model-count cap,
  and worker defaults follow usable replicas rather than fixed 1/2/4 values.
- Ordinary direct embedding performs no per-input tokenizer requests and no
  implicit calibration. Effective context drives explicit character
  recommendations; native microbatch is not substituted as its ceiling.
  The 2,500-text corpus and category 80% / mixed 90% margins are unchanged.
- Transfer and cleanup release library-owned input/result/HTTP/task/Future and
  retired-generation references. Primary failures, cancellation, repeated Stop,
  shared-owner survival and actual transport drain remain explicit.

This does not promise constant RSS while input or consumer-retained results
grow, size a new budget automatically from RAM, silently spill to disk,
truncate texts or introduce another scheduler.

## Public interfaces

```text
service.aiter_texts(texts, *, input_type="document", timeout=None,
                    cancel=None, close_source=False) -> EmbeddingRecordStream
adapter.aiter_texts(texts, *, input_type="document", timeout=None,
                    cancel=None, close_source=False) -> EmbeddingRecordStream

service.astream_texts(windows, *, input_type="document", max_pending_windows=None,
                      timeout=None, cancel=None, close_source=False)
adapter.prepare_texts(texts, *, input_type="document", timeout=None, cancel=None)
adapter.embed_prepared(inputs, *, timeout=None, cancel=None)
```

Use `async with` or `aclose()` for deterministic early exit. Dropping the flat
facade also signals Stop and schedules the inner cleanup on a live owning
loop. Cleanup retains real work until drain; a closed loop cannot execute it
and uncooperative user code cannot safely be killed.

Automatic recoverable errors retain `EmbeddingChunkSplitRequiredError`.
For flat input, `window_index` identifies the global record index; nested
indices/spans are relative to that occurrence. Explicit iteration can continue
with later inputs. Consumer source splitting, graph and storage stay external.

See [embedding contracts](../EMBEDDING-CONTRACTS.md),
[async input/results](../ASYNC-EMBEDDING-WINDOWS.md),
[automatic use](../AUTOMATIC-EMBEDDING.md) and
[capacity/ownership](../CONCURRENCY-AND-DIAGNOSTICS.md).

## Verification evidence

The progressive-facade focused suite passed **16 tests in 17.71 seconds** in a
local development run. It includes an actual shared-engine 10,000-text flat
stream with ordered output and no remaining owners, result reservations or
active leases; weak-reference checks distinguish output transfer from retained
past values. Additional focused cases exercise first delivery before source
exhaustion, completed-undelivered results and a dropped facade during blocked
HTTP whose lease remains owned until real drain. This local result is not a
replacement for final repository/package verification.

The final repository suite passed **2286 tests in 107.44 seconds**.
Four warnings come from deliberately unverified HTTPS loopback fixtures.
Ruff checks, formatting of all 138 Python files and `git diff --check` passed.
The implementation commit is `e6cbffc06af35bd994db23980e32775e11ffd22f`; documentation is committed separately.

The focused resource suite passed **28 tests in 28.46 seconds**. It covers
10,000 synchronous D768 vectors, one 10,000-row async D768 window, 10,000
progressive D2 records, repeated-cycle allocation attribution, retained errors,
shared-owner cancellation and real executor retirement across 5 → 7 → 2
verified replicas. Three additional inert-stream cases check that repeated
close releases a never-acquired source without reading it.

A normalized wheel was built and installed in a clean Python environment
outside the checkout. Imports resolved to installed site-packages; dependency
checks passed. The installed smoke returned 10,000 ordered progressive records
and left no owners, result reservations or handoff tickets. It used synthetic
borrowed clients, including zero-delay responses; partial POSTs and fresh
preflight renewals depend on arrival timing. These checks do not constitute a
server throughput benchmark or a claim of fewer control HTTP calls.

Wheel: `runtime_llama-0.15.0-py3-none-any.whl`.

SHA-256: `4fcfca0e369ce55d94c74af74e1ba53bce921472fd635ae4b1b339a23cdaa49f`.

Release staging checks compare all wheel and source-archive payloads with the
committed source, verify normalized archive headers, corpus checksums, licenses,
public API signatures, examples, relative links and private-data exclusions.
The release manifest records the final source revision, artifact hashes and
verification boundaries. `SHA256SUMS` covers all published payloads and the
manifest; anonymous publication checks compare the downloaded assets and raw
source with those staged bytes.

Memory verification observes actual ownership, weak references/finalizers and
repeated-cycle allocations. RSS is reported with scope when measured, not used
as an invented MiB acceptance threshold. Caller-held outputs, live backlog,
intentional caches and worker-process memory are distinguished from references
to completed library operations.

## Model and application boundary

The earlier numerical deviation remains recorded as the operator-accepted
known profile limitation (`FIND-000007`); unchanged strict gates are not
rewritten as a numerical pass. These tests do not establish a retrieval-quality
result, a throughput gain or pure GPU timing. Any separately performed live
qualification must state its actual model/build/topology and measurement scope.
No fresh 0.15 live inference is claimed by the deterministic checks above.

The optional server-observation module remains a client/provider contract;
it does not install or require a native-server extension. No server capability
or consumer integration is accepted merely by publishing the wheel.

## Publication and licensing

Apache-2.0, copyright Davide Mele, NOTICE and upstream MIT attribution remain.
Any optional native helper assets retain their original provenance and complete
third-party notices. Reused helper binaries do not imply a new native-platform
execution, VPN enrollment or inference qualification.

Public artifacts exclude deployment secrets, private addresses, local paths,
private live source texts, raw vectors and caches. Publication does not update
consumer installations or perform PyPI distribution unless separately stated
with actual evidence.
