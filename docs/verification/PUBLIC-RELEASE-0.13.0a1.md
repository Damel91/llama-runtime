# Public prerelease 0.13.0a1

The stakeholder explicitly authorized pushing and publishing 0.13.0a1 and then
informing the existing CodingCastle chat on 2026-10-23. This is a GitHub
**prerelease**; 0.11.1 remains the latest stable version. No consumer code, pin,
installation, service or job is changed by this publication.

Flower publication authority: `REQ-000043`, `GOAL-000025`, `MILE-000026`,
`CHANGE-000028`, `PACKET-000029`. Implementation and qualification remain under
`REQ-000042` / `PACKET-000028`, with evidence 211 and 212. The earlier numerical
finding `FIND-000007` stays open; publication does not accept or resolve it.

## Exact artifact and public source

The wheel is the same artifact used for clean installation and live checks:

```text
runtime_llama-0.13.0a1-py3-none-any.whl
SHA256 c1cb36802579be6bc58624f657f87ad64d7d7305218b3d8bd022b9635c56ce3f
```

Its embedded README retains the historical local-candidate wording from the
qualification build. The current repository, release notes and public
`RELEASE-MANIFEST.json` record the subsequent publication authorization.
The wheel is not rebuilt to change that metadata. All 39 runtime payload
entries, including the typing marker, are byte-identical to the tagged source.
The source archive includes the current publication documentation and retains
the tested runtime code and selected corpus unchanged.

The seven public assets are the wheel, source archive, three optional native
helper archives, `RELEASE-MANIFEST.json` and `SHA256SUMS`. Packaging validation
checks provenance, private-data exclusions, licenses, archive contents and
runtime/source equivalence. Publication verification checks main/tag identity,
prerelease status, latest stable, and anonymous download hashes for every asset.
No PyPI upload is part of this release.

## Embedding contract

Only an explicit character-budget request can calibrate the 2,500 selected
synthetic texts: the 500 densest measured Nomic examples in each category.
Individual category budgets use their weighted mean characters/prepared-token
ratio at 80%; `mixed` uses the complete selected mean at 90%. Recommendations
remain fixed during failure recovery. Startup without a budget request and
ordinary embedding do not tokenize or calibrate.

A direct native HTTP 500 starts bounded failed-subbatch isolation in the
existing scheduler. Successful child vectors are retained and never replayed
by isolation. An isolated rejected singleton returns
`EmbeddingChunkSplitRequiredError`, containing successful original indices and
two lossless midpoint source halves for each rejected occurrence. A generic
500 is not proof of context overflow, and one halving need not be sufficient.
No tokenizer request, learned-budget read/write or recalibration is performed
by this fallback. The consumer retains completed vectors and retries only the
requested halves, with its own finite progress/retry bounds.

Ordered asynchronous windows preserve prior successful windows and remain
usable after a caught recoverable split event. Cancellation, deadlines,
identity/topology checks, result credits, drained close and the existing
scheduler remain in force. The previous exact-diagnostic overflow type remains
importable, but automatic native-500 recovery uses the new partial-results
contract. Consumer migration belongs to its own chat and lifecycle.

See the [public API](../AUTOMATIC-EMBEDDING.md) and
[consumer migration example](../HALVING-CONSUMER.md).

## Verification and limits

The [qualification record](HARD-CORPUS-HALVING-0.13.0a1.md) records 2,031 passing
tests, Ruff checks, builds and isolated wheel installation. A real local HTTP
barrier fixture proves overlapping requests on two aliases with one slot each;
a healthy alias completes while another isolates its failed input.

The live qualification used one loaded Nomic replica, context/native batch/
microbatch 512 and one slot. Cold calibration counted exactly 2,500 texts in
12.5469 seconds. The completed warm continuation processed 511 texts in
2.5971 seconds using two embedding POSTs, ten metadata GETs and no tokenizer
calls. The earlier observation was 3.2882 seconds with the same request counts.
These are client wall observations, not a controlled speedup or pure GPU time.

Two good original vectors survived isolation. The rejected text needed a
second halving round before all four fragments succeeded. The mixed budget
remained 1,101 characters throughout. Retained good vectors and same-alias
text-versus-token-ID comparisons had maximum absolute error zero, within the
unchanged `1e-5` / `0.999999` thresholds. The single-alias evidence does not
close the earlier cross-replica finding or qualify every model/profile.

## Optional helper and licenses

The three optional tsnet archives reuse the verified v0.6.0 executables
byte-for-byte. Only the versioned containers and provenance text are updated;
sources and dependency/license inventory are unchanged. Archives retain the
project Apache-2.0 license, Davide Mele copyright, NOTICE, Tailscale license
and third-party notices. No new native build, OS execution, live VPN enrollment
or VPN inference qualification is claimed.
