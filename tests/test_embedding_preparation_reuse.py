"""Public native embedding APIs reuse exact tokens without reusing admission."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest

from runtime_llama.client import ClientHTTPError
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError
from runtime_llama.adapters.embedding import LlamaCppEmbeddingAdapter

from test_embedding_contracts import (
    NAME,
    ContractClient,
    configured,
    vector,
)


class ReuseClient(ContractClient):
    """Native token arrays round-trip through an exact deterministic tokenizer."""

    supports_embedding_token_ids = True

    def __init__(self, replicas=2, **kwargs):
        self.vocab = 0x110002
        self.token_text = {}
        self.wire_inputs = []
        super().__init__(replicas, **kwargs)

    def set_aliases(self, aliases, **kwargs):
        super().set_aliases(aliases, **kwargs)
        for entry in self.catalog:
            entry["meta"]["n_vocab"] = self.vocab

    def tokenize(self, content, **kwargs):
        super().tokenize(content, **kwargs)
        tokens = (0, *(ord(character) + 2 for character in content), 1)
        with self.lock:
            self.token_text[tokens] = content
        return {"tokens": list(tokens)}

    def embeddings(self, input, **kwargs):
        with self.lock:
            self.wire_inputs.append(input)
            decoded = [
                self.token_text[tuple(row)] if isinstance(row, list) else row
                for row in input
            ]
        return super().embeddings(decoded, **kwargs)


def runtime(client, **changes):
    return configured(client, embedding_batch_wait_sec=0, timeout=10, **changes)


def assert_native_rows(client):
    assert client.wire_inputs
    assert all(
        isinstance(row, list) and row[0] == 0 and row[-1] == 1
        for request in client.wire_inputs
        for row in request
    )


@pytest.mark.parametrize(
    "limits",
    [
        {"embedding_preparation_cache_entries": 0},
        {"embedding_preparation_cache_bytes": 0},
        {
            "embedding_preparation_cache_entries": 0,
            "embedding_preparation_cache_bytes": 0,
        },
    ],
)
def test_configured_zero_cache_retains_no_tokens_and_preserves_exact_outputs(limits):
    client = ReuseClient(1)
    adapter = runtime(client, **limits)
    try:
        assert (
            adapter.embed_query("same") == adapter.embed_query("same") == vector("same")
        )
        assert len(client.token_calls) == 2
        view = adapter.health_status()["preparation_cache"]
        assert view["entries"] == view["bytes"] == view["hits"] == 0
        assert_native_rows(client)
    finally:
        adapter.close()


def test_configured_small_lru_evicts_exact_tokens_and_close_drains_storage():
    client = ReuseClient(1)
    adapter = runtime(client, embedding_preparation_cache_entries=1)
    try:
        for text in ("a", "b", "b", "a"):
            assert adapter.embed_query(text) == vector(text)
        assert [text for text, _ in client.token_calls] == ["a", "b", "a"]
        view = adapter.health_status()["preparation_cache"]
        assert view["entries"] == 1 and view["evictions"] == 2
        assert view["max_entries"] == 1 and view["hits"] == 1
    finally:
        adapter.close()
    view = adapter._preparation_cache.snapshot()
    assert view["closed"] and view["entries"] == view["bytes"] == 0


@pytest.mark.parametrize(
    "configured_bytes,pending_bytes,expected",
    [(4096, 65536, 4096), (32768, 65536, 16384), (0, 65536, 0)],
)
def test_configured_cache_bytes_are_separate_and_clipped_to_pending_fraction(
    configured_bytes, pending_bytes, expected
):
    client = ReuseClient(1)
    adapter = runtime(
        client,
        embedding_preparation_cache_bytes=configured_bytes,
        embedding_max_pending_bytes=pending_bytes,
        embedding_max_response_bytes=1024,
    )
    try:
        assert adapter._preparation_cache.snapshot()["max_bytes"] == expected
        assert adapter.embedding_preparation_cache_bytes == configured_bytes
        assert adapter.embedding_max_response_bytes == 1024
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "name,value",
    [
        ("embedding_max_response_bytes", None),
        ("embedding_max_response_bytes", 0),
        ("embedding_max_response_bytes", True),
        ("embedding_preparation_cache_entries", -1),
        ("embedding_preparation_cache_entries", True),
        ("embedding_preparation_cache_bytes", None),
        ("embedding_preparation_cache_bytes", 1.5),
    ],
)
def test_direct_adapter_validation_rejects_invalid_bypassed_profiles_before_client(
    monkeypatch, name, value
):
    client = ReuseClient(1)
    adapter = runtime(client)
    profile = adapter.profile
    adapter.close()
    # Defend the direct constructor even if a consumer bypasses the frozen
    # profile's own validation or supplies a structurally compatible object.
    object.__setattr__(profile, name, value)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid budgets must fail before creating HTTP clients")

    monkeypatch.setattr("runtime_llama.adapters.embedding.LlamaCppClient", forbidden)
    with pytest.raises(ValueError, match=name):
        LlamaCppEmbeddingAdapter(profile)


def test_raw_and_prepared_public_calls_reuse_tokens_across_verified_aliases():
    client = ReuseClient()
    adapter = runtime(client)
    try:
        prepared = adapter.prepare_texts(["città 🙂"])
        assert prepared[0].token_count == len("città 🙂") + 2
        assert len(client.token_calls) == 1
        assert adapter.embed_texts(["città 🙂"] * 4) == [vector("città 🙂")] * 4
        assert {alias for alias, _ in client.embedding_calls} == {"embed_0", "embed_1"}
        assert len(client.token_calls) == 1
        client.set_aliases(["copy-a", "copy-b"])
        assert adapter.embed_texts(["città 🙂"]) == [vector("città 🙂")]
        assert adapter.embed_prepared(prepared) == [vector("città 🙂")]
        assert client.embedding_calls[-1][0] in {"copy-a", "copy-b"}
        assert len(client.token_calls) == 1
        assert adapter.health_status()["preparation_cache"]["hits"] >= 3
        assert_native_rows(client)
    finally:
        adapter.close()


def test_same_prefixed_text_still_has_distinct_task_and_policy_keys():
    client = ReuseClient(1)
    adapter = runtime(client, embedding_prefixes={"document": "P: ", "query": "P: "})
    try:
        document = adapter.prepare_texts(["x"])[0]
        query = adapter.prepare_query("x")
        raw = adapter.prepare_texts(["P: x"], input_type="raw")[0]
        assert [document.text, query.text, raw.text] == ["P: x"] * 3
        assert len(client.token_calls) == 3
        assert adapter.prepare_query("x").token_count == len("P: x") + 2
        assert adapter.prepare_texts(["x"])[0].token_count == len("P: x") + 2
        assert adapter.prepare_texts(["P: x"], input_type="raw")[0].token_count == 6
        assert len(client.token_calls) == 3
        assert adapter.embed_prepared([document, query, raw]) == [vector("P: x")] * 3
        assert len(client.token_calls) == 3
        assert all(
            options["add_special"] is True
            and options["parse_special"] is True
            and options["with_pieces"] is False
            for _, options in client.token_calls
        )
        assert_native_rows(client)
    finally:
        adapter.close()


def test_different_configured_prefixes_do_not_share_adapter_cache_or_double_prefix():
    client = ReuseClient(1)
    adapters = [
        runtime(client, embedding_prefixes={"query": prefix})
        for prefix in ("Q: ", "query: ")
    ]
    try:
        for adapter, prefix in zip(adapters, ("Q: ", "query: ")):
            prepared = adapter.prepare_query("hello")
            assert prepared.text == prefix + "hello"
            assert adapter.embed_query("hello") == vector(prefix + "hello")
            assert adapter.embed_prepared([prepared]) == [vector(prefix + "hello")]
        assert [text for text, _ in client.token_calls] == ["Q: hello", "query: hello"]
        assert [texts for _, texts in client.embedding_calls] == [
            ("Q: hello",),
            ("Q: hello",),
            ("query: hello",),
            ("query: hello",),
        ]
    finally:
        for adapter in adapters:
            adapter.close()


@pytest.mark.parametrize("metadata", ["build", "vocab"])
def test_changed_tokenizer_build_or_vocabulary_requires_a_new_exact_tokenization(
    metadata,
):
    client = ReuseClient(1)
    adapter = runtime(client)
    try:
        assert adapter.embed_query("same") == vector("same")
        assert len(client.token_calls) == 1
        if metadata == "build":
            client.builds["embed_0"] = "native-build-new"
        else:
            client.catalog[0]["meta"]["n_vocab"] += 1
        assert adapter.embed_query("same") == vector("same")
        assert len(client.token_calls) == 2
        assert adapter.embed_query("same") == vector("same")
        assert len(client.token_calls) == 2
        assert adapter.health_status()["preparation_cache"]["entries"] == 2
    finally:
        adapter.close()


@pytest.mark.parametrize("unknown", ["identity", "build"])
def test_unknown_identity_or_build_never_caches_or_reuses_prepared_counts(unknown):
    client = ReuseClient(1, verified=unknown != "identity")
    if unknown == "build":
        client.builds["embed_0"] = None
    adapter = runtime(client, model="embed_0" if unknown == "identity" else NAME)
    try:
        prepared = adapter.prepare_query("same")
        assert adapter.embed_query("same") == vector("same")
        assert adapter.embed_prepared([prepared]) == [vector("same")]
        assert len(client.token_calls) == 3
        view = adapter.health_status()["preparation_cache"]
        assert view["entries"] == view["bytes"] == view["hits"] == 0
        assert all(
            isinstance(row, str) for request in client.wire_inputs for row in request
        )
    finally:
        adapter.close()


def test_changed_pinned_weights_reject_before_cache_or_inference():
    client = ReuseClient(1)
    adapter = runtime(client)
    try:
        prepared = adapter.prepare_query("same")
        prior = adapter._preparation_cache.snapshot()
        client.catalog[0]["tags"][0] = "noxi-content-sha256-v1:" + "c" * 64
        with pytest.raises(LLMBackendError, match="identity_changed|model_not_loaded"):
            adapter.embed_prepared([prepared])
        assert len(client.token_calls) == 1
        assert not client.embedding_calls
        assert adapter._preparation_cache.snapshot() == prior
    finally:
        adapter.close()


def test_cached_tokens_do_not_grant_inference_after_loaded_context_is_reduced():
    client = ReuseClient(1)
    adapter = runtime(client)
    try:
        assert adapter.embed_query("longer") == vector("longer")
        client.contexts["embed_0"] = 3
        prepared = adapter.prepare_query("longer")
        assert prepared.token_count == 8
        assert prepared.context_token_limit == 3
        assert prepared.fits_context is False
        assert len(client.token_calls) == 1
        for invoke in (
            lambda: adapter.embed_query("longer"),
            lambda: adapter.embed_prepared([prepared]),
        ):
            with pytest.raises(LLMBackendError, match="exceeds_context"):
                invoke()
        assert len(client.token_calls) == 1
        assert len(client.embedding_calls) == 1
        assert adapter.health_status()["preparation_cache"]["hits"] >= 3
    finally:
        adapter.close()


def test_oversized_count_is_measurable_reusable_and_never_sent_to_inference():
    client = ReuseClient(1, context=4)
    adapter = runtime(client)
    try:
        first = adapter.prepare_query("oversized")
        again = adapter.prepare_query("oversized")
        assert first.token_count == again.token_count == 11
        assert first.fits_context is again.fits_context is False
        assert first._token_ids is again._token_ids is None
        assert len(client.token_calls) == 1
        with pytest.raises(LLMBackendError, match="exceeds_context"):
            adapter.embed_prepared([again])
        assert len(client.token_calls) == 1
        assert not client.embedding_calls
        # A later larger operator-owned context permits the same exact tokens.
        client.contexts["embed_0"] = 32
        assert adapter.embed_query("oversized") == vector("oversized")
        assert len(client.token_calls) == 1
        assert_native_rows(client)
    finally:
        adapter.close()


@pytest.mark.parametrize("foreign", [False, True])
def test_warmed_cache_never_bypasses_prepared_seal_validation(foreign):
    client = ReuseClient(1)
    adapter = runtime(client)
    other = runtime(client)
    try:
        measured = adapter.prepare_query("warm")
        assert adapter.embed_query("warm") == vector("warm")
        untrusted = (
            other.prepare_query("warm")
            if foreign
            else replace(measured, token_count=measured.token_count + 1)
        )
        calls, posts = len(client.token_calls), len(client.embedding_calls)
        with pytest.raises(LLMBackendError, match="invalid_preparation"):
            adapter.embed_prepared([untrusted])
        assert len(client.token_calls) == calls
        assert len(client.embedding_calls) == posts
    finally:
        adapter.close()
        other.close()


def test_every_distinct_cold_input_still_tokenizes_once_before_repeated_reuse():
    client = ReuseClient()
    adapter = runtime(client)
    incoming = [f"input-{index:03d}" for index in range(511)]
    try:
        measured = adapter.prepare_texts(incoming)
        assert len(measured) == 511
        assert Counter(text for text, _ in client.token_calls) == Counter(incoming)
        assert len(client.token_calls) == 511
        assert adapter.embed_texts(incoming) == [vector(text) for text in incoming]
        assert len(client.token_calls) == 511
        assert adapter.health_status()["preparation_cache"]["hits"] == 511
        assert_native_rows(client)
    finally:
        adapter.close()


@pytest.mark.parametrize("failure", ["backend", "cancel", "timeout"])
def test_unsuccessful_tokenizations_never_enter_cache_and_recovery_retries(failure):
    client = ReuseClient(1)
    adapter = runtime(client)
    stop = Event()
    try:
        if failure == "backend":
            client.token_failure = ClientHTTPError(500, error_type="server_error")
            with pytest.raises(LLMBackendError):
                adapter.prepare_query("retry")
            client.token_failure = None
        else:
            client.token_gate.clear()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    adapter.prepare_query,
                    "retry",
                    cancel=stop.is_set,
                    timeout=0.05 if failure == "timeout" else 5,
                )
                assert client.token_started.wait(2)
                if failure == "cancel":
                    stop.set()
                expected = (
                    LLMTimeoutError if failure == "timeout" else LLMCancelledError
                )
                with pytest.raises(expected):
                    future.result(3)
            client.token_gate.set()
            stop.clear()
        assert adapter._preparation_cache.snapshot()["entries"] == 0
        assert adapter.prepare_query("retry").token_count == 7
        assert len(client.token_calls) == 2
        assert adapter._preparation_cache.snapshot()["entries"] == 1
        assert not client.embedding_calls
    finally:
        client.token_gate.set()
        adapter.close()


def test_public_cache_diagnostics_observe_bound_eviction_oversize_skip_and_close():
    client = ReuseClient(1)
    adapter = runtime(client, embedding_max_pending_bytes=8192)
    try:
        # Exercise LRU turnover without retaining twenty sealed public records
        # in a single 8 KiB output window. Each result transfers before the next.
        for index in range(20):
            adapter.prepare_query(f"text-{index:02d}")
        # Larger than the 2 KiB cache, but the public sealed result still fits
        # its independent 8 KiB result limit.
        adapter.prepare_query("x" * 40)
        view = adapter.health_status()["preparation_cache"]
        assert view["max_entries"] == 1024
        assert view["max_bytes"] == 2048
        assert 0 < view["entries"] <= view["max_entries"]
        assert 0 < view["bytes"] <= view["max_bytes"]
        assert view["peak_bytes"] <= view["max_bytes"]
        assert view["evictions"] > 0
        assert view["oversized_skips"] > 0
        adapter.close()
        closed = adapter._preparation_cache.snapshot()
        assert closed["closed"]
        assert closed["entries"] == closed["bytes"] == 0
    finally:
        adapter.close()
