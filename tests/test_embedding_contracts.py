"""Public embedding contract tests: total budgets, exact preparation and drain."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import math
from threading import Event, Lock
import time
from urllib.parse import quote

import pytest

from runtime_llama import PreparedEmbeddingInput, build_embedding_adapter
from runtime_llama.adapters.embedding import LlamaCppEmbeddingAdapter
from runtime_llama.client import ClientHTTPError
from runtime_llama.configuration import AdapterProfile
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError

CONTENT = "a" * 64
SEMANTIC = "b" * 64
NAME = "contract-embedding.gguf"


def vector(text):
    return [float(sum(map(ord, text))), float(len(text) + 1)]


def eventually(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("fixture condition was not reached")
        time.sleep(0.002)


class ContractClient:
    """Deterministic external server with independent context and HTTP capacity."""

    def __init__(self, replicas=2, *, context=2048, native_batch=2048, verified=True):
        self.lock = Lock()
        self.catalog = []
        self.contexts = {}
        self.batches = {}
        self.builds = {}
        self.counts = {}
        self.token_calls = []
        self.embedding_calls = []
        self.props_calls = []
        self.token_active = self.token_peak = 0
        self.embed_active = self.embed_peak = 0
        self.by_alias = Counter()
        self.alias_peak = Counter()
        self.token_gate = Event()
        self.token_gate.set()
        self.embed_gate = Event()
        self.embed_gate.set()
        self.token_started = Event()
        self.embed_started = Event()
        self.failure = None
        self.token_failure = None
        self.closed = False
        self.delay = 0
        self.set_aliases(
            [f"embed_{i}" for i in range(replicas)],
            context=context,
            native_batch=native_batch,
            verified=verified,
        )

    def set_aliases(self, aliases, *, context=2048, native_batch=2048, verified=True):
        self.catalog = []
        for alias in aliases:
            tags = (
                [
                    "noxi-content-sha256-v1:" + CONTENT,
                    "noxi-embedding-sha256-v1:" + SEMANTIC,
                    "noxi-model-name-v1:" + quote(NAME, safe=""),
                ]
                if verified
                else []
            )
            args = ["llama-server"]
            if native_batch is not None:
                args += ["--batch-size", str(native_batch)]
            self.catalog.append(
                {
                    "id": alias,
                    "aliases": [],
                    "tags": tags,
                    "status": {"value": "loaded", "args": args},
                    # Neither this training metadata nor ubatch is HTTP capacity.
                    "context_length": 999999,
                    "meta": {"n_ctx_train": 999999},
                }
            )
            self.contexts[alias] = context
            self.batches[alias] = native_batch
            self.builds[alias] = "test-build-1"

    def health(self, **_kwargs):
        return {"status": "ok"}

    def models(self, **_kwargs):
        return {"data": deepcopy(self.catalog)}

    def props(self, model=None, **_kwargs):
        self.props_calls.append(model)
        model = model or self.catalog[0]["id"]
        return {
            "model_alias": model,
            "total_slots": 1,
            "default_generation_settings": {"n_ctx": self.contexts[model]},
            "build_info": self.builds[model],
        }

    @staticmethod
    def _wait(gate, *, cancel=None, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while not gate.wait(0.002):
            if cancel and cancel():
                raise LLMCancelledError()
            if deadline is not None and time.monotonic() >= deadline:
                raise LLMTimeoutError("test_transport_timeout")
        if cancel and cancel():
            raise LLMCancelledError()

    def tokenize(self, content, *, cancel=None, timeout=None, **kwargs):
        with self.lock:
            self.token_active += 1
            self.token_peak = max(self.token_peak, self.token_active)
            self.token_calls.append((content, dict(kwargs)))
            self.token_started.set()
        try:
            self._wait(self.token_gate, cancel=cancel, timeout=timeout)
            if self.token_failure:
                raise self.token_failure
            return {"tokens": list(range(self.counts.get(content, len(content) + 1)))}
        finally:
            with self.lock:
                self.token_active -= 1

    def embeddings(self, input, *, model, cancel=None, timeout=None, **_kwargs):
        texts = tuple(input)
        with self.lock:
            self.embed_active += 1
            self.embed_peak = max(self.embed_peak, self.embed_active)
            self.by_alias[model] += 1
            self.alias_peak[model] = max(self.alias_peak[model], self.by_alias[model])
            self.embedding_calls.append((model, texts))
            self.embed_started.set()
        try:
            self._wait(self.embed_gate, cancel=cancel, timeout=timeout)
            if self.delay:
                time.sleep(self.delay * (sum(map(ord, texts[0])) % 3))
            if self.failure:
                error = self.failure(model, texts)
                if error:
                    raise error
            return {
                "data": list(
                    reversed(
                        [
                            {"index": index, "embedding": vector(text)}
                            for index, text in enumerate(texts)
                        ]
                    )
                )
            }
        finally:
            with self.lock:
                self.embed_active -= 1
                self.by_alias[model] -= 1

    def close(self):
        self.closed = True


def configured(client, **changes):
    values = {
        "adapter_type": "embedding",
        "model": NAME,
        "embedding_replica_discovery": True,
        "batch_size": 256,
        # Keep packing assertions independent from readiness/thread timing.
        "embedding_batch_wait_sec": 0.5,
        "embedding_request_parallelism": 4,
        "embedding_queue_size": 2,
        "token_count_parallelism": 2,
        "timeout": 2,
        "normalize": False,
        # ContractClient's server fixture always returns two-dimensional vectors.
        # Tests of cold/unknown width opt out explicitly with dimension=None.
        "dimension": 2,
    }
    values.update(changes)
    return LlamaCppEmbeddingAdapter(
        AdapterProfile.from_config("documents", values), client=client
    )


def texts(count, start=0):
    return [f"item-{index:05d}" for index in range(start, start + count)]


def assert_once(client, expected):
    observed = [text for _, batch in client.embedding_calls for text in batch]
    assert Counter(observed) == Counter(expected)
    assert all(client.alias_peak[alias] <= 1 for alias in client.alias_peak)


@pytest.mark.parametrize(
    ("replicas", "count", "cap", "quotas"),
    [
        (2, 256, 4, [128, 128]),
        (4, 256, 4, [64, 64, 64, 64]),
        (4, 256, 2, [64, 64, 64, 64]),
        (2, 255, 4, [127, 128]),
        (4, 255, 4, [63, 64, 64, 64]),
        (4, 3, 4, [1, 1, 1]),
        (4, 1, 4, [1]),
    ],
)
def test_total_text_budget_is_balanced_over_usable_replicas(
    replicas, count, cap, quotas
):
    client = ContractClient(replicas, context=50000, native_batch=50000)
    client.delay = 0.001
    adapter = configured(client, embedding_request_parallelism=cap)
    incoming = texts(count)
    try:
        assert adapter.embed_texts(incoming) == [vector(text) for text in incoming]
        assert sorted(len(batch) for _, batch in client.embedding_calls) == quotas
        assert len({alias for alias, _ in client.embedding_calls}) == len(quotas)
        assert_once(client, incoming)
        assert client.embed_peak <= min(replicas, cap)
    finally:
        adapter.close()


def test_multiple_total_budget_windows_preserve_every_original_offset():
    client = ContractClient(context=50000, native_batch=50000)
    adapter = configured(client)
    incoming = texts(513)
    try:
        assert adapter.embed_texts(incoming) == [vector(text) for text in incoming]
        assert sorted(len(batch) for _, batch in client.embedding_calls) == [
            1,
            128,
            128,
            128,
            128,
        ]
        assert_once(client, incoming)
    finally:
        adapter.close()


def test_native_step_budget_does_not_split_http_arrays():
    client = ContractClient(context=2048, native_batch=2048)
    incoming = texts(256)
    client.counts = dict.fromkeys(incoming, 32)
    for entry in client.catalog:
        entry["status"]["args"] += ["--ubatch-size", "64"]
    adapter = configured(client)
    try:
        assert adapter.embed_texts(incoming) == [vector(text) for text in incoming]
        assert len(client.embedding_calls) == 2
        assert Counter(alias for alias, _ in client.embedding_calls) == {
            "embed_0": 1,
            "embed_1": 1,
        }
        assert all(len(batch) == 128 for _, batch in client.embedding_calls)
        assert all(
            sum(client.counts[t] for t in batch) == 4096
            for _, batch in client.embedding_calls
        )
        assert len(client.token_calls) == 256
        assert_once(client, incoming)
    finally:
        adapter.close()


@pytest.mark.parametrize("native,configured_limit", [(2048, 96), (96, 2048)])
def test_http_aggregate_budget_uses_explicit_policy_independently_of_native(
    native, configured_limit
):
    client = ContractClient(context=4096, native_batch=native)
    incoming = texts(9)
    client.counts = dict.fromkeys(incoming, 32)
    adapter = configured(client, batch_token_limit=configured_limit)
    try:
        assert adapter.embed_texts(incoming) == [vector(text) for text in incoming]
        assert all(
            sum(client.counts[t] for t in batch) <= configured_limit
            for _, batch in client.embedding_calls
        )
        assert_once(client, incoming)
    finally:
        adapter.close()


def test_public_preparation_measures_boundary_oversize_and_reuses_exact_count():
    client = ContractClient()
    client.counts = {"D: exact": 2048, "D: over": 2049}
    adapter = configured(client, embedding_prefixes={"document": "D: ", "query": "Q: "})
    try:
        measured = adapter.prepare_texts(["exact", "over"])
        assert isinstance(measured, tuple)
        assert all(isinstance(item, PreparedEmbeddingInput) for item in measured)
        exact, over = measured
        assert (
            exact.text,
            exact.token_count,
            exact.context_token_limit,
            exact.batch_token_limit,
        ) == ("D: exact", 2048, 2048, None)
        assert exact.fits_context and exact.fits_input
        assert exact.fits_request is None
        assert not over.fits_context and not over.fits_input
        assert over.fits_request is None
        assert over.token_count == 2049 and over.max_input_tokens == 2048
        assert exact.identity_verified and exact.content_sha256 == CONTENT
        assert exact.embedding_sha256 == SEMANTIC
        assert not client.embedding_calls
        calls = len(client.token_calls)
        assert adapter.embed_prepared([exact]) == [vector("D: exact")]
        assert len(client.token_calls) == calls
        prior = len(client.embedding_calls)
        with pytest.raises(LLMBackendError, match="exceeds_context|exceeds_batch"):
            adapter.embed_prepared([over])
        assert len(client.embedding_calls) == prior
        assert adapter.embed_query("ok") == vector("Q: ok")
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "input_type,prefix", [("document", "D: "), ("query", "Q: "), ("raw", "")]
)
def test_public_task_preparation_is_literal_including_specials_unicode_and_empty(
    input_type, prefix
):
    client = ContractClient()
    adapter = configured(client, embedding_prefixes={"document": "D: ", "query": "Q: "})
    incoming = ["  città 🔬 <bos>\n", "", "D: already prefixed"]
    try:
        measured = adapter.prepare_texts(incoming, input_type=input_type)
        assert [item.text for item in measured] == [prefix + text for text in incoming]
        assert [item.token_count for item in measured] == [
            len(prefix + text) + 1 for text in incoming
        ]
        assert all(item.input_type == input_type for item in measured)
        for _, options in client.token_calls:
            assert options["add_special"] is True
            assert options["parse_special"] is True
            assert options["with_pieces"] is False
        assert adapter.embed_prepared(measured) == [
            vector(prefix + text) for text in incoming
        ]
        assert len(client.token_calls) == len(incoming)
    finally:
        adapter.close()


def test_empty_query_is_measurable_and_empty_collections_need_no_io():
    client = ContractClient()
    adapter = configured(client, embedding_prefixes={"query": "Q: "})
    try:
        assert adapter.prepare_texts([]) == ()
        assert adapter.embed_prepared([]) == []
        assert adapter.embed_texts([]) == []
        assert adapter.embed_query("") == []
        assert (
            not client.token_calls
            and not client.embedding_calls
            and not client.props_calls
        )
        measured = adapter.prepare_query("")
        assert measured.text == "Q: " and measured.token_count == 4
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("text", "changed"),
        ("token_count", 1),
        ("input_type", "raw"),
        ("context_token_limit", 99999),
        ("batch_token_limit", 99999),
        ("content_sha256", "c" * 64),
    ],
)
def test_altered_preparation_is_not_a_count_authority(field, value):
    client = ContractClient()
    adapter = configured(client)
    try:
        prepared = adapter.prepare_query("original")
        with pytest.raises(FrozenInstanceError):
            prepared.text = "mutated"
        forged = replace(prepared, **{field: value})
        with pytest.raises(LLMBackendError, match="invalid_preparation"):
            adapter.embed_prepared([forged])
        assert not client.embedding_calls
    finally:
        adapter.close()


def test_foreign_adapter_preparation_is_rejected_without_embedding():
    client = ContractClient()
    first, second = configured(client), configured(client)
    try:
        prepared = first.prepare_query("original")
        with pytest.raises(LLMBackendError, match="invalid_preparation"):
            second.embed_prepared([prepared])
        assert not client.embedding_calls
    finally:
        first.close()
        second.close()


def test_same_identity_new_topology_reuses_count_but_fresh_limit_can_reject():
    client = ContractClient()
    adapter = configured(client)
    try:
        prepared = adapter.prepare_texts(["original"])
        calls = len(client.token_calls)
        client.set_aliases(["copy_renamed"], context=2048, native_batch=2048)
        assert adapter.embed_prepared(prepared) == [vector("original")]
        assert len(client.token_calls) == calls
        assert client.embedding_calls[-1][0] == "copy_renamed"
        client.contexts["copy_renamed"] = 4
        previous = len(client.embedding_calls)
        with pytest.raises(LLMBackendError, match="exceeds_context"):
            adapter.embed_prepared(prepared)
        assert len(client.embedding_calls) == previous
    finally:
        adapter.close()


def test_changed_runtime_build_recounts_and_unverified_fallback_never_reuses():
    for verified in (True, False):
        client = ContractClient(1, verified=verified)
        adapter = configured(client, model=NAME if verified else "embed_0")
        try:
            measured = adapter.prepare_query("original")
            initial = len(client.token_calls)
            if verified:
                client.builds["embed_0"] = "test-build-2"
            assert adapter.embed_prepared([measured]) == [vector("original")]
            assert len(client.token_calls) == initial + 1
        finally:
            adapter.close()


def test_prepared_identity_change_is_rejected_before_inference():
    client = ContractClient()
    adapter = configured(client)
    try:
        measured = adapter.prepare_query("original")
        for entry in client.catalog:
            entry["tags"][0] = "noxi-content-sha256-v1:" + "c" * 64
        with pytest.raises(LLMBackendError, match="identity_changed|model_not_loaded"):
            adapter.embed_prepared([measured])
        assert not client.embedding_calls
    finally:
        adapter.close()


def test_context_valid_but_aggregate_invalid_input_is_measurable_never_sent():
    client = ContractClient(context=4096, native_batch=64)
    client.counts["over request"] = 65
    adapter = configured(client, batch_token_limit=64)
    try:
        measured = adapter.prepare_texts(["over request"])[0]
        assert measured.fits_context and not measured.fits_request
        assert measured.max_input_tokens == 64
        with pytest.raises(LLMBackendError, match="exceeds_batch"):
            adapter.embed_prepared([measured])
        assert not client.embedding_calls
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "argv",
    [
        ["--batch-size", "0"],
        ["--batch-size", "2048", "-b", "2048"],
        ["--batch-size"],
        ["--batch-size", "bad"],
        ["--batch-size=2048", "--batch-size=4096"],
        ["--batch-size", True],
    ],
)
def test_invalid_or_ambiguous_native_batch_metadata_fails_closed(argv):
    client = ContractClient()
    for entry in client.catalog:
        entry["status"]["args"] = ["llama-server", *argv]
    adapter = configured(client)
    try:
        with pytest.raises(LLMBackendError, match="batch|metadata"):
            adapter.prepare_query("test")
        assert not client.embedding_calls
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "argv",
    [["--batch-size=64"], ["-b", "64"], ["--batch-size", "64", "--ubatch-size", "2"]],
)
def test_supported_native_batch_argv_forms_are_diagnostic_not_http_limits(argv):
    client = ContractClient()
    for entry in client.catalog:
        entry["status"]["args"] = ["llama-server", *argv]
    adapter = configured(client)
    try:
        assert adapter.prepare_query("test").batch_token_limit is None
        assert adapter._primary.loaded_batch_token_limit == 64
    finally:
        adapter.close()


def test_absent_native_batch_remains_unknown_not_training_context_or_microbatch():
    client = ContractClient(native_batch=None)
    for entry in client.catalog:
        entry["status"]["args"] += ["--ubatch-size", "16", "--", "--batch-size", "1"]
    adapter = configured(client)
    try:
        prepared = adapter.prepare_query("test")
        assert prepared.context_token_limit == 2048
        assert prepared.batch_token_limit is None
        assert prepared.fits_request is None
    finally:
        adapter.close()


def test_shared_bulk_preparation_bounds_tokenizer_concurrency_and_close_drains():
    client = ContractClient(4)
    client.token_gate.clear()
    adapter = configured(
        client, batch_size=8, token_count_parallelism=2, embedding_queue_size=1
    )
    with ThreadPoolExecutor(max_workers=4) as callers:
        jobs = [
            callers.submit(adapter.prepare_texts, texts(32, i * 100)) for i in range(3)
        ]
        assert client.token_started.wait(2)
        eventually(lambda: client.token_active == 2)
        time.sleep(0.02)
        assert client.token_peak == 2
        assert adapter._scheduler.snapshot()["queued"] <= 1
        closed = callers.submit(adapter.close)
        closed.result(timeout=3)
        for job in jobs:
            with pytest.raises((LLMCancelledError, LLMBackendError)):
                job.result(timeout=3)
    assert client.token_active == client.embed_active == 0
    assert not client.embedding_calls
    assert adapter._scheduler.snapshot()["in_flight"] == 0
    assert adapter._scheduler.snapshot()["queued"] == 0


def test_preparation_cancellation_is_isolated_from_another_shared_call():
    client = ContractClient()
    client.token_gate.clear()
    cancelled = Event()
    adapter = configured(client, token_count_parallelism=1, embedding_queue_size=1)
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(
                adapter.prepare_texts, texts(16), cancel=cancelled.is_set
            )
            assert client.token_started.wait(2)
            second = callers.submit(adapter.prepare_query, "independent")
            cancelled.set()
            with pytest.raises(LLMCancelledError):
                first.result(timeout=2)
            client.token_gate.set()
            assert second.result(timeout=2).text == "independent"
        assert client.token_active == 0
    finally:
        client.token_gate.set()
        adapter.close()


def test_preparation_queue_and_transport_use_one_finite_deadline():
    client = ContractClient()
    client.token_gate.clear()
    adapter = configured(client, token_count_parallelism=1, embedding_queue_size=0)
    started = time.monotonic()
    try:
        with pytest.raises(LLMTimeoutError):
            adapter.prepare_texts(texts(32), timeout=0.06)
        assert time.monotonic() - started < 0.5
        assert client.token_active == 0
        assert not client.embedding_calls
        client.token_gate.set()
        assert adapter.prepare_query("recovered").text == "recovered"
    finally:
        client.token_gate.set()
        adapter.close()


def test_partial_quota_failure_does_not_replay_successful_outputs_and_recovers():
    client = ContractClient(context=50000, native_batch=50000)
    client.failure = lambda alias, batch: (
        ClientHTTPError(500) if alias == "embed_1" else None
    )
    adapter = configured(client)
    incoming = texts(256)
    try:
        with pytest.raises(ClientHTTPError):
            adapter.embed_texts(incoming)
        observed = [text for _, batch in client.embedding_calls for text in batch]
        assert all(count == 1 for count in Counter(observed).values())
        assert client.embed_active == 0
        assert adapter._scheduler.snapshot()["in_flight"] == 0
        client.failure = None
        assert adapter.embed_query("recovered") == vector("recovered")
    finally:
        adapter.close()


def test_topology_replacement_waits_for_active_call_without_replaying_positions():
    client = ContractClient(context=50000, native_batch=50000)
    client.embed_gate.clear()
    adapter = configured(client)
    incoming = texts(256)
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(adapter.embed_texts, incoming)
            eventually(lambda: client.embed_active == 2)
            client.set_aliases(["embed_0"], context=50000, native_batch=50000)
            second = callers.submit(adapter.embed_texts, texts(3, 1000))
            time.sleep(0.02)
            assert not second.done()
            client.embed_gate.set()
            assert first.result(timeout=3) == [vector(text) for text in incoming]
            assert second.result(timeout=3) == [vector(text) for text in texts(3, 1000)]
        assert_once(client, incoming + texts(3, 1000))
        assert set(adapter.health_status()["replicas"]) == {"embed_0"}
    finally:
        client.embed_gate.set()
        adapter.close()


@pytest.mark.parametrize("timeout", [0, -1, math.inf, math.nan, True])
def test_public_preparation_rejects_nonfinite_or_invalid_timeout_before_io(timeout):
    client = ContractClient()
    adapter = configured(client)
    try:
        with pytest.raises(ValueError):
            adapter.prepare_query("x", timeout=timeout)
        assert (
            not client.token_calls
            and not client.embedding_calls
            and not client.props_calls
        )
    finally:
        adapter.close()


def test_public_factory_exports_contract_type_without_private_imports():
    adapter = build_embedding_adapter(
        {"model": "external", "base_url": "http://127.0.0.1:1"}
    )
    try:
        assert callable(adapter.prepare_texts)
        assert callable(adapter.prepare_query)
        assert callable(adapter.embed_prepared)
        assert PreparedEmbeddingInput.__name__ == "PreparedEmbeddingInput"
    finally:
        adapter.close()


@pytest.mark.parametrize("incoming", ["not a bulk", b"not a bulk", 5, ["ok", 5]])
def test_prepare_bulk_requires_a_sequence_of_strings_before_io(incoming):
    client = ContractClient()
    adapter = configured(client)
    try:
        with pytest.raises(ValueError):
            adapter.prepare_texts(incoming)
        assert (
            not client.token_calls
            and not client.props_calls
            and not client.embedding_calls
        )
    finally:
        adapter.close()


@pytest.mark.parametrize("incoming", ["not preparations", b"not preparations", 5])
def test_embed_prepared_rejects_nonsequence_or_string_bulk_before_io(incoming):
    client = ContractClient()
    adapter = configured(client)
    try:
        with pytest.raises(ValueError):
            adapter.embed_prepared(incoming)
        assert (
            not client.token_calls
            and not client.props_calls
            and not client.embedding_calls
        )
    finally:
        adapter.close()


def test_prepared_seal_and_effective_limit_cannot_be_replaced():
    client = ContractClient()
    adapter = configured(client)
    try:
        prepared = adapter.prepare_query("original")
        for forged in (
            replace(prepared, _seal=b"wrong"),
            replace(prepared, max_input_tokens=999999),
            replace(prepared, identity_verified=False),
        ):
            with pytest.raises(LLMBackendError, match="invalid_preparation"):
                adapter.embed_prepared([forged])
        assert not client.embedding_calls
    finally:
        adapter.close()


@pytest.mark.parametrize("tokens", [None, [True], [-1], [1.5], "tokens"])
def test_bad_native_exact_count_never_becomes_an_inference_permit(tokens):
    client = ContractClient()
    client.tokenize = lambda *_args, **_kwargs: {"tokens": tokens}
    adapter = configured(client)
    try:
        with pytest.raises(LLMBackendError, match="tokenization"):
            adapter.prepare_query("bad count")
        assert not client.embedding_calls
    finally:
        adapter.close()


def test_tokenizer_transport_failure_is_typed_drained_and_next_call_recovers():
    client = ContractClient()
    client.token_failure = ClientHTTPError(500)
    adapter = configured(client)
    try:
        with pytest.raises(ClientHTTPError):
            adapter.prepare_texts(texts(32))
        assert client.token_active == 0 and not client.embedding_calls
        assert adapter._scheduler.snapshot()["in_flight"] == 0
        client.token_failure = None
        assert adapter.prepare_query("recovery").text == "recovery"
    finally:
        adapter.close()


def test_waiting_preparation_deadline_does_not_stop_independent_active_caller():
    client = ContractClient(1)
    client.token_gate.clear()
    adapter = configured(
        client,
        token_count_parallelism=1,
        embedding_request_parallelism=1,
        embedding_queue_size=0,
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(adapter.prepare_query, "independent", timeout=2)
            assert client.token_started.wait(2)
            second = callers.submit(adapter.prepare_query, "waiter", timeout=0.05)
            with pytest.raises(LLMTimeoutError):
                second.result(timeout=1)
            assert not first.done()
            client.token_gate.set()
            assert first.result(timeout=2).text == "independent"
        assert client.token_active == 0 and not client.embedding_calls
    finally:
        client.token_gate.set()
        adapter.close()


def test_probe_rejects_numerical_instability_without_relaxing_thresholds():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "probes"
        / "qualify_embedding_contracts.py"
    )
    spec = importlib.util.spec_from_file_location("embedding_contract_probe", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    first = [1.0] + [0.0] * 767
    same = probe.compare(first, first)
    assert same["maximum_absolute_error"] == 0
    assert same["minimum_cosine"] == 1
    changed = [0.0, 1.0] + [0.0] * 766
    with pytest.raises(AssertionError, match="unchanged threshold"):
        probe.compare(first, changed)
    assert (
        probe.MAX_ABS,
        probe.MIN_COSINE,
        probe.MAX_NORM_ERROR,
        probe.MAX_RAW_PROJECTION_ERROR,
    ) == (1e-5, 0.999999, 1e-6, 1e-12)


def test_fresh_explicit_http_budget_can_reject_previously_measured_input():
    client = ContractClient()
    client.counts["original"] = 65
    adapter = configured(client)
    try:
        prepared = adapter.prepare_texts(["original"])
        calls = len(client.token_calls)
        assert prepared[0].fits_input is True
        adapter.batch_token_limit = 64
        with pytest.raises(LLMBackendError, match="exceeds_batch"):
            adapter.embed_prepared(prepared)
        assert len(client.token_calls) == calls
        assert not client.embedding_calls
    finally:
        adapter.close()


@pytest.mark.parametrize("value", [True, 0, -1, 64.0, "64"])
def test_supported_native_property_must_be_a_positive_integer(value):
    client = ContractClient()
    original = client.props
    client.props = lambda **kwargs: {**original(**kwargs), "n_batch": value}
    adapter = configured(client)
    try:
        with pytest.raises(LLMBackendError, match="batch"):
            adapter.prepare_query("test")
        assert not client.embedding_calls
    finally:
        adapter.close()


def test_conflicting_supported_property_and_loaded_argv_is_typed_failure():
    client = ContractClient(native_batch=2048)
    original = client.props
    client.props = lambda **kwargs: {**original(**kwargs), "n_batch": 64}
    adapter = configured(client)
    try:
        with pytest.raises(LLMBackendError, match="ambiguous_loaded_batch"):
            adapter.prepare_query("test")
        assert not client.embedding_calls
    finally:
        adapter.close()


def test_option_looking_model_value_is_not_a_batch_declaration():
    client = ContractClient()
    for entry in client.catalog:
        entry["status"]["args"] = [
            "llama-server",
            "--model",
            "--batch-size",
            "--batch-size",
            "64",
        ]
    adapter = configured(client)
    try:
        assert adapter.prepare_query("test").batch_token_limit is None
        assert adapter._primary.loaded_batch_token_limit == 64
    finally:
        adapter.close()


def test_fully_unknown_limits_remain_unknown_on_public_preparation():
    client = ContractClient(context=None, native_batch=None)
    adapter = configured(client)
    try:
        value = adapter.prepare_query("test")
        assert value.context_token_limit is None
        assert value.batch_token_limit is None
        assert value.max_input_tokens is None
        assert value.fits_context is value.fits_request is value.fits_input is None
    finally:
        adapter.close()


@pytest.mark.parametrize("input_type", [None, 1, True, "invalid"])
def test_public_preparation_rejects_invalid_task_before_io(input_type):
    client = ContractClient()
    adapter = configured(client)
    try:
        with pytest.raises(ValueError):
            adapter.prepare_texts(["test"], input_type=input_type)
        assert (
            not client.token_calls
            and not client.props_calls
            and not client.embedding_calls
        )
    finally:
        adapter.close()


def test_extremely_large_native_batch_decimal_is_a_typed_metadata_failure():
    client = ContractClient()
    for entry in client.catalog:
        entry["status"]["args"] = ["llama-server", "--batch-size", "9" * 5000]
    adapter = configured(client)
    try:
        with pytest.raises(LLMBackendError, match="batch"):
            adapter.prepare_query("test")
        assert not client.embedding_calls
    finally:
        adapter.close()


def test_verified_identity_with_unknown_tokenizer_build_never_reuses_count():
    client = ContractClient()
    client.builds = dict.fromkeys(client.builds, None)
    client.counts["original"] = 8
    adapter = configured(client)
    try:
        prepared = adapter.prepare_query("original")
        assert prepared.identity_verified and prepared.token_count == 8
        initial = len(client.token_calls)
        client.counts["original"] = 2049
        with pytest.raises(LLMBackendError, match="exceeds_context|exceeds_batch"):
            adapter.embed_prepared([prepared])
        assert len(client.token_calls) == initial + 1
        assert not client.embedding_calls
    finally:
        adapter.close()
