from concurrent.futures import ThreadPoolExecutor
import math
from threading import Event, Lock
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from runtime_llama.adapters.embedding import LlamaCppEmbeddingAdapter
from runtime_llama.client import ClientHTTPError, LlamaCppClient
from runtime_llama.configuration import AdapterProfile
from runtime_llama.errors import LLMBackendError, LLMTimeoutError


def profile(**changes):
    values = dict(
        adapter_type="embedding",
        role_id="documents",
        model="embed-model",
        base_url="https://remote.example/vpn",
        timeout=7.0,
        model_config=SimpleNamespace(
            context_length=None, verify_ssl=True, instance_count=1
        ),
        generation=SimpleNamespace(api_key=None, api_key_env=None),
        batch_size=32,
        # These tests exercise full-batch validation/recovery; zero-delay
        # progressive dispatch is covered separately by the queue-engine tests.
        embedding_batch_wait_sec=0.5,
        batch_token_limit=None,
        token_count_parallelism=1,
        normalize=False,
        # The default fake server returns two dimensions. Tests of cold
        # dimension discovery opt out explicitly; full-batch contract tests
        # should not accidentally exercise the one-row bootstrap instead.
        dimension=2,
        max_input_tokens=None,
        embedding_request_parallelism=1,
        embedding_queue_size=16,
        embedding_replicas=(),
    )
    values.update(changes)
    return SimpleNamespace(**values)


def payload(vectors, indexes=None):
    return {
        "data": [
            {"index": index, "embedding": vector}
            for index, vector in zip(
                range(len(vectors)) if indexes is None else indexes,
                vectors,
            )
        ]
    }


class Client:
    def __init__(self, *, responses=None, context=32, counts=None, total_slots=1):
        self.responses = list(responses or [])
        self.context = context
        self.total_slots = total_slots
        self.counts = counts or {}
        self.embedding_calls = []
        self.tokenize_calls = []
        self.props_calls = []
        self.events = []
        self.active = self.max_active = 0
        self.lock = Lock()
        self.delay = 0
        self.closed = False

    def props(self, model=None, **kwargs):
        self.props_calls.append((model, kwargs))
        if isinstance(self.context, Exception):
            raise self.context
        return {
            "default_generation_settings": {"n_ctx": self.context},
            "n_ctx_train": 99999,
            "total_slots": self.total_slots,
        }

    def tokenize(self, content, **kwargs):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.tokenize_calls.append((content, kwargs))
            self.events.append(("tokenize", content))
        try:
            if self.delay:
                time.sleep(self.delay)
            return {
                "tokens": list(
                    range(self.counts.get(content, len(content.split()) + 1))
                )
            }
        finally:
            with self.lock:
                self.active -= 1

    def embeddings(self, input, model=None, **kwargs):
        self.embedding_calls.append((list(input), model, kwargs))
        self.events.append(("embedding", list(input)))
        if self.responses:
            result = self.responses.pop(0)
            if isinstance(result, Exception):
                raise result
            if callable(result):
                return result(input)
            return result
        return payload([[float(len(text)), 1.0] for text in input])

    def health(self, **kwargs):
        return {"status": "ok"}

    def models(self, **kwargs):
        return {"data": [{"id": "embed-model", "status": {"value": "loaded"}}]}

    def close(self):
        self.closed = True


def test_batch_order_and_tokenizer_match_server_embedding_options():
    client = Client(responses=[payload([[3, 4], [1, 0]], indexes=[1, 0])])
    adapter = LlamaCppEmbeddingAdapter(profile(normalize=True), client=client)
    assert adapter.embed_texts(["first", "second"]) == [[1, 0], [0.6, 0.8]]
    assert adapter.embedding_dim == 2
    assert client.embedding_calls[0][0:2] == (["first", "second"], "embed-model")
    for _, options in client.tokenize_calls:
        assert options == dict(
            model="embed-model",
            add_special=True,
            parse_special=True,
            with_pieces=False,
            timeout=options["timeout"],
            cancel=options["cancel"],
            cancel_wait=True,
        )
        assert 0 < options["timeout"] <= 7.0
        assert callable(options["cancel"])
    assert "autoload" not in client.embedding_calls[0][2]


def test_empty_inputs_do_not_contact_server():
    client = Client()
    adapter = LlamaCppEmbeddingAdapter(profile(), client=client)
    assert adapter.embed_texts([]) == []
    assert adapter.embed_query("") == []
    assert (
        not client.props_calls
        and not client.tokenize_calls
        and not client.embedding_calls
    )


def test_preflight_and_embedding_are_progressive_by_item_batch():
    client = Client()
    adapter = LlamaCppEmbeddingAdapter(
        profile(
            batch_size=2,
            embedding_queue_size=1,
            embedding_max_pending_tokens=8,
        ),
        client=client,
    )
    incoming = ["one", "two", "three", "four", "five", "six"]
    first_started, second_prepared, release_first = Event(), Event(), Event()
    tokenize, embeddings = client.tokenize, client.embeddings

    def count(content, **kwargs):
        if content == "four":
            second_prepared.set()
        return tokenize(content, **kwargs)

    def request(input, **kwargs):
        # Persistent queues may combine ready records from internal windows;
        # the first HTTP, rather than one specific composition, owns this gate.
        if not first_started.is_set():
            first_started.set()
            assert release_first.wait(3)
        return embeddings(input, **kwargs)

    client.tokenize, client.embeddings = count, request
    try:
        with ThreadPoolExecutor(max_workers=1) as callers:
            pending = callers.submit(adapter.embed_texts, incoming)
            try:
                assert first_started.wait(2)
                assert second_prepared.wait(2)
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    pressure = adapter._engine.snapshot()
                    if (
                        pressure["pending_tokens"] == 8
                        and pressure["embedding_in_flight"] == 1
                        and pressure["preparation_results_waiting"] > 0
                    ):
                        break
                    time.sleep(0.001)
                else:
                    pytest.fail("prepared records did not reach the token budget")
                assert not pending.done()
                # Preparation admission may retire before HTTP drain. The
                # explicit token budget, including ready and active records,
                # bounds retained work; one worker handoff can await credit.
                assert pressure["peak_pending_tokens"] <= 8
                assert sum(pressure["ready_by_replica"].values()) == 2
                assert len(client.tokenize_calls) < len(incoming)
            finally:
                release_first.set()
            assert pending.result(timeout=3) == [
                [float(len(text)), 1.0] for text in incoming
            ]
        assert [len(call[0]) for call in client.embedding_calls] == [2, 2, 2]
        assert sorted(
            text for batch, _, _ in client.embedding_calls for text in batch
        ) == sorted(incoming)
        assert sorted(text for text, _ in client.tokenize_calls) == sorted(incoming)
        for position, (kind, batch) in enumerate(client.events):
            if kind == "embedding":
                assert all(
                    ("tokenize", text) in client.events[:position] for text in batch
                )
    finally:
        release_first.set()
        adapter.close()


def test_exact_counts_pack_http_requests_and_reject_indivisible_overbudget():
    client = Client(counts={"a": 2, "b": 1, "c": 2, "long": 6})
    adapter = LlamaCppEmbeddingAdapter(
        profile(batch_size=4, batch_token_limit=3), client=client
    )
    try:
        adapter.embed_texts(["a", "b", "c"])
        assert [call[0] for call in client.embedding_calls] == [["a", "b"], ["c"]]
        before = len(client.embedding_calls)
        with pytest.raises(LLMBackendError, match="input_exceeds_batch_budget"):
            adapter.embed_texts(["a", "long"])
        assert len(client.embedding_calls) == before
        assert adapter.embed_query("b")
    finally:
        adapter.close()


def test_context_is_minimum_of_observed_and_configured_limits():
    client = Client(context=4, counts={"too long": 5})
    adapter = LlamaCppEmbeddingAdapter(
        profile(
            max_input_tokens=9,
            model_config=SimpleNamespace(context_length=7, verify_ssl=True),
        ),
        client=client,
    )
    assert adapter.max_input_tokens == 4
    with pytest.raises(LLMBackendError, match="input_exceeds_context"):
        adapter.embed_query("too long")
    assert not client.embedding_calls


def test_unknown_loaded_context_stays_unknown_and_refreshes_after_server_change():
    client = Client(context=None)
    adapter = LlamaCppEmbeddingAdapter(profile(), client=client)
    assert adapter.max_input_tokens is None
    adapter.embed_query("a")
    assert adapter.health_status()["loaded_context_length"] is None
    client.context = 2
    with pytest.raises(LLMBackendError, match="input_exceeds_context"):
        adapter.embed_query("three word input")


def test_optional_props_failure_does_not_invent_metadata():
    client = Client(context=ClientHTTPError(404))
    adapter = LlamaCppEmbeddingAdapter(profile(max_input_tokens=8), client=client)
    assert adapter.max_input_tokens == 8
    assert adapter.embed_query("a")
    assert adapter.health_status()["loaded_context_length"] is None


@pytest.mark.parametrize("status", [401, 429, 500, 503])
def test_props_errors_are_not_mistaken_for_an_optional_missing_endpoint(status):
    client = Client(context=ClientHTTPError(status))
    adapter = LlamaCppEmbeddingAdapter(profile(max_input_tokens=8), client=client)
    with pytest.raises(ClientHTTPError):
        adapter.embed_query("a")
    assert not client.tokenize_calls and not client.embedding_calls
    adapter.close()


@pytest.mark.parametrize("configured_limit", [None, 8])
def test_timed_out_props_refresh_invalidates_observed_context(configured_limit):
    client = Client(context=4)
    adapter = LlamaCppEmbeddingAdapter(
        profile(max_input_tokens=configured_limit), client=client
    )
    assert adapter.max_input_tokens == 4

    client.context = LLMTimeoutError("test props timeout")
    status = adapter.health_status()
    assert status["loaded_context_length"] is None
    assert status["max_input_tokens"] == configured_limit
    assert adapter.max_input_tokens == configured_limit
    with pytest.raises(LLMTimeoutError):
        adapter.embed_query("a")
    assert not client.embedding_calls


def test_tokenizer_preflight_has_bounded_parallelism():
    client = Client()
    client.delay = 0.02
    adapter = LlamaCppEmbeddingAdapter(
        profile(token_count_parallelism=3), client=client
    )
    assert len(adapter.embed_texts(["a", "b", "c", "d"])) == 4
    assert client.max_active == 3


def test_concurrent_metadata_refresh_cannot_remove_another_calls_observed_limit():
    client = Client(context=4, counts={"long": 5})
    counting_started, continue_counting = Event(), Event()
    tokenize = client.tokenize

    def controlled_tokenize(content, **kwargs):
        if content == "long":
            counting_started.set()
            assert continue_counting.wait(3)
        return tokenize(content, **kwargs)

    client.tokenize = controlled_tokenize
    adapter = LlamaCppEmbeddingAdapter(
        profile(embedding_request_parallelism=2, token_count_parallelism=2),
        client=client,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(adapter.embed_query, "long")
        assert counting_started.wait(3)
        client.context = None
        assert adapter.embed_query("short")
        continue_counting.set()
        with pytest.raises(LLMBackendError, match="input_exceeds_context"):
            pending.result(timeout=3)
    assert [call[0] for call in client.embedding_calls] == [["short"]]


@pytest.mark.parametrize("tokens", [None, "123", [True], [-1], [0.5], [{"id": 1}]])
def test_invalid_tokenization_never_reaches_embedding(tokens):
    client = Client()
    client.tokenize = lambda *args, **kwargs: {"tokens": tokens}
    adapter = LlamaCppEmbeddingAdapter(profile(), client=client)
    with pytest.raises(LLMBackendError, match="invalid_tokenization"):
        adapter.embed_query("private input")
    assert not client.embedding_calls


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [True],
        ["1"],
        [None],
        [float("nan")],
        [float("inf")],
        [-float("inf")],
        [[1.0]],
        [10**1000],
    ],
)
def test_vectors_must_be_nonempty_finite_numeric_arrays(vectors):
    client = Client(responses=[payload([vectors])])
    adapter = LlamaCppEmbeddingAdapter(profile(dimension=None), client=client)
    with pytest.raises(LLMBackendError):
        adapter.embed_query("private input")
    assert adapter.embedding_dim == 0


@pytest.mark.parametrize("indexes", [[0, 0], [0, 2], [True, 1], [0.0, 1], ["0", 1]])
def test_indexes_are_unique_in_range_integers(indexes):
    client = Client(responses=[payload([[1], [2]], indexes=indexes)])
    # Declare the one-dimensional fixture so both rows reach native index
    # validation in one POST; an unknown width would first require a seed.
    adapter = LlamaCppEmbeddingAdapter(profile(dimension=1), client=client)
    with pytest.raises(LLMBackendError):
        adapter.embed_texts(["a", "b"])
    assert adapter.embedding_dim == 1


def test_dimension_learning_is_atomic_across_a_failing_call():
    client = Client(
        responses=[
            payload([[1, 2]]),
            payload([[1, float("nan")]]),
            payload([[1, 2, 3]]),
        ]
    )
    adapter = LlamaCppEmbeddingAdapter(
        profile(batch_size=1, dimension=None), client=client
    )
    with pytest.raises(LLMBackendError):
        adapter.embed_texts(["a", "b"])
    assert adapter.embedding_dim == 0
    assert adapter.embed_query("c") == [1, 2, 3]
    assert adapter.embedding_dim == 3


def test_configured_dimension_and_batch_dimensions_are_enforced():
    client = Client(responses=[payload([[1, 2, 3]])])
    adapter = LlamaCppEmbeddingAdapter(profile(dimension=2), client=client)
    with pytest.raises(LLMBackendError, match="dimension_mismatch"):
        adapter.embed_query("a")
    assert adapter.embedding_dim == 2
    client = Client(responses=[payload([[1], [1, 2]])])
    with pytest.raises(LLMBackendError, match="dimension_mismatch"):
        LlamaCppEmbeddingAdapter(profile(dimension=1), client=client).embed_texts(
            ["a", "b"]
        )


def test_normalization_handles_extreme_finite_values_and_zero_vectors():
    client = Client(responses=[payload([[1e308, 1e308], [0, 0]])])
    result = LlamaCppEmbeddingAdapter(
        profile(normalize=True), client=client
    ).embed_texts(["a", "b"])
    assert all(math.isfinite(v) for v in result[0])
    assert sum(v * v for v in result[0]) == pytest.approx(1.0)
    assert result[1] == [0, 0]


def test_context_capacity_error_splits_and_remembers_bounded_batch_size():
    client = Client(
        responses=[ClientHTTPError(400, error_type="exceed_context_size_error")]
    )
    adapter = LlamaCppEmbeddingAdapter(profile(batch_size=4), client=client)
    assert len(adapter.embed_texts(["a", "b", "c", "d"])) == 4
    assert [len(call[0]) for call in client.embedding_calls] == [4, 2, 2]
    assert adapter.health_status()["effective_batch_size"] == 2
    adapter.embed_texts(["a", "b", "c", "d"])
    assert [len(call[0]) for call in client.embedding_calls[-2:]] == [2, 2]


@pytest.mark.parametrize(
    "status,error_type",
    [
        (400, "invalid_request_error"),
        (401, None),
        (404, None),
        (500, "server_error"),
        (503, None),
    ],
)
def test_unqualified_http_errors_are_not_retried_or_split(status, error_type):
    client = Client(responses=[ClientHTTPError(status, error_type=error_type)])
    with pytest.raises(ClientHTTPError):
        LlamaCppEmbeddingAdapter(profile(), client=client).embed_texts(["a", "b"])
    assert len(client.embedding_calls) == 1


def test_single_input_context_error_is_not_truncated_or_retried():
    client = Client(
        responses=[ClientHTTPError(400, error_type="exceed_context_size_error")]
    )
    with pytest.raises(ClientHTTPError):
        LlamaCppEmbeddingAdapter(profile(), client=client).embed_query(
            "keep full input"
        )
    assert [call[0] for call in client.embedding_calls] == [["keep full input"]]


def test_adaptive_children_do_not_count_as_capacity_recovery_successes():
    client = Client(
        responses=[ClientHTTPError(400, error_type="exceed_context_size_error")]
    )
    adapter = LlamaCppEmbeddingAdapter(profile(batch_size=4), client=client)
    adapter.embed_texts(["a", "b", "c", "d"])
    assert adapter.health_status()["batch_recovery"]["successes"] == 0
    for _ in range(32):
        adapter.embed_texts(["a", "b"])
    assert adapter.health_status()["effective_batch_size"] == 4


def test_flapping_capacity_requires_longer_stability_before_each_recovery():
    client = Client()
    adapter = LlamaCppEmbeddingAdapter(profile(batch_size=4), client=client)
    texts = ["first", "second", "third", "fourth"]

    for required_successes in (32, 64, 128):
        client.responses.append(
            ClientHTTPError(400, error_type="exceed_context_size_error")
        )
        assert len(adapter.embed_texts(texts)) == 4
        reduced = adapter.health_status()
        assert reduced["effective_batch_size"] == 2
        assert reduced["batch_recovery"]["successes"] == 0
        assert reduced["batch_recovery"]["success_window"] == required_successes

        # A flapping backend must not immediately receive another large batch.
        start = len(client.embedding_calls)
        for _ in range(required_successes - 1):
            assert len(adapter.embed_texts(texts[:2])) == 2
        assert adapter.health_status()["effective_batch_size"] == 2
        assert all(len(call[0]) == 2 for call in client.embedding_calls[start:])

        assert len(adapter.embed_texts(texts[:2])) == 2
        assert adapter.health_status()["effective_batch_size"] == 4
        assert len(adapter.embed_texts(texts)) == 4
        assert client.embedding_calls[-1][0] == texts


def test_success_started_before_capacity_reduction_cannot_restore_capacity_early():
    client = Client(
        responses=[ClientHTTPError(400, error_type="exceed_context_size_error")],
        total_slots=2,
    )
    adapter = LlamaCppEmbeddingAdapter(
        profile(batch_size=8, embedding_request_parallelism=2), client=client
    )
    assert len(adapter.embed_texts([f"initial-{index}" for index in range(8)])) == 8
    assert adapter.health_status()["effective_batch_size"] == 4
    assert adapter._primary.total_slots == 2
    assert adapter._scheduler.snapshot()["effective_parallelism"] == 1
    assert (
        adapter._scheduler.snapshot()["replicas"]["embed-model"]["max_in_flight"] == 1
    )

    old_texts = [f"old-{index}" for index in range(4)]
    old_started, release_old = Event(), Event()
    embeddings = client.embeddings

    def controlled_embeddings(input, **kwargs):
        if input == old_texts:
            old_started.set()
            assert release_old.wait(10)
        return embeddings(input, **kwargs)

    client.embeddings = controlled_embeddings
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(adapter.embed_texts, old_texts)
        try:
            assert old_started.wait(3)
            # One original lease per replica prevents a second concurrent
            # POST, even though native props advertise two sequence slots.
            # Inject the adaptive-generation transition directly to retain
            # the stale-success regression without inventing HTTP capacity.
            adapter._tighten_adaptive_batch_size(2, replica=adapter._primary)
            assert adapter.health_status()["effective_batch_size"] == 2
            release_old.set()
            assert len(pending.result(timeout=3)) == 4
            status = adapter.health_status()
            assert status["effective_batch_size"] == 2
            assert status["batch_recovery"]["successes"] == 0

            # Complete all but one of the new stability window's successes.
            for _ in range(63):
                assert len(adapter.embed_texts(["stable-a", "stable-b"])) == 2
            assert adapter.health_status()["batch_recovery"]["successes"] == 63

            assert len(adapter.embed_texts(["stable-a", "stable-b"])) == 2
            assert adapter.health_status()["effective_batch_size"] == 4
        finally:
            release_old.set()


def test_timeout_and_cancel_stay_request_local_and_load_is_readiness_only():
    client = Client()

    def cancel():
        return False

    adapter = LlamaCppEmbeddingAdapter(profile(), client=client)
    assert adapter.load()
    adapter.embed_query("a", timeout=2.5, cancel=cancel)
    options = client.embedding_calls[0][2]
    assert options == {
        "timeout": options["timeout"],
        "cancel": options["cancel"],
        "cancel_wait": True,
    }
    assert 0 < options["timeout"] <= 2.5
    assert callable(options["cancel"])
    assert options["cancel"]() is False
    assert callable(client.tokenize_calls[0][1]["cancel"])
    assert 0 < client.props_calls[-1][1]["timeout"] <= 2.5
    adapter.close()
    assert not adapter.is_ready()
    with pytest.raises(LLMBackendError, match="closed"):
        adapter.embed_query("a")
    assert not client.closed  # injected client remains caller-owned


def test_multiple_client_owned_instances_are_rejected():
    p = profile(
        model_config=SimpleNamespace(
            instance_count=2, context_length=None, verify_ssl=True
        )
    )
    with pytest.raises(ValueError, match="instance_count"):
        LlamaCppEmbeddingAdapter(p, client=Client())


def test_actual_profile_and_http_client_integrate_without_runtime_or_sdk():
    from io import BytesIO
    import json

    import requests
    from urllib3.response import HTTPResponse

    class Response(requests.Response):
        def __init__(self, data):
            super().__init__()
            body = json.dumps(data).encode("utf-8")
            self.status_code = 200
            self.encoding = "utf-8"
            self.headers.update(
                {"Content-Type": "application/json", "Content-Length": str(len(body))}
            )
            self.raw = HTTPResponse(
                body=BytesIO(body),
                headers=self.headers,
                preload_content=False,
                decode_content=False,
            )
            self.closed = False

        def close(self):
            try:
                super().close()
            finally:
                self.closed = True

    class Session:
        def __init__(self):
            self.calls, self.responses = [], []

        def request(self, **kwargs):
            self.calls.append(kwargs)
            path = urlsplit(kwargs["url"]).path
            if path.endswith("/v1/models"):
                result = {
                    "data": [{"id": "embed-model", "status": {"value": "loaded"}}]
                }
            elif path.endswith("/props"):
                result = {"default_generation_settings": {"n_ctx": 4}}
            elif path.endswith("/tokenize"):
                assert kwargs["json"]["add_special"] is True
                assert kwargs["json"]["parse_special"] is True
                assert kwargs["json"]["model"] == "embed-model"
                result = {"tokens": [1, 2, 3]}
            elif path.endswith("/v1/embeddings"):
                assert kwargs["json"] == {
                    "input": ["one", "two"],
                    "model": "embed-model",
                }
                result = payload([[3, 4], [0, 1]], indexes=[1, 0])
            else:
                raise AssertionError(path)
            response = Response(result)
            self.responses.append(response)
            return response

    configured = AdapterProfile.from_config(
        "documents",
        {
            "adapter_type": "embedding",
            "model": "embed-model",
            "base_url": "https://remote.example/vpn",
            "timeout": 2.0,
            "normalize": True,
            "dimension": 2,
            "model_context_length": 12,
            "max_input_tokens": 9,
            "embedding_batch_wait_sec": 0.5,
        },
    )
    session = Session()
    client = LlamaCppClient(configured.base_url, session=session)
    adapter = LlamaCppEmbeddingAdapter(configured, client=client)
    assert adapter.batch_size == 32
    assert adapter.embed_texts(["one", "two"]) == [[0, 1], [0.6, 0.8]]
    assert adapter.max_input_tokens == 4
    assert all(
        call["verify"] is True and 0 < call["timeout"] <= 2.0 for call in session.calls
    )
    assert all("autoload" not in call["url"] for call in session.calls)
    assert all(response.closed for response in session.responses)
