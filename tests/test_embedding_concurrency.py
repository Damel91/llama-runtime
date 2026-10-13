"""Deterministic qualification against real local HTTP connections."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Event, Lock, Thread, enumerate as threads
import time

import pytest

from runtime_llama.adapters.embedding import LlamaCppEmbeddingAdapter
from runtime_llama.client import ClientHTTPError, LlamaCppClient
from runtime_llama.configuration import AdapterProfile
from runtime_llama.errors import LLMBackendError, LLMCancelledError


def eventually(check, timeout=3):
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() >= deadline:
            raise AssertionError("local fixture condition not reached")
        time.sleep(0.005)


def vectors(texts):
    return [[float(text.rsplit("-", 1)[-1]), 1.0] for text in texts]


def response(texts):
    return {
        "data": list(
            reversed(
                [
                    {"index": index, "embedding": vector}
                    for index, vector in enumerate(vectors(texts))
                ]
            )
        )
    }


class State:
    def __init__(self):
        self.lock = Lock()
        self.release = Event()
        self.started = Event()
        self.active = self.peak = 0
        self.by_model = Counter()
        self.peak_by_model = Counter()
        self.calls = []
        self.tokenize_calls = []
        self.props_calls = []
        self.contexts = {"embed": 64, "embed_1": 64}
        self.props_overrides = {}
        self.health_code = 200
        self.catalog = [
            {"id": model, "status": {"value": "loaded"}} for model in self.contexts
        ]
        self.token_counts = {}
        self.action = lambda body: (200, response(body["input"]))

    def embedding(self, body):
        model = body["model"]
        with self.lock:
            self.active += 1
            self.by_model[model] += 1
            self.peak = max(self.peak, self.active)
            self.peak_by_model[model] = max(
                self.peak_by_model[model], self.by_model[model]
            )
            self.calls.append((model, tuple(body["input"])))
            self.started.set()
        try:
            return self.action(body)
        finally:
            with self.lock:
                self.active -= 1
                self.by_model[model] -= 1


@contextmanager
def server():
    state = State()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def send_json(self, status, body):
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            try:
                self.wfile.write(encoded)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            from urllib.parse import parse_qs, urlsplit

            path = urlsplit(self.path)
            if path.path == "/health":
                body = {"status": "ok"}
                self.send_json(state.health_code, body)
                return
            elif path.path == "/v1/models":
                body = {"data": state.catalog}
            elif path.path == "/props":
                model = parse_qs(path.query)["model"][0]
                with state.lock:
                    state.props_calls.append(model)
                body = {
                    "model_alias": model,
                    "total_slots": 2,
                    "default_generation_settings": {"n_ctx": state.contexts[model]},
                }
                body.update(state.props_overrides.get(model, {}))
            else:
                raise AssertionError(path.path)
            self.send_json(200, body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                with state.lock:
                    state.tokenize_calls.append((body["model"], body["content"]))
                count = state.token_counts.get(body["content"], 2)
                self.send_json(200, {"tokens": list(range(count))})
            elif self.path == "/v1/embeddings":
                self.send_json(*state.embedding(body))
            else:
                raise AssertionError(self.path)

        def log_message(self, *_args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    http.daemon_threads = True
    serving = Thread(target=http.serve_forever, daemon=True)
    serving.start()
    try:
        yield state, f"http://127.0.0.1:{http.server_port}"
    finally:
        state.release.set()
        http.shutdown()
        http.server_close()
        serving.join(timeout=2)


def configured(base_url, **changes):
    values = {
        "adapter_type": "embedding",
        "model": "embed",
        "base_url": base_url,
        "timeout": 2,
        "batch_size": 2,
        "embedding_request_parallelism": 4,
        "embedding_queue_size": 2,
        "embedding_replicas": [
            {"model": "embed", "max_in_flight": 2},
            {"model": "embed_1", "max_in_flight": 2},
        ],
    }
    values.update(changes)
    return AdapterProfile.from_config("documents", values)


def test_real_http_four_in_flight_two_replicas_and_bounded_shared_queue():
    with server() as (state, url):

        def action(body):
            assert state.release.wait(3)
            # Complete both batches and individual vectors in different orders.
            time.sleep(0.002 * (int(body["input"][0].split("-")[-1]) % 4))
            return 200, response(body["input"])

        state.action = action
        adapter = LlamaCppEmbeddingAdapter(configured(url))
        first = [f"item-{index}" for index in range(16)]
        second = [f"item-{index}" for index in range(100, 116)]
        try:
            with ThreadPoolExecutor(max_workers=2) as callers:
                a = callers.submit(adapter.embed_texts, first)
                b = callers.submit(adapter.embed_texts, second)
                eventually(lambda: state.active == 4)
                eventually(lambda: adapter._scheduler.snapshot()["queued"] == 2)
                assert state.peak == 4
                assert state.peak_by_model == {"embed": 2, "embed_1": 2}
                state.release.set()
                assert a.result(timeout=5) == vectors(first)
                assert b.result(timeout=5) == vectors(second)
            status = adapter.health_status()
            assert status["scheduler"]["peak_in_flight"] == 4
            assert status["scheduler"]["peak_queued"] == 2
            assert status["scheduler"]["queued"] == 0
            assert status["scheduler"]["in_flight"] == 0
            assert status["model_ready"] and status["server_healthy"]
            observed = [text for _, batch in state.calls for text in batch]
            assert Counter(observed) == Counter(first + second)
        finally:
            state.release.set()
            adapter.close()


def test_replica_specific_context_and_token_packing_use_the_selected_alias():
    with server() as (state, url):
        state.contexts = {"embed": 4, "embed_1": 8}
        state.token_counts = {f"item-{index}": 3 for index in range(8)}
        adapter = LlamaCppEmbeddingAdapter(
            configured(
                url,
                batch_size=4,
                embedding_request_parallelism=2,
                embedding_replicas=[
                    {"model": "embed", "max_in_flight": 1, "batch_token_limit": 4},
                    {"model": "embed_1", "max_in_flight": 1, "batch_token_limit": 6},
                ],
            )
        )
        try:
            texts = [f"item-{index}" for index in range(8)]
            assert adapter.embed_texts(texts) == vectors(texts)
            assert {model for model, _ in state.calls} == {"embed", "embed_1"}
            for model, batch in state.calls:
                assert len(batch) <= (1 if model == "embed" else 2)
                assert all((model, text) in state.tokenize_calls for text in batch)
            status = adapter.health_status()
            assert status["replicas"]["embed"]["max_input_tokens"] == 4
            assert status["replicas"]["embed_1"]["max_input_tokens"] == 8
            assert status["replicas"]["embed"]["observed_total_slots"] == 2
        finally:
            adapter.close()


def test_capacity_split_is_local_to_replica_and_does_not_replay_successful_batch():
    with server() as (state, url):

        def action(body):
            if body["model"] == "embed" and len(body["input"]) > 2:
                return 400, {"error": {"type": "exceed_context_size_error"}}
            return 200, response(body["input"])

        state.action = action
        # Total budget eight produces one four-text quota on each alias.
        adapter = LlamaCppEmbeddingAdapter(configured(url, batch_size=8))
        texts = [f"item-{index}" for index in range(8)]
        try:
            assert adapter.embed_texts(texts) == vectors(texts)
            assert Counter(state.calls) == {
                ("embed", tuple(texts[:4])): 1,
                ("embed", tuple(texts[:2])): 1,
                ("embed", tuple(texts[2:4])): 1,
                ("embed_1", tuple(texts[4:])): 1,
            }
            status = adapter.health_status()
            assert status["replicas"]["embed"]["effective_batch_size"] == 2
            assert status["replicas"]["embed_1"]["effective_batch_size"] == 8
        finally:
            adapter.close()


def test_real_http_failure_stops_unsent_batches_and_next_call_recovers():
    with server() as (state, url):

        def action(body):
            if body["input"] == ["item-0", "item-1"]:
                return 500, {"error": {"type": "server_error"}}
            return 200, response(body["input"])

        state.action = action
        adapter = LlamaCppEmbeddingAdapter(
            configured(url, embedding_request_parallelism=1)
        )
        try:
            with pytest.raises(ClientHTTPError):
                adapter.embed_texts([f"item-{index}" for index in range(30)])
            assert state.calls == [("embed", ("item-0", "item-1"))]
            assert adapter.embedding_dim == 0
            assert adapter.embed_query("item-99") == [99, 1]
        finally:
            adapter.close()


def test_cancellation_drains_one_call_without_cancelling_another_borrowed_client():
    with server() as (state, url):

        def action(body):
            if body["input"][0] == "item-0":
                assert state.release.wait(3)
            return 200, response(body["input"])

        state.action = action
        client = LlamaCppClient(url, timeout=2)
        adapter = LlamaCppEmbeddingAdapter(
            configured(url, batch_size=1, embedding_request_parallelism=2),
            client=client,
        )
        cancel = Event()
        try:
            with ThreadPoolExecutor(max_workers=2) as callers:
                stopped = callers.submit(
                    adapter.embed_texts,
                    [f"item-{index}" for index in range(16)],
                    cancel=cancel.is_set,
                )
                assert state.started.wait(2)
                cancel.set()
                other = callers.submit(adapter.embed_query, "item-99")
                assert other.result(timeout=3) == [99, 1]
                state.release.set()
                with pytest.raises(LLMCancelledError):
                    stopped.result(timeout=3)
            adapter.close()
            assert client.health() == {"status": "ok"}
            assert sum(batch == ("item-0",) for _, batch in state.calls) == 1
            assert not any("item-15" in batch for _, batch in state.calls)
        finally:
            state.release.set()
            adapter.close()
            client.close()


def test_cancelled_preheader_request_keeps_its_slot_and_zero_queue_backpressures():
    with server() as (state, url):

        def action(body):
            if body["input"] == ["item-0"]:
                assert state.release.wait(3)
            return 200, response(body["input"])

        state.action = action
        adapter = LlamaCppEmbeddingAdapter(
            configured(
                url,
                batch_size=1,
                embedding_request_parallelism=1,
                embedding_queue_size=0,
                embedding_replicas=[],
            )
        )
        cancel = Event()
        try:
            with ThreadPoolExecutor(max_workers=2) as callers:
                first = callers.submit(
                    adapter.embed_query, "item-0", cancel=cancel.is_set
                )
                assert state.started.wait(2)
                cancel.set()
                second = callers.submit(adapter.embed_query, "item-99")
                eventually(lambda: len(adapter._calls) == 2)
                time.sleep(0.08)
                assert not first.done() and not second.done()
                assert state.calls == [("embed", ("item-0",))]
                assert adapter._scheduler.snapshot()["queued"] == 0
                state.release.set()
                with pytest.raises(LLMCancelledError):
                    first.result(timeout=3)
                assert second.result(timeout=3) == [99, 1]
            assert state.peak == 1
            assert adapter._scheduler.snapshot()["peak_queued"] == 0
        finally:
            state.release.set()
            adapter.close()


def test_observed_replica_context_rejects_input_before_embedding_request():
    with server() as (state, url):
        state.contexts["embed"] = 2
        state.token_counts["item-0"] = 3
        adapter = LlamaCppEmbeddingAdapter(
            configured(url, embedding_replicas=[], max_input_tokens=8)
        )
        try:
            with pytest.raises(LLMBackendError, match="tokens=3:limit=2"):
                adapter.embed_query("item-0")
            assert state.calls == []
            assert adapter.embedding_dim == 0
        finally:
            adapter.close()


def test_close_waits_for_http_cleanup_and_leaves_no_adapter_workers():
    with server() as (state, url):

        def action(body):
            assert state.release.wait(3)
            return 200, response(body["input"])

        state.action = action
        adapter = LlamaCppEmbeddingAdapter(configured(url))
        existing = {thread.ident for thread in threads()}
        with ThreadPoolExecutor(max_workers=2) as callers:
            result = callers.submit(
                adapter.embed_texts, [f"item-{index}" for index in range(40)]
            )
            eventually(lambda: state.active == 4)
            closed = callers.submit(adapter.close)
            eventually(adapter._closing.is_set)
            # The close call retains its HTTP permits until the gated request ends.
            assert not closed.done()
            state.release.set()
            closed.result(timeout=4)
            with pytest.raises((LLMBackendError, LLMCancelledError)):
                result.result(timeout=4)
        assert not [
            thread
            for thread in threads()
            if thread.ident not in existing and thread.name.startswith("runtime_llama_")
        ]
        assert state.peak == 4
        assert len(state.calls) == 4
        adapter.close()
        with pytest.raises(LLMBackendError, match="closed"):
            adapter.embed_query("item-99")


@pytest.mark.parametrize("status", ["missing", "unloaded", "direct", "unknown"])
def test_readiness_requires_selected_model_availability_beyond_server_health(status):
    with server() as (state, url):
        if status == "missing":
            state.catalog = []
        elif status == "unloaded":
            state.catalog = [{"id": "embed", "status": {"value": "unloaded"}}]
        else:
            state.catalog = [{"id": "embed"}]
            if status == "unknown":
                state.props_overrides["embed"] = {"model_alias": None}
        adapter = LlamaCppEmbeddingAdapter(configured(url, embedding_replicas=[]))
        try:
            health = adapter.health_status()
            assert health["server_healthy"] is True
            assert health["model_ready"] is (status == "direct")
            assert adapter.load() is (status == "direct")
        finally:
            adapter.close()


@pytest.mark.parametrize("transition", ["unloaded", "missing", "server_down"])
def test_readiness_transition_invalidates_previous_loaded_metadata(transition):
    with server() as (state, url):
        adapter = LlamaCppEmbeddingAdapter(configured(url, embedding_replicas=[]))
        try:
            first = adapter.health_status()
            assert first["model_ready"]
            assert first["loaded_context_length"] == 64
            assert first["replicas"]["embed"]["observed_total_slots"] == 2
            props_calls = list(state.props_calls)
            if transition == "unloaded":
                state.catalog = [{"id": "embed", "status": {"value": "unloaded"}}]
            elif transition == "missing":
                state.catalog = []
            else:
                state.health_code = 503
            current = adapter.health_status()
            assert current["model_ready"] is False
            assert current["loaded_context_length"] is None
            assert current["max_input_tokens"] is None
            assert current["replicas"]["embed"]["observed_total_slots"] is None
            assert state.props_calls == props_calls
        finally:
            adapter.close()


@pytest.mark.parametrize("catalog_status", [True, False])
@pytest.mark.parametrize("problem", ["mismatch", "sleeping"])
def test_incoherent_or_sleeping_props_do_not_qualify_model(catalog_status, problem):
    with server() as (state, url):
        if not catalog_status:
            state.catalog = [{"id": "embed"}]
        state.props_overrides["embed"] = (
            {"model_alias": "wrong-model"}
            if problem == "mismatch"
            else {"is_sleeping": True}
        )
        adapter = LlamaCppEmbeddingAdapter(configured(url, embedding_replicas=[]))
        try:
            health = adapter.health_status()
            assert health["server_healthy"] is True
            assert health["model_ready"] is False
            assert health["loaded_context_length"] is None
            assert health["replicas"]["embed"]["observed_total_slots"] is None
            with pytest.raises(LLMBackendError):
                adapter.embed_query("item-0")
            assert state.calls == []
        finally:
            adapter.close()


def test_direct_server_canonical_model_and_configured_alias_are_verified_together():
    with server() as (state, url):
        state.catalog = [{"id": "canonical", "aliases": ["embed"]}]
        state.props_overrides["embed"] = {"model_alias": "canonical"}
        adapter = LlamaCppEmbeddingAdapter(configured(url, embedding_replicas=[]))
        try:
            # Embedding is valid even before an explicit readiness observation.
            assert adapter.embed_query("item-0") == [0, 1]
            assert adapter.is_ready()
            assert adapter.health_status()["model_ready"]
        finally:
            adapter.close()
