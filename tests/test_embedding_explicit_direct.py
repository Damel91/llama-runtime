"""One selected native target uses the existing engine without tokenization."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import gc
import json
from threading import Event, Thread
from urllib.parse import parse_qs, urlsplit
import weakref

import pytest

from runtime_llama import EmbeddingChunkSplitRequiredError, build_embedding_adapter
from runtime_llama.adapters.embedding import _DirectInput
from runtime_llama.client import ClientHTTPError
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError

from test_embedding_async_stream import assert_drained
from test_embedding_contracts import ContractClient, configured, eventually, vector


class NativeClient(ContractClient):
    """Content tags deliberately absent; only observed target metadata exists."""

    def __init__(self):
        super().__init__(1, verified=False)
        self.builds["embed_0"] = "b12345-aabbcc"
        self.health_value = {"status": "ok"}
        self.props_change = None
        self.catalog_change = None
        self.catalog_calls = 0
        self.metadata_gate = Event()
        self.metadata_gate.set()
        self.metadata_started = Event()

    def health(self, **kwargs):
        self.metadata_started.set()
        self._wait(
            self.metadata_gate,
            cancel=kwargs.get("cancel"),
            timeout=kwargs.get("timeout"),
        )
        return deepcopy(self.health_value)

    def models(self, **kwargs):
        self.catalog_calls += 1
        result = super().models(**kwargs)
        if self.catalog_change:
            result = self.catalog_change(self.catalog_calls, result)
        return result

    def props(self, model=None, **kwargs):
        result = super().props(model=model, **kwargs)
        if self.props_change:
            result = self.props_change(model, result)
        return result


@pytest.fixture
def make_adapter():
    adapters = []

    def create(client=None, **changes):
        client = client or NativeClient()
        values = {
            "model": "embed_0",
            "embedding_replica_discovery": False,
            "embedding_batch_wait_sec": 0,
        }
        values.update(changes)
        adapter = configured(client, **values)
        adapters.append(adapter)
        return adapter, client

    yield create
    for adapter in adapters:
        adapter.close()


def test_direct_511_uses_same_engine_with_no_identity_or_token_counts(
    make_adapter, monkeypatch
):
    adapter, client = make_adapter(batch_size=256, embedding_batch_wait_sec=0.5)
    engine, scheduler = adapter._engine, adapter._scheduler
    seen = []
    prepare = adapter._engine_prepare

    def observe(*args, **kwargs):
        record = prepare(*args, **kwargs)
        seen.append(record)
        return record

    def forbidden(*_args, **_kwargs):
        pytest.fail("direct strings cannot use tokenizer or preparation cache")

    monkeypatch.setattr(adapter, "_engine_prepare", observe)
    monkeypatch.setattr(adapter._preparation_cache, "get", forbidden)
    monkeypatch.setattr(adapter._preparation_cache, "put", forbidden)
    monkeypatch.setattr(client, "tokenize", forbidden)
    incoming = [f"item-{index}" for index in range(511)]
    assert adapter.embed_texts_direct(incoming) == [vector(text) for text in incoming]
    assert adapter._engine is engine and adapter._scheduler is scheduler
    assert sum(len(batch) for _, batch in client.embedding_calls) == 511
    assert {alias for alias, _ in client.embedding_calls} == {"embed_0"}
    assert all(len(batch) <= 256 for _, batch in client.embedding_calls)
    assert client.embed_peak == 1
    assert len(seen) == 511
    assert all(type(record) is _DirectInput for record in seen)
    assert all(record.identity_verified is False for record in seen)
    assert all(
        record.content_sha256 is record.embedding_sha256 is record._token_ids is None
        for record in seen
    )
    assert all(record.token_count == record.reserved_tokens == 0 for record in seen)
    assert adapter._pinned_identity is None and not adapter._uses_verified_pool
    assert_drained(adapter, client)


@pytest.mark.parametrize("task", ["document", "query", "raw"])
def test_prefix_once_duplicates_unicode_and_empty_text(make_adapter, task):
    adapter, client = make_adapter(
        embedding_prefixes={"query": "query: ", "document": "doc: "}
    )
    inputs = ["東京🙂", "same", "same", "", "doc: already present"]
    prefix = "" if task == "raw" else {"query": "query: ", "document": "doc: "}[task]
    assert adapter.embed_texts_direct(inputs, input_type=task) == [
        vector(prefix + text) for text in inputs
    ]
    assert Counter(
        text for _, batch in client.embedding_calls for text in batch
    ) == Counter(prefix + text for text in inputs)
    assert not client.token_calls


def test_exact_methods_remain_exact_before_and_after_direct(make_adapter):
    adapter, client = make_adapter()
    prepared = adapter.prepare_texts(["exact-before"])
    assert adapter.embed_prepared(prepared) == [vector("exact-before")]
    assert adapter.embed_texts_direct(["direct"]) == [vector("direct")]
    assert adapter.embed_texts(["exact-after"]) == [vector("exact-after")]
    assert [content for content, _ in client.token_calls] == [
        "exact-before",
        # Unknown weights do not authorize reusing the earlier preparation.
        "exact-before",
        "exact-after",
    ]
    assert_drained(adapter, client)


@pytest.mark.parametrize(
    "changes",
    [
        {"embedding_replica_discovery": True},
        {"embedding_replicas": [{"model": "embed_0"}, {"model": "embed_1"}]},
        {"embedding_input_format": "tokens"},
    ],
)
def test_explicit_path_never_disables_pool_or_token_policy(make_adapter, changes):
    adapter, client = make_adapter(**changes)
    with pytest.raises(
        LLMBackendError, match="single_target_required|text_input_required"
    ):
        adapter.embed_texts_direct(["source"])
    assert not client.token_calls and not client.embedding_calls
    assert client.catalog_calls == 0


@pytest.mark.parametrize("inputs", ["one", b"one", ["good", 123], [None]])
def test_invalid_inputs_fail_before_native_requests(make_adapter, inputs):
    adapter, client = make_adapter()
    with pytest.raises((ValueError, TypeError)):
        adapter.embed_texts_direct(inputs)
    assert client.catalog_calls == 0
    assert not client.token_calls and not client.embedding_calls


def test_empty_input_is_inert(make_adapter):
    adapter, client = make_adapter()
    assert adapter.embed_texts_direct([]) == []
    assert client.catalog_calls == 0 and not client.embedding_calls


@pytest.mark.parametrize(
    "mutation",
    [
        "unhealthy",
        "strata",
        "conflict",
        "unknown_backend",
        "sleeping",
        "wrong_props_model",
        "missing",
        "ambiguous",
        "unloaded",
        "failed",
        "not_embedding",
    ],
)
def test_incompatible_target_fails_before_embedding(make_adapter, mutation):
    client = NativeClient()
    if mutation == "unhealthy":
        client.health_value["status"] = "loading"
    elif mutation == "strata":
        client.health_value["service"] = "strata"
        client.builds["embed_0"] = "Strata 0.1.52"
    elif mutation == "conflict":
        client.health_value["service"] = "strata"
    elif mutation == "unknown_backend":
        client.builds["embed_0"] = "unknown custom server"
    elif mutation == "sleeping":
        client.props_change = lambda _, props: {**props, "is_sleeping": True}
    elif mutation == "wrong_props_model":
        client.props_change = lambda _, props: {**props, "model_alias": "other"}
    elif mutation == "missing":
        client.catalog[0]["id"] = "other"
    elif mutation == "ambiguous":
        client.catalog.append(deepcopy(client.catalog[0]))
    elif mutation == "unloaded":
        client.catalog[0]["status"]["value"] = "unloaded"
    elif mutation == "failed":
        client.catalog[0]["status"]["failed"] = True
    elif mutation == "not_embedding":
        client.catalog[0]["capabilities"] = {"embeddings": False}
    adapter, _ = make_adapter(client)
    with pytest.raises(LLMBackendError):
        adapter.embed_texts_direct(["must not infer"])
    assert not client.token_calls and not client.embedding_calls
    assert_drained(adapter, client)


def test_statusless_singleton_root_props_attests_without_model_select(make_adapter):
    client = NativeClient()
    del client.catalog[0]["status"]
    adapter, _ = make_adapter(client)
    assert adapter.embed_texts_direct(["source"]) == [vector("source")]
    assert client.props_calls and all(model is None for model in client.props_calls)
    assert not client.token_calls


def test_statusless_router_never_receives_a_model_selecting_props(make_adapter):
    client = NativeClient()
    del client.catalog[0]["status"]
    client.props_change = lambda _, props: {**props, "role": "router"}
    adapter, _ = make_adapter(client)
    with pytest.raises(LLMBackendError, match="target_unverified"):
        adapter.embed_texts_direct(["source"])
    assert client.props_calls and all(model is None for model in client.props_calls)
    assert not client.embedding_calls


def test_other_catalog_models_never_join_the_selected_lane(make_adapter):
    client = NativeClient()
    client.set_aliases(["embed_0", "embed_1"], verified=False)
    client.builds = {key: "b12345-aabbcc" for key in client.builds}
    # Root router properties identify no selected model. Selected properties are
    # allowed only because the catalog explicitly attests embed_0 as loaded.
    client.props_change = lambda model, props: (
        {**props, "role": "router", "model_alias": ""} if model is None else props
    )
    adapter, _ = make_adapter(client)
    assert adapter.embed_texts_direct(["one", "two"]) == [vector("one"), vector("two")]
    assert set(adapter._replicas) == {"embed_0"}
    assert {model for model, _ in client.embedding_calls} == {"embed_0"}
    assert not client.token_calls


@pytest.mark.parametrize("change", ["canonical", "build", "unloaded", "ambiguous"])
def test_fresh_dispatch_observation_rejects_target_changes(make_adapter, change):
    client = NativeClient()

    def changed(number, catalog):
        if number < 2:
            return catalog
        entry = catalog["data"][0]
        if change == "canonical":
            entry["id"] = "replacement"
            entry["aliases"] = ["embed_0"]
        elif change == "build":
            client.builds["embed_0"] = "b54321-ddeeff"
        elif change == "unloaded":
            entry["status"]["value"] = "unloaded"
        else:
            catalog["data"].append(deepcopy(entry))
        return catalog

    client.catalog_change = changed
    adapter, _ = make_adapter(client)
    with pytest.raises(LLMBackendError):
        adapter.embed_texts_direct(["source"])
    assert client.catalog_calls >= 2
    assert not client.token_calls and not client.embedding_calls
    assert_drained(adapter, client)


def test_halving_preserves_indexed_vectors_and_unverified_call_correlation(
    make_adapter,
):
    adapter, client = make_adapter(batch_size=256, embedding_batch_wait_sec=0.5)
    client.failure = lambda _model, batch: (
        ClientHTTPError(500) if "bad chunk" in batch else None
    )
    errors = []
    for _ in range(2):
        with pytest.raises(EmbeddingChunkSplitRequiredError) as caught:
            adapter.embed_texts_direct(["first", "bad chunk", "last"])
        error = caught.value
        errors.append(error)
        assert error.failed_input_indices == (1,)
        assert [(row.input_index, row.embedding) for row in error.completed] == [
            (0, tuple(vector("first"))),
            (2, tuple(vector("last"))),
        ]
        assert error.inputs[0].halves[0].char_end == len("bad chunk") // 2
        assert error.inputs[0].halves[1].char_end == len("bad chunk")
        assert error.representation_id.startswith("unverified-call:")
        assert len(error.representation_id.removeprefix("unverified-call:")) == 32
        assert not hasattr(error, "content_sha256")
        assert not hasattr(error, "embedding_sha256")
        assert error.original_status_code == 500
        assert_drained(adapter, client)
    assert errors[0].representation_id != errors[1].representation_id
    assert not client.token_calls


def test_direct_cancellation_does_not_retire_shared_post_survivor(make_adapter):
    adapter, client = make_adapter(batch_size=8, embedding_batch_wait_sec=0.5)
    client.metadata_gate.clear()
    client.embed_gate.clear()
    cancelled = Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            adapter.embed_texts_direct, ["owner-a"], cancel=cancelled.is_set
        )
        assert client.metadata_started.wait(2)
        second = pool.submit(adapter.embed_texts_direct, ["owner-b"])
        eventually(lambda: len(adapter._calls) == 2)
        client.metadata_gate.set()
        assert client.embed_started.wait(2)
        assert client.embedding_calls == [("embed_0", ("owner-a", "owner-b"))]
        cancelled.set()
        adapter._engine.wake()
        client.embed_gate.set()
        with pytest.raises(LLMCancelledError):
            first.result(3)
        assert second.result(3) == [vector("owner-b")]
    assert_drained(adapter, client)
    assert not client.token_calls


def test_exact_peer_remains_usable_when_explicit_native_proof_fails(make_adapter):
    client = NativeClient()
    client.builds["embed_0"] = "unidentified test server"
    client.metadata_gate.clear()
    adapter, _ = make_adapter(client)
    with ThreadPoolExecutor(max_workers=2) as pool:
        direct = pool.submit(adapter.embed_texts_direct, ["direct"])
        assert client.metadata_started.wait(2)
        exact = pool.submit(adapter.embed_texts, ["exact"])
        eventually(lambda: len(adapter._calls) == 2)
        client.metadata_gate.set()
        with pytest.raises(LLMBackendError, match="target_unverified"):
            direct.result(3)
        assert exact.result(3) == [vector("exact")]
    assert [text for text, _ in client.token_calls] == ["exact"]
    assert client.embedding_calls == [("embed_0", ("exact",))]
    assert_drained(adapter, client)


@pytest.mark.parametrize("phase", ["metadata", "embedding"])
def test_cancel_timeout_and_close_drain_owned_work(make_adapter, phase):
    for mode in ("cancel", "timeout", "close"):
        adapter, client = make_adapter()
        gate = client.metadata_gate if phase == "metadata" else client.embed_gate
        started = (
            client.metadata_started if phase == "metadata" else client.embed_started
        )
        gate.clear()
        cancel = Event()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(
                adapter.embed_texts_direct,
                ["source"],
                cancel=cancel.is_set,
                timeout=0.08 if mode == "timeout" else 2,
            )
            assert started.wait(2)
            if mode == "cancel":
                cancel.set()
                adapter._engine.wake()
            elif mode == "close":
                adapter.close()
            expected = {
                "cancel": LLMCancelledError,
                "timeout": LLMTimeoutError,
                "close": LLMBackendError,
            }[mode]
            with pytest.raises(expected):
                pending.result(3)
            gate.set()
        assert_drained(adapter, client)


def test_10000_texts_release_owned_records_without_new_quota(make_adapter, monkeypatch):
    adapter, client = make_adapter(batch_size=None, timeout=15)
    references = []
    prepare = adapter._engine_prepare

    def observed(*args, **kwargs):
        result = prepare(*args, **kwargs)
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(adapter, "_engine_prepare", observed)
    incoming = [f"source-{index}" for index in range(10000)]
    assert adapter.embed_texts_direct(incoming) == [vector(text) for text in incoming]
    assert len(references) == 10000
    snapshot = adapter._engine.snapshot()
    for key in (
        "pending_bytes",
        "retained_result_bytes",
        "result_reserved_bytes",
        "embedding_in_flight",
    ):
        assert snapshot[key] == 0
    assert adapter._intake.snapshot()["owners"] == 0
    assert not client.token_calls
    assert_drained(adapter, client)

    def collected():
        # Unreachable Future/callback cycles belong to Python's collector;
        # a retained live SDK root would keep these sentinels alive after GC.
        gc.collect()
        return all(reference() is None for reference in references)

    eventually(collected)


@pytest.fixture
def native_http():
    state = {"gets": [], "posts": [], "tokenize": 0, "active": 0, "closed": False}
    catalog = {"data": [{"id": "bare-native", "aliases": []}]}
    props = {
        "model_alias": "bare-native",
        "build_info": "b12345-aabbcc",
        "total_slots": 4,
        "default_generation_settings": {"n_ctx": 2048},
    }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def reply(self, value, status=200):
            wire = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(wire)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(wire)
            self.close_connection = True

        def do_GET(self):
            state["gets"].append(self.path)
            route = urlsplit(self.path)
            if route.path == "/health":
                self.reply({"status": "ok"})
            elif route.path == "/v1/models":
                self.reply(catalog)
            elif route.path == "/props":
                assert not parse_qs(route.query), (
                    "singleton props must not select/load a model"
                )
                self.reply(props)
            else:
                self.reply({"error": "unpermitted"}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["posts"].append((self.path, body))
            if self.path == "/tokenize":
                state["tokenize"] += 1
                self.reply({"error": "tokenizer forbidden"}, 500)
                return
            if self.path != "/v1/embeddings":
                self.reply({"error": "unpermitted"}, 404)
                return
            assert body["model"] == "bare-native"
            state["active"] += 1
            try:
                self.reply(
                    {
                        "data": [
                            {"index": index, "embedding": vector(text)}
                            for index, text in reversed(list(enumerate(body["input"])))
                        ]
                    }
                )
            finally:
                state["active"] -= 1

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01))
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        state["closed"] = not thread.is_alive()
        assert state["active"] == 0 and state["closed"]


def test_real_native_http_statusless_catalog_has_no_tokenizer(native_http):
    url, state = native_http
    adapter = build_embedding_adapter(
        {
            "model": "bare-native",
            "base_url": url,
            "normalize": False,
            "timeout": 3,
            "embedding_prefixes": {"document": "doc: "},
        }
    )
    try:
        result = adapter.embed_texts_direct(["one", "東京", "one"])
        assert result == [vector("doc: " + text) for text in ("one", "東京", "one")]
        assert state["tokenize"] == 0
        assert {path for path, _ in state["posts"]} == {"/v1/embeddings"}
        assert {urlsplit(path).path for path in state["gets"]} == {
            "/health",
            "/v1/models",
            "/props",
        }
        assert not adapter._calls
    finally:
        adapter.close()
    assert state["active"] == 0
