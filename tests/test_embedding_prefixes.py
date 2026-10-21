from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from runtime_llama.adapters.embedding import LlamaCppEmbeddingAdapter
from runtime_llama.client import ClientHTTPError
from runtime_llama.configuration import AdapterProfile
from runtime_llama.errors import LLMBackendError, LLMCancelledError


def configured(**changes):
    values = {
        "adapter_type": "embedding",
        "model": "embed-model",
        "base_url": "https://remote.example/vpn",
        "timeout": 7.0,
        "embedding_prefixes": {"query": "Q: ", "document": "D: "},
    }
    values.update(changes)
    return AdapterProfile.from_config("documents", values)


def vector(text):
    return [float(len(text)), float(sum(map(ord, text)))]


class RecordingClient:
    def __init__(self, *, context=4096, counts=None, responses=(), total_slots=1):
        self.context = context
        self.total_slots = total_slots
        self.counts = counts or {}
        self.responses = list(responses)
        self.props_calls = []
        self.tokenize_calls = []
        self.embedding_calls = []
        self.lock = Lock()

    def props(self, model=None, **kwargs):
        with self.lock:
            self.props_calls.append((model, kwargs))
        return {
            "default_generation_settings": {"n_ctx": self.context},
            "total_slots": self.total_slots,
        }

    def tokenize(self, content, **kwargs):
        with self.lock:
            self.tokenize_calls.append((content, kwargs))
        return {"tokens": list(range(self.counts.get(content, len(content))))}

    def embeddings(self, input, model=None, **kwargs):
        with self.lock:
            self.embedding_calls.append((list(input), model, kwargs))
            result = self.responses.pop(0) if self.responses else None
        if isinstance(result, Exception):
            raise result
        return {
            "data": [
                {"index": index, "embedding": vector(text)}
                for index, text in enumerate(input)
            ]
        }


@pytest.fixture
def make_adapter():
    adapters = []

    def create(client=None, **changes):
        client = client or RecordingClient()
        adapter = LlamaCppEmbeddingAdapter(configured(**changes), client=client)
        adapters.append(adapter)
        return adapter, client

    yield create
    for adapter in adapters:
        adapter.close()


def test_task_prefixes_match_tokenizer_and_embedding_inputs(make_adapter):
    adapter, client = make_adapter(
        embedding_prefixes={
            "query": "task: search result | query: ",
            "document": "title: none | text: ",
        }
    )
    query = "task: search result | query: Where is the API?"
    documents = ["title: none | text:  API docs\n", "title: none | text: "]
    queries = [
        "task: search result | query: first",
        "task: search result | query: second",
    ]

    assert adapter.embed_query("Where is the API?") == vector(query)
    assert adapter.embed_texts([" API docs\n", ""]) == [vector(t) for t in documents]
    assert adapter.embed_texts(["first", "second"], input_type="query") == [
        vector(t) for t in queries
    ]
    expected = [query, *documents, *queries]
    assert [content for content, _ in client.tokenize_calls] == expected
    assert [
        content for batch, _, _ in client.embedding_calls for content in batch
    ] == expected
    assert all(
        options["add_special"] and options["parse_special"]
        for _, options in client.tokenize_calls
    )


def test_prefix_concatenation_preserves_whitespace_without_inserting_separator(
    make_adapter,
):
    adapter, client = make_adapter(
        embedding_prefixes={"query": "query:", "document": " \n doc:\t"}
    )
    assert adapter.embed_query("  text\n") == vector("query:  text\n")
    assert adapter.embed_texts([" text \t"])[0] == vector(" \n doc:\t text \t")
    assert [batch for batch, _, _ in client.embedding_calls] == [
        ["query:  text\n"],
        [" \n doc:\t text \t"],
    ]


def test_raw_mode_is_explicit_and_does_not_guess_preformatted_inputs(make_adapter):
    adapter, client = make_adapter()
    prepared = "D: existing prefix"
    assert adapter.embed_texts([prepared], input_type="raw") == [vector(prepared)]
    assert adapter.embed_texts([prepared]) == [vector("D: " + prepared)]
    assert [batch for batch, _, _ in client.embedding_calls] == [
        [prepared],
        ["D: " + prepared],
    ]


def test_empty_defaults_preserve_existing_inputs_for_each_task(make_adapter):
    adapter, client = make_adapter(embedding_prefixes={})
    text = "  unchanged \n\t"
    for task in ("document", "query", "raw"):
        assert adapter.embed_texts([text], input_type=task) == [vector(text)]
    assert adapter.embed_query(text) == vector(text)
    assert all(batch == [text] for batch, _, _ in client.embedding_calls)


def test_each_model_profile_owns_its_task_prefixes(make_adapter):
    client = RecordingClient()
    nomic, _ = make_adapter(
        client,
        model="nomic",
        embedding_prefixes={
            "query": "search_query: ",
            "document": "search_document: ",
        },
    )
    gemma, _ = make_adapter(
        client,
        model="gemma",
        embedding_prefixes={
            "query": "task: search result | query: ",
            "document": "title: none | text: ",
        },
    )
    nomic.embed_query("same input")
    gemma.embed_query("same input")
    nomic.embed_texts(["same input"])
    gemma.embed_texts(["same input"])
    assert [(batch, model) for batch, model, _ in client.embedding_calls] == [
        (["search_query: same input"], "nomic"),
        (["task: search result | query: same input"], "gemma"),
        (["search_document: same input"], "nomic"),
        (["title: none | text: same input"], "gemma"),
    ]


def test_old_duck_typed_profiles_without_prefix_configuration_still_work():
    original = configured()
    legacy = SimpleNamespace(
        **{
            name: getattr(original, name)
            for name in (
                "adapter_type",
                "model_config",
                "role_id",
                "model",
                "base_url",
                "timeout",
                "generation",
                "batch_size",
                "batch_token_limit",
                "token_count_parallelism",
                "normalize",
                "dimension",
                "max_input_tokens",
                "embedding_request_parallelism",
                "embedding_queue_size",
                "embedding_replicas",
            )
        }
    )
    client = RecordingClient()
    adapter = LlamaCppEmbeddingAdapter(legacy, client=client)
    try:
        assert adapter.embed_query(" query ") == vector(" query ")
        assert adapter.embed_texts([" document "]) == [vector(" document ")]
    finally:
        adapter.close()


@pytest.mark.parametrize("input_type", [None, "queries", "Document", "", 1, [], {}])
@pytest.mark.parametrize("texts", [[], ["text"]])
def test_invalid_input_type_is_rejected_before_server_access(
    make_adapter, input_type, texts
):
    adapter, client = make_adapter()
    with pytest.raises(ValueError, match="input_type must be document, query or raw"):
        adapter.embed_texts(texts, input_type=input_type)
    assert (
        not client.props_calls
        and not client.tokenize_calls
        and not client.embedding_calls
    )


def test_empty_inputs_remain_empty_even_with_nonempty_prefixes(make_adapter):
    adapter, client = make_adapter()
    assert adapter.embed_query("") == []
    for task in ("document", "query", "raw"):
        assert adapter.embed_texts([], input_type=task) == []
    assert (
        not client.props_calls
        and not client.tokenize_calls
        and not client.embedding_calls
    )


def test_prefix_counts_toward_input_limit_and_raw_call_recovers(make_adapter):
    client = RecordingClient(context=4, counts={"x": 2, "Q: x": 5})
    adapter, _ = make_adapter(client)
    with pytest.raises(LLMBackendError, match="input_exceeds_context:tokens=5:limit=4"):
        adapter.embed_query("x")
    assert not client.embedding_calls
    assert adapter.embedding_dim == 0
    assert adapter.embed_texts(["x"], input_type="raw") == [vector("x")]
    assert [content for content, _ in client.tokenize_calls] == ["Q: x", "x"]


def test_batch_packing_counts_complete_prefixed_inputs(make_adapter):
    client = RecordingClient(counts={"a": 1, "b": 1, "D: a": 3, "D: b": 3})
    adapter, _ = make_adapter(client, batch_size=4, batch_token_limit=4)
    assert adapter.embed_texts(["a", "b"]) == [vector("D: a"), vector("D: b")]
    assert [batch for batch, _, _ in client.embedding_calls] == [["D: a"], ["D: b"]]
    assert [content for content, _ in client.tokenize_calls] == ["D: a", "D: b"]


def test_adaptive_capacity_split_does_not_apply_prefix_twice(make_adapter):
    client = RecordingClient(
        responses=[ClientHTTPError(400, error_type="exceed_context_size_error")]
    )
    adapter, _ = make_adapter(client, batch_size=4, embedding_batch_wait_sec=0.5)
    expected = ["D: a", "D: b", "D: c", "D: d"]
    assert adapter.embed_texts(["a", "b", "c", "d"]) == [vector(t) for t in expected]
    assert [batch for batch, _, _ in client.embedding_calls] == [
        expected,
        expected[:2],
        expected[2:],
    ]
    assert [content for content, _ in client.tokenize_calls] == expected
    assert adapter.embed_texts(["a", "b", "c", "d"]) == [vector(t) for t in expected]
    assert [batch for batch, _, _ in client.embedding_calls[-2:]] == [
        expected[:2],
        expected[2:],
    ]


def test_prefixes_are_shared_by_replicas_with_stable_result_order(make_adapter):
    class OrderedClient(RecordingClient):
        def __init__(self):
            super().__init__()
            self.first_started, self.release_first = Event(), Event()
            self.completions = []

        def embeddings(self, input, model=None, **kwargs):
            result = super().embeddings(input, model=model, **kwargs)
            if input == ["D: item-0"]:
                self.first_started.set()
                assert self.release_first.wait(3)
            elif input == ["D: item-1"]:
                assert self.first_started.wait(3)
                with self.lock:
                    self.completions.append("item-1")
                self.release_first.set()
                return result
            with self.lock:
                self.completions.append(input[0].removeprefix("D: "))
            return result

    client = OrderedClient()
    adapter, _ = make_adapter(
        client,
        model="embed-0",
        batch_size=1,
        embedding_request_parallelism=2,
        embedding_replicas=[
            {"model": "embed-0", "max_in_flight": 1},
            {"model": "embed-1", "max_in_flight": 1},
        ],
    )
    inputs = [f"item-{i}" for i in range(6)]
    assert adapter.embed_texts(inputs) == [vector("D: " + text) for text in inputs]
    assert client.completions.index("item-1") < client.completions.index("item-0")
    assert {model for _, model, _ in client.embedding_calls} == {"embed-0", "embed-1"}
    assert Counter(
        (text, options["model"]) for text, options in client.tokenize_calls
    ) == Counter(
        (text, model) for batch, model, _ in client.embedding_calls for text in batch
    )


def test_concurrent_query_cancellation_does_not_change_document_task(make_adapter):
    class ControlledClient(RecordingClient):
        def __init__(self):
            super().__init__(total_slots=2)
            self.query_started, self.release_query = Event(), Event()

        def embeddings(self, input, model=None, **kwargs):
            result = super().embeddings(input, model=model, **kwargs)
            if input == ["Q: hold"]:
                self.query_started.set()
                assert self.release_query.wait(3)
                if kwargs["cancel"]():
                    raise LLMCancelledError()
            return result

    client = ControlledClient()
    adapter, _ = make_adapter(client, embedding_request_parallelism=2)
    cancel = Event()
    with ThreadPoolExecutor(max_workers=2) as callers:
        query = callers.submit(
            adapter.embed_query, "hold", timeout=1.25, cancel=cancel.is_set
        )
        try:
            assert client.query_started.wait(3)
            cancel.set()
            document = callers.submit(adapter.embed_texts, ["body"], timeout=2.5)
            assert document.result(timeout=3) == [vector("D: body")]
        finally:
            client.release_query.set()
        with pytest.raises(LLMCancelledError):
            query.result(timeout=3)
    assert [batch for batch, _, _ in client.embedding_calls] == [
        ["Q: hold"],
        ["D: body"],
    ]
    for (_, _, options), bound in zip(client.embedding_calls, (1.25, 2.5)):
        assert 0 < options["timeout"] <= bound
    assert adapter.embed_query("recovered") == vector("Q: recovered")
