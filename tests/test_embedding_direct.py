"""Direct string intake shares the exact adapter's queues and ownership."""

import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from threading import Event

import pytest

from runtime_llama import PreparedEmbeddingInput
from runtime_llama.adapters.embedding import (
    _Call,
    _DirectEmbeddingPolicy,
    _DirectInput,
)
from runtime_llama.client import ClientHTTPError
from runtime_llama.embedding_recovery import _FailedOwnerPost
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError

from test_embedding_async_stream import WindowSource, assert_drained, run, wait_thread
from test_embedding_contracts import (
    CONTENT,
    SEMANTIC,
    ContractClient,
    configured,
    eventually,
    texts,
    vector,
)


def policy(adapter, *, input_type="document", **changes):
    fields = {
        "content_sha256": CONTENT,
        "embedding_sha256": SEMANTIC,
        "tokenizer_build": "test-build-1",
        "policy": adapter._policy(input_type),
        "prefix": (
            ""
            if input_type == "raw"
            else getattr(adapter._embedding_prefixes, input_type)
        ),
    }
    fields.update(changes)
    return _DirectEmbeddingPolicy(**fields)


@pytest.fixture
def make_adapter():
    adapters = []

    def create(client=None, **changes):
        client = client or ContractClient()
        changes.setdefault("embedding_batch_wait_sec", 0)
        adapter = configured(client, **changes)
        adapters.append(adapter)
        return adapter, client

    yield create
    for adapter in adapters:
        adapter.close()


def test_direct_511_texts_skip_tokenizer_and_preserve_balanced_order(make_adapter):
    adapter, client = make_adapter(embedding_batch_wait_sec=0.5)
    client.supports_embedding_token_ids = True
    incoming = texts(511)
    result = adapter._embed_texts_direct(incoming, policy=policy(adapter))
    assert result == [vector(text) for text in incoming]
    assert not client.token_calls
    assert sorted(len(batch) for _, batch in client.embedding_calls) == [
        127,
        128,
        128,
        128,
    ]
    assert len({model for model, _ in client.embedding_calls}) == 2
    assert all(client.alias_peak[model] <= 1 for model in client.alias_peak)
    status = adapter.health_status()
    assert status["engine"]["token_accounting"] == "may_include_context_reservations"
    assert status["engine"]["preparation_work_includes_direct_reservations"] is True
    assert_drained(adapter, client)


@pytest.mark.parametrize("task", ["document", "query", "raw"])
def test_prefix_once_unicode_and_repeated_occurrences(make_adapter, task):
    adapter, client = make_adapter(
        embedding_prefixes={"query": "query: ", "document": "doc: "},
    )
    chosen = policy(adapter, input_type=task)
    incoming = ["a", "東京🙂", "a", "doc: already in source", "", "e\u0301"]
    assert adapter._embed_texts_direct(incoming, policy=chosen, input_type=task) == [
        vector(chosen.prefix + text) for text in incoming
    ]
    assert Counter(
        text for _, batch in client.embedding_calls for text in batch
    ) == Counter(chosen.prefix + text for text in incoming)
    assert not client.token_calls


def test_exact_public_preparations_remain_exact_after_direct_call(make_adapter):
    adapter, client = make_adapter()
    assert adapter._embed_texts_direct(["auto"], policy=policy(adapter)) == [
        vector("auto")
    ]
    prepared = adapter.prepare_texts(["exact"])
    assert type(prepared[0]) is PreparedEmbeddingInput
    assert prepared[0].token_count == len("exact") + 1
    assert adapter.embed_prepared(prepared) == [vector("exact")]
    assert adapter.embed_texts(["ordinary"]) == [vector("ordinary")]
    assert [text for text, _ in client.token_calls] == ["exact", "ordinary"]
    with pytest.raises(LLMBackendError, match="invalid_preparation"):
        adapter.embed_prepared([object()])


def test_cold_direct_call_needs_no_calibration_or_tokenizer(make_adapter, monkeypatch):
    adapter, client = make_adapter(ContractClient(1))

    def forbidden(*_args, **_kwargs):
        pytest.fail("ordinary direct inference must never call the tokenizer")

    monkeypatch.setattr(client, "tokenize", forbidden)
    chosen = policy(adapter)
    assert set(chosen.__dataclass_fields__) == {
        "content_sha256",
        "embedding_sha256",
        "tokenizer_build",
        "policy",
        "prefix",
    }
    assert adapter._embed_texts_direct(["cold input"], policy=chosen) == [
        vector("cold input")
    ]
    assert client.embedding_calls == [("embed_0", ("cold input",))]


def test_direct_records_reserve_context_without_inspecting_source_length(
    make_adapter, monkeypatch
):
    adapter, client = make_adapter(embedding_prefixes={"document": "long prefix: "})
    seen = []
    prepare = adapter._engine_prepare

    def record(*args, **kwargs):
        result = prepare(*args, **kwargs)
        seen.append(result)
        return result

    monkeypatch.setattr(adapter, "_engine_prepare", record)
    chosen = policy(adapter)
    incoming = ["1234567", "", "x" * 10000]
    assert adapter._embed_texts_direct(incoming, policy=chosen) == [
        vector(chosen.prefix + text) for text in incoming
    ]
    assert [item.reserved_tokens for item in seen] == [2048, 2048, 2048]
    assert all(item.token_count == item.reserved_tokens for item in seen)
    assert all(type(item) is _DirectInput for item in seen)
    assert all(item._token_ids is None for item in seen)
    assert all(item._direct_policy is chosen for item in seen)
    assert not client.token_calls


@pytest.mark.parametrize(("budget", "expected"), [(4096, [2, 2]), (8192, [4])])
def test_context_reservations_conservatively_pack_explicit_http_budget(
    make_adapter, budget, expected
):
    adapter, client = make_adapter(
        ContractClient(1), batch_token_limit=budget, embedding_batch_wait_sec=0.5
    )
    incoming = ["x" * 12 + str(index) for index in range(4)]
    assert adapter._embed_texts_direct(incoming, policy=policy(adapter)) == [
        vector(text) for text in incoming
    ]
    assert sorted(len(batch) for _, batch in client.embedding_calls) == expected
    assert not client.token_calls


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"content_sha256": "c" * 64}, "direct_identity_changed"),
        ({"embedding_sha256": "d" * 64}, "direct_identity_changed"),
        ({"tokenizer_build": "another-build"}, "direct_tokenizer_changed"),
        ({"prefix": "wrong"}, "direct_policy_mismatch"),
        ({"policy": "another-policy"}, "direct_policy_mismatch"),
    ],
)
def test_policy_identity_build_and_prefix_mismatch_fail_before_post(
    make_adapter, changes, message
):
    adapter, client = make_adapter()
    with pytest.raises(LLMBackendError, match=message):
        adapter._embed_texts_direct(["source"], policy=policy(adapter, **changes))
    assert not client.token_calls and not client.embedding_calls
    assert_drained(adapter, client)


def test_unverified_alias_fallback_is_not_accepted_for_automatic_mode(make_adapter):
    adapter, client = make_adapter(ContractClient(1, verified=False), model="embed_0")
    with pytest.raises(LLMBackendError, match="direct_identity_unverified"):
        adapter._embed_texts_direct(["source"], policy=policy(adapter))
    assert not client.token_calls and not client.embedding_calls


def test_missing_loaded_context_is_explicit_failure(make_adapter, monkeypatch):
    adapter, client = make_adapter()
    original = client.props

    def props(**kwargs):
        result = original(**kwargs)
        result["default_generation_settings"] = {}
        return result

    monkeypatch.setattr(client, "props", props)
    with pytest.raises(LLMBackendError, match="direct_context_unknown"):
        adapter._embed_texts_direct(["source"], policy=policy(adapter))
    assert not client.token_calls and not client.embedding_calls


@pytest.mark.parametrize("native_limit", [False, True])
def test_direct_reservation_uses_context_and_microbatch_without_local_admission(
    make_adapter, monkeypatch, native_limit
):
    client = ContractClient(1, context=100)
    if native_limit:
        client.catalog[0]["status"]["args"] += ["--ubatch-size", "10"]
    else:
        client.contexts["embed_0"] = 10
    adapter, _ = make_adapter(client)
    reservations = []
    prepare = adapter._engine_prepare

    def record(*args, **kwargs):
        result = prepare(*args, **kwargs)
        reservations.append(result.reserved_tokens)
        return result

    monkeypatch.setattr(adapter, "_engine_prepare", record)
    text = "x" * 31
    assert adapter._embed_texts_direct([text], policy=policy(adapter)) == [vector(text)]
    assert reservations == [10]
    assert not client.token_calls
    assert client.embedding_calls == [("embed_0", (text,))]


def test_fresh_prepost_context_reduction_does_not_reject_old_reservation(
    make_adapter, monkeypatch
):
    adapter, client = make_adapter(ContractClient(1))
    prepare = adapter._engine_prepare

    def reducing(*args, **kwargs):
        result = prepare(*args, **kwargs)
        client.contexts["embed_0"] = 10
        return result

    monkeypatch.setattr(adapter, "_engine_prepare", reducing)
    assert adapter._embed_texts_direct(["source"], policy=policy(adapter)) == [
        vector("source")
    ]
    assert not client.token_calls
    assert client.embedding_calls == [("embed_0", ("source",))]


def test_fresh_prepost_tokenizer_change_rejects_reservation(make_adapter, monkeypatch):
    adapter, client = make_adapter(ContractClient(1))
    prepare = adapter._engine_prepare

    def changing(*args, **kwargs):
        result = prepare(*args, **kwargs)
        client.builds["embed_0"] = "changed-build"
        return result

    monkeypatch.setattr(adapter, "_engine_prepare", changing)
    with pytest.raises(LLMBackendError, match="direct_tokenizer_changed"):
        adapter._embed_texts_direct(["source"], policy=policy(adapter))
    assert not client.token_calls and not client.embedding_calls


def test_fresh_prepost_microbatch_reduction_does_not_reject_old_reservation(
    make_adapter, monkeypatch
):
    adapter, client = make_adapter(ContractClient(1))
    prepare = adapter._engine_prepare

    def reducing(*args, **kwargs):
        result = prepare(*args, **kwargs)
        client.catalog[0]["status"]["args"] += ["--ubatch-size", "10"]
        return result

    monkeypatch.setattr(adapter, "_engine_prepare", reducing)
    assert adapter._embed_texts_direct(["source"], policy=policy(adapter)) == [
        vector("source")
    ]
    assert not client.token_calls
    assert client.embedding_calls == [("embed_0", ("source",))]


def test_reservation_cannot_be_issued_as_exact_public_preparation(make_adapter):
    adapter, client = make_adapter(ContractClient(1))
    call = _Call(adapter, None)
    call._direct_policy = policy(adapter)
    context = adapter._engine_start(call)
    try:
        record = adapter._engine_prepare(
            ("source", "document"),
            "embed_0",
            call,
            context.observations["embed_0"],
            context.conservative,
        )
        with pytest.raises(LLMBackendError, match="direct_preparation_not_exact"):
            adapter._engine_result(call, "prepare", [record])
        assert not client.token_calls
    finally:
        adapter._engine_finish(call, context.joined)


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(400, "exceed_context_size_error"), (500, "server_error")],
)
def test_native_rejection_is_typed_without_retry_or_truncation(
    make_adapter, status, error_type
):
    adapter, client = make_adapter(ContractClient(1))
    error = ClientHTTPError(status, error_type=error_type)
    client.failure = lambda *_: error
    expected = _FailedOwnerPost if status == 500 else ClientHTTPError
    with pytest.raises(expected) as raised:
        adapter._embed_texts_direct(["one", "two"], policy=policy(adapter))
    if status == 500:
        assert raised.value.original is error
        assert [row.input_index for row in raised.value.rows] == [0, 1]
    else:
        assert raised.value is error
    assert client.embedding_calls == [("embed_0", ("one", "two"))]
    assert not client.token_calls
    assert_drained(adapter, client)


def test_byte_bound_is_still_hard_for_direct_source(make_adapter):
    adapter, client = make_adapter(embedding_max_request_bytes=80)
    with pytest.raises(LLMBackendError, match="request_bytes"):
        adapter._embed_texts_direct(["🙂" * 30], policy=policy(adapter))
    assert not client.token_calls and not client.embedding_calls


def test_token_id_only_profile_does_not_silently_change_format(make_adapter):
    adapter, client = make_adapter(embedding_input_format="tokens")
    with pytest.raises(LLMBackendError, match="direct_text_input_required"):
        adapter._embed_texts_direct(["source"], policy=policy(adapter))
    assert not client.token_calls and not client.embedding_calls


@pytest.mark.parametrize(
    "invalid",
    [
        {"content_sha256": "not-a-digest"},
        {"embedding_sha256": "A" * 64},
        {"tokenizer_build": ""},
        {"policy": ""},
        {"prefix": None},
    ],
)
def test_invalid_direct_policy_is_rejected(make_adapter, invalid):
    adapter, _ = make_adapter()
    with pytest.raises(ValueError):
        policy(adapter, **invalid)


def test_policy_is_frozen_and_invalid_input_never_reaches_server(make_adapter):
    adapter, client = make_adapter()
    chosen = policy(adapter)
    with pytest.raises(FrozenInstanceError):
        chosen.prefix = "other"
    for incoming in ("text", b"text", [None], [1], iter(["text"])):
        with pytest.raises(ValueError):
            adapter._embed_texts_direct(incoming, policy=chosen)
    assert not client.token_calls and not client.embedding_calls


def test_sync_cancellation_and_timeout_drain_same_engine(make_adapter):
    adapter, client = make_adapter(ContractClient(1))
    chosen = policy(adapter)
    client.embed_gate.clear()
    cancelled = Event()
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(
            adapter._embed_texts_direct,
            ["source"],
            policy=chosen,
            cancel=cancelled.is_set,
        )
        assert client.embed_started.wait(2)
        cancelled.set()
        with pytest.raises(LLMCancelledError):
            future.result(3)
    assert_drained(adapter, client)
    client.embed_started.clear()
    with pytest.raises(LLMTimeoutError):
        adapter._embed_texts_direct(["other"], policy=chosen, timeout=0.1)
    assert_drained(adapter, client)
    assert not client.token_calls


def test_async_direct_stream_is_inert_ordered_and_uses_shared_bridge():
    async def scenario():
        client = ContractClient()
        adapter = configured(client, embedding_batch_wait_sec=0)
        source = WindowSource([["one", "one"], [], ["東京🙂"], ["last"]])
        chosen = policy(adapter)
        stream = adapter._astream_texts_direct(source, policy=chosen)
        assert not client.props_calls and not client.token_calls
        assert source.pulls == 0
        try:
            async with stream:
                result = [window async for window in stream]
            assert result == [
                [vector("one"), vector("one")],
                [],
                [vector("東京🙂")],
                [vector("last")],
            ]
            assert stream._bridge is adapter._stream_bridge()
            assert not client.token_calls
            assert_drained(adapter, client)
        finally:
            await adapter.aclose()

    run(scenario)


def test_async_cancel_and_close_drain_direct_http():
    async def scenario():
        client = ContractClient(1)
        client.embed_gate.clear()
        adapter = configured(client, embedding_batch_wait_sec=0)
        source = WindowSource([["source"], ["pending"]])
        stream = adapter._astream_texts_direct(source, policy=policy(adapter))
        next_result = asyncio.create_task(anext(stream))
        try:
            await wait_thread(client.embed_started)
            next_result.cancel()
            with pytest.raises(asyncio.CancelledError):
                await next_result
            await stream.aclose()
            assert_drained(adapter, client)
            assert not client.token_calls
        finally:
            await adapter.aclose()

    run(scenario)


def test_exact_and_direct_owners_share_global_concurrency(make_adapter):
    adapter, client = make_adapter(
        ContractClient(1), embedding_request_parallelism=1, embedding_queue_size=2
    )
    chosen = policy(adapter)
    client.embed_gate.clear()
    with ThreadPoolExecutor(2) as executor:
        direct = executor.submit(adapter._embed_texts_direct, ["auto"], policy=chosen)
        assert client.embed_started.wait(2)
        exact = executor.submit(adapter.embed_texts, ["exact"])
        eventually(lambda: bool(client.token_calls))
        assert client.embed_peak == 1
        client.embed_gate.set()
        assert direct.result(3) == [vector("auto")]
        assert exact.result(3) == [vector("exact")]
    assert [text for text, _ in client.token_calls] == ["exact"]
    assert client.embed_peak == 1
    assert_drained(adapter, client)


@pytest.mark.parametrize("cancellation", ["probe", "task"])
def test_shared_direct_post_retires_only_cancelled_owner(cancellation):
    from test_embedding_shared_lifecycle import SharedScenario

    async def scenario():
        stop = Event()
        shared = SharedScenario(cancel_a=stop.is_set)
        shared.streams = [
            shared.adapter._astream_texts_direct(
                source,
                policy=policy(shared.adapter),
                cancel=cancel,
                timeout=timeout,
                close_source=True,
            )
            for source, cancel, timeout in zip(
                shared.sources, (stop.is_set, None), (5, 10)
            )
        ]
        async with shared:
            owner = shared.calls["a"]
            shared.client.retired_owner = owner.stop.is_set
            if cancellation == "probe":
                stop.set()
                expected = LLMCancelledError
            else:
                shared.tasks[0].cancel()
                expected = asyncio.CancelledError
            await wait_thread(shared.client.retired_owner_polled)
            assert not shared.tasks[0].done()
            assert not shared.client.transport_cancelled.is_set()
            assert shared.adapter._scheduler.snapshot()["in_flight"] == 1
            shared.client.shared_gate.set()
            with pytest.raises(expected):
                await shared.tasks[0]
            assert await shared.tasks[1] == [vector("b-0"), vector("b-1")]
            shared.assert_one_shared_post()
            assert [text for text, _ in shared.client.token_calls] == ["prime"]
            await shared.streams[1].aclose()
            assert_drained(shared.adapter, shared.client)

    run(scenario)
