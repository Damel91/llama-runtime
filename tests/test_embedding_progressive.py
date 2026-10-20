"""Progressive preparation and native token input contracts, entirely offline."""

import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from threading import Event, Lock

import pytest

import runtime_llama.adapters.embedding as embedding_module
from runtime_llama.client import ClientHTTPError
from runtime_llama.embedding_preparation import issue_preparation
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError

from test_embedding_async_stream import (
    WindowSource,
    checkpoint,
    run,
    wait_thread,
    wait_until,
)
from test_embedding_contracts import (
    CONTENT,
    ContractClient,
    configured,
    eventually,
    texts,
    vector,
)


class BatchClock:
    """Only the coalescing clock moves; real call deadlines remain finite."""

    def __init__(self):
        self.value = 0.0
        self.sampled = Event()
        self.lock = Lock()

    def __call__(self):
        with self.lock:
            value = self.value
        self.sampled.set()
        return value

    def advance(self, value):
        with self.lock:
            assert value >= self.value
            self.value = value


class TokenClient(ContractClient):
    """Fake native client: IDs map back to text, without implicit BOS repair."""

    supports_embedding_token_ids = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_gates = {}
        self.token_done = {}
        self.ids_by_text = {}
        self.text_by_ids = {}
        self.wire_calls = []
        self.events = []
        self.wire_failure = None

    def gate(self, text):
        self.token_gates[text] = Event()
        self.token_done[text] = Event()
        return self.token_gates[text]

    def tokenize(self, content, *, cancel=None, timeout=None, **kwargs):
        with self.lock:
            self.token_active += 1
            self.token_peak = max(self.token_peak, self.token_active)
            self.token_calls.append((content, dict(kwargs)))
            self.events.append(("token-start", content))
            self.token_started.set()
        try:
            self._wait(
                self.token_gates.get(content, self.token_gate),
                cancel=cancel,
                timeout=timeout,
            )
            if self.token_failure:
                raise self.token_failure
            with self.lock:
                if content not in self.ids_by_text:
                    count = self.counts.get(content, len(content) + 1)
                    identifier = 100 + len(self.ids_by_text)
                    ids = (
                        ()
                        if count == 0
                        else (1,)
                        if count == 1
                        else (1, *([identifier] * (count - 2)), 2)
                    )
                    self.ids_by_text[content] = ids
                    self.text_by_ids[ids] = content
                ids = self.ids_by_text[content]
                self.events.append(("token-complete", content))
                self.token_done.setdefault(content, Event()).set()
            return {"tokens": list(ids)}
        finally:
            with self.lock:
                self.token_active -= 1

    def embeddings(self, input, *, model, **kwargs):
        incoming = deepcopy(input)
        with self.lock:
            self.wire_calls.append((model, incoming))
            self.events.append(("embedding-wire", incoming))
        if self.wire_failure is not None:
            raise self.wire_failure
        if not isinstance(input, list) or not input:
            raise AssertionError("embedding wire input must be a nonempty list")
        if all(type(item) is int for item in input):
            rows = [input]
        elif all(isinstance(item, list) for item in input):
            rows = input
        elif all(isinstance(item, str) for item in input):
            return super().embeddings(input, model=model, **kwargs)
        else:
            raise AssertionError("invalid or mixed embedding wire shape")
        if any(
            not row or any(type(token) is not int or token < 0 for token in row)
            for row in rows
        ):
            raise AssertionError("invalid native token sequence")
        decoded = [self.text_by_ids[tuple(row)] for row in rows]
        return super().embeddings(decoded, model=model, **kwargs)


def adapter_for(client, **changes):
    values = {
        "embedding_batch_wait_sec": 0.5,
        "embedding_input_format": "auto",
        "timeout": 5,
    }
    values.update(changes)
    return configured(client, **values)


def wire_kind(payload):
    return "text" if all(isinstance(item, str) for item in payload) else "tokens"


def wire_text_count(payload):
    return 1 if payload and all(type(item) is int for item in payload) else len(payload)


def assert_once(client, incoming):
    assert Counter(
        text for _, batch in client.embedding_calls for text in batch
    ) == Counter(incoming)
    assert all(peak <= 1 for peak in client.alias_peak.values())


def drain_assertions(adapter, client):
    assert not adapter._calls
    assert client.token_active == client.embed_active == 0
    snapshot = adapter._scheduler.snapshot()
    assert snapshot["queued"] == snapshot["in_flight"] == 0


def test_progressive_dispatch_precedes_completion_of_its_quota(monkeypatch):
    client = TokenClient(1)
    incoming = texts(5)
    blocked = client.gate(incoming[-1])
    adapter = adapter_for(client, token_count_parallelism=1)
    clock = BatchClock()
    monkeypatch.setattr(adapter, "_batch_now", clock)
    try:
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(adapter.embed_texts, incoming)
            assert client.token_started.wait(2)
            eventually(lambda: len(client.token_calls) == len(incoming))
            assert clock.sampled.wait(2)
            clock.advance(0.5)
            assert client.embed_started.wait(2)
            assert not client.token_done[incoming[-1]].is_set()
            assert not pending.done()
            assert all(incoming[-1] not in batch for _, batch in client.embedding_calls)
            blocked.set()
            assert pending.result(3) == [vector(text) for text in incoming]
        assert_once(client, incoming)
    finally:
        blocked.set()
        adapter.close()


def test_ready_other_alias_progresses_despite_out_of_order_tokenization(monkeypatch):
    client = TokenClient(2)
    incoming = texts(6)
    blocked = client.gate(incoming[0])
    adapter = adapter_for(client, token_count_parallelism=2)
    clock = BatchClock()
    monkeypatch.setattr(adapter, "_batch_now", clock)
    try:
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(adapter.embed_texts, incoming)
            assert client.embed_started.wait(2)
            assert all(incoming[0] not in batch for _, batch in client.embedding_calls)
            assert not pending.done()
            blocked.set()
            assert pending.result(3) == [vector(text) for text in incoming]
        assert_once(client, incoming)
        assert Counter(text for text, _ in client.token_calls) == Counter(incoming)
    finally:
        blocked.set()
        adapter.close()


def test_complete_ready_quotas_coalesce_without_waiting_or_request_storm(monkeypatch):
    client = TokenClient(2)
    incoming = texts(256)
    adapter = adapter_for(client)
    clock = BatchClock()
    monkeypatch.setattr(adapter, "_batch_now", clock)
    try:
        assert adapter.embed_texts(incoming) == [vector(text) for text in incoming]
        assert len(client.wire_calls) == 2
        assert (
            sorted(wire_text_count(payload) for _, payload in client.wire_calls)
            == [len(incoming) // 2] * 2
        )
        assert_once(client, incoming)
    finally:
        adapter.close()


def test_batch_timer_is_anchored_to_first_ready_record_not_later_arrivals(monkeypatch):
    client = TokenClient(1)
    incoming = texts(4)
    second, last = client.gate(incoming[1]), client.gate(incoming[-1])
    adapter = adapter_for(client, token_count_parallelism=1)
    clock = BatchClock()
    monkeypatch.setattr(adapter, "_batch_now", clock)
    try:
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(adapter.embed_texts, incoming)
            eventually(lambda: len(client.token_calls) == 2)
            assert clock.sampled.wait(2)
            clock.advance(0.4)
            second.set()
            eventually(lambda: len(client.token_calls) == 4)
            clock.advance(0.5)
            assert client.embed_started.wait(2)
            assert not last.is_set() and not pending.done()
            last.set()
            assert pending.result(3) == [vector(text) for text in incoming]
    finally:
        second.set()
        last.set()
        adapter.close()


@pytest.mark.parametrize(
    ("replicas", "cap", "count"),
    [(1, 1, 0), (4, 4, 1), (4, 2, 3), (2, 2, 255), (2, 2, 256), (4, 2, 257)],
)
def test_progressive_balanced_quotas_and_odd_tails_preserve_every_position(
    monkeypatch, replicas, cap, count
):
    client = TokenClient(replicas)
    incoming = texts(count)
    adapter = adapter_for(client, embedding_request_parallelism=cap)
    monkeypatch.setattr(adapter, "_batch_now", BatchClock())
    try:
        assert adapter.embed_texts(incoming) == [vector(text) for text in incoming]
        assert_once(client, incoming)
        assert all(wire_text_count(payload) > 0 for _, payload in client.wire_calls)
        assert client.embed_peak <= cap
    finally:
        adapter.close()


@pytest.mark.parametrize("native_batch", [4, 2048])
@pytest.mark.parametrize("trigger", ["http-budget", "adaptive-cap"])
def test_progressive_flush_respects_explicit_http_budget_or_adaptive_text_cap(
    monkeypatch, native_batch, trigger
):
    client = TokenClient(1, native_batch=native_batch)
    incoming = texts(5)
    client.counts.update(dict.fromkeys(incoming, 5))
    adapter = adapter_for(
        client, batch_token_limit=10 if trigger == "http-budget" else None
    )
    monkeypatch.setattr(adapter, "_batch_now", BatchClock())
    try:
        if trigger == "adaptive-cap":
            adapter.health_status()
            for replica in adapter._replicas.values():
                replica.adaptive_batch_size = 2
        assert adapter.embed_texts(incoming) == [vector(text) for text in incoming]
        assert sorted(wire_text_count(payload) for _, payload in client.wire_calls) == [
            1,
            2,
            2,
        ]
        if trigger == "http-budget":
            assert all(
                sum(len(row) for row in payload) <= 10
                for _, payload in client.wire_calls
            )
    finally:
        adapter.close()


@pytest.mark.parametrize("wait", [0, -1, True, float("inf"), float("nan"), "slow"])
def test_invalid_coalescing_wait_is_rejected_before_transport(wait):
    client = TokenClient()
    with pytest.raises(ValueError):
        adapter = adapter_for(client, embedding_batch_wait_sec=wait)
        try:
            adapter.embed_texts(["one"])
        finally:
            adapter.close()
    assert not client.token_calls and not client.wire_calls


def test_progressive_ready_pressure_remains_bounded_across_shared_callers(monkeypatch):
    client = TokenClient(1)
    client.embed_gate.clear()
    adapter = adapter_for(
        client,
        embedding_request_parallelism=1,
        embedding_queue_size=0,
        token_count_parallelism=1,
        timeout=5,
    )
    monkeypatch.setattr(adapter, "_batch_now", BatchClock())
    cancel = Event()
    try:
        with ThreadPoolExecutor(4) as pool:
            jobs = [
                pool.submit(
                    adapter.embed_texts, texts(20, i * 100), cancel=cancel.is_set
                )
                for i in range(4)
            ]
            assert client.embed_started.wait(2)
            eventually(lambda: len(adapter._calls) == len(jobs))
            snapshot = adapter._scheduler.snapshot()
            assert snapshot["queued"] == 0 and snapshot["in_flight"] == 1
            assert client.token_peak <= 1 and client.embed_peak <= 1
            # One configured lookahead quota remains leased through its HTTP
            # drain; additional callers cannot prepare unbounded ready data.
            acquired = adapter._preparation_permits.acquire(blocking=False)
            if acquired:
                adapter._preparation_permits.release()
            assert not acquired
            assert len(client.token_calls) <= 20
            cancel.set()
            for job in jobs:
                with pytest.raises(LLMCancelledError):
                    job.result(3)
        drain_assertions(adapter, client)
    finally:
        cancel.set()
        client.embed_gate.set()
        adapter.close()


def test_single_worker_stages_do_not_deadlock_under_progressive_preparation(
    monkeypatch,
):
    client = TokenClient(1)
    adapter = adapter_for(
        client,
        token_count_parallelism=1,
        embedding_request_parallelism=1,
        embedding_queue_size=0,
    )
    monkeypatch.setattr(adapter, "_batch_now", BatchClock())
    try:
        with ThreadPoolExecutor(2) as pool:
            jobs = [
                pool.submit(adapter.embed_texts, texts(17, i * 100)) for i in range(2)
            ]
            for i, job in enumerate(jobs):
                assert job.result(3) == [vector(text) for text in texts(17, i * 100)]
        drain_assertions(adapter, client)
    finally:
        adapter.close()


def test_late_oversize_drains_prior_success_without_publishing_or_replaying(
    monkeypatch,
):
    client = TokenClient(1)
    incoming = ["first", "late-oversize"]
    client.counts[incoming[-1]] = 2049
    blocked = client.gate(incoming[-1])
    adapter = adapter_for(client, token_count_parallelism=1)
    clock = BatchClock()
    monkeypatch.setattr(adapter, "_batch_now", clock)
    try:
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(adapter.embed_texts, incoming)
            eventually(lambda: len(client.token_calls) == 2)
            assert clock.sampled.wait(2)
            clock.advance(0.5)
            assert client.embed_started.wait(2)
            blocked.set()
            with pytest.raises(LLMBackendError, match="exceeds"):
                pending.result(3)
        assert client.embedding_calls == [("embed_0", ("first",))]
        drain_assertions(adapter, client)
        assert adapter.embed_query("healthy") == vector("healthy")
    finally:
        blocked.set()
        adapter.close()


def test_duplicate_token_reuse_is_local_to_verified_task_identity(monkeypatch):
    client = TokenClient(1)
    adapter = adapter_for(
        client, embedding_prefixes={"query": "Q: ", "document": "D: "}
    )
    monkeypatch.setattr(adapter, "_batch_now", BatchClock())
    try:
        assert adapter.embed_texts(["same"] * 5) == [vector("D: same")] * 5
        assert len(client.token_calls) == 1
        assert adapter.embed_query("same") == vector("Q: same")
        assert [text for text, _ in client.token_calls] == ["D: same", "Q: same"]
        assert_once(client, ["D: same"] * 5 + ["Q: same"])
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "task,prefix", [("raw", ""), ("query", "Q: "), ("document", "D: ")]
)
@pytest.mark.parametrize("text", ["hello", "中文 🧪", "<bos> literal"])
def test_native_ids_keep_prefix_and_special_tokens_exactly_once(task, prefix, text):
    client = TokenClient(1)
    adapter = adapter_for(
        client, embedding_prefixes={"query": "Q: ", "document": "D: "}
    )
    try:
        prepared = adapter.prepare_texts([text], input_type=task)
        record = prepared[0]
        ids = client.ids_by_text[prefix + text]
        assert record._token_ids == ids and type(record._token_ids) is tuple
        assert record.token_count == len(ids)
        assert adapter.embed_prepared(prepared) == [vector(prefix + text)]
        assert len(client.token_calls) == 1
        assert client.wire_calls[-1][1] == [list(ids)]
        assert ids[0] == 1 and ids[-1] == 2
        assert client.token_calls[0][1]["add_special"] is True
        assert client.token_calls[0][1]["parse_special"] is True
    finally:
        adapter.close()


def test_nested_token_arrays_keep_cardinality_and_response_order(monkeypatch):
    client = TokenClient(1)
    incoming = ["a", "longer", "another"]
    adapter = adapter_for(client)
    monkeypatch.setattr(adapter, "_batch_now", BatchClock())
    try:
        assert adapter.embed_texts(incoming) == [vector(text) for text in incoming]
        assert client.wire_calls == [
            ("embed_0", [list(client.ids_by_text[text]) for text in incoming])
        ]
        assert len({len(row) for row in client.wire_calls[0][1]}) > 1
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "tokens", [[True], [-1], [1.5], ["1"], [[1]], [2**31], None, "tokens"]
)
def test_invalid_native_ids_never_become_prepared_authority_or_embedding(tokens):
    client = TokenClient(1)
    client.tokenize = lambda *_args, **_kwargs: {"tokens": tokens}
    adapter = adapter_for(client)
    try:
        with pytest.raises(LLMBackendError):
            adapter.prepare_query("invalid")
        assert not client.wire_calls
    finally:
        adapter.close()


def test_prepared_token_ids_are_detached_from_mutable_tokenizer_payload():
    client = TokenClient(1)
    payloads = []
    original = client.tokenize

    def tokenize(*args, **kwargs):
        payload = original(*args, **kwargs)
        payloads.append(payload)
        return payload

    client.tokenize = tokenize
    adapter = adapter_for(client)
    try:
        record = adapter.prepare_query("stable")
        ids = record._token_ids
        payloads[0]["tokens"][:] = [999]
        with pytest.raises(FrozenInstanceError):
            record._token_ids = (999,)
        assert adapter.embed_prepared([record]) == [vector("stable")]
        assert client.wire_calls[-1][1] == [list(ids)]
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "change", ["id", "order", "count", "policy", "build", "identity"]
)
def test_token_id_record_seal_binds_every_inference_authority(change):
    client = TokenClient(1)
    adapter = adapter_for(client)
    try:
        record = adapter.prepare_query("sealed")
        fields = {
            "id": {"_token_ids": (9, *record._token_ids[1:])},
            "order": {"_token_ids": tuple(reversed(record._token_ids))},
            "count": {"token_count": record.token_count + 1},
            "policy": {"_policy": "other"},
            "build": {"_tokenizer_build": "other"},
            "identity": {"content_sha256": "f" * 64},
        }[change]
        with pytest.raises(LLMBackendError):
            adapter.embed_prepared([replace(record, **fields)])
        assert not client.wire_calls
    finally:
        adapter.close()


@pytest.mark.parametrize("change", ["rename", "weights", "build", "unknown-build"])
def test_prepared_ids_follow_fresh_topology_identity_and_tokenizer_observation(change):
    client = TokenClient(1)
    adapter = adapter_for(client)
    try:
        prepared = adapter.prepare_query("same")
        counted = len(client.token_calls)
        if change == "rename":
            client.set_aliases(["renamed"])
        elif change == "weights":
            client.catalog[0]["tags"][0] = "noxi-content-sha256-v1:" + (
                "c" * 64 if CONTENT != "c" * 64 else "d" * 64
            )
        else:
            client.builds["embed_0"] = (
                None if change == "unknown-build" else "test-build-2"
            )
        if change == "weights":
            with pytest.raises(LLMBackendError):
                adapter.embed_prepared([prepared])
            assert not client.wire_calls
        else:
            assert adapter.embed_prepared([prepared]) == [vector("same")]
            if change == "rename":
                assert len(client.token_calls) == counted
                assert client.wire_calls[0][0] == "renamed"
            else:
                assert len(client.token_calls) > counted
                if change == "unknown-build":
                    assert wire_kind(client.wire_calls[0][1]) == "text"
    finally:
        adapter.close()


def test_count_only_prepared_record_can_use_only_a_safe_compatible_path():
    client = TokenClient(1)
    adapter = adapter_for(client)
    try:
        record = adapter.prepare_query("legacy-count")
        fields = asdict(record)
        fields.pop("_seal")
        fields["_token_ids"] = None
        legacy = issue_preparation(adapter._preparation_key, **fields)
        counted = len(client.token_calls)
        assert adapter.embed_prepared([legacy]) == [vector("legacy-count")]
        if wire_kind(client.wire_calls[0][1]) == "tokens":
            assert len(client.token_calls) > counted
    finally:
        adapter.close()


def test_preparation_reuse_does_not_repeat_tokenization_or_internal_sealing(
    monkeypatch,
):
    client = TokenClient(1)
    sealed = []
    original = embedding_module.issue_preparation

    def issue(*args, **kwargs):
        sealed.append(kwargs["text"])
        return original(*args, **kwargs)

    monkeypatch.setattr(embedding_module, "issue_preparation", issue)
    adapter = adapter_for(client)
    try:
        assert adapter.embed_texts(["ordinary-a", "ordinary-b"])
        assert sealed == []
        records = adapter.prepare_texts(["public-a", "public-b"])
        counted = len(client.token_calls)
        assert adapter.embed_prepared(records) == [
            vector("public-a"),
            vector("public-b"),
        ]
        assert len(client.token_calls) == counted
        assert sealed == ["public-a", "public-b"]
    finally:
        adapter.close()


@pytest.mark.parametrize("mode", ["auto", "text", "tokens"])
def test_zero_token_preparation_is_measurable_without_synthetic_bos(mode):
    client = TokenClient(1)
    client.counts["zero"] = 0
    adapter = adapter_for(client, embedding_input_format=mode)
    try:
        record = adapter.prepare_query("zero")
        assert record.token_count == 0 and record._token_ids in (None, ())
        if mode == "tokens":
            with pytest.raises(LLMBackendError):
                adapter.embed_prepared([record])
            assert not client.wire_calls
        else:
            assert adapter.embed_prepared([record]) == [vector("zero")]
            assert client.wire_calls[-1][1] == ["zero"]
        assert len(client.token_calls) == 1
    finally:
        adapter.close()


@pytest.mark.parametrize("capability", [None, False, 1, "true", True])
def test_auto_requires_explicit_native_contract_not_truthy_custom_capability(
    capability,
):
    client = TokenClient(1)
    client.supports_embedding_token_ids = capability
    adapter = adapter_for(client)
    try:
        assert adapter.embed_query("capability") == vector("capability")
        assert wire_kind(client.wire_calls[0][1]) == (
            "tokens" if capability is True else "text"
        )
    finally:
        adapter.close()


@pytest.mark.parametrize("mode", ["auto", "text"])
def test_unknown_custom_clients_keep_text_and_old_consumer_semantics(mode):
    client = ContractClient(1)
    adapter = adapter_for(client, embedding_input_format=mode)
    try:
        assert adapter.embed_texts(["one", "two"]) == [vector("one"), vector("two")]
        assert client.embedding_calls == [("embed_0", ("one", "two"))]
    finally:
        adapter.close()


@pytest.mark.parametrize("mode", [None, True, "binary", 1])
def test_invalid_token_input_format_is_local_and_performs_no_http(mode):
    client = TokenClient(1)
    with pytest.raises(ValueError):
        adapter = adapter_for(client, embedding_input_format=mode)
        try:
            adapter.embed_query("invalid")
        finally:
            adapter.close()
    assert not client.token_calls and not client.wire_calls


@pytest.mark.parametrize(
    "failure",
    [ClientHTTPError(400), ClientHTTPError(500), LLMTimeoutError("ambiguous")],
)
def test_token_input_failure_never_replays_as_text_or_another_inference(failure):
    client = TokenClient(1)
    client.wire_failure = failure
    adapter = adapter_for(client, embedding_input_format="tokens")
    try:
        with pytest.raises(type(failure)):
            adapter.embed_query("once-only")
        assert len(client.wire_calls) == 1
        assert wire_kind(client.wire_calls[0][1]) == "tokens"
        drain_assertions(adapter, client)
    finally:
        adapter.close()


@pytest.mark.parametrize("mode", ["auto", "text", "tokens"])
def test_native_context_error_does_not_replay_possibly_computed_siblings(
    monkeypatch, mode
):
    client = TokenClient(1)
    incoming = ["first", "second", "third", "fourth"]
    computed_before_error = []
    error = ClientHTTPError(400, error_type="exceed_context_size_error")

    def fail(_alias, batch):
        if len(batch) == len(incoming):
            # Native may finish one task before discovering a later task's
            # context failure: a rejected HTTP response is not a replay permit.
            computed_before_error.append(batch[0])
            return error
        return None

    client.failure = fail
    adapter = adapter_for(client, embedding_input_format=mode)
    monkeypatch.setattr(adapter, "_batch_now", BatchClock())
    try:
        with pytest.raises(ClientHTTPError) as observed:
            adapter.embed_texts(incoming)
        assert observed.value is error
        assert computed_before_error == [incoming[0]]
        assert len(client.wire_calls) == 1
        assert client.embedding_calls == [("embed_0", tuple(incoming))]
        assert wire_kind(client.wire_calls[0][1]) == (
            "text" if mode == "text" else "tokens"
        )
        drain_assertions(adapter, client)
        assert adapter.embed_query("healthy") == vector("healthy")
    finally:
        adapter.close()


def test_unknown_custom_context_recovery_remains_compatible(monkeypatch):
    client = TokenClient(1)
    client.supports_embedding_token_ids = None
    incoming = ["first", "second", "third", "fourth"]
    client.failure = lambda _alias, batch: (
        ClientHTTPError(400, error_type="exceed_context_size_error")
        if len(batch) == len(incoming)
        else None
    )
    adapter = adapter_for(client)
    monkeypatch.setattr(adapter, "_batch_now", BatchClock())
    try:
        assert adapter.embed_texts(incoming) == [vector(text) for text in incoming]
        assert len(client.wire_calls) == 3
        assert all(wire_kind(payload) == "text" for _, payload in client.wire_calls)
        drain_assertions(adapter, client)
    finally:
        adapter.close()


def test_auto_capability_and_identity_decisions_do_not_leak_between_adapters():
    native, unknown = TokenClient(1), TokenClient(1)
    unknown.supports_embedding_token_ids = None
    first, second = adapter_for(native), adapter_for(unknown)
    try:
        assert first.embed_query("same") == second.embed_query("same") == vector("same")
        assert wire_kind(native.wire_calls[0][1]) == "tokens"
        assert wire_kind(unknown.wire_calls[0][1]) == "text"
        unknown.supports_embedding_token_ids = True
        unknown.builds["embed_0"] = "new-build"
        assert second.embed_query("next") == vector("next")
        assert wire_kind(unknown.wire_calls[-1][1]) == "tokens"
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("missing", ["identity", "tokenizer-build"])
def test_auto_without_verified_identity_or_build_keeps_text_and_recounts(missing):
    client = TokenClient(1, verified=missing != "identity")
    if missing == "tokenizer-build":
        client.builds["embed_0"] = None
    adapter = adapter_for(client, model="embed_0")
    try:
        record = adapter.prepare_query("not-authoritative")
        counted = len(client.token_calls)
        assert adapter.embed_prepared([record]) == [vector("not-authoritative")]
        assert adapter.embed_prepared([record]) == [vector("not-authoritative")]
        assert len(client.token_calls) == counted + 2
        assert all(wire_kind(payload) == "text" for _, payload in client.wire_calls)
    finally:
        adapter.close()


def test_mixed_native_sync_and_async_windows_share_existing_limits():
    async def scenario():
        client = TokenClient(2)
        client.embed_gate.clear()
        adapter = adapter_for(
            client, embedding_request_parallelism=2, embedding_queue_size=0, timeout=5
        )
        stream = adapter.astream_texts(WindowSource([["async-a", "async-b"]]))
        first = asyncio.create_task(anext(stream))
        sync = None
        try:
            await wait_until(lambda: client.embed_active == 2)
            sync = asyncio.create_task(asyncio.to_thread(adapter.embed_texts, ["sync"]))
            await wait_until(lambda: len(adapter._calls) == 2)
            assert client.embed_peak == 2
            client.embed_gate.set()
            assert await first == [vector("async-a"), vector("async-b")]
            assert await sync == [vector("sync")]
            assert all(
                wire_kind(payload) == "tokens" for _, payload in client.wire_calls
            )
            assert_once(client, ["async-a", "async-b", "sync"])
        finally:
            client.embed_gate.set()
            await asyncio.gather(
                *(job for job in [first, sync] if job is not None),
                return_exceptions=True,
            )
            await stream.aclose()
            await adapter.aclose()

    run(scenario)


@pytest.mark.parametrize("phase", ["tokenize", "http"])
def test_progressive_worker_cancellation_drains_without_stopping_another_caller(phase):
    client = TokenClient(2)
    if phase == "tokenize":
        client.gate("cancelled")
    else:
        client.embed_gate.clear()
    adapter = adapter_for(client, timeout=5)
    cancel = Event()
    try:
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(adapter.embed_query, "cancelled", cancel=cancel.is_set)
            assert (
                client.token_started if phase == "tokenize" else client.embed_started
            ).wait(2)
            second = pool.submit(adapter.embed_query, "healthy")
            cancel.set()
            with pytest.raises(LLMCancelledError):
                first.result(3)
            client.embed_gate.set()
            assert second.result(3) == vector("healthy")
        drain_assertions(adapter, client)
    finally:
        cancel.set()
        client.embed_gate.set()
        for gate in client.token_gates.values():
            gate.set()
        adapter.close()


def test_progressive_close_drains_queued_ids_and_rejects_new_intake():
    async def scenario():
        client = TokenClient(1)
        client.embed_gate.clear()
        adapter = adapter_for(
            client, embedding_request_parallelism=1, embedding_queue_size=1, timeout=5
        )
        source = WindowSource([["first"], ["second"], ["third"]])
        stream = adapter.astream_texts(source, close_source=True)
        pending = asyncio.create_task(anext(stream))
        try:
            await wait_thread(client.embed_started)
            await asyncio.wait_for(adapter.aclose(), 3)
            await asyncio.gather(pending, return_exceptions=True)
            assert source.close_calls == 1
            drain_assertions(adapter, client)
            with pytest.raises(LLMBackendError, match="closed"):
                await asyncio.to_thread(adapter.embed_query, "late")
        finally:
            client.embed_gate.set()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            await adapter.aclose()

    run(scenario)


def test_progressive_timeout_is_one_window_deadline_including_coalescing(monkeypatch):
    client = TokenClient(1)
    blocked = client.gate("last")
    adapter = adapter_for(
        client, embedding_batch_wait_sec=10, token_count_parallelism=1
    )
    monkeypatch.setattr(adapter, "_batch_now", BatchClock())
    try:
        with pytest.raises(LLMTimeoutError):
            adapter.embed_texts(["first", "last"], timeout=0.05)
        assert not client.wire_calls
        drain_assertions(adapter, client)
    finally:
        blocked.set()
        adapter.close()


@pytest.mark.parametrize("change", ["context", "identity", "build"])
def test_progressive_unsent_records_refresh_limits_identity_and_build(
    monkeypatch, change
):
    client = TokenClient(1)
    client.counts.update({"first": 4, "last": 4})
    blocked = client.gate("last")
    adapter = adapter_for(client, token_count_parallelism=1)
    clock = BatchClock()
    monkeypatch.setattr(adapter, "_batch_now", clock)
    try:
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(adapter.embed_texts, ["first", "last"])
            eventually(lambda: len(client.token_calls) == 2)
            assert clock.sampled.wait(2)
            clock.advance(0.5)
            assert client.embed_started.wait(2)
            eventually(lambda: client.embed_active == 0)
            if change == "context":
                client.contexts["embed_0"] = 2
            elif change == "identity":
                client.catalog[0]["tags"][0] = "noxi-content-sha256-v1:" + "c" * 64
            else:
                client.builds["embed_0"] = "changed-after-first"
            blocked.set()
            with pytest.raises(LLMBackendError):
                pending.result(3)
        assert client.embedding_calls == [("embed_0", ("first",))]
        drain_assertions(adapter, client)
    finally:
        blocked.set()
        adapter.close()


def test_paused_consumer_retains_native_windows_without_speculative_source_reads():
    async def scenario():
        client = TokenClient(1)
        adapter = adapter_for(
            client, embedding_request_parallelism=2, embedding_queue_size=0
        )
        source = WindowSource([["one"], ["two"], ["three"]])
        stream = adapter.astream_texts(source, max_pending_windows=2)
        try:
            async with stream:
                await wait_until(
                    lambda: (
                        len(client.embedding_calls) == 2 and client.embed_active == 0
                    )
                )
                await checkpoint()
                assert source.pulls == 2
                assert await anext(stream) == [vector("one")]
                assert await anext(stream) == [vector("two")]
                assert await anext(stream) == [vector("three")]
                with pytest.raises(StopAsyncIteration):
                    await anext(stream)
            assert_once(client, ["one", "two", "three"])
        finally:
            await stream.aclose()
            await adapter.aclose()

    run(scenario)
