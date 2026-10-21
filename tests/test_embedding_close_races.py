"""Deterministic close races and busy-replica admission regressions."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from runtime_llama.errors import LLMBackendError

from test_embedding_contracts import ContractClient, configured, eventually, vector


@pytest.mark.parametrize("phase", ["metadata", "preparation", "queued"])
@pytest.mark.parametrize("prior_failure", [False, True])
def test_close_during_pending_work_preserves_typed_first_failure(
    monkeypatch, phase, prior_failure
):
    client = ContractClient(1)
    client.token_gate.clear()
    adapter = configured(
        client,
        timeout=5,
        token_count_parallelism=1,
        embedding_request_parallelism=1,
        embedding_queue_size=0,
    )
    metadata_entered, release_metadata = Event(), Event()
    props = client.props
    expected = LLMBackendError("test_existing_backend_failure")

    def observed_props(*args, **kwargs):
        if phase == "metadata":
            metadata_entered.set()
            client._wait(
                release_metadata,
                cancel=kwargs.get("cancel"),
                timeout=kwargs.get("timeout"),
            )
        return props(*args, **kwargs)

    monkeypatch.setattr(client, "props", observed_props)
    try:
        with ThreadPoolExecutor(max_workers=4) as callers:
            pending = [callers.submit(adapter.embed_query, "one")]
            try:
                assert (
                    metadata_entered if phase == "metadata" else client.token_started
                ).wait(2)
                if phase == "queued":
                    pending.append(callers.submit(adapter.embed_query, "two"))
                    eventually(lambda: len(adapter._calls) == 2)
                    assert client.token_active == 1
                    assert len(client.token_calls) == 1
                if prior_failure:
                    # A child can already have failed before close interrupts
                    # another active or queued part of the same call.
                    with adapter._calls_lock:
                        calls = tuple(adapter._calls)
                    for call in calls:
                        call.fail(expected)
                closing = callers.submit(adapter.close)
                for future in pending:
                    with pytest.raises(LLMBackendError) as failure:
                        future.result(timeout=4)
                    if prior_failure:
                        assert failure.value is expected
                    else:
                        assert str(failure.value) == "llamacpp_embedding_adapter_closed"
                closing.result(timeout=4)
            finally:
                release_metadata.set()
                client.token_gate.set()
        assert not adapter._calls
        assert adapter._scheduler.snapshot()["in_flight"] == 0
        assert client.token_active == client.embed_active == 0
        assert not client.embedding_calls
    finally:
        release_metadata.set()
        client.token_gate.set()
        adapter.close()


def test_busy_replica_does_not_block_collection_of_other_prepared_quota(monkeypatch):
    client = ContractClient(2)
    incoming = ["a0", "a1", "b0", "b1"]
    client.counts.update(dict.fromkeys(incoming, 2))
    adapter = configured(
        client,
        batch_size=4,
        batch_token_limit=2,
        embedding_request_parallelism=2,
        embedding_queue_size=0,
        timeout=5,
    )
    a_first_entered, b_second_entered, release_a = Event(), Event(), Event()
    tokenize, embeddings = client.tokenize, client.embeddings

    def prepare(content, *, model=None, **kwargs):
        if model == "embed_1":
            # B is still preparing when A becomes ready; its eventual ready
            # records must advance while A's next HTTP cannot acquire a slot.
            client._wait(
                a_first_entered,
                cancel=kwargs.get("cancel"),
                timeout=kwargs.get("timeout"),
            )
        return tokenize(content, model=model, **kwargs)

    def request(input, *, model, **kwargs):
        result = embeddings(input, model=model, **kwargs)
        if model == "embed_0" and input == ["a0"]:
            a_first_entered.set()
            client._wait(
                release_a,
                cancel=kwargs.get("cancel"),
                timeout=kwargs.get("timeout"),
            )
        if model == "embed_1" and input == ["b1"]:
            b_second_entered.set()
        return result

    monkeypatch.setattr(client, "tokenize", prepare)
    monkeypatch.setattr(client, "embeddings", request)
    try:
        with ThreadPoolExecutor(max_workers=1) as callers:
            pending = callers.submit(adapter.embed_texts, incoming)
            try:
                assert a_first_entered.wait(2)
                assert b_second_entered.wait(2)
                assert not pending.done()
                assert not release_a.is_set()
            finally:
                release_a.set()
            assert pending.result(timeout=3) == [vector(text) for text in incoming]
        assert Counter(client.embedding_calls) == Counter(
            (f"embed_{index // 2}", (text,)) for index, text in enumerate(incoming)
        )
        assert all(value == 1 for value in client.alias_peak.values())
        assert adapter._scheduler.snapshot()["peak_in_flight"] == 2
    finally:
        release_a.set()
        adapter.close()
