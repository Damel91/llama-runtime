"""Admission for a shared wire request remains local to each logical owner."""

from dataclasses import replace

from runtime_llama.adapters.embedding import _Call
from runtime_llama.errors import LLMBackendError

from test_embedding_contracts import ContractClient, configured, vector


def test_shared_fresh_validation_is_linear_and_does_not_repeat_tokenization():
    client = ContractClient(1)
    adapter = configured(client)
    left, right = _Call(adapter, None), _Call(adapter, None)
    context = adapter._engine_start(left)
    observation = context.observations["embed_0"]
    admitted = []
    original = adapter._admit

    def admit(record, observed):
        admitted.append(record.text)
        return original(record, observed)

    adapter._admit = admit
    try:
        records = [
            adapter._engine_prepare(
                (text, "document"), "embed_0", call, observation, context.conservative
            )
            for text, call in [("alpha", left), ("beta", right)]
        ]
        for record in records:
            adapter._engine_validate(record, "embed_0", observation)
        group = adapter._engine_batch_call([left, right])
        result = adapter._engine_run_shared(
            "embed_0", records, [left, right], group, observation
        )
        assert result == [(0, vector("alpha")), (1, vector("beta"))]
        assert admitted == ["alpha", "beta", "alpha", "beta"]
        assert len(client.token_calls) == 2
        assert client.embedding_calls == [("embed_0", ("alpha", "beta"))]
    finally:
        adapter._engine_finish(left, context.joined)
        adapter.close()


def test_old_owner_input_limit_does_not_leak_into_another_owner():
    client = ContractClient(1)
    adapter = configured(client)
    left, right = _Call(adapter, None), _Call(adapter, None)
    context = adapter._engine_start(left)
    observation = context.observations["embed_0"]
    try:
        records = [
            adapter._engine_prepare(
                (text, "document"), "embed_0", call, observation, context.conservative
            )
            for text, call in [("a", left), ("long-document", right)]
        ]
        records[0] = replace(records[0], context_token_limit=2, max_input_tokens=2)
        rows = adapter._engine_run_shared(
            "embed_0",
            records,
            [left, right],
            adapter._engine_batch_call([left, right]),
            observation,
        )
        assert rows == [(0, vector("a")), (1, vector("long-document"))]
        assert left.failure is right.failure is None
    finally:
        adapter._engine_finish(left, context.joined)
        adapter.close()


def test_dimension_expectation_failure_is_confined_to_its_owner():
    client = ContractClient(1)
    adapter = configured(client)
    left, right = _Call(adapter, None), _Call(adapter, None)
    left.dimension = 3
    context = adapter._engine_start(left)
    observation = context.observations["embed_0"]
    try:
        records = [
            adapter._engine_prepare(
                (text, "document"), "embed_0", call, observation, context.conservative
            )
            for text, call in [("alpha", left), ("beta", right)]
        ]
        rows = adapter._engine_run_shared(
            "embed_0",
            records,
            [left, right],
            adapter._engine_batch_call([left, right]),
            observation,
        )
        assert len(rows) == 2  # Wire indexes remain complete; engine drops left.
        assert isinstance(left.failure, LLMBackendError)
        assert "dimension_mismatch" in str(left.failure)
        assert right.failure is None and right.dimension == 2
    finally:
        adapter._engine_finish(left, context.joined)
        adapter.close()
