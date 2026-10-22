"""A sibling's unchanged minimum cannot hide the failed replica's drift."""

import pytest

from runtime_llama import EmbeddingInputOverflowError, LLMBackendError
from runtime_llama.client import ClientHTTPError

from test_automatic_embedding import make_service as make_service, remote
from test_automatic_embedding_recovery import assert_partition, failing_native


@pytest.mark.parametrize("changed_limit", ["context", "microbatch"])
def test_failed_replica_limit_drift_is_rejected_even_with_same_pool_minimum(
    make_service, changed_limit
):
    client = remote(replicas=2, context=64)
    for entry in client.catalog:
        entry["status"]["args"] += ["--ubatch-size", "64"]
    failing_native(client)
    original = client.tokenize
    changed = False

    def drift(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            changed = True
            if changed_limit == "context":
                client.contexts["embed_0"] = 256
            else:
                client.catalog[0]["status"]["args"][-1] = "256"
        return result

    client.tokenize = drift
    service, _ = make_service(client, request_parallelism=1, batch_size=1)
    with pytest.raises(LLMBackendError, match="recovery_snapshot_changed"):
        service.embed_texts(["x" * 120])
    assert changed and client.token_calls
    assert len(client.embedding_calls) == 1
    assert client.embedding_calls[0][0] == "embed_0"
    # Replica 1 still bounds the pool at 64 despite replica 0's changed value.
    assert client.contexts["embed_1"] == 64
    assert client.catalog[1]["status"]["args"][-1] == "64"
    assert not service.config.cache_dir.exists()
    assert not service._operations
    assert client.token_active == client.embed_active == 0


def test_verified_fragments_fit_every_usable_replica_before_consumer_retry(
    make_service,
):
    client = remote(replicas=2, context=128)
    client.contexts["embed_1"] = 64
    failing_native(client)
    service, _ = make_service(client, request_parallelism=1, batch_size=1)
    source = "α🙂漢字" * 50
    with pytest.raises(EmbeddingInputOverflowError) as caught:
        service.embed_texts([source])
    error = caught.value
    assert client.embedding_calls[0][0] == "embed_0"
    assert len(client.embedding_calls) == 1
    assert error.inputs[0].token_count > 128
    assert error.inputs[0].token_limit == error.budget.effective_context_tokens == 64
    assert_partition(error.inputs[0], source, client)
    pieces = [
        source[span.char_start : span.char_end] for span in error.inputs[0].fragments
    ]
    assert all(span.token_count <= 64 for span in error.inputs[0].fragments)
    previous_counts = len(client.token_calls)
    assert len(service.embed_texts(pieces)) == len(pieces)
    # The next dispatch rotates onto the smaller replica. Exact advice must
    # remain valid there without another failed POST or diagnostic tokenizer.
    assert client.embedding_calls[1][0] == "embed_1"
    assert len(client.token_calls) == previous_counts
    assert not service._operations


@pytest.mark.parametrize("diagnostic_failure", ["http503", "malformed_tokens"])
def test_unavailable_exact_diagnostics_preserve_original500_without_learning(
    make_service, diagnostic_failure
):
    client = remote(replicas=1, context=64)
    original = failing_native(client)
    if diagnostic_failure == "http503":
        client.token_failure = ClientHTTPError(503, error_type="unavailable_error")
    else:
        tokenize = client.tokenize

        def malformed(*args, **kwargs):
            tokenize(*args, **kwargs)
            return {"tokens": ["invalid-token-id"]}

        client.tokenize = malformed
    service, _ = make_service(client)
    with pytest.raises(ClientHTTPError) as caught:
        service.embed_texts(["x" * 120])
    assert caught.value is original
    assert caught.value.status_code == 500
    assert caught.value.error_type == "server_error"
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert len(client.embedding_calls) == len(client.token_calls) == 1
    assert not service.config.cache_dir.exists()
    assert not service._operations
    assert client.token_active == client.embed_active == 0
