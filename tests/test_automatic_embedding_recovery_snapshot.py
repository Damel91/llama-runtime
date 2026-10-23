"""Native isolation keeps identity checks and never revises statistical means."""

import pytest

from runtime_llama import EmbeddingChunkSplitRequiredError, LLMBackendError
from runtime_llama.client import ClientHTTPError

from test_automatic_embedding import make_service as make_service, remote
from test_automatic_embedding_recovery import assert_halves, failing_native


@pytest.mark.parametrize("changed_identity", ["tokenizer", "content"])
def test_native_isolation_rejects_changed_identity_before_child_post(
    make_service, changed_identity
):
    client = remote(replicas=2, context=64)
    original = ClientHTTPError(500, error_type="server_error")
    changed = False

    def failure(model, rows):
        nonlocal changed
        if not changed:
            changed = True
            if changed_identity == "tokenizer":
                client.builds[model] = "changed-build"
            else:
                client.catalog[0]["tags"] = [
                    "noxi-content-sha256-v1:" + "c" * 64
                    if tag.startswith("noxi-content-sha256-v1:")
                    else tag
                    for tag in client.catalog[0]["tags"]
                ]
        return original

    client.failure = failure
    service, _ = make_service(client, request_parallelism=1, batch_size=2)
    with pytest.raises(LLMBackendError, match="identity_changed|tokenizer_changed"):
        service.embed_texts(["okay", "x" * 80])
    assert changed and len(client.embedding_calls) == 1
    assert client.embedding_calls[0][0] == "embed_0"
    assert not client.token_calls and not service.config.cache_dir.exists()
    assert not service._operations and client.embed_active == client.token_active == 0


def test_midpoint_advice_does_not_claim_fit_on_different_capacity_replica(
    make_service,
):
    client = remote(replicas=2, context=128)
    client.contexts["embed_1"] = 64
    failing_native(client)
    service, _ = make_service(client, request_parallelism=1, batch_size=2)
    source = "α🙂漢字" * 50
    with pytest.raises(EmbeddingChunkSplitRequiredError) as caught:
        service.embed_texts([source])
    assert client.embedding_calls[0][0] == "embed_0"
    pieces = assert_halves(caught.value.inputs[0], source)
    assert not hasattr(caught.value, "budget")
    # These halves fit the larger replica, but not the smaller. An explicit
    # consumer retry may therefore request another halving without pretending
    # that any estimated/exact token admission happened in the library.
    assert all(64 < len("search_document: " + piece) + 1 <= 128 for piece in pieces)
    with pytest.raises(EmbeddingChunkSplitRequiredError) as repeated:
        service.embed_texts(pieces)
    assert repeated.value.failed_input_indices == (0, 1)
    for item in repeated.value.inputs:
        assert_halves(item, pieces[item.input_index])
    assert client.embedding_calls[1][0] == "embed_1"
    assert not client.token_calls and not service.config.cache_dir.exists()
    assert not service._operations


@pytest.mark.parametrize("tokenizer_failure", ["http503", "malformed_tokens"])
def test_unavailable_tokenizer_does_not_affect_native_isolation(
    make_service, tokenizer_failure
):
    client = remote(replicas=1, context=64)
    failing_native(client)
    if tokenizer_failure == "http503":
        client.token_failure = ClientHTTPError(503, error_type="unavailable_error")
    else:

        def malformed(*args, **kwargs):
            pytest.fail("native isolation must not call an unavailable tokenizer")

        client.tokenize = malformed
    service, _ = make_service(client)
    with pytest.raises(EmbeddingChunkSplitRequiredError) as caught:
        service.embed_texts(["x" * 80, "okay"])
    assert_halves(caught.value.inputs[0], "x" * 80)
    assert [row.input_index for row in caught.value.completed] == [1]
    assert not client.token_calls and not service.config.cache_dir.exists()
    assert not service._operations
    assert client.token_active == client.embed_active == 0
