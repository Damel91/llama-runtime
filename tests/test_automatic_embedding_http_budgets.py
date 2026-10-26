"""Public automatic-service transport budgets and direct queue reservations."""

from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import math
import struct

import pytest

from runtime_llama import (
    AutomaticEmbeddingConfig,
    AutomaticEmbeddingService,
    LlamaCppClient,
)
from runtime_llama.errors import LLMBackendError

from test_automatic_embedding import remote
from test_embedding_contracts import eventually, vector
from test_optional_stream_idle_transport import endpoint as endpoint


_MIB = 1024 * 1024


@pytest.fixture(scope="module")
def large_vectors():
    """Normalized float32 values retain long JSON spellings after promotion."""
    norm = math.sqrt(2048 * (0.1 * 0.1 + 0.2 * 0.2))
    pair = [
        struct.unpack("f", struct.pack("f", value / norm))[0] for value in (0.1, 0.2)
    ]
    base = pair * 2048
    vectors = []
    for index in range(128):
        row = list(base)
        row[index] = -row[index]
        vectors.append(row)
    body = json.dumps(
        {
            "object": "list",
            "data": [
                {"index": index, "embedding": vectors[index]}
                for index in reversed(range(128))
            ],
        },
        separators=(",", ":"),
    ).encode()
    assert 8 * _MIB < len(body) < 16 * _MIB
    assert len(gzip.compress(body)) < 8 * _MIB
    assert abs(math.sqrt(math.fsum(value * value for value in base)) - 1) <= 1e-6
    return vectors, body


@pytest.fixture
def facade_endpoint(endpoint, large_vectors):
    def start(*, compressed=False, incomplete=False, large=False):
        vectors, _ = large_vectors
        server = remote(replicas=1, context=2048)
        headers = []

        def send(handler, body, *, compress=False, truncate=False):
            wire = gzip.compress(body) if compress else body
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header(
                "Content-Length", str(len(wire) + (17 if truncate else 0))
            )
            if compress:
                handler.send_header("Content-Encoding", "gzip")
            if truncate or handler.headers.get("Connection") == "close":
                handler.send_header("Connection", "close")
            if truncate:
                handler.close_connection = True
            handler.end_headers()
            handler.wfile.write(wire)
            handler.wfile.flush()

        def respond(handler, body):
            handler.connection.settimeout(3)
            headers.append(handler.headers.get("Connection"))
            if handler.path == "/health":
                payload = server.health()
            elif handler.path == "/v1/models":
                payload = server.models()
            elif handler.path.startswith("/props"):
                payload = server.props(model="embed_0")
            else:
                assert handler.path == "/v1/embeddings", handler.path
                assert isinstance(body["input"], list)
                assert all(isinstance(text, str) for text in body["input"])
                rows = body["input"]
                if large:
                    # Reply to the actual requested cardinality, including the
                    # one-row dimension bootstrap. Distinct source vectors
                    # remain correlated even if the request is safely split.
                    payload = {
                        "object": "list",
                        "data": [
                            {
                                "index": index,
                                "embedding": vectors[
                                    int(rows[index].removeprefix("item-"))
                                    if rows[index].startswith("item-")
                                    else 0
                                ],
                            }
                            for index in reversed(range(len(rows)))
                        ],
                    }
                    send(
                        handler,
                        json.dumps(payload, separators=(",", ":")).encode(),
                        compress=compressed,
                        truncate=incomplete,
                    )
                    return
                payload = {
                    "data": [
                        {"index": index, "embedding": [1.0, 0.0]}
                        for index in range(len(body["input"]))
                    ]
                }
            send(
                handler,
                json.dumps(payload).encode(),
                compress=compressed and handler.path == "/v1/embeddings",
                truncate=incomplete and handler.path == "/v1/embeddings",
            )

        base_url, seen = endpoint(respond, keepalive=True)
        return base_url, seen, headers

    return start


def service_for(base_url, **changes):
    return AutomaticEmbeddingService(
        AutomaticEmbeddingConfig(
            base_url=base_url,
            timeout=5,
            batch_size=128,
            request_parallelism=1,
            embedding_prefixes={"document": "", "query": ""},
            **changes,
        )
    )


def assert_drained(service):
    adapter = service._adapter
    assert not service._operations and not adapter._calls
    assert not service._client._streams
    scheduler = adapter._scheduler.snapshot()
    assert scheduler["in_flight"] == scheduler["queued"] == 0
    state = adapter._engine_health_snapshot()
    for key in (
        "pending_bytes",
        "pending_tokens",
        "retained_result_bytes",
        "result_reserved_bytes",
        "embedding_in_flight",
        "tokenization_in_flight",
        "preparation_results_waiting",
        "http_results_waiting",
    ):
        assert state[key] == 0, (key, state)
    if adapter.embedding_max_pending_bytes is not None:
        assert state["peak_pending_bytes"] <= adapter.embedding_max_pending_bytes
    if adapter.embedding_max_pending_tokens is not None:
        assert state["peak_pending_tokens"] <= adapter.embedding_max_pending_tokens


def posts(seen):
    return [body for path, body, _ in seen if path == "/v1/embeddings"]


@pytest.mark.parametrize("compressed", [False, True])
def test_public_explicit_response_cap_rejects_large_body_and_drains_without_replay(
    facade_endpoint, large_vectors, compressed
):
    expected, _ = large_vectors
    base_url, seen, _ = facade_endpoint(compressed=compressed, large=True)
    # Isolate the configured JSON cap from conservative decoded-vector and
    # handoff admission. This budget funds the same full 128-row wire payload
    # for both the rejecting 8 MiB and accepting 16 MiB configurations.
    service = service_for(
        base_url,
        embedding_capacity_limits=True,
        embedding_max_response_bytes=8 * _MIB,
        embedding_max_pending_bytes=512 * _MIB,
    )
    incoming = [f"item-{index:03}" for index in range(128)]
    try:
        warmup = service.embed_query("warmup")
        assert warmup == pytest.approx(expected[0], abs=1e-5)
        assert_drained(service)
        seen.clear()
        with pytest.raises(LLMBackendError, match="json_response_too_large"):
            service.embed_texts(incoming)
        assert posts(seen) == [{"input": incoming, "model": "embed_0"}]
        assert_drained(service)
        assert service.embed_query("healthy") == pytest.approx(expected[0], abs=1e-5)
        assert len(posts(seen)) == 2
        assert_drained(service)
    finally:
        service.close()
    assert not any(path == "/tokenize" for path, _, _ in seen)


def test_public_larger_response_cap_accepts_ordered_128_by_4096_vectors(
    facade_endpoint, large_vectors
):
    expected, _ = large_vectors
    base_url, seen, _ = facade_endpoint(large=True)
    service = service_for(
        base_url,
        embedding_capacity_limits=True,
        embedding_max_response_bytes=16 * _MIB,
        embedding_max_pending_bytes=512 * _MIB,
    )
    incoming = [f"item-{index:03}" for index in range(128)]
    try:
        warmup = service.embed_query("warmup")
        assert warmup == pytest.approx(expected[0], abs=1e-5)
        assert_drained(service)
        seen.clear()
        result = service.embed_texts(incoming)
        assert len(result) == 128
        for actual, source in zip(result, expected):
            assert len(actual) == 4096
            assert all(math.isfinite(value) for value in actual)
            assert max(abs(a - b) for a, b in zip(actual, source)) <= 1e-5
            assert (
                abs(math.sqrt(math.fsum(value * value for value in actual)) - 1) <= 1e-6
            )
        assert posts(seen) == [{"input": incoming, "model": "embed_0"}]
        assert_drained(service)
    finally:
        service.close()


@pytest.mark.parametrize("compressed", [False, True])
def test_public_incomplete_response_fails_once_and_drains(facade_endpoint, compressed):
    base_url, seen, _ = facade_endpoint(compressed=compressed, incomplete=True)
    service = service_for(base_url, embedding_max_response_bytes=16 * _MIB)
    try:
        with pytest.raises(LLMBackendError, match="transport_failure"):
            service.embed_query("incomplete")
        assert posts(seen) == [{"input": ["incomplete"], "model": "embed_0"}]
        assert_drained(service)
    finally:
        service.close()


@pytest.mark.parametrize("policy", ["close", "reuse"])
def test_public_owned_http_policy_is_selected_and_connects(facade_endpoint, policy):
    base_url, seen, headers = facade_endpoint()
    service = service_for(base_url, http_connection_policy=policy)
    try:
        assert service.embed_query("first") == [1.0, 0.0]
        assert service.embed_query("second") == [1.0, 0.0]
        assert service._client.connection_policy == policy
        assert set(headers) == {"close" if policy == "close" else "keep-alive"}
        health_peers = [peer for path, _, peer in seen if path == "/health"]
        assert len(health_peers) >= 2
        if policy == "close":
            assert len(set(health_peers)) == len(health_peers)
        else:
            assert len(set(health_peers)) < len(health_peers)
        assert_drained(service)
    finally:
        service.close()
    assert service._client._closed


def test_public_injected_client_keeps_policy_and_ownership(facade_endpoint):
    base_url, seen, headers = facade_endpoint()
    client = LlamaCppClient(
        base_url, trust_env=False, connection_policy="reuse", timeout=5
    )
    service = AutomaticEmbeddingService(
        AutomaticEmbeddingConfig(base_url=base_url, http_connection_policy="close"),
        client=client,
    )
    try:
        assert service.embed_query("source") == [1.0, 0.0]
        assert client.connection_policy == "reuse"
        assert set(headers) == {"keep-alive"}
        assert_drained(service)
        service.close()
        assert not client._closed
        assert client.health() == {"status": "ok"}
        assert len(posts(seen)) == 1
    finally:
        service.close()
        client.close()


@pytest.mark.parametrize(("limit", "cardinality"), [(None, 256), (65536 * 256, 256)])
def test_public_direct_inputs_do_not_reserve_a_whole_context_each(limit, cardinality):
    client = remote(replicas=1, context=65536)
    client.catalog[0]["status"]["args"] = [
        "llama-server",
        "--batch-size",
        "65536",
        "--ubatch-size",
        "65536",
    ]
    changes = {} if limit is None else {"embedding_max_pending_tokens": limit}
    service = AutomaticEmbeddingService(
        AutomaticEmbeddingConfig(
            base_url="http://runtime.invalid:8080",
            request_parallelism=1,
            batch_size=256,
            normalize=False,
            **changes,
        ),
        client=client,
    )
    incoming = [f"short-{index}" for index in range(256)]
    try:
        # Measure token-reservation cardinality after the independent one-row
        # dimension bootstrap has established the decoded result allowance.
        assert service.embed_query("warmup") == vector("search_query: warmup")
        client.embedding_calls.clear()
        assert service.embed_texts(incoming) == [
            vector("search_document: " + text) for text in incoming
        ]
        assert [len(rows) for _, rows in client.embedding_calls] == [cardinality] * (
            256 // cardinality
        )
        assert not client.token_calls
        adapter = service._adapter
        state = adapter._engine_health_snapshot()
        assert state["pending_bytes"] == state["pending_tokens"] == 0
        assert state["peak_pending_tokens"] == 0
        assert state["working_bytes_limit"] is None
        assert not adapter._calls
        assert adapter._scheduler.snapshot()["in_flight"] == 0
    finally:
        service.close()


def test_high_context_owners_share_explicit_batches_without_fake_token_charges():
    client = remote(replicas=1, context=65536)
    client.catalog[0]["status"]["args"] = [
        "llama-server",
        "--batch-size",
        "65536",
        "--ubatch-size",
        "65536",
    ]
    client.embed_gate.clear()
    service = AutomaticEmbeddingService(
        AutomaticEmbeddingConfig(
            base_url="http://runtime.invalid:8080",
            request_parallelism=1,
            batch_size=4,
            embedding_capacity_limits=True,
            embedding_max_pending_tokens=4 * 65536,
            embedding_max_pending_bytes=128 * 1024,
            normalize=False,
        ),
        client=client,
    )
    first = [f"owner-a-{index}" for index in range(64)]
    second = [f"owner-b-{index}" for index in range(64)]
    try:
        with ThreadPoolExecutor(2) as executor:
            a = executor.submit(service.embed_texts, first)
            assert client.embed_started.wait(2)
            b = executor.submit(service.embed_texts, second)
            eventually(lambda: len(service._adapter._calls) == 2)
            eventually(
                lambda: (
                    service._adapter._engine_health_snapshot()["retained_plans"] >= 2
                )
            )
            assert not a.done() and not b.done()
            client.embed_gate.set()
            assert a.result(4) == [vector("search_document: " + text) for text in first]
            assert b.result(4) == [
                vector("search_document: " + text) for text in second
            ]
        submitted = [text for _, rows in client.embedding_calls for text in rows]
        assert len(submitted) == len(set(submitted)) == 128
        assert min(i for i, text in enumerate(submitted) if "owner-b-" in text) < max(
            i for i, text in enumerate(submitted) if "owner-a-" in text
        )
        state = service._adapter._engine_health_snapshot()
        assert state["pending_bytes"] == state["pending_tokens"] == 0
        assert state["peak_pending_tokens"] == 0
        assert state["peak_pending_bytes"] <= 128 * 1024
        assert client.embed_active == 0 and not client.token_calls
    finally:
        client.embed_gate.set()
        service.close()
