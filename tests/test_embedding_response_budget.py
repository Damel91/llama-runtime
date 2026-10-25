"""Embedding response capacity is independent of input and pending budgets."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Event, Lock, Thread
import time

import pytest

from runtime_llama.adapters.embedding import LlamaCppEmbeddingAdapter
from runtime_llama.configuration import AdapterProfile
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError


LARGE_VECTOR = [1.0] + [0.0] * 1535


@contextmanager
def endpoint(*, compressed=False, truncated=False):
    seen, lock = [], Lock()
    slow_started, release = Event(), Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def send_json(self, body, *, incomplete=False):
            decoded = json.dumps(body).encode()
            wire = gzip.compress(decoded) if compressed else decoded
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header(
                "Content-Length", str(len(wire) + (5 if incomplete else 0))
            )
            if incomplete:
                self.send_header("Connection", "close")
                self.close_connection = True
            if compressed:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            try:
                self.wfile.write(wire)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == "/health":
                self.send_json({"status": "ok"})
            elif self.path == "/v1/models":
                self.send_json({"data": [{"id": "embed"}]})
            else:
                assert self.path.startswith("/props?")
                self.send_json(
                    {"model_alias": "embed", "total_slots": 2, "n_ctx": 2048}
                )

        def do_POST(self):
            wire = self.rfile.read(int(self.headers["Content-Length"]))
            body = json.loads(wire)
            with lock:
                seen.append((self.path, len(wire), body))
            if self.path == "/tokenize":
                self.send_json({"tokens": [1, 2]})
                return
            assert self.path == "/v1/embeddings"
            if body["input"] in (["slow"], ["drip"]):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "100000")
                self.end_headers()
                slow_started.set()
                if body["input"] == ["drip"]:
                    deadline = time.monotonic() + 3
                    while not release.is_set() and time.monotonic() < deadline:
                        try:
                            self.wfile.write(b" ")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            break
                        release.wait(0.03)
                else:
                    release.wait(3)
                return
            self.send_json(
                {
                    "data": [
                        {
                            "index": index,
                            "embedding": [1.0, 0.0]
                            if text == "healthy"
                            else LARGE_VECTOR,
                        }
                        for index, text in enumerate(body["input"])
                    ]
                },
                incomplete=truncated,
            )

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    http.daemon_threads = True
    serving = Thread(target=http.serve_forever, daemon=True)
    serving.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", seen, slow_started
    finally:
        release.set()
        http.shutdown()
        http.server_close()
        serving.join(2)
        assert not serving.is_alive()


def adapter_for(url, **changes):
    return LlamaCppEmbeddingAdapter(
        AdapterProfile.from_config(
            "ingest",
            {
                "adapter_type": "embedding",
                "model": "embed",
                "base_url": url,
                "timeout": 2,
                "batch_size": 1,
                "embedding_input_format": "text",
                "embedding_request_parallelism": 2,
                "embedding_max_request_bytes": 256,
                "embedding_max_response_bytes": 16384,
                **changes,
            },
        )
    )


@pytest.mark.parametrize("compressed", [False, True])
def test_large_valid_vector_accepts_separate_response_budget(compressed):
    with endpoint(compressed=compressed) as (url, seen, _):
        # The response limit is independent of the tiny HTTP input limit.
        # Materialized outputs now also need explicit result/handoff credits.
        adapter = adapter_for(
            url, embedding_max_pending_bytes=1024 * 1024, dimension=1536
        )
        try:
            assert adapter.embed_query("x") == LARGE_VECTOR
            assert len(json.dumps({"data": LARGE_VECTOR})) > 4096
            assert all(size <= 256 for _, size, _ in seen)
            assert len([row for row in seen if row[0] == "/v1/embeddings"]) == 1
            assert all("max_response_bytes" not in body for _, _, body in seen)
            assert not adapter._client._streams
        finally:
            adapter.close()


@pytest.mark.parametrize("compressed", [False, True])
def test_response_overflow_is_typed_and_not_replayed_or_partially_published(
    compressed,
):
    with endpoint(compressed=compressed) as (url, seen, _):
        adapter = adapter_for(url, embedding_max_response_bytes=256)
        try:
            with pytest.raises(LLMBackendError, match="json_response_too_large"):
                adapter.embed_texts(["x", "y"])
            posts = [body for path, _, body in seen if path == "/v1/embeddings"]
            # Independent requests already dispatched may drain, but none is
            # replayed after a decoded-body overflow or exposed as a result.
            assert 1 <= len(posts) <= 2
            assert len({tuple(body["input"]) for body in posts}) == len(posts)
            assert not adapter._client._streams
        finally:
            adapter.close()


def test_concurrent_response_overflow_does_not_fail_healthy_owner():
    with endpoint() as (url, seen, _):
        adapter = adapter_for(url, embedding_max_response_bytes=256)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                oversized = executor.submit(adapter.embed_query, "oversized")
                healthy = executor.submit(adapter.embed_query, "healthy")
                with pytest.raises(LLMBackendError, match="json_response_too_large"):
                    oversized.result(2)
                assert healthy.result(2) == [1.0, 0.0]
            assert sorted(
                body["input"][0] for path, _, body in seen if path == "/v1/embeddings"
            ) == ["healthy", "oversized"]
            assert not adapter._client._streams
        finally:
            adapter.close()


@pytest.mark.parametrize("compressed", [False, True])
def test_incomplete_embedding_response_fails_without_retry_or_partial_results(
    compressed,
):
    with endpoint(compressed=compressed, truncated=True) as (url, seen, _):
        adapter = adapter_for(url)
        try:
            with pytest.raises(LLMBackendError):
                adapter.embed_query("x")
            assert len([row for row in seen if row[0] == "/v1/embeddings"]) == 1
            assert not adapter._client._streams
        finally:
            adapter.close()


@pytest.mark.parametrize("termination", ["cancel", "timeout"])
@pytest.mark.parametrize("style", ["slow", "drip"])
def test_separate_response_budget_keeps_idle_body_cancellable_and_drained(
    termination, style
):
    with endpoint() as (url, seen, started):
        adapter = adapter_for(url)
        stop = Event()
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                slow = executor.submit(
                    adapter.embed_query,
                    style,
                    cancel=stop.is_set,
                    timeout=0.3 if termination == "timeout" else 2,
                )
                assert started.wait(1)
                healthy = executor.submit(adapter.embed_query, "healthy")
                if termination == "cancel":
                    stop.set()
                expected = (
                    LLMCancelledError if termination == "cancel" else LLMTimeoutError
                )
                with pytest.raises(expected):
                    slow.result(1)
                assert healthy.result(1) == [1.0, 0.0]
            assert len([row for row in seen if row[0] == "/v1/embeddings"]) == 2
            assert not adapter._client._streams
        finally:
            stop.set()
            adapter.close()
