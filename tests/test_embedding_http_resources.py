"""Native embedding/tokenizer response bounds apply before JSON decoding."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread

import pytest

from runtime_llama.client import LlamaCppClient
from runtime_llama.errors import LLMBackendError


@pytest.fixture
def resource_server():
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            size = 1024 if body.get("large") else 2
            payload = (
                {"tokens": list(range(size))}
                if self.path == "/tokenize"
                else {"data": [{"index": 0, "embedding": [0.0] * size}]}
            )
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", bodies
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


@pytest.mark.parametrize("method", ["tokenize", "embeddings"])
def test_native_helpers_enforce_response_budget_without_forwarding_it(
    resource_server, method
):
    endpoint, bodies = resource_server
    client = LlamaCppClient(endpoint, timeout=2)
    argument = "text" if method == "tokenize" else ["text"]
    try:
        helper = getattr(client, method)
        assert isinstance(helper(argument, max_response_bytes=128), dict)
        with pytest.raises(LLMBackendError, match="json_response_too_large"):
            helper(argument, large=True, max_response_bytes=128, cancel_wait=True)
        assert len(bodies) == 2
        assert all("max_response_bytes" not in body for body in bodies)
        assert not client._streams
    finally:
        client.close()
