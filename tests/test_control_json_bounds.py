"""Control JSON limits cover wire, inflation, framing and real cleanup."""

from __future__ import annotations

import gzip
import json
from threading import Event
import time
import zlib

import pytest

from runtime_llama.client import (
    ClientHTTPError,
    LlamaCppClient,
    _bounded_json_body,
)
import runtime_llama.client as transport
from runtime_llama.errors import LLMBackendError, LLMTimeoutError
from test_optional_stream_idle_transport import endpoint as endpoint


def send(handler, body, *, encoding=None, status=200, chunked=False, extra=0):
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    if encoding:
        handler.send_header("Content-Encoding", encoding)
    if chunked:
        handler.send_header("Transfer-Encoding", "chunked")
    else:
        handler.send_header("Content-Length", str(len(body) + extra))
    handler.end_headers()
    try:
        if chunked:
            for index in range(0, len(body), 19):
                part = body[index : index + 19]
                handler.wfile.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
            handler.wfile.write(b"0\r\n\r\n")
        else:
            handler.wfile.write(body)
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        # A bounded reader intentionally closes before an excessive body ends.
        return
    if extra:
        handler.close_connection = True


@pytest.mark.parametrize("method", ["health", "models", "props"])
@pytest.mark.parametrize("style", ["plain", "gzip", "chunked"])
def test_control_helpers_bound_before_json_decode_and_never_retry(
    endpoint, monkeypatch, method, style
):
    payload = json.dumps({"private": "x" * 10000}).encode()
    encoded = gzip.compress(payload) if style == "gzip" else payload
    decoded = []
    original = transport.requests.Response.json

    def decode(response, **kwargs):
        decoded.append(True)
        return original(response, **kwargs)

    monkeypatch.setattr(transport.requests.Response, "json", decode)
    base_url, observed = endpoint(
        lambda handler, _body: send(
            handler,
            encoded,
            encoding="gzip" if style == "gzip" else None,
            chunked=style == "chunked",
        )
    )
    with LlamaCppClient(base_url, trust_env=False, timeout=1) as api:
        with pytest.raises(LLMBackendError, match="json_response_too_large"):
            getattr(api, method)(max_response_bytes=256, cancel_wait=True)
        assert not api._streams
    assert decoded == []
    assert len(observed) == 1


def test_inflater_receives_hard_output_limit_before_bomb_materializes(monkeypatch):
    compressed = gzip.compress(b"x" * 2_000_000)
    assert len(compressed) < 4096
    limits = []
    factory = zlib.decompressobj

    class Decoder:
        def __init__(self, *args):
            self.real = factory(*args)

        def decompress(self, data, max_length=0):
            limits.append(max_length)
            result = self.real.decompress(data, max_length)
            assert len(result) <= 4097
            return result

        def __getattr__(self, name):
            return getattr(self.real, name)

    monkeypatch.setattr(transport.zlib, "decompressobj", Decoder)
    with pytest.raises(LLMBackendError, match="json_response_too_large"):
        _bounded_json_body(iter([compressed]), "gzip", 4096)
    assert limits == [4097]


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "raw-deflate"])
def test_compressed_json_supports_fragmented_wire_and_integrity(encoding):
    body = b'{"value":"' + b"x" * 200 + b'"}'
    if encoding == "gzip":
        encoded = gzip.compress(body)
    elif encoding == "deflate":
        encoded = zlib.compress(body)
    else:
        compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
        encoded = compressor.compress(body) + compressor.flush()
    name = "deflate" if encoding == "raw-deflate" else encoding
    assert _bounded_json_body(iter(bytes([c]) for c in encoded), name, 512) == body
    with pytest.raises((LLMBackendError, zlib.error)):
        _bounded_json_body(iter([encoded[:-1]]), name, 512)


def test_gzip_members_share_one_total_envelope():
    body = gzip.compress(b'{"value":') + gzip.compress(b"1}")
    assert _bounded_json_body(iter([body]), "gzip", 128) == b'{"value":1}'
    with pytest.raises(LLMBackendError, match="json_response_too_large"):
        _bounded_json_body(iter([gzip.compress(b"x" * 300)] * 2), "gzip", 512)


def test_tiny_gzip_exact_decoded_boundary_and_finite_empty_member_wire_guard():
    body = b'{"ok":1}'
    encoded = gzip.compress(body)
    assert len(encoded) > len(body)
    assert _bounded_json_body(iter([encoded]), "gzip", len(body)) == body
    padded = gzip.compress(b"") * 4000 + encoded
    with pytest.raises(LLMBackendError, match="json_response_too_large"):
        _bounded_json_body(iter([padded]), "gzip", len(body))


@pytest.mark.parametrize("style", ["truncated", "malformed", "bad-gzip", "br"])
def test_invalid_framing_or_encoding_drains_and_does_not_replay(endpoint, style):
    base_url, observed = endpoint(
        lambda handler, _body: send(
            handler,
            b"invalid" if style in {"malformed", "bad-gzip"} else b'{"ok":1}',
            encoding="gzip" if style == "bad-gzip" else "br" if style == "br" else None,
            extra=10 if style == "truncated" else 0,
        )
    )
    with LlamaCppClient(base_url, trust_env=False, timeout=1) as api:
        with pytest.raises(LLMBackendError):
            api.health(max_response_bytes=128, cancel_wait=True)
        assert not api._streams
    assert len(observed) == 1


def test_error_body_has_small_envelope_and_preserves_only_status(endpoint, monkeypatch):
    decoded = []
    monkeypatch.setattr(
        transport.requests.Response,
        "json",
        lambda *_args, **_kwargs: decoded.append(True),
    )
    base_url, observed = endpoint(
        lambda handler, _body: send(handler, b"x" * 100000, status=503)
    )
    with LlamaCppClient(base_url, trust_env=False) as api:
        with pytest.raises(ClientHTTPError) as caught:
            api.health(max_response_bytes=1024 * 1024, cancel_wait=True)
        assert caught.value.status_code == 503
        assert caught.value.error_code is None
        assert not api._streams
    assert decoded == []
    assert len(observed) == 1


def test_opaque_borrowed_response_cannot_silently_drop_finite_bound():
    class Response:
        status_code = 200

        def __init__(self):
            self.decoded = False
            self.closed = False

        def json(self):
            self.decoded = True
            return {"ok": True}

        def close(self):
            self.closed = True

    class Session:
        def __init__(self):
            self.response = Response()
            self.calls = 0

        def request(self, **kwargs):
            self.calls += 1
            return self.response

    session = Session()
    with LlamaCppClient("http://chosen.example", session=session) as api:
        with pytest.raises(LLMBackendError, match="bounded_response_required"):
            api.health(max_response_bytes=128, cancel_wait=True)
        assert not session.response.decoded
        assert session.response.closed
        assert api.health() == {"ok": True}
    assert session.calls == 2


def test_finite_opaque_error_keeps_status_without_decoding_unbounded_body():
    class Response:
        status_code = 503
        decoded = False
        closed = False

        def json(self):
            self.decoded = True
            return {"error": {"code": "must_not_be_observed"}}

        def close(self):
            self.closed = True

    class Session:
        response = Response()
        calls = 0

        def request(self, **kwargs):
            self.calls += 1
            return self.response

    borrowed = Session()
    with LlamaCppClient("http://chosen.example", session=borrowed) as api:
        with pytest.raises(ClientHTTPError) as caught:
            api.health(max_response_bytes=128, cancel_wait=True)
        assert caught.value.status_code == 503
        assert caught.value.error_code is None
    assert borrowed.calls == 1
    assert not borrowed.response.decoded
    assert borrowed.response.closed


def test_bounded_control_slow_drip_keeps_original_deadline_and_drains(endpoint):
    finished = Event()

    def respond(handler, _body):
        handler.send_response(200)
        handler.send_header("Content-Length", "100000")
        handler.end_headers()
        try:
            for _ in range(1000):
                handler.wfile.write(b" ")
                handler.wfile.flush()
                time.sleep(0.01)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            finished.set()

    base_url, observed = endpoint(respond)
    start = time.monotonic()
    with LlamaCppClient(base_url, trust_env=False, timeout=0.15) as api:
        with pytest.raises(LLMTimeoutError):
            api.health(max_response_bytes=200000, cancel_wait=True)
        assert not api._streams
    assert time.monotonic() - start < 1.5
    assert finished.wait(1)
    assert len(observed) == 1
