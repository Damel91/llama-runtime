"""Isolated HTTP client adapted from llamatelemetry's MIT LlamaCppClient.

Source: llamatelemetry/llamatelemetry, commit
 a0a1df88e3b143d1e38e241b2599e82c56092a0b, llamatelemetry/api/client.py.
The constructor, headers, request dispatch and endpoint/payload mappings derive
from that client. Typed projections and package initialization are excluded;
SSE decoding, cancellation, closure and sanitized errors are adaptations.
No model/server bootstrap is performed. [DONE] terminates transport only.

Upstream copyright (c) 2024 Waqas Muhammad; MIT license reproduced in
_vendor/llamatelemetry.LICENSE. Upstream source SHA-256:
7f9c93f9852012ad08f992ea1ee7b5c1260789556af05410398c24250c5f878c.
"""

from __future__ import annotations

import codecs
import json as json_module
import math
from queue import Empty, Full, Queue
import re
from threading import Event, Lock, Thread
import time
from typing import Any, Callable, Iterator
from urllib.parse import urlencode, urlsplit
import weakref

import requests
from urllib3.exceptions import NewConnectionError
from urllib3.exceptions import TimeoutError as Urllib3TimeoutError

from .errors import LLMBackendError, LLMCancelledError, LLMTimeoutError

CancellationProbe = Callable[[], bool]
_SAFE_CODE = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
_END = object()
_MAX_FRAME = 8 * 1024 * 1024


class ClientHTTPError(LLMBackendError):
    """HTTP status and bounded structured codes; never provider message/body."""

    def __init__(
        self,
        status_code: int,
        *,
        error_code: str | None = None,
        error_type: str | None = None,
    ) -> None:
        super().__init__(f"llama_server_http_error:status={status_code}")
        self.status_code = status_code
        self.error_code = error_code
        self.error_type = error_type


def _cancelled(cancel: CancellationProbe | None) -> None:
    if cancel is None:
        return
    try:
        requested = bool(cancel())
    except Exception as exc:
        raise LLMBackendError(
            "llama_server_cancellation_probe_failed"
        ) from RuntimeError(type(exc).__name__)
    if requested:
        raise LLMCancelledError()


def _timeout(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("timeout must be a positive finite number")
    return result


def _is_timeout(exc: BaseException, depth: int = 0) -> bool:
    # urllib3 also subclasses connection-refused/DNS failures from TimeoutError.
    if isinstance(exc, NewConnectionError):
        return False
    if isinstance(exc, (requests.exceptions.Timeout, Urllib3TimeoutError)):
        return True
    if depth >= 8:
        return False
    nested = (exc.__cause__, exc.__context__, *exc.args)
    return any(
        isinstance(item, BaseException) and _is_timeout(item, depth + 1)
        for item in nested
        if item is not exc
    )


def _safe_failure(exc: Exception) -> Exception:
    if isinstance(exc, (LLMBackendError, LLMTimeoutError, LLMCancelledError)):
        return exc
    if _is_timeout(exc):
        error = LLMTimeoutError("llama_server_transport_timeout")
    else:
        error = LLMBackendError("llama_server_transport_failure")
    # Preserve the failure classification without exposing request URLs, bodies,
    # authorization headers, JSON source or raw requests exception messages.
    error.__cause__ = RuntimeError(type(exc).__name__)
    error.__suppress_context__ = True
    return error


def _code(value: Any) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    return value if isinstance(value, str) and _SAFE_CODE.fullmatch(value) else None


def _http_error(
    response: Any, read_json: Callable[[], Any] | None = None
) -> ClientHTTPError:
    code = kind = None
    try:
        payload = (read_json or response.json)()
        error = payload.get("error", payload) if isinstance(payload, dict) else {}
        if isinstance(error, dict):
            code, kind = _code(error.get("code")), _code(error.get("type"))
    except Exception:
        pass
    return ClientHTTPError(int(response.status_code), error_code=code, error_type=kind)


def _sse(chunks: Iterator[bytes | str]) -> Iterator[dict[str, Any]]:
    """Decode SSE data across UTF-8, CR/LF and HTTP chunk boundaries."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    parts: list[str] = []
    line_size = frame_size = 0
    data: list[str] = []
    first = True
    skip_lf = False
    done = False

    def frame() -> dict[str, Any] | None:
        nonlocal data, done, frame_size
        frame_size = 0
        if not data:
            return None
        raw = "\n".join(data)
        data = []
        if raw.strip() == "[DONE]":
            done = True
            return None
        try:

            def reject_constant(_value: str) -> Any:
                raise ValueError("nonfinite JSON number")

            payload = json_module.loads(raw, parse_constant=reject_constant)
        except (ValueError, TypeError) as exc:
            raise LLMBackendError("llama_server_malformed_sse_json") from RuntimeError(
                type(exc).__name__
            )
        if not isinstance(payload, dict):
            raise LLMBackendError("llama_server_sse_payload_not_object")
        return payload

    def line(value: str) -> dict[str, Any] | None:
        nonlocal frame_size
        if value == "":
            return frame()
        if value.startswith("data:") or value == "data":
            value = value[5:] if value.startswith("data:") else ""
            value = value[1:] if value.startswith(" ") else value
            data.append(value)
            frame_size += len(value) + 1
        return None

    def texts() -> Iterator[str]:
        try:
            for chunk in chunks:
                yield (
                    decoder.decode(chunk, final=False)
                    if isinstance(chunk, bytes)
                    else chunk
                )
            yield decoder.decode(b"", final=True)
        except UnicodeError as exc:
            raise LLMBackendError("llama_server_sse_invalid_utf8") from RuntimeError(
                type(exc).__name__
            )

    for text in texts():
        if first and text:
            text = text.removeprefix("\ufeff")
            first = False
        if skip_lf and text:
            text = text[1:] if text.startswith("\n") else text
            skip_lf = False
        if not text:
            continue
        segments = re.split(r"(\r\n|\r|\n)", text)
        for index, segment in enumerate(segments):
            if index % 2 == 0:
                if segment:
                    parts.append(segment)
                    line_size += len(segment)
                    if line_size + frame_size > _MAX_FRAME:
                        raise LLMBackendError("llama_server_sse_frame_too_large")
            else:
                payload = line("".join(parts))
                parts = []
                line_size = 0
                skip_lf = segment == "\r" and index == len(segments) - 2
                if payload is not None:
                    yield payload
                if done:
                    return
    if parts:
        payload = line("".join(parts))
        if payload is not None:
            yield payload
    if not done:
        payload = frame()
        if payload is not None:
            yield payload


class _HTTPCall:
    """An exclusively leased session and its response stay owned by one reader."""

    def __init__(
        self,
        client: LlamaCppClient,
        method: str,
        path: str,
        body: Any,
        timeout: float,
        cancel: CancellationProbe | None,
        cancel_wait: bool,
        max_response_bytes: int | None = None,
    ) -> None:
        self._client, self._method, self._path = client, method, path
        self._body, self._timeout, self._cancel = body, timeout, cancel
        self._cancel_wait = cancel_wait
        self._max_response_bytes = max_response_bytes
        self._stop = Event()
        self._session: Any = None
        self._response: Any = None
        self._close_lock = Lock()
        self._close_started = False
        self._cleanup_started = Event()
        self._response_closed = Event()
        self._closer: Thread | None = None
        self._worker: Thread

    def start(self) -> None:
        self._worker.start()

    def _close_response(self) -> None:
        with self._close_lock:
            response = self._response
            if response is None or self._close_started:
                return
            self._close_started = True
        try:
            response.close()
        except Exception:
            pass
        finally:
            self._response_closed.set()

    def _send(self) -> Any:
        self._session = self._client._lease_session()
        if self._stop.is_set():
            return None
        # Stream the HTTP body even for JSON: response.json() runs in this
        # reader, while the caller can notice cancellation during body I/O.
        self._response = self._client._send(
            self._session,
            self._method,
            self._path,
            self._body,
            self._timeout,
            self._stop,
        )
        return self._response

    def _finish(self) -> None:
        self._close_response()
        if self._response is not None:
            # Another closer can still be waiting for the reader's socket
            # operation. Never return that session to the pool before it ends.
            self._response_closed.wait()
        if self._session is not None:
            self._client._release_session(self._session)

    def _chunks(self) -> Iterator[bytes | str]:
        """Check stop between network reads, including a body that never ends.

        read1 returns available data after one underlying read instead of
        waiting to fill a large buffer. Older urllib3 falls back to one-byte
        iter_content. The HTTP timeout still bounds a completely idle read.
        """
        raw = getattr(self._response, "raw", None)
        read1 = getattr(raw, "read1", None)
        if callable(read1):
            while not self._stop.is_set():
                chunk = read1(65536, decode_content=True)
                if self._stop.is_set() or not chunk:
                    return
                yield chunk
        else:
            for chunk in self._response.iter_content(chunk_size=1):
                if self._stop.is_set():
                    return
                yield chunk

    def _read_json(self) -> Any:
        if (
            isinstance(self._response, requests.Response)
            and getattr(self._response, "raw", None) is not None
        ):
            body = bytearray()
            for chunk in self._chunks():
                if (
                    self._max_response_bytes is not None
                    and len(body) + len(chunk) > self._max_response_bytes
                ):
                    raise LLMBackendError("llama_server_json_response_too_large")
                body.extend(chunk)
            if self._stop.is_set():
                raise LLMCancelledError()
            # Preserve requests' established encoding/JSON decoding behavior,
            # after cooperatively reading the body instead of response.json()
            # performing one uninterruptible full-body read.
            self._response._content = bytes(body)
            self._response._content_consumed = True
        elif isinstance(self._response, requests.Response):
            content = self._response.content
            if (
                self._max_response_bytes is not None
                and len(content) > self._max_response_bytes
            ):
                raise LLMBackendError("llama_server_json_response_too_large")
        return self._response.json()

    def _signal_close(self) -> None:
        self._stop.set()
        with self._close_lock:
            if self._cleanup_started.is_set():
                return
            self._cleanup_started.set()
            self._closer = Thread(
                target=self._close_response,
                daemon=True,
                name="runtime_llama_http_close",
            )
            self._closer.start()

    def _join(self, timeout: float | None = None) -> None:
        self._worker.join(timeout=timeout)
        if self._closer is not None:
            self._closer.join(timeout=timeout)

    def close(self) -> None:
        self._signal_close()
        # A finite HTTP timeout bounds a read that requests cannot interrupt
        # before headers arrive. Callers choosing cancel_wait retain their
        # concurrency permit until cleanup finishes; ordinary cancel is prompt.
        self._join(timeout=None if self._cancel_wait else 0.1)


class _JSONCall(_HTTPCall):
    """A single JSON response, with cancellable waiting and tracked cleanup."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self._done = Event()
        self._result: Any = None
        self._error: Exception | None = None
        self._worker = Thread(
            target=self._read, daemon=True, name="runtime_llama_json_reader"
        )

    def _read(self) -> None:
        try:
            if self._stop.is_set() or self._send() is None:
                return
            if self._stop.is_set():
                return
            if not 200 <= int(self._response.status_code) < 300:
                raise _http_error(self._response, self._read_json)
            payload = self._read_json()
            if not isinstance(payload, (dict, list)):
                raise LLMBackendError("llama_server_json_payload_not_object_or_array")
            self._result = payload
        except Exception as exc:
            if not self._stop.is_set():
                self._error = _safe_failure(exc)
        finally:
            self._finish()
            self._done.set()
            self._client._forget(self)

    def result(self) -> dict[str, Any] | list[Any]:
        deadline = time.monotonic() + self._timeout
        try:
            while True:
                _cancelled(self._cancel)
                if self._stop.is_set():
                    raise LLMBackendError("llama_server_client_closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LLMTimeoutError("llama_server_json_response_timeout")
                if not self._done.wait(timeout=min(0.05, remaining)):
                    continue
                _cancelled(self._cancel)
                if self._stop.is_set():
                    raise LLMBackendError("llama_server_client_closed")
                if self._error is not None:
                    raise self._error
                return self._result
        except BaseException:
            self.close()
            raise


class _SSEStream(_HTTPCall, Iterator[dict[str, Any]]):
    """Per-call SSE reader with idle cancellation and tracked cleanup."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self._queue: Queue[Any] = Queue(maxsize=64)
        self._worker = Thread(
            target=self._read, daemon=True, name="runtime_llama_sse_reader"
        )

    def __iter__(self) -> _SSEStream:
        return self

    def _put(self, item: Any) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return
            except Full:
                continue

    def _read(self) -> None:
        try:
            if self._stop.is_set() or self._send() is None:
                return
            if self._stop.is_set():
                return
            if not 200 <= int(self._response.status_code) < 300:
                raise _http_error(self._response, self._read_json)
            for payload in _sse(self._chunks()):
                if self._stop.is_set():
                    break
                self._put(payload)
        except Exception as exc:
            if not self._stop.is_set():
                self._put(_safe_failure(exc))
        finally:
            self._finish()
            self._put(_END)
            self._client._forget(self)

    def __next__(self) -> dict[str, Any]:
        if self._stop.is_set():
            raise StopIteration
        deadline = time.monotonic() + self._timeout
        try:
            while True:
                _cancelled(self._cancel)
                if self._stop.is_set():
                    raise StopIteration
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LLMTimeoutError("llama_server_stream_idle_timeout")
                try:
                    value = self._queue.get(timeout=min(0.05, remaining))
                except Empty:
                    continue
                _cancelled(self._cancel)
                if value is _END:
                    self.close()
                    raise StopIteration
                if isinstance(value, Exception):
                    raise value
                return value
        except BaseException:
            self.close()
            raise


class LlamaCppClient:
    """HTTP only: raw llama-server native and /v1 responses, no implicit retry."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 600.0,
        verify_ssl: bool = True,
        session: Any = None,
        *,
        trust_env: bool = True,
    ) -> None:
        invalid_url = (
            "base_url must be an HTTP(S) server URL without "
            "credentials/query/fragment/whitespace/backslashes"
        )
        try:
            if not isinstance(base_url, str) or any(
                char.isspace()
                or ord(char) < 32
                or 127 <= ord(char) <= 159
                or char == "\\"
                for char in base_url
            ):
                raise ValueError
            parsed = urlsplit(base_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or parsed.port == 0
            ):
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError(invalid_url) from None
        if not isinstance(trust_env, bool):
            raise ValueError("trust_env must be a boolean")
        self.trust_env = trust_env
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = _timeout(timeout)
        self.verify_ssl = verify_ssl
        self._owns_session = session is None
        self._session = session if session is not None else self._new_session()
        self._session_pool = [self._session] if self._owns_session else []
        self._pool_lock = Lock()
        self._dispatch_lock = Lock()
        self._streams: weakref.WeakSet[_HTTPCall] = weakref.WeakSet()
        self._streams_lock = Lock()
        self._closed = False

    def _new_session(self) -> Any:
        session = requests.Session()
        # An explicit policy applies to every owned lease, including sessions
        # created later for concurrent requests. Borrowed sessions are untouched.
        session.trust_env = self.trust_env
        session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
        session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
        return session

    @staticmethod
    def _close_session(session: Any) -> None:
        try:
            session.close()
        except Exception:
            pass

    def _lease_session(self) -> Any:
        with self._pool_lock:
            if self._closed:
                raise LLMBackendError("llama_server_client_closed")
            if not self._owns_session:
                return self._session
            return (
                self._session_pool.pop() if self._session_pool else self._new_session()
            )

    def _release_session(self, session: Any) -> None:
        if not self._owns_session:
            return
        with self._pool_lock:
            if not self._closed:
                self._session_pool.append(session)
                return
        self._close_session(session)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _send(
        self,
        session: Any,
        method: str,
        path: str,
        body: Any,
        timeout: float,
        stop: Event,
    ) -> Any:
        locked = False
        try:
            if not self._owns_session:
                # requests mutates Session cookies during dispatch. A borrowed
                # Session cannot be cloned or reconfigured by this client; guard
                # dispatch through headers, then allow independent body reads.
                while not self._dispatch_lock.acquire(timeout=0.05):
                    if stop.is_set():
                        raise LLMBackendError("llama_server_client_closed")
                locked = True
            if stop.is_set():
                raise LLMBackendError("llama_server_client_closed")
            return session.request(
                method=method,
                url=f"{self.base_url}{path}",
                headers=self._headers(),
                json=body,
                timeout=timeout,
                verify=self.verify_ssl,
                stream=True,
                allow_redirects=False,
            )
        finally:
            if locked:
                self._dispatch_lock.release()

    def _forget(self, stream: _HTTPCall) -> None:
        with self._streams_lock:
            self._streams.discard(stream)

    def request(
        self,
        method: str,
        path: str,
        json: Any = None,
        stream: bool = False,
        cancel: CancellationProbe | None = None,
        *,
        timeout: float | None = None,
        cancel_wait: bool = False,
        max_response_bytes: int | None = None,
    ) -> dict[str, Any] | list[Any] | Iterator[dict[str, Any]]:
        """Read JSON or SSE; optionally bound decompressed JSON/error bodies.

        ``max_response_bytes`` does not change the existing SSE frame limit.
        The bound is enforced while reading network responses, before buffering
        additional bytes. Borrowed custom response objects own their decoding.
        """
        _cancelled(cancel)
        route = urlsplit(path)
        if (
            not path.startswith("/")
            or path.startswith("//")
            or "\\" in path
            or route.scheme
            or route.netloc
            or route.fragment
        ):
            raise ValueError(
                "path must be an absolute server path, optionally with a query"
            )
        bound = self.timeout if timeout is None else _timeout(timeout)
        if not isinstance(cancel_wait, bool):
            raise ValueError("cancel_wait must be a boolean")
        if max_response_bytes is not None and (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes <= 0
        ):
            raise ValueError("max_response_bytes must be a positive integer")
        if self._closed:
            raise LLMBackendError("llama_server_client_closed")
        reader = _SSEStream if stream else _JSONCall
        result = reader(
            self, method, path, json, bound, cancel, cancel_wait, max_response_bytes
        )
        with self._streams_lock:
            if self._closed:
                raise LLMBackendError("llama_server_client_closed")
            self._streams.add(result)
            result.start()
        return result if stream else result.result()

    def chat_completion(
        self,
        messages: list[Any],
        model: str | None = None,
        stream: bool = False,
        cancel: CancellationProbe | None = None,
        *,
        timeout: float | None = None,
        cancel_wait: bool = False,
        **kwargs: Any,
    ) -> Any:
        data = {"messages": messages, **kwargs, "stream": stream}
        if model is not None:
            data["model"] = model
        return self.request(
            "POST",
            "/v1/chat/completions",
            json=data,
            stream=stream,
            cancel=cancel,
            timeout=timeout,
            cancel_wait=cancel_wait,
        )

    def complete(
        self,
        prompt: Any,
        stream: bool = False,
        cancel: CancellationProbe | None = None,
        *,
        timeout: float | None = None,
        cancel_wait: bool = False,
        **kwargs: Any,
    ) -> Any:
        data = {"prompt": prompt, **kwargs, "stream": stream}
        return self.request(
            "POST",
            "/completion",
            json=data,
            stream=stream,
            cancel=cancel,
            timeout=timeout,
            cancel_wait=cancel_wait,
        )

    def embeddings(
        self,
        input: Any,
        model: str | None = None,
        cancel: CancellationProbe | None = None,
        *,
        timeout: float | None = None,
        cancel_wait: bool = False,
        **kwargs: Any,
    ) -> Any:
        data = {"input": input, **kwargs}
        if model is not None:
            data["model"] = model
        return self.request(
            "POST",
            "/v1/embeddings",
            json=data,
            cancel=cancel,
            timeout=timeout,
            cancel_wait=cancel_wait,
        )

    def tokenize(
        self,
        content: Any,
        cancel: CancellationProbe | None = None,
        *,
        timeout: float | None = None,
        cancel_wait: bool = False,
        **kwargs: Any,
    ) -> Any:
        return self.request(
            "POST",
            "/tokenize",
            json={"content": content, **kwargs},
            cancel=cancel,
            timeout=timeout,
            cancel_wait=cancel_wait,
        )

    def detokenize(
        self,
        tokens: list[int],
        cancel: CancellationProbe | None = None,
        *,
        timeout: float | None = None,
        cancel_wait: bool = False,
        **kwargs: Any,
    ) -> Any:
        return self.request(
            "POST",
            "/detokenize",
            json={"tokens": tokens, **kwargs},
            cancel=cancel,
            timeout=timeout,
            cancel_wait=cancel_wait,
        )

    def health(
        self,
        cancel: CancellationProbe | None = None,
        *,
        timeout: float | None = None,
        cancel_wait: bool = False,
    ) -> Any:
        return self.request(
            "GET", "/health", cancel=cancel, timeout=timeout, cancel_wait=cancel_wait
        )

    def props(
        self,
        model: str | None = None,
        cancel: CancellationProbe | None = None,
        *,
        timeout: float | None = None,
        cancel_wait: bool = False,
        max_response_bytes: int | None = None,
    ) -> Any:
        query = "?" + urlencode({"model": model}) if model is not None else ""
        return self.request(
            "GET",
            "/props" + query,
            cancel=cancel,
            timeout=timeout,
            cancel_wait=cancel_wait,
            max_response_bytes=max_response_bytes,
        )

    def models(
        self,
        cancel: CancellationProbe | None = None,
        *,
        timeout: float | None = None,
        cancel_wait: bool = False,
        max_response_bytes: int | None = None,
    ) -> Any:
        return self.request(
            "GET",
            "/v1/models",
            cancel=cancel,
            timeout=timeout,
            cancel_wait=cancel_wait,
            max_response_bytes=max_response_bytes,
        )

    def close(self) -> None:
        with self._streams_lock:
            self._closed = True
            streams = list(self._streams)
        for stream in streams:
            stream._signal_close()
        # Signal every reader before draining any of them. A request awaiting
        # headers can keep this call waiting up to its finite HTTP timeout.
        for stream in streams:
            stream._join()
        with self._pool_lock:
            sessions, self._session_pool = self._session_pool, []
        for session in sessions:
            self._close_session(session)

    def __enter__(self) -> LlamaCppClient:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
