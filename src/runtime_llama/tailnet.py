"""Opt-in lifecycle for the separately built userspace tailnet helper.

Importing or constructing these objects does not launch a process, create node
state or enroll a device. The consumer supplies configuration and the binary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import re
import secrets
import signal
import subprocess
from threading import Event, Lock, Thread, current_thread
import time
from typing import Any, Callable
from urllib.parse import urlsplit
from weakref import WeakSet

from runtime_llama.client import LlamaCppClient
from runtime_llama.errors import LLMBackendError, LLMCancelledError
from runtime_llama.model_discovery import ModelCatalog, discover_models

_PROTOCOL = 1
_BACKEND = "1.104.1"
_MAX_CONFIG = 65536
_MAX_STATUS = 8192
_HELPER_ERRORS = {
    "invalid_config",
    "state_in_use",
    "state_mismatch",
    "state_invalid",
    "tailnet_start_failed",
    "tailnet_start_timeout",
    "proxy_start_failed",
    "startup_cancelled",
    "invalid_command",
    "proxy_failed",
    "shutdown_timeout",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate helper protocol field")
        result[key] = value
    return result


class TailnetError(LLMBackendError):
    """Sanitized helper lifecycle failure, independent of provider messages."""


def _positive(value: Any, name: str) -> float:
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError
        number = float(value)
        if not math.isfinite(number) or not 1e-9 <= number <= 86400:
            raise ValueError
    except (ValueError, OverflowError, TypeError):
        raise ValueError(
            f"{name} must be at least 1 nanosecond and at most 86400 seconds"
        ) from None
    return number


def _absolute_path(value: Any, name: str) -> str:
    try:
        text = os.fspath(value)
        if not isinstance(text, str) or not text or "\0" in text:
            raise ValueError
        if not Path(text).is_absolute():
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be an absolute path") from None
    return text


@dataclass(frozen=True)
class TailnetConfig:
    """Consumer-owned node identity and finite lifecycle budgets.

    ``control_url=None`` selects Tailscale; a Headscale URL selects that
    coordination server. ``auth_key=None`` supports previously enrolled state.
    The state directory is dedicated to this helper and locked by the Go side.
    No credentials are resolved implicitly from environment variables.
    """

    executable: str | os.PathLike[str]
    state_dir: str | os.PathLike[str]
    hostname: str
    control_url: str | None = None
    auth_key: str | None = field(default=None, repr=False)
    ephemeral: bool = False
    startup_timeout_sec: float = 30.0
    dial_timeout_sec: float = 10.0
    shutdown_timeout_sec: float = 5.0

    def __post_init__(self) -> None:
        for name in ("executable", "state_dir"):
            object.__setattr__(self, name, _absolute_path(getattr(self, name), name))
        if not isinstance(self.hostname, str) or not re.fullmatch(
            r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", self.hostname
        ):
            raise ValueError("hostname must be a DNS label")
        if not isinstance(self.ephemeral, bool):
            raise ValueError("ephemeral must be a boolean")
        if self.auth_key is not None and (
            not isinstance(self.auth_key, str)
            or not 1 <= len(self.auth_key) <= 8192
            or any(c.isspace() or ord(c) < 32 for c in self.auth_key)
        ):
            raise ValueError("auth_key must be a nonempty bounded token")
        if self.control_url is not None:
            try:
                value = self.control_url
                if not isinstance(value, str) or any(
                    c.isspace() or ord(c) < 32 or ord(c) == 127 or c == "\\"
                    for c in value
                ):
                    raise ValueError
                parsed = urlsplit(value)
                if (
                    parsed.scheme != "https"
                    or not parsed.hostname
                    or parsed.username is not None
                    or parsed.password is not None
                    or parsed.query
                    or parsed.fragment
                    or parsed.port == 0
                    or parsed.path not in {"", "/"}
                ):
                    raise ValueError
            except (ValueError, TypeError):
                raise ValueError(
                    "control_url must be a credential-free HTTPS root URL"
                ) from None
            object.__setattr__(self, "control_url", value.rstrip("/"))
        for name in (
            "startup_timeout_sec",
            "dial_timeout_sec",
            "shutdown_timeout_sec",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))

    def _payload(self, token: str) -> bytes:
        payload = (
            json.dumps(
                {
                    "protocol": _PROTOCOL,
                    "state_dir": self.state_dir,
                    "hostname": self.hostname,
                    "auth_key": self.auth_key or "",
                    "control_url": self.control_url or "",
                    "ephemeral": self.ephemeral,
                    "proxy_token": token,
                    "startup_timeout_sec": self.startup_timeout_sec,
                    "dial_timeout_sec": self.dial_timeout_sec,
                    "shutdown_timeout_sec": self.shutdown_timeout_sec,
                },
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        if len(payload) > _MAX_CONFIG:
            raise ValueError("tailnet helper configuration exceeds protocol limit")
        return payload


def _child_environment() -> dict[str, str]:
    ignored = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    return {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("TS_", "TSNET_")) and key.lower() not in ignored
    }


class TailnetHelper:
    """One explicit helper process; closing it drains clients created via client().

    Use ``with TailnetHelper(config) as network`` or call ``start()`` explicitly.
    ``proxy_url`` is ephemeral secret transport configuration, not node state.
    A helper is single-use: failure never restarts or replays inference.
    """

    def __init__(self, config: TailnetConfig) -> None:
        if not isinstance(config, TailnetConfig):
            raise TypeError("config must be TailnetConfig")
        self.config = config
        self._lock = Lock()
        self._failure_lock = Lock()
        self._closing = Event()
        self._shutdown = Event()
        self._failed = Event()
        self._shutdown_error = Event()
        self._config_sent = Event()
        self._ready = Event()
        self._closed = False
        self._process: subprocess.Popen[bytes] | None = None
        self._threads: list[Thread] = []
        self._events: Queue[dict[str, Any]] = Queue(maxsize=1)
        self._clients: WeakSet[LlamaCppClient] = WeakSet()
        self._discovery_calls: set[Event] = set()
        self._proxy_url: str | None = None
        self._failure_cleanup: Thread | None = None

    def __repr__(self) -> str:
        state = "closed" if self._closed else "ready" if self._ready.is_set() else "new"
        return f"TailnetHelper(state={state!r})"

    def _signal_process(self, *, kill: bool) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "nt":
                process.kill() if kill else process.terminate()
            else:
                os.killpg(process.pid, signal.SIGKILL if kill else signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    def _fail(self) -> None:
        with self._failure_lock:
            self._failed.set()
            if self._ready.is_set() and not self._closing.is_set():
                self._signal_process(kill=False)
                if self._failure_cleanup is None:
                    self._failure_cleanup = Thread(
                        target=self._close_after_failure,
                        name="runtime_llama_tailnet_failure",
                        daemon=True,
                    )
                    self._failure_cleanup.start()

    def _close_after_failure(self) -> None:
        try:
            self.close()
        except TailnetError:
            pass

    def _read_status(self) -> None:
        process = self._process
        seen = False
        try:
            line = bytearray()
            while True:
                char = process.stdout.read(1)
                if not char:
                    if not self._closing.is_set():
                        self._fail()
                    return
                if char != b"\n":
                    line.extend(char)
                    if len(line) > _MAX_STATUS:
                        self._fail()
                        return
                    continue
                value = json.loads(
                    line.decode("utf-8"), object_pairs_hook=_unique_object
                )
                line.clear()
                if not isinstance(value, dict):
                    self._fail()
                    return
                if seen:
                    if self._closing.is_set() and value.get("status") == "closed":
                        continue
                    if self._closing.is_set() and value.get("status") == "error":
                        self._shutdown_error.set()
                    self._fail()
                    return
                seen = True
                self._events.put_nowait(value)
        except Exception:
            if not self._closing.is_set():
                self._fail()

    def _drain_stderr(self) -> None:
        try:
            while self._process.stderr.read(16384):
                pass
        except (OSError, ValueError):
            pass

    def _write_control(self, payload: bytes) -> None:
        pipe = self._process.stdin
        try:
            remaining = memoryview(payload)
            while remaining:
                written = pipe.write(remaining)
                if not written:
                    raise OSError
                remaining = remaining[written:]
            pipe.flush()
            self._config_sent.set()
            self._shutdown.wait()
            if self._process.poll() is None:
                pipe.write(b'{"protocol":1,"command":"shutdown"}\n')
                pipe.flush()
        except (OSError, ValueError):
            if not self._closing.is_set():
                self._fail()
        finally:
            try:
                pipe.close()
            except (OSError, ValueError):
                pass

    @staticmethod
    def _check_cancel(probe: Callable[[], bool] | None) -> None:
        if probe is None:
            return
        try:
            requested = bool(probe())
        except Exception:
            raise TailnetError("tailnet_cancellation_probe_failed") from None
        if requested:
            raise LLMCancelledError("tailnet_start_cancelled")

    def _validate_ready(self, value: dict[str, Any], token: str) -> str:
        if type(value.get("protocol")) is not int or value["protocol"] != _PROTOCOL:
            raise TailnetError("tailnet_helper_protocol_error")
        if value.get("status") == "error":
            code = value.get("code")
            known = (
                code
                if isinstance(code, str) and code in _HELPER_ERRORS
                else "start_failed"
            )
            raise TailnetError("tailnet_helper_" + known)
        if set(value) != {"protocol", "status", "proxy_url", "backend_version"}:
            raise TailnetError("tailnet_helper_protocol_error")
        if value["status"] != "ready" or value["backend_version"] != _BACKEND:
            raise TailnetError("tailnet_helper_protocol_error")
        try:
            raw = value["proxy_url"]
            if not isinstance(raw, str):
                raise ValueError
            parsed = urlsplit(raw)
            port = parsed.port
            if (
                not port
                or parsed.scheme != "http"
                or parsed.hostname != "127.0.0.1"
                or raw != f"http://127.0.0.1:{port}"
            ):
                raise ValueError
        except (ValueError, TypeError):
            raise TailnetError("tailnet_helper_protocol_error") from None
        return f"http://runtime:{token}@127.0.0.1:{port}"

    def start(
        self, *, cancellation_probe: Callable[[], bool] | None = None
    ) -> TailnetHelper:
        self._check_cancel(cancellation_probe)
        with self._lock:
            if self._closed or self._closing.is_set():
                raise TailnetError("tailnet_helper_closed")
            if self._ready.is_set():
                self._assert_running()
                return self
            token = secrets.token_urlsafe(32)
            payload = self.config._payload(token)
            deadline = time.monotonic() + self.config.startup_timeout_sec
            try:
                self._process = subprocess.Popen(
                    [self.config.executable],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    env=_child_environment(),
                    start_new_session=os.name != "nt",
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        | subprocess.CREATE_NO_WINDOW
                        if os.name == "nt"
                        else 0
                    ),
                )
            except OSError:
                self._closed = True
                raise TailnetError("tailnet_helper_launch_failed") from None
            self._threads = [
                Thread(
                    target=self._read_status,
                    name="runtime_llama_tailnet_status",
                    daemon=True,
                ),
                Thread(
                    target=self._drain_stderr,
                    name="runtime_llama_tailnet_stderr",
                    daemon=True,
                ),
                Thread(
                    target=self._write_control,
                    args=(payload,),
                    name="runtime_llama_tailnet_control",
                    daemon=True,
                ),
            ]
            try:
                for thread in self._threads:
                    thread.start()
                status = None
                while True:
                    self._check_cancel(cancellation_probe)
                    if self._closing.is_set():
                        raise TailnetError("tailnet_helper_closed")
                    if status is None:
                        try:
                            status = self._events.get_nowait()
                        except Empty:
                            pass
                    if status is not None and status.get("status") == "error":
                        self._validate_ready(status, token)
                    if self._failed.is_set() or self._process.poll() is not None:
                        raise TailnetError("tailnet_helper_start_failed")
                    if status is not None and self._config_sent.is_set():
                        self._proxy_url = self._validate_ready(status, token)
                        self._ready.set()
                        self._assert_running()
                        return self
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TailnetError("tailnet_helper_start_timeout")
                    self._failed.wait(min(0.02, remaining))
            except BaseException:
                self._closing.set()
                self._close_locked()
                raise

    def _assert_running(self) -> None:
        if (
            not self._ready.is_set()
            or self._closed
            or self._closing.is_set()
            or self._failed.is_set()
            or self._process is None
            or self._process.poll() is not None
        ):
            raise TailnetError("tailnet_helper_unavailable")

    @property
    def proxy_url(self) -> str:
        """Secret, ephemeral explicit route for SDK profiles and discovery."""
        with self._lock:
            self._assert_running()
            return self._proxy_url

    @property
    def running(self) -> bool:
        with self._lock:
            try:
                self._assert_running()
            except TailnetError:
                return False
            return True

    def client(self, base_url: str, **kwargs: Any) -> LlamaCppClient:
        """Create an owned raw HTTP client, also closed when this helper closes."""
        if "proxy_url" in kwargs or kwargs.get("session") is not None:
            raise ValueError("tailnet client route and session are managed")
        if kwargs.get("trust_env", False) is not False:
            raise ValueError("tailnet client ignores ambient transport configuration")
        kwargs.pop("trust_env", None)
        with self._lock:
            self._assert_running()
            client = LlamaCppClient(
                base_url, proxy_url=self._proxy_url, trust_env=False, **kwargs
            )
            self._clients.add(client)
            return client

    def discover_models(self, base_url: str, **kwargs: Any) -> ModelCatalog:
        """Read-only discovery over this explicit route; no model administration."""
        if "proxy_url" in kwargs:
            raise ValueError("tailnet discovery route is managed")
        caller_probe = kwargs.pop("cancellation_probe", None)
        done = Event()
        with self._lock:
            self._assert_running()
            proxy_url = self._proxy_url
            self._discovery_calls.add(done)

        def cancelled() -> bool:
            return self._closing.is_set() or (
                caller_probe is not None and bool(caller_probe())
            )

        try:
            return discover_models(
                base_url,
                proxy_url=proxy_url,
                cancellation_probe=cancelled,
                **kwargs,
            )
        finally:
            done.set()
            with self._lock:
                self._discovery_calls.discard(done)

    def _close_locked(self) -> None:
        if self._closed:
            return
        self._closing.set()
        process = self._process
        client_threads = []
        client_failed = Event()

        def close_client(client: LlamaCppClient) -> None:
            try:
                client.close()
            except Exception:
                client_failed.set()

        for client in list(self._clients):
            thread = Thread(
                target=close_client,
                args=(client,),
                name="runtime_llama_tailnet_client_close",
                daemon=True,
            )
            client_threads.append(thread)
            thread.start()
        self._clients.clear()
        self._shutdown.set()
        deadline = time.monotonic() + self.config.shutdown_timeout_sec
        hard_deadline = deadline + 2
        if process is not None:
            try:
                process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                self._signal_process(kill=False)
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self._signal_process(kill=True)
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        raise TailnetError("tailnet_helper_cleanup_timeout") from None
        for thread in [*self._threads, *client_threads]:
            if thread is not current_thread() and thread.ident is not None:
                thread.join(timeout=max(0, hard_deadline - time.monotonic()))
        discovery_pending = any(
            not done.wait(timeout=max(0, hard_deadline - time.monotonic()))
            for done in self._discovery_calls
        )
        self._discovery_calls.clear()
        self._closed = True
        self._proxy_url = None
        workers_alive = any(
            thread.is_alive() for thread in [*self._threads, *client_threads]
        )
        if process is not None:
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None and not workers_alive:
                    pipe.close()
        if workers_alive or client_failed.is_set() or discovery_pending:
            raise TailnetError("tailnet_helper_cleanup_timeout")
        if self._shutdown_error.is_set():
            raise TailnetError("tailnet_helper_shutdown_failed")

    def close(self) -> None:
        self._closing.set()
        with self._lock:
            self._close_locked()
        cleanup = self._failure_cleanup
        if cleanup is not None and cleanup is not current_thread():
            cleanup.join(timeout=1)
            if cleanup.is_alive():
                raise TailnetError("tailnet_helper_cleanup_timeout")

    def __enter__(self) -> TailnetHelper:
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.close()
