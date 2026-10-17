"""Call-owned socket interruption for streaming without a read deadline.

Session leases remain exclusive. Adapter and pool subclasses are scoped to
each owned Session; urllib3's shared class maps are never mutated.
"""

from __future__ import annotations

import socket
from threading import Event, Lock
from typing import Any

from requests.adapters import HTTPAdapter

try:
    from urllib3.util.ssltransport import SSLTransport
except ImportError:
    SSLTransport = ()


class RequestInterrupt:
    def __init__(self, stop: Event) -> None:
        self._stop = stop
        self._lock = Lock()
        self._connection: Any = None
        self._socket: Any = None
        self._detached = False

    @staticmethod
    def _shutdown(sock: Any) -> None:
        # HTTPS through an HTTPS proxy uses TLS-in-TLS. SSLTransport lacks
        # shutdown(), but exposes its known underlying socket transport.
        for _ in range(8):
            if not isinstance(sock, SSLTransport):
                break
            sock = sock.socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except (OSError, ValueError):
                pass

    def capture(self, connection: Any) -> None:
        with self._lock:
            if self._detached:
                raise OSError("runtime_llama_request_aborted")
            self._connection = connection
            current = getattr(connection, "sock", None)
            if current is not None:
                # http.client can clear connection.sock after Connection: close
                # headers while its buffered response still owns the socket.
                self._socket = current
            if self._stop.is_set():
                self._shutdown(self._socket)
                raise OSError("runtime_llama_request_aborted")

    def abort(self) -> None:
        with self._lock:
            if self._detached:
                return
            current = getattr(self._connection, "sock", None)
            if current is not None:
                self._socket = current
            self._shutdown(self._socket)

    def detach(self) -> None:
        with self._lock:
            self._detached = True
            connection = self._connection
            if getattr(connection, "_runtime_llama_interrupt", None) is self:
                connection._runtime_llama_interrupt = None
            self._connection = self._socket = None


class InterruptibleAdapter(HTTPAdapter):
    """Preserve Requests TLS/proxy handling, adding only a leased call hook."""

    def __init__(self) -> None:
        self.interrupt: RequestInterrupt | None = None
        self._scoped_pools: dict[type, type] = {}
        super().__init__(max_retries=0)

    def _scope_pool(self, pool_class: type) -> type:
        if pool_class in self._scoped_pools:
            return self._scoped_pools[pool_class]
        adapter = self
        connection_class = pool_class.ConnectionCls

        class ScopedConnection(connection_class):
            def connect(self) -> None:
                interrupt = getattr(self, "_runtime_llama_interrupt", None)
                if interrupt is not None:
                    interrupt.capture(self)
                super().connect()
                if interrupt is not None:
                    # Cancellation during bounded connect/TLS must be noticed
                    # before entering an unbounded response-header read.
                    interrupt.capture(self)

        class ScopedPool(pool_class):
            ConnectionCls = ScopedConnection

            def _make_request(self, conn: Any, *args: Any, **kwargs: Any) -> Any:
                interrupt = adapter.interrupt
                conn._runtime_llama_interrupt = interrupt
                if interrupt is not None:
                    # This also captures a warm keep-alive socket, for which
                    # connect() will not run again.
                    interrupt.capture(conn)
                return super()._make_request(conn, *args, **kwargs)

        self._scoped_pools[pool_class] = ScopedPool
        return ScopedPool

    def _scope_manager(self, manager: Any) -> None:
        if getattr(manager, "_runtime_llama_scoped", False):
            return
        manager.pool_classes_by_scheme = {
            scheme: self._scope_pool(pool_class)
            for scheme, pool_class in manager.pool_classes_by_scheme.items()
        }
        manager._runtime_llama_scoped = True

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        super().init_poolmanager(*args, **kwargs)
        self._scope_manager(self.poolmanager)

    def proxy_manager_for(self, *args: Any, **kwargs: Any) -> Any:
        manager = super().proxy_manager_for(*args, **kwargs)
        self._scope_manager(manager)
        return manager


def bind_interrupt(session: Any, interrupt: RequestInterrupt) -> None:
    """Install once on an exclusively leased, library-owned Session."""
    for scheme in ("http://", "https://"):
        previous = session.adapters[scheme]
        if not isinstance(previous, InterruptibleAdapter):
            adapter = InterruptibleAdapter()
            session.mount(scheme, adapter)
            previous.close()
        else:
            adapter = previous
        adapter.interrupt = interrupt


def unbind_interrupt(session: Any, interrupt: RequestInterrupt) -> None:
    for adapter in session.adapters.values():
        if isinstance(adapter, InterruptibleAdapter) and adapter.interrupt is interrupt:
            adapter.interrupt = None
