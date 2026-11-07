"""Optional ASGI application: native SDK inference plus local browser control."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hmac
from importlib.resources import files
from ipaddress import ip_address
import os
import secrets
import time
from urllib.parse import urlsplit
from uuid import uuid4

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from runtime_llama.relay.admin_contract import ADMIN_API_VERSION, ADMIN_TOKEN_HEADER
from runtime_llama.relay.bridge import ThreadBridge
from runtime_llama.relay.config import (
    RelayConfigConflict,
    RelayConfigError,
    RelayConfigStore,
)
from runtime_llama.relay.protocol import (
    ChatStreamProjection,
    RelayError,
    chat_payload,
    embedding_payload,
    map_error,
    parse_chat,
    parse_embedding,
    sse,
)
from runtime_llama.relay.service import RelayService


_SSE_KEEPALIVE_SECONDS = 15.0


class _SafeErrors:
    """Map failures before Starlette's generic error handler re-raises to logs.

    This is a streaming ASGI boundary, not a body-buffering middleware.
    Cancellation and already-started response transport failures keep their
    normal propagation; SDK/SSE failures are handled by the response itself.
    """

    def __init__(self, app, handler):
        self.app, self.handler = app, handler

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = False

        async def track(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, track)
        except Exception as error:
            if started:
                raise
            response = await self.handler(None, error)
            await response(scope, receive, send)


class _RelayStream(Response):
    media_type = "text/event-stream"

    def __init__(self, bridge, projection, cleanup):
        super().__init__(
            status_code=200,
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            media_type=self.media_type,
        )
        # Streaming has no Content-Length; the ASGI server owns HTTP framing.
        self.raw_headers = [
            (name, value)
            for name, value in self.raw_headers
            if name != b"content-length"
        ]
        self.bridge, self.projection, self.cleanup = bridge, projection, cleanup

    async def __call__(self, scope, receive, send):
        async def send_chunk(chunk):
            await send({"type": "http.response.body", "body": chunk, "more_body": True})

        async def next_event():
            if self.projection.finished:
                return await anext(self.bridge)
            # A keepalive timeout must not cancel the SDK invocation or
            # start another read from the bridge.
            pending = asyncio.create_task(anext(self.bridge))
            try:
                while not pending.done():
                    ready, _ = await asyncio.wait(
                        (pending,), timeout=_SSE_KEEPALIVE_SECONDS
                    )
                    if not ready:
                        await send_chunk(b": keep-alive\n\n")
            except BaseException:
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                raise
            return pending.result()

        async def stream():
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": self.raw_headers,
                }
            )
            await send_chunk(b": relay connected\n\n")
            try:
                while True:
                    try:
                        event = await next_event()
                    except StopAsyncIteration:
                        break
                    for chunk in self.projection.consume(event):
                        await send_chunk(chunk)
                if not self.projection.finished:
                    raise RelayError(502, "relay_stream_ended_without_terminal")
            except Exception as error:
                await send_chunk(sse(map_error(error).payload()))
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        async def disconnected():
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    self.bridge.cancel()
                    return

        tasks = {
            asyncio.create_task(stream()),
            asyncio.create_task(disconnected()),
            asyncio.create_task(self.bridge.wait_cancelled()),
        }
        try:
            done, _pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if not task.cancelled():
                    task.result()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.bridge.aclose()
            self.cleanup()


def create_app(config_path=None, initial_config=None, service=None):
    """Build the optional app; listeners/helper/model loads remain explicit."""
    if service is not None and (config_path is not None or initial_config is not None):
        raise ValueError("relay_service_conflicts_with_configuration")
    service = (
        service
        if service is not None
        else RelayService(RelayConfigStore(config_path, initial_config))
    )
    admin_token = secrets.token_urlsafe(32)
    bridges = {}

    def cancel_active():
        # Nonblocking hook for a server's first shutdown signal. ASGI servers
        # may wait for requests before entering lifespan shutdown, so quiet
        # calls must receive Stop before that wait rather than a new time cap.
        service.registry.cancel_all()
        for bridge in tuple(bridges.values()):
            bridge.cancel()

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            cancel_active()
            active = tuple(bridges.values())
            for bridge in active:
                await bridge.aclose()
            bridges.clear()
            async with ThreadBridge(lambda _emit, _cancel: service.close()) as worker:
                await worker.result()

    def authority(request):
        header = request.headers.get("host", "")
        try:
            parsed = urlsplit("//" + header)
            host = parsed.hostname
            parsed.port
        except ValueError:
            raise RelayError(403, "relay_admin_host_denied") from None
        listener = service.config_store.snapshot().config["listener"]
        allowed = set(listener["allowed_hosts"])
        active_host = service.listener["host"]
        if active_host not in {"0.0.0.0", "::"}:
            allowed.add(active_host.strip("[]"))
        if (
            host not in allowed
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise RelayError(403, "relay_admin_host_denied")
        origin = request.headers.get("origin")
        if origin is not None:
            expected = f"{request.url.scheme}://{header}"
            if origin != expected:
                raise RelayError(403, "relay_admin_origin_denied")
        if request.headers.get("sec-fetch-site") in {"cross-site", "same-site"}:
            raise RelayError(403, "relay_admin_origin_denied")

    def admin_access(request, *, session=False):
        authority(request)
        if not session and not hmac.compare_digest(
            request.headers.get(ADMIN_TOKEN_HEADER, "").encode("utf-8"),
            admin_token.encode("utf-8"),
        ):
            raise RelayError(401, "relay_admin_token_required")

    def is_loopback(host):
        if host == "localhost":
            return True
        try:
            return ip_address((host or "").strip("[]")).is_loopback
        except ValueError:
            return False

    def inference_access(request):
        access = service.config_store.snapshot().config["access"]
        key = access.get("api_key")
        if access.get("api_key_env"):
            key = os.environ.get(access["api_key_env"])
            if not key:
                raise RelayError(503, "relay_access_environment_unavailable")
        if key is not None and not hmac.compare_digest(
            request.headers.get("authorization", "").encode("utf-8"),
            ("Bearer " + key).encode("utf-8"),
        ):
            raise RelayError(401, "relay_api_key_required")

    async def body(request):
        if (
            request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            != "application/json"
        ):
            raise RelayError(415, "relay_json_content_type_required")
        try:
            return await request.json()
        except (ValueError, UnicodeError):
            raise RelayError(400, "relay_invalid_json") from None

    async def blocking(request, function, *, request_id=None):
        request_id = request_id or uuid4().hex
        bridge = ThreadBridge(lambda _emit, cancel: function(cancel))
        bridges[request_id] = bridge

        async def disconnected():
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    bridge.cancel()
                    return

        result = asyncio.create_task(bridge.result())
        watcher = asyncio.create_task(disconnected())
        try:
            done, _pending = await asyncio.wait(
                (result, watcher), return_when=asyncio.FIRST_COMPLETED
            )
            if watcher in done and not result.done():
                bridge.cancel()
                raise RelayError(499, "relay_request_cancelled")
            return await result
        finally:
            watcher.cancel()
            if not result.done():
                result.cancel()
            await asyncio.gather(result, watcher, return_exceptions=True)
            await bridge.aclose()
            bridges.pop(request_id, None)

    async def chat(request):
        inference_access(request)
        parsed = parse_chat(await body(request))
        request_id, created = uuid4().hex, int(time.time())
        if not parsed.stream:
            result = await blocking(
                request,
                lambda cancel: service.chat(
                    parsed, cancel_event=cancel, request_id=request_id
                ),
                request_id=request_id,
            )
            return JSONResponse(
                chat_payload(
                    result, request_id=request_id, created=created, model=parsed.model
                )
            )
        bridge = ThreadBridge(
            lambda emit, cancel: service.chat(
                parsed, emit=emit, cancel_event=cancel, request_id=request_id
            )
        )
        bridges[request_id] = bridge
        return _RelayStream(
            bridge,
            ChatStreamProjection(
                request_id=request_id,
                created=created,
                model=parsed.model,
                include_usage=parsed.include_usage,
            ),
            lambda: bridges.pop(request_id, None),
        )

    async def embeddings(request):
        inference_access(request)
        parsed = parse_embedding(await body(request))
        request_id = uuid4().hex
        vectors = await blocking(
            request,
            lambda cancel: service.embed(
                parsed, cancel_event=cancel, request_id=request_id
            ),
            request_id=request_id,
        )
        return JSONResponse(embedding_payload(vectors, parsed))

    async def models(request):
        inference_access(request)
        return JSONResponse(await blocking(request, service.model_list))

    async def health(_request):
        return JSONResponse({"status": "ok", "service": "llama-runtime-relay"})

    def admin_json(value):
        return JSONResponse(
            value,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    async def admin(request):
        path = request.url.path
        admin_access(request, session=path == "/api/admin/session")
        if path == "/api/admin/session":
            return admin_json(
                {"admin_token": admin_token, "api_version": ADMIN_API_VERSION}
            )
        if path == "/api/admin/config":
            if request.method == "GET":
                return admin_json(service.config_store.snapshot().public())
            payload = await body(request)
            if not isinstance(payload, dict) or set(payload) != {
                "expected_revision",
                "config",
            }:
                raise RelayError(400, "relay_config_write_shape_invalid")
            return admin_json(
                await blocking(
                    request,
                    lambda _cancel: service.replace_config(
                        payload["expected_revision"], payload["config"]
                    ),
                )
            )
        if path == "/api/admin/status":
            return admin_json(service.status())
        if path == "/api/admin/catalog":
            return admin_json(service.catalog_snapshot())
        if path == "/api/admin/catalog/refresh":
            return admin_json(await blocking(request, service.refresh_catalog))
        if path == "/api/admin/requests":
            return admin_json(service.registry.snapshot())
        if path == "/api/admin/requests/last-completed":
            service.registry.clear_last_completed()
            return admin_json(service.registry.snapshot())
        if path.endswith("/cancel"):
            identifier = request.path_params["request_id"]
            cancelled = service.cancel_request(identifier)
            bridge = bridges.get(identifier)
            if bridge is not None:
                bridge.cancel()
            if not cancelled:
                raise RelayError(404, "relay_request_not_active")
            return admin_json({"request_id": identifier, "cancelled": True})
        if path in {"/api/admin/tailnet/start", "/api/admin/tailnet/stop"}:
            method = (
                service.start_tailnet
                if path.endswith("/start")
                else service.stop_tailnet
            )
            return admin_json(await blocking(request, lambda _cancel: method()))
        if path in {"/api/admin/opencode", "/api/admin/opencode/apply"}:
            apply = path.endswith("/apply")
            if apply and not (
                request.client
                and is_loopback(request.client.host)
                and is_loopback(service.listener["host"])
                and is_loopback(request.url.hostname)
            ):
                raise RelayError(403, "relay_opencode_apply_local_only")
            authority_url = str(request.base_url).rstrip("/") + "/v1"
            if request.method == "POST":
                payload = await body(request)
                return admin_json(
                    await blocking(
                        request,
                        lambda cancel: service.export_configured_opencode(
                            payload, authority_url, cancel, apply=apply
                        ),
                    )
                )
            return admin_json(
                service.export_opencode(
                    request.query_params.get("version", "v2"),
                    request.query_params.getlist("models"),
                    authority_url,
                )
            )
        raise RelayError(404, "relay_admin_route_not_found")

    async def static(request):
        authority(request)
        name = {
            "/": "index.html",
            "/static/app.js": "app.js",
            "/static/style.css": "style.css",
        }.get(request.url.path)
        if name is None:
            raise RelayError(404, "relay_asset_not_found")
        try:
            payload = files("runtime_llama.relay").joinpath("static", name).read_bytes()
        except FileNotFoundError:
            raise RelayError(503, "relay_web_assets_unavailable") from None
        content_type = {
            "index.html": "text/html",
            "app.js": "text/javascript",
            "style.css": "text/css",
        }[name]
        return Response(
            payload,
            media_type=content_type,
            headers={
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
            },
        )

    async def handle_error(_request, error):
        if isinstance(error, RelayConfigConflict):
            mapped = RelayError(
                409,
                "relay_config_revision_conflict",
                details={"actual_revision": error.actual_revision},
            )
        elif isinstance(error, RelayConfigError):
            mapped = RelayError(400, str(error))
        else:
            mapped = map_error(error)
        return JSONResponse(
            mapped.payload(),
            status_code=mapped.status,
            headers={"Cache-Control": "no-store"},
        )

    routes = [
        Route("/health", health),
        Route("/v1/models", models),
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/v1/embeddings", embeddings, methods=["POST"]),
    ]
    from runtime_llama.relay.admin_contract import ADMIN_ROUTES

    for method, path in ADMIN_ROUTES:
        routes.append(Route(path, admin, methods=[method]))
    routes += [
        Route("/", static),
        Route("/static/app.js", static),
        Route("/static/style.css", static),
    ]
    app = Starlette(
        routes=routes,
        lifespan=lifespan,
        middleware=[Middleware(_SafeErrors, handler=handle_error)],
        exception_handlers={
            RelayError: handle_error,
            RelayConfigError: handle_error,
        },
    )
    app.state.relay_service, app.state.relay_bridges = service, bridges
    app.state.relay_cancel_all = cancel_active
    return app
