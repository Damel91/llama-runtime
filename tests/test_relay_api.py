"""Real ASGI/native HTTP boundaries; fixtures never execute proposed tools."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Event, Lock, Thread

import httpx
import pytest
from starlette.testclient import TestClient

from runtime_llama.relay.app import create_app
from runtime_llama.relay.config import RelayConfigStore, default_config
from runtime_llama.relay.service import RelayService
from runtime_llama.usage import LLMCallResult, LLMStreamEvent
from test_generation import _chunk


ORIGIN = "http://127.0.0.1:8088"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    }
]


@pytest.fixture
def native_servers():
    owned = []

    def start(kind="strata", *, responses=None, key=None):
        state = {"gets": [], "posts": [], "active": 0}
        lock = Lock()
        model = "chat-model" if kind == "strata" else "embedding-model"

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def reply(self, value, status=200, *, sse=False):
                wire = value if isinstance(value, bytes) else json.dumps(value).encode()
                self.send_response(status)
                self.send_header(
                    "Content-Type", "text/event-stream" if sse else "application/json"
                )
                self.send_header("Content-Length", str(len(wire)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(wire)
                self.close_connection = True

            def authorized(self):
                if key and self.headers.get("Authorization") != "Bearer " + key:
                    self.reply({"error": "fixture unauthorized"}, 401)
                    return False
                return True

            def do_GET(self):
                with lock:
                    state["gets"].append((self.path, self.headers.get("Authorization")))
                if not self.authorized():
                    return
                if self.path == "/health":
                    value = {"status": "ok"}
                    if kind == "strata":
                        value.update(service="strata", loaded=True, model=model)
                elif self.path == "/props":
                    value = {
                        "build_info": "Strata 0.1.52"
                        if kind == "strata"
                        else "b12345-aabbcc",
                        "model_alias": model,
                        "is_sleeping": False,
                        "total_slots": 4,
                        "default_generation_settings": {"n_ctx": 2048},
                    }
                elif self.path == "/v1/models":
                    value = {
                        "data": [
                            {
                                "id": model,
                                "status": {"value": "loaded"},
                                "capabilities": {
                                    "chat": kind == "strata",
                                    "embeddings": kind != "strata",
                                },
                            }
                        ]
                    }
                else:
                    self.reply({"error": "unexpected endpoint"}, 404)
                    return
                self.reply(value)

            def do_POST(self):
                request = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                with lock:
                    index = len(state["posts"])
                    state["posts"].append(
                        (self.path, request, self.headers.get("Authorization"))
                    )
                    state["active"] += 1
                try:
                    if not self.authorized():
                        return
                    if self.path == "/v1/embeddings":
                        self.reply(
                            {
                                "data": [
                                    {
                                        "index": i,
                                        "embedding": [float(len(text)), float(i + 1)],
                                    }
                                    for i, text in reversed(
                                        list(enumerate(request["input"]))
                                    )
                                ]
                            }
                        )
                    elif self.path == "/v1/chat/completions":
                        frames = (
                            responses(index, request)
                            if responses
                            else [
                                _chunk(content="native-answer"),
                                _chunk(finish="stop"),
                            ]
                        )
                        if isinstance(frames, int):
                            self.reply({"error": "private native failure"}, frames)
                        else:
                            wire = (
                                b"".join(
                                    b"data: " + json.dumps(frame).encode() + b"\n\n"
                                    for frame in frames
                                )
                                + b"data: [DONE]\n\n"
                            )
                            self.reply(wire, sse=True)
                    else:
                        self.reply({"error": "forbidden extra endpoint"}, 500)
                finally:
                    with lock:
                        state["active"] -= 1

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = False
        thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01))
        thread.start()
        owned.append((server, thread, state))
        return f"http://127.0.0.1:{server.server_port}", state

    yield start
    for server, thread, state in owned:
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive() and state["active"] == 0


@contextmanager
def client(config=None, service=None, **client_options):
    app = (
        create_app(initial_config=config)
        if service is None
        else create_app(service=service)
    )
    with TestClient(
        app, base_url=ORIGIN, raise_server_exceptions=False, **client_options
    ) as api:
        yield api, app
    assert app.state.relay_bridges == {}


def config_for(*endpoints):
    config = default_config()
    config["runtime_endpoints"] = [
        {"name": f"native-{i}", "base_url": url, **settings}
        for i, (url, settings) in enumerate(endpoints)
    ]
    return config


def token(api):
    response = api.get("/api/admin/session", headers={"Origin": ORIGIN})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    return {"X-Relay-Admin-Token": response.json()["admin_token"], "Origin": ORIGIN}


def test_opencode_apply_only_changes_the_local_provider_and_default(
    native_servers, tmp_path, monkeypatch
):
    url, upstream = native_servers(key="upstream-secret")
    config = config_for((url, {"api_key": "upstream-secret"}))
    config["access"]["api_key"] = "private-relay-key"
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(tmp_path))
    target = tmp_path / "opencode.jsonc"
    original = '{\n  // Keep these settings\n  "providers": {"other": {"settings": {"apiKey": "other-private-key"}},},\n  "model": "other/default",\n  "permissions": [],\n}\n'
    target.write_text(original, encoding="utf-8")
    with client(config, client=("127.0.0.1", 40001)) as (api, _app):
        headers = token(api)
        catalog = api.post("/api/admin/catalog/refresh", headers=headers).json()
        model = catalog["runtimes"][0]["models"][0]["id"]
        payload = {
            "version": "v2",
            "catalog_revision": catalog["revision"],
            "default_model": model,
            "models": {
                model: {"observed_context": 2048, "context": 2048, "output": 512}
            },
        }
        preview = api.post("/api/admin/opencode", headers=headers, json=payload)
        assert preview.status_code == 200 and target.read_text() == original
        response = api.post("/api/admin/opencode/apply", headers=headers, json=payload)
        assert response.status_code == 200
        value = response.json()
        assert value["changed"] and value["model"] == "llama-relay/" + model
        assert value["filename"] == target.name
        assert (tmp_path / value["backup"]).read_text() == original
        assert not any(
            key in response.text
            for key in ("other-private-key", "private-relay-key", "upstream-secret")
        )
        written = target.read_text()
        assert all(
            segment in written
            for segment in (
                "// Keep these settings",
                "other-private-key",
                '"permissions": []',
                '"apiKey": "{env:LLAMA_RELAY_API_KEY}"',
                '"output": 512',
                '"baseURL": "' + ORIGIN + '/v1"',
            )
        )
        repeated = api.post("/api/admin/opencode/apply", headers=headers, json=payload)
        assert repeated.status_code == 200 and repeated.json()["changed"] is False
        assert target.read_text() == written

        payload["models"][model] = {
            "observed_context": 4096,
            "context": 4096,
            "output": 512,
        }
        stale = api.post("/api/admin/opencode/apply", headers=headers, json=payload)
        assert stale.status_code == 409
        assert stale.json()["error"]["code"] == "relay_opencode_context_changed"
        assert target.read_text() == written and not upstream["posts"]


@pytest.mark.parametrize(
    "listener,peer,status,code",
    [
        ("127.0.0.1", "192.0.2.12", 403, "relay_opencode_apply_local_only"),
        ("0.0.0.0", "127.0.0.1", 403, "relay_opencode_apply_local_only"),
        ("127.0.0.1", "127.0.0.1", 400, "relay_opencode_apply_v2_required"),
    ],
)
def test_opencode_apply_requires_loopback_and_v2(
    listener, peer, status, code, tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(tmp_path))
    config = default_config()
    config["listener"]["host"] = listener
    with client(config, client=(peer, 40001)) as (api, _app):
        path = "/api/admin/opencode/apply"
        payload = {
            "version": "v1",
            "catalog_revision": 0,
            "default_model": "chat",
            "models": {},
        }
        assert api.post(path, json=payload).status_code == 401
        response = api.post(path, json=payload, headers=token(api))
        assert response.status_code == status
        assert response.json()["error"]["code"] == code
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "version,provider_key,options_key",
    [
        ("v1", "provider", "options"),
        ("v2", "providers", "settings"),
    ],
)
def test_configured_opencode_reads_remote_context_and_exports_through_relay(
    native_servers, version, provider_key, options_key
):
    url, upstream = native_servers(key="upstream-secret")
    config = config_for((url, {"api_key": "upstream-secret"}))
    config["access"]["api_key"] = "local-secret"
    with client(config) as (api, _app):
        headers = token(api)
        catalog = api.post("/api/admin/catalog/refresh", headers=headers).json()
        model = catalog["runtimes"][0]["models"][0]
        assert model["opencode_chat"] and model["loaded_context"] == 2048
        payload = {
            "version": version,
            "catalog_revision": catalog["revision"],
            "default_model": model["id"],
            "models": {
                model["id"]: {"observed_context": 2048, "context": 2048, "output": 512}
            },
        }
        assert api.post("/api/admin/opencode", json=payload).status_code == 401
        response = api.post("/api/admin/opencode", headers=headers, json=payload)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert (
            "upstream-secret" not in response.text
            and "local-secret" not in response.text
        )
        exported = response.json()["config"]
        assert exported["model"] == "llama-relay/" + model["id"]
        provider = exported[provider_key]["llama-relay"]
        assert provider[options_key] == {
            "baseURL": ORIGIN + "/v1",
            "apiKey": "{env:LLAMA_RELAY_API_KEY}",
        }
        assert provider["models"][model["id"]]["limit"] == {
            "context": 2048,
            "output": 512,
        }
        assert [path for path, _key in upstream["gets"]] == [
            "/health",
            "/props",
            "/v1/models",
        ] * 2
        assert not upstream["posts"]


def test_static_assets_and_inert_admin_reads_are_served_without_native_probe():
    with client() as (api, app):
        page = api.get("/")
        assert page.status_code == 200 and "/static/app.js" in page.text
        for path in ("/static/app.js", "/static/style.css"):
            response = api.get(path)
            assert response.status_code == 200
            assert (
                "frame-ancestors 'none'" in response.headers["content-security-policy"]
            )
            assert response.headers["x-content-type-options"] == "nosniff"
        headers = token(api)
        assert (
            api.get("/api/admin/catalog", headers=headers).json()["observed_at"] is None
        )
        assert api.get("/api/admin/status", headers=headers).status_code == 200
        assert app.state.relay_service.registry.snapshot()["active"] == []


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "foreign.example"},
        {"Origin": "https://foreign.example"},
        {"Origin": ORIGIN + ".foreign.example"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
        {"Host": "user@localhost:8088"},
        {"Host": "[broken"},
    ],
)
def test_foreign_sites_cannot_bootstrap_admin_session(headers):
    with client() as (api, _):
        response = api.get("/api/admin/session", headers=headers)
        assert response.status_code == (
            400 if headers.get("Host") == "[broken" else 403
        )
        assert "admin_token" not in response.text
        assert (
            api.options(
                "/api/admin/config",
                headers={
                    "Origin": "https://foreign.example",
                    "Access-Control-Request-Method": "PUT",
                },
            ).headers.get("access-control-allow-origin")
            is None
        )


def test_admin_token_revision_conflict_and_secret_preservation(tmp_path):
    url = "http://127.0.0.1:8080"
    config = config_for((url, {"api_key": "backend-private"}))
    store = RelayConfigStore(tmp_path / "relay.json", initial_config=config)
    service = RelayService(store)
    with client(service=service) as (api, _):
        assert api.get("/api/admin/config").status_code == 401
        headers = token(api)
        initial = api.get("/api/admin/config", headers=headers).json()
        assert "backend-private" not in json.dumps(initial)
        assert "/runtime_endpoints/0/api_key" in initial["secret_fields"]
        initial["config"]["llm_defaults"]["temperature"] = 0
        payload = {
            "expected_revision": initial["revision"],
            "config": initial["config"],
        }
        saved = api.put("/api/admin/config", headers=headers, json=payload)
        assert saved.status_code == 200
        assert (
            service.config_store.snapshot().config["runtime_endpoints"][0]["api_key"]
            == "backend-private"
        )
        assert service.config_store.snapshot().settings("m", "llm")["temperature"] == 0
        assert (
            api.put("/api/admin/config", headers=headers, json=payload).status_code
            == 409
        )
        assert api.post("/api/admin/tailnet/start", headers=headers).status_code == 400
        assert (
            api.post(
                "/api/admin/requests/not-active/cancel", headers=headers
            ).status_code
            == 404
        )


def test_relay_key_is_separate_from_native_key_and_export(native_servers):
    url, state = native_servers(key="native-private")
    config = config_for((url, {"api_key": "native-private"}))
    config["access"] = {"api_key": "relay-private"}
    with client(config) as (api, _):
        assert api.get("/v1/models").status_code == 401
        assert (
            api.get(
                "/v1/models", headers={"Authorization": "Bearer native-private"}
            ).status_code
            == 401
        )
        assert state["gets"] == []
        assert (
            api.get(
                "/v1/models", headers={"Authorization": "Bearer relay-private"}
            ).json()["data"][0]["id"]
            == "chat-model"
        )
        assert {key for _, key in state["gets"]} == {"Bearer native-private"}
        headers = token(api)
        exported = api.get("/api/admin/opencode?version=v2", headers=headers)
        assert exported.status_code == 200
        assert (
            "relay-private" not in exported.text
            and "native-private" not in exported.text
        )
        assert "LLAMA_RELAY_API_KEY" in exported.text


def test_listener_edit_keeps_active_custom_origin_until_explicit_restart():
    config = default_config()
    config["listener"].update(host="127.0.0.2", allowed_hosts=["localhost"])
    service = RelayService(RelayConfigStore(initial_config=config))
    app = create_app(service=service)
    origin = "http://127.0.0.2:8088"
    with TestClient(app, base_url=origin) as api:
        session = api.get("/api/admin/session", headers={"Origin": origin})
        assert session.status_code == 200
        headers = {
            "Origin": origin,
            "X-Relay-Admin-Token": session.json()["admin_token"],
        }
        current = api.get("/api/admin/config", headers=headers).json()
        current["config"]["listener"]["host"] = "127.0.0.3"
        saved = api.put(
            "/api/admin/config",
            headers=headers,
            json={
                "expected_revision": current["revision"],
                "config": current["config"],
            },
        )
        assert saved.status_code == 200 and saved.json()["restart_required"] is True
        assert service.listener["host"] == "127.0.0.2"
        assert api.get("/api/admin/status", headers=headers).status_code == 200
        assert api.get("/", headers={"Origin": origin}).status_code == 200
        assert (
            api.get(
                "/api/admin/session", headers={"Host": "127.0.0.3:8088"}
            ).status_code
            == 403
        )


@pytest.mark.parametrize(
    "payload,status",
    [
        ({"model": "x", "messages": [], "stream": False}, 400),
        ({"model": "x", "messages": [{"role": "user", "content": "x"}], "n": 2}, 400),
        (
            {
                "model": "x",
                "messages": [{"role": "user", "content": "x"}],
                "runtime_endpoints": [],
            },
            400,
        ),
    ],
)
def test_unsupported_chat_is_rejected_before_native_work(payload, status):
    with client() as (api, _):
        response = api.post("/v1/chat/completions", json=payload)
        assert response.status_code == status and response.json()["error"]["code"]


def test_bad_wire_shapes_and_unavailable_native_are_explicit():
    with client() as (api, _):
        assert (
            api.post(
                "/v1/chat/completions",
                content="{}",
                headers={"Content-Type": "text/plain"},
            ).status_code
            == 415
        )
        assert (
            api.post(
                "/v1/chat/completions",
                content="{",
                headers={"Content-Type": "application/json"},
            ).status_code
            == 400
        )
        assert (
            api.post(
                "/v1/embeddings", json={"model": "x", "input": [[1, 2]]}
            ).status_code
            == 400
        )
        assert (
            api.post(
                "/v1/embeddings", json={"model": "x", "input": "x", "dimensions": 2}
            ).status_code
            == 400
        )
        response = api.post(
            "/v1/chat/completions",
            json={"model": "missing", "messages": [{"role": "user", "content": "x"}]},
        )
        assert response.status_code == 404


def test_unexpected_provider_error_is_mapped_without_reraising_private_message():
    class FailingService(RelayService):
        def model_list(self, cancel_event=None):
            raise RuntimeError("PRIVATE-PROVIDER-CONTENT")

    service = FailingService()
    app = create_app(service=service)
    with TestClient(app, base_url=ORIGIN, raise_server_exceptions=True) as api:
        response = api.get("/v1/models")
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "relay_internal_error"
        assert "PRIVATE" not in response.text
    assert app.state.relay_bridges == {}


def test_chat_json_native_sdk_preserves_structured_history_and_options(native_servers):
    url, state = native_servers()
    messages = [
        {"role": "system", "content": "system fixture"},
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "prior",
                    "type": "function",
                    "function": {"name": "inspect", "arguments": '{"key":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "prior", "content": "fixture result"},
        {"role": "user", "content": "continue"},
    ]
    with client(config_for((url, {}))) as (api, app):
        response = api.post(
            "/v1/chat/completions",
            json={
                "model": "chat-model",
                "messages": messages,
                "temperature": 0,
                "max_tokens": 32,
                "stop": ["END"],
            },
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["choices"][0]["message"]["content"] == "native-answer"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert "usage" not in result
        request = state["posts"][0][1]
        assert request["messages"] == messages
        assert (
            request["temperature"] == 0
            and request["max_tokens"] == 32
            and request["stop"] == ["END"]
        )
        assert len(state["posts"]) == 1
        assert app.state.relay_service.registry.snapshot()["active"] == []


def test_sse_recovery_stages_rejected_content_and_budget_resets(native_servers):
    def responses(index, _request):
        if index == 0:
            return [
                _chunk(content="NEVER-PUBLISH"),
                _chunk(finish="stop"),
                {
                    "choices": [],
                    "strata_rejected_tool_calls": [
                        {"name": "wrong", "reason": "undeclared"}
                    ],
                },
            ]
        return [
            _chunk(reasoning="accepted-reasoning"),
            _chunk(
                tools=[
                    {
                        "index": 0,
                        "id": "accepted-call",
                        "type": "function",
                        "function": {"name": "inspect", "arguments": '{"key":"ok"}'},
                    }
                ]
            ),
            _chunk(finish="tool_calls"),
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                },
            },
        ]

    url, state = native_servers(responses=responses)
    payload = {
        "model": "chat-model",
        "messages": [{"role": "user", "content": "fixture"}],
        "tools": TOOLS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    with client(config_for((url, {}))) as (api, app):
        response = api.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert "NEVER-PUBLISH" not in response.text
        assert "tool_recovery" not in response.text
        frames = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert all(frame.get("usage", "missing") is None for frame in frames[:-1])
        assert len({(f["id"], f["created"], f["model"]) for f in frames}) == 1
        assert any(
            f["choices"]
            and f["choices"][0]["delta"].get("tool_calls", [{}])[0].get("id")
            == "accepted-call"
            for f in frames
        )
        assert (
            next(
                f["choices"][0]["finish_reason"]
                for f in frames
                if f["choices"] and f["choices"][0]["finish_reason"]
            )
            == "tool_calls"
        )
        assert frames[-1]["choices"] == [] and frames[-1]["usage"]["total_tokens"] == 10
        assert response.text.endswith("data: [DONE]\n\n")
        assert len(state["posts"]) == 2
        assert state["posts"][1][1]["messages"] != payload["messages"]
        summary = app.state.relay_service.registry.snapshot()["last_completed"]
        assert summary["corrections_used"] == 1
        assert "NEVER-PUBLISH" not in json.dumps(summary)
        # New incoming HTTP invocation has independent correction accounting.
        assert api.post("/v1/chat/completions", json=payload).status_code == 200
        assert len(state["posts"]) == 3
        assert (
            app.state.relay_service.registry.snapshot()["last_completed"][
                "corrections_used"
            ]
            == 0
        )


@pytest.mark.parametrize("stream", [False, True])
def test_native_error_is_not_success_or_retried(native_servers, stream):
    url, state = native_servers(responses=lambda _i, _r: 500)
    with client(config_for((url, {}))) as (api, _):
        response = api.post(
            "/v1/chat/completions",
            json={
                "model": "chat-model",
                "messages": [{"role": "user", "content": "x"}],
                "stream": stream,
            },
        )
        assert response.status_code == (200 if stream else 502)
        assert (
            "error" in response.text and "private native failure" not in response.text
        )
        assert (
            "[DONE]" not in response.text
            and '"finish_reason":"stop"' not in response.text
        )
        assert len(state["posts"]) == 1


def test_embedding_direct_http_order_prefix_and_no_tokenizer(native_servers):
    url, state = native_servers("llamacpp")
    config = config_for((url, {}))
    config["embedding_defaults"].update(
        normalize=False,
        batch_size=256,
        embedding_prefixes={"query": "query: ", "document": "document: "},
    )
    with client(config) as (api, _):
        response = api.post(
            "/v1/embeddings",
            json={
                "model": "embedding-model",
                "input": ["same", "日本語", "same"],
                "input_type": "query",
            },
        )
        assert response.status_code == 200, response.text
        rows = response.json()["data"]
        assert [row["index"] for row in rows] == [0, 1, 2]
        assert [row["embedding"][0] for row in rows] == [
            len("query: " + text) for text in ["same", "日本語", "same"]
        ]
        assert "usage" not in response.json()
        assert {p[0] for p in state["posts"]} == {"/v1/embeddings"}
        assert [
            text for _, request, _ in state["posts"] for text in request["input"]
        ] == ["query: same", "query: 日本語", "query: same"]


class ControlledService(RelayService):
    """Owned SDK-work stand-in for ASGI disconnect/backpressure tests only."""

    def __init__(self, *, first_event=False):
        super().__init__()
        self.started, self.release, self.stopped = Event(), Event(), Event()
        self.first_event = first_event

    def chat(self, request, *, emit=None, cancel_event=None, request_id=None):
        with self._operation(
            "chat", request.model, cancel_event, request_id
        ) as operation:
            self.started.set()
            try:
                if emit and self.first_event:
                    for _ in range(3):
                        emit(LLMStreamEvent("content", "fixture"))
                while not self.release.wait(0.002):
                    operation.check()
                operation.check()
                if emit:
                    emit(LLMStreamEvent("terminal", terminal_reason="completed"))
                else:
                    return LLMCallResult("fixture")
            finally:
                self.stopped.set()


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.002)


async def invoke(app, *, stream, messages, sent, send_block=None):
    payload = json.dumps(
        {
            "model": "fixture",
            "messages": [{"role": "user", "content": "fixture"}],
            "stream": stream,
        }
    ).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [
            (b"host", b"127.0.0.1:8088"),
            (b"content-type", b"application/json"),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 8088),
    }
    await messages.put({"type": "http.request", "body": payload, "more_body": False})

    async def send(message):
        sent.append(message)
        if (
            send_block is not None
            and message["type"] == "http.response.body"
            and message.get("body")
        ):
            await send_block.wait()

    await app(scope, messages.get, send)


@pytest.mark.parametrize("stream", [False, True])
def test_quiet_prefill_disconnect_and_asgi_task_cancel_drain(stream):
    async def scenario(task_cancel):
        service = ControlledService()
        app = create_app(service=service)
        messages, sent = asyncio.Queue(), []
        task = asyncio.create_task(
            invoke(app, stream=stream, messages=messages, sent=sent)
        )
        try:
            await until(service.started.is_set)
            if task_cancel:
                task.cancel()
            else:
                await messages.put({"type": "http.disconnect"})
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 3)
            assert service.stopped.is_set()
            assert service.registry.snapshot()["active"] == []
            assert app.state.relay_bridges == {}
        finally:
            service.release.set()
            await asyncio.gather(task, return_exceptions=True)
            service.close()

    asyncio.run(scenario(False))
    asyncio.run(scenario(True))


def test_stop_interrupts_backpressured_send_and_does_not_cancel_sibling():
    async def scenario():
        service = ControlledService(first_event=True)
        app = create_app(service=service)
        queues, sent = [asyncio.Queue(), asyncio.Queue()], [[], []]
        blocked = asyncio.Event()
        tasks = [
            asyncio.create_task(
                invoke(app, stream=True, messages=q, sent=out, send_block=blocked)
            )
            for q, out in zip(queues, sent)
        ]
        try:
            await until(lambda: len(service.registry.snapshot()["active"]) == 2)
            identifier = service.registry.snapshot()["active"][0]["request_id"]
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url=ORIGIN
            ) as api:
                session = (await api.get("/api/admin/session")).json()
                response = await api.post(
                    f"/api/admin/requests/{identifier}/cancel",
                    headers={"X-Relay-Admin-Token": session["admin_token"]},
                )
                assert response.status_code == 200
            await until(lambda: len(service.registry.snapshot()["active"]) == 1)
            await until(lambda: sum(task.done() for task in tasks) == 1)
            assert sum(task.done() for task in tasks) == 1
            await queues[1 if tasks[0].done() else 0].put({"type": "http.disconnect"})
            await asyncio.wait_for(asyncio.gather(*tasks), 3)
            assert (
                service.registry.snapshot()["active"] == []
                and app.state.relay_bridges == {}
            )
        finally:
            blocked.set()
            service.release.set()
            for q in queues:
                await q.put({"type": "http.disconnect"})
            await asyncio.gather(*tasks, return_exceptions=True)
            service.close()

    asyncio.run(scenario())


def test_server_shutdown_hook_stops_quiet_calls_before_lifespan_wait():
    async def scenario():
        service = ControlledService()
        app = create_app(service=service)
        messages, sent = asyncio.Queue(), []
        task = asyncio.create_task(
            invoke(app, stream=True, messages=messages, sent=sent)
        )
        try:
            await until(service.started.is_set)
            app.state.relay_cancel_all()
            app.state.relay_cancel_all()
            await asyncio.wait_for(task, 3)
            assert service.stopped.is_set()
            assert (
                app.state.relay_bridges == {}
                and service.registry.snapshot()["active"] == []
            )
        finally:
            service.release.set()
            await asyncio.gather(task, return_exceptions=True)
            service.close()

    asyncio.run(scenario())
