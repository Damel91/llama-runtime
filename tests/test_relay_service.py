"""Relay routing reuses actual SDK projections; lifecycle peers are isolated."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
from threading import Event

import pytest

from runtime_llama import build_adapter, build_embedding_adapter
from runtime_llama.errors import LLMCancelledError, LLMBackendError
from runtime_llama.model_discovery import (
    ModelCapabilities,
    ModelCatalog,
    ModelIdentity,
    ModelObservation,
    ObservedInt,
)
from runtime_llama.runtime_discovery import RuntimeCatalog, RuntimeObservation
from runtime_llama.relay.config import RelayConfigStore, RelayConfigConflict
from runtime_llama.relay.protocol import (
    ChatStreamProjection,
    RelayError,
    chat_payload,
    embedding_payload,
    parse_chat,
    parse_embedding,
    map_error,
)
from runtime_llama.relay.service import RelayService
from runtime_llama.usage import LLMCallResult, LLMStreamEvent

from test_tool_recovery import SequenceClient, TOOLS, rejected, valid
from test_embedding_explicit_direct import NativeClient


def observe(identifier, *, chat=True, embedding=False, verified=False, state="loaded"):
    return ModelObservation(
        identifier,
        (),
        state,
        False,
        ModelIdentity(
            "a" * 64 if verified else None, "b" * 64 if verified else None, identifier
        ),
        ModelCapabilities(chat, embedding),
    )


class Discovery:
    def __init__(self, *, embedding=False, verified=False):
        self.calls = []
        self.embedding, self.verified = embedding, verified
        self.gate = None
        self.started = Event()
        self.state = "loaded"

    def __call__(self, endpoints, *, timeout, cancellation_probe):
        self.calls.append(tuple(endpoints))
        self.started.set()
        if self.gate is not None:
            while not self.gate.wait(0.005):
                if cancellation_probe():
                    raise LLMCancelledError()
        rows = []
        for endpoint in endpoints:
            identifier = "embed_0" if self.embedding else endpoint.name + "-chat"
            model = observe(
                identifier,
                chat=not self.embedding,
                embedding=self.embedding,
                verified=self.verified,
                state=self.state,
            )
            catalog = ModelCatalog((model,), (), 3, 0.001)
            rows.append(
                RuntimeObservation(
                    endpoint,
                    "llamacpp" if self.embedding else "strata",
                    catalog,
                    (),
                    3,
                    0.001,
                    True,
                    None if self.embedding else identifier,
                )
            )
        return RuntimeCatalog(tuple(rows), 0.001)


def config(**changes):
    return {
        "runtime_endpoints": [{"name": "direct", "base_url": "http://fixture.invalid"}],
        **changes,
    }


def test_relay_calls_actual_sdk_recovery_once_and_hides_rejected_attempt():
    client = SequenceClient([rejected(), valid()])
    seen = []

    def factory(profile):
        seen.append(deepcopy(profile))
        return build_adapter(profile, client=client)

    service = RelayService(
        RelayConfigStore(initial_config=config()),
        discovery=Discovery(),
        adapter_factory=factory,
    )
    request = parse_chat(
        {
            "model": "direct-chat",
            "messages": [{"role": "user", "content": "synthetic"}],
            "tools": TOOLS,
            "stream": True,
        }
    )
    events = []
    service.chat(request, emit=events.append)
    terminal = events[-1]
    assert terminal.tool_recovery.corrections_used == 1
    assert len(client.calls) == client.streams_closed == 2
    assert len(seen) == 1 and seen[0]["backend"] == "strata"
    projection = ChatStreamProjection(
        request_id="fixture", created=1, model=request.model
    )
    wire = b"".join(chunk for event in events for chunk in projection.consume(event))
    assert b"not executed" not in wire and b"tool_recovery" not in wire
    assert b"pi_write" in wire and b"data: [DONE]" in wire
    last = service.registry.snapshot()["last_completed"]
    assert last["corrections_used"] == 1 and last["status"] == "completed"
    assert service.registry.snapshot()["active"] == []
    service.close()


def test_blocking_sdk_tool_result_history_and_model_overrides_are_preserved():
    client = SequenceClient([valid()])
    profiles = []

    def factory(profile):
        profiles.append(deepcopy(profile))
        return build_adapter(profile, client=client)

    source = config(
        llm_defaults={"temperature": 0.2},
        model_overrides={"direct-chat": {"llm": {"temperature": 0.4}}},
    )
    service = RelayService(
        RelayConfigStore(initial_config=source),
        discovery=Discovery(),
        adapter_factory=factory,
    )
    history = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "old",
                    "type": "function",
                    "function": {"name": "pi_write", "arguments": '{"path":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "old", "content": "done"},
        {"role": "user", "content": "next"},
    ]
    request = parse_chat(
        {
            "model": "direct-chat",
            "messages": history,
            "tools": TOOLS,
            "temperature": 0,
            "max_tokens": 25,
        }
    )
    result = service.chat(request)
    assert client.calls[0]["messages"] == history
    assert client.calls[0]["temperature"] == 0 and client.calls[0]["max_tokens"] == 25
    assert profiles[0]["temperature"] == 0.4
    payload = chat_payload(result, request_id="fixture", created=1, model=request.model)
    assert payload["choices"][0]["message"]["tool_calls"][0]["id"] == "call_valid"
    assert "usage" not in payload
    service.close()


def test_bare_embeddings_use_new_public_direct_method_without_tokenization():
    client = NativeClient()
    adapters = []

    def factory(profile):
        adapter = build_embedding_adapter(profile, client=client)
        adapters.append(adapter)
        return adapter

    service = RelayService(
        RelayConfigStore(
            initial_config=config(
                embedding_defaults={
                    "normalize": False,
                    "embedding_prefixes": {"query": "query: ", "document": "doc: "},
                }
            )
        ),
        discovery=Discovery(embedding=True),
        embedding_factory=factory,
    )
    request = parse_embedding(
        {
            "model": "embed_0",
            "input": ["a", "duplicate", "duplicate"],
            "input_type": "query",
        }
    )
    vectors = service.embed(request)
    payload = embedding_payload(vectors, request)
    assert len(payload["data"]) == 3 and [row["index"] for row in payload["data"]] == [
        0,
        1,
        2,
    ]
    assert not client.token_calls
    assert all(
        text.startswith("query: ")
        for _alias, batch in client.embedding_calls
        for text in batch
    )
    assert len(adapters) == 1
    service.embed(request)
    assert len(adapters) == 1
    service.close()


def test_verified_embeddings_share_one_service_and_catalog_change_retires_it():
    discovery = Discovery(embedding=True, verified=True)
    created = []

    class Embeddings:
        def __init__(self, profile):
            self.profile, self.closed = profile, False
            created.append(self)

        def embed_texts(self, texts, **options):
            assert not self.closed
            return [[float(len(text)), 1.0] for text in texts]

        def close(self):
            self.closed = True

    service = RelayService(
        RelayConfigStore(initial_config=config()),
        discovery=discovery,
        automatic_embedding_factory=Embeddings,
    )
    request = parse_embedding({"model": "embed_0", "input": ["x"]})
    with ThreadPoolExecutor(8) as pool:
        assert all(
            result == [[1.0, 1.0]]
            for result in pool.map(lambda _: service.embed(request), range(20))
        )
    assert len(created) == 1
    discovery.state = "unloaded"
    service.refresh_catalog()
    assert created[0].closed
    assert not service._cached
    service.close()


def test_catalog_reads_are_explicit_and_config_replacement_does_not_mutate_old_call():
    discovery = Discovery()
    discovery.gate = Event()
    client = SequenceClient([valid()])
    profiles = []

    def factory(profile):
        profiles.append(profile)
        return build_adapter(profile, client=client)

    service = RelayService(
        RelayConfigStore(initial_config=config(llm_defaults={"temperature": 0.1})),
        discovery=discovery,
        adapter_factory=factory,
    )
    assert service.catalog_snapshot()["observed_at"] is None and not discovery.calls
    request = parse_chat(
        {
            "model": "direct-chat",
            "messages": [{"role": "user", "content": "fixture"}],
            "tools": TOOLS,
        }
    )
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(service.chat, request)
        assert discovery.started.wait(2)
        response = service.replace_config(0, config(llm_defaults={"temperature": 0.9}))
        discovery.gate.set()
        assert future.result(3).tool_calls
    assert response["revision"] == 1 and profiles[0]["temperature"] == 0.1
    assert service.catalog_snapshot()["revision"] == 1
    assert service.catalog_snapshot()["observed_at"] is None
    with pytest.raises(RelayConfigConflict):
        service.replace_config(0, config())
    service.close()


class Helper:
    def __init__(self, configuration):
        self.configuration = configuration
        self.started = self.closed = False
        self.proxy_url = "http://runtime:" + "z" * 32 + "@127.0.0.1:17891"

    def start(self):
        self.started = True

    def close(self):
        self.closed = True


def vpn_config(tmp_path):
    return config(
        runtime_endpoints=[
            {"name": "direct", "base_url": "http://fixture.invalid"},
            {"name": "vpn", "base_url": "http://peer.fixture", "use_tailnet": True},
        ],
        tailnet={
            "executable": str(tmp_path / "helper"),
            "state_dir": str(tmp_path / "state"),
            "hostname": "relay-fixture",
        },
    )


def test_helper_is_explicit_and_never_falls_back_to_direct(tmp_path):
    discovery = Discovery()
    helpers = []

    def helper_factory(configuration):
        helper = Helper(configuration)
        helpers.append(helper)
        return helper

    service = RelayService(
        RelayConfigStore(initial_config=vpn_config(tmp_path)),
        discovery=discovery,
        tailnet_factory=helper_factory,
    )
    first = service.refresh_catalog()
    assert len(discovery.calls[0]) == 1
    assert first["runtimes"][1]["diagnostics"] == ["tailnet_not_running"]
    assert not helpers
    service.start_tailnet()
    assert helpers[0].started
    result = service.refresh_catalog()
    assert len(discovery.calls[1]) == 2
    assert discovery.calls[1][1].proxy_url == helpers[0].proxy_url
    assert "z" * 32 not in json.dumps(result)
    service.stop_tailnet()
    assert helpers[0].closed
    service.refresh_catalog()
    assert len(discovery.calls[-1]) == 1
    service.close()


def test_stopping_helper_cancels_only_selected_vpn_and_drains_old_adapter(tmp_path):
    running = {name: Event() for name in ("direct-chat", "vpn-chat")}
    release_direct = Event()
    adapters = []

    class Adapter:
        def __init__(self, profile):
            self.profile, self.closed = profile, False
            adapters.append(self)

        def generate_with_metadata(self, _messages, cancellation_probe, **_options):
            model = self.profile["model"]
            running[model].set()
            while not release_direct.wait(0.005):
                if cancellation_probe():
                    raise LLMCancelledError()
            return LLMCallResult("done")

        def close(self):
            self.closed = True
            self.profile.clear()

    service = RelayService(
        RelayConfigStore(initial_config=vpn_config(tmp_path)),
        discovery=Discovery(),
        adapter_factory=Adapter,
        tailnet_factory=Helper,
    )
    service.start_tailnet()

    def make(model):
        return parse_chat(
            {"model": model, "messages": [{"role": "user", "content": "fixture"}]}
        )

    with ThreadPoolExecutor(2) as pool:
        direct = pool.submit(service.chat, make("direct-chat"))
        vpn = pool.submit(service.chat, make("vpn-chat"))
        assert all(event.wait(2) for event in running.values())
        service.stop_tailnet()
        with pytest.raises(LLMCancelledError):
            vpn.result(2)
        assert not direct.done()
        assert any(adapter.closed for adapter in adapters)
        release_direct.set()
        assert direct.result(2).text == "done"
    assert not service._retired and not service._helper_resources
    service.close()


def test_opencode_export_is_versioned_and_never_exports_the_relay_key():
    service = RelayService(
        RelayConfigStore(
            initial_config=config(access={"api_key": "private-relay-key"})
        ),
        discovery=Discovery(),
    )
    service.refresh_catalog()
    for version in ("v1", "v2"):
        result = service.export_opencode(
            version, ["direct-chat"], "http://localhost:8088/v1"
        )
        assert "private-relay-key" not in json.dumps(result)
        assert "{env:LLAMA_RELAY_API_KEY}" in json.dumps(result)
        assert result["config"]["$schema"] == "https://opencode.ai/config.json"
    exported = service.export_opencode("v2", [], "http://localhost:8088/v1")
    assert (
        exported["config"]["providers"]["llama-relay"]["package"]
        == "@opencode/ai/providers/openai-compatible"
    )
    service.replace_config(0, config())
    service.refresh_catalog()
    assert "relay-local" in json.dumps(
        service.export_opencode("v2", [], "http://localhost:8088/v1")
    )
    service.close()


def test_configured_opencode_separates_training_context_and_rejects_stale_choices():
    context = 8192

    def discover(endpoints, **_kwargs):
        models = (
            replace(
                observe("coder/a"),
                loaded_context=ObservedInt(context, "catalog.meta.n_ctx"),
            ),
            replace(
                observe("coder-b"),
                training_context=ObservedInt(131072, "catalog.meta.n_ctx_train"),
            ),
            observe("embedding", chat=False, embedding=True),
        )
        return RuntimeCatalog(
            (
                RuntimeObservation(
                    endpoints[0],
                    "llamacpp",
                    ModelCatalog(models, (), 3, 0.0),
                    (),
                    3,
                    0.0,
                    True,
                ),
            ),
            0.0,
        )

    service = RelayService(
        RelayConfigStore(initial_config=config()), discovery=discover
    )
    base_url = "http://localhost:8088/v1"
    try:
        catalog = service.refresh_catalog()
        assert catalog["runtimes"][0]["models"][1]["loaded_context"] is None
        payload = {
            "version": "v2",
            "catalog_revision": catalog["revision"],
            "default_model": "coder-b",
            "models": {
                "coder/a": {"observed_context": 8192, "context": 8192, "output": 1024},
                "coder-b": {"observed_context": None, "context": 4096, "output": 512},
            },
        }
        exported = service.export_configured_opencode(payload, base_url)["config"]
        assert exported["model"] == "llama-relay/coder-b"
        assert exported["providers"]["llama-relay"]["models"] == {
            "coder/a": {"name": "coder/a", "limit": {"context": 8192, "output": 1024}},
            "coder-b": {"name": "coder-b", "limit": {"context": 4096, "output": 512}},
        }

        for field, value, code in [
            ("output", 8192, "relay_opencode_output_invalid"),
            ("output", True, "relay_opencode_output_invalid"),
            ("context", 131072, "relay_opencode_context_invalid"),
        ]:
            invalid = deepcopy(payload)
            invalid["models"]["coder/a"][field] = value
            with pytest.raises(RelayError) as error:
                service.export_configured_opencode(invalid, base_url)
            assert (error.value.status, error.value.code) == (400, code)
        with pytest.raises(RelayError, match="relay_opencode_default_model_invalid"):
            service.export_configured_opencode(
                {**payload, "default_model": "embedding"}, base_url
            )
        with pytest.raises(RelayError, match="relay_opencode_model_not_available"):
            service.export_configured_opencode(
                {
                    **payload,
                    "models": {
                        **payload["models"],
                        "embedding": payload["models"]["coder-b"],
                    },
                },
                base_url,
            )

        context = 4096
        with pytest.raises(RelayError, match="relay_opencode_context_changed") as error:
            service.export_configured_opencode(payload, base_url)
        assert error.value.status == 409
        service.replace_config(0, config())
        with pytest.raises(RelayError, match="relay_opencode_catalog_changed") as error:
            service.export_configured_opencode(payload, base_url)
        assert error.value.status == 409
    finally:
        service.close()


def test_stream_terminal_follows_adapter_close_and_failed_close_has_no_success():
    adapters = []

    class Adapter:
        def __init__(self, _profile):
            self.closed = False
            self.fail_close = True
            self.close_calls = 0
            adapters.append(self)

        def stream_with_metadata(self, *_args, **_kwargs):
            yield LLMStreamEvent(kind="content", text="accepted")
            yield LLMStreamEvent(kind="terminal", terminal_reason="completed")

        def close(self):
            self.close_calls += 1
            if self.fail_close:
                raise RuntimeError("fixture close still owns its worker")
            self.closed = True

    service = RelayService(
        RelayConfigStore(initial_config=config()),
        discovery=Discovery(),
        adapter_factory=Adapter,
    )
    request = parse_chat(
        {
            "model": "direct-chat",
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
        }
    )
    events = []
    with pytest.raises(RelayError, match="relay_resource_close_failed"):
        service.chat(request, emit=events.append)
    assert [event.kind for event in events] == ["content"]
    assert service.registry.snapshot()["last_completed"]["status"] == "failed"
    assert len(service._retired) == 1 and not adapters[0].closed
    assert adapters[0].close_calls == 1
    adapters[0].fail_close = False
    service.close()
    assert adapters[0].close_calls == 2 and adapters[0].closed
    assert not service._retired


def test_successful_stream_terminal_observes_actual_adapter_drain():
    closed = Event()

    class Adapter:
        def __init__(self, _profile):
            pass

        def stream_with_metadata(self, *_args, **_kwargs):
            yield LLMStreamEvent(kind="terminal", terminal_reason="completed")

        def close(self):
            closed.set()

    service = RelayService(
        RelayConfigStore(initial_config=config()),
        discovery=Discovery(),
        adapter_factory=Adapter,
    )
    request = parse_chat(
        {
            "model": "direct-chat",
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
        }
    )
    observed = []

    def emit(event):
        assert closed.is_set()
        assert service.registry.snapshot()["active"] == []
        observed.append(event)

    service.chat(request, emit=emit)
    assert len(observed) == 1
    service.close()


def test_failed_helper_stop_retains_ownership_and_close_retries_pending_node(tmp_path):
    helpers = []

    class FailingHelper(Helper):
        def __init__(self, configuration):
            super().__init__(configuration)
            self.fail_close = True
            self.close_calls = 0
            helpers.append(self)

        def close(self):
            self.close_calls += 1
            if self.fail_close:
                raise RuntimeError("fixture child still running")
            super().close()

    service = RelayService(
        RelayConfigStore(initial_config=vpn_config(tmp_path)),
        discovery=Discovery(),
        tailnet_factory=FailingHelper,
    )
    service.start_tailnet()
    with pytest.raises(RelayError, match="relay_resource_close_failed"):
        service.stop_tailnet()
    assert service.status()["tailnet"]["state"] == "failed"
    assert service._pending_helper_stops and service._retired_helpers
    assert helpers[0].close_calls == 1
    with pytest.raises(RelayError, match="previous_tailnet_stop_failed"):
        service.start_tailnet()
    helpers[0].fail_close = False
    service.close()
    assert helpers[0].closed and helpers[0].close_calls == 2
    assert not service._pending_helper_stops and not service._retired_helpers


@pytest.mark.parametrize(
    "body",
    [
        {"model": "m", "messages": [], "stream": True},
        {"model": "m", "messages": [{"role": "user", "content": "x"}], "n": 2},
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "api_key": "secret",
        },
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 4,
            "max_completion_tokens": 4,
        },
    ],
)
def test_bad_chat_shapes_are_explicit_before_sdk_or_discovery(body):
    with pytest.raises(RelayError):
        parse_chat(body)


def test_projection_omits_unknown_usage_and_never_invents_success_on_error():
    projection = ChatStreamProjection(
        request_id="f", created=1, model="m", include_usage=True
    )
    assert (
        projection.consume(
            LLMStreamEvent(kind="tool_recovery", data={"next_attempt": 1})
        )
        == []
    )
    assert projection.consume(LLMStreamEvent(kind="timings")) == []
    wire = b"".join(
        projection.consume(
            LLMStreamEvent(
                kind="terminal",
                terminal_reason="completed",
                data={"finish_reason": "stop"},
            )
        )
    )
    assert b'"usage":null' in wire and b"[DONE]" in wire
    assert b"prompt_tokens" not in wire and b"total_tokens" not in wire
    with pytest.raises(RelayError, match="incomplete"):
        ChatStreamProjection(request_id="f", created=1, model="m").consume(
            LLMStreamEvent(kind="terminal", terminal_reason="cancelled")
        )
    mapped = map_error(LLMBackendError("upstream private key content"))
    assert mapped.status == 502 and "private key" not in json.dumps(mapped.payload())
    assert (
        map_error(LLMBackendError("private_credential_value")).code
        == "relay_upstream_error"
    )


def test_base64_embedding_is_float32_and_token_shapes_are_not_silently_accepted():
    import base64
    import struct

    request = parse_embedding(
        {"model": "m", "input": "text", "encoding_format": "base64"}
    )
    result = embedding_payload([[0.5, -1.0]], request)
    assert struct.unpack("<ff", base64.b64decode(result["data"][0]["embedding"])) == (
        0.5,
        -1.0,
    )
    assert "usage" not in result
    for value in ([1, 2], [[1, 2]]):
        with pytest.raises(RelayError):
            parse_embedding({"model": "m", "input": value})
    with pytest.raises(RelayError):
        parse_embedding({"model": "m", "input": "text", "dimensions": 4})
