from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread

import pytest

from runtime_llama.adapters.llama import LlamaCppAdapter
from runtime_llama.adapters.strata import StrataAdapter
from runtime_llama.client import ClientHTTPError
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError


def chunk(content=None, *, finish=None, **delta):
    if content is not None:
        delta["content"] = content
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


class Client:
    def __init__(self, text="done"):
        self.payloads = [chunk(text), chunk(finish="stop")]
        self.calls = []
        self.closed_streams = 0
        self.closed = False
        self.gets = []
        self.health_payload = {
            "status": "ok",
            "service": "strata",
            "loaded": True,
            "model": "qwen",
        }
        self.catalog = {
            "data": [
                {"id": "qwen", "aliases": ["alias"], "status": {"value": "loaded"}}
            ]
        }
        self.properties = {
            "model_alias": "qwen",
            "is_sleeping": False,
            "default_generation_settings": {"n_ctx": 8192},
            "total_slots": 1,
        }

    def chat_completion(self, messages, **options):
        self.calls.append({"messages": deepcopy(messages), **deepcopy(options)})

        def stream():
            try:
                for payload in self.payloads:
                    if isinstance(payload, BaseException):
                        raise payload
                    yield deepcopy(payload)
            finally:
                self.closed_streams += 1

        return stream()

    def health(self):
        self.gets.append(("health", None))
        return deepcopy(self.health_payload)

    def models(self):
        self.gets.append(("models", None))
        return deepcopy(self.catalog)

    def props(self, model=None):
        self.gets.append(("props", model))
        return deepcopy(self.properties)

    def close(self):
        self.closed = True


SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
    "additionalProperties": False,
}


def test_shared_transport_preserves_history_sampling_reasoning_and_usage():
    client = Client()
    client.payloads = [
        chunk(reasoning_content="reason"),
        chunk("done"),
        chunk(finish="stop"),
        {
            "choices": [],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
        },
    ]
    adapter = StrataAdapter(
        model="qwen",
        client=client,
        reasoning="low",
        top_k=20,
        min_p=0.05,
        top_p=0.9,
        repeat_penalty=1.05,
        stream_idle_timeout_sec=None,
    )
    messages = [
        {"role": "developer", "content": "policy"},
        {"role": "assistant", "content": "prior", "reasoning_content": "thought"},
        {"role": "user", "content": "go"},
    ]
    result = adapter.generate_with_metadata(
        messages, request_options={"return_progress": True}
    )
    assert isinstance(adapter, LlamaCppAdapter)
    assert result.text == "done" and result.reasoning_text == "reason"
    assert result.usage.total_tokens == 10
    call = client.calls[0]
    assert call["messages"] == messages
    assert call["stream"] is True and call["idle_timeout"] is None
    assert call["reasoning_effort"] == "low"
    assert call["chat_template_kwargs"] == {"enable_thinking": True}
    assert call["top_k"] == 20 and call["min_p"] == 0.05
    assert call["top_p"] == 0.9 and call["repeat_penalty"] == 1.05
    assert call["return_progress"] is True
    assert client.gets == [] and client.closed_streams == 1


def test_schema_mapping_and_local_validation_use_common_api():
    client = Client('{"value":42}')
    adapter = StrataAdapter(
        model="qwen", client=client, output_mode="json", json_schema=SCHEMA
    )
    assert adapter.invoke("go") == '{"value":42}'
    assert client.calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "runtime_llama_response",
            "schema": SCHEMA,
            "strict": True,
        },
    }
    assert SCHEMA["required"] == ["value"]


@pytest.mark.parametrize(
    "text", ['{"unexpected":42}', '{"value":"wrong"}', '{"value":42,"extra":1}']
)
def test_server_schema_validation_omission_never_becomes_success(text):
    client = Client(text)
    adapter = StrataAdapter(
        model="qwen", client=client, output_mode="json", json_schema=SCHEMA
    )
    with pytest.raises(LLMBackendError, match="strata_json_schema_mismatch") as caught:
        adapter.generate_with_metadata("go")
    assert text not in str(caught.value)
    assert caught.value.diagnostics.terminal_reason == "backend_error"
    assert client.closed_streams == 1
    client.payloads = [chunk('{"value":2}'), chunk(finish="stop")]
    assert adapter.generate("recover") == '{"value":2}'
    assert client.closed_streams == 2


def test_unsatisfiable_schema_negative_canary_is_checked_locally():
    client = Client('{"value":42}')
    adapter = StrataAdapter(model="qwen", client=client)
    with pytest.raises(LLMBackendError, match="strata_json_schema_mismatch"):
        adapter.generate(
            "go",
            request_options={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "impossible",
                        "schema": {**SCHEMA, "not": {}},
                    },
                }
            },
        )


def test_local_schema_references_and_draft_validation():
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": "#/$defs/object",
        "$defs": {"object": SCHEMA},
    }
    adapter = StrataAdapter(
        model="qwen",
        client=Client('{"value":42}'),
        output_mode="json",
        json_schema=schema,
    )
    assert adapter.generate("go") == '{"value":42}'


@pytest.mark.parametrize(
    "reference", ["https://schemas.example.net/secret", "file:///private/secret"]
)
def test_schema_cannot_trigger_external_reference_fetch(reference):
    client = Client('{"value":42}')
    adapter = StrataAdapter(
        model="qwen",
        client=client,
        output_mode="json",
        json_schema={"type": "object", "$ref": reference},
    )
    with pytest.raises(ValueError, match="local_references"):
        adapter.generate("go")
    assert client.calls == []


def test_invalid_schema_is_rejected_before_http():
    client = Client("{}")
    adapter = StrataAdapter(
        model="qwen", client=client, output_mode="json", json_schema={"type": "unknown"}
    )
    with pytest.raises(ValueError, match="strata_invalid_json_schema"):
        adapter.generate("go")
    assert client.calls == []


@pytest.mark.parametrize("dialect", [[], {}, 42, "https://schemas.example.net/unknown"])
def test_schema_dialect_is_validated_without_lookup_or_type_error(dialect):
    client = Client("{}")
    adapter = StrataAdapter(
        model="qwen",
        client=client,
        output_mode="json",
        json_schema={"$schema": dialect, "type": "object"},
    )
    with pytest.raises(ValueError, match="strata_.*json_schema"):
        adapter.generate("go")
    assert client.calls == []


def test_cyclic_python_schema_input_does_not_hang_reference_scan():
    schema = {"type": "object"}
    schema["properties"] = {"child": schema}
    client = Client("{}")
    adapter = StrataAdapter(
        model="qwen", client=client, output_mode="json", json_schema=schema
    )
    with pytest.raises(ValueError, match="strata_invalid_json_schema"):
        adapter.generate("go")
    assert client.calls == []


@pytest.mark.parametrize(
    "text", ["not-json", "[]", "null", '{"x":NaN}', '{"x":Infinity}']
)
def test_json_object_requires_one_strict_json_object(text):
    client = Client(text)
    adapter = StrataAdapter(model="qwen", client=client, output_mode="json")
    with pytest.raises(LLMBackendError, match="strata_.*json"):
        adapter.generate("go")
    assert client.closed_streams == 1


def test_failed_structured_stream_has_no_success_terminal_and_closes():
    client = Client('{"wrong":42}')
    adapter = StrataAdapter(
        model="qwen", client=client, output_mode="json", json_schema=SCHEMA
    )
    stream = adapter.stream_with_metadata("go")
    events = []
    with pytest.raises(LLMBackendError, match="strata_json_schema_mismatch"):
        for event in stream:
            events.append(event)
    assert all(event.kind != "terminal" for event in events)
    assert client.closed_streams == 1


def test_truncated_structured_output_is_partial_rather_than_schema_failure():
    client = Client()
    client.payloads = [chunk('{"value":'), chunk(finish="length")]
    adapter = StrataAdapter(
        model="qwen", client=client, output_mode="json", json_schema=SCHEMA
    )
    result = adapter.generate_with_metadata("go")
    assert result.text == '{"value":' and result.terminal_reason == "length_cap"
    assert client.closed_streams == 1


def test_missing_local_reference_raises_typed_error_without_details():
    client = Client('{"value":42}')
    adapter = StrataAdapter(
        model="qwen",
        client=client,
        output_mode="json",
        json_schema={"$ref": "#/$defs/missing"},
    )
    with pytest.raises(LLMBackendError, match="strata_json_schema_unresolvable"):
        adapter.generate("go")
    assert client.closed_streams == 1


def test_tool_fragments_keep_two_calls_and_correlated_history():
    client = Client()
    client.payloads = [
        chunk(
            tool_calls=[
                {
                    "index": i,
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a":'},
                }
                for i in range(2)
            ]
        ),
        chunk(
            tool_calls=[
                {"index": i, "function": {"arguments": str(i) + "}"}} for i in range(2)
            ]
        ),
        chunk(finish="tool_calls"),
    ]
    adapter = StrataAdapter(model="qwen", client=client)
    result = adapter.generate_with_metadata(
        "go",
        tools=[{"type": "function", "function": {"name": "add"}}],
        tool_choice="required",
    )
    assert [call["id"] for call in result.tool_calls] == ["call_0", "call_1"]
    assert [
        json.loads(call["function"]["arguments"]) for call in result.tool_calls
    ] == [{"a": 0}, {"a": 1}]
    assert result.terminal_reason == "tool_calls" and client.closed_streams == 1
    history = [
        {"role": "assistant", "content": None, "tool_calls": list(result.tool_calls)},
        {"role": "tool", "tool_call_id": "call_0", "content": "0"},
    ]
    client.payloads = [chunk("answer"), chunk(finish="stop")]
    assert adapter.generate(history) == "answer"
    assert client.calls[-1]["messages"] == history


def test_cancellation_before_post_and_during_stream_keep_client_recoverable():
    client = Client()
    adapter = StrataAdapter(model="qwen", client=client)
    with pytest.raises(LLMCancelledError):
        adapter.generate("go", cancellation_probe=lambda: True)
    assert client.calls == []
    cancel = False
    stream = adapter.stream_with_metadata("go", cancellation_probe=lambda: cancel)
    assert next(stream).kind == "content"
    cancel = True
    with pytest.raises(LLMCancelledError):
        next(stream)
    assert client.closed_streams == 1
    assert adapter.generate("recover") == "done"


@pytest.mark.parametrize(
    "error",
    [LLMTimeoutError("timeout"), ClientHTTPError(503, error_code="unavailable")],
)
def test_typed_transport_failure_closes_and_next_request_works(error):
    client = Client()
    client.payloads = [error]
    adapter = StrataAdapter(model="qwen", client=client)
    with pytest.raises(type(error)):
        adapter.generate("go")
    assert client.closed_streams == 1
    client.payloads = [chunk("recovered"), chunk(finish="stop")]
    assert adapter.generate("recover") == "recovered"


@pytest.mark.parametrize("model", ["qwen", "alias"])
def test_health_proves_canonical_alias_readiness_without_model_query(model):
    client = Client()
    client.catalog["data"].append(
        {
            "id": "alias",
            "alias_of": "qwen",
            "aliases": ["alias"],
            "status": {"value": "loaded"},
        }
    )
    adapter = StrataAdapter(model=model, client=client)
    status = adapter.health_status()
    assert status["backend"] == "strata"
    assert status["model_ready"] is True
    assert status["loaded_context_length"] == 8192 and status["observed_slots"] == 1
    assert status["process_lifecycle"] == "external"
    assert client.gets == [("health", None), ("models", None), ("props", None)]
    assert client.calls == []


def test_health_does_not_inspect_model_props_when_unloaded():
    client = Client()
    client.health_payload["loaded"] = False
    client.catalog["data"][0]["status"]["value"] = "unloaded"
    status = StrataAdapter(model="qwen", client=client).health_status()
    assert status["model_ready"] is False and status["model_available"] is True
    assert status["reason"] == "model_not_loaded"
    assert client.gets == [("health", None), ("models", None)]


def test_conflicting_alias_row_is_not_accepted_as_same_loaded_model():
    client = Client()
    client.catalog["data"].append(
        {
            "id": "alias",
            "alias_of": "qwen",
            "aliases": ["alias"],
            "status": {"value": "unloaded"},
        }
    )
    status = StrataAdapter(model="alias", client=client).health_status()
    assert status["model_ready"] is False and status["reason"] == "metadata_unavailable"
    assert client.gets == [("health", None), ("models", None)]


def test_health_does_not_accept_another_service_or_stale_props():
    client = Client()
    client.health_payload["service"] = "other"
    adapter = StrataAdapter(model="qwen", client=client)
    assert adapter.health_status()["model_ready"] is False
    assert client.gets == [("health", None)]
    client.health_payload["service"] = "strata"
    client.properties["model_alias"] = "other-model"
    status = adapter.health_status()
    assert (
        status["model_ready"] is False and status["reason"] == "model_metadata_mismatch"
    )
    assert status["loaded_context_length"] is None


def test_borrowed_client_remains_caller_owned():
    client = Client()
    adapter = StrataAdapter(model="qwen", client=client)
    adapter.close()
    assert client.closed is False


def test_early_structured_iterator_close_drains_without_validation_or_terminal():
    client = Client('{"unfinished":')
    adapter = StrataAdapter(
        model="qwen", client=client, output_mode="json", json_schema=SCHEMA
    )
    stream = adapter.stream_with_metadata("go")
    assert next(stream).kind == "content"
    stream.close()
    assert stream.gi_frame is None
    assert client.closed_streams == 1


def test_real_http_schema_mapping_negative_canary_and_recovery(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    requests_seen = []
    responses = ['{"unexpected":42}', '{"value":42}']

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            size = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(size))
            requests_seen.append((self.path, request))
            text = responses.pop(0)
            payload = (
                b"".join(
                    b"data: " + json.dumps(item).encode() + b"\n\n"
                    for item in [chunk(text), chunk(finish="stop")]
                )
                + b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    adapter = StrataAdapter(
        model="qwen",
        base_url=f"http://127.0.0.1:{server.server_port}",
        output_mode="json",
        json_schema=SCHEMA,
    )
    assert adapter._client.trust_env is False
    assert adapter._client._session.trust_env is False
    try:
        with pytest.raises(LLMBackendError, match="strata_json_schema_mismatch"):
            adapter.generate("conflicting prompt", timeout=2)
        assert adapter.generate("recover", timeout=2) == '{"value":42}'
    finally:
        adapter.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert len(requests_seen) == 2
    for path, request in requests_seen:
        assert path == "/v1/chat/completions"
        assert request["model"] == "qwen" and request["stream"] is True
        assert request["response_format"]["type"] == "json_schema"
        assert request["response_format"]["json_schema"]["schema"] == SCHEMA
        assert "n_ctx" not in request and "auto_load" not in request
