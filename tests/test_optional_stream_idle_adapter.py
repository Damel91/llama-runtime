from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event
import time

import pytest

from runtime_llama import AdapterProfile, LlamaCppClient, build_adapter
from runtime_llama.adapters.llama import LlamaCppAdapter
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError
from test_generation import _Client, _chunk
from test_optional_stream_idle_transport import endpoint as endpoint, headers


def frame(payload):
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


FIRST = frame(_chunk(content="first"))
LAST = (
    frame(_chunk(content="second")) + frame(_chunk(finish="stop")) + b"data: [DONE]\n\n"
)


def quiet_chat(endpoint, phase, *, observe_auth=None):
    entered, release = Event(), Event()
    endpoint.release_on_cleanup(release)

    def respond(handler, _body):
        if observe_auth is not None:
            observe_auth.append(handler.headers.get("Authorization"))
        if phase == "headers":
            entered.set()
            release.wait(timeout=3)
        headers(handler)
        if phase == "between":
            handler.wfile.write(FIRST)
            handler.wfile.flush()
        if phase != "headers":
            entered.set()
            release.wait(timeout=3)
        if phase != "between":
            handler.wfile.write(FIRST)
        handler.wfile.write(LAST)
        handler.wfile.flush()

    base_url, observed = endpoint(respond)
    return base_url, observed, entered, release


def configured(base_url="http://127.0.0.1:8080", *, idle=None, client=None):
    return build_adapter(
        {"model": "chosen", "base_url": base_url, "stream_idle_timeout_sec": idle},
        client=client,
    )


@pytest.mark.parametrize("idle", [None, 30.0, 12.0])
def test_adapter_idle_configuration_maps_to_transport_without_payload_leak(idle):
    client = _Client()
    adapter = configured(idle=idle, client=client)
    result = adapter.generate_with_metadata("fixture")
    call = client.calls[0]
    assert call["timeout"] == idle
    if idle is None:
        assert "idle_timeout" in call and call["idle_timeout"] is None
    else:
        assert "idle_timeout" not in call
    assert result.transport.idle_timeout_sec == idle
    assert (
        json.loads(json.dumps(result.transport.to_dict()))["idle_timeout_sec"] == idle
    )
    adapter.close()
    assert not client.client_closed


def test_omitted_configuration_stays_finite_and_per_call_none_inherits_disabled():
    default_client = _Client()
    default = LlamaCppAdapter(model="chosen", client=default_client)
    assert default.generate_with_metadata("fixture").transport.idle_timeout_sec == 30.0
    assert "idle_timeout" not in default_client.calls[0]
    disabled_client = _Client()
    disabled = configured(client=disabled_client)
    assert (
        disabled.generate_with_metadata(
            "fixture", timeout=None
        ).transport.idle_timeout_sec
        is None
    )
    assert disabled_client.calls[0]["idle_timeout"] is None


def test_positive_call_override_restores_finite_transport_and_evidence():
    client = _Client()
    result = configured(client=client).generate_with_metadata("fixture", timeout=0.2)
    assert client.calls[0]["timeout"] == 0.2
    assert "idle_timeout" not in client.calls[0]
    assert result.transport.idle_timeout_sec == 0.2


@pytest.mark.parametrize(
    "value",
    [0, -1, True, False, "12", float("nan"), float("inf"), -float("inf"), 10**1000],
)
def test_invalid_call_timeout_rejected_before_transport(value):
    client = _Client()
    with pytest.raises(ValueError, match="timeout must be positive and finite"):
        configured(client=client).generate_with_metadata("fixture", timeout=value)
    assert client.calls == []


@pytest.mark.parametrize(
    "field",
    [
        "idle_timeout",
        "stream_idle_timeout_sec",
        "cancel_wait",
        "max_response_bytes",
        "trust_env",
        "verify_ssl",
        "api_key",
        "api_key_env",
    ],
)
@pytest.mark.parametrize("location", ["request_options", "extra_body"])
def test_transport_controls_cannot_enter_generation_payload(field, location):
    client = _Client()
    if location == "extra_body":
        adapter = LlamaCppAdapter(
            model="chosen", client=client, extra_body={field: None}
        )
        options = {}
    else:
        adapter = configured(client=client)
        options = {"request_options": {field: None}}
    with pytest.raises(ValueError, match="overrides_managed_fields"):
        adapter.generate_with_metadata("fixture", **options)
    assert client.calls == []


@pytest.mark.parametrize("metadata_stream", [False, True])
def test_pre_cancel_reports_disabled_idle_truthfully_without_a_request(metadata_stream):
    client = _Client()
    adapter = configured(client=client)
    with pytest.raises(LLMCancelledError) as caught:
        if metadata_stream:
            list(
                adapter.stream_with_metadata("fixture", cancellation_probe=lambda: True)
            )
        else:
            adapter.generate_with_metadata("fixture", cancellation_probe=lambda: True)
    assert caught.value.diagnostics.idle_timeout_sec is None
    assert caught.value.diagnostics.to_dict()["idle_timeout_sec"] is None
    assert client.calls == []


@pytest.mark.parametrize("phase", ["headers", "first", "between"])
def test_owned_factory_client_waits_through_quiet_stream_and_returns_null_evidence(
    endpoint, phase
):
    base_url, observed, entered, release = quiet_chat(endpoint, phase)
    adapter = configured(base_url)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(adapter.generate_with_metadata, "fixture")
            assert entered.wait(timeout=1)
            time.sleep(0.08)
            assert not pending.done()
            release.set()
            result = pending.result(timeout=1)
        assert result.text == "firstsecond"
        assert result.transport.idle_timeout_sec is None
        assert (
            json.loads(json.dumps(result.transport.to_dict()))["idle_timeout_sec"]
            is None
        )
        assert len(observed) == 1 and observed[0][0] == "/v1/chat/completions"
        assert "idle_timeout" not in observed[0][1]
        assert "timeout" not in observed[0][1]
        assert "stream_idle_timeout_sec" not in observed[0][1]
    finally:
        release.set()
        adapter.close()


@pytest.mark.parametrize("phase", ["headers", "first", "between"])
def test_disabled_adapter_cancellation_aborts_quiet_stream_with_nullable_evidence(
    endpoint, phase
):
    base_url, observed, entered, release = quiet_chat(endpoint, phase)
    adapter, cancel = configured(base_url), Event()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                adapter.generate_with_metadata,
                "fixture",
                cancellation_probe=cancel.is_set,
            )
            assert entered.wait(timeout=1)
            cancel.set()
            with pytest.raises(LLMCancelledError) as caught:
                pending.result(timeout=1)
        assert caught.value.diagnostics.idle_timeout_sec is None
        assert caught.value.diagnostics.to_dict()["idle_timeout_sec"] is None
        assert len(observed) == 1
        started = time.monotonic()
        adapter.close()
        assert time.monotonic() - started < 0.5
    finally:
        release.set()
        adapter.close()


@pytest.mark.parametrize("phase", ["headers", "first", "between"])
def test_owned_adapter_close_interrupts_stream_without_waiting_for_server(
    endpoint, phase
):
    base_url, observed, entered, release = quiet_chat(endpoint, phase)
    adapter = configured(base_url)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(adapter.generate_with_metadata, "fixture")
            assert entered.wait(timeout=1)
            started = time.monotonic()
            adapter.close()
            assert time.monotonic() - started < 0.5
            with pytest.raises(LLMBackendError):
                pending.result(timeout=1)
        assert len(observed) == 1
    finally:
        release.set()
        adapter.close()


@pytest.mark.parametrize("configured_idle,call_timeout", [(0.04, None), (None, 0.04)])
def test_finite_configuration_or_call_override_still_times_out(
    endpoint, configured_idle, call_timeout
):
    base_url, observed, entered, release = quiet_chat(endpoint, "first")
    adapter = configured(base_url, idle=configured_idle)
    try:
        with pytest.raises(LLMTimeoutError) as caught:
            adapter.generate_with_metadata("fixture", timeout=call_timeout)
        assert entered.is_set() and len(observed) == 1
        assert caught.value.diagnostics.idle_timeout_sec == 0.04
    finally:
        release.set()
        adapter.close()


def test_noxi_borrowed_owned_session_client_ignores_ambient_auth_and_proxy(
    endpoint, monkeypatch, tmp_path
):
    netrc = tmp_path / "netrc"
    netrc.write_text("machine 127.0.0.1 login wrong password wrong\n")
    monkeypatch.setenv("NETRC", str(netrc))
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    auth = []
    base_url, observed, entered, release = quiet_chat(
        endpoint, "headers", observe_auth=auth
    )
    profile = AdapterProfile.from_config(
        "chat",
        {
            "model": "chosen",
            "base_url": base_url,
            "stream_idle_timeout_sec": None,
            "api_key": "explicit-bearer",
        },
    )
    client = LlamaCppClient(
        profile.base_url,
        timeout=profile.timeout,
        trust_env=False,
        api_key=profile.generation.api_key,
    )
    adapter = build_adapter(profile.to_llm_config(), client=client)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(adapter.generate_with_metadata, "fixture")
            assert entered.wait(timeout=1)
            adapter.close()  # The injected client belongs to the bridge.
            assert not pending.done()
            release.set()
            result = pending.result(timeout=1)
        assert result.text == "firstsecond"
        assert result.transport.idle_timeout_sec is None
        assert auth == ["Bearer explicit-bearer"]
        assert len(observed) == 1
    finally:
        release.set()
        adapter.close()
        client.close()


def test_disabled_idle_inherits_callers_finite_connection_budget(endpoint):
    base_url, observed, entered, release = quiet_chat(endpoint, "first")
    client = LlamaCppClient(base_url, timeout=0.04, trust_env=False)
    adapter = configured(base_url, client=client)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(adapter.generate_with_metadata, "fixture")
            assert entered.wait(timeout=1)
            time.sleep(0.1)
            assert not pending.done()
            release.set()
            result = pending.result(timeout=1)
        assert result.text == "firstsecond"
        assert result.transport.idle_timeout_sec is None
        assert client.timeout == 0.04
        assert len(observed) == 1
    finally:
        release.set()
        adapter.close()
        client.close()
