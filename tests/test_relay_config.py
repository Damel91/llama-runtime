"""Configuration/admin producer contracts without sockets or web dependencies."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from runtime_llama.relay.admin_contract import (
    ADMIN_API_VERSION,
    ADMIN_ROUTES,
    ADMIN_TOKEN_HEADER,
    admin_contract,
)
from runtime_llama.relay.config import (
    RelayConfigConflict,
    RelayConfigError,
    RelayConfigStore,
    default_config,
    validate_config,
)


def endpoint(name="primary", **changes):
    return {"name": name, "base_url": "http://fixture.invalid:8080/v1", **changes}


def test_import_remains_independent_of_web_packages(tmp_path):
    script = """
import importlib.abc
import sys
class BlockWeb(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'starlette', 'uvicorn', 'anyio'}:
            raise AssertionError('relay foundation imported a web dependency')
sys.meta_path.insert(0, BlockWeb())
import runtime_llama.relay
from runtime_llama.relay.config import RelayConfigStore
assert RelayConfigStore().snapshot().revision == 0
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert not list(tmp_path.iterdir())


def test_construction_and_validation_do_not_persist_or_connect(tmp_path, monkeypatch):
    from runtime_llama.client import LlamaCppClient

    def network_forbidden(*args, **kwargs):
        pytest.fail("configuration validation attempted HTTP")

    monkeypatch.setattr(LlamaCppClient, "health", network_forbidden)
    monkeypatch.setattr(LlamaCppClient, "models", network_forbidden)
    monkeypatch.setattr(LlamaCppClient, "props", network_forbidden)
    path = tmp_path / "not-created" / "relay.json"
    source = {"runtime_endpoints": [endpoint(use_tailnet=True)]}
    store = RelayConfigStore(path, source)
    snapshot = store.snapshot()
    source["runtime_endpoints"][0]["name"] = "changed"
    assert snapshot.config["runtime_endpoints"][0]["name"] == "primary"
    assert (
        snapshot.config["runtime_endpoints"][0]["base_url"]
        == "http://fixture.invalid:8080"
    )
    assert snapshot.config["runtime_endpoints"][0]["use_tailnet"] is True
    assert snapshot.config["llm_defaults"] == {}
    assert not path.parent.exists()


def test_snapshot_nested_values_and_settings_are_independent():
    store = RelayConfigStore(
        initial_config={
            "llm_defaults": {
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": True},
            },
            "model_overrides": {"chat": {"llm": {"top_k": 0}}},
        }
    )
    snapshot = store.snapshot()
    value = snapshot.config
    value["llm_defaults"]["chat_template_kwargs"]["enable_thinking"] = False
    settings = snapshot.settings("chat", "llm")
    assert settings["temperature"] == settings["top_k"] == 0
    assert settings["chat_template_kwargs"] == {"enable_thinking": True}
    settings["chat_template_kwargs"].clear()
    assert snapshot.settings("chat", "llm")["chat_template_kwargs"]
    with pytest.raises(TypeError):
        snapshot._data["listener"]["port"] = 9000


def test_nullable_idle_is_explicit_and_zero_values_survive():
    store = RelayConfigStore(
        initial_config={
            "llm_defaults": {
                "stream_idle_timeout_sec": None,
                "temperature": 0,
                "top_k": 0,
            },
            "embedding_defaults": {"queue_size": 0, "batch_size": 10000},
        }
    )
    data = store.snapshot().config
    assert data["llm_defaults"]["stream_idle_timeout_sec"] is None
    assert data["llm_defaults"]["temperature"] == data["llm_defaults"]["top_k"] == 0
    assert data["embedding_defaults"]["queue_size"] == 0
    assert data["embedding_defaults"]["batch_size"] == 10000


def test_secret_redaction_preserves_schema_property_names_and_repr(tmp_path):
    secret = "fixture-backend-key"
    proxy = "http://runtime:" + "x" * 32 + "@127.0.0.1:7777"
    config = {
        "runtime_endpoints": [endpoint(api_key=secret, proxy_url=proxy)],
        "access": {"api_key": "fixture-local-key"},
        "tailnet": {
            "executable": str(tmp_path / "helper"),
            "state_dir": str(tmp_path / "node-state"),
            "hostname": "relay-fixture",
            "auth_key": "fixture-node-key",
        },
        "llm_defaults": {
            "json_schema": {
                "type": "object",
                "properties": {"api_key": {"type": "string"}},
            }
        },
    }
    snapshot = RelayConfigStore(initial_config=config).snapshot()
    public = snapshot.public()
    exported = json.dumps(public)
    assert all(
        value not in exported
        for value in [secret, proxy, "fixture-local-key", "fixture-node-key"]
    )
    assert set(public["secret_fields"]) == {
        "/runtime_endpoints/0/api_key",
        "/runtime_endpoints/0/proxy_url",
        "/access/api_key",
        "/tailnet/auth_key",
    }
    assert "api_key" in public["config"]["llm_defaults"]["json_schema"]["properties"]
    assert "fixture" not in repr(snapshot)
    assert snapshot.config["runtime_endpoints"][0]["api_key"] == secret


def test_omitted_secret_preserves_by_endpoint_name_after_reorder():
    store = RelayConfigStore(
        initial_config={
            "runtime_endpoints": [
                endpoint("alpha", api_key="alpha-secret"),
                endpoint("beta", api_key="beta-secret"),
            ]
        }
    )
    public = store.snapshot().public()["config"]
    public["runtime_endpoints"].reverse()
    updated = store.replace(0, public)
    assert [
        (item["name"], item["api_key"]) for item in updated.config["runtime_endpoints"]
    ] == [("beta", "beta-secret"), ("alpha", "alpha-secret")]
    assert len(updated.public()["secret_fields"]) == 2


def test_clear_replace_and_renamed_endpoint_never_inherit_other_secrets():
    store = RelayConfigStore(
        initial_config={
            "runtime_endpoints": [endpoint("alpha", api_key="alpha-secret")],
            "access": {"api_key": "old-local-secret"},
        }
    )
    public = store.snapshot().public()["config"]
    public["runtime_endpoints"][0]["api_key"] = None
    public["access"]["api_key"] = "new-local-secret"
    snapshot = store.replace(0, public)
    assert snapshot.config["runtime_endpoints"][0]["api_key"] is None
    assert snapshot.config["access"]["api_key"] == "new-local-secret"
    public = snapshot.public()["config"]
    public["runtime_endpoints"][0]["name"] = "another"
    snapshot = store.replace(1, public)
    assert not snapshot.config["runtime_endpoints"][0].get("api_key")


def test_switching_secret_to_environment_requires_explicit_clear():
    store = RelayConfigStore(initial_config={"access": {"api_key": "old-secret"}})
    public = store.snapshot().public()["config"]
    public["access"]["api_key_env"] = "RELAY_FIXTURE_KEY"
    with pytest.raises(RelayConfigError, match="relay_access_authentication_invalid"):
        store.replace(0, public)
    public["access"]["api_key"] = None
    assert (
        store.replace(0, public).config["access"]["api_key_env"] == "RELAY_FIXTURE_KEY"
    )


def test_persistence_roundtrip_private_mode_and_revision_conflict(tmp_path):
    path = tmp_path / "data" / "relay.json"
    store = RelayConfigStore(path)
    config = store.snapshot().config
    config["runtime_endpoints"] = [endpoint(api_key="stored-secret")]
    first = store.replace(0, config)
    assert first.revision == 1
    assert (
        json.loads(path.read_text())["config"]["runtime_endpoints"][0]["api_key"]
        == "stored-secret"
    )
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    reloaded = RelayConfigStore(path)
    assert reloaded.snapshot().config == first.config
    assert reloaded.snapshot().revision == 1
    with pytest.raises(RelayConfigConflict) as error:
        reloaded.replace(0, default_config())
    assert (error.value.expected_revision, error.value.actual_revision) == (0, 1)


def test_failed_atomic_replace_keeps_previous_file_snapshot_and_no_temporary(
    tmp_path, monkeypatch
):
    path = tmp_path / "relay.json"
    store = RelayConfigStore(path)
    store.replace(0, default_config())
    original = path.read_bytes()

    def failure(*args):
        raise OSError("fixture failed rename")

    monkeypatch.setattr(os, "replace", failure)
    with pytest.raises(OSError):
        store.replace(1, {"llm_defaults": {"temperature": 0}})
    assert path.read_bytes() == original
    assert store.snapshot().revision == 1
    assert not list(tmp_path.glob(".relay-config-*"))


def test_concurrent_writers_cannot_lose_revision(tmp_path):
    store = RelayConfigStore(tmp_path / "config.json")

    def write(temperature):
        try:
            return store.replace(
                0, {"llm_defaults": {"temperature": temperature}}
            ).revision
        except RelayConfigConflict:
            return "conflict"

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(write, [0.1, 0.2]))
    assert sorted(results, key=str) == [1, "conflict"]
    assert RelayConfigStore(store.path).snapshot().revision == 1


@pytest.mark.parametrize(
    "config",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"unknown": 1},
        {"listener": {"port": 0}},
        {"listener": {"port": True}},
        {"listener": {"allowed_hosts": ["*"]}},
        {"runtime_endpoints": [endpoint(use_tailnet=1)]},
        {"runtime_endpoints": [endpoint(), endpoint()]},
        {
            "runtime_endpoints": [
                endpoint(base_url="http://user:password@fixture.invalid")
            ]
        },
        {"llm_defaults": {"api_key": "not-an-endpoint-key"}},
        {"llm_defaults": {"temperature": float("nan")}},
        {"llm_defaults": {"instance_count": 2}},
        {"llm_defaults": {"stream_idle_timeout_sec": 0}},
        {"embedding_defaults": {"input_type": []}},
        {"model_overrides": {"model": {"llm": []}}},
        {"model_overrides": {"model": {"embedding": 5}}},
        {"tailnet": {"executable": "relative-path"}},
    ],
)
def test_invalid_configuration_has_safe_error_and_no_effect(config, tmp_path):
    store = RelayConfigStore(tmp_path / "config.json")
    with pytest.raises(RelayConfigError) as error:
        store.replace(0, config)
    assert "password" not in str(error.value)
    assert "not-an-endpoint-key" not in str(error.value)
    assert store.snapshot().revision == 0
    assert not store.path.exists()


def test_admin_contract_is_versioned_and_independent():
    assert ADMIN_API_VERSION == 1
    assert ADMIN_TOKEN_HEADER == "X-Relay-Admin-Token"
    contract = admin_contract()
    assert contract["schema_version"] == 1
    assert len(contract["routes"]) == len(ADMIN_ROUTES)
    assert ("DELETE", "/api/admin/requests/last-completed") in ADMIN_ROUTES
    assert contract["opencode_versions"] == ["v1", "v2"]
    contract["config_fields"].clear()
    assert "runtime_endpoints" in admin_contract()["config_fields"]


def test_config_has_no_catalog_or_model_count_quota():
    overrides = {f"model-{index}": {"llm": {"top_k": 0}} for index in range(1100)}
    validated = validate_config({"model_overrides": overrides})
    assert len(validated["model_overrides"]) == 1100
    assert deepcopy(validated) == validated
