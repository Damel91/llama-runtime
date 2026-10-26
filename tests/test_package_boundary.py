import subprocess
import sys
import os
from pathlib import Path

import pytest

from runtime_llama import (
    AdapterProfile,
    LLMBackendError,
    build_adapter,
    build_embedding_adapter,
)


def test_factory_preserves_profile_and_never_sends_http(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("factory must not send HTTP")

    monkeypatch.setattr("requests.Session.request", forbidden)
    generation = build_adapter(
        {"model": "remote-model", "base_url": "https://inference.example.net"}
    )
    embedding = build_embedding_adapter(
        {"model": "remote-embedding", "base_url": "https://inference.example.net"}
    )
    assert generation.model_name == "remote-model"
    assert embedding.model_name == "remote-embedding"
    generation.close()
    embedding.close()


def test_factories_reject_wrong_role():
    profile = AdapterProfile.from_config(
        "embed", {"adapter_type": "embedding", "model": "embedding"}
    )
    with pytest.raises(ValueError):
        build_adapter(profile)


def test_import_without_model_or_sdk_packages():
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'lmstudio', 'openai', 'llamatelemetry', 'torch', 'transformers', 'flow_of_work_mcp', 'codingcastle'}:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
import runtime_llama
assert runtime_llama.__version__ == '0.15.0'
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=environment
    )
    assert completed.returncode == 0, completed.stderr


def test_cancellation_probe_failure_does_not_leak_traceback():
    import traceback

    def broken():
        raise RuntimeError("PRIVATE_PROBE_DETAILS_901")

    adapter = build_adapter({"model": "remote-model"})
    try:
        with pytest.raises(LLMBackendError) as caught:
            adapter.invoke_with_ctx("hello", cancellation_probe=broken)
        assert "PRIVATE_PROBE_DETAILS_901" not in "".join(
            traceback.format_exception(caught.value)
        )
    finally:
        adapter.close()
