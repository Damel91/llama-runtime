"""Reentrant consumer callbacks fail finitely without mutating/draining self."""

import json
import subprocess
import sys

import pytest


_CHILD = r"""
import asyncio
import json
import sys
import threading

sys.path[:0] = json.loads(sys.argv[3])

from runtime_llama.automatic_embedding import AutomaticEmbeddingService
from runtime_llama.errors import LLMBackendError
from test_automatic_embedding import remote

mode = sys.argv[1]
client = remote()
service = AutomaticEmbeddingService({
    "base_url": "http://fixture.invalid:8080",
    "cache_dir": sys.argv[2],
    "timeout": 0.5,
    "calibration_timeout": 0.5,
    "normalize": False,
}, client=client)
checks = 0
armed = False
entered = False
caught = []
callback_thread = None


def cancel():
    global checks, entered, callback_thread
    checks += 1
    trigger = (
        mode in ("sync_initialize", "sync_embed", "stream_readiness", "caught") and checks == 2
        or mode == "before_registration" and checks == 1
        or mode in ("calibration_worker", "stream_http", "sync_http") and armed and not entered
    )
    if trigger:
        entered = True
        callback_thread = threading.current_thread().name
        if mode == "caught":
            try:
                service.close()
            except LLMBackendError as error:
                caught.append(str(error))
        else:
            service.close()
    return False


if mode == "calibration_worker":
    original = client.tokenize
    def tokenize(*args, **kwargs):
        global armed
        armed = True
        return original(*args, **kwargs)
    client.tokenize = tokenize
elif mode in ("stream_http", "sync_http"):
    original = client.embeddings
    def embeddings(*args, **kwargs):
        global armed
        armed = True
        return original(*args, **kwargs)
    client.embeddings = embeddings


async def stream():
    async def windows():
        yield ["one"]
    cursor = service.astream_texts(windows(), cancel=cancel, timeout=0.5)
    try:
        return await asyncio.wait_for(anext(cursor), timeout=1)
    finally:
        await cursor.aclose()


error = None
try:
    if mode.startswith("stream_"):
        asyncio.run(stream())
    elif mode == "calibration_worker":
        service.character_budget(cancel=cancel, timeout=0.5)
    elif mode in ("sync_embed", "sync_http"):
        service.embed_texts(["one"], cancel=cancel, timeout=0.5)
    else:
        service.initialize(cancel=cancel, timeout=0.5)
except LLMBackendError as failure:
    error = str(failure)

# Reject before mutating closing state or stopping unrelated work. Recovery and
# ordinary external closure still own a fully drained, usable service.
assert entered
assert not service._closed and not service._closing.is_set()
assert not service._operations and not service._profile_flights
assert client.token_active == client.embed_active == 0
assert service.initialize().available
service.close()
assert service._closed
assert not client.closed
print(json.dumps({
    "mode": mode,
    "error": error,
    "caught": caught,
    "thread": callback_thread,
    "recovered_and_closed": True,
}))
"""


@pytest.mark.parametrize(
    "mode",
    [
        "before_registration",
        "sync_initialize",
        "sync_embed",
        "sync_http",
        "calibration_worker",
        "stream_readiness",
        "stream_http",
        "caught",
    ],
)
def test_reentrant_close_rejected_without_deadlock_or_service_mutation(tmp_path, mode):
    # A future regression must fail this child within a finite deadline rather
    # than deadlocking pytest or leaving its embedding/calibration workers alive.
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, mode, str(tmp_path), json.dumps(sys.path)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout)
    code = "llamacpp_automatic_embedding_close_from_cancel_callback"
    if mode == "caught":
        assert observed["error"] is None and observed["caught"] == [code]
    else:
        assert observed["error"] == code and not observed["caught"]
    assert observed["recovered_and_closed"]
    if mode == "calibration_worker":
        assert observed["thread"].startswith("llama-calibration_")
    elif mode == "stream_readiness":
        assert observed["thread"].startswith("asyncio_")
    elif mode in ("stream_http", "sync_http"):
        assert observed["thread"].startswith("runtime_llama_embedding_")
