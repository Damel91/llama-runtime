"""Automatic streams reuse admission without caching dispatch authority."""

import asyncio
from collections import Counter
from threading import Event
from types import SimpleNamespace

import pytest

from runtime_llama.embedding_stream import EmbeddingWindowStream
from runtime_llama.errors import LLMBackendError, LLMCancelledError

from test_automatic_embedding import make_service as make_service, remote
from test_embedding_async_stream import WindowSource, run, wait_thread, wait_until
from test_embedding_contracts import vector


def tracked_remote(replicas=2):
    client = remote(replicas=replicas)
    counters = Counter()
    for name in ("health", "models", "props"):
        original = getattr(client, name)

        def observed(*args, _name=name, _original=original, **kwargs):
            with client.lock:
                counters[_name] += 1
            return _original(*args, **kwargs)

        setattr(client, name, observed)
    return client, counters


def expected(window):
    return [vector("search_document: " + text) for text in window]


def test_windows_reuse_one_admission_but_each_post_observes_properties(make_service):
    async def scenario():
        client, counters = tracked_remote()
        service, _ = make_service(client, batch_size=4, request_parallelism=2)
        await asyncio.to_thread(service.embed_texts, ["prime0", "prime1"])
        baseline = counters.copy()
        first_post = len(client.embedding_calls)
        windows = [[f"window-{i}-text-{j}" for j in range(4)] for i in range(8)]
        async with service.astream_texts(
            WindowSource(windows), max_pending_windows=2
        ) as stream:
            result = [window async for window in stream]
        assert result == [expected(window) for window in windows]
        posts = client.embedding_calls[first_post:]
        assert len(posts) == 16 and all(len(batch) == 2 for _, batch in posts)
        # The initial stream readiness remains fresh. Its later windows do not
        # repeat health/discovery/all-replica properties before preparation.
        assert counters["health"] - baseline["health"] == 1
        # Two readiness observations plus one fresh properties request per POST.
        assert counters["props"] - baseline["props"] == 2 + len(posts)
        assert client.alias_peak == {"embed_0": 1, "embed_1": 1}
        assert not client.token_calls and service._adapter._pool_users == 0
        await service.aclose()

    run(scenario)


def test_slow_alias_preflight_does_not_block_next_window_on_other_alias(
    make_service, monkeypatch
):
    async def scenario():
        client, counters = tracked_remote()
        service, _ = make_service(
            client, batch_size=4, request_parallelism=4, timeout=5
        )
        await asyncio.to_thread(service.embed_texts, ["prime0", "prime1"])
        baseline = counters.copy()
        first_post = len(client.embedding_calls)
        properties = client.props
        observed = Counter()
        blocked, release = Event(), Event()

        def slow_properties(model=None, **kwargs):
            observed[model] += 1
            # Each alias has one readiness observation. The next observation
            # belongs to dispatch, so a new window must not wait for this one.
            if model == "embed_1" and observed[model] == 2:
                blocked.set()
                client._wait(
                    release,
                    cancel=kwargs.get("cancel"),
                    timeout=kwargs.get("timeout"),
                )
            return properties(model, **kwargs)

        monkeypatch.setattr(client, "props", slow_properties)
        windows = [[f"window-{i}-text-{j}" for j in range(4)] for i in range(3)]
        stream = service.astream_texts(WindowSource(windows), max_pending_windows=2)
        pending = asyncio.create_task(anext(stream))
        try:
            await wait_thread(blocked)
            await wait_until(
                lambda: (
                    sum(
                        model == "embed_0"
                        for model, _ in client.embedding_calls[first_post:]
                    )
                    >= 2
                )
            )
            assert not release.is_set() and not pending.done()
            assert counters["health"] - baseline["health"] == 1
            release.set()
            results = [await pending]
            results.extend([window async for window in stream])
            assert results == [expected(window) for window in windows]
            assert service._adapter._pool_users == 0
        finally:
            release.set()
            await stream.aclose()
            await service.aclose()

    run(scenario)


def test_open_idle_scope_never_owns_a_pool_reference(make_service):
    async def scenario():
        service, _ = make_service()
        source = WindowSource([["later"]], block_at=0)
        stream = service.astream_texts(source, close_source=True)
        try:
            await stream.__aenter__()
            await source.wait_pulls(1)
            assert service._adapter._pool_users == 0
            source.release.set()
            assert await anext(stream) == expected(["later"])
            await stream.aclose()
            assert service._adapter._pool_users == 0
            assert not service._streams and source.close_calls == 1
        finally:
            source.release.set()
            await stream.aclose()
            await service.aclose()

    run(scenario)


def test_late_window_still_rejects_fresh_unloaded_model(make_service):
    async def scenario():
        client = remote(replicas=1)
        service, _ = make_service(client, batch_size=2)
        await asyncio.to_thread(service.embed_texts, ["prime"])
        first_post = len(client.embedding_calls)
        source = WindowSource([["first", "second"], ["must-not-send"]])
        async with service.astream_texts(source, max_pending_windows=1) as stream:
            assert await anext(stream) == expected(["first", "second"])
            client.catalog[0]["status"]["value"] = "unloaded"
            with pytest.raises(LLMBackendError, match="not_loaded"):
                await anext(stream)
        assert len(client.embedding_calls) == first_post + 1
        assert service._adapter._pool_users == 0 and not client.token_calls
        await service.aclose()

    run(scenario)


def test_each_stream_gets_its_own_scope_and_sync_calls_stay_unscoped(
    make_service, monkeypatch
):
    async def scenario():
        service, _ = make_service(batch_size=4)
        await asyncio.to_thread(service.embed_texts, ["prime0", "prime1"])
        adapter = service._adapter
        submit = adapter._stream_submit
        scopes = []

        def record_scope(window, *, call, **kwargs):
            scopes.append(getattr(call, "_admission_scope", None))
            return submit(window, call=call, **kwargs)

        monkeypatch.setattr(adapter, "_stream_submit", record_scope)
        first = service.astream_texts(WindowSource([["one"], ["two"]]))
        second = service.astream_texts(WindowSource([["three"]]))
        async with first:
            assert [window async for window in first] == [
                expected(["one"]),
                expected(["two"]),
            ]
        async with second:
            assert [window async for window in second] == [expected(["three"])]
        assert scopes[:2] == [first._admission_scope] * 2
        assert scopes[2:] == [second._admission_scope]
        assert first._admission_scope is not second._admission_scope
        assert not hasattr(adapter, "_admission_scope")
        assert await asyncio.to_thread(service.embed_texts, ["sync"]) == expected(
            ["sync"]
        )
        assert adapter._pool_users == 0
        await service.aclose()

    run(scenario)


def test_stream_cancellation_drains_all_scoped_window_references(make_service):
    async def scenario():
        client = remote()
        service, _ = make_service(client, batch_size=4, timeout=5)
        await asyncio.to_thread(service.embed_texts, ["prime0", "prime1"])
        client.embed_started.clear()
        client.embed_gate.clear()
        cancelled = Event()
        source = WindowSource([["one", "two"], ["three", "four"]])
        stream = service.astream_texts(
            source, cancel=cancelled.is_set, close_source=True
        )
        pending = asyncio.create_task(anext(stream))
        try:
            await wait_thread(client.embed_started)
            cancelled.set()
            with pytest.raises(LLMCancelledError):
                await pending
            assert client.embed_active == service._adapter._pool_users == 0
            assert not service._streams and not service._adapter._calls
            assert source.close_calls == 1
        finally:
            client.embed_gate.set()
            await stream.aclose()
            await service.aclose()

    run(scenario)


def test_exact_stream_constructor_cannot_accept_direct_admission_scope():
    adapter = SimpleNamespace(profile=SimpleNamespace(timeout=5))
    with pytest.raises(ValueError, match="direct streams"):
        EmbeddingWindowStream(
            adapter,
            WindowSource([["one"]]),
            prepared=True,
            _admission_scope=object(),
        )
