"""Synchronous waiter interruption retires one owner and waits for real drain."""

import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent
from threading import Event, Thread

import pytest

from runtime_llama.errors import LLMBackendError, LLMCancelledError

from test_embedding_contracts import ContractClient, configured


_SIGNAL_SCENARIO = dedent(
    r"""
    from collections import Counter
    from concurrent.futures import Future, ThreadPoolExecutor
    import os
    import signal
    import sys
    from threading import Event, Thread, current_thread, main_thread

    from runtime_llama.errors import LLMCancelledError
    from test_embedding_contracts import ContractClient, configured, eventually, vector
    from test_embedding_shared_lifecycle import SharedClient

    phase = sys.argv[1]
    client = SharedClient() if phase == "shared" else ContractClient(1)
    client.supports_embedding_token_ids = True
    adapter = configured(client, dimension=2, batch_size=4,
        embedding_input_format="text", embedding_batch_wait_sec=0,
        embedding_request_parallelism=1, embedding_queue_size=4, timeout=5)
    active, waiting, cancelled, drain, returned = (Event() for _ in range(5))
    woke_once, woke_twice = Event(), Event()
    captured, wakeups, thread_errors = {}, [], []
    first_interrupt = KeyboardInterrupt("first real SIGINT")
    signals = []

    def interrupt(signum, frame):
        signals.append(signum)
        raise first_interrupt if len(signals) == 1 else KeyboardInterrupt("repeat SIGINT")

    signal.signal(signal.SIGINT, interrupt)
    submit, wake = adapter._submit_execute, adapter._engine.wake

    def observed_submit(*args, **kwargs):
        future = submit(*args, **kwargs)
        if current_thread() is main_thread():
            captured.update(call=kwargs["call"], future=future)
            result = future.result
            def observed_result(*args, **kwargs):
                waiting.set()
                return result(*args, **kwargs)
            future.result = observed_result
        return future

    def observed_wake():
        wake()
        wakeups.append(True)
        woke_once.set()
        if len(wakeups) >= 2:
            woke_twice.set()

    adapter._submit_execute, adapter._engine.wake = observed_submit, observed_wake
    tokenizer, embeddings, props = client.tokenize, client.embeddings, client.props
    never = Event()

    def cancelled_worker(run):
        active.set()
        try:
            return run()
        except LLMCancelledError:
            cancelled.set()
            assert drain.wait(3), "actual worker drain was not released"
            raise

    if phase == "preparation":
        client.token_gate.clear()
        def tokenize(content, **kwargs):
            return cancelled_worker(lambda: tokenizer(content, **kwargs))
        client.tokenize = tokenize
    elif phase == "http":
        client.embed_gate.clear()
        def embed(input, **kwargs):
            return cancelled_worker(lambda: embeddings(input, **kwargs))
        client.embeddings = embed
    elif phase == "metadata":
        def observe(*args, **kwargs):
            def request():
                client._wait(never, cancel=kwargs.get("cancel"), timeout=kwargs.get("timeout"))
                return props(*args, **kwargs)
            return cancelled_worker(request)
        client.props = observe

    pool = ThreadPoolExecutor(max_workers=2)
    prime = survivor = None
    if phase == "shared":
        prime = pool.submit(adapter.embed_texts, ["prime"])
        assert client.prime_started.wait(2)
        survivor = pool.submit(adapter.embed_texts, ["b-0", "b-1"])

    def send_signals():
        try:
            assert waiting.wait(2)
            future, call = captured["future"], captured["call"]
            assert isinstance(future, Future) and future.running()
            if phase == "shared":
                eventually(lambda: sum(adapter._engine.snapshot()["ready_by_replica"].values()) == 4)
                client.prime_gate.set()
                assert client.shared_started.wait(2)
                assert Counter(client.shared_posts[0]) == Counter(["a-0", "a-1", "b-0", "b-1"])
                client.retired_owner = call.stop.is_set
            else:
                assert active.wait(2)
            os.kill(os.getpid(), signal.SIGINT)
            assert woke_once.wait(2), "abandoned owner did not wake coordinator"
            assert call.stop.is_set()
            if phase == "shared":
                assert client.retired_owner_polled.wait(2)
                assert not client.transport_cancelled.is_set()
            else:
                assert cancelled.wait(2)
            assert not future.done() and not returned.is_set()
            os.kill(os.getpid(), signal.SIGINT)
            assert woke_twice.wait(2), "repeated interrupt abandoned cleanup"
            assert not future.done() and not returned.is_set()
            if phase in ("http", "shared"):
                assert adapter._scheduler.snapshot()["in_flight"] == 1
            drain.set()
            if phase == "shared":
                client.shared_gate.set()
        except BaseException as error:
            thread_errors.append(error)
        finally:
            drain.set()
            if phase == "shared":
                client.release_all()

    sender = Thread(target=send_signals, name="subprocess_signal_sender", daemon=True)
    sender.start()
    try:
        try:
            adapter.embed_texts(["a-0", "a-1"] if phase == "shared" else ["interrupted"])
        except KeyboardInterrupt as error:
            returned.set()
            assert error is first_interrupt, "cleanup replaced the first interrupt"
            assert drain.is_set(), "synchronous call returned before actual worker drain"
            assert captured["future"].done()
        else:
            raise AssertionError("SIGINT did not interrupt the synchronous waiter")
        sender.join(4)
        assert not sender.is_alive()
        if thread_errors:
            raise thread_errors[0]
        assert len(signals) == 2 and len(wakeups) >= 2
        if phase == "shared":
            assert survivor.result(2) == [vector("b-0"), vector("b-1")]
            assert prime.result(2) == [vector("prime")]
            assert len(client.shared_posts) == 1
            assert not client.transport_cancelled.is_set()
        else:
            assert len(client.embedding_calls) == (1 if phase == "http" else 0)
        assert not adapter._closed and not client.closed
        assert not adapter._calls and adapter._pool_users == 0
        assert adapter._scheduler.snapshot()["in_flight"] == 0
        view = adapter._engine.snapshot()
        for name in ("pending_bytes", "pending_tokens", "retained_plans", "result_reserved_bytes"):
            assert view[name] == 0, (name, view)
        print("PASS: " + phase + " real repeated SIGINT drains only its logical owner")
    finally:
        drain.set()
        never.set()
        client.token_gate.set()
        client.embed_gate.set()
        if phase == "shared":
            client.release_all()
        adapter.close()
        pool.shutdown(wait=True)
        sender.join(4)
    """
)


@pytest.mark.skipif(
    os.name != "posix", reason="real POSIX SIGINT stays in child process"
)
@pytest.mark.parametrize("phase", ["metadata", "preparation", "http", "shared"])
def test_real_repeated_sigint_retires_one_owner_and_waits_actual_drain(phase, tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root / "tests"), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-u", "-c", _SIGNAL_SCENARIO, phase],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: " + phase in result.stdout


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_baseexception_from_real_active_future_preserves_first_error_through_repeated_cleanup(
    interruption,
):
    client = ContractClient(1)
    client.embed_gate.clear()
    adapter = configured(client, embedding_batch_wait_sec=0, timeout=5)
    original = interruption("first waiter interruption")
    cancelled, release = Event(), Event()
    captured, attempts, wakeups = {}, [], []
    submit, request, wake = (
        adapter._submit_execute,
        client.embeddings,
        adapter._engine.wake,
    )

    def observed_submit(*args, **kwargs):
        future = submit(*args, **kwargs)
        captured.update(call=kwargs["call"], future=future)
        result = future.result

        def interrupted_result(*args, **kwargs):
            attempts.append(True)
            if len(attempts) == 1:
                assert client.embed_started.wait(2)
                raise original
            if len(attempts) == 2:
                raise SystemExit("second waiter interruption")
            return result(*args, **kwargs)

        future.result = interrupted_result
        return future

    def cancelled_request(*args, **kwargs):
        try:
            return request(*args, **kwargs)
        except LLMCancelledError:
            cancelled.set()
            assert release.wait(3)
            raise

    def observed_wake():
        wakeups.append(True)
        wake()

    def finish_drain():
        assert cancelled.wait(3)
        assert captured["call"].stop.is_set()
        assert not captured["future"].done()
        release.set()

    adapter._submit_execute = observed_submit
    adapter._engine.wake = observed_wake
    client.embeddings = cancelled_request
    drainer = Thread(target=finish_drain, daemon=True)
    drainer.start()
    try:
        with pytest.raises(interruption) as caught:
            adapter.embed_query("active")
        assert caught.value is original
        assert release.is_set() and captured["future"].done()
        assert len(attempts) >= 3 and len(wakeups) >= 2
        assert not adapter._calls
        assert adapter._scheduler.snapshot()["in_flight"] == 0
    finally:
        release.set()
        client.embed_gate.set()
        adapter.close()
        drainer.join(4)
    assert not drainer.is_alive()


@pytest.mark.parametrize("phase", ["preparation", "http"])
def test_normal_completed_worker_failure_keeps_typed_error_without_waiter_stop(phase):
    client = ContractClient(1)
    adapter = configured(client, embedding_batch_wait_sec=0)
    expected = LLMBackendError("worker typed first failure")
    if phase == "preparation":
        client.token_failure = expected
    else:
        client.failure = lambda _alias, _texts: expected

    def no_waiter_wakeup():
        raise AssertionError("a completed worker failure was treated as abandonment")

    adapter._engine.wake = no_waiter_wakeup
    try:
        with pytest.raises(LLMBackendError) as caught:
            adapter.embed_query("failure")
        assert caught.value is expected
        assert not adapter._calls
        assert adapter._scheduler.snapshot()["in_flight"] == 0
    finally:
        adapter.close()
