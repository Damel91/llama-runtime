"""Global ownership admission is independent of HTTP replica capacity."""

import asyncio
from collections.abc import Sequence
from threading import Event, Thread
import time

import pytest

from runtime_llama.embedding_resources import (
    ByteBudget,
    EmbeddingIntake,
    EmbeddingResourceLimitError,
)
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError


class Call:
    def __init__(self, timeout=3):
        self.deadline = time.monotonic() + timeout
        self.stop = Event()

    def check(self):
        if self.stop.is_set():
            raise LLMCancelledError()
        if time.monotonic() >= self.deadline:
            raise LLMTimeoutError("intake_timeout")

    def remaining(self):
        self.check()
        return self.deadline - time.monotonic()


def eventually(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.001)


def test_sync_and_async_share_the_same_fifo_owner_budget():
    async def scenario():
        intake = EmbeddingIntake(1, 1024)
        first = intake.acquire_owner(Call())
        sync_entered, sync_release = Event(), Event()
        sync_call, async_call = Call(), Call()

        def sync_owner():
            lease = intake.acquire_owner(sync_call)
            sync_entered.set()
            try:
                assert sync_release.wait(3)
            finally:
                lease.release()

        thread = Thread(target=sync_owner)
        thread.start()
        try:
            await asyncio.to_thread(
                eventually, lambda: intake.snapshot()["owner_waiters"] == 1
            )
            waiting = asyncio.create_task(intake.acquire_owner_async(async_call))
            await asyncio.to_thread(
                eventually, lambda: intake.snapshot()["owner_waiters"] == 2
            )
            first.release()
            assert await asyncio.to_thread(sync_entered.wait, 2)
            assert not waiting.done()
            sync_release.set()
            last = await asyncio.wait_for(waiting, 2)
            assert intake.acquire_owner(async_call) is last
            assert intake.snapshot()["owners"] == 1
            assert intake.snapshot()["peak_owners"] == 1
            last.release()
            last.release()
            assert intake.snapshot()["owners"] == 0
        finally:
            first.release()
            sync_release.set()
            await asyncio.to_thread(thread.join, 3)
            assert not thread.is_alive()

    asyncio.run(scenario())


def test_snapshot_is_immutable_once_and_retains_credits_until_owner_transfer():
    intake = EmbeddingIntake(2, 2048)
    lease = intake.acquire_owner(Call())
    original = ["original", "two"]
    frozen = lease.snapshot(original)
    original[0] = "mutated"
    assert frozen == ("original", "two")
    assert lease.snapshot(frozen) is frozen
    held = intake.snapshot()["snapshot_bytes"]
    assert held > sum(len(text) for text in frozen)
    with pytest.raises(LLMBackendError, match="snapshot_already_set"):
        lease.snapshot(["different"])
    assert intake.snapshot()["snapshot_bytes"] == held
    lease.release()
    assert intake.snapshot()["snapshot_bytes"] == 0
    assert intake.snapshot()["owners"] == 0


def test_impossible_reference_array_is_rejected_before_reading_source():
    class HugeSequence(Sequence):
        def __len__(self):
            return 10**9

        def __getitem__(self, index):
            raise AssertionError("oversized source was read")

    intake = EmbeddingIntake(1, 512)
    lease = intake.acquire_owner(Call())
    try:
        with pytest.raises(EmbeddingResourceLimitError) as failure:
            lease.snapshot(HugeSequence())
        assert failure.value.resource == "snapshot_bytes"
        assert failure.value.limit == 512
        assert failure.value.requested > 512
        assert "reduce" in failure.value.remedy
        assert intake.snapshot()["snapshot_bytes"] == 0
        assert not intake.snapshot()["snapshot_building"]
    finally:
        lease.release()


def test_impossible_text_fails_without_leaking_partial_snapshot():
    intake = EmbeddingIntake(1, 256)
    lease = intake.acquire_owner(Call())
    try:
        with pytest.raises(EmbeddingResourceLimitError, match="snapshot_bytes_limit"):
            lease.snapshot(["small", "large" * 100])
        view = intake.snapshot()
        assert view["snapshot_bytes"] == 0
        assert view["owners"] == 1
        assert not view["snapshot_building"]
        assert lease.snapshot(["retry"]) == ("retry",)
    finally:
        lease.release()


def test_serial_builder_prevents_partial_window_byte_deadlock():
    intake = EmbeddingIntake(3, 210)
    first = intake.acquire_owner(Call())
    first.snapshot(["one"])
    leases = [intake.acquire_owner(Call()), intake.acquire_owner(Call())]
    completed = [Event(), Event()]
    failures = []

    def build(index):
        try:
            assert leases[index].snapshot([str(index)]) == (str(index),)
            completed[index].set()
        except BaseException as exc:
            failures.append(exc)

    threads = [Thread(target=build, args=(index,)) for index in range(2)]
    threads[0].start()
    eventually(lambda: intake.snapshot()["snapshot_building"])
    threads[1].start()
    try:
        eventually(lambda: intake.snapshot()["snapshot_waiters"] == 1)
        assert not completed[0].is_set() and not completed[1].is_set()
        first.release()
        assert completed[0].wait(2)
        assert not completed[1].is_set()
        leases[0].release()
        assert completed[1].wait(2)
        assert intake.snapshot()["peak_snapshot_bytes"] <= 210
        assert not failures
    finally:
        first.release()
        for lease in leases:
            lease.release()
        for thread in threads:
            thread.join(3)
            assert not thread.is_alive()
    assert intake.snapshot()["snapshot_bytes"] == 0


def test_async_snapshot_cancellation_releases_builder_and_partial_bytes():
    async def scenario():
        intake = EmbeddingIntake(3, 210)
        first = intake.acquire_owner(Call())
        first.snapshot(["held"])
        lease = await intake.acquire_owner_async(Call())
        pending = asyncio.create_task(lease.snapshot_async(["waiting"]))
        try:
            await asyncio.to_thread(
                eventually, lambda: intake.snapshot()["snapshot_building"]
            )
            held = intake.snapshot()["snapshot_bytes"]
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert intake.snapshot()["snapshot_bytes"] == held
            assert not intake.snapshot()["snapshot_building"]
            assert intake.snapshot()["snapshot_waiters"] == 0
            assert intake.snapshot()["owners"] == 2
            first.release()
            assert await lease.snapshot_async(["healthy"]) == ("healthy",)
        finally:
            first.release()
            lease.release()
        assert intake.snapshot()["owners"] == 0
        assert intake.snapshot()["snapshot_bytes"] == 0

    asyncio.run(scenario())


def test_async_owner_cancel_and_deadline_remove_waiters_without_new_lease():
    async def scenario():
        intake = EmbeddingIntake(1, 1024)
        first = intake.acquire_owner(Call())
        try:
            waiting = asyncio.create_task(intake.acquire_owner_async(Call()))
            await asyncio.to_thread(
                eventually, lambda: intake.snapshot()["owner_waiters"] == 1
            )
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert intake.snapshot()["owner_waiters"] == 0
            with pytest.raises(LLMTimeoutError):
                await intake.acquire_owner_async(Call(timeout=0.01))
            assert intake.snapshot()["owners"] == 1
            assert intake.snapshot()["owner_waiters"] == 0
            assert not intake._signals
        finally:
            first.release()

    asyncio.run(scenario())


def test_close_wakes_waiters_without_releasing_live_owner_storage():
    intake = EmbeddingIntake(1, 1024)
    first = intake.acquire_owner(Call())
    first.snapshot(["held"])
    failure = []

    def wait():
        try:
            intake.acquire_owner(Call())
        except BaseException as exc:
            failure.append(exc)

    thread = Thread(target=wait)
    thread.start()
    try:
        eventually(lambda: intake.snapshot()["owner_waiters"] == 1)
        used = intake.snapshot()["snapshot_bytes"]
        intake.close()
        thread.join(2)
        assert not thread.is_alive()
        assert len(failure) == 1
        assert isinstance(failure[0], LLMBackendError)
        assert "closed" in str(failure[0])
        assert intake.snapshot()["owners"] == 1
        assert intake.snapshot()["snapshot_bytes"] == used
        with pytest.raises(LLMBackendError, match="closed"):
            intake.acquire_owner(Call())
    finally:
        first.release()
        thread.join(3)
    assert intake.snapshot()["owners"] == intake.snapshot()["snapshot_bytes"] == 0


def test_byte_handoff_and_control_progress_are_independent_bounded_pools():
    notifications = []
    handoff = ByteBudget(512, notify=lambda: notifications.append("released"))
    control = ByteBudget(128, name="control_bytes")
    first = handoff.try_acquire(400)
    second = handoff.try_acquire(112, family="response")
    assert handoff.try_acquire(1) is None
    progress = control.try_acquire(128, family="control")
    assert progress is not None
    assert handoff.snapshot()["used"] == handoff.snapshot()["peak"] == 512
    with pytest.raises(EmbeddingResourceLimitError) as failure:
        handoff.try_acquire(513)
    assert failure.value.resource == "handoff_bytes"
    assert failure.value.requested == 513
    first.release()
    first.release()
    assert notifications == ["released"]
    handoff.close()
    with pytest.raises(LLMBackendError, match="closed"):
        handoff.try_acquire(1)
    assert handoff.snapshot()["used"] == 112
    second.release()
    progress.release()
    assert handoff.snapshot()["used"] == handoff.snapshot()["leases"] == 0
    assert control.snapshot()["used"] == 0


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_resource_configuration_requires_positive_integer_limits(value):
    with pytest.raises(ValueError):
        EmbeddingIntake(value, 1024)
    with pytest.raises(ValueError):
        EmbeddingIntake(1, value)
    with pytest.raises(ValueError):
        ByteBudget(value)


def test_sync_drain_waits_real_release_and_leaves_async_cleanup_to_aclose():
    async def scenario():
        intake = EmbeddingIntake(2, 1024)
        sync_lease = intake.acquire_owner(Call())
        async_lease = await intake.acquire_owner_async(Call())
        started, drained = Event(), Event()

        def close():
            intake.close()
            started.set()
            intake.drain_sync()
            drained.set()

        thread = Thread(target=close)
        thread.start()
        try:
            assert await asyncio.to_thread(started.wait, 2)
            assert not drained.is_set()
            assert intake.snapshot()["owners"] == 2
            sync_lease.release()
            assert await asyncio.to_thread(drained.wait, 2)
            assert intake.snapshot()["owners"] == 1
            assert not async_lease.released
            async_lease.release()
        finally:
            sync_lease.release()
            async_lease.release()
            await asyncio.to_thread(thread.join, 3)
            assert not thread.is_alive()

    asyncio.run(scenario())


def test_sync_drain_preserves_first_interrupt_until_ownership_really_ends(monkeypatch):
    intake = EmbeddingIntake(1, 1024)
    acquired, release = Event(), Event()

    def owner():
        lease = intake.acquire_owner(Call())
        acquired.set()
        try:
            assert release.wait(3)
        finally:
            lease.release()

    thread = Thread(target=owner)
    thread.start()
    assert acquired.wait(2)
    intake.close()
    attempts = []
    original_wait = intake._condition.wait

    def wait():
        attempts.append(True)
        if len(attempts) == 1:
            raise KeyboardInterrupt("primary")
        if len(attempts) == 2:
            raise SystemExit("secondary")
        release.set()
        original_wait()

    monkeypatch.setattr(intake._condition, "wait", wait)
    try:
        with pytest.raises(KeyboardInterrupt, match="primary"):
            intake.drain_sync()
        assert len(attempts) == 3
        assert intake.snapshot()["owners"] == 0
    finally:
        release.set()
        thread.join(3)
        assert not thread.is_alive()


def test_reentrant_sync_close_is_rejected_before_closed_state_or_false_release():
    intake = EmbeddingIntake(1, 1024)
    lease = intake.acquire_owner(Call())
    lease.snapshot(["owned"])
    before = intake.snapshot()
    with pytest.raises(LLMBackendError, match="close_from_active_intake"):
        intake.assert_can_drain_sync()
    after = intake.snapshot()
    assert not after["closed"]
    assert after["owners"] == 1
    assert after["snapshot_bytes"] == before["snapshot_bytes"]
    assert not lease.released
    intake.close()
    with pytest.raises(LLMBackendError, match="close_from_active_intake"):
        intake.drain_sync()
    assert not lease.released
    lease.release()
    intake.assert_can_drain_sync()
    intake.drain_sync()
    assert intake.snapshot()["owners"] == 0
