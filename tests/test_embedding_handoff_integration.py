"""Coordinator handoff ownership and progress under genuine resource pressure."""

import asyncio
from collections import Counter, deque
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Event
from types import SimpleNamespace

from runtime_llama.embedding_engine import _Batch, _Entry, _Plan, _State
from runtime_llama.embedding_handoff import HandoffReservations

from test_embedding_async_stream import WindowSource, run, wait_until
from test_embedding_contracts import ContractClient, configured, vector


@dataclass
class WireRecord:
    _text_wire_bytes: int = 8
    _tokens_wire_bytes: int | None = None
    token_count: int = 1


def test_oldest_becoming_ready_reopens_blocked_handoff_without_freeing_normal_bytes(
    monkeypatch,
):
    async def scenario():
        client = ContractClient(1)
        old_entered, release_old = Event(), Event()
        tokenize = client.tokenize

        def prepare(content, **kwargs):
            if content == "older":
                old_entered.set()
                client._wait(
                    release_old,
                    cancel=kwargs.get("cancel"),
                    timeout=kwargs.get("timeout"),
                )
            return tokenize(content, **kwargs)

        monkeypatch.setattr(client, "tokenize", prepare)
        adapter = configured(
            client,
            dimension=2,
            batch_size=1,
            embedding_request_parallelism=1,
            embedding_queue_size=1,
            embedding_max_pending_owners=2,
            token_count_parallelism=2,
            embedding_batch_wait_sec=0,
            timeout=8,
        )
        engine = adapter._engine
        reservations = HandoffReservations(65536, 8192, notify=engine.wake)
        engine._handoff = reservations
        held = []
        while (ticket := reservations.try_acquire([WireRecord()], 2)) is not None:
            held.append(ticket)
        assert held
        normal_used = reservations.snapshot()["normal_used"]
        selected_families = []
        acquire = reservations.try_acquire

        def observed(*args, **kwargs):
            ticket = acquire(*args, **kwargs)
            if ticket is not None:
                selected_families.append(ticket.family)
            return ticket

        monkeypatch.setattr(reservations, "try_acquire", observed)
        stream = adapter.astream_texts(
            WindowSource([["older"], ["younger"]]), max_pending_windows=2
        )
        pending = asyncio.create_task(anext(stream))
        try:
            await wait_until(old_entered.is_set)
            await wait_until(lambda: "embed_0" in engine._handoff_blocked)
            blocked = engine._handoff_blocked["embed_0"]
            assert blocked[-1] is False
            assert reservations.snapshot()["normal_used"] == normal_used
            assert not client.embedding_calls
            # No handoff bytes or oldest-owner identity changes here. Only its
            # newly ready membership can invalidate the previous blocked epoch.
            release_old.set()
            assert await asyncio.wait_for(pending, 3) == [vector("older")]
            assert await asyncio.wait_for(anext(stream), 3) == [vector("younger")]
            assert reservations.snapshot()["normal_used"] == normal_used
            assert selected_families == ["progress", "progress"]
            assert Counter(
                text for _, batch in client.embedding_calls for text in batch
            ) == Counter(["older", "younger"])
            assert [batch for _, batch in client.embedding_calls] == [
                ("older",),
                ("younger",),
            ]
            assert reservations.snapshot()["progress_used"] == 0
        finally:
            release_old.set()
            for ticket in held:
                ticket.release()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            await stream.aclose()
            await adapter.aclose()

    run(scenario)


def test_drained_http_ticket_stays_owned_until_async_result_credit_transfer():
    client = ContractClient(1)
    adapter = configured(client, dimension=2)
    engine = adapter._engine
    call = adapter._stream_call(None, 5)
    state = _State(["input"], "embed", call, Future(), True)
    plan = _Plan(state, "embed_0", 0, 1, 1, deque(), http_active=1)
    record = WireRecord()
    entry = _Entry(plan, 0, record, 64, 0)
    ticket = engine._handoff.try_acquire([record], 2)
    batch = _Batch([entry], (state,), SimpleNamespace(), ticket)
    completed = Future()
    completed.set_result([(0, [1.0, 2.0])])
    state.pending = engine._http_active = 1
    engine._bytes, engine._tokens = 64, 1
    try:
        engine._http_completion(batch, completed)
        assert completed.done() and engine._http_active == 0
        assert not ticket.released
        assert len(engine._http_publications) == 1
        assert engine._handoff.snapshot()["used"] == ticket.amount
        engine._publish_http()
        assert not ticket.released and state.outputs is None
        # Result ownership is reserved before the handoff envelope is released.
        assert engine._reserve_result(state)
        assert state.result_credit > 0
        engine._publish_http()
        assert state.outputs == [[1.0, 2.0]]
        assert ticket.released
        assert not engine._http_publications
        assert engine._handoff.snapshot()["used"] == 0
        assert state.result_credit <= engine._result_reserved
    finally:
        ticket.release()
        adapter.close()


def test_completed_async_result_switches_from_handoff_to_delivery_ownership():
    async def scenario():
        client = ContractClient(1)
        adapter = configured(client, dimension=2, embedding_batch_wait_sec=0)
        stream = adapter.astream_texts(WindowSource([["held"]]), max_pending_windows=1)
        try:
            await stream.__aenter__()
            await wait_until(
                lambda: bool(stream._pending) and stream._pending[0].future.done()
            )
            assert adapter._engine._handoff.snapshot()["used"] == 0
            assert adapter._engine.snapshot()["retained_result_bytes"] > 0
            assert adapter._intake.snapshot()["owners"] == 1
            assert await anext(stream) == [vector("held")]
            await stream.aclose()
            await wait_until(
                lambda: adapter._engine.snapshot()["retained_result_bytes"] == 0
            )
            assert adapter._intake.snapshot()["owners"] == 0
        finally:
            await stream.aclose()
            await adapter.aclose()

    run(scenario)
