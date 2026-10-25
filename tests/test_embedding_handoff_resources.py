"""Response handoff reserves progress independently of ready-record storage."""

from dataclasses import dataclass
import json
import sys
from threading import Thread

import pytest

from runtime_llama.client import _bounded_json_body
from runtime_llama.embedding_handoff import HandoffReservations
from runtime_llama.embedding_resources import EmbeddingResourceLimitError
from runtime_llama.errors import LLMBackendError


@dataclass
class Record:
    _text_wire_bytes: int = 64
    _tokens_wire_bytes: int | None = None


def test_normal_admission_precedes_progress_and_retains_parallel_owner_capacity():
    pool = HandoffReservations(4096, 8192)
    tickets = []
    try:
        # No owner semaphore is introduced: several POSTs of one owner may use
        # distinct replica leases as long as their resource envelopes fit.
        while (ticket := pool.try_acquire([Record()], dimension=2)) is not None:
            tickets.append(ticket)
        assert len(tickets) > 1
        assert all(ticket.family == "normal" for ticket in tickets)
        assert pool.snapshot()["normal_used"] <= pool.normal_limit
        assert pool.snapshot()["progress_used"] == 0
        progress_tickets = []
        while (
            progress := pool.try_acquire([Record()], dimension=2, progress=True)
        ) is not None:
            assert progress.family == "progress"
            progress_tickets.append(progress)
        assert progress_tickets
        tickets.extend(progress_tickets)
        assert pool.try_acquire([Record()], dimension=2) is None
        assert pool.try_acquire([Record()], dimension=2, progress=True) is None
        assert pool.snapshot()["used"] <= pool.limit
        assert pool.snapshot()["progress_used"] <= pool.progress_limit
    finally:
        for ticket in tickets:
            ticket.release()
    view = pool.snapshot()
    assert view["used"] == view["tickets"] == 0
    assert view["normal_peak"] > 0 and view["progress_peak"] > 0


def test_progress_eligible_owner_still_uses_normal_when_it_is_available():
    pool = HandoffReservations(4096, 8192)
    ticket = pool.try_acquire([Record()], dimension=2, progress=True)
    try:
        assert ticket.family == "normal"
        assert pool.snapshot()["progress_used"] == 0
    finally:
        ticket.release()


def test_every_candidate_fits_reserved_progress_before_it_can_enter_normal():
    pool = HandoffReservations(4096, 8192)
    rows = [Record(_text_wire_bytes=16)] * 50
    assert pool.estimate(rows, dimension=2) < pool.normal_limit
    with pytest.raises(EmbeddingResourceLimitError) as failure:
        pool.try_acquire(rows, dimension=2)
    assert failure.value.resource == "handoff_bytes"
    assert failure.value.limit == pool.progress_limit
    assert failure.value.requested > pool.progress_limit
    assert pool.snapshot()["used"] == 0
    # The same coordinator can trim its candidate without duplicating records.
    ticket = pool.try_acquire(rows[:25], dimension=2)
    try:
        assert ticket.amount <= pool.max_batch_bytes
    finally:
        ticket.release()


def test_an_indivisible_unknown_or_known_response_is_explicitly_impossible():
    pool = HandoffReservations(4096, 8192)
    with pytest.raises(EmbeddingResourceLimitError, match="handoff_bytes_limit"):
        pool.try_acquire([Record(_text_wire_bytes=pool.progress_limit + 1)])
    with pytest.raises(EmbeddingResourceLimitError, match="handoff_bytes_limit"):
        pool.try_acquire([Record()], dimension=1024)
    assert pool.snapshot()["used"] == pool.snapshot()["tickets"] == 0


def test_response_limit_and_unknown_dimension_envelope_are_finite():
    pool = HandoffReservations(65536, 2048)
    rows = [Record(_text_wire_bytes=20, _tokens_wire_bytes=80)] * 2
    assert pool.response_limit == min(2048, 65536 // 16)
    request = 64 + 2 * 81
    cold_cap = min(pool.response_limit, (pool.progress_limit - request) // 65)
    unknown = request + 65 * cold_cap
    assert pool.estimate(rows) == unknown
    known = request + pool.response_limit + 64 + (96 * 8 + 64) * 2
    assert pool.estimate(rows, dimension=8) == known
    ticket = pool.try_acquire(rows)
    try:
        assert ticket.response_limit == cold_cap
        assert ticket.response_limit < pool.response_limit
        assert ticket.amount <= pool.progress_limit
        assert ticket.amount == unknown
    finally:
        ticket.release()


def test_ready_entries_and_direct_records_have_the_same_envelope():
    @dataclass
    class Entry:
        record: Record

    pool = HandoffReservations(4096, 8192)
    record = Record()
    assert pool.estimate([record], 2) == pool.estimate([Entry(record)], 2)


def test_floor_is_fixed_and_does_not_expand_input_or_working_budget():
    small = HandoffReservations(128, 8192)
    larger = HandoffReservations(131072, 8192)
    assert small.limit == 65536
    assert small.normal_limit == 49152 and small.progress_limit == 16384
    assert larger.limit == 131072
    assert small.response_limit == 4096
    assert small.snapshot()["used"] == 0


def test_ticket_release_is_idempotent_across_threads_and_close_keeps_ownership():
    changed = []
    pool = HandoffReservations(4096, 8192, notify=lambda: changed.append(True))
    ticket = pool.try_acquire([Record()], dimension=2)
    used = pool.snapshot()["used"]
    pool.close()
    assert pool.snapshot()["used"] == used
    with pytest.raises(LLMBackendError, match="closed"):
        pool.try_acquire([Record()], dimension=2)
    threads = [Thread(target=ticket.release) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()
    assert len(changed) == 2  # one close and one actual release
    assert pool.snapshot()["used"] == pool.snapshot()["tickets"] == 0


@pytest.mark.parametrize("dimension", [0, -1, True, 1.5])
def test_invalid_dimension_does_not_reserve_handoff(dimension):
    pool = HandoffReservations(4096, 8192)
    with pytest.raises(ValueError):
        pool.try_acquire([Record()], dimension=dimension)
    assert pool.snapshot()["used"] == 0


def test_empty_candidates_and_invalid_wire_costs_are_not_admitted():
    pool = HandoffReservations(4096, 8192)
    with pytest.raises(ValueError):
        pool.try_acquire([])
    with pytest.raises(ValueError):
        pool.try_acquire([Record(_text_wire_bytes=-1)])
    assert pool.snapshot()["used"] == 0


def test_cold_short_numeric_json_materialization_fits_its_reserved_envelope():
    pool = HandoffReservations(65536, 8192)
    ticket = pool.try_acquire([Record(_text_wire_bytes=8)])
    try:
        encoded = json.dumps(
            {"data": [{"index": 0, "embedding": [0.0] * 40}]},
            separators=(",", ":"),
        ).encode()
        assert len(encoded) <= ticket.response_limit
        decoded = json.loads(encoded)

        def owned_bytes(value):
            if isinstance(value, dict):
                return sys.getsizeof(value) + sum(
                    owned_bytes(key) + owned_bytes(item) for key, item in value.items()
                )
            if isinstance(value, list):
                return sys.getsizeof(value) + sum(owned_bytes(item) for item in value)
            return sys.getsizeof(value)

        materialized = owned_bytes(decoded)
        assert materialized > len(encoded)
        assert materialized <= 64 * ticket.response_limit
        assert ticket.amount <= pool.progress_limit
    finally:
        ticket.release()


def test_cold_response_cap_rejects_body_before_any_json_decode(monkeypatch):
    import runtime_llama.client as client_module

    pool = HandoffReservations(4096, 8192)
    ticket = pool.try_acquire([Record(_text_wire_bytes=8)])
    decoded = []

    def forbidden(*args, **kwargs):
        decoded.append(True)
        raise AssertionError("over-budget response was decoded")

    monkeypatch.setattr(client_module.json_module, "loads", forbidden)
    try:
        assert ticket.response_limit < pool.response_limit
        body = b"[" + b"0.0," * ticket.response_limit + b"0.0]"
        with pytest.raises(LLMBackendError, match="json_response_too_large"):
            raw = _bounded_json_body(iter([body]), "", ticket.response_limit)
            client_module.json_module.loads(raw)
        assert not decoded
    finally:
        ticket.release()
