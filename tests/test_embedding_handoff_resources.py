"""Handoff ownership is observed; only explicit independent budgets constrain it."""

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


def test_explicit_pool_retains_parallel_owner_capacity_without_a_hidden_partition():
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
        assert pool.try_acquire([Record()], dimension=2) is None
        assert pool.try_acquire([Record()], dimension=2, progress=True) is None
        assert pool.snapshot()["used"] <= pool.limit
        assert pool.progress_limit is None
    finally:
        for ticket in tickets:
            ticket.release()
    view = pool.snapshot()
    assert view["used"] == view["tickets"] == 0
    assert view["normal_peak"] > 0 and view["progress_peak"] == 0


def test_progress_eligible_owner_still_uses_normal_when_it_is_available():
    pool = HandoffReservations(4096, 8192)
    ticket = pool.try_acquire([Record()], dimension=2, progress=True)
    try:
        assert ticket.family == "normal"
        assert pool.snapshot()["progress_used"] == 0
    finally:
        ticket.release()


def test_a_candidate_can_use_the_full_explicit_pool_without_quartering_or_halving():
    pool = HandoffReservations(4096, 8192)
    rows = [Record(_text_wire_bytes=16)] * 50
    with pytest.raises(EmbeddingResourceLimitError) as failure:
        pool.try_acquire(rows, dimension=2)
    assert failure.value.resource == "handoff_bytes"
    assert failure.value.limit == 4096
    assert failure.value.requested > pool.limit
    assert pool.snapshot()["used"] == 0
    admissible = rows[:10]
    assert pool.limit // 4 < pool.estimate(admissible, dimension=2) <= pool.limit
    ticket = pool.try_acquire(admissible, dimension=2)
    try:
        assert ticket.amount == pool.estimate(admissible, dimension=2)
    finally:
        ticket.release()


def test_an_indivisible_unknown_or_known_response_is_explicitly_impossible():
    pool = HandoffReservations(4096, 8192)
    with pytest.raises(EmbeddingResourceLimitError, match="handoff_bytes_limit"):
        pool.try_acquire([Record(_text_wire_bytes=pool.limit + 1)])
    with pytest.raises(EmbeddingResourceLimitError, match="handoff_bytes_limit"):
        pool.try_acquire([Record()], dimension=1024)
    assert pool.snapshot()["used"] == pool.snapshot()["tickets"] == 0


def test_explicit_body_limit_is_independent_of_handoff_size_and_unknown_width():
    pool = HandoffReservations(65536, 2048)
    rows = [Record(_text_wire_bytes=20, _tokens_wire_bytes=80)] * 2
    assert pool.response_limit == 2048
    request = 64 + 2 * 81
    assert pool.estimate(rows) == request
    known = request + 64 + (40 * 8 + 96) * 2
    assert pool.estimate(rows, dimension=8) == known
    ticket = pool.try_acquire(rows)
    try:
        assert ticket.response_limit == pool.response_limit
        assert ticket.amount == request
    finally:
        ticket.release()


def test_ready_entries_and_direct_records_have_the_same_envelope():
    @dataclass
    class Entry:
        record: Record

    pool = HandoffReservations(4096, 8192)
    record = Record()
    assert pool.estimate([record], 2) == pool.estimate([Entry(record)], 2)


def test_explicit_small_limit_is_not_raised_and_response_limit_is_not_divided():
    small = HandoffReservations(128, 8192)
    larger = HandoffReservations(131072, 8192)
    assert small.limit == small.normal_limit == 128
    assert small.progress_limit is None
    assert larger.limit == 131072
    assert small.response_limit == larger.response_limit == 8192
    assert small.snapshot()["used"] == 0


def test_omitted_limits_accept_observed_work_beyond_the_old_64_mib_budget():
    pool = HandoffReservations(None, None)
    rows = [Record()] * 10000
    ticket = pool.try_acquire(rows, dimension=768)
    try:
        assert ticket.amount > 64 * 1024 * 1024
        assert pool.limit is None and ticket.response_limit is None
        assert pool.snapshot()["tickets"] == 1
    finally:
        ticket.release()
    assert pool.snapshot()["used"] == pool.snapshot()["tickets"] == 0


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


def test_cold_numeric_body_uses_only_the_explicit_body_limit_and_releases_ticket():
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
        # The accounting observation is not a claim about exact Python RSS;
        # no 64x decoder multiplier or cold-width body shrink is imposed.
        assert ticket.response_limit == 8192
    finally:
        ticket.release()


def test_explicit_response_cap_rejects_body_before_any_json_decode(monkeypatch):
    import runtime_llama.client as client_module

    pool = HandoffReservations(4096, 128)
    ticket = pool.try_acquire([Record(_text_wire_bytes=8)])
    decoded = []

    def forbidden(*args, **kwargs):
        decoded.append(True)
        raise AssertionError("over-budget response was decoded")

    monkeypatch.setattr(client_module.json_module, "loads", forbidden)
    try:
        assert ticket.response_limit == pool.response_limit == 128
        body = b"[" + b"0.0," * ticket.response_limit + b"0.0]"
        with pytest.raises(LLMBackendError, match="json_response_too_large"):
            raw = _bounded_json_body(iter([body]), "", ticket.response_limit)
            client_module.json_module.loads(raw)
        assert not decoded
    finally:
        ticket.release()
