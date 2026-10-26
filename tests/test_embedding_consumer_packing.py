"""Ordered stream windows share row fairness without delaying their head."""

import asyncio
from collections import Counter
import random
from threading import Event

from runtime_llama.embedding_engine import _Preflight, _ReadyQueue

from test_embedding_async_stream import WindowSource, run, wait_until
from test_embedding_contracts import ContractClient, configured, vector
from test_embedding_incremental_packing import (
    engine_fixture,
    entry,
    owner,
    previous_pack,
)


def window(group, index, size, observation=None):
    state = owner(observation)
    state.call._packing_group = group
    state.call._window_index = index
    state.ready["model"] = _ReadyQueue(
        entry(state, index=position) for position in range(size)
    )
    return state


def test_oldest_ready_windows_fill_one_consumer_turn_across_source_boundaries():
    engine, _ = engine_fixture(cap=5)
    group = object()
    older, newer = window(group, 0, 2), window(group, 1, 4)
    empty = window(group, -1, 0)
    failed = window(group, -2, 2)
    failed.call.failure = RuntimeError("cancelled oldest window")
    selected, bounded = engine._pack(
        "model", older.context.observations["model"], [newer, failed, empty, older]
    )
    assert bounded
    assert [item.plan.state for item in selected] == [older] * 2 + [newer] * 3


def test_distinct_consumers_keep_turns_across_unequal_prefetch_and_rotated_states():
    engine, _ = engine_fixture(cap=1)
    first, second = object(), object()
    a0, b0 = window(first, 0, 20), window(second, 0, 20)
    a1, b1 = window(first, 1, 20), window(second, 1, 20)
    a2 = window(first, 2, 20)
    sync = owner()
    sync.ready["model"] = _ReadyQueue(entry(sync, index=i) for i in range(20))
    states = [a0, b0, a1, sync, b1, a2]
    engine._packing_turns.prune(states, ("model",))
    observed = []
    for index in range(18):
        offset = index % len(states)
        rotated = states[offset:] + states[:offset]
        selected, _ = engine._pack("model", a0.context.observations["model"], rotated)
        state = selected[0].plan.state
        observed.append(getattr(state.call, "_packing_group", state))
        # Repeated speculative selection never consumes another turn.
        assert (
            engine._pack("model", a0.context.observations["model"], rotated)[0]
            == selected
        )
        engine._packing_turns.committed("model", selected)
        assert state.ready["model"].popleft() is selected[0]
    assert observed == [first, second, sync] * 6


def test_turns_advance_after_last_row_and_only_after_successful_dispatch():
    engine, _ = engine_fixture(cap=4)
    states = [window(object(), 0, 8) for _ in range(3)]
    observation = states[0].context.observations["model"]
    selected, _ = engine._pack("model", observation, states)
    assert [item.plan.state for item in selected] == [*states, states[0]]
    scheduler = engine.adapter._scheduler
    real_dispatch = scheduler.try_dispatch
    scheduler.try_dispatch = lambda *_args, **_kwargs: None
    engine._shared_task = lambda *_args: None
    engine._preflights["model"] = _Preflight(observation, tuple(states))
    engine._dispatch(states)
    assert engine._pack("model", observation, states)[0] == selected
    scheduler.try_dispatch = real_dispatch
    engine._preflights["model"] = _Preflight(observation, tuple(states))
    engine._dispatch(states)
    assert engine._pack("model", observation, states)[0][0].plan.state is states[1]


def test_alias_turns_and_retired_owner_bookkeeping_are_independent():
    engine, _ = engine_fixture(cap=1)
    states = [window(object(), 0, 2) for _ in range(3)]
    for state in states:
        state.ready["other"] = state.ready["model"]
    groups = engine._packing_turns.groups("model", states)
    engine._packing_turns.groups("other", states)
    engine._packing_turns.committed("model", [next(iter(groups[0]))])
    assert engine._packing_turns.groups("model", states)[0].states == (states[1],)
    assert engine._packing_turns.groups("other", states)[0].states == (states[0],)
    engine._packing_turns.prune(states[1:], ("model",))
    assert tuple(engine._packing_turns._aliases) == ("model",)
    assert set(engine._packing_turns._aliases["model"]) == {
        state.call._packing_group for state in states[1:]
    }
    engine._packing_turns.prune([], ("model",))
    assert not engine._packing_turns._aliases


def test_new_consumer_joins_after_waiting_consumers():
    engine, _ = engine_fixture(cap=1)
    first, waiting, newcomer = [window(object(), 0, 2) for _ in range(3)]
    observation = first.context.observations["model"]
    selected, _ = engine._pack("model", observation, [first, waiting])
    engine._packing_turns.committed("model", selected)
    engine._packing_turns.prune([first, waiting, newcomer], ("model",))
    assert (
        engine._pack("model", observation, [newcomer, first, waiting])[0][0].plan.state
        is waiting
    )


def test_filtered_preflight_scope_keeps_other_consumers_out_and_retains_their_turns():
    engine, _ = engine_fixture(cap=1)
    states = [window(object(), 0, 2) for _ in range(3)]
    engine._packing_turns.prune(states, ("model",))
    observation = states[0].context.observations["model"]
    selected, _ = engine._pack("model", observation, [states[1]])
    assert selected[0].plan.state is states[1]
    assert engine._pack("model", observation, states)[0][0].plan.state is states[0]
    engine._packing_turns.committed("model", selected)
    assert engine._pack("model", observation, states)[0][0].plan.state is states[2]


def test_group_summaries_match_flattened_reference_with_mixed_ids_and_wire_limits():
    rng = random.Random(451_028)
    for case in range(500):
        engine, failures = engine_fixture(
            cap=rng.randint(1, 30),
            request_bytes=rng.randint(35, 900),
            native=bool(case % 2),
        )
        states, groups = [], {}
        for _ in range(rng.randint(1, 4)):
            group = object()
            members = []
            for index in range(rng.randint(1, 4)):
                state = window(group, index, 0)
                for position in range(rng.randint(0, 8)):
                    item = entry(
                        state,
                        index=position,
                        tokens=rng.randint(1, 30),
                        limit=rng.choice([None, 30, 90]),
                        text=rng.randint(2, 100),
                        ids=rng.choice([None, 2, 40, 150]),
                    )
                    if rng.randrange(8) == 0:
                        item.record._token_ids = ()
                    state.ready["model"].append(item)
                members.append(state)
            groups[group] = members
            states.extend(members)
        rng.shuffle(states)
        engine._packing_turns.prune(states, ("model",))
        observation = states[0].context.observations["model"]
        observation.batch_token_limit = rng.choice([None, 40, 70, 200])
        reference = []
        for group in dict.fromkeys(state.call._packing_group for state in states):
            shadow = owner(observation)
            shadow.ready["model"] = _ReadyQueue(
                item for state in groups[group] for item in state.ready["model"]
            )
            reference.append(shadow)
        expected, bounded, rejected = previous_pack(engine, observation, reference)
        assert engine._ready("model", observation, states) == bool(
            bounded or rejected
        ), case
        actual, actual_bounded = engine._pack("model", observation, states)
        assert actual == expected and actual_bounded == bounded, case
        assert len(failures) == len(rejected), case


class UnequalReplicaClient(ContractClient):
    def __init__(self):
        super().__init__(2, context=50000, native_batch=50000)
        self.slow_gates = []
        self.fast_finished = 0
        self.release_all = False

    def embeddings(self, input, *, model, cancel=None, timeout=None, **kwargs):
        if model == "embed_1":
            gate = Event()
            with self.lock:
                self.slow_gates.append(gate)
                if self.release_all:
                    gate.set()
            self._wait(gate, cancel=cancel, timeout=timeout)
        result = super().embeddings(
            input, model=model, cancel=cancel, timeout=timeout, **kwargs
        )
        if model == "embed_0":
            with self.lock:
                self.fast_finished += 1
        return result

    def unblock(self):
        with self.lock:
            self.release_all = True
            for gate in self.slow_gates:
                gate.set()


def test_streams_with_different_prefetch_and_a_sync_call_share_each_model_batch():
    async def scenario():
        client = ContractClient(1, context=50000, native_batch=50000)
        client.embed_gate.clear()
        adapter = configured(
            client,
            batch_size=3,
            dimension=2,
            embedding_request_parallelism=1,
            embedding_queue_size=8,
            token_count_parallelism=4,
            embedding_batch_wait_sec=0,
            timeout=10,
        )
        engine = adapter._engine
        dispatch = engine._dispatch
        ready = False

        def prepare_all_consumers(states):
            nonlocal ready
            if not ready:
                count = sum(
                    len(queue) for state in states for queue in state.ready.values()
                )
                if count < 36:
                    return
                ready = True
            dispatch(states)

        engine._dispatch = prepare_all_consumers
        windows = {
            "a": [[f"a{w}:{i}" for i in range(6)] for w in range(3)],
            "b": [[f"b{w}:{i}" for i in range(6)] for w in range(2)],
        }
        streams = {
            key: adapter.astream_texts(
                WindowSource(value), max_pending_windows=len(value)
            )
            for key, value in windows.items()
        }

        async def consume(stream):
            async with stream:
                return [result async for result in stream]

        tasks = {
            key: asyncio.create_task(consume(stream)) for key, stream in streams.items()
        }
        sync_inputs = [f"s:{i}" for i in range(6)]
        sync = asyncio.create_task(asyncio.to_thread(adapter.embed_texts, sync_inputs))
        try:
            await wait_until(lambda: bool(client.embedding_calls))
            assert Counter(text[0] for text in client.embedding_calls[0][1]) == {
                "a": 1,
                "b": 1,
                "s": 1,
            }
            client.embed_gate.set()
            assert await sync == [vector(text) for text in sync_inputs]
            for key, task in tasks.items():
                assert await task == [
                    [vector(text) for text in window] for window in windows[key]
                ]
            assert len(client.embedding_calls) == 12
            assert all(
                Counter(text[0] for text in batch) == {"a": 1, "b": 1, "s": 1}
                for _, batch in client.embedding_calls[:6]
            )
            assert all(len(batch) == 3 for _, batch in client.embedding_calls)
            assert not engine._packing_turns._aliases
        finally:
            client.embed_gate.set()
            await asyncio.gather(sync, *tasks.values(), return_exceptions=True)
            for stream in streams.values():
                await stream.aclose()
            await adapter.aclose()

    run(scenario)


def test_oldest_window_completion_refills_fast_replica_before_next_slow_result():
    async def scenario():
        client = UnequalReplicaClient()
        adapter = configured(
            client,
            batch_size=256,
            dimension=2,
            embedding_request_parallelism=4,
            embedding_queue_size=0,
            token_count_parallelism=4,
            embedding_batch_wait_sec=0,
            timeout=10,
        )
        engine = adapter._engine
        dispatch = engine._dispatch
        initial_ready = False

        def admit_both_initial_windows(states):
            nonlocal initial_ready
            if not initial_ready:
                count = sum(
                    len(queue) for state in states for queue in state.ready.values()
                )
                if count < 512:
                    return
                initial_ready = True
            dispatch(states)

        engine._dispatch = admit_both_initial_windows
        windows = [[f"w{w}:{i:03d}" for i in range(256)] for w in range(8)]
        source = WindowSource(windows)
        stream = adapter.astream_texts(source, max_pending_windows=2)
        delivered = []

        async def consume():
            async with stream:
                async for result in stream:
                    delivered.append(result)

        consumer = asyncio.create_task(consume())
        try:
            await wait_until(
                lambda: len(client.slow_gates) == 1 and client.fast_finished == 2
            )
            assert source.pulls == 2
            assert not delivered
            client.slow_gates[0].set()
            await wait_until(
                lambda: len(client.slow_gates) == 2 and client.fast_finished == 3
            )
            assert source.pulls == 3
            assert len(delivered) == 1
            assert len(stream._pending) == 2
            assert not client.slow_gates[1].is_set()
            snapshot = engine.snapshot()
            assert snapshot["embedding_in_flight"] == 1
            assert snapshot["ready_by_replica"]["embed_0"] == 0
            assert snapshot["peak_active_plans"] <= 4
            assert adapter.embedding_max_pending_bytes is None
            assert snapshot["peak_pending_bytes"] > 0
            assert snapshot["peak_result_reserved_bytes"] > 0
            client.unblock()
            await asyncio.wait_for(consumer, 5)
            assert delivered == [
                [vector(text) for text in window] for window in windows
            ]
            assert len(client.embedding_calls) == 16
            assert Counter(len(batch) for _, batch in client.embedding_calls) == {
                128: 16
            }
            assert not engine._packing_turns._aliases
            assert engine.snapshot()["result_reserved_bytes"] == 0
        finally:
            client.unblock()
            await asyncio.gather(consumer, return_exceptions=True)
            await stream.aclose()
            await adapter.aclose()

    run(scenario)
