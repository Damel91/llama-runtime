"""Persistent bounded preparation and record queues over one HTTP scheduler.

The coordinator owns state, never HTTP. Workers publish completion events;
inference batches are selected only when the shared scheduler can start them.
"""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Condition, Thread, current_thread
import sys
import time
from typing import Any

from runtime_llama.errors import LLMBackendError


@dataclass(frozen=True)
class EngineContext:
    joined: bool
    observations: dict[str, Any]
    conservative: tuple[int | None, int | None]


@dataclass(eq=False)
class _Plan:
    state: Any
    model: str
    offset: int
    size: int
    cap: int
    groups: deque
    finished: int = 0
    preparing: int = 0
    http_active: int = 0
    admitted: bool = True
    preparation_lease: bool = True
    seeded: bool = False
    seed_published: bool = False


@dataclass(eq=False)
class _Entry:
    plan: _Plan
    index: int
    record: Any
    byte_cost: int
    ready_at: float


class _ReadyQueue(deque):
    """Ready rows plus bounded summaries, never a frozen packing candidate.

    The wire-ID prefix is an integer ledger only. It lets readiness account
    for the format switch at the first row without reusable IDs without
    walking all previously prepared rows after every tokenizer completion.
    Its consumed prefix is compacted geometrically, so storage follows the
    live queue and append/pop work is amortized constant time.
    """

    __slots__ = (
        "tokens",
        "text_bytes",
        "limits",
        "minimum_limit",
        "_id_prefix",
        "_head",
        "_missing_ids",
    )

    def __init__(self, entries=()):
        super().__init__()
        self.tokens = self.text_bytes = self._head = 0
        self.limits = Counter()
        self.minimum_limit = None
        self._id_prefix = [0]
        self._missing_ids = deque()
        for entry in entries:
            self.append(entry)

    def append(self, entry):
        record = entry.record
        self.tokens += record.token_count
        self.text_bytes += record._text_wire_bytes
        limit = record.batch_token_limit
        if limit is not None:
            self.limits[limit] += 1
            self.minimum_limit = (
                limit if self.minimum_limit is None else min(self.minimum_limit, limit)
            )
        wire = record._tokens_wire_bytes
        if wire is None or not record._token_ids:
            self._missing_ids.append(len(self._id_prefix) - 1)
        self._id_prefix.append(self._id_prefix[-1] + (wire or 0))
        super().append(entry)

    def popleft(self):
        entry = super().popleft()
        record = entry.record
        self.tokens -= record.token_count
        self.text_bytes -= record._text_wire_bytes
        limit = record.batch_token_limit
        if limit is not None:
            self.limits[limit] -= 1
            if not self.limits[limit]:
                del self.limits[limit]
                if limit == self.minimum_limit:
                    self.minimum_limit = min(self.limits, default=None)
        if self._missing_ids and self._missing_ids[0] == self._head:
            self._missing_ids.popleft()
        self._head += 1
        if self._head >= len(self):
            consumed = self._head
            base = self._id_prefix[consumed]
            self._id_prefix = [value - base for value in self._id_prefix[consumed:]]
            self._missing_ids = deque(index - consumed for index in self._missing_ids)
            self._head = 0
        return entry

    def clear(self):
        super().clear()
        self.tokens = self.text_bytes = self._head = 0
        self.limits.clear()
        self.minimum_limit = None
        self._id_prefix = [0]
        self._missing_ids.clear()

    @property
    def first_missing_ids(self):
        return self._missing_ids[0] - self._head if self._missing_ids else None

    def id_prefix_bytes(self, count):
        return self._id_prefix[self._head + count] - self._id_prefix[self._head]


@dataclass(eq=False)
class _Publication:
    plan: _Plan
    positions: deque[int]
    record: Any
    byte_cost: int


@dataclass(eq=False)
class _Batch:
    entries: list[_Entry]
    owners: tuple[Any, ...]
    call: Any


@dataclass(eq=False)
class _MetadataRound:
    members: dict[str, tuple[Any, ...]]
    catalog: Any = None
    waiting: deque = field(default_factory=deque)


@dataclass(frozen=True)
class _Preflight:
    observation: Any
    members: tuple[Any, ...]


@dataclass(eq=False)
class _State:
    inputs: Any
    operation: str
    call: Any
    future: Future
    retain_result: bool
    context: EngineContext | None = None
    outputs: list | None = None
    cursor: int = 0
    descriptors: deque = field(default_factory=deque)
    plans: list = field(default_factory=list)
    ready: dict = field(default_factory=dict)
    pending: int = 0
    starting: bool = False
    finishing: bool = False
    blocked_at: tuple[int, int] | None = None
    checking: bool = False
    checked_at: float = 0.0
    failure_handled: bool = False
    result_credit: int = 0
    plan_rotation: int = 0


class PersistentEmbeddingEngine:
    """Adapter-scoped record ownership; EmbeddingScheduler owns HTTP leases.

    Raw and ready queue credits are bounded separately from transient worker
    responses. At most T tokenizer results can await publication; no further
    tokenization starts until that bounded handoff makes progress. Native
    response limits bound those transient results before JSON materialization.
    """

    def __init__(self, adapter):
        self.adapter = adapter
        self.parallelism = adapter.embedding_request_parallelism
        self.tokenizers = adapter.token_count_parallelism
        self.control_workers = max(1, self.parallelism)
        self.capacity = max(1, self.parallelism + adapter.embedding_queue_size)
        self.max_bytes = getattr(
            adapter, "embedding_max_pending_bytes", 64 * 1024 * 1024
        )
        self.max_tokens = getattr(adapter, "embedding_max_pending_tokens", 1024 * 1024)
        self.max_request_bytes = getattr(
            adapter, "embedding_max_request_bytes", 8 * 1024 * 1024
        )
        self._condition = Condition()
        self._incoming = deque()
        self._events = deque()
        self._states: deque[_State] = deque()
        # Persistent physical lanes own the queues. State.ready is only an
        # owner index into these lanes, not a second per-window scheduler.
        self._lanes: dict[str, dict[_State, _ReadyQueue]] = {}
        self._publications: deque[_Publication] = deque()
        self._http_publications: deque = deque()
        self._observing: set[str] = set()
        self._preflights: dict[str, _Preflight] = {}
        self._metadata_rounds: deque[_MetadataRound] = deque()
        self._retained: dict[Any, int] = {}
        self._released_calls: set[Any] = set()
        self._release_results = deque()
        self._closed = False
        self._thread = None
        self._control = self._tokenizer = None
        self._control_active = self._token_active = self._http_active = 0
        self._plans = self._bytes = self._tokens = self._raw_bytes = 0
        self._descriptor_count = 0
        self._unknown_seeds = 0
        self._preparation_blocked = False
        self._peak_bytes = self._peak_tokens = self._peak_plans = 0
        self._peak_token_active = self._peak_http_active = 0
        self._result_bytes = self._peak_result_bytes = 0
        self._result_reserved = self._peak_result_reserved = 0
        self._vector_dim = adapter.embedding_dim or None
        self._progress = self._rotation = 0
        self._metadata_availability = 0
        self._notification = 0
        self._plan_counts = Counter()
        self._preparing_by_model = Counter()
        self._retained_plan_count = 0
        self._view_signature = None
        self._packing_candidate_visits = 0
        self._view = {}
        self._update_view()

    def submit(self, inputs, *, operation, call, retain_result=False) -> Future:
        """Register borrowed input storage; a Future completes after real drain."""
        future = Future()
        # Prevent cancellation of a wrapper from falsely completing owned work.
        future.set_running_or_notify_cancel()
        state = _State(inputs, operation, call, future, retain_result)
        with self._condition:
            if self._closed:
                raise LLMBackendError("llamacpp_embedding_adapter_closed")
            self._incoming.append(state)
            if self._thread is None:
                self._control = ThreadPoolExecutor(
                    max_workers=self.control_workers,
                    thread_name_prefix="runtime_llama_control",
                )
                self._tokenizer = ThreadPoolExecutor(
                    max_workers=self.tokenizers,
                    thread_name_prefix="runtime_llama_tokenizer",
                )
                self._thread = Thread(
                    target=self._run,
                    name="runtime_llama_embedding_coordinator",
                    daemon=True,
                )
                self._thread.start()
            self._notify_locked()
        return future

    def _notify_locked(self):
        # Remember notifications delivered while the coordinator is working,
        # not just those delivered after it has entered Condition.wait().
        self._notification += 1
        self._condition.notify_all()

    def wake(self):
        """Notify cancellation without introducing a polling coordinator per call."""
        with self._condition:
            self._notify_locked()

    def metadata_available(self):
        """Publish external pool/gate availability without metadata polling."""
        with self._condition:
            if self._closed:
                return
            self._metadata_availability += 1
            self._notify_locked()

    def release_result(self, call):
        with self._condition:
            if self._closed and (self._thread is None or not self._thread.is_alive()):
                return
            self._release_results.append(call)
            self._notify_locked()

    @staticmethod
    def _resource(name):
        return LLMBackendError(f"llamacpp_embedding_{name}_limit")

    def _event(self, kind, state, payload, future):
        with self._condition:
            self._events.append((kind, state, payload, future))
            self._notify_locked()

    def _task(self, executor, kind, state, payload, run):
        state.pending += 1
        if kind == "prepare":
            self._token_active += 1
            self._peak_token_active = max(self._peak_token_active, self._token_active)
        else:
            self._control_active += 1
        try:
            future = executor.submit(run)
        except BaseException as exc:
            future = Future()
            future.set_exception(exc)
        future.add_done_callback(lambda done: self._event(kind, state, payload, done))

    def _shared_task(self, kind, owners, payload, run):
        owners = tuple(dict.fromkeys(owners))
        for owner in owners:
            owner.pending += 1
        self._control_active += 1
        try:
            future = self._control.submit(run)
        except BaseException as exc:
            future = Future()
            future.set_exception(exc)
        future.add_done_callback(lambda done: self._event(kind, owners, payload, done))

    def _fail(self, state, failure):
        state.call.fail(failure)
        if state.failure_handled:
            return
        state.failure_handled = True
        state.descriptors.clear()
        state.cursor = len(state.inputs)
        for entries in state.ready.values():
            while entries:
                self._release_entry(entries.popleft())
        for plan in state.plans:
            if plan.admitted:
                self._descriptor_count -= sum(
                    len(positions) for _, positions in plan.groups
                )
            plan.groups.clear()
        self._publications = deque(
            item for item in self._publications if item.plan.state is not state
        )
        self._http_publications = deque(
            item for item in self._http_publications if item[0] is not state
        )

    def _release_entry(self, entry):
        self._bytes -= entry.byte_cost
        self._tokens -= entry.record.token_count

    def _completions(self, events):
        for kind, state, payload, future in events:
            if kind == "http":
                self._http_completion(payload, future)
                continue
            if kind in ("start_shared", "catalog_shared", "observe_shared"):
                self._shared_completion(kind, state, payload, future)
                continue
            state.pending -= 1
            if kind == "prepare":
                self._token_active -= 1
                plan, positions, raw_cost = payload
                plan.preparing -= 1
                self._preparing_by_model[plan.model] -= 1
                if not self._preparing_by_model[plan.model]:
                    del self._preparing_by_model[plan.model]
                self._raw_bytes -= raw_cost
                self._bytes -= raw_cost
            else:
                self._control_active -= 1
                if kind == "check":
                    state.checking = False
                    state.checked_at = time.monotonic()
            try:
                result = future.result()
            except BaseException as exc:
                self._progress += 1
                self._fail(state, exc)
                if kind == "finish":
                    self._finish_future(state, None)
                continue
            if kind != "check":
                self._progress += 1
            if kind == "finish":
                self._finish_future(state, result)
                continue
            if state.call.failure is not None:
                continue
            if kind == "prepare":
                if state.operation == "prepare":
                    for position in positions:
                        state.outputs[position] = result
                        plan.finished += 1
                else:
                    try:
                        self.adapter._engine_validate(
                            result, plan.model, state.context.observations[plan.model]
                        )
                        count = result.token_count
                        cost = self._record_bytes(result)
                        if count > self.max_tokens:
                            raise self._resource("pending_tokens")
                        if cost > self.max_bytes:
                            raise self._resource("pending_bytes")
                        self._publications.append(
                            _Publication(plan, deque(positions), result, cost)
                        )
                    except BaseException as exc:
                        self._fail(state, exc)

    def _shared_completion(self, kind, owners, payload, future):
        self._control_active -= 1
        for owner in owners:
            owner.pending -= 1
            if kind == "start_shared":
                owner.starting = False
        if kind == "observe_shared":
            self._observing.discard(payload)
        try:
            result = future.result()
        except BaseException as exc:
            if kind == "catalog_shared":
                self._observing.difference_update(payload.members)
            for owner in owners:
                self._fail(owner, exc)
            self._progress += 1
            return
        if kind == "start_shared":
            if result is None:
                for owner in owners:
                    # Availability can change while the worker returns None.
                    # Preserve the marker from submission, never completion.
                    owner.blocked_at = payload
                return
            if len(result) != len(owners):
                for owner in owners:
                    self._fail(
                        owner, LLMBackendError("llamacpp_embedding_admission_mismatch")
                    )
                return
            for owner, context in zip(owners, result):
                if context is None:
                    self._fail(
                        owner,
                        owner.call.failure
                        or LLMBackendError("llamacpp_embedding_admission_mismatch"),
                    )
                    continue
                owner.context = context
                aliases = set(context.observations)
                for alias in tuple(self._lanes):
                    if alias not in aliases and not self._lanes[alias]:
                        del self._lanes[alias]
                for alias in aliases:
                    self._lanes.setdefault(alias, {})
                if owner.retain_result:
                    if 64 + 96 * len(owner.inputs) > self.max_bytes:
                        self._fail(owner, self._resource("result_bytes"))
                else:
                    owner.outputs = [None] * len(owner.inputs)
        elif kind == "catalog_shared":
            payload.catalog = result
            payload.waiting = deque(payload.members)
            self._metadata_rounds.append(payload)
        else:
            self._preflights[payload] = _Preflight(result, owners)
        self._progress += 1

    def _advance_metadata(self):
        while self._metadata_rounds and self._control_active < self.control_workers:
            metadata = self._metadata_rounds[0]
            model = metadata.waiting.popleft()
            if not metadata.waiting:
                self._metadata_rounds.popleft()
            owners = tuple(
                owner
                for owner in metadata.members[model]
                if self._metadata_member(owner, model)
            )
            if not owners:
                self._observing.discard(model)
                continue
            group = self.adapter._engine_batch_call(
                tuple(owner.call for owner in owners)
            )
            self._shared_task(
                "observe_shared",
                owners,
                model,
                lambda model=model, group=group, catalog=metadata.catalog: (
                    self.adapter._engine_observe(model, group, catalog=catalog)
                ),
            )

    def _http_completion(self, batch, future):
        self._http_active -= 1
        self._progress += 1
        for owner in batch.owners:
            owner.pending -= 1
        for entry in batch.entries:
            entry.plan.http_active -= 1
            self._release_entry(entry)
        try:
            rows = future.result()
            indexed = {}
            for index, vector in rows:
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < len(batch.entries)
                    or index in indexed
                ):
                    raise LLMBackendError(
                        "llamacpp_embedding_count_or_response_mismatch"
                    )
                indexed[index] = vector
        except BaseException as exc:
            for owner in batch.owners:
                self._fail(owner, exc)
            return
        destinations = {owner: [] for owner in batch.owners}
        vectors = {owner: [] for owner in batch.owners}
        for index, entry in enumerate(batch.entries):
            owner = entry.plan.state
            if owner.call.failure is not None:
                continue
            if index not in indexed:
                self._fail(
                    owner,
                    LLMBackendError("llamacpp_embedding_count_or_response_mismatch"),
                )
                continue
            destinations[owner].append((entry.plan, entry.index))
            vectors[owner].append(indexed[index])
        for owner in batch.owners:
            if owner.call.failure is not None:
                continue
            result = vectors[owner]
            if result and self._vector_dim is None:
                self._vector_dim = len(result[0])
            if owner.retain_result and not owner.result_credit:
                self._http_publications.append((owner, destinations[owner], result))
            else:
                self._publish_http_result(owner, destinations[owner], result)

    @staticmethod
    def _record_bytes(record):
        ids = getattr(record, "_token_ids", None)
        return (
            640 + sys.getsizeof(record.text) + (40 * len(ids) if ids is not None else 0)
        )

    @staticmethod
    def _item_key(item):
        return (item.text, item.input_type) if hasattr(item, "token_count") else item

    def _borrowed_key(self, inputs, index):
        if hasattr(inputs, "prefix") and hasattr(inputs, "indices"):
            value = inputs.texts[inputs.indices[index]]
            return (value if isinstance(value, str) else str(value), inputs.input_type)
        return self._item_key(inputs[index])

    def _plan(self, state):
        if not state.descriptors:
            stop = min(len(state.inputs), state.cursor + self.adapter.batch_size)
            offset = state.cursor
            for model, size in self.adapter._scheduler.plan_window(stop - offset):
                state.descriptors.append((model, offset, size, size))
                offset += size
            state.cursor = stop
        descriptors = list(state.descriptors)
        selected = min(
            range(len(descriptors)),
            key=lambda index: self._plan_counts[descriptors[index][0]],
        )
        model, offset, remaining, cap = descriptors.pop(selected)
        state.descriptors = deque(descriptors)
        # Bound descriptor storage even when a consumer configures huge windows.
        # This is a bound on small index descriptors, not an estimate of text
        # sizes. Prefix copies are admitted by their actual UTF-8 bytes below.
        size = min(remaining, max(1, self.max_bytes // 1024 - self._descriptor_count))
        if size < remaining:
            state.descriptors.appendleft((model, offset + size, remaining - size, cap))
        observed = state.context.observations[model]
        deduplicate = (
            observed.identity_key is not None and observed.tokenizer_build is not None
        )
        if deduplicate:
            groups, keys = [], {}
            for index in range(offset, offset + size):
                key = self._borrowed_key(state.inputs, index)
                if key in keys:
                    groups[keys[key]][1].append(index)
                else:
                    keys[key] = len(groups)
                    groups.append((index, [index]))
        else:
            groups = [(index, [index]) for index in range(offset, offset + size)]
        plan = _Plan(state, model, offset, size, cap, deque(groups))
        state.plans.append(plan)
        # Descriptor balancing counts retained plans, including suspended ones,
        # exactly as the former scan over owner.plans did. Preparation leases
        # are an independent resource and do not alter this ownership ledger.
        self._plan_counts[model] += 1
        self._retained_plan_count += 1
        self._plans += 1
        self._descriptor_count += size
        self._peak_plans = max(self._peak_plans, self._plans)

    def _publish(self):
        for _ in range(len(self._publications)):
            publication = self._publications.popleft()
            plan, record = publication.plan, publication.record
            if plan.state.call.failure is not None:
                continue
            if not plan.admitted or (
                plan.state.retain_result
                and self._vector_dim is not None
                and not plan.state.result_credit
            ):
                self._publications.append(publication)
                continue
            while publication.positions:
                if (
                    plan.state.retain_result
                    and self._vector_dim is None
                    and plan.seed_published
                ):
                    break
                if (
                    self._bytes + publication.byte_cost > self.max_bytes
                    or self._tokens + record.token_count > self.max_tokens
                ):
                    break
                position = publication.positions.popleft()
                entry = _Entry(
                    plan,
                    position,
                    record,
                    publication.byte_cost,
                    self.adapter._batch_now(),
                )
                lane = self._lanes.setdefault(plan.model, {})
                queue = lane.get(plan.state)
                if queue is None:
                    queue = lane[plan.state] = _ReadyQueue()
                plan.state.ready[plan.model] = queue
                queue.append(entry)
                plan.seed_published = True
                self._bytes += entry.byte_cost
                self._tokens += record.token_count
                self._peak_bytes = max(self._peak_bytes, self._bytes)
                self._peak_tokens = max(self._peak_tokens, self._tokens)
            if publication.positions:
                self._publications.append(publication)

    def _prepare(self, states):
        self._preparation_blocked = False
        if any(
            item.plan.admitted
            and not (
                item.plan.state.retain_result
                and self._vector_dim is not None
                and not item.plan.state.result_credit
            )
            and not (
                item.plan.state.retain_result
                and self._vector_dim is None
                and item.plan.seed_published
            )
            for item in self._publications
        ):
            self._preparation_blocked = True
            return
        owner_rotation = 0
        while self._token_active < self.tokenizers:
            selected = None
            choices = []
            for offset in range(len(states)):
                state = states[(owner_rotation + offset) % len(states)]
                if state.context is None or state.call.failure is not None:
                    continue
                if state.retain_result:
                    if self._vector_dim is not None and not state.result_credit:
                        continue
                    if (
                        self._vector_dim is None
                        and self._unknown_seeds >= self.parallelism
                    ):
                        continue
                start = state.plan_rotation % max(1, len(state.plans))
                for offset in range(len(state.plans)):
                    index = (start + offset) % len(state.plans)
                    plan = state.plans[index]
                    if not plan.groups or not plan.admitted:
                        continue
                    if state.retain_result and self._vector_dim is None and plan.seeded:
                        continue
                    choices.append((state, plan, index))
            # A fast worker continues feeding a lane that has no tokenizer
            # active instead of joining another lane's slow in-flight inputs.
            # The stable owner order still rotates fairly within each lane.
            choices.sort(key=lambda item: self._preparing_by_model[item[1].model])
            for state, plan, index in choices:
                if state.call.failure is not None:
                    continue
                item_index, positions = plan.groups[0]
                text = self._borrowed_key(state.inputs, item_index)[0]
                prefix = getattr(state.inputs, "prefix", "")
                if len(text) + len(prefix) > self.max_request_bytes:
                    self._fail(state, self._resource("request_bytes"))
                    continue
                try:
                    utf8_bytes = len(text.encode("utf-8")) + len(prefix.encode("utf-8"))
                except UnicodeError:
                    self._fail(
                        state,
                        LLMBackendError("llamacpp_embedding_invalid_utf8_input"),
                    )
                    continue
                # Mixed-width Python strings can use four bytes per
                # character even when most UTF-8 bytes are ASCII.
                raw_cost = 336 + 4 * (len(text) + len(prefix))
                if raw_cost > self.max_bytes:
                    self._fail(state, self._resource("pending_bytes"))
                    continue
                if utf8_bytes > self.max_request_bytes:
                    self._fail(state, self._resource("request_bytes"))
                    continue
                if self._bytes + raw_cost <= self.max_bytes:
                    selected = state, plan, item_index, positions, raw_cost
                    state.plan_rotation = index + 1
                    break
                self._preparation_blocked = True
            if selected is None:
                return
            state, plan, item_index, positions, raw_cost = selected
            plan.groups.popleft()
            self._descriptor_count -= len(positions)
            if state.retain_result and self._vector_dim is None:
                plan.seeded = True
                self._unknown_seeds += 1
            plan.preparing += 1
            self._preparing_by_model[plan.model] += 1
            self._bytes += raw_cost
            self._raw_bytes += raw_cost
            self._peak_bytes = max(self._peak_bytes, self._bytes)
            item = state.inputs[item_index]
            self._task(
                self._tokenizer,
                "prepare",
                state,
                (plan, positions, raw_cost),
                lambda item=item, plan=plan, state=state: self.adapter._engine_prepare(
                    item,
                    plan.model,
                    state.call,
                    state.context.observations[plan.model],
                    state.context.conservative,
                ),
            )
            owner_rotation += 1

    def _owner_eligible(self, state, model):
        return (
            state.context is not None
            and not state.finishing
            and not state.future.done()
            and state.call.failure is None
            and model in state.context.observations
            and (
                not state.retain_result
                or state.result_credit
                or self._vector_dim is None
            )
        )

    def _metadata_member(self, state, model):
        return (
            self._owner_eligible(state, model)
            and state.operation != "prepare"
            and (
                bool(state.ready.get(model))
                or state.cursor < len(state.inputs)
                or any(item[0] == model for item in state.descriptors)
                or any(
                    plan.model == model and (plan.groups or plan.preparing)
                    for plan in state.plans
                )
                or any(
                    item.plan.state is state and item.plan.model == model
                    for item in self._publications
                )
            )
        )

    def _pending_admission(self, state):
        """Registered, useful work whose admission can still make progress.

        A topology-busy admission must let existing pool users drain. Result
        credit waiters likewise cannot hold up owners that need to complete
        before the consumer can release their credits.
        """
        return (
            state.context is None
            and not state.finishing
            and not state.future.done()
            and state.call.failure is None
            and state.operation != "prepare"
            and bool(state.inputs)
            and (state.starting or state.blocked_at is None)
            and (
                not state.retain_result
                or state.result_credit
                or self._vector_dim is None
            )
        )

    def _can_produce(self, model, states):
        """Whether useful admitted work can still fill this physical lane."""
        for state in states:
            if self._pending_admission(state):
                return True
            if state.operation == "prepare" or not self._owner_eligible(state, model):
                continue
            if any(
                plan.admitted
                and plan.model == model
                and (plan.groups or plan.preparing)
                for plan in state.plans
            ):
                return True
            if any(
                item.plan.state is state
                and item.plan.model == model
                and item.plan.admitted
                for item in self._publications
            ):
                return True
            # Admission runs after dispatch. Do not prematurely flush just
            # before an already registered producer can obtain a free plan.
            if (
                self._plans < self.capacity
                and self._descriptor_count < max(1, self.max_bytes // 1024)
                and (
                    state.cursor < len(state.inputs)
                    or any(item[0] == model for item in state.descriptors)
                )
            ):
                return True
        return False

    def _forced_flush(self, selected, bounded):
        if not selected:
            return False
        if bounded or self._preparation_blocked or self._tokens >= self.max_tokens:
            return True
        if self._vector_dim is None and any(
            item.plan.state.retain_result for item in selected
        ):
            return True
        wait = self.adapter.embedding_batch_wait_sec
        return (
            bool(wait)
            and self.adapter._batch_now() - min(entry.ready_at for entry in selected)
            >= wait
        )

    def _eligible(self, model, selected, bounded, states):
        return bool(selected) and (
            self._forced_flush(selected, bounded)
            or not self._can_produce(model, states)
        )

    def _ready(self, model, observation, states):
        """Decide whether packing can make progress from queue summaries.

        Readiness never freezes a fair prefix: owner order, metadata scope,
        loaded limits and scheduler capacity are reconsidered at dispatch.
        With fewer than the text ceiling all ready rows would be examined,
        so their ledgers detect an exact token/wire boundary without repeatedly
        traversing the same growing prefix after each preparation completion.
        """
        owners = [
            state
            for state in states
            if self._owner_eligible(state, model) and state.ready.get(model)
        ]
        if not owners:
            return False
        queues = [state.ready[model] for state in owners]
        count = sum(map(len, queues))
        cap = self.adapter._current_batch_size(
            self.adapter._scheduler.text_cap(self.adapter.batch_size),
            self.adapter._replicas[model],
        )
        if count >= cap:
            return True
        budget = observation.batch_token_limit
        for queue in queues:
            limit = queue.minimum_limit
            if limit is not None:
                budget = limit if budget is None else min(budget, limit)
        if budget is not None and sum(queue.tokens for queue in queues) >= budget:
            return True
        overhead = self.adapter._engine_payload_overhead(model)
        native_ids = (
            self.adapter.embedding_input_format != "text"
            and getattr(self.adapter._client, "supports_embedding_token_ids", None)
            is True
        )
        missing = (
            min(
                (
                    (queue.first_missing_ids, index)
                    for index, queue in enumerate(queues)
                    if queue.first_missing_ids is not None
                ),
                default=None,
            )
            if native_ids
            else None
        )
        if not native_ids or missing is not None:
            text_size = overhead + sum(queue.text_bytes for queue in queues)
            if text_size + 2 * (count - 1) >= self.max_request_bytes:
                return True
        if native_ids:
            # Fair rotation reaches the first row without reusable IDs at
            # (round, owner). Earlier queues contribute one more row in that
            # round; exhausted queues naturally stop contributing. All later
            # rows use the complete text payload instead of token IDs.
            if missing is None:
                counts = [len(queue) for queue in queues]
            else:
                round_index, owner_index = missing
                counts = [
                    min(len(queue), round_index + (index < owner_index))
                    for index, queue in enumerate(queues)
                ]
            native_count = sum(counts)
            if native_count:
                native_size = overhead + sum(
                    queue.id_prefix_bytes(size) for queue, size in zip(queues, counts)
                )
                if native_size + 2 * (native_count - 1) >= self.max_request_bytes:
                    return True
        if self._preparation_blocked or self._tokens >= self.max_tokens:
            return True
        if self._vector_dim is None and any(state.retain_result for state in owners):
            return True
        if not self._can_produce(model, states):
            return True
        wait = self.adapter.embedding_batch_wait_sec
        return (
            bool(wait)
            and self.adapter._batch_now() - min(queue[0].ready_at for queue in queues)
            >= wait
        )

    def _pack(self, model, observation, states):
        """Fairly gather owners without summing their source-window ceilings."""
        cap = self.adapter._current_batch_size(
            self.adapter._scheduler.text_cap(self.adapter.batch_size),
            self.adapter._replicas[model],
        )
        owners = deque(
            (state, iter(state.ready[model]))
            for state in states
            if self._owner_eligible(state, model) and state.ready.get(model)
        )
        selected, tokens = [], 0
        budget = observation.batch_token_limit
        text_bytes = token_bytes = self.adapter._engine_payload_overhead(model)
        native_ids = (
            self.adapter.embedding_input_format != "text"
            and getattr(self.adapter._client, "supports_embedding_token_ids", None)
            is True
        )
        while owners:
            state, entries = owners.popleft()
            try:
                entry = next(entries)
            except StopIteration:
                continue
            owners.append((state, entries))
            self._packing_candidate_visits += 1
            record = entry.record
            limit = record.batch_token_limit
            effective = (
                budget
                if limit is None
                else limit
                if budget is None
                else min(budget, limit)
            )
            if (
                selected
                and effective is not None
                and tokens + record.token_count > effective
            ):
                return selected, True
            row_text = record._text_wire_bytes
            row_tokens = record._tokens_wire_bytes
            use_ids = native_ids and row_tokens is not None and bool(record._token_ids)
            candidate_text = text_bytes + row_text + (2 if selected else 0)
            candidate_tokens = token_bytes + (row_tokens or 0) + (2 if selected else 0)
            size = candidate_tokens if use_ids else candidate_text
            if size > self.max_request_bytes:
                if selected:
                    return selected, True
                self._fail(state, self._resource("request_bytes"))
                return [], False
            selected.append(entry)
            tokens += record.token_count
            text_bytes, token_bytes, native_ids = (
                candidate_text,
                candidate_tokens,
                use_ids,
            )
            budget = effective
            if (
                len(selected) >= cap
                or (budget is not None and tokens >= budget)
                or size >= self.max_request_bytes
            ):
                return selected, True
        return selected, False

    def _dispatch(self, states):
        scheduler = self.adapter._scheduler
        for model, preflight in tuple(self._preflights.items()):
            # A completed observation belongs only to members that existed
            # before its catalog request. New owners require another round.
            members = set(preflight.members)
            scope = [state for state in states if state in members]
            self._preflights.pop(model, None)
            if not scheduler.has_capacity(model):
                continue
            observation = preflight.observation
            if not self._ready(model, observation, scope):
                continue
            selected, bounded = self._pack(model, observation, scope)
            # Frozen membership limits which rows this snapshot may send;
            # it does not prove the whole model queue has reached its tail.
            # Renew a partial round when a later owner can help fill it,
            # rather than sending the old members' artificially short tail.
            if not self._forced_flush(selected, bounded) and any(
                state not in members
                and (
                    self._metadata_member(state, model)
                    or self._pending_admission(state)
                )
                for state in states
            ):
                continue
            if not self._eligible(model, selected, bounded, scope):
                continue
            try:
                distinct = tuple(dict.fromkeys(entry.plan.state for entry in selected))
                group = self.adapter._engine_batch_call(
                    tuple(owner.call for owner in distinct)
                )
                batch = _Batch(selected, distinct, group)
                future = scheduler.try_dispatch(
                    lambda _alias, batch=batch, model=model, observed=observation: (
                        self.adapter._engine_run_shared(
                            model,
                            [item.record for item in batch.entries],
                            [item.plan.state.call for item in batch.entries],
                            batch.call,
                            observed,
                        )
                    ),
                    group.check_local,
                    model=model,
                )
                if future is None:
                    continue
            except BaseException as exc:
                for owner in dict.fromkeys(entry.plan.state for entry in selected):
                    self._fail(owner, exc)
                continue
            for entry in selected:
                assert entry.plan.state.ready[model].popleft() is entry
                entry.plan.http_active += 1
            for owner in distinct:
                owner.pending += 1
            self._http_active += 1
            self._peak_http_active = max(self._peak_http_active, self._http_active)
            future.add_done_callback(
                lambda done, batch=batch: self._event("http", None, batch, done)
            )
        if self._control_active >= self.control_workers:
            return
        budget = (
            self.parallelism
            - scheduler.snapshot()["in_flight"]
            - len(self._observing)
            - len(self._preflights)
        )
        members = {}
        for model in dict.fromkeys(model for state in states for model in state.ready):
            if len(members) >= budget:
                break
            if (
                model in self._observing
                or model in self._preflights
                or not scheduler.has_capacity(model)
            ):
                continue
            scope = tuple(
                state for state in states if self._metadata_member(state, model)
            )
            producers = scope + tuple(
                state for state in states if self._pending_admission(state)
            )
            owner = next((state for state in scope if state.ready.get(model)), None)
            if owner is None:
                continue
            if not self._ready(model, owner.context.observations[model], producers):
                continue
            selected, bounded = self._pack(
                model, owner.context.observations[model], scope
            )
            if self._eligible(model, selected, bounded, producers):
                members[model] = scope
        if members:
            metadata = _MetadataRound(members)
            owners = tuple(
                dict.fromkeys(owner for scope in members.values() for owner in scope)
            )
            self._observing.update(members)
            self._shared_task(
                "catalog_shared",
                owners,
                metadata,
                lambda owners=owners: self.adapter._engine_catalog_shared(
                    tuple(owner.call for owner in owners)
                ),
            )

    def _complete(self, state):
        failure = state.call.failure
        try:
            result = (
                None
                if failure is not None
                else self.adapter._engine_result(
                    state.call, state.operation, state.outputs
                )
            )
        finally:
            self.adapter._engine_finish(
                state.call, state.context.joined if state.context else False
            )
        return result

    @staticmethod
    def _output_bytes(outputs):
        return 64 + sum(96 + 40 * len(vector) for vector in outputs)

    def _reserve_result(self, state):
        if not state.retain_result:
            return True
        dimension = self._vector_dim
        if dimension is None and state.inputs:
            # The shared scheduler bounds all first HTTPs by P, also when one
            # owner is using several replicas. Their bounded responses may
            # discover dimension concurrently.
            # Its response stays in the HTTP handoff until the whole window has
            # result credits; it is not an unaccounted retained public result.
            return True
        wanted = (
            64 if not state.inputs else 64 + len(state.inputs) * (96 + 40 * dimension)
        )
        if wanted > self.max_bytes:
            self._fail(state, self._resource("result_bytes"))
            return False
        delta = wanted - state.result_credit
        if self._result_reserved + delta > self.max_bytes:
            return False
        self._result_reserved += delta
        state.result_credit = wanted
        if state.outputs is None:
            state.outputs = [None] * len(state.inputs)
        self._peak_result_reserved = max(
            self._peak_result_reserved, self._result_reserved
        )
        return True

    @staticmethod
    def _publish_http_result(state, destinations, vectors):
        for (plan, index), vector in zip(destinations, vectors):
            state.outputs[index] = vector
            plan.finished += 1

    def _publish_http(self):
        for _ in range(len(self._http_publications)):
            state, destinations, vectors = self._http_publications.popleft()
            if state.call.failure is not None:
                continue
            if state.result_credit:
                self._publish_http_result(state, destinations, vectors)
            else:
                self._http_publications.append((state, destinations, vectors))

    def _sync_result_admission(self):
        """Dormant uncredited windows cannot monopolize the working pool.

        Unknown-dimension bootstrap prepares at most P seed records. Unsent
        seeds can move back to that bounded handoff when a different window
        gets result credits first. Their borrowed index plans release lookahead
        admission until the owner can reserve its complete result window.
        """
        if self._vector_dim is None:
            return
        for state in self._states:
            if not state.retain_result or state.call.failure is not None:
                continue
            if not state.result_credit:
                for entries in state.ready.values():
                    while entries:
                        entry = entries.popleft()
                        self._release_entry(entry)
                        self._publications.append(
                            _Publication(
                                entry.plan,
                                deque([entry.index]),
                                entry.record,
                                entry.byte_cost,
                            )
                        )
                for plan in state.plans:
                    if plan.admitted and not plan.preparing and not plan.http_active:
                        if plan.preparation_lease:
                            self._plans -= 1
                            plan.preparation_lease = False
                        self._descriptor_count -= sum(
                            len(positions) for _, positions in plan.groups
                        )
                        plan.admitted = False
        for state in self._states:
            if (
                not state.retain_result
                or not state.result_credit
                or state.call.failure is not None
            ):
                continue
            for plan in state.plans:
                count = sum(len(positions) for _, positions in plan.groups)
                if (
                    not plan.admitted
                    and (not plan.groups or self._plans < self.capacity)
                    and self._descriptor_count + count <= max(1, self.max_bytes // 1024)
                ):
                    plan.admitted = True
                    if plan.groups:
                        self._plans += 1
                        plan.preparation_lease = True
                    self._descriptor_count += count

    def _finish_future(self, state, result):
        self._states.remove(state)
        for lane in self._lanes.values():
            lane.pop(state, None)
        if (
            state.retain_result
            and state.call.failure is None
            and state.call not in self._released_calls
        ):
            cost = self._output_bytes(result)
            if cost > state.result_credit:
                # Dimensions have been validated before result publication.
                # A larger object must never silently exceed its reservation.
                self._fail(state, self._resource("result_bytes"))
                self._result_reserved -= state.result_credit
            else:
                self._result_reserved -= state.result_credit - cost
                self._retained[state.call] = cost
                self._result_bytes += cost
                self._peak_result_bytes = max(
                    self._peak_result_bytes, self._result_bytes
                )
        else:
            self._result_reserved -= state.result_credit
        self._released_calls.discard(state.call)
        self._update_view()
        if state.call.failure is not None:
            state.future.set_exception(state.call.failure)
        else:
            state.future.set_result(result)

    def _finish(self, states):
        self._release_preparation_leases(states)
        for state in states:
            if state.finishing:
                continue
            if (
                state.retain_result
                and not state.result_credit
                and state.call.failure is None
            ):
                continue
            for plan in tuple(state.plans):
                if state.call.failure is not None or plan.finished == plan.size:
                    if not plan.preparing and (
                        state.call.failure is None or not state.pending
                    ):
                        state.plans.remove(plan)
                        self._progress += 1
                        self._retained_plan_count -= 1
                        self._plan_counts[plan.model] -= 1
                        if not self._plan_counts[plan.model]:
                            del self._plan_counts[plan.model]
                        if plan.seeded:
                            self._unknown_seeds -= 1
                            plan.seeded = False
                        if plan.preparation_lease:
                            self._plans -= 1
            failed = state.call.failure is not None
            exhausted = state.cursor == len(state.inputs) and not state.descriptors
            if state.pending or (
                not failed and (state.context is None or not exhausted or state.plans)
            ):
                continue
            if self._control_active >= self.control_workers:
                continue
            state.finishing = True
            self._task(
                self._control,
                "finish",
                state,
                None,
                lambda state=state: self._complete(state),
            )

    def _release_preparation_leases(self, states):
        for state in states:
            if (
                state.retain_result
                and not state.result_credit
                and state.call.failure is None
            ):
                continue
            for plan in state.plans:
                if plan.preparation_lease and not plan.groups and not plan.preparing:
                    self._plans -= 1
                    plan.preparation_lease = False

    def _admit_plans(self, states):
        while self._plans < self.capacity:
            admitted = False
            for state in states:
                if (
                    state.context is not None
                    and not state.finishing
                    and state.call.failure is None
                    and self._plans < self.capacity
                    and (state.cursor < len(state.inputs) or state.descriptors)
                    and self._descriptor_count < max(1, self.max_bytes // 1024)
                    and (
                        not state.retain_result
                        or state.result_credit
                        or self._vector_dim is None
                    )
                    and not (
                        state.retain_result
                        and self._vector_dim is None
                        and len(state.plans)
                        >= min(len(state.context.observations), self.parallelism)
                    )
                ):
                    try:
                        self._plan(state)
                        admitted = True
                    except BaseException as exc:
                        self._fail(state, exc)
            if not admitted:
                break

    def _start_cohorts(self, states):
        with self._condition:
            marker = self._progress, self._metadata_availability
        waiting = [
            state
            for state in states
            if state.context is None
            and not state.starting
            and not state.finishing
            and state.call.failure is None
            and state.blocked_at != marker
        ]
        while waiting and self._control_active < self.control_workers:
            cohort = tuple(waiting[: self.capacity])
            del waiting[: self.capacity]
            for state in cohort:
                state.starting = True
            self._shared_task(
                "start_shared",
                cohort,
                marker,
                lambda cohort=cohort: self.adapter._engine_start_shared(
                    tuple(state.call for state in cohort)
                ),
            )

    def _update_view(self):
        ready = tuple(
            (model, sum(len(queue) for queue in owners.values()))
            for model, owners in self._lanes.items()
        )
        signature = (
            self._bytes,
            self._peak_bytes,
            self._tokens,
            self._peak_tokens,
            self._plans,
            self._retained_plan_count,
            self._peak_plans,
            self._token_active,
            self._peak_token_active,
            self._http_active,
            self._peak_http_active,
            self._control_active,
            len(self._observing),
            len(self._preflights),
            len(self._metadata_rounds),
            self._packing_candidate_visits,
            len(self._publications),
            len(self._http_publications),
            self._result_bytes,
            self._peak_result_bytes,
            self._result_reserved,
            self._peak_result_reserved,
            ready,
        )
        if signature == self._view_signature:
            return
        view = {
            "pending_bytes": self._bytes,
            "peak_pending_bytes": self._peak_bytes,
            "pending_tokens": self._tokens,
            "peak_pending_tokens": self._peak_tokens,
            "active_plans": self._plans,
            "active_preparation_plans": self._plans,
            "retained_plans": self._retained_plan_count,
            "peak_active_plans": self._peak_plans,
            "tokenization_in_flight": self._token_active,
            "peak_tokenization_in_flight": self._peak_token_active,
            "embedding_in_flight": self._http_active,
            "peak_embedding_in_flight": self._peak_http_active,
            "control_in_flight": self._control_active,
            "metadata_targets": len(self._observing) + len(self._preflights),
            "metadata_rounds_waiting": len(self._metadata_rounds),
            "packing_candidate_visits": self._packing_candidate_visits,
            "preparation_results_waiting": len(self._publications),
            "http_results_waiting": len(self._http_publications),
            "retained_result_bytes": self._result_bytes,
            "peak_retained_result_bytes": self._peak_result_bytes,
            "result_reserved_bytes": self._result_reserved,
            "peak_result_reserved_bytes": self._peak_result_reserved,
            "http_parallelism_bound": self.parallelism
            + self.tokenizers
            + self.control_workers,
            "ready_by_replica": dict(ready),
        }
        with self._condition:
            self._view = view
            self._view_signature = signature

    def snapshot(self):
        with self._condition:
            return {
                **self._view,
                "ready_by_replica": dict(self._view["ready_by_replica"]),
            }

    def _next_wait(self):
        """Only unnotifiable consumer callbacks and actual deadlines need timers.

        Worker completions, stop, metadata availability and result credits all
        carry a remembered notification. Custom coalescing clocks are probed
        finitely because advancing one cannot notify this condition itself.
        """
        now = time.monotonic()
        wakeups = []
        for state in self._states:
            if state.finishing or state.call.failure is not None:
                continue
            wakeups.append(max(0.0, state.call.deadline - now))
            if (
                state.call.cancel is not None
                and not state.checking
                and self._control_active < self.control_workers
            ):
                wakeups.append(max(0.0, state.checked_at + 0.05 - now))
        wait = self.adapter.embedding_batch_wait_sec
        if wait and self._control_active < self.control_workers:
            clock = self.adapter._batch_now
            normal_clock = (
                getattr(getattr(clock, "__func__", None), "__qualname__", "")
                == "LlamaCppEmbeddingAdapter._batch_now"
            )
            batch_now = clock()
            for model, owners in self._lanes.items():
                if (
                    model in self._observing
                    or model in self._preflights
                    or not self.adapter._scheduler.has_capacity(model)
                ):
                    continue
                oldest = min(
                    (queue[0].ready_at for queue in owners.values() if queue),
                    default=None,
                )
                if oldest is None:
                    continue
                remaining = wait - (batch_now - oldest)
                if remaining > 0:
                    wakeups.append(remaining if normal_clock else min(remaining, 0.05))
                # An elapsed timer was already handled in this cycle. If it
                # cannot progress, a resource/metadata event must unblock it.
        return min(wakeups, default=None)

    def _run(self):
        while True:
            with self._condition:
                notification = self._notification
                progress = self._progress
                self._states.extend(self._incoming)
                self._incoming.clear()
                events, self._events = self._events, deque()
                releases, self._release_results = self._release_results, deque()
            for call in releases:
                cost = self._retained.pop(call, 0)
                self._result_bytes -= cost
                self._result_reserved -= cost
                if not cost and any(state.call is call for state in self._states):
                    self._released_calls.add(call)
            self._completions(events)
            states = list(self._states)
            if states:
                offset = self._rotation % len(states)
                states = states[offset:] + states[:offset]
                self._rotation += 1
            for state in self._states:
                if state.call.failure is not None:
                    self._fail(state, state.call.failure)
                else:
                    try:
                        state.call.check_local()
                    except BaseException as exc:
                        self._fail(state, exc)
                if state.call.failure is None and (
                    state.retain_result or state.context is not None
                ):
                    self._reserve_result(state)
            self._sync_result_admission()
            self._publish_http()
            self._publish()
            # Decide readiness only after exhausted preparation credits have
            # admitted the next registered producers. Otherwise a false tail
            # would fetch metadata and then discard it while the batch fills.
            self._release_preparation_leases(states)
            self._admit_plans(states)
            self._prepare(states)
            self._advance_metadata()
            self._dispatch(states)
            self._finish(states)
            for state in states:
                if state.finishing or state.call.failure is not None:
                    continue
                if (
                    state.call.cancel is not None
                    and not state.checking
                    and time.monotonic() - state.checked_at >= 0.05
                    and self._control_active < self.control_workers
                ):
                    state.checking = True
                    self._task(self._control, "check", state, None, state.call.check)
            self._start_cohorts(states)
            self._update_view()
            # A local transition can release admission without scheduling a
            # worker (for example retiring a seed plan). Settle that transition
            # once more before waiting for external work.
            changed = progress != self._progress
            timeout = self._next_wait()
            with self._condition:
                if self._closed and not self._states and not self._incoming:
                    return
                if (
                    self._events
                    or self._incoming
                    or self._release_results
                    or notification != self._notification
                    or changed
                ):
                    continue
                self._condition.wait(timeout)

    def close(self):
        with self._condition:
            self._closed = True
            for state in (*self._incoming, *self._states):
                state.call.stop.set()
            self._notify_locked()
            coordinator = self._thread
        if coordinator is not None:
            if coordinator is current_thread():
                raise LLMBackendError("llamacpp_embedding_close_from_coordinator")
            coordinator.join()
            self._control.shutdown(wait=True, cancel_futures=True)
            self._tokenizer.shutdown(wait=True, cancel_futures=True)
        self._retained.clear()
        self._preflights.clear()
        self._metadata_rounds.clear()
        self._observing.clear()
        self._released_calls.clear()
        self._release_results.clear()
        self._result_bytes = self._result_reserved = 0
        self._update_view()
