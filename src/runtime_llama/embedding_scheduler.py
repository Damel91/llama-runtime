"""Adapter-scoped, bounded admission for external embedding requests."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Condition
from typing import Any, Callable

from runtime_llama.errors import LLMBackendError


@dataclass(eq=False)
class _Job:
    run: Callable[[str], Any]
    check: Callable[[], None]
    future: Future
    model: str


class EmbeddingScheduler:
    """Plan text quotas and admit ready affine jobs on one bounded queue."""

    def __init__(self, parallelism: int, queue_size: int, replicas: dict[str, int]):
        self.parallelism, self.queue_size = parallelism, queue_size
        self._limits = dict(replicas)
        self._aliases = tuple(replicas)
        self._replica_active = dict.fromkeys(replicas, 0)
        self._replica_peak = dict.fromkeys(replicas, 0)
        self._condition = Condition()
        self._queue: deque[_Job] = deque()
        self._active = self._peak_active = self._peak_queued = 0
        self._next_alias = 0
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=parallelism, thread_name_prefix="runtime_llama_embedding"
        )

    def plan_window(self, text_count: int) -> tuple[tuple[str, int], ...]:
        """Rotate distinct destinations for one total text budget.

        Occupancy only affects admission, not the quota or topology. The caller
        owns the contiguous input slices corresponding to the returned sizes.
        """
        if (
            not isinstance(text_count, int)
            or isinstance(text_count, bool)
            or text_count < 0
        ):
            raise ValueError("text_count must be a nonnegative integer")
        with self._condition:
            if self._closed:
                raise LLMBackendError("llamacpp_embedding_adapter_closed")
            if text_count == 0:
                return ()
            count = min(text_count, len(self._aliases), self.parallelism)
            if count == 0:
                raise LLMBackendError("llamacpp_embedding_no_usable_replica")
            base, remainder = divmod(text_count, count)
            planned = tuple(
                (
                    self._aliases[(self._next_alias + offset) % len(self._aliases)],
                    base + (offset < remainder),
                )
                for offset in range(count)
            )
            self._next_alias = (self._next_alias + count) % len(self._aliases)
            return planned

    def text_cap(self, text_budget: int) -> int:
        """Return one configured quota ceiling without rotating destinations.

        Small source windows keep their own distribution credits, but do not
        reduce the ceiling of an HTTP batch assembled from several windows.
        """
        if (
            not isinstance(text_budget, int)
            or isinstance(text_budget, bool)
            or text_budget <= 0
        ):
            raise ValueError("text_budget must be a positive integer")
        with self._condition:
            if self._closed:
                raise LLMBackendError("llamacpp_embedding_adapter_closed")
            count = min(text_budget, len(self._aliases), self.parallelism)
            if not count:
                raise LLMBackendError("llamacpp_embedding_no_usable_replica")
            return (text_budget + count - 1) // count

    def _can_dispatch(self, model: str) -> bool:
        return (
            self._active < self.parallelism
            and self._replica_active[model] < self._limits[model]
        )

    def update_limits(self, replicas: dict[str, int]) -> None:
        """Apply observed capacities without replaying or aborting active jobs.

        Topology belongs to the adapter's drained replacement boundary. A
        reduction can leave already active work above the new limit; subsequent
        admission waits for it to drain. Increasing a limit wakes eligible jobs.
        """
        if any(
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
            for limit in replicas.values()
        ):
            raise ValueError("replica limits must be positive integers")
        limits = dict(replicas)
        with self._condition:
            if self._closed:
                raise LLMBackendError("llamacpp_embedding_adapter_closed")
            if limits.keys() != self._limits.keys():
                raise LLMBackendError("llamacpp_embedding_scheduler_topology_changed")
            self._limits = limits
            self._pump()
            self._condition.notify_all()

    @staticmethod
    def _fail(job: _Job, exc: BaseException) -> None:
        try:
            job.future.set_exception(exc)
        except InvalidStateError:
            # A caller may cancel its Future before it has been dispatched.
            pass

    def _dispatch(self, job: _Job) -> None:
        self._active += 1
        self._replica_active[job.model] += 1
        self._peak_active = max(self._peak_active, self._active)
        self._replica_peak[job.model] = max(
            self._replica_peak[job.model], self._replica_active[job.model]
        )
        self._executor.submit(self._run, job)

    def _run(self, job: _Job) -> None:
        running = False
        result = None
        failure = None
        try:
            running = job.future.set_running_or_notify_cancel()
            if running:
                try:
                    job.check()
                    result = job.run(job.model)
                except BaseException as exc:
                    failure = exc
        finally:
            with self._condition:
                self._active -= 1
                self._replica_active[job.model] -= 1
                self._pump()
                self._condition.notify_all()
        # Completion is the drain boundary: the job's transport has returned
        # and its scheduler permits have been released before callers wake.
        if running:
            if failure is not None:
                job.future.set_exception(failure)
            else:
                job.future.set_result(result)

    def _pump(self) -> None:
        while self._queue and not self._closed:
            candidate = None
            for job in tuple(self._queue):
                if job.future.cancelled():
                    self._queue.remove(job)
                    continue
                try:
                    job.check()
                except BaseException as exc:
                    self._queue.remove(job)
                    self._fail(job, exc)
                    continue
                # The oldest eligible job wins. Skipping a busy alias cannot
                # overtake an older admitted job for that same alias.
                if candidate is None and self._can_dispatch(job.model):
                    candidate = job
            if candidate is None:
                break
            self._queue.remove(candidate)
            self._dispatch(candidate)

    def submit(
        self,
        run: Callable[[str], Any],
        check: Callable[[], None],
        *,
        model: str,
    ) -> Future:
        """Admit a ready affine job; blocked submitters stay outside the queue."""
        job = _Job(run, check, Future(), model)
        with self._condition:
            while True:
                if self._try_admit(job):
                    return job.future
                self._condition.wait(0.05)

    def try_submit(
        self,
        run: Callable[[str], Any],
        check: Callable[[], None],
        *,
        model: str,
    ) -> Future | None:
        """Admit a ready job or return immediately when bounded capacity is full.

        An unadmitted job has no queue position or execution side effect. The
        coordinator can therefore try a different alias without head-of-line
        blocking, using the same admission and fairness policy as ``submit``.
        """
        job = _Job(run, check, Future(), model)
        with self._condition:
            return job.future if self._try_admit(job) else None

    def has_capacity(self, model: str) -> bool:
        """Observe immediate admission without reserving or enqueueing work."""
        with self._condition:
            return (
                not self._closed
                and model in self._limits
                and self._can_dispatch(model)
                and not any(job.model == model for job in self._queue)
            )

    def try_dispatch(
        self,
        run: Callable[[str], Any],
        check: Callable[[], None],
        *,
        model: str,
    ) -> Future | None:
        """Start only with actual capacity; never freeze a batch in the queue.

        Persistent record queues own work that cannot start yet. Their caller
        selects the current batch immediately before this operation and keeps
        ownership when it returns None. Cancellation callbacks run outside the
        scheduler lock; the worker checks again before touching the transport.
        """
        check()
        job = _Job(run, check, Future(), model)
        with self._condition:
            if self._closed:
                raise LLMBackendError("llamacpp_embedding_adapter_closed")
            if model not in self._limits:
                raise LLMBackendError("llamacpp_embedding_unknown_replica")
            if not self._can_dispatch(model) or any(
                queued.model == model for queued in self._queue
            ):
                return None
            self._dispatch(job)
            return job.future

    def _try_admit(self, job: _Job) -> bool:
        """Shared admission while holding the scheduler condition lock."""
        job.check()
        if self._closed:
            raise LLMBackendError("llamacpp_embedding_adapter_closed")
        if job.model not in self._limits:
            raise LLMBackendError("llamacpp_embedding_unknown_replica")
        self._pump()
        same_alias_queued = any(queued.model == job.model for queued in self._queue)
        if self._can_dispatch(job.model) and not same_alias_queued:
            self._dispatch(job)
            return True
        if len(self._queue) < self.queue_size:
            self._queue.append(job)
            self._peak_queued = max(self._peak_queued, len(self._queue))
            return True
        return False

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "in_flight": self._active,
                "queued": len(self._queue),
                "peak_in_flight": self._peak_active,
                "peak_queued": self._peak_queued,
                "replicas": {
                    alias: {
                        "in_flight": self._replica_active[alias],
                        "peak_in_flight": self._replica_peak[alias],
                        "max_in_flight": self._limits[alias],
                    }
                    for alias in self._aliases
                },
            }

    def close(self) -> None:
        with self._condition:
            self._closed = True
            while self._queue:
                self._fail(
                    self._queue.popleft(),
                    LLMBackendError("llamacpp_embedding_adapter_closed"),
                )
            self._condition.notify_all()
        self._executor.shutdown(wait=True)
