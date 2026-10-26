"""Adapter-scoped, bounded admission for external embedding requests."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
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
    admitted: bool = False


class EmbeddingScheduler:
    """Plan text quotas and admit ready affine jobs on one bounded queue."""

    def __init__(
        self, parallelism: int | None, queue_size: int | None, replicas: dict[str, int]
    ):
        if parallelism is not None and (
            not isinstance(parallelism, int)
            or isinstance(parallelism, bool)
            or parallelism <= 0
        ):
            raise ValueError("parallelism must be None or a positive integer")
        if queue_size is not None and (
            not isinstance(queue_size, int)
            or isinstance(queue_size, bool)
            or queue_size < 0
        ):
            raise ValueError("queue_size must be a nonnegative integer")
        self._validate_limits(replicas)
        self.requested_parallelism = parallelism
        self.parallelism = (
            len(replicas) if parallelism is None else min(parallelism, len(replicas))
        )
        self.queue_size = queue_size
        self._requested_limits = dict(replicas)
        self._limits = dict.fromkeys(replicas, 1)
        self._aliases = tuple(replicas)
        self._alias_positions = {
            alias: index for index, alias in enumerate(self._aliases)
        }
        self._replica_active = dict.fromkeys(replicas, 0)
        self._replica_peak = dict.fromkeys(replicas, 0)
        self._condition = Condition()
        self._queue: deque[_Job] = deque()
        self._active = self._peak_active = self._peak_queued = 0
        self._next_alias = 0
        self._next_dispatch_alias = 0
        self._closed = False
        self._executor = ThreadPoolExecutor(
            # An unavailable topology owns no work or worker thread. The
            # executor stays lazy, with a legal constructor size for N=0.
            max_workers=max(1, self.parallelism),
            thread_name_prefix="runtime_llama_embedding",
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
            count = min(text_count, len(self._aliases))
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
            # Full-width windows must rotate the larger odd quotas too.
            self._next_alias = (self._next_alias + (remainder or count)) % len(
                self._aliases
            )
            return planned

    def text_cap(self, text_budget: int) -> int:
        """Return one configured quota ceiling without rotating destinations.

        Small source windows keep their own distribution credits, but do not
        reduce the ceiling of an HTTP batch assembled from several windows.
        """
        if text_budget is None:
            return None
        if (
            not isinstance(text_budget, int)
            or isinstance(text_budget, bool)
            or text_budget <= 0
        ):
            raise ValueError("text_budget must be a positive integer")
        with self._condition:
            if self._closed:
                raise LLMBackendError("llamacpp_embedding_adapter_closed")
            count = min(text_budget, len(self._aliases))
            if not count:
                raise LLMBackendError("llamacpp_embedding_no_usable_replica")
            return (text_budget + count - 1) // count

    def _can_dispatch(self, model: str) -> bool:
        return (
            self._active < self.parallelism
            and self._replica_active[model] < self._limits[model]
        )

    def dispatch_order(self, models: Iterable[str] | None = None) -> tuple[str, ...]:
        """Read alias turns without consuming one before an actual dispatch.

        The coordinator uses this same order for observation and admission.
        Stale aliases from a previous local topology are ignored. Quota and
        consumer turns are independent of this scheduling cursor.
        """
        with self._condition:
            if self._closed:
                # A coordinator may have captured this local epoch just before
                # drained replacement. Like has_capacity(), this is only a
                # readiness observation and must not kill that coordinator.
                return ()
            ordered = (
                self._aliases[self._next_dispatch_alias :]
                + self._aliases[: self._next_dispatch_alias]
            )
            if models is None:
                return ordered
            selected = set(models)
            return tuple(alias for alias in ordered if alias in selected)

    @staticmethod
    def _validate_limits(replicas: dict[str, int]) -> None:
        if any(
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
            for limit in replicas.values()
        ):
            raise ValueError("replica limits must be positive integers")

    def update_limits(self, replicas: dict[str, int]) -> None:
        """Refresh requested caps without multiplying an original alias lease.

        Topology belongs to the adapter's drained replacement boundary.
        Native slots and accepted positive legacy caps never increase HTTP
        admission beyond one original job for each verified replica.
        """
        self._validate_limits(replicas)
        limits = dict(replicas)
        with self._condition:
            if self._closed:
                raise LLMBackendError("llamacpp_embedding_adapter_closed")
            if limits.keys() != self._limits.keys():
                raise LLMBackendError("llamacpp_embedding_scheduler_topology_changed")
            self._requested_limits = limits
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
        try:
            self._executor.submit(self._run, job)
        except BaseException as exc:
            self._active -= 1
            self._replica_active[job.model] -= 1
            # A partially enqueued orphan must not retain the rejected batch
            # and its owners while waiting for a future executor worker.
            job.run = lambda _model: None
            job.check = lambda: None
            self._condition.notify_all()
            if isinstance(exc, Exception) and not isinstance(exc, LLMBackendError):
                raise LLMBackendError(
                    "llamacpp_embedding_scheduler_submission_failed"
                ) from exc
            raise
        # Workers first acquire this same condition. In particular an executor
        # that enqueues before a thread-start failure cannot run abandoned work
        # later, after these reservations have already been rolled back.
        job.admitted = True
        self._peak_active = max(self._peak_active, self._active)
        self._replica_peak[job.model] = max(
            self._replica_peak[job.model], self._replica_active[job.model]
        )
        self._next_dispatch_alias = (self._alias_positions[job.model] + 1) % len(
            self._aliases
        )

    def _run(self, job: _Job) -> None:
        with self._condition:
            if not job.admitted:
                return
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
            candidates = {}
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
                # Retain FIFO per alias; a perpetually ready first alias must
                # not consume every global permit when the guard is below N.
                if self._can_dispatch(job.model):
                    candidates.setdefault(job.model, job)
            order = self.dispatch_order(candidates)
            if not order:
                break
            candidate = candidates[order[0]]
            self._queue.remove(candidate)
            try:
                self._dispatch(candidate)
            except BaseException as exc:
                # Pump also runs while a previous worker is releasing its
                # lease. A rejected queued dispatch must fail its own Future,
                # never prevent completion of the already drained worker.
                self._fail(candidate, exc)

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
        if self.queue_size is None or len(self._queue) < self.queue_size:
            self._queue.append(job)
            self._peak_queued = max(self._peak_queued, len(self._queue))
            return True
        return False

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "replica_count": len(self._aliases),
                "requested_parallelism": self.requested_parallelism,
                "effective_parallelism": self.parallelism,
                "capacity_reason": (
                    "unavailable"
                    if not self._aliases
                    else "configured_guard"
                    if self.requested_parallelism is not None
                    and self.requested_parallelism < len(self._aliases)
                    else "replica_count"
                ),
                "in_flight": self._active,
                "queued": len(self._queue),
                "peak_in_flight": self._peak_active,
                "peak_queued": self._peak_queued,
                "replicas": {
                    alias: {
                        "in_flight": self._replica_active[alias],
                        "peak_in_flight": self._replica_peak[alias],
                        "max_in_flight": self._limits[alias],
                        "requested_max_in_flight": self._requested_limits[alias],
                        "capacity_reason": "one_original_post_per_replica",
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
