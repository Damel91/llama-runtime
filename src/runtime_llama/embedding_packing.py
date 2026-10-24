"""Consumer turns over views of the engine's existing physical ready queues."""

from collections import deque
from itertools import chain


def _consumer(state):
    group = getattr(state.call, "_packing_group", None)
    return state if group is None else group


class _ConsumerRows:
    """Concatenate one consumer's oldest ready windows without copying rows."""

    def __init__(self, states, model):
        self.states = tuple(
            sorted(states, key=lambda state: getattr(state.call, "_window_index", 0))
        )
        self.queues = tuple(state.ready[model] for state in self.states)

    def __iter__(self):
        return chain.from_iterable(self.queues)

    def __len__(self):
        return sum(map(len, self.queues))

    @property
    def first_missing_ids(self):
        offset = 0
        for queue in self.queues:
            missing = queue.first_missing_ids
            if missing is not None:
                return offset + missing
            offset += len(queue)
        return None

    def id_prefix_bytes(self, count):
        total = 0
        for queue in self.queues:
            size = min(count, len(queue))
            total += queue.id_prefix_bytes(size)
            count -= size
            if not count:
                break
        return total


class _ConsumerTurns:
    """Per-alias row fairness; only an actual dispatch consumes a turn.

    The deques hold consumer identities, never rows or frozen batches. Stream
    identities are opaque tokens; independent calls use their live state as
    their identity. Pruning follows current engine ownership, so completed
    calls and retired aliases cannot remain reachable from this bookkeeping.
    """

    def __init__(self):
        self._aliases = {}

    def prune(self, states, aliases):
        active = dict.fromkeys(_consumer(state) for state in states)
        for model in tuple(self._aliases):
            if model not in aliases or not active:
                del self._aliases[model]
                continue
            self._aliases[model] = deque(
                group for group in self._aliases[model] if group in active
            )
        if active:
            for model in aliases:
                self._aliases.setdefault(model, deque())
        for turns in self._aliases.values():
            known = set(turns)
            turns.extend(group for group in active if group not in known)

    def groups(self, model, states):
        owners = {}
        for state in states:
            owners.setdefault(_consumer(state), []).append(state)
        turns = self._aliases.setdefault(model, deque())
        known = set(turns)
        turns.extend(group for group in owners if group not in known)
        return [
            _ConsumerRows(owners[group], model) for group in turns if group in owners
        ]

    def committed(self, model, entries):
        turns = self._aliases[model]
        last = _consumer(entries[-1].plan.state)
        turns.rotate(-turns.index(last) - 1)
