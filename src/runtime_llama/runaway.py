"""Bounded, text-delta based runaway detection for streamed local inference."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import hashlib
from typing import Deque, Dict, Optional, Tuple


@dataclass(frozen=True)
class RunawayDiagnostics:
    channel: str
    char_offset: int
    window_chars: int
    repeat_count: int
    detector: str

    def to_dict(self) -> Dict[str, int | str]:
        return {
            "channel": self.channel,
            "char_offset": self.char_offset,
            "window_chars": self.window_chars,
            "repeat_count": self.repeat_count,
            "detector": self.detector,
        }


@dataclass
class _ChannelState:
    total_chars: int = 0
    trailing: str = ""
    rolling_tail: str = ""
    hashes: Deque[Tuple[int, bytes]] = field(default_factory=deque)
    counts: Counter[bytes] = field(default_factory=Counter)
    in_whitespace: bool = False


class RunawayDetector:
    """Detect repeated normalized text without model-token or prompt access.

    Backends expose transport deltas, not portable tokenizer IDs.  The detector
    therefore keeps a bounded normalized character representation per output
    channel.  It never emits or logs the generated text.
    """

    def __init__(
        self,
        *,
        min_chars: int,
        window_chars: int,
        block_chars: int,
        repeat_threshold: int,
        ngram_chars: int,
    ) -> None:
        values = {
            "min_chars": min_chars,
            "window_chars": window_chars,
            "block_chars": block_chars,
            "repeat_threshold": repeat_threshold,
            "ngram_chars": ngram_chars,
        }
        if any(not isinstance(value, int) or value <= 0 for value in values.values()):
            raise ValueError("runaway detector settings must be positive integers")
        if window_chars < max(block_chars * 2, ngram_chars * repeat_threshold):
            raise ValueError(
                "runaway_window_chars is too small for its detector settings"
            )
        self._min_chars = min_chars
        self._window_chars = window_chars
        self._block_chars = block_chars
        self._repeat_threshold = repeat_threshold
        self._ngram_chars = ngram_chars
        self._cycle_chars = ngram_chars * repeat_threshold
        self._max_delta_chars = min(window_chars, 4096)
        self._channels: Dict[str, _ChannelState] = {}

    @staticmethod
    def _normalize_delta(state: _ChannelState, text: str) -> str:
        """Normalize one delta without making output depend on chunk splits."""
        normalized: list[str] = []
        for char in str(text or "").casefold():
            if char.isspace():
                if not state.in_whitespace:
                    normalized.append(" ")
                state.in_whitespace = True
                continue
            normalized.append(char)
            state.in_whitespace = False
        value = "".join(normalized)
        return value.lstrip() if state.total_chars == 0 else value

    @staticmethod
    def _hash(block: str) -> bytes:
        return hashlib.blake2s(
            block.encode("utf-8", errors="replace"), digest_size=12
        ).digest()

    def observe(self, channel: str, text: str) -> Optional[RunawayDiagnostics]:
        key = str(channel or "content")
        state = self._channels.get(key)
        if state is None:
            state = _ChannelState()
            self._channels[key] = state
        normalized = self._normalize_delta(state, text)
        if not normalized:
            return None

        # A pathological transport chunk must not turn a bounded guard into an
        # unbounded per-event operation. The recent suffix is the useful signal.
        state.total_chars += len(normalized)
        if len(normalized) > self._max_delta_chars:
            normalized = normalized[-self._max_delta_chars :]
            # Do not bridge a rolling span across text intentionally omitted
            # from one oversized transport event.
            state.rolling_tail = ""

        processed_start = state.total_chars - len(normalized)
        scan = state.rolling_tail + normalized
        scan_start = processed_start - len(state.rolling_tail)
        cutoff = state.total_chars - self._window_chars

        while state.hashes and state.hashes[0][0] <= cutoff:
            _, stale = state.hashes.popleft()
            state.counts[stale] -= 1
            if state.counts[stale] <= 0:
                del state.counts[stale]

        last_start = len(scan) - self._block_chars
        for index in range(max(0, last_start + 1)):
            end_offset = scan_start + index + self._block_chars
            # Previously observed windows end at or before processed_start.
            # Windows outside the final bounded history need not enter state.
            if end_offset <= processed_start or end_offset <= cutoff:
                continue
            block = scan[index : index + self._block_chars]
            digest = self._hash(block)
            state.hashes.append((end_offset, digest))
            state.counts[digest] += 1
            repeat_count = state.counts[digest]
            if (
                state.total_chars >= self._min_chars
                and repeat_count >= self._repeat_threshold
            ):
                return RunawayDiagnostics(
                    channel=str(channel or "content"),
                    char_offset=state.total_chars,
                    window_chars=self._window_chars,
                    repeat_count=repeat_count,
                    detector="rolling_repeated_block",
                )

        tail_size = max(0, self._block_chars - 1)
        state.rolling_tail = scan[-tail_size:] if tail_size else ""
        state.trailing = (state.trailing + normalized)[-self._cycle_chars :]

        if state.total_chars < self._min_chars:
            return None

        ngram_size = self._ngram_chars
        required = self._cycle_chars
        if len(state.trailing) >= required:
            ngram = state.trailing[-ngram_size:]
            if ngram and state.trailing.endswith(ngram * self._repeat_threshold):
                return RunawayDiagnostics(
                    channel=str(channel or "content"),
                    char_offset=state.total_chars,
                    window_chars=self._window_chars,
                    repeat_count=self._repeat_threshold,
                    detector="ngram_cycle",
                )
        return None
