from itertools import count

import pytest

import runtime_llama.runaway as runaway


def test_channel_state_is_allocated_only_when_a_channel_first_appears(monkeypatch):
    original = runaway._ChannelState
    created = []

    def state():
        value = original()
        created.append(value)
        return value

    monkeypatch.setattr(runaway, "_ChannelState", state)
    detector = runaway.RunawayDetector(
        min_chars=100,
        window_chars=64,
        block_chars=16,
        repeat_threshold=4,
        ngram_chars=2,
    )
    for index in range(30):
        detector.observe("content" if index % 2 else "", f"content-{index}|")
        detector.observe("reasoning", f"plan-{index}|")
    assert len(created) == 2
    assert set(detector._channels) == {"content", "reasoning"}


@pytest.mark.parametrize("window", [12, 64])
@pytest.mark.parametrize("chunk_size", [1, 2, 100])
@pytest.mark.parametrize("text", ["AbC" * 4, "ßK" * 4, "ab \n\t" * 4])
def test_ngram_cycle_at_exact_suffix_and_window_boundaries(window, chunk_size, text):
    detector = runaway.RunawayDetector(
        min_chars=12,
        window_chars=window,
        block_chars=2,
        repeat_threshold=4,
        ngram_chars=3,
    )
    # Isolate the ngram guard: distinct digests keep the independent repeated
    # block guard from detecting the same repetition before the cycle is full.
    unique = count()
    detector._hash = lambda block: next(unique).to_bytes(8, "big")
    diagnostics = None
    for offset in range(0, len(text), chunk_size):
        diagnostics = detector.observe("reasoning", text[offset : offset + chunk_size])
        assert len(detector._channels["reasoning"].trailing) <= 12
        if diagnostics is not None:
            break
    assert diagnostics.detector == "ngram_cycle"
    assert diagnostics.char_offset == 12
    assert diagnostics.repeat_count == 4
    assert diagnostics.channel == "reasoning"
    assert diagnostics.window_chars == window


@pytest.mark.parametrize("chunk_size", [1, 3, 100])
def test_cycle_threshold_waits_for_minimum_chars_without_losing_suffix(chunk_size):
    detector = runaway.RunawayDetector(
        min_chars=13,
        window_chars=12,
        block_chars=2,
        repeat_threshold=4,
        ngram_chars=3,
    )
    unique = count()
    detector._hash = lambda block: next(unique).to_bytes(8, "big")
    text = "abc" * 4 + "a"
    diagnostics = None
    for offset in range(0, len(text), chunk_size):
        diagnostics = detector.observe("content", text[offset : offset + chunk_size])
        if diagnostics:
            break
    assert diagnostics.detector == "ngram_cycle"
    assert diagnostics.char_offset == 13
    assert detector._channels["content"].trailing == "bca" * 4


def test_small_cycle_suffix_does_not_shorten_the_hash_history():
    detector = runaway.RunawayDetector(
        min_chars=512,
        window_chars=64,
        block_chars=8,
        repeat_threshold=4,
        ngram_chars=2,
    )
    for delta in ("abcdefghijklmno", "pqrstuvwxyz012345", "6789ABCDEFGHIJK"):
        assert detector.observe("content", delta) is None
    state = detector._channels["content"]
    assert len(state.trailing) == 8
    assert len(state.rolling_tail) == 7
    assert len(state.hashes) > len(state.trailing)
    assert sum(state.counts.values()) == len(state.hashes)
