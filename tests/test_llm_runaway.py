from types import SimpleNamespace

import pytest

from runtime_llama import BaseLLMAdapter, LLMRunawayError, LLMStreamEvent
from runtime_llama.runaway import RunawayDetector


class _EventAdapter(BaseLLMAdapter):
    def __init__(self, events, profile=None):
        self._events = events
        self._role_profile = profile

    @property
    def role_id(self):
        return "diagnostic_review"

    @property
    def model_name(self):
        return "fake-local"

    @property
    def role_profile(self):
        return self._role_profile

    def _iter_stream_events_with_ctx(self, messages, **kwargs):
        del messages, kwargs
        yield from self._events

    def is_ready(self):
        return True

    def health_status(self):
        return {"healthy": True}


def _profile(**overrides):
    values = {
        "runaway_min_chars": 8,
        "runaway_window_chars": 64,
        "runaway_block_chars": 4,
        "runaway_repeat_threshold": 3,
        "runaway_ngram_chars": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_runaway_detector_trips_on_repeated_reasoning_without_text_leakage():
    adapter = _EventAdapter(
        [
            LLMStreamEvent(kind="reasoning", text="plan"),
            LLMStreamEvent(kind="reasoning", text="plan"),
            LLMStreamEvent(kind="reasoning", text="plan"),
        ],
        _profile(),
    )

    with pytest.raises(LLMRunawayError) as caught:
        adapter.invoke_with_ctx([{"role": "user", "content": "review"}])

    assert adapter.get_last_terminal_reason() == "runaway"
    diagnostics = caught.value.diagnostics
    assert diagnostics["channel"] == "reasoning"
    assert diagnostics["repeat_count"] >= 3
    assert "plan" not in str(diagnostics)


def test_runaway_detector_trips_on_misaligned_paragraph_across_irregular_deltas():
    detector = RunawayDetector(
        min_chars=96,
        window_chars=512,
        block_chars=32,
        repeat_threshold=3,
        ngram_chars=12,
    )
    paragraph = (
        "Recheck process ownership before cleanup, preserve the active child, "
        "and then reconsider the same transition boundary. "
    )
    # The recurrence period is deliberately not divisible by block_chars and
    # transport boundaries move between repetitions.
    stream = paragraph * 3
    sizes = [7, 19, 5, 31, 11, 23]
    offset = 0
    diagnostics = None
    index = 0
    while offset < len(stream) and diagnostics is None:
        size = sizes[index % len(sizes)]
        diagnostics = detector.observe("reasoning", stream[offset : offset + size])
        offset += size
        index += 1

    assert diagnostics is not None
    assert diagnostics.channel == "reasoning"
    assert diagnostics.detector == "rolling_repeated_block"
    assert diagnostics.repeat_count >= 3
    assert diagnostics.char_offset < len(stream)


def test_runaway_normalization_is_invariant_to_whitespace_delta_boundaries():
    settings = dict(
        min_chars=48,
        window_chars=256,
        block_chars=16,
        repeat_threshold=3,
        ngram_chars=8,
    )
    contiguous = RunawayDetector(**settings)
    segmented = RunawayDetector(**settings)
    text = "inspect the same lifecycle boundary and preserve ownership " * 3

    contiguous_result = contiguous.observe("reasoning", text)
    segmented_result = None
    for delta in [
        "inspect the same ",
        "lifecycle\n",
        "boundary and preserve ownership inspect",
        " the same lifecycle boundary ",
        "and preserve ownership inspect the same lifecycle boundary",
        " and preserve ownership ",
    ]:
        segmented_result = segmented.observe("reasoning", delta)
        if segmented_result is not None:
            break

    assert contiguous_result is not None
    assert segmented_result is not None
    assert contiguous_result.detector == segmented_result.detector


def test_runaway_detector_keeps_rolling_state_bounded_after_long_stream():
    detector = RunawayDetector(
        min_chars=32,
        window_chars=256,
        block_chars=16,
        repeat_threshold=256,
        ngram_chars=1,
    )

    for index in range(2_000):
        assert detector.observe("reasoning", f"segment-{index:05d}-payload ") is None

    state = detector._channels["reasoning"]
    assert len(state.trailing) <= 256
    assert len(state.rolling_tail) < 16
    assert len(state.hashes) <= 256
    assert sum(state.counts.values()) == len(state.hashes)


def test_runaway_detector_does_not_trip_on_novel_stream_and_preserves_length_cap():
    adapter = _EventAdapter(
        [
            LLMStreamEvent(kind="content", text="alpha "),
            LLMStreamEvent(kind="content", text="bravo "),
            LLMStreamEvent(kind="content", text="charlie "),
            LLMStreamEvent(kind="content", text="delta"),
            LLMStreamEvent(kind="terminal", terminal_reason="length_cap"),
        ],
        _profile(runaway_repeat_threshold=4),
    )

    result = adapter.invoke_with_ctx([{"role": "user", "content": "review"}])

    assert result.text == "alpha bravo charlie delta"
    assert result.terminal_reason == "length_cap"
    assert adapter.get_last_terminal_reason() == "length_cap"


def test_detector_rejects_invalid_bounded_configuration():
    with pytest.raises(ValueError, match="too small"):
        RunawayDetector(
            min_chars=1,
            window_chars=8,
            block_chars=8,
            repeat_threshold=4,
            ngram_chars=4,
        )


def test_runaway_closes_event_generator_before_returning_error():
    closed = {"value": False}

    def events():
        try:
            yield LLMStreamEvent(kind="content", text="loop")
            yield LLMStreamEvent(kind="content", text="loop")
            yield LLMStreamEvent(kind="content", text="loop")
        finally:
            closed["value"] = True

    adapter = _EventAdapter(events(), _profile())

    with pytest.raises(LLMRunawayError):
        adapter.invoke([{"role": "user", "content": "review"}])

    assert closed["value"] is True
