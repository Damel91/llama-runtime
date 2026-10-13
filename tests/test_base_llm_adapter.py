from typing import Dict, Iterator, List

from runtime_llama import BaseLLMAdapter


class _DummyAdapter(BaseLLMAdapter):
    @property
    def role_id(self) -> str:
        return "test_role"

    @property
    def model_name(self) -> str:
        return "test_model"

    def invoke(self, messages: List) -> str:
        return "ok"

    def stream(self, messages: List) -> Iterator[str]:
        yield "ok"

    def is_ready(self) -> bool:
        return True

    def health_status(self) -> Dict:
        return {"healthy": True}


def test_strip_thinking_text_removes_closed_and_unclosed_blocks():
    adapter = _DummyAdapter()

    assert adapter._strip_thinking_text("a<think>secret</think>b") == "ab"
    assert adapter._strip_thinking_text("a<think>secret") == "a"


def test_stream_thinking_stripper_handles_cross_token_boundaries():
    adapter = _DummyAdapter()
    stripper = adapter._new_thinking_stream_stripper()
    visible = []

    for token in ["he", "<thi", "nk>secret", "</thi", "nk>llo"]:
        visible.extend(stripper.feed(token))
    visible.extend(stripper.flush())

    assert "".join(visible) == "hello"


def test_stream_thinking_stripper_drops_unclosed_thinking_tail():
    adapter = _DummyAdapter()
    stripper = adapter._new_thinking_stream_stripper()
    visible = []

    for token in ["visible", "<think>", "hidden"]:
        visible.extend(stripper.feed(token))
    visible.extend(stripper.flush())

    assert "".join(visible) == "visible"


def test_extract_content_and_reasoning_handles_dict_payload():
    content, reasoning = _DummyAdapter._extract_content_and_reasoning(
        {"content": "visible", "reasoning_content": "thinking"}
    )

    assert content == "visible"
    assert reasoning == "thinking"


def test_extract_content_and_reasoning_handles_object_payload():
    from types import SimpleNamespace

    payload = SimpleNamespace(content="visible", reasoning_content="thinking")
    content, reasoning = _DummyAdapter._extract_content_and_reasoning(payload)

    assert content == "visible"
    assert reasoning == "thinking"


def test_extract_content_and_reasoning_handles_none_and_missing_fields():
    assert _DummyAdapter._extract_content_and_reasoning(None) == ("", "")
    assert _DummyAdapter._extract_content_and_reasoning({}) == ("", "")
    assert _DummyAdapter._extract_content_and_reasoning(
        {"content": None, "reasoning_content": None}
    ) == ("", "")
