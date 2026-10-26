"""Completed exact-token reuse has bounded storage and no owner lifecycle."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from threading import Barrier, Event

import pytest

import runtime_llama.embedding_preparation_cache as module
from runtime_llama.embedding_preparation_cache import PreparationCache, PreparationKey


def key(text="input", **changes):
    values = {
        "text": text,
        "input_type": "document",
        "policy": "add_special=true;parse_special=true;prefix=doc:",
        "content_sha256": "a" * 64,
        "embedding_sha256": "b" * 64,
        "tokenizer_build": "native-build-1",
        "vocab_size": 32000,
    }
    values.update(changes)
    return PreparationKey(**values)


def entry_cost(cache_key, tokens):
    cache = PreparationCache(1, 1024 * 1024)
    assert cache.put(cache_key, tokens)
    return cache.snapshot()["bytes"]


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("text", "another input"),
        ("input_type", "query"),
        ("policy", "add_special=false;parse_special=true;prefix=doc:"),
        ("content_sha256", "c" * 64),
        ("embedding_sha256", "d" * 64),
        ("tokenizer_build", "native-build-2"),
        ("vocab_size", 64000),
        ("vocab_size", None),
    ],
)
def test_every_tokenization_identity_field_is_required_for_a_hit(field, changed):
    cache = PreparationCache(8, 1024 * 1024)
    original = key()
    assert cache.put(original, (1, 7, 2))
    assert cache.get(replace(original, **{field: changed})) is None
    assert cache.get(key()) == (1, 7, 2)
    assert cache.snapshot()["hits"] == cache.snapshot()["misses"] == 1


def test_key_does_not_include_alias_context_or_admission_permission():
    assert tuple(PreparationKey.__dataclass_fields__) == (
        "text",
        "input_type",
        "policy",
        "content_sha256",
        "embedding_sha256",
        "tokenizer_build",
        "vocab_size",
    )
    original = key()
    with pytest.raises(FrozenInstanceError):
        original.text = "changed"
    with pytest.raises((AttributeError, TypeError)):
        original.context_length = 2048


def test_full_unicode_text_comparison_survives_hash_collisions(monkeypatch):
    monkeypatch.setattr(PreparationKey, "__hash__", lambda self: 1)
    cache = PreparationCache(8, 1024 * 1024)
    texts = ["é", "e\u0301", "東京🙂", "東京🙃", "", "raw\x00value"]
    for index, text in enumerate(texts):
        assert cache.put(key(text), (index,))
    for index, text in enumerate(texts):
        assert cache.get(key(text)) == (index,)
    assert cache.get(key("not present")) is None
    assert cache.snapshot()["entries"] == len(texts)


def test_empty_token_tuple_is_a_hit_and_valid_values_remain_immutable():
    cache = PreparationCache(4, 1024 * 1024)
    assert cache.put(key(""), ())
    assert cache.get(key("")) == ()
    tokens = (0, 1, 2**31 - 1)
    assert cache.put(key(), tokens)
    assert cache.get(key()) is tokens
    assert cache.snapshot()["hits"] == 2


def test_get_refreshes_lru_and_replacement_updates_cost_without_duplicate_entries():
    cache = PreparationCache(2, 1024 * 1024)
    assert cache.put(key("a"), (1,))
    assert cache.put(key("b"), (2,))
    assert cache.get(key("a")) == (1,)
    assert cache.put(key("c"), (3,))
    assert cache.get(key("b")) is None
    assert cache.get(key("a")) == (1,)
    assert cache.put(key("a"), (1, 2, 3))
    assert cache.snapshot()["bytes"] == entry_cost(key("a"), (1, 2, 3)) + entry_cost(
        key("c"), (3,)
    )
    view = cache.snapshot()
    assert view["entries"] == view["peak_entries"] == 2
    assert view["puts"] == 4
    assert view["evictions"] == 1


def test_byte_budget_evicts_lru_even_when_entry_count_has_capacity():
    cost = entry_cost(key("a"), (1, 2))
    cache = PreparationCache(100, cost * 2)
    for text in ("a", "b", "c"):
        assert cache.put(key(text), (1, 2))
        assert cache.snapshot()["bytes"] <= cost * 2
    assert cache.get(key("a")) is None
    assert cache.get(key("b")) == cache.get(key("c")) == (1, 2)
    view = cache.snapshot()
    assert view["entries"] == 2
    assert view["bytes"] == view["peak_bytes"] == cost * 2
    assert view["evictions"] == 1


def test_oversized_text_or_tokens_skip_retention_without_evicting_useful_entries():
    cost = entry_cost(key(), (1,))
    cache = PreparationCache(4, cost)
    assert cache.put(key(), (1,))
    assert not cache.put(key("🙂" * 10000), (2,))
    assert not cache.put(key("tokens"), tuple(range(10000)))
    assert cache.get(key()) == (1,)
    view = cache.snapshot()
    assert view["bytes"] == cost
    assert view["entries"] == 1
    assert view["oversized_skips"] == 2
    assert view["evictions"] == 0


@pytest.mark.parametrize(("entries", "budget"), [(0, 1024), (8, 0), (0, 0)])
def test_zero_budget_disables_retention(entries, budget):
    cache = PreparationCache(entries, budget)
    assert not cache.put(key(), (1,))
    assert cache.get(key()) is None
    assert cache.snapshot()["entries"] == cache.snapshot()["bytes"] == 0


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "10"])
@pytest.mark.parametrize("field", ["max_entries", "max_bytes"])
def test_invalid_budgets_are_rejected(field, invalid):
    values = {"max_entries": 10, "max_bytes": 1024}
    values[field] = invalid
    with pytest.raises(ValueError):
        PreparationCache(**values)


@pytest.mark.parametrize("tokens", [[1], (True,), (-1,), (2**31,), (1.0,), ("1",)])
def test_mutable_or_invalid_token_values_never_enter_the_cache(tokens):
    cache = PreparationCache(4, 1024 * 1024)
    with pytest.raises(ValueError):
        cache.put(key(), tokens)
    assert cache.snapshot()["entries"] == cache.snapshot()["bytes"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"text": None},
        {"input_type": "other"},
        {"policy": ""},
        {"content_sha256": "path/to/model.gguf"},
        {"embedding_sha256": "b" * 63},
        {"tokenizer_build": ""},
        {"vocab_size": 0},
        {"vocab_size": True},
    ],
)
def test_ineligible_key_metadata_is_rejected(change):
    with pytest.raises(ValueError):
        key(**change)


def test_key_and_value_validation_and_size_estimation_happen_outside_cache_lock(
    monkeypatch,
):
    cache = PreparationCache(4, 1024 * 1024)
    validate_key, estimate = module._key, module._entry_bytes
    observations = []

    def outside_key(value):
        assert not cache._lock.locked()
        observations.append("key")
        return validate_key(value)

    def outside_cost(cache_key, tokens):
        assert not cache._lock.locked()
        observations.append("cost")
        return estimate(cache_key, tokens)

    monkeypatch.setattr(module, "_key", outside_key)
    monkeypatch.setattr(module, "_entry_bytes", outside_cost)
    assert cache.put(key(), (1, 2))
    assert cache.get(key()) == (1, 2)
    assert observations == ["key", "cost", "key"]


def test_concurrent_get_put_keeps_both_storage_limits_and_complete_values():
    workers = 8
    barrier = Barrier(workers)
    budget = entry_cost(key("00"), (1, 2)) * 8
    cache = PreparationCache(5, budget)

    def work(worker):
        barrier.wait(3)
        for index in range(200):
            text = f"{(index + worker) % 32:02d}"
            expected = (int(text), int(text) + 1)
            assert cache.put(key(text), expected)
            assert cache.get(key(text)) in (None, expected)
            view = cache.snapshot()
            assert 0 <= view["entries"] <= 5
            assert 0 <= view["bytes"] <= budget

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(work, range(workers)))
    view = cache.snapshot()
    assert view["puts"] == workers * 200
    assert view["hits"] + view["misses"] == workers * 200
    assert view["peak_entries"] <= 5
    assert view["peak_bytes"] <= budget
    assert view["evictions"] > 0


def test_close_clears_storage_and_rejects_late_worker_publication():
    cache = PreparationCache(4, 1024 * 1024)
    assert cache.put(key(), (1, 2))
    started, release = Event(), Event()

    def late_worker():
        started.set()
        assert release.wait(3)
        return cache.put(key("late"), (3, 4))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(late_worker)
        assert started.wait(3)
        cache.close()
        release.set()
        assert future.result(3) is False
    cache.close()
    assert cache.get(key()) is None
    view = cache.snapshot()
    assert view["closed"]
    assert view["entries"] == view["bytes"] == 0
    assert view["puts"] == 1
    assert view["peak_entries"] == 1
    assert view["peak_bytes"] > 0


def test_snapshot_is_detached_and_has_only_bounded_storage_counters():
    cache = PreparationCache(2, 8192)
    assert set(cache.snapshot()) == {
        "enabled",
        "entries",
        "bytes",
        "peak_entries",
        "peak_bytes",
        "max_entries",
        "max_bytes",
        "hits",
        "misses",
        "puts",
        "evictions",
        "oversized_skips",
        "closed",
    }
    changed = cache.snapshot()
    changed["entries"] = 100
    assert cache.snapshot()["entries"] == 0
