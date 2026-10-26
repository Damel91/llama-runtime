"""Statistical character budgets remain distinct from exact inference limits."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
from threading import Event, Lock

import pytest

import runtime_llama.embedding_calibration as module
from runtime_llama.embedding_calibration import (
    CALIBRATION_CATEGORIES,
    CATEGORY_SAFETY_FACTOR,
    MIXED_SAFETY_FACTOR,
    CalibrationProfile,
    CalibrationStats,
    load_existing_calibration,
    load_or_calibrate,
)
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError


def binding(**changes):
    result = {
        "content_sha256": "a" * 64,
        "embedding_sha256": "b" * 64,
        "tokenizer_build": "native-test-1",
        "vocab_size": 32000,
        "input_type": "document",
        "policy": "prefix=search_document: ;add_special=true;parse_special=true",
    }
    result.update(changes)
    return result


def check():
    pass


@pytest.fixture
def small_corpus(monkeypatch):
    records = tuple(
        module._CorpusRecord(
            category, f"{category} {number}: café 東京🙂 " * (number + 1)
        )
        for category in CALIBRATION_CATEGORIES
        for number in range(3)
    )
    manifest = {
        "sample_count": len(records),
        "corpus_sha256": "c" * 64,
        "categories": {
            category: {
                "count": 3,
                "total_characters": sum(
                    len(record.text)
                    for record in records
                    if record.category == category
                ),
            }
            for category in CALIBRATION_CATEGORIES
        },
    }

    def load(check):
        check()
        return manifest, records

    monkeypatch.setattr(module, "_load_corpus", load)
    monkeypatch.setattr(module, "_load_manifest", lambda check: (None, manifest))
    return manifest, records


def fake_tokens(text):
    # Prefix and native special-token overhead are already in the denominator.
    return math.ceil(len(text) / 4) + 7


def test_packaged_corpus_is_complete_unique_varied_and_reproducible(tmp_path):
    manifest, records = module._load_corpus(check)
    assert len(records) == len({record.text for record in records}) == 2500
    assert {record.category for record in records} == set(CALIBRATION_CATEGORIES)
    assert all(
        sum(record.category == category for record in records) == 500
        for category in CALIBRATION_CATEGORIES
    )
    for category in CALIBRATION_CATEGORIES:
        lengths = [
            len(record.text) for record in records if record.category == category
        ]
        assert max(lengths) > min(lengths) * 2
    assert any("\n" in record.text for record in records)
    assert any("東京" in record.text or "图书馆" in record.text for record in records)
    assert any(
        len(record.text.encode("utf-8")) > len(record.text) for record in records
    )
    path = Path(__file__).resolve().parents[1] / "tools/generate_calibration_corpus.py"
    specification = importlib.util.spec_from_file_location("corpus_generator", path)
    generator = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(generator)
    regenerated = generator.generate(tmp_path)
    assert regenerated == manifest
    shipped = Path(module.__file__).with_name("calibration_corpus")
    for name in [
        "manifest.json",
        *(category + ".jsonl" for category in CALIBRATION_CATEGORIES),
    ]:
        assert (tmp_path / name).read_bytes() == (shipped / name).read_bytes()


def _generator():
    path = Path(__file__).resolve().parents[1] / "tools/generate_calibration_corpus.py"
    specification = importlib.util.spec_from_file_location(
        "hard_corpus_generator", path
    )
    generator = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(generator)
    return generator


def test_selected_corpus_is_exactly_the_worst_five_hundred_in_every_category():
    generator = _generator()
    receipt_bytes = generator.SELECTION_RECEIPT.read_bytes()
    receipt = json.loads(receipt_bytes)
    originals = generator.original_records()
    selected = generator.select_hardest(originals, receipt)
    measurements = {row["id"]: row for row in receipt["measurements"]}
    assert len(measurements) == 10000
    assert sum(len(rows) for rows in selected.values()) == 2500
    for category in CALIBRATION_CATEGORIES:
        ranked = sorted(
            (row for row in measurements.values() if row["category"] == category),
            key=lambda row: (
                Fraction(row["source_characters"], row["tokens"]),
                row["id"],
            ),
        )
        assert receipt["selected"][category] == [row["id"] for row in ranked[:500]]
        assert [record["text"] for record in selected[category]] == [
            originals[category][int(row["id"].split("-")[1]) - 1]["text"]
            for row in ranked[:500]
        ]
    manifest, _ = module._load_corpus(check)
    assert manifest["schema"] == "runtime-llama-calibration-corpus-v2"
    assert manifest["corpus_sha256"] != generator.SOURCE_CORPUS_SHA256
    assert (
        manifest["selection"]["receipt_sha256"]
        == hashlib.sha256(receipt_bytes).hexdigest()
    )
    assert manifest["selection"]["measurement_binding"] == receipt["binding"]


@pytest.mark.parametrize(
    "damage",
    [
        lambda receipt: receipt["measurements"].pop(),
        lambda receipt: receipt["measurements"][0].update(tokens=True),
        lambda receipt: receipt["measurements"][0].update(text_sha256="a" * 64),
        lambda receipt: receipt["selected"]["code"].reverse(),
        lambda receipt: receipt.update(source_corpus_sha256="b" * 64),
    ],
)
def test_offline_hard_selection_rejects_missing_counts_tampering_and_wrong_ranks(
    damage,
):
    generator = _generator()
    receipt = json.loads(generator.SELECTION_RECEIPT.read_bytes())
    damage(receipt)
    with pytest.raises(ValueError):
        generator.select_hardest(generator.original_records(), receipt)


def test_full_first_use_calls_exactly_twenty_five_hundred_then_warm_start_calls_none(
    tmp_path,
):
    count, lock = 0, Lock()

    def tokenize(text):
        nonlocal count
        with lock:
            count += 1
        return fake_tokens(text)

    first = load_or_calibrate(tmp_path, binding(), tokenize, check)
    assert count == first.stats_for().sample_count == 2500
    second = load_or_calibrate(
        tmp_path, binding(), lambda text: pytest.fail("warm cache tokenized"), check
    )
    assert first == second
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_source_characters_and_full_native_token_count_are_not_double_adjusted(
    tmp_path, small_corpus
):
    manifest, records = small_corpus
    seen = []

    def tokenize(text):
        seen.append(text)
        return fake_tokens(text)

    profile = load_or_calibrate(tmp_path, binding(), tokenize, check)
    assert sorted(seen) == sorted(record.text for record in records)
    total_chars = sum(len(record.text) for record in records)
    total_tokens = sum(fake_tokens(record.text) for record in records)
    weighted = total_chars / total_tokens
    mixed = profile.stats_for("mixed")
    assert mixed.total_characters == total_chars
    assert mixed.total_tokens == total_tokens
    assert mixed.chars_per_token == weighted
    assert mixed.arithmetic_mean_chars_per_token == pytest.approx(
        sum(len(record.text) / fake_tokens(record.text) for record in records)
        / len(records)
    )
    assert profile.safe_chars_per_token() == weighted * 0.90
    assert profile.max_input_chars(512) == math.floor(512 * weighted * 0.90)
    assert CATEGORY_SAFETY_FACTOR == 0.80
    assert MIXED_SAFETY_FACTOR == 0.90
    assert profile.corpus_sha256 == manifest["corpus_sha256"]
    for category in CALIBRATION_CATEGORIES:
        selected = [record for record in records if record.category == category]
        statistics = profile.stats_for(category)
        assert statistics.sample_count == 3
        assert statistics.chars_per_token == sum(
            len(row.text) for row in selected
        ) / sum(fake_tokens(row.text) for row in selected)
        assert (
            profile.safe_chars_per_token(category) == statistics.chars_per_token * 0.80
        )
        assert profile.max_input_chars(512, category) == math.floor(
            512 * statistics.chars_per_token * 0.80
        )
    assert mixed.chars_per_token != pytest.approx(mixed.arithmetic_mean_chars_per_token)


def test_profile_statistics_and_nested_binding_are_immutable(tmp_path, small_corpus):
    original = binding(metadata={"flags": [True, "value"]})
    profile = load_or_calibrate(tmp_path, original, fake_tokens, check)
    original["metadata"]["flags"].append("changed")
    assert profile.binding["metadata"]["flags"] == (True, "value")
    with pytest.raises(TypeError):
        profile.binding["content_sha256"] = "changed"
    with pytest.raises(TypeError):
        profile.statistics["mixed"] = profile.stats_for()
    with pytest.raises(FrozenInstanceError):
        profile.stats_for().total_tokens = 0


@pytest.mark.parametrize("category", ["unknown", None, True, 1, []])
def test_unknown_category_rejected(tmp_path, small_corpus, category):
    profile = load_or_calibrate(tmp_path, binding(), fake_tokens, check)
    with pytest.raises(ValueError):
        profile.stats_for(category)


@pytest.mark.parametrize("context", [0, -1, True, None, 1.5, "512"])
def test_invalid_context_budget_rejected(tmp_path, small_corpus, context):
    profile = load_or_calibrate(tmp_path, binding(), fake_tokens, check)
    with pytest.raises(ValueError):
        profile.max_input_chars(context)


@pytest.mark.parametrize(
    "change",
    [
        {"content_sha256": "d" * 64},
        {"embedding_sha256": "e" * 64},
        {"tokenizer_build": "native-test-2"},
        {"vocab_size": 64000},
        {"input_type": "query"},
        {"policy": "prefix=search_query: ;add_special=true;parse_special=true"},
    ],
)
def test_every_tokenizer_identity_field_invalidates_cache(
    tmp_path, small_corpus, change
):
    first = load_or_calibrate(tmp_path, binding(), fake_tokens, check)
    calls = []
    second = load_or_calibrate(
        tmp_path,
        binding(**change),
        lambda text: calls.append(text) or fake_tokens(text),
        check,
    )
    assert len(calls) == 15
    assert first.profile_id != second.profile_id


def test_context_changes_recompute_budget_without_recalibrating(tmp_path, small_corpus):
    profile = load_or_calibrate(tmp_path, binding(), fake_tokens, check)
    warm = load_or_calibrate(
        tmp_path, binding(), lambda text: pytest.fail("tokenized"), check
    )
    assert warm.max_input_chars(8192) > profile.max_input_chars(512) * 15


def test_safety_policy_changes_reuse_raw_cached_measurements(
    tmp_path, small_corpus, monkeypatch
):
    first = load_or_calibrate(tmp_path, binding(), fake_tokens, check)
    cached_bytes = (tmp_path / (first.profile_id + ".json")).read_bytes()
    first_mixed_budget = first.max_input_chars(512)
    first_code_budget = first.max_input_chars(512, "code")
    monkeypatch.setattr(module, "MIXED_SAFETY_FACTOR", 0.95)
    monkeypatch.setattr(module, "CATEGORY_SAFETY_FACTOR", 0.70)
    warm = load_or_calibrate(
        tmp_path, binding(), lambda text: pytest.fail("policy change tokenized"), check
    )
    assert warm == first
    assert warm.max_input_chars(512) > first_mixed_budget
    assert warm.max_input_chars(512, "code") < first_code_budget
    assert warm.safe_chars_per_token() == warm.stats_for().chars_per_token * 0.95
    assert (
        warm.safe_chars_per_token("code")
        == warm.stats_for("code").chars_per_token * 0.70
    )
    assert (tmp_path / (warm.profile_id + ".json")).read_bytes() == cached_bytes
    assert b"SAFETY" not in cached_bytes and b"safety_factor" not in cached_bytes


def test_corpus_change_invalidates_all_category_statistics(tmp_path, small_corpus):
    first = load_or_calibrate(tmp_path, binding(), fake_tokens, check)
    small_corpus[0]["corpus_sha256"] = "f" * 64
    second = load_or_calibrate(tmp_path, binding(), fake_tokens, check)
    assert first.profile_id != second.profile_id


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None, "7"])
def test_bad_native_counts_never_persist(tmp_path, small_corpus, value):
    with pytest.raises(LLMBackendError, match="invalid_token_count"):
        load_or_calibrate(tmp_path, binding(), lambda text: value, check)
    assert not list(tmp_path.iterdir())


def test_cancellation_drains_workers_and_never_publishes_partial_cache(
    tmp_path, small_corpus
):
    cancelled, entered, release = Event(), Event(), Event()
    active, lock = 0, Lock()

    def bounded_check():
        if cancelled.is_set():
            raise LLMCancelledError()

    def tokenize(text):
        nonlocal active
        with lock:
            active += 1
        entered.set()
        release.wait(2)
        with lock:
            active -= 1
        return fake_tokens(text)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            load_or_calibrate, tmp_path, binding(), tokenize, bounded_check
        )
        assert entered.wait(2)
        cancelled.set()
        release.set()
        with pytest.raises(LLMCancelledError):
            future.result(2)
    assert active == 0
    assert not list(tmp_path.iterdir())


def test_pending_tokenizations_never_exceed_worker_budget(tmp_path, small_corpus):
    lock = Lock()
    active, peak = 0, 0
    barrier = Event()

    def tokenize(text):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 3:
                barrier.set()
        assert barrier.wait(2)
        with lock:
            active -= 1
        return fake_tokens(text)

    load_or_calibrate(tmp_path, binding(), tokenize, check, workers=3)
    assert peak == 3
    assert active == 0


def test_same_process_same_key_shares_first_calibration(tmp_path, small_corpus):
    started, release = Event(), Event()
    calls, lock = [], Lock()

    def tokenize(text):
        with lock:
            calls.append(text)
        started.set()
        assert release.wait(2)
        return fake_tokens(text)

    with ThreadPoolExecutor(max_workers=3) as executor:
        leader = executor.submit(
            load_or_calibrate, tmp_path, binding(), tokenize, check
        )
        assert started.wait(2)
        waiter = executor.submit(
            load_or_calibrate, tmp_path, binding(), tokenize, check
        )
        release.set()
        first, second = leader.result(3), waiter.result(3)
    assert first == second
    assert len(calls) == len(set(calls)) == 15


def test_waiter_cancellation_does_not_stop_leader(tmp_path, small_corpus):
    started, release, waiter_started, cancelled = Event(), Event(), Event(), Event()

    def tokenize(text):
        started.set()
        assert release.wait(2)
        return fake_tokens(text)

    def waiter_check():
        waiter_started.set()
        if cancelled.is_set():
            raise LLMCancelledError()

    with ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(
            load_or_calibrate, tmp_path, binding(), tokenize, check
        )
        assert started.wait(2)
        waiter = executor.submit(
            load_or_calibrate, tmp_path, binding(), tokenize, waiter_check
        )
        assert waiter_started.wait(2)
        cancelled.set()
        with pytest.raises(LLMCancelledError):
            waiter.result(2)
        release.set()
        assert leader.result(3).stats_for().sample_count == 15


def test_final_identity_validation_happens_after_counts_before_cache(
    tmp_path, small_corpus
):
    calls = []

    def tokenize(text):
        calls.append(text)
        return fake_tokens(text)

    def validate():
        assert len(calls) == 15
        assert not list(tmp_path.iterdir())
        raise LLMBackendError("model changed")

    with pytest.raises(LLMBackendError, match="model changed"):
        load_or_calibrate(tmp_path, binding(), tokenize, check, validate=validate)
    assert not list(tmp_path.iterdir())
    assert not module._FLIGHTS
    profile = load_or_calibrate(tmp_path, binding(), tokenize, check)
    assert profile.stats_for().sample_count == 15


def test_atomic_cache_write_is_private_and_contains_only_statistics(
    tmp_path, small_corpus
):
    profile = load_or_calibrate(tmp_path, binding(), fake_tokens, check)
    path = tmp_path / (profile.profile_id + ".json")
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    payload = path.read_text()
    assert "café" not in payload and "token_ids" not in payload
    assert set(json.loads(payload)) == {
        "schema",
        "profile_id",
        "binding",
        "corpus_sha256",
        "statistics",
    }
    assert not list(tmp_path.glob(".calibration-*"))


def test_atomic_replace_failure_removes_temporary_file(
    tmp_path, small_corpus, monkeypatch
):
    def fail(*args):
        raise OSError("filesystem unavailable")

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(LLMBackendError, match="cache_write_failed"):
        load_or_calibrate(tmp_path, binding(), fake_tokens, check)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "damage",
    [
        lambda value: value.update(schema="unknown"),
        lambda value: value.update(binding={"different": "binding"}),
        lambda value: value["statistics"]["mixed"].update(total_tokens=0),
        lambda value: value["statistics"]["mixed"].update(chars_per_token=float("nan")),
        lambda value: value["statistics"]["code"].update(sample_count=True),
        lambda value: value["statistics"]["code"].update(total_characters=10),
        lambda value: value["statistics"]["mixed"].update(total_tokens=999999),
        lambda value: value["statistics"].pop("code"),
    ],
)
def test_corrupted_or_inconsistent_cache_is_recalibrated(
    tmp_path, small_corpus, damage
):
    original = load_or_calibrate(tmp_path, binding(), fake_tokens, check)
    path = tmp_path / (original.profile_id + ".json")
    payload = json.loads(path.read_text())
    damage(payload)
    path.write_text(json.dumps(payload))
    calls = []
    result = load_or_calibrate(
        tmp_path, binding(), lambda text: calls.append(text) or fake_tokens(text), check
    )
    assert result == original
    assert len(calls) == 15


@pytest.mark.parametrize("payload", [b"{", b"[1,2]", b"x" * (128 * 1024 + 1)])
def test_invalid_or_oversized_cache_is_recalibrated(tmp_path, small_corpus, payload):
    original = load_or_calibrate(tmp_path, binding(), fake_tokens, check)
    path = tmp_path / (original.profile_id + ".json")
    path.write_bytes(payload)
    calls = []
    assert (
        load_or_calibrate(
            tmp_path,
            binding(),
            lambda text: calls.append(text) or fake_tokens(text),
            check,
        )
        == original
    )
    assert len(calls) == 15


@pytest.mark.parametrize("workers", [0, -1, True, None, 1.5])
def test_invalid_worker_budget_rejected(tmp_path, small_corpus, workers):
    with pytest.raises(ValueError):
        load_or_calibrate(tmp_path, binding(), fake_tokens, check, workers=workers)


def test_explicit_worker_count_above_old_ceiling_is_supported(tmp_path, small_corpus):
    profile = load_or_calibrate(tmp_path, binding(), fake_tokens, check, workers=33)
    assert profile.stats_for("mixed").sample_count == 15


def test_binding_and_cache_above_old_byte_ceilings_reuse_completed_calibration(
    tmp_path, small_corpus
):
    selected = {**binding(), "prefix": "x" * 150000}
    calls = []

    def tokenize(text):
        calls.append(text)
        return fake_tokens(text)

    first = load_or_calibrate(tmp_path, selected, tokenize, check)
    assert len(calls) == 15
    assert load_or_calibrate(tmp_path, selected, tokenize, check) == first
    assert len(calls) == 15
    assert first.max_input_chars(2**31) > 0


def test_native_positive_token_count_has_no_local_integer_ceiling(
    tmp_path, small_corpus
):
    profile = load_or_calibrate(tmp_path, binding(), lambda _text: 2**31, check)
    assert profile.stats_for("mixed").total_tokens == 15 * 2**31


def test_calibration_binding_rejects_cycles_instead_of_fixed_depth():
    nested = "value"
    for _ in range(20):
        nested = {"nested": nested}
    assert module._binding_payload({"values": nested}) == {"values": nested}
    circular = {}
    circular["self"] = circular
    with pytest.raises(ValueError, match="cycle"):
        module._binding_payload(circular)


@pytest.mark.parametrize(
    "invalid", [{}, {1: "value"}, {"value": object()}, {"value": float("nan")}]
)
def test_invalid_binding_rejected_before_tokenizing(tmp_path, small_corpus, invalid):
    with pytest.raises(ValueError):
        load_or_calibrate(
            tmp_path, invalid, lambda text: pytest.fail("tokenized"), check
        )


def test_binding_fingerprint_is_canonical_and_contains_corpus_digest(
    tmp_path, small_corpus
):
    profile = load_or_calibrate(tmp_path, binding(), fake_tokens, check)
    expected = hashlib.sha256(
        module._canonical(
            {
                "schema": module._PROFILE_SCHEMA,
                "corpus_sha256": small_corpus[0]["corpus_sha256"],
                "binding": binding(),
            }
        )
    ).hexdigest()
    assert profile.profile_id == expected
    reverse = dict(reversed(list(binding().items())))
    assert (
        load_or_calibrate(
            tmp_path, reverse, lambda text: pytest.fail("tokenized"), check
        )
        == profile
    )


@pytest.mark.parametrize("target", ["manifest.json", "code.jsonl"])
def test_modified_shipped_corpus_is_rejected_before_remote_calls(
    tmp_path, monkeypatch, target
):
    original = module._read_bounded

    def tamper(resource, limit=None):
        payload = original(resource, limit)
        if resource.name == target:
            if target == "manifest.json":
                manifest = json.loads(payload)
                manifest["categories"]["code"]["file"] = "../outside.jsonl"
                return json.dumps(manifest).encode()
            return b"!" + payload[1:]
        return payload

    monkeypatch.setattr(module, "_read_bounded", tamper)
    with pytest.raises(LLMBackendError, match="invalid_corpus"):
        load_or_calibrate(
            tmp_path,
            binding(),
            lambda text: pytest.fail("tokenized corrupt corpus"),
            check,
        )
    assert not list(tmp_path.iterdir())


def test_cached_binding_checks_json_types_not_python_boolean_integer_equality(
    tmp_path, small_corpus
):
    original = load_or_calibrate(tmp_path, binding(vocab_size=1), fake_tokens, check)
    path = tmp_path / (original.profile_id + ".json")
    payload = json.loads(path.read_text())
    payload["binding"]["vocab_size"] = True
    path.write_text(json.dumps(payload))
    calls = []
    assert (
        load_or_calibrate(
            tmp_path,
            binding(vocab_size=1),
            lambda text: calls.append(text) or fake_tokens(text),
            check,
        )
        == original
    )
    assert len(calls) == 15


@pytest.mark.parametrize("nested", [False, True])
def test_existing_file_cache_parent_fails_before_any_tokenization(
    tmp_path, small_corpus, nested
):
    parent = tmp_path / "ordinary-file"
    parent.write_text("preserve this file")
    cache = parent / "nested-cache" if nested else parent
    calls = []
    with pytest.raises(LLMBackendError, match="cache_write_failed"):
        load_or_calibrate(
            cache,
            binding(),
            lambda text: calls.append(text) or fake_tokens(text),
            check,
        )
    assert not calls
    assert parent.read_text() == "preserve this file"
    assert list(tmp_path.iterdir()) == [parent]
    assert not module._FLIGHTS


def test_denied_temporary_creation_fails_before_any_tokenization(
    tmp_path, small_corpus, monkeypatch
):
    def denied(*args, **kwargs):
        raise PermissionError("cache directory does not permit writes")

    monkeypatch.setattr(module.tempfile, "mkstemp", denied)
    calls = []
    with pytest.raises(LLMBackendError, match="cache_write_failed"):
        load_or_calibrate(
            tmp_path,
            binding(),
            lambda text: calls.append(text) or fake_tokens(text),
            check,
        )
    assert not calls and not list(tmp_path.iterdir())
    assert not module._FLIGHTS


def test_failed_probe_flush_cleans_up_before_any_tokenization(
    tmp_path, small_corpus, monkeypatch
):
    def failed_flush(descriptor):
        raise OSError("no space available")

    monkeypatch.setattr(module.os, "fsync", failed_flush)
    calls = []
    with pytest.raises(LLMBackendError, match="cache_write_failed"):
        load_or_calibrate(
            tmp_path,
            binding(),
            lambda text: calls.append(text) or fake_tokens(text),
            check,
        )
    assert not calls and not list(tmp_path.iterdir())
    assert not module._FLIGHTS


def test_existing_directory_profile_destination_fails_before_any_tokenization(
    tmp_path, small_corpus
):
    original = load_or_calibrate(tmp_path, binding(), fake_tokens, check)
    destination = tmp_path / (original.profile_id + ".json")
    destination.unlink()
    destination.mkdir()
    calls = []
    with pytest.raises(LLMBackendError, match="cache_write_failed"):
        load_or_calibrate(
            tmp_path,
            binding(),
            lambda text: calls.append(text) or fake_tokens(text),
            check,
        )
    assert not calls and list(tmp_path.iterdir()) == [destination]
    assert not list(destination.iterdir()) and not module._FLIGHTS


@pytest.mark.parametrize(
    "failure", [LLMCancelledError, LLMTimeoutError, KeyboardInterrupt]
)
def test_interrupted_preflight_cleans_private_probe_and_never_tokenizes(
    tmp_path, small_corpus, monkeypatch, failure
):
    original = module.tempfile.mkstemp
    created = []

    def capture(*args, **kwargs):
        descriptor, name = original(*args, **kwargs)
        created.append((descriptor, Path(name)))
        return descriptor, name

    def interrupted():
        if created and created[-1][1].exists():
            raise failure("owner stopped before calibration")

    monkeypatch.setattr(module.tempfile, "mkstemp", capture)
    calls = []
    with pytest.raises(failure, match="owner stopped"):
        load_or_calibrate(
            tmp_path,
            binding(),
            lambda text: calls.append(text) or fake_tokens(text),
            interrupted,
        )
    assert len(created) == 1
    assert not calls and not list(tmp_path.iterdir()) and not module._FLIGHTS
    with pytest.raises(OSError):
        os.fstat(created[0][0])


@pytest.mark.parametrize("failure", [LLMCancelledError, LLMTimeoutError])
def test_stop_after_final_flush_cleans_cache_temp_and_preserves_typed_failure(
    tmp_path, small_corpus, monkeypatch, failure
):
    original = module.os.fsync
    flushes = []

    def capture(descriptor):
        original(descriptor)
        flushes.append(descriptor)

    def interrupted():
        # The first flush belongs to preflight, the second to the final cache.
        if len(flushes) == 2:
            raise failure("owner stopped before cache publication")

    monkeypatch.setattr(module.os, "fsync", capture)
    calls = []
    with pytest.raises(failure, match="owner stopped"):
        load_or_calibrate(
            tmp_path,
            binding(),
            lambda text: calls.append(text) or fake_tokens(text),
            interrupted,
        )
    assert len(calls) == 15 and len(flushes) == 2
    assert not list(tmp_path.iterdir()) and not module._FLIGHTS


def test_complete_readable_cache_does_not_require_directory_write_permission(
    tmp_path, small_corpus, monkeypatch
):
    directory = tmp_path / "readable-cache"
    original = load_or_calibrate(directory, binding(), fake_tokens, check)
    path = directory / (original.profile_id + ".json")
    before, modified = path.read_bytes(), path.stat().st_mtime_ns

    def denied(*args, **kwargs):
        pytest.fail("a complete cache must not attempt a persistence preflight")

    monkeypatch.setattr(module.tempfile, "mkstemp", denied)
    path.chmod(0o400)
    directory.chmod(0o500)
    try:
        warm = load_or_calibrate(
            directory,
            binding(),
            lambda text: pytest.fail("warm cache tokenized"),
            check,
        )
        assert warm == original
        assert path.read_bytes() == before and path.stat().st_mtime_ns == modified
        assert list(directory.iterdir()) == [path]
    finally:
        directory.chmod(0o700)
        path.chmod(0o600)


@pytest.mark.parametrize("failure", [LLMCancelledError, LLMTimeoutError])
def test_stopped_warm_cache_read_never_rewrites_complete_profile(
    tmp_path, small_corpus, monkeypatch, failure
):
    original = load_or_calibrate(tmp_path, binding(), fake_tokens, check)
    path = tmp_path / (original.profile_id + ".json")
    before, modified = path.read_bytes(), path.stat().st_mtime_ns
    load = module._load_cached
    observed = []

    def cached(*args, **kwargs):
        result = load(*args, **kwargs)
        observed.append(result)
        return result

    def interrupted():
        if observed:
            raise failure("owner stopped after cache read")

    monkeypatch.setattr(module, "_load_cached", cached)
    with pytest.raises(failure, match="owner stopped"):
        load_or_calibrate(
            tmp_path,
            binding(),
            lambda text: pytest.fail("warm cache tokenized"),
            interrupted,
        )
    assert observed == [original]
    assert path.read_bytes() == before and path.stat().st_mtime_ns == modified
    assert list(tmp_path.iterdir()) == [path]


def test_preflight_writes_private_generic_probe_and_removes_it_before_remote_work(
    tmp_path, small_corpus, monkeypatch
):
    original = module.os.fsync
    flushed = []

    def capture(descriptor):
        value = os.fstat(descriptor)
        flushed.append((value.st_size, stat.S_IMODE(value.st_mode)))
        original(descriptor)

    def tokenize(text):
        assert flushed and flushed[0][0] > 0
        assert not list(tmp_path.glob(".calibration-preflight-*"))
        return fake_tokens(text)

    monkeypatch.setattr(module.os, "fsync", capture)
    profile = load_or_calibrate(tmp_path, binding(), tokenize, check)
    assert len(flushed) == 2
    if os.name != "nt":
        assert all(mode == 0o600 for _, mode in flushed)
    path = tmp_path / (profile.profile_id + ".json")
    assert list(tmp_path.iterdir()) == [path]
    assert b"write-probe" not in path.read_bytes()


def test_mixed_mean_character_recommendation_does_not_guarantee_code_category_fit():
    # Aggregate counts retained in the public 0.11.0 qualification. This is a
    # mathematical cross-category example, not a claim about any specific input.
    code_ratio = 946784 / 344037
    mixed_ratio = 6838911 / 2088271
    statistics = {
        "code": CalibrationStats(
            2000,
            946784,
            344037,
            code_ratio,
            2.790804390271928,
            2.350515463917526,
            3.256637168141593,
        ),
        "mixed": CalibrationStats(
            10000,
            6838911,
            2088271,
            mixed_ratio,
            3.356641988786965,
            1.2115384615384615,
            4.882352941176471,
        ),
    }
    profile = CalibrationProfile("math-example", binding(), "c" * 64, statistics)
    historical_code_budget = math.floor(512 * code_ratio * 0.80)
    historical_mixed_budget = math.floor(512 * mixed_ratio * 0.90)
    assert historical_code_budget == 1127 and historical_mixed_budget == 1509
    assert historical_mixed_budget / code_ratio == pytest.approx(548.3318613326799)
    assert historical_mixed_budget / code_ratio > 512
    # Each selected category now uses its own mean. The mixed recommendation
    # remains an empirical aggregate, not exact admission for every category.
    assert profile.max_input_chars(512, "code") == historical_code_budget
    assert profile.max_input_chars(512, "mixed") == historical_mixed_budget
    assert CATEGORY_SAFETY_FACTOR == 0.80 and MIXED_SAFETY_FACTOR == 0.90


def test_existing_profile_read_uses_only_manifest_and_never_materializes_corpus(
    tmp_path, small_corpus, monkeypatch
):
    original = load_or_calibrate(tmp_path, binding(), fake_tokens, check)
    path = tmp_path / (original.profile_id + ".json")
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("warm-only profile lookup materialized/calibrated source records")

    monkeypatch.setattr(module, "_load_corpus", forbidden)
    monkeypatch.setattr(module, "_calibrate", forbidden)
    monkeypatch.setattr(module, "_preflight_cache", forbidden)
    assert load_existing_calibration(tmp_path, binding(), check) == original
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_existing_profile_lookup_is_inert_when_cache_missing(tmp_path, small_corpus):
    missing = tmp_path / "never-created"
    assert load_existing_calibration(missing, binding(), check) is None
    assert not missing.exists()


def test_existing_profile_lookup_does_not_repair_invalid_cache(tmp_path, small_corpus):
    original = load_or_calibrate(tmp_path, binding(), fake_tokens, check)
    path = tmp_path / (original.profile_id + ".json")
    path.write_bytes(b"invalid cache")
    assert load_existing_calibration(tmp_path, binding(), check) is None
    assert path.read_bytes() == b"invalid cache"


def test_existing_profile_lookup_does_not_load_a_different_binding(
    tmp_path, small_corpus
):
    load_or_calibrate(tmp_path, binding(), fake_tokens, check)
    assert (
        load_existing_calibration(tmp_path, binding(input_type="query"), check) is None
    )


def test_existing_profile_lookup_checks_cancellation_without_creating_state(
    tmp_path, small_corpus
):
    def stopped():
        raise LLMCancelledError("stopped profile lookup")

    with pytest.raises(LLMCancelledError, match="stopped profile lookup"):
        load_existing_calibration(tmp_path, binding(), stopped)
    assert not list(tmp_path.iterdir())


def test_packaged_manifest_lookup_preserves_timeout_type():
    calls = []

    def expired():
        calls.append(1)
        if len(calls) > 1:
            raise LLMTimeoutError("manifest deadline")

    with pytest.raises(LLMTimeoutError, match="manifest deadline"):
        module._load_manifest(expired)
