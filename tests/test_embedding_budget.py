"""Persistent learned character ceilings are scalar, monotone and scoped."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import stat
from threading import Event
import time

import pytest

import runtime_llama.embedding_budget as module
from runtime_llama.embedding_budget import (
    EMBEDDING_BUDGET_POLICY_ID,
    LearnedBudgetState,
    budget_context_scope_id,
    load_learned_budget,
    lower_learned_budget,
)
from runtime_llama.errors import LLMBackendError, LLMCancelledError, LLMTimeoutError


def binding(**changes):
    value = {
        "content_sha256": "a" * 64,
        "embedding_sha256": "b" * 64,
        "tokenizer_build": "native-test-1",
        "vocab_size": 32000,
        "input_type": "document",
        "policy": "add_special=true;parse_special=true;prefix=search_document: ",
        "prefix": "search_document: ",
    }
    value.update(changes)
    return value


def check():
    pass


def test_fixed_hard_corpus_mean_policy_cannot_reuse_minimum_or_learned_policy():
    previous = {
        **module._POLICY,
        "basis": "minimum-source-characters-per-fully-prepared-token-across-all-10000",
        "learned_limit": "absolute-characters-without-percentage-reapplication",
    }
    assert (
        EMBEDDING_BUDGET_POLICY_ID
        != hashlib.sha256(module._canonical(previous)).hexdigest()
    )
    assert module._POLICY["basis"] == (
        "weighted-mean-source-characters-per-fully-prepared-token-selected-hard-2500"
    )
    assert module._POLICY["category_factor"] == 0.80
    assert module._POLICY["mixed_factor"] == 0.90
    assert (
        module._POLICY["learned_limit"]
        == "fixed-initial-recommendation-no-inference-updates"
    )


def state_path(directory, context=512, **changes):
    return directory / (
        "budget-" + budget_context_scope_id(binding(**changes), context) + ".json"
    )


def bounded_check(seconds=5):
    deadline = time.monotonic() + seconds

    def remaining():
        if time.monotonic() >= deadline:
            raise LLMTimeoutError("bounded budget operation expired")

    return remaining


def _process_reduce(directory, candidates, barrier, results):
    try:
        barrier.wait(8)
        states = [
            lower_learned_budget(directory, binding(), 512, candidate, bounded_check())
            for candidate in candidates
        ]
        results.put({"ceilings": [state.max_input_chars for state in states]})
    except BaseException as exc:
        results.put({"error": type(exc).__name__})


def _process_hold_lock(lock_path, entered, release):
    with module._locked(Path(lock_path), bounded_check(8)):
        entered.set()
        if not release.wait(8):
            raise RuntimeError("parent did not release held test lock")


def test_missing_read_is_inert_and_never_creates_a_directory(tmp_path):
    missing = tmp_path / "absent"
    assert load_learned_budget(missing, binding(), 512, check) is None
    assert not missing.exists()


def test_absolute_limit_only_decreases_and_percentages_are_never_reapplied(tmp_path):
    first = lower_learned_budget(tmp_path, binding(), 512, 150, check)
    assert type(first) is LearnedBudgetState
    assert first.max_input_chars == 150 and first.budget_revision == 1
    assert first.effective_context_tokens == 512
    assert first.context_scope_id == budget_context_scope_id(binding(), 512)
    original = state_path(tmp_path).read_bytes()
    assert lower_learned_budget(tmp_path, binding(), 512, 150, check) == first
    assert lower_learned_budget(tmp_path, binding(), 512, 200, check) == first
    assert state_path(tmp_path).read_bytes() == original
    second = lower_learned_budget(tmp_path, binding(), 512, 100, check)
    assert second.max_input_chars == 100 and second.budget_revision == 2
    assert load_learned_budget(tmp_path, binding(), 512, check) == second
    assert second.max_input_chars not in (80, 90)
    with pytest.raises(FrozenInstanceError):
        second.max_input_chars = 200


def test_categories_aliases_and_connection_details_do_not_create_new_learning_spaces(
    tmp_path,
):
    first = lower_learned_budget(tmp_path, binding(category="code"), 512, 150, check)
    assert (
        load_learned_budget(
            tmp_path,
            binding(category="mixed", alias="replica-two", endpoint="runtime.example"),
            512,
            check,
        )
        == first
    )
    second = lower_learned_budget(tmp_path, binding(category="prose"), 512, 100, check)
    assert second.context_scope_id == first.context_scope_id
    assert second.budget_revision == 2


@pytest.mark.parametrize(
    "change",
    [
        {"content_sha256": "c" * 64},
        {"embedding_sha256": "d" * 64},
        {"tokenizer_build": "native-test-2"},
        {"vocab_size": 64000},
        {"vocab_size": None},
        {"input_type": "query", "prefix": "search_query: ", "policy": "query-policy"},
        {"prefix": "other: "},
        {"policy": "different-special-token-policy"},
    ],
)
def test_every_verified_tokenizer_binding_change_uses_a_different_scope(
    tmp_path, change
):
    first = lower_learned_budget(tmp_path, binding(), 512, 100, check)
    assert load_learned_budget(tmp_path, binding(**change), 512, check) is None
    changed = lower_learned_budget(tmp_path, binding(**change), 512, 200, check)
    assert changed.context_scope_id != first.context_scope_id
    assert changed.budget_revision == 1
    assert load_learned_budget(tmp_path, binding(), 512, check) == first


def test_context_changes_are_explicit_independent_scopes_without_hidden_scaling(
    tmp_path,
):
    original = lower_learned_budget(tmp_path, binding(), 512, 100, check)
    assert load_learned_budget(tmp_path, binding(), 8192, check) is None
    changed = lower_learned_budget(tmp_path, binding(), 8192, 300, check)
    assert changed.context_scope_id != original.context_scope_id
    assert changed.max_input_chars == 300 and changed.budget_revision == 1
    assert load_learned_budget(tmp_path, binding(), 512, check) == original


def test_state_contains_only_digest_and_scalar_fields_with_private_permissions(
    tmp_path,
):
    value = binding(
        endpoint="private.example", api_key="synthetic-secret", text="never-retain"
    )
    state = lower_learned_budget(tmp_path, value, 512, 100, check)
    path = state_path(tmp_path)
    payload = path.read_bytes()
    parsed = json.loads(payload)
    assert set(parsed) == {
        "schema",
        "policy_id",
        "binding_sha256",
        "context_scope_id",
        "budget_revision",
        "max_input_chars",
        "effective_context_tokens",
    }
    assert parsed["policy_id"] == EMBEDDING_BUDGET_POLICY_ID
    assert parsed["context_scope_id"] == state.context_scope_id
    assert not any(
        marker in payload
        for marker in (
            b"private.example",
            b"synthetic-secret",
            b"never-retain",
            b"search_document",
            b"token_ids",
            b"split_plan",
            b"source_text",
            b"native-test",
        )
    )
    assert path.with_suffix(".lock").read_bytes() == b"\x00"
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.with_suffix(".lock").stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".budget-tmp-*"))


@pytest.mark.parametrize(
    "damage",
    [
        lambda value: value.update(schema="unknown"),
        lambda value: value.update(policy_id="different"),
        lambda value: value.update(binding_sha256="f" * 64),
        lambda value: value.update(context_scope_id="f" * 64),
        lambda value: value.update(effective_context_tokens=True),
        lambda value: value.update(effective_context_tokens=8192),
        lambda value: value.update(max_input_chars=0),
        lambda value: value.update(max_input_chars=True),
        lambda value: value.update(max_input_chars=1.5),
        lambda value: value.update(budget_revision=True),
        lambda value: value.update(budget_revision=0),
        lambda value: value.update(extra="unexpected"),
    ],
)
def test_invalid_state_is_not_silently_discarded_or_replaced(tmp_path, damage):
    lower_learned_budget(tmp_path, binding(), 512, 100, check)
    path = state_path(tmp_path)
    value = json.loads(path.read_bytes())
    damage(value)
    payload = json.dumps(value).encode()
    path.write_bytes(payload)
    with pytest.raises(LLMBackendError, match="learned_budget_invalid"):
        load_learned_budget(tmp_path, binding(), 512, check)
    with pytest.raises(LLMBackendError, match="learned_budget_invalid"):
        lower_learned_budget(tmp_path, binding(), 512, 200, check)
    assert path.read_bytes() == payload


@pytest.mark.parametrize("payload", [b"{", b"[]", b"x" * 16385, b'{"a":1,"a":2}'])
def test_malformed_or_duplicate_state_does_not_reset_learning(tmp_path, payload):
    lower_learned_budget(tmp_path, binding(), 512, 100, check)
    state_path(tmp_path).write_bytes(payload)
    with pytest.raises(LLMBackendError, match="learned_budget_invalid"):
        lower_learned_budget(tmp_path, binding(), 512, 200, check)
    assert state_path(tmp_path).read_bytes() == payload


def test_retained_invalid_state_error_does_not_retain_large_decoder_payload(tmp_path):
    lower_learned_budget(tmp_path, binding(), 512, 100, check)
    path = state_path(tmp_path)
    payload = b'{"schema":"' + b"unclosed-state-value" * 2048
    assert len(payload) > 16384
    path.write_bytes(payload)
    with pytest.raises(LLMBackendError, match="learned_budget_invalid") as caught:
        load_learned_budget(tmp_path, binding(), 512, check)
    error = caught.value
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback = error.__traceback__
    observed_reader = False
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_code.co_name == "_read":
            observed_reader = True
            assert frame.f_locals["payload"] is None
            assert frame.f_locals["value"] is None
        traceback = traceback.tb_next
    assert observed_reader
    assert path.read_bytes() == payload


def test_readable_state_load_does_not_need_write_permission(tmp_path, monkeypatch):
    directory = tmp_path / "cache"
    first = lower_learned_budget(directory, binding(), 512, 100, check)
    path = state_path(directory)
    before, modified = path.read_bytes(), path.stat().st_mtime_ns
    path.chmod(0o400)
    directory.chmod(0o500)

    def forbidden(*args, **kwargs):
        pytest.fail("read-only state load attempted a write")

    monkeypatch.setattr(module.tempfile, "mkstemp", forbidden)
    try:
        assert load_learned_budget(directory, binding(), 512, check) == first
        assert path.read_bytes() == before and path.stat().st_mtime_ns == modified
    finally:
        directory.chmod(0o700)
        path.chmod(0o600)


@pytest.mark.parametrize("failure", [LLMCancelledError, LLMTimeoutError])
def test_cancellation_and_timeout_before_commit_keep_previous_state_complete(
    tmp_path, monkeypatch, failure
):
    first = lower_learned_budget(tmp_path, binding(), 512, 300, check)
    path = state_path(tmp_path)
    before = path.read_bytes()
    original = module.tempfile.mkstemp
    created = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        created.append(Path(result[1]))
        return result

    def stopped():
        if created:
            raise failure("owner stopped before commit")

    monkeypatch.setattr(module.tempfile, "mkstemp", capture)
    with pytest.raises(failure, match="owner stopped"):
        lower_learned_budget(tmp_path, binding(), 512, 200, stopped)
    assert path.read_bytes() == before
    assert load_learned_budget(tmp_path, binding(), 512, check) == first
    assert not list(tmp_path.glob(".budget-tmp-*"))


def test_cancellation_after_atomic_commit_does_not_undo_valid_decrease(
    tmp_path, monkeypatch
):
    lower_learned_budget(tmp_path, binding(), 512, 300, check)
    committed = Event()
    original = module.os.replace

    def replace(*args):
        original(*args)
        committed.set()

    def stopped():
        if committed.is_set():
            raise LLMCancelledError("cancelled after publication")

    monkeypatch.setattr(module.os, "replace", replace)
    second = lower_learned_budget(tmp_path, binding(), 512, 200, stopped)
    assert committed.is_set() and second.budget_revision == 2
    assert load_learned_budget(tmp_path, binding(), 512, check) == second


def test_failed_replace_keeps_original_and_cleans_complete_temporary_state(
    tmp_path, monkeypatch
):
    first = lower_learned_budget(tmp_path, binding(), 512, 300, check)
    before = state_path(tmp_path).read_bytes()

    def failed(*args):
        raise OSError("synthetic filesystem failure")

    monkeypatch.setattr(module.os, "replace", failed)
    with pytest.raises(LLMBackendError, match="learned_budget_storage_failed"):
        lower_learned_budget(tmp_path, binding(), 512, 200, check)
    assert state_path(tmp_path).read_bytes() == before
    assert load_learned_budget(tmp_path, binding(), 512, check) == first
    assert not list(tmp_path.glob(".budget-tmp-*"))


def test_parallel_threads_converge_on_minimum_and_preserve_monotone_revisions(tmp_path):
    start = Event()
    candidates = [900, 300, 600, 100, 800, 200, 400, 500]

    def update(candidate):
        assert start.wait(3)
        return lower_learned_budget(
            tmp_path, binding(), 512, candidate, bounded_check()
        )

    with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
        futures = [executor.submit(update, value) for value in candidates]
        start.set()
        states = [future.result(5) for future in futures]
    final = load_learned_budget(tmp_path, binding(), 512, check)
    assert final.max_input_chars == min(candidates)
    assert 1 <= final.budget_revision <= len(candidates)
    by_revision = {}
    for state in states:
        assert (
            by_revision.setdefault(state.budget_revision, state.max_input_chars)
            == state.max_input_chars
        )
    ordered = [ceiling for _, ceiling in sorted(by_revision.items())]
    assert all(left > right for left, right in zip(ordered, ordered[1:]))


def test_parallel_processes_converge_without_partial_state(tmp_path):
    context = multiprocessing.get_context("spawn")
    barrier, results = context.Barrier(2), context.Queue()
    processes = [
        context.Process(
            target=_process_reduce, args=(str(tmp_path), values, barrier, results)
        )
        for values in ([900, 700, 500], [800, 600, 400])
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(12)
            assert process.exitcode == 0
        reports = [results.get(timeout=2) for _ in processes]
        assert all("error" not in report for report in reports)
        final = load_learned_budget(tmp_path, binding(), 512, check)
        assert final.max_input_chars == 400
        assert 1 <= final.budget_revision <= 6
        assert not list(tmp_path.glob(".budget-tmp-*"))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(2)
        results.close()
        results.join_thread()


def test_waiting_on_another_process_lock_is_cancellable_and_does_not_modify_state(
    tmp_path,
):
    first = lower_learned_budget(tmp_path, binding(), 512, 300, check)
    before = state_path(tmp_path).read_bytes()
    context = multiprocessing.get_context("spawn")
    entered, release = context.Event(), context.Event()
    process = context.Process(
        target=_process_hold_lock,
        args=(str(state_path(tmp_path).with_suffix(".lock")), entered, release),
    )
    try:
        process.start()
        assert entered.wait(8)
        checks = []

        def cancelled():
            checks.append(1)
            if len(checks) >= 7:
                raise LLMCancelledError("cancelled waiting for process lock")

        with pytest.raises(LLMCancelledError, match="waiting for process lock"):
            lower_learned_budget(tmp_path, binding(), 512, 200, cancelled)
        assert state_path(tmp_path).read_bytes() == before
        assert load_learned_budget(tmp_path, binding(), 512, check) == first
    finally:
        release.set()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(2)
    assert process.exitcode == 0
    assert (
        lower_learned_budget(
            tmp_path, binding(), 512, 200, bounded_check()
        ).max_input_chars
        == 200
    )


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None, "512"])
def test_invalid_context_is_rejected_without_creating_state(tmp_path, value):
    with pytest.raises(ValueError):
        lower_learned_budget(tmp_path, binding(), value, 100, check)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None, "100"])
def test_invalid_character_ceiling_is_rejected_without_creating_state(tmp_path, value):
    with pytest.raises(ValueError):
        lower_learned_budget(tmp_path, binding(), 512, value, check)
    assert not list(tmp_path.iterdir())


def test_budget_scalar_state_has_no_implicit_numeric_or_file_size_ceiling(tmp_path):
    context, characters = 2**40, 2**60
    identity = binding(vocab_size=2**40)
    state = lower_learned_budget(tmp_path, identity, context, characters, check)
    assert load_learned_budget(tmp_path, identity, context, check) == state
    path = state_path(tmp_path, context, vocab_size=2**40)
    value = json.loads(path.read_bytes())
    value["budget_revision"] = 2**60
    # Legal JSON whitespace is not a resource policy or a schema violation.
    payload = json.dumps(value, indent=4096).encode()
    assert len(payload) > 16384
    path.write_bytes(payload)
    loaded = load_learned_budget(tmp_path, identity, context, check)
    assert loaded.budget_revision == 2**60
    assert loaded.max_input_chars == characters
    assert loaded.effective_context_tokens == context


@pytest.mark.parametrize(
    "field",
    [
        "content_sha256",
        "embedding_sha256",
        "tokenizer_build",
        "input_type",
        "policy",
        "prefix",
        "vocab_size",
    ],
)
def test_incomplete_binding_cannot_create_a_learned_space(tmp_path, field):
    value = binding()
    del value[field]
    with pytest.raises(ValueError):
        lower_learned_budget(tmp_path, value, 512, 100, check)
    assert not list(tmp_path.iterdir())
