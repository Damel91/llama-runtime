"""Native execution observations cannot manufacture HTTP request budgets."""

from dataclasses import FrozenInstanceError

import pytest

from runtime_llama.embedding_metadata import (
    LoadedExecutionMetadata,
    loaded_batch_limit,
    loaded_execution_metadata,
)
from runtime_llama.errors import LLMBackendError


def loaded(*args):
    return {"status": {"value": "loaded", "args": ["llama-server", *args]}}


def test_native_execution_values_are_distinct_immutable_observations():
    observed = loaded_execution_metadata(
        {"n_batch": 4096, "n_ubatch": 512, "total_slots": 4},
        loaded("--batch-size", "4096", "--ubatch-size", "512"),
    )
    assert observed == LoadedExecutionMetadata(4096, 512, 4)
    assert not hasattr(observed, "batch_token_limit")
    assert not hasattr(observed, "can_split")
    with pytest.raises(FrozenInstanceError):
        observed.n_batch = 1


@pytest.mark.parametrize(
    "props,entry",
    [
        (None, None),
        ({}, {}),
        ({"default_generation_settings": {"n_ctx": 8192}}, None),
        ({"model_path": "--batch-size=99", "n_ctx_train": 8192}, None),
        ({}, {"preset": "--batch-size 99 --ubatch-size 11"}),
        ({}, loaded("--parallel", "4", "--ctx-size", "8192")),
        ({}, loaded("--model", "--batch-size", "--alias", "--ubatch-size")),
        ({}, loaded("--", "--batch-size", "99", "--ubatch-size", "11")),
    ],
)
def test_unpublished_or_unrelated_values_remain_unknown(props, entry):
    assert loaded_execution_metadata(props, entry) == LoadedExecutionMetadata()


@pytest.mark.parametrize(
    "args",
    [
        ("--batch-size", "2048", "--ubatch-size", "256"),
        ("-b", "2048", "-ub", "256"),
        ("--batch-size=2048", "--ubatch-size=256"),
    ],
)
def test_native_batch_and_microbatch_supported_argv_spellings(args):
    assert loaded_execution_metadata({}, loaded(*args)) == LoadedExecutionMetadata(
        2048, 256, None
    )


def test_root_default_settings_and_loaded_argv_can_confirm_same_values():
    props = {
        "n_batch": 2048,
        "n_ubatch": 256,
        "default_generation_settings": {"n_batch": 2048, "n_ubatch": 256},
    }
    assert loaded_execution_metadata(
        props, loaded("-b", "2048", "-ub", "256")
    ) == LoadedExecutionMetadata(2048, 256)


@pytest.mark.parametrize("name", ["n_batch", "n_ubatch"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 64.0, "64", None])
def test_malformed_supported_native_properties_are_typed_failures(name, value):
    with pytest.raises(LLMBackendError, match="invalid_loaded_batch"):
        loaded_execution_metadata({name: value})


@pytest.mark.parametrize("name,option", [("n_batch", "-b"), ("n_ubatch", "-ub")])
def test_conflicting_supported_values_do_not_select_a_convenient_limit(name, option):
    with pytest.raises(LLMBackendError, match="ambiguous_loaded_batch"):
        loaded_execution_metadata({name: 128}, loaded(option, "256"))
    with pytest.raises(LLMBackendError, match="ambiguous_loaded_batch"):
        loaded_execution_metadata(
            {name: 128, "default_generation_settings": {name: 256}}
        )


@pytest.mark.parametrize("option", ["-b", "-ub", "--batch-size", "--ubatch-size"])
def test_repeated_native_argv_is_ambiguous_even_when_values_match(option):
    with pytest.raises(LLMBackendError, match="ambiguous_loaded_batch"):
        loaded_execution_metadata({}, loaded(option, "128", option, "128"))


@pytest.mark.parametrize("option", ["--batch-size", "--ubatch-size"])
@pytest.mark.parametrize("value", ["", "-1", "0", "1.5", "+4", "--model"])
def test_native_argv_sizes_require_positive_decimal_values(option, value):
    with pytest.raises(LLMBackendError, match="invalid_loaded_batch"):
        loaded_execution_metadata({}, loaded(option + "=" + value))


@pytest.mark.parametrize("option", ["--batch-size", "--ubatch-size"])
def test_missing_native_option_argument_is_not_unknown_capacity(option):
    with pytest.raises(LLMBackendError, match="invalid_loaded_batch"):
        loaded_execution_metadata({}, loaded(option))


@pytest.mark.parametrize("args", [None, "--batch-size 128", [128], {"-b": "128"}])
def test_malformed_loaded_argv_is_a_typed_failure(args):
    with pytest.raises(LLMBackendError, match="invalid_loaded_batch"):
        loaded_execution_metadata({}, {"status": {"args": args}})


@pytest.mark.parametrize("value", [True, False, 0, -1, 2.0, "2", None])
def test_unverified_slot_values_cannot_authorize_more_concurrency(value):
    assert loaded_execution_metadata({"total_slots": value}).total_slots is None


def test_native_batch_compatibility_helper_is_only_the_native_observation():
    assert loaded_batch_limit({"n_batch": 128, "n_ubatch": 32}) == 128
    assert loaded_batch_limit({"n_ubatch": 32}) is None


def test_batch_looking_values_consumed_by_other_options_are_not_declarations():
    observed = loaded_execution_metadata(
        {}, loaded("--model", "--batch-size", "--alias", "--ubatch-size", "-b", "64")
    )
    assert observed == LoadedExecutionMetadata(n_batch=64)
