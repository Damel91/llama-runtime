"""Explicit replica lists obey the same content authority as discovery."""

from copy import deepcopy

import pytest

from runtime_llama.errors import LLMBackendError
from runtime_llama.replica_discovery import ReplicaIdentity, verify_explicit_pool


CONTENT = "a" * 64
SEMANTIC = "b" * 64
OTHER = "c" * 64


def entry(model, *, content=CONTENT, semantic=SEMANTIC, loaded=True, aliases=()):
    tags = []
    if content is not None:
        tags.append(f"noxi-content-sha256-v1:{content}")
    if semantic is not None:
        tags.append(f"noxi-embedding-sha256-v1:{semantic}")
    return {
        "id": model,
        "aliases": list(aliases),
        "status": {"value": "loaded" if loaded else "unloaded"},
        "tags": tags,
    }


def test_explicit_selection_is_sorted_and_excludes_unconfigured_equivalent_copies():
    catalog = {
        "data": [entry("c"), entry("a"), entry("b"), entry("llm", content=OTHER)]
    }
    original = deepcopy(catalog)
    selected = verify_explicit_pool(catalog, ["b", "a"])
    assert selected.topology == (
        ("a", (CONTENT, SEMANTIC)),
        ("b", (CONTENT, SEMANTIC)),
    )
    assert selected.identity == ReplicaIdentity(CONTENT, SEMANTIC)
    assert catalog == original


def test_synonyms_and_repeated_selectors_cannot_multiply_one_destination():
    selected = verify_explicit_pool(
        {"data": [entry("a", aliases=("synonym", "other")), entry("b")]},
        ["a", "synonym", "other", "a"],
    )
    assert [replica.model for replica in selected.replicas] == ["a"]
    assert selected.replicas[0].aliases == frozenset({"a", "synonym", "other"})


def test_single_unverified_canonical_target_stays_conservative_after_deduplication():
    selected = verify_explicit_pool(
        {"data": [entry("a", content=None, semantic=None, aliases=("synonym",))]},
        ["a", "synonym"],
    )
    assert len(selected.replicas) == 1
    assert selected.identity is None and selected.fallback_alias == "a"


@pytest.mark.parametrize(
    "second",
    [
        entry("b", content=OTHER),
        entry("b", semantic=OTHER),
        entry("b", content=None, semantic=None),
    ],
)
def test_distinct_explicit_destinations_require_matching_complete_identities(second):
    first = entry("a")
    first["path"] = second["path"] = "/same/file.gguf"
    first["meta"] = second["meta"] = {"n_embd": 768}
    with pytest.raises(LLMBackendError, match="ambiguous_model_identity"):
        verify_explicit_pool({"data": [first, second]}, ["a", "b"])


@pytest.mark.parametrize("second", [entry("b", loaded=False), entry("other")])
def test_every_explicit_destination_must_be_present_and_loaded(second):
    with pytest.raises(LLMBackendError, match="model_not_loaded"):
        verify_explicit_pool({"data": [entry("a"), second]}, ["a", "b"])


@pytest.mark.parametrize("changes", [{"content": OTHER}, {"semantic": OTHER}])
def test_pinned_identity_rejects_alias_reuse_without_selecting_other_copies(changes):
    with pytest.raises(LLMBackendError, match="content_identity_changed"):
        verify_explicit_pool(
            {"data": [entry("a", **changes), entry("other")]},
            ["a"],
            pinned=ReplicaIdentity(CONTENT, SEMANTIC),
        )


def test_pinned_selection_cannot_fall_back_to_unconfigured_same_identity_alias():
    with pytest.raises(LLMBackendError, match="model_not_loaded"):
        verify_explicit_pool(
            {"data": [entry("other")]},
            ["a"],
            pinned=ReplicaIdentity(CONTENT, SEMANTIC),
        )


@pytest.mark.parametrize(
    "catalog,selectors",
    [
        ({"data": [entry("a"), entry("b", aliases=("a",))]}, ["a"]),
        (
            {"data": [entry("a", aliases=("alias",)), entry("b", aliases=("a",))]},
            ["alias"],
        ),
        ({"data": [entry("a"), entry("a")]}, ["a"]),
    ],
)
def test_routing_collisions_are_rejected_including_unloaded_peers(catalog, selectors):
    catalog["data"][-1]["status"] = {"value": "unloaded"}
    with pytest.raises(LLMBackendError, match="ambiguous_model"):
        verify_explicit_pool(catalog, selectors)


def test_malformed_selected_identity_never_degrades_to_single_fallback():
    with pytest.raises(LLMBackendError, match="incomplete_content_identity"):
        verify_explicit_pool({"data": [entry("a", semantic=None)]}, ["a"])


def test_unrelated_malformed_identity_is_not_a_candidate():
    selected = verify_explicit_pool(
        {"data": [entry("a"), entry("b"), entry("other", semantic=None)]},
        ["b", "a"],
    )
    assert [replica.model for replica in selected.replicas] == ["a", "b"]


@pytest.mark.parametrize("catalog", [None, {}, {"data": None}, {"data": [3]}])
def test_invalid_catalog_is_a_typed_failure(catalog):
    with pytest.raises(LLMBackendError, match="invalid_model_catalog"):
        verify_explicit_pool(catalog, ["a"])


@pytest.mark.parametrize("selectors", [[], [""], [None], ["a", 3], None, "a", 3])
def test_invalid_explicit_selector_list_is_a_typed_failure(selectors):
    with pytest.raises(LLMBackendError, match="invalid_model_selector"):
        verify_explicit_pool({"data": [entry("a")]}, selectors)
