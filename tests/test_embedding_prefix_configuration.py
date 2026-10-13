"""Model task prefixes are literal, immutable adapter configuration."""

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import pytest

from runtime_llama import AdapterProfile, EmbeddingPrefixes, build_embedding_adapter


def profile(**settings):
    return AdapterProfile.from_config(
        "ingest", {"adapter_type": "embedding", "model": "embed_0", **settings}
    )


def test_empty_prefix_defaults_preserve_common_llm_and_embedding_profiles():
    parsed = profile()
    llm = AdapterProfile.from_config("coder", {"model": "llm"})
    assert parsed.embedding_prefixes == EmbeddingPrefixes()
    assert llm.embedding_prefixes == EmbeddingPrefixes()
    assert EmbeddingPrefixes.from_value({}) == EmbeddingPrefixes()


def test_public_typed_prefixes_are_immutable_and_preserve_literal_whitespace():
    prefixes = EmbeddingPrefixes(query="  query:\t", document="title: none | text: ")
    parsed = replace(profile(), embedding_prefixes=prefixes)
    assert parsed.embedding_prefixes is prefixes
    assert EmbeddingPrefixes.from_value(prefixes) is prefixes
    assert prefixes.query == "  query:\t"
    assert prefixes.document == "title: none | text: "
    with pytest.raises(FrozenInstanceError):
        prefixes.query = "changed"


def test_mapping_is_canonicalized_without_retaining_mutable_input():
    settings = {"query": "search_query: ", "document": "search_document: "}
    parsed = profile(embedding_prefixes=settings)
    settings["query"] = "changed"
    assert isinstance(parsed.embedding_prefixes, EmbeddingPrefixes)
    assert parsed.embedding_prefixes.query == "search_query: "
    assert parsed.embedding_prefixes.document == "search_document: "
    assert EmbeddingPrefixes.from_value(MappingProxyType(settings)).query == "changed"


def test_role_override_replaces_the_whole_prefix_mapping():
    defaults = {
        "adapter_type": "embedding",
        "model": "embed_0",
        "embedding_prefixes": {"query": "default query: ", "document": "default doc: "},
    }
    inherited = AdapterProfile.from_config("ingest", {}, defaults)
    overridden = AdapterProfile.from_config(
        "search", {"embedding_prefixes": {"query": "task: search | query: "}}, defaults
    )
    assert inherited.embedding_prefixes == EmbeddingPrefixes(
        query="default query: ", document="default doc: "
    )
    assert overridden.embedding_prefixes == EmbeddingPrefixes(
        query="task: search | query: "
    )


def test_factory_accepts_prefix_mapping_and_typed_profile_without_http():
    parsed = profile(embedding_prefixes=EmbeddingPrefixes(query="query: "))
    for config in (
        {"model": "embed_0", "embedding_prefixes": {"query": "query: "}},
        parsed,
        parsed.to_embedding_config(),
    ):
        adapter = build_embedding_adapter(config, client=object())
        try:
            assert adapter.profile.embedding_prefixes == EmbeddingPrefixes(
                query="query: "
            )
        finally:
            adapter.close()


def test_prefixes_do_not_change_external_server_physical_identity():
    original = profile()
    changed = profile(embedding_prefixes={"query": "query: ", "document": "document: "})
    assert (
        original.model_config.physical_identity
        == changed.model_config.physical_identity
    )


@pytest.mark.parametrize("field", ["query", "document"])
@pytest.mark.parametrize("value", [None, True, 1, 1.5, b"prefix", [], {}])
def test_prefix_values_must_be_strings(field, value):
    with pytest.raises(ValueError, match="must be a string"):
        profile(embedding_prefixes={field: value})


@pytest.mark.parametrize(
    "value",
    [None, "query: ", ["query: "], True, 1, {"query_prefix": "query: "}, {"raw": ""}],
)
def test_malformed_and_unknown_prefix_options_are_rejected(value):
    with pytest.raises(ValueError, match="only query/document"):
        profile(embedding_prefixes=value)


@pytest.mark.parametrize("prefixes", [{"query": "query: "}, {"document": " "}])
def test_nonempty_prefixes_require_an_embedding_profile(prefixes):
    with pytest.raises(ValueError, match="requires an embedding profile"):
        AdapterProfile.from_config(
            "coder", {"model": "llm", "embedding_prefixes": prefixes}
        )
