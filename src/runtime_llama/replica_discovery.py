"""Read-only selection of loaded replicas using server content identities.

The Noxi tags identify the bytes of the model and its embedding semantics.
Neither a path, a display name nor a vector width establishes equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
from urllib.parse import unquote_to_bytes

from runtime_llama.errors import LLMBackendError

_CONTENT = "noxi-content-sha256-v1:"
_EMBEDDING = "noxi-embedding-sha256-v1:"
_NAME = "noxi-model-name-v1:"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def catalog_aliases(entry: dict[str, Any]) -> set[str]:
    values = [entry.get("id")]
    if isinstance(entry.get("aliases"), list):
        values.extend(entry["aliases"])
    return {value for value in values if isinstance(value, str) and value}


def is_loaded(entry: dict[str, Any]) -> bool:
    status = entry.get("status")
    return (status.get("value") if isinstance(status, dict) else status) == "loaded"


@dataclass(frozen=True)
class ReplicaIdentity:
    content_sha256: str
    embedding_sha256: str

    @property
    def key(self) -> tuple[str, str]:
        return self.content_sha256, self.embedding_sha256


@dataclass(frozen=True)
class DiscoveredReplica:
    model: str
    aliases: frozenset[str]
    identity: ReplicaIdentity | None
    model_name: str | None


@dataclass(frozen=True)
class EmbeddingPool:
    replicas: tuple[DiscoveredReplica, ...]
    identity: ReplicaIdentity | None
    fallback_alias: str | None = None

    @property
    def topology(self) -> tuple[tuple[str, tuple[str, str] | None], ...]:
        return tuple(
            (replica.model, replica.identity.key if replica.identity else None)
            for replica in self.replicas
        )


def _tag_values(entry: dict[str, Any]) -> tuple[ReplicaIdentity | None, str | None]:
    tags = entry.get("tags", [])
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise LLMBackendError("llamacpp_embedding_invalid_identity_tags")
    values: dict[str, str] = {}
    for tag in tags:
        for prefix in (_CONTENT, _EMBEDDING, _NAME):
            if tag.startswith(prefix):
                value = tag[len(prefix) :]
                if prefix in values and values[prefix] != value:
                    raise LLMBackendError("llamacpp_embedding_ambiguous_identity_tags")
                values[prefix] = value
    content, embedding = values.get(_CONTENT), values.get(_EMBEDDING)
    if (content is None) != (embedding is None):
        raise LLMBackendError("llamacpp_embedding_incomplete_content_identity")
    if content is not None and (
        not _SHA256.fullmatch(content) or not _SHA256.fullmatch(embedding or "")
    ):
        raise LLMBackendError("llamacpp_embedding_invalid_content_identity")
    name = values.get(_NAME)
    if name is not None:
        try:
            name = unquote_to_bytes(name).decode("utf-8", errors="strict")
        except (UnicodeError, ValueError):
            raise LLMBackendError("llamacpp_embedding_invalid_model_name") from None
        if not name or "/" in name or "\\" in name:
            raise LLMBackendError("llamacpp_embedding_invalid_model_name")
    return (
        ReplicaIdentity(content, embedding) if content is not None else None,
        name,
    )


def parse_replica(entry: dict[str, Any]) -> DiscoveredReplica:
    model = entry.get("id")
    if not isinstance(model, str) or not model:
        raise LLMBackendError("llamacpp_embedding_invalid_model_catalog")
    identity, name = _tag_values(entry)
    return DiscoveredReplica(model, frozenset(catalog_aliases(entry)), identity, name)


def discover_embedding_pool(
    catalog: Any,
    selector: str,
    *,
    pinned: ReplicaIdentity | None = None,
    fallback_alias: str | None = None,
) -> EmbeddingPool:
    """Resolve an alias, a model basename or ``sha256:<content digest>``.

    Missing content metadata permits only a single exact alias. That fallback
    remains single-target for the adapter lifetime, even if metadata appears.
    """
    entries = catalog.get("data") if isinstance(catalog, dict) else None
    if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
        raise LLMBackendError("llamacpp_embedding_invalid_model_catalog")
    aliases = [entry for entry in entries if selector in catalog_aliases(entry)]
    if len(aliases) > 1:
        raise LLMBackendError("llamacpp_embedding_ambiguous_model")
    parsed: list[DiscoveredReplica] = []
    for entry in entries:
        if not is_loaded(entry):
            continue
        try:
            parsed.append(parse_replica(entry))
        except LLMBackendError:
            # A malformed exact target must not silently degrade to fallback.
            tags = entry.get("tags", [])
            related = entry in aliases
            if isinstance(tags, list):
                for tag in tags:
                    if not isinstance(tag, str):
                        continue
                    if tag == _CONTENT + selector.removeprefix(
                        "sha256:"
                    ) and selector.startswith("sha256:"):
                        related = True
                    if tag.startswith(_NAME):
                        try:
                            related = (
                                related
                                or unquote_to_bytes(tag[len(_NAME) :]).decode("utf-8")
                                == selector
                            )
                        except UnicodeError:
                            pass
            if related:
                raise
            # Malformed unrelated models are not candidates for equivalence.
    if len({replica.model for replica in parsed}) != len(parsed):
        raise LLMBackendError("llamacpp_embedding_ambiguous_model")
    if fallback_alias is not None:
        matches = [r for r in parsed if fallback_alias in r.aliases]
        if len(matches) != 1:
            raise LLMBackendError("llamacpp_embedding_model_not_loaded")
        # This adapter already operated without proven model identity. A later
        # metadata upgrade cannot retrospectively prove earlier vectors.
        replica = matches[0]
        return EmbeddingPool((replica,), None, fallback_alias)
    anchor = next((r for r in parsed if selector in r.aliases), None)
    if aliases and anchor is None and pinned is None:
        raise LLMBackendError("llamacpp_embedding_model_not_loaded")
    if anchor is not None:
        if pinned is not None and anchor.identity != pinned:
            raise LLMBackendError("llamacpp_embedding_content_identity_changed")
        if anchor.identity is None:
            return EmbeddingPool((anchor,), None, selector)
        identity = anchor.identity
    elif pinned is not None:
        identity = pinned
    else:
        if selector.startswith("sha256:"):
            digest = selector.removeprefix("sha256:")
            if not _SHA256.fullmatch(digest):
                raise LLMBackendError("llamacpp_embedding_invalid_model_selector")
            matches = [
                r for r in parsed if r.identity and r.identity.content_sha256 == digest
            ]
        else:
            matches = [r for r in parsed if r.model_name == selector]
        identities = {r.identity for r in matches}
        if not matches:
            raise LLMBackendError("llamacpp_embedding_model_not_loaded")
        if None in identities or len(identities) != 1:
            raise LLMBackendError("llamacpp_embedding_ambiguous_model_identity")
        identity = matches[0].identity
    replicas = tuple(
        sorted((r for r in parsed if r.identity == identity), key=lambda r: r.model)
    )
    if not replicas:
        raise LLMBackendError("llamacpp_embedding_model_not_loaded")
    # Alias collisions could route a request to a different catalogue entry.
    for replica in replicas:
        if sum(replica.model in catalog_aliases(entry) for entry in entries) != 1:
            raise LLMBackendError("llamacpp_embedding_ambiguous_model")
    return EmbeddingPool(replicas, identity)


def verify_loaded_replica(
    catalog: Any, model: str, identity: ReplicaIdentity | None
) -> None:
    entries = catalog.get("data") if isinstance(catalog, dict) else None
    if not isinstance(entries, list):
        raise LLMBackendError("llamacpp_embedding_invalid_model_catalog")
    matches = [
        e for e in entries if isinstance(e, dict) and model in catalog_aliases(e)
    ]
    if len(matches) != 1 or not is_loaded(matches[0]):
        raise LLMBackendError("llamacpp_embedding_model_not_loaded")
    parsed = parse_replica(matches[0])
    if identity is not None and parsed.identity != identity:
        raise LLMBackendError("llamacpp_embedding_content_identity_changed")
