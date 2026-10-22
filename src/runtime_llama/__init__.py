"""Functional contracts for an externally managed llama.cpp HTTP server."""

from runtime_llama.contracts import (
    BaseLLMAdapter,
    CancellationProbe,
    EmbeddingAdapter,
    LLMAdapter,
)
from runtime_llama.configuration import (
    AdapterProfile,
    LlamaServerConfig,
    LlamaGenerationConfig,
    LlamaRoleConfig,
    ModelRoleProfile,
    EmbeddingPrefixes,
    EmbeddingReplicaConfig,
)
from runtime_llama.embedding_preparation import (
    EmbeddingInputType,
    PreparedEmbeddingInput,
)
from runtime_llama.embedding_stream import EmbeddingWindowStream
from runtime_llama.diagnostics import ThinkingDiagnostic, ThinkingDiagnosticsConfig
from runtime_llama.errors import (
    LLMBackendError,
    LLMCancelledError,
    LLMRunawayError,
    LLMTimeoutError,
)
from runtime_llama.usage import (
    LLMCallResult,
    LLMStreamEvent,
    LLMTransportAttempt,
    LLMUsageStats,
    LLMServerTimings,
)
from runtime_llama.mcp import (
    MCPCallResult,
    MCPEphemeralIntegration,
    MCPIntegration,
    MCPInvalidToolCall,
    MCPInvocationOptions,
    MCPPluginIntegration,
    MCPProviderInfo,
    MCPStreamEvent,
    MCPToolCall,
)
from runtime_llama.client import ClientHTTPError, LlamaCppClient
from runtime_llama.model_discovery import (
    ModelCapabilities,
    ModelCatalog,
    ModelIdentity,
    ModelObservation,
    ModelState,
    ObservationSource,
    ObservedInt,
    discover_models,
)
from runtime_llama.adapters.llama import LlamaCppAdapter
from runtime_llama.adapters.embedding import LlamaCppEmbeddingAdapter
from runtime_llama.factory import build_adapter, build_embedding_adapter
from runtime_llama.tailnet import TailnetConfig, TailnetError, TailnetHelper

__version__ = "0.10.3"

__all__ = [
    "TailnetConfig",
    "TailnetError",
    "TailnetHelper",
    "BaseLLMAdapter",
    "CancellationProbe",
    "EmbeddingAdapter",
    "LLMAdapter",
    "AdapterProfile",
    "LlamaServerConfig",
    "LlamaGenerationConfig",
    "LlamaRoleConfig",
    "ModelRoleProfile",
    "EmbeddingPrefixes",
    "EmbeddingInputType",
    "PreparedEmbeddingInput",
    "EmbeddingWindowStream",
    "EmbeddingReplicaConfig",
    "ThinkingDiagnostic",
    "ThinkingDiagnosticsConfig",
    "LLMBackendError",
    "LLMCancelledError",
    "LLMRunawayError",
    "LLMTimeoutError",
    "LLMCallResult",
    "LLMStreamEvent",
    "LLMTransportAttempt",
    "LLMUsageStats",
    "LLMServerTimings",
    "MCPCallResult",
    "MCPEphemeralIntegration",
    "MCPIntegration",
    "MCPInvalidToolCall",
    "MCPInvocationOptions",
    "MCPPluginIntegration",
    "MCPProviderInfo",
    "MCPStreamEvent",
    "MCPToolCall",
    "ClientHTTPError",
    "LlamaCppClient",
    "ModelCapabilities",
    "ModelCatalog",
    "ModelIdentity",
    "ModelObservation",
    "ModelState",
    "ObservationSource",
    "ObservedInt",
    "discover_models",
    "LlamaCppAdapter",
    "LlamaCppEmbeddingAdapter",
    "build_adapter",
    "build_embedding_adapter",
]
