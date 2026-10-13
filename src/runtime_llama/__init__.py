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
from runtime_llama.adapters.llama import LlamaCppAdapter
from runtime_llama.adapters.embedding import LlamaCppEmbeddingAdapter
from runtime_llama.factory import build_adapter, build_embedding_adapter

__version__ = "0.4.1"

__all__ = [
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
    "LlamaCppAdapter",
    "LlamaCppEmbeddingAdapter",
    "build_adapter",
    "build_embedding_adapter",
]
