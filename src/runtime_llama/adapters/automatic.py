"""Request-local LLM routing from observed loaded models, never model loading."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
from threading import Event, Lock, get_ident
from typing import Any, Iterator, Mapping, Optional

from runtime_llama._proxy import proxy_secrets
from runtime_llama.adapters.llama import LlamaCppAdapter
from runtime_llama.adapters.strata import StrataAdapter
from runtime_llama.client import CancellationProbe, LlamaCppClient, _cancelled
from runtime_llama.configuration import AdapterProfile
from runtime_llama.errors import LLMBackendError, LLMCancelledError
from runtime_llama.diagnostics import ThinkingDiagnosticsConfig, ThinkingDiagnosticSink
from runtime_llama.runtime_discovery import (
    RuntimeCatalog,
    RuntimeEndpoint,
    RuntimeModelSelection,
    discover_runtimes,
)
from runtime_llama.usage import LLMStreamEvent


@dataclass(eq=False)
class _Operation:
    owner_thread: int
    stop: Event = field(default_factory=Event)
    discovery_done: Event = field(default_factory=Event)
    client: LlamaCppClient | None = None


class AutoRuntimeAdapter(LlamaCppAdapter):
    """Choose llama.cpp or Strata for the requested, already loaded model.

    Endpoint configuration belongs to deployment. A new read-only probe selects
    the backend for each generation; that selection is fixed for the entire
    request and included in its normalized result/events. Concurrent requests
    never change one another's route. Catalog/selection properties describe the
    latest observation, not a future readiness guarantee or an active lease.
    """

    def __init__(self, profile: AdapterProfile) -> None:
        if profile.adapter_type != "llm" or profile.backend != "auto":
            raise ValueError("automatic runtime adapter requires an auto LLM profile")
        self._runtime_endpoints = profile.runtime_endpoints or (
            RuntimeEndpoint(
                name="default",
                base_url=profile.base_url,
                api_key=profile.generation.api_key,
                api_key_env=profile.generation.api_key_env,
                verify_ssl=profile.model_config.verify_ssl,
                proxy_url=profile.model_config.proxy_url,
                http_connection_policy=profile.model_config.http_connection_policy,
            ),
        )
        self._runtime_lock = Lock()
        self._runtime_closed = False
        self._runtime_operations: set[_Operation] = set()
        self._current_catalog: RuntimeCatalog | None = None
        self._selected_runtime: RuntimeModelSelection | None = None
        super().__init__(
            model=profile.model,
            role_id=profile.role_id,
            role_profile=profile,
            model_config=profile.model_config,
            role_config=profile.role,
            generation_config=profile.generation,
        )
        # Parent initialization establishes the common API/guards without any
        # network I/O. Actual endpoint clients belong to individual operations.
        self._client.close()

    @property
    def _diagnostic_secrets(self) -> tuple[str, ...]:
        secrets = self._base_diagnostic_secrets
        for endpoint in self._runtime_endpoints:
            token = endpoint.api_key or (
                os.environ.get(endpoint.api_key_env) if endpoint.api_key_env else None
            )
            secrets += ((token,) if token else ()) + proxy_secrets(endpoint.proxy_url)
        return secrets

    @_diagnostic_secrets.setter
    def _diagnostic_secrets(self, value: tuple[str, ...]) -> None:
        self._base_diagnostic_secrets = value

    @classmethod
    def from_profile(
        cls,
        profile: AdapterProfile,
        *,
        client: LlamaCppClient | None = None,
    ) -> AutoRuntimeAdapter:
        if client is not None:
            raise ValueError("automatic runtime routing owns its endpoint clients")
        return cls(profile)

    @property
    def current_catalog(self) -> RuntimeCatalog | None:
        with self._runtime_lock:
            return self._current_catalog

    @property
    def selected_runtime(self) -> RuntimeModelSelection | None:
        with self._runtime_lock:
            return self._selected_runtime

    def _begin(self, cancel: CancellationProbe | None) -> _Operation:
        _cancelled(cancel)
        operation = _Operation(get_ident())
        with self._runtime_lock:
            if self._runtime_closed:
                raise LLMBackendError("runtime_adapter_closed")
            self._runtime_operations.add(operation)
            self._current_catalog = None
            self._selected_runtime = None
        return operation

    @staticmethod
    def _cancel(
        operation: _Operation, cancel: CancellationProbe | None
    ) -> CancellationProbe:
        def cancelled() -> bool:
            if operation.stop.is_set():
                return True
            requested = bool(cancel and cancel())
            return requested or operation.stop.is_set()

        return cancelled

    def _raise_if_recovery_retired(self) -> None:
        with self._runtime_lock:
            closed = self._runtime_closed
        if closed:
            raise LLMCancelledError()

    def _finish(self, operation: _Operation) -> None:
        operation.discovery_done.set()
        with self._runtime_lock:
            client = operation.client
        try:
            if client is not None:
                client.close()
        finally:
            # Remain discoverable by concurrent close() through actual drain.
            # Removing the operation before client.close() would let close()
            # return while an owned reader/session was still being cleaned up.
            with self._runtime_lock:
                operation.client = None
                self._runtime_operations.discard(operation)

    def _discover(
        self,
        operation: _Operation,
        cancel: CancellationProbe | None,
        timeout: float | None = None,
    ) -> RuntimeCatalog:
        try:
            catalog = discover_runtimes(
                self._runtime_endpoints,
                timeout=(
                    self.role_profile.runtime_discovery_timeout
                    if timeout is None
                    else timeout
                ),
                cancellation_probe=self._cancel(operation, cancel),
            )
            _cancelled(self._cancel(operation, cancel))
            with self._runtime_lock:
                if self._runtime_closed:
                    raise LLMCancelledError()
                self._current_catalog = catalog
                self._selected_runtime = None
            return catalog
        finally:
            operation.discovery_done.set()

    def discover_models(
        self,
        *,
        timeout: float | None = None,
        cancellation_probe: CancellationProbe | None = None,
    ) -> RuntimeCatalog:
        """Probe every configured endpoint and list each observed model catalog."""
        operation = self._begin(cancellation_probe)
        try:
            return self._discover(operation, cancellation_probe, timeout)
        finally:
            self._finish(operation)

    def _iter_stream_events_with_ctx(
        self,
        messages: Any,
        *,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        output_format: Optional[str] = None,
        timeout: Optional[float] = None,
        cancellation_probe: Optional[CancellationProbe] = None,
        request_options: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[LLMStreamEvent]:
        yield from self._iter_selected_events(
            messages,
            num_ctx=num_ctx,
            num_predict=num_predict,
            output_format=output_format,
            timeout=timeout,
            cancellation_probe=cancellation_probe,
            request_options=request_options,
        )

    def _iter_invocation_events(
        self,
        messages: Any,
        *,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        output_format: Optional[str] = None,
        timeout: Optional[float] = None,
        cancellation_probe: Optional[CancellationProbe] = None,
        request_options: Optional[Mapping[str, Any]] = None,
        thinking_diagnostics: Optional[ThinkingDiagnosticsConfig] = None,
        thinking_sink: Optional[ThinkingDiagnosticSink] = None,
    ) -> Iterator[LLMStreamEvent]:
        yield from self._iter_selected_events(
            messages,
            num_ctx=num_ctx,
            num_predict=num_predict,
            output_format=output_format,
            timeout=timeout,
            cancellation_probe=cancellation_probe,
            request_options=request_options,
            invocation_projection=True,
            thinking_diagnostics=thinking_diagnostics,
            thinking_sink=thinking_sink,
        )

    def _iter_selected_events(
        self,
        messages: Any,
        *,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        output_format: Optional[str] = None,
        timeout: Optional[float] = None,
        cancellation_probe: Optional[CancellationProbe] = None,
        request_options: Optional[Mapping[str, Any]] = None,
        invocation_projection: bool = False,
        thinking_diagnostics: Optional[ThinkingDiagnosticsConfig] = None,
        thinking_sink: Optional[ThinkingDiagnosticSink] = None,
    ) -> Iterator[LLMStreamEvent]:
        idle = self._stream_idle(timeout)
        self._raise_if_cancelled(cancellation_probe, timeout=idle)
        operation = self._begin(cancellation_probe)
        cancel = self._cancel(operation, cancellation_probe)
        events = None
        selection = None
        try:
            catalog = self._discover(operation, cancellation_probe)
            selection = catalog.select(self.model_name)
            endpoint = selection.runtime.endpoint
            token = endpoint.resolved_api_key()
            observed = selection.model.loaded_context
            model_config = replace(
                self._model_config,
                model=selection.model.id,
                base_url=endpoint.base_url,
                context_length=(observed.value if observed is not None else None),
                runtime_identity=endpoint.name,
                verify_ssl=endpoint.verify_ssl,
                proxy_url=endpoint.proxy_url,
                http_connection_policy=endpoint.http_connection_policy,
            )
            selected_profile = replace(
                self.role_profile,
                model_config=model_config,
                generation=replace(
                    self._generation_config, api_key=token, api_key_env=None
                ),
                backend_id=selection.backend,
                runtime_endpoints=(),
            )
            client = LlamaCppClient(
                endpoint.base_url,
                api_key=token,
                timeout=30.0 if idle is None else idle,
                verify_ssl=endpoint.verify_ssl,
                trust_env=False,
                proxy_url=endpoint.proxy_url,
                connection_policy=endpoint.http_connection_policy,
            )
            with self._runtime_lock:
                closed = self._runtime_closed or operation.stop.is_set()
                if not closed:
                    operation.client = client
                    self._current_catalog = catalog
                    self._selected_runtime = selection
            if closed:
                client.close()
                raise LLMCancelledError()
            adapter_type = (
                StrataAdapter if selection.backend == "strata" else LlamaCppAdapter
            )
            child = adapter_type.from_profile(selected_profile, client=client)
            child._requested_context_ceiling = self._requested_context_ceiling
            child._observed_context_length = (
                observed.value if observed is not None else None
            )
            generate_events = (
                child._iter_invocation_events
                if invocation_projection
                else child._iter_stream_events_with_ctx
            )
            events = generate_events(
                messages,
                num_ctx=num_ctx,
                num_predict=num_predict,
                output_format=output_format,
                timeout=timeout,
                cancellation_probe=cancel,
                request_options=request_options,
                **(
                    {
                        "thinking_diagnostics": thinking_diagnostics,
                        "thinking_sink": thinking_sink,
                    }
                    if invocation_projection
                    else {}
                ),
            )
            for event in events:
                _cancelled(cancel)
                yield replace(event, runtime_selection=selection)
        except BaseException as error:
            if selection is not None and not isinstance(error, GeneratorExit):
                error.runtime_selection = selection
            raise
        finally:
            try:
                close = getattr(events, "close", None)
                if callable(close):
                    close()
            finally:
                self._finish(operation)

    def health_status(self) -> dict[str, Any]:
        """Read readiness from the same complete discovery used for routing."""
        selection = None
        reason = "runtime_model_unavailable"
        operation = self._begin(None)
        try:
            catalog = self._discover(operation, None)
            try:
                selection = catalog.select(self.model_name)
                reason = "ready"
            except LLMBackendError as exc:
                reason = str(exc)
            observed = selection.model.loaded_context if selection is not None else None
            context = observed.value if observed is not None else None
            with self._runtime_lock:
                if self._runtime_closed or operation.stop.is_set():
                    raise LLMCancelledError()
                self._current_catalog = catalog
                self._selected_runtime = selection
                self._observed_context_length = context
            ready = selection is not None
            healthy = any(item.available for item in catalog.runtimes)
            available = any(
                model.id == self.model_name or self.model_name in model.aliases
                for runtime in catalog.runtimes
                if runtime.catalog is not None
                for model in runtime.catalog.models
            )
            return {
                "backend": selection.backend if ready else "auto",
                "role_id": self.role_id,
                "model": self.model_name,
                "healthy": healthy,
                "server_healthy": healthy,
                "backend_ready": ready,
                "model_available": available,
                "model_ready": ready,
                "model_state": "loaded" if ready else "unknown",
                "reason": reason,
                "loaded_context_length": context,
                "configured_context_length": self._model_config.context_length,
                "runtime_selection": selection,
                "process_lifecycle": "external",
                "model_loading": "server_owned",
            }
        finally:
            self._finish(operation)

    def close(self) -> None:
        """Stop discovery and drain active HTTP even for a paused generator."""
        with self._runtime_lock:
            self._runtime_closed = True
            operations = tuple(self._runtime_operations)
            for operation in operations:
                operation.stop.set()
            clients = tuple(
                operation.client
                for operation in operations
                if operation.client is not None
            )
        for client in clients:
            client.close()
        for operation in operations:
            if operation.owner_thread != get_ident():
                operation.discovery_done.wait()
        with self._runtime_lock:
            self._runtime_operations.difference_update(operations)
            self._selected_runtime = None
            self._current_catalog = None


__all__ = ["AutoRuntimeAdapter"]
