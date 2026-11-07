"""Relay-owned routing, configuration generations and SDK invocations."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from threading import Event, RLock, Lock, get_ident
import time

from runtime_llama import (
    AutomaticEmbeddingService,
    build_adapter,
    build_embedding_adapter,
)
from runtime_llama.errors import LLMBackendError, LLMCancelledError
from runtime_llama.runtime_discovery import RuntimeEndpoint, discover_runtimes
from runtime_llama.tailnet import TailnetConfig, TailnetHelper
from runtime_llama.relay.config import RelayConfigStore
from runtime_llama.relay.opencode_config import apply_opencode_v2
from runtime_llama.relay.state import (
    RelayGeneration,
    RelayGenerationManager,
    RelayRequestRegistry,
)
from runtime_llama.relay.protocol import RelayError, map_error, usage_payload


class _Operation:
    def __init__(self, service, kind, model, external_cancel, request_id):
        self.service = service
        self.model = model
        self.lease = service.generations.acquire()
        self.snapshot = self.lease.snapshot
        try:
            self.request = service.registry.start(
                kind, model, self.snapshot.revision, request_id=request_id
            )
        except BaseException:
            self.lease.release()
            raise
        self.external_cancel = external_cancel
        self.helper_epoch = None
        self.helper_lease = self.resource_lease = None
        self.resource_generation = None
        self.captured_helper = (
            service._helper
            if any(
                item.get("use_tailnet")
                for item in self.snapshot.config["runtime_endpoints"]
            )
            else None
        )
        if self.captured_helper is not None:
            try:
                self.helper_lease = self.captured_helper[2].acquire()
            except BaseException:
                service.registry.finish(self.request.request_id, status="failed")
                self.lease.release()
                raise
        self.done = Event()
        self.owner_thread = get_ident()

    def cancelled(self):
        return self.request.cancelled() or bool(
            self.external_cancel and self.external_cancel.is_set()
        )

    def check(self):
        if self.cancelled():
            raise LLMCancelledError()

    def cancel(self):
        self.request.cancel_event.set()
        if self.external_cancel is not None:
            self.external_cancel.set()


class RelayService:
    """Explicit service owner; construction performs no network or helper start.

    Factories are dependency-injection boundaries for tests and integrations;
    production defaults always use the actual SDK, never a simulated backend.
    """

    def __init__(
        self,
        config_store=None,
        *,
        listener=None,
        adapter_factory=build_adapter,
        embedding_factory=build_embedding_adapter,
        automatic_embedding_factory=AutomaticEmbeddingService,
        discovery=discover_runtimes,
        tailnet_factory=TailnetHelper,
    ):
        self.config_store = (
            config_store if config_store is not None else RelayConfigStore()
        )
        self.registry = RelayRequestRegistry()
        self.generations = RelayGenerationManager(self.config_store.snapshot())
        self.listener = deepcopy(
            listener or self.config_store.snapshot().config["listener"]
        )
        self._adapter_factory = adapter_factory
        self._embedding_factory = embedding_factory
        self._automatic_embedding_factory = automatic_embedding_factory
        self._discovery = discovery
        self._tailnet_factory = tailnet_factory
        self._lock, self._tailnet_transition = RLock(), Lock()
        self._closed = False
        self._operations = {}
        self._cached = {}
        self._retired = set()
        self._retired_helpers = set()
        self._helper_resources = {}
        self._pending_helper_stops = {}
        self._helper_epochs = {}
        self._stopped_helper_epochs = set()
        self._helper = None
        self._helper_epoch = 0
        self._tailnet_state = "stopped"
        self._tailnet_error = None
        self._catalog = {
            "revision": self.config_store.snapshot().revision,
            "observed_at": None,
            "runtimes": [],
            "http_requests": 0,
            "elapsed_ms": None,
        }

    def _forget(self, generation):
        with self._lock:
            self._retired.discard(generation)
            self._retired_helpers.discard(generation)
            self._helper_resources.pop(generation, None)
            epoch = self._helper_epochs.pop(generation, None)
            if epoch is not None:
                self._stopped_helper_epochs.discard(epoch)
            for epoch, helper in tuple(self._pending_helper_stops.items()):
                if helper[2] is generation:
                    self._pending_helper_stops.pop(epoch, None)

    def _new_generation(self, snapshot):
        return RelayGeneration(snapshot, on_drained=self._forget)

    def _retire(self, generation):
        with self._lock:
            self._retired.add(generation)
        generation.retire()

    def _retire_helper(self, generation):
        with self._lock:
            self._retired_helpers.add(generation)
        self._retire(generation)

    @staticmethod
    def _wait_resource_drain(generation):
        # This observes closure; it does not retry close or add a request budget.
        while not generation.wait_drained(0.05):
            if generation.diagnostics()["close_error_types"]:
                raise RelayError(503, "relay_resource_close_failed")

    def cancel_request(self, request_id):
        with self._lock:
            operation = self._operations.get(request_id)
            if operation is None or not self.registry.cancel(request_id):
                return False
            operation.cancel()
            return True

    @contextmanager
    def _operation(self, kind, model, cancel_event=None, request_id=None):
        with self._lock:
            if self._closed:
                raise RelayError(503, "relay_service_closed")
            operation = _Operation(self, kind, model, cancel_event, request_id)
            self._operations[operation.request.request_id] = operation
        status, error_code, primary_failed = "completed", None, False
        try:
            operation.check()
            yield operation
            operation.check()
        except BaseException as error:
            primary_failed = True
            status = "cancelled" if isinstance(error, LLMCancelledError) else "failed"
            error_code = map_error(error).code
            raise
        finally:
            cleanup_error = None
            try:
                for lease in (
                    operation.resource_lease,
                    operation.helper_lease,
                    operation.lease,
                ):
                    if lease is not None:
                        try:
                            lease.release()
                        except BaseException as error:
                            if cleanup_error is None:
                                cleanup_error = error
                generation = operation.resource_generation
                if (
                    generation is not None
                    and generation.diagnostics()["close_error_types"]
                    and cleanup_error is None
                ):
                    cleanup_error = RelayError(503, "relay_resource_close_failed")
                if cleanup_error is not None and status == "completed":
                    status, error_code = "failed", map_error(cleanup_error).code
            finally:
                self.registry.finish(
                    operation.request.request_id,
                    status=status,
                    **({"error_code": error_code} if error_code else {}),
                )
                with self._lock:
                    self._operations.pop(operation.request.request_id, None)
                operation.done.set()
            if cleanup_error is not None and not primary_failed:
                raise cleanup_error

    def _endpoints(self, operation):
        snapshot = operation.snapshot
        endpoints, unavailable = [], []
        helper_lease, epoch = operation.helper_lease, None
        with self._lock:
            helper = operation.captured_helper
            if helper is not None:
                epoch = helper[0]
            if helper_lease is None or epoch in self._stopped_helper_epochs:
                helper = None
            for raw in snapshot.config["runtime_endpoints"]:
                value = {key: item for key, item in raw.items() if key != "use_tailnet"}
                if raw.get("use_tailnet"):
                    if helper is None:
                        unavailable.append(
                            {
                                "name": raw["name"],
                                "endpoint": {
                                    "name": raw["name"],
                                    "base_url": raw["base_url"],
                                    "use_tailnet": True,
                                },
                                "backend": None,
                                "available": False,
                                "diagnostics": ["tailnet_not_running"],
                                "models": [],
                            }
                        )
                        continue
                    epoch, node, _generation = helper
                    value["proxy_url"] = node.proxy_url
                endpoints.append(RuntimeEndpoint.from_value(value))
        return endpoints, unavailable, helper_lease, epoch

    def _discover_for(self, operation):
        endpoints, unavailable, helper_lease, helper_epoch = self._endpoints(operation)
        try:
            catalog = self._discovery(
                endpoints,
                timeout=operation.snapshot.settings(operation.model, "llm").get(
                    "runtime_discovery_timeout", 5.0
                ),
                cancellation_probe=operation.cancelled,
            )
            operation.check()
            rendered = self._render_catalog(operation.snapshot, catalog, unavailable)
            retired = []
            with self._lock:
                if (
                    self.config_store.snapshot().revision == operation.snapshot.revision
                    and not self._closed
                ):
                    self._catalog = rendered
                    loaded = {
                        (
                            runtime.endpoint.name,
                            item.id,
                            item.identity.content_sha256,
                            item.identity.embedding_sha256,
                        )
                        for runtime in catalog.runtimes
                        if runtime.available and runtime.catalog is not None
                        for item in runtime.catalog.loaded_models
                    }
                    stale = [
                        key
                        for key in self._cached
                        if key[0] == operation.snapshot.revision
                        and (key[1], key[2], key[4], key[5]) not in loaded
                    ]
                    retired = [self._cached.pop(key) for key in stale]
            for generation in retired:
                self._retire(generation)
            return catalog, helper_lease, helper_epoch, rendered
        except BaseException:
            if helper_lease is not None:
                helper_lease.release()
                operation.helper_lease = None
                operation.captured_helper = None
            raise

    @staticmethod
    def _render_catalog(snapshot, catalog, unavailable):
        endpoint_settings = {
            entry["name"]: entry for entry in snapshot.config["runtime_endpoints"]
        }
        rows = []
        for runtime in catalog.runtimes:
            models = []
            for model in runtime.catalog.models if runtime.catalog else ():
                context = (
                    model.loaded_context.value
                    if model.loaded_context is not None
                    else None
                )
                models.append(
                    {
                        "id": model.id,
                        "aliases": list(model.aliases),
                        "state": model.state,
                        "loaded": model.state == "loaded",
                        "failed": model.failed,
                        "capabilities": {
                            "chat": model.capabilities.chat,
                            "embeddings": model.capabilities.embeddings,
                        },
                        "opencode_chat": (
                            runtime.available
                            and model.state == "loaded"
                            and model.failed is not True
                            and model.capabilities.chat is not False
                            and (
                                runtime.backend != "strata"
                                or model.id == runtime.llm_model_id
                            )
                        ),
                        "context": context,
                        "loaded_context": context,
                        "training_context": model.training_context.value
                        if model.training_context is not None
                        else None,
                        "embedding_width": model.model_embedding_width.value
                        if model.model_embedding_width is not None
                        else None,
                    }
                )
            rows.append(
                {
                    "name": runtime.endpoint.name,
                    "endpoint": {
                        "name": runtime.endpoint.name,
                        "base_url": runtime.endpoint.base_url,
                        "use_tailnet": endpoint_settings[runtime.endpoint.name].get(
                            "use_tailnet", False
                        ),
                    },
                    "backend": runtime.backend,
                    "available": runtime.available,
                    "diagnostics": list(runtime.diagnostics),
                    "models": models,
                }
            )
        by_name = {row["name"]: row for row in (*rows, *unavailable)}
        return {
            "revision": snapshot.revision,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "runtimes": [
                by_name[item["name"]] for item in snapshot.config["runtime_endpoints"]
            ],
            "http_requests": catalog.http_requests,
            "elapsed_ms": catalog.elapsed_seconds * 1000.0,
        }

    def catalog_snapshot(self):
        with self._lock:
            return deepcopy(self._catalog)

    def refresh_catalog(self, cancel_event=None):
        with self._operation("catalog", "", cancel_event) as operation:
            _catalog, helper_lease, _epoch, rendered = self._discover_for(operation)
            try:
                return rendered
            finally:
                if helper_lease is not None:
                    helper_lease.release()

    def model_list(self, cancel_event=None):
        catalog = self.refresh_catalog(cancel_event)
        names = dict.fromkeys(
            model["id"]
            for runtime in catalog["runtimes"]
            for model in runtime["models"]
        )
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": created,
                    "owned_by": "llama-runtime",
                }
                for name in names
            ],
        }

    def _select(self, operation, model, purpose):
        catalog, helper_lease, helper_epoch, _rendered = self._discover_for(operation)
        try:
            selection = catalog.select(model, purpose=purpose)
            endpoint = selection.runtime.endpoint
            configured = next(
                item
                for item in operation.snapshot.config["runtime_endpoints"]
                if item["name"] == endpoint.name
            )
            with self._lock:
                if configured.get("use_tailnet"):
                    if (
                        helper_epoch in self._stopped_helper_epochs
                        or operation.captured_helper is None
                    ):
                        raise LLMCancelledError()
                    operation.helper_epoch = helper_epoch
                    operation.helper_lease, helper_lease = helper_lease, None
            self.registry.update(
                operation.request.request_id,
                backend=selection.backend,
                endpoint=endpoint.name,
                selected_model=selection.model.id,
            )
            return selection
        finally:
            if helper_lease is not None:
                helper_lease.release()
                operation.helper_lease = None
                operation.captured_helper = None

    def _per_call_adapter(self, operation, request, selection):
        endpoint = selection.runtime.endpoint
        settings = operation.snapshot.settings(request.model, "llm")
        if "response_format" in request.options:
            settings["output_mode"] = "text"
            settings.pop("json_schema", None)
        profile = {
            **settings,
            "model": selection.model.id,
            "backend": selection.backend,
            "base_url": endpoint.base_url,
            "api_key": endpoint.resolved_api_key(),
            "api_key_env": None,
            "verify_ssl": endpoint.verify_ssl,
            "proxy_url": endpoint.proxy_url,
            "http_connection_policy": endpoint.http_connection_policy,
        }
        generation = self._new_generation(operation.snapshot)
        operation.resource_generation = generation
        operation.resource_lease = generation.acquire()
        if operation.helper_epoch is not None:
            with self._lock:
                self._helper_resources[generation] = operation.helper_epoch
        try:
            adapter = operation.resource_lease.resource(
                "adapter", lambda: self._adapter_factory(profile)
            )
        finally:
            self._retire(generation)
        return adapter

    def _record_event(self, operation, event):
        updates = {}
        if event.usage is not None:
            updates["usage"] = usage_payload(event.usage)
        if event.server_timings is not None:
            updates["server_timings"] = event.server_timings.to_dict()
        if event.tool_recovery is not None:
            report = event.tool_recovery
            updates.update(
                corrections_used=report.corrections_used,
                reasons=[
                    reason for attempt in report.attempts for reason in attempt.reasons
                ],
                coordinator_elapsed_ms=report.elapsed_ms,
                observed_usage=usage_payload(report.observed_usage),
                total_usage=usage_payload(report.total_usage),
                usage_complete=report.usage_complete,
                unknown_usage_fields=list(report.unknown_usage_fields),
            )
        if getattr(event, "terminal_reason", ""):
            updates["terminal_reason"] = event.terminal_reason
        if updates:
            self.registry.update(operation.request.request_id, **updates)

    def chat(self, request, *, emit=None, cancel_event=None, request_id=None):
        terminal = None
        with self._operation(
            "chat", request.model, cancel_event, request_id
        ) as operation:
            selection = self._select(operation, request.model, "llm")
            adapter = self._per_call_adapter(operation, request, selection)
            options = {
                "request_options": request.options,
                "cancellation_probe": operation.cancelled,
            }
            if request.num_predict is not None:
                options["num_predict"] = request.num_predict
            if emit is None:
                result = adapter.generate_with_metadata(request.messages, **options)
                self._record_event(operation, result)
                return result
            events = adapter.stream_with_metadata(request.messages, **options)
            try:
                for event in events:
                    operation.check()
                    self._record_event(operation, event)
                    if event.kind == "terminal":
                        terminal = event
                    else:
                        emit(event)
            finally:
                events.close()
        # Success is not sent before per-invocation adapters have really closed.
        if terminal is not None:
            emit(terminal)

    def _embedding(self, operation, model, selection):
        endpoint = selection.runtime.endpoint
        identity = selection.model.identity
        verified = (
            identity.content_sha256 is not None
            and identity.embedding_sha256 is not None
        )
        settings = operation.snapshot.settings(model, "embedding")
        settings.setdefault("normalize", True)
        settings.setdefault("timeout", 120.0)
        input_type = settings.pop("input_type", "document")
        key = (
            operation.snapshot.revision,
            endpoint.name,
            selection.model.id,
            operation.helper_epoch,
            identity.content_sha256,
            identity.embedding_sha256,
        )
        with self._lock:
            if self._closed:
                raise LLMCancelledError()
            generation = self._cached.get(key)
            if generation is None:
                generation = self._new_generation(operation.snapshot)
                if self.config_store.snapshot().revision == operation.snapshot.revision:
                    self._cached[key] = generation
                if operation.helper_epoch is not None:
                    self._helper_resources[generation] = operation.helper_epoch
            operation.resource_lease = generation.acquire()
            operation.resource_generation = generation
            stale = [
                old_key
                for old_key in self._cached
                if old_key[:3] == key[:3] and old_key != key
            ]
            retired = [self._cached.pop(old_key) for old_key in stale]
        for old in retired:
            self._retire(old)

        def build():
            common = {
                "base_url": endpoint.base_url,
                "model": selection.model.id,
                "api_key": endpoint.resolved_api_key(),
                "verify_ssl": endpoint.verify_ssl,
                "proxy_url": endpoint.proxy_url,
                "http_connection_policy": endpoint.http_connection_policy,
            }
            if verified:
                return self._automatic_embedding_factory({**settings, **common})
            renamed = {
                "request_parallelism": "embedding_request_parallelism",
                "queue_size": "embedding_queue_size",
            }
            direct = {
                renamed.get(name, name): value for name, value in settings.items()
            }
            return self._embedding_factory(
                {**direct, **common, "embedding_replica_discovery": False}
            )

        adapter = operation.resource_lease.resource("embedding", build)
        with self._lock:
            still_cached = self._cached.get(key) is generation
        if not still_cached:
            self._retire(generation)
        return adapter, verified, input_type

    def embed(self, request, *, cancel_event=None, request_id=None):
        with self._operation(
            "embedding", request.model, cancel_event, request_id
        ) as operation:
            selection = self._select(operation, request.model, "embedding")
            adapter, verified, default_task = self._embedding(
                operation, request.model, selection
            )
            method = adapter.embed_texts if verified else adapter.embed_texts_direct
            return method(
                request.inputs,
                input_type=request.input_type or default_task,
                cancel=operation.cancelled,
            )

    def replace_config(self, expected_revision, config):
        with self._lock:
            if self._closed:
                raise RelayError(503, "relay_service_closed")
            old = self.config_store.snapshot()
            snapshot = self.config_store.replace(expected_revision, config)
            self.generations.replace(snapshot)
            retired = list(self._cached.values())
            self._cached.clear()
            helper = None
            if old.config["tailnet"] != snapshot.config["tailnet"]:
                helper, self._helper = self._helper, None
                self._helper_epoch += 1
                self._tailnet_state = "stopped"
            self._catalog = {
                "revision": snapshot.revision,
                "observed_at": None,
                "runtimes": [],
                "http_requests": 0,
                "elapsed_ms": None,
            }
        for generation in retired:
            self._retire(generation)
        if helper is not None:
            self._retire_helper(helper[2])
        return {
            **snapshot.public(),
            "restart_required": self._restart_required(snapshot),
        }

    def _restart_required(self, snapshot):
        desired = snapshot.config["listener"]
        return any(desired[name] != self.listener[name] for name in ("host", "port"))

    def status(self):
        with self._lock:
            snapshot = self.config_store.snapshot()
            return {
                "revision": snapshot.revision,
                "listener": deepcopy(self.listener),
                "restart_required": self._restart_required(snapshot),
                "tailnet": {
                    "configured": snapshot.config["tailnet"] is not None,
                    "state": self._tailnet_state,
                    "running": self._helper is not None,
                    "error_code": self._tailnet_error,
                },
                "requests": self.registry.snapshot(),
            }

    def start_tailnet(self):
        with self._tailnet_transition:
            with self._lock:
                if self._closed:
                    raise RelayError(503, "relay_service_closed")
                if self._helper is not None:
                    return self.status()["tailnet"]
                if self._pending_helper_stops:
                    raise RelayError(409, "relay_previous_tailnet_stop_failed")
                if any(
                    generation.diagnostics()["close_error_types"]
                    for generation in self._retired_helpers
                ):
                    raise RelayError(409, "relay_previous_tailnet_stop_failed")
                if any(
                    generation.diagnostics()["close_error_types"]
                    for generation in self._helper_resources
                ):
                    raise RelayError(409, "relay_previous_tailnet_stop_failed")
                snapshot = self.config_store.snapshot()
                config = snapshot.config["tailnet"]
                if config is None:
                    raise RelayError(400, "relay_tailnet_not_configured")
                self._tailnet_state = "starting"
            generation = self._new_generation(snapshot)
            lease = generation.acquire()
            try:
                helper = lease.resource(
                    "helper", lambda: self._tailnet_factory(TailnetConfig(**config))
                )
                helper.start()
                with self._lock:
                    if (
                        self._closed
                        or self.config_store.snapshot().config["tailnet"] != config
                    ):
                        raise RelayError(409, "relay_tailnet_config_changed")
                    self._helper_epoch += 1
                    self._helper = (self._helper_epoch, helper, generation)
                    self._helper_epochs[generation] = self._helper_epoch
                    self._tailnet_state, self._tailnet_error = "running", None
            except BaseException:
                self._retire_helper(generation)
                with self._lock:
                    self._tailnet_state, self._tailnet_error = (
                        "failed",
                        "relay_tailnet_start_failed",
                    )
                raise
            finally:
                lease.release()
            return self.status()["tailnet"]

    def stop_tailnet(self):
        with self._tailnet_transition:
            with self._lock:
                helper, self._helper = self._helper, None
                if helper is not None:
                    self._pending_helper_stops[helper[0]] = helper
                    self._stopped_helper_epochs.add(helper[0])
                stops = tuple(self._pending_helper_stops.values())
                self._tailnet_state = "stopping" if stops else "stopped"
            try:
                for epoch, _node, generation in stops:
                    with self._lock:
                        for operation in self._operations.values():
                            if operation.helper_epoch == epoch:
                                operation.cancel()
                        keys = [key for key in self._cached if key[3] == epoch]
                        for key in keys:
                            self._cached.pop(key)
                        owned = tuple(
                            item
                            for item, owner in self._helper_resources.items()
                            if owner == epoch
                        )
                    for item in owned:
                        self._retire(item)
                    for item in owned:
                        self._wait_resource_drain(item)
                    self._retire_helper(generation)
                    self._wait_resource_drain(generation)
                    with self._lock:
                        self._pending_helper_stops.pop(epoch, None)
                with self._lock:
                    retry = tuple(self._retired_helpers)
                for generation in retry:
                    generation.retire()
                    self._wait_resource_drain(generation)
            except RelayError:
                with self._lock:
                    self._tailnet_state, self._tailnet_error = (
                        "failed",
                        "relay_resource_close_failed",
                    )
                raise
            with self._lock:
                self._tailnet_state, self._tailnet_error = "stopped", None
            return self.status()["tailnet"]

    def close(self):
        with self._tailnet_transition:
            with self._lock:
                self._closed = True
                self.registry.close()
                operations = tuple(self._operations.values())
                for operation in operations:
                    operation.cancel()
                retired = tuple(self._cached.values())
                self._cached.clear()
                helper, self._helper = self._helper, None
                helper_generations = {
                    value[2] for value in self._pending_helper_stops.values()
                }
                helper_generations.update(self._retired_helpers)
                if helper is not None:
                    helper_generations.add(helper[2])
            for operation in operations:
                if operation.owner_thread != get_ident():
                    operation.done.wait()
            for generation in retired:
                self._retire(generation)
            with self._lock:
                remaining = tuple(self._retired - helper_generations)
            # Proxy-using adapters stop before the helper owning their credentials.
            for generation in remaining:
                generation.retire()
                self._wait_resource_drain(generation)
            for generation in helper_generations:
                self._retire_helper(generation)
                self._wait_resource_drain(generation)
            self.generations.close()
            if not self.generations.wait_drained(0):
                raise RuntimeError("relay_configuration_drain_incomplete")

    def export_opencode(self, version, models, base_url):
        if version not in {"v1", "v2"}:
            raise RelayError(400, "relay_opencode_version_unsupported")
        catalog = self.catalog_snapshot()
        names = dict.fromkeys(
            model["id"]
            for runtime in catalog["runtimes"]
            if runtime["available"]
            for model in runtime["models"]
            if model["loaded"] and model["capabilities"]["chat"] is not False
        )
        selected = list(names) if not models else list(dict.fromkeys(models))
        if not selected:
            raise RelayError(400, "relay_catalog_has_no_available_chat_models")
        if any(model not in names for model in selected):
            raise RelayError(400, "relay_opencode_model_not_available")
        entries = {model: {"name": model} for model in selected}
        access = self.config_store.snapshot().config["access"]
        return self._opencode_config(version, entries, base_url, access)

    def export_configured_opencode(
        self, payload, base_url, cancel_event=None, *, apply=False
    ):
        if not isinstance(payload, dict) or payload.keys() != {
            "version",
            "catalog_revision",
            "default_model",
            "models",
        }:
            raise RelayError(400, "relay_opencode_export_shape_invalid")
        version, revision = payload["version"], payload["catalog_revision"]
        if apply and version != "v2":
            raise RelayError(400, "relay_opencode_apply_v2_required")
        if version not in ("v1", "v2"):
            raise RelayError(400, "relay_opencode_version_unsupported")
        if type(revision) is not int or revision < 0:
            raise RelayError(400, "relay_opencode_catalog_revision_invalid")
        models, default = payload["models"], payload["default_model"]
        if not isinstance(models, dict) or not models:
            raise RelayError(400, "relay_opencode_model_selection_invalid")
        if not isinstance(default, str) or default not in models:
            raise RelayError(400, "relay_opencode_default_model_invalid")
        for limits in models.values():
            self._validate_opencode_limits(limits)

        entries = {}
        with self._operation("catalog", "", cancel_event) as operation:
            if operation.snapshot.revision != revision:
                raise RelayError(409, "relay_opencode_catalog_changed")
            catalog, _lease, _epoch, _rendered = self._discover_for(operation)
            for model, limits in models.items():
                operation.check()
                try:
                    selected = catalog.select(model, purpose="llm").model
                except LLMBackendError:
                    raise RelayError(
                        409, "relay_opencode_model_not_available"
                    ) from None
                if selected.id != model:
                    raise RelayError(409, "relay_opencode_model_not_available")
                observed = selected.loaded_context
                effective = observed.value if observed is not None else None
                if effective != limits["observed_context"]:
                    raise RelayError(409, "relay_opencode_context_changed")
                entries[model] = {
                    "name": model,
                    "limit": {"context": limits["context"], "output": limits["output"]},
                }
            with self._lock:
                if self.config_store.snapshot().revision != revision:
                    raise RelayError(409, "relay_opencode_catalog_changed")
                exported = self._opencode_config(
                    version,
                    entries,
                    base_url,
                    operation.snapshot.config["access"],
                    default,
                )
                if not apply:
                    return exported
                operation.check()
                config = exported["config"]
                return {
                    **apply_opencode_v2(config),
                    "model": config["model"],
                    "warnings": exported["warnings"],
                }

    @staticmethod
    def _validate_opencode_limits(limits):
        if not isinstance(limits, dict) or limits.keys() != {
            "observed_context",
            "context",
            "output",
        }:
            raise RelayError(400, "relay_opencode_model_limits_invalid")
        observed, context, output = (
            limits["observed_context"],
            limits["context"],
            limits["output"],
        )
        if (
            type(context) is not int
            or context <= 0
            or (
                observed is not None
                and (type(observed) is not int or observed != context)
            )
        ):
            raise RelayError(400, "relay_opencode_context_invalid")
        if type(output) is not int or not 0 < output < context:
            raise RelayError(400, "relay_opencode_output_invalid")

    @staticmethod
    def _opencode_config(version, entries, base_url, access, default=None):
        authenticated = bool(access.get("api_key") or access.get("api_key_env"))
        environment_name = access.get("api_key_env") or "LLAMA_RELAY_API_KEY"
        api_key = "{env:" + environment_name + "}" if authenticated else "relay-local"
        warnings = []
        if authenticated:
            warnings.append(
                f"Set {environment_name} in OpenCode's environment; the relay key is never exported."
            )
        if default is None:
            warnings.append(
                "Model limits/capabilities not proved by the runtime are not invented; review the selected OpenCode version's defaults."
            )
        else:
            warnings.append(
                "Review OpenCode's defaults for capabilities the relay did not observe."
            )
        if version == "v1":
            config = {
                "$schema": "https://opencode.ai/config.json",
                "provider": {
                    "llama-relay": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "llama-runtime relay",
                        "options": {"baseURL": base_url, "apiKey": api_key},
                        "models": entries,
                    }
                },
            }
        else:
            config = {
                "$schema": "https://opencode.ai/config.json",
                "providers": {
                    "llama-relay": {
                        "package": "@opencode/ai/providers/openai-compatible",
                        "name": "llama-runtime relay",
                        "settings": {"baseURL": base_url, "apiKey": api_key},
                        "models": entries,
                    }
                },
            }
        if default is not None:
            config["model"] = f"llama-relay/{default}"
        return {
            "version": version,
            "config": config,
            "filename": f"opencode-relay-{version}.json",
            "warnings": warnings,
        }
