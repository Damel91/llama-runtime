"""Source-attributed audit context at the relay's first-party boundaries."""

from dataclasses import dataclass, field
import sqlite3
from threading import Lock

from runtime_llama.relay.audit_store import (
    AuditStore,
    RETENTION_DAYS,
    RETENTION_EVENTS,
    audit_label,
    audit_model_label,
)


def declared_id(value):
    """Treat opaque client labels as declared metadata, never as identity."""
    return audit_label(value)


class RelaySecurityAudit:
    """Private store availability is independent of inference availability."""

    def __init__(self, config_path):
        path = (
            config_path.with_name(config_path.name + ".audit.sqlite3")
            if config_path is not None
            else None
        )
        self._lock = Lock()
        self._dropped = 0
        self._mode = "persistent" if path is not None else "memory"
        try:
            self.store = AuditStore(path)
        except (OSError, ValueError, sqlite3.Error):
            self.store = None

    def append(self, **fields):
        if self.store is None:
            return
        try:
            self.store.append(**fields)
        except (OSError, ValueError, sqlite3.Error):
            with self._lock:
                self._dropped += 1

    def observation(self):
        with self._lock:
            return {
                "storage": self._mode if self.store is not None else "unavailable",
                "dropped_events": self._dropped,
                "retention_days": RETENTION_DAYS,
                "retention_events": RETENTION_EVENTS,
                "coverage": {
                    "model_proposal": "relay_native",
                    "model_output": "relay_native",
                    "relay_decision": "relay",
                    "executor_dispatch": None,
                    "executor_outcome": None,
                    "network_connection": None,
                },
            }

    def close(self):
        if self.store is not None:
            self.store.close()


@dataclass
class AuditContext:
    """Request-local metadata. The context never receives prompts or arguments."""

    audit: RelaySecurityAudit
    request_id: str
    session_id: str | None = None
    correlation_id: str | None = None
    model: str | None = None
    runtime: str | None = None
    _approved: dict = field(default_factory=dict, init=False)

    def route(self, model, runtime):
        self.model, self.runtime = audit_model_label(model), audit_label(runtime)

    def record(self, phase, outcome, *, source, **fields):
        for key in ("tool_call_id", "tool_name"):
            if key in fields:
                fields[key] = audit_label(fields[key])
        if phase == "relay_decision" and outcome == "approved":
            if fields.get("reason") == "tool_policy_accepted":
                fields["verified_scope"] = "declared_tool_schema"
            call_id = declared_id(fields.get("tool_call_id"))
            if call_id is not None:
                self._approved[call_id] = {
                    key: fields[key]
                    for key in (
                        "tool_name",
                        "policy_version",
                        "attempt_index",
                        "proposal_index",
                    )
                    if key in fields
                }
        self.audit.append(
            request_id=self.request_id,
            correlation_id=self.correlation_id or self.request_id,
            session_id=self.session_id,
            session_source="client_declared" if self.session_id else None,
            model=self.model,
            runtime=self.runtime,
            phase=phase,
            outcome=outcome,
            source=source,
            **fields,
        )

    def forwarded(self, identifiers):
        for identifier in identifiers:
            call_id = declared_id(identifier)
            approved = self._approved.pop(call_id, None)
            self.record(
                "relay_decision",
                "forwarded",
                source="relay_transport",
                tool_call_id=call_id if approved is not None else None,
                reason="relay_handoff",
                **(approved or {}),
            )
