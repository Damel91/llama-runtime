"""Private, bounded SQLite history for content-free relay audit events."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time, timedelta, timezone
from ipaddress import ip_address, ip_network
import os
from pathlib import Path
import re
import sqlite3
import stat
from threading import Lock


RETENTION_DAYS = 30
RETENTION_EVENTS = 20_000
_PAGE_SIZE = 100
_APPLICATION_ID = 0x4C524155
_SCHEMA_VERSION = 1

_FIELDS = (
    "request_id",
    "correlation_id",
    "session_id",
    "session_source",
    "tool_call_id",
    "model",
    "runtime",
    "source",
    "tool_name",
    "phase",
    "outcome",
    "reason",
    "policy_version",
    "verified_scope",
    "attempt_index",
    "proposal_index",
    "destination_host",
    "destination_port",
    "destination_class",
    "connector_id",
)
_NUMBER_FIELDS = {"attempt_index", "proposal_index", "destination_port"}
_REQUIRED_FIELDS = {"source", "phase", "outcome"}
_COLUMNS = ("id", "timestamp", *_FIELDS)
_COLUMN_SQL = ", ".join(_COLUMNS)
_INSERT_SQL = (
    f"INSERT INTO relay_audit (timestamp, {', '.join(_FIELDS)}) "
    f"VALUES ({', '.join('?' for _ in range(len(_FIELDS) + 1))})"
)
_FILTER_FIELDS = {
    "from",
    "to",
    "session_id",
    "model",
    "tool_name",
    "phase",
    "outcome",
    "destination",
}

_SOURCE_PHASES = {
    "relay_native": {"model_proposal", "model_output"},
    "relay_policy": {"relay_decision"},
    "relay_transport": {"relay_decision"},
    "executor": {"executor_dispatch", "executor_outcome"},
    "network_monitor": {"network_connection"},
}
_PHASE_OUTCOMES = {
    "model_proposal": {"proposed"},
    "model_output": {"literal_markup"},
    "relay_decision": {"approved", "blocked", "withheld", "forwarded"},
    "executor_dispatch": {"dispatched", "blocked", "denied", "failed"},
    "executor_outcome": {
        "completed",
        "succeeded",
        "failed",
        "cancelled",
        "blocked",
        "denied",
        "unknown",
    },
    "network_connection": {"observed", "connected", "failed", "blocked", "denied"},
}
_OUTCOMES = frozenset().union(*_PHASE_OUTCOMES.values())
_REASONS = frozenset(
    {
        "cancelled",
        "connection_reset",
        "content_filter",
        "early_eof",
        "executor_cancelled",
        "executor_completed",
        "executor_dispatched",
        "executor_failed",
        "executor_policy_denied",
        "idle_timeout",
        "incomplete_tool_response",
        "invalid_tool_arguments_json",
        "length_cap",
        "literal_tool_markup",
        "llama_conflicting_tool_call_identity",
        "llama_duplicate_tool_call_identity",
        "llama_malformed_completed_tool_call",
        "llama_malformed_tool_call_delta",
        "llama_malformed_tool_call_function",
        "llama_missing_tool_call_identity",
        "llama_missing_tool_call_index",
        "malformed_completed_tool_call",
        "malformed_completed_tool_calls",
        "malformed_native_tool_delta",
        "malformed_native_tool_rejection",
        "native_attempt_interrupted",
        "native_rejected_tool_call",
        "native_tool_protocol_accepted",
        "network_blocked",
        "network_connected",
        "network_connection_failed",
        "network_observed",
        "no_declared_tool_authority",
        "queue_wait_timeout",
        "relay_handoff",
        "relay_incomplete_generation",
        "relay_internal_error",
        "relay_malformed_sdk_tool_delta",
        "relay_request_cancelled",
        "relay_upstream_error",
        "relay_upstream_timeout",
        "repair_tool_call_missing",
        "required_tool_call_missing",
        "server_error",
        "server_side_tool_effects_possible",
        "specific_tool_choice_violated",
        "strata_tool_call_incomplete",
        "synthetic_dispatch",
        "synthetic_success",
        "tool_arguments_not_object",
        "tool_arguments_schema_mismatch",
        "tool_choice_none_violated",
        "tool_finish_without_calls",
        "tool_policy_accepted",
        "tool_policy_blocked",
        "tool_protocol_error",
        "tool_protocol_rejected",
        "tool_recovery_exhausted",
        "tool_recovery_incomplete",
        "tool_recovery_unsafe",
        "tool_schema_unresolvable",
        "undeclared_tool_name",
        "upstream_server_error",
        "visible_output_cap",
    }
)
_SESSION_SOURCES = {"client_declared", "trusted_executor"}
_LABEL = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*(?::[A-Za-z0-9_][A-Za-z0-9_.-]*)*")
_SECRET_PREFIX = re.compile(
    r"(?:sk-|gh[pousr]_|github_pat_|xox[baprs]-|tskey-|eyJ[A-Za-z0-9_-]*\.)", re.I
)
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_PRIVATE_NETWORKS = tuple(
    ip_network(value)
    for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
)
_SHARED_NETWORK = ip_network("100.64.0.0/10")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def audit_label(value: object) -> str | None:
    """Return a bounded audit label, omitting titles, paths and secret forms."""
    if (
        not isinstance(value, str)
        or len(value) > 160
        or not _LABEL.fullmatch(value)
        or _SECRET_PREFIX.match(value)
    ):
        return None
    return value


def audit_model_label(value: object) -> str | None:
    """Admit also a verified runtime catalog's single namespace/model label."""
    if isinstance(value, str) and len(value) <= 160 and value.count("/") == 1:
        namespace, model = value.split("/")
        if ":" not in namespace and audit_label(namespace) and audit_label(model):
            return value
    return audit_label(value)


def _label(value: object) -> str:
    label = audit_label(value)
    if label is None:
        raise ValueError("relay_audit_label_invalid")
    return label


def _destination_host(value: object) -> tuple[str, str | None]:
    if not isinstance(value, str) or len(value) > 254 or "%" in value:
        raise ValueError("relay_audit_destination_invalid")
    try:
        address = ip_address(value)
    except ValueError:
        host = value.lower().removesuffix(".")
        if not (
            host.isascii()
            and len(host) <= 253
            and any(char.isalpha() for char in host)
            and all(_DNS_LABEL.fullmatch(label) for label in host.split("."))
            and not _SECRET_PREFIX.match(host)
        ):
            raise ValueError("relay_audit_destination_invalid") from None
        return host, None

    address = getattr(address, "ipv4_mapped", None) or address
    if address.is_loopback:
        classification = "loopback"
    elif (
        address.is_link_local
        or address in _SHARED_NETWORK
        or any(address in network for network in _PRIVATE_NETWORKS)
    ):
        classification = "private"
    else:
        classification = (
            "public" if address.is_global and not address.is_multicast else None
        )
    return str(address), classification


def _destination_port(value: object) -> int:
    if type(value) is not int or not 1 <= value <= 65535:
        raise ValueError("relay_audit_destination_invalid")
    return value


def _destination_filter(value: object) -> tuple[str, int | None]:
    if not isinstance(value, str) or len(value) > 262:
        raise ValueError("relay_audit_destination_invalid")
    if value.startswith("["):
        match = re.fullmatch(r"\[([a-fA-F0-9:.]+)\](?::([0-9]{1,5}))?", value)
        if match is None or ":" not in match[1]:
            raise ValueError("relay_audit_destination_invalid")
        host, port = match.groups()
    elif value.count(":") == 1:
        host, port = value.rsplit(":", 1)
    else:
        host, port = value, None
    normalized, _class = _destination_host(host)
    if port is None:
        return normalized, None
    if not re.fullmatch(r"[0-9]{1,5}", port):
        raise ValueError("relay_audit_destination_invalid")
    return normalized, _destination_port(int(port))


def _event(fields: dict) -> dict:
    if fields.keys() - set(_FIELDS) or not _REQUIRED_FIELDS <= fields.keys():
        raise ValueError("relay_audit_fields_invalid")
    value = {key: fields.get(key) for key in _FIELDS}
    source, phase, outcome = (value[key] for key in ("source", "phase", "outcome"))
    if not all(type(item) is str for item in (source, phase, outcome)) or (
        phase not in _SOURCE_PHASES.get(source, ())
        or outcome not in _PHASE_OUTCOMES[phase]
    ):
        raise ValueError("relay_audit_event_kind_invalid")

    for key in (
        "request_id",
        "correlation_id",
        "session_id",
        "tool_call_id",
        "runtime",
        "tool_name",
        "policy_version",
        "verified_scope",
        "connector_id",
    ):
        if value[key] is not None:
            value[key] = _label(value[key])
    if value["model"] is not None and audit_model_label(value["model"]) is None:
        raise ValueError("relay_audit_label_invalid")
    if phase == "relay_decision" and (
        (source == "relay_transport") != (outcome == "forwarded")
    ):
        raise ValueError("relay_audit_event_kind_invalid")
    session_source = value["session_source"]
    if session_source is not None and (
        type(session_source) is not str or session_source not in _SESSION_SOURCES
    ):
        raise ValueError("relay_audit_session_source_invalid")
    if (value["session_id"] is None) != (session_source is None):
        raise ValueError("relay_audit_session_source_invalid")
    if session_source == "trusted_executor" and source != "executor":
        raise ValueError("relay_audit_session_source_invalid")
    reason = value["reason"]
    if reason is not None and (type(reason) is not str or reason not in _REASONS):
        raise ValueError("relay_audit_reason_invalid")
    scope = value["verified_scope"]
    if (
        scope is not None
        and source.startswith("relay_")
        and (source, outcome, reason, scope)
        != ("relay_policy", "approved", "tool_policy_accepted", "declared_tool_schema")
    ):
        raise ValueError("relay_audit_verified_scope_invalid")
    for key in ("attempt_index", "proposal_index"):
        index = value[key]
        if index is not None and (
            type(index) is not int or not 0 <= index <= 2**63 - 1
        ):
            raise ValueError("relay_audit_index_invalid")

    host, port, classification = (
        value[key]
        for key in ("destination_host", "destination_port", "destination_class")
    )
    if source == "network_monitor":
        host, computed = _destination_host(host)
        port = _destination_port(port)
        if classification is not None and classification != computed:
            raise ValueError("relay_audit_destination_invalid")
        value.update(
            destination_host=host, destination_port=port, destination_class=computed
        )
    elif any(item is not None for item in (host, port, classification)):
        raise ValueError("relay_audit_destination_requires_monitor")
    return value


def _date_boundary(value: object, *, upper: bool) -> str:
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError("relay_audit_time_invalid")
    try:
        if _ISO_DATE.fullmatch(value):
            boundary = datetime.combine(
                date.fromisoformat(value), time.max if upper else time.min, timezone.utc
            )
        else:
            if "T" not in value:
                raise ValueError
            boundary = datetime.fromisoformat(value)
            if boundary.tzinfo is None:
                raise ValueError
        return _timestamp(boundary)
    except (ValueError, OverflowError):
        raise ValueError("relay_audit_time_invalid") from None


def _criteria(
    filters: Mapping[str, object], before: int | None = None
) -> tuple[str, tuple]:
    if not isinstance(filters, Mapping) or filters.keys() - _FILTER_FIELDS:
        raise ValueError("relay_audit_filters_invalid")
    conditions, args = [], []

    def add(condition, value):
        conditions.append(condition)
        args.append(value)

    if before is not None:
        if type(before) is not int or not 0 < before < 2**63:
            raise ValueError("relay_audit_cursor_invalid")
        add("id < ?", before)
    for key in ("session_id", "tool_name"):
        if key in filters:
            add(f"{key} = ?", _label(filters[key]))
    if "model" in filters:
        model = audit_model_label(filters["model"])
        if model is None:
            raise ValueError("relay_audit_label_invalid")
        add("model = ?", model)
    for key, choices in (("phase", _PHASE_OUTCOMES), ("outcome", _OUTCOMES)):
        if key in filters:
            value = filters[key]
            if type(value) is not str or value not in choices:
                raise ValueError("relay_audit_filter_value_invalid")
            add(f"{key} = ?", value)
    since = _date_boundary(filters["from"], upper=False) if "from" in filters else None
    until = _date_boundary(filters["to"], upper=True) if "to" in filters else None
    if since is not None:
        add("timestamp >= ?", since)
    if until is not None:
        add("timestamp <= ?", until)
    if since is not None and until is not None and since > until:
        raise ValueError("relay_audit_time_range_invalid")
    if "destination" in filters:
        host, port = _destination_filter(filters["destination"])
        add("destination_host = ?", host)
        if port is not None:
            add("destination_port = ?", port)
    return (" WHERE " + " AND ".join(conditions) if conditions else "", tuple(args))


def _validate_file(info: os.stat_result) -> None:
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise OSError("relay_audit_file_invalid")
    if os.name == "posix" and (
        info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise OSError("relay_audit_file_not_private")


def _open_file(path: Path) -> tuple[os.stat_result, bool]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = path.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or (
        os.name == "posix" and (parent.st_uid != os.geteuid() or parent.st_mode & 0o022)
    ):
        raise OSError("relay_audit_directory_not_private")
    for suffix in ("-journal", "-wal", "-shm"):
        try:
            info = path.with_name(path.name + suffix).lstat()
        except FileNotFoundError:
            continue
        _validate_file(info)
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        new = True
    except FileExistsError:
        _validate_file(path.lstat())
        descriptor = os.open(path, flags)
        new = False
    try:
        if new and os.name == "posix":
            os.fchmod(descriptor, 0o600)
        info = os.fstat(descriptor)
        _validate_file(info)
        if not new and not info.st_size:
            raise OSError("relay_audit_file_invalid")
        return info, new
    finally:
        os.close(descriptor)


class AuditStore:
    """Accept typed local events and retain a bounded, restart-safe history."""

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self._lock = Lock()
        if path is None:
            connection = sqlite3.connect(":memory:", check_same_thread=False)
            new = True
        else:
            file_path = Path(path).expanduser().absolute()
            info, new = _open_file(file_path)
            connection = sqlite3.connect(
                file_path.as_uri() + "?mode=rw", uri=True, check_same_thread=False
            )
        try:
            if path is not None:
                current = file_path.lstat()
                _validate_file(current)
                if not os.path.samestat(current, info):
                    raise OSError("relay_audit_file_changed")
            connection.row_factory = sqlite3.Row
            self._connection = connection
            if new:
                self._initialize()
            else:
                self._validate_schema()
            with connection:
                connection.execute("PRAGMA secure_delete = ON")
                self._prune(_now())
        except BaseException:
            connection.close()
            raise

    def _initialize(self) -> None:
        fields = ", ".join(
            key
            + (" INTEGER" if key in _NUMBER_FIELDS else " TEXT")
            + (" NOT NULL" if key in _REQUIRED_FIELDS else "")
            for key in _FIELDS
        )
        with self._connection as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE relay_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                f"timestamp TEXT NOT NULL, {fields})"
            )
            connection.execute(
                "CREATE INDEX relay_audit_time ON relay_audit (timestamp)"
            )
            connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    def _validate_schema(self) -> None:
        connection = self._connection
        application_id = connection.execute("PRAGMA application_id").fetchone()[0]
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        columns = tuple(
            row["name"] for row in connection.execute("PRAGMA table_info(relay_audit)")
        )
        if (application_id, version, columns) != (
            _APPLICATION_ID,
            _SCHEMA_VERSION,
            _COLUMNS,
        ):
            raise OSError("relay_audit_database_invalid")

    def _prune(self, now: datetime) -> None:
        connection = self._connection
        connection.execute(
            "DELETE FROM relay_audit WHERE timestamp < ?",
            (_timestamp(now - timedelta(days=RETENTION_DAYS)),),
        )
        connection.execute(
            "DELETE FROM relay_audit WHERE id <= (SELECT max(id) FROM relay_audit) - ?",
            (RETENTION_EVENTS,),
        )

    def append(self, **fields: object) -> dict:
        """Validate one typed observation and persist it without raw payloads."""
        fields = _event(fields)
        with self._lock, self._connection as connection:
            now = _now()
            timestamp = _timestamp(now)
            identifier = connection.execute(
                _INSERT_SQL, (timestamp, *(fields[key] for key in _FIELDS))
            ).lastrowid
            self._prune(now)
        return {"id": identifier, "timestamp": timestamp, **fields}

    def query(
        self,
        filters: Mapping[str, object],
        *,
        before: int | None = None,
        limit: int = _PAGE_SIZE,
    ) -> tuple[list[dict], int | None]:
        """Read newest first; the next cursor excludes the last returned ID."""
        if type(limit) is not int or not 1 <= limit <= _PAGE_SIZE:
            raise ValueError("relay_audit_page_size_invalid")
        where, args = _criteria(filters, before)
        with self._lock, self._connection as connection:
            self._prune(_now())
            rows = connection.execute(
                f"SELECT {_COLUMN_SQL} FROM relay_audit{where} ORDER BY id DESC LIMIT ?",
                (*args, limit + 1),
            ).fetchall()
        page = [dict(item) for item in rows[:limit]]
        return page, page[-1]["id"] if len(rows) > limit else None

    def export_rows(self, filters: Mapping[str, object]) -> list[dict]:
        """Return one filtered snapshot of the complete retained window."""
        where, args = _criteria(filters)
        with self._lock, self._connection as connection:
            self._prune(_now())
            rows = connection.execute(
                f"SELECT {_COLUMN_SQL} FROM relay_audit{where} ORDER BY id DESC", args
            ).fetchall()
        return [dict(item) for item in rows]

    def close(self) -> None:
        with self._lock:
            self._connection.close()
