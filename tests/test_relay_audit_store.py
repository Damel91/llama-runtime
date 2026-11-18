from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import os
import sqlite3
import stat

import pytest

from runtime_llama.relay import audit_store
from runtime_llama.relay.audit_store import AuditStore, audit_label, audit_model_label


def proposal(store, **fields):
    return store.append(
        source="relay_native",
        phase="model_proposal",
        outcome="proposed",
        **fields,
    )


def test_persistence_and_private_sqlite_ownership(tmp_path):
    path = tmp_path / "private" / "config.json.audit.sqlite3"
    first = AuditStore(path)
    event = proposal(
        first,
        request_id="request-one",
        correlation_id="trace-1",
        session_id="session-1",
        session_source="client_declared",
        tool_call_id="call-1",
        model="Qwen/Qwen.3.8",
        runtime="strata",
        tool_name="read_file",
        attempt_index=0,
        proposal_index=0,
    )
    assert first._connection.execute("PRAGMA secure_delete").fetchone()[0] == 1
    first.close()
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    second = AuditStore(path)
    try:
        assert second.query({"session_id": "session-1"}) == ([event], None)
        assert second.export_rows({"tool_name": "read_file"}) == [event]
        assert second.export_rows({"model": "Qwen/Qwen.3.8"}) == [event]
        assert proposal(second)["id"] > event["id"]
        assert datetime.fromisoformat(event["timestamp"]).tzinfo == timezone.utc
    finally:
        second.close()


def test_paging_and_filters_are_consistent_during_append(monkeypatch):
    clock = datetime(2026, 11, 18, tzinfo=timezone.utc)
    monkeypatch.setattr(audit_store, "_now", lambda: clock)
    store = AuditStore()
    try:
        first = proposal(
            store, session_id="session-1", session_source="client_declared"
        )
        proposal(store, session_id="session-2", session_source="client_declared")
        clock += timedelta(days=1)
        third = proposal(
            store, session_id="session-1", session_source="client_declared"
        )
        filters = {
            "session_id": "session-1",
            "phase": "model_proposal",
            "outcome": "proposed",
        }
        page, before = store.query(filters, limit=1)
        assert page == [third] and before == third["id"]
        latest = proposal(
            store, session_id="session-1", session_source="client_declared"
        )
        assert store.query(filters, limit=1, before=before) == ([first], None)
        assert store.export_rows(
            {**filters, "from": "2026-11-19", "to": "2026-11-19"}
        ) == [latest, third]
        assert store.export_rows(
            {
                "from": "2026-11-18T01:00:00+01:00",
                "to": "2026-11-18T00:00:00Z",
                **filters,
            }
        ) == [first]
    finally:
        store.close()


def test_count_and_age_retention_is_applied_at_write_read_and_reopen(
    tmp_path, monkeypatch
):
    clock = datetime(2026, 11, 18, tzinfo=timezone.utc)
    monkeypatch.setattr(audit_store, "_now", lambda: clock)
    monkeypatch.setattr(audit_store, "RETENTION_EVENTS", 3)
    path = tmp_path / "audit.sqlite3"
    store = AuditStore(path)
    try:
        rows = [proposal(store) for _ in range(5)]
        assert store.export_rows({}) == list(reversed(rows[-3:]))
        assert (
            store._connection.execute("SELECT count(*) FROM relay_audit").fetchone()[0]
            == 3
        )
        clock += timedelta(days=audit_store.RETENTION_DAYS, microseconds=1)
        assert store.query({}) == ([], None)
        after_prune = proposal(store)
        assert after_prune["id"] > rows[-1]["id"]
    finally:
        store.close()

    clock += timedelta(days=audit_store.RETENTION_DAYS, microseconds=1)
    reopened = AuditStore(path)
    try:
        assert (
            reopened._connection.execute("SELECT count(*) FROM relay_audit").fetchone()[
                0
            ]
            == 0
        )
        assert reopened.export_rows({}) == []
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "host, canonical, classification, match",
    [
        ("127.0.0.1", "127.0.0.1", "loopback", "127.0.0.1:443"),
        ("10.23.45.67", "10.23.45.67", "private", "10.23.45.67"),
        ("100.64.1.2", "100.64.1.2", "private", "100.64.1.2"),
        ("::ffff:10.23.45.67", "10.23.45.67", "private", "[::ffff:10.23.45.67]:443"),
        ("2001:4860:0::1", "2001:4860::1", "public", "[2001:4860::1]:443"),
        (
            "WORKSTATION.example.",
            "workstation.example",
            None,
            "workstation.example:443",
        ),
        ("192.0.2.10", "192.0.2.10", None, "192.0.2.10"),
    ],
)
def test_trusted_network_destination_is_canonical_and_classified_locally(
    host, canonical, classification, match
):
    store = AuditStore()
    try:
        row = store.append(
            source="network_monitor",
            phase="network_connection",
            outcome="connected",
            destination_host=host,
            destination_port=443,
            reason="network_connected",
            connector_id="connector-1",
        )
        assert row["destination_host"] == canonical
        assert row["destination_class"] == classification
        assert store.query({"destination": match}) == ([row], None)
        assert store.query({"destination": "workstation.invalid"}) == ([], None)
    finally:
        store.close()


@pytest.mark.parametrize(
    "fields",
    [
        {"arguments": "PRIVATE_ARGUMENTS"},
        {"thinking": "PRIVATE_THINKING"},
        {"authorization": "PRIVATE_HEADER"},
        {"model": "https://private.invalid/model"},
        {"tool_name": "/home/private/model"},
        {"tool_call_id": "sk-privatekey"},
        {"reason": "arbitrary_prose_word"},
        {"attempt_index": True},
        {"proposal_index": 2**63},
        {"session_id": "unattributed-claim"},
        {"session_id": "untrusted-auth", "session_source": "trusted_executor"},
        {"destination_host": "10.23.45.67"},
        {"verified_scope": "declared_tool_schema"},
    ],
)
def test_unrecognized_or_unsafe_fields_do_not_persist(fields):
    store = AuditStore()
    try:
        with pytest.raises(ValueError, match="^relay_audit_") as caught:
            proposal(store, **fields)
        for raw in fields.values():
            assert str(raw) not in str(caught.value)
        assert store.export_rows({}) == []
    finally:
        store.close()


def test_authority_phases_and_unknown_fields_stay_distinct():
    store = AuditStore()
    try:
        literal = store.append(
            source="relay_native",
            phase="model_output",
            outcome="literal_markup",
            reason="literal_tool_markup",
        )
        approved = store.append(
            source="relay_policy",
            phase="relay_decision",
            outcome="approved",
            reason="tool_policy_accepted",
            verified_scope="declared_tool_schema",
        )
        executed = store.append(
            source="executor",
            phase="executor_outcome",
            outcome="completed",
            reason="executor_completed",
            session_id="executor-session",
            session_source="trusted_executor",
        )
        assert store.query({"phase": "model_output"}) == ([literal], None)
        assert store.export_rows({"outcome": "completed"}) == [executed]
        assert approved["verified_scope"] == "declared_tool_schema"
        assert not executed["destination_host"]
        with pytest.raises(ValueError, match="relay_audit_event_kind_invalid"):
            store.append(
                source="relay_native", phase="executor_outcome", outcome="completed"
            )
        with pytest.raises(ValueError, match="relay_audit_event_kind_invalid"):
            store.append(
                source="relay_transport", phase="relay_decision", outcome="approved"
            )
        with pytest.raises(ValueError, match="relay_audit_event_kind_invalid"):
            store.append(
                source="relay_policy", phase="relay_decision", outcome="forwarded"
            )
        with pytest.raises(ValueError, match="relay_audit_destination_invalid"):
            store.append(
                source="network_monitor",
                phase="network_connection",
                outcome="connected",
                destination_host="10.23.45.67",
                destination_port=443,
                destination_class="public",
            )
    finally:
        store.close()


@pytest.mark.parametrize(
    "filters, options",
    [
        ({"unexpected": "fixture"}, {}),
        ({"phase": ["model_proposal"]}, {}),
        ({"outcome": "made_up"}, {}),
        ({"destination": "https://private.invalid"}, {}),
        ({"destination": "workstation.invalid:0"}, {}),
        ({"from": "2026-11-18T00:00:00"}, {}),
        ({"from": "2026-11-18\u0000T00:00:00Z"}, {}),
        ({"from": "2026-11-19", "to": "2026-11-18"}, {}),
        ({}, {"before": True}),
        ({}, {"before": 2**63}),
        ({}, {"limit": 101}),
    ],
)
def test_invalid_filters_are_rejected_without_sql_or_reflection(filters, options):
    store = AuditStore()
    try:
        with pytest.raises(ValueError, match="^relay_audit_") as caught:
            store.query(filters, **options)
        assert "private.invalid" not in str(caught.value)
    finally:
        store.close()


def test_store_rejects_foreign_database_and_readable_existing_file(tmp_path):
    path = tmp_path / "not-an-audit.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE unrelated (id INTEGER)")
    connection.close()
    path.chmod(0o600)
    original = path.read_bytes()
    with pytest.raises(OSError, match="relay_audit_database_invalid"):
        AuditStore(path)
    assert path.read_bytes() == original
    if os.name == "posix":
        path.chmod(0o644)
        with pytest.raises(OSError, match="relay_audit_file_not_private"):
            AuditStore(path)
        assert path.read_bytes() == original


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions and symlinks")
@pytest.mark.parametrize("sidecar", ["", "-journal", "-shm", "-wal"])
def test_symlinked_database_and_sidecars_are_rejected(tmp_path, sidecar):
    path = tmp_path / "audit.sqlite3"
    target = tmp_path / "unrelated"
    target.write_bytes(b"untouched")
    (tmp_path / (path.name + sidecar)).symlink_to(target)
    with pytest.raises(OSError, match="relay_audit_file_invalid"):
        AuditStore(path)
    assert target.read_bytes() == b"untouched"


def test_concurrent_append_and_read_without_event_loss(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite3")

    def append(index):
        proposal(store, request_id=f"request-{index}")
        page, _ = store.query({}, limit=10)
        assert page == sorted(page, key=lambda row: row["id"], reverse=True)

    try:
        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(append, range(25)))
        assert {row["request_id"] for row in store.export_rows({})} == {
            f"request-{index}" for index in range(25)
        }
    finally:
        store.close()


def test_optional_label_projection_allows_identifiers_and_omits_unsafe_values():
    assert audit_label("request:trace-1") == "request:trace-1"
    assert audit_label("http://private.invalid") is None
    assert audit_label("sk-strata-sensitive") is None
    assert audit_label("typing a title") is None
    assert audit_label("x" * 161) is None
    assert audit_model_label("Qwen/Qwen.3.8:tag") == "Qwen/Qwen.3.8:tag"
    assert audit_model_label("/Qwen/Qwen.3.8") is None
    assert audit_model_label("../Qwen.3.8") is None
    assert audit_model_label("C:/Qwen.3.8") is None
