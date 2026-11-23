/* Security audit view: filters, paged reads, event cards and JSONL export.
   Pagination cursors and filter parameters keep the baseline contract. */

import { $, node, formatNumber, detailPair } from "./dom.js";
import { state } from "./state.js";
import { api } from "./api.js";
import { renderRequestCards } from "./activity.js";

const auditPhases = {
  model_proposal: "Model proposal",
  model_output: "Model output",
  relay_decision: "Relay decision",
  executor_dispatch: "Executor dispatch",
  executor_outcome: "Executor outcome",
  network_connection: "Outbound connection",
};
const auditSources = {
  relay_native: "Native runtime",
  relay: "Relay",
  relay_policy: "Relay policy",
  relay_transport: "Relay transport",
  executor: "Executor",
  network_monitor: "Network monitor",
};

function auditTime(value) {
  return value ? value.replace("T", " ").replace(/(?:Z|\+00:00)$/, " UTC") : "Unknown";
}

function auditPath(path, filters, before = null) {
  const query = new URLSearchParams(filters);
  if (before !== null) query.set("before", String(before));
  return query.size ? `${path}?${query}` : path;
}

function auditFilters() {
  const filters = {};
  for (const [name, value] of new FormData($("audit-filter"))) {
    const text = value.trim();
    if (!text) continue;
    filters[name] = name === "from" || name === "to" ? new Date(`${text}Z`).toISOString() : text;
  }
  if (filters.from && filters.to && filters.from > filters.to) throw new Error("From must be at or before To.");
  return filters;
}

function auditEventCard(event) {
  const card = node("article", "telemetry-request audit-event");
  const detail = node("details");
  const heading = node("summary");
  const time = node("time", "audit-timestamp", auditTime(event.timestamp));
  time.dateTime = event.timestamp;
  const outcome = event.outcome.replaceAll("_", " ");
  const negative = ["blocked", "denied", "failed"].includes(event.outcome);
  heading.append(
    node("strong", "", auditPhases[event.phase] || event.phase),
    node("span", "cell-secondary", [event.tool_name, event.model].filter(Boolean).join(" · ") || "No tool or model recorded"),
    node("span", `state-pill${negative ? " error" : ""}`, outcome),
    time,
  );
  const facts = node("dl", "detail-grid");
  const host = event.destination_host;
  const destination = host && (event.destination_port ? `${host.includes(":") ? `[${host}]` : host}:${event.destination_port}` : host);
  const sessionSource = {client_declared: "Declared by client", trusted_executor: "Trusted executor"};
  for (const [label, value] of [
    ["Source", auditSources[event.source] || event.source], ["Runtime", event.runtime], ["Reason", event.reason],
    ["Event ID", event.id], ["Request ID", event.request_id], ["Correlation ID", event.correlation_id],
    ["Session ID", event.session_id], ["Session source", sessionSource[event.session_source]], ["Tool call ID", event.tool_call_id],
    ["Policy version", event.policy_version], ["Verified scope", event.verified_scope],
    ["Attempt index", event.attempt_index], ["Proposal index", event.proposal_index],
    ["Destination", destination], ["Destination class", event.destination_class], ["Connector", event.connector_id],
  ]) if (value !== undefined && value !== null) detailPair(facts, label, value);
  detail.append(heading, facts);
  card.append(detail);
  return card;
}

function renderAuditControls() {
  const {data, stale, checkedAt, dirty, operation, cursors} = state.audit;
  $("audit-apply").disabled = operation !== null;
  $("audit-refresh").disabled = dirty || operation !== null;
  $("audit-export").disabled = dirty || operation !== null || !data || data.storage === "unavailable";
  $("audit-previous").disabled = dirty || operation !== null || cursors.length < 2;
  $("audit-next").disabled = dirty || operation !== null || data?.next_before == null;
  $("audit-progress").hidden = operation === null;
  $("audit-progress").textContent = operation === "read" ? "Reading audit…" : operation === "export" ? "Preparing JSONL…" : "";
  $("audit-draft").hidden = !dirty;
  $("audit-observed").textContent = stale
    ? checkedAt ? `Update failed. Showing the page read at ${auditTime(checkedAt)}.` : "Update failed. Audit is unavailable."
    : checkedAt ? `Read at ${auditTime(checkedAt)}.` : "Apply filters or refresh to read the audit.";
}

export function renderAudit() {
  renderAuditControls();
  const {data, cursors} = state.audit;
  if (!data) return;
  const storageLabels = {persistent: "Persistent storage", memory: "In memory", unavailable: "Audit storage unavailable"};
  $("audit-storage").textContent = storageLabels[data.storage] || "Storage unknown";
  $("audit-retention").textContent = `Retention up to ${formatNumber(data.retention_days)} days and ${formatNumber(data.retention_events)} events.`;
  $("audit-dropped").textContent = `${formatNumber(data.dropped_events)} events dropped.`;
  $("audit-coverage").hidden = false;
  const coverage = document.createDocumentFragment();
  for (const [phase, label] of Object.entries(auditPhases)) {
    const item = node("div");
    const source = data.coverage[phase];
    detailPair(item, label, source ? auditSources[source] || source : "Unobserved");
    coverage.append(item);
  }
  $("audit-coverage").replaceChildren(coverage);
  $("audit-empty").textContent = data.storage === "unavailable" ? "No audit page is available." : "No retained events match these filters.";
  $("audit-empty").hidden = data.events.length > 0;
  $("audit-page").textContent = `Page ${cursors.length} · ${formatNumber(data.events.length)} events`;
  renderRequestCards($("audit-event-list"), data.events, auditEventCard, "id");
}

function auditError(error) {
  const message = error.code === "relay_audit_filter_invalid" ? "Check the filters. Use a hostname or IP and an optional port for Destination."
    : error.code === "relay_audit_storage_unavailable" ? "Audit storage is unavailable."
    : error.message || "The audit request failed.";
  $("audit-error").textContent = message;
  $("audit-error").hidden = false;
}

async function readAudit(filters = state.audit.filters, cursors = state.audit.cursors) {
  const audit = state.audit;
  if (audit.operation) return;
  const epoch = state.pageEpoch;
  const edit = audit.edit;
  audit.operation = "read";
  $("audit-error").hidden = true;
  renderAuditControls();
  try {
    const page = await api(auditPath("/audit", filters, cursors.at(-1)));
    if (epoch !== state.pageEpoch) return;
    audit.data = page;
    audit.filters = filters;
    audit.cursors = cursors;
    audit.checkedAt = new Date().toISOString();
    audit.dirty = edit !== audit.edit;
    audit.stale = false;
  } catch (error) {
    if (epoch !== state.pageEpoch) return;
    audit.stale = true;
    auditError(error);
  } finally {
    if (epoch === state.pageEpoch) { audit.operation = null; renderAudit(); }
  }
}

async function exportAudit() {
  const audit = state.audit;
  if (audit.operation || audit.dirty || !audit.data || audit.data.storage === "unavailable") return;
  const epoch = state.pageEpoch;
  audit.operation = "export";
  $("audit-error").hidden = true;
  renderAuditControls();
  try {
    const blob = await api(auditPath("/audit/export", audit.filters), {format: "jsonl"});
    if (epoch !== state.pageEpoch) return;
    const url = URL.createObjectURL(blob);
    const link = node("a");
    link.href = url;
    link.download = "relay-security-audit.jsonl";
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  } catch (error) {
    if (epoch === state.pageEpoch) auditError(error);
  } finally {
    if (epoch === state.pageEpoch) { audit.operation = null; renderAuditControls(); }
  }
}

export function bindAudit() {
  const audit = state.audit;
  $("open-audit").addEventListener("click", () => $("audit-heading").focus());
  $("audit-filter").addEventListener("input", () => {
    audit.edit += 1;
    audit.dirty = true;
    renderAuditControls();
  });
  $("audit-filter").addEventListener("submit", (event) => {
    event.preventDefault();
    try { void readAudit(auditFilters(), [null]); }
    catch (error) { auditError(error); }
  });
  $("audit-refresh").addEventListener("click", () => void readAudit());
  $("audit-export").addEventListener("click", () => void exportAudit());
  $("audit-previous").addEventListener("click", () => void readAudit(audit.filters, audit.cursors.slice(0, -1)));
  $("audit-next").addEventListener("click", () => void readAudit(audit.filters, [...audit.cursors, audit.data.next_before]));
  renderAudit();
}
