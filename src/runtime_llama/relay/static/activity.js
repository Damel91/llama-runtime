/* Requests and telemetry views: status line, request cards, retained recent
   observations and the telemetry tables. */

import { $, node, button, unknown, formatNumber, formatMs, formatDate, detailPair, notice } from "./dom.js";
import { state, busy, refreshLocalState } from "./state.js";
import { api } from "./api.js";

const requestCards = new WeakMap();

export function renderRequestObservation() {
  $("requests-observed").textContent = state.requestsStale
    ? state.requestsObserved ? `Update failed. Showing the snapshot from ${formatDate(state.requestsObserved)}.` : "Requests are unavailable. Try refreshing."
    : state.requestsObserved ? `Updated ${formatDate(state.requestsObserved)}.` : "Waiting for the first snapshot.";
}

export function renderStatus() {
  const available = state.connection === "connected";
  $("service-state").textContent = available ? "Connected" : state.connection === "connecting" ? "Connecting" : "Connection unavailable";
  $("service-state").className = `status-tag${available ? " ready" : state.connection === "unavailable" ? " error" : ""}`;
  $("api-url").textContent = `${window.location.origin}/v1`;
  $("footer-state").textContent = available
    ? `Local relay connected. Saved configuration revision ${state.status?.revision ?? state.revision}.`
    : state.connection === "connecting" ? "Connecting to the local relay." : "The local administration connection is unavailable.";
  if (!state.status) return;
  $("restart-notice").hidden = state.status.restart_required !== true;
  const helper = state.status.tailnet || {};
  const helperState = helper.state || helper.status || (helper.running === true ? "Running" : helper.running === false ? "Stopped" : "Unknown");
  $("tailnet-state").textContent = helperState;
  $("tailnet-state").className = `status-tag${["running", "ready"].includes(String(helperState).toLowerCase()) ? " ready" : ""}`;
  $("start-tailnet").disabled = !state.config?.tailnet || ["running", "ready", "starting"].includes(String(helperState).toLowerCase());
  $("stop-tailnet").disabled = ["stopped", "not_configured", "disabled"].includes(String(helperState).toLowerCase());
}

function requestCard(summary, active) {
  const card = node("article", "request-summary"); const header = node("div", "request-summary-header"); const identity = node("div");
  identity.append(node("h3", "", summary.selected_model || summary.model || "Model not reported"), node("p", "technical-value", `${summary.kind || "Request"} · ${summary.request_id || "ID not reported"}`));
  header.append(identity);
  if (active) { const cancel = button(summary.status === "cancelling" ? "Stopping" : "Stop request", () => busy(cancel, async () => { await api(`/requests/${encodeURIComponent(summary.request_id)}/cancel`, {method: "POST"}); notice("Cancellation requested for this invocation."); await refreshLocalState({afterPending: true}).catch(() => {}); }), "danger", true); cancel.disabled = summary.status === "cancelling"; header.append(cancel); }
  else header.append(node("span", `state-pill${["completed", "tool_calls"].includes(summary.status) ? " ready" : ["failed", "error"].includes(summary.status) ? " error" : ""}`, summary.status || "Unknown outcome"));
  card.append(header); const grid = node("dl", "detail-grid");
  for (const [label, value] of [
    ["Status", summary.status], ["Started", formatDate(summary.started_at)], ["Runtime", summary.backend], ["Connection", summary.endpoint],
    ["Configuration", summary.generation], ["Client elapsed", formatMs(summary.elapsed_ms)], ["Corrections used", summary.corrections_used], ["Terminal reason", summary.terminal_reason],
    ["Accepted tokens", formatNumber(summary.usage?.total_tokens)], ["Observed attempt tokens", formatNumber(summary.observed_usage?.total_tokens)], ["All-attempt tokens", formatNumber(summary.total_usage?.total_tokens)], ["Coordinator wall time", formatMs(summary.coordinator_elapsed_ms)],
  ]) detailPair(grid, label, unknown(value));
  card.append(grid);
  const details = node("details", "request-detail"); details.append(node("summary", "", "Recorded diagnostics"), node("pre", "", JSON.stringify(summary, null, 2))); card.append(details);
  return card;
}

export function renderRequestCards(container, summaries, makeCard, id = "request_id") {
  const cards = new Map(Array.from(container.children, (card) => [card.dataset.requestId, card]));
  const wanted = new Set(summaries.map((summary) => String(summary[id])));
  for (const [key, card] of cards) if (!wanted.has(key)) card.remove();
  summaries.forEach((summary, index) => {
    const key = String(summary[id]);
    const previous = cards.get(key);
    const signature = JSON.stringify(summary);
    if (previous && requestCards.get(previous) === signature) return;
    const card = makeCard(summary);
    card.dataset.requestId = key;
    requestCards.set(card, signature);
    if (!previous) { container.insertBefore(card, container.children[index] ?? null); return; }
    const focused = document.activeElement;
    const previousDetails = previous.querySelector("details");
    const details = card.querySelector("details");
    details.open = previousDetails.open;
    const focusInDetails = previousDetails.contains(focused);
    const focusOnAction = previous.contains(focused) && !focusInDetails;
    previous.replaceWith(card);
    if (focusInDetails) details.querySelector("summary").focus({preventScroll: true});
    if (focusOnAction) (card.querySelector("button:not(:disabled)") || details.querySelector("summary")).focus({preventScroll: true});
  });
}

export function renderRequests() {
  const active = state.requests.active || [];
  renderRequestCards($("active-requests"), active, (summary) => requestCard(summary, true));
  $("requests-empty").hidden = active.length > 0;
  if (state.view !== "telemetry" || !state.telemetry) renderActiveCount(active.length);
  renderRequestCards($("last-request"), state.requests.last_completed ? [state.requests.last_completed] : [], (summary) => requestCard(summary, false));
  $("last-empty").hidden = Boolean(state.requests.last_completed); $("clear-last").disabled = !state.requests.last_completed;
}

function renderActiveCount(count) {
  $("active-nav-count").hidden = count === 0;
  $("active-nav-count").textContent = String(count);
}

export function renderTelemetryObservation() {
  const snapshot = state.telemetry;
  $("telemetry-observed").textContent = state.telemetryStale
    ? snapshot ? `Update failed. Showing the snapshot from ${formatDate(snapshot.observed_at)}.` : "Telemetry is unavailable. Try refreshing."
    : snapshot ? `Observed ${formatDate(snapshot.observed_at)}.` : "Waiting for the first snapshot.";
}

function inferenceKind(kind) {
  return kind === "chat" ? "Chat" : kind === "embedding" ? "Embedding" : unknown(kind);
}

function telemetryDuration(value) {
  return typeof value === "number" && value >= 1000 ? `${(value / 1000).toLocaleString(undefined, {maximumFractionDigits: 2})} s` : formatMs(value);
}

function tokensPerSecond(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toLocaleString(undefined, {maximumFractionDigits: 1})} tok/s`
    : "Unknown";
}

function renderTelemetryRows(id, rows) {
  const fragment = document.createDocumentFragment();
  for (const values of rows) {
    const row = node("tr");
    values.forEach((value, index) => {
      const cell = node(index === 0 ? "th" : "td", "", value);
      if (index === 0) cell.scope = "row";
      row.append(cell);
    });
    fragment.append(row);
  }
  $(id).replaceChildren(fragment);
}

function telemetryRequestCard(summary) {
  const card = node("article", "telemetry-request");
  const timing = summary.server_timings;
  const identity = node("strong", "", summary.selected_model || summary.model || "Model not reported");
  const title = node("summary");
  title.append(
    identity,
    node("span", "cell-secondary", `${inferenceKind(summary.kind)} · ${summary.endpoint || "Connection not selected"}`),
    node("span", `state-pill${summary.status === "failed" ? " error" : summary.status === "completed" ? " ready" : ""}`, unknown(summary.status)),
    node("span", "telemetry-request-time", telemetryDuration(summary.elapsed_ms)),
  );
  const facts = node("dl", "detail-grid");
  const values = [
    ["Runtime", unknown(summary.backend)], ["Phase", unknown(summary.phase)],
    ["Started", formatDate(summary.started_at)], ["Finished", formatDate(summary.finished_at)],
    ["Relay elapsed", telemetryDuration(summary.elapsed_ms)], ["Discovery elapsed", telemetryDuration(summary.discovery_ms)],
    ["Discovery HTTP calls", formatNumber(summary.discovery_http_requests)],
  ];
  if (summary.kind === "chat") {
    values.push(
      ["First visible text at relay", telemetryDuration(summary.first_text_ms)],
      ["Prompt tokens", formatNumber(summary.usage?.prompt_tokens)],
      ["Output tokens", formatNumber(summary.usage?.completion_tokens)],
      ["Accepted tokens", formatNumber(summary.usage?.total_tokens)],
      ["Observed attempt tokens", formatNumber(summary.observed_usage?.total_tokens)],
      ["All-attempt tokens", formatNumber(summary.total_usage?.total_tokens)],
      ["Attempt token coverage", summary.usage_complete === true ? "Complete" : summary.usage_complete === false ? "Partial" : "Unknown"],
      ["Tool corrections", formatNumber(summary.corrections_used)],
      ["Native prefill", telemetryDuration(timing?.prompt_ms)],
      ["Native prefill rate", tokensPerSecond(timing?.prompt_per_second)],
      ["Native decode", telemetryDuration(timing?.predicted_ms)],
      ["Native decode rate", tokensPerSecond(timing?.predicted_per_second)],
      ["Native cached tokens", formatNumber(timing?.cache_n)],
    );
  } else if (summary.kind === "embedding") {
    values.push(["Input texts", formatNumber(summary.embedding_inputs)], ["Output vectors", formatNumber(summary.embedding_outputs)]);
  }
  if (summary.terminal_reason) values.push(["Terminal reason", summary.terminal_reason]);
  if (summary.error_code) values.push(["Error code", summary.error_code]);
  if (summary.reasons?.length) values.push(["Recovery reasons", [...new Set(summary.reasons)].join(", ")]);
  for (const [label, value] of values) detailPair(facts, label, value);
  const detail = node("details"); detail.append(title, facts);
  card.append(detail);
  return card;
}

export function renderTelemetry() {
  const snapshot = state.telemetry;
  const {active, counters, models, recent, embedding_http: embedding} = snapshot;
  renderTelemetryObservation();
  $("telemetry-range").textContent = `Relay process started ${formatDate(snapshot.started_at)}.`;
  $("telemetry-catalog").textContent = snapshot.catalog_observed_at ? `Catalog observed ${formatDate(snapshot.catalog_observed_at)}.` : "No catalog observation yet.";
  $("telemetry-active").textContent = formatNumber(active.total);
  $("telemetry-active-types").textContent = `${formatNumber(active.chat)} chat · ${formatNumber(active.embedding)} embedding`;
  for (const outcome of ["completed", "failed", "cancelled"]) $(`telemetry-${outcome}`).textContent = formatNumber(counters[outcome]);
  if (state.view === "telemetry") renderActiveCount(active.total);
  renderTelemetryRows("telemetry-model-rows", models.map((row) => [
    row.model, unknown(row.backend), unknown(row.endpoint), inferenceKind(row.kind),
    formatNumber(row.active), formatNumber(row.counters.completed), formatNumber(row.counters.failed), formatNumber(row.counters.cancelled),
    telemetryDuration(row.average_elapsed_ms), formatNumber(row.samples),
  ]));
  $("telemetry-model-empty").hidden = models.length > 0;
  $("telemetry-model-table").hidden = models.length === 0;
  renderRequestCards($("telemetry-active-rows"), active.requests, telemetryRequestCard);
  renderRequestCards($("telemetry-recent-rows"), recent, telemetryRequestCard);
  $("telemetry-active-empty").hidden = active.requests.length > 0;
  $("telemetry-recent-empty").hidden = recent.length > 0;
  $("telemetry-active-replicas").textContent = `${formatNumber(embedding.active_replicas)} active replicas · peak ${formatNumber(embedding.peak_active_replicas)}.`;
  const httpLabels = {health: "Health", catalog: "Catalog", props: "Properties", tokenize: "Tokenization", embedding: "Embedding POST"};
  const httpCounts = (row) => [row.count, row.failed, row.active, row.peak].map(formatNumber);
  renderTelemetryRows("telemetry-http-rows", Object.entries(embedding.by_kind).map(([kind, counts]) => [httpLabels[kind] || kind, ...httpCounts(counts)]));
  renderTelemetryRows("telemetry-replica-rows", embedding.replicas.map((row) => [row.model, row.endpoint, ...httpCounts(row)]));
  $("telemetry-replica-empty").hidden = embedding.replicas.length > 0;
  $("telemetry-replica-table").hidden = embedding.replicas.length === 0;
}
