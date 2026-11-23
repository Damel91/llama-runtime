/* Observed model catalog: rows, filters, table, runtime rail, detail and refresh. */

import { $, node, button, unknown, formatNumber, formatDate, notice, detailPair } from "./dom.js";
import { state, busy } from "./state.js";
import { api } from "./api.js";

export function rowsFromCatalog(catalog) {
  return (catalog.runtimes || []).flatMap((runtime, runtimeIndex) =>
    (runtime.models || []).map((model, modelIndex) => ({
      key: JSON.stringify([runtime.name || runtime.endpoint?.name || runtimeIndex, model.id, modelIndex]),
      runtime, model,
    }))
  );
}

function modelAvailability(row) {
  if (row.runtime.available === false) return "unavailable";
  if (row.model.failed === true) return "failed";
  if (row.model.loaded === true) return "loaded";
  if (row.model.loaded === false) return "not-loaded";
  return "unknown";
}

function capabilityLabel(model) {
  const labels = [];
  if (model.capabilities?.chat === true) labels.push("Chat");
  if (model.capabilities?.embeddings === true) labels.push("Embeddings");
  if (!labels.length) labels.push(model.capabilities?.chat === false && model.capabilities?.embeddings === false ? "Unsupported" : "Unknown");
  return labels;
}

export function visibleRows() {
  const query = $("model-search").value.trim().toLocaleLowerCase();
  const kind = $("model-kind").value;
  const availability = $("model-availability").value;
  return rowsFromCatalog(state.catalog).filter((row) => {
    const haystack = [row.model.id, ...(row.model.aliases || []), row.runtime.name, row.runtime.backend, row.runtime.endpoint?.base_url].join(" ").toLocaleLowerCase();
    if (query && !haystack.includes(query)) return false;
    if (kind !== "all" && (kind === "unknown" ? !capabilityLabel(row.model).includes("Unknown") : row.model.capabilities?.[kind] !== true)) return false;
    return availability === "all" || (availability === "unavailable" ? ["unavailable", "failed"].includes(modelAvailability(row)) : modelAvailability(row) === availability);
  });
}

export function renderModels() {
  const allRows = rowsFromCatalog(state.catalog);
  const rows = visibleRows();
  $("model-rows").replaceChildren();
  for (const row of rows) {
    const tr = node("tr", row.key === state.selectedRow ? "selected" : "");
    const checkCell = node("td", "checkbox-column");
    const check = node("input"); check.type = "checkbox"; check.dataset.exportModel = row.model.id;
    check.setAttribute("aria-label", `Select ${row.model.id} for OpenCode export`);
    check.disabled = row.model.opencode_chat !== true;
    check.checked = !check.disabled && state.selectedModels.has(row.model.id);
    check.addEventListener("change", () => {
      check.checked ? state.selectedModels.add(row.model.id) : state.selectedModels.delete(row.model.id);
      for (const input of document.querySelectorAll("[data-export-model]:not(:disabled)")) if (input.dataset.exportModel === row.model.id) input.checked = check.checked;
      updateModelSelection();
    });
    checkCell.append(check); tr.append(checkCell);
    const nameCell = node("td");
    const nameButton = node("button", "model-name-button", row.model.id); nameButton.type = "button";
    nameButton.addEventListener("click", () => { state.selectedRow = row.key; renderModels(); renderModelDetail(row); });
    nameCell.append(nameButton);
    const alias = (row.model.aliases || []).join(", ");
    if (alias) nameCell.append(node("div", "cell-secondary", alias));
    tr.append(nameCell);
    const runtimeCell = node("td", "", row.runtime.backend || "Unknown");
    runtimeCell.append(node("div", "cell-secondary", row.runtime.name || row.runtime.endpoint?.name || "Unnamed endpoint")); tr.append(runtimeCell);
    const availabilityCell = node("td");
    const availability = modelAvailability(row);
    availabilityCell.append(node("span", `state-pill ${availability === "loaded" ? "ready" : ["unavailable", "failed"].includes(availability) ? "error" : ""}`, {
      loaded: "Loaded", "not-loaded": "Not loaded", unavailable: "Endpoint unavailable", failed: "Failed", unknown: "Unknown",
    }[availability])); tr.append(availabilityCell);
    const capabilityCell = node("td", "capabilities");
    for (const label of capabilityLabel(row.model)) capabilityCell.append(node("span", `capability-tag${label === "Unknown" ? " unknown" : ""}`, label)); tr.append(capabilityCell);
    tr.append(node("td", "", formatNumber(row.model.context ?? row.model.loaded_context))); $("model-rows").append(tr);
  }
  const observed = Boolean(state.catalog.observed_at);
  $("models-empty").hidden = rows.length > 0;
  if (allRows.length && !rows.length) {
    $("models-empty").querySelector("h3").textContent = "No models match these filters";
    $("models-empty").querySelector("p").textContent = "Change the search, capability or availability filter to see the rest of the observed catalog.";
    $("models-empty").querySelector("button").hidden = true;
  } else {
    $("models-empty").querySelector("h3").textContent = observed ? "No models were reported" : "No observed models yet";
    $("models-empty").querySelector("p").textContent = observed ? "Inspect the connection observations above, or load models through your server manager and refresh the catalog." : "Add a connection, then refresh the catalog. Discovery reads runtime metadata; it does not load models.";
    $("models-empty").querySelector("button").hidden = false;
  }
  $("model-count").textContent = observed ? `${rows.length.toLocaleString()} of ${allRows.length.toLocaleString()} observed model entries. Unknown capabilities and context remain unknown.` : "No catalog observation.";
  $("catalog-observed").textContent = observed ? `Observed ${formatDate(state.catalog.observed_at)} · configuration revision ${unknown(state.catalog.revision)}${state.catalog.revision !== state.revision ? " (settings have changed; refresh to update)" : ""}` : "No discovery has run yet.";
  renderRuntimeRail(); updateModelSelection();
  if (state.selectedRow && !allRows.some((row) => row.key === state.selectedRow)) { state.selectedRow = null; $("model-detail").hidden = true; }
  const datalist = $("known-models"); datalist.replaceChildren();
  for (const id of new Set([...allRows.map((row) => row.model.id), ...Object.keys(state.config?.model_overrides || {})])) { const option = node("option"); option.value = id; datalist.append(option); }
}

function renderRuntimeRail() {
  const rail = $("runtime-rail"); rail.replaceChildren();
  for (const runtime of state.catalog.runtimes || []) {
    const item = node("div", `runtime-node ${runtime.available === true ? "available" : runtime.available === false ? "unavailable" : ""}`);
    const marker = node("span", "node-status");
    marker.setAttribute("aria-hidden", "true");
    const availability = runtime.available === true ? "Available" : runtime.available === false ? "Unavailable" : "Availability unknown";
    item.append(marker, node("strong", "", runtime.name || runtime.endpoint?.name || "Unnamed endpoint"), node("span", "", runtime.backend || "Unknown runtime"), node("span", "runtime-availability", availability));
    item.title = [runtime.endpoint?.base_url, ...(runtime.diagnostics || [])].filter(Boolean).join("\n");
    rail.append(item);
  }
}

function updateModelSelection() {
  const candidates = [...document.querySelectorAll("[data-export-model]:not(:disabled)")];
  const checked = candidates.filter((input) => input.checked).length;
  $("select-visible-models").checked = candidates.length > 0 && checked === candidates.length;
  $("select-visible-models").indeterminate = checked > 0 && checked < candidates.length;
  $("select-visible-models").disabled = candidates.length === 0;
}

function renderModelDetail(row) {
  const detail = $("model-detail"); detail.replaceChildren(); detail.hidden = false;
  const header = node("div", "detail-heading"); header.append(node("h3", "", row.model.id));
  const close = node("button", "icon-button", "×"); close.type = "button"; close.setAttribute("aria-label", "Close selected model"); close.addEventListener("click", () => { state.selectedRow = null; detail.hidden = true; renderModels(); }); header.append(close); detail.append(header);
  const grid = node("dl", "detail-grid");
  for (const [label, value] of [
    ["Connection", row.runtime.name || row.runtime.endpoint?.name], ["Runtime", row.runtime.backend], ["Server", row.runtime.endpoint?.base_url],
    ["Loaded", row.model.loaded === true ? "Yes" : row.model.loaded === false ? "No" : "Unknown"], ["Loaded context", formatNumber(row.model.context ?? row.model.loaded_context)],
    ["Training context", formatNumber(row.model.training_context)], ["Embedding width", formatNumber(row.model.embedding_width)], ["Aliases", row.model.aliases?.join(", ") || "None reported"],
  ]) detailPair(grid, label, unknown(value));
  detail.append(grid);
  const actions = node("div", "detail-actions"); actions.append(button("Configure model settings", () => state.hooks.openModelSettings(row.model.id))); detail.append(actions);
  detail.append(node("p", "detail-note", "These are runtime observations, not a reservation or an instruction to load the model. Context and GPU allocation remain on the server."));
}

export async function refreshCatalog(control = $("refresh-catalog")) {
  await busy(control, async () => {
    state.catalog = await api("/catalog/refresh", {method: "POST"});
    if (state.exportDraft) state.hooks.invalidateExport();
    const ids = new Set(rowsFromCatalog(state.catalog).filter((row) => row.model.opencode_chat === true).map((row) => row.model.id));
    state.selectedModels = new Set([...state.selectedModels].filter((id) => ids.has(id)));
    renderModels(); state.renderers.connections();
    notice(`Catalog refreshed: ${rowsFromCatalog(state.catalog).length.toLocaleString()} observed model entries. No model was loaded.`);
  });
}
