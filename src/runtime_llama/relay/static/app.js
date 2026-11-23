/* The console uses only the versioned local admin API. No browser-side inference.
   This module is the entry point: theme, module wiring, event bindings and boot.
   No work logic lives here. */

import { $, notice, copyText } from "./dom.js";
import { api } from "./api.js";
import {
  state, clone, markDirty, busy, navigate, schedulePoll, watchView,
  refreshVisibleView, refreshLocalState, refreshTelemetry,
} from "./state.js";
import { loadConfig, reloadConfig, saveConfig, changeConfig } from "./config.js";
import { renderModels, visibleRows, refreshCatalog } from "./catalog.js";
import { renderConnections, editEndpoint, showEndpointAuth, applyEndpoint } from "./connections.js";
import { openSettingsTab, openModelSettings, renderOverrides, renderTailnet, renderSettings } from "./settings.js";
import { renderStatus, renderRequests, renderRequestObservation, renderTelemetry, renderTelemetryObservation } from "./activity.js";
import { renderAudit, bindAudit } from "./audit.js";
import { invalidateExport, openExport, submitExport, downloadExport } from "./export.js";

const THEME_KEY = "llama-runtime.relay.theme";

function setupTheme() {
  const picker = $("relay-theme");
  const apply = (value) => {
    const theme = value === "dark" ? "dark" : "light";
    document.documentElement.dataset.theme = theme;
    picker.value = theme;
    return theme;
  };
  let saved;
  try { saved = window.localStorage.getItem(THEME_KEY); }
  catch { /* Theme selection still works when storage is disabled. */ }
  apply(saved);
  picker.addEventListener("change", () => {
    const theme = apply(picker.value);
    try { window.localStorage.setItem(THEME_KEY, theme); }
    catch { /* Keep the current choice for this page. */ }
  });
}

function wire() {
  state.hooks.api = api;
  state.hooks.invalidateExport = invalidateExport;
  state.hooks.openModelSettings = openModelSettings;
  state.renderers.status = renderStatus;
  state.renderers.requests = renderRequests;
  state.renderers.requestObservation = renderRequestObservation;
  state.renderers.telemetry = renderTelemetry;
  state.renderers.telemetryObservation = renderTelemetryObservation;
  state.renderers.connections = renderConnections;
  state.renderers.settings = renderSettings;
  state.renderers.models = renderModels;
}

function bind() {
  bindAudit();
  function focusWorkspace() { $("workspace").focus({preventScroll: true}); }
  for (const link of document.querySelectorAll("[data-view]")) link.addEventListener("click", (event) => {
    if (!event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey) focusWorkspace();
  });
  for (const control of document.querySelectorAll("[data-navigate]")) control.addEventListener("click", () => {
    window.location.hash = control.dataset.navigate;
    focusWorkspace();
  });
  $("skip-workspace").addEventListener("click", (event) => {
    event.preventDefault();
    $("workspace").focus();
  });
  window.addEventListener("hashchange", () => { if (window.location.hash !== "#workspace") navigate(window.location.hash.slice(1)); });
  function restoreVisibleState() {
    if (!document.hidden && watchView()) refreshVisibleView().catch(() => {});
    schedulePoll();
  }
  document.addEventListener("visibilitychange", restoreVisibleState);
  window.addEventListener("pagehide", () => {
    state.pageEpoch += 1;
    state.pageActive = false;
    state.token = null;
    state.sessionPending = null;
    state.localRefresh = null;
    state.telemetryRefresh = null;
    state.audit.operation = null;
    schedulePoll();
  });
  window.addEventListener("pageshow", (event) => {
    if (event.persisted) {
      state.pageActive = true;
      renderAudit();
      if (watchView()) restoreVisibleState();
      else refreshLocalState().catch(() => {});
    }
  });
  window.addEventListener("beforeunload", (event) => { if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });
  $("save-config").addEventListener("click", () => changeConfig("save", saveConfig));
  for (const id of ["reload-config", "resolve-conflict"]) $(id).addEventListener("click", () => changeConfig("reload", reloadConfig));
  $("copy-api").addEventListener("click", () => busy($("copy-api"), async () => { await copyText(`${window.location.origin}/v1`); notice("Local API URL copied."); }));
  for (const id of ["model-search", "model-kind", "model-availability"]) $(id).addEventListener(id === "model-search" ? "input" : "change", renderModels);
  $("select-visible-models").addEventListener("change", () => { const selected = $("select-visible-models").checked; for (const row of visibleRows()) if (row.model.opencode_chat === true) selected ? state.selectedModels.add(row.model.id) : state.selectedModels.delete(row.model.id); renderModels(); });
  $("refresh-catalog").addEventListener("click", () => refreshCatalog());
  $("add-endpoint").addEventListener("click", () => editEndpoint());
  for (const control of document.querySelectorAll('[data-action="add-endpoint"]')) control.addEventListener("click", () => editEndpoint());
  for (const id of ["endpoint-auth", "endpoint-proxy-action"]) $(id).addEventListener("change", showEndpointAuth);
  $("endpoint-form").addEventListener("submit", applyEndpoint);
  $("endpoint-dialog").addEventListener("close", () => { $("endpoint-key").value = ""; $("endpoint-proxy").value = ""; });
  for (const control of document.querySelectorAll("[data-close-dialog]")) control.addEventListener("click", () => $(control.dataset.closeDialog).close());
  $("refresh-requests").addEventListener("click", () => busy($("refresh-requests"), refreshLocalState));
  $("refresh-telemetry").addEventListener("click", () => busy($("refresh-telemetry"), refreshTelemetry));
  $("clear-last").addEventListener("click", () => busy($("clear-last"), async () => { await api("/requests/last-completed", {method: "DELETE"}); notice("Last-completed summary cleared."); await refreshLocalState({afterPending: true}).catch(() => {}); }));
  for (const tab of document.querySelectorAll("[data-settings-tab]")) {
    tab.addEventListener("click", () => openSettingsTab(tab.dataset.settingsTab));
    tab.addEventListener("keydown", (event) => { if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return; event.preventDefault(); const tabs = [...document.querySelectorAll("[data-settings-tab]")]; let index = tabs.indexOf(tab); index = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length; openSettingsTab(tabs[index].dataset.settingsTab); tabs[index].focus(); });
  }
  $("edit-override").addEventListener("click", () => renderOverrides($("override-model").value));
  $("override-model").addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); renderOverrides($("override-model").value); } });
  $("tailnet-enabled").addEventListener("change", () => {
    if ($("tailnet-enabled").checked) state.config.tailnet = state.tailnetDraft || {executable: "", state_dir: "", hostname: ""};
    else { state.tailnetDraft = clone(state.config.tailnet); state.config.tailnet = null; }
    markDirty(); renderTailnet(); renderStatus();
  });
  for (const [id, action] of [["start-tailnet", "start"], ["stop-tailnet", "stop"]]) $(id).addEventListener("click", () => busy($(id), async () => { await api(`/tailnet/${action}`, {method: "POST"}); await refreshLocalState(); notice(action === "start" ? "Helper start requested using saved settings. Refresh the catalog to observe its connections." : "Helper stop completed. Private-network connections do not fall back to direct access."); }));
  $("open-export").addEventListener("click", openExport);
  $("export-form").addEventListener("submit", (event) => { event.preventDefault(); void submitExport(); });
  $("apply-export").addEventListener("click", () => { void submitExport(true); });
  $("export-form").addEventListener("input", invalidateExport);
  $("export-form").addEventListener("change", invalidateExport);
  $("export-dialog").addEventListener("close", () => { state.exportDraft = null; invalidateExport(); });
  $("copy-export").addEventListener("click", () => busy($("copy-export"), async () => { if (state.exportValue) { await copyText(JSON.stringify(state.exportValue.config, null, 2) + "\n"); notice("OpenCode configuration copied. Existing files were not changed."); } }));
  $("download-export").addEventListener("click", downloadExport);
}

async function boot() {
  setupTheme();
  wire();
  bind(); navigate(window.location.hash.slice(1));
  try {
    loadConfig(await api("/config"));
    const results = await Promise.allSettled([api("/catalog"), refreshVisibleView()]);
    if (results[0].status === "fulfilled") { state.catalog = results[0].value; renderModels(); renderConnections(); }
    for (const result of results) if (result.status === "rejected") notice(result.reason.message, true);
    schedulePoll();
  } catch (error) {
    state.connection = "unavailable";
    renderStatus();
    notice(error.message || "Could not connect to the local relay. Reload settings to reconnect.", true);
  }
}

boot();
