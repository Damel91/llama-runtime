/* Shared console state, dirty/save state, navigation, polling and snapshot refresh.
   Renderers and the admin transport arrive from the entry module, so this module
   imports only dom.js and the graph stays acyclic. */

import { $, notice } from "./dom.js";

export const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
export const clone = (value) => JSON.parse(JSON.stringify(value));

export const state = {
  token: null, sessionPending: null, pageActive: true, pageEpoch: 0,
  revision: null, config: null, configAction: null, secretFields: [], endpointSecrets: new Map(),
  catalog: {revision: null, observed_at: null, runtimes: []}, status: null,
  requests: {active: [], last_completed: null}, view: "models", settingsTab: "generation",
  connection: "connecting", observations: 0, connectionObservation: 0,
  requestsObserved: null, requestsStale: false, localRefresh: null,
  telemetry: null, telemetryStale: false, telemetryRefresh: null,
  audit: {data: null, filters: {}, cursors: [null], checkedAt: null, dirty: false, edit: 0, operation: null, stale: false},
  dirty: false, errors: new Map(), rawFields: new Map(), selectedRow: null,
  selectedModels: new Set(), endpointIndex: null, exportDraft: null, exportValue: null, exportApplying: false, poll: null, tailnetDraft: null,
  renderers: {}, hooks: {},
};

export function markDirty() {
  state.dirty = true;
  updateSaveState();
}

export function updateSaveState() {
  $("dirty-label").hidden = !state.dirty;
  const updating = state.configAction !== null;
  $("save-config").disabled = updating || !state.config || !state.dirty || state.errors.size > 0;
  $("reload-config").disabled = updating;
  $("resolve-conflict").disabled = updating;
  $("config-progress").hidden = !updating;
  $("config-progress").textContent = state.configAction === "save" ? "Saving settings…" : updating ? "Reloading settings…" : "";
  for (const id of ["view-connections", "view-settings", "endpoint-dialog"]) {
    $(id).inert = updating;
  }
}

export async function busy(control, operation) {
  if (control.disabled) return;
  control.disabled = true;
  control.setAttribute("aria-busy", "true");
  try { await operation(); }
  catch (error) { notice(error.message || "The local relay could not complete this action.", true); }
  finally {
    control.disabled = false; control.removeAttribute("aria-busy"); updateSaveState();
    if (state.status) state.renderers.status();
    $("clear-last").disabled = !state.requests.last_completed;
  }
}

function refreshSnapshot(key, path, update, stale) {
  if (state[key]) return state[key];
  const epoch = state.pageEpoch;
  const observation = ++state.observations;
  const recordConnection = (connection) => {
    if (observation < state.connectionObservation) return;
    state.connectionObservation = observation;
    state.connection = connection;
    state.renderers.status();
  };
  const refresh = state.hooks.api(path).then((value) => {
    if (epoch !== state.pageEpoch) return;
    update(value);
    recordConnection("connected");
  }).catch((error) => {
    if (epoch !== state.pageEpoch) return;
    stale();
    recordConnection("unavailable");
    throw error;
  }).finally(() => { if (state[key] === refresh) state[key] = null; });
  state[key] = refresh;
  return refresh;
}

export function refreshLocalState({afterPending = false} = {}) {
  if (afterPending && state.localRefresh) return state.localRefresh.catch(() => {}).then(() => refreshLocalState());
  return refreshSnapshot("localRefresh", "/status", (value) => {
    state.status = value;
    state.requests = value.requests;
    state.requestsObserved = new Date().toISOString();
    state.requestsStale = false;
    state.renderers.requests();
    state.renderers.requestObservation();
  }, () => { state.requestsStale = true; state.renderers.requestObservation(); });
}

export function refreshTelemetry() {
  return refreshSnapshot("telemetryRefresh", "/telemetry", (value) => {
    state.telemetry = value;
    state.telemetryStale = false;
    state.renderers.telemetry();
  }, () => { state.telemetryStale = true; state.renderers.telemetryObservation(); });
}

export function watchView() { return state.view === "requests" || state.view === "telemetry"; }

export function refreshVisibleView() {
  return state.view === "telemetry" ? refreshTelemetry() : refreshLocalState();
}

export function navigate(view) {
  const descriptions = {
    models: ["Models", "The models your configured runtimes actually report."],
    connections: ["Connections", "Connect native runtimes without moving their models or processes."],
    requests: ["Requests", "Inspect the work owned by this relay and its latest completion."],
    telemetry: ["Telemetry", "Observe latency, tool recovery, and embedding traffic through this relay."],
    settings: ["Settings", "Configure the local relay while active requests keep their own settings."],
  };
  state.view = own(descriptions, view) ? view : "models";
  for (const section of document.querySelectorAll(".view")) section.hidden = section.id !== `view-${state.view}`;
  for (const link of document.querySelectorAll("[data-view]")) {
    if (link.dataset.view === state.view) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  $("view-title").textContent = descriptions[state.view][0];
  $("view-description").textContent = descriptions[state.view][1];
  if (watchView() && state.config && !document.hidden) refreshVisibleView().catch(() => {});
  schedulePoll();
}

export function schedulePoll() {
  if (state.poll !== null) window.clearTimeout(state.poll);
  state.poll = null;
  if (!watchView() || document.hidden || !state.pageActive || !state.config) return;
  state.poll = window.setTimeout(async () => {
    try { await refreshVisibleView(); }
    catch { /* The last snapshot remains visible with its update status. */ }
    finally { schedulePoll(); }
  }, 2000);
}
