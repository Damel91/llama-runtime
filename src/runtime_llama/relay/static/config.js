/* Configuration draft, validation and save workflow, plus the shared field-edit
   helpers used by the settings panels. Revision conflicts and secret handling
   keep the baseline behavior. */

import { $, node, notice, field } from "./dom.js";
import { state, clone, own, markDirty, updateSaveState, refreshLocalState } from "./state.js";
import { api } from "./api.js";

export function optionalValue(target, key, raw, kind = "number") {
  if (raw === "") { delete target[key]; return; }
  if (raw === "__null__") { target[key] = null; return; }
  if (kind === "boolean") { target[key] = raw === "true"; return; }
  if (kind === "string") { target[key] = raw; return; }
  const value = Number(raw);
  if (!Number.isFinite(value) || (kind === "integer" && !Number.isInteger(value))) {
    throw new Error(kind === "integer" ? "Enter a whole number or leave this field empty." : "Enter a finite number or leave this field empty.");
  }
  target[key] = value;
}

export function secretValue(target, key, action, value) {
  if (action === "keep") return;
  if (action === "clear") { target[key] = null; return; }
  if (action !== "replace" || typeof value !== "string" || !value) {
    throw new Error("Enter a value to replace the saved secret, or choose Keep current.");
  }
  target[key] = value;
}

function savePayload(revision, config) {
  if (!Number.isInteger(revision) || revision < 0) throw new Error("Reload settings before saving.");
  return {expected_revision: revision, config: clone(config)};
}

export function endpointDraft(previous, values, savedSecrets = {}) {
  const result = clone(previous || {});
  const renamed = previous && previous.name !== values.name.trim();
  result.name = values.name.trim();
  result.base_url = values.base_url.trim();
  if (!result.name || !result.base_url) throw new Error("Enter a connection name and server origin.");
  result.backend = values.backend;
  result.verify_ssl = Boolean(values.verify_ssl);
  result.use_tailnet = Boolean(values.use_tailnet);
  result.http_connection_policy = values.http_connection_policy;
  if (values.auth === "env") {
    if (!values.api_key_env.trim()) throw new Error("Enter the API key environment variable name.");
    result.api_key = null;
    result.api_key_env = values.api_key_env.trim();
  } else if (values.auth === "key") {
    delete result.api_key_env;
    if (values.api_key) result.api_key = values.api_key;
    else if (typeof result.api_key === "string" && result.api_key) { /* Keep an explicitly staged value. */ }
    else if (savedSecrets.api_key && !renamed) delete result.api_key;
    else if (!result.api_key) throw new Error("Enter the API key. A renamed connection cannot reuse its old saved key.");
  } else {
    result.api_key = null;
    delete result.api_key_env;
  }
  if (renamed && savedSecrets.proxy_url && values.proxy_action === "keep" && !result.proxy_url) {
    throw new Error("Re-enter the proxy URL or choose No proxy when renaming this connection.");
  }
  secretValue(result, "proxy_url", values.proxy_action, values.proxy_url);
  return result;
}

export function loadConfig(value) {
  state.revision = value.revision;
  if (state.exportDraft && state.exportDraft.catalog.revision !== state.revision) state.hooks.invalidateExport();
  state.config = clone(value.config);
  state.secretFields = [...(value.secret_fields || [])];
  state.endpointSecrets = new Map(state.config.runtime_endpoints.map((endpoint, index) => [endpoint.name, {
    api_key: state.secretFields.includes(`/runtime_endpoints/${index}/api_key`),
    proxy_url: state.secretFields.includes(`/runtime_endpoints/${index}/proxy_url`),
  }]));
  state.dirty = false;
  state.tailnetDraft = null;
  state.errors.clear();
  state.rawFields.clear();
  $("config-revision").textContent = `Revision ${state.revision}`;
  $("conflict-notice").hidden = true;
  updateSaveState();
  state.renderers.connections();
  state.renderers.settings();
}

export async function reloadConfig(discard = true) {
  if (discard && state.dirty && !window.confirm("Discard unsaved changes and reload the saved relay settings?")) return;
  loadConfig(await api("/config"));
  notice("Saved settings loaded.");
  await refreshLocalState({afterPending: true}).catch(() => {});
}

export async function saveConfig() {
  if (state.errors.size) throw new Error("Correct the highlighted fields before saving.");
  try {
    const value = await api("/config", {method: "PUT", body: savePayload(state.revision, state.config)});
    loadConfig(value);
    $("restart-notice").hidden = value.restart_required !== true;
    notice("Settings saved. New requests use the new configuration; active requests keep their original settings.");
    await refreshLocalState({afterPending: true}).catch(() => {});
  } catch (error) {
    if (error.status === 409) {
      $("conflict-notice").hidden = false;
      notice("Your draft has not been overwritten. Reload saved settings to resolve the revision conflict.", true);
    } else throw error;
  }
}

export async function changeConfig(action, run) {
  if (state.configAction) return;
  state.configAction = action;
  updateSaveState();
  try { await run(); }
  catch (error) { notice(error.message, true); }
  finally { state.configAction = null; updateSaveState(); }
}

export const fieldKey = (scope, key) => JSON.stringify([scope, key]);

export function clearScopeDiagnostics(scope) {
  for (const values of [state.errors, state.rawFields]) for (const key of [...values.keys()]) {
    try { if (JSON.parse(key)[0] === scope) values.delete(key); } catch (_) { /* Other field domains retain their own state. */ }
  }
}

export function onValueEdit(control, update) {
  control.addEventListener("input", update);
  control.addEventListener("change", update);
}

export function editField(control, errorKey, mutate) {
  const error = node("span", "field-error"); error.hidden = true;
  onValueEdit(control, () => {
    try { mutate(control.value); state.errors.delete(errorKey); error.hidden = true; control.removeAttribute("aria-invalid"); }
    catch (failure) { state.errors.set(errorKey, failure.message); error.textContent = failure.message; error.hidden = false; control.setAttribute("aria-invalid", "true"); }
    markDirty();
  });
  if (state.errors.has(errorKey)) { error.textContent = state.errors.get(errorKey); error.hidden = false; control.setAttribute("aria-invalid", "true"); }
  return error;
}

export function numberField(grid, settings, scope, key, label, help, integer = false) {
  const wrap = field(label, help); const input = node("input"); input.type = "number"; input.step = integer ? "1" : "any";
  input.placeholder = "Inherit default"; input.value = settings[key] ?? "";
  const error = editField(input, fieldKey(scope, key), (raw) => {
    if (input.validity.badInput) throw new Error("Enter a valid number or leave this field empty.");
    optionalValue(settings, key, raw, integer ? "integer" : "number");
  }); wrap.append(input, error); grid.append(wrap);
}

export function selectField(grid, settings, scope, key, label, options, help) {
  const wrap = field(label, help); const input = node("select");
  for (const [value, text] of [["", "Inherit default"], ...options]) { const option = node("option", "", text); option.value = value; input.append(option); }
  input.value = own(settings, key) ? String(settings[key]) : "";
  const error = editField(input, fieldKey(scope, key), (raw) => optionalValue(settings, key, raw, options.every(([value]) => value === "true" || value === "false") ? "boolean" : "string"));
  wrap.append(input, error); grid.append(wrap);
}

export function jsonField(grid, settings, scope, key, label, help) {
  const wrap = field(label, help); wrap.classList.add("span-all"); const input = node("textarea", "code-editor"); input.rows = 4; input.spellcheck = false;
  const errorKey = fieldKey(scope, key); input.value = state.rawFields.get(errorKey) ?? (own(settings, key) ? JSON.stringify(settings[key], null, 2) : ""); input.placeholder = "Inherit default";
  const error = editField(input, errorKey, (raw) => {
    state.rawFields.set(errorKey, raw);
    if (!raw.trim()) { delete settings[key]; return; }
    let value;
    try { value = JSON.parse(raw); } catch (_) { throw new Error("Enter a valid JSON object, or leave this field empty."); }
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("This setting requires a JSON object.");
    settings[key] = value;
  }); wrap.append(input, error); grid.append(wrap);
}
