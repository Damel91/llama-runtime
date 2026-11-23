/* OpenCode export dialog: draft lifecycle, token fields, generate/apply requests
   and download. The exported payload and sanitization stay server-side. */

import { $, node, notice, formatNumber } from "./dom.js";
import { state } from "./state.js";
import { api } from "./api.js";
import { rowsFromCatalog } from "./catalog.js";

export function invalidateExport() {
  if (state.exportDraft) state.exportDraft.edit += 1;
  state.exportValue = null;
  $("export-output").value = "";
  $("copy-export").disabled = true;
  $("download-export").disabled = true;
  clearExportFeedback();
  updateExportActions();
}

function clearExportFeedback() {
  $("export-error").hidden = true;
  $("export-applied").hidden = true;
  $("export-warnings").replaceChildren();
  $("export-warnings").hidden = true;
}

function updateExportActions() {
  const draft = state.exportDraft;
  const unavailable = !draft?.fields.size || Boolean(draft.pending);
  const generate = $("generate-export");
  const apply = $("apply-export");
  generate.disabled = unavailable;
  apply.hidden = $("export-version").value !== "v2";
  apply.disabled = unavailable || state.exportApplying;
  for (const [control, busy] of [[generate, draft?.pending === "generate"], [apply, state.exportApplying]]) {
    if (busy) control.setAttribute("aria-busy", "true");
    else control.removeAttribute("aria-busy");
  }
}

function exportTokenField(title, help, value = null) {
  const label = node("label", "", title);
  const input = node("input");
  input.type = "number";
  input.min = "1";
  input.step = "1";
  input.required = true;
  if (value !== null) {
    input.value = String(value);
    input.readOnly = true;
  }
  label.append(input, node("small", "", help));
  return {label, input};
}

export function openExport() {
  const fields = new Map();
  const catalog = state.catalog;
  state.exportDraft = {catalog, fields, edit: 0, pending: null};
  invalidateExport();
  const settings = $("export-model-settings");
  const defaultModel = $("export-default");
  const prompt = node("option", "", "Choose a model");
  prompt.value = "";
  defaultModel.replaceChildren(prompt);
  settings.replaceChildren();
  for (const {model} of rowsFromCatalog(catalog)) {
    const id = model.id;
    if (model.opencode_chat !== true || fields.has(id) || (state.selectedModels.size && !state.selectedModels.has(id))) continue;
    const option = node("option", "", id);
    option.value = id;
    defaultModel.append(option);
    const observedContext = model.loaded_context ?? null;
    const context = observedContext === null
      ? exportTokenField("Context (tokens)", "The server did not report its loaded context. Enter the effective value for one conversation.")
      : exportTokenField("Loaded context (tokens)", "Reported by the server for the loaded model.", observedContext);
    const output = exportTokenField("Maximum output tokens", "Per generation. Choose a value below the context.");
    const fieldset = node("fieldset", "export-model");
    const grid = node("div", "form-grid");
    grid.append(context.label, output.label);
    fieldset.append(node("legend", "", id), grid);
    settings.append(fieldset);
    fields.set(id, {observedContext, context: context.input, output: output.input});
  }
  defaultModel.value = fields.size === 1 ? fields.keys().next().value : "";
  defaultModel.disabled = fields.size === 0;
  updateExportActions();
  $("export-selection").textContent = fields.size
    ? `${fields.size} chat ${fields.size === 1 ? "model" : "models"} to export from ${state.selectedModels.size ? "your selection" : "the observed catalog"}.`
    : "No exportable chat models. Close this dialog and refresh the catalog or change your selection.";
  $("export-dialog").showModal();
  $("export-version").focus();
}

function exportTokenCount(input, message) {
  const value = input.valueAsNumber;
  if (Number.isSafeInteger(value) && value > 0) return value;
  input.focus();
  throw new Error(message);
}

function exportPayload(draft) {
  if (draft.catalog !== state.catalog || draft.catalog.revision !== state.revision) {
    throw new Error("The catalog or saved settings changed. Close this dialog, refresh the catalog and reopen the export.");
  }
  const defaultModel = $("export-default").value;
  if (!draft.fields.has(defaultModel)) {
    $("export-default").focus();
    throw new Error("Choose the default model for OpenCode.");
  }
  const models = Object.fromEntries([...draft.fields].map(([id, {observedContext, context, output}]) => {
    const contextTokens = exportTokenCount(context, `Enter a positive whole number for the context of ${id}.`);
    const outputTokens = exportTokenCount(output, `Enter a positive whole number for the maximum output of ${id}.`);
    if (outputTokens >= contextTokens) {
      output.focus();
      throw new Error(`Maximum output for ${id} must be below its context of ${formatNumber(contextTokens)} tokens.`);
    }
    return [id, {observed_context: observedContext, context: contextTokens, output: outputTokens}];
  }));
  return {version: $("export-version").value, catalog_revision: draft.catalog.revision, default_model: defaultModel, models};
}

export async function submitExport(apply = false) {
  const draft = state.exportDraft;
  if (!draft || draft.pending || !draft.fields.size) return;
  if (apply && (state.exportApplying || $("export-version").value !== "v2")) return;
  if (apply) clearExportFeedback();
  else invalidateExport();
  const edit = draft.edit;
  const current = () => state.exportDraft === draft && draft.edit === edit && $("export-dialog").open;
  try {
    const payload = exportPayload(draft);
    draft.pending = apply ? "apply" : "generate";
    if (apply) state.exportApplying = true;
    updateExportActions();
    const value = await api(apply ? "/opencode/apply" : "/opencode", {method: "POST", body: payload});
    if (apply) {
      const result = value.changed
        ? `Saved OpenCode V2 settings in ${value.filename}. Reopen OpenCode to load them.`
        : `${value.filename} already contains these OpenCode V2 settings.`;
      const message = value.backup ? `${result} Previous settings saved in ${value.backup}.` : result;
      if (!current()) { notice(message); return; }
      $("export-applied").textContent = message;
      $("export-applied").hidden = false;
    } else if (current()) {
      state.exportValue = {config: value.config, filename: value.filename || `opencode-relay-${payload.version}.json`};
      $("export-output").value = JSON.stringify(value.config, null, 2) + "\n";
      $("copy-export").disabled = false;
      $("download-export").disabled = false;
    }
    if (!current()) return;
    for (const warning of value.warnings || []) $("export-warnings").append(node("div", "", warning));
    $("export-warnings").hidden = !(value.warnings || []).length;
  } catch (error) {
    const messages = {
      relay_opencode_apply_local_only: "Open the relay panel on this computer to apply settings to OpenCode.",
      relay_opencode_apply_v2_required: "Select OpenCode V2 to apply settings.",
      relay_opencode_apply_invalid_export: "Refresh the model catalog and reopen this dialog before applying settings.",
      relay_opencode_config_invalid: "Check the OpenCode configuration file for syntax or settings errors before applying.",
      relay_opencode_config_conflict: "Review the OpenCode configuration file and its location before trying again.",
      relay_opencode_config_write_failed: "Check write access to the OpenCode configuration folder, then try again.",
    };
    const message = messages[error.code] || (error.status === 409
      ? "A model or the saved settings changed. Close this dialog, refresh the catalog and reopen the export."
      : error.message);
    if (!current()) { if (apply) notice(message, true); return; }
    $("export-error").textContent = message;
    $("export-error").hidden = false;
  } finally {
    draft.pending = null;
    if (apply) state.exportApplying = false;
    updateExportActions();
  }
}

export function downloadExport() {
  if (!state.exportValue) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(state.exportValue.config, null, 2) + "\n"], {type: "application/json"}));
  const link = node("a"); link.href = url; link.download = state.exportValue.filename.replace(/[\\/]/g, "_"); document.body.append(link); link.click(); link.remove(); window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
