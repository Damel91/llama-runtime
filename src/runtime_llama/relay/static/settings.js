/* Settings panels: LLM and embedding defaults, per-model overrides, the private
   network helper and local access. Field editing reuses the config.js helpers. */

import { $, node, button, field, group } from "./dom.js";
import { state, clone, own, markDirty, navigate } from "./state.js";
import { optionalValue, secretValue, fieldKey, clearScopeDiagnostics, onValueEdit, editField, numberField, selectField, jsonField } from "./config.js";

function textSetting(grid, target, key, label, help, options = {}) {
  const wrap = field(label, help); const input = node(options.multiline ? "textarea" : "input"); if (!options.multiline) input.type = options.secret ? "password" : "text";
  input.value = target[key] ?? ""; input.autocomplete = options.secret ? "new-password" : "off"; if (options.placeholder) input.placeholder = options.placeholder;
  onValueEdit(input, () => { if (input.value === "" && options.optional) delete target[key]; else target[key] = options.trim === false ? input.value : input.value.trim(); markDirty(); });
  wrap.append(input); grid.append(wrap); return input;
}

function secretSetting(grid, target, key, label, pointer) {
  const wrap = node("div", "field"); wrap.append(node("span", "", label), node("small", "", state.secretFields.includes(pointer) || target[key] ? "A value is configured. It is never read back; Keep current leaves it unchanged." : "No saved value is reported."));
  const row = node("div", "secret-edit"); const action = node("select");
  action.setAttribute("aria-label", `${label} change action`);
  for (const [value, text] of [["keep", "Keep current"], ["replace", "Replace"], ["clear", "Remove"]]) { const option = node("option", "", text); option.value = value; action.append(option); }
  const input = node("input"); input.type = "password"; input.autocomplete = "new-password"; input.placeholder = "New value"; input.disabled = true; input.setAttribute("aria-label", `New ${label.toLowerCase()}`);
  const errorKey = `secret:${pointer}`; const error = editField(input, errorKey, () => secretValue(target, key, action.value, input.value));
  action.addEventListener("change", () => {
    input.disabled = action.value !== "replace";
    if (action.value === "clear") { target[key] = null; state.errors.delete(errorKey); error.hidden = true; input.removeAttribute("aria-invalid"); }
    else if (action.value === "keep") { delete target[key]; input.value = ""; state.errors.delete(errorKey); error.hidden = true; input.removeAttribute("aria-invalid"); }
    else { const message = "Enter a new secret value."; state.errors.set(errorKey, message); error.textContent = message; error.hidden = false; input.setAttribute("aria-invalid", "true"); input.focus(); }
    markDirty();
  }); row.append(action, input); wrap.append(row, error); grid.append(wrap);
}

function renderLLM(container, settings, scope) {
  container.replaceChildren();
  group(container, "Sampling", "Exact values are preserved. These settings change generation behavior, not the model’s loaded context or device allocation.", (grid) => {
    numberField(grid, settings, scope, "temperature", "Temperature", "Finite value; the SDK default is used when omitted.");
    numberField(grid, settings, scope, "top_p", "Top-p", "When set: greater than 0 and at most 1.");
    numberField(grid, settings, scope, "top_k", "Top-k", "Optional nonnegative whole number.", true);
    numberField(grid, settings, scope, "min_p", "Min-p", "When set: from 0 to 1.");
    numberField(grid, settings, scope, "repeat_penalty", "Repetition penalty", "Optional positive value.");
    numberField(grid, settings, scope, "num_predict", "Output token budget", "An explicit per-request generation budget.", true);
  });
  group(container, "Reasoning and waits", "Tool recovery keeps the existing initial generation plus at most three corrections. Requests and corrections remain on the selected runtime.", (grid) => {
    selectField(grid, settings, scope, "reasoning", "Reasoning policy", [["auto", "Auto"], ["off", "Off"], ["on", "On"], ["low", "Low"], ["medium", "Medium"], ["high", "High"]], "Availability depends on the native model and template.");
    const wrap = node("div", "field"); wrap.append(node("span", "", "Stream idle wait"), node("small", "", "No idle limit permits quiet prefill; connection and discovery waits remain finite."));
    const row = node("div", "optional-value"); const choice = node("select");
    choice.setAttribute("aria-label", "Stream idle wait policy");
    for (const [value, text] of [["inherit", "Inherit default"], ["none", "No idle limit"], ["seconds", "Seconds"]]) { const option = node("option", "", text); option.value = value; choice.append(option); }
    const input = node("input"); input.type = "number"; input.step = "any"; input.setAttribute("aria-label", "Stream idle seconds"); input.placeholder = "Seconds";
    choice.value = own(settings, "stream_idle_timeout_sec") ? settings.stream_idle_timeout_sec === null ? "none" : "seconds" : "inherit";
    input.value = settings.stream_idle_timeout_sec ?? ""; input.disabled = choice.value !== "seconds";
    const update = () => {
      input.disabled = choice.value !== "seconds";
      if (choice.value === "seconds" && (!input.value || !Number.isFinite(Number(input.value)) || Number(input.value) <= 0)) throw new Error("Enter a positive finite idle timeout, or choose another wait policy.");
      optionalValue(settings, "stream_idle_timeout_sec", choice.value === "inherit" ? "" : choice.value === "none" ? "__null__" : input.value);
    };
    const errorKey = fieldKey(scope, "stream_idle_timeout_sec");
    const error = editField(input, errorKey, update);
    choice.addEventListener("change", () => { try { update(); state.errors.delete(errorKey); error.hidden = true; } catch (failure) { state.errors.set(errorKey, failure.message); error.textContent = failure.message; error.hidden = false; } markDirty(); });
    row.append(choice, input); wrap.append(row, error); grid.append(wrap);
    numberField(grid, settings, scope, "runtime_discovery_timeout", "Discovery timeout, seconds", "Finite timeout for observing configured runtimes.");
    numberField(grid, settings, scope, "context_ceiling", "Logical context ceiling", "Optional consumer ceiling; does not reload server context.", true);
    selectField(grid, settings, scope, "output_mode", "Output format", [["text", "Text"], ["json", "JSON"]], "A native schema or JSON support is required for structured output.");
  });
  const advanced = node("details", "advanced-settings"); advanced.append(node("summary", "", "Advanced generation settings")); const content = node("div");
  const grid = node("div", "form-grid");
  for (const [key, title, help] of [["chat_template_kwargs", "Chat template options", "Native template options as an object."], ["json_schema", "JSON Schema", "Optional structured-output schema; external references are not fetched."], ["extra_body", "Native request extensions", "Explicit supported native extensions. Managed fields remain protected."], ["thinking_diagnostics", "Thinking diagnostics", "Optional capture settings. The request summaries do not expose captured text."]]) jsonField(grid, settings, scope, key, title, help);
  for (const [key, label] of [["runaway_min_chars", "Runaway minimum characters"], ["runaway_window_chars", "Runaway window characters"], ["runaway_block_chars", "Runaway block characters"], ["runaway_repeat_threshold", "Runaway repetition threshold"], ["runaway_ngram_chars", "Runaway n-gram characters"]]) numberField(grid, settings, scope, key, label, "Existing SDK protection; omission keeps its default.", true);
  content.append(grid); advanced.append(content); container.append(advanced);
}

function renderEmbedding(container, settings, scope) {
  container.replaceChildren();
  group(container, "Embedding requests", "Direct texts use the existing embedding engine. This console does not start calibration or tokenize each text.", (grid) => {
    selectField(grid, settings, scope, "input_type", "Input task", [["document", "Document"], ["query", "Query"], ["raw", "Raw"]], "Chooses the model’s configured query/document prefix, or no task prefix for raw input.");
    selectField(grid, settings, scope, "normalize", "Normalize vectors", [["true", "Enabled"], ["false", "Disabled"]], "Keep the declared vector normalization contract.");
    numberField(grid, settings, scope, "timeout", "Request deadline, seconds", "Finite overall embedding timeout.");
    numberField(grid, settings, scope, "batch_size", "Total batch text target", "Optional target distributed over verified replicas. It does not cap the submitted text count.", true);
    numberField(grid, settings, scope, "request_parallelism", "Global HTTP guard", "Optional explicit positive guard. Omitted capacity follows usable replicas.", true);
    const wrap = field("Task prefix source", "Custom prefixes are literal strings, including spaces and line breaks."); const choice = node("select");
    for (const [value, text] of [["inherit", "Automatic / inherited model prefixes"], ["custom", "Custom query and document prefixes"]]) { const option = node("option", "", text); option.value = value; choice.append(option); }
    choice.value = own(settings, "embedding_prefixes") ? "custom" : "inherit";
    const prefixes = node("div", "prefix-fields");
    const controls = {};
    for (const type of ["query", "document"]) { const prefixWrap = field(`${type === "query" ? "Query" : "Document"} prefix`); const input = node("textarea"); input.rows = 2; input.spellcheck = false; input.value = settings.embedding_prefixes?.[type] ?? ""; controls[type] = input; onValueEdit(input, () => { if (choice.value === "custom") { settings.embedding_prefixes = {query: controls.query.value, document: controls.document.value}; markDirty(); } }); prefixWrap.append(input); prefixes.append(prefixWrap); }
    prefixes.hidden = choice.value !== "custom";
    choice.addEventListener("change", () => { prefixes.hidden = choice.value !== "custom"; if (choice.value === "custom") settings.embedding_prefixes = {query: controls.query.value, document: controls.document.value}; else delete settings.embedding_prefixes; markDirty(); });
    wrap.append(choice); grid.append(wrap, prefixes);
  });
  const advanced = node("details", "advanced-settings"); advanced.append(node("summary", "", "Explicit capacity and preparation policies")); const grid = node("div", "form-grid");
  selectField(grid, settings, scope, "embedding_capacity_limits", "Apply explicit capacity policies", [["false", "Disabled"], ["true", "Enabled"]], "Existing scalar capacity fields are inactive unless deliberately enabled.");
  selectField(grid, settings, scope, "embedding_preparation_cache_enabled", "Preparation cache", [["false", "Disabled"], ["true", "Enabled"]], "An explicit cache policy; released at adapter close.");
  for (const [key, label] of [["queue_size", "Queue size"], ["embedding_max_pending_owners", "Pending owners"], ["embedding_max_pending_bytes", "Pending bytes"], ["embedding_max_pending_tokens", "Pending tokens"], ["embedding_max_request_bytes", "Native request bytes"], ["embedding_max_response_bytes", "Native response bytes"], ["embedding_max_handoff_bytes", "HTTP handoff bytes"], ["embedding_isolation_max_requests", "Isolation request policy"]]) numberField(grid, settings, scope, key, label, "Optional operator policy; no value is supplied automatically.", true);
  advanced.append(grid); container.append(advanced);
}

export function openSettingsTab(name) {
  state.settingsTab = name;
  for (const tab of document.querySelectorAll("[data-settings-tab]")) { const active = tab.dataset.settingsTab === name; tab.setAttribute("aria-selected", String(active)); tab.tabIndex = active ? 0 : -1; }
  for (const panel of document.querySelectorAll(".settings-panel")) panel.hidden = panel.id !== `settings-${name}`;
}

export function openModelSettings(id) {
  window.location.hash = "settings"; navigate("settings"); openSettingsTab("overrides"); $("override-model").value = id; renderOverrides(id); $("override-model").focus();
}

export function renderOverrides(id) {
  const target = $("override-fields"); target.replaceChildren();
  if (!id.trim()) { target.append(node("p", "section-note", "Choose an exact model selector first.")); return; }
  id = id.trim();
  const actual = state.config.model_overrides[id] || {};
  const working = {llm: clone(actual.llm || {}), embedding: clone(actual.embedding || {})};
  const title = node("div", "section-heading"); title.append(node("h3", "", id), button("Reset model overrides", () => {
    delete state.config.model_overrides[id];
    clearScopeDiagnostics(`model:${id}:llm`); clearScopeDiagnostics(`model:${id}:embedding`);
    markDirty(); renderOverrides(id);
  })); target.append(title);
  target.append(node("p", "section-note", "Blank fields inherit the relay defaults. Prefixes are applied once. Changing settings here does not load or reload the model."));
  const commit = () => {
    const value = {};
    for (const kind of ["llm", "embedding"]) if (Object.keys(working[kind]).length) value[kind] = working[kind];
    if (Object.keys(value).length) state.config.model_overrides[id] = value; else delete state.config.model_overrides[id];
  };
  const llm = node("details", "advanced-settings"); llm.open = true; llm.append(node("summary", "", "Generation overrides")); const llmContent = node("div"); renderLLM(llmContent, working.llm, `model:${id}:llm`); llm.append(llmContent);
  const embedding = node("details", "advanced-settings"); embedding.append(node("summary", "", "Embedding overrides")); const embeddingContent = node("div"); renderEmbedding(embeddingContent, working.embedding, `model:${id}:embedding`); embedding.append(embeddingContent);
  const editor = node("div"); editor.append(llm, embedding); onValueEdit(editor, commit);
  target.append(editor);
}

export function renderTailnet() {
  const container = $("tailnet-fields"); container.replaceChildren(); $("tailnet-enabled").checked = state.config.tailnet !== null;
  if (!state.config.tailnet) { container.append(node("p", "section-note", "Direct connections do not need the helper. Configure it only for endpoints using the embedded private network.")); return; }
  const settings = state.config.tailnet;
  group(container, "Node configuration", "Save these settings before starting the helper. Existing enrollment state can be reused without a new enrollment key.", (grid) => {
    textSetting(grid, settings, "executable", "Helper executable", "Absolute path to the matching optional helper binary.");
    textSetting(grid, settings, "state_dir", "Node state directory", "An absolute, dedicated private directory owned by this operator.");
    textSetting(grid, settings, "hostname", "Node hostname", "The private-network node’s DNS label.");
    textSetting(grid, settings, "control_url", "Headscale coordinator URL", "Leave empty for Tailscale; Headscale requires an HTTPS origin.", {optional: true});
    secretSetting(grid, settings, "auth_key", "Enrollment key", "/tailnet/auth_key");
    selectField(grid, settings, "tailnet", "ephemeral", "Ephemeral node", [["false", "No"], ["true", "Yes"]], "Choose explicitly for this node’s intended lifetime.");
    for (const [key, label] of [["startup_timeout_sec", "Startup timeout, seconds"], ["dial_timeout_sec", "Dial timeout, seconds"], ["shutdown_timeout_sec", "Shutdown timeout, seconds"]]) numberField(grid, settings, "tailnet", key, label, "Omission keeps the helper’s existing finite default.");
  });
}

function renderAccess() {
  const container = $("access-fields"); container.replaceChildren();
  group(container, "Listener", "Saved address changes require an explicit relay restart. The current browser stays connected to its existing address.", (grid) => {
    textSetting(grid, state.config.listener, "host", "Bind host", "Loopback is the default.");
    numberField(grid, state.config.listener, "listener", "port", "Listening port", "A TCP port from 1 to 65535.", true);
    const wrap = field("Allowed hostnames", "One hostname per line. Wildcards are not accepted."); wrap.classList.add("span-all"); const input = node("textarea"); input.rows = 3; input.value = (state.config.listener.allowed_hosts || []).join("\n"); onValueEdit(input, () => { state.config.listener.allowed_hosts = input.value.split(/\r?\n/).map((value) => value.trim()).filter(Boolean); markDirty(); }); wrap.append(input); grid.append(wrap);
  });
  group(container, "Inference API authentication", "This key protects consumer access to /v1. It is distinct from backend API keys and the browser’s local administration session.", (grid) => {
    secretSetting(grid, state.config.access, "api_key", "Relay API key", "/access/api_key");
    textSetting(grid, state.config.access, "api_key_env", "Relay API key environment variable", "Use a literal key or an environment reference, not both. Remove a saved key explicitly when switching to an environment variable.", {optional: true});
  });
}

export function renderSettings() {
  if (!state.config) return;
  renderLLM($("llm-default-fields"), state.config.llm_defaults, "global:llm");
  renderEmbedding($("embedding-default-fields"), state.config.embedding_defaults, "global:embedding");
  renderTailnet(); renderAccess(); state.renderers.models();
  if ($("override-model").value) renderOverrides($("override-model").value);
  openSettingsTab(state.settingsTab);
}
