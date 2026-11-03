/* The console uses only the versioned local admin API. No browser-side inference. */
"use strict";

const RelayConsole = (() => {
  const PREFIX = "/api/admin";
  const TOKEN_HEADER = "X-Relay-Admin-Token";
  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const state = {
    token: null, revision: null, config: null, secretFields: [], endpointSecrets: new Map(),
    catalog: {revision: null, observed_at: null, runtimes: []}, status: null,
    requests: {active: [], last_completed: null}, view: "models", settingsTab: "generation",
    dirty: false, errors: new Map(), rawFields: new Map(), selectedRow: null,
    selectedModels: new Set(), endpointIndex: null, exportValue: null, poll: null, tailnetDraft: null,
  };

  function optionalValue(target, key, raw, kind = "number") {
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

  function secretValue(target, key, action, value) {
    if (action === "keep") return;
    if (action === "clear") { target[key] = null; return; }
    if (action !== "replace" || typeof value !== "string" || !value) {
      throw new Error("Enter a value to replace the saved secret, or choose Keep current.");
    }
    target[key] = value;
  }

  function rowsFromCatalog(catalog) {
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

  function savePayload(revision, config) {
    if (!Number.isInteger(revision) || revision < 0) throw new Error("Reload settings before saving.");
    return {expected_revision: revision, config: clone(config)};
  }

  function endpointDraft(previous, values, savedSecrets = {}) {
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

  const $ = (id) => document.getElementById(id);
  function node(tag, className, text) {
    const value = document.createElement(tag);
    if (className) value.className = className;
    if (text !== undefined) value.textContent = String(text);
    return value;
  }
  const unknown = (value) => value === null || value === undefined ? "Unknown" : String(value);
  const formatNumber = (value) => typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : "Unknown";
  const formatMs = (value) => typeof value === "number" && Number.isFinite(value) ? `${value.toLocaleString(undefined, {maximumFractionDigits: 1})} ms` : "Unknown";
  function formatDate(value) {
    if (!value) return "Unknown";
    const date = new Date(value);
    return Number.isNaN(date.valueOf()) ? "Unknown" : date.toLocaleString();
  }
  function notice(message, isError = false) {
    $("notice").textContent = message;
    $("notice").className = `notice${isError ? " error" : ""}`;
    $("notice").hidden = false;
  }
  function markDirty() {
    state.dirty = true;
    updateSaveState();
  }
  function updateSaveState() {
    $("dirty-label").hidden = !state.dirty;
    $("save-config").disabled = !state.config || !state.dirty || state.errors.size > 0;
  }
  function button(label, handler, variant = "secondary", small = false) {
    const value = node("button", `button ${variant}${small ? " small" : ""}`, label);
    value.type = "button";
    value.addEventListener("click", handler);
    return value;
  }
  async function busy(control, operation) {
    if (control.disabled) return;
    control.disabled = true;
    control.setAttribute("aria-busy", "true");
    try { await operation(); }
    catch (error) { notice(error.message || "The local relay could not complete this action.", true); }
    finally {
      control.disabled = false; control.removeAttribute("aria-busy"); updateSaveState();
      if (state.status) renderStatus();
      $("clear-last").disabled = !state.requests.last_completed;
    }
  }
  async function api(path, {method = "GET", body, token = true} = {}) {
    const headers = {Accept: "application/json"};
    if (token && state.token) headers[TOKEN_HEADER] = state.token;
    if (body !== undefined) headers["Content-Type"] = "application/json";
    const response = await fetch(PREFIX + path, {
      method, headers, credentials: "same-origin", cache: "no-store", redirect: "error",
      ...(body !== undefined ? {body: JSON.stringify(body)} : {}),
    });
    let value;
    try { value = await response.json(); }
    catch (_) { throw new Error("The relay returned an unreadable response. Reload the console to reconnect."); }
    if (!response.ok) {
      const error = new Error(value.error?.message || value.error?.code || `The relay returned HTTP ${response.status}.`);
      error.status = response.status;
      error.code = value.error?.code;
      throw error;
    }
    return value;
  }
  function loadConfig(value) {
    state.revision = value.revision;
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
    renderConnections();
    renderSettings();
  }
  async function reloadConfig(discard = true) {
    if (discard && state.dirty && !window.confirm("Discard unsaved changes and reload the saved relay settings?")) return;
    const session = await api("/session", {token: false});
    if (session.api_version !== 1 || !session.admin_token) throw new Error("This console requires admin API version 1.");
    state.token = session.admin_token;
    loadConfig(await api("/config"));
    await refreshLocalState();
    notice("Saved settings loaded.");
  }
  async function saveConfig() {
    document.activeElement?.blur();
    if (state.errors.size) throw new Error("Correct the highlighted fields before saving.");
    try {
      const value = await api("/config", {method: "PUT", body: savePayload(state.revision, state.config)});
      loadConfig(value);
      $("restart-notice").hidden = value.restart_required !== true;
      notice("Settings saved. New requests use the new configuration; active requests keep their original settings.");
      await refreshLocalState();
    } catch (error) {
      if (error.status === 409) {
        $("conflict-notice").hidden = false;
        notice("Your draft has not been overwritten. Reload saved settings to resolve the revision conflict.", true);
      } else throw error;
    }
  }
  async function refreshLocalState() {
    const results = await Promise.allSettled([api("/status"), api("/requests")]);
    if (results[0].status === "fulfilled") { state.status = results[0].value; renderStatus(); }
    if (results[1].status === "fulfilled") { state.requests = results[1].value; renderRequests(); }
    const failed = results.find((result) => result.status === "rejected");
    if (failed) throw failed.reason;
  }
  function renderStatus() {
    if (!state.status) return;
    $("service-state").textContent = "Connected";
    $("service-state").className = "status-tag ready";
    $("api-url").textContent = `${window.location.origin}/v1`;
    $("footer-state").textContent = `Local relay connected. Saved configuration revision ${state.status.revision ?? state.revision}.`;
    $("restart-notice").hidden = state.status.restart_required !== true;
    const helper = state.status.tailnet || {};
    const helperState = helper.state || helper.status || (helper.running === true ? "Running" : helper.running === false ? "Stopped" : "Unknown");
    $("tailnet-state").textContent = helperState;
    $("tailnet-state").className = `status-tag${["running", "ready"].includes(String(helperState).toLowerCase()) ? " ready" : ""}`;
    $("start-tailnet").disabled = !state.config?.tailnet || ["running", "ready", "starting"].includes(String(helperState).toLowerCase());
    $("stop-tailnet").disabled = ["stopped", "not_configured", "disabled"].includes(String(helperState).toLowerCase());
  }
  function navigate(view, focus = false) {
    const descriptions = {
      models: ["Models", "The models your configured runtimes actually report."],
      connections: ["Connections", "Connect native runtimes without moving their models or processes."],
      requests: ["Requests", "Inspect the work owned by this relay and its latest completion."],
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
    if (state.view === "requests" && state.token) refreshLocalState().catch((error) => notice(error.message, true));
    if (focus) $("workspace").focus({preventScroll: true});
    schedulePoll();
  }
  function schedulePoll() {
    if (state.poll !== null) window.clearTimeout(state.poll);
    state.poll = null;
    if (state.view !== "requests" || document.hidden || !state.token) return;
    state.poll = window.setTimeout(async () => {
      try { await refreshLocalState(); }
      catch (error) { notice(error.message, true); }
      finally { schedulePoll(); }
    }, 2000);
  }
  function capabilityLabel(model) {
    const labels = [];
    if (model.capabilities?.chat === true) labels.push("Chat");
    if (model.capabilities?.embeddings === true) labels.push("Embeddings");
    if (!labels.length) labels.push(model.capabilities?.chat === false && model.capabilities?.embeddings === false ? "Unsupported" : "Unknown");
    return labels;
  }
  function visibleRows() {
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
  function renderModels() {
    const allRows = rowsFromCatalog(state.catalog);
    const rows = visibleRows();
    $("model-rows").replaceChildren();
    for (const row of rows) {
      const tr = node("tr", row.key === state.selectedRow ? "selected" : "");
      const checkCell = node("td", "checkbox-column");
      const check = node("input"); check.type = "checkbox"; check.dataset.exportModel = row.model.id;
      check.setAttribute("aria-label", `Select ${row.model.id} for OpenCode export`);
      check.checked = state.selectedModels.has(row.model.id);
      check.disabled = row.model.capabilities?.chat === false || modelAvailability(row) !== "loaded";
      check.addEventListener("change", () => {
        check.checked ? state.selectedModels.add(row.model.id) : state.selectedModels.delete(row.model.id);
        for (const input of document.querySelectorAll("[data-export-model]")) if (input.dataset.exportModel === row.model.id) input.checked = check.checked;
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
      item.append(node("span", "node-status"), node("strong", "", runtime.name || runtime.endpoint?.name || "Unnamed endpoint"), node("span", "", runtime.backend || "Unknown runtime"));
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
    $("export-selection").textContent = state.selectedModels.size ? `${state.selectedModels.size} selected model selector${state.selectedModels.size === 1 ? "" : "s"}. The export uses saved relay settings and observed capabilities.` : "No models checked: use the exportable observed chat catalog. Unknown client capabilities are reported in the export warnings.";
  }
  function detailPair(container, label, value, technical = false) {
    container.append(node("dt", "", label), node("dd", technical ? "technical-value" : "", value));
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
    const actions = node("div", "detail-actions"); actions.append(button("Configure model settings", () => openModelSettings(row.model.id))); detail.append(actions);
    detail.append(node("p", "detail-note", "These are runtime observations, not a reservation or an instruction to load the model. Context and GPU allocation remain on the server."));
  }
  async function refreshCatalog(control = $("refresh-catalog")) {
    await busy(control, async () => {
      state.catalog = await api("/catalog/refresh", {method: "POST"});
      const ids = new Set(rowsFromCatalog(state.catalog).map((row) => row.model.id));
      state.selectedModels = new Set([...state.selectedModels].filter((id) => ids.has(id)));
      renderModels(); renderConnections();
      notice(`Catalog refreshed: ${rowsFromCatalog(state.catalog).length.toLocaleString()} observed model entries. No model was loaded.`);
    });
  }
  function renderConnections() {
    if (!state.config) return;
    const list = $("endpoint-list"); list.replaceChildren();
    state.config.runtime_endpoints.forEach((endpoint, index) => {
      const row = node("article", "connection-row");
      const info = node("div", "connection-info"); info.append(node("h3", "", endpoint.name), node("p", "connection-url technical-value", endpoint.base_url));
      const meta = node("div", "connection-meta");
      for (const label of [endpoint.backend === "auto" ? "Automatic runtime" : endpoint.backend, endpoint.use_tailnet ? "Embedded private network" : "Direct connection", endpoint.verify_ssl === false ? "TLS verification disabled" : "TLS verified when HTTPS"] ) meta.append(node("span", "capability-tag", label));
      const observed = (state.catalog.runtimes || []).find((runtime) => (runtime.name || runtime.endpoint?.name) === endpoint.name);
      if (observed) meta.append(node("span", `state-pill${observed.available === true ? " ready" : observed.available === false ? " error" : ""}`, observed.available === true ? "Available at last observation" : observed.available === false ? "Unavailable at last observation" : "Unknown"));
      info.append(meta);
      if (observed?.diagnostics?.length) info.append(node("p", "cell-secondary", observed.diagnostics.join(", ")));
      row.append(info);
      const actions = node("div", "connection-actions"); actions.append(button("Edit", () => editEndpoint(index), "secondary", true), button("Remove", () => {
        state.config.runtime_endpoints.splice(index, 1); markDirty(); renderConnections(); notice("Connection removed from the draft. Save changes to apply it to new requests.");
      }, "danger", true)); row.append(actions); list.append(row);
    });
    $("connections-empty").hidden = state.config.runtime_endpoints.length > 0;
  }
  function showEndpointAuth() {
    $("endpoint-env-field").hidden = $("endpoint-auth").value !== "env";
    $("endpoint-key-field").hidden = $("endpoint-auth").value !== "key";
    $("endpoint-proxy-field").hidden = $("endpoint-proxy-action").value !== "replace";
  }
  function editEndpoint(index = null) {
    state.endpointIndex = index;
    const endpoint = index === null ? {} : state.config.runtime_endpoints[index];
    const saved = state.endpointSecrets.get(endpoint.name) || {};
    $("endpoint-dialog-title").textContent = index === null ? "Add connection" : "Edit connection";
    $("endpoint-name").value = endpoint.name || ""; $("endpoint-url").value = endpoint.base_url || "";
    $("endpoint-backend").value = endpoint.backend || "auto"; $("endpoint-transport").value = endpoint.use_tailnet ? "tailnet" : "direct";
    $("endpoint-tls").checked = endpoint.verify_ssl !== false; $("endpoint-http").value = endpoint.http_connection_policy || "close";
    $("endpoint-auth").value = endpoint.api_key_env ? "env" : endpoint.api_key || saved.api_key ? "key" : "none";
    $("endpoint-key-env").value = endpoint.api_key_env || ""; $("endpoint-key").value = ""; $("endpoint-proxy").value = "";
    $("endpoint-key-help").textContent = saved.api_key || endpoint.api_key ? "A key is configured. Leave empty to keep it under the same connection name." : "Enter a key. It will not be read back by the browser.";
    $("endpoint-proxy-action").value = "keep"; $("endpoint-error").hidden = true; showEndpointAuth();
    $("endpoint-dialog").showModal(); $("endpoint-name").focus();
  }
  function applyEndpoint(event) {
    event.preventDefault();
    try {
      const previous = state.endpointIndex === null ? null : state.config.runtime_endpoints[state.endpointIndex];
      const result = endpointDraft(previous, {
        name: $("endpoint-name").value, base_url: $("endpoint-url").value, backend: $("endpoint-backend").value,
        verify_ssl: $("endpoint-tls").checked, use_tailnet: $("endpoint-transport").value === "tailnet", http_connection_policy: $("endpoint-http").value,
        auth: $("endpoint-auth").value, api_key_env: $("endpoint-key-env").value, api_key: $("endpoint-key").value,
        proxy_action: $("endpoint-proxy-action").value, proxy_url: $("endpoint-proxy").value,
      }, state.endpointSecrets.get(previous?.name) || {});
      if (state.config.runtime_endpoints.some((endpoint, index) => endpoint.name === result.name && index !== state.endpointIndex)) throw new Error("Choose a unique connection name.");
      if (state.endpointIndex === null) state.config.runtime_endpoints.push(result); else state.config.runtime_endpoints[state.endpointIndex] = result;
      $("endpoint-dialog").close(); markDirty(); renderConnections(); notice("Connection applied to the draft. Save changes, then refresh the catalog to observe it.");
    } catch (error) { $("endpoint-error").textContent = error.message; $("endpoint-error").hidden = false; }
  }
  function field(label, help) {
    const wrap = node("label", "field"); wrap.append(node("span", "", label));
    if (help) wrap.append(node("small", "", help));
    return wrap;
  }
  const fieldKey = (scope, key) => JSON.stringify([scope, key]);
  function clearScopeDiagnostics(scope) {
    for (const values of [state.errors, state.rawFields]) for (const key of [...values.keys()]) {
      try { if (JSON.parse(key)[0] === scope) values.delete(key); } catch (_) { /* Other field domains retain their own state. */ }
    }
  }
  function group(container, title, description, render) {
    const section = node("section", "setting-group"); section.append(node("h3", "", title));
    if (description) section.append(node("p", "", description));
    const grid = node("div", "form-grid"); render(grid); section.append(grid); container.append(section);
  }
  function onValueEdit(control, update) {
    control.addEventListener("input", update);
    control.addEventListener("change", update);
  }
  function editField(control, errorKey, mutate) {
    const error = node("span", "field-error"); error.hidden = true;
    onValueEdit(control, () => {
      try { mutate(control.value); state.errors.delete(errorKey); error.hidden = true; control.removeAttribute("aria-invalid"); }
      catch (failure) { state.errors.set(errorKey, failure.message); error.textContent = failure.message; error.hidden = false; control.setAttribute("aria-invalid", "true"); }
      markDirty();
    });
    if (state.errors.has(errorKey)) { error.textContent = state.errors.get(errorKey); error.hidden = false; control.setAttribute("aria-invalid", "true"); }
    return error;
  }
  function numberField(grid, settings, scope, key, label, help, integer = false) {
    const wrap = field(label, help); const input = node("input"); input.type = "number"; input.step = integer ? "1" : "any";
    input.placeholder = "Inherit default"; input.value = settings[key] ?? "";
    const error = editField(input, fieldKey(scope, key), (raw) => {
      if (input.validity.badInput) throw new Error("Enter a valid number or leave this field empty.");
      optionalValue(settings, key, raw, integer ? "integer" : "number");
    }); wrap.append(input, error); grid.append(wrap);
  }
  function selectField(grid, settings, scope, key, label, options, help) {
    const wrap = field(label, help); const input = node("select");
    for (const [value, text] of [["", "Inherit default"], ...options]) { const option = node("option", "", text); option.value = value; input.append(option); }
    input.value = own(settings, key) ? String(settings[key]) : "";
    const error = editField(input, fieldKey(scope, key), (raw) => optionalValue(settings, key, raw, options.every(([value]) => value === "true" || value === "false") ? "boolean" : "string"));
    wrap.append(input, error); grid.append(wrap);
  }
  function jsonField(grid, settings, scope, key, label, help) {
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
  function openSettingsTab(name) {
    state.settingsTab = name;
    for (const tab of document.querySelectorAll("[data-settings-tab]")) { const active = tab.dataset.settingsTab === name; tab.setAttribute("aria-selected", String(active)); tab.tabIndex = active ? 0 : -1; }
    for (const panel of document.querySelectorAll(".settings-panel")) panel.hidden = panel.id !== `settings-${name}`;
  }
  function openModelSettings(id) {
    window.location.hash = "settings"; navigate("settings"); openSettingsTab("overrides"); $("override-model").value = id; renderOverrides(id); $("override-model").focus();
  }
  function renderOverrides(id) {
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
  function renderTailnet() {
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
  function renderSettings() {
    if (!state.config) return;
    renderLLM($("llm-default-fields"), state.config.llm_defaults, "global:llm");
    renderEmbedding($("embedding-default-fields"), state.config.embedding_defaults, "global:embedding");
    renderTailnet(); renderAccess(); renderModels();
    if ($("override-model").value) renderOverrides($("override-model").value);
    openSettingsTab(state.settingsTab);
  }
  function requestCard(summary, active) {
    const card = node("article", "request-summary"); const header = node("div", "request-summary-header"); const identity = node("div");
    identity.append(node("h3", "", summary.selected_model || summary.model || "Model not reported"), node("p", "technical-value", `${summary.kind || "Request"} · ${summary.request_id || "ID not reported"}`));
    header.append(identity);
    if (active) { const cancel = button(summary.status === "cancelling" ? "Stopping" : "Stop request", () => busy(cancel, async () => { await api(`/requests/${encodeURIComponent(summary.request_id)}/cancel`, {method: "POST"}); await refreshLocalState(); notice("Cancellation requested for this invocation."); }), "danger", true); cancel.disabled = summary.status === "cancelling"; header.append(cancel); }
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
  function renderRequests() {
    const active = state.requests.active || []; const list = $("active-requests"); list.replaceChildren();
    for (const summary of active) list.append(requestCard(summary, true));
    $("requests-empty").hidden = active.length > 0; $("active-nav-count").hidden = active.length === 0; $("active-nav-count").textContent = String(active.length);
    $("last-request").replaceChildren();
    if (state.requests.last_completed) $("last-request").append(requestCard(state.requests.last_completed, false));
    $("last-empty").hidden = Boolean(state.requests.last_completed); $("clear-last").disabled = !state.requests.last_completed;
  }
  async function copyText(value) {
    if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(value);
    else { const input = node("textarea"); input.value = value; input.setAttribute("aria-label", "Text to copy"); document.body.append(input); input.select(); const copied = document.execCommand("copy"); input.remove(); if (!copied) throw new Error("Your browser could not copy this value. Select it and copy manually."); }
  }
  function openExport() {
    updateModelSelection(); $("export-dialog").showModal(); $("export-version").focus();
  }
  async function generateExport() {
    const query = new URLSearchParams({version: $("export-version").value});
    for (const id of state.selectedModels) query.append("models", id);
    const value = await api(`/opencode?${query.toString()}`);
    state.exportValue = {config: value.config, filename: value.filename || `opencode-relay-${$("export-version").value}.json`};
    $("export-output").value = JSON.stringify(value.config, null, 2) + "\n";
    $("copy-export").disabled = false; $("download-export").disabled = false;
    $("export-warnings").replaceChildren();
    for (const warning of value.warnings || []) $("export-warnings").append(node("div", "", warning));
    $("export-warnings").hidden = !(value.warnings || []).length;
  }
  function downloadExport() {
    if (!state.exportValue) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(state.exportValue.config, null, 2) + "\n"], {type: "application/json"}));
    const link = node("a"); link.href = url; link.download = state.exportValue.filename.replace(/[\\/]/g, "_"); document.body.append(link); link.click(); link.remove(); window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }
  function bind() {
    for (const link of document.querySelectorAll("[data-view]")) link.addEventListener("click", () => navigate(link.dataset.view, true));
    for (const control of document.querySelectorAll("[data-navigate]")) control.addEventListener("click", () => { window.location.hash = control.dataset.navigate; navigate(control.dataset.navigate, true); });
    window.addEventListener("hashchange", () => navigate(window.location.hash.slice(1)));
    document.addEventListener("visibilitychange", schedulePoll);
    window.addEventListener("pagehide", () => { if (state.poll !== null) window.clearTimeout(state.poll); state.token = null; });
    window.addEventListener("beforeunload", (event) => { if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });
    $("save-config").addEventListener("click", () => busy($("save-config"), saveConfig));
    for (const id of ["reload-config", "resolve-conflict"]) $(id).addEventListener("click", () => busy($(id), () => reloadConfig(true)));
    $("copy-api").addEventListener("click", () => busy($("copy-api"), async () => { await copyText(`${window.location.origin}/v1`); notice("Local API URL copied."); }));
    for (const id of ["model-search", "model-kind", "model-availability"]) $(id).addEventListener(id === "model-search" ? "input" : "change", renderModels);
    $("select-visible-models").addEventListener("change", () => { const selected = $("select-visible-models").checked; for (const row of visibleRows()) if (row.model.capabilities?.chat !== false && modelAvailability(row) === "loaded") selected ? state.selectedModels.add(row.model.id) : state.selectedModels.delete(row.model.id); renderModels(); });
    $("refresh-catalog").addEventListener("click", () => refreshCatalog());
    $("add-endpoint").addEventListener("click", () => editEndpoint());
    for (const control of document.querySelectorAll('[data-action="add-endpoint"]')) control.addEventListener("click", () => editEndpoint());
    for (const id of ["endpoint-auth", "endpoint-proxy-action"]) $(id).addEventListener("change", showEndpointAuth);
    $("endpoint-form").addEventListener("submit", applyEndpoint);
    $("endpoint-dialog").addEventListener("close", () => { $("endpoint-key").value = ""; $("endpoint-proxy").value = ""; });
    for (const control of document.querySelectorAll("[data-close-dialog]")) control.addEventListener("click", () => $(control.dataset.closeDialog).close());
    $("refresh-requests").addEventListener("click", () => busy($("refresh-requests"), refreshLocalState));
    $("clear-last").addEventListener("click", () => busy($("clear-last"), async () => { await api("/requests/last-completed", {method: "DELETE"}); await refreshLocalState(); notice("Last-completed summary cleared."); }));
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
    $("open-export").addEventListener("click", openExport); $("generate-export").addEventListener("click", () => busy($("generate-export"), generateExport));
    $("copy-export").addEventListener("click", () => busy($("copy-export"), async () => { if (state.exportValue) { await copyText(JSON.stringify(state.exportValue.config, null, 2) + "\n"); notice("OpenCode configuration copied. Existing files were not changed."); } }));
    $("download-export").addEventListener("click", downloadExport);
  }
  async function boot() {
    bind(); navigate(window.location.hash.slice(1));
    try {
      const session = await api("/session", {token: false});
      if (session.api_version !== 1 || !session.admin_token) throw new Error("This console requires admin API version 1.");
      state.token = session.admin_token;
      loadConfig(await api("/config"));
      const results = await Promise.allSettled([api("/catalog"), refreshLocalState()]);
      if (results[0].status === "fulfilled") { state.catalog = results[0].value; renderModels(); renderConnections(); }
      for (const result of results) if (result.status === "rejected") notice(result.reason.message, true);
      schedulePoll();
    } catch (error) {
      $("service-state").textContent = "Not connected"; $("service-state").className = "status-tag error";
      $("footer-state").textContent = "The local administration connection is unavailable.";
      notice(error.message || "Could not connect to the local relay. Reload settings to reconnect.", true);
    }
  }

  return {boot, optionalValue, secretValue, endpointDraft, savePayload, rowsFromCatalog, modelAvailability};
})();

if (typeof module !== "undefined" && module.exports) module.exports = RelayConsole;
if (typeof document !== "undefined") RelayConsole.boot();
