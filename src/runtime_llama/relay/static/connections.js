/* Runtime connections view: endpoint rows, availability observations and the
   add/edit dialog. Draft mutation goes through config.js endpointDraft. */

import { $, node, notice, button } from "./dom.js";
import { state, markDirty } from "./state.js";
import { endpointDraft } from "./config.js";

export function renderConnections() {
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

export function showEndpointAuth() {
  $("endpoint-env-field").hidden = $("endpoint-auth").value !== "env";
  $("endpoint-key-field").hidden = $("endpoint-auth").value !== "key";
  $("endpoint-proxy-field").hidden = $("endpoint-proxy-action").value !== "replace";
}

export function editEndpoint(index = null) {
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

export function applyEndpoint(event) {
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
