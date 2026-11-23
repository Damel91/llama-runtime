/* Admin transport and browser session. The token lifecycle stays here; the page
   state it reads comes from state.js. No DOM access: session end is reported to
   the caller through the thrown error. */

import { state } from "./state.js";

const PREFIX = "/api/admin";
const TOKEN_HEADER = "X-Relay-Admin-Token";

async function adminRequest(path, {method = "GET", body, format = "json"} = {}, token) {
  const headers = {Accept: format === "jsonl" ? "application/x-ndjson" : "application/json"};
  if (token) headers[TOKEN_HEADER] = token;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(PREFIX + path, {
    method, headers, credentials: "same-origin", cache: "no-store", redirect: "error",
    ...(body !== undefined ? {body: JSON.stringify(body)} : {}),
  });
  if (response.ok && format === "jsonl") return response.blob();
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

function adminSession() {
  if (state.token) return Promise.resolve(state.token);
  if (!state.pageActive) return Promise.reject(new Error("The console is inactive."));
  if (state.sessionPending) return state.sessionPending;
  const epoch = state.pageEpoch;
  const attempt = adminRequest("/session").then((session) => {
    if (epoch !== state.pageEpoch) throw new Error("The console session changed.");
    if (session.api_version !== 1 || !session.admin_token) throw new Error("This console requires admin API version 1.");
    state.token = session.admin_token;
    return state.token;
  }).finally(() => { if (state.sessionPending === attempt) state.sessionPending = null; });
  state.sessionPending = attempt;
  return attempt;
}

export async function api(path, options) {
  const token = await adminSession();
  try { return await adminRequest(path, options, token); }
  catch (error) {
    if (error.status !== 401 || error.code !== "relay_admin_token_required") throw error;
    if (state.token === token) state.token = null;
    return adminRequest(path, options, await adminSession());
  }
}
