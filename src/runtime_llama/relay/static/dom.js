/* DOM construction, formatting and clipboard helpers. No shared state, no admin calls. */

export const $ = (id) => document.getElementById(id);

export function node(tag, className, text) {
  const value = document.createElement(tag);
  if (className) value.className = className;
  if (text !== undefined) value.textContent = String(text);
  return value;
}

export const unknown = (value) => value === null || value === undefined ? "Unknown" : String(value);
export const formatNumber = (value) => typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : "Unknown";
export const formatMs = (value) => typeof value === "number" && Number.isFinite(value) ? `${value.toLocaleString(undefined, {maximumFractionDigits: 1})} ms` : "Unknown";

export function formatDate(value) {
  if (!value) return "Unknown";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "Unknown" : date.toLocaleString();
}

export function notice(message, isError = false) {
  $("notice").textContent = message;
  $("notice").className = `notice${isError ? " error" : ""}`;
  $("notice").hidden = false;
}

export function button(label, handler, variant = "secondary", small = false) {
  const value = node("button", `button ${variant}${small ? " small" : ""}`, label);
  value.type = "button";
  value.addEventListener("click", handler);
  return value;
}

export function field(label, help) {
  const wrap = node("label", "field"); wrap.append(node("span", "", label));
  if (help) wrap.append(node("small", "", help));
  return wrap;
}

export function group(container, title, description, render) {
  const section = node("section", "setting-group"); section.append(node("h3", "", title));
  if (description) section.append(node("p", "", description));
  const grid = node("div", "form-grid"); render(grid); section.append(grid); container.append(section);
}

export function detailPair(container, label, value, technical = false) {
  container.append(node("dt", "", label), node("dd", technical ? "technical-value" : "", value));
}

export async function copyText(value) {
  if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(value);
  else { const input = node("textarea"); input.value = value; input.setAttribute("aria-label", "Text to copy"); document.body.append(input); input.select(); const copied = document.execCommand("copy"); input.remove(); if (!copied) throw new Error("Your browser could not copy this value. Select it and copy manually."); }
}
