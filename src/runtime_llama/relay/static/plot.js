/* Telemetry plot panel: a standalone time-series view of the observations the
   relay currently retains. Every render derives only from the snapshot array it
   receives: no accumulation between snapshots, no admin calls, no external
   chart library. Missing native values are never estimated. */

import { $ } from "./dom.js";
import { state } from "./state.js";

const SVG_NS = "http://www.w3.org/2000/svg";

export const PLOT_METRICS = [
  {key: "prompt_tokens", label: "Prompt tokens", value: (summary) => summary.usage?.prompt_tokens, format: (value) => `${Math.round(value).toLocaleString()} tokens`},
  {key: "output_tokens", label: "Output tokens", value: (summary) => summary.usage?.completion_tokens, format: (value) => `${Math.round(value).toLocaleString()} tokens`},
  {key: "total_tokens", label: "Total tokens", value: (summary) => summary.usage?.total_tokens, format: (value) => `${Math.round(value).toLocaleString()} tokens`},
  {key: "prefill_rate", label: "Native prefill rate", value: (summary) => summary.server_timings?.prompt_per_second, format: (value) => `${value.toLocaleString(undefined, {maximumFractionDigits: 1})} tok/s`},
  {key: "decode_rate", label: "Native generation rate", value: (summary) => summary.server_timings?.predicted_per_second, format: (value) => `${value.toLocaleString(undefined, {maximumFractionDigits: 1})} tok/s`},
  {key: "relay_latency", label: "Relay latency", value: (summary) => summary.elapsed_ms, format: (value) => `${value.toLocaleString(undefined, {maximumFractionDigits: 1})} ms`},
  {key: "first_text", label: "Time to first visible text", value: (summary) => summary.first_text_ms, format: (value) => `${value.toLocaleString(undefined, {maximumFractionDigits: 1})} ms`},
];
const DEFAULT_METRIC = "total_tokens";

let bound = false;

function svgNode(tag, attributes, text) {
  const value = document.createElementNS(SVG_NS, tag);
  for (const [name, attribute] of Object.entries(attributes)) value.setAttribute(name, attribute);
  if (text !== undefined) value.textContent = text;
  return value;
}

function metricFor(select) {
  return PLOT_METRICS.find((metric) => metric.key === select.value) || PLOT_METRICS.find((metric) => metric.key === DEFAULT_METRIC);
}

function observationTime(summary) {
  const stamp = Date.parse(summary.finished_at || summary.started_at);
  return Number.isFinite(stamp) ? stamp : null;
}

function timeLabel(millis, span) {
  const date = new Date(millis);
  return span < 6 * 3600 * 1000
    ? date.toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit", second: "2-digit"})
    : date.toLocaleString(undefined, {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"});
}

function valueLabel(value) {
  const magnitude = Math.abs(value);
  if (magnitude >= 1000000) return `${(value / 1000000).toLocaleString(undefined, {maximumFractionDigits: 1})}M`;
  if (magnitude >= 1000) return `${(value / 1000).toLocaleString(undefined, {maximumFractionDigits: magnitude >= 10000 ? 0 : 1})}k`;
  return value.toLocaleString(undefined, {maximumFractionDigits: magnitude < 10 ? 1 : 0});
}

function niceStep(range) {
  const raw = range / 3;
  const magnitude = 10 ** Math.floor(Math.log10(raw || 1));
  for (const multiple of [1, 2, 5, 10]) if (raw <= magnitude * multiple) return magnitude * multiple;
  return magnitude * 10;
}

function renderEmpty(status, message) {
  $("plot-canvas").replaceChildren();
  status.textContent = message;
}

export function renderTelemetryPlot() {
  const select = $("plot-metric");
  const status = $("plot-status");
  if (!select || !status) return;
  if (!bound) {
    bound = true;
    select.addEventListener("change", () => renderTelemetryPlot());
  }
  const metric = metricFor(select);
  if (select.value !== metric.key) select.value = metric.key;
  const recent = state.telemetry?.recent || [];
  if (!recent.length) {
    renderEmpty(status, "No retained observations. The plot stays empty while the relay holds none; nothing is kept across relay restarts.");
    return;
  }
  const points = [];
  for (const summary of recent) {
    const time = observationTime(summary);
    const value = metric.value(summary);
    if (time !== null && typeof value === "number" && Number.isFinite(value)) points.push({time, value});
  }
  points.sort((left, right) => left.time - right.time);
  const missing = recent.length - points.length;
  const suffix = missing ? ` ${missing} without a value for this metric; missing native values are not estimated.` : "";
  if (!points.length) {
    renderEmpty(status, `${recent.length.toLocaleString()} retained observations, none reporting a value for ${metric.label.toLowerCase()}.${suffix}`);
    return;
  }
  const width = 720, height = 240, left = 58, right = 14, top = 14, bottom = 30;
  let x0 = points[0].time, x1 = points.at(-1).time;
  if (x1 === x0) { x0 -= 1000; x1 += 1000; }
  let y0 = Math.min(...points.map((point) => point.value));
  let y1 = Math.max(...points.map((point) => point.value));
  if (y1 === y0) { const pad = Math.max(Math.abs(y1) * 0.05, 1); y0 -= pad; y1 += pad; }
  const span = x1 - x0;
  const plot = (time, value) => [
    left + ((time - x0) / span) * (width - left - right),
    height - bottom - ((value - y0) / (y1 - y0)) * (height - top - bottom),
  ];
  const svg = svgNode("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${metric.label}: ${points.length} retained observations plotted over time.`, class: "telemetry-plot-svg"});
  const step = niceStep(y1 - y0);
  for (let value = Math.ceil(y0 / step) * step; value <= y1 + step * 0.001; value += step) {
    const [, y] = plot(x0, value);
    svg.append(
      svgNode("line", {x1: left, x2: width - right, y1: y, y2: y, class: "plot-grid"}),
      svgNode("text", {x: left - 8, y: y + 4, "text-anchor": "end", class: "plot-label"}, valueLabel(value)),
    );
  }
  for (let index = 0; index <= 3; index += 1) {
    const time = x0 + (span * index) / 3;
    const [x] = plot(time, y0);
    svg.append(
      svgNode("line", {x1: x, x2: x, y1: height - bottom, y2: height - bottom + 5, class: "plot-tick"}),
      svgNode("text", {x, y: height - bottom + 18, "text-anchor": index === 0 ? "start" : index === 3 ? "end" : "middle", class: "plot-label"}, timeLabel(time, span)),
    );
  }
  svg.append(
    svgNode("text", {x: left, y: 10, class: "plot-label"}, metric.label),
    svgNode("polyline", {points: points.map((point) => plot(point.time, point.value).join(",")).join(" "), fill: "none", class: "plot-series"}),
  );
  for (const point of points) {
    const [x, y] = plot(point.time, point.value);
    const dot = svgNode("circle", {cx: x, cy: y, r: 3, class: "plot-point"});
    dot.append(svgNode("title", {}, `${metric.format(point.value)} · ${new Date(point.time).toLocaleString()}`));
    svg.append(dot);
  }
  $("plot-canvas").replaceChildren(svg);
  status.textContent = `${recent.length.toLocaleString()} retained observations · ${points.length.toLocaleString()} with a value for ${metric.label.toLowerCase()}.${suffix}`;
}
