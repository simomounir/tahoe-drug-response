// Drug response explorer (phase 5c spec). Data: data/index.json + one shard per drug, written by `make demo-data`.
// Every value reaches the page through textContent or SVG attributes; nothing is parsed as HTML.
"use strict";

const MODELS = [
  { key: "global_mean", label: "per-dose mean", color: "var(--series-1)", shape: "square" },
  { key: "ridge", label: "ridge", color: "var(--series-2)", shape: "diamond" },
  { key: "neural", label: "neural network", color: "var(--series-3)", shape: "triangle" },
];
const SPLITS = {
  random: ["random", "This exact condition (cell line, drug, dose, plate) was held out at random; other conditions with the same drug or cell line were usually in training."],
  unseen_cell_line: ["new cell line", "The whole cell line was held out: the model never saw it during training."],
  unseen_drug: ["new drug", "The whole drug (and near-identical molecules) was held out: the model never saw it during training."],
  both_unseen: ["new drug + new line", "Both the drug and the cell line were held out: the pre-registered test."],
};
const TOP = 20;
const SVGNS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);

let index, byLine = {}, shards = {}, tested = {}, state = { cid: null, split: "both_unseen", all: false };

function el(tag, attrs = {}, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
function option(value, text) { const o = document.createElement("option"); o.value = value; o.textContent = text; return o; }
function fmt(v, d = 2) { return v === null || v === undefined ? "—" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d); }
function score(v) { return v === null || v === undefined ? "—" : v.toFixed(3); }

async function shard(drug) {
  if (!shards[drug]) {
    shards[drug] = fetch("data/" + index.drugs[drug].shard).then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status} loading data for ${index.drugs[drug].name}`);
      return r.json();
    });
    shards[drug].catch(() => delete shards[drug]);  // let a later attempt retry
  }
  return shards[drug];
}

function doseLabel(c, siblings) {
  const unit = c[4] === "uM" ? "µM" : c[4];
  const dup = siblings.filter((s) => s[3] === c[3]).length > 1;
  return `${c[3]} ${unit}` + (dup ? ` · ${c[5]}` : "");
}
function isTested(cid) { return tested[state.split].has(cid); }

function fillSelectors() {
  const line = $("line").value, drug = $("drug").value;
  const drugs = Object.keys(byLine[line] || {}).sort((a, b) => index.drugs[a].name.localeCompare(index.drugs[b].name));
  $("drug").replaceChildren(...drugs.map((d) => option(d, index.drugs[d].name)));
  $("drug").value = drugs.includes(drug) ? drug : drugs[0];
  const conds = (byLine[line][$("drug").value] || []).slice().sort((a, b) => a[3] - b[3]);
  $("dose").replaceChildren(...conds.map((c) => option(c[0], doseLabel(c, conds) + (isTested(c[0]) ? "" : " (not tested in this split)"))));
  if (conds.some((c) => c[0] === state.cid)) $("dose").value = state.cid;
  state.cid = $("dose").value;
}

function select(cid) {
  const c = index.conditions.find((x) => x[0] === cid);
  if (!c) return;
  state.cid = cid;
  $("line").value = c[1];
  fillSelectors();
  $("drug").value = c[2];
  fillSelectors();
  $("dose").value = cid;
  state.cid = cid;
  render();
}

function markers(svg, shape, x, y, color, tip) {
  const s = 5;
  let m;
  if (shape === "square") m = el("rect", { x: x - s, y: y - s, width: 2 * s, height: 2 * s, fill: color });
  else if (shape === "diamond") m = el("polygon", { points: `${x},${y - s - 1} ${x + s + 1},${y} ${x},${y + s + 1} ${x - s - 1},${y}`, fill: color });
  else if (shape === "triangle") m = el("polygon", { points: `${x},${y - s - 1} ${x + s + 1},${y + s} ${x - s - 1},${y + s}`, fill: color });
  else m = el("circle", { cx: x, cy: y, r: 6, fill: "currentColor" });
  m.appendChild(el("title", {}, tip));
  svg.appendChild(m);
}

function drawChart(genes, measured, pred) {
  // Drawn at the container's real width so labels stay 12 px on a phone as on a desktop.
  const width = Math.max(320, Math.round($("chart").clientWidth || 720));
  const n = genes.length, rowH = 24, left = width < 480 ? 76 : 96, right = 12, top = 8, height = top + n * rowH + 36;
  const values = [...measured, ...MODELS.flatMap((m) => pred[m.key] || [])];
  let lo = Math.min(0, ...values), hi = Math.max(0, ...values);
  const pad = (hi - lo) * 0.05 || 1; lo -= pad; hi += pad;
  const x = (v) => left + ((v - lo) / (hi - lo)) * (width - left - right);
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Measured versus predicted log fold-change per gene" });
  const step = [0.5, 1, 2, 5, 10].find((s) => (hi - lo) / s <= 8) || 20;
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) {
    svg.appendChild(el("line", { x1: x(t), x2: x(t), y1: top, y2: top + n * rowH, stroke: "var(--grid)", "stroke-width": t === 0 ? 1.5 : 0.8 }));
    svg.appendChild(el("text", { x: x(t), y: top + n * rowH + 18, "text-anchor": "middle", "font-size": 12, fill: "currentColor" }, (Math.round(t * 10) / 10).toString()));
  }
  svg.appendChild(el("text", { x: (left + width - right) / 2, y: height - 2, "text-anchor": "middle", "font-size": 12, fill: "currentColor" }, "logFC vs DMSO"));
  genes.forEach((g, i) => {
    const y = top + i * rowH + rowH / 2;
    svg.appendChild(el("text", { x: left - 8, y: y + 4, "text-anchor": "end", "font-size": 12, fill: "currentColor" }, g));
    MODELS.forEach((m) => { if (pred[m.key]) markers(svg, m.shape, x(pred[m.key][i]), y, m.color, `${g} · ${m.label}: ${fmt(pred[m.key][i])}`); });
    markers(svg, "circle", x(measured[i]), y, null, `${g} · measured: ${fmt(measured[i])}`);
  });
  $("chart").replaceChildren(svg);
}

async function render() {
  try { await draw(); } catch (e) {
    $("title").textContent = "Could not load this condition.";
    $("subtitle").textContent = `${e.message}. Try again or pick another drug.`;
    $("chart").replaceChildren(); $("scores").tBodies[0].replaceChildren();
  }
}

async function draw() {
  const c = index.conditions.find((x) => x[0] === state.cid);
  const data = (await shard(c[2])).conditions[state.cid];
  const split = data.splits[state.split];
  document.querySelectorAll("#splits button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.split === state.split)));
  $("split-note").textContent = SPLITS[state.split][1];
  const dose = `${c[3]} ${c[4] === "uM" ? "µM" : c[4]}`;
  const siblings = byLine[c[1]][c[2]];
  $("title").textContent = `${index.drugs[c[2]].name.trim()} at ${doseLabel(c, siblings)} in ${index.lines[c[1]]}`;
  const med = index.medians[state.split] || {};
  const body = $("scores").tBodies[0];
  body.replaceChildren(...MODELS.map((m) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th"); th.scope = "row";
    const sw = document.createElement("span"); sw.className = "swatch"; sw.style.background = m.color;
    th.append(sw, m.label);
    const a = document.createElement("td"); a.textContent = split ? score(split.score[m.key]) : "—";
    const e = document.createElement("td"); e.textContent = split ? score(split.scored[m.key]) : "—";
    const b = document.createElement("td"); b.textContent = score(med[m.key]);
    tr.append(th, a, e, b);
    return tr;
  }));
  history.replaceState(null, "", `#${encodeURIComponent(state.cid)}/${state.split}`);
  if (!split) {
    $("subtitle").textContent = `This condition was not a test condition in any repeat of the “${SPLITS[state.split][0]}” split, so there is no held-out prediction to show. Try another split or “Random condition”.`;
    $("chart").replaceChildren(); $("chart-note").textContent = ""; $("values").tHead.replaceChildren(); $("values").tBodies[0].replaceChildren();
    return;
  }
  $("subtitle").textContent = `Predictions from repeat ${split.repeat + 1} of the “${SPLITS[state.split][0]}” split, the first in which this condition was a test condition. ${data.genes.length} differentially expressed genes.`;
  const order = data.genes.map((_, i) => i).sort((a, b) => Math.abs(data.measured[b]) - Math.abs(data.measured[a]) || a - b)
    .slice(0, state.all ? data.genes.length : TOP).sort((a, b) => data.measured[b] - data.measured[a]);
  const genes = order.map((i) => index.genes[data.genes[i]]);
  const measured = order.map((i) => data.measured[i] / 100);
  const pred = Object.fromEntries(MODELS.map((m) => [m.key, split.pred[m.key] ? order.map((i) => split.pred[m.key][i] / 100) : null]));
  drawChart(genes, measured, pred);
  $("chart-note").textContent = (state.all ? `All ${genes.length} DE genes` : `The ${genes.length} most strongly changed DE genes`) + ", most up-regulated first. Markers can overlap when models predict the same value; the numbers are in the table below.";
  const head = document.createElement("tr");
  ["gene", "measured", ...MODELS.map((m) => m.label)].forEach((h) => { const th = document.createElement("th"); th.scope = "col"; th.textContent = h; head.append(th); });
  $("values").tHead.replaceChildren(head);
  $("values").tBodies[0].replaceChildren(...genes.map((g, k) => {
    const tr = document.createElement("tr");
    [g, fmt(measured[k]), ...MODELS.map((m) => fmt(pred[m.key] && pred[m.key][k]))].forEach((v, j) => {
      const td = document.createElement(j === 0 ? "th" : "td"); if (j === 0) td.scope = "row"; td.textContent = v; tr.append(td);
    });
    return tr;
  }));
}

async function main() {
  index = await (await fetch("data/index.json")).json();
  for (const c of index.conditions) ((byLine[c[1]] ??= {})[c[2]] ??= []).push(c);
  for (const s of Object.keys(SPLITS)) tested[s] = new Set(index.tested[s]);
  const lines = Object.keys(byLine).sort((a, b) => index.lines[a].localeCompare(index.lines[b]));
  $("line").replaceChildren(...lines.map((l) => option(l, index.lines[l])));
  $("splits").replaceChildren(...Object.entries(SPLITS).map(([k, [label]]) => {
    const b = document.createElement("button"); b.type = "button"; b.dataset.split = k; b.textContent = label;
    b.addEventListener("click", () => { state.split = k; fillSelectors(); render(); });
    return b;
  }));
  const legend = [["measured", "currentColor"], ...MODELS.map((m) => [m.label, m.color])];
  $("legend").replaceChildren(...legend.map(([t, c]) => { const s = document.createElement("span"); const w = document.createElement("span"); w.className = "swatch"; w.style.background = c; w.style.borderRadius = t === "measured" ? "50%" : "0"; s.append(w, t); return s; }));
  $("line").addEventListener("change", () => { fillSelectors(); render(); });
  $("drug").addEventListener("change", () => { fillSelectors(); render(); });
  $("dose").addEventListener("change", () => { state.cid = $("dose").value; render(); });
  $("all").addEventListener("click", () => { state.all = !state.all; $("all").setAttribute("aria-pressed", String(state.all)); render(); });
  $("random").addEventListener("click", () => { const pool = [...tested[state.split]]; select(pool[Math.floor(Math.random() * pool.length)]); });
  let cid = "", split = "";
  try { [cid, split] = decodeURIComponent(location.hash.slice(1)).split("/"); } catch { /* malformed link: fall back to the default */ }
  if (split && SPLITS[split]) state.split = split;
  select(index.conditions.some((c) => c[0] === cid) ? cid : (index.featured || index.conditions[0][0]));
}

let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => state.cid && render(), 150); });

main().catch((e) => { $("title").textContent = "Could not load the demo data."; $("subtitle").textContent = String(e); });
