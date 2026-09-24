/* SEAF web dashboard - single-page port of the Streamlit app.
 * The trained models run in a Web Worker (engine.js); all state lives in
 * memory for this viewer and resets on reload (like the free-tier Python app). */
(function () {
  "use strict";

  // ------------------------------------------------------------------ helpers
  const $ = (s, el) => (el || document).querySelector(s);
  const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
  const f3 = (x) => (x == null || isNaN(x) ? "—" : Number(x).toFixed(3));
  const now = () => Date.now() / 1000;
  const INDEX = JSON.parse($("#index-data").textContent);
  const GLOSS = {
    SHAP: "SHAP splits a prediction into per-feature contributions that add up exactly to the model's output: an exact receipt of how the decision was reached.",
    trust: "Trust T (0-1) averages Confidence, (1 - Anomaly) and (1 - Stability deviation). Low trust sends the decision to a human reviewer.",
    sigma: "σ deviation: how many standard deviations a feature's SHAP contribution is from its usual value on normal traffic. |σ| above ~3 is unusual.",
    C: "Confidence C = max(p, 1-p): how sure the model is about its own decision.",
    A: "Anomaly A: how unusual the explanation looks compared with normal model reasoning (Isolation Forest on standardised SHAP vectors), scaled 0-1 on clean data.",
    S: "Stability deviation S: inputs are nudged by ±2% eight times; S measures how unusually stable or volatile the explanation is compared with normal decisions.",
    threshold: "Decisions with trust below this threshold go to review. Reviewer outcomes re-tune it to catch the most attacks within a 10% false-alarm budget.",
    pp: "Prediction-preserving attack: inputs change so the decision stays the same but its stated reasons change. Output-only audits cannot see it.",
    evasion: "Evasion attack: inputs are nudged just enough to flip the decision.",
    fpr: "False-positive rate: share of legitimate decisions sent to review.",
    det: "Detection rate: share of attacked decisions sent to review.",
    auc: "ROC-AUC: probability that a random attacked record looks more suspicious than a random clean one. 0.5 = chance, 1.0 = perfect.",
    drift: "Drift: the population slowly changes (e.g. blood pressure readings creep up). Attacks arrive as sudden, isolated deviations instead.",
  };
  const tip = (k) => `<span class="tip" tabindex="0" data-tip="${esc(GLOSS[k] || k)}">?</span>`;
  const badge = (t, tone = "neutral", dot = true) => `<span class="badge ${tone}">${dot ? '<span class="dot"></span>' : ""}${esc(t)}</span>`;
  const vbadge = (v) => (v === "Accept" ? badge("Accept", "safe") : badge("Review", "review"));
  function kpi(label, value, delta, tone = "accent", dcls = "", tk = null) {
    const c = { accent: "var(--accent)", safe: "var(--safe)", review: "var(--review)", attack: "var(--attack)", drift: "var(--drift)", neutral: "var(--border-strong)" }[tone];
    return `<div class="kpi" style="--kc:${c}"><div class="l">${esc(label)}${tk ? tip(tk) : ""}</div><div class="v">${esc(value)}</div><div class="d ${dcls}">${delta == null ? "&nbsp;" : esc(delta)}</div></div>`;
  }
  const kpis = (arr) => `<div class="kpis">${arr.join("")}</div>`;
  const section = (t, tk) => `<div class="section-title">${esc(t)}${tk ? tip(tk) : ""}</div>`;
  const callout = (h, warn) => `<div class="callout${warn ? " warn" : ""}">${h}</div>`;
  const head = (eyebrow, title, sub) => `<div class="page-head"><div class="eyebrow">${esc(eyebrow)}</div><h1>${esc(title)}</h1>${sub ? `<p>${sub}</p>` : ""}</div>`;
  function verdictPanel(v, T, thr, sub) {
    const ok = v === "Accept";
    const s = sub || (ok ? `Trust ${f3(T)} ≥ threshold ${f3(thr)}: explanation consistent with normal reasoning.`
                         : `Trust ${f3(T)} < threshold ${f3(thr)}: explanation deviates from normal reasoning.`);
    return `<div class="verdict ${ok ? "safe" : "review"}"><div class="ic">${ok ? "✓" : "!"}</div><div><div class="t">${ok ? "Accept" : "Route to review"}</div><div class="s">${esc(s)}</div></div></div>`;
  }
  function toast(msg, tone = "") {
    const box = $("#toasts"), el = document.createElement("div");
    el.className = "toast " + tone; el.textContent = msg; box.appendChild(el);
    while (box.children.length > 4) box.firstChild.remove();
    setTimeout(() => el.remove(), 4200);
  }
  function mkRng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const rand = mkRng((Date.now() & 0xffffff) ^ 0x9e37);

  // ------------------------------------------------------------------ theme + plotly
  function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
  function col() {
    return { text: css("--text"), text2: css("--text-2"), muted: css("--muted"), grid: css("--grid"), surface: css("--surface"),
             surface2: css("--surface-2"), border: css("--border-strong"), accent: css("--accent"), safe: css("--safe"),
             review: css("--review"), attack: css("--attack"), drift: css("--drift"), s: [css("--s1"), css("--s2"), css("--s3"), css("--s4")] };
  }
  function rgba(hex, a) { const h = hex.replace("#", ""); const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16); return `rgba(${r},${g},${b},${a})`; }
  function plot(el, data, layout, react) {
    if (!window.Plotly || !el) return;
    const c = col();
    const base = {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", margin: { l: 8, r: 8, t: 30, b: 8 },
      font: { family: "Inter, system-ui, sans-serif", size: 12.5, color: c.text2 }, colorway: c.s,
      xaxis: { gridcolor: c.grid, zerolinecolor: c.border, tickfont: { color: c.muted }, automargin: true, title: { font: { color: c.muted, size: 12 } } },
      yaxis: { gridcolor: c.grid, zerolinecolor: c.border, tickfont: { color: c.muted }, automargin: true, title: { font: { color: c.muted, size: 12 } } },
      legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1, bgcolor: "rgba(0,0,0,0)", font: { color: c.text2, size: 12 } },
      hoverlabel: { bgcolor: c.surface2, bordercolor: c.border, font: { family: "Inter, sans-serif", color: c.text } },
      bargap: 0.28,
    };
    const merged = deepMerge(base, layout || {});
    // Plotly >= 3 sizes responsive charts from their container: pin the height.
    if (merged.height) el.style.height = merged.height + "px";
    (react ? Plotly.react : Plotly.newPlot)(el, data, merged, { displayModeBar: false, responsive: true });
  }
  function deepMerge(a, b) {
    const o = Object.assign({}, a);
    for (const k of Object.keys(b)) o[k] = b[k] && typeof b[k] === "object" && !Array.isArray(b[k]) && a[k] && typeof a[k] === "object" ? deepMerge(a[k], b[k]) : b[k];
    return o;
  }
  const MONO = "JetBrains Mono, ui-monospace, Menlo, Consolas, monospace";

  // ------------------------------------------------------------------ worker
  const WORKER_SRC = `
    const M = {};
    self.onmessage = (e) => {
      const { id, op, a } = e.data;
      try {
        let r;
        if (op === "load") { M[a.key] = SEAF.load(JSON.parse(a.text)); r = true; }
        else if (op === "score") r = SEAF.score(M[a.key], a.X, a.opts);
        else if (op === "pp") { const at = SEAF.attackPP(M[a.key], a.x, a.o); r = { at, s: SEAF.score(M[a.key], [at.x_orig, at.x_adv]) }; }
        else if (op === "ev") { const at = SEAF.attackEvasion(M[a.key], a.x, a.o); r = { at, s: SEAF.score(M[a.key], [at.x_orig, at.x_adv]) }; }
        self.postMessage({ id, r });
      } catch (err) { self.postMessage({ id, err: String(err && err.stack || err) }); }
    };`;
  const worker = new Worker(URL.createObjectURL(new Blob([$("#engine").textContent, WORKER_SRC], { type: "text/javascript" })));
  let rpcId = 0; const pending = new Map();
  worker.onmessage = (e) => { const p = pending.get(e.data.id); pending.delete(e.data.id); e.data.err ? p.reject(new Error(e.data.err)) : p.resolve(e.data.r); };
  const call = (op, a) => new Promise((resolve, reject) => { const id = ++rpcId; pending.set(id, { resolve, reject }); worker.postMessage({ id, op, a }); });

  // ------------------------------------------------------------------ state
  const S = {
    page: "home", domain: "finance", dom: {}, loading: {}, results: null,
    thr: {}, thrHist: {}, audit: [], nextId: 1, reviewDelta: null, demo: null,
    live: null, predict: {}, attack: {}, auditSel: null, analyticsTab: "paper", reviewPage: 0,
  };

  async function ensureDomain(key) {
    if (S.dom[key]) return S.dom[key];
    if (S.loading[key]) return S.loading[key];
    S.loading[key] = (async () => {
      const res = await fetch(new URL(`data/${key}.json`, location.href));
      if (!res.ok) throw new Error(`Could not load data/${key}.json (${res.status})`);
      const text = await res.text();
      await call("load", { key, text });
      const d = JSON.parse(text);
      delete d.rf; delete d.iso;
      d.recIndex = new Map(d.records.idx.map((v, i) => [v, i]));
      S.dom[key] = d;
      S.thr[key] = d.seaf.default_threshold;
      S.thrHist[key] = [{ ts: now() - 86400, t: d.seaf.default_threshold, reason: "initial: 5th pct of baseline trust" }];
      seed(d);
      return d;
    })();
    return S.loading[key];
  }
  function seed(d) {
    const r = mkRng(11), n = d.seed.length, t0 = now() - 86400;
    const order = d.seed.map((_, i) => i).sort(() => r() - 0.5);
    order.forEach((i, k) => {
      const rec = d.seed[i];
      let status = rec.verdict === "Review" ? "pending" : "auto_accepted";
      if (rec.verdict === "Review" && r() < 0.55) status = rec.sim_label === "attack" ? "confirmed_attack" : "approved";
      const x = d.features.map((f) => rec.features[f]);
      S.audit.push({ id: S.nextId++, ts: t0 + (k + 1) * (82800 / n), domain: d.key, source: "seed", ref: rec.ref, pred: rec.pred,
        p: rec.proba, C: rec.C, A: rec.A, S: rec.S, T: rec.T, threshold: rec.threshold, verdict: rec.verdict, status,
        sim: rec.sim_label, top: rec.top_deviations, x, phi: d.features.map((f) => rec.phi[f]), base: rec.base_value });
    });
  }
  function logRec(d, r, source, ref, sim) {
    const thr = S.thr[d.key], verdict = r.T < thr ? "Review" : "Accept";
    const e = { id: S.nextId++, ts: now(), domain: d.key, source, ref, pred: r.pred, p: r.p, C: r.C, A: r.A, S: r.S, T: r.T,
      threshold: thr, verdict, status: verdict === "Review" ? "pending" : "auto_accepted", sim, top: r.top, x: r.x, phi: r.phi, base: r.base };
    S.audit.push(e);
    return e;
  }

  // ------------------------------------------------------------------ domain helpers
  function disp(d, j, v) {
    const f = d.features[j];
    if (d.categorical[f]) return d.categorical[f][Math.round(v)] ?? String(v);
    if (d.value_labels[f] && d.value_labels[f][String(Math.round(v))] != null) return d.value_labels[f][String(Math.round(v))];
    if (d.display_units[f]) return (v * d.display_units[f][0]).toFixed(1) + " " + d.display_units[f][1];
    if (d.integer_idx.indexOf(j) >= 0) return Math.round(v).toLocaleString("en-US");
    return Number(v).toLocaleString("en-US", { maximumFractionDigits: 2 });
  }
  const lab = (d, f) => d.labels[f] || f;
  const recX = (d, evalIdx) => d.records.X[d.recIndex.get(evalIdx)].slice();
  const verdictOf = (d, T) => (T < S.thr[d.key] ? "Review" : "Accept");
  function pools(d) {
    const rev = S.audit.filter((a) => a.domain === d.key && a.source !== "seed");
    return {
      tc: d.pools.tune_clean.concat(rev.filter((a) => a.status === "approved").map((a) => a.T)),
      ta: d.pools.tune_attack.concat(rev.filter((a) => a.status === "confirmed_attack").map((a) => a.T)),
      hc: d.pools.held_clean, ha: d.pools.held_attack,
    };
  }
  const heldRates = (d, t) => { const p = pools(d); return SEAF.rates(t, p.hc, p.ha); };
  function recalibrate(d, reason, budget = 0.10) {
    const old = S.thr[d.key], p = pools(d);
    const t = SEAF.selectThreshold(p.tc, p.ta, budget, old);
    S.thr[d.key] = t;
    const r = SEAF.rates(t, p.hc, p.ha);
    S.thrHist[d.key].push({ ts: now(), t, reason, det: r.det, fpr: r.fpr });
    return { old, t, r };
  }

  // ------------------------------------------------------------------ charts
  function gauge(el, p, names) {
    const c = col(), clr = p >= 0.5 ? c.attack : c.safe;
    plot(el, [{ type: "indicator", mode: "gauge+number", value: p * 100, number: { suffix: "%", font: { family: MONO, size: 34, color: c.text } },
      gauge: { axis: { range: [0, 100], tickvals: [0, 50, 100], tickfont: { size: 10, color: c.muted } }, bar: { color: clr, thickness: 0.28 },
        bgcolor: c.surface2, borderwidth: 0, steps: [{ range: [0, 50], color: rgba(c.safe, 0.1) }, { range: [50, 100], color: rgba(c.attack, 0.1) }],
        threshold: { line: { color: c.text2, width: 2 }, thickness: 0.8, value: 50 } }, domain: { x: [0, 1], y: [0.1, 1] } }],
      { height: 210, margin: { l: 22, r: 22, t: 12, b: 6 }, annotations: [
        { x: 0.02, y: 0, text: names[0], showarrow: false, font: { size: 11, color: c.muted }, xanchor: "left" },
        { x: 0.98, y: 0, text: names[1], showarrow: false, font: { size: 11, color: c.muted }, xanchor: "right" }] });
  }
  function radar(el, r, ref) {
    const c = col(), cats = ["Confidence", "Normal reasoning", "Stable explanation", "Confidence"];
    const v = [r.C, 1 - r.A, 1 - r.S, r.C], rv = [ref.C, 1 - ref.A, 1 - ref.S, ref.C];
    plot(el, [{ type: "scatterpolar", r: rv, theta: cats, name: "Typical clean", line: { color: c.muted, dash: "dot", width: 1.5 } },
      { type: "scatterpolar", r: v, theta: cats, name: "This decision", fill: "toself", line: { color: c.accent, width: 2 }, fillcolor: rgba(c.accent, 0.18) }],
      { height: 250, polar: { bgcolor: "rgba(0,0,0,0)", radialaxis: { range: [0, 1], showticklabels: false, gridcolor: c.grid }, angularaxis: { gridcolor: c.grid, tickfont: { size: 11, color: c.text2 } } },
        margin: { l: 34, r: 34, t: 26, b: 16 }, legend: { y: -0.12, yanchor: "top", x: 0.5, xanchor: "center" } });
  }
  function trustBar(el, r, thr) {
    const c = col();
    const parts = [["Confidence C/3", r.C / 3, c.s[0]], ["Normal reasoning (1−A)/3", (1 - r.A) / 3, c.s[2]], ["Stable explanation (1−S)/3", (1 - r.S) / 3, c.s[3]]];
    plot(el, parts.map(([n, v, clr]) => ({ type: "bar", orientation: "h", y: ["Trust"], x: [v], name: n, text: [v.toFixed(2)], textposition: "inside",
      insidetextanchor: "middle", textfont: { family: MONO, size: 11, color: "#fff" }, marker: { color: clr, line: { color: c.surface, width: 2 } },
      hovertemplate: n + ": %{x:.3f}<extra></extra>" })),
      { barmode: "stack", height: 165, xaxis: { range: [0, 1], dtick: 0.25 }, yaxis: { visible: false }, margin: { l: 8, r: 8, t: 22, b: 6 },
        legend: { y: -0.5, yanchor: "top", x: 0, xanchor: "left" },
        shapes: [{ type: "line", x0: thr, x1: thr, y0: -0.5, y1: 0.5, line: { color: c.review, width: 2, dash: "dash" } }],
        annotations: [{ x: thr, y: 0.64, text: "threshold " + f3(thr), showarrow: false, font: { size: 11, color: c.review } },
          { x: r.T, y: -0.64, text: "T = " + f3(r.T), showarrow: false, xanchor: r.T > 0.8 ? "right" : "left", font: { family: MONO, size: 12, color: c.text } }] });
  }
  function waterfall(el, d, r, top = 9) {
    const c = col(), order = r.phi.map((_, j) => j).sort((a, b) => Math.abs(r.phi[b]) - Math.abs(r.phi[a]));
    const keep = order.slice(0, top), rest = order.slice(top);
    let ys = keep.map((j) => `${lab(d, d.features[j])} = ${disp(d, j, r.x[j])}`).reverse();
    let xs = keep.map((j) => r.phi[j]).reverse();
    if (rest.length) { ys = [`${rest.length} other features`].concat(ys); xs = [rest.reduce((s, j) => s + r.phi[j], 0)].concat(xs); }
    let cum = r.base, hi = Math.max(r.base, r.p), lo = 0;
    xs.forEach((v) => { cum += v; hi = Math.max(hi, cum); lo = Math.min(lo, cum); });
    plot(el, [{ type: "waterfall", orientation: "h", measure: ["absolute"].concat(xs.map(() => "relative"), ["total"]),
      y: ["Base rate"].concat(ys, ["Prediction"]), x: [r.base].concat(xs, [0]),
      text: [r.base.toFixed(3)].concat(xs.map((v) => (v >= 0 ? "+" : "") + v.toFixed(3)), [r.p.toFixed(3)]), textposition: "outside",
      textfont: { family: MONO, size: 11, color: c.text2 }, connector: { line: { color: c.border, width: 1, dash: "dot" } },
      increasing: { marker: { color: c.attack } }, decreasing: { marker: { color: c.safe } }, totals: { marker: { color: c.accent } },
      hovertemplate: "%{y}<br>contribution %{x:+.4f}<extra></extra>" }],
      { height: 90 + 30 * (xs.length + 2), showlegend: false, xaxis: { title: { text: "probability of positive class" }, range: [lo - 0.02, Math.min(hi + 0.14, 1.15)] }, margin: { l: 8, r: 36, t: 10, b: 8 } });
  }
  function devBars(el, d, top, h = 170) {
    const c = col(), t = top.slice().reverse();
    const xs = t.map((v) => v.sigma), m = Math.max(3.5, Math.max(...xs.map(Math.abs)) * 1.3);
    plot(el, [{ type: "bar", orientation: "h", x: xs, y: t.map((v) => lab(d, v.feature)), text: xs.map((v) => (v >= 0 ? "+" : "") + v.toFixed(1) + "σ"),
      textposition: "outside", textfont: { family: MONO, size: 11, color: c.text2 },
      marker: { color: xs.map((v) => (Math.abs(v) >= 3 ? c.attack : Math.abs(v) >= 2 ? c.review : c.muted)) }, hovertemplate: "%{y}: %{x:+.2f}σ<extra></extra>" }],
      { height: h, showlegend: false, xaxis: { range: [-m, m], zeroline: true }, margin: { l: 4, r: 4, t: 4, b: 4 } });
  }
  function shapCompare(el, d, p0, p1, top = 10) {
    const c = col(), order = p0.map((_, j) => j).sort((a, b) => Math.abs(p0[b]) + Math.abs(p1[b]) - Math.abs(p0[a]) - Math.abs(p1[a])).slice(0, top).reverse();
    const ys = order.map((j) => lab(d, d.features[j]));
    plot(el, [{ type: "bar", orientation: "h", y: ys, x: order.map((j) => p0[j]), name: "Original", marker: { color: c.muted } },
      { type: "bar", orientation: "h", y: ys, x: order.map((j) => p1[j]), name: "Attacked", marker: { color: c.attack } }],
      { barmode: "group", height: 110 + 32 * ys.length, xaxis: { title: { text: "SHAP contribution" }, zeroline: true } });
  }
  function trustCompare(el, a, b, thr) {
    const c = col(), cats = ["C", "1 − A", "1 − S", "Trust T"];
    plot(el, [{ type: "bar", x: cats, y: [a.C, 1 - a.A, 1 - a.S, a.T], name: "Original", marker: { color: c.muted } },
      { type: "bar", x: cats, y: [b.C, 1 - b.A, 1 - b.S, b.T], name: "Attacked", marker: { color: c.attack } }],
      { barmode: "group", height: 240, yaxis: { range: [0, 1.05] },
        shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: thr, y1: thr, line: { color: c.review, width: 1.5, dash: "dash" } }],
        annotations: [{ xref: "paper", x: 1, y: thr, text: "threshold " + f3(thr), showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 11, color: c.review } }] });
  }
  function grouped(el, cats, series, o = {}) {
    const c = col(), names = Object.keys(series), nOurs = names.filter((n) => !/paper/i.test(n)).length;
    const traces = names.map((n, i) => {
      const ref = /paper/i.test(n), slot = ref ? i - nOurs : i;
      const t = { type: "bar", x: cats, y: series[n], name: n, opacity: ref ? 0.55 : 1, text: series[n].map((v) => (o.fmt ? o.fmt(v) : v.toFixed(3))),
        textposition: "outside", textfont: { family: MONO, size: 10.5, color: c.text2 },
        marker: { color: c.s[slot % 4], pattern: ref ? { shape: "/", fgcolor: c.surface, size: 6 } : undefined, line: { color: c.surface, width: 2 } } };
      if (o.errors && o.errors[n]) t.error_y = { type: "data", array: o.errors[n].map((e, k) => e[1] - series[n][k]), arrayminus: o.errors[n].map((e, k) => series[n][k] - e[0]), color: c.text2, thickness: 1.2, width: 4 };
      return t;
    });
    const shapes = o.ref == null ? [] : [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: o.ref, y1: o.ref, line: { color: c.muted, width: 1, dash: "dot" } }];
    plot(el, traces, { barmode: "group", height: o.h || 310, yaxis: { range: o.range || [0, 1.02], title: { text: o.ytitle || "ROC-AUC" }, tickformat: o.tickformat }, shapes });
  }

  // ------------------------------------------------------------------ shell
  const PAGES = [["home", "Home"], ["predict", "Predict"], ["attack", "Attack Simulator"], ["monitor", "Live Monitor"],
    ["review", "Review Queue"], ["analytics", "Analytics"], ["audit", "Audit Log"], ["about", "About & Admin"]];
  function shell() {
    $("#nav").innerHTML = PAGES.map(([k, t]) => `<button data-p="${k}" ${S.page === k ? 'aria-current="page"' : ""}>${t}</button>`).join("");
    $$("#nav button").forEach((b) => (b.onclick = () => go(b.dataset.p)));
    const sel = $("#domain");
    sel.innerHTML = Object.keys(INDEX).map((k) => `<option value="${k}" ${k === S.domain ? "selected" : ""}>${esc(INDEX[k].title)} · ${esc(INDEX[k].subtitle)}</option>`).join("");
    sel.onchange = () => setDomain(sel.value);
    const d = S.dom[S.domain];
    const pend = S.audit.filter((a) => a.domain === S.domain && a.status === "pending").length;
    $("#pill").innerHTML = d ? `threshold <span class="mono">${f3(S.thr[S.domain])}</span> · ${pend} pending` : "loading…";
  }
  function go(p) {
    if (S.live && S.live.running && p !== "monitor") { S.live.running = false; toast("Live stream paused"); }
    S.page = p; render(); window.scrollTo(0, 0);
  }
  async function setDomain(k) {
    if (S.live) { S.live.running = false; S.live = null; }
    S.domain = k; S.demo = S.demo ? { step: 0 } : null; S.predict = {}; S.attack = {};
    render();
  }
  async function render() {
    shell();
    const main = $("#main");
    if (!S.dom[S.domain]) {
      main.innerHTML = `<div class="card"><div class="muted">Loading the ${esc(INDEX[S.domain].title)} model (Random Forest ${INDEX[S.domain].n_nodes.toLocaleString()} nodes + Isolation Forest) into your browser…</div><div class="skeleton"></div><div class="skeleton"></div></div>`;
      try { await ensureDomain(S.domain); } catch (e) { main.innerHTML = `<div class="card attack"><h4>Couldn't load the model</h4><div class="muted">${esc(e.message)}. Reload the page to try again.</div></div>`; return; }
      shell();
    }
    const d = S.dom[S.domain];
    try { await VIEWS[S.page](main, d); } catch (e) {
      console.error(e);
      main.innerHTML = `<div class="card attack"><h4>Something went wrong on this page</h4><div class="muted">${esc(e.message)}. Try another page or reload.</div></div>`;
    }
  }

  // ------------------------------------------------------------------ HOME
  const STAGES = [["01", "Input", "Loan, patient or employee record"], ["02", "Model", "Random Forest makes the decision"], ["03", "SHAP", "Exact per-feature receipt of <i>why</i>"],
    ["04", "Validation", "Isolation Forest vs normal reasoning"], ["05", "Trust", "Confidence · anomaly · stability"]];
  async function home(main, d) {
    const pipe = STAGES.map(([n, h, t]) => `<div class="stage"><div class="n">${n}</div><div class="h">${h}</div><div class="d">${t}</div></div><div class="link"></div>`).join("") +
      `<div class="stage"><div class="n">06</div><div class="h">Decision</div><div class="row" style="margin-top:6px;gap:6px">${badge("Accept", "safe")}${badge("Review", "review")}</div></div>`;
    const ex = S.results && S.results.experiments, cd = ex && ex.cross_domain;
    const pooled = cd && cd.feedback_pooled.heldout;
    main.innerHTML = `
      <section class="hero"><span class="chip">Explanation-space anomaly detection</span>
        <h1>Don't just explain AI decisions. <span class="grad">Verify the explanation.</span></h1>
        <p>SEAF treats every SHAP explanation as a security signal: if a decision's stated reasons don't look like the model's normal reasoning, it goes to a human instead of being silently accepted.</p>
        <div class="pipeline">${pipe}</div></section>
      <div class="row" style="margin:16px 0 4px">
        <button class="btn primary" id="startDemo">▶ Start guided demo</button>
        <button class="btn" data-go="monitor">Open live monitor</button><button class="btn" data-go="attack">Attack simulator</button></div>
      <div id="demo"></div>
      ${section("Choose a domain")}
      <div class="grid g3">${Object.keys(INDEX).map((k) => { const m = INDEX[k]; return `<button class="dcard ${k === S.domain ? "active" : ""}" data-dom="${k}">
        <div class="t">${esc(m.title)}</div><div class="muted">${esc(m.subtitle)} · ${m.n_rows.toLocaleString()} records</div>
        <div class="stats"><div><b>${m.accuracy.toFixed(3)}</b>accuracy</div><div><b>${m.roc_auc.toFixed(3)}</b>ROC-AUC</div><div><b>${m.n_features}</b>features</div></div></button>`; }).join("")}</div>
      ${section("Key results", "auc")}
      ${cd ? kpis([
        kpi("Prediction-preserving attacks that succeeded", `${cd.attack_success}/${cd.attack_attempted}`, "decision unchanged, reasons displaced", "attack", "", "pp"),
        kpi("Healthcare trust-score AUC", ex.domains.healthcare.signal_auc.trust_equal.toFixed(3), "equal-weight C · A · S", "accent", "", "auc"),
        kpi("Stability AUC spread", `${cd.stability.fixed_range.toFixed(2)} → ${cd.stability.absolute_range.toFixed(2)}`, "fixed → |deviation|", "drift"),
        kpi("Held-out detection after recalibration", pct(pooled.after.detection), `from ${pct(pooled.before.detection)} (pooled)`, "safe", "up", "det"),
        kpi("Decisions in this session's audit log", S.audit.length.toLocaleString(), `${S.audit.filter((a) => a.status === "pending").length} awaiting review`, "neutral")]) : ""}
      <details class="card" style="margin-top:12px"><summary style="cursor:pointer;font-weight:600">How to read SEAF (plain language)</summary>
        <ul class="muted" style="margin:10px 0 0;padding-left:18px;font-size:13.5px;line-height:1.7">
          <li><b>SHAP</b> splits every prediction into per-feature contributions that add up <i>exactly</i> to the model's output: a receipt for the decision.</li>
          <li><b>Normal reasoning baseline</b>: SHAP receipts of clean historical decisions. An Isolation Forest learns what they usually look like.</li>
          <li><b>Trust score T</b> averages the model's confidence, how normal the receipt looks, and how stable it is when inputs are nudged by ±2%.</li>
          <li><b>Low trust → human review.</b> The reviewer's verdict tunes the alert threshold, never the baseline.</li>
          <li><b>σ deviation</b>: how many standard deviations a feature's contribution is from its usual value.</li></ul></details>`;
    $("#startDemo").onclick = () => { S.demo = { step: 0 }; renderDemo(d); $("#demo").scrollIntoView({ behavior: "smooth", block: "start" }); };
    $$("[data-go]", main).forEach((b) => (b.onclick = () => go(b.dataset.go)));
    $$("[data-dom]", main).forEach((b) => (b.onclick = () => setDomain(b.dataset.dom)));
    if (S.demo) renderDemo(d);
  }
  const STEPS = ["Normal decision", "Attack", "SEAF flags it", "Reviewer confirms", "Threshold updates"];
  async function renderDemo(d) {
    const box = $("#demo"); if (!box) return;
    const D = S.demo;
    box.innerHTML = section("Guided demo · " + d.title) + `<div class="stepper">${STEPS.map((s, i) => `<div class="step ${i < D.step ? "done" : i === D.step ? "now" : ""}"><b>STEP ${i + 1}</b>${s}</div>`).join("")}</div><div class="card" id="demoBody"><div class="skeleton"></div><div class="skeleton"></div></div><div class="row" style="margin-top:10px;justify-content:space-between" id="demoNav"></div>`;
    const ai = d.attacks.eval_idx.indexOf(d.demo_idx);
    if (!D.clean) {
      if (ai < 0) { $("#demoBody").innerHTML = callout("No demo pair for this domain. Try another domain.", true); return; }
      const [c, a] = await call("score", { key: d.key, X: [recX(d, d.demo_idx), d.attacks.X[ai]] });
      Object.assign(D, { clean: c, attack: a, thr: S.thr[d.key] });
    }
    const c = D.clean, a = D.attack, names = d.class_names, thr = D.thr, body = $("#demoBody");
    if (D.step === 0) {
      body.innerHTML = `<h3 style="margin-bottom:12px">A legitimate ${esc(d.subtitle.toLowerCase())} request arrives</h3>
        <div class="split"><div><div id="dg" class="chart"></div>${verdictPanel(verdictOf(d, c.T), c.T, thr)}</div><div id="dw" class="chart"></div></div>
        <div style="margin-top:12px">${callout(`The model predicts <b>${esc(names[c.pred])}</b> (${esc(d.verb)} ${f3(c.p)}). The SHAP receipt adds up exactly: base rate ${f3(c.base)} + contributions = ${f3(c.p)}. Trust <b>${f3(c.T)}</b> is above the threshold, so the decision is accepted automatically.`)}</div>`;
      gauge($("#dg"), c.p, names); waterfall($("#dw"), d, c, 7);
    } else if (D.step === 1) {
      const changed = c.x.filter((v, j) => Math.abs(v - a.x[j]) > 1e-9).length;
      body.innerHTML = `<h3 style="margin-bottom:12px">An adversary edits the inputs, keeping the decision but changing its reasons</h3>
        ${kpis([kpi("Prediction", names[a.pred], "unchanged", "safe"), kpi("Probability change", (a.p - c.p >= 0 ? "+" : "") + (a.p - c.p).toFixed(3), "within the ±0.06 budget", "neutral"), kpi("Features edited", String(changed), "each within ±35%", "review")])}
        <div id="dc" class="chart"></div>${callout(`An output-only audit sees the <b>same decision</b>. But the explanation, the recorded justification, has been rearranged. This is a <b>prediction-preserving attack</b>. ${tip("pp")}`)}`;
      shapCompare($("#dc"), d, c.phi, a.phi, 8);
    } else if (D.step === 2) {
      if (!D.logged) D.logged = logRec(d, a, "demo", "demo-attack", "attack").id;
      body.innerHTML = `<h3 style="margin-bottom:12px">SEAF compares the new explanation with normal model reasoning</h3>
        <div class="grid g2"><div>${verdictPanel(verdictOf(d, a.T), a.T, thr)}<div id="dt" class="chart"></div></div><div><div class="muted">Top deviating features (σ from normal) ${tip("sigma")}</div><div id="dd" class="chart"></div></div></div>
        ${callout(`Trust fell from <b>${f3(c.T)}</b> to <b>${f3(a.T)}</b>. The decision is in the review queue (audit #${D.logged}) with the per-feature evidence above.`)}`;
      trustCompare($("#dt"), c, a, thr); devBars($("#dd"), d, a.top, 230);
    } else if (D.step === 3) {
      body.innerHTML = `<h3 style="margin-bottom:8px">A human reviewer examines the audit record</h3><div class="muted">Audit record #${D.logged} · ${esc(d.title)} · trust <span class="mono">${f3(a.T)}</span></div>
        <div id="dd" class="chart"></div>${D.confirmed ? callout("Confirmed. Continue to see the effect on the detection threshold.") : `<button class="btn primary" id="confirmDemo">Confirm attack</button>`}`;
      devBars($("#dd"), d, a.top, 210);
      if (!D.confirmed) $("#confirmDemo").onclick = () => {
        const rec = S.audit.find((x) => x.id === D.logged); rec.status = "confirmed_attack"; rec.note = "guided demo";
        const before = heldRates(d, S.thr[d.key]);
        const r = recalibrate(d, "guided demo: confirmed attack");
        Object.assign(D, { confirmed: true, cal: r, before, step: 4 });
        toast(`Attack confirmed · threshold ${f3(r.old)} → ${f3(r.t)}`, "safe");
        shell(); renderDemo(d);
      };
    } else {
      const r = D.cal;
      if (!r) { body.innerHTML = callout("Confirm the attack in step 4 first.", true); }
      else {
        const dt = r.t - r.old;
        body.innerHTML = `<h3 style="margin-bottom:12px">The confirmed outcome recalibrates the threshold</h3>
          ${kpis([kpi("Threshold", f3(r.t), `${dt >= 0 ? "+" : ""}${dt.toFixed(3)} vs ${f3(r.old)}`, "accent", dt > 0 ? "up" : dt < 0 ? "down" : "", "threshold"),
            kpi("Held-out detection", pct(r.r.det), `${(100 * (r.r.det - D.before.det)).toFixed(1)} pp`, "safe", "up", "det"),
            kpi("Held-out false positives", pct(r.r.fpr), `${(100 * (r.r.fpr - D.before.fpr)).toFixed(1)} pp`, "review", "warn", "fpr")])}
          ${callout("The threshold is re-chosen to catch the most attacks within a 10% false-alarm budget, and the result is measured on <b>held-out</b> records never used to choose it. The baseline of normal reasoning is <b>never</b> changed automatically.")}`;
      }
    }
    $("#demoNav").innerHTML = `<div class="row"><button class="btn" id="dBack" ${D.step === 0 ? "disabled" : ""}>← Back</button>${D.step < 4 ? `<button class="btn primary" id="dNext" ${D.step === 3 && !D.confirmed ? "disabled" : ""}>Next →</button>` : ""}</div><button class="btn" id="dClose">Close demo</button>`;
    $("#dBack").onclick = () => { D.step--; renderDemo(d); };
    if ($("#dNext")) $("#dNext").onclick = () => { D.step++; renderDemo(d); };
    $("#dClose").onclick = () => { S.demo = null; box.innerHTML = ""; };
  }

  // ------------------------------------------------------------------ PREDICT
  function formHTML(d, x, key) {
    return `<div class="form-grid">${d.features.map((f, j) => {
      const L = esc(lab(d, f)), id = `${key}_${j}`;
      if (d.categorical[f]) return `<label class="f" for="${id}">${L}<select class="inp" id="${id}" data-j="${j}">${d.categorical[f].map((c, i) => `<option value="${i}" ${Math.round(x[j]) === i ? "selected" : ""}>${esc(c)}</option>`).join("")}</select></label>`;
      if (d.choices[f] && d.numeric_idx.indexOf(j) < 0) {
        let opts = d.choices[f].slice(); if (opts.indexOf(x[j]) < 0) opts = opts.concat([x[j]]).sort((a, b) => a - b);
        const vl = d.value_labels[f] || {};
        return `<label class="f" for="${id}">${L}<select class="inp" id="${id}" data-j="${j}">${opts.map((o) => `<option value="${o}" ${o === x[j] ? "selected" : ""}>${esc(vl[String(Math.round(o))] || o)}</option>`).join("")}</select></label>`;
      }
      if (d.display_units[f]) { const [fac, unit] = d.display_units[f]; return `<label class="f" for="${id}">${esc(L.split(" (")[0])} (${esc(unit)})<input class="inp mono" type="number" step="0.1" id="${id}" data-j="${j}" data-fac="${fac}" value="${(x[j] * fac).toFixed(1)}"></label>`; }
      const r = d.ranges[f];
      return `<label class="f" for="${id}" title="Observed range ${r[0].toLocaleString()} – ${r[1].toLocaleString()}">${L}<input class="inp mono" type="number" id="${id}" data-j="${j}" value="${x[j]}" step="any"></label>`;
    }).join("")}</div>`;
  }
  function readForm(d, root) {
    const x = new Array(d.features.length).fill(0);
    $$("[data-j]", root).forEach((el) => { const j = +el.dataset.j; let v = parseFloat(el.value); if (el.dataset.fac) v = Math.round(v / parseFloat(el.dataset.fac)); x[j] = isNaN(v) ? 0 : v; });
    return x;
  }
  async function predict(main, d) {
    const P = S.predict;
    if (!P.x) { P.x = recX(d, d.records.idx[0]); P.src = "eval-" + d.records.idx[0]; }
    main.innerHTML = head(`${d.title} · ${d.subtitle}`, "Predict & verify",
      `Score one record. SEAF shows the model's decision, its SHAP explanation, and whether that explanation looks like the model's normal reasoning ${tip("trust")}.`) +
      `<div class="split"><div class="card"><h3 style="font-size:16px">Input record</h3>
        <div class="row" style="margin:12px 0 6px"><button class="btn sm" id="pClean">Random clean</button><button class="btn sm" id="pAtk">Random attacked</button><button class="btn sm" id="pReset">Reset</button></div>
        <div class="muted" style="margin-bottom:10px">Loaded: <span class="mono">${esc(P.src)}</span> ${P.src.startsWith("adv") ? badge("attacked sample", "attack", false) : ""}</div>
        <form id="pForm">${formHTML(d, P.x, "pf")}<button class="btn primary block" style="margin-top:14px" type="submit">⚡ Score with SEAF</button></form></div>
        <div id="pOut"></div></div>`;
    const load = (x, src) => { P.x = x; P.src = src; P.res = null; predict(main, d); };
    $("#pClean").onclick = () => { const i = d.records.idx[Math.floor(rand() * d.records.idx.length)]; load(recX(d, i), "eval-" + i); };
    $("#pAtk").onclick = () => { const k = Math.floor(rand() * d.attacks.X.length); load(d.attacks.X[k].slice(), "adv-" + d.attacks.eval_idx[k]); };
    $("#pReset").onclick = () => load(recX(d, d.records.idx[0]), "eval-" + d.records.idx[0]);
    $("#pForm").onsubmit = async (e) => {
      e.preventDefault();
      P.x = readForm(d, $("#pForm"));
      $("#pOut").innerHTML = `<div class="card"><div class="muted">Running TreeSHAP (1 + ${d.seaf.k} jittered copies) and the Isolation Forest in your browser…</div><div class="skeleton"></div><div class="skeleton" style="height:180px"></div></div>`;
      const [r] = await call("score", { key: d.key, X: [P.x] });
      const e2 = logRec(d, r, "predict", P.src, P.src.startsWith("adv") ? "attack" : null);
      P.res = { r, id: e2.id, thr: e2.threshold };
      if (e2.verdict === "Review") toast(`Decision #${e2.id} routed to review (T=${f3(r.T)})`, "attack");
      shell(); showPredict(d);
    };
    if (P.res) showPredict(d);
    else $("#pOut").innerHTML = `<div class="empty"><h4>No decision yet</h4><div>Edit the record, or load a random one, and press <b>Score with SEAF</b>. You'll see the prediction, its SHAP explanation and a trust verdict.</div></div>`;
  }
  function showPredict(d) {
    const { r, id, thr } = S.predict.res, names = d.class_names, v = r.T < thr ? "Review" : "Accept";
    const sum = r.phi.reduce((s, x) => s + x, 0);
    const b = d.seaf.baseline_median;
    $("#pOut").innerHTML = verdictPanel(v, r.T, thr) + `<div style="height:12px"></div>` +
      kpis([kpi("Prediction", names[r.pred], "audit #" + id, "accent"), kpi(d.verb[0].toUpperCase() + d.verb.slice(1), f3(r.p), null, "neutral"), kpi("Trust T", f3(r.T), "threshold " + f3(thr), v === "Accept" ? "safe" : "review", "", "trust")]) +
      `<div class="grid g2"><div>${section("Probability")}<div id="pg" class="chart"></div></div><div>${section("Trust signals", "trust")}<div id="pr" class="chart"></div></div></div>
      ${section("Trust composition")}<div class="muted">C = ${f3(r.C)} ${tip("C")} &nbsp; A = ${f3(r.A)} ${tip("A")} &nbsp; S = ${f3(r.S)} ${tip("S")}</div><div id="pt" class="chart"></div>
      ${section("Why? SHAP explanation", "SHAP")}<div id="pw" class="chart"></div>
      ${callout(`Summation identity: base rate <span class="mono">${r.base.toFixed(4)}</span> + Σφ <span class="mono">${(sum >= 0 ? "+" : "") + sum.toFixed(4)}</span> = <span class="mono">${r.p.toFixed(4)}</span> (error ${Math.abs(r.base + sum - r.p).toExponential(1)}). Red bars push towards <b>${esc(names[1])}</b>, green towards <b>${esc(names[0])}</b>.`)}
      ${section("Audit evidence: most deviating features", "sigma")}<div id="pd" class="chart"></div>
      ${v === "Review" ? `<button class="btn" data-go="review">Open review queue →</button>` : ""}`;
    gauge($("#pg"), r.p, names); radar($("#pr"), r, b); trustBar($("#pt"), r, thr); waterfall($("#pw"), d, r); devBars($("#pd"), d, r.top, 210);
    $$("[data-go]", $("#pOut")).forEach((el) => (el.onclick = () => go(el.dataset.go)));
  }

  // ------------------------------------------------------------------ ATTACK
  async function attack(main, d) {
    const A = S.attack;
    A.idx = A.idx == null ? d.records.idx[0] : A.idx; A.kind = A.kind || "pp";
    A.budget = A.budget || 0.35; A.n = A.n || 40; A.tau = A.tau || 0.06;
    const nums = d.numeric_idx.map((j) => lab(d, d.features[j]));
    main.innerHTML = head(`${d.title} · red team`, "Attack simulator",
      `Act as the adversary. A <b>prediction-preserving</b> attack ${tip("pp")} keeps the decision but rewrites its reasons; an <b>evasion</b> attack ${tip("evasion")} flips the decision with the smallest nudge. Then see whether SEAF notices.`) +
      `<div class="card"><div class="grid g3">
        <div><h4 style="margin-bottom:10px">1 · Pick a record</h4><div class="row"><select class="inp" id="aRec" style="flex:1">${d.records.idx.map((i, k) => `<option value="${i}" ${i === A.idx ? "selected" : ""}>Record #${i} · actual: ${esc(d.class_names[d.records.y[k]])}</option>`).join("")}</select><button class="btn sm" id="aRand">Random</button></div></div>
        <div><h4 style="margin-bottom:10px">2 · Choose the attack</h4><div class="seg" id="aKind"><button data-k="pp" aria-pressed="${A.kind === "pp"}">Prediction-preserving</button><button data-k="ev" aria-pressed="${A.kind === "ev"}">Evasion</button></div>
          <div class="muted" style="margin-top:8px">Perturbs ${nums.length} continuous features: ${esc(nums.slice(0, 6).join(", "))}${nums.length > 6 ? "…" : ""}</div></div>
        <div><h4 style="margin-bottom:10px">3 · Budget</h4>${A.kind === "pp" ? `
          <label class="f">Max change per feature: <span class="mono" id="vb">±${A.budget.toFixed(2)}</span><input type="range" id="aB" min="0.05" max="0.5" step="0.05" value="${A.budget}"></label>
          <label class="f">Candidates: <span class="mono" id="vn">${A.n}</span><input type="range" id="aN" min="10" max="120" step="10" value="${A.n}"></label>
          <label class="f">Max probability change |Δp|: <span class="mono" id="vt">${A.tau.toFixed(2)}</span><input type="range" id="aT" min="0.01" max="0.15" step="0.01" value="${A.tau}"></label>` :
          `<div class="muted">Searches radii 2% → 75% with 300 candidates each and keeps the smallest perturbation that flips the decision.</div>`}</div></div>
        <div class="row" style="margin-top:14px"><button class="btn primary" id="aGo">🚀 Launch attack</button><div id="aProg" style="flex:1;min-width:200px"></div></div></div>
      <div id="aOut" style="margin-top:18px"></div>`;
    $("#aRec").onchange = (e) => { A.idx = +e.target.value; };
    $("#aRand").onclick = () => { A.idx = d.records.idx[Math.floor(rand() * d.records.idx.length)]; $("#aRec").value = A.idx; };
    $$("#aKind button").forEach((b) => (b.onclick = () => { A.kind = b.dataset.k; attack(main, d); }));
    if ($("#aB")) {
      $("#aB").oninput = (e) => { A.budget = +e.target.value; $("#vb").textContent = "±" + A.budget.toFixed(2); };
      $("#aN").oninput = (e) => { A.n = +e.target.value; $("#vn").textContent = A.n; };
      $("#aT").oninput = (e) => { A.tau = +e.target.value; $("#vt").textContent = A.tau.toFixed(2); };
    }
    $("#aGo").onclick = async () => {
      const prog = $("#aProg"), stepsTxt = A.kind === "pp"
        ? [`Generating ${A.n} candidates at ±${Math.round(A.budget * 100)}%`, "Screening with predict_proba (keep class, |Δp| ≤ " + A.tau.toFixed(2) + ")", "Explaining survivors with TreeSHAP", "Scoring original and attacked record with SEAF"]
        : ["Searching radii 2% → 75%", "Screening candidates for a flipped decision", "Scoring original and attacked record with SEAF"];
      let k = 0;
      const tick = setInterval(() => { k = Math.min(k + 1, stepsTxt.length - 1); prog.innerHTML = `<div class="muted">${esc(stepsTxt[k])}…</div><div class="progress"><div style="width:${(100 * (k + 1)) / (stepsTxt.length + 1)}%"></div></div>`; }, 350);
      prog.innerHTML = `<div class="muted">${esc(stepsTxt[0])}…</div><div class="progress"><div style="width:10%"></div></div>`;
      $("#aGo").disabled = true;
      const t0 = performance.now();
      const res = await call(A.kind, { key: d.key, x: recX(d, A.idx), o: A.kind === "pp" ? { n: A.n, budget: A.budget, tau: A.tau, seed: Math.floor(rand() * 1e9) } : { seed: Math.floor(rand() * 1e9) } });
      clearInterval(tick);
      $("#aGo").disabled = false;
      prog.innerHTML = `<div class="muted">${res.at.success ? "✓ Attack succeeded" : "✗ No candidate met the constraints"} in ${((performance.now() - t0) / 1000).toFixed(2)} s</div>`;
      A.out = { at: res.at, r0: res.s[0], r1: res.s[1], thr: S.thr[d.key], logged: null, idx: A.idx };
      if (res.at.success) toast(res.s[1].T < S.thr[d.key] ? "SEAF flagged the attacked decision" : "The attack slipped past the current threshold", res.s[1].T < S.thr[d.key] ? "safe" : "attack");
      showAttack(d);
    };
    if (A.out) showAttack(d);
    else $("#aOut").innerHTML = `<div class="empty"><h4>No attack run yet</h4><div>Pick a record and launch an attack. You'll see the original and attacked decisions side by side.</div></div>`;
  }
  function showAttack(d) {
    const { at, r0, r1, thr } = S.attack.out, names = d.class_names, out = $("#aOut");
    if (!at.success) { out.innerHTML = callout("No candidate satisfied the constraints for this record. Try a larger budget, more candidates, or another record.", true); return; }
    const v0 = r0.T < thr ? "Review" : "Accept", v1 = r1.T < thr ? "Review" : "Accept";
    const rows = r0.phi.map((_, j) => j).sort((a, b) => Math.abs(r1.phi[b] - r0.phi[b]) - Math.abs(r1.phi[a] - r0.phi[a])).map((j) => {
      const a = at.x_orig[j], b = at.x_adv[j], ch = Math.abs(a - b) > 1e-9, rel = ch && Math.abs(a) > 1e-9 ? ((b - a) / Math.abs(a)) * 100 : 0;
      return `<tr class="${ch ? "changed" : ""}"><td>${esc(lab(d, d.features[j]))}</td><td class="num">${esc(disp(d, j, a))}</td><td class="num">${esc(disp(d, j, b))}</td><td class="num">${ch ? `<span class="${rel > 0 ? "up" : "dn"}">${rel >= 0 ? "+" : ""}${rel.toFixed(1)}%</span>` : "—"}</td><td class="num">${r0.phi[j].toFixed(4)}</td><td class="num">${r1.phi[j].toFixed(4)}</td><td class="num">${(r1.phi[j] - r0.phi[j] >= 0 ? "+" : "") + (r1.phi[j] - r0.phi[j]).toFixed(4)}</td></tr>`;
    }).join("");
    const caught = v0 === "Accept" && v1 === "Review";
    out.innerHTML = section("Outcome") + kpis([
      kpi("Decision", names[r1.pred], r0.pred === r1.pred ? "unchanged" : "flipped from " + names[r0.pred], r0.pred === r1.pred ? "safe" : "attack"),
      kpi("Probability change", (r1.p - r0.p >= 0 ? "+" : "") + (r1.p - r0.p).toFixed(3), `${f3(r0.p)} → ${f3(r1.p)}`, "neutral"),
      kpi("Trust change", (r1.T - r0.T >= 0 ? "+" : "") + (r1.T - r0.T).toFixed(3), `${f3(r0.T)} → ${f3(r1.T)}`, r1.T < r0.T ? "attack" : "safe", r1.T < r0.T ? "down" : "up", "trust"),
      kpi("Attribution displacement", at.displacement ? at.displacement.toFixed(2) + "σ" : "—", `mean input change ${(100 * at.perturbation).toFixed(1)}%`, "review", "", "sigma")]) +
      `<div class="grid g2">${[["Original", r0, v0, ""], ["Attacked", r1, v1, "attack"]].map(([t, r, v, cls], i) => `<div class="card ${cls}"><div class="row" style="justify-content:space-between"><h4>${t}</h4>${vbadge(v)}</div>
        <div class="muted" style="margin-top:6px">Prediction <b>${esc(names[r.pred])}</b> · p = <span class="mono">${f3(r.p)}</span> · T = <span class="mono">${f3(r.T)}</span> · C ${r.C.toFixed(2)} · A ${r.A.toFixed(2)} · S ${r.S.toFixed(2)}</div><div id="ad${i}" class="chart"></div></div>`).join("")}</div>
      <div style="margin-top:12px">${caught ? callout(`<b>Caught.</b> The decision stayed <b>${esc(names[r1.pred])}</b>, but SEAF saw that the explanation no longer looks like normal reasoning and routed it to review (trust ${f3(r1.T)} < ${f3(thr)}).`)
        : v1 === "Accept" ? callout(`<b>Missed at the current threshold</b> (${f3(thr)}). Detection is probabilistic: the paper reports AUCs of 0.6-0.87, which supports triage rather than automatic rejection. Reviewer feedback can raise the threshold.`, true)
        : callout("Both decisions fall below the threshold: the original was already low-trust.", true)}</div>
      <div class="split-r"><div>${section("SHAP before vs after", "SHAP")}<div id="acmp" class="chart"></div></div><div>${section("Trust signals before vs after", "trust")}<div id="atr" class="chart"></div>${at.trace && at.trace.length ? section("Random search progress") + '<div id="atc" class="chart"></div>' : ""}</div></div>
      ${section("What changed")}<div class="tablewrap"><table class="t"><thead><tr><th>Feature</th><th>Original</th><th>Attacked</th><th>Change</th><th>SHAP before</th><th>SHAP after</th><th>Δ SHAP</th></tr></thead><tbody>${rows}</tbody></table></div>
      <div style="margin-top:14px">${S.attack.out.logged ? badge("Logged as audit #" + S.attack.out.logged, "accent") : `<button class="btn" id="aLog">Send attacked decision to the audit log / review queue</button>`}</div>`;
    devBars($("#ad0"), d, r0.top, 185); devBars($("#ad1"), d, r1.top, 185); shapCompare($("#acmp"), d, r0.phi, r1.phi); trustCompare($("#atr"), r0, r1, thr);
    if ($("#atc")) { const c = col(); plot($("#atc"), [{ type: "scatter", mode: "lines", y: at.trace, x: at.trace.map((_, i) => i + 1), line: { color: c.attack, width: 2, shape: "hv" }, fill: "tozeroy", fillcolor: rgba(c.attack, 0.1), hovertemplate: "candidate %{x}<br>best %{y:.2f}σ<extra></extra>" }], { height: 180, showlegend: false, xaxis: { title: { text: "surviving candidate #" } }, yaxis: { title: { text: "best displacement (σ)" } } }); }
    if ($("#aLog")) $("#aLog").onclick = () => { const e = logRec(d, r1, "attack", "sim-" + S.attack.out.idx, "attack"); S.attack.out.logged = e.id; toast(`Logged as audit #${e.id}${e.verdict === "Review" ? " · pending review" : ""}`); shell(); showAttack(d); };
  }

  // ------------------------------------------------------------------ MONITOR
  const SPEEDS = { Slow: [2000, 1], Normal: [1250, 1], Fast: [900, 2] };
  function newLive(d) {
    const numSorted = d.numeric_idx.slice().sort((a, b) => d.importance[b] - d.importance[a]);
    const thr = S.thr[d.key];
    return { running: false, speed: "Normal", rate: 0.08, burst: 0, drift: false, driftIdx: numSorted.slice(0, 2), driftRate: 0.004,
      level: 0, fullStab: true, hist: [], mon: new SEAF.Monitor({ expected: Math.max(heldRates(d, thr).fpr, 0.02) }), thr0: thr, last: "normal", timer: null, busy: false };
  }
  async function monitor(main, d) {
    if (!S.live || S.live.key !== d.key) { S.live = newLive(d); S.live.key = d.key; }
    const L = S.live;
    main.innerHTML = head(`${d.title} · real-time`, "Live monitor",
      `A simulated stream of decisions from the evaluation split, scored live in your browser. Inject attacks or start a gradual drift and watch SEAF separate a sudden attack spike from slow population change ${tip("drift")}.`) +
      `<div class="card"><div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:20px">
        <div><h4 style="margin-bottom:10px">Stream</h4><div class="row"><button class="btn primary" id="mRun">${L.running ? "❚❚ Pause" : "▶ Start"}</button><button class="btn" id="mReset">Reset</button></div>
          <div class="seg" id="mSpeed" style="margin-top:10px">${Object.keys(SPEEDS).map((s) => `<button data-s="${s}" aria-pressed="${L.speed === s}">${s}</button>`).join("")}</div></div>
        <div><h4 style="margin-bottom:10px">Attacks</h4><label class="f">Injection rate: <span class="mono" id="vr">${Math.round(L.rate * 100)}%</span><input type="range" id="mRate" min="0" max="0.5" step="0.01" value="${L.rate}"></label>
          <button class="btn sm" id="mBurst" style="margin-top:8px">⚡ Inject burst ×20</button></div>
        <div><h4 style="margin-bottom:10px">Drift</h4><label class="switch"><input type="checkbox" id="mDrift" ${L.drift ? "checked" : ""}> Start drift</label>
          <div class="row" style="margin-top:8px;gap:6px">${d.numeric_idx.map((j) => `<label class="switch" style="font-size:12.5px"><input type="checkbox" data-dj="${j}" ${L.driftIdx.indexOf(j) >= 0 ? "checked" : ""}>${esc(lab(d, d.features[j]))}</label>`).join("")}</div>
          <label class="f" style="margin-top:8px">Drift speed: <span class="mono" id="vd">${L.driftRate.toFixed(3)}</span>/record<input type="range" id="mDR" min="0.001" max="0.01" step="0.001" value="${L.driftRate}"></label></div>
        <div><h4 style="margin-bottom:10px">Performance</h4><label class="switch"><input type="checkbox" id="mStab" ${L.fullStab ? "checked" : ""}> Stability on every record</label>
          <div class="muted" style="margin-top:6px">Stability costs ${d.seaf.k} extra TreeSHAP evaluations per record. Off = measure every 3rd record and use the baseline median for the rest.</div>
          <button class="btn sm" id="mRecal" style="margin-top:8px">Recalibrate threshold</button></div></div></div>
      <div id="live" style="margin-top:14px"></div>`;
    $("#mRun").onclick = () => { L.running = !L.running; $("#mRun").textContent = L.running ? "❚❚ Pause" : "▶ Start"; if (L.running) loop(d); else drawLive(d); };
    $("#mReset").onclick = () => { L.running = false; S.live = null; monitor(main, d); };
    $$("#mSpeed button").forEach((b) => (b.onclick = () => { L.speed = b.dataset.s; $$("#mSpeed button").forEach((x) => x.setAttribute("aria-pressed", x === b)); }));
    $("#mRate").oninput = (e) => { L.rate = +e.target.value; $("#vr").textContent = Math.round(L.rate * 100) + "%"; };
    $("#mBurst").onclick = () => { L.burst = 20; toast("Injecting a burst of 20 attacks", "attack"); };
    $("#mDrift").onchange = (e) => { L.drift = e.target.checked; };
    $$("[data-dj]").forEach((c) => (c.onchange = () => { L.driftIdx = $$("[data-dj]").filter((x) => x.checked).map((x) => +x.dataset.dj); }));
    $("#mDR").oninput = (e) => { L.driftRate = +e.target.value; $("#vd").textContent = L.driftRate.toFixed(3); };
    $("#mStab").onchange = (e) => { L.fullStab = e.target.checked; };
    $("#mRecal").onclick = () => { const r = recalibrate(d, "recalibrated from Live Monitor"); toast(`Threshold ${f3(r.old)} → ${f3(r.t)}`); shell(); drawLive(d); };
    drawLive(d, true);
    if (L.running) loop(d);
  }
  async function loop(d) {
    const L = S.live;
    if (!L || !L.running || L.busy || S.page !== "monitor" || L.key !== d.key) return;
    L.busy = true;
    try {
      const [interval, per] = SPEEDS[L.speed];
      const ev = [];
      for (let i = 0; i < per; i++) {
        L.level = L.drift ? Math.min(0.45, L.level + L.driftRate) : Math.max(0, L.level - 4 * L.driftRate);
        const atk = L.burst > 0 || rand() < L.rate;
        if (L.burst > 0) L.burst--;
        if (atk && d.attacks.X.length) { const k = Math.floor(rand() * d.attacks.X.length); ev.push({ x: d.attacks.X[k].slice(), ref: "adv-" + d.attacks.eval_idx[k], label: "attack" }); }
        else {
          const k = Math.floor(rand() * d.records.idx.length), x = d.records.X[k].slice();
          if (L.level > 0) L.driftIdx.forEach((j) => (x[j] *= 1 + L.level));
          ev.push({ x, ref: "eval-" + d.records.idx[k], label: L.level > 0.02 ? "drift" : "clean" });
        }
      }
      const n0 = L.hist.length;
      const res = [];
      for (let i = 0; i < ev.length; i++) {
        const stab = L.fullStab || (n0 + i) % 3 === 0;
        res.push((await call("score", { key: d.key, X: [ev[i].x], opts: { stability: stab } }))[0]);
      }
      if (!S.live || S.live !== L) return;
      res.forEach((r, i) => {
        const e = logRec(d, r, "stream", ev[i].ref, ev[i].label);
        const zin = d.numeric_idx.map((j) => (r.x[j] - d.seaf.input_mean[j]) / d.seaf.input_scale[j]);
        const snap = L.mon.update(e.verdict === "Review", zin, r.T);
        L.hist.push({ n: n0 + i + 1, id: e.id, T: r.T, p: r.p, pred: r.pred, flagged: e.verdict === "Review", label: ev[i].label, threshold: e.threshold, ...snap });
        if (ev[i].label === "attack" && e.verdict === "Review") toast(`Attack flagged · audit #${e.id} · T=${f3(r.T)}`, "attack");
        if (snap.state !== L.last) {
          if (snap.state === "spike") toast("Sudden drop in trust: possible attack burst", "attack");
          else if (snap.state === "drift") toast("Gradual drift detected: population shifting", "drift");
          L.last = snap.state;
        }
      });
      if (L.hist.length > 400) L.hist = L.hist.slice(-400);
      shell(); drawLive(d);
      L.busy = false;
      if (L.running) L.timer = setTimeout(() => loop(d), interval);
    } catch (e) { L.busy = false; L.running = false; toast("Stream stopped: " + e.message, "attack"); }
  }
  function drawLive(d, first) {
    const L = S.live, box = $("#live"); if (!box || !L) return;
    const H = L.hist, n = H.length, thr = S.thr[d.key], names = d.class_names;
    const st = n ? H[n - 1].state : "normal";
    const flagged = H.filter((h) => h.flagged).length, atk = H.filter((h) => h.label === "attack"), cl = H.filter((h) => h.label !== "attack");
    const caught = atk.filter((h) => h.flagged).length, fa = cl.filter((h) => h.flagged).length;
    const conf = S.audit.filter((a) => a.domain === d.key && a.source === "stream" && a.status === "confirmed_attack").length;
    const dthr = thr - L.thr0, [interval, per] = SPEEDS[L.speed];
    const sb = { normal: badge("Normal", "safe"), spike: badge("Attack spike", "attack"), drift: badge("Gradual drift", "drift") }[st];
    const feed = H.slice(-12).reverse().map((h) => `<div class="feed-row ${h.flagged && h.label === "attack" ? "hot" : h.flagged ? "flag" : ""}"><span class="id">#${h.id}</span>
      <span>${esc(names[h.pred])} <span class="mono muted">p=${h.p.toFixed(2)}</span> ${h.label === "attack" ? badge("attack", "attack", false) : h.label === "drift" ? badge("drift", "drift", false) : ""}</span>
      <span class="tt" style="color:${h.flagged ? "var(--review)" : "var(--safe)"}">${f3(h.T)}</span>${h.flagged ? badge("Review", "review") : badge("Accept", "safe")}</div>`).join("");
    const html = `<div class="statusbar"><span class="live-dot ${L.running ? "" : "off"}"></span><b>${L.running ? "LIVE" : "PAUSED"}</b><span class="muted">· ${esc(d.title)} · ${per} record(s) every ${(interval / 1000).toFixed(2)} s</span><span style="flex:1"></span>${sb}${L.level > 0.005 ? badge(`drift +${Math.round(100 * L.level)}%`, "drift") : ""}${L.burst > 0 ? badge("burst in progress", "attack") : ""}</div>` +
      kpis([kpi("Processed", n.toLocaleString(), L.running ? `${(per / (interval / 1000)).toFixed(1)} rec/s target` : "paused", "accent"),
        kpi("Flagged for review", String(flagged), `${n ? ((100 * flagged) / n).toFixed(1) : "0.0"}% of stream`, "review"),
        kpi("Injected attacks caught", `${caught}/${atk.length}`, `${atk.length ? Math.round((100 * caught) / atk.length) : 0}% detection`, "attack", "", "det"),
        kpi("False alarms", String(fa), `${cl.length ? ((100 * fa) / cl.length).toFixed(1) : "0.0"}% of clean/drifted`, "neutral", "", "fpr"),
        kpi("Confirmed by reviewers", String(conf), "stream decisions", "safe"),
        kpi("Current threshold", f3(thr), Math.abs(dthr) > 1e-9 ? `${dthr >= 0 ? "+" : ""}${dthr.toFixed(3)} since start` : "unchanged", "review", dthr > 0 ? "up" : dthr < 0 ? "down" : "", "threshold")]) +
      `<div class="split"><div>${section("Decision feed")}${n ? `<div class="feed">${feed}</div><div class="muted" style="margin-top:6px">Attack / drift tags are the simulator's ground truth, shown for the demo only. SEAF never sees them.</div>` : `<div class="empty"><h4>Stream idle</h4><div>Press <b>Start</b> to begin feeding decisions.</div></div>`}</div>
        <div>${section("Trust over time", "trust")}<div id="lt" class="chart"></div>${section("Rolling flag rate: spike vs drift")}<div id="lr" class="chart"></div>
        <div class="muted"><b>Red shading = spike</b>: the last 20 decisions' flag rate jumps, or their trust is significantly lower (rank test) than the preceding window. <b>Violet shading = drift</b>: the mean of standardised inputs has moved consistently away from where the stream started. Random-sign attack edits average out; population drift does not. Bursts show up best in <b>Healthcare</b>.</div></div></div>`;
    box.innerHTML = html;
    const c = col(), T = H.slice(-200);
    const tr = [{ type: "scatter", mode: "lines", x: T.map((h) => h.n), y: T.map((h) => h.threshold), name: "Threshold", line: { color: c.review, width: 1.5, dash: "dash" }, hoverinfo: "skip" },
      { type: "scatter", mode: "lines", x: T.map((h) => h.n), y: T.map((h, i) => { const w = T.slice(Math.max(0, i - 9), i + 1); return w.reduce((s, v) => s + v.T, 0) / w.length; }), name: "Rolling mean", line: { color: c.accent, width: 2 }, hoverinfo: "skip" }];
    [["clean", c.safe, "Clean", "circle"], ["drift", c.drift, "Drifted", "diamond"], ["attack", c.attack, "Injected attack", "x"]].forEach(([l, clr, nm, sym]) => {
      const pts = T.filter((h) => h.label === l);
      if (pts.length) tr.push({ type: "scatter", mode: "markers", x: pts.map((h) => h.n), y: pts.map((h) => h.T), name: nm, marker: { color: clr, size: l === "clean" ? 6 : 8, symbol: sym, opacity: l === "clean" ? 0.6 : 0.95 }, hovertemplate: "#%{x} · T=%{y:.3f}<extra>" + nm + "</extra>" });
    });
    plot($("#lt"), tr, { height: 270, xaxis: { title: { text: "decision #" } }, yaxis: { title: { text: "trust T" }, range: [0, 1] } });
    const shapes = [];
    ["spike", "drift"].forEach((s) => {
      let start = null;
      T.forEach((h, i) => {
        const on = h.state === s;
        if (on && start === null) start = i;
        if ((!on || i === T.length - 1) && start !== null) { const end = on ? i : i - 1; shapes.push({ type: "rect", xref: "x", yref: "paper", x0: T[start].n - 0.5, x1: T[end].n + 0.5, y0: 0, y1: 1, fillcolor: rgba(s === "spike" ? c.attack : c.drift, 0.16), line: { width: 0 }, layer: "below" }); start = null; }
      });
    });
    shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1, y0: L.mon.o.expected, y1: L.mon.o.expected, line: { color: c.muted, width: 1, dash: "dot" } });
    plot($("#lr"), [{ type: "scatter", mode: "lines", x: T.map((h) => h.n), y: T.map((h) => h.shortRate), name: "Recent 20", line: { color: c.accent, width: 2 } },
      { type: "scatter", mode: "lines", x: T.map((h) => h.n), y: T.map((h) => h.longRate), name: "Last 100", line: { color: c.muted, width: 2, dash: "dot" } }],
      { height: 240, shapes, xaxis: { title: { text: "decision #" } }, yaxis: { title: { text: "flag rate" }, range: [0, 1], tickformat: ".0%" } });
  }

  // ------------------------------------------------------------------ REVIEW
  function review(main, d) {
    const thr = S.thr[d.key], hr = heldRates(d, thr), R = S.reviewDelta && S.reviewDelta.key === d.key ? S.reviewDelta : null;
    const mine = S.audit.filter((a) => a.domain === d.key);
    const cnt = (s) => mine.filter((a) => a.status === s).length;
    const dl = (nw, old, p) => { const df = nw - old; return Math.abs(df) < 1e-12 ? ["no change", ""] : [p ? `${df >= 0 ? "+" : ""}${(100 * df).toFixed(1)} pp` : `${df >= 0 ? "+" : ""}${df.toFixed(3)}`, df > 0 ? "up" : "down"]; };
    const [td, tc] = R ? dl(R.t1, R.t0) : [null, ""], [dd, dc] = R ? dl(R.d1, R.d0, true) : [null, ""], [fd, fc] = R ? dl(R.f1, R.f0, true) : [null, ""];
    S.rq = S.rq || { src: "All", sort: "trust", truth: true, budget: 0.10 };
    const Q = S.rq;
    let pend = mine.filter((a) => a.status === "pending" && (Q.src === "All" || a.source === Q.src));
    pend = Q.sort === "trust" ? pend.sort((a, b) => a.T - b.T) : pend.sort((a, b) => b.id - a.id);
    const pages = Math.max(1, Math.ceil(pend.length / 8)); S.reviewPage = Math.min(S.reviewPage, pages - 1);
    const view = pend.slice(S.reviewPage * 8, S.reviewPage * 8 + 8);
    main.innerHTML = head(`${d.title} · human in the loop`, "Review queue",
      `Low-trust decisions wait here with their audit evidence. Each verdict re-tunes the alert threshold ${tip("threshold")} and is measured on held-out data. The normal-reasoning baseline is never changed automatically.`) +
      kpis([kpi("Pending review", String(cnt("pending")), `${mine.length} decisions logged`, "review"),
        kpi("Threshold", f3(thr), td || "flag when T < threshold", "accent", tc, "threshold"),
        kpi("Held-out detection", pct(hr.det), dd || "attacks caught", "safe", dc, "det"),
        kpi("Held-out false positives", pct(hr.fpr), fd || `budget ${pct(Q.budget, 0)}`, "attack", fc === "up" ? "warn" : fc, "fpr"),
        kpi("Reviewed", String(cnt("approved") + cnt("confirmed_attack")), `${cnt("confirmed_attack")} confirmed attacks`, "neutral")]) +
      (R ? callout(`<b>${R.outcome === "confirmed_attack" ? "Confirmed attack" : "Approved"} #${R.id}.</b> Threshold ${f3(R.t0)} → <b>${f3(R.t1)}</b>; held-out detection ${pct(R.d0)} → <b>${pct(R.d1)}</b>; false positives ${pct(R.f0)} → <b>${pct(R.f1)}</b>. Chosen to maximise detection within the ${pct(Q.budget, 0)} false-alarm budget on the tuning pool; measured on held-out records never used for selection.`) : "") +
      `<div class="row" style="margin:14px 0">
        <label class="f">Source<select class="inp" id="rSrc">${["All", "stream", "predict", "attack", "demo", "seed"].map((s) => `<option ${s === Q.src ? "selected" : ""}>${s}</option>`).join("")}</select></label>
        <label class="f">Sort<select class="inp" id="rSort"><option value="trust" ${Q.sort === "trust" ? "selected" : ""}>Lowest trust first</option><option value="new" ${Q.sort === "new" ? "selected" : ""}>Newest first</option></select></label>
        <label class="switch" style="margin-top:18px"><input type="checkbox" id="rTruth" ${Q.truth ? "checked" : ""}> Show simulation label</label>
        <label class="f" style="min-width:200px">False-alarm budget: <span class="mono" id="vbud">${Q.budget.toFixed(2)}</span><input type="range" id="rBud" min="0.02" max="0.3" step="0.01" value="${Q.budget}"></label></div>
      <div id="rList" class="grid">${view.length ? "" : `<div class="empty"><h4>Queue is clear</h4><div>No decisions are waiting. Run the <b>Live Monitor</b> with attacks enabled, or launch an attack in the <b>Attack Simulator</b>, to generate flagged decisions.</div></div>`}</div>
      ${pages > 1 ? `<div class="row" style="margin-top:12px"><button class="btn sm" id="rPrev" ${S.reviewPage === 0 ? "disabled" : ""}>← Prev</button><span class="muted">Page ${S.reviewPage + 1} of ${pages}</span><button class="btn sm" id="rNext" ${S.reviewPage >= pages - 1 ? "disabled" : ""}>Next →</button></div>` : ""}`;
    const list = $("#rList");
    view.forEach((a) => {
      const el = document.createElement("div");
      el.className = "card review";
      const truth = Q.truth && a.sim ? badge("sim: " + a.sim, a.sim === "attack" ? "attack" : a.sim === "drift" ? "drift" : "neutral") : "";
      el.innerHTML = `<div class="grid" style="grid-template-columns:minmax(0,1.2fr) minmax(0,1.6fr) minmax(150px,.7fr);align-items:center">
        <div><div class="row" style="gap:8px"><span class="mono" style="font-size:15px;font-weight:600">#${a.id}</span>${badge("Review", "review")}${badge(a.source, "accent", false)}${truth}</div>
          <div class="muted" style="margin-top:6px">${new Date(a.ts * 1000).toLocaleString()} · ref <span class="mono">${esc(a.ref || "—")}</span></div>
          <div style="margin-top:8px">Prediction <b>${esc(d.class_names[a.pred])}</b> · p=<span class="mono">${f3(a.p)}</span></div>
          <div class="mono" style="margin-top:4px;font-size:13px">T ${f3(a.T)} &lt; ${f3(a.threshold)}</div><div class="muted mono">C ${a.C.toFixed(2)} · A ${a.A.toFixed(2)} · S ${a.S.toFixed(2)}</div></div>
        <div><div class="muted">Top deviating features (σ) ${tip("sigma")}</div><div class="chart" data-dev="${a.id}"></div></div>
        <div class="grid" style="gap:8px"><button class="btn primary" data-act="confirmed_attack" data-id="${a.id}">Confirm attack</button><button class="btn" data-act="approved" data-id="${a.id}">Approve</button></div></div>`;
      list.appendChild(el);
      devBars($(`[data-dev="${a.id}"]`, el), d, a.top, 160);
    });
    $$("[data-act]", list).forEach((b) => (b.onclick = () => {
      const id = +b.dataset.id, rec = S.audit.find((x) => x.id === id), t0 = S.thr[d.key], h0 = heldRates(d, t0);
      rec.status = b.dataset.act; rec.note = "reviewer";
      const r = recalibrate(d, `review #${id}: ${b.dataset.act === "confirmed_attack" ? "confirmed attack" : "approved"}`, Q.budget);
      S.reviewDelta = { key: d.key, id, outcome: b.dataset.act, t0, t1: r.t, d0: h0.det, d1: r.r.det, f0: h0.fpr, f1: r.r.fpr };
      toast(`${b.dataset.act === "confirmed_attack" ? "Attack confirmed" : "Decision approved"} · threshold ${f3(t0)} → ${f3(r.t)}`, b.dataset.act === "confirmed_attack" ? "attack" : "safe");
      shell(); review(main, d);
    }));
    $("#rSrc").onchange = (e) => { Q.src = e.target.value; S.reviewPage = 0; review(main, d); };
    $("#rSort").onchange = (e) => { Q.sort = e.target.value; review(main, d); };
    $("#rTruth").onchange = (e) => { Q.truth = e.target.checked; review(main, d); };
    $("#rBud").oninput = (e) => { Q.budget = +e.target.value; $("#vbud").textContent = Q.budget.toFixed(2); };
    if ($("#rPrev")) { $("#rPrev").onclick = () => { S.reviewPage--; review(main, d); }; $("#rNext").onclick = () => { S.reviewPage++; review(main, d); }; }
  }

  // ------------------------------------------------------------------ ANALYTICS
  function analytics(main, d) {
    const ex = S.results && S.results.experiments, P = S.results && S.results.paper;
    main.innerHTML = head("Evidence", "Analytics", `Results reproduced by <code>scripts/run_experiments.py</code> in the Python project, side by side with the paper's values, plus live statistics from this session's audit log. AUCs ${tip("auc")} are paired: each attacked record is compared with its own clean original.`) +
      `<div class="seg" id="anTab" style="margin-bottom:14px"><button data-t="paper" aria-pressed="${S.analyticsTab === "paper"}">Paper experiments</button><button data-t="live" aria-pressed="${S.analyticsTab === "live"}">Live audit stats</button></div><div id="anBody"></div>`;
    $$("#anTab button").forEach((b) => (b.onclick = () => { S.analyticsTab = b.dataset.t; analytics(main, d); }));
    const body = $("#anBody");
    if (S.analyticsTab === "paper") {
      if (!ex) { body.innerHTML = callout("Results are still loading.", true); return; }
      const DOM = ["finance", "healthcare", "recruitment"], DL = DOM.map((k) => INDEX[k].title), D = ex.domains, cd = ex.cross_domain;
      const tbl = (hdr, rows) => `<div class="tablewrap"><table class="t"><thead><tr>${hdr.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c, i) => `<td class="${i ? "num" : ""}">${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
      const hc = D.healthcare.stability_auc, sens = ex.sensitivity && ex.sensitivity.recruitment_unbalanced, mon = S.results.monitor;
      body.innerHTML = kpis([kpi("Attack success", `${cd.attack_success}/${cd.attack_attempted}`, `paper ${P.attack.success}/${P.attack.total}`, "attack", "", "pp"),
        kpi("Mean |Δp|", DOM.map((k) => D[k].attack.mean_abs_dp.toFixed(3)).join(" / "), "paper 0.027 / 0.031 / 0.032", "neutral"),
        kpi("Stability AUC range", `${cd.stability.fixed_range.toFixed(3)} → ${cd.stability.absolute_range.toFixed(3)}`, "paper 0.350 → 0.137", "drift"),
        kpi("Summation identity error", Math.max(...DOM.map((k) => D[k].fig2.identity_error)).toExponential(1), "base + Σφ = p", "safe", "", "SHAP")]) +
        section("1 · Explanation-space vs input-space detection") + `<div class="split-r"><div id="c1" class="chart"></div><div>${tbl(["Domain", "Expl. (ours)", "Input (ours)", "5-seed expl / input", "Paper"], DOM.map((k) => [INDEX[k].title, D[k].space_auc.explanation.toFixed(3), D[k].space_auc.input.toFixed(3), `${D[k].seed_spread.explanation.mean.toFixed(3)} / ${D[k].seed_spread.input.mean.toFixed(3)}`, `${P.space_auc[k].explanation.toFixed(3)} / ${P.space_auc[k].input.toFixed(3)}`]))}<div class="muted" style="margin-top:8px">Error bars: 95% bootstrap CI over the 60 attack/original pairs. With 60 pairs the intervals are about ±0.07 wide, so single-run gaps of that size are within sampling noise.</div></div></div>` +
        section("2 · Stability signal: fixed direction vs absolute deviation") + `<div class="split-r"><div id="c2" class="chart"></div><div>${callout(`<b>Inversion.</b> In healthcare the naive signal scores <b>${hc.fixed.toFixed(3)}</b> (below 0.5): attacked records look <i>more</i> stable. Prediction spread under the same ±2% jitter is <b>${hc.pred_dispersion_attack.toFixed(3)}</b> for attacked vs <b>${hc.pred_dispersion_clean.toFixed(3)}</b> for originals (paper 0.018 vs 0.031): the attack parks records in flat regions of the decision surface. Scoring |deviation| from the baseline median treats unusually stable and unusually volatile explanations alike.`)}</div></div>` +
        section("3 · Per-signal and combined trust AUC") + `<div class="split-r"><div id="c3" class="chart"></div><div>${tbl(["Signal"].concat(DL.map((t) => t + " (ours / paper)")), [["confidence", "Confidence 1−C"], ["anomaly", "Anomaly A"], ["stability", "Stability S"], ["trust_equal", "Equal-weight trust"], ["trust_fitted", "Fitted trust"]].map(([k, n]) => [n].concat(DOM.map((dm) => `${D[dm].signal_auc[k].toFixed(3)} / ${P.signal_auc[dm][k].toFixed(3)}`))))}</div></div>` +
        section("4 · Feedback loop: threshold recalibration") + `<div class="split-r"><div id="c4" class="chart"></div><div>${tbl(["Domain", "Threshold", "Held-out detection", "Held-out FPR"], DOM.map((k) => { const f = D[k].feedback, h = f.heldout; return [INDEX[k].title, `${f3(f.default_threshold)} → ${f3(f.recalibrated_threshold)}`, `${pct(h.detection_before)} → ${pct(h.detection_after)}`, `${pct(h.fpr_before)} → ${pct(h.fpr_after)}`]; }).concat([["Pooled", "per domain", `${pct(cd.feedback_pooled.heldout.before.detection)} → ${pct(cd.feedback_pooled.heldout.after.detection)}`, `${pct(cd.feedback_pooled.heldout.before.fpr)} → ${pct(cd.feedback_pooled.heldout.after.fpr)}`]]))}<div class="muted" style="margin-top:8px">Paper: detection 10% → 40%, FPR 5% → 13% on held-out data.</div></div></div>` +
        section("5 · Baseline contamination (dose-response)") + `<div class="split-r"><div id="c5" class="chart"></div><div>${callout("Wrongly cleared attacks are appended to the clean baseline before the detector is refitted. Degradation is gradual: isolated review mistakes are tolerable, a systematically compromised review process is not. That's why baseline refits are manual and restricted to reviewer-verified records.")}</div></div>` +
        (sens ? section("Sensitivity: recruitment without class weighting") + callout(`The paper's recruitment metrics (recall 0.106) are reproduced almost exactly <b>without</b> class weighting: recall ${sens.model.recall.toFixed(3)}, AUC ${sens.model.roc_auc.toFixed(3)} (TN/FP/FN/TP ${sens.model.tn}/${sens.model.fp}/${sens.model.fn}/${sens.model.tp}). On that model explanation vs input AUC is <b>${sens.space_auc.explanation.toFixed(3)}</b> vs <b>${sens.space_auc.input.toFixed(3)}</b> (paper 0.635 vs 0.558) and equal-weight trust AUC <b>${sens.signal_auc.trust_equal.toFixed(3)}</b> (paper 0.614). The deployed model uses class_weight='balanced' as specified.`) : "") +
        section("Live-monitor classifier (simulation, 4 seeds × 260 decisions)") + tbl(["Domain", "False spike (clean)", "False drift (clean)", "Burst of 20 detected", "Drift detected"], DOM.map((k) => { const m = mon.domains[k], c4 = (a, f) => `${a.filter(f).length}/${a.length}`; return [INDEX[k].title, c4(m.clean, (r) => r.spike_steps > 0), c4(m.clean, (r) => r.drift_steps > 0), c4(m.burst, (r) => r.spike_in_burst_window), c4(m.drift, (r) => r.first_drift != null)]; })) +
        section("Protected model performance (Table II)") + tbl(["Domain", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], DOM.map((k) => [INDEX[k].title].concat(["accuracy", "precision", "recall", "f1", "roc_auc"].map((m) => `${D[k].model[m].toFixed(4)} (${P.model[k][m].toFixed(4)})`)))) + `<div class="muted" style="margin-top:6px">Ours (paper) · held-out test set.</div>`;
      grouped($("#c1"), DL, { "Explanation (ours)": DOM.map((k) => D[k].space_auc.explanation), "Input (ours)": DOM.map((k) => D[k].space_auc.input), "Explanation (paper)": DOM.map((k) => P.space_auc[k].explanation), "Input (paper)": DOM.map((k) => P.space_auc[k].input) },
        { ref: 0.5, range: [0.3, 1.02], errors: { "Explanation (ours)": DOM.map((k) => D[k].space_auc.explanation_ci), "Input (ours)": DOM.map((k) => D[k].space_auc.input_ci) }, h: 330 });
      grouped($("#c2"), DL, { "Fixed direction (ours)": DOM.map((k) => D[k].stability_auc.fixed), "Absolute deviation (ours)": DOM.map((k) => D[k].stability_auc.absolute), "Fixed (paper)": DOM.map((k) => P.stability_auc[k].fixed), "Absolute (paper)": DOM.map((k) => P.stability_auc[k].absolute) }, { ref: 0.5 });
      const sig = ["confidence", "anomaly", "stability", "trust_equal", "trust_fitted"];
      grouped($("#c3"), ["Confidence", "Anomaly", "Stability", "Equal trust", "Fitted trust"], Object.fromEntries(DOM.map((k) => [INDEX[k].title, sig.map((s) => D[k].signal_auc[s])])), { ref: 0.5, h: 320 });
      grouped($("#c4"), DL.concat(["Pooled"]), { "Default threshold": DOM.map((k) => D[k].feedback.heldout.detection_before).concat([cd.feedback_pooled.heldout.before.detection]), Recalibrated: DOM.map((k) => D[k].feedback.heldout.detection_after).concat([cd.feedback_pooled.heldout.after.detection]) }, { ref: null, ytitle: "held-out detection", fmt: (v) => Math.round(v * 100) + "%", tickformat: ".0%", range: [0, 0.6], h: 290 });
      const c = col();
      plot($("#c5"), DOM.map((k, i) => ({ type: "scatter", mode: "lines+markers", x: D[k].contamination.map((z) => z.level), y: D[k].contamination.map((z) => z.auc), name: INDEX[k].title, line: { color: c.s[i], width: 2 }, marker: { size: 8, line: { color: c.surface, width: 2 } } })),
        { height: 290, xaxis: { title: { text: "attacks added to baseline (share of baseline size)" }, tickformat: ".0%" }, yaxis: { title: { text: "explanation-space AUC" }, range: [0.3, 1] } });
    } else {
      const A = S.audit.filter((a) => a.domain === d.key), n = A.length, c = col();
      const fl = A.filter((a) => a.verdict === "Review").length;
      body.innerHTML = kpis([kpi("Decisions", n.toLocaleString(), esc(d.title), "accent"), kpi("Flag rate", pct(fl / Math.max(n, 1)), `${fl} routed to review`, "review"),
        kpi("Confirmed attacks", String(A.filter((a) => a.status === "confirmed_attack").length), "by reviewers", "attack"),
        kpi("Approved after review", String(A.filter((a) => a.status === "approved").length), "false alarms cleared", "safe"),
        kpi("Mean trust", f3(A.reduce((s, a) => s + a.T, 0) / Math.max(n, 1)), null, "neutral", "", "trust")]) +
        `<div class="split-r"><div>${section("Decisions over time")}<div id="l1" class="chart"></div></div><div>${section("Review status")}<div id="l2" class="chart"></div></div></div>
        <div class="grid g2"><div>${section("Trust by reviewer outcome")}<div id="l3" class="chart"></div><div class="muted">Clean = auto-accepted or approved; attacked = reviewer-confirmed.</div></div><div>${section("Threshold history")}<div id="l4" class="chart"></div></div></div>`;
      const span = n > 1 ? A[A.length - 1].ts - A[0].ts : 0, bucket = span > 6 * 3600 ? 3600 : span > 1800 ? 300 : 60;
      const bk = {}; A.forEach((a) => { const b = Math.floor(a.ts / bucket) * bucket; bk[b] = bk[b] || { Accept: 0, Review: 0 }; bk[b][a.verdict]++; });
      const xs = Object.keys(bk).sort().map((t) => new Date(+t * 1000));
      plot($("#l1"), ["Accept", "Review"].map((v) => ({ type: "bar", name: v, x: xs, y: Object.keys(bk).sort().map((t) => bk[t][v]), marker: { color: v === "Accept" ? c.safe : c.review } })), { barmode: "stack", height: 280, yaxis: { title: { text: "decisions" } } });
      const sts = [["auto_accepted", "Auto-accepted", c.safe], ["pending", "Pending", c.review], ["approved", "Approved", c.accent], ["confirmed_attack", "Confirmed attack", c.attack]];
      plot($("#l2"), [{ type: "pie", hole: 0.66, sort: false, labels: sts.map((s) => s[1]), values: sts.map((s) => A.filter((a) => a.status === s[0]).length), marker: { colors: sts.map((s) => s[2]), line: { color: c.surface, width: 2 } }, textinfo: "none" }],
        { height: 250, margin: { l: 8, r: 8, t: 8, b: 8 }, legend: { orientation: "v", x: 1, y: 0.5, xanchor: "left" }, annotations: [{ text: `<b>${n}</b><br>decisions`, showarrow: false, font: { family: MONO, size: 15, color: c.text } }] });
      const tc = A.filter((a) => a.status === "approved" || a.status === "auto_accepted").map((a) => a.T), ta = A.filter((a) => a.status === "confirmed_attack").map((a) => a.T);
      plot($("#l3"), [{ type: "histogram", x: tc, name: "Clean", xbins: { start: 0, end: 1, size: 0.025 }, histnorm: "probability", marker: { color: rgba(c.safe, 0.6) } }, { type: "histogram", x: ta, name: "Attacked", xbins: { start: 0, end: 1, size: 0.025 }, histnorm: "probability", marker: { color: rgba(c.attack, 0.6) } }],
        { barmode: "overlay", height: 260, xaxis: { title: { text: "trust T" }, range: [0, 1] }, yaxis: { tickformat: ".0%" }, shapes: [{ type: "line", x0: S.thr[d.key], x1: S.thr[d.key], yref: "paper", y0: 0, y1: 1, line: { color: c.review, width: 2, dash: "dash" } }] });
      const h = S.thrHist[d.key];
      plot($("#l4"), [{ type: "scatter", mode: "lines+markers", x: h.map((z) => new Date(z.ts * 1000)), y: h.map((z) => z.t), text: h.map((z) => z.reason), line: { shape: "hv", color: c.accent, width: 2 }, marker: { size: 8, line: { color: c.surface, width: 2 } }, hovertemplate: "%{y:.3f}<br>%{text}<extra></extra>" }], { height: 260, yaxis: { title: { text: "threshold" } } });
    }
  }

  // ------------------------------------------------------------------ AUDIT
  function audit(main, d) {
    S.af = S.af || { dom: S.domain, verdict: "", status: "", source: "", q: "", tmin: 0, tmax: 1 };
    const F = S.af;
    main.innerHTML = head("Accountability", "Audit log", "Every decision SEAF has scored in this session: prediction, trust components, verdict and review outcome, with the full attribution record kept for adjudication.") +
      `<div class="card"><div class="row">
        <label class="f">Domain<select class="inp" id="fDom"><option value="">All</option>${Object.keys(S.dom).map((k) => `<option value="${k}" ${F.dom === k ? "selected" : ""}>${esc(INDEX[k].title)}</option>`).join("")}</select></label>
        <label class="f">Verdict<select class="inp" id="fV"><option value="">All</option><option ${F.verdict === "Accept" ? "selected" : ""}>Accept</option><option ${F.verdict === "Review" ? "selected" : ""}>Review</option></select></label>
        <label class="f">Status<select class="inp" id="fS"><option value="">All</option>${["auto_accepted", "pending", "approved", "confirmed_attack"].map((s) => `<option ${F.status === s ? "selected" : ""}>${s}</option>`).join("")}</select></label>
        <label class="f">Source<select class="inp" id="fSrc"><option value="">All</option>${["seed", "predict", "attack", "stream", "demo"].map((s) => `<option ${F.source === s ? "selected" : ""}>${s}</option>`).join("")}</select></label>
        <label class="f" style="flex:1;min-width:180px">Search<input class="inp" id="fQ" placeholder="audit id, record ref, note…" value="${esc(F.q)}"></label>
        <label class="f">Trust ≥ <input class="inp mono" id="fMin" type="number" step="0.05" min="0" max="1" value="${F.tmin}" style="width:90px"></label>
        <label class="f">Trust ≤ <input class="inp mono" id="fMax" type="number" step="0.05" min="0" max="1" value="${F.tmax}" style="width:90px"></label></div></div>
      <div class="row" style="margin:12px 0;justify-content:space-between"><div class="muted" id="aCount"></div><button class="btn sm" id="aCopy">Copy filtered rows as CSV</button></div>
      <div class="tablewrap" style="max-height:460px;overflow:auto"><table class="t"><thead><tr><th>#</th><th>Time</th><th>Domain</th><th>Source</th><th>Ref</th><th>Pred</th><th>p</th><th>C</th><th>A</th><th>S</th><th>Trust</th><th>Thr.</th><th>Verdict</th><th>Status</th><th>Top deviations</th></tr></thead><tbody id="aBody"></tbody></table></div>
      <div id="aDetail" style="margin-top:14px"></div>`;
    const rowsOf = () => S.audit.filter((a) => (!F.dom || a.domain === F.dom) && (!F.verdict || a.verdict === F.verdict) && (!F.status || a.status === F.status) && (!F.source || a.source === F.source) &&
      a.T >= F.tmin && a.T <= F.tmax && (!F.q || String(a.id) === F.q.replace("#", "") || (a.ref || "").toLowerCase().includes(F.q.toLowerCase()) || (a.note || "").toLowerCase().includes(F.q.toLowerCase()))).sort((a, b) => b.id - a.id);
    const draw = () => {
      const rows = rowsOf();
      $("#aCount").innerHTML = `${rows.length.toLocaleString()} of ${S.audit.length.toLocaleString()} decisions ${badge(rows.filter((r) => r.verdict === "Review").length + " flagged", "review")}`;
      $("#aBody").innerHTML = rows.slice(0, 400).map((a) => { const dd = S.dom[a.domain]; return `<tr class="click ${S.auditSel === a.id ? "sel" : ""}" data-id="${a.id}"><td class="num">${a.id}</td><td>${new Date(a.ts * 1000).toLocaleString()}</td><td>${esc(INDEX[a.domain].title)}</td><td>${esc(a.source)}</td><td class="mono">${esc(a.ref || "")}</td><td>${esc(dd.class_names[a.pred])}</td><td class="num">${f3(a.p)}</td><td class="num">${a.C.toFixed(3)}</td><td class="num">${a.A.toFixed(3)}</td><td class="num">${a.S.toFixed(3)}</td><td class="num"><span class="bar-cell" style="width:${Math.round(a.T * 50)}px"></span> ${a.T.toFixed(3)}</td><td class="num">${a.threshold.toFixed(3)}</td><td>${vbadge(a.verdict)}</td><td>${esc(a.status)}</td><td>${esc(a.top.slice(0, 3).map((t) => `${t.feature} ${t.sigma >= 0 ? "+" : ""}${t.sigma.toFixed(1)}σ`).join(", "))}</td></tr>`; }).join("");
      $$("#aBody tr").forEach((tr) => (tr.onclick = () => { S.auditSel = +tr.dataset.id; draw(); detail(); }));
    };
    const detail = () => {
      const a = S.audit.find((x) => x.id === S.auditSel); if (!a) { $("#aDetail").innerHTML = `<div class="muted">Select a row to inspect its full audit record.</div>`; return; }
      const dd = S.dom[a.domain];
      $("#aDetail").innerHTML = `<h3 style="font-size:17px;margin-bottom:10px">Audit record #${a.id}</h3><div class="grid g2"><div class="card"><div class="muted">${esc(INDEX[a.domain].title)} · ${esc(a.source)} · ${new Date(a.ts * 1000).toLocaleString()}</div><div class="row" style="margin:8px 0">${vbadge(a.verdict)}${badge(a.status, "neutral")}</div><div class="mono" style="font-size:13px">p=${a.p.toFixed(4)} · T=${a.T.toFixed(4)} · thr=${a.threshold.toFixed(3)}</div>
        <div class="tablewrap" style="margin-top:10px;max-height:260px;overflow:auto"><table class="t"><thead><tr><th>Feature</th><th>Value</th><th>SHAP</th></tr></thead><tbody>${dd.features.map((f, j) => `<tr><td>${esc(lab(dd, f))}</td><td class="num">${esc(disp(dd, j, a.x[j]))}</td><td class="num">${a.phi[j].toFixed(4)}</td></tr>`).join("")}</tbody></table></div></div><div class="card"><div class="muted">Top deviating features (σ) ${tip("sigma")}</div><div id="aDev" class="chart"></div></div></div>`;
      devBars($("#aDev"), dd, a.top, 230);
    };
    const bind = (id, k, num) => ($("#" + id).oninput = (e) => { F[k] = num ? +e.target.value : e.target.value; draw(); });
    bind("fDom", "dom"); bind("fV", "verdict"); bind("fS", "status"); bind("fSrc", "source"); bind("fQ", "q"); bind("fMin", "tmin", true); bind("fMax", "tmax", true);
    $("#aCopy").onclick = async () => {
      const rows = rowsOf(), hdr = "id,time,domain,source,ref,pred,p,C,A,S,T,threshold,verdict,status,top_deviations";
      const csv = [hdr].concat(rows.map((a) => [a.id, new Date(a.ts * 1000).toISOString(), a.domain, a.source, a.ref || "", a.pred, a.p, a.C, a.A, a.S, a.T, a.threshold, a.verdict, a.status, `"${a.top.slice(0, 3).map((t) => t.feature + " " + t.sigma.toFixed(2)).join("; ")}"`].join(","))).join("\n");
      try { await navigator.clipboard.writeText(csv); toast(`Copied ${rows.length} rows as CSV`, "safe"); }
      catch (e) { const ta = document.createElement("textarea"); ta.value = csv; ta.className = "inp mono"; ta.rows = 8; $("#aDetail").prepend(ta); ta.select(); toast("Clipboard blocked: the CSV is selected below, press Ctrl/Cmd+C"); }
    };
    draw(); detail();
  }

  // ------------------------------------------------------------------ ABOUT
  function about(main, d) {
    const dep = S.results && S.results.deployment, thr = S.thr[d.key], hr = heldRates(d, thr);
    main.innerHTML = head("Operations", "About & admin", "This page is a temporary in-browser deployment of the SEAF Python project. The trained models were exported and run here in JavaScript.") +
      `<div class="grid g2"><div class="card"><h4>What runs where</h4><ul class="muted" style="padding-left:18px;line-height:1.7;margin:8px 0 0">
        <li>The <b>Random Forest</b> (300 trees, depth 10) and the <b>explanation-space Isolation Forest</b> are exported from the trained artifacts and evaluated in a Web Worker.</li>
        <li><b>TreeSHAP</b> is a port of the path-dependent algorithm shap uses; the ±2% stability jitter re-implements numpy's PCG64 generator. Verified against the Python library: p, SHAP and anomaly scores match to ~1e-15, trust to &lt;1e-4.</li>
        <li>Attacks, threshold recalibration and the drift / spike monitor are ports of <code>seaf/attack.py</code>, <code>seaf/feedback.py</code> and <code>seaf/drift.py</code>.</li>
        <li>The audit log, review queue and thresholds live <b>in memory for you only</b> and reset when you reload the page. Nobody else sees your reviews.</li>
        <li>Not included here: CSV upload and baseline refit (they need Python training). Use the Streamlit app for those.</li></ul></div>
        <div class="card"><h4>Threshold · ${esc(d.title)}</h4>${kpis([kpi("Current threshold", f3(thr), "default " + f3(d.seaf.default_threshold), "accent", "", "threshold"), kpi("Held-out detection", pct(hr.det), null, "safe", "", "det"), kpi("Held-out FPR", pct(hr.fpr), null, "review", "", "fpr")])}
          <div class="row"><button class="btn primary" id="adRecal">Recalibrate now (10% budget)</button><button class="btn" id="adReset">Reset to default</button><button class="btn" id="adWipe">Reset demo data</button></div>
          <div class="tablewrap" style="margin-top:12px;max-height:220px;overflow:auto"><table class="t"><thead><tr><th>Time</th><th>Threshold</th><th>Reason</th></tr></thead><tbody>${S.thrHist[d.key].slice().reverse().map((h) => `<tr><td>${new Date(h.ts * 1000).toLocaleTimeString()}</td><td class="num">${f3(h.t)}</td><td>${esc(h.reason)}</td></tr>`).join("")}</tbody></table></div></div></div>
      ${section("Models loaded in this browser")}<div class="tablewrap"><table class="t"><thead><tr><th>Domain</th><th>Records</th><th>Features</th><th>RF nodes</th><th>Accuracy</th><th>ROC-AUC</th><th>Baseline n</th><th>Loaded</th></tr></thead><tbody>${Object.keys(INDEX).map((k) => `<tr><td>${esc(INDEX[k].title)}</td><td class="num">${INDEX[k].n_rows.toLocaleString()}</td><td class="num">${INDEX[k].n_features}</td><td class="num">${INDEX[k].n_nodes.toLocaleString()}</td><td class="num">${INDEX[k].accuracy.toFixed(4)}</td><td class="num">${INDEX[k].roc_auc.toFixed(4)}</td><td class="num">${INDEX[k].n_baseline}</td><td>${S.dom[k] ? badge("yes", "safe") : badge("on demand", "neutral", false)}</td></tr>`).join("")}</tbody></table></div>
      ${dep ? section("Python deployment footprint (measured)") + `<div class="muted">Container RAM after visiting all pages: 220 MiB · cold start ≈ 5 s · scoring one record with stability: ${Object.keys(dep.domains).map((k) => `${INDEX[k].title} ${dep.domains[k].score_one_with_stability_ms} ms`).join(", ")}.</div>` : ""}
      ${section("Source")}<div class="muted">GitHub: <span class="mono">shivanshkumar0512/minor_project_122</span>, branch <span class="mono">claude/vibrant-goldberg-e8x8q3</span>. The full Streamlit app, experiments and deployment files (Hugging Face Spaces Docker, Streamlit Community Cloud) live there.</div>`;
    $("#adRecal").onclick = () => { const r = recalibrate(d, "manual recalibration (budget 0.10)"); toast(`Threshold ${f3(r.old)} → ${f3(r.t)}`); shell(); about(main, d); };
    $("#adReset").onclick = () => { S.thr[d.key] = d.seaf.default_threshold; S.thrHist[d.key].push({ ts: now(), t: d.seaf.default_threshold, reason: "manual reset to default" }); shell(); about(main, d); };
    $("#adWipe").onclick = () => { S.audit = []; S.nextId = 1; S.reviewDelta = null; S.demo = null; S.live = null; Object.values(S.dom).forEach((dm) => { S.thr[dm.key] = dm.seaf.default_threshold; S.thrHist[dm.key] = [{ ts: now(), t: dm.seaf.default_threshold, reason: "reset" }]; seed(dm); }); toast("Demo data reset", "safe"); shell(); about(main, d); };
  }

  const VIEWS = { home, predict, attack, monitor, review, analytics, audit, about };

  // ------------------------------------------------------------------ boot
  fetch(new URL("data/results.json", location.href)).then((r) => r.json()).then((j) => { S.results = j; if (S.page === "home" || S.page === "analytics" || S.page === "about") render(); }).catch(() => {});
  const mq = window.matchMedia("(prefers-color-scheme: light)");
  (mq.addEventListener ? mq.addEventListener("change", () => render()) : null);
  new MutationObserver(() => render()).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  const start = location.hash.replace("#", "");
  if (VIEWS[start]) S.page = start;
  render();
})();
