/* SEAF engine for the browser: a JavaScript port of the seaf/ Python library.
 *
 *   model = SEAF.load(domainJson)             // decode the exported forests
 *   SEAF.score(model, [x1, x2], {stability})  // -> [{p, pred, phi, C, A, S, T, z, top, ...}]
 *
 * TreeSHAP is the path-dependent algorithm of Lundberg et al. (the one shap's
 * C++ TreeExplainer uses), ported with the same flat "unique path" buffers, so
 * attributions match the Python artifacts to floating-point precision.
 * The stability jitter re-implements numpy's default_rng (SeedSequence + PCG64),
 * so S and T are bit-identical to the Python app as well.
 */
(function (root) {
  "use strict";

  // ---------------------------------------------------------------- decoding
  function b64ToBuf(s) {
    if (typeof atob === "function") {
      const bin = atob(s), u = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
      return u.buffer;
    }
    const b = Buffer.from(s, "base64");           // node (tests)
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  }
  function decodeForest(f) {
    const out = { n_trees: f.n_trees, offsets: f.offsets, max_samples: f.max_samples };
    out.feat = new Int16Array(b64ToBuf(f.feat));
    out.thr = new Float64Array(b64ToBuf(f.thr));
    out.left = new Int32Array(b64ToBuf(f.left));
    out.right = new Int32Array(b64ToBuf(f.right));
    if (f.value) out.value = new Float64Array(b64ToBuf(f.value));
    if (f.cover) out.cover = new Float64Array(b64ToBuf(f.cover));
    if (f.nsamp) out.nsamp = new Int32Array(b64ToBuf(f.nsamp));
    // max depth (for the TreeSHAP buffer)
    let maxd = 0;
    const depth = new Int32Array(out.feat.length);
    for (let t = 0; t < f.n_trees; t++) {
      const stack = [out.offsets[t]];
      depth[out.offsets[t]] = 0;
      while (stack.length) {
        const n = stack.pop();
        if (out.left[n] >= 0) {
          depth[out.left[n]] = depth[out.right[n]] = depth[n] + 1;
          stack.push(out.left[n], out.right[n]);
        } else if (depth[n] > maxd) maxd = depth[n];
      }
    }
    out.maxDepth = maxd;
    return out;
  }

  function load(d) {
    const m = Object.assign({}, d);
    m.rf = decodeForest(d.rf);
    m.iso = decodeForest(d.iso);
    m.nf = d.features.length;
    const s = (m.maxDepth = m.rf.maxDepth) + 2;
    m.buf = { fd: new Int32Array(s * (s + 1) / 2 + s), zf: new Float64Array(s * (s + 1) / 2 + s),
              of: new Float64Array(s * (s + 1) / 2 + s), pw: new Float64Array(s * (s + 1) / 2 + s) };
    delete m.rf_raw;
    return m;
  }

  // ---------------------------------------------------------------- forest
  function leafOf(F, root, x) {
    let n = root;
    while (F.left[n] >= 0) n = x[F.feat[n]] <= F.thr[n] ? F.left[n] : F.right[n];
    return n;
  }
  function predict(m, x) {
    const F = m.rf;
    let s = 0;
    for (let t = 0; t < F.n_trees; t++) s += F.value[leafOf(F, F.offsets[t], x)];
    return s / F.n_trees;
  }

  // ---------------------------------------------------------------- TreeSHAP
  function extendPath(B, b, ud, zfrac, ofrac, fi) {
    B.fd[b + ud] = fi; B.zf[b + ud] = zfrac; B.of[b + ud] = ofrac; B.pw[b + ud] = ud === 0 ? 1 : 0;
    for (let i = ud - 1; i >= 0; i--) {
      B.pw[b + i + 1] += ofrac * B.pw[b + i] * (i + 1) / (ud + 1);
      B.pw[b + i] = zfrac * B.pw[b + i] * (ud - i) / (ud + 1);
    }
  }
  function unwindPath(B, b, ud, pi) {
    const o = B.of[b + pi], z = B.zf[b + pi];
    let next = B.pw[b + ud];
    for (let i = ud - 1; i >= 0; i--) {
      if (o !== 0) {
        const tmp = B.pw[b + i];
        B.pw[b + i] = next * (ud + 1) / ((i + 1) * o);
        next = tmp - B.pw[b + i] * z * (ud - i) / (ud + 1);
      } else {
        B.pw[b + i] = B.pw[b + i] * (ud + 1) / (z * (ud - i));
      }
    }
    for (let i = pi; i < ud; i++) { B.fd[b + i] = B.fd[b + i + 1]; B.zf[b + i] = B.zf[b + i + 1]; B.of[b + i] = B.of[b + i + 1]; }
  }
  function unwoundSum(B, b, ud, pi) {
    const o = B.of[b + pi], z = B.zf[b + pi];
    let next = B.pw[b + ud], total = 0;
    for (let i = ud - 1; i >= 0; i--) {
      if (o !== 0) {
        const tmp = next * (ud + 1) / ((i + 1) * o);
        total += tmp;
        next = B.pw[b + i] - tmp * z * ((ud - i) / (ud + 1));
      } else if (z !== 0) {
        total += (B.pw[b + i] / z) / ((ud - i) / (ud + 1));
      }
    }
    return total;
  }
  function recurse(F, B, x, phi, node, pb, ud, pz, po, pf) {
    const b = pb + ud + 1;
    for (let i = 0; i < ud; i++) { B.fd[b + i] = B.fd[pb + i]; B.zf[b + i] = B.zf[pb + i]; B.of[b + i] = B.of[pb + i]; B.pw[b + i] = B.pw[pb + i]; }
    extendPath(B, b, ud, pz, po, pf);
    const l = F.left[node];
    if (l < 0) {
      const v = F.value[node];
      for (let i = 1; i <= ud; i++) {
        const w = unwoundSum(B, b, ud, i);
        phi[B.fd[b + i]] += w * (B.of[b + i] - B.zf[b + i]) * v;
      }
      return;
    }
    const r = F.right[node], sf = F.feat[node];
    const hot = x[sf] <= F.thr[node] ? l : r, cold = hot === l ? r : l;
    const w = F.cover[node], hz = F.cover[hot] / w, cz = F.cover[cold] / w;
    let iz = 1, io = 1, pi = 0;
    for (; pi <= ud; pi++) if (B.fd[b + pi] === sf) break;
    if (pi !== ud + 1) { iz = B.zf[b + pi]; io = B.of[b + pi]; unwindPath(B, b, ud, pi); ud -= 1; }
    recurse(F, B, x, phi, hot, b, ud + 1, hz * iz, io, sf);
    recurse(F, B, x, phi, cold, b, ud + 1, cz * iz, 0, sf);
  }
  function shap(m, x) {
    const F = m.rf, phi = new Float64Array(m.nf);
    for (let t = 0; t < F.n_trees; t++) recurse(F, m.buf, x, phi, F.offsets[t], 0, 0, 1, 1, -1);
    for (let j = 0; j < m.nf; j++) phi[j] /= F.n_trees;
    return phi;
  }

  // ---------------------------------------------------------------- isolation forest
  function avgPath(n) {
    if (n <= 1) return 0;
    if (n === 2) return 1;
    return 2 * (Math.log(n - 1) + 0.5772156649015329) - 2 * (n - 1) / n;
  }
  function anomaly(m, z) {
    const F = m.iso;
    let depthSum = 0;
    for (let t = 0; t < F.n_trees; t++) {
      let n = F.offsets[t], d = 0;
      while (F.left[n] >= 0) { n = z[F.feat[n]] <= F.thr[n] ? F.left[n] : F.right[n]; d++; }
      depthSum += d + avgPath(F.nsamp[n]);
    }
    return Math.pow(2, -depthSum / (F.n_trees * avgPath(F.max_samples)));
  }

  // ---------------------------------------------------------------- stability
  /* Bit-exact port of numpy.random.default_rng([seed, crc32(row)]).uniform(...)
   * (SeedSequence -> PCG64 XSL-RR -> 53-bit doubles), so the jitter - and hence
   * S and T - match the Python library exactly. */
  const CRC_TABLE = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
    return t;
  })();
  function crc32(bytes) {
    let c = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }
  const M32 = 0xffffffffn, M64 = (1n << 64n) - 1n, M128 = (1n << 128n) - 1n;
  const PCG_MULT = 0x2360ED051FC65DA44385DF649FCCF645n;
  function seedSequenceState(entropy, nWords) {
    const mul = (a, b) => Number((BigInt(a) * BigInt(b)) & M32);
    const hc = { v: 0x43b0d7e5 };
    const hashmix = (value) => {
      value = (value ^ hc.v) >>> 0; hc.v = mul(hc.v, 0x931e8875);
      value = mul(value, hc.v); return (value ^ (value >>> 16)) >>> 0;
    };
    const mix = (x, y) => {
      let r = Number((BigInt(0xca01f9dd) * BigInt(x) - BigInt(0x4973f715) * BigInt(y)) & M32);
      return (r ^ (r >>> 16)) >>> 0;
    };
    const pool = [0, 0, 0, 0];
    for (let i = 0; i < 4; i++) pool[i] = hashmix(i < entropy.length ? entropy[i] : 0);
    for (let s = 0; s < 4; s++) for (let d = 0; d < 4; d++) if (s !== d) pool[d] = mix(pool[d], hashmix(pool[s]));
    for (let s = 4; s < entropy.length; s++) for (let d = 0; d < 4; d++) pool[d] = mix(pool[d], hashmix(entropy[s]));
    let hb = 0x8b51f9dd;
    const out = [];
    for (let i = 0; i < nWords; i++) {
      let v = (pool[i % 4] ^ hb) >>> 0; hb = mul(hb, 0x58f38ded);
      v = mul(v, hb); out.push((v ^ (v >>> 16)) >>> 0);
    }
    return out;
  }
  function PCG64(entropy) {
    const w = seedSequenceState(entropy, 8).map(BigInt);
    const u64 = (i) => w[2 * i] | (w[2 * i + 1] << 32n);        // little-endian uint64 view
    const initstate = (u64(0) << 64n) | u64(1), initseq = (u64(2) << 64n) | u64(3);
    let state = 0n;
    const inc = ((initseq << 1n) | 1n) & M128;
    const step = () => { state = (state * PCG_MULT + inc) & M128; };
    step(); state = (state + initstate) & M128; step();
    this.next64 = () => {
      step();
      const rot = state >> 122n, v = ((state >> 64n) ^ state) & M64;
      return ((v >> rot) | (v << ((64n - rot) & 63n))) & M64;
    };
    this.random = () => Number(this.next64() >> 11n) / 9007199254740992;
  }
  function jitterFactors(m, x) {
    const rowBytes = new Uint8Array(new Float64Array(x).buffer);
    const rng = new PCG64([42, crc32(rowBytes)]);
    const k = m.seaf.k, sc = m.seaf.jitter, out = [];
    for (let c = 0; c < k; c++) out.push(m.numeric_idx.map(() => -sc + 2 * sc * rng.random()));
    return out;
  }
  function dispersion(m, x) {
    const k = m.seaf.k, F = jitterFactors(m, x), P = [];
    for (let c = 0; c < k; c++) {
      const xj = x.slice();
      m.numeric_idx.forEach((j, q) => { xj[j] *= 1 + F[c][q]; });
      P.push(shap(m, xj));
    }
    let tot = 0;
    for (let j = 0; j < m.nf; j++) {
      let mu = 0; for (let c = 0; c < k; c++) mu += P[c][j] / m.seaf.phi_scale[j];
      mu /= k;
      let v = 0; for (let c = 0; c < k; c++) { const d = P[c][j] / m.seaf.phi_scale[j] - mu; v += d * d; }
      tot += Math.sqrt(v / k);
    }
    return tot / m.nf;
  }
  function mulberry32(a) {                     // cheap RNG for attacks / stream only
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------------------------------------------------------------- trust
  const clip01 = (v) => Math.min(1, Math.max(0, v));
  function score(m, X, opts) {
    const stab = !opts || opts.stability !== false, S = m.seaf;
    return X.map((x) => {
      const p = predict(m, x), phi = shap(m, x);
      const z = Array.from(phi, (v, j) => (v - S.phi_mean[j]) / S.phi_scale[j]);
      const aRaw = anomaly(m, z);
      const disp = stab ? dispersion(m, x) : null;
      const C = Math.max(p, 1 - p);
      const A = clip01((aRaw - S.A_lo) / (S.A_hi - S.A_lo));
      const Sn = stab ? clip01((Math.abs(disp - S.m_ref) - S.S_lo) / (S.S_hi - S.S_lo)) : S.S_neutral;
      const T = (C + (1 - A) + (1 - Sn)) / 3;
      const order = z.map((v, j) => j).sort((a, b) => Math.abs(z[b]) - Math.abs(z[a])).slice(0, 5);
      return { x: x.slice(), p, pred: p >= 0.5 ? 1 : 0, phi: Array.from(phi), base: S.expected_value, z,
               anomaly_raw: aRaw, dispersion: disp, C, A, S: Sn, T, stability_measured: stab,
               top: order.map((j) => ({ feature: m.features[j], sigma: z[j] })) };
    });
  }

  // ---------------------------------------------------------------- attacks
  function perturb(m, x, budget, rng) {
    const c = x.slice();
    for (const j of m.numeric_idx) {
      c[j] = x[j] * (1 + (rng() * 2 - 1) * budget);
      if (m.integer_idx.indexOf(j) >= 0) c[j] = Math.round(c[j]);
    }
    return c;
  }
  function relChange(m, x, c) {
    let s = 0;
    for (const j of m.numeric_idx) s += Math.abs(c[j] - x[j]) / (Math.abs(x[j]) > 1e-9 ? Math.abs(x[j]) : 1);
    return s / m.numeric_idx.length;
  }
  /* Prediction-preserving random search (paper 3.4): screen with predict, SHAP only survivors. */
  function attackPP(m, x, o) {
    o = Object.assign({ n: 40, budget: 0.35, tau: 0.06, seed: 1 }, o || {});
    const rng = mulberry32(o.seed), p0 = predict(m, x), phi0 = shap(m, x);
    let best = null, bestD = -1, surv = 0;
    const trace = [];
    for (let i = 0; i < o.n; i++) {
      const c = perturb(m, x, o.budget, rng), p = predict(m, c);
      if ((p >= 0.5) !== (p0 >= 0.5) || Math.abs(p - p0) > o.tau) continue;
      surv++;
      const ph = shap(m, c);
      let d = 0;
      for (let j = 0; j < m.nf; j++) { const v = (ph[j] - phi0[j]) / m.seaf.phi_scale[j]; d += v * v; }
      d = Math.sqrt(d);
      if (d > bestD) { bestD = d; best = { x: c, p }; }
      trace.push(bestD);
    }
    if (!best) return { success: false, x_orig: x, x_adv: x.slice(), p0, p1: p0, n: o.n, survivors: 0, trace };
    return { success: true, kind: "pp", x_orig: x, x_adv: best.x, p0, p1: best.p, displacement: bestD,
             perturbation: relChange(m, x, best.x), n: o.n, survivors: surv, trace };
  }
  function attackEvasion(m, x, o) {
    o = Object.assign({ n: 300, seed: 1, radii: [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75] }, o || {});
    const rng = mulberry32(o.seed), p0 = predict(m, x);
    let tried = 0;
    for (const r of o.radii) {
      let best = null, bestR = Infinity, flips = 0;
      for (let i = 0; i < o.n; i++) {
        const c = perturb(m, x, r, rng), p = predict(m, c);
        tried++;
        if ((p >= 0.5) !== (p0 >= 0.5)) {
          flips++;
          const rc = relChange(m, x, c);
          if (rc < bestR) { bestR = rc; best = { x: c, p }; }
        }
      }
      if (best) return { success: true, kind: "evasion", x_orig: x, x_adv: best.x, p0, p1: best.p,
                         perturbation: bestR, n: tried, survivors: flips, radius: r, trace: [] };
    }
    return { success: false, kind: "evasion", x_orig: x, x_adv: x.slice(), p0, p1: p0, n: tried, survivors: 0, trace: [] };
  }

  // ---------------------------------------------------------------- feedback
  function rates(t, clean, attack) {
    const det = attack.length ? attack.filter((v) => v < t).length / attack.length : 0;
    const fpr = clean.length ? clean.filter((v) => v < t).length / clean.length : 0;
    return { det, fpr };
  }
  /* max detection s.t. FPR <= budget; ties -> lowest threshold (seaf/feedback.py). */
  function selectThreshold(clean, attack, budget, fallback) {
    const tc = clean.slice().sort((a, b) => a - b);
    if (!tc.length) return fallback == null ? 0.5 : fallback;
    const nAllowed = Math.floor(budget * tc.length);
    const tMax = nAllowed < tc.length ? tc[nAllowed] : Infinity;
    const caught = attack.filter((v) => v < tMax);
    if (!caught.length) return fallback != null && fallback < tMax ? fallback : Math.min(tMax, 1);
    const top = Math.max.apply(null, caught);
    const above = tc.filter((v) => v > top).concat(attack.filter((v) => v > top));
    const nxt = above.length ? Math.min(Math.min.apply(null, above), tMax) : Math.min(top + 1e-3, tMax);
    return (top + nxt) / 2;
  }

  // ---------------------------------------------------------------- drift monitor (seaf/drift.py)
  function mwuZ(recent, ref) {
    const n1 = recent.length, n2 = ref.length, all = recent.concat(ref), n = n1 + n2;
    const idx = all.map((v, i) => i).sort((a, b) => all[a] - all[b]);
    const ranks = new Array(n);
    let tie = 0;
    for (let i = 0; i < n;) {
      let j = i;
      while (j + 1 < n && all[idx[j + 1]] === all[idx[i]]) j++;
      const r = (i + j + 2) / 2, c = j - i + 1;
      for (let k = i; k <= j; k++) ranks[idx[k]] = r;
      tie += c * c * c - c;
      i = j + 1;
    }
    let u = 0; for (let i = 0; i < n1; i++) u += ranks[i];
    u -= n1 * (n1 + 1) / 2;
    const sd = Math.sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1))));
    return sd > 0 ? (u - n1 * n2 / 2) / sd : 0;
  }
  function Monitor(o) {
    o = Object.assign({ short: 20, long: 100, expected: 0.05, shiftT: 3.5, trustZ: 2.5, minRef: 30,
                        refSize: 40, recent: 30, debounce: 3 }, o || {});
    this.o = o; this.flags = []; this.trust = []; this.zRef = []; this.zRecent = []; this.raw = []; this.state = "normal";
  }
  Monitor.prototype.update = function (flag, zin, trust) {
    const o = this.o;
    this.flags.push(flag ? 1 : 0); if (this.flags.length > o.long) this.flags.shift();
    this.trust.push(trust); if (this.trust.length > o.long) this.trust.shift();
    if (this.zRef.length < o.refSize) this.zRef.push(zin);
    else { this.zRecent.push(zin); if (this.zRecent.length > o.recent) this.zRecent.shift(); }
    const f = this.flags, mean = (a) => a.reduce((s, v) => s + v, 0) / Math.max(a.length, 1);
    const shortRate = mean(f.slice(-o.short)), longRate = mean(f);
    const refOk = f.length >= o.short + o.minRef;
    const refRate = refOk ? mean(f.slice(0, -o.short)) : longRate;
    const tz = refOk ? mwuZ(this.trust.slice(-o.short), this.trust.slice(0, -o.short)) : 0;
    let shift = 0;
    if (this.zRecent.length >= o.recent && this.zRef.length >= o.refSize) {
      const d = zin.length;
      for (let j = 0; j < d; j++) {
        const R = this.zRef.map((r) => r[j]), Q = this.zRecent.map((r) => r[j]);
        const mR = mean(R), mQ = mean(Q);
        const vR = mean(R.map((v) => (v - mR) ** 2)), vQ = mean(Q.map((v) => (v - mQ) ** 2));
        const se = Math.max(Math.sqrt(vR / R.length + vQ / Q.length), 0.05);
        shift = Math.max(shift, Math.abs(mQ - mR) / se);
      }
    }
    let raw = "normal";
    if (shift >= o.shiftT) raw = "drift";
    else if ((refOk && shortRate >= Math.max(3 * o.expected, refRate + 0.2)) || tz <= -o.trustZ) raw = "spike";
    this.raw.push(raw); if (this.raw.length > o.debounce) this.raw.shift();
    if (this.raw.length === o.debounce && this.raw.every((r) => r === raw)) this.state = raw;
    return { shortRate, longRate, refRate, trustZ: tz, shift, state: this.state };
  };

  const api = { load, predict, shap, anomaly, dispersion, score, attackPP, attackEvasion, rates,
                selectThreshold, Monitor, mwuZ, mulberry32, PCG64, crc32 };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.SEAF = api;
})(typeof self !== "undefined" ? self : this);
