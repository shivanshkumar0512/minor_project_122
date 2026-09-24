// node web/tests/verify_engine.js <reference.json>  - compares the JS engine with the Python library
const fs = require("fs"), path = require("path");
const SEAF = require("../src/engine.js");
const ref = JSON.parse(fs.readFileSync(process.argv[2]));
let ok = true;
for (const key of Object.keys(ref)) {
  const m = SEAF.load(JSON.parse(fs.readFileSync(path.join(__dirname, "..", "data", key + ".json"))));
  const R = ref[key];
  let dp = 0, dphi = 0, dA = 0, dT = 0, dS = 0, dD = 0;
  const t0 = Date.now();
  const res = SEAF.score(m, R.X);
  const ms = (Date.now() - t0) / R.X.length;
  res.forEach((r, i) => {
    dp = Math.max(dp, Math.abs(r.p - R.p[i]));
    r.phi.forEach((v, j) => (dphi = Math.max(dphi, Math.abs(v - R.phi[i][j]))));
    dA = Math.max(dA, Math.abs(r.anomaly_raw - R.A_raw[i]));
    dT = Math.max(dT, Math.abs(r.T - R.T[i]));
    dS = Math.max(dS, Math.abs(r.S - R.S[i]));
    dD = Math.max(dD, Math.abs(r.dispersion - R.disp[i]) / R.disp[i]);
  });
  const sumErr = Math.max(...res.map((r) => Math.abs(r.base + r.phi.reduce((a, b) => a + b, 0) - r.p)));
  console.log(`${key}: max|dp|=${dp.toExponential(1)} max|dphi|=${dphi.toExponential(1)} max|dA|=${dA.toExponential(1)} ` +
              `identity=${sumErr.toExponential(1)} | max|dS|=${dS.toExponential(1)} max rel disp=${dD.toFixed(2)} max|dT|=${dT.toExponential(1)} | ${ms.toFixed(0)} ms/record`);
  if (dp > 1e-9 || dphi > 1e-9 || dA > 1e-9 || dT > 1e-4) ok = false;
}
console.log(ok ? "MATCH: p, phi, anomaly exact (1e-9); stability and trust within 1e-4" : "MISMATCH");
process.exit(ok ? 0 : 1);
