// Referee: fuse several pose models' answers joint by joint.
//
// Each candidate (a model, or a model run on the mirrored image) gives the 17
// COCO keypoints of the same person. For every joint, each candidate's answer
// is scored by a small logistic model trained on human-labelled COCO people
// (see tools/posebench): its own confidence, how far it is from the consensus,
// how many other candidates agree with it, which model it came from, and arm
// geometry for the wrists. The most likely answer wins (averaged with the
// answers that agree with it), and its probability of being within 5% of body
// height of the truth becomes the joint's confidence.
//
// Mirrored exactly by tools/swingdb/referee.py (parity-tested).

export const COCO_GROUP = [0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6];
const NGROUP = 7;

function finite(v) {
  return Number.isFinite(v);
}

function median(values) {
  const v = values.filter(finite).sort((a, b) => a - b);
  if (!v.length) return NaN;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

/** Body height from the consensus pose: per-joint median over candidates. */
export function consensusHeight(cands) {
  let lo = Infinity;
  let hi = -Infinity;
  for (let j = 0; j < 17; j++) {
    const y = median(cands.map((c) => c[j][1]));
    if (finite(y)) {
      lo = Math.min(lo, y);
      hi = Math.max(hi, y);
    }
  }
  return Math.max(1, (hi - lo) * 1.1);
}

/** Raw features of candidate c at joint j (same order as the training code). */
export function rawFeatures(cands, c, j, h) {
  const [x, y, conf] = cands[c][j];
  const mx = median(cands.map((p) => p[j][0]));
  const my = median(cands.map((p) => p[j][1]));
  const dMed = Math.hypot(x - mx, y - my) / h;
  const others = [];
  for (let o = 0; o < cands.length; o++) {
    if (o === c || !finite(cands[o][j][0])) continue;
    others.push(Math.hypot(x - cands[o][j][0], y - cands[o][j][1]) / h);
  }
  const agree5 = others.filter((d) => d < 0.05).length;
  const agree10 = others.filter((d) => d < 0.1).length;
  const nearest = others.length ? Math.min(...others) : 1;
  let confRank = 0;
  for (const p of cands) if (finite(p[j][2]) && p[j][2] > conf) confRank++;
  let geo = 0;
  let wd = 0;
  if (j === 9 || j === 10) {
    const [e, s] = j === 9 ? [7, 5] : [8, 6];
    const P = cands[c];
    const fa = Math.hypot(x - P[e][0], y - P[e][1]);
    const ua = Math.hypot(P[e][0] - P[s][0], P[e][1] - P[s][1]);
    geo = fa / Math.max(ua, 1e-3);
    const ow = j === 9 ? 10 : 9;
    wd = Math.hypot(x - P[ow][0], y - P[ow][1]) / h;
  }
  return [conf, dMed, agree5, agree10, nearest, confRank, geo, wd];
}

/** Feature vector fed to the logistic model (one-hots, raw features, interactions). */
export function featureVector(raw, c, j, K) {
  const [conf, dMed, , , nearest] = raw;
  const M = Array.from({ length: K }, (_, k) => (k === c ? 1 : 0));
  const G = Array.from({ length: NGROUP }, (_, g) => (g === COCO_GROUP[j] ? 1 : 0));
  const ld = Math.log(dMed + 1e-3);
  return [...M, ...G, ...raw, ld, Math.log(nearest + 1e-3), ...G.map((g) => conf * g), ...G.map((g) => ld * g), ...M.map((m) => conf * m)];
}

/**
 * Fuse candidates: `cands` is a list (in the referee's candidate order) of
 * 17x3 arrays [x, y, conf] in pixels (NaN rows when a model gave nothing).
 * The most likely answer wins; when the referee has a `soft_radius`, the
 * answers within that distance of it (a fraction of body height) are averaged,
 * weighted by their probabilities. Returns 17x3 [x, y, probability of the
 * winner] and the index of the winning candidate per joint.
 */
export function fuse(cands, referee) {
  const K = cands.length;
  const usable = cands.filter((c) => c && c.some((q) => finite(q[0])));
  if (!usable.length) return null;
  if (K !== referee.candidates.length) throw new Error(`referee expects ${referee.candidates.length} candidates, got ${K}`);
  const full = cands.map((c) => c || Array.from({ length: 17 }, () => [NaN, NaN, NaN]));
  const h = consensusHeight(full);
  const radius = referee.soft_radius || 0;
  const out = [];
  const chosen = [];
  for (let j = 0; j < 17; j++) {
    let best = -1;
    let bestP = -1;
    const probs = new Array(K).fill(NaN);
    for (let c = 0; c < K; c++) {
      if (!finite(full[c][j][0])) continue;
      const z = featureVector(rawFeatures(full, c, j, h), c, j, K);
      let s = referee.bias;
      for (let i = 0; i < z.length; i++) s += referee.weights[i] * z[i];
      const p = 1 / (1 + Math.exp(-s));
      probs[c] = p;
      if (p > bestP) {
        bestP = p;
        best = c;
      }
    }
    chosen.push(best);
    if (best < 0) {
      out.push([NaN, NaN, 0]);
      continue;
    }
    let [x, y] = full[best][j];
    if (radius > 0) {
      let sw = 0;
      let sx = 0;
      let sy = 0;
      for (let c = 0; c < K; c++) {
        if (!finite(probs[c]) || Math.hypot(full[c][j][0] - x, full[c][j][1] - y) / h > radius) continue;
        sw += probs[c];
        sx += probs[c] * full[c][j][0];
        sy += probs[c] * full[c][j][1];
      }
      x = sx / sw;
      y = sy / sw;
    }
    out.push([x, y, bestP]);
  }
  return { keypoints: out, chosen };
}
