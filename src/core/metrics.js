// Per-frame swing metrics computed on canonical frames (see sequence.js).
// Lengths are in torso lengths (TL); angles in degrees. Everything here is
// what a camera perpendicular to the pitch path can actually see: stride,
// weight shift, head movement, hand path, apparent joint angles and how much
// the hips/shoulders narrow as they rotate.

import { C, kx, ky, kc } from './body25.js';
import { angle3, clamp, DEG, derivative, gaussianSmooth, fillGaps } from './math.js';

const MIN = 0.05;

export function P(f, j) {
  return kc(f, j) > MIN ? [kx(f, j), ky(f, j)] : null;
}

export function mid(f, a, b) {
  const pa = P(f, a);
  const pb = P(f, b);
  if (pa && pb) return [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2];
  return pa || pb;
}

export function hands(f) {
  return mid(f, C.fWrist, C.bWrist);
}

function jointAngle(f, a, b, c) {
  const pa = P(f, a);
  const pb = P(f, b);
  const pc = P(f, c);
  return pa && pb && pc ? angle3(pa[0], pa[1], pb[0], pb[1], pc[0], pc[1]) : NaN;
}

/** Reference values taken from the stance frame of the same sequence. */
/**
 * Half-width (frames) of the window averaged for stance values: the stance is
 * still, so averaging ±60 ms beats single-frame keypoint noise.
 */
export function stanceHalfWindow(fps) {
  return Math.max(0, Math.round(0.06 * fps));
}

export function stanceContext(frames, stanceIndex, halfWindow = 0) {
  const lo = Math.max(0, stanceIndex - halfWindow);
  const hi = Math.min(frames.length - 1, stanceIndex + halfWindow);
  const avgOf = (fn) => {
    let sx = 0;
    let sy = 0;
    let n = 0;
    for (let i = lo; i <= hi; i++) {
      const p = fn(frames[i]);
      if (p) {
        sx += p[0];
        sy += p[1];
        n++;
      }
    }
    return n ? [sx / n, sy / n] : null;
  };
  const fh = avgOf((f) => P(f, C.fHip));
  const bh = avgOf((f) => P(f, C.bHip));
  const fs = avgOf((f) => P(f, C.fShoulder));
  const bs = avgOf((f) => P(f, C.bShoulder));
  return {
    nose: avgOf((f) => P(f, C.nose) || P(f, C.neck)) || [0, 1],
    fAnkle: avgOf((f) => P(f, C.fAnkle)) || [0.4, -1.6],
    hands: avgOf(hands) || [0, 1],
    hipW: fh && bh ? fh[0] - bh[0] : NaN,
    shW: fs && bs ? fs[0] - bs[0] : NaN,
  };
}

/** Rotation estimate from how much a left-right segment narrows vs. the stance. */
function turnFromWidth(w, w0) {
  if (!(w0 > 0.12) || !Number.isFinite(w)) return NaN;
  return Math.acos(clamp(w / w0, -1, 1)) * DEG;
}

export const METRICS = {
  headX: { label: 'Head drift (toward pitcher)', unit: 'len' },
  headY: { label: 'Head height change', unit: 'len' },
  handsX: { label: 'Hands (toward pitcher)', unit: 'len' },
  handsY: { label: 'Hands height', unit: 'len' },
  handsHeight: { label: 'Hands above shoulders', unit: 'len' },
  handsDepth: { label: 'Hands vs. back shoulder', unit: 'len' },
  handsToFrontHip: { label: 'Hands vs. front hip', unit: 'len' },
  handReach: { label: 'Hands distance from neck', unit: 'len' },
  extensionX: { label: 'Hands out toward pitcher (vs. neck)', unit: 'len' },
  hipTravel: { label: 'Hip travel (toward pitcher)', unit: 'len' },
  hipHeight: { label: 'Hip height change', unit: 'len' },
  stride: { label: 'Front foot travel', unit: 'len' },
  frontFootLift: { label: 'Front foot lift', unit: 'len' },
  stanceWidth: { label: 'Stance width', unit: 'len' },
  weightShift: { label: 'Hips vs. center of feet', unit: 'len' },
  headOverFeet: { label: 'Head vs. center of feet', unit: 'len' },
  posture: { label: 'Head height above feet', unit: 'len' },
  hipTurn: { label: 'Hip turn (apparent)', unit: 'deg' },
  shoulderTurn: { label: 'Shoulder turn (apparent)', unit: 'deg' },
  fKnee: { label: 'Front knee angle', unit: 'deg' },
  bKnee: { label: 'Back knee angle', unit: 'deg' },
  fElbow: { label: 'Lead elbow angle', unit: 'deg' },
  bElbow: { label: 'Back elbow angle', unit: 'deg' },
  trunkTilt: { label: 'Spine tilt (toward pitcher)', unit: 'deg' },
  shoulderDrop: { label: 'Back shoulder below front', unit: 'len' },
  backElbowHeight: { label: 'Back elbow vs. back shoulder', unit: 'len' },
};

/** Compute all metrics for one canonical frame. */
export function frameMetrics(f, ctx) {
  const nose = P(f, C.nose) || P(f, C.neck);
  const neck = P(f, C.neck);
  const hip = P(f, C.midHip);
  const h = hands(f);
  const fa = P(f, C.fAnkle);
  const ba = P(f, C.bAnkle);
  const feet = fa && ba ? [(fa[0] + ba[0]) / 2, (fa[1] + ba[1]) / 2] : null;
  const fh = P(f, C.fHip);
  const bh = P(f, C.bHip);
  const fs = P(f, C.fShoulder);
  const bs = P(f, C.bShoulder);
  const be = P(f, C.bElbow);
  const m = {};
  m.headX = nose ? nose[0] - ctx.nose[0] : NaN;
  m.headY = nose ? nose[1] - ctx.nose[1] : NaN;
  m.handsX = h ? h[0] : NaN;
  m.handsY = h ? h[1] : NaN;
  m.handsHeight = h && neck ? h[1] - neck[1] : NaN;
  m.handsDepth = h && bs ? h[0] - bs[0] : NaN;
  m.handsToFrontHip = h && fh ? h[0] - fh[0] : NaN;
  m.handReach = h && neck ? Math.hypot(h[0] - neck[0], h[1] - neck[1]) : NaN;
  m.extensionX = h && neck ? h[0] - neck[0] : NaN;
  m.hipTravel = hip ? hip[0] : NaN;
  m.hipHeight = hip ? hip[1] : NaN;
  m.stride = fa ? fa[0] - ctx.fAnkle[0] : NaN;
  m.frontFootLift = fa ? fa[1] - ctx.fAnkle[1] : NaN;
  m.stanceWidth = fa && ba ? fa[0] - ba[0] : NaN;
  m.weightShift = hip && feet ? hip[0] - feet[0] : NaN;
  m.headOverFeet = nose && feet ? nose[0] - feet[0] : NaN;
  m.posture = nose && feet ? nose[1] - Math.min(fa[1], ba[1]) : NaN;
  m.hipTurn = fh && bh ? turnFromWidth(fh[0] - bh[0], ctx.hipW) : NaN;
  m.shoulderTurn = fs && bs ? turnFromWidth(fs[0] - bs[0], ctx.shW) : NaN;
  m.fKnee = jointAngle(f, C.fHip, C.fKnee, C.fAnkle);
  m.bKnee = jointAngle(f, C.bHip, C.bKnee, C.bAnkle);
  m.fElbow = jointAngle(f, C.fShoulder, C.fElbow, C.fWrist);
  m.bElbow = jointAngle(f, C.bShoulder, C.bElbow, C.bWrist);
  m.trunkTilt = neck && hip ? Math.atan2(neck[0] - hip[0], neck[1] - hip[1]) * DEG : NaN;
  m.shoulderDrop = fs && bs ? fs[1] - bs[1] : NaN;
  m.backElbowHeight = be && bs ? be[1] - bs[1] : NaN;
  return m;
}

/**
 * Metric time series for a canonical sequence, plus derived kinematics
 * (hand speed and horizontal hand velocity, both TL/s).
 */
export function computeSeries(frames, stanceIndex, fps) {
  const ctx = stanceContext(frames, stanceIndex, stanceHalfWindow(fps));
  const per = frames.map((f) => frameMetrics(f, ctx));
  const series = {};
  for (const key of Object.keys(METRICS)) series[key] = per.map((m) => m[key]);
  const hx = gaussianSmooth(fillGaps(series.handsX), fps / 60);
  const hy = gaussianSmooth(fillGaps(series.handsY), fps / 60);
  const vx = derivative(hx).map((v) => v * fps);
  const vy = derivative(hy).map((v) => v * fps);
  series.handVx = vx;
  series.handSpeed = vx.map((v, i) => Math.hypot(v, vy[i]));
  series.ctx = ctx;
  return series;
}

/** Value of a series at a (possibly fractional) index, averaged over ±radius frames. */
export function valueAt(series, index, radius = 0) {
  const i = Math.round(index);
  const vals = [];
  for (let k = i - radius; k <= i + radius; k++) {
    const v = series[k];
    if (Number.isFinite(v)) vals.push(v);
  }
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : NaN;
}

export function maxIn(series, from, to) {
  let best = NaN;
  for (let i = Math.max(0, Math.round(from)); i <= Math.min(series.length - 1, Math.round(to)); i++) {
    if (Number.isFinite(series[i]) && !(series[i] <= best)) best = series[i];
  }
  return best;
}

export function minIn(series, from, to) {
  let best = NaN;
  for (let i = Math.max(0, Math.round(from)); i <= Math.min(series.length - 1, Math.round(to)); i++) {
    if (Number.isFinite(series[i]) && !(series[i] >= best)) best = series[i];
  }
  return best;
}
