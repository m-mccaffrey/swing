// Full swing comparison: prepare sequences, find the swing beats (stance,
// load, foot plant, contact, extension, finish), line the swings up beat to
// beat, score each part of the swing and build chart series.
//
// Time is tempo-free: each swing runs on its own clock (estimateSwingFps), so
// slow motion, frame rate and how fast someone swings don't matter. Between
// two beats, time is stretched linearly ("sync the beginning and the end" of
// every part of the swing).

import { KP, C, NUM_KP } from './body25.js';
import { canonicalize, resampleFrames, estimateSwingFps } from './sequence.js';
import { computeSeries, stanceHalfWindow } from './metrics.js';
import { detectPhases, sanitizePhases, refinePhases, PHASE_KEYS, pickSwingFps } from './phases.js';
import { dtw, pathMaps } from './dtw.js';
import { stancePose, poseDistance, similarityFromDistance, proportionScales, rescaleBones } from './match.js';

/** Swing-clock rate both swings are resampled to when proposing beats. */
export const ALIGN_FPS = 60;

/**
 * Canonicalize a raw BODY_25 pixel sequence and precompute what the
 * comparison needs.
 * @param {object} o
 * @param {number[][]} o.frames raw BODY_25 frames (pixels)
 * @param {number} o.fps analyzed frames per second of *video* time
 * @param {number} [o.swingFps] frames per swing-second (estimated if omitted)
 * @param {number} o.stanceIndex
 * @param {'left'|'right'} o.pitcherSide
 * @param {object} [o.phases] known beat frames (pro database entries)
 */
export function prepareSwing({ frames, fps, swingFps = null, stanceIndex, pitcherSide, phases = null }) {
  const sf = swingFps ?? (phases ? pickSwingFps(estimateSwingFps(frames, fps), phases) : estimateSwingFps(frames, fps));
  const { frames: canon, transform } = canonicalize(frames, { pitcherSide, stanceIndex, fps: sf });
  const series = computeSeries(canon, stanceIndex, sf);
  const detected = detectPhases(canon, sf, stanceIndex, series);
  return {
    raw: frames,
    canon,
    transform,
    fps,
    swingFps: sf,
    stanceIndex,
    pitcherSide,
    series,
    phases: phases ? sanitizePhases({ ...phases, stance: stanceIndex }, canon.length) : detected,
    detectedPhases: detected,
    stancePose: stancePose(canon, stanceIndex, stanceHalfWindow(sf)),
  };
}

/**
 * Piecewise-linear map from one set of beat frames to another. Before the
 * first beat everything maps to the first beat; after the last one time runs
 * on at the last segment's rate.
 */
export function beatWarp(from, to) {
  return (x) => {
    if (x <= from[0]) return to[0];
    for (let k = 1; k < from.length; k++) {
      if (x <= from[k]) {
        const span = from[k] - from[k - 1];
        const u = span > 0 ? (x - from[k - 1]) / span : 1;
        return to[k - 1] + u * (to[k] - to[k - 1]);
      }
    }
    const n = from.length;
    let k = n - 1;
    while (k > 0 && !(from[k] - from[k - 1] > 0)) k--;
    const rate = k > 0 ? (to[k] - to[k - 1]) / (from[k] - from[k - 1]) || 1 : 1;
    return to[n - 1] + (x - from[n - 1]) * rate;
  };
}

// Joints used for alignment and posture scoring.
const ALIGN_JOINTS = [
  [KP.Nose, 1], [KP.Neck, 0.8], [C.fShoulder, 0.8], [C.bShoulder, 0.8], [C.fElbow, 0.8], [C.bElbow, 0.8],
  [C.fWrist, 2], [C.bWrist, 2], [KP.MidHip, 1], [C.fHip, 0.6], [C.bHip, 0.6], [C.fKnee, 0.8], [C.bKnee, 0.8],
  [C.fAnkle, 1.2], [C.bAnkle, 1],
];
const VEL_JOINTS = [[C.fWrist, 1], [C.bWrist, 1], [KP.MidHip, 0.5], [C.fAnkle, 0.5]];
const VEL_TAU = 0.05; // swing-seconds: turns TL/s into TL for mixing with positions

function alignFeatures(frames, fps) {
  const s = frames[0];
  return frames.map((f, i) => {
    const prev = frames[Math.max(0, i - 1)];
    const next = frames[Math.min(frames.length - 1, i + 1)];
    const span = (Math.min(frames.length - 1, i + 1) - Math.max(0, i - 1)) / fps || 1;
    const feat = [];
    for (const [j, w] of ALIGN_JOINTS) {
      const ok = f[j * 3 + 2] > 0.05 && s[j * 3 + 2] > 0.05;
      feat.push(ok ? [f[j * 3] - s[j * 3], f[j * 3 + 1] - s[j * 3 + 1], w] : null);
    }
    for (const [j, w] of VEL_JOINTS) {
      const ok = prev[j * 3 + 2] > 0.05 && next[j * 3 + 2] > 0.05;
      feat.push(
        ok
          ? [((next[j * 3] - prev[j * 3]) / span) * VEL_TAU, ((next[j * 3 + 1] - prev[j * 3 + 1]) / span) * VEL_TAU, w]
          : null,
      );
    }
    return feat;
  });
}

function featureCost(a, b) {
  let s = 0;
  let ws = 0;
  for (let k = 0; k < a.length; k++) {
    const p = a[k];
    const q = b[k];
    if (p && q) {
      const dx = p[0] - q[0];
      const dy = p[1] - q[1];
      s += p[2] * (dx * dx + dy * dy);
      ws += p[2];
    }
  }
  return ws ? Math.sqrt(s / ws) : 1;
}

/** Posture of a frame relative to its own MidHip (for scoring body positions). */
function relToHip(f) {
  const g = f.slice();
  const hx = f[KP.MidHip * 3];
  const hy = f[KP.MidHip * 3 + 1];
  for (let j = 0; j < NUM_KP; j++) {
    g[j * 3] -= hx;
    g[j * 3 + 1] -= hy;
  }
  return g;
}

const POSTURE_WEIGHTS = Object.fromEntries(ALIGN_JOINTS.map(([j, w]) => [j, w]));
/** Posture distance (TL) at which a phase score is ≈ 61%. */
export const POSTURE_SIGMA = 0.24;

export const SEGMENTS = [
  { key: 'load', from: 'stance', to: 'load', label: 'Load' },
  { key: 'stride', from: 'load', to: 'footPlant', label: 'Stride' },
  { key: 'swing', from: 'footPlant', to: 'contact', label: 'Launch → contact' },
  { key: 'follow', from: 'contact', to: 'finish', label: 'Extension & finish' },
];

// Joint groups reported in "largest difference" summaries. Positions are
// relative to a reference joint so they describe body shape, not travel.
const DEVIATION_GROUPS = [
  { key: 'hands', label: 'hands', joints: [C.fWrist, C.bWrist], ref: KP.Neck },
  { key: 'backElbow', label: 'back elbow', joints: [C.bElbow], ref: C.bShoulder },
  { key: 'head', label: 'head', joints: [KP.Nose], ref: KP.MidHip },
  { key: 'frontKnee', label: 'front knee', joints: [C.fKnee], ref: C.fHip },
  { key: 'backKnee', label: 'back knee', joints: [C.bKnee], ref: C.bHip },
  { key: 'frontFoot', label: 'front foot', joints: [C.fAnkle], ref: KP.MidHip },
];

function groupPoint(f, g) {
  let x = 0;
  let y = 0;
  let n = 0;
  for (const j of g.joints) {
    if (f[j * 3 + 2] > 0.05) {
      x += f[j * 3];
      y += f[j * 3 + 1];
      n++;
    }
  }
  if (!n || !(f[g.ref * 3 + 2] > 0.05)) return null;
  return [x / n - f[g.ref * 3], y / n - f[g.ref * 3 + 1]];
}

/**
 * Propose the user's beats by aligning the movement with the pro's (DTW on
 * movement-from-stance features at a common swing-clock rate), then refine
 * each beat locally from the user's own motion.
 */
export function proposeBeats(user, pro, proFit) {
  const proEnd = Math.min(pro.canon.length - 1, pro.phases.finish + Math.round(0.1 * pro.swingFps));
  const P = resampleFrames(proFit, pro.swingFps, ALIGN_FPS, pro.stanceIndex, proEnd);
  const U = resampleFrames(user.canon, user.swingFps, ALIGN_FPS, user.stanceIndex, user.canon.length - 1);
  const fp = alignFeatures(P.frames, ALIGN_FPS);
  const fu = alignFeatures(U.frames, ALIGN_FPS);
  const n = P.frames.length;
  const m = U.frames.length;
  const { path, cost } = dtw((i, j) => featureCost(fp[i], fu[j]), n, m, { stepPenalty: 0.03 });
  const { proToUser } = pathMaps(path, n, m);
  const proToRes = (idx) => Math.max(0, Math.min(n - 1, Math.round(((idx - pro.stanceIndex) / pro.swingFps) * ALIGN_FPS)));
  const beats = {};
  for (const key of PHASE_KEYS) {
    beats[key] = key === 'stance' ? user.stanceIndex : Math.round(U.srcIndex[Math.max(0, Math.min(m - 1, Math.round(proToUser[proToRes(pro.phases[key])])))]);
  }
  const aligned = sanitizePhases(beats, user.canon.length);
  return { aligned, refined: refinePhases(user, aligned, pro), cost };
}

/**
 * Compare a prepared user swing with a prepared pro swing.
 * @param {object} user from prepareSwing()
 * @param {object} pro from prepareSwing()
 * @param {{phases?: object}} [opts] the user's own beats (e.g. adjusted by
 *   hand); proposed automatically when omitted
 */
export function compareSwing(user, pro, { phases: fixed = null } = {}) {
  // Give the pro the user's limb proportions so body shape, not build, is compared.
  const scales = proportionScales(pro.stancePose, user.stancePose);
  const proFit = pro.canon.map((f) => rescaleBones(f, scales));

  // The automatic beats are always proposed, so hand-set ones can be shown against them.
  const proposal = proposeBeats(user, pro, proFit);
  const phases = fixed ? sanitizePhases({ ...fixed, stance: user.stanceIndex }, user.canon.length) : proposal.refined;

  // Beat-to-beat time maps between the two swings (original frame indices).
  const pb = PHASE_KEYS.map((k) => pro.phases[k]);
  const ub = PHASE_KEYS.map((k) => phases[k]);
  const proToUser = beatWarp(pb, ub);
  const userToPro = beatWarp(ub, pb);
  const clampU = (j) => Math.max(0, Math.min(user.canon.length - 1, Math.round(j)));
  const clampP = (i) => Math.max(0, Math.min(pro.canon.length - 1, Math.round(i)));

  // Posture score over each part of the swing, sampled evenly between beats.
  const SAMPLES = 24;
  const segments = SEGMENTS.map((seg) => {
    const dists = [];
    const dev = Object.fromEntries(DEVIATION_GROUPS.map((g) => [g.key, { dx: 0, dy: 0, n: 0 }]));
    for (let k = 0; k <= SAMPLES; k++) {
      const u = k / SAMPLES;
      const pf = proFit[clampP(pro.phases[seg.from] + u * (pro.phases[seg.to] - pro.phases[seg.from]))];
      const uf = user.canon[clampU(phases[seg.from] + u * (phases[seg.to] - phases[seg.from]))];
      const { distance } = poseDistance(relToHip(uf), relToHip(pf), POSTURE_WEIGHTS);
      if (Number.isFinite(distance)) dists.push(distance);
      for (const g of DEVIATION_GROUPS) {
        const pu = groupPoint(uf, g);
        const pp = groupPoint(pf, g);
        if (pu && pp) {
          dev[g.key].dx += pu[0] - pp[0];
          dev[g.key].dy += pu[1] - pp[1];
          dev[g.key].n++;
        }
      }
    }
    const meanDist = dists.length ? dists.reduce((x, y) => x + y, 0) / dists.length : Infinity;
    const deviations = DEVIATION_GROUPS.map((g) => {
      const d = dev[g.key];
      const dx = d.n ? d.dx / d.n : 0;
      const dy = d.n ? d.dy / d.n : 0;
      return { key: g.key, label: g.label, dx, dy, d: Math.hypot(dx, dy) };
    }).sort((x, y) => y.d - x.d);
    return { ...seg, distance: meanDist, score: similarityFromDistance(meanDist, POSTURE_SIGMA), deviations };
  });

  // Chart series on the pro's swing clock (swing-seconds relative to contact).
  const chartKeys = ['headX', 'headY', 'handsX', 'handsY', 'hipTravel', 'stride', 'hipTurn', 'shoulderTurn', 'fKnee', 'trunkTilt'];
  const t = [];
  const chart = Object.fromEntries(chartKeys.map((k) => [k, { user: [], pro: [] }]));
  const proEnd = Math.min(pro.canon.length - 1, pro.phases.finish + Math.round(0.1 * pro.swingFps));
  for (let i = pro.stanceIndex; i <= proEnd; i++) {
    t.push((i - pro.phases.contact) / pro.swingFps);
    const uIdx = clampU(proToUser(i));
    for (const k of chartKeys) {
      chart[k].pro.push(pro.series[k][i]);
      chart[k].user.push(user.series[k][uIdx]);
    }
  }
  const phaseTimes = Object.fromEntries(PHASE_KEYS.map((k) => [k, (pro.phases[k] - pro.phases.contact) / pro.swingFps]));

  const weights = { load: 1, stride: 1, swing: 1.5, follow: 1 };
  let ws = 0;
  let ss = 0;
  for (const seg of segments) {
    ss += seg.score * weights[seg.key];
    ws += weights[seg.key];
  }

  return {
    phases,
    autoPhases: proposal.refined,
    alignedPhases: proposal.aligned,
    proScales: scales,
    segments,
    swingScore: ss / ws,
    alignCost: proposal.cost,
    endUserIndex: clampU(proToUser(proEnd)),
    // Frame lookup for synchronized playback: user original index → pro original index.
    userToProOrig: (uIdx) => clampP(userToPro(uIdx)),
    chart: { t, series: chart, phaseTimes },
  };
}
