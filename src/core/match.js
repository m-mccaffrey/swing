// Stance matching: compare the user's canonical stance with every pro stance
// in the database and rank them.

import { KP, NUM_KP, emptyFrame } from './body25.js';

/** Joint weights for stance similarity (hands and lower half matter most). */
export const STANCE_WEIGHTS = {
  [KP.Nose]: 1.0,
  [KP.Neck]: 0.8,
  [KP.RShoulder]: 1.0,
  [KP.LShoulder]: 1.0,
  [KP.RElbow]: 1.0,
  [KP.LElbow]: 1.0,
  [KP.RWrist]: 1.5,
  [KP.LWrist]: 1.5,
  [KP.RHip]: 0.6,
  [KP.LHip]: 0.6,
  [KP.RKnee]: 1.0,
  [KP.LKnee]: 1.0,
  [KP.RAnkle]: 1.2,
  [KP.LAnkle]: 1.2,
};

/** Similarity falloff: distance (in torso lengths) at which similarity ≈ 61%. */
export const STANCE_SIGMA = 0.16;

/**
 * Average a few canonical frames around the stance and express every joint
 * relative to that frame's MidHip.
 */
export function stancePose(frames, stanceIndex, halfWindow = 1) {
  const out = emptyFrame();
  const lo = Math.max(0, stanceIndex - halfWindow);
  const hi = Math.min(frames.length - 1, stanceIndex + halfWindow);
  for (let j = 0; j < NUM_KP; j++) {
    let sx = 0;
    let sy = 0;
    let sc = 0;
    let n = 0;
    for (let i = lo; i <= hi; i++) {
      const f = frames[i];
      const c = f[j * 3 + 2];
      const hc = f[KP.MidHip * 3 + 2];
      if (c > 0.05 && hc > 0.05) {
        sx += f[j * 3] - f[KP.MidHip * 3];
        sy += f[j * 3 + 1] - f[KP.MidHip * 3 + 1];
        sc += c;
        n++;
      }
    }
    if (n) {
      out[j * 3] = sx / n;
      out[j * 3 + 1] = sy / n;
      out[j * 3 + 2] = sc / n;
    }
  }
  return out;
}

/**
 * Weighted RMS joint distance between two MidHip-relative poses, plus a
 * per-joint breakdown. Joints missing in either pose are skipped.
 */
export function poseDistance(a, b, weights = STANCE_WEIGHTS) {
  let s = 0;
  let ws = 0;
  const perJoint = {};
  for (const [jStr, w] of Object.entries(weights)) {
    const j = Number(jStr);
    if (a[j * 3 + 2] > 0.05 && b[j * 3 + 2] > 0.05) {
      const dx = a[j * 3] - b[j * 3];
      const dy = a[j * 3 + 1] - b[j * 3 + 1];
      const d2 = dx * dx + dy * dy;
      perJoint[j] = { dx, dy, d: Math.sqrt(d2) };
      s += w * d2;
      ws += w;
    }
  }
  return { distance: ws ? Math.sqrt(s / ws) : Infinity, perJoint, coverage: ws };
}

export function similarityFromDistance(d, sigma = STANCE_SIGMA) {
  if (!Number.isFinite(d)) return 0;
  return 100 * Math.exp(-(d * d) / (2 * sigma * sigma));
}

// Skeleton tree rooted at MidHip: [child, parent].
const BONES = [
  [KP.Neck, KP.MidHip], [KP.Nose, KP.Neck], [KP.REye, KP.Nose], [KP.LEye, KP.Nose], [KP.REar, KP.REye], [KP.LEar, KP.LEye],
  [KP.RShoulder, KP.Neck], [KP.RElbow, KP.RShoulder], [KP.RWrist, KP.RElbow],
  [KP.LShoulder, KP.Neck], [KP.LElbow, KP.LShoulder], [KP.LWrist, KP.LElbow],
  [KP.RHip, KP.MidHip], [KP.RKnee, KP.RHip], [KP.RAnkle, KP.RKnee], [KP.RBigToe, KP.RAnkle], [KP.RSmallToe, KP.RBigToe], [KP.RHeel, KP.RAnkle],
  [KP.LHip, KP.MidHip], [KP.LKnee, KP.LHip], [KP.LAnkle, KP.LKnee], [KP.LBigToe, KP.LAnkle], [KP.LSmallToe, KP.LBigToe], [KP.LHeel, KP.LAnkle],
];

/** Bone lengths of a reference pose (e.g. the user's stance). */
export function boneLengths(ref) {
  const out = {};
  for (const [c, p] of BONES) {
    if (ref[c * 3 + 2] > 0.05 && ref[p * 3 + 2] > 0.05) out[c] = Math.hypot(ref[c * 3] - ref[p * 3], ref[c * 3 + 1] - ref[p * 3 + 1]);
  }
  return out;
}

/**
 * Retarget a pose onto another body's proportions: keep every bone's
 * direction from `frame` but use the lengths in `lengths`, starting from the
 * frame's MidHip. A pro drawn this way has the user's limb lengths, so feet,
 * hands and head line up regardless of age or build.
 */
export function retarget(frame, lengths) {
  if (!lengths || !(frame[KP.MidHip * 3 + 2] > 0.05)) return frame;
  const out = frame.slice();
  for (const [c, p] of BONES) {
    if (!(frame[c * 3 + 2] > 0.05) || !(frame[p * 3 + 2] > 0.05) || !(out[p * 3 + 2] > 0.05)) continue;
    const dx = frame[c * 3] - frame[p * 3];
    const dy = frame[c * 3 + 1] - frame[p * 3 + 1];
    const d = Math.hypot(dx, dy);
    const L = lengths[c] ?? d;
    const k = d > 1e-9 ? L / d : 0;
    out[c * 3] = out[p * 3] + dx * k;
    out[c * 3 + 1] = out[p * 3 + 1] + dy * k;
  }
  return out;
}

/**
 * Per-bone length ratios that give `fromPose` the proportions of `toPose`
 * (both measured at the stance). Clamped so a bad detection can't explode.
 */
export function proportionScales(fromPose, toPose) {
  const a = boneLengths(fromPose);
  const b = boneLengths(toPose);
  const out = {};
  for (const k of Object.keys(a)) if (b[k] && a[k] > 1e-6) out[k] = Math.min(2, Math.max(0.5, b[k] / a[k]));
  return out;
}

/**
 * Scale every bone of a frame by a per-bone factor, keeping its direction and
 * its *current* length otherwise. Unlike retarget() this preserves the
 * foreshortening that shows rotation (hips and shoulders narrowing as they
 * turn), so it is safe to apply to every frame of a swing.
 */
export function rescaleBones(frame, scales) {
  if (!scales || !(frame[KP.MidHip * 3 + 2] > 0.05)) return frame;
  const out = frame.slice();
  for (const [c, p] of BONES) {
    if (!(frame[c * 3 + 2] > 0.05) || !(frame[p * 3 + 2] > 0.05) || !(out[p * 3 + 2] > 0.05)) continue;
    const k = scales[c] ?? 1;
    out[c * 3] = out[p * 3] + (frame[c * 3] - frame[p * 3]) * k;
    out[c * 3 + 1] = out[p * 3 + 1] + (frame[c * 3 + 1] - frame[p * 3 + 1]) * k;
  }
  return out;
}

/**
 * Rank pros by stance similarity.
 * @param {number[]} userPose from stancePose()
 * @param {{id:string, stancePose:number[]}[]} pros
 */
export function rankStances(userPose, pros) {
  return pros
    .map((pro) => {
      // Compare shapes, not proportions: give the pro the user's bone lengths.
      const { distance, perJoint } = poseDistance(userPose, rescaleBones(pro.stancePose, proportionScales(pro.stancePose, userPose)));
      return { pro, distance, similarity: similarityFromDistance(distance), perJoint };
    })
    .sort((a, b) => a.distance - b.distance);
}
