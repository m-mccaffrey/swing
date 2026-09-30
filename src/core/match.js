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

/**
 * Rank pros by stance similarity.
 * @param {number[]} userPose from stancePose()
 * @param {{id:string, stancePose:number[]}[]} pros
 */
export function rankStances(userPose, pros) {
  return pros
    .map((pro) => {
      const { distance, perJoint } = poseDistance(userPose, pro.stancePose);
      return { pro, distance, similarity: similarityFromDistance(distance), perJoint };
    })
    .sort((a, b) => a.distance - b.distance);
}
