// Fixing the tracking by hand ("pins").
//
// When the tracker puts a joint in the wrong place, the user drags it to where
// it really is on one frame. That point is trusted from then on (PIN_CONF): the
// smoothing spline goes through it, hand repair never moves it and left/right
// flicker repair never swaps it.
//
// A tracking mistake usually lasts several frames, so the joint is also
// re-picked in the neighbouring frames from the pose models' other answers
// (the candidates the referee chose between, kept from the analysis). From the
// pin outward, each frame takes the answer closest to where the joint is
// heading, until the tracker's own answer agrees again for two frames, the
// answers run out or jump away, or the next pin of that joint.

import { COCO_TO_BODY25, KP, PIN_CONF, kc, kx, ky, pinned, setKp } from './body25.js';
import { torsoLength } from './sequence.js';

const FOLLOWED_CONF = 0.6; // a joint re-picked next to a pin
const FOLLOW_MAX_SEC = 0.5; // how far from a pin the re-picking may reach
const AGREE_TL = 0.1; // the tracker agrees with the followed path (torso lengths)
const GATE_TL = 0.6; // the nearest answer is further than this from where the joint is heading: stop
const FEET = {
  [KP.LAnkle]: [KP.LBigToe, KP.LSmallToe, KP.LHeel],
  [KP.RAnkle]: [KP.RBigToe, KP.RSmallToe, KP.RHeel],
};

/** Joints the user can drag: everything except the midpoints (Neck, MidHip), which follow. */
export const PINNABLE = Array.from({ length: 25 }, (_, j) => j).filter((j) => j !== KP.Neck && j !== KP.MidHip);

/** Valid pins, one per frame and joint (the last one given wins), in the order they were last placed. */
export function normalizePins(pins = []) {
  const byKey = new Map();
  for (const p of pins) {
    if (!(Number.isInteger(p.frame) && PINNABLE.includes(p.joint) && Number.isFinite(p.x) && Number.isFinite(p.y))) continue;
    const key = `${p.frame}:${p.joint}`;
    byKey.delete(key);
    byKey.set(key, p);
  }
  return [...byKey.values()];
}

/** Move joint j of frame f to (x, y); an ankle carries its heel and toes along. */
function moveJoint(f, j, x, y, c) {
  f.touched = true;
  const dx = x - kx(f, j);
  const dy = y - ky(f, j);
  const had = kc(f, j) > 0;
  setKp(f, j, x, y, c);
  for (const t of FEET[j] || []) if (had && kc(f, t) > 0 && !pinned(f, t)) setKp(f, t, kx(f, t) + dx, ky(f, t) + dy, kc(f, t));
}

/** Re-pick joint j outward from the pin at frame t0 (dir = +1 or -1), stopping before frame `stop`. */
function follow(out, frames, candidates, j, t0, dir, stop, maxSteps, tl) {
  const c = COCO_TO_BODY25.indexOf(j);
  if (c < 0) return;
  let prev = [kx(out[t0], j), ky(out[t0], j)];
  let vel = [0, 0];
  let agree = 0;
  for (let s = 1; s <= maxSteps; s++) {
    const u = t0 + dir * s;
    if (u < 0 || u >= out.length || u === stop) break;
    const own = kc(frames[u], j) > 0 ? [kx(frames[u], j), ky(frames[u], j)] : null;
    const options = [];
    for (const cand of candidates[u] || []) if (cand && Number.isFinite(cand[c]?.[0])) options.push([cand[c][0], cand[c][1]]);
    if (own) options.push(own);
    if (!options.length) break;
    const aim = [prev[0] + vel[0], prev[1] + vel[1]];
    let best = null;
    let bestD = Infinity;
    for (const o of options) {
      const d = Math.hypot(o[0] - aim[0], o[1] - aim[1]);
      if (d < bestD) {
        bestD = d;
        best = o;
      }
    }
    if (bestD > GATE_TL * tl) break;
    if (own && Math.hypot(best[0] - own[0], best[1] - own[1]) < AGREE_TL * tl) {
      if (++agree >= 2) break; // the tracker is back on the joint
    } else {
      agree = 0;
      moveJoint(out[u], j, best[0], best[1], Math.max(FOLLOWED_CONF, Math.min(1, kc(frames[u], j))));
    }
    vel = [0.5 * vel[0] + 0.5 * (best[0] - prev[0]), 0.5 * vel[1] + 0.5 * (best[1] - prev[1])];
    prev = best;
  }
}

/**
 * Frames with the user's pins applied: `pins` is a list of {frame, joint
 * (BODY_25), x, y} in video pixels; `candidates[i]` (optional) the pose models'
 * answers at frame i, each 17 COCO keypoints [x, y, conf] or null.
 */
export function applyPins(frames, pins, { candidates = null, fps = 30 } = {}) {
  const list = normalizePins(pins).filter((p) => p.frame >= 0 && p.frame < frames.length);
  if (!list.length) return frames;
  const out = frames.map((f) => f.slice());
  for (const p of list) moveJoint(out[p.frame], p.joint, p.x, p.y, PIN_CONF);
  if (candidates) {
    const tl = torsoLength(frames) || 100;
    const maxSteps = Math.max(1, Math.round(FOLLOW_MAX_SEC * fps));
    const byJoint = new Map();
    for (const p of list) byJoint.set(p.joint, [...(byJoint.get(p.joint) || []), p.frame]);
    for (const [j, unsorted] of byJoint) {
      const ts = unsorted.sort((a, b) => a - b);
      ts.forEach((t, k) => {
        follow(out, frames, candidates, j, t, 1, k + 1 < ts.length ? ts[k + 1] : -1, maxSteps, tl);
        follow(out, frames, candidates, j, t, -1, k > 0 ? ts[k - 1] : -1, maxSteps, tl);
      });
    }
  }
  // The midpoints follow their joints.
  for (const f of out) {
    if (!f.touched) continue;
    delete f.touched;
    for (const [mid, a, b] of [[KP.Neck, KP.LShoulder, KP.RShoulder], [KP.MidHip, KP.LHip, KP.RHip]]) {
      if (kc(f, a) > 0 && kc(f, b) > 0) setKp(f, mid, (kx(f, a) + kx(f, b)) / 2, (ky(f, a) + ky(f, b)) / 2, Math.min(1, kc(f, a), kc(f, b)));
    }
  }
  return out;
}

/** The pinnable joint nearest to (x, y) in frame f, within `radius` (same units), or -1. */
export function nearestJoint(f, x, y, radius) {
  let best = -1;
  let bestD = radius;
  for (const j of PINNABLE) {
    if (!(kc(f, j) > 0)) continue;
    const d = Math.hypot(kx(f, j) - x, ky(f, j) - y);
    if (d < bestD) {
      bestD = d;
      best = j;
    }
  }
  return best;
}
