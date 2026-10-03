// Sequence-level processing: left/right label repair, gap filling, smoothing,
// orientation (which side of the frame the pitcher is on), stance-frame
// suggestion, and canonicalization into a view-independent coordinate frame.
//
// Canonical coordinates:
//   * x points toward the pitcher, y points up. Meters are not known, so the
//     unit is a body-proportional "torso length" (TL): (torso + thigh + shin)
//     at the stance / BODY_TO_TORSO, i.e. about one Neck→MidHip for an adult.
//   * origin is the MidHip at the stance frame.
//   * "L*" BODY_25 joints are the FRONT side (closest to the pitcher), "R*"
//     the BACK side. A left-handed hitter therefore looks like a right-handed
//     one after canonicalization, and a video filmed from the hitter's back is
//     (orthographically) the mirror of one filmed from the chest, which the
//     flip undoes. That is what lets any side view be compared to any other.

import { KP, NUM_KP, LR_GROUPS, PIN_CONF, kx, ky, kc, emptyFrame, pinned, swapLR } from './body25.js';
import { clamp, fillGaps, gaussianSmooth, median, mean, derivative, robustSpline } from './math.js';

const MIN_CONF = 0.25;

function pt(f, j) {
  return f[j * 3 + 2] > MIN_CONF ? [f[j * 3], f[j * 3 + 1]] : null;
}

function midOf(f, a, b) {
  const pa = pt(f, a);
  const pb = pt(f, b);
  if (pa && pb) return [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2];
  return pa || pb;
}

/**
 * Whole-body length (pixels): Neck→MidHip plus the average thigh and shin,
 * divided by BODY_TO_TORSO so the unit still reads as an adult torso length.
 * Using the legs too means kids (relatively shorter legs) and adults scale
 * consistently from head to toe, not just at the torso.
 */
export const BODY_TO_TORSO = 2.75;

export function bodyScale(frames, center, halfWindow = 2) {
  const lo = Math.max(0, center - halfWindow);
  const hi = Math.min(frames.length, center + halfWindow + 1);
  const seg = (f, a, b) => {
    const p = pt(f, a);
    const q = pt(f, b);
    return p && q ? Math.hypot(p[0] - q[0], p[1] - q[1]) : NaN;
  };
  const avg = (a, b) => (Number.isFinite(a) && Number.isFinite(b) ? (a + b) / 2 : Number.isFinite(a) ? a : b);
  const d = [];
  for (let i = lo; i < hi; i++) {
    const f = frames[i];
    const torso = seg(f, KP.Neck, KP.MidHip);
    const thigh = avg(seg(f, KP.LHip, KP.LKnee), seg(f, KP.RHip, KP.RKnee));
    const shin = avg(seg(f, KP.LKnee, KP.LAnkle), seg(f, KP.RKnee, KP.RAnkle));
    if ([torso, thigh, shin].every(Number.isFinite)) d.push((torso + thigh + shin) / BODY_TO_TORSO);
  }
  const m = median(d);
  return Number.isFinite(m) && m > 1e-6 ? m : torsoLength(frames, center, halfWindow);
}

/** Median Neck→MidHip distance (pixels) over a window of frames. */
export function torsoLength(frames, center = null, halfWindow = 3) {
  const lo = center == null ? 0 : Math.max(0, center - halfWindow);
  const hi = center == null ? frames.length : Math.min(frames.length, center + halfWindow + 1);
  const d = [];
  for (let i = lo; i < hi; i++) {
    const n = pt(frames[i], KP.Neck);
    const h = pt(frames[i], KP.MidHip);
    if (n && h) d.push(Math.hypot(n[0] - h[0], n[1] - h[1]));
  }
  const m = median(d);
  if (Number.isFinite(m) && m > 1e-6) return m;
  return center == null ? NaN : torsoLength(frames, null);
}

/** Cost of assigning `f`'s group joints (optionally swapped) to reference frame `ref`. */
function groupCost(f, ref, pairs, swapped) {
  let cost = 0;
  let n = 0;
  for (const [a, b] of pairs) {
    for (const [src, dst] of swapped ? [[a, b], [b, a]] : [[a, a], [b, b]]) {
      const p = pt(f, src);
      const q = pt(ref, dst);
      if (p && q) {
        cost += Math.hypot(p[0] - q[0], p[1] - q[1]);
        n++;
      }
    }
  }
  return n ? cost / n : NaN;
}

/**
 * Fix left/right label flicker (common in side views) by walking outward from
 * a reference frame and, per joint group, keeping whichever labelling is most
 * continuous with the previous (already corrected) frame.
 */
export function fixLeftRightFlicker(frames, refIndex = 0) {
  const out = frames.map((f) => f.slice());
  const walk = (from, to, step) => {
    let prev = out[from];
    for (let i = from + step; step > 0 ? i <= to : i >= to; i += step) {
      let f = out[i];
      for (const pairs of Object.values(LR_GROUPS)) {
        if (pairs.some(([a, b]) => pinned(f, a) || pinned(f, b))) continue; // the user said which is which
        const keep = groupCost(f, prev, pairs, false);
        const swap = groupCost(f, prev, pairs, true);
        if (Number.isFinite(keep) && Number.isFinite(swap) && swap < keep * 0.7) f = swapLR(f, pairs);
      }
      out[i] = f;
      // Carry forward the last good reference for joints missing in this frame.
      const merged = prev.slice();
      for (let j = 0; j < NUM_KP; j++) {
        if (f[j * 3 + 2] > MIN_CONF) {
          merged[j * 3] = f[j * 3];
          merged[j * 3 + 1] = f[j * 3 + 1];
          merged[j * 3 + 2] = f[j * 3 + 2];
        }
      }
      prev = merged;
    }
  };
  if (!out.length) return out;
  const r = clamp(refIndex, 0, out.length - 1);
  walk(r, out.length - 1, 1);
  walk(r, 0, -1);
  return out;
}

/**
 * Fill short gaps and smooth every keypoint track with a robust smoothing
 * spline (math.js robustSpline): confident detections pull the path, doubtful
 * ones barely do, sudden accelerations are penalized, and a detection far off
 * the path (a joint that jumped for a frame or two) is ignored. Filled or
 * rejected points get a low (but non-zero) confidence so later steps can
 * down-weight them and the overlay draws them as estimates.
 */
export const SMOOTH_CUTOFF_HZ = 6; // swing-time Hz
export const OUTLIER_TL = 0.2; // further off the path than this: ignored
const PIN_WEIGHT = 25; // a point placed by hand: the path goes (almost) through it

export function cleanSequence(frames, { fps = 30, maxGapSec = 0.2 } = {}) {
  const n = frames.length;
  const out = frames.map(() => emptyFrame());
  const maxGap = Math.max(1, Math.round(maxGapSec * fps));
  // Cutoff in swing-time Hz, so smoothing is the same at any frame rate.
  const q = fps / (2 * Math.PI * SMOOTH_CUTOFF_HZ);
  const lambda = q * q * q * q;
  const k = OUTLIER_TL * (torsoLength(frames) || 100);
  for (let j = 0; j < NUM_KP; j++) {
    const xs = frames.map((f) => f[j * 3]);
    const ys = frames.map((f) => f[j * 3 + 1]);
    const cs = frames.map((f) => f[j * 3 + 2]);
    const ws = cs.map((c) => (c >= PIN_CONF ? PIN_WEIGHT : c > MIN_CONF ? c : 0));
    const fixed = cs.map((c) => c >= PIN_CONF);
    // Fit each run of detections whose gaps are short enough to bridge.
    let i = 0;
    while (i < n) {
      if (!(ws[i] > 0)) {
        i++;
        continue;
      }
      let last = i;
      for (let m = i + 1; m < n && m - last - 1 <= maxGap; m++) if (ws[m] > 0) last = m;
      const fit = robustSpline([xs.slice(i, last + 1), ys.slice(i, last + 1)], ws.slice(i, last + 1), lambda, k, 3, fixed.slice(i, last + 1));
      for (let m = i; m <= last; m++) {
        const q = m - i;
        out[m][j * 3] = fit.xs[0][q];
        out[m][j * 3 + 1] = fit.xs[1][q];
        out[m][j * 3 + 2] = ws[m] > 0 && fit.keep[q] >= 0.5 ? cs[m] : 0.2;
      }
      i = last + 1;
    }
  }
  return out;
}

/**
 * Both hands hold the bat until well after contact, so the wrists stay within
 * about a hand's width of each other. Pose models often lose the far hand
 * behind the body (low confidence, position guessed) or blur both hands at
 * launch, and a straight-line fill across the fastest part of the swing cuts
 * the corner of the hand path. From the start until shortly after the hands'
 * peak speed (some hitters let go with the top hand in the finish):
 *  - a doubtful wrist is put next to a believable one, at the hands' offset
 *    interpolated from the frames where both were seen together (its own
 *    guess is ignored, even when it lands near the other hand);
 *  - of two confident wrists that disagree, the less believable one (lower
 *    confidence, odd forearm length; if that doesn't decide, the one that
 *    jumped since the last frame) is moved the same way;
 *  - two doubtful wrists that agree with each other are kept as they are;
 *  - two doubtful wrists that disagree are filled in from nearby frames.
 * A wrist counts as doubtful when its confidence is low or its forearm length
 * is far from usual (a confident pose model can still put a hand on the bat).
 * Moved or kept-doubtful points get ESTIMATED_CONF: high enough to be used
 * downstream, low enough to be drawn as estimates.
 */
export const HANDS_APART_TL = 0.45;
const HANDS_NOISE_TL = 0.15;
export const ESTIMATED_CONF = 0.26;
const DOUBTFUL_CONF = 0.2; // below MIN_CONF: treated as missing and filled in
const HAND_RELEASE_SEC = 0.15;
const MISPLACED_MAX_SEC = 0.2;

export function repairHands(frames, fps) {
  const n = frames.length;
  const tl = torsoLength(frames);
  if (!n || !(tl > 0)) return frames;
  // "Apart" means further apart than this hitter's hands usually are (their
  // spacing on the bat as the camera sees it), plus room for keypoint noise.
  const spacing = [];
  for (const f of frames) {
    if (f[KP.LWrist * 3 + 2] > MIN_CONF && f[KP.RWrist * 3 + 2] > MIN_CONF) {
      const d = Math.hypot(f[KP.LWrist * 3] - f[KP.RWrist * 3], f[KP.LWrist * 3 + 1] - f[KP.RWrist * 3 + 1]) / tl;
      if (d <= HANDS_APART_TL) spacing.push(d);
    }
  }
  const usual = median(spacing);
  const D = tl * Math.min(HANDS_APART_TL, (Number.isFinite(usual) ? clamp(usual, 0.05, 0.3) : 0.3) + HANDS_NOISE_TL);
  // First over the whole clip, only to find the hands' peak speed without
  // spikes from a misplaced hand; then for real, up to just after the peak.
  const speed = handSpeedSeries(handsTogether(frames, n - 1, D, fps), fps);
  let peak = 0;
  for (let i = 1; i < n; i++) if ((speed[i] || 0) > (speed[peak] || 0)) peak = i;
  return handsTogether(frames, Math.min(n - 1, peak + Math.round(HAND_RELEASE_SEC * fps)), D, fps);
}

function handsTogether(frames, end, D, fps) {
  const n = frames.length;
  const W = [KP.LWrist, KP.RWrist];
  const E = [KP.LElbow, KP.RElbow];
  const has = (f, j) => f[j * 3 + 2] > 0;
  const good = (f, j) => f[j * 3 + 2] > MIN_CONF;
  const dist = (f, a, b) => Math.hypot(f[a * 3] - f[b * 3], f[a * 3 + 1] - f[b * 3 + 1]);

  // Hands' offset (back wrist minus front wrist) where both were seen
  // together, and forearm lengths: learned from the whole clip, used up to `end`.
  const ox = new Array(n).fill(NaN);
  const oy = new Array(n).fill(NaN);
  const fore = [[], []];
  for (let i = 0; i < n; i++) {
    const f = frames[i];
    if (good(f, W[0]) && good(f, W[1]) && dist(f, W[0], W[1]) <= D) {
      ox[i] = f[W[1] * 3] - f[W[0] * 3];
      oy[i] = f[W[1] * 3 + 1] - f[W[0] * 3 + 1];
    }
    for (const k of [0, 1]) if (good(f, W[k]) && good(f, E[k])) fore[k].push(dist(f, W[k], E[k]));
  }
  const fx = fillGaps(ox);
  const fy = fillGaps(oy);
  const foreLen = fore.map((d) => median(d));
  // Two confident hands apart for longer than a glitch are really apart (a
  // hand off the bat): only short stretches count as a misplaced hand.
  const apart = frames.map((f) => good(f, W[0]) && good(f, W[1]) && dist(f, W[0], W[1]) > D);
  const glitch = new Array(n).fill(false);
  for (let i = 0; i < n; ) {
    let j = i;
    while (j < n && apart[j]) j++;
    if (j > i && j - i <= MISPLACED_MAX_SEC * fps) for (let k = i; k < j; k++) glitch[k] = true;
    i = Math.max(j, i + 1);
  }
  // How believable a wrist is: its confidence, less a penalty for an odd forearm.
  const belief = (f, k) => {
    let b = f[W[k] * 3 + 2];
    if (good(f, E[k]) && foreLen[k] > 0) b -= 0.5 * Math.min(1, Math.abs(dist(f, W[k], E[k]) / foreLen[k] - 1));
    return b;
  };

  const out = frames.map((f) => f.slice());
  for (let i = 0; i <= end; i++) {
    const f = out[i];
    const offX = Number.isFinite(fx[i]) ? fx[i] : 0;
    const offY = Number.isFinite(fy[i]) ? fy[i] : 0;
    // Put wrist k next to the other one (never a wrist placed by hand).
    const place = (k) => {
      if (pinned(f, W[k])) return;
      const a = W[1 - k];
      const sign = k === 1 ? 1 : -1;
      f[W[k] * 3] = f[a * 3] + sign * offX;
      f[W[k] * 3 + 1] = f[a * 3 + 1] + sign * offY;
      f[W[k] * 3 + 2] = ESTIMATED_CONF;
    };
    // A confident wrist with an impossible forearm is not trusted.
    const g0 = belief(f, 0) > MIN_CONF;
    const g1 = belief(f, 1) > MIN_CONF;
    const together = has(f, W[0]) && has(f, W[1]) && dist(f, W[0], W[1]) <= D;
    if (pinned(f, W[0]) || pinned(f, W[1])) {
      // A hand placed by hand is right, wherever the other one is.
      if (!(g0 && g1)) place(g0 ? 1 : 0);
    } else if (g0 && g1) {
      if (!together && (glitch[i] || !apart[i])) {
        const b0 = belief(f, 0);
        const b1 = belief(f, 1);
        if (Math.abs(b0 - b1) >= 0.15) place(b0 < b1 ? 0 : 1);
        else if (i > 0) {
          // Equally sure: the one that jumped away from where the hands just were is wrong.
          const prev = out[i - 1];
          const jump = (k) => Math.hypot(f[W[k] * 3] - prev[W[k] * 3], f[W[k] * 3 + 1] - prev[W[k] * 3 + 1]);
          const [j0, j1] = [jump(0), jump(1)];
          if (Math.max(j0, j1) > D && Math.max(j0, j1) > 2 * Math.min(j0, j1)) place(j0 > j1 ? 0 : 1);
        }
      }
    } else if (g0 || g1) {
      // Even a guess that lands near the other hand is a guess: use the believable hand.
      place(g0 ? 1 : 0);
    } else if (together) {
      for (const j of W) f[j * 3 + 2] = Math.max(f[j * 3 + 2], ESTIMATED_CONF);
    } else {
      // Neither is believable: leave both to be filled in from nearby frames.
      for (const j of W) f[j * 3 + 2] = Math.min(f[j * 3 + 2], DOUBTFUL_CONF);
    }
  }
  return out;
}

/** Wrist-midpoint speed series in torso-lengths per second (image space). */
export function handSpeedSeries(frames, fps) {
  const tl = torsoLength(frames) || 1;
  const hx = frames.map((f) => midOf(f, KP.LWrist, KP.RWrist)?.[0] ?? NaN);
  const hy = frames.map((f) => midOf(f, KP.LWrist, KP.RWrist)?.[1] ?? NaN);
  const sx = gaussianSmooth(fillGaps(hx), fps / 60);
  const sy = gaussianSmooth(fillGaps(hy), fps / 60);
  const dx = derivative(sx);
  const dy = derivative(sy);
  return dx.map((v, i) => (Math.hypot(v, dy[i]) * fps) / tl);
}

/** Whole-body motion energy (TL/s): mean speed of wrists, ankles, nose, hips. */
export function motionEnergy(frames, fps) {
  const tl = torsoLength(frames) || 1;
  const joints = [KP.LWrist, KP.RWrist, KP.LAnkle, KP.RAnkle, KP.Nose, KP.MidHip, KP.LKnee, KP.RKnee];
  const per = joints.map((j) => {
    const xs = gaussianSmooth(fillGaps(frames.map((f) => (kc(f, j) > MIN_CONF ? kx(f, j) : NaN))), fps / 30);
    const ys = gaussianSmooth(fillGaps(frames.map((f) => (kc(f, j) > MIN_CONF ? ky(f, j) : NaN))), fps / 30);
    const dx = derivative(xs);
    const dy = derivative(ys);
    return dx.map((v, i) => (Math.hypot(v, dy[i]) * fps) / tl);
  });
  return frames.map((_, i) => mean(per.map((s) => s[i]).filter(Number.isFinite)));
}

/**
 * Length (seconds) of the "hand burst" of a typical swing: the stretch around
 * the fastest hand movement where the hands move faster than a quarter of
 * their peak speed (roughly launch to finish). It defines the app's unit of
 * time, so this is a convention, calibrated so the reference swing runs at
 * its true frame rate.
 */
export const SWING_BURST_SEC = 0.586;

function burstOf(frames, swingFps) {
  const s = handSpeedSeries(frames, swingFps);
  let p = 0;
  for (let i = 1; i < s.length; i++) if ((s[i] || 0) > (s[p] || 0)) p = i;
  const thr = 0.25 * (s[p] || 0);
  if (!(thr > 0)) return null;
  let a = p;
  let b = p;
  while (a - 1 >= 0 && s[a - 1] > thr) a--;
  while (b + 1 < s.length && s[b + 1] > thr) b++;
  // Sub-frame edges by linear interpolation of the threshold crossings.
  const fa = a > 0 ? (s[a] - thr) / (s[a] - s[a - 1]) : 0;
  const fb = b < s.length - 1 ? (s[b] - thr) / (s[b] - s[b + 1]) : 0;
  return { length: b - a + fa + fb, a, b };
}

function handTravel(frames, a, b, halfWindow) {
  const tl = torsoLength(frames) || 1;
  const avg = (c) => {
    let x = 0;
    let y = 0;
    let n = 0;
    for (let i = Math.max(0, c - halfWindow); i <= Math.min(frames.length - 1, c + halfWindow); i++) {
      const h = midOf(frames[i], KP.LWrist, KP.RWrist);
      if (h) {
        x += h[0];
        y += h[1];
        n++;
      }
    }
    return n ? [x / n, y / n] : null;
  };
  const pa = avg(a);
  const pb = avg(b);
  return pa && pb ? Math.hypot(pb[0] - pa[0], pb[1] - pa[1]) / tl : 0;
}

/**
 * The swing's own clock: frames per "swing second", measured from how long
 * the hand burst lasts. A 4x slow-motion replay, a 240 fps clip and a slower
 * youth swing all come out on the same time scale, so nobody has to enter a
 * slow-motion factor and swings are compared tempo-free. Every time constant
 * in the analysis (smoothing, windows, alignment rate) uses this unit.
 *
 * Several starting smoothing scales are tried and the self-consistent one
 * whose burst moves the hands furthest wins (a keypoint-noise spike can be
 * self-consistent too, but it doesn't go anywhere).
 */
export function estimateSwingFps(frames, fps) {
  let best = null;
  for (const k of [0.5, 1, 2, 4, 8, 16]) {
    let est = fps * k;
    let prev = est;
    let burst = null;
    let hands = frames;
    for (let it = 0; it < 3; it++) {
      // Lost or misplaced hands would distort the burst: time the repaired ones.
      hands = repairHands(frames, est);
      burst = burstOf(hands, est);
      if (!burst || !(burst.length > 0)) break;
      prev = est;
      est = burst.length / SWING_BURST_SEC;
    }
    if (!burst || !(burst.length > 0)) continue;
    const ok = Math.abs(Math.log(est / prev)) < 0.25;
    const travel = handTravel(hands, burst.a, burst.b, Math.max(1, Math.round(est * 0.02)));
    if (!best || (ok && (!best.ok || travel > best.travel))) best = { est, ok, travel };
  }
  // A clip is never faster than real time: allow a quick swing, no less.
  return best && Number.isFinite(best.est) ? clamp(best.est, Math.max(5, 0.7 * fps), 5000) : fps;
}

/**
 * Suggest the stance (set-up) frame: the end of the last quiet period before
 * the fastest hand movement in the clip (the swing).
 */
export function suggestStanceFrame(frames, fps) {
  const n = frames.length;
  if (n < 3) return 0;
  const speed = handSpeedSeries(frames, fps);
  let peak = 0;
  for (let i = 1; i < n; i++) if ((speed[i] || 0) > (speed[peak] || 0)) peak = i;
  const energy = motionEnergy(frames, fps);
  const sorted = energy.filter(Number.isFinite).sort((a, b) => a - b);
  const p20 = sorted[Math.floor(sorted.length * 0.2)] ?? 0;
  const thr = Math.max(0.3, p20 * 1.8);
  const minRun = Math.max(2, Math.round(0.12 * fps));
  let run = 0;
  // Walk backward from shortly before the swing looking for a quiet run.
  for (let i = peak - Math.round(0.1 * fps); i >= 0; i--) {
    if (energy[i] < thr) {
      run++;
      if (run >= minRun) {
        // i is the start of the run; the stance is the quiet frame right before the load.
        let end = i + run - 1;
        while (end + 1 < peak && energy[end + 1] < thr) end++;
        // Back off slightly from the first hint of the load.
        return Math.max(i, end - Math.round(0.08 * fps));
      }
    } else {
      run = 0;
    }
  }
  return clamp(peak - Math.round(1.0 * fps), 0, n - 1);
}

/**
 * Guess whether the pitcher is to the right or left of the frame using
 * independent cues: head turned toward the pitcher, hands held back away from
 * the pitcher, stride direction, and the direction the hands travel through
 * the swing. Returns { side, confidence, votes }.
 */
export function detectPitcherSide(frames, stanceIndex, fps) {
  const n = frames.length;
  const s = clamp(stanceIndex, 0, n - 1);
  const tl = torsoLength(frames, s) || torsoLength(frames) || 1;
  const win = Math.max(1, Math.round(0.15 * fps));
  const votes = [];

  // 1. Head turn: the nose sits on the pitcher's side of the ears.
  const head = [];
  for (let i = Math.max(0, s - win); i <= Math.min(n - 1, s + win); i++) {
    const nose = pt(frames[i], KP.Nose);
    const ears = midOf(frames[i], KP.LEar, KP.REar);
    if (nose && ears) head.push(nose[0] - ears[0]);
  }
  if (head.length) votes.push({ cue: 'head turn', value: clamp(median(head) / (0.15 * tl), -1, 1), weight: 1 });

  // 2. Hands are held back, away from the pitcher, at the stance.
  const hands = midOf(frames[s], KP.LWrist, KP.RWrist);
  const hip = pt(frames[s], KP.MidHip);
  if (hands && hip) votes.push({ cue: 'hands held back', value: clamp((hip[0] - hands[0]) / (0.3 * tl), -1, 1), weight: 1 });

  // 3. Stride: the ankle that travels farthest moves toward the pitcher.
  let bestDx = 0;
  for (const j of [KP.LAnkle, KP.RAnkle]) {
    const a0 = pt(frames[s], j);
    if (!a0) continue;
    for (let i = s; i < n; i++) {
      const a = pt(frames[i], j);
      if (a && Math.abs(a[0] - a0[0]) > Math.abs(bestDx)) bestDx = a[0] - a0[0];
    }
  }
  if (bestDx) votes.push({ cue: 'stride direction', value: clamp(bestDx / (0.6 * tl), -1, 1), weight: 1.5 });

  // 4. Swing: the hands' biggest excursion from the stance is toward the pitcher.
  if (hands) {
    let best = 0;
    for (let i = s; i < n; i++) {
      const h = midOf(frames[i], KP.LWrist, KP.RWrist);
      if (h && Math.abs(h[0] - hands[0]) > Math.abs(best)) best = h[0] - hands[0];
    }
    if (best) votes.push({ cue: 'hand path', value: clamp(best / (0.8 * tl), -1, 1), weight: 1.5 });
  }

  const wsum = votes.reduce((a, v) => a + v.weight, 0) || 1;
  const score = votes.reduce((a, v) => a + v.value * v.weight, 0) / wsum;
  return { side: score >= 0 ? 'right' : 'left', confidence: Math.min(1, Math.abs(score)), votes };
}

function xOf(f, j) {
  return f[j * 3 + 2] > MIN_CONF ? f[j * 3] : NaN;
}

const MAX_ROLL = (12 * Math.PI) / 180;
const MIN_ROLL = (1 * Math.PI) / 180;

/**
 * Camera roll (radians, image coordinates) from the ground line: at the
 * stance both feet are planted on level ground, so the line through the big
 * toes (or the ankles if the toes aren't found) should be horizontal. Returns
 * 0 when the feet are too close together, the estimate is tiny, or it is
 * implausibly large (a raised foot rather than a tilted camera).
 */
export function estimateRoll(frames, stanceIndex, fps = 30) {
  const n = frames.length;
  const s = clamp(stanceIndex, 0, n - 1);
  const tl = torsoLength(frames, s);
  if (!Number.isFinite(tl)) return 0;
  // The feet don't move during the stance: use ±150 ms so keypoint noise averages out.
  const hw = Math.max(2, Math.round(0.15 * fps));
  const lineAngles = (a, b) => {
    const out = [];
    for (let i = Math.max(0, s - hw); i <= Math.min(n - 1, s + hw); i++) {
      const p = pt(frames[i], a);
      const q = pt(frames[i], b);
      if (!p || !q) continue;
      const [l, r] = p[0] <= q[0] ? [p, q] : [q, p];
      if (r[0] - l[0] < 0.5 * tl) continue;
      out.push(Math.atan2(r[1] - l[1], r[0] - l[0]));
    }
    return out;
  };
  let angles = lineAngles(KP.LBigToe, KP.RBigToe);
  if (angles.length < 3) angles = lineAngles(KP.LAnkle, KP.RAnkle);
  const m = median(angles);
  if (!Number.isFinite(m) || Math.abs(m) < MIN_ROLL || Math.abs(m) > MAX_ROLL) return 0;
  return m;
}

function rotateFrames(frames, angle, cx, cy) {
  const c = Math.cos(angle);
  const sn = Math.sin(angle);
  return frames.map((f) => {
    const g = f.slice();
    for (let j = 0; j < NUM_KP; j++) {
      if (g[j * 3 + 2] > 0) {
        const dx = f[j * 3] - cx;
        const dy = f[j * 3 + 1] - cy;
        g[j * 3] = cx + dx * c - dy * sn;
        g[j * 3 + 1] = cy + dx * sn + dy * c;
      }
    }
    return g;
  });
}

/**
 * Canonicalize a raw pixel-space sequence. Returns
 * { frames, transform } where frames are in canonical TL units (see top of
 * file) and transform maps canonical points back into the source image.
 */
export function canonicalize(rawFrames, { pitcherSide = 'right', stanceIndex = 0, fps = 30, clean = true } = {}) {
  const n = rawFrames.length;
  if (!n) throw new Error('No frames to analyze');
  const s = clamp(stanceIndex, 0, n - 1);
  const sx = pitcherSide === 'left' ? -1 : 1;

  // 1. Flip so the pitcher is toward +x (still pixel units, y down).
  let frames = rawFrames.map((f) => {
    const g = f.slice();
    for (let j = 0; j < NUM_KP; j++) g[j * 3] = sx * f[j * 3];
    return g;
  });

  // 1b. Level a tilted camera using the ground line at the stance.
  const roll = estimateRoll(frames, s, fps);
  let cx = 0;
  let cy = 0;
  if (roll) {
    const win0 = frames.slice(Math.max(0, s - 2), Math.min(n, s + 3));
    cx = median(win0.map((f) => xOf(f, KP.MidHip)));
    cy = median(win0.map((f) => (kc(f, KP.MidHip) > MIN_CONF ? ky(f, KP.MidHip) : NaN)));
    if (!Number.isFinite(cx) || !Number.isFinite(cy)) {
      cx = 0;
      cy = 0;
    }
    frames = rotateFrames(frames, -roll, cx, cy);
  }

  // 2. Decide front/back labels per joint group from geometry at the stance:
  //    the front side is the one closer to the pitcher (larger x).
  const win = [];
  for (let i = Math.max(0, s - 2); i <= Math.min(n - 1, s + 2); i++) win.push(frames[i]);
  const sideScore = (pairs) => {
    let score = 0;
    for (const [r, l] of pairs) {
      const d = median(win.map((f) => xOf(f, l) - xOf(f, r)));
      if (Number.isFinite(d)) score += d;
    }
    return score;
  };
  const armScore = sideScore([[KP.RShoulder, KP.LShoulder], [KP.RElbow, KP.LElbow]]);
  const legScore = sideScore([[KP.RHip, KP.LHip], [KP.RKnee, KP.LKnee], [KP.RAnkle, KP.LAnkle]]);
  const swapArms = armScore < 0;
  const swapLegs = legScore < 0;
  if (swapArms || swapLegs) {
    frames = frames.map((f) => {
      let g = f;
      if (swapArms) g = swapLR(g, [...LR_GROUPS.arms, ...LR_GROUPS.head]);
      if (swapLegs) g = swapLR(g, LR_GROUPS.legs);
      return g;
    });
  }

  // 3. Repair frame-to-frame flicker and lost hands, then fill gaps and smooth.
  frames = fixLeftRightFlicker(frames, s);
  if (clean) frames = cleanSequence(repairHands(frames, fps), { fps });

  // 4. Normalize: origin at stance MidHip, unit = stance torso length, y up.
  const scale = bodyScale(frames, s, 2);
  if (!Number.isFinite(scale)) throw new Error('Could not find the hitter’s torso (neck and hips) in the stance frame');
  const originFrame = frames[s];
  let ox = kc(originFrame, KP.MidHip) > 0 ? kx(originFrame, KP.MidHip) : NaN;
  let oy = kc(originFrame, KP.MidHip) > 0 ? ky(originFrame, KP.MidHip) : NaN;
  if (!Number.isFinite(ox)) {
    ox = median(frames.map((f) => xOf(f, KP.MidHip)));
    oy = median(frames.map((f) => (kc(f, KP.MidHip) > MIN_CONF ? ky(f, KP.MidHip) : NaN)));
  }
  const canon = frames.map((f) => {
    const g = emptyFrame();
    for (let j = 0; j < NUM_KP; j++) {
      const c = f[j * 3 + 2];
      if (c > 0) {
        g[j * 3] = (f[j * 3] - ox) / scale;
        g[j * 3 + 1] = -(f[j * 3 + 1] - oy) / scale;
        g[j * 3 + 2] = c;
      }
    }
    return g;
  });

  return {
    frames: canon,
    // ox/oy are in flipped, leveled pixel space; roll/cx/cy undo the leveling.
    transform: { sx, ox, oy, scale, roll, cx, cy, swapArms, swapLegs, stanceIndex: s, pitcherSide },
  };
}

/** Map a canonical point back to source-image pixel coordinates. */
export function canonToImage(t, x, y) {
  let px = t.ox + x * t.scale;
  let py = t.oy - y * t.scale;
  if (t.roll) {
    const dx = px - t.cx;
    const dy = py - t.cy;
    const c = Math.cos(t.roll);
    const s = Math.sin(t.roll);
    px = t.cx + dx * c - dy * s;
    py = t.cy + dx * s + dy * c;
  }
  return [t.sx * px, py];
}

/**
 * Linearly resample frames from srcFps to dstFps. Returns { frames, srcIndex }
 * where srcIndex[k] is the (fractional) source index of output frame k.
 */
export function resampleFrames(frames, srcFps, dstFps, startIndex = 0, endIndex = frames.length - 1) {
  const out = [];
  const srcIndex = [];
  const dur = (endIndex - startIndex) / srcFps;
  const count = Math.max(1, Math.floor(dur * dstFps + 1e-6) + 1);
  for (let k = 0; k < count; k++) {
    const si = startIndex + (k / dstFps) * srcFps;
    const i0 = Math.floor(si);
    const i1 = Math.min(endIndex, i0 + 1);
    const t = si - i0;
    const a = frames[i0];
    const b = frames[i1];
    const g = emptyFrame();
    for (let j = 0; j < NUM_KP; j++) {
      const ca = a[j * 3 + 2];
      const cb = b[j * 3 + 2];
      if (ca > 0 && cb > 0) {
        g[j * 3] = a[j * 3] + (b[j * 3] - a[j * 3]) * t;
        g[j * 3 + 1] = a[j * 3 + 1] + (b[j * 3 + 1] - a[j * 3 + 1]) * t;
        g[j * 3 + 2] = Math.min(ca, cb);
      } else if (ca > 0 || cb > 0) {
        const src = t < 0.5 && ca > 0 ? a : cb > 0 ? b : a;
        g[j * 3] = src[j * 3];
        g[j * 3 + 1] = src[j * 3 + 1];
        g[j * 3 + 2] = src[j * 3 + 2] * 0.5;
      }
    }
    out.push(g);
    srcIndex.push(si);
  }
  return { frames: out, srcIndex };
}

/** Fraction of frames where the core body (neck, hips, a wrist, an ankle) was found. */
export function detectionCoverage(frames) {
  if (!frames.length) return 0;
  let ok = 0;
  for (const f of frames) {
    if (pt(f, KP.Neck) && pt(f, KP.MidHip) && (pt(f, KP.LWrist) || pt(f, KP.RWrist)) && (pt(f, KP.LAnkle) || pt(f, KP.RAnkle))) ok++;
  }
  return ok / frames.length;
}
