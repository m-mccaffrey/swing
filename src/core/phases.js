// Swing phase definitions and heuristic phase detection on a canonical
// sequence. Pro database entries carry hand-checked phase frames; for user
// swings phases are transferred from the matched pro through the DTW
// alignment and then refined locally with the event detectors below.

import { computeSeries } from './metrics.js';
import { argMax, argMin, clamp } from './math.js';

export const PHASES = [
  { key: 'stance', label: 'Stance', blurb: 'Set-up position before any movement.' },
  { key: 'load', label: 'Load', blurb: 'Peak of the gather: leg kick at its highest / hands furthest back.' },
  { key: 'footPlant', label: 'Foot plant', blurb: 'Front foot lands; the swing is about to launch.' },
  { key: 'contact', label: 'Contact', blurb: 'Bat meets the ball (peak hand speed toward the pitcher).' },
  { key: 'extension', label: 'Extension', blurb: 'Arms extend through the ball after contact.' },
  { key: 'finish', label: 'Finish', blurb: 'Balanced follow-through.' },
];

export const PHASE_KEYS = PHASES.map((p) => p.key);

// Beat times of the reference swing (placeholder A at its true frame rate),
// in swing-seconds. Only the spans between them are used.
const REF_BEAT_SEC = { load: 45 / 60, footPlant: 61 / 60, contact: 75 / 60, extension: 80 / 60 };

/**
 * Frames per swing-second implied by a set of beats: their load→contact and
 * foot plant→extension spans against the reference swing. NaN if unusable.
 */
export function swingFpsFromBeats(p) {
  if (!p) return NaN;
  const r = [['load', 'contact'], ['footPlant', 'extension']]
    .map(([a, b]) => (p[b] - p[a]) / (REF_BEAT_SEC[b] - REF_BEAT_SEC[a]))
    .filter((v) => Number.isFinite(v) && v > 0);
  return r.length ? r.reduce((s, v) => s + v, 0) / r.length : NaN;
}

/**
 * The swing clock to use when beats are known (a pro entry, or beats set by
 * hand): the hand-burst clock, unless the beats disagree with it by more than
 * 1.5x. That happens when the pose model loses the hands for most of the
 * swing, and then the reviewed beats are the better witness.
 */
export function pickSwingFps(handFps, beats) {
  const b = swingFpsFromBeats(beats);
  return Number.isFinite(b) && !(Math.abs(Math.log(handFps / b)) <= Math.log(1.5)) ? b : handFps;
}

export function phaseLabel(key) {
  return PHASES.find((p) => p.key === key)?.label ?? key;
}

/** Make sure phase frames are present, in order and inside [0, n). */
export function sanitizePhases(phases, n) {
  const out = {};
  let last = 0;
  for (const key of PHASE_KEYS) {
    let v = Math.round(phases?.[key]);
    if (!Number.isFinite(v)) v = last;
    v = clamp(v, last, n - 1);
    out[key] = v;
    last = v;
  }
  return out;
}

/** Load: peak of the leg kick if there is one, otherwise hands furthest back. */
export function findLoad(series, lo, hi) {
  const lift = argMax(series.frontFootLift, lo, hi + 1);
  if (lift >= 0 && series.frontFootLift[lift] > 0.08) return lift;
  return argMin(series.handsX, lo, hi + 1);
}

/** Foot plant: front foot has finished most of its travel and is back down. */
export function findFootPlant(series, lo, hi, stance, contact) {
  let maxStride = -Infinity;
  for (let i = stance; i <= contact; i++) if (Number.isFinite(series.stride[i])) maxStride = Math.max(maxStride, series.stride[i]);
  if (!(maxStride > 0.12)) return -1;
  for (let i = Math.max(stance, lo); i <= hi; i++) {
    if (series.stride[i] >= 0.9 * maxStride && !(series.frontFootLift[i] > 0.12)) return i;
  }
  return -1;
}

/**
 * Contact: peak horizontal hand speed toward the pitcher. Uses the centroid of
 * the peak (frames above 80% of it, weighted by how far above) rather than the
 * single fastest frame, which keypoint jitter can move by a frame or two.
 */
export function findContact(series, lo, hi) {
  const v = series.handVx;
  const peak = argMax(v, lo, hi + 1);
  if (peak < 0) return peak;
  const thr = 0.8 * v[peak];
  let a = peak;
  let b = peak;
  while (a - 1 >= Math.max(0, lo) && v[a - 1] >= thr) a--;
  while (b + 1 <= Math.min(v.length - 1, hi) && v[b + 1] >= thr) b++;
  let sw = 0;
  let st = 0;
  for (let i = a; i <= b; i++) {
    const w = v[i] - thr;
    sw += w;
    st += w * i;
  }
  return sw > 0 ? Math.round(st / sw) : peak;
}

/**
 * Detect phases from a canonical sequence.
 * @param {number[][]} frames canonical frames
 * @param {number} fps frames per swing-second (see estimateSwingFps)
 * @param {number} stanceIndex chosen stance frame
 */
export function detectPhases(frames, fps, stanceIndex = 0, series = computeSeries(frames, stanceIndex, fps)) {
  const n = frames.length;
  const s = clamp(stanceIndex, 0, n - 1);
  const sec = (t) => Math.max(1, Math.round(t * fps));

  const peak = findContact(series, s + 1, n - 1);
  const contact = clamp(peak < 0 ? n - 1 : peak, s + 3, n - 1);

  let footPlant = findFootPlant(series, s, contact, s, contact);
  if (footPlant < 0) footPlant = contact - sec(0.2);
  footPlant = clamp(footPlant, s + 2, contact - 1);

  const load = clamp(findLoad(series, s, footPlant), s + 1, footPlant - 1);

  // Extension: hands furthest from the neck shortly after contact.
  let extension = argMax(series.handReach, contact, contact + sec(0.2));
  if (extension < 0) extension = contact + sec(0.08);
  extension = clamp(extension, contact + 1, n - 1);

  // Finish: hands slow down after the swing, no later than ~0.6 s after contact.
  const peakSpeed = series.handSpeed[contact] || 1;
  let finish = -1;
  for (let i = extension + sec(0.12); i < Math.min(n, contact + sec(0.6)); i++) {
    if (series.handSpeed[i] < 0.2 * peakSpeed) {
      finish = i;
      break;
    }
  }
  if (finish < 0) finish = Math.min(n - 1, contact + sec(0.45));
  finish = clamp(finish, extension, n - 1);

  return sanitizePhases({ stance: s, load, footPlant, contact, extension, finish }, n);
}

/**
 * Refine phases estimated by alignment: search for the event signature near
 * each estimate, then apply the offset between the pro's labelled frame and
 * the same detector run on the pro, so user and pro phases mean the same thing.
 */
export function refinePhases(user, estimate, pro, windowSec = 0.15) {
  const out = { ...estimate };
  const wu = Math.max(1, Math.round(windowSec * user.swingFps));
  const wp = Math.max(1, Math.round(windowSec * pro.swingFps));
  const finders = {
    contact: (S, lo, hi) => findContact(S, lo, hi),
    footPlant: (S, lo, hi, st, c) => findFootPlant(S, lo, hi, st, c),
    load: (S, lo, hi) => findLoad(S, lo, hi),
  };
  for (const key of ['contact', 'footPlant', 'load']) {
    const find = finders[key];
    const pc = pro.phases.contact;
    const hp = find(pro.series, pro.phases[key] - wp, pro.phases[key] + wp, pro.stanceIndex, pc);
    if (hp < 0) continue;
    const offsetSec = (pro.phases[key] - hp) / pro.swingFps;
    const uc = key === 'contact' ? estimate.contact + wu : out.contact;
    const hu = find(user.series, out[key] - wu, out[key] + wu, user.stanceIndex, uc);
    if (hu < 0) continue;
    out[key] = hu + Math.round(offsetSec * user.swingFps);
  }
  // Keep the later phases after the refined contact.
  const gap = (a, b) => Math.max(1, Math.round((pro.phases[b] - pro.phases[a]) / pro.swingFps * user.swingFps));
  if (out.extension <= out.contact) out.extension = out.contact + gap('contact', 'extension');
  if (out.finish <= out.extension) out.finish = out.extension + gap('extension', 'finish');
  return sanitizePhases(out, user.canon.length);
}
