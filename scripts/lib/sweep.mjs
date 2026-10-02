// Normalization sweep harness: render the same synthetic swing under many
// recording conditions (camera, framing, frame rate, slow motion, body size,
// noise...) and run it through the full user pipeline against the reference
// pro. A perfect normalization reports "matches" on every check.
//
// Used by tests/normalization.test.mjs and scripts/normalization-sweep.mjs.

import { makeSpec, renderSwing, archetypeEntry, ARCHETYPES } from '../../src/core/synth.js';
import { prepareSwing, compareSwing } from '../../src/core/compare.js';
import { suggestStanceFrame, detectPitcherSide, estimateSwingFps } from '../../src/core/sequence.js';
import { rankStances, retarget, boneLengths } from '../../src/core/match.js';
import { evaluateFeedback } from '../../src/core/feedback.js';
import { prepareEntry } from '../../src/core/db.js';
import { PHASE_KEYS } from '../../src/core/phases.js';
import { KP, NUM_KP, swapLR, LR_GROUPS } from '../../src/core/body25.js';
import { rng } from '../../src/core/math.js';

const BASE_SPEC = ARCHETYPES['synthetic-a'].spec;
/** Reference recording: 60 fps real time, 720p, orthographic side view. */
export const REF_RENDER = { fps: 60, preRoll: 0.8, postRoll: 0.5, seed: 3 };

let proCache = null;
/** The reference pro: placeholder A exactly as stored in the database. */
export function referencePro() {
  if (!proCache) proCache = prepareEntry(archetypeEntry('synthetic-a'));
  return proCache;
}

// ------------------------------------------------------------------ 2D post-processing

export function transformFrames(frames, fn) {
  return frames.map((f) => {
    const g = f.slice();
    for (let j = 0; j < NUM_KP; j++) {
      if (g[j * 3 + 2] > 0) {
        const [x, y] = fn(g[j * 3], g[j * 3 + 1]);
        g[j * 3] = x;
        g[j * 3 + 1] = y;
      }
    }
    return g;
  });
}

/** Change limb proportions (2D bone lengths) while keeping every joint angle. */
export function scaleLimbs(frames, { legs = 1, arms = 1, torso = 1 }) {
  const L = [KP.LKnee, KP.RKnee, KP.LAnkle, KP.RAnkle];
  const A = [KP.LElbow, KP.RElbow, KP.LWrist, KP.RWrist];
  return frames.map((f) => {
    const b = boneLengths(f);
    for (const j of L) if (b[j]) b[j] *= legs;
    for (const j of A) if (b[j]) b[j] *= arms;
    if (b[KP.Neck]) b[KP.Neck] *= torso;
    return retarget(f, b);
  });
}

/** Random keypoint dropouts (confidence 0) on a fraction of joints. */
export function dropout(frames, p, seed = 9) {
  const r = rng(seed);
  return frames.map((f) => {
    const g = f.slice();
    for (let j = 0; j < NUM_KP; j++) if (r() < p) g[j * 3 + 2] = 0;
    return g;
  });
}

/** Swap left/right legs (or arms) on a fraction of frames, like pose-model flicker. */
export function flicker(frames, p, seed = 13) {
  const r = rng(seed);
  return frames.map((f) => (r() < p ? swapLR(f, r() < 0.5 ? LR_GROUPS.legs : LR_GROUPS.arms) : f));
}

// ------------------------------------------------------------------ pose-model hand failures

const torsoPx = (f) => Math.hypot(f[KP.Neck * 3] - f[KP.MidHip * 3], f[KP.Neck * 3 + 1] - f[KP.MidHip * 3 + 1]) || 100;
const gaussR = (r) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());

/**
 * The far hand lost behind the body from the load until just after contact:
 * low confidence and a wandering guess about `offTL` torso lengths off.
 * `r` is the renderSwing() result (for the true phases).
 */
export function hiddenHand(frames, r, { joint = KP.RWrist, conf = 0.1, offTL = 0.8, seed = 5 } = {}) {
  const rand = rng(seed);
  let ang = rand() * 2 * Math.PI;
  const a = r.phases.load;
  const b = Math.min(frames.length - 1, r.phases.contact + 3);
  return frames.map((f, i) => {
    if (i < a || i > b) return f;
    const g = f.slice();
    ang += (rand() - 0.5) * 0.6;
    g[joint * 3] += Math.cos(ang) * offTL * torsoPx(f);
    g[joint * 3 + 1] += Math.sin(ang) * offTL * torsoPx(f);
    g[joint * 3 + 2] = conf;
    return g;
  });
}

/** Both hands motion-blurred from foot plant to extension: low confidence, positions roughly right. */
export function blurredHands(frames, r, { conf = 0.15, jitterTL = 0.04, seed = 6 } = {}) {
  const rand = rng(seed);
  return frames.map((f, i) => {
    if (i < r.phases.footPlant || i > r.phases.extension) return f;
    const g = f.slice();
    for (const j of [KP.LWrist, KP.RWrist]) {
      g[j * 3] += gaussR(rand) * jitterTL * torsoPx(f);
      g[j * 3 + 1] += gaussR(rand) * jitterTL * torsoPx(f);
      g[j * 3 + 2] = conf;
    }
    return g;
  });
}

/** One hand confidently in the wrong place around contact (snapped to the bat or the other arm). */
export function wrongHand(frames, r, { joint = KP.LWrist, conf = 0.55, offTL = 0.7 } = {}) {
  return frames.map((f, i) => {
    if (i < r.phases.contact - 3 || i > r.phases.contact + 1) return f;
    const g = f.slice();
    g[joint * 3] -= offTL * torsoPx(f);
    g[joint * 3 + 1] += 0.3 * torsoPx(f);
    g[joint * 3 + 2] = conf;
    return g;
  });
}

/** Confident single-frame jumps of random joints between load and extension (pose-model glitches). */
export function glitches(frames, r, { count = 4, offTL = 0.5, seed = 7 } = {}) {
  const rand = rng(seed);
  const joints = [KP.LWrist, KP.RWrist, KP.LElbow, KP.RElbow, KP.LKnee, KP.LAnkle, KP.Nose, KP.Neck];
  const out = frames.map((f) => f.slice());
  for (let q = 0; q < count; q++) {
    const j = joints[Math.floor(rand() * joints.length)];
    const i = r.phases.load + Math.floor(rand() * (r.phases.extension - r.phases.load));
    const ang = rand() * 2 * Math.PI;
    out[i][j * 3] += Math.cos(ang) * offTL * torsoPx(frames[i]);
    out[i][j * 3 + 1] += Math.sin(ang) * offTL * torsoPx(frames[i]);
  }
  return out;
}

// ------------------------------------------------------------------ cases

/**
 * Render a user swing for a condition.
 * @param {object} c
 * @param {object} [c.spec] makeSpec overrides on top of placeholder A
 * @param {object} [c.render] renderSwing options on top of REF_RENDER
 * @param {(frames:number[][], meta:object)=>number[][]} [c.post] 2D post-processing
 */
export function renderCase(c = {}) {
  const spec = makeSpec({ ...BASE_SPEC, ...(c.spec || {}), tracks: { ...(BASE_SPEC.tracks || {}), ...(c.spec?.tracks || {}) } });
  const render = { ...REF_RENDER, ...(c.render || {}), camera: { ...(c.render?.camera || {}) } };
  const r = renderSwing(spec, render);
  const frames = c.post ? c.post(r.frames, r) : r.frames;
  return {
    frames,
    fps: r.fps,
    trueSpeed: r.speedFactor, // the app is never told this
    width: r.width,
    height: r.height,
    truePhases: r.phases,
    trueSide: r.pitcherSide,
    spec,
  };
}

/**
 * Run a rendered case through the user pipeline (auto stance and pitcher
 * side unless given) against the pro, and score it against ground truth.
 */
export function evaluateCase(cs, pro = referencePro(), { stanceShiftSec = 0 } = {}) {
  // Like the app: only the video frame rate is known; the swing sets its own clock.
  const swingFps = estimateSwingFps(cs.frames, cs.fps);
  let stance = suggestStanceFrame(cs.frames, swingFps);
  if (stanceShiftSec) stance = Math.max(0, Math.round(stance + stanceShiftSec * cs.fps * cs.trueSpeed));
  const side = detectPitcherSide(cs.frames, stance, swingFps);
  const user = prepareSwing({ frames: cs.frames, fps: cs.fps, swingFps, stanceIndex: stance, pitcherSide: side.side });
  const [rank] = rankStances(user.stancePose, [pro]);
  const cmp = compareSwing(user, pro.prep);
  const items = evaluateFeedback({ ...user, phases: cmp.phases }, pro.prep, { proName: 'Pro' });
  // Phase timing error in real milliseconds (true frame spacing).
  const msPerFrame = 1000 / (cs.fps * cs.trueSpeed);
  const phaseErrMs = {};
  for (const k of PHASE_KEYS.filter((k) => k !== 'stance')) phaseErrMs[k] = Math.abs(cmp.phases[k] - cs.truePhases[k]) * msPerFrame;
  const worst = items.reduce((a, b) => (b.score > a.score ? b : a), items[0]);
  return {
    swingFps,
    // Swing clock vs. the true real-time rate (tempo changes show up here on purpose).
    clockRatio: swingFps / (cs.fps * cs.trueSpeed),
    sideCorrect: side.side === cs.trueSide,
    sideConfidence: side.confidence,
    stance,
    stanceSimilarity: rank.similarity,
    swingScore: cmp.swingScore,
    segments: Object.fromEntries(cmp.segments.map((s) => [s.key, s.score])),
    phases: cmp.phases,
    phaseErrMs,
    maxPhaseErrMs: Math.max(...Object.values(phaseErrMs)),
    msPerFrame,
    items,
    byId: Object.fromEntries(items.map((i) => [i.id, i])),
    notGood: items.filter((i) => i.severity !== 'good').map((i) => i.id),
    worst: { id: worst.id, ratio: worst.score },
  };
}

/** A pro recorded under a condition, prepared with its true phases. */
export function proFromCase(c) {
  const cs = renderCase(c);
  const prep = prepareSwing({ frames: cs.frames, fps: cs.fps, stanceIndex: cs.truePhases.stance, pitcherSide: cs.trueSide, phases: cs.truePhases });
  return { id: 'pro-variant', name: 'Pro (variant)', prep, stancePose: prep.stancePose, meta: {} };
}

/** Systematic keypoint offsets like a different pose model (e.g. OpenPose vs MediaPipe). */
export function poseModelBias(frames) {
  return frames.map((f) => {
    const g = f.slice();
    const tl = Math.hypot(f[KP.Neck * 3] - f[KP.MidHip * 3], f[KP.Neck * 3 + 1] - f[KP.MidHip * 3 + 1]) || 100;
    g[KP.Neck * 3 + 1] -= 0.06 * tl; // neck sits higher
    for (const [h, sgn] of [[KP.LHip, 1], [KP.RHip, -1]]) {
      g[h * 3 + 1] -= 0.05 * tl; // hips higher...
      g[h * 3] += sgn * 0.04 * tl * Math.sign(f[KP.LHip * 3] - f[KP.RHip * 3] || 1); // ...and wider
    }
    g[KP.MidHip * 3 + 1] -= 0.05 * tl;
    for (const j of [KP.Nose, KP.LEye, KP.REye]) g[j * 3 + 1] += 0.02 * tl;
    return g;
  });
}

// ------------------------------------------------------------------ the sweep

const ref = { camera: {} };
const cam = (camera, extra = {}) => ({ render: { ...extra, camera: { ...ref.camera, ...camera } } });

/**
 * Conditions grouped by what the app should do with them:
 *   invariant  — must not change any result (normalization's job)
 *   robust     — nuisances that can't be fully removed; results should degrade gracefully
 *   sensitive  — real swing differences; the named checks must react (and others stay quiet)
 *   limit      — known limits of a single 2D side camera, kept to document how results degrade
 */
export const CONDITIONS = [
  // ---- space: framing and camera placement
  { group: 'space', kind: 'invariant', name: 'reference (same recording as the pro)', case: {} },
  { group: 'space', kind: 'invariant', name: 'hitter off-center (+400 px, -120 px)', case: cam({ u0: 1040, vGround: 545 }) },
  { group: 'space', kind: 'invariant', name: 'camera twice as far (half size)', case: cam({ pxPerM: 145, vGround: 600 }) },
  { group: 'space', kind: 'invariant', name: 'camera closer (1.6x size)', case: cam({ pxPerM: 465, vGround: 900, height: 1000 }) },
  { group: 'space', kind: 'invariant', name: '4K resolution', case: cam({ width: 3840, height: 2160, pxPerM: 870, u0: 1920, vGround: 1995 }) },
  { group: 'space', kind: 'invariant', name: '480p resolution', case: cam({ width: 854, height: 480, pxPerM: 193, u0: 427, vGround: 443 }) },
  { group: 'space', kind: 'invariant', name: 'portrait phone video (1080x1920)', case: cam({ width: 1080, height: 1920, pxPerM: 560, u0: 470, vGround: 1560 }) },
  { group: 'space', kind: 'invariant', name: 'filmed from behind (mirror view)', case: cam({ pitcherSide: 'left' }) },
  { group: 'space', kind: 'invariant', name: 'left-handed hitter, chest view', case: { spec: { bats: 'L' } } },
  { group: 'space', kind: 'invariant', name: 'left-handed hitter, from behind', case: { spec: { bats: 'L' }, ...cam({ pitcherSide: 'right' }) } },
  // ---- space: body
  { group: 'body', kind: 'invariant', name: 'shorter hitter (1.50 m, same proportions)', case: { spec: { heightM: 1.5 } } },
  { group: 'body', kind: 'invariant', name: 'taller hitter (2.00 m)', case: { spec: { heightM: 2.0 } } },
  { group: 'body', kind: 'invariant', name: 'child proportions (legs -20%, arms -10%)', case: { post: (f) => scaleLimbs(f, { legs: 0.8, arms: 0.9 }) } },
  { group: 'body', kind: 'invariant', name: 'long legs (+15%)', case: { post: (f) => scaleLimbs(f, { legs: 1.15 }) } },
  // ---- time
  { group: 'time', kind: 'invariant', name: '30 fps', case: { render: { fps: 30 } } },
  { group: 'time', kind: 'invariant', name: '24 fps', case: { render: { fps: 24 } } },
  { group: 'time', kind: 'invariant', name: '120 fps', case: { render: { fps: 120 } } },
  { group: 'time', kind: 'invariant', name: '240 fps', case: { render: { fps: 240 } } },
  { group: 'time', kind: 'invariant', name: '4x slow motion (120 fps capture played at 30 fps), nothing entered', case: { render: { fps: 30, speedFactor: 4 } } },
  { group: 'time', kind: 'invariant', name: '8x slow motion (240 fps capture played at 30 fps), nothing entered', case: { render: { fps: 30, speedFactor: 8 } } },
  { group: 'time', kind: 'invariant', name: '8x slow motion played at 60 fps, nothing entered', case: { render: { fps: 60, speedFactor: 8 } } },
  { group: 'time', kind: 'invariant', name: '16x slow motion with 3 px jitter, nothing entered', case: { render: { fps: 30, speedFactor: 16, noisePx: 3 } } },
  { group: 'time', kind: 'invariant', name: 'slow motion thinned to a frame budget (4x at 60 fps, every 4th frame)', case: { render: { fps: 15, speedFactor: 4 } } },
  { group: 'time', kind: 'invariant', name: 'swing 25% slower overall (tempo is not compared)', case: { spec: { tempo: 1.25 } } },
  { group: 'time', kind: 'invariant', name: 'swing 20% quicker overall', case: { spec: { tempo: 0.8 } } },
  { group: 'time', kind: 'invariant', name: 'long lead-in with bat waggle (3 s)', case: { spec: { idle: 0.04 }, render: { preRoll: 3 } } },
  { group: 'time', kind: 'invariant', name: 'clip starts right at the stance', case: { render: { preRoll: 0.05 } } },
  { group: 'time', kind: 'invariant', name: 'long tail after the finish (3 s)', case: { render: { postRoll: 3 } } },
  { group: 'time', kind: 'invariant', name: 'stance picked 0.15 s early', case: {}, opts: { stanceShiftSec: -0.15 } },
  { group: 'time', kind: 'invariant', name: 'stance picked 0.1 s late', case: {}, opts: { stanceShiftSec: 0.1 } },
  // ---- robustness
  { group: 'robust', kind: 'robust', name: 'keypoint jitter 2 px', case: { render: { noisePx: 2 } } },
  { group: 'robust', kind: 'robust', name: 'keypoint jitter 5 px', case: { render: { noisePx: 5 } } },
  { group: 'robust', kind: 'robust', name: '5% keypoint dropouts', case: { post: (f) => dropout(f, 0.05) } },
  { group: 'robust', kind: 'robust', name: 'left/right label flicker on 8% of frames', case: { post: (f) => flicker(f, 0.08) } },
  { group: 'robust', kind: 'robust', name: 'camera tilted 3°', case: cam({ roll: 3 }) },
  { group: 'robust', kind: 'robust', name: 'camera tilted 8°', case: cam({ roll: 8 }) },
  { group: 'robust', kind: 'robust', name: 'camera 10° off perpendicular', case: cam({ yaw: 10 }) },
  { group: 'robust', kind: 'limit', name: 'camera 25° off perpendicular', case: cam({ yaw: 25 }) },
  { group: 'robust', kind: 'robust', name: 'perspective camera 8 m away', case: cam({ distance: 8 }) },
  { group: 'robust', kind: 'robust', name: 'perspective camera 4 m away', case: cam({ distance: 4 }) },
  { group: 'robust', kind: 'robust', name: 'different pose model (neck/hips placed differently)', case: { post: (f) => poseModelBias(f) } },
  { group: 'robust', kind: 'robust', name: 'far hand lost behind the body, load to contact (wild low-confidence guesses)', case: { post: (f, r) => hiddenHand(f, r) } },
  { group: 'robust', kind: 'robust', name: 'both hands motion-blurred, foot plant to extension (low confidence)', case: { post: (f, r) => blurredHands(f, r) } },
  { group: 'robust', kind: 'robust', name: 'one hand confidently misplaced around contact', case: { post: (f, r) => wrongHand(f, r) } },
  { group: 'robust', kind: 'robust', name: 'confident one-frame glitches: 4 joints jump half a torso length', case: { post: (f, r) => glitches(f, r) } },
  { group: 'robust', kind: 'robust', name: 'all three hand failures at 30 fps with 3 px jitter', case: { render: { fps: 30, noisePx: 3 }, post: (f, r) => wrongHand(blurredHands(hiddenHand(f, r), r), r) } },
  {
    group: 'robust', kind: 'robust', name: 'realistic phone video: portrait 30 fps, 1.40 m kid, 4° tilt, jitter, dropouts, waggle',
    case: {
      spec: { heightM: 1.4, idle: 0.03 },
      render: { fps: 30, preRoll: 2, noisePx: 2.5, camera: { width: 1080, height: 1920, pxPerM: 700, u0: 500, vGround: 1500, roll: 4 } },
      post: (f) => dropout(f, 0.03),
    },
  },
  // ---- the pro recorded differently from the user (reference user)
  { group: 'pro', kind: 'invariant', name: 'pro at 30 fps, filmed from behind, 4K', proCase: { render: { fps: 30, camera: { pitcherSide: 'left', width: 3840, height: 2160, pxPerM: 870, u0: 1920, vGround: 1995 } } } },
  { group: 'pro', kind: 'invariant', name: 'pro is a 2.0 m left-hander in 8x slow motion', proCase: { spec: { bats: 'L', heightM: 2.0 }, render: { fps: 30, speedFactor: 8 } } },
  { group: 'pro', kind: 'robust', name: 'pro keypoints from a different pose model', proCase: { post: (f) => poseModelBias(f) } },
  // ---- sensitivity: one real difference at a time
  {
    group: 'sensitive', kind: 'sensitive', name: 'longer stride (+15 cm)',
    case: { spec: { tracks: { frontAnkle: { stride: [0.03, 0.66, 0.14], footPlant: [0.02, 0.85, 0], finish: [0.02, 0.85, 0] }, pelvis: { footPlant: [0, 0.16, 0.8], launch: [0.01, 0.17, 0.8], contact: [0.02, 0.18, 0.82], extension: [0.03, 0.21, 0.83], follow: [0.03, 0.25, 0.84], finish: [0.02, 0.28, 0.84] } } } },
    expect: ['plant.stride'],
    // The longer stride moves the hips, and the head with them: a real consequence.
    allow: ['plant.head', 'contact.head'],
  },
  {
    group: 'sensitive', kind: 'sensitive', name: 'hands set 15 cm lower in the stance',
    case: { spec: { tracks: { hands: { stance: [0.4, -0.25, 1.25], loadStart: [0.38, -0.27, 1.27] } } } },
    expect: ['stance.handsHeight'],
  },
  {
    group: 'sensitive', kind: 'sensitive', name: 'hips open 25° less at contact',
    case: { spec: { tracks: { pelvisYaw: { contact: 20, extension: 40 } } } },
    expect: ['contact.hips'],
    allow: ['contact.sequence'],
  },
];

export function runSweep(filter = () => true) {
  return CONDITIONS.filter(filter).map((c) => {
    const pro = c.proCase ? proFromCase(c.proCase) : referencePro();
    const cs = renderCase(c.case || {});
    const res = evaluateCase(cs, pro, c.opts);
    return { ...c, cs, res };
  });
}
