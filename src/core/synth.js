// Procedural baseball swing generator.
//
// Builds a simple 3D skeleton (rigid segments, two-bone IK for arms and legs)
// driven by keyframed swing parameters, then projects it orthographically
// onto a camera placed perpendicular to the pitch path, producing BODY_25
// frames exactly like a pose estimator would. It is used to create the
// placeholder "pro" database entries (until real OpenPose data is added) and
// the demo swing, and in tests where ground truth is needed.
//
// World frame (right-handed hitter): X points from the hitter toward the
// plate (toward the camera), Y points toward the pitcher, Z is up, meters.

import { KP, emptyFrame, setKp, swapLR } from './body25.js';
import { makeMonotoneInterpolator, rng, gauss, clamp } from './math.js';

const RAD = Math.PI / 180;
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => {
  const l = len(a);
  return l > 1e-9 ? mul(a, 1 / l) : [0, 0, 1];
};

function rotZ(d) {
  const c = Math.cos(d * RAD);
  const s = Math.sin(d * RAD);
  return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
}
function rotY(d) {
  const c = Math.cos(d * RAD);
  const s = Math.sin(d * RAD);
  return [[c, 0, s], [0, 1, 0], [-s, 0, c]];
}
function rotX(d) {
  const c = Math.cos(d * RAD);
  const s = Math.sin(d * RAD);
  return [[1, 0, 0], [0, c, -s], [0, s, c]];
}
function matMul(A, B) {
  return A.map((row) => [0, 1, 2].map((j) => row[0] * B[0][j] + row[1] * B[1][j] + row[2] * B[2][j]));
}
function apply(M, v) {
  return [dot(M[0], v), dot(M[1], v), dot(M[2], v)];
}

/** Two-bone IK. Returns the middle joint and the (reach-clamped) end joint. */
function ik2(root, target, l1, l2, pole) {
  const d = sub(target, root);
  const dist = clamp(len(d), Math.abs(l1 - l2) + 1e-3, l1 + l2 - 1e-4);
  const dir = unit(d);
  const cosA = clamp((l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist), -1, 1);
  const sinA = Math.sqrt(1 - cosA * cosA);
  let bend = sub(pole, mul(dir, dot(pole, dir)));
  if (len(bend) < 1e-6) bend = cross(dir, [0, 1, 0]);
  bend = unit(bend);
  const midJ = add(root, add(mul(dir, l1 * cosA), mul(bend, l1 * sinA)));
  return { mid: midJ, end: add(root, mul(dir, dist)) };
}

/** Swing timeline (seconds after the stance) for the base archetype. */
export const BASE_TIMES = {
  stance: 0,
  loadStart: 0.25,
  load: 0.55,
  stride: 0.7,
  footPlant: 0.82,
  launch: 0.9,
  contact: 1.05,
  extension: 1.13,
  follow: 1.25,
  finish: 1.45,
};

/**
 * Keyframed parameters for a 1.88 m right-handed hitter. Keys are phase names
 * from the timeline; values are interpolated with monotone cubics.
 */
export const BASE_TRACKS = {
  pelvis: {
    stance: [0, -0.02, 0.9], loadStart: [0, -0.03, 0.9], load: [0, -0.08, 0.92], footPlant: [0, 0.1, 0.82],
    launch: [0.01, 0.11, 0.82], contact: [0.02, 0.12, 0.84], extension: [0.03, 0.15, 0.85], follow: [0.03, 0.19, 0.86],
    finish: [0.02, 0.22, 0.86],
  },
  pelvisYaw: { stance: 0, loadStart: -3, load: -15, footPlant: 2, launch: 15, contact: 45, extension: 60, follow: 72, finish: 85 },
  torsoYaw: { stance: 0, loadStart: -5, load: -25, footPlant: -28, launch: -18, contact: 25, extension: 60, follow: 95, finish: 120 },
  torsoBend: { stance: 20, load: 22, footPlant: 22, contact: 25, extension: 24, finish: 12 },
  torsoTilt: { stance: 3, load: 0, footPlant: 3, launch: 8, contact: 20, extension: 22, finish: 10 },
  headYaw: { stance: 72, contact: 72, extension: 68, finish: 50 },
  headPitch: { stance: 10, contact: 15, finish: 5 },
  hands: {
    stance: [0.4, -0.25, 1.4], loadStart: [0.38, -0.27, 1.42], load: [0.3, -0.36, 1.47], footPlant: [0.26, -0.38, 1.42],
    launch: [0.3, -0.3, 1.28], contact: [0.55, 0.08, 1.0], extension: [0.6, 0.3, 1.05], follow: [0.4, 0.4, 1.35],
    finish: [-0.05, 0.25, 1.55],
  },
  handle: {
    stance: [0, -0.3, 1], load: [0.1, -0.6, 0.8], footPlant: [0.2, -0.6, 0.8], launch: [0.4, -0.7, 0.4],
    contact: [0.9, 0.35, -0.1], extension: [0.5, 0.8, 0.1], follow: [-0.2, 0.6, 0.6], finish: [-0.6, -0.3, 0.5],
  },
  frontAnkle: { stance: [0, 0.4, 0], loadStart: [0, 0.4, 0], load: [0.05, 0.32, 0.3], stride: [0.03, 0.58, 0.14], footPlant: [0.02, 0.7, 0], finish: [0.02, 0.7, 0] },
  frontFootYaw: { stance: 0, load: -10, footPlant: 30, contact: 35, finish: 45 },
  frontFootPitch: { stance: 0, load: 25, stride: 10, footPlant: 0 },
  backAnkle: { stance: [0, -0.4, 0], contact: [0, -0.38, 0], finish: [0.05, -0.3, 0] },
  backFootYaw: { stance: 0, launch: 0, contact: 40, finish: 70 },
  backFootPitch: { stance: 0, launch: 0, contact: 30, finish: 60 },
  leadElbowPole: { stance: [0.3, 0.3, -1], contact: [0, 0.2, -1], finish: [0.3, 0.5, -0.3] },
  backElbowPole: { stance: [0.2, -0.5, -1], contact: [0, -0.2, -1], finish: [0.2, 0.2, -1] },
};

/**
 * Build a swing spec from the base archetype plus overrides:
 * { heightM, bats: 'R'|'L', idle, times: {...}, tracks: { trackName: { phase: value } } }
 */
export function makeSpec(over = {}) {
  const tracks = {};
  for (const [k, v] of Object.entries(BASE_TRACKS)) tracks[k] = { ...v, ...(over.tracks?.[k] || {}) };
  // `tempo` stretches everything after the stance (1.2 = a 20% slower swing).
  const tempo = over.tempo ?? 1;
  const times = Object.fromEntries(Object.entries({ ...BASE_TIMES, ...(over.times || {}) }).map(([k, t]) => [k, t * tempo]));
  return {
    heightM: over.heightM ?? 1.88,
    bats: over.bats ?? 'R',
    idle: over.idle ?? 0,
    times,
    tracks,
  };
}

function buildInterpolators(spec) {
  const out = {};
  for (const [name, keys] of Object.entries(spec.tracks)) {
    const entries = Object.entries(keys)
      .map(([phase, v]) => [spec.times[phase], v])
      .filter(([t]) => Number.isFinite(t))
      .sort((a, b) => a[0] - b[0]);
    const xs = entries.map((e) => e[0]);
    if (Array.isArray(entries[0][1])) {
      const comps = [0, 1, 2].map((c) => makeMonotoneInterpolator(xs, entries.map((e) => e[1][c])));
      out[name] = (t) => comps.map((f) => f(t));
    } else {
      out[name] = makeMonotoneInterpolator(xs, entries.map((e) => e[1]));
    }
  }
  return out;
}

/** 3D joint positions (BODY_25 indices, right-handed labels) at time t. */
export function poseAt(spec, t, I = buildInterpolators(spec)) {
  const s = spec.heightM / 1.88;
  const L = {
    thigh: 0.46 * s, shank: 0.46 * s, upper: 0.31 * s, fore: 0.27 * s, spine: 0.55 * s,
    shHalf: 0.19 * s, hipHalf: 0.12 * s, head: 0.19 * s,
  };
  const scaled = (v) => mul(v, s);
  // Idle waggle before the stance fades to zero at t = 0.
  const fade = t < 0 ? Math.min(1, -t / 0.4) : 0;
  const amp = spec.idle * fade;
  const w = 2 * Math.PI * 1.1 * t;
  const sway = [0, amp * 0.4 * Math.sin(w * 0.7), 0];
  const waggle = [amp * 0.3 * Math.sin(w + 0.5), amp * Math.sin(w), amp * 0.6 * Math.sin(w + 1)];

  const pelvis = add(scaled(I.pelvis(t)), sway);
  const Rp = rotZ(I.pelvisYaw(t));
  const Rt = matMul(rotZ(I.torsoYaw(t)), matMul(rotY(I.torsoBend(t)), rotX(I.torsoTilt(t))));
  const lHip = add(pelvis, apply(Rp, [0, L.hipHalf, 0]));
  const rHip = add(pelvis, apply(Rp, [0, -L.hipHalf, 0]));
  const neck = add(pelvis, apply(Rt, [0, 0, L.spine]));
  const lSh = add(neck, apply(Rt, [0, L.shHalf, 0]));
  const rSh = add(neck, apply(Rt, [0, -L.shHalf, 0]));

  // Head: mostly upright, face turned toward the pitcher.
  const upT = apply(Rt, [0, 0, 1]);
  const headC = add(neck, mul(unit(add(mul(upT, 0.4), [0, 0, 0.6])), L.head));
  const hy = I.headYaw(t) * RAD;
  const hp = I.headPitch(t) * RAD;
  const face = [Math.cos(hy) * Math.cos(hp), Math.sin(hy) * Math.cos(hp), -Math.sin(hp)];
  const left = unit(cross([0, 0, 1], face));
  const up = cross(face, left);
  const hs = s;
  const nose = add(headC, add(mul(face, 0.1 * hs), mul(up, -0.01 * hs)));
  const lEye = add(headC, add(add(mul(face, 0.08 * hs), mul(left, 0.033 * hs)), mul(up, 0.035 * hs)));
  const rEye = add(headC, add(add(mul(face, 0.08 * hs), mul(left, -0.033 * hs)), mul(up, 0.035 * hs)));
  const lEar = add(headC, add(add(mul(left, 0.075 * hs), mul(face, -0.01 * hs)), mul(up, 0.01 * hs)));
  const rEar = add(headC, add(add(mul(left, -0.075 * hs), mul(face, -0.01 * hs)), mul(up, 0.01 * hs)));

  // Arms: both hands on the bat handle; right (top) hand above the left.
  const handsC = add(scaled(I.hands(t)), waggle);
  const handle = unit(I.handle(t));
  const rWristT = add(handsC, mul(handle, 0.045 * s));
  const lWristT = add(handsC, mul(handle, -0.045 * s));
  const la = ik2(lSh, lWristT, L.upper, L.fore, I.leadElbowPole(t));
  const ra = ik2(rSh, rWristT, L.upper, L.fore, I.backElbowPole(t));

  // Feet: ankle height follows heel lift so the toes stay on the ground.
  const foot = (ankleKey, yawKey, pitchKey, outward) => {
    const a = I[ankleKey](t);
    const yaw = I[yawKey](t) * RAD;
    const p = Math.max(0, I[pitchKey](t)) * RAD;
    const ground = 0.02 + 0.16 * Math.sin(p) + 0.06 * Math.cos(p);
    const ankle = [a[0] * s, a[1] * s, (a[2] + ground) * s];
    const f = [Math.cos(yaw), Math.sin(yaw), 0];
    const lat = [-Math.sin(yaw) * outward, Math.cos(yaw) * outward, 0];
    const off = (fa, za) => add(mul(f, (fa * Math.cos(p) + za * Math.sin(p)) * s), [0, 0, (-fa * Math.sin(p) + za * Math.cos(p)) * s]);
    const toe = add(ankle, off(0.16, -0.06));
    return {
      ankle,
      bigToe: add(toe, mul(lat, -0.02 * s)),
      smallToe: add(add(toe, mul(lat, 0.035 * s)), mul(f, -0.03 * s)),
      heel: add(ankle, off(-0.06, -0.06)),
    };
  };
  const lf = foot('frontAnkle', 'frontFootYaw', 'frontFootPitch', 1);
  const rf = foot('backAnkle', 'backFootYaw', 'backFootPitch', -1);
  const ll = ik2(lHip, lf.ankle, L.thigh, L.shank, [1, 0.25, 0.1]);
  const rl = ik2(rHip, rf.ankle, L.thigh, L.shank, [1, -0.25, 0.1]);

  const P = new Array(25);
  P[KP.Nose] = nose;
  P[KP.Neck] = neck;
  P[KP.RShoulder] = rSh;
  P[KP.RElbow] = ra.mid;
  P[KP.RWrist] = ra.end;
  P[KP.LShoulder] = lSh;
  P[KP.LElbow] = la.mid;
  P[KP.LWrist] = la.end;
  P[KP.MidHip] = pelvis;
  P[KP.RHip] = rHip;
  P[KP.RKnee] = rl.mid;
  P[KP.RAnkle] = rl.end;
  P[KP.LHip] = lHip;
  P[KP.LKnee] = ll.mid;
  P[KP.LAnkle] = ll.end;
  P[KP.REye] = rEye;
  P[KP.LEye] = lEye;
  P[KP.REar] = rEar;
  P[KP.LEar] = lEar;
  P[KP.LBigToe] = lf.bigToe;
  P[KP.LSmallToe] = lf.smallToe;
  P[KP.LHeel] = lf.heel;
  P[KP.RBigToe] = rf.bigToe;
  P[KP.RSmallToe] = rf.smallToe;
  P[KP.RHeel] = rf.heel;
  return P;
}

/**
 * Render a swing to BODY_25 pixel frames.
 * @param {object} spec from makeSpec()
 * @param {object} o
 * @param {number} [o.fps=60] video frames per second of video time
 * @param {number} [o.speedFactor=1] slow motion (4 → each video second covers 0.25 s)
 * @param {number} [o.preRoll=0.3] seconds of real time before the stance
 * @param {number} [o.postRoll=0.3] seconds after the finish
 * @param {object} [o.camera] { width, height, pxPerM, u0, vGround, pitcherSide }
 * @param {number} [o.noisePx=0] keypoint jitter (pixels, 1σ)
 * @param {number} [o.seed=7]
 */
export function renderSwing(spec, o = {}) {
  const fps = o.fps ?? 60;
  const speed = o.speedFactor ?? 1;
  const pre = o.preRoll ?? 0.3;
  const post = o.postRoll ?? 0.3;
  const cam = {
    width: 1280,
    height: 720,
    pxPerM: 290,
    u0: 640,
    vGround: 665,
    pitcherSide: spec.bats === 'L' ? 'left' : 'right',
    ...(o.camera || {}),
  };
  const rand = rng(o.seed ?? 7);
  const I = buildInterpolators(spec);
  const t0 = -pre;
  const t1 = spec.times.finish + post;
  const dt = 1 / (fps * speed);
  const count = Math.floor((t1 - t0) / dt) + 1;
  const sx = cam.pitcherSide === 'left' ? -1 : 1;
  // Camera imperfections for robustness tests:
  //   yaw      degrees the camera is turned away from perpendicular to the pitch path
  //   roll     degrees the camera is tilted (image rotation)
  //   distance meters to the hitter for a perspective camera (default: orthographic)
  //   camHeight meters above the ground of a perspective camera's lens
  const yaw = ((cam.yaw ?? 0) * Math.PI) / 180;
  const roll = ((cam.roll ?? 0) * Math.PI) / 180;
  const camH = cam.camHeight ?? 1.1;
  const project = (p) => {
    let X = p[0];
    let Y = p[1];
    if (yaw) [X, Y] = [X * Math.cos(yaw) - Y * Math.sin(yaw), X * Math.sin(yaw) + Y * Math.cos(yaw)];
    let k = cam.pxPerM;
    let u;
    let v;
    if (cam.distance) {
      k = (cam.pxPerM * cam.distance) / (cam.distance - X);
      u = cam.u0 + sx * k * Y;
      v = cam.vGround - cam.pxPerM * camH - k * (p[2] - camH);
    } else {
      u = cam.u0 + sx * k * Y;
      v = cam.vGround - k * p[2];
    }
    if (roll) {
      const cx = cam.width / 2;
      const cy = cam.height / 2;
      const du = u - cx;
      const dv = v - cy;
      u = cx + du * Math.cos(roll) - dv * Math.sin(roll);
      v = cy + du * Math.sin(roll) + dv * Math.cos(roll);
    }
    return [u, v];
  };
  const frames = [];
  const times = [];
  for (let k = 0; k < count; k++) {
    const t = t0 + k * dt;
    const P = poseAt(spec, t, I);
    let f = emptyFrame();
    // Joints turned away from the camera (smaller X) get lower confidence.
    const depthRef = P[KP.Neck][0];
    for (let j = 0; j < 25; j++) {
      const p = P[j];
      let conf = 0.92 - 0.3 * clamp((depthRef - p[0] - 0.05) / 0.3, 0, 1);
      if (j === KP.LSmallToe || j === KP.RSmallToe) conf *= 0.6;
      conf = clamp(conf + gauss(rand) * 0.02, 0.05, 0.99);
      const [u, v] = project(p);
      setKp(f, j, u + gauss(rand) * (o.noisePx ?? 0), v + gauss(rand) * (o.noisePx ?? 0), conf);
    }
    // A left-handed hitter is the mirror image of a right-handed one.
    if (spec.bats === 'L') f = swapLR(f);
    frames.push(f.map((x) => Math.round(x * 100) / 100));
    times.push(t);
  }
  const frameOf = (t) => clamp(Math.round((t - t0) / dt), 0, count - 1);
  const phases = {};
  for (const key of ['stance', 'load', 'footPlant', 'contact', 'extension', 'finish']) phases[key] = frameOf(spec.times[key]);
  return {
    frames,
    fps,
    speedFactor: speed,
    width: cam.width,
    height: cam.height,
    pitcherSide: cam.pitcherSide,
    stanceFrame: phases.stance,
    phases,
    times,
  };
}

/** Three clearly synthetic placeholder archetypes for the starter database. */
export const ARCHETYPES = {
  'synthetic-a': {
    meta: {
      name: 'Placeholder A — leg kick, hands high',
      team: 'Synthetic',
      bats: 'R',
      notes: 'Procedurally generated placeholder. Medium-width stance, big leg kick, hands loaded high by the back shoulder.',
    },
    spec: {},
    render: { fps: 60, preRoll: 0.2, postRoll: 0.25, seed: 11 },
  },
  'synthetic-b': {
    meta: {
      name: 'Placeholder B — wide & crouched, toe tap',
      team: 'Synthetic',
      bats: 'R',
      notes: 'Procedurally generated placeholder. Wide, crouched stance, small toe-tap stride, lower hands.',
    },
    spec: {
      tracks: {
        pelvis: { stance: [0, -0.01, 0.8], loadStart: [0, -0.02, 0.8], load: [0, -0.06, 0.81], footPlant: [0, 0.06, 0.78], launch: [0.01, 0.07, 0.78], contact: [0.02, 0.08, 0.8], extension: [0.03, 0.1, 0.81], follow: [0.03, 0.13, 0.82], finish: [0.02, 0.15, 0.82] },
        torsoBend: { stance: 28, load: 28, footPlant: 28, contact: 28, extension: 26, finish: 15 },
        hands: { stance: [0.38, -0.3, 1.24], loadStart: [0.37, -0.31, 1.25], load: [0.3, -0.38, 1.3], footPlant: [0.27, -0.4, 1.3], launch: [0.3, -0.32, 1.2] },
        frontAnkle: { stance: [0, 0.5, 0], loadStart: [0, 0.5, 0], load: [0.04, 0.47, 0.1], stride: [0.03, 0.6, 0.06], footPlant: [0.02, 0.66, 0], finish: [0.02, 0.66, 0] },
        backAnkle: { stance: [0, -0.5, 0], contact: [0, -0.48, 0], finish: [0.05, -0.4, 0] },
        frontFootPitch: { stance: 0, load: 10, stride: 5, footPlant: 0 },
      },
    },
    render: { fps: 60, preRoll: 0.2, postRoll: 0.25, seed: 23, camera: { width: 1920, height: 1080, pxPerM: 420, u0: 960, vGround: 1000 } },
  },
  'synthetic-c': {
    meta: {
      name: 'Placeholder C — upright, big stride (LHH)',
      team: 'Synthetic',
      bats: 'L',
      notes: 'Procedurally generated placeholder. Left-handed, filmed with the pitcher on the left. Narrow upright stance, long stride, hands high.',
    },
    spec: {
      bats: 'L',
      heightM: 1.83,
      tracks: {
        pelvis: { stance: [0, -0.02, 0.93], loadStart: [0, -0.03, 0.93], load: [0, -0.07, 0.94], footPlant: [0, 0.14, 0.82], launch: [0.01, 0.15, 0.82], contact: [0.02, 0.16, 0.84], extension: [0.03, 0.19, 0.85], follow: [0.03, 0.22, 0.86], finish: [0.02, 0.25, 0.86] },
        torsoBend: { stance: 12, load: 15, footPlant: 18, contact: 22, extension: 22, finish: 10 },
        hands: { stance: [0.36, -0.2, 1.55], loadStart: [0.35, -0.22, 1.56], load: [0.3, -0.32, 1.56], footPlant: [0.27, -0.35, 1.48] },
        frontAnkle: { stance: [0, 0.3, 0], loadStart: [0, 0.3, 0], load: [0.05, 0.26, 0.22], stride: [0.03, 0.62, 0.12], footPlant: [0.02, 0.78, 0], finish: [0.02, 0.78, 0] },
        backAnkle: { stance: [0, -0.3, 0], contact: [0, -0.28, 0], finish: [0.05, -0.22, 0] },
      },
    },
    render: { fps: 60, preRoll: 0.2, postRoll: 0.25, seed: 37 },
  },
};

/** Database entry (swing-db/v1) for a synthetic archetype. */
export function archetypeEntry(id) {
  const a = ARCHETYPES[id];
  const spec = makeSpec(a.spec);
  const r = renderSwing(spec, a.render);
  return {
    schema: 'swing-db/v1',
    id,
    ...a.meta,
    synthetic: true,
    source: 'Procedurally generated by src/core/synth.js — replace with real OpenPose data',
    keypointFormat: 'BODY_25',
    fps: r.fps,
    speedFactor: 1,
    image: { width: r.width, height: r.height },
    orientation: { pitcherSide: r.pitcherSide, view: 'open side (facing the hitter’s chest)' },
    stanceFrame: r.stanceFrame,
    phases: r.phases,
    frames: r.frames.map((f) => ({ pose_keypoints_2d: f })),
  };
}

/**
 * The built-in demo "user" swing: an amateur-looking swing with a few classic
 * flaws (long stride, head drift, shoulders opening early, soft front leg,
 * slower swing), filmed in portrait at 30 fps with a waggle before the stance.
 */
export function demoUserSwing() {
  const spec = makeSpec({
    heightM: 1.76,
    idle: 0.03,
    times: { footPlant: 0.86, launch: 0.96, contact: 1.16, extension: 1.25, follow: 1.38, finish: 1.6 },
    tracks: {
      pelvis: {
        stance: [0, 0.0, 0.88], loadStart: [0, -0.01, 0.88], load: [0, -0.03, 0.89], stride: [0, 0.12, 0.84],
        footPlant: [0, 0.2, 0.78], launch: [0.01, 0.24, 0.77], contact: [0.02, 0.28, 0.77], extension: [0.03, 0.3, 0.78],
        follow: [0.03, 0.31, 0.8], finish: [0.02, 0.32, 0.82],
      },
      pelvisYaw: { stance: 0, loadStart: -2, load: -8, footPlant: 5, launch: 12, contact: 30, extension: 45, follow: 60, finish: 70 },
      torsoYaw: { stance: 0, loadStart: -3, load: -12, footPlant: -2, launch: 15, contact: 45, extension: 70, follow: 90, finish: 100 },
      torsoTilt: { stance: 3, load: 1, footPlant: 2, launch: 3, contact: 6, extension: 8, finish: 5 },
      hands: {
        stance: [0.4, -0.24, 1.36], loadStart: [0.39, -0.25, 1.37], load: [0.36, -0.28, 1.38], footPlant: [0.36, -0.22, 1.3],
        launch: [0.42, -0.12, 1.15], contact: [0.6, 0.22, 0.98], extension: [0.62, 0.42, 1.0], follow: [0.45, 0.45, 1.2],
        finish: [0.05, 0.35, 1.35],
      },
      frontAnkle: { stance: [0, 0.38, 0], loadStart: [0, 0.38, 0], load: [0.04, 0.34, 0.16], stride: [0.03, 0.7, 0.08], footPlant: [0.02, 0.88, 0], finish: [0.02, 0.88, 0] },
      backAnkle: { stance: [0, -0.4, 0], contact: [0, -0.38, 0], finish: [0.05, -0.32, 0] },
    },
  });
  const r = renderSwing(spec, {
    fps: 30,
    preRoll: 1.0,
    postRoll: 0.7,
    noisePx: 1.5,
    seed: 5,
    camera: { width: 1080, height: 1920, pxPerM: 560, u0: 470, vGround: 1560, pitcherSide: 'right' },
  });
  return { ...r, spec };
}
