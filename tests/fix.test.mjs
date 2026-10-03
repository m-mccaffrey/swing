import { test } from 'node:test';
import assert from 'node:assert/strict';
import { KP, COCO_TO_BODY25, PIN_CONF, kx, ky, kc, setKp, toOpenPoseJSON } from '../src/core/body25.js';
import { applyPins, nearestJoint, normalizePins } from '../src/core/fix.js';
import { repairHands, torsoLength } from '../src/core/sequence.js';
import { prepareSwing } from '../src/core/compare.js';
import { makeSpec, renderSwing } from '../src/core/synth.js';

const W = KP.RWrist;
const C = COCO_TO_BODY25.indexOf(W);

/** A swing whose tracker put the right wrist `off` px too high on frames [from, to). */
function mistracked(from, to, off) {
  const s = renderSwing(makeSpec({}), { fps: 60, preRoll: 0.6, postRoll: 0.4, seed: 31 });
  const truth = s.frames;
  const frames = truth.map((f, i) => {
    const g = f.slice();
    if (i >= from && i < to) g[W * 3 + 1] -= off;
    return g;
  });
  // Three models' answers per frame: two follow the tracker's mistake, one is right.
  const candidates = truth.map((f, i) => {
    const coco = COCO_TO_BODY25.map((j) => [frames[i][j * 3], frames[i][j * 3 + 1], 0.9]);
    const right = coco.map((p) => p.slice());
    right[C] = [kx(f, W), ky(f, W), 0.5];
    return [coco, coco.map((p) => p.slice()), right];
  });
  return { s, truth, frames, candidates };
}

test('pins are validated and de-duplicated, latest last', () => {
  const pins = normalizePins([
    { frame: 5, joint: KP.RWrist, x: 1, y: 2 },
    { frame: 2, joint: KP.LKnee, x: 1, y: 2 },
    { frame: 5, joint: KP.RWrist, x: 3, y: 4 },
    { frame: 1, joint: KP.Neck, x: 1, y: 2 }, // a midpoint: not draggable
    { frame: 1, joint: KP.LWrist, x: NaN, y: 2 },
  ]);
  assert.deepEqual(pins, [{ frame: 2, joint: KP.LKnee, x: 1, y: 2 }, { frame: 5, joint: KP.RWrist, x: 3, y: 4 }]); // in order of last placement
});

test('a pin re-picks the wrist from the right model until the tracker agrees again', () => {
  const { s, truth, frames, candidates } = mistracked(40, 52, 60);
  const t = 40;
  const fixed = applyPins(frames, [{ frame: t, joint: W, x: kx(truth[t], W), y: ky(truth[t], W) }], { candidates, fps: s.fps });
  assert.equal(kc(fixed[t], W), PIN_CONF);
  for (let i = t + 1; i < 52; i++) {
    assert.ok(Math.hypot(kx(fixed[i], W) - kx(truth[i], W), ky(fixed[i], W) - ky(truth[i], W)) < 1e-9, `frame ${i}`);
    assert.ok(kc(fixed[i], W) >= 0.6 && kc(fixed[i], W) <= 1);
  }
  // Untouched outside the mistake, and the other joints are untouched everywhere.
  for (const i of [t - 3, 60]) assert.deepEqual(fixed[i], frames[i]);
  assert.equal(kx(fixed[45], KP.LWrist), kx(frames[45], KP.LWrist));
});

test('without the models\' answers only the pinned frame changes', () => {
  const { s, truth, frames } = mistracked(40, 52, 60);
  const fixed = applyPins(frames, [{ frame: 44, joint: W, x: kx(truth[44], W), y: ky(truth[44], W) }], { fps: s.fps });
  assert.equal(ky(fixed[44], W), ky(truth[44], W));
  assert.deepEqual(fixed[45], frames[45]);
});

test('an ankle pin carries the heel and toes, and the hips\' midpoint follows a hip pin', () => {
  const { s, frames } = mistracked(0, 0, 0);
  const f = frames[10];
  const fixed = applyPins(frames, [
    { frame: 10, joint: KP.LAnkle, x: kx(f, KP.LAnkle) + 7, y: ky(f, KP.LAnkle) - 3 },
    { frame: 10, joint: KP.RHip, x: kx(f, KP.RHip) + 10, y: ky(f, KP.RHip) },
  ], { fps: s.fps })[10];
  for (const j of [KP.LHeel, KP.LBigToe]) {
    assert.ok(Math.abs(kx(fixed, j) - kx(f, j) - 7) < 1e-9 && Math.abs(ky(fixed, j) - ky(f, j) + 3) < 1e-9);
  }
  assert.ok(Math.abs(kx(fixed, KP.MidHip) - (kx(f, KP.LHip) + kx(f, KP.RHip) + 10) / 2) < 1e-9);
  assert.equal(toOpenPoseJSON(fixed).people[0].pose_keypoints_2d[KP.LAnkle * 3 + 2], 1); // exported as a plain 1
});

test('hand repair never moves a pinned wrist, and puts a lost hand next to it', () => {
  const { s, truth } = mistracked(0, 0, 0);
  const frames = truth.map((f) => f.slice());
  const i = 50;
  const tl = torsoLength(frames);
  setKp(frames[i], KP.LWrist, kx(frames[i], KP.LWrist) + 0.8 * tl, ky(frames[i], KP.LWrist), PIN_CONF); // apart, but placed by hand
  setKp(frames[i], KP.RWrist, kx(frames[i], KP.RWrist), ky(frames[i], KP.RWrist), 0.9);
  let out = repairHands(frames, s.fps);
  assert.equal(kx(out[i], KP.LWrist), kx(frames[i], KP.LWrist));
  assert.equal(kx(out[i], KP.RWrist), kx(frames[i], KP.RWrist)); // both believable: left as they are
  setKp(frames[i], KP.RWrist, 0, 0, 0.05); // the other hand lost
  out = repairHands(frames, s.fps);
  assert.equal(kc(out[i], KP.LWrist), PIN_CONF);
  assert.ok(Math.hypot(kx(out[i], KP.RWrist) - kx(out[i], KP.LWrist), ky(out[i], KP.RWrist) - ky(out[i], KP.LWrist)) < 0.5 * tl);
});

/** Canonical position of image point (x, y) under prepareSwing's transform. */
function toCanon(t, x, y) {
  let px = t.sx * x;
  let py = y;
  if (t.roll) {
    const c = Math.cos(-t.roll);
    const s = Math.sin(-t.roll);
    [px, py] = [t.cx + c * (px - t.cx) - s * (py - t.cy), t.cy + s * (px - t.cx) + c * (py - t.cy)];
  }
  return [(px - t.ox) / t.scale, -(py - t.oy) / t.scale];
}

test('the analysis goes through a pin even where the pose models disagree with it', () => {
  const { s, frames } = mistracked(0, 0, 0);
  const tl = torsoLength(frames);
  for (const [i, off] of [[55, 0.25], [20, 1.2]]) {
    // Pin the wrist well off the path the models agree on (off torso lengths up and forward).
    const target = [kx(frames[i], W) + off * tl, ky(frames[i], W) - off * tl];
    const pinnedFrames = applyPins(frames, [{ frame: i, joint: W, x: target[0], y: target[1] }], { fps: s.fps });
    const prep = prepareSwing({ frames: pinnedFrames, fps: s.fps, stanceIndex: s.phases.stance, pitcherSide: 'right' });
    const want = toCanon(prep.transform, ...target);
    // The right wrist may be relabelled as the front or back hand.
    const err = Math.min(...[KP.RWrist, KP.LWrist].map((j) => Math.hypot(prep.canon[i][j * 3] - want[0], prep.canon[i][j * 3 + 1] - want[1])));
    assert.ok(err < 0.03, `off ${off}: ${err} torso lengths from the pin`);
  }
});

test('the nearest draggable joint is found within the radius', () => {
  const { frames } = mistracked(0, 0, 0);
  const f = frames[0];
  assert.equal(nearestJoint(f, kx(f, KP.LKnee) + 2, ky(f, KP.LKnee), 10), KP.LKnee);
  assert.equal(nearestJoint(f, -1000, -1000, 10), -1);
  assert.notEqual(nearestJoint(f, kx(f, KP.Neck), ky(f, KP.Neck), 1), KP.Neck);
});
