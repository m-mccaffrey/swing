import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fuse, consensusHeight, featureVector, rawFeatures } from '../src/core/referee.js';
import { KP, COCO_TO_BODY25, fusedToBody25, emptyFrame, setKp, kx, ky, kc } from '../src/core/body25.js';
import { PoseEngine, squareFromBox } from '../src/pose/engine.js';

const referee = (q) => JSON.parse(readFileSync(new URL(`../models/referee-${q}.json`, import.meta.url)));

/** A standing person, COCO-17 [x, y], about 400 px tall. */
function person(dx = 0) {
  const p = [
    [300, 100], [310, 92], [290, 92], [320, 98], [280, 98], // head
    [340, 150], [260, 150], [350, 220], [250, 220], [355, 285], [245, 285], // arms
    [325, 290], [275, 290], [330, 390], [270, 390], [335, 490], [265, 490], // legs
  ];
  return p.map(([x, y]) => [x + dx, y]);
}
const withConf = (pts, c = 0.8) => pts.map(([x, y]) => [x, y, c]);

test('referee files match the feature vector', () => {
  for (const q of ['best', 'fast']) {
    const r = referee(q);
    const K = r.candidates.length;
    const z = featureVector(rawFeatures(Array.from({ length: K }, () => withConf(person())), 0, 9, 440), 0, 9, K);
    assert.equal(r.weights.length, z.length, q);
    assert.ok(Number.isFinite(r.bias));
  }
});

const near = (a, b, tol = 1e-6) => a.every((v, i) => Math.abs(v - b[i]) < tol);

test('the referee takes the answer the other models agree on', () => {
  for (const q of ['best', 'fast']) {
    const r = referee(q);
    const K = r.candidates.length;
    if (K < 3) continue; // two models can't outvote each other
    const good = withConf(person());
    const bad = withConf(person());
    bad[9] = [500, 120, 0.95]; // one model throws the left wrist away, confidently
    for (let c = 0; c < K; c++) {
      const cands = Array.from({ length: K }, () => good.map((p) => p.slice()));
      cands[c] = bad;
      const { keypoints, chosen } = fuse(cands, r);
      assert.notEqual(chosen[9], c, `${q}: candidate ${c} outlier chosen`);
      assert.ok(near(keypoints[9].slice(0, 2), [355, 285]), `${q}: ${keypoints[9]}`);
      assert.ok(keypoints[9][2] > 0.5 && keypoints[9][2] < 1);
    }
  }
});

test('answers that agree with the winner are averaged', () => {
  const r = referee('best');
  const K = r.candidates.length;
  const cands = Array.from({ length: K }, (_, c) => withConf(person()).map(([x, y, v], j) => (j === 9 ? [x + c, y, v] : [x, y, v])));
  const soft = fuse(cands, r).keypoints[9];
  const hard = fuse(cands, { ...r, soft_radius: 0 });
  assert.ok(soft[0] > 355 && soft[0] < 355 + K - 1, `averaged x ${soft[0]}`);
  assert.equal(hard.keypoints[9][0], 355 + hard.chosen[9]);
  assert.equal(soft[2], hard.keypoints[9][2]); // confidence is the winner's
});

test('the referee copes with missing models and joints', () => {
  const r = referee('fast');
  const a = withConf(person());
  a[15] = [NaN, NaN, NaN];
  const { keypoints, chosen } = fuse([a, null], r);
  assert.equal(chosen[0], 0);
  assert.equal(chosen[15], -1); // nobody saw the ankle
  assert.ok(Number.isNaN(keypoints[15][0]) && keypoints[15][2] === 0);
  assert.equal(fuse([null, null], r), null);
  assert.throws(() => fuse([a], r), /expects 2/);
});

test('body height comes from the consensus pose', () => {
  const h = consensusHeight([withConf(person()), withConf(person(5)), withConf(person(-5))]);
  assert.ok(Math.abs(h - (490 - 92) * 1.1) < 1e-9);
});

test('fused COCO joints become BODY_25, with MediaPipe feet moved to the fused ankles', () => {
  const fused = withConf(person(), 0.7);
  const mp = emptyFrame();
  setKp(mp, KP.LAnkle, 330, 480, 0.9); // MediaPipe's ankle is 5 px left and 10 px up
  setKp(mp, KP.LBigToe, 360, 500, 0.8);
  setKp(mp, KP.LHeel, 320, 495, 0.6);
  const f = fusedToBody25(fused, mp);
  COCO_TO_BODY25.forEach((j, c) => assert.deepEqual([kx(f, j), ky(f, j), kc(f, j)], fused[c]));
  assert.deepEqual([kx(f, KP.Neck), ky(f, KP.Neck), kc(f, KP.Neck)], [300, 150, 0.7]);
  assert.deepEqual([kx(f, KP.MidHip), ky(f, KP.MidHip)], [300, 290]);
  assert.deepEqual([kx(f, KP.LBigToe), ky(f, KP.LBigToe), kc(f, KP.LBigToe)], [365, 510, 0.8]);
  assert.deepEqual([kx(f, KP.LHeel), ky(f, KP.LHeel)], [325, 505]);
  assert.equal(kc(f, KP.RBigToe), 0); // MediaPipe had no right foot
  assert.equal(kc(fusedToBody25(fused), KP.LBigToe), 0);
});

/** The tracker's decisions, without any models. */
function tracker() {
  const e = Object.create(PoseEngine.prototype);
  e.reset();
  return e;
}

test('the tracker starts on the most prominent person and stays with them', () => {
  const e = tracker();
  const hitter = [200, 100, 150, 400, 0.9];
  const catcher = [500, 300, 160, 200, 0.95];
  assert.deepEqual(e.pick([catcher, hitter]), hitter);
  e.crop = squareFromBox(...hitter.slice(0, 4));
  // Re-checks keep the hitter even when someone bigger walks in elsewhere.
  const umpire = [700, 50, 220, 480, 0.99];
  assert.deepEqual(e.pick([umpire, catcher, [210, 105, 150, 395, 0.6]]), [210, 105, 150, 395, 0.6]);
  assert.equal(e.pick([umpire, catcher]), null); // hitter not found: keep the crop
});

test('the tracker follows the pose, limited per frame, and finds the same hitter after losing them', () => {
  const e = tracker();
  e.crop = squareFromBox(200, 100, 150, 400);
  const before = { ...e.crop };
  // A pose far away and much bigger (a glitch) only moves the crop a little.
  const glitch = withConf(person(2000).map(([x, y]) => [x * 3, y * 3]));
  e.follow(glitch);
  assert.ok(e.crop.side <= before.side * 1.25 + 1e-9);
  const moved = Math.hypot(e.crop.x0 + e.crop.side / 2 - (before.x0 + before.side / 2), e.crop.y0 + e.crop.side / 2 - (before.y0 + before.side / 2));
  assert.ok(moved <= 0.3 * before.side * Math.SQRT2 + 1e-9);
  // Too few confident joints: lost; the next detection must be near the last place.
  e.follow(withConf(person(), 0.05));
  assert.equal(e.crop, null);
  assert.ok(e.last);
  const farAway = [1500, 100, 150, 400, 0.99];
  assert.equal(e.pick([farAway]), null);
  const near = [e.last.x0 + 40, e.last.y0 + 40, 150, 400, 0.5];
  assert.deepEqual(e.pick([farAway, near]), near);
});
