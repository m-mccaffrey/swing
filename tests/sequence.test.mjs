import { test } from 'node:test';
import assert from 'node:assert/strict';
import { canonicalize, fixLeftRightFlicker, detectPitcherSide, suggestStanceFrame, resampleFrames, canonToImage, repairHands } from '../src/core/sequence.js';
import { renderSwing, makeSpec } from '../src/core/synth.js';
import { hiddenHand, wrongHand } from '../scripts/lib/sweep.mjs';
import { swapLR, LR_GROUPS, KP } from '../src/core/body25.js';
import { archetype, demoUserSwing, mirror, reframe, maxAbsDiff } from './helpers.mjs';

test('pitcher side is detected for every archetype and the demo swing', () => {
  for (const id of ['synthetic-a', 'synthetic-b', 'synthetic-c']) {
    const { entry, frames } = archetype(id);
    const d = detectPitcherSide(frames, entry.stanceFrame, entry.fps);
    assert.equal(d.side, entry.orientation.pitcherSide, id);
  }
  const demo = demoUserSwing();
  assert.equal(detectPitcherSide(demo.frames, demo.stanceFrame, demo.fps).side, 'right');
  const { entry, frames } = archetype('synthetic-a');
  assert.equal(detectPitcherSide(mirror(frames, 1280), entry.stanceFrame, 60).side, 'left');
});

test('canonical form is independent of camera side, framing and handedness', () => {
  const { entry, frames } = archetype('synthetic-a');
  const base = canonicalize(frames, { pitcherSide: 'right', stanceIndex: entry.stanceFrame, fps: 60 }).frames;
  // Mirror image (left-handed hitter / filmed from the other side), zoomed and shifted.
  const other = reframe(mirror(frames, 1280), 1.7, -300, 120);
  const canon = canonicalize(other, { pitcherSide: 'left', stanceIndex: entry.stanceFrame, fps: 60 }).frames;
  assert.ok(maxAbsDiff(base, canon) < 1e-6, `max diff ${maxAbsDiff(base, canon)}`);
  // Stance MidHip is the origin; the unit is about one adult torso length.
  const s = base[entry.stanceFrame];
  assert.ok(Math.abs(s[KP.MidHip * 3]) < 1e-9 && Math.abs(s[KP.MidHip * 3 + 1]) < 1e-9);
  const torso = Math.hypot(s[KP.Neck * 3] - s[KP.MidHip * 3], s[KP.Neck * 3 + 1] - s[KP.MidHip * 3 + 1]);
  assert.ok(Math.abs(torso - 1) < 0.15, `torso ${torso}`);
  // Front (L) side is toward the pitcher (+x) at the stance.
  assert.ok(s[KP.LAnkle * 3] > s[KP.RAnkle * 3]);
  assert.ok(s[KP.LShoulder * 3] > s[KP.RShoulder * 3]);
});

test('labels swapped by the pose model are repaired', () => {
  const { entry, frames } = archetype('synthetic-a');
  const base = canonicalize(frames, { pitcherSide: 'right', stanceIndex: entry.stanceFrame, fps: 60, clean: false }).frames;
  // Swap the whole body in every frame (e.g. a back view) plus flicker the legs on a few frames.
  const broken = frames.map((f, i) => {
    let g = swapLR(f);
    if (i % 17 === 5) g = swapLR(g, LR_GROUPS.legs);
    return g;
  });
  const fixed = canonicalize(broken, { pitcherSide: 'right', stanceIndex: entry.stanceFrame, fps: 60, clean: false }).frames;
  assert.ok(maxAbsDiff(base, fixed) < 1e-6, `max diff ${maxAbsDiff(base, fixed)}`);
});

test('flicker repair keeps a clean sequence unchanged', () => {
  const { entry, frames } = archetype('synthetic-b');
  const out = fixLeftRightFlicker(frames, entry.stanceFrame);
  assert.equal(maxAbsDiff(frames, out), 0);
});

test('suggested stance is in the quiet set-up before the load', () => {
  const demo = demoUserSwing();
  const s = suggestStanceFrame(demo.frames, demo.fps);
  // The demo waggles until the stance (frame 30) and starts loading at +0.25 s (frame ~37).
  assert.ok(s >= demo.stanceFrame - 3 && s <= demo.stanceFrame + 8, `suggested ${s}`);
  const { entry, frames } = archetype('synthetic-c');
  const s2 = suggestStanceFrame(frames, entry.fps);
  assert.ok(s2 >= 0 && s2 < entry.phases.load - 10, `suggested ${s2}`);
});

test('resampling preserves timing and canonToImage inverts the normalization', () => {
  const { entry, frames } = archetype('synthetic-a');
  const r = resampleFrames(frames, 60, 30);
  assert.equal(r.frames.length, Math.floor((frames.length - 1) / 2) + 1);
  assert.deepEqual(r.frames[3].slice(0, 2), frames[6].slice(0, 2));
  const c = canonicalize(frames, { pitcherSide: 'right', stanceIndex: entry.stanceFrame, fps: 60, clean: false });
  const f = c.frames[40];
  const [x, y] = canonToImage(c.transform, f[KP.LWrist * 3], f[KP.LWrist * 3 + 1]);
  assert.ok(Math.abs(x - frames[40][KP.LWrist * 3]) < 1e-6 && Math.abs(y - frames[40][KP.LWrist * 3 + 1]) < 1e-6);
});

// ---- hand repair (repairHands)

function handCase(post) {
  const r = renderSwing(makeSpec({}), { fps: 60, preRoll: 0.8, postRoll: 0.5, seed: 31 });
  return { r, frames: post ? post(r.frames.map((f) => f.slice()), r) : r.frames };
}
const wristErr = (a, b, j) => Math.hypot(a[j * 3] - b[j * 3], a[j * 3 + 1] - b[j * 3 + 1]);
const torso = (f) => Math.hypot(f[KP.Neck * 3] - f[KP.MidHip * 3], f[KP.Neck * 3 + 1] - f[KP.MidHip * 3 + 1]);

test('hand repair leaves clean hands alone', () => {
  const { frames } = handCase();
  assert.equal(maxAbsDiff(repairHands(frames, 60), frames), 0);
});

test('a hand lost behind the body is put back next to the other hand', () => {
  const { r, frames } = handCase((f, r) => hiddenHand(f, r, { offTL: 1 }));
  const out = repairHands(frames, 60);
  for (let i = r.phases.load; i <= r.phases.contact + 3; i++) {
    assert.ok(out[i][KP.RWrist * 3 + 2] > 0.25, `frame ${i} still doubtful`);
    assert.ok(wristErr(out[i], r.frames[i], KP.RWrist) < 0.25 * torso(r.frames[i]), `frame ${i}: ${wristErr(out[i], r.frames[i], KP.RWrist).toFixed(0)} px off`);
    assert.equal(wristErr(out[i], r.frames[i], KP.LWrist), 0); // the visible hand is untouched
  }
});

test('a confidently misplaced hand is moved back', () => {
  const { r, frames } = handCase((f, r) => wrongHand(f, r, { offTL: 0.9 }));
  const out = repairHands(frames, 60);
  for (let i = r.phases.contact - 3; i <= r.phases.contact + 1; i++) {
    assert.ok(wristErr(frames[i], r.frames[i], KP.LWrist) > 0.5 * torso(r.frames[i]));
    assert.ok(wristErr(out[i], r.frames[i], KP.LWrist) < 0.25 * torso(r.frames[i]), `frame ${i}`);
  }
});

test('two doubtful hands that agree are kept, and a release after contact is left alone', () => {
  const { r, frames } = handCase((f, r) => {
    for (let i = r.phases.footPlant; i <= r.phases.contact; i++) for (const j of [KP.LWrist, KP.RWrist]) f[i][j * 3 + 2] = 0.15;
    // Top hand lets go in the finish: far from the other hand, both confident.
    for (let i = r.phases.finish; i < f.length; i++) f[i][KP.RWrist * 3 + 1] -= 0.8 * torso(f[i]);
    return f;
  });
  const out = repairHands(frames, 60);
  for (let i = r.phases.footPlant; i <= r.phases.contact; i++) {
    for (const j of [KP.LWrist, KP.RWrist]) {
      assert.equal(wristErr(out[i], frames[i], j), 0);
      assert.ok(out[i][j * 3 + 2] > 0.25);
    }
  }
  for (let i = r.phases.finish; i < frames.length; i++) assert.equal(wristErr(out[i], frames[i], KP.RWrist), 0);
});

test('hands that are really apart for a while (a hand off the bat) are not pulled together', () => {
  const { r, frames } = handCase((f, r) => {
    // Top hand off the bat for half a second before the stance, both hands clearly seen.
    for (let i = Math.max(0, r.phases.stance - 30); i < r.phases.stance; i++) f[i][KP.RWrist * 3 + 1] += 0.9 * torso(f[i]);
    return f;
  });
  const out = repairHands(frames, 60);
  for (let i = Math.max(0, r.phases.stance - 30); i < r.phases.stance; i++) assert.equal(wristErr(out[i], frames[i], KP.RWrist), 0, `frame ${i}`);
});

test('the cleanup spline follows the swing, ignores one-frame jumps and bridges gaps on a curve', async () => {
  const { robustSpline } = await import('../src/core/math.js');
  const n = 60;
  const z = Array.from({ length: n }, (_, i) => 100 * Math.sin(i / 8));
  const w = z.map(() => 0.9);
  const noisy = z.slice();
  noisy[20] += 80; // a confident jump
  for (let i = 35; i < 41; i++) w[i] = 0; // a gap
  const { xs, keep } = robustSpline([noisy], w, 0.5, 30);
  assert.equal(keep[20], 0);
  for (let i = 0; i < n; i++) assert.ok(Math.abs(xs[0][i] - z[i]) < 1.5, `sample ${i}: ${xs[0][i].toFixed(1)} vs ${z[i].toFixed(1)}`);
});
