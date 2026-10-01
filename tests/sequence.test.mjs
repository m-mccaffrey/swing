import { test } from 'node:test';
import assert from 'node:assert/strict';
import { canonicalize, fixLeftRightFlicker, detectPitcherSide, suggestStanceFrame, resampleFrames, canonToImage } from '../src/core/sequence.js';
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
