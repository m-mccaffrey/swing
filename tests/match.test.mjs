import { test } from 'node:test';
import assert from 'node:assert/strict';
import { prepareSwing } from '../src/core/compare.js';
import { rankStances } from '../src/core/match.js';
import { archetype, mirror, reframe } from './helpers.mjs';

const IDS = ['synthetic-a', 'synthetic-b', 'synthetic-c'];

function prep(id, transform = (f) => f, side = null) {
  const { entry, frames } = archetype(id);
  const p = prepareSwing({ frames: transform(frames), fps: entry.fps, stanceIndex: entry.stanceFrame, pitcherSide: side || entry.orientation.pitcherSide, phases: entry.phases });
  return { id, name: entry.name, prep: p, stancePose: p.stancePose };
}

test('every archetype is its own best stance match', () => {
  const pros = IDS.map((id) => prep(id));
  for (const p of pros) {
    const r = rankStances(p.stancePose, pros);
    assert.equal(r[0].pro.id, p.id);
    assert.ok(r[0].similarity > 99);
    assert.ok(r[1].similarity < 90, `${p.id} vs ${r[1].pro.id}: ${r[1].similarity}`);
  }
});

test('stance match survives mirroring and re-framing', () => {
  const pros = IDS.map((id) => prep(id));
  const w = archetype('synthetic-b').entry.image.width;
  const user = prep('synthetic-b', (f) => reframe(mirror(f, w), 0.6, 80, -40), 'left');
  const r = rankStances(user.stancePose, pros);
  assert.equal(r[0].pro.id, 'synthetic-b');
  assert.ok(r[0].similarity > 99);
});

test('a child-proportioned hitter still matches, with feet on the same ground', async () => {
  const { retarget, boneLengths } = await import('../src/core/match.js');
  const { KP } = await import('../src/core/body25.js');
  const pros = IDS.map((id) => prep(id));
  // Same swing as A, but legs 25% shorter relative to the torso (like a kid).
  const legs = [KP.LKnee, KP.RKnee, KP.LAnkle, KP.RAnkle];
  const kid = prep('synthetic-a', (frames) =>
    frames.map((f) => {
      const L = boneLengths(f);
      for (const j of legs) L[j] *= 0.75;
      return retarget(f, L);
    }),
  );
  const r = rankStances(kid.stancePose, pros);
  assert.equal(r[0].pro.id, 'synthetic-a');
  assert.ok(r[0].similarity > 90, `similarity ${r[0].similarity}`);
  const fitted = retarget(pros[0].stancePose, boneLengths(kid.stancePose));
  const ankleGap = Math.abs(fitted[KP.LAnkle * 3 + 1] - kid.stancePose[KP.LAnkle * 3 + 1]);
  assert.ok(ankleGap < 0.02, `ankle gap ${ankleGap}`);
});
