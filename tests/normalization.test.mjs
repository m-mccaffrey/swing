// Normalization sweep: the same swing recorded under many conditions must
// compare as identical to itself, nuisances must degrade gracefully, and real
// differences must be caught by the right check (see scripts/lib/sweep.mjs and
// docs/normalization-sweep.md).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { runSweep, renderCase, evaluateCase } from '../scripts/lib/sweep.mjs';

const results = runSweep();
const describe = (r) => `${r.name}: score ${r.res.swingScore.toFixed(1)}, stance ${r.res.stanceSimilarity.toFixed(1)}, phase err ${r.res.maxPhaseErrMs.toFixed(0)} ms, flags [${r.res.notGood}]`;

for (const r of results.filter((r) => r.kind === 'invariant')) {
  test(`invariant: ${r.name}`, () => {
    const x = r.res;
    assert.ok(x.sideCorrect, `pitcher side wrong — ${describe(r)}`);
    assert.deepEqual(x.notGood, [], describe(r));
    assert.ok(x.swingScore >= 95, describe(r));
    assert.ok(x.stanceSimilarity >= 95, describe(r));
    // Phases within ~2 frames of the coarser recording, and never worse than 35 ms.
    assert.ok(x.maxPhaseErrMs <= Math.max(35, 2 * x.msPerFrame), describe(r));
  });
}

for (const r of results.filter((r) => r.kind === 'robust')) {
  test(`robust: ${r.name}`, () => {
    const x = r.res;
    assert.ok(x.sideCorrect, describe(r));
    assert.ok(x.notGood.length <= 1, describe(r));
    assert.ok(x.worst.ratio < 1.5, describe(r));
    assert.ok(x.swingScore >= 90, describe(r));
  });
}

for (const r of results.filter((r) => r.kind === 'sensitive')) {
  test(`sensitive: ${r.name}`, () => {
    const x = r.res;
    for (const id of r.expect) assert.notEqual(x.byId[id].severity, 'good', `${id} should be flagged — ${describe(r)}`);
    const allowed = new Set([...r.expect, ...(r.allow || [])]);
    const extra = x.notGood.filter((id) => !allowed.has(id));
    assert.deepEqual(extra, [], `unrelated checks flagged — ${describe(r)}`);
  });
}

test('limits are detected or documented: ignored slow motion is at least spotted', () => {
  const r = results.find((r) => r.name.includes('suggestion ignored'));
  assert.equal(r.res.slowmoSuggested, 4);
  const yaw = results.find((r) => r.name.includes('25° off'));
  assert.ok(yaw.res.swingScore < 95, 'expected the 25° off-axis camera to degrade (documented limit)');
});

test('keypoint jitter: false alarms stay rare across noise seeds', () => {
  const seeds = [3, 8, 15, 21, 33, 47, 52, 61];
  for (const [noisePx, maxFlags] of [[2, 0], [5, 2]]) {
    let flags = 0;
    for (const seed of seeds) flags += evaluateCase(renderCase({ render: { noisePx, seed } })).notGood.length;
    assert.ok(flags <= maxFlags, `${noisePx}px jitter: ${flags} false flags over ${seeds.length} seeds`);
  }
});

test('camera tilt is corrected from the ground line', () => {
  for (const roll of [-6, 3, 8]) {
    const r = evaluateCase(renderCase({ render: { noisePx: 2, camera: { roll } } }));
    assert.deepEqual(r.notGood, [], `roll ${roll}°: [${r.notGood}]`);
    assert.ok(r.stanceSimilarity > 97, `roll ${roll}°: stance ${r.stanceSimilarity}`);
  }
});
