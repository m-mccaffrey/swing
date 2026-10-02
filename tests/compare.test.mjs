import { test } from 'node:test';
import assert from 'node:assert/strict';
import { prepareSwing, compareSwing } from '../src/core/compare.js';
import { evaluateFeedback, summarize } from '../src/core/feedback.js';
import { detectPitcherSide, suggestStanceFrame } from '../src/core/sequence.js';
import { PHASE_KEYS } from '../src/core/phases.js';
import { archetype, demoUserSwing } from './helpers.mjs';

function proPrep(id) {
  const { entry, frames } = archetype(id);
  return prepareSwing({ frames, fps: entry.fps, stanceIndex: entry.stanceFrame, pitcherSide: entry.orientation.pitcherSide, phases: entry.phases });
}

function demoPrep() {
  const d = demoUserSwing();
  const stance = suggestStanceFrame(d.frames, d.fps);
  const side = detectPitcherSide(d.frames, stance, d.fps).side;
  return { d, user: prepareSwing({ frames: d.frames, fps: d.fps, stanceIndex: stance, pitcherSide: side }) };
}

test('a swing compared with itself scores 100 with identical phases and no issues', () => {
  const p = proPrep('synthetic-a');
  const c = compareSwing(p, p);
  assert.ok(c.swingScore > 99.9);
  assert.deepEqual(c.phases, p.phases);
  const items = evaluateFeedback({ ...p, phases: c.phases }, p, { proName: 'A' });
  assert.ok(items.length >= 25);
  assert.ok(items.every((i) => i.severity === 'good'), items.filter((i) => i.severity !== 'good').map((i) => i.id).join());
});

test('heuristic phase detection matches the labelled pro phases', () => {
  for (const id of ['synthetic-a', 'synthetic-b', 'synthetic-c']) {
    const { entry } = archetype(id);
    const p = proPrep(id);
    for (const k of ['load', 'footPlant', 'contact']) {
      assert.ok(Math.abs(p.detectedPhases[k] - entry.phases[k]) <= 3, `${id} ${k}: ${p.detectedPhases[k]} vs ${entry.phases[k]}`);
    }
  }
});

test('demo swing: phases are transferred accurately through the alignment', () => {
  const { d, user } = demoPrep();
  const c = compareSwing(user, proPrep('synthetic-a'));
  for (const k of PHASE_KEYS.filter((k) => k !== 'stance')) {
    assert.ok(Math.abs(c.phases[k] - d.phases[k]) <= 2, `${k}: ${c.phases[k]} vs true ${d.phases[k]}`);
  }
  assert.ok(c.swingScore > 50 && c.swingScore < 95, `score ${c.swingScore}`);
});

test('demo swing: feedback finds the flaws that were built into it', () => {
  const { user } = demoPrep();
  const pro = proPrep('synthetic-a');
  const c = compareSwing(user, pro);
  const items = evaluateFeedback({ ...user, phases: c.phases }, pro, { proName: 'Pro' });
  const byId = Object.fromEntries(items.map((i) => [i.id, i]));
  const flagged = (id, dir) => {
    const it = byId[id];
    assert.ok(it, `${id} missing`);
    assert.notEqual(it.severity, 'good', `${id} should be flagged (${it.user} vs ${it.pro})`);
    if (dir) assert.equal(Math.sign(it.delta), dir, `${id} direction`);
  };
  flagged('plant.stride', +1); // longer stride
  flagged('contact.head', +1); // head drifts toward the pitcher
  flagged('contact.tilt', +1); // not staying behind the ball
  flagged('contact.shoulders', +1); // shoulders open early
  flagged('load.legLift', -1); // smaller leg lift
  assert.equal(byId['contact.sequence'].severity, 'major');
  const s = summarize(items);
  assert.equal(s.priorities.length, 3);
  assert.ok(s.strengths.length > 5);
});

test('every feedback item carries evidence: measured joints and in-range frames', async () => {
  const { EVIDENCE } = await import('../src/core/feedback.js');
  const { user } = demoPrep();
  const pro = proPrep('synthetic-a');
  const c = compareSwing(user, pro);
  const items = evaluateFeedback({ ...user, phases: c.phases }, pro, { proName: 'Pro' });
  for (const it of items) {
    assert.ok(EVIDENCE[it.id], `no evidence spec for ${it.id}`);
    const ev = it.evidence;
    assert.ok(ev.joints.length > 0, it.id);
    assert.ok(ev.userFrame >= 0 && ev.userFrame < user.canon.length, `${it.id} user frame ${ev.userFrame}`);
    assert.ok(ev.proFrame >= 0 && ev.proFrame < pro.canon.length, `${it.id} pro frame ${ev.proFrame}`);
  }
  // Leg lift is read at the highest point of the front foot, not the load frame.
  const leg = items.find((i) => i.id === 'load.legLift');
  const lift = pro.series.frontFootLift;
  assert.equal(lift[leg.evidence.proFrame], Math.max(...lift.slice(pro.phases.stance, pro.phases.footPlant + 1).filter(Number.isFinite)));
});

test('hand-set beats are used as given, and the automatic beats stay available', () => {
  const { user } = demoPrep();
  const pro = proPrep('synthetic-a');
  const auto = compareSwing(user, pro);
  assert.deepEqual(auto.autoPhases, auto.phases);
  const beats = { ...auto.phases, contact: auto.phases.contact - 2, finish: auto.phases.finish + 3 };
  const fixed = compareSwing(user, pro, { phases: beats });
  assert.deepEqual(fixed.phases, beats);
  assert.deepEqual(fixed.autoPhases, auto.phases);
  // The contact segments are anchored at the moved beat: the pro's contact frame maps to it.
  assert.equal(fixed.userToProOrig(beats.contact), pro.phases.contact);
  assert.notEqual(fixed.swingScore, auto.swingScore);
  // Beats are kept in order and inside the clip whatever comes in.
  const messy = compareSwing(user, pro, { phases: { ...beats, load: beats.footPlant + 5, finish: 1e6 } });
  let last = -1;
  for (const k of PHASE_KEYS) {
    assert.ok(messy.phases[k] >= last && messy.phases[k] < user.canon.length, `${k}: ${messy.phases[k]}`);
    last = messy.phases[k];
  }
});

test('beats give the swing clock when the hands could not time the swing', async () => {
  const { swingFpsFromBeats, pickSwingFps } = await import('../src/core/phases.js');
  const ref = { stance: 12, load: 45, footPlant: 61, contact: 75, extension: 80, finish: 99 }; // reference swing at 60
  assert.ok(Math.abs(swingFpsFromBeats(ref) - 60) < 1e-9);
  const slow = Object.fromEntries(Object.entries(ref).map(([k, v]) => [k, v * 8]));
  assert.ok(Math.abs(swingFpsFromBeats(slow) - 480) < 1e-6);
  assert.equal(pickSwingFps(62, ref), 62); // agree: keep the hands' clock
  assert.equal(pickSwingFps(8, slow), swingFpsFromBeats(slow)); // hands lost: trust the beats
  assert.equal(pickSwingFps(62, null), 62);
});

test('the fitted pro stands on the user\'s ground, whatever the proportions', async () => {
  const { renderCase, scaleLimbs, referencePro } = await import('../scripts/lib/sweep.mjs');
  const { lowestFoot } = await import('../src/core/compare.js');
  const cs = renderCase({ post: (f) => scaleLimbs(f, { legs: 0.8, arms: 0.9 }) });
  const user = prepareSwing({ frames: cs.frames, fps: cs.fps, stanceIndex: cs.truePhases.stance, pitcherSide: cs.trueSide });
  const pro = referencePro().prep;
  const c = compareSwing(user, pro);
  const ground = lowestFoot(user.canon[user.stanceIndex]);
  for (let j = pro.stanceIndex; j <= pro.phases.finish; j++) {
    assert.ok(Math.abs(lowestFoot(c.proFit[j]) - ground) < 0.02, `pro frame ${j}: ${(lowestFoot(c.proFit[j]) - ground).toFixed(3)} TL off the ground`);
  }
});
