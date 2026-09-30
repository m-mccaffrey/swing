import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { poseAt, makeSpec, BASE_TIMES } from '../src/core/synth.js';
import { KP } from '../src/core/body25.js';
import { validateEntry, prepareEntry, makeEntry, entryFrames } from '../src/core/db.js';
import { archetype } from './helpers.mjs';

const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

test('synthetic skeleton keeps constant bone lengths through the swing', () => {
  const spec = makeSpec();
  const bones = [[KP.LShoulder, KP.LElbow], [KP.LElbow, KP.LWrist], [KP.RHip, KP.RKnee], [KP.RKnee, KP.RAnkle], [KP.LHip, KP.LKnee], [KP.LKnee, KP.LAnkle], [KP.Neck, KP.MidHip]];
  const ref = poseAt(spec, 0);
  for (let t = -0.2; t <= BASE_TIMES.finish; t += 0.05) {
    const P = poseAt(spec, t);
    for (const [a, b] of bones) assert.ok(Math.abs(dist3(P[a], P[b]) - dist3(ref[a], ref[b])) < 1e-3, `bone ${a}-${b} at ${t}`);
  }
});

test('database index and entries are valid and loadable', () => {
  const dir = new URL('../data/pros/', import.meta.url);
  const index = JSON.parse(readFileSync(new URL('index.json', dir), 'utf8'));
  assert.ok(index.entries.length >= 1);
  const files = new Set(readdirSync(dir));
  const ids = new Set();
  for (const meta of index.entries) {
    assert.ok(files.has(meta.file), `missing ${meta.file}`);
    const entry = JSON.parse(readFileSync(new URL(meta.file, dir), 'utf8'));
    assert.equal(entry.id, meta.id);
    assert.ok(!ids.has(entry.id), 'duplicate id');
    ids.add(entry.id);
    const p = prepareEntry(entry);
    assert.equal(p.prep.phases.contact, entry.phases.contact);
  }
});

test('validation rejects broken entries with readable errors', () => {
  const { entry } = archetype('synthetic-a');
  assert.throws(() => validateEntry({ ...entry, orientation: { pitcherSide: 'up' } }), /pitcherSide/);
  assert.throws(() => validateEntry({ ...entry, stanceFrame: 9999 }), /stanceFrame/);
  assert.throws(() => validateEntry({ ...entry, phases: { ...entry.phases, contact: 2 } }), /order/);
  assert.throws(() => validateEntry({ ...entry, name: '' }), /name/);
});

test('makeEntry round-trips frames and accepts OpenPose documents', () => {
  const { entry, frames } = archetype('synthetic-c');
  const e = makeEntry({ id: 'x', name: 'X', fps: 60, pitcherSide: 'left', stanceFrame: entry.stanceFrame, phases: entry.phases, frames });
  validateEntry(e);
  assert.equal(entryFrames(e).length, frames.length);
  const docs = { ...e, frames: frames.map((f) => ({ version: 1.3, people: [{ pose_keypoints_2d: f }] })) };
  assert.deepEqual(entryFrames(docs)[10], frames[10]);
});
