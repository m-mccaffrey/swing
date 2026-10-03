import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { entryFrames } from '../src/core/db.js';
import { detectPitcherSide, estimateSwingFps } from '../src/core/sequence.js';
import { pickSwingFps } from '../src/core/phases.js';

const dir = new URL('../data/pros/', import.meta.url);
const index = JSON.parse(readFileSync(new URL('index.json', dir)));

// A pitcher side stored on the wrong side flips the whole comparison (the pro's
// back foot is measured as the front one), so every entry's stored side must
// agree with what its own swing says at its stance whenever that's clear.
for (const meta of index.entries) {
  test(`${meta.id}: stored pitcher side agrees with the swing`, () => {
    const e = JSON.parse(readFileSync(new URL(meta.file, dir)));
    const frames = entryFrames(e);
    const sf = e.phases ? pickSwingFps(estimateSwingFps(frames, e.fps), e.phases) : estimateSwingFps(frames, e.fps);
    const d = detectPitcherSide(frames, e.stanceFrame ?? 0, sf);
    if (d.confidence < 0.5) return; // not clear enough to judge
    assert.equal(e.orientation.pitcherSide, d.side, `the swing says the pitcher is to the ${d.side} (${Math.round(d.confidence * 100)}% sure)`);
  });
}
