#!/usr/bin/env node
// Prints JS detection results on a few test swings as JSON, so the Python
// port in tools/swingdb/analysis.py can be checked against it
// (tools/tests/test_swingdb.py).
import { archetypeEntry, demoUserSwing, makeSpec, renderSwing } from '../src/core/synth.js';
import { entryFrames } from '../src/core/db.js';
import { swapLR, LR_GROUPS, NUM_KP } from '../src/core/body25.js';
import { suggestStanceFrame, detectPitcherSide, canonicalize, estimateSwingFps } from '../src/core/sequence.js';
import { detectPhases } from '../src/core/phases.js';

const cases = [];
for (const id of ['synthetic-a', 'synthetic-b', 'synthetic-c']) {
  const e = archetypeEntry(id);
  cases.push({ name: id, frames: entryFrames(e), fps: e.fps });
}
const demo = demoUserSwing();
cases.push({ name: 'demo', frames: demo.frames, fps: demo.fps });
// Mirrored, with label flicker and dropouts, to exercise the repair paths.
cases.push({
  name: 'demo-mirrored-flicker',
  fps: demo.fps,
  frames: demo.frames.map((f, i) => {
    let g = f.slice();
    for (let j = 0; j < NUM_KP; j++) g[j * 3] = demo.width - g[j * 3];
    g = swapLR(g);
    if (i % 11 === 4) g = swapLR(g, LR_GROUPS.legs);
    if (i % 13 === 6) for (const j of [4, 7]) g[j * 3 + 2] = 0;
    return g;
  }),
});

// Tilted camera (exercises the ground-line leveling) and slow motion.
const tilted = renderSwing(makeSpec({ idle: 0.03 }), { fps: 30, preRoll: 1, postRoll: 0.5, noisePx: 1, camera: { roll: 6 }, seed: 21 });
cases.push({ name: 'tilted-6deg', frames: tilted.frames, fps: tilted.fps });
const slow = renderSwing(makeSpec({ bats: 'L' }), { fps: 30, speedFactor: 4, preRoll: 0.6, postRoll: 0.4, camera: { roll: -4 }, seed: 22 });
cases.push({ name: 'lhh-slowmo4-tilted', frames: slow.frames, fps: slow.fps });
const noisySlow = renderSwing(makeSpec({ tempo: 1.2 }), { fps: 30, speedFactor: 8, preRoll: 0.5, postRoll: 0.3, noisePx: 4, seed: 23 });
cases.push({ name: 'noisy-slowmo8-slow-tempo', frames: noisySlow.frames, fps: noisySlow.fps });

// Each case is analyzed the way the tools do it: only the video fps is known,
// and everything runs on the swing's own clock.
const out = cases.map((c) => {
  const swingFps = estimateSwingFps(c.frames, c.fps);
  const stance = suggestStanceFrame(c.frames, swingFps);
  const side = detectPitcherSide(c.frames, stance, swingFps);
  const { frames: canon } = canonicalize(c.frames, { pitcherSide: side.side, stanceIndex: stance, fps: swingFps });
  const phases = detectPhases(canon, swingFps, stance);
  return { name: c.name, fps: c.fps, swingFps, frames: c.frames, stance, side: side.side, confidence: side.confidence, phases, canon };
});
process.stdout.write(JSON.stringify(out));
