#!/usr/bin/env node
// Prints JS detection results on a few test swings as JSON, so the Python
// port in tools/swingdb/analysis.py can be checked against it
// (tools/tests/test_swingdb.py).
import { archetypeEntry, demoUserSwing } from '../src/core/synth.js';
import { entryFrames } from '../src/core/db.js';
import { swapLR, LR_GROUPS, NUM_KP } from '../src/core/body25.js';
import { suggestStanceFrame, detectPitcherSide, canonicalize } from '../src/core/sequence.js';
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

const out = cases.map((c) => {
  const stance = suggestStanceFrame(c.frames, c.fps);
  const side = detectPitcherSide(c.frames, stance, c.fps);
  const { frames: canon } = canonicalize(c.frames, { pitcherSide: side.side, stanceIndex: stance, fps: c.fps });
  const phases = detectPhases(canon, c.fps, stance);
  return { name: c.name, fps: c.fps, frames: c.frames, stance, side: side.side, confidence: side.confidence, phases, canon };
});
process.stdout.write(JSON.stringify(out));
