#!/usr/bin/env node
// Prints JS detection results on a few test swings as JSON, so the Python
// port in tools/swingdb/analysis.py can be checked against it
// (tools/tests/test_swingdb.py). With the argument "referee" it prints the
// referee's fused poses on random candidate sets instead (tools/swingdb/referee.py).
import { readFileSync } from 'node:fs';
import { archetypeEntry, demoUserSwing, makeSpec, renderSwing } from '../src/core/synth.js';
import { entryFrames } from '../src/core/db.js';
import { swapLR, LR_GROUPS, NUM_KP, COCO_TO_BODY25, fusedToBody25 } from '../src/core/body25.js';
import { fuse, zoomCrop } from '../src/core/referee.js';
import { suggestStanceFrame, detectPitcherSide, canonicalize, estimateSwingFps } from '../src/core/sequence.js';
import { detectPhases } from '../src/core/phases.js';
import { hiddenHand, blurredHands, wrongHand } from './lib/sweep.mjs';

/** Candidate sets around demo-swing poses: noise, outliers, swapped sides, missing joints and models. */
function refereeCases() {
  let seed = 7;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };
  const demo = demoUserSwing();
  const out = [];
  for (const quality of ['best', 'fast']) {
    const referee = JSON.parse(readFileSync(new URL(`../models/referee-${quality}.json`, import.meta.url)));
    const K = referee.candidates.length;
    for (let n = 0; n < 12; n++) {
      const f = demo.frames[(n * 7) % demo.frames.length];
      const truth = COCO_TO_BODY25.map((j) => [f[j * 3], f[j * 3 + 1]]);
      const cands = [];
      for (let c = 0; c < K; c++) {
        if (n % 5 === 3 && c === K - 1) {
          cands.push(null); // a model that found nobody
          continue;
        }
        const noise = 2 + 10 * rand();
        const cand = truth.map(([x, y]) => [x + noise * (rand() - 0.5), y + noise * (rand() - 0.5), rand()]);
        if (rand() < 0.4) [cand[9], cand[10]] = [cand[10], cand[9]]; // wrists swapped
        if (rand() < 0.3) cand[9 + (n % 2)] = [cand[9][0] + 80 * rand(), cand[9][1] - 60 * rand(), 0.9]; // wild wrist
        if (rand() < 0.2) cand[15] = [NaN, NaN, NaN]; // missing ankle
        cands.push(cand);
      }
      const res = fuse(cands, referee);
      const mp = n % 2 ? f : null; // feet from a MediaPipe frame on every other case
      out.push({ quality, cands, mp, result: res, body25: res && fusedToBody25(res.keypoints, mp), zoom: zoomCrop(cands) });
    }
    out.push({ quality, cands: Array.from({ length: K }, () => null), mp: null, result: fuse(Array.from({ length: K }, () => null), referee), body25: null });
  }
  return out;
}

process.stdout.write(JSON.stringify(process.argv[2] === 'referee' ? refereeCases() : swingCases()));

function swingCases() {
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
  // Pose-model hand failures (exercises repairHands): hidden far hand, blur, a misplaced hand.
  const hands = renderSwing(makeSpec({}), { fps: 30, preRoll: 0.8, postRoll: 0.5, noisePx: 3, seed: 24 });
  cases.push({ name: 'hand-failures-30fps', frames: wrongHand(blurredHands(hiddenHand(hands.frames, hands), hands), hands), fps: hands.fps });
  const hands2 = renderSwing(makeSpec({ bats: 'L' }), { fps: 60, preRoll: 0.6, postRoll: 0.4, seed: 25 });
  cases.push({ name: 'hidden-hand-lhh-60fps', frames: hiddenHand(hands2.frames, hands2, { joint: 4, offTL: 1.1, seed: 8 }), fps: hands2.fps });
  // Top hand off the bat during the set-up (really apart: must be left alone).
  const offBat = renderSwing(makeSpec({ idle: 0.03 }), { fps: 60, preRoll: 1.2, postRoll: 0.4, noisePx: 1, seed: 26 });
  cases.push({
    name: 'hand-off-bat-setup',
    fps: offBat.fps,
    frames: offBat.frames.map((f, i) => {
      if (i >= offBat.phases.stance - 10) return f;
      const g = f.slice();
      g[4 * 3 + 1] += 0.9 * Math.hypot(f[3] - f[24], f[4] - f[25]);
      return g;
    }),
  });

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
  return out;
}
