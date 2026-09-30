import { archetypeEntry, demoUserSwing } from '../src/core/synth.js';
import { entryFrames } from '../src/core/db.js';
import { swapLR, NUM_KP } from '../src/core/body25.js';

export function archetype(id) {
  const e = archetypeEntry(id);
  return { entry: e, frames: entryFrames(e) };
}

export { demoUserSwing };

/** Mirror frames horizontally inside an image of width w and swap L/R labels. */
export function mirror(frames, w) {
  return frames.map((f) => {
    const g = f.slice();
    for (let j = 0; j < NUM_KP; j++) if (g[j * 3 + 2] > 0) g[j * 3] = w - g[j * 3];
    return swapLR(g);
  });
}

/** Scale and translate frames (simulates a different camera distance/framing). */
export function reframe(frames, s, dx, dy) {
  return frames.map((f) => {
    const g = f.slice();
    for (let j = 0; j < NUM_KP; j++) {
      if (g[j * 3 + 2] > 0) {
        g[j * 3] = g[j * 3] * s + dx;
        g[j * 3 + 1] = g[j * 3 + 1] * s + dy;
      }
    }
    return g;
  });
}

export function maxAbsDiff(a, b, conf = 0.1) {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < NUM_KP; j++) {
      if (a[i][j * 3 + 2] > conf && b[i][j * 3 + 2] > conf) {
        m = Math.max(m, Math.abs(a[i][j * 3] - b[i][j * 3]), Math.abs(a[i][j * 3 + 1] - b[i][j * 3 + 1]));
      }
    }
  }
  return m;
}
