import { test } from 'node:test';
import assert from 'node:assert/strict';
import { dtw, pathMaps } from '../src/core/dtw.js';

test('identical sequences align on the diagonal', () => {
  const a = Array.from({ length: 40 }, (_, i) => Math.sin(i / 5));
  const { path, cost, endJ } = dtw((i, j) => Math.abs(a[i] - a[j]), 40, 40);
  assert.equal(endJ, 39);
  assert.ok(cost < 1e-9);
  assert.ok(path.every(([i, j]) => i === j));
});

test('a slowed-down copy with a trailing tail is matched (open end)', () => {
  const f = (t) => Math.sin(t * 3) + t;
  const pro = Array.from({ length: 50 }, (_, i) => f(i / 49));
  // User: 1.5x slower, followed by 30 frames of standing still.
  const user = Array.from({ length: 75 }, (_, j) => f(j / 74)).concat(new Array(30).fill(f(1) + 0.8));
  const { path, endJ } = dtw((i, j) => Math.abs(pro[i] - user[j]), pro.length, user.length);
  assert.ok(Math.abs(endJ - 74) <= 2, `endJ ${endJ}`);
  const { proToUser, userToPro } = pathMaps(path, pro.length, user.length);
  assert.ok(Math.abs(proToUser[25] - 25 * 1.5) <= 2, `mid ${proToUser[25]}`);
  assert.equal(userToPro[user.length - 1], pro.length - 1);
});
