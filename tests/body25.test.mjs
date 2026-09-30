import { test } from 'node:test';
import assert from 'node:assert/strict';
import { KP, mediapipeToBody25, coco18ToBody25, toBody25, swapLR, framesFromOpenPoseDocs, toOpenPoseJSON, emptyFrame, setKp } from '../src/core/body25.js';

function mpLandmarks() {
  return Array.from({ length: 33 }, (_, i) => ({ x: i / 100, y: 1 - i / 100, z: 0, visibility: 0.9 }));
}

test('MediaPipe landmarks map onto BODY_25 in pixels', () => {
  const f = mediapipeToBody25(mpLandmarks(), 1000, 500);
  assert.equal(f.length, 75);
  // Nose = landmark 0.
  assert.deepEqual(f.slice(0, 3), [0, 500, 0.9]);
  // Neck = midpoint of shoulders 11 and 12.
  assert.ok(Math.abs(f[KP.Neck * 3] - 115) < 1e-9);
  assert.ok(Math.abs(f[KP.Neck * 3 + 1] - 500 * (1 - 0.115)) < 1e-9);
  // RWrist = 16, LAnkle = 27, RHeel = 30.
  assert.ok(Math.abs(f[KP.RWrist * 3] - 160) < 1e-9);
  assert.ok(Math.abs(f[KP.LAnkle * 3] - 270) < 1e-9);
  assert.ok(Math.abs(f[KP.RHeel * 3] - 300) < 1e-9);
  // Small toes are approximated with reduced confidence.
  assert.ok(f[KP.LSmallToe * 3 + 2] < 0.9);
});

test('COCO-18 keypoints convert to BODY_25 with a MidHip', () => {
  const coco = new Array(54).fill(0);
  const set = (i, x, y) => {
    coco[i * 3] = x;
    coco[i * 3 + 1] = y;
    coco[i * 3 + 2] = 0.8;
  };
  set(1, 100, 100); // neck
  set(8, 90, 200); // RHip
  set(11, 110, 200); // LHip
  const f = coco18ToBody25(coco);
  assert.deepEqual(f.slice(KP.MidHip * 3, KP.MidHip * 3 + 3), [100, 200, 0.8]);
  assert.equal(toBody25(coco).length, 75);
  assert.throws(() => toBody25([1, 2, 3]));
});

test('swapLR exchanges left and right joints only', () => {
  const f = emptyFrame();
  setKp(f, KP.LWrist, 1, 2, 0.5);
  setKp(f, KP.Nose, 7, 8, 0.9);
  const g = swapLR(f);
  assert.deepEqual(g.slice(KP.RWrist * 3, KP.RWrist * 3 + 3), [1, 2, 0.5]);
  assert.deepEqual(g.slice(0, 3), [7, 8, 0.9]);
});

test('OpenPose documents: lock on to the main hitter and follow them', () => {
  const person = (cx, size, conf = 0.9) => {
    const f = emptyFrame();
    for (let j = 0; j < 25; j++) setKp(f, j, cx + (j % 3) * size * 0.1, 100 + j * size * 0.1, conf);
    setKp(f, KP.MidHip, cx, 100 + size, conf);
    return f;
  };
  const docs = [
    { people: [{ pose_keypoints_2d: person(100, 30) }, { pose_keypoints_2d: person(500, 200) }] },
    { people: [{ pose_keypoints_2d: person(505, 200) }, { pose_keypoints_2d: person(110, 60) }] },
    { people: [] },
  ];
  const frames = framesFromOpenPoseDocs(docs);
  assert.equal(frames.length, 3);
  assert.equal(frames[0][KP.MidHip * 3], 500);
  assert.equal(frames[1][KP.MidHip * 3], 505);
  assert.equal(frames[2][KP.MidHip * 3 + 2], 0);
  const doc = toOpenPoseJSON(frames[0]);
  assert.equal(doc.people[0].pose_keypoints_2d.length, 75);
});
