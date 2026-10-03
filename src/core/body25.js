// OpenPose BODY_25 keypoint format: definitions, conversion from MediaPipe
// BlazePose (33 landmarks) and COCO-18, and OpenPose JSON read/write.
//
// A "frame" everywhere in this app is a flat array of 75 numbers:
// [x0, y0, c0, x1, y1, c1, ... x24, y24, c24], exactly like OpenPose's
// `pose_keypoints_2d`. Missing keypoints are (0, 0, 0), as in OpenPose.

export const NUM_KP = 25;

export const KP = Object.freeze({
  Nose: 0,
  Neck: 1,
  RShoulder: 2,
  RElbow: 3,
  RWrist: 4,
  LShoulder: 5,
  LElbow: 6,
  LWrist: 7,
  MidHip: 8,
  RHip: 9,
  RKnee: 10,
  RAnkle: 11,
  LHip: 12,
  LKnee: 13,
  LAnkle: 14,
  REye: 15,
  LEye: 16,
  REar: 17,
  LEar: 18,
  LBigToe: 19,
  LSmallToe: 20,
  LHeel: 21,
  RBigToe: 22,
  RSmallToe: 23,
  RHeel: 24,
});

export const KP_NAMES = Object.keys(KP);

// After canonicalization the "L" joints are always the hitter's FRONT side
// (closest to the pitcher) and the "R" joints the BACK side, regardless of
// handedness. These aliases make the analysis code read naturally.
export const C = Object.freeze({
  nose: KP.Nose,
  neck: KP.Neck,
  midHip: KP.MidHip,
  fShoulder: KP.LShoulder,
  fElbow: KP.LElbow,
  fWrist: KP.LWrist,
  bShoulder: KP.RShoulder,
  bElbow: KP.RElbow,
  bWrist: KP.RWrist,
  fHip: KP.LHip,
  fKnee: KP.LKnee,
  fAnkle: KP.LAnkle,
  bHip: KP.RHip,
  bKnee: KP.RKnee,
  bAnkle: KP.RAnkle,
  fHeel: KP.LHeel,
  fToe: KP.LBigToe,
  bHeel: KP.RHeel,
  bToe: KP.RBigToe,
});

/** Index pairs swapped when mirroring left/right labels. */
export const LR_PAIRS = [
  [KP.RShoulder, KP.LShoulder],
  [KP.RElbow, KP.LElbow],
  [KP.RWrist, KP.LWrist],
  [KP.RHip, KP.LHip],
  [KP.RKnee, KP.LKnee],
  [KP.RAnkle, KP.LAnkle],
  [KP.REye, KP.LEye],
  [KP.REar, KP.LEar],
  [KP.RBigToe, KP.LBigToe],
  [KP.RSmallToe, KP.LSmallToe],
  [KP.RHeel, KP.LHeel],
];

/** Joint groups used for per-group left/right flicker correction. */
export const LR_GROUPS = {
  arms: LR_PAIRS.slice(0, 3),
  legs: [LR_PAIRS[3], LR_PAIRS[4], LR_PAIRS[5], LR_PAIRS[8], LR_PAIRS[9], LR_PAIRS[10]],
  head: [LR_PAIRS[6], LR_PAIRS[7]],
};

/** OpenPose's BODY_25 render pairs. */
export const PAIRS = [
  [1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11],
  [8, 12], [12, 13], [13, 14], [1, 0], [0, 15], [15, 17], [0, 16], [16, 18],
  [14, 19], [19, 20], [14, 21], [11, 22], [22, 23], [11, 24],
];

/** OpenPose's BODY_25 per-keypoint render colors. */
export const COLORS = [
  [255, 0, 85], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
  [85, 255, 0], [0, 255, 0], [255, 0, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
  [0, 170, 255], [0, 85, 255], [0, 0, 255], [255, 0, 170], [170, 0, 255], [255, 0, 255],
  [85, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 255, 255], [0, 255, 255],
  [0, 255, 255],
];

export function emptyFrame() {
  return new Array(NUM_KP * 3).fill(0);
}

export function kx(f, j) {
  return f[j * 3];
}
export function ky(f, j) {
  return f[j * 3 + 1];
}
export function kc(f, j) {
  return f[j * 3 + 2];
}
export function setKp(f, j, x, y, c) {
  f[j * 3] = x;
  f[j * 3 + 1] = y;
  f[j * 3 + 2] = c;
}
export function has(f, j, minConf = 0.05) {
  return f[j * 3 + 2] > minConf;
}

/**
 * Confidence of a point the user placed by hand (src/core/fix.js): above any
 * model's, so it survives every step that moves points with their confidence,
 * and the steps that would doubt it leave it alone.
 */
export const PIN_CONF = 2;
export function pinned(f, j) {
  return f[j * 3 + 2] >= PIN_CONF;
}

export function swapLR(frame, pairs = LR_PAIRS) {
  const f = frame.slice();
  for (const [a, b] of pairs) {
    for (let k = 0; k < 3; k++) {
      f[a * 3 + k] = frame[b * 3 + k];
      f[b * 3 + k] = frame[a * 3 + k];
    }
  }
  return f;
}

// MediaPipe BlazePose landmark indices.
const MP = {
  nose: 0, leftEye: 2, rightEye: 5, leftEar: 7, rightEar: 8,
  leftShoulder: 11, rightShoulder: 12, leftElbow: 13, rightElbow: 14,
  leftWrist: 15, rightWrist: 16, leftHip: 23, rightHip: 24, leftKnee: 25, rightKnee: 26,
  leftAnkle: 27, rightAnkle: 28, leftHeel: 29, rightHeel: 30, leftFootIndex: 31, rightFootIndex: 32,
};

/**
 * Convert one MediaPipe pose (33 normalized landmarks) to a BODY_25 frame in
 * pixel coordinates. `visibility` becomes the OpenPose confidence. MediaPipe
 * has no separate small toe, so it is approximated from the foot index and heel
 * with reduced confidence.
 */
export function mediapipeToBody25(landmarks, width, height) {
  const f = emptyFrame();
  if (!landmarks || landmarks.length < 33) return f;
  const conf = (l) => {
    const v = l.visibility ?? l.presence ?? 1;
    return Number.isFinite(v) ? Math.max(0.001, Math.min(1, v)) : 0.5;
  };
  const put = (j, i, cScale = 1) => {
    const l = landmarks[i];
    setKp(f, j, l.x * width, l.y * height, conf(l) * cScale);
  };
  const putMid = (j, a, b) => {
    const la = landmarks[a];
    const lb = landmarks[b];
    setKp(f, j, ((la.x + lb.x) / 2) * width, ((la.y + lb.y) / 2) * height, Math.min(conf(la), conf(lb)));
  };
  put(KP.Nose, MP.nose);
  putMid(KP.Neck, MP.leftShoulder, MP.rightShoulder);
  put(KP.RShoulder, MP.rightShoulder);
  put(KP.RElbow, MP.rightElbow);
  put(KP.RWrist, MP.rightWrist);
  put(KP.LShoulder, MP.leftShoulder);
  put(KP.LElbow, MP.leftElbow);
  put(KP.LWrist, MP.leftWrist);
  putMid(KP.MidHip, MP.leftHip, MP.rightHip);
  put(KP.RHip, MP.rightHip);
  put(KP.RKnee, MP.rightKnee);
  put(KP.RAnkle, MP.rightAnkle);
  put(KP.LHip, MP.leftHip);
  put(KP.LKnee, MP.leftKnee);
  put(KP.LAnkle, MP.leftAnkle);
  put(KP.REye, MP.rightEye);
  put(KP.LEye, MP.leftEye);
  put(KP.REar, MP.rightEar);
  put(KP.LEar, MP.leftEar);
  put(KP.LBigToe, MP.leftFootIndex);
  put(KP.LHeel, MP.leftHeel);
  put(KP.RBigToe, MP.rightFootIndex);
  put(KP.RHeel, MP.rightHeel);
  // Small toe: a little behind the big toe along the foot, low confidence.
  for (const [small, big, heel] of [
    [KP.LSmallToe, MP.leftFootIndex, MP.leftHeel],
    [KP.RSmallToe, MP.rightFootIndex, MP.rightHeel],
  ]) {
    const b = landmarks[big];
    const h = landmarks[heel];
    setKp(f, small, (b.x * 0.8 + h.x * 0.2) * width, (b.y * 0.8 + h.y * 0.2) * height, conf(b) * 0.5);
  }
  return f;
}

/** Convert a COCO-18 (OpenPose COCO model, 54 values) frame to BODY_25. */
export function coco18ToBody25(kp) {
  const f = emptyFrame();
  const map = {
    0: KP.Nose, 1: KP.Neck, 2: KP.RShoulder, 3: KP.RElbow, 4: KP.RWrist, 5: KP.LShoulder,
    6: KP.LElbow, 7: KP.LWrist, 8: KP.RHip, 9: KP.RKnee, 10: KP.RAnkle, 11: KP.LHip,
    12: KP.LKnee, 13: KP.LAnkle, 14: KP.REye, 15: KP.LEye, 16: KP.REar, 17: KP.LEar,
  };
  for (const [src, dst] of Object.entries(map)) {
    const s = Number(src) * 3;
    setKp(f, dst, kp[s], kp[s + 1], kp[s + 2]);
  }
  if (kc(f, KP.RHip) > 0 && kc(f, KP.LHip) > 0) {
    setKp(
      f,
      KP.MidHip,
      (kx(f, KP.RHip) + kx(f, KP.LHip)) / 2,
      (ky(f, KP.RHip) + ky(f, KP.LHip)) / 2,
      Math.min(kc(f, KP.RHip), kc(f, KP.LHip)),
    );
  }
  return f;
}

/** Normalize any pose_keypoints_2d array (BODY_25 or COCO-18) to BODY_25. */
export function toBody25(kp) {
  if (!Array.isArray(kp)) throw new Error('pose_keypoints_2d must be an array');
  if (kp.length === 75) return kp.map((v) => (Number.isFinite(v) ? v : 0));
  if (kp.length === 54) return coco18ToBody25(kp);
  if (kp.length === 0) return emptyFrame();
  throw new Error(`Unsupported keypoint count ${kp.length / 3} (expected BODY_25 or COCO-18)`);
}

/** Wrap a frame as a standard OpenPose per-frame JSON document. */
export function toOpenPoseJSON(frame) {
  return {
    version: 1.3,
    people: [
      {
        person_id: [-1],
        pose_keypoints_2d: frame.map((v, i) => Math.round((i % 3 === 2 ? Math.min(1, v) : v) * 1000) / 1000),
        face_keypoints_2d: [],
        hand_left_keypoints_2d: [],
        hand_right_keypoints_2d: [],
        pose_keypoints_3d: [],
        face_keypoints_3d: [],
        hand_left_keypoints_3d: [],
        hand_right_keypoints_3d: [],
      },
    ],
  };
}

/** Rough "center" of a person used for tracking across frames. */
function personCenter(f) {
  for (const j of [KP.MidHip, KP.Neck]) if (kc(f, j) > 0.05) return [kx(f, j), ky(f, j)];
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (let j = 0; j < NUM_KP; j++) {
    if (kc(f, j) > 0.05) {
      sx += kx(f, j);
      sy += ky(f, j);
      n++;
    }
  }
  return n ? [sx / n, sy / n] : null;
}

function personSize(f) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let conf = 0;
  for (let j = 0; j < NUM_KP; j++) {
    if (kc(f, j) > 0.05) {
      minX = Math.min(minX, kx(f, j));
      maxX = Math.max(maxX, kx(f, j));
      minY = Math.min(minY, ky(f, j));
      maxY = Math.max(maxY, ky(f, j));
      conf += kc(f, j);
    }
  }
  if (!Number.isFinite(minX)) return 0;
  return (maxY - minY + (maxX - minX) * 0.5) * (conf / NUM_KP);
}

/**
 * Turn a list of OpenPose per-frame JSON documents into one BODY_25 frame per
 * document. With several people in a frame (catcher, umpire...) we lock on
 * to the most prominent person in the first frame and then follow whoever is
 * closest to them.
 */
export function framesFromOpenPoseDocs(docs) {
  const out = [];
  let prev = null;
  for (const doc of docs) {
    const people = (doc?.people || [])
      .map((p) => toBody25(p.pose_keypoints_2d || []))
      .filter((f) => personCenter(f));
    if (!people.length) {
      out.push(emptyFrame());
      continue;
    }
    let pick;
    if (!prev) {
      pick = people.reduce((a, b) => (personSize(b) > personSize(a) ? b : a));
    } else {
      const pc = personCenter(prev);
      pick = people.reduce((a, b) => {
        const ca = personCenter(a);
        const cb = personCenter(b);
        return Math.hypot(cb[0] - pc[0], cb[1] - pc[1]) < Math.hypot(ca[0] - pc[0], ca[1] - pc[1]) ? b : a;
      });
    }
    out.push(pick);
    prev = pick;
  }
  return out;
}

/** Number of confidently detected keypoints in a frame. */
export function detectedCount(f, minConf = 0.3) {
  let n = 0;
  for (let j = 0; j < NUM_KP; j++) if (kc(f, j) >= minConf) n++;
  return n;
}

/** COCO-17 keypoint -> BODY_25 index. */
export const COCO_TO_BODY25 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11];
const FEET = [
  [KP.LAnkle, [KP.LBigToe, KP.LSmallToe, KP.LHeel]],
  [KP.RAnkle, [KP.RBigToe, KP.RSmallToe, KP.RHeel]],
];

/**
 * BODY_25 from fused COCO-17 keypoints ([x, y, prob] in pixels) plus, for the
 * heels and toes, a MediaPipe BODY_25 frame of the same person: its feet are
 * moved with the fused ankles. Mirrored by fused_to_body25() in tools/swingdb.
 */
export function fusedToBody25(fused, mpFrame = null) {
  const f = emptyFrame();
  COCO_TO_BODY25.forEach((j, c) => {
    const [x, y, p] = fused[c];
    if (Number.isFinite(x) && Number.isFinite(y)) setKp(f, j, x, y, p);
  });
  for (const [mid, a, b] of [[KP.Neck, KP.LShoulder, KP.RShoulder], [KP.MidHip, KP.LHip, KP.RHip]]) {
    if (kc(f, a) > 0 && kc(f, b) > 0) setKp(f, mid, (kx(f, a) + kx(f, b)) / 2, (ky(f, a) + ky(f, b)) / 2, Math.min(kc(f, a), kc(f, b)));
  }
  if (mpFrame) {
    for (const [ankle, toes] of FEET) {
      if (!(kc(mpFrame, ankle) > 0 && kc(f, ankle) > 0)) continue;
      const dx = kx(f, ankle) - kx(mpFrame, ankle);
      const dy = ky(f, ankle) - ky(mpFrame, ankle);
      for (const t of toes) if (kc(mpFrame, t) > 0) setKp(f, t, kx(mpFrame, t) + dx, ky(mpFrame, t) + dy, kc(mpFrame, t));
    }
  }
  return f;
}
