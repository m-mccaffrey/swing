// Pose engine: find the hitter, follow them, and fuse several pose models.
// Same pipeline as tools/swingdb/engine.py (the pro database), so users and
// pros are measured the same way:
//   1. A square crop around the hitter: EfficientDet-Lite0 (MediaPipe object
//      detector) finds people to start with, every RECHECK frames and when
//      tracking is lost; otherwise the crop follows the previous frame's pose.
//   2. On the crop, the models the referee was trained with: MoveNet Thunder
//      (TensorFlow.js) and MediaPipe Pose (heavy), each also on the mirrored
//      crop, plus EfficientPose ("best"), or MoveNet and MediaPipe once ("fast");
//      then MoveNet again on a close-up of the arms where those answers put them.
//   3. The referee (models/referee-*.json) picks each joint's best answer and
//      averages it with the answers that agree.
//   4. BODY_25: the fused joints plus MediaPipe's heels and toes.

import { fuse, zoomCrop, ZOOM_JOINTS } from '../core/referee.js';
import { emptyFrame, fusedToBody25, mediapipeToBody25 } from '../core/body25.js';

const TF_VERSION = '4.22.0';
const LOCAL_TF = new URL('../../vendor/tfjs/', import.meta.url).href;
const CDN_TF = `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@${TF_VERSION}/dist/`;
const CDN_TF_WASM = `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${TF_VERSION}/dist/`;
const MODELS_DIR = new URL('../../models/', import.meta.url).href;

const MOVENET_SIZE = 256;
const EP_SIZE = 368;
const MP_CROP = 384;
const CROP_MARGIN = 0.1;
const POSE_MARGIN = 0.2;
const RECHECK = 30;
const FLIP_COCO = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]];
const MP_PAIRS = [[1, 4], [2, 5], [3, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22],
  [23, 24], [25, 26], [27, 28], [29, 30], [31, 32]];
const MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
// EfficientPose (MPII order: head, neck, r_shoulder, r_elbow, r_wrist, chest, l_shoulder, l_elbow, l_wrist,
// center, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle) -> COCO; it has no eyes or ears.
const EP_TO_COCO = [[0, 0], [5, 6], [6, 2], [7, 7], [8, 3], [9, 8], [10, 4], [11, 13], [12, 10], [13, 14], [14, 11], [15, 15], [16, 12]];
const EP_FLIP = [0, 1, 6, 7, 8, 5, 2, 3, 4, 9, 13, 14, 15, 10, 11, 12];

let tfPromise = null;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const el = document.createElement('script');
    el.src = src;
    el.onload = resolve;
    el.onerror = () => {
      el.remove();
      reject(new Error(`Could not load ${src}`));
    };
    document.head.append(el);
  });
}

/** Load a script from the vendored copy, else the CDN; returns the directory used. */
async function loadVendored(file, cdnDir) {
  try {
    await loadScript(LOCAL_TF + file);
    return LOCAL_TF;
  } catch {
    await loadScript(cdnDir + file);
    return cdnDir;
  }
}

/**
 * TensorFlow.js (vendored, else the CDN). WebGL when there's a real GPU;
 * otherwise (software rendering: no GPU, or a blocklisted one) WebAssembly,
 * which is several times faster than WebGL emulated on the CPU. Resolves to
 * { tf, gpu } so the MediaPipe models can pick their delegate the same way.
 */
export function loadTf() {
  if (!tfPromise) {
    tfPromise = (async () => {
      if (!globalThis.tf?.loadGraphModel) await loadVendored('tf.min.js', CDN_TF);
      const { tf } = globalThis;
      // TF.js refuses WebGL contexts with a "major performance caveat" (software rendering).
      try {
        if (await tf.setBackend('webgl')) {
          await tf.ready();
          return { tf, gpu: true };
        }
      } catch {
        /* no usable GPU */
      }
      try {
        const dir = await loadVendored('tf-backend-wasm.min.js', CDN_TF_WASM);
        tf.wasm.setWasmPaths(dir === LOCAL_TF ? LOCAL_TF : CDN_TF_WASM);
        if (await tf.setBackend('wasm')) {
          await tf.ready();
          return { tf, gpu: false };
        }
      } catch (e) {
        console.warn('TensorFlow.js WebAssembly backend unavailable', e);
      }
      await tf.setBackend('cpu');
      await tf.ready();
      return { tf, gpu: false };
    })();
    tfPromise.catch(() => {
      tfPromise = null;
    });
  }
  return tfPromise;
}

export function squareFromBox(x, y, w, h, margin = CROP_MARGIN) {
  const half = (Math.max(w, h) / 2) * (1 + 2 * margin);
  return { x0: x + w / 2 - half, y0: y + h / 2 - half, side: 2 * half };
}

function squareFromPoints(pts, margin = POSE_MARGIN) {
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const half = (Math.max(maxX - minX, maxY - minY) / 2) * (1 + 2 * margin);
  return { x0: (minX + maxX) / 2 - half, y0: (minY + maxY) / 2 - half, side: 2 * half };
}

/** Draw the (possibly out-of-frame) square crop of src (W x H) into ctx at size x size, black outside the frame. */
function drawCrop(ctx, src, W, H, crop, size, mirror) {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, size, size);
  const s = size / crop.side;
  if (mirror) ctx.setTransform(-1, 0, 0, 1, size, 0);
  const sx = Math.max(0, crop.x0);
  const sy = Math.max(0, crop.y0);
  const ex = Math.min(W, crop.x0 + crop.side);
  const ey = Math.min(H, crop.y0 + crop.side);
  if (ex > sx && ey > sy) ctx.drawImage(src, sx, sy, ex - sx, ey - sy, (sx - crop.x0) * s, (sy - crop.y0) * s, (ex - sx) * s, (ey - sy) * s);
  ctx.restore();
}

function makeCanvas(size) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(size, size);
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  return c;
}

/**
 * Sub-pixel peak of a heatmap channel from its maximum `v` at (x, y) and the
 * neighbours [left, right, up, down] (NaN off the edge): a parabola through
 * each axis, as tools/swingdb/engine.py heatmap_peaks().
 */
export function refinePeak(x, y, v, [l, r, u, d], W, H) {
  let dx = 0;
  let dy = 0;
  if (x > 0 && x < W - 1) {
    const den = l - 2 * v + r;
    if (den < 0) dx = (0.5 * (l - r)) / den;
  }
  if (y > 0 && y < H - 1) {
    const den = u - 2 * v + d;
    if (den < 0) dy = (0.5 * (u - d)) / den;
  }
  return [x + dx, y + dy, v];
}

export class PoseEngine {
  constructor({ tf, movenet, effpose, pose, detector, referee, quality }) {
    Object.assign(this, { tf, movenet, effpose, pose, detector, referee, quality });
    this.mnCanvas = makeCanvas(MOVENET_SIZE);
    this.mnCtx = this.mnCanvas.getContext('2d', { willReadFrequently: true });
    this.epCanvas = makeCanvas(EP_SIZE);
    this.epCtx = this.epCanvas.getContext('2d', { willReadFrequently: true });
    this.mpCanvas = makeCanvas(MP_CROP);
    this.mpCtx = this.mpCanvas.getContext('2d');
    this.reset();
  }

  /** Forget the tracked hitter (call before a new clip). */
  reset() {
    this.crop = null; // square {x0, y0, side} around the hitter for this frame
    this.last = null; // last crop that held the hitter (to find them again after losing them)
    this.sinceCheck = 0;
  }

  people(src) {
    const r = this.detector.detect(src);
    return (r.detections || []).map((d) => {
      const b = d.boundingBox;
      return [b.originX, b.originY, b.width, b.height, d.categories?.[0]?.score ?? 0];
    });
  }

  /**
   * The hitter's box: near the current crop while tracking, near the last place
   * they were seen after losing them (never someone else across the frame),
   * else the most prominent person.
   */
  pick(boxes) {
    if (!boxes.length) return null;
    const ref = this.crop || this.last;
    if (ref) {
      const cx = ref.x0 + ref.side / 2;
      const cy = ref.y0 + ref.side / 2;
      const score = (b) => -Math.hypot(b[0] + b[2] / 2 - cx, b[1] + b[3] / 2 - cy) / ref.side
        - Math.abs(Math.log((Math.max(b[2], b[3]) * (1 + 2 * CROP_MARGIN)) / ref.side));
      const best = boxes.reduce((a, b) => (score(b) > score(a) ? b : a));
      return score(best) > (this.crop ? -0.6 : -1.2) ? best : null;
    }
    return boxes.reduce((a, b) => (b[2] * b[3] * Math.sqrt(b[4]) > a[2] * a[3] * Math.sqrt(a[4]) ? b : a));
  }

  async movenetAt(src, crop, mirror) {
    const { tf } = this;
    drawCrop(this.mnCtx, src, this.W, this.H, crop, MOVENET_SIZE, false);
    const input = tf.tidy(() => {
      let t = tf.browser.fromPixels(this.mnCanvas);
      if (mirror) t = tf.reverse(t, 1);
      return t.expandDims(0).toInt();
    });
    const res = this.movenet.execute(input);
    const kp = await res.data();
    tf.dispose([input, res]);
    const p = [];
    for (let j = 0; j < 17; j++) {
      const u = mirror ? 1 - kp[j * 3 + 1] : kp[j * 3 + 1];
      p.push([crop.x0 + u * crop.side, crop.y0 + kp[j * 3] * crop.side, kp[j * 3 + 2]]);
    }
    if (mirror) for (const [a, b] of FLIP_COCO) [p[a], p[b]] = [p[b], p[a]];
    return p;
  }

  async effposeAt(src, crop, mirror) {
    const { tf } = this;
    drawCrop(this.epCtx, src, this.W, this.H, crop, EP_SIZE, false);
    const S = EP_SIZE;
    // Peak of each of the 16 heatmaps and its 4 neighbours, found on the GPU (the
    // heatmaps are 368 x 368 x 16; reading them back whole would be slow).
    const [idxT, valT] = tf.tidy(() => {
      let t = tf.browser.fromPixels(this.epCanvas).toFloat().div(127.5).sub(1);
      if (mirror) t = tf.reverse(t, 1);
      const hm = this.effpose.execute(t.expandDims(0)).reshape([S * S, 16]).transpose(); // 16 x S*S
      const idx = hm.argMax(1).toInt();
      const offsets = tf.tensor1d([0, -1, 1, -S, S], 'int32');
      const pos = tf.clipByValue(idx.expandDims(1).add(offsets.expandDims(0)), 0, S * S - 1);
      const ch = tf.range(0, 16, 1, 'int32').expandDims(1).tile([1, 5]);
      return [idx, tf.gatherND(hm, tf.stack([ch, pos], 2))];
    });
    const [idx, vals] = await Promise.all([idxT.data(), valT.data()]);
    tf.dispose([idxT, valT]);
    let pk = [];
    for (let c = 0; c < 16; c++) {
      const x = idx[c] % S;
      const y = Math.floor(idx[c] / S);
      const v = vals.slice(c * 5, c * 5 + 5);
      pk.push(refinePeak(x, y, v[0], [v[1], v[2], v[3], v[4]], S, S));
    }
    if (mirror) pk = EP_FLIP.map((e) => [S - 1 - pk[e][0], pk[e][1], pk[e][2]]);
    const s = crop.side / S;
    const p = Array.from({ length: 17 }, () => [NaN, NaN, NaN]);
    for (const [c, e] of EP_TO_COCO) p[c] = [crop.x0 + pk[e][0] * s, crop.y0 + pk[e][1] * s, pk[e][2]];
    for (const c of [1, 2, 3, 4]) p[c] = [p[0][0], p[0][1], 0]; // no eyes or ears: at the head, with no confidence
    return p;
  }

  mediapipeAt(src, crop, mirror) {
    drawCrop(this.mpCtx, src, this.W, this.H, crop, MP_CROP, mirror);
    const r = this.pose.detect(this.mpCanvas);
    if (!r?.landmarks?.length) return null;
    let best = null;
    for (const lms of r.landmarks) {
      const q = lms.map((l) => [l.x * MP_CROP, l.y * MP_CROP, l.visibility ?? 0.5]);
      if (mirror) {
        for (const p of q) p[0] = MP_CROP - 1 - p[0];
        for (const [a, b] of MP_PAIRS) [q[a], q[b]] = [q[b], q[a]];
      }
      const mx = q.reduce((s, p) => s + p[0], 0) / q.length;
      const my = q.reduce((s, p) => s + p[1], 0) / q.length;
      const d = Math.hypot(mx - MP_CROP / 2, my - MP_CROP / 2);
      if (!best || d < best.d) best = { d, q };
    }
    const s = crop.side / MP_CROP;
    return best.q.map(([x, y, v]) => [crop.x0 + x * s, crop.y0 + y * s, v]);
  }

  /**
   * BODY_25 frame (pixels) of the hitter in `src` (a video element or canvas).
   * The models' answers the referee chose between stay in `lastCandidates`
   * (referee order, 17 COCO keypoints each, or null), for fixing by hand later.
   */
  async process(src) {
    this.lastCandidates = null;
    const W = src.videoWidth || src.width;
    const H = src.videoHeight || src.height;
    this.W = W;
    this.H = H;
    if (!this.crop || this.sinceCheck >= RECHECK) {
      const box = this.pick(this.people(src));
      if (box) this.crop = squareFromBox(box[0], box[1], box[2], box[3]);
      this.sinceCheck = 0;
    }
    if (!this.crop) return emptyFrame();
    this.sinceCheck++;
    const cands = [];
    const mp33 = [];
    for (const name of this.referee.candidates) {
      const [model, variant] = name.split('-');
      const mirror = variant === 'mirrored';
      if (name === 'movenet-zoom') {
        // A closer look at the arms, around where the answers so far put them.
        const zoom = zoomCrop(cands);
        const p = zoom ? await this.movenetAt(src, zoom, false) : null;
        cands.push(p && p.map((q, j) => (ZOOM_JOINTS.includes(j) ? q : [NaN, NaN, NaN])));
      } else if (model === 'mediapipe') {
        const q = this.mediapipeAt(src, this.crop, mirror);
        mp33.push(q);
        cands.push(q ? MP_TO_COCO.map((i) => q[i]) : null);
      } else if (model === 'movenet') {
        cands.push(await this.movenetAt(src, this.crop, mirror));
      } else if (model === 'efficientpose') {
        cands.push(await this.effposeAt(src, this.crop, mirror));
      } else {
        throw new Error(`unknown referee candidate ${name}`);
      }
    }
    const res = fuse(cands, this.referee);
    if (!res) {
      this.crop = null;
      return emptyFrame();
    }
    this.lastCandidates = cands;
    const ref = mp33.find((q) => q);
    const mpFrame = ref ? mediapipeToBody25(ref.map(([x, y, v]) => ({ x: x / W, y: y / H, visibility: v })), W, H) : null;
    const frame = fusedToBody25(res.keypoints, mpFrame);
    this.follow(res.keypoints);
    return frame;
  }

  follow(fused) {
    const good = fused.filter((p) => p[2] > 0.3 && Number.isFinite(p[0]));
    const core = [5, 6, 11, 12].map((j) => fused[j][2]);
    if (good.length < 6 || core.reduce((a, b) => a + b, 0) / 4 < 0.15) {
      this.crop = null; // lost: look for the hitter again next frame
      return;
    }
    const next = squareFromPoints(good);
    const prev = this.crop;
    this.last = prev;
    const side = Math.min(Math.max(next.side, prev.side * 0.8), prev.side * 1.25);
    const lim = 0.3 * prev.side;
    const clampShift = (v) => Math.max(-lim, Math.min(lim, v));
    const cx = prev.x0 + prev.side / 2 + clampShift(next.x0 + next.side / 2 - (prev.x0 + prev.side / 2));
    const cy = prev.y0 + prev.side / 2 + clampShift(next.y0 + next.side / 2 - (prev.y0 + prev.side / 2));
    this.crop = { x0: cx - side / 2, y0: cy - side / 2, side };
  }
}

/** Build an engine. `vision` is the MediaPipe tasks module + fileset from detector.js. */
export async function createEngine({ mod, vision, poseModel, detectorModel, quality = 'best', onStatus = () => {} }) {
  onStatus('Loading pose models…');
  const { tf, gpu } = await loadTf();
  const referee = await fetch(`${MODELS_DIR}referee-${quality}.json`).then((r) => {
    if (!r.ok) throw new Error(`Could not load the referee (${r.status})`);
    return r.json();
  });
  const uses = (model) => referee.candidates.some((c) => c.startsWith(model));
  const [movenet, effpose] = await Promise.all([
    tf.loadGraphModel(`${MODELS_DIR}movenet-thunder.json`),
    uses('efficientpose') ? tf.loadGraphModel(`${MODELS_DIR}efficientpose.json`) : null,
  ]);
  const make = async (delegate) => Promise.all([
    mod.PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: poseModel, delegate },
      runningMode: 'IMAGE',
      numPoses: 2,
      minPoseDetectionConfidence: 0.1,
      minPosePresenceConfidence: 0.1,
    }),
    mod.ObjectDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: detectorModel, delegate },
      runningMode: 'IMAGE',
      maxResults: 8,
      scoreThreshold: 0.25,
      categoryAllowlist: ['person'],
    }),
  ]);
  let pose;
  let detector;
  try {
    // Without a real GPU, MediaPipe's WebAssembly (CPU) path beats emulated WebGL too.
    [pose, detector] = await make(gpu ? 'GPU' : 'CPU');
  } catch {
    [pose, detector] = await make('CPU');
  }
  // Warm up the TensorFlow.js models so the first frame isn't slow.
  tf.tidy(() => movenet.execute(tf.zeros([1, MOVENET_SIZE, MOVENET_SIZE, 3], 'int32')));
  if (effpose) tf.tidy(() => effpose.execute(tf.zeros([1, EP_SIZE, EP_SIZE, 3])));
  return new PoseEngine({ tf, movenet, effpose, pose, detector, referee, quality });
}
