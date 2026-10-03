// In-browser pose estimation, converted to OpenPose BODY_25 frames.
//
// "best" and "fast" run the pose engine (engine.js): person detector, crop
// tracking, MoveNet + MediaPipe and the learned referee, the same pipeline the
// Python tool uses for the pro database. "heavy"/"full"/"lite" are the older
// MediaPipe-only path (whole frame, VIDEO mode).
//
// The MediaPipe runtime and models are served from ./vendor when the site was
// built with `npm run build` (GitHub Pages workflow) and fall back to the
// public CDN / model bucket otherwise.

import { mediapipeToBody25, emptyFrame } from '../core/body25.js';
import { createEngine } from './engine.js';

export const MEDIAPIPE_VERSION = '1.0.1';
const LOCAL_MP = new URL('../../vendor/mediapipe/', import.meta.url).href;
const LOCAL_MODELS = new URL('../../vendor/models/', import.meta.url).href;
const CDN_MP = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/`;
const MODEL_BUCKET = 'https://storage.googleapis.com/mediapipe-models/';
const DETECTOR_FILE = 'efficientdet_lite0.tflite';

export const MODELS = {
  best: { label: 'Best (two models, cross-checked)', engine: true },
  fast: { label: 'Fast (two models)', engine: true },
  heavy: { label: 'MediaPipe only, heavy (older method)', file: 'pose_landmarker_heavy.task' },
  full: { label: 'MediaPipe only, full (older method)', file: 'pose_landmarker_full.task' },
  lite: { label: 'MediaPipe only, lite (older method)', file: 'pose_landmarker_lite.task' },
};

function remoteModelUrl(file) {
  if (file === DETECTOR_FILE) return `${MODEL_BUCKET}object_detector/efficientdet_lite0/float32/1/${file}`;
  return `${MODEL_BUCKET}pose_landmarker/${file.replace('.task', '')}/float16/1/${file}`;
}

let visionPromise = null;

/** Load the MediaPipe vision bundle (local first, then CDN). */
function loadVision() {
  if (!visionPromise) {
    visionPromise = (async () => {
      try {
        const mod = await import(`${LOCAL_MP}vision_bundle.mjs`);
        return { mod, wasm: `${LOCAL_MP}wasm` };
      } catch {
        const mod = await import(`${CDN_MP}vision_bundle.mjs`);
        return { mod, wasm: `${CDN_MP}wasm` };
      }
    })();
    visionPromise.catch(() => {
      visionPromise = null;
    });
  }
  return visionPromise;
}

async function modelPath(file) {
  const local = `${LOCAL_MODELS}${file}`;
  try {
    const res = await fetch(local, { method: 'HEAD' });
    if (res.ok) return local;
  } catch {
    /* fall through to remote */
  }
  return remoteModelUrl(file);
}

const landmarkers = new Map();

/**
 * Create (or reuse) a detector for `quality`: the pose engine for "best" and
 * "fast", else a MediaPipe PoseLandmarker in VIDEO mode. GPU first, then CPU.
 */
export async function getLandmarker(quality = 'best', onStatus = () => {}) {
  if (!MODELS[quality]) quality = 'best';
  if (landmarkers.has(quality)) return landmarkers.get(quality);
  onStatus('Loading pose models…');
  const { mod, wasm } = await loadVision();
  const vision = await mod.FilesetResolver.forVisionTasks(wasm);
  if (MODELS[quality].engine) {
    const [poseModel, detectorModel] = await Promise.all([modelPath(MODELS.heavy.file), modelPath(DETECTOR_FILE)]);
    const engine = await createEngine({ mod, vision, poseModel, detectorModel, quality, onStatus });
    const wrapped = { engine };
    landmarkers.set(quality, wrapped);
    return wrapped;
  }
  const modelAssetPath = await modelPath(MODELS[quality].file);
  const make = (delegate) =>
    mod.PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath, delegate },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.4,
      minPosePresenceConfidence: 0.4,
      minTrackingConfidence: 0.4,
    });
  let lm;
  try {
    lm = await make('GPU');
  } catch {
    lm = await make('CPU');
  }
  const wrapped = { landmarker: lm, lastTs: 0 };
  landmarkers.set(quality, wrapped);
  return wrapped;
}

/** Seek a video element and resolve once the frame is ready. */
export function seekVideo(video, t) {
  return new Promise((resolve) => {
    if (Math.abs(video.currentTime - t) < 1e-4 && video.readyState >= 2) {
      resolve();
      return;
    }
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      video.removeEventListener('seeked', onSeeked);
      resolve();
    };
    const onSeeked = () => {
      if (typeof video.requestVideoFrameCallback === 'function') {
        video.requestVideoFrameCallback(() => finish());
        setTimeout(finish, 80);
      } else {
        finish();
      }
    };
    video.addEventListener('seeked', onSeeked);
    video.currentTime = t;
    setTimeout(finish, 3000);
  });
}

/**
 * Estimate the native frame rate of a video by briefly playing it (muted)
 * and watching presented frames. Falls back to 30.
 */
export async function estimateVideoFps(video) {
  if (typeof video.requestVideoFrameCallback !== 'function') return 30;
  const times = [];
  const wasMuted = video.muted;
  video.muted = true;
  try {
    await new Promise((resolve) => {
      const stop = setTimeout(resolve, 900);
      const onFrame = (_now, meta) => {
        times.push(meta.mediaTime);
        if (times.length >= 12) {
          clearTimeout(stop);
          resolve();
        } else {
          video.requestVideoFrameCallback(onFrame);
        }
      };
      video.requestVideoFrameCallback(onFrame);
      video.play().catch(resolve);
    });
  } finally {
    video.pause();
    video.muted = wasMuted;
  }
  const d = [];
  for (let i = 1; i < times.length; i++) if (times[i] > times[i - 1]) d.push(times[i] - times[i - 1]);
  if (d.length < 3) return 30;
  d.sort((a, b) => a - b);
  const fps = 1 / d[Math.floor(d.length / 2)];
  // Snap to common rates.
  const common = [24, 25, 30, 48, 50, 60, 90, 120, 240];
  const best = common.reduce((a, b) => (Math.abs(b - fps) < Math.abs(a - fps) ? b : a));
  return Math.abs(best - fps) / best < 0.08 ? best : Math.round(fps);
}

/**
 * Run pose detection over [start, end] of a video at `fps` samples/second.
 * With the pose engine, `candidates[i]` holds the models' answers at frame i
 * (see PoseEngine.process), named by `candidateNames`.
 * @returns {Promise<{frames:number[][], times:number[], fps:number, width:number, height:number, candidates?:Array, candidateNames?:string[]}>}
 */
export async function analyzeVideo(video, detector, { fps = 30, start = 0, end = video.duration, onProgress, signal } = {}) {
  const width = video.videoWidth;
  const height = video.videoHeight;
  const dur = Math.max(0, Math.min(end, video.duration) - start);
  const count = Math.max(1, Math.floor(dur * fps) + 1);
  const frames = [];
  const times = [];
  video.pause();
  const { engine } = detector;
  const candidates = engine ? [] : null;
  engine?.reset(); // a new clip: find the hitter again
  // MediaPipe needs strictly increasing timestamps per landmarker instance.
  const base = (detector.lastTs || 0) + 1000;
  for (let k = 0; k < count; k++) {
    if (signal?.aborted) throw new DOMException('Analysis cancelled', 'AbortError');
    const t = Math.min(video.duration - 1e-3, start + k / fps);
    await seekVideo(video, t);
    const ts = base + Math.round((t - start) * 1000);
    let frame = emptyFrame();
    try {
      if (engine) {
        engine.lastCandidates = null;
        frame = await engine.process(video);
      } else {
        const res = detector.landmarker.detectForVideo(video, ts);
        const lm = res?.landmarks?.[0];
        if (lm) frame = mediapipeToBody25(lm, width, height);
      }
    } catch (e) {
      console.warn('Pose detection failed on frame', k, e);
    }
    detector.lastTs = ts;
    // Kept (to 0.01 px) so a joint fixed by hand can be re-picked from the other answers nearby.
    candidates?.push(engine.lastCandidates?.map((c) => c && c.map((q) => q.map((v) => Math.round(v * 100) / 100))) ?? null);
    frames.push(frame);
    times.push(t);
    onProgress?.(k + 1, count, frame, t);
    // Yield so the page stays responsive.
    if (k % 4 === 3) await new Promise((r) => setTimeout(r, 0));
  }
  if (engine) return { frames, times, fps, width, height, candidates, candidateNames: engine.referee.candidates };
  return { frames, times, fps, width, height };
}
