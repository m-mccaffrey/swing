// In-browser pose estimation with MediaPipe Pose Landmarker, converted to
// OpenPose BODY_25 frames. The MediaPipe runtime and models are served from
// ./vendor when the site was built with `npm run build` (GitHub Pages
// workflow) and fall back to the public CDN / model bucket otherwise.

import { mediapipeToBody25, emptyFrame } from '../core/body25.js';

export const MEDIAPIPE_VERSION = '1.0.1';
const LOCAL_MP = new URL('../../vendor/mediapipe/', import.meta.url).href;
const LOCAL_MODELS = new URL('../../vendor/models/', import.meta.url).href;
const CDN_MP = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/`;
const MODEL_BUCKET = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/';

export const MODELS = {
  lite: { label: 'Fast (lite)', file: 'pose_landmarker_lite.task' },
  full: { label: 'Balanced (full)', file: 'pose_landmarker_full.task' },
  heavy: { label: 'Most accurate (heavy, slow)', file: 'pose_landmarker_heavy.task' },
};

function remoteModelUrl(key) {
  return `${MODEL_BUCKET}pose_landmarker_${key}/float16/1/${MODELS[key].file}`;
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

async function modelPath(key) {
  const local = `${LOCAL_MODELS}${MODELS[key].file}`;
  try {
    const res = await fetch(local, { method: 'HEAD' });
    if (res.ok) return local;
  } catch {
    /* fall through to remote */
  }
  return remoteModelUrl(key);
}

const landmarkers = new Map();

/**
 * Create (or reuse) a PoseLandmarker in VIDEO mode. Tries the GPU delegate
 * first and falls back to CPU.
 */
export async function getLandmarker(quality = 'full', onStatus = () => {}) {
  if (landmarkers.has(quality)) return landmarkers.get(quality);
  onStatus('Loading pose model…');
  const { mod, wasm } = await loadVision();
  const vision = await mod.FilesetResolver.forVisionTasks(wasm);
  const modelAssetPath = await modelPath(quality);
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
 * @returns {Promise<{frames:number[][], times:number[], fps:number, width:number, height:number}>}
 */
export async function analyzeVideo(video, detector, { fps = 30, start = 0, end = video.duration, onProgress, signal } = {}) {
  const width = video.videoWidth;
  const height = video.videoHeight;
  const dur = Math.max(0, Math.min(end, video.duration) - start);
  const count = Math.max(1, Math.floor(dur * fps) + 1);
  const frames = [];
  const times = [];
  video.pause();
  // MediaPipe needs strictly increasing timestamps per landmarker instance.
  const base = detector.lastTs + 1000;
  for (let k = 0; k < count; k++) {
    if (signal?.aborted) throw new DOMException('Analysis cancelled', 'AbortError');
    const t = Math.min(video.duration - 1e-3, start + k / fps);
    await seekVideo(video, t);
    const ts = base + Math.round((t - start) * 1000);
    let frame = emptyFrame();
    try {
      const res = detector.landmarker.detectForVideo(video, ts);
      const lm = res?.landmarks?.[0];
      if (lm) frame = mediapipeToBody25(lm, width, height);
    } catch (e) {
      console.warn('Pose detection failed on frame', k, e);
    }
    detector.lastTs = ts;
    frames.push(frame);
    times.push(t);
    onProgress?.(k + 1, count, frame, t);
    // Yield so the page stays responsive.
    if (k % 4 === 3) await new Promise((r) => setTimeout(r, 0));
  }
  return { frames, times, fps, width, height };
}
