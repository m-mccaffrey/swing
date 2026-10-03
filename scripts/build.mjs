#!/usr/bin/env node
// Assemble the static site in dist/ for GitHub Pages:
//   * copies the app (HTML, CSS, JS, data, assets)
//   * copies the pose engine's models (models/: MoveNet for TensorFlow.js and
//     the referees; the ONNX copy is for the Python tool only)
//   * vendors the MediaPipe Tasks runtime and TensorFlow.js from node_modules
//   * downloads the MediaPipe models (cached in .cache/models) so the deployed
//     site doesn't depend on third-party CDNs at runtime.
// If a model can't be downloaded the app falls back to Google's model bucket.
import { cpSync, mkdirSync, rmSync, existsSync, writeFileSync, copyFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'dist');
const BUCKET = 'https://storage.googleapis.com/mediapipe-models';
const models = [
  ...['lite', 'full', 'heavy'].map((m) => [`pose_landmarker_${m}.task`, `${BUCKET}/pose_landmarker/pose_landmarker_${m}/float16/1/pose_landmarker_${m}.task`]),
  ['efficientdet_lite0.tflite', `${BUCKET}/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite`],
];
const skipModels = process.argv.includes('--no-models');

rmSync(dist, { recursive: true, force: true });
mkdirSync(dist, { recursive: true });
for (const item of ['index.html', 'builder.html', 'css', 'src', 'data', 'assets']) {
  cpSync(join(root, item), join(dist, item), { recursive: true });
}
cpSync(join(root, 'models'), join(dist, 'models'), { recursive: true, filter: (f) => !f.endsWith('.onnx') });
writeFileSync(join(dist, '.nojekyll'), '');

// MediaPipe runtime.
const mp = join(root, 'node_modules', '@mediapipe', 'tasks-vision');
if (existsSync(mp)) {
  const out = join(dist, 'vendor', 'mediapipe');
  mkdirSync(join(out, 'wasm'), { recursive: true });
  copyFileSync(join(mp, 'vision_bundle.mjs'), join(out, 'vision_bundle.mjs'));
  for (const f of readdirSync(join(mp, 'wasm'))) {
    if (f.includes('module_internal')) continue; // only the classic loaders are used
    copyFileSync(join(mp, 'wasm', f), join(out, 'wasm', f));
  }
  console.log('vendored @mediapipe/tasks-vision');
} else {
  console.warn('node_modules/@mediapipe/tasks-vision missing (run npm ci); the site will load MediaPipe from the CDN');
}

// TensorFlow.js (runs MoveNet): the library plus its WebAssembly backend for
// devices without a usable GPU.
const tfjs = join(root, 'node_modules', '@tensorflow');
const tfFiles = [
  ['tfjs', 'tf.min.js'],
  ['tfjs-backend-wasm', 'tf-backend-wasm.min.js'],
  ['tfjs-backend-wasm', 'tfjs-backend-wasm.wasm'],
  ['tfjs-backend-wasm', 'tfjs-backend-wasm-simd.wasm'],
  ['tfjs-backend-wasm', 'tfjs-backend-wasm-threaded-simd.wasm'],
];
if (tfFiles.every(([pkg, f]) => existsSync(join(tfjs, pkg, 'dist', f)))) {
  mkdirSync(join(dist, 'vendor', 'tfjs'), { recursive: true });
  for (const [pkg, f] of tfFiles) copyFileSync(join(tfjs, pkg, 'dist', f), join(dist, 'vendor', 'tfjs', f));
  console.log('vendored @tensorflow/tfjs');
} else {
  console.warn('node_modules/@tensorflow/tfjs* missing (run npm ci); the site will load TensorFlow.js from the CDN');
}

// MediaPipe models.
if (!skipModels) {
  const cache = join(root, '.cache', 'models');
  mkdirSync(cache, { recursive: true });
  const out = join(dist, 'vendor', 'models');
  mkdirSync(out, { recursive: true });
  for (const [file, url] of models) {
    const cached = join(cache, file);
    if (!existsSync(cached) || statSync(cached).size < 1e6) {
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        writeFileSync(cached, Buffer.from(await res.arrayBuffer()));
        console.log(`downloaded ${file}`);
      } catch (e) {
        console.warn(`could not download ${file} (${e.message}); the app will fetch it from Google at runtime`);
        continue;
      }
    }
    copyFileSync(cached, join(out, file));
  }
}
console.log(`built ${dist}`);
