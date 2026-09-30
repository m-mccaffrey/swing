#!/usr/bin/env node
// Assemble the static site in dist/ for GitHub Pages:
//   * copies the app (HTML, CSS, JS, data, assets)
//   * vendors the MediaPipe Tasks runtime from node_modules
//   * downloads the pose models (cached in .cache/models) so the deployed
//     site doesn't depend on third-party CDNs at runtime.
// If a model can't be downloaded the app falls back to Google's model bucket.
import { cpSync, mkdirSync, rmSync, existsSync, writeFileSync, copyFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'dist');
const models = ['lite', 'full', 'heavy'];
const skipModels = process.argv.includes('--no-models');

rmSync(dist, { recursive: true, force: true });
mkdirSync(dist, { recursive: true });
for (const item of ['index.html', 'builder.html', 'css', 'src', 'data', 'assets']) {
  cpSync(join(root, item), join(dist, item), { recursive: true });
}
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

// Pose models.
if (!skipModels) {
  const cache = join(root, '.cache', 'models');
  mkdirSync(cache, { recursive: true });
  const out = join(dist, 'vendor', 'models');
  mkdirSync(out, { recursive: true });
  for (const m of models) {
    const file = `pose_landmarker_${m}.task`;
    const cached = join(cache, file);
    if (!existsSync(cached) || statSync(cached).size < 1e6) {
      const url = `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${m}/float16/1/${file}`;
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
