#!/usr/bin/env node
// Tiny static server for local development and tests.
//   node scripts/serve.mjs [--root dist] [--port 8080]
// With --root . (npm run dev) the /vendor paths are mapped to node_modules and
// the model cache so no build step is needed.
import { createServer } from 'node:http';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { join, extname, resolve, dirname, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, '..');
const args = process.argv.slice(2);
const opt = (n, d) => (args.includes(`--${n}`) ? args[args.indexOf(`--${n}`) + 1] : d);
const root = resolve(repo, opt('root', 'dist'));
const port = Number(opt('port', process.env.PORT || 8080));

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8', '.svg': 'image/svg+xml',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.wasm': 'application/wasm', '.task': 'application/octet-stream',
  '.tflite': 'application/octet-stream', '.bin': 'application/octet-stream',
  '.mp4': 'video/mp4', '.webm': 'video/webm', '.map': 'application/json',
};

const MAPS = [
  ['/vendor/mediapipe/wasm/', join(repo, 'node_modules/@mediapipe/tasks-vision/wasm/')],
  ['/vendor/mediapipe/', join(repo, 'node_modules/@mediapipe/tasks-vision/')],
  ['/vendor/models/', join(repo, '.cache/models/')],
  ['/vendor/tfjs/', join(repo, 'node_modules/@tensorflow/tfjs/dist/')],
  ['/vendor/tfjs/', join(repo, 'node_modules/@tensorflow/tfjs-backend-wasm/dist/')],
];

function resolvePath(urlPath) {
  const p = decodeURIComponent(urlPath.split('?')[0]);
  const candidates = [];
  candidates.push(join(root, normalize(p)));
  for (const [prefix, dir] of MAPS) if (p.startsWith(prefix)) candidates.push(join(dir, normalize(p.slice(prefix.length))));
  for (let c of candidates) {
    if (!c.startsWith(root) && !MAPS.some(([, d]) => c.startsWith(d))) continue;
    if (existsSync(c) && statSync(c).isDirectory()) c = join(c, 'index.html');
    if (existsSync(c) && statSync(c).isFile()) return c;
  }
  return null;
}

createServer((req, res) => {
  const file = resolvePath(req.url);
  if (!file) {
    res.writeHead(404, { 'content-type': 'text/plain' });
    res.end('Not found');
    return;
  }
  const size = statSync(file).size;
  const type = TYPES[extname(file)] || 'application/octet-stream';
  const range = req.headers.range && /bytes=(\d*)-(\d*)/.exec(req.headers.range);
  if (range) {
    const start = range[1] ? Number(range[1]) : 0;
    const end = range[2] ? Number(range[2]) : size - 1;
    res.writeHead(206, { 'content-type': type, 'content-range': `bytes ${start}-${end}/${size}`, 'accept-ranges': 'bytes', 'content-length': end - start + 1 });
    if (req.method === 'HEAD') return res.end();
    createReadStream(file, { start, end }).pipe(res);
    return;
  }
  res.writeHead(200, { 'content-type': type, 'content-length': size, 'accept-ranges': 'bytes', 'cache-control': 'no-cache' });
  if (req.method === 'HEAD') return res.end();
  createReadStream(file).pipe(res);
}).listen(port, () => console.log(`Serving ${root} at http://localhost:${port}/`));
