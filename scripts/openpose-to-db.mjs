#!/usr/bin/env node
// Convert a folder of OpenPose per-frame JSON files (the output of
// `openpose --write_json <dir>`) into a swing database entry and register it
// in data/pros/index.json.
//
//   node scripts/openpose-to-db.mjs <json-dir> --name "Player Name" --fps 60 \
//        [--id player-name] [--bats R|L] [--team ...] [--pitcher right|left] \
//        [--stance N] [--width 1280 --height 720] [--source URL]
//
// Anything not given (pitcher side, stance frame, phases) is detected
// automatically; check the result in builder.html before relying on it.
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { framesFromOpenPoseDocs } from '../src/core/body25.js';
import { detectPitcherSide, suggestStanceFrame, canonicalize, estimateSwingFps } from '../src/core/sequence.js';
import { detectPhases } from '../src/core/phases.js';
import { makeEntry, validateEntry, slugify } from '../src/core/db.js';

const args = process.argv.slice(2);
const dirArg = args.find((a) => !a.startsWith('--'));
const opt = (name, def) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
if (!dirArg || !opt('name') || !opt('fps')) {
  console.error('usage: node scripts/openpose-to-db.mjs <json-dir> --name "Player" --fps 60 [options]');
  process.exit(1);
}
const files = readdirSync(dirArg).filter((f) => f.endsWith('.json')).sort();
if (!files.length) throw new Error(`No .json files in ${dirArg}`);
const docs = files.map((f) => JSON.parse(readFileSync(join(dirArg, f), 'utf8')));
const frames = framesFromOpenPoseDocs(docs);
const fps = Number(opt('fps'));
// Detection runs on the swing's own clock, so slow motion needs no setting.
const swingFps = estimateSwingFps(frames, fps);
const stance = opt('stance') != null ? Number(opt('stance')) : suggestStanceFrame(frames, swingFps);
const pitcherSide = opt('pitcher') || detectPitcherSide(frames, stance, swingFps).side;
const { frames: canon } = canonicalize(frames, { pitcherSide, stanceIndex: stance, fps: swingFps });
const phases = detectPhases(canon, swingFps, stance);
const name = opt('name');
const id = opt('id', slugify(name));
const entry = validateEntry(
  makeEntry({
    id, name, team: opt('team', ''), bats: opt('bats', ''), source: opt('source', ''),
    fps, width: Number(opt('width', 0)) || undefined, height: Number(opt('height', 0)) || undefined,
    pitcherSide, stanceFrame: stance, phases, frames,
  }),
);
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dbDir = join(root, 'data', 'pros');
writeFileSync(join(dbDir, `${id}.json`), JSON.stringify(entry) + '\n');
const indexPath = join(dbDir, 'index.json');
const index = existsSync(indexPath) ? JSON.parse(readFileSync(indexPath, 'utf8')) : { schema: 'swing-db-index/v1', entries: [] };
index.entries = index.entries.filter((e) => e.id !== id);
index.entries.push({ id, file: `${id}.json`, name, bats: entry.bats });
writeFileSync(indexPath, JSON.stringify(index, null, 2) + '\n');
console.log(`wrote data/pros/${id}.json: ${frames.length} frames, pitcher ${pitcherSide}, stance ${stance}, phases ${JSON.stringify(phases)}`);
