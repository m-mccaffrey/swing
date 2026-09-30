// Pro swing database: loading, validation and preparation of entries.
//
// Entry format (swing-db/v1), one JSON file per swing in data/pros/:
// {
//   "schema": "swing-db/v1",
//   "id": "unique-id", "name": "Player Name", "team": "...", "bats": "R" | "L",
//   "keypointFormat": "BODY_25",
//   "fps": 60,                       // frames per second of the video
//   "speedFactor": 1,                // >1 if the source was slow motion
//   "image": { "width": 1280, "height": 720 },
//   "orientation": { "pitcherSide": "right" | "left" },
//   "stanceFrame": 12,
//   "phases": { "stance": 12, "load": 45, "footPlant": 61, "contact": 75, "extension": 80, "finish": 99 },
//   "frames": [ { "pose_keypoints_2d": [x0, y0, c0, ... x24, y24, c24] }, ... ]
// }
// `frames` may also hold raw OpenPose per-frame documents ({ "people": [...] })
// or bare keypoint arrays; COCO-18 keypoints are converted to BODY_25.

import { framesFromOpenPoseDocs, toBody25 } from './body25.js';
import { prepareSwing } from './compare.js';
import { PHASE_KEYS } from './phases.js';

export const SCHEMA = 'swing-db/v1';
const LOCAL_KEY = 'swingmatch.localPros.v1';

/** Extract BODY_25 frames from an entry, whatever the frame encoding. */
export function entryFrames(entry) {
  const frames = entry.frames;
  if (!Array.isArray(frames) || !frames.length) throw new Error('Entry has no frames');
  if (Array.isArray(frames[0])) return frames.map(toBody25);
  if (frames[0] && Array.isArray(frames[0].people)) return framesFromOpenPoseDocs(frames);
  return frames.map((f) => toBody25(f.pose_keypoints_2d || []));
}

/** Throw a readable error if an entry can't be used. Returns the entry. */
export function validateEntry(entry) {
  const where = entry?.id ? `Entry "${entry.id}"` : 'Entry';
  if (!entry || typeof entry !== 'object') throw new Error('Entry is not an object');
  if (entry.schema && entry.schema !== SCHEMA) throw new Error(`${where}: unsupported schema ${entry.schema}`);
  if (!entry.id) throw new Error(`${where}: missing "id"`);
  if (!entry.name) throw new Error(`${where}: missing "name"`);
  if (!(entry.fps > 0)) throw new Error(`${where}: "fps" must be a positive number`);
  const side = entry.orientation?.pitcherSide;
  if (side !== 'left' && side !== 'right') throw new Error(`${where}: orientation.pitcherSide must be "left" or "right"`);
  const frames = entryFrames(entry);
  if (frames.length < 10) throw new Error(`${where}: needs at least 10 frames`);
  if (!(entry.stanceFrame >= 0 && entry.stanceFrame < frames.length)) throw new Error(`${where}: stanceFrame out of range`);
  if (entry.phases) {
    let last = -1;
    for (const k of PHASE_KEYS) {
      const v = entry.phases[k];
      if (v == null) continue;
      if (!(v >= 0 && v < frames.length)) throw new Error(`${where}: phase "${k}" out of range`);
      if (v < last) throw new Error(`${where}: phases must be in order (${PHASE_KEYS.join(' → ')})`);
      last = v;
    }
  }
  return entry;
}

/** Canonicalize an entry and precompute everything the comparison needs. */
export function prepareEntry(entry) {
  validateEntry(entry);
  const frames = entryFrames(entry);
  const prep = prepareSwing({
    frames,
    fps: entry.fps,
    speedFactor: entry.speedFactor || 1,
    stanceIndex: entry.stanceFrame,
    pitcherSide: entry.orientation.pitcherSide,
    phases: entry.phases || null,
  });
  return {
    id: entry.id,
    name: entry.name,
    meta: {
      team: entry.team || '',
      bats: entry.bats || '',
      notes: entry.notes || '',
      source: entry.source || '',
      synthetic: !!entry.synthetic,
      image: entry.image || null,
    },
    entry,
    prep,
    stancePose: prep.stancePose,
  };
}

async function fetchJSON(url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`Could not load ${url} (${res.status})`);
  return res.json();
}

/**
 * Load every entry listed in data/pros/index.json plus entries saved in this
 * browser. Entries that fail to load are reported, not fatal.
 */
export async function loadDatabase(base = 'data/pros/') {
  const errors = [];
  const pros = [];
  let index = { entries: [] };
  try {
    index = await fetchJSON(`${base}index.json`);
  } catch (e) {
    errors.push(e.message);
  }
  const loaded = await Promise.all(
    (index.entries || []).map(async (meta) => {
      try {
        const entry = await fetchJSON(`${base}${meta.file}`);
        return prepareEntry(entry);
      } catch (e) {
        errors.push(`${meta.id || meta.file}: ${e.message}`);
        return null;
      }
    }),
  );
  for (const p of loaded) if (p) pros.push(p);
  for (const entry of getLocalEntries()) {
    try {
      const p = prepareEntry(entry);
      p.meta.local = true;
      if (!pros.some((q) => q.id === p.id)) pros.push(p);
    } catch (e) {
      errors.push(`Local entry ${entry.id}: ${e.message}`);
    }
  }
  return { pros, errors };
}

function storage() {
  try {
    return globalThis.localStorage || null;
  } catch {
    return null;
  }
}

/** Entries saved from the database builder in this browser. */
export function getLocalEntries() {
  try {
    const raw = storage()?.getItem(LOCAL_KEY);
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

export function saveLocalEntry(entry) {
  validateEntry(entry);
  const list = getLocalEntries().filter((e) => e.id !== entry.id);
  list.push(entry);
  try {
    storage()?.setItem(LOCAL_KEY, JSON.stringify(list));
    return true;
  } catch {
    return false;
  }
}

export function removeLocalEntry(id) {
  try {
    storage()?.setItem(LOCAL_KEY, JSON.stringify(getLocalEntries().filter((e) => e.id !== id)));
  } catch {
    /* storage unavailable */
  }
}

/** Build a database entry from analyzed frames (used by the builder and exports). */
export function makeEntry({ id, name, team = '', bats = '', notes = '', source = '', fps, speedFactor = 1, width, height, pitcherSide, stanceFrame, phases, frames }) {
  const round = (v) => Math.round(v * 100) / 100;
  return {
    schema: SCHEMA,
    id,
    name,
    team,
    bats,
    notes,
    source,
    keypointFormat: 'BODY_25',
    fps,
    speedFactor,
    image: width && height ? { width, height } : undefined,
    orientation: { pitcherSide },
    stanceFrame,
    phases,
    frames: frames.map((f) => ({ pose_keypoints_2d: f.map((v, i) => (i % 3 === 2 ? Math.round(v * 1000) / 1000 : round(v))) })),
  };
}

export function slugify(s) {
  return String(s || '')
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60);
}
