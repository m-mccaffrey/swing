#!/usr/bin/env node
// Regenerate the synthetic placeholder entries in data/pros/ and make sure
// they are listed in data/pros/index.json (real entries are left untouched).
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ARCHETYPES, archetypeEntry } from '../src/core/synth.js';
import { validateEntry } from '../src/core/db.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dir = join(root, 'data', 'pros');
const indexPath = join(dir, 'index.json');
const index = existsSync(indexPath) ? JSON.parse(readFileSync(indexPath, 'utf8')) : { schema: 'swing-db-index/v1', entries: [] };

for (const id of Object.keys(ARCHETYPES)) {
  const entry = validateEntry(archetypeEntry(id));
  const file = `${id}.json`;
  writeFileSync(join(dir, file), JSON.stringify(entry) + '\n');
  const meta = { id, file, name: entry.name, bats: entry.bats, synthetic: true };
  const i = index.entries.findIndex((e) => e.id === id);
  if (i >= 0) index.entries[i] = meta;
  else index.entries.push(meta);
  console.log(`wrote data/pros/${file} (${entry.frames.length} frames)`);
}
writeFileSync(indexPath, JSON.stringify(index, null, 2) + '\n');
console.log(`index: ${index.entries.length} entries`);
