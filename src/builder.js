// Database builder: video / OpenPose JSON / existing entry → reviewed
// swing-db/v1 entry.

import { framesFromOpenPoseDocs, NUM_KP } from './core/body25.js';
import { suggestStanceFrame, detectPitcherSide, canonicalize } from './core/sequence.js';
import { detectPhases, PHASES, sanitizePhases } from './core/phases.js';
import { makeEntry, validateEntry, entryFrames, saveLocalEntry, getLocalEntries, removeLocalEntry, slugify } from './core/db.js';
import { getLandmarker, analyzeVideo, estimateVideoFps, seekVideo } from './pose/detector.js';
import { Stage } from './ui/stage.js';
import { drawSkeleton, drawScene, canonicalBounds, fitCanvas } from './ui/draw.js';

const $ = (id) => document.getElementById(id);

const st = {
  video: null,
  videoFps: 30,
  frames: null,
  times: null,
  fps: 30,
  width: 0,
  height: 0,
  stance: 0,
  phases: null,
  pitcherSide: 'right',
  index: 0,
  canon: null,
  bounds: null,
  stage: null,
};

function error(msg) {
  $('error').textContent = msg || '';
  $('error').hidden = !msg;
}

function readJSON(file) {
  return file.text().then((t) => JSON.parse(t));
}

function download(name, data) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data)], { type: 'application/json' }));
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(a.href);
    a.remove();
  }, 1000);
}

// ------------------------------------------------------------ sources

async function onVideo(file) {
  if (!file) return;
  error('');
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.src = URL.createObjectURL(file);
  try {
    await new Promise((res, rej) => {
      video.addEventListener('loadeddata', res, { once: true });
      video.addEventListener('error', () => rej(new Error('This browser cannot play that video.')), { once: true });
    });
  } catch (e) {
    error(e.message);
    return;
  }
  st.video = video;
  $('b-start').value = '0';
  $('b-end').value = Math.min(video.duration, 10).toFixed(1);
  $('b-analyze').disabled = false;
  st.videoFps = await estimateVideoFps(video).catch(() => 30);
  $('m-fps').value = String(st.videoFps);
  $('b-progress-text').textContent = `${file.name}: ${video.videoWidth}×${video.videoHeight}, ${video.duration.toFixed(1)} s, about ${st.videoFps} fps.`;
  $('b-progress').hidden = false;
  $('b-progress').querySelector('span').style.width = '0%';
  if (!$('m-source').value) $('m-source').value = file.name;
}

async function analyze() {
  const v = st.video;
  if (!v) return;
  error('');
  $('b-analyze').disabled = true;
  const bar = $('b-progress').querySelector('span');
  try {
    const det = await getLandmarker($('b-quality').value, (s) => ($('b-progress-text').textContent = s));
    const afps = $('b-afps').value === 'auto' ? Math.min(60, st.videoFps) : Number($('b-afps').value);
    const res = await analyzeVideo(v, det, {
      fps: afps,
      start: Number($('b-start').value) || 0,
      end: Number($('b-end').value) || v.duration,
      onProgress: (k, n) => {
        bar.style.width = `${(100 * k) / n}%`;
        $('b-progress-text').textContent = `Frame ${k} of ${n}`;
      },
    });
    $('m-fps').value = String(afps);
    setFrames({ frames: res.frames, times: res.times, fps: afps, width: res.width, height: res.height });
  } catch (e) {
    console.error(e);
    error(`Pose detection failed: ${e.message || e}`);
  } finally {
    $('b-analyze').disabled = false;
  }
}

async function onOpenPose(files) {
  if (!files?.length) return;
  error('');
  try {
    const sorted = [...files].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    const docs = await Promise.all(sorted.map(readJSON));
    const frames = framesFromOpenPoseDocs(docs);
    const fps = Number($('b-op-fps').value) || 30;
    $('m-fps').value = String(fps);
    st.video = null;
    setFrames({ frames, fps, ...imageSizeFromFrames(frames) });
  } catch (e) {
    error(`Could not read the OpenPose files: ${e.message}`);
  }
}

async function onEntry(file) {
  if (!file) return;
  error('');
  try {
    const entry = validateEntry(await readJSON(file));
    const frames = entryFrames(entry);
    $('m-name').value = entry.name || '';
    $('m-id').value = entry.id || '';
    $('m-team').value = entry.team || '';
    $('m-bats').value = entry.bats || 'R';
    $('m-source').value = entry.source || '';
    $('m-notes').value = entry.notes || '';
    $('m-fps').value = String(entry.fps);
    $('m-speed').value = String(entry.speedFactor || 1);
    st.video = null;
    const size = entry.image?.width ? { width: entry.image.width, height: entry.image.height } : imageSizeFromFrames(frames);
    setFrames({ frames, fps: entry.fps, ...size, stance: entry.stanceFrame, phases: entry.phases, pitcherSide: entry.orientation.pitcherSide });
  } catch (e) {
    error(`Could not load that entry: ${e.message}`);
  }
}

function imageSizeFromFrames(frames) {
  let maxX = 0;
  let maxY = 0;
  for (const f of frames) {
    for (let j = 0; j < NUM_KP; j++) {
      if (f[j * 3 + 2] > 0.05) {
        maxX = Math.max(maxX, f[j * 3]);
        maxY = Math.max(maxY, f[j * 3 + 1]);
      }
    }
  }
  // OpenPose pixel coordinates: assume a standard frame that contains them.
  if (maxX <= 1.01 && maxY <= 1.01) return { width: 1, height: 1 };
  const std = [[1280, 720], [1920, 1080], [3840, 2160], [720, 1280], [1080, 1920]];
  const fit = std.find(([w, h]) => w >= maxX && h >= maxY);
  return fit ? { width: fit[0], height: fit[1] } : { width: Math.ceil(maxX * 1.05), height: Math.ceil(maxY * 1.05) };
}

// ------------------------------------------------------------ editing

function realFps() {
  return st.fps * (Number($('m-speed').value) || 1);
}

function setFrames({ frames, times = null, fps, width, height, stance = null, phases = null, pitcherSide = null }) {
  if (!frames?.length) {
    error('No frames found.');
    return;
  }
  st.frames = frames;
  st.fps = fps;
  st.times = times || frames.map((_, i) => i / fps);
  st.width = width;
  st.height = height;
  st.stance = stance ?? suggestStanceFrame(frames, realFps());
  st.pitcherSide = pitcherSide ?? detectPitcherSide(frames, st.stance, realFps()).side;
  document.querySelector(`#b-side input[value="${st.pitcherSide}"]`).checked = true;
  recanon();
  st.phases = phases ? sanitizePhases({ ...phases, stance: st.stance }, frames.length) : detectPhases(st.canon, realFps(), st.stance);
  st.stage?.destroy();
  st.stage = new Stage($('b-stage'), { video: st.video, width, height });
  $('b-slider').max = String(frames.length - 1);
  $('b-edit').hidden = false;
  $('b-meta').hidden = false;
  renderPhases();
  updateSnippet();
  goTo(st.stance);
  $('b-edit').scrollIntoView({ behavior: 'smooth' });
}

function recanon() {
  try {
    st.canon = canonicalize(st.frames, { pitcherSide: st.pitcherSide, stanceIndex: st.stance, fps: realFps() }).frames;
    st.bounds = canonicalBounds([st.canon]);
  } catch (e) {
    st.canon = null;
    error(`${e.message}. Choose a stance frame where the whole body is visible.`);
  }
}

function renderPhases() {
  const box = $('b-phases');
  box.textContent = '';
  for (const ph of PHASES) {
    const name = document.createElement('span');
    name.textContent = ph.label;
    const val = document.createElement('span');
    val.className = 'pf';
    val.textContent = `#${st.phases[ph.key]}`;
    const go = document.createElement('button');
    go.className = 'btn ghost';
    go.type = 'button';
    go.textContent = 'Go';
    go.addEventListener('click', () => goTo(st.phases[ph.key]));
    const set = document.createElement('button');
    set.className = 'btn';
    set.type = 'button';
    set.textContent = 'Set';
    set.setAttribute('aria-label', `Set ${ph.label} to current frame`);
    set.addEventListener('click', () => {
      st.phases[ph.key] = st.index;
      if (ph.key === 'stance') {
        st.stance = st.index;
        recanon();
      }
      renderPhases();
      goTo(st.index);
    });
    box.append(name, val, go, set);
  }
  let last = -1;
  const bad = PHASES.some((p) => {
    const v = st.phases[p.key];
    const out = v < last;
    last = v;
    return out;
  });
  if (bad) error('Phase frames must be in order: stance, load, foot plant, contact, extension, finish.');
  else if ($('error').textContent.startsWith('Phase frames')) error('');
}

async function goTo(i) {
  i = Math.max(0, Math.min(st.frames.length - 1, i));
  st.index = i;
  $('b-slider').value = String(i);
  const phase = PHASES.filter((p) => st.phases[p.key] === i).map((p) => p.label);
  $('b-frame-label').textContent = `Frame ${i} of ${st.frames.length - 1} · ${st.times[i].toFixed(2)} s${phase.length ? ` · ${phase.join(', ')}` : ''}`;
  if (st.video) await seekVideo(st.video, st.times[i]);
  if (st.index !== i) return;
  st.stage.draw((ctx, map, u) => drawSkeleton(ctx, st.frames[i], { map, lineWidth: 3 * u, radius: 3 * u, outline: 'rgba(0,0,0,0.45)' }));
  const c = $('b-canon');
  const w = c.clientWidth || 400;
  const h = c.clientHeight || 300;
  const ctx = fitCanvas(c, w, h);
  if (st.canon) drawScene(ctx, w, h, { bounds: st.bounds, user: st.canon[i], userTrail: { frames: st.canon, from: st.phases.stance, to: st.phases.finish } });
}

function currentEntry() {
  const name = $('m-name').value.trim();
  if (!name) throw new Error('Enter the player name first.');
  const id = $('m-id').value.trim() || slugify(name);
  const phases = { ...st.phases, stance: st.stance };
  return validateEntry(
    makeEntry({
      id,
      name,
      team: $('m-team').value.trim(),
      bats: $('m-bats').value,
      notes: $('m-notes').value.trim(),
      source: $('m-source').value.trim(),
      fps: Number($('m-fps').value) || st.fps,
      speedFactor: Number($('m-speed').value) || 1,
      width: st.width,
      height: st.height,
      pitcherSide: st.pitcherSide,
      stanceFrame: st.stance,
      phases,
      frames: st.frames,
    }),
  );
}

function updateSnippet() {
  const name = $('m-name').value.trim() || 'Player Name';
  const id = $('m-id').value.trim() || slugify(name);
  $('b-snippet').textContent = JSON.stringify({ id, file: `${id}.json`, name, bats: $('m-bats').value }, null, 0);
}

function renderLocal() {
  const ul = $('b-local');
  ul.textContent = '';
  const list = getLocalEntries();
  if (!list.length) {
    const li = document.createElement('li');
    li.className = 'muted small';
    li.textContent = 'Nothing saved yet.';
    ul.appendChild(li);
    return;
  }
  for (const e of list) {
    const li = document.createElement('li');
    li.className = 'fb';
    const name = document.createElement('span');
    name.className = 'fb-msg';
    name.textContent = `${e.name} (${e.id})`;
    const rm = document.createElement('button');
    rm.className = 'btn ghost';
    rm.type = 'button';
    rm.textContent = 'Remove';
    rm.addEventListener('click', () => {
      removeLocalEntry(e.id);
      renderLocal();
    });
    li.append(rm, name);
    ul.appendChild(li);
  }
}

// ------------------------------------------------------------ wiring

$('b-video').addEventListener('change', (e) => onVideo(e.target.files[0]));
$('b-openpose').addEventListener('change', (e) => onOpenPose(e.target.files));
$('b-entry').addEventListener('change', (e) => onEntry(e.target.files[0]));
$('b-analyze').addEventListener('click', analyze);
$('b-slider').addEventListener('input', () => goTo(Number($('b-slider').value)));
$('b-prev').addEventListener('click', () => goTo(st.index - 1));
$('b-next').addEventListener('click', () => goTo(st.index + 1));
$('b-redetect').addEventListener('click', () => {
  recanon();
  if (st.canon) st.phases = detectPhases(st.canon, realFps(), st.stance);
  renderPhases();
  goTo(st.index);
});
document.querySelectorAll('#b-side input').forEach((r) =>
  r.addEventListener('change', () => {
    st.pitcherSide = r.value;
    recanon();
    goTo(st.index);
  }),
);
for (const id of ['m-name', 'm-id', 'm-bats']) $(id).addEventListener('input', updateSnippet);
$('m-fps').addEventListener('change', () => {
  if (st.frames && !st.video) {
    st.fps = Number($('m-fps').value) || st.fps;
    st.times = st.frames.map((_, i) => i / st.fps);
  }
});
$('b-download').addEventListener('click', () => {
  try {
    const e = currentEntry();
    error('');
    download(`${e.id}.json`, e);
  } catch (err) {
    error(err.message);
  }
});
$('b-save').addEventListener('click', () => {
  try {
    const e = currentEntry();
    error('');
    const ok = saveLocalEntry(e);
    $('b-saved').textContent = ok
      ? `Saved “${e.name}” in this browser. It now appears in the Analyze page’s rankings.`
      : 'Could not save (browser storage is full or disabled). Download the file instead.';
    renderLocal();
  } catch (err) {
    error(err.message);
  }
});
renderLocal();
window.swingBuilder = { st, setFrames };
