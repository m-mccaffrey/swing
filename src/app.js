// Main page controller: upload → pose detection → stance pick → stance
// match → swing comparison → feedback.

import { loadDatabase, makeEntry } from './core/db.js';
import { prepareSwing, compareSwing } from './core/compare.js';
import { detectPitcherSide, suggestStanceFrame, canonToImage, detectionCoverage, estimateSwingFps } from './core/sequence.js';
import { rankStances } from './core/match.js';
import { evaluateFeedback, summarize, formatValue, formatDelta, TORSO_TO_HEIGHT } from './core/feedback.js';
import { PHASES, phaseLabel, pickSwingFps } from './core/phases.js';
import { demoUserSwing } from './core/synth.js';
import { NUM_KP, kx, ky, pinned } from './core/body25.js';
import { applyPins, nearestJoint, normalizePins } from './core/fix.js';
import { getLandmarker, analyzeVideo, estimateVideoFps } from './pose/detector.js';
import { Stage, FramePlayer } from './ui/stage.js';
import { drawSkeleton, drawScene, canonicalBounds, fitCanvas, cssVar, drawHighlights, drawPoints } from './ui/draw.js';
import { lineChart } from './ui/charts.js';
import * as library from './ui/library.js';

const $ = (id) => document.getElementById(id);

const state = {
  pros: [],
  video: null,
  videoUrl: null,
  videoFps: 30,
  isDemo: false,
  analysis: null, // { frames, times, fps, width, height, candidates?, candidateNames? }
  pins: [], // joints fixed by hand: {frame, joint, x, y} in video pixels
  fixedFrames: null, // analysis frames with the pins applied (cache)
  fixing: false, // the stance stage is in "fix a joint" mode
  fixFrame: 0, // frame shown while fixing (the stance pick stays put)
  drag: null, // joint being dragged: {joint, x, y}
  swingFps: 30, // the swing's own clock (frames per swing-second)
  userBeats: null, // beats the user adjusted by hand (frame indices), or null for automatic
  beatKey: 'contact', // beat selected in the beat editor
  stanceIndex: 0,
  suggestedStance: 0,
  sideDetect: null,
  sideManual: false, // the user chose the pitcher side (else it follows the stance pick)
  pitcherSide: 'right',
  user: null,
  ranking: [],
  pro: null,
  comparison: null,
  items: [],
  evidence: null, // feedback item whose evidence is on screen
  file: null, // the uploaded File (saved with the swing)
  swingId: null, // id of the saved swing being shown
  thumbPending: false,
  abort: null,
};

const stages = {};
let player = null;

// ---------------------------------------------------------------- helpers

function show(id, visible = true) {
  $(id).hidden = !visible;
}

function scrollTo(id) {
  $(id).scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function heightInches() {
  const ft = Number($('set-ft').value) || 5;
  const inch = Number($('set-in').value) || 0;
  return Math.max(36, Math.min(96, ft * 12 + inch));
}

/** The analysis frames with the joints fixed by hand: what everything downstream uses. */
function userFrames() {
  const a = state.analysis;
  if (!state.pins.length) return a.frames;
  if (!state.fixedFrames) state.fixedFrames = applyPins(a.frames, state.pins, { candidates: a.candidates, fps: a.fps });
  return state.fixedFrames;
}

function torsoIn() {
  return heightInches() * TORSO_TO_HEIGHT;
}

function showError(msg) {
  let el = $('error-banner');
  if (!el) {
    el = document.createElement('p');
    el.id = 'error-banner';
    el.className = 'notice error';
    el.setAttribute('role', 'alert');
    document.querySelector('.hero').appendChild(el);
  }
  el.textContent = msg;
  el.hidden = !msg;
  if (msg) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function download(name, data) {
  const blob = new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(a.href);
    a.remove();
  }, 1000);
}

function stageLabel() {
  return state.isDemo ? 'Demo swing (keypoints only, no video)' : 'Saved swing (video not stored, keypoints only)';
}

function fmtTime(t) {
  return `${t.toFixed(2)} s`;
}

// ---------------------------------------------------------------- database

async function initDatabase() {
  const { pros, errors } = await loadDatabase('data/pros/');
  state.pros = pros;
  const synth = pros.filter((p) => p.meta.synthetic).length;
  const local = pros.filter((p) => p.meta.local).length;
  let msg = `${pros.length} pro swing${pros.length === 1 ? '' : 's'} in the database`;
  const parts = [];
  if (synth) parts.push(`${synth} synthetic placeholder${synth === 1 ? '' : 's'}`);
  if (local) parts.push(`${local} saved in this browser`);
  if (parts.length) msg += ` (${parts.join(', ')})`;
  msg += '.';
  if (errors.length) msg += ` ${errors.length} entr${errors.length === 1 ? 'y' : 'ies'} could not be loaded.`;
  $('db-status').textContent = msg;
  if (errors.length) console.warn('Database load errors:', errors);
}

// ---------------------------------------------------------------- step 1: upload

function resetFlow() {
  player?.pause();
  state.abort?.abort();
  for (const id of ['step-clip', 'step-progress', 'step-stance', 'results']) show(id, false);
  show('step-upload');
  showError('');
  renderSavedList();
}

async function onFile(file) {
  if (!file) return;
  if (!file.type.startsWith('video/') && !/\.(mp4|mov|webm|m4v|avi|mkv)$/i.test(file.name)) {
    showError('That file does not look like a video. Please choose an MP4, MOV or WebM file.');
    return;
  }
  resetFlow();
  if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
  state.isDemo = false;
  state.file = file;
  state.swingId = null;
  state.userBeats = null;
  state.videoUrl = URL.createObjectURL(file);
  const video = document.createElement('video');
  video.preload = 'auto';
  video.muted = true;
  video.playsInline = true;
  video.src = state.videoUrl;
  state.video = video;
  try {
    await new Promise((resolve, reject) => {
      video.addEventListener('loadeddata', resolve, { once: true });
      video.addEventListener('error', () => reject(new Error('This browser cannot play that video format. Try an MP4 (H.264) file.')), { once: true });
    });
  } catch (e) {
    showError(e.message);
    return;
  }
  stages.clip = new Stage($('clip-stage'), { video, width: video.videoWidth, height: video.videoHeight });
  video.controls = true;
  const dur = video.duration;
  $('clip-start').value = '0';
  $('clip-end').value = Math.min(dur, 12).toFixed(1);
  $('clip-start').max = dur.toFixed(1);
  $('clip-end').max = dur.toFixed(1);
  $('clip-info').textContent = `${file.name}: ${video.videoWidth}×${video.videoHeight}, ${dur.toFixed(1)} s.`;
  show('step-upload', false);
  show('step-clip');
  scrollTo('step-clip');
  state.videoFps = await estimateVideoFps(video).catch(() => 30);
  video.currentTime = 0;
  $('clip-info').textContent += ` About ${state.videoFps} fps.${dur > 12 ? ' Long video: only the first 12 s are selected; adjust below.' : ''}`;
}

function analysisFps() {
  const v = $('set-fps').value;
  if (v !== 'auto') return Number(v);
  return Math.min(60, Math.max(15, state.videoFps || 30));
}

async function runAnalysis() {
  const video = state.video;
  const start = Math.max(0, Number($('clip-start').value) || 0);
  const end = Math.min(video.duration, Number($('clip-end').value) || video.duration);
  if (!(end - start >= 0.5)) {
    showError('Please select at least half a second of video.');
    return;
  }
  showError('');
  video.controls = false;
  video.pause();
  show('step-clip', false);
  show('step-progress');
  scrollTo('step-progress');
  const abort = new AbortController();
  state.abort = abort;
  const canvas = $('progress-canvas');
  const setProgress = (p) => {
    $('progress-bar').firstElementChild.style.width = `${Math.round(p * 100)}%`;
    $('progress-bar').setAttribute('aria-valuenow', String(Math.round(p * 100)));
  };
  setProgress(0);
  try {
    const quality = $('set-quality').value;
    const detector = await getLandmarker(quality, (s) => ($('progress-status').textContent = s));
    const fps = analysisFps();
    $('progress-status').textContent = 'Detecting your pose in each frame…';
    const t0 = performance.now();
    const res = await analyzeVideo(video, detector, {
      fps,
      start,
      end,
      signal: abort.signal,
      onProgress: (k, n, frame) => {
        setProgress(k / n);
        const el = (performance.now() - t0) / 1000;
        const eta = (el / k) * (n - k);
        $('progress-detail').textContent = `Frame ${k} of ${n} · about ${Math.ceil(eta)} s left`;
        const w = canvas.clientWidth || 320;
        const h = canvas.clientHeight || 320;
        const ctx = fitCanvas(canvas, w, h);
        const s = Math.min(w / video.videoWidth, h / video.videoHeight);
        const ox = (w - video.videoWidth * s) / 2;
        const oy = (h - video.videoHeight * s) / 2;
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, w, h);
        ctx.drawImage(video, ox, oy, video.videoWidth * s, video.videoHeight * s);
        drawSkeleton(ctx, frame, { map: (x, y) => [ox + x * s, oy + y * s], lineWidth: 2.5, radius: 2.5 });
      },
    });
    state.analysis = res;
    saveNewSwing({ start, end, quality });
    afterAnalysis();
  } catch (e) {
    if (e.name === 'AbortError') {
      show('step-progress', false);
      show('step-clip');
      video.controls = true;
      return;
    }
    console.error(e);
    show('step-progress', false);
    show('step-clip');
    video.controls = true;
    showError(`Pose detection failed: ${e.message || e}. Check your connection (the pose model is downloaded on first use) and try again.`);
  } finally {
    state.abort = null;
  }
}

function runDemo() {
  resetFlow();
  const d = demoUserSwing();
  state.isDemo = true;
  state.swingId = null;
  state.userBeats = null;
  state.video = null;
  $('set-ft').value = '5';
  $('set-in').value = '9';
  state.analysis = { frames: d.frames, times: d.times.map((t) => t - d.times[0]), fps: d.fps, width: d.width, height: d.height };
  show('step-upload', false);
  afterAnalysis();
}

// ---------------------------------------------------------------- step 3: stance

function afterAnalysis(saved = null) {
  const a = state.analysis;
  show('step-progress', false);
  state.pins = normalizePins(saved?.pins || []);
  state.fixedFrames = null;
  const frames = userFrames();
  const coverage = detectionCoverage(frames);
  if (coverage < 0.2) {
    show('step-upload');
    showError('We could not find a person in most of the video. Make sure your whole body is visible, the lighting is good, and you are the only person in the frame.');
    return;
  }
  // The swing's own clock: slow motion and frame rate need no setting.
  state.swingFps = estimateSwingFps(frames, a.fps);
  state.suggestedStance = suggestStanceFrame(frames, state.swingFps);
  state.stanceIndex = saved?.stanceIndex ?? state.suggestedStance;
  state.sideManual = Boolean(saved?.sideManual);
  state.pitcherSide = saved?.pitcherSide ?? 'right';
  updateSide();
  const note = $('coverage-note');
  if (coverage < 0.8) {
    note.hidden = false;
    note.textContent = `Your body was only found in ${Math.round(coverage * 100)}% of frames. Results may be less reliable; a clearer, steadier video helps.`;
  } else note.hidden = true;

  makeStanceStage();
  const slider = $('stance-slider');
  slider.max = String(a.frames.length - 1);
  slider.value = String(state.stanceIndex);
  setFixing(false);
  show('step-stance');
  scrollTo('step-stance');
  showStanceFrame(state.stanceIndex);
}

function makeStanceStage() {
  const a = state.analysis;
  stages.stance?.destroy();
  stages.stance = new Stage($('stance-stage'), { video: state.video, width: a.width, height: a.height, label: stageLabel() });
  attachFixHandlers(stages.stance);
}

/**
 * Detect the pitcher side at the current stance pick. It follows the stance
 * until the user chooses a side; after that it only says when the swing
 * disagrees. (A side detected at the wrong stance, e.g. in the finish, would
 * flip the whole comparison.)
 */
function updateSide() {
  state.sideDetect = detectPitcherSide(userFrames(), state.stanceIndex, state.swingFps);
  const { side, confidence } = state.sideDetect;
  const conf = Math.round(confidence * 100);
  if (!state.sideManual) state.pitcherSide = side;
  document.querySelector(`#pitcher-side input[value="${state.pitcherSide}"]`).checked = true;
  const hint = $('pitcher-hint');
  hint.classList.toggle('warn', false);
  if (!state.sideManual) {
    hint.textContent = `Detected from your stride, hand path, hand position and head turn at this stance (${conf}% sure). Change it if it is wrong.`;
    if (conf < 40) {
      hint.textContent = `Detected at this stance, but only ${conf}% sure: check it. The pitcher is on the side your front foot steps toward.`;
      hint.classList.toggle('warn', true);
    }
  } else if (side !== state.pitcherSide && conf >= 30) {
    hint.textContent = `You chose ${state.pitcherSide}, but at this stance the swing looks like the pitcher is to the ${side} (${conf}% sure). Your front foot should step toward the pitcher.`;
    hint.classList.toggle('warn', true);
  } else {
    hint.textContent = 'Set by you.';
  }
}

async function showStanceFrame(i) {
  const a = state.analysis;
  i = Math.max(0, Math.min(a.frames.length - 1, i));
  if (state.fixing) state.fixFrame = i;
  else if (state.stanceIndex !== i) {
    state.stanceIndex = i;
    updateSide();
  }
  $('stance-slider').value = String(i);
  $('stance-frame-label').textContent = state.fixing
    ? `Fixing frame ${i + 1} of ${a.frames.length} · ${fmtTime(a.times[i])} · your stance is frame ${state.stanceIndex + 1}`
    : `Frame ${i + 1} of ${a.frames.length} · ${fmtTime(a.times[i])}${i === state.suggestedStance ? ' · suggested stance' : ''}`;
  if (state.video) {
    await new Promise((resolve) => {
      const v = state.video;
      if (Math.abs(v.currentTime - a.times[i]) < 1e-4) return resolve();
      const done = () => {
        v.removeEventListener('seeked', done);
        resolve();
      };
      v.addEventListener('seeked', done);
      v.currentTime = a.times[i];
      setTimeout(done, 1500);
    });
  }
  if ((state.fixing ? state.fixFrame : state.stanceIndex) !== i) return;
  drawStanceStage(i);
  if (state.thumbPending && state.swingId && state.video) {
    state.thumbPending = false;
    const thumb = library.thumbnail(state.video);
    if (thumb) library.updateSwing(state.swingId, { thumb }).then(renderSavedList).catch(() => {});
  }
}

// ---------------------------------------------------------------- fixing joints by hand

/** Enter or leave "fix a joint" mode on the stance stage (it starts at frame `at`). */
function setFixing(on, at = state.stanceIndex) {
  state.fixing = on;
  state.drag = null;
  if (on) state.fixFrame = at;
  const overlay = stages.stance?.overlay;
  if (overlay) {
    overlay.style.pointerEvents = on ? 'auto' : '';
    overlay.style.touchAction = on ? 'none' : '';
    overlay.style.cursor = on ? 'crosshair' : '';
  }
  $('fix-toggle').textContent = on ? 'Done fixing' : 'Fix a joint';
  $('fix-toggle').setAttribute('aria-pressed', String(on));
  $('fix-hint').hidden = !on;
  updateFixStatus();
}

function updateFixStatus() {
  const n = state.pins.length;
  $('fix-undo').hidden = !n;
  $('fix-count').textContent = n ? `${n} joint${n > 1 ? 's' : ''} fixed by hand.` : '';
}

/** Replace the pins, then redo everything that depends on the frames. */
function setPins(pins) {
  const a = state.analysis;
  state.pins = normalizePins(pins);
  state.fixedFrames = null;
  state.swingFps = estimateSwingFps(userFrames(), a.fps);
  updateSide();
  if (state.swingId) library.updateSwing(state.swingId, { pins: state.pins }).then(renderSavedList).catch(() => {});
  updateFixStatus();
  drawStanceStage(state.fixing ? state.fixFrame : state.stanceIndex);
}

function drawStanceStage(i) {
  const f = userFrames()[i].slice();
  const d = state.drag;
  if (d) {
    f[d.joint * 3] = d.x;
    f[d.joint * 3 + 1] = d.y;
  }
  stages.stance.draw((ctx, map, u) => {
    drawSkeleton(ctx, f, { map, lineWidth: 3 * u, radius: 3 * u, outline: 'rgba(0,0,0,0.45)', hollowBelow: state.fixing ? 0.3 : 0 });
    drawPinMarks(ctx, map, f, u, d?.joint);
  });
}

/** Joints fixed by hand (and the one being dragged): a white square. */
function drawPinMarks(ctx, map, f, u, extra = -1) {
  ctx.save();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2 * u;
  for (let j = 0; j < NUM_KP; j++) {
    if (!pinned(f, j) && j !== extra) continue;
    const [x, y] = map(kx(f, j), ky(f, j));
    ctx.strokeRect(x - 5 * u, y - 5 * u, 10 * u, 10 * u);
  }
  ctx.restore();
}

/** Dragging a joint on the stance stage, in fix mode. */
function attachFixHandlers(stage) {
  const toImage = (e) => {
    const r = stage.overlay.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * stage.width, ((e.clientY - r.top) / r.height) * stage.height];
  };
  stage.overlay.addEventListener('pointerdown', (e) => {
    if (!state.fixing) return;
    const [x, y] = toImage(e);
    const radius = (24 / stage.overlay.getBoundingClientRect().width) * stage.width;
    const joint = nearestJoint(userFrames()[state.fixFrame], x, y, radius);
    if (joint < 0) return;
    e.preventDefault();
    stage.overlay.setPointerCapture(e.pointerId);
    state.drag = { joint, x, y };
    drawStanceStage(state.fixFrame);
  });
  stage.overlay.addEventListener('pointermove', (e) => {
    if (!state.drag) return;
    [state.drag.x, state.drag.y] = toImage(e);
    drawStanceStage(state.fixFrame);
  });
  const finish = (e) => {
    const d = state.drag;
    if (!d) return;
    state.drag = null;
    if (e.type === 'pointercancel') return drawStanceStage(state.fixFrame);
    setPins([...state.pins, { frame: state.fixFrame, joint: d.joint, x: Math.round(d.x * 10) / 10, y: Math.round(d.y * 10) / 10 }]);
  };
  stage.overlay.addEventListener('pointerup', finish);
  stage.overlay.addEventListener('pointercancel', finish);
}

// ---------------------------------------------------------------- compare

/** Prepare the user's swing, on the beats' clock when hand-set beats disagree with the hands'. */
function prepareUser() {
  const a = state.analysis;
  state.user = prepareSwing({
    frames: userFrames(),
    fps: a.fps,
    swingFps: state.userBeats ? pickSwingFps(state.swingFps, state.userBeats) : state.swingFps,
    stanceIndex: state.stanceIndex,
    pitcherSide: state.pitcherSide,
  });
}

function compare() {
  const a = state.analysis;
  if (!state.pros.length) {
    showError('The pro database is empty or failed to load, so there is nothing to compare against.');
    return;
  }
  // Hand-set beats belong to a stance; a new stance starts from automatic beats.
  if (state.userBeats && state.userBeats.stance !== state.stanceIndex) state.userBeats = null;
  try {
    prepareUser();
  } catch (e) {
    showError(`${e.message}. Pick a frame where your whole body is clearly visible.`);
    return;
  }
  showError('');
  if (state.swingId) {
    library.updateSwing(state.swingId, { stanceIndex: state.stanceIndex, pitcherSide: state.pitcherSide, sideManual: state.sideManual, height: heightInches(), beats: state.userBeats }).catch(() => {});
  }
  state.ranking = rankStances(state.user.stancePose, state.pros);
  // The video element moves into the results player, so the stance step closes.
  show('step-stance', false);
  show('results');
  selectPro(state.ranking[0].pro);
  scrollTo('results');
}

function reopenStance(fixAt = null) {
  player?.pause();
  show('results', false);
  makeStanceStage();
  show('step-stance');
  scrollTo('step-stance');
  setFixing(fixAt != null, fixAt);
  showStanceFrame(fixAt ?? state.stanceIndex);
}

/** Pro frame j fitted to the user's proportions and standing on the user's ground. */
function proAt(j) {
  return state.comparison.proFit[j];
}

function selectPro(pro) {
  state.evidence = null;
  $('evidence-note').hidden = true;
  state.pro = pro;
  runComparison();
  renderResults();
}

/** Compare with the selected pro, using hand-set beats when there are any. */
function runComparison() {
  const pro = state.pro;
  state.comparison = compareSwing(state.user, pro.prep, state.userBeats ? { phases: state.userBeats } : {});
  const userForRules = { ...state.user, phases: state.comparison.phases };
  state.items = evaluateFeedback(userForRules, pro.prep, { proName: pro.name.split(' — ')[0] });
}

// ---------------------------------------------------------------- results

function renderRanking() {
  const list = $('pro-list');
  list.textContent = '';
  state.ranking.forEach((r, k) => {
    const li = document.createElement('li');
    li.className = 'pro-item';
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('aria-pressed', String(r.pro.id === state.pro.id));
    const rank = document.createElement('span');
    rank.className = 'pro-rank';
    rank.textContent = `#${k + 1}`;
    const name = document.createElement('span');
    name.className = 'pro-name';
    name.textContent = r.pro.name;
    if (k === 0) {
      const badge = document.createElement('span');
      badge.className = 'badge best';
      badge.textContent = 'Best match';
      name.appendChild(badge);
    }
    for (const [flag, text] of [['synthetic', 'Synthetic'], ['local', 'This browser']]) {
      if (!r.pro.meta[flag]) continue;
      const badge = document.createElement('span');
      badge.className = 'badge';
      badge.textContent = text;
      name.appendChild(badge);
    }
    const score = document.createElement('span');
    score.className = 'pro-score';
    score.textContent = `${Math.round(r.similarity)}%`;
    const bar = document.createElement('span');
    bar.className = 'bar';
    const fill = document.createElement('span');
    fill.style.width = `${Math.max(2, r.similarity)}%`;
    bar.appendChild(fill);
    const meta = document.createElement('span');
    meta.className = 'pro-meta';
    meta.textContent = [r.pro.meta.team, r.pro.meta.bats ? `Bats ${r.pro.meta.bats}` : '', 'stance similarity'].filter(Boolean).join(' · ');
    b.append(rank, name, score, bar, meta);
    b.addEventListener('click', () => {
      if (r.pro.id !== state.pro.id) selectPro(r.pro);
    });
    li.appendChild(b);
    list.appendChild(li);
  });
}

function renderStanceCanvas() {
  const canvas = $('stance-canvas');
  const u = state.user.canon[state.user.stanceIndex];
  const p = proAt(state.pro.prep.stanceIndex);
  const w = canvas.clientWidth || 480;
  const h = canvas.clientHeight || 360;
  const ctx = fitCanvas(canvas, w, h);
  drawScene(ctx, w, h, { bounds: canonicalBounds([[u], [p]], 0.35), user: u, pro: p, caption: 'Stance' });
  $('stance-pro-name').textContent = state.pro.name;
}

function valuesText(it) {
  const ti = torsoIn();
  const pro = state.pro.name.split(' — ')[0];
  const scaled = it.unit === 'len' ? ' (scaled to your size)' : '';
  return `You ${formatValue(it.user, it.unit, ti)} · ${pro} ${formatValue(it.pro, it.unit, ti)}${scaled} · difference ${formatDelta(it.delta, it.unit, ti)}`;
}

const SEV_LABEL = { good: 'Matches', minor: 'Minor', major: 'Work on' };

function feedbackRow(it) {
  const row = document.createElement('button');
  row.type = 'button';
  row.className = 'fb';
  row.title = 'Show the frames this was measured on';
  row.addEventListener('click', () => showEvidence(it, row));
  const sev = document.createElement('span');
  sev.className = `sev ${it.severity}`;
  sev.textContent = SEV_LABEL[it.severity];
  const msg = document.createElement('span');
  msg.className = 'fb-msg';
  msg.textContent = it.message;
  const vals = document.createElement('span');
  vals.className = 'fb-vals';
  vals.textContent = `${it.label}: ${valuesText(it)}`;
  row.append(sev, msg, vals);
  if (it.tip) {
    const tip = document.createElement('span');
    tip.className = 'fb-tip';
    tip.textContent = it.tip;
    row.appendChild(tip);
  }
  const look = document.createElement('span');
  look.className = 'fb-look';
  look.textContent = 'Show frames →';
  row.appendChild(look);
  return row;
}

const SEV_ORDER = { major: 0, minor: 1, good: 2 };

function renderFeedback() {
  const items = state.items;
  const stanceBox = $('stance-feedback');
  stanceBox.textContent = '';
  items
    .filter((i) => i.phase === 'stance')
    .sort((a, b) => SEV_ORDER[a.severity] - SEV_ORDER[b.severity] || b.score - a.score)
    .forEach((it) => stanceBox.appendChild(feedbackRow(it)));

  const { priorities, strengths } = summarize(items.filter((i) => i.phase !== 'stance').concat(items.filter((i) => i.phase === 'stance' && i.severity === 'major')), 3);
  const pr = $('priorities');
  pr.textContent = '';
  if (!priorities.length) {
    const p = document.createElement('p');
    p.textContent = `Your swing lines up closely with ${state.pro.name} on every check. Nice work.`;
    pr.appendChild(p);
  }
  priorities.forEach((it, k) => {
    const card = document.createElement('article');
    card.className = 'priority';
    const ph = document.createElement('div');
    ph.className = 'ph';
    ph.textContent = `Priority ${k + 1} · ${phaseLabel(it.phase)}`;
    const h = document.createElement('h3');
    h.textContent = it.message;
    const v = document.createElement('p');
    v.textContent = valuesText(it);
    const tip = document.createElement('p');
    tip.textContent = it.tip;
    const sev = document.createElement('span');
    sev.className = `sev ${it.severity}`;
    sev.textContent = SEV_LABEL[it.severity];
    const look = document.createElement('button');
    look.type = 'button';
    look.className = 'btn ghost small-btn';
    look.textContent = 'Show me the frames';
    look.addEventListener('click', () => showEvidence(it, look));
    card.append(ph, h, sev, v, tip, look);
    pr.appendChild(card);
  });

  const st = $('strengths');
  st.textContent = '';
  if (strengths.length) {
    const h = document.createElement('h3');
    h.textContent = `Where you already match ${state.pro.name.split(' — ')[0]}`;
    const ul = document.createElement('ul');
    for (const it of strengths.slice(0, 10)) {
      const li = document.createElement('li');
      const pill = document.createElement('button');
      pill.type = 'button';
      pill.className = 'pill';
      pill.textContent = `✓ ${it.label}`;
      pill.title = 'Show the frames this was measured on';
      pill.addEventListener('click', () => showEvidence(it, pill));
      li.appendChild(pill);
      ul.appendChild(li);
    }
    st.append(h, ul);
  }

  const groups = $('phase-feedback');
  groups.textContent = '';
  for (const ph of PHASES) {
    if (ph.key === 'stance') continue;
    const list = items.filter((i) => i.phase === ph.key).sort((a, b) => SEV_ORDER[a.severity] - SEV_ORDER[b.severity] || b.score - a.score);
    if (!list.length) continue;
    const g = document.createElement('div');
    g.className = 'phase-group';
    const h = document.createElement('h3');
    h.textContent = ph.label;
    const blurb = document.createElement('p');
    blurb.className = 'muted small';
    blurb.textContent = ph.blurb;
    const box = document.createElement('div');
    box.className = 'feedback-list';
    list.forEach((it) => box.appendChild(feedbackRow(it)));
    g.append(h, blurb, box);
    groups.appendChild(g);
  }
  $('timing-note').textContent =
    'Tempo is not compared: each swing runs on its own clock, so slow motion, frame rate and swing speed don’t matter. Positions are compared at matching beats; adjust the beats under the player if one looks off.';
}

function renderScores() {
  const c = state.comparison;
  const r = state.ranking.find((x) => x.pro.id === state.pro.id);
  const box = $('scores');
  box.textContent = '';
  const tile = (k, v, hero = false) => {
    const d = document.createElement('div');
    d.className = `score-tile${hero ? ' hero' : ''}`;
    const vv = document.createElement('div');
    vv.className = 'v';
    vv.textContent = v;
    const kk = document.createElement('div');
    kk.className = 'k';
    kk.textContent = k;
    d.append(vv, kk);
    box.appendChild(d);
  };
  tile(`Swing match vs ${state.pro.name.split(' — ')[0]}`, `${Math.round(c.swingScore)}%`, true);
  tile('Stance', `${Math.round(r.similarity)}%`);
  for (const s of c.segments) tile(s.label, `${Math.round(s.score)}%`);
}

function renderTimeline() {
  const a = state.analysis;
  const n = a.frames.length;
  const tl = $('timeline');
  tl.max = String(n - 1);
  const marks = $('phase-marks');
  marks.textContent = '';
  const chips = $('phase-chips');
  chips.textContent = '';
  let lastPct = -100;
  let lastLow = false;
  for (const ph of PHASES) {
    const i = state.comparison.phases[ph.key];
    const m = document.createElement('span');
    const pct = (i / Math.max(1, n - 1)) * 100;
    // Stagger labels that would overlap the previous one.
    const low = pct - lastPct < 9 && !lastLow;
    if (low) m.classList.add('low');
    lastPct = pct;
    lastLow = low;
    m.style.left = `${pct}%`;
    m.textContent = ph.key === 'footPlant' ? 'Plant' : ph.label;
    marks.appendChild(m);
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    chip.dataset.phase = ph.key;
    const edited = state.userBeats && ph.key !== 'stance' && state.comparison.phases[ph.key] !== state.comparison.autoPhases[ph.key];
    chip.textContent = `${ph.label} · ${fmtTime(a.times[i])}${edited ? ' ✎' : ''}`;
    chip.addEventListener('click', () => {
      clearEvidence();
      if (ph.key !== 'stance') selectBeat(ph.key);
      player.seek(i);
    });
    chips.appendChild(chip);
  }
  renderBeatEditor();
}

// ---------------------------------------------------------------- beat editor

function selectBeat(key) {
  state.beatKey = key;
  renderBeatEditor();
}

function renderBeatEditor(note = '') {
  const sel = $('beat-select');
  if (sel.value !== state.beatKey) sel.value = state.beatKey;
  const i = state.comparison.phases[state.beatKey];
  const auto = state.comparison.autoPhases[state.beatKey];
  const where = `${phaseLabel(state.beatKey)} is at frame ${i + 1} (${fmtTime(state.analysis.times[i])})`;
  const status = !state.userBeats
    ? `Automatic beats. ${where}.`
    : i !== auto
      ? `Beats adjusted by hand. ${where}; detected at frame ${auto + 1}.`
      : `Beats adjusted by hand. ${where}, as detected.`;
  $('beat-status').textContent = note ? `${note} ${status}` : status;
  $('beat-reset').disabled = !state.userBeats;
}

/** Move one beat to a frame (kept between its neighbours) and recompare. */
function setBeat(key, frame) {
  const phases = { ...state.comparison.phases };
  const order = PHASES.map((p) => p.key);
  const k = order.indexOf(key);
  const lo = phases[order[k - 1]] + 1;
  const hi = k + 1 < order.length ? phases[order[k + 1]] - 1 : state.analysis.frames.length - 1;
  if (lo > hi) {
    renderBeatEditor(`No room to move ${phaseLabel(key)}: move a neighbouring beat first.`);
    return;
  }
  const f = Math.max(lo, Math.min(hi, Math.round(frame)));
  phases[key] = f;
  applyBeats(phases);
  player.seek(f);
  if (f !== Math.round(frame)) {
    renderBeatEditor(`${phaseLabel(key)} has to stay between ${phaseLabel(order[k - 1])} and ${k + 1 < order.length ? phaseLabel(order[k + 1]) : 'the end of the clip'}, so it went as far as it can.`);
  }
}

function applyBeats(beats) {
  clearEvidence();
  state.userBeats = beats;
  if (state.swingId) library.updateSwing(state.swingId, { beats }).catch(() => {});
  retimeUser();
  runComparison();
  refreshResults();
}

/** Re-prepare the user's swing if its clock should change with the beats. */
function retimeUser() {
  const want = state.userBeats ? pickSwingFps(state.swingFps, state.userBeats) : state.swingFps;
  if (Math.abs(Math.log(want / state.user.swingFps)) > 1e-9) {
    prepareUser();
    state.sceneBounds = null;
  }
}

function refreshResults() {
  renderScores();
  renderFeedback();
  renderTimeline();
  renderCharts();
  state.sceneBounds = null;
  drawResultFrame(player.index);
}

function currentPhase(i) {
  let cur = null;
  for (const ph of PHASES) if (i >= state.comparison.phases[ph.key]) cur = ph.key;
  return cur;
}

function drawResultFrame(i) {
  const user = state.user;
  const pro = state.pro.prep;
  const c = state.comparison;
  const a = state.analysis;
  $('timeline').value = String(i);
  const cur = currentPhase(i);
  for (const chip of $('phase-chips').children) chip.setAttribute('aria-pressed', String(chip.dataset.phase === cur));
  // Evidence mode: show exactly the two frames a feedback check compared.
  const ev = state.evidence?.evidence;
  const evidenceHere = ev && i === ev.userFrame;
  const j = evidenceHere ? ev.proFrame : Math.round(c.userToProOrig(i));
  const ghostOn = $('ghost-toggle').checked;
  const toImg = (f) => {
    const g = f.slice();
    for (let k = 0; k < NUM_KP; k++) {
      const [x, y] = canonToImage(user.transform, f[k * 3], f[k * 3 + 1]);
      g[k * 3] = x;
      g[k * 3 + 1] = y;
    }
    return g;
  };
  const proColor = cssVar('--series-pro', '#eb6834');
  const userColor = cssVar('--series-user', '#2a78d6');
  const showGhost = ghostOn && (i >= user.stanceIndex || evidenceHere);
  stages.result.draw((ctx, map, u) => {
    if (showGhost) drawSkeleton(ctx, toImg(proAt(j)), { map, color: proColor, lineWidth: 3 * u, radius: 0, alpha: 0.85, outline: 'rgba(0,0,0,0.35)' });
    const mine = toImg(user.canon[i]);
    drawSkeleton(ctx, mine, { map, lineWidth: 3 * u, radius: 3 * u, outline: 'rgba(0,0,0,0.45)', hollowBelow: 0.3 });
    drawPinMarks(ctx, map, mine, u);
    if ($('raw-toggle').checked) drawPoints(ctx, a.frames[i], { map, radius: 2.5 * u });
    if (evidenceHere) {
      const o = { pair: ev.pair, scale: u };
      if (showGhost) drawHighlights(ctx, map, toImg(proAt(j)), ev.joints, proColor, { ...o, stanceFrame: ev.fromStance ? toImg(proAt(ev.proStance)) : null });
      drawHighlights(ctx, map, toImg(user.canon[i]), ev.joints, userColor, { ...o, stanceFrame: ev.fromStance ? toImg(user.canon[ev.userStance]) : null });
    }
  });
  const canvas = $('scene-canvas');
  const w = canvas.clientWidth || 480;
  const h = canvas.clientHeight || 360;
  const ctx = fitCanvas(canvas, w, h);
  if (!state.sceneBounds) {
    state.sceneBounds = canonicalBounds([
      user.canon.slice(user.stanceIndex, c.endUserIndex + 1),
      c.proFit.slice(pro.stanceIndex, pro.phases.finish + 1),
    ]);
  }
  const proFrame = proAt(i >= user.stanceIndex || evidenceHere ? j : pro.stanceIndex);
  const smap = drawScene(ctx, w, h, {
    bounds: state.sceneBounds,
    user: user.canon[i],
    pro: proFrame,
    userTrail: evidenceHere ? null : { frames: user.canon, from: user.stanceIndex, to: c.endUserIndex },
    proTrail: evidenceHere ? null : { frames: c.proFit, from: pro.stanceIndex, to: pro.phases.finish },
    caption: evidenceHere
      ? `${state.evidence.label} · ${phaseLabel(state.evidence.phase)}`
      : `${cur ? phaseLabel(cur) : 'Before stance'} · ${fmtTime(a.times[i])}`,
  });
  if (evidenceHere) {
    const o = { pair: ev.pair };
    drawHighlights(ctx, smap, proFrame, ev.joints, proColor, { ...o, stanceFrame: ev.fromStance ? proAt(ev.proStance) : null });
    drawHighlights(ctx, smap, user.canon[i], ev.joints, userColor, { ...o, stanceFrame: ev.fromStance ? user.canon[ev.userStance] : null });
  }
}

/** Jump the player to the frames a feedback check was measured on. */
async function showEvidence(it, el) {
  if (!it.evidence || !player) return;
  player.pause();
  state.evidence = it;
  document.querySelectorAll('.evidence-active').forEach((x) => x.classList.remove('evidence-active'));
  el?.classList.add('evidence-active');
  const ev = it.evidence;
  const a = state.analysis;
  const pro = state.pro.prep;
  const proShort = state.pro.name.split(' — ')[0];
  $('evidence-note').hidden = false;
  $('evidence-title').textContent = `${it.label} · ${phaseLabel(it.phase)}: ${it.message}`;
  const how = [
    `Your frame ${ev.userFrame + 1} (${fmtTime(a.times[ev.userFrame])}) vs ${proShort}’s frame ${ev.proFrame + 1} (${(ev.proFrame / pro.fps).toFixed(2)} s into their clip).`,
    `${valuesText(it)}.`,
    `Rings mark the joints measured${ev.fromStance ? '; dashed circles and arrows show how far they moved from the stance.' : ev.pair ? '; the dashed line is the distance or angle measured.' : '.'}`,
  ];
  if (it.unit === 'sec') how.push(`Timing runs from ${it.id === 'timing.swing' ? 'foot plant to contact' : 'load to foot plant'}; the frame shown is the end of that span.`);
  $('evidence-body').textContent = how.join(' ');
  $('h-swing').scrollIntoView({ behavior: 'smooth', block: 'start' });
  await player.seek(ev.userFrame);
}

function clearEvidence() {
  if (!state.evidence) return;
  state.evidence = null;
  $('evidence-note').hidden = true;
  document.querySelectorAll('.evidence-active').forEach((x) => x.classList.remove('evidence-active'));
}

function renderPlayer() {
  const a = state.analysis;
  player?.pause();
  stages.result?.destroy();
  stages.result = new Stage($('result-stage'), { video: state.video, width: a.width, height: a.height, label: stageLabel() });
  $('ghost-caption').hidden = false;
  state.sceneBounds = null;
  player = new FramePlayer({
    video: state.video,
    times: a.times,
    onFrame: drawResultFrame,
    onState: (playing) => {
      $('play-btn').innerHTML = playing ? '&#10074;&#10074;' : '&#9654;';
      $('play-btn').setAttribute('aria-label', playing ? 'Pause' : 'Play');
    },
  });
  renderTimeline();
  player.seek(state.user.stanceIndex);
}

const CHARTS = [
  { key: 'headX', title: 'Head movement toward the pitcher', unit: 'len' },
  { key: 'headY', title: 'Head height change', unit: 'len' },
  { key: 'handsX', title: 'Hands: toward the pitcher', unit: 'len' },
  { key: 'handsY', title: 'Hands: height', unit: 'len' },
  { key: 'stride', title: 'Front foot travel (stride)', unit: 'len' },
  { key: 'hipTurn', title: 'Hip turn (apparent)', unit: 'deg' },
  { key: 'shoulderTurn', title: 'Shoulder turn (apparent)', unit: 'deg' },
  { key: 'fKnee', title: 'Front knee angle (apparent)', unit: 'deg' },
];

function renderCharts() {
  const box = $('charts');
  box.textContent = '';
  const { t, series, phaseTimes } = state.comparison.chart;
  const ti = torsoIn();
  const proShort = state.pro.name.split(' — ')[0];
  for (const c of CHARTS) {
    const div = document.createElement('div');
    box.appendChild(div);
    lineChart(div, {
      title: c.title,
      t,
      phaseTimes,
      format: (v) => formatValue(v, c.unit, ti),
      series: [
        { name: 'You', short: 'You', colorVar: '--series-user', values: series[c.key].user },
        { name: proShort, short: 'Pro', colorVar: '--series-pro', values: series[c.key].pro },
      ],
    });
  }
  // Table view of every check.
  const tableBox = $('data-table');
  tableBox.textContent = '';
  const scroll = document.createElement('div');
  scroll.className = 'table-scroll';
  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  for (const h of ['Phase', 'Check', 'You', proShort, 'Difference', 'Status', '']) {
    const th = document.createElement('th');
    th.textContent = h;
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  const tbody = document.createElement('tbody');
  const order = Object.fromEntries(PHASES.map((p, k) => [p.key, k]));
  for (const it of state.items.slice().sort((a, b) => order[a.phase] - order[b.phase])) {
    const tr = document.createElement('tr');
    for (const v of [phaseLabel(it.phase), it.label, formatValue(it.user, it.unit, ti), formatValue(it.pro, it.unit, ti), formatDelta(it.delta, it.unit, ti), SEV_LABEL[it.severity]]) {
      const td = document.createElement('td');
      td.textContent = v;
      tr.appendChild(td);
    }
    const td = document.createElement('td');
    const look = document.createElement('button');
    look.type = 'button';
    look.className = 'btn ghost small-btn';
    look.textContent = 'Show';
    look.setAttribute('aria-label', `Show frames for ${it.label}`);
    look.addEventListener('click', () => showEvidence(it, look));
    td.appendChild(look);
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
  table.append(thead, tbody);
  scroll.appendChild(table);
  tableBox.appendChild(scroll);
}

function renderResults() {
  renderRanking();
  renderStanceCanvas();
  renderScores();
  renderFeedback();
  renderPlayer();
  renderCharts();
}

// ---------------------------------------------------------------- export

function exportPose() {
  const a = state.analysis;
  const entry = makeEntry({
    id: `my-swing-${new Date().toISOString().slice(0, 10)}`,
    name: 'My swing',
    notes: 'Exported from Swing Match. Same format as the pro database (swing-db/v1).',
    source: state.isDemo ? 'Swing Match demo swing (synthetic)' : 'Swing Match pose tracking (BODY_25)',
    fps: a.fps,
    width: a.width,
    height: a.height,
    pitcherSide: state.pitcherSide,
    stanceFrame: state.user.stanceIndex,
    phases: state.comparison.phases,
    frames: userFrames().map((f) => f.map((v, k) => (k % 3 === 2 ? Math.min(1, v) : v))),
  });
  download('my-swing-body25.json', entry);
}

function exportReport() {
  const r = state.ranking.map((x) => ({ id: x.pro.id, name: x.pro.name, stanceSimilarity: Math.round(x.similarity * 10) / 10 }));
  download('swing-report.json', {
    generatedAt: new Date().toISOString(),
    heightInches: heightInches(),
    pitcherSide: state.pitcherSide,
    stanceFrame: state.user.stanceIndex,
    stanceRanking: r,
    comparedWith: { id: state.pro.id, name: state.pro.name },
    swingScore: Math.round(state.comparison.swingScore * 10) / 10,
    segments: state.comparison.segments.map((s) => ({ key: s.key, label: s.label, score: Math.round(s.score * 10) / 10 })),
    phases: Object.fromEntries(Object.entries(state.comparison.phases).map(([k, i]) => [k, { frame: i, time: state.analysis.times[i] }])),
    feedback: state.items.map(({ id, phase, label, unit, user, pro, delta, severity, message, tip }) => ({ id, phase, label, unit, user, pro, delta, severity, message, tip })),
    units: { len: 'torso lengths (body-proportional, at stance)', deg: 'degrees' },
  });
}

// ---------------------------------------------------------------- saved swings

async function saveNewSwing({ start, end, quality }) {
  const a = state.analysis;
  const id = `swing-${Date.now()}`;
  const record = {
    id,
    name: state.file?.name || 'Swing',
    createdAt: Date.now(),
    analysis: a,
    clip: { start, end },
    quality,
    stanceIndex: null,
    pitcherSide: null,
    height: heightInches(),
    thumb: null,
    hasVideo: false,
  };
  try {
    await library.putSwing(record);
    state.swingId = id;
    state.thumbPending = true;
    library.requestPersistence();
    if (state.file && (await library.putVideo(id, state.file))) {
      await library.updateSwing(id, { hasVideo: true, videoSize: state.file.size, videoType: state.file.type });
    }
    renderSavedList();
  } catch (e) {
    console.warn('Could not save the swing in this browser:', e);
  }
}

function fmtDate(ms) {
  try {
    return new Date(ms).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
  } catch {
    return '';
  }
}

async function renderSavedList() {
  let list = [];
  try {
    list = await library.listSwings();
  } catch {
    $('saved-swings').hidden = true;
    return;
  }
  const ul = $('saved-list');
  ul.textContent = '';
  $('saved-swings').hidden = !list.length;
  for (const rec of list) {
    const li = document.createElement('li');
    li.className = 'saved-item';
    const img = document.createElement('img');
    img.className = 'saved-thumb';
    img.alt = '';
    if (rec.thumb) img.src = rec.thumb;
    const text = document.createElement('div');
    const name = document.createElement('div');
    name.className = 'saved-name';
    name.textContent = rec.name;
    const meta = document.createElement('div');
    meta.className = 'saved-meta';
    const a = rec.analysis || {};
    const dur = a.times?.length ? a.times[a.times.length - 1] - a.times[0] : 0;
    meta.textContent = [
      fmtDate(rec.createdAt),
      `${dur.toFixed(1)} s`,
      rec.hasVideo ? 'video saved' : 'poses only',
      rec.stanceIndex != null ? 'stance picked' : '',
      rec.beats ? 'beats adjusted' : '',
      rec.pins?.length ? `${rec.pins.length} joint${rec.pins.length > 1 ? 's' : ''} fixed` : '',
    ]
      .filter(Boolean)
      .join(' · ');
    text.append(name, meta);
    const actions = document.createElement('div');
    actions.className = 'saved-actions';
    const open = document.createElement('button');
    open.type = 'button';
    open.className = 'btn primary small-btn';
    open.textContent = 'Open';
    open.addEventListener('click', () => openSaved(rec.id));
    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'btn ghost small-btn';
    del.textContent = 'Delete';
    del.setAttribute('aria-label', `Delete ${rec.name}`);
    del.addEventListener('click', async () => {
      if (!confirm(`Delete the saved swing “${rec.name}” from this browser?`)) return;
      await library.deleteSwing(rec.id).catch(() => {});
      if (state.swingId === rec.id) state.swingId = null;
      renderSavedList();
    });
    actions.append(open, del);
    li.append(img, text, actions);
    ul.appendChild(li);
  }
}

async function openSaved(id) {
  let rec;
  try {
    rec = await library.getSwing(id);
  } catch (e) {
    showError(`Could not open that swing: ${e.message}`);
    return;
  }
  if (!rec) {
    renderSavedList();
    return;
  }
  resetFlow();
  state.isDemo = false;
  state.file = null;
  state.swingId = rec.id;
  state.thumbPending = !rec.thumb;
  state.analysis = rec.analysis;
  state.userBeats = rec.beats || null;
  if (rec.height) {
    $('set-ft').value = String(Math.floor(rec.height / 12));
    $('set-in').value = String(rec.height % 12);
  }
  if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
  state.videoUrl = null;
  state.video = null;
  if (rec.hasVideo) {
    const blob = await library.getVideo(rec.id).catch(() => null);
    if (blob) {
      state.videoUrl = URL.createObjectURL(blob);
      const video = document.createElement('video');
      video.preload = 'auto';
      video.muted = true;
      video.playsInline = true;
      video.src = state.videoUrl;
      const ok = await new Promise((resolve) => {
        video.addEventListener('loadeddata', () => resolve(true), { once: true });
        video.addEventListener('error', () => resolve(false), { once: true });
      });
      if (ok) state.video = video;
    }
  }
  show('step-upload', false);
  afterAnalysis({ stanceIndex: rec.stanceIndex, pitcherSide: rec.pitcherSide, sideManual: rec.sideManual, pins: rec.pins });
  // Stance already picked last time: go straight to the results.
  if (rec.stanceIndex != null && !$('step-stance').hidden) {
    await dbReady;
    compare();
  }
}

// ---------------------------------------------------------------- wiring

function wire() {
  const input = $('file-input');
  input.addEventListener('change', () => onFile(input.files[0]));
  const dz = $('dropzone');
  dz.addEventListener('dragover', (e) => {
    e.preventDefault();
    dz.classList.add('drag');
  });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
  dz.addEventListener('drop', (e) => {
    e.preventDefault();
    dz.classList.remove('drag');
    onFile(e.dataTransfer.files[0]);
  });
  $('demo-btn').addEventListener('click', runDemo);
  $('analyze-btn').addEventListener('click', runAnalysis);
  $('clip-cancel').addEventListener('click', () => {
    input.value = '';
    resetFlow();
  });
  $('cancel-btn').addEventListener('click', () => state.abort?.abort());

  const slider = $('stance-slider');
  slider.addEventListener('input', () => showStanceFrame(Number(slider.value)));
  const shown = () => (state.fixing ? state.fixFrame : state.stanceIndex);
  $('stance-prev').addEventListener('click', () => showStanceFrame(shown() - 1));
  $('stance-next').addEventListener('click', () => showStanceFrame(shown() + 1));
  $('stance-suggest').addEventListener('click', () => showStanceFrame(state.suggestedStance));
  document.querySelectorAll('#pitcher-side input').forEach((r) =>
    r.addEventListener('change', () => {
      state.pitcherSide = r.value;
      state.sideManual = true;
      updateSide();
    }),
  );
  $('compare-btn').addEventListener('click', () => {
    setFixing(false);
    compare();
  });
  $('restance-btn').addEventListener('click', () => reopenStance());
  $('fix-toggle').addEventListener('click', () => {
    setFixing(!state.fixing, state.stanceIndex);
    showStanceFrame(state.fixing ? state.fixFrame : state.stanceIndex);
  });
  $('fix-undo').addEventListener('click', () => {
    setPins(state.pins.slice(0, -1));
  });
  $('fix-here').addEventListener('click', () => reopenStance(player?.index ?? state.stanceIndex));

  $('play-btn').addEventListener('click', () => {
    clearEvidence();
    if (player.playing) player.pause();
    else {
      const from = player.index >= state.comparison.endUserIndex ? state.user.stanceIndex : player.index;
      player.play(Number($('play-rate').value), from, Math.min(state.analysis.frames.length - 1, state.comparison.endUserIndex + 3));
    }
  });
  $('timeline').addEventListener('input', () => {
    clearEvidence();
    player.seek(Number($('timeline').value));
  });
  $('beat-select').addEventListener('change', () => {
    selectBeat($('beat-select').value);
    clearEvidence();
    player.seek(state.comparison.phases[state.beatKey]);
  });
  $('beat-set').addEventListener('click', () => setBeat(state.beatKey, player.index));
  $('beat-prev').addEventListener('click', () => setBeat(state.beatKey, state.comparison.phases[state.beatKey] - 1));
  $('beat-next').addEventListener('click', () => setBeat(state.beatKey, state.comparison.phases[state.beatKey] + 1));
  $('beat-reset').addEventListener('click', () => {
    clearEvidence();
    state.userBeats = null;
    if (state.swingId) library.updateSwing(state.swingId, { beats: null }).catch(() => {});
    retimeUser();
    runComparison();
    refreshResults();
  });
  $('evidence-clear').addEventListener('click', () => {
    clearEvidence();
    drawResultFrame(player.index);
  });
  $('ghost-toggle').addEventListener('change', () => drawResultFrame(player.index));
  $('raw-toggle').addEventListener('change', () => drawResultFrame(player.index));
  $('export-pose').addEventListener('click', exportPose);
  $('export-report').addEventListener('click', exportReport);
  $('restart-btn').addEventListener('click', () => {
    input.value = '';
    resetFlow();
    scrollTo('step-upload');
  });
  for (const id of ['set-ft', 'set-in']) {
    $(id).addEventListener('change', () => {
      if (state.swingId) library.updateSwing(state.swingId, { height: heightInches() }).catch(() => {});
      if (state.comparison) {
        renderFeedback();
        renderCharts();
      }
    });
  }
  window.addEventListener('resize', () => {
    if (state.comparison && !$('results').hidden) {
      renderStanceCanvas();
      drawResultFrame(player.index);
    }
  });
  window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener?.('change', () => {
    if (state.comparison) {
      renderStanceCanvas();
      drawResultFrame(player.index);
    }
  });
}

wire();
renderSavedList();
const dbReady = initDatabase().catch((e) => {
  $('db-status').textContent = `Could not load the pro database: ${e.message}`;
});

// Exposed for automated checks and debugging.
window.swingMatch = { state, runDemo, compare, openSaved, renderSavedList };
