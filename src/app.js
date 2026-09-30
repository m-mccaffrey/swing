// Main page controller: upload → pose detection → stance pick → stance
// match → swing comparison → feedback.

import { loadDatabase, makeEntry } from './core/db.js';
import { prepareSwing, compareSwing } from './core/compare.js';
import { detectPitcherSide, suggestStanceFrame, canonToImage, detectionCoverage } from './core/sequence.js';
import { rankStances } from './core/match.js';
import { evaluateFeedback, summarize, formatValue, formatDelta, TORSO_TO_HEIGHT } from './core/feedback.js';
import { PHASES, phaseLabel } from './core/phases.js';
import { demoUserSwing } from './core/synth.js';
import { NUM_KP } from './core/body25.js';
import { getLandmarker, analyzeVideo, estimateVideoFps } from './pose/detector.js';
import { Stage, FramePlayer } from './ui/stage.js';
import { drawSkeleton, drawScene, canonicalBounds, fitCanvas, cssVar } from './ui/draw.js';
import { lineChart } from './ui/charts.js';

const $ = (id) => document.getElementById(id);

const state = {
  pros: [],
  video: null,
  videoUrl: null,
  videoFps: 30,
  isDemo: false,
  analysis: null, // { frames, times, fps, width, height, speedFactor }
  stanceIndex: 0,
  suggestedStance: 0,
  sideDetect: null,
  pitcherSide: 'right',
  user: null,
  ranking: [],
  pro: null,
  comparison: null,
  items: [],
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

function torsoIn() {
  return heightInches() * TORSO_TO_HEIGHT;
}

function speedFactor() {
  return Number($('set-speed').value) || 1;
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
    state.analysis = { ...res, speedFactor: speedFactor() };
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
  state.video = null;
  $('set-speed').value = '1';
  $('set-ft').value = '5';
  $('set-in').value = '9';
  state.analysis = { frames: d.frames, times: d.times.map((t) => t - d.times[0]), fps: d.fps, width: d.width, height: d.height, speedFactor: 1 };
  show('step-upload', false);
  afterAnalysis();
}

// ---------------------------------------------------------------- step 3: stance

function afterAnalysis() {
  const a = state.analysis;
  show('step-progress', false);
  const coverage = detectionCoverage(a.frames);
  if (coverage < 0.2) {
    show('step-upload');
    showError('We could not find a person in most of the video. Make sure your whole body is visible, the lighting is good, and you are the only person in the frame.');
    return;
  }
  const realFps = a.fps * a.speedFactor;
  state.suggestedStance = suggestStanceFrame(a.frames, realFps);
  state.stanceIndex = state.suggestedStance;
  state.sideDetect = detectPitcherSide(a.frames, state.stanceIndex, realFps);
  state.pitcherSide = state.sideDetect.side;
  document.querySelector(`#pitcher-side input[value="${state.pitcherSide}"]`).checked = true;
  const conf = Math.round(state.sideDetect.confidence * 100);
  $('pitcher-hint').textContent = `Auto-detected from your head turn, hand position, stride and swing direction (${conf}% confident). Change it if it is wrong.`;
  const note = $('coverage-note');
  if (coverage < 0.8) {
    note.hidden = false;
    note.textContent = `Your body was only found in ${Math.round(coverage * 100)}% of frames. Results may be less reliable; a clearer, steadier video helps.`;
  } else note.hidden = true;

  stages.stance?.destroy();
  stages.stance = new Stage($('stance-stage'), { video: state.video, width: a.width, height: a.height, label: 'Demo swing (keypoints only, no video)' });
  const slider = $('stance-slider');
  slider.max = String(a.frames.length - 1);
  slider.value = String(state.stanceIndex);
  show('step-stance');
  scrollTo('step-stance');
  showStanceFrame(state.stanceIndex);
}

async function showStanceFrame(i) {
  const a = state.analysis;
  i = Math.max(0, Math.min(a.frames.length - 1, i));
  state.stanceIndex = i;
  $('stance-slider').value = String(i);
  $('stance-frame-label').textContent = `Frame ${i + 1} of ${a.frames.length} · ${fmtTime(a.times[i])}${i === state.suggestedStance ? ' · suggested stance' : ''}`;
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
  if (state.stanceIndex !== i) return;
  stages.stance.draw((ctx, map, u) => drawSkeleton(ctx, a.frames[i], { map, lineWidth: 3 * u, radius: 3 * u, outline: 'rgba(0,0,0,0.45)' }));
}

// ---------------------------------------------------------------- compare

function compare() {
  const a = state.analysis;
  if (!state.pros.length) {
    showError('The pro database is empty or failed to load, so there is nothing to compare against.');
    return;
  }
  try {
    state.user = prepareSwing({
      frames: a.frames,
      fps: a.fps,
      speedFactor: a.speedFactor,
      stanceIndex: state.stanceIndex,
      pitcherSide: state.pitcherSide,
    });
  } catch (e) {
    showError(`${e.message}. Pick a frame where your whole body is clearly visible.`);
    return;
  }
  showError('');
  state.ranking = rankStances(state.user.stancePose, state.pros);
  // The video element moves into the results player, so the stance step closes.
  show('step-stance', false);
  show('results');
  selectPro(state.ranking[0].pro);
  scrollTo('results');
}

function reopenStance() {
  player?.pause();
  show('results', false);
  const a = state.analysis;
  stages.stance?.destroy();
  stages.stance = new Stage($('stance-stage'), { video: state.video, width: a.width, height: a.height, label: 'Demo swing (keypoints only, no video)' });
  show('step-stance');
  scrollTo('step-stance');
  showStanceFrame(state.stanceIndex);
}

function selectPro(pro) {
  state.pro = pro;
  state.comparison = compareSwing(state.user, pro.prep);
  const userForRules = { ...state.user, phases: state.comparison.phases };
  state.items = evaluateFeedback(userForRules, pro.prep, { proName: pro.name.split(' — ')[0], timingKnown: true });
  renderResults();
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
  const p = state.pro.prep.canon[state.pro.prep.stanceIndex];
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
  const row = document.createElement('div');
  row.className = 'fb';
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
    const tip = document.createElement('p');
    tip.className = 'fb-tip';
    tip.textContent = it.tip;
    row.appendChild(tip);
  }
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
    card.append(ph, h, sev, v, tip);
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
      li.textContent = `✓ ${it.label}`;
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
  const sf = state.analysis.speedFactor;
  $('timing-note').textContent =
    sf === 1
      ? 'Timing checks assume your video plays in real time. If it was recorded in slow motion, set "Video speed" in the settings and analyze again.'
      : `Timing adjusted for ${sf}× slow motion.`;
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
  for (const ph of PHASES) {
    const i = state.comparison.phases[ph.key];
    const m = document.createElement('span');
    m.style.left = `${(i / Math.max(1, n - 1)) * 100}%`;
    m.textContent = ph.key === 'footPlant' ? 'Plant' : ph.label;
    marks.appendChild(m);
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    chip.dataset.phase = ph.key;
    chip.textContent = `${ph.label} · ${fmtTime(a.times[i])}`;
    chip.addEventListener('click', () => player.seek(i));
    chips.appendChild(chip);
  }
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
  const j = Math.round(c.userToProOrig(i));
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
  stages.result.draw((ctx, map, u) => {
    if (ghostOn && i >= user.stanceIndex) drawSkeleton(ctx, toImg(pro.canon[j]), { map, color: proColor, lineWidth: 3 * u, radius: 0, alpha: 0.85, outline: 'rgba(0,0,0,0.35)' });
    drawSkeleton(ctx, toImg(user.canon[i]), { map, lineWidth: 3 * u, radius: 3 * u, outline: 'rgba(0,0,0,0.45)' });
  });
  const canvas = $('scene-canvas');
  const w = canvas.clientWidth || 480;
  const h = canvas.clientHeight || 360;
  const ctx = fitCanvas(canvas, w, h);
  if (!state.sceneBounds) {
    state.sceneBounds = canonicalBounds([
      user.canon.slice(user.stanceIndex, c.endUserIndex + 1),
      pro.canon.slice(pro.stanceIndex, pro.phases.finish + 1),
    ]);
  }
  drawScene(ctx, w, h, {
    bounds: state.sceneBounds,
    user: user.canon[i],
    pro: i >= user.stanceIndex ? pro.canon[j] : pro.canon[pro.stanceIndex],
    userTrail: { frames: user.canon, from: user.stanceIndex, to: c.endUserIndex },
    proTrail: { frames: pro.canon, from: pro.stanceIndex, to: pro.phases.finish },
    caption: `${cur ? phaseLabel(cur) : 'Before stance'} · ${fmtTime(a.times[i])}`,
  });
}

function renderPlayer() {
  const a = state.analysis;
  player?.pause();
  stages.result?.destroy();
  stages.result = new Stage($('result-stage'), { video: state.video, width: a.width, height: a.height, label: 'Demo swing (keypoints only, no video)' });
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
  for (const h of ['Phase', 'Check', 'You', proShort, 'Difference', 'Status']) {
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
    source: state.isDemo ? 'Swing Match demo swing (synthetic)' : 'Swing Match (MediaPipe Pose → BODY_25)',
    fps: a.fps,
    speedFactor: a.speedFactor,
    width: a.width,
    height: a.height,
    pitcherSide: state.pitcherSide,
    stanceFrame: state.user.stanceIndex,
    phases: state.comparison.phases,
    frames: a.frames,
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
    units: { len: 'torso lengths (Neck→MidHip at stance)', deg: 'degrees', sec: 'seconds' },
  });
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
  $('stance-prev').addEventListener('click', () => showStanceFrame(state.stanceIndex - 1));
  $('stance-next').addEventListener('click', () => showStanceFrame(state.stanceIndex + 1));
  $('stance-suggest').addEventListener('click', () => showStanceFrame(state.suggestedStance));
  document.querySelectorAll('#pitcher-side input').forEach((r) =>
    r.addEventListener('change', () => {
      state.pitcherSide = r.value;
    }),
  );
  $('compare-btn').addEventListener('click', compare);
  $('restance-btn').addEventListener('click', reopenStance);

  $('play-btn').addEventListener('click', () => {
    if (player.playing) player.pause();
    else {
      const from = player.index >= state.comparison.endUserIndex ? state.user.stanceIndex : player.index;
      player.play(Number($('play-rate').value), from, Math.min(state.analysis.frames.length - 1, state.comparison.endUserIndex + 3));
    }
  });
  $('timeline').addEventListener('input', () => player.seek(Number($('timeline').value)));
  $('ghost-toggle').addEventListener('change', () => drawResultFrame(player.index));
  $('export-pose').addEventListener('click', exportPose);
  $('export-report').addEventListener('click', exportReport);
  $('restart-btn').addEventListener('click', () => {
    input.value = '';
    resetFlow();
    scrollTo('step-upload');
  });
  for (const id of ['set-ft', 'set-in']) {
    $(id).addEventListener('change', () => {
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
initDatabase().catch((e) => {
  $('db-status').textContent = `Could not load the pro database: ${e.message}`;
});

// Exposed for automated checks and debugging.
window.swingMatch = { state, runDemo, compare };
