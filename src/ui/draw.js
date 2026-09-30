// Canvas drawing: BODY_25 skeletons (OpenPose colors or a single color),
// the canonical side-by-side comparison scene, and hand-path trails.

import { PAIRS, COLORS, NUM_KP, KP } from '../core/body25.js';

export function cssVar(name, fallback = '#888') {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** Resize a canvas' backing store to its CSS size × devicePixelRatio. */
export function fitCanvas(canvas, cssWidth, cssHeight) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(1, Math.round(cssWidth * dpr));
  const h = Math.max(1, Math.round(cssHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/**
 * Draw one BODY_25 frame.
 * @param {CanvasRenderingContext2D} ctx
 * @param {number[]} frame
 * @param {object} o
 * @param {(x:number,y:number)=>[number,number]} [o.map] frame → canvas coordinates
 * @param {string|null} [o.color] single color; null uses OpenPose colors
 */
export function drawSkeleton(ctx, frame, { map = (x, y) => [x, y], color = null, alpha = 1, lineWidth = 4, radius = 3.5, minConf = 0.1, dash = null, outline = null } = {}) {
  if (!frame) return;
  const ok = (j) => frame[j * 3 + 2] > minConf;
  const P = (j) => map(frame[j * 3], frame[j * 3 + 1]);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  if (dash) ctx.setLineDash(dash);
  const rgb = (j) => `rgb(${COLORS[j].join(',')})`;
  if (outline) {
    ctx.strokeStyle = outline;
    ctx.lineWidth = lineWidth + 3;
    for (const [a, b] of PAIRS) {
      if (!ok(a) || !ok(b)) continue;
      const [x1, y1] = P(a);
      const [x2, y2] = P(b);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  }
  ctx.lineWidth = lineWidth;
  for (const [a, b] of PAIRS) {
    if (!ok(a) || !ok(b)) continue;
    const [x1, y1] = P(a);
    const [x2, y2] = P(b);
    ctx.strokeStyle = color || rgb(b);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  if (radius > 0) {
    for (let j = 0; j < NUM_KP; j++) {
      if (!ok(j)) continue;
      const [x, y] = P(j);
      ctx.fillStyle = color || rgb(j);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}

function handsPoint(f) {
  const a = f[KP.LWrist * 3 + 2] > 0.1;
  const b = f[KP.RWrist * 3 + 2] > 0.1;
  if (a && b) return [(f[KP.LWrist * 3] + f[KP.RWrist * 3]) / 2, (f[KP.LWrist * 3 + 1] + f[KP.RWrist * 3 + 1]) / 2];
  if (a) return [f[KP.LWrist * 3], f[KP.LWrist * 3 + 1]];
  if (b) return [f[KP.RWrist * 3], f[KP.RWrist * 3 + 1]];
  return null;
}

/** Polyline through the hands' positions for frames [from, to]. */
export function drawTrail(ctx, frames, from, to, { map, color, alpha = 1, lineWidth = 2 }) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  let started = false;
  for (let i = Math.max(0, from); i <= Math.min(frames.length - 1, to); i++) {
    const p = handsPoint(frames[i]);
    if (!p) continue;
    const [x, y] = map(p[0], p[1]);
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else ctx.lineTo(x, y);
  }
  if (started) ctx.stroke();
  ctx.restore();
}

/** Bounds (canonical units) that contain all confident joints of the given frames. */
export function canonicalBounds(frameLists, margin = 0.25) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const frames of frameLists) {
    for (const f of frames) {
      for (let j = 0; j < NUM_KP; j++) {
        if (f[j * 3 + 2] > 0.1) {
          minX = Math.min(minX, f[j * 3]);
          maxX = Math.max(maxX, f[j * 3]);
          minY = Math.min(minY, f[j * 3 + 1]);
          maxY = Math.max(maxY, f[j * 3 + 1]);
        }
      }
    }
  }
  if (!Number.isFinite(minX)) return { minX: -2, maxX: 2, minY: -2, maxY: 2 };
  return { minX: minX - margin, maxX: maxX + margin, minY: minY - margin, maxY: maxY + margin };
}

/** Map canonical (x right = pitcher, y up) into a w×h box preserving aspect. */
export function canonicalMapper(bounds, w, h, pad = 12) {
  const bw = bounds.maxX - bounds.minX;
  const bh = bounds.maxY - bounds.minY;
  const s = Math.min((w - 2 * pad) / bw, (h - 2 * pad) / bh);
  const ox = pad + (w - 2 * pad - bw * s) / 2;
  const oy = pad + (h - 2 * pad - bh * s) / 2;
  return (x, y) => [ox + (x - bounds.minX) * s, oy + (bounds.maxY - y) * s];
}

/**
 * Canonical comparison scene: ground line, pitcher arrow, both skeletons and
 * optional hand-path trails.
 */
export function drawScene(ctx, w, h, { bounds, user, pro, userTrail, proTrail, caption }) {
  const map = canonicalMapper(bounds, w, h, 16);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = cssVar('--surface-1', '#fff');
  ctx.fillRect(0, 0, w, h);
  // Hairline grid every 0.5 TL.
  ctx.strokeStyle = cssVar('--grid', '#e1e0d9');
  ctx.lineWidth = 1;
  for (let gx = Math.ceil(bounds.minX * 2) / 2; gx <= bounds.maxX; gx += 0.5) {
    const [x] = map(gx, 0);
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, 0);
    ctx.lineTo(Math.round(x) + 0.5, h);
    ctx.stroke();
  }
  for (let gy = Math.ceil(bounds.minY * 2) / 2; gy <= bounds.maxY; gy += 0.5) {
    const [, y] = map(0, gy);
    ctx.beginPath();
    ctx.moveTo(0, Math.round(y) + 0.5);
    ctx.lineTo(w, Math.round(y) + 0.5);
    ctx.stroke();
  }
  // Pitcher direction.
  ctx.fillStyle = cssVar('--text-muted', '#898781');
  ctx.font = '12px system-ui, -apple-system, "Segoe UI", sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText('toward pitcher →', w - 10, 18);
  if (caption) {
    ctx.textAlign = 'left';
    ctx.fillText(caption, 10, 18);
  }
  const userColor = cssVar('--series-user', '#2a78d6');
  const proColor = cssVar('--series-pro', '#eb6834');
  const surface = cssVar('--surface-1', '#fff');
  if (proTrail) drawTrail(ctx, proTrail.frames, proTrail.from, proTrail.to, { map, color: proColor, alpha: 0.35 });
  if (userTrail) drawTrail(ctx, userTrail.frames, userTrail.from, userTrail.to, { map, color: userColor, alpha: 0.35 });
  if (pro) drawSkeleton(ctx, pro, { map, color: proColor, lineWidth: 4, radius: 3, outline: surface, alpha: 0.9 });
  if (user) drawSkeleton(ctx, user, { map, color: userColor, lineWidth: 4, radius: 3, outline: surface });
  return map;
}
