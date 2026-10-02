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
export function drawSkeleton(ctx, frame, { map = (x, y) => [x, y], color = null, alpha = 1, lineWidth = 4, radius = 3.5, minConf = 0.1, dash = null, outline = null, hollowBelow = 0 } = {}) {
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
      ctx.beginPath();
      if (frame[j * 3 + 2] < hollowBelow) {
        // An estimate (the pose model wasn't sure): a ring instead of a dot.
        ctx.arc(x, y, radius * 1.5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,255,255,0.85)';
        ctx.fill();
        ctx.lineWidth = Math.max(1.5, radius * 0.7);
        ctx.strokeStyle = color || rgb(j);
        ctx.stroke();
      } else {
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = color || rgb(j);
        ctx.fill();
      }
    }
  }
  ctx.restore();
}

/**
 * The pose model's own keypoints, unprocessed: small white dots, fainter and
 * hollow where the model reported low confidence.
 */
export function drawPoints(ctx, frame, { map = (x, y) => [x, y], radius = 2.5, minConf = 0.25 } = {}) {
  if (!frame) return;
  ctx.save();
  for (let j = 0; j < NUM_KP; j++) {
    const c = frame[j * 3 + 2];
    if (!(c > 0)) continue;
    const [x, y] = map(frame[j * 3], frame[j * 3 + 1]);
    ctx.globalAlpha = 0.45 + 0.55 * Math.min(1, c);
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.lineWidth = Math.max(1, radius * 0.6);
    ctx.strokeStyle = 'rgba(0,0,0,0.8)';
    ctx.stroke();
    if (c > minConf) {
      ctx.fillStyle = '#fff';
      ctx.fill();
    } else {
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = Math.max(1, radius * 0.4);
      ctx.stroke();
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

/**
 * Evidence overlay: ring the measured joints, join pairs (widths, angles,
 * heights between two joints) and, for movements measured from the stance,
 * mark where the joints were at the stance with an arrow to where they are now.
 */
export function drawHighlights(ctx, map, frame, joints, color, { pair = false, stanceFrame = null, scale = 1 } = {}) {
  if (!frame || !joints?.length) return;
  const ok = (f, j) => f && f[j * 3 + 2] > 0.05;
  const P = (f, j) => map(f[j * 3], f[j * 3 + 1]);
  ctx.save();
  ctx.lineCap = 'round';
  const surface = 'rgba(255,255,255,0.9)';
  if (pair) {
    const pts = joints.filter((j) => ok(frame, j)).map((j) => P(frame, j));
    ctx.strokeStyle = color;
    ctx.lineWidth = 3 * scale;
    ctx.setLineDash([6 * scale, 4 * scale]);
    ctx.beginPath();
    pts.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (stanceFrame) {
    for (const j of joints) {
      if (!ok(frame, j) || !ok(stanceFrame, j)) continue;
      const [x0, y0] = P(stanceFrame, j);
      const [x1, y1] = P(frame, j);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2 * scale;
      ctx.setLineDash([3 * scale, 3 * scale]);
      ctx.beginPath();
      ctx.arc(x0, y0, 6 * scale, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      const d = Math.hypot(x1 - x0, y1 - y0);
      if (d > 10 * scale) {
        const ux = (x1 - x0) / d;
        const uy = (y1 - y0) / d;
        const ex = x1 - ux * 10 * scale;
        const ey = y1 - uy * 10 * scale;
        ctx.beginPath();
        ctx.moveTo(x0 + ux * 6 * scale, y0 + uy * 6 * scale);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - ux * 6 * scale - uy * 4 * scale, ey - uy * 6 * scale + ux * 4 * scale);
        ctx.lineTo(ex - ux * 6 * scale + uy * 4 * scale, ey - uy * 6 * scale - ux * 4 * scale);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
      }
    }
  }
  for (const j of joints) {
    if (!ok(frame, j)) continue;
    const [x, y] = P(frame, j);
    ctx.lineWidth = 5 * scale;
    ctx.strokeStyle = surface;
    ctx.beginPath();
    ctx.arc(x, y, 9 * scale, 0, Math.PI * 2);
    ctx.stroke();
    ctx.lineWidth = 2.5 * scale;
    ctx.strokeStyle = color;
    ctx.stroke();
  }
  ctx.restore();
}
