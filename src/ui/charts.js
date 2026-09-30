// Minimal SVG line charts (user vs. pro over the aligned swing timeline)
// with a crosshair tooltip, legend and direct end labels.

const NS = 'http://www.w3.org/2000/svg';

function el(name, attrs = {}, parent = null) {
  const e = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (parent) parent.appendChild(e);
  return e;
}

function niceTicks(min, max, count = 4) {
  const span = max - min || 1;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || 10 * mag;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(Math.round(v / step) * step);
  return ticks;
}

const PHASE_SHORT = { load: 'Load', footPlant: 'Plant', contact: 'Contact', finish: 'Finish' };

/**
 * Render a line chart into `container` (re-renders on resize).
 * @param {HTMLElement} container
 * @param {object} o
 * @param {string} o.title
 * @param {number[]} o.t x values (seconds relative to contact)
 * @param {{name:string, colorVar:string, values:number[]}[]} o.series
 * @param {Record<string, number>} [o.phaseTimes]
 * @param {(v:number)=>string} o.format value formatter
 */
export function lineChart(container, o) {
  container.classList.add('chart');
  container.textContent = '';
  const head = document.createElement('div');
  head.className = 'chart-head';
  const h = document.createElement('h4');
  h.textContent = o.title;
  head.appendChild(h);
  const legend = document.createElement('div');
  legend.className = 'legend';
  for (const s of o.series) {
    const item = document.createElement('span');
    item.className = 'legend-item';
    const key = document.createElement('span');
    key.className = 'line-key';
    key.style.background = `var(${s.colorVar})`;
    item.append(key, document.createTextNode(s.name));
    legend.appendChild(item);
  }
  head.appendChild(legend);
  container.appendChild(head);
  if (o.subtitle) {
    const sub = document.createElement('p');
    sub.className = 'chart-sub';
    sub.textContent = o.subtitle;
    container.appendChild(sub);
  }
  const wrap = document.createElement('div');
  wrap.className = 'chart-plot';
  container.appendChild(wrap);
  const tip = document.createElement('div');
  tip.className = 'chart-tip';
  tip.hidden = true;
  wrap.appendChild(tip);

  const draw = () => {
    wrap.querySelectorAll('svg').forEach((s) => s.remove());
    const W = Math.max(240, wrap.clientWidth);
    const H = o.height || 170;
    const m = { l: 44, r: 58, t: 14, b: 24 };
    const svg = el('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': o.title });
    wrap.insertBefore(svg, tip);
    const t = o.t;
    const xs = [t[0], t[t.length - 1]];
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of o.series) for (const v of s.values) if (Number.isFinite(v)) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    if (!Number.isFinite(lo)) {
      lo = 0;
      hi = 1;
    }
    const padY = (hi - lo) * 0.12 || 1;
    lo -= padY;
    hi += padY;
    const X = (v) => m.l + ((v - xs[0]) / (xs[1] - xs[0] || 1)) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);

    // Grid + y ticks.
    for (const v of niceTicks(lo, hi, 4)) {
      el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: 'grid' }, svg);
      const lab = el('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', class: 'tick' }, svg);
      lab.textContent = o.format(v);
    }
    // x ticks (seconds from contact).
    for (const v of niceTicks(xs[0], xs[1], 5)) {
      const lab = el('text', { x: X(v), y: H - 6, 'text-anchor': 'middle', class: 'tick' }, svg);
      lab.textContent = Math.abs(v) < 1e-9 ? '0 s' : `${v > 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}`;
    }
    el('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: 'axis' }, svg);
    // Phase markers.
    for (const [k, label] of Object.entries(PHASE_SHORT)) {
      const pt = o.phaseTimes?.[k];
      if (!Number.isFinite(pt)) continue;
      el('line', { x1: X(pt), x2: X(pt), y1: m.t, y2: H - m.b, class: 'phase-line' }, svg);
      const lab = el('text', { x: X(pt) + 3, y: m.t + 8, class: 'phase-tick' }, svg);
      lab.textContent = label;
    }
    // Series lines + end labels.
    const ends = [];
    for (const s of o.series) {
      let d = '';
      let pen = false;
      let last = null;
      s.values.forEach((v, i) => {
        if (!Number.isFinite(v)) {
          pen = false;
          return;
        }
        d += `${pen ? 'L' : 'M'}${X(t[i]).toFixed(1)},${Y(v).toFixed(1)}`;
        pen = true;
        last = [X(t[i]), Y(v)];
      });
      el('path', { d, class: 'series', style: `stroke: var(${s.colorVar})` }, svg);
      if (last) ends.push({ s, x: last[0], y: last[1] });
    }
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].ly = ends[i - 1].ly ?? ends[i - 1].y + 13;
    for (const e of ends) {
      const ly = e.ly ?? e.y;
      if (ly !== e.y) el('line', { x1: e.x + 2, y1: e.y, x2: e.x + 8, y2: ly, class: 'leader' }, svg);
      const lab = el('text', { x: e.x + 10, y: ly + 4, class: 'end-label' }, svg);
      lab.textContent = e.s.short || e.s.name;
    }

    // Crosshair + tooltip.
    const cross = el('line', { y1: m.t, y2: H - m.b, class: 'crosshair', visibility: 'hidden' }, svg);
    const dots = o.series.map((s) => el('circle', { r: 4, class: 'cross-dot', style: `fill: var(${s.colorVar})`, visibility: 'hidden' }, svg));
    const hit = el('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent', tabindex: 0 }, svg);
    const show = (i) => {
      const x = X(t[i]);
      cross.setAttribute('x1', x);
      cross.setAttribute('x2', x);
      cross.setAttribute('visibility', 'visible');
      tip.textContent = '';
      const tt = document.createElement('div');
      tt.className = 'tip-time';
      tt.textContent = Math.abs(t[i]) < 0.005 ? 'at contact' : `${t[i] > 0 ? '+' : '−'}${Math.abs(t[i]).toFixed(2)} s from contact`;
      tip.appendChild(tt);
      o.series.forEach((s, k) => {
        const v = s.values[i];
        const row = document.createElement('div');
        row.className = 'tip-row';
        const key = document.createElement('span');
        key.className = 'line-key';
        key.style.background = `var(${s.colorVar})`;
        const val = document.createElement('strong');
        val.textContent = o.format(v);
        const name = document.createElement('span');
        name.className = 'tip-name';
        name.textContent = s.name;
        row.append(key, val, name);
        tip.appendChild(row);
        if (Number.isFinite(v)) {
          dots[k].setAttribute('cx', x);
          dots[k].setAttribute('cy', Y(v));
          dots[k].setAttribute('visibility', 'visible');
        } else dots[k].setAttribute('visibility', 'hidden');
      });
      tip.hidden = false;
      const left = Math.min(W - tip.offsetWidth - 4, Math.max(4, x + 12));
      tip.style.left = `${left}px`;
      tip.style.top = `${m.t}px`;
    };
    const hide = () => {
      cross.setAttribute('visibility', 'hidden');
      dots.forEach((d) => d.setAttribute('visibility', 'hidden'));
      tip.hidden = true;
    };
    const nearest = (clientX) => {
      const r = svg.getBoundingClientRect();
      const x = clientX - r.left;
      let best = 0;
      for (let i = 1; i < t.length; i++) if (Math.abs(X(t[i]) - x) < Math.abs(X(t[best]) - x)) best = i;
      return best;
    };
    let focusIdx = t.findIndex((v) => v >= 0);
    hit.addEventListener('pointermove', (ev) => show(nearest(ev.clientX)));
    hit.addEventListener('pointerleave', hide);
    hit.addEventListener('focus', () => show(Math.max(0, focusIdx)));
    hit.addEventListener('blur', hide);
    hit.addEventListener('keydown', (ev) => {
      if (ev.key === 'ArrowRight' || ev.key === 'ArrowLeft') {
        focusIdx = Math.max(0, Math.min(t.length - 1, focusIdx + (ev.key === 'ArrowRight' ? 1 : -1)));
        show(focusIdx);
        ev.preventDefault();
      }
    });
  };
  draw();
  if (typeof ResizeObserver === 'function') {
    let lastW = wrap.clientWidth;
    const ro = new ResizeObserver(() => {
      if (Math.abs(wrap.clientWidth - lastW) > 4) {
        lastW = wrap.clientWidth;
        draw();
      }
    });
    ro.observe(wrap);
  }
}
