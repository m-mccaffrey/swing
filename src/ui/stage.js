// A video (or blank backdrop) with a canvas overlay sized to the video's
// pixel grid, plus a small player that keeps an overlay in sync.

import { fitCanvas } from './draw.js';

export class Stage {
  /**
   * @param {HTMLElement} wrap container (.stage-wrap)
   * @param {{video?: HTMLVideoElement|null, width: number, height: number, label?: string}} o
   */
  constructor(wrap, { video = null, width, height, label = 'Keypoints only (no video)' }) {
    this.wrap = wrap;
    this.label = label;
    this.video = video;
    this.width = width;
    this.height = height;
    wrap.textContent = '';
    this.el = document.createElement('div');
    this.el.className = 'stage';
    const ar = width / height;
    this.el.style.aspectRatio = `${width} / ${height}`;
    this.el.style.width = `min(100%, calc(70vh * ${ar.toFixed(4)}))`;
    if (video) {
      video.controls = false;
      video.playsInline = true;
      video.muted = true;
      this.el.appendChild(video);
    } else {
      this.bg = document.createElement('canvas');
      this.el.appendChild(this.bg);
    }
    this.overlay = document.createElement('canvas');
    this.overlay.className = 'overlay';
    this.el.appendChild(this.overlay);
    wrap.appendChild(this.el);
    this.drawFn = null;
    if (typeof ResizeObserver === 'function') {
      this.ro = new ResizeObserver(() => this.redraw());
      this.ro.observe(this.el);
    }
  }

  /** Map image pixel coordinates to overlay CSS pixels. */
  mapper() {
    const w = this.el.clientWidth;
    const h = this.el.clientHeight;
    const sx = w / this.width;
    const sy = h / this.height;
    return (x, y) => [x * sx, y * sy];
  }

  /** Scale factor for line widths so skeletons look the same at any size. */
  unit() {
    return Math.max(0.6, Math.min(2, this.el.clientWidth / 480));
  }

  draw(fn) {
    this.drawFn = fn;
    this.redraw();
  }

  redraw() {
    const w = this.el.clientWidth;
    const h = this.el.clientHeight;
    if (!w || !h) return;
    if (this.bg) {
      const g = fitCanvas(this.bg, w, h);
      const grad = g.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, '#20262e');
      grad.addColorStop(0.75, '#1a1f25');
      grad.addColorStop(1, '#2b3a2a');
      g.fillStyle = grad;
      g.fillRect(0, 0, w, h);
      g.fillStyle = 'rgba(255,255,255,0.35)';
      g.font = '12px system-ui, sans-serif';
      g.fillText(this.label, 10, 20);
    }
    const ctx = fitCanvas(this.overlay, w, h);
    ctx.clearRect(0, 0, w, h);
    if (this.drawFn) this.drawFn(ctx, this.mapper(), this.unit());
  }

  destroy() {
    this.ro?.disconnect();
  }
}

/**
 * Plays through analyzed frames. With a video it plays the real video and
 * reports the nearest analyzed frame; without one it advances a virtual clock.
 */
export class FramePlayer {
  constructor({ video = null, times, onFrame, onState = () => {} }) {
    this.video = video;
    this.times = times;
    this.onFrame = onFrame;
    this.onState = onState;
    this.playing = false;
    this.index = 0;
    this.endIndex = times.length - 1;
    this.raf = 0;
  }

  nearest(t) {
    const ts = this.times;
    let lo = 0;
    let hi = ts.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (ts[mid] <= t) lo = mid;
      else hi = mid;
    }
    return Math.abs(ts[hi] - t) < Math.abs(ts[lo] - t) ? hi : lo;
  }

  async seek(i) {
    this.pause();
    this.index = Math.max(0, Math.min(this.times.length - 1, i));
    if (this.video) {
      const t = this.times[this.index];
      await new Promise((resolve) => {
        if (Math.abs(this.video.currentTime - t) < 1e-4) return resolve();
        const done = () => {
          this.video.removeEventListener('seeked', done);
          resolve();
        };
        this.video.addEventListener('seeked', done);
        this.video.currentTime = t;
        setTimeout(done, 1500);
      });
    }
    this.onFrame(this.index);
  }

  play(rate = 0.5, from = null, to = this.times.length - 1) {
    if (this.playing) return;
    if (from != null) this.index = from;
    if (this.index >= to) this.index = from ?? 0;
    this.endIndex = to;
    this.playing = true;
    this.onState(true);
    const startT = this.times[this.index];
    const endT = this.times[to];
    if (this.video) {
      this.video.playbackRate = rate;
      this.video.currentTime = startT;
      this.video.play().catch(() => this.pause());
      const tick = () => {
        if (!this.playing) return;
        const t = this.video.currentTime;
        this.index = this.nearest(t);
        this.onFrame(this.index);
        if (t >= endT || this.video.ended) {
          this.pause();
          return;
        }
        this.raf = requestAnimationFrame(tick);
      };
      this.raf = requestAnimationFrame(tick);
    } else {
      let last = performance.now();
      let t = startT;
      const tick = (now) => {
        if (!this.playing) return;
        t += ((now - last) / 1000) * rate;
        last = now;
        this.index = this.nearest(t);
        this.onFrame(this.index);
        if (t >= endT) {
          this.pause();
          return;
        }
        this.raf = requestAnimationFrame(tick);
      };
      this.raf = requestAnimationFrame(tick);
    }
  }

  pause() {
    if (!this.playing) return;
    this.playing = false;
    cancelAnimationFrame(this.raf);
    this.video?.pause();
    this.onState(false);
  }
}
