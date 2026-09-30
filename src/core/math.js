// Small numeric helpers shared by the analysis pipeline. Pure functions, no DOM.

export const DEG = 180 / Math.PI;

export function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function mean(arr) {
  if (!arr.length) return NaN;
  let s = 0;
  for (const v of arr) s += v;
  return s / arr.length;
}

export function median(arr) {
  const v = arr.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return NaN;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

export function std(arr) {
  const m = mean(arr);
  return Math.sqrt(mean(arr.map((v) => (v - m) ** 2)));
}

export function argMax(arr, from = 0, to = arr.length) {
  let best = -1;
  let bestV = -Infinity;
  for (let i = Math.max(0, from); i < Math.min(arr.length, to); i++) {
    if (Number.isFinite(arr[i]) && arr[i] > bestV) {
      bestV = arr[i];
      best = i;
    }
  }
  return best;
}

export function argMin(arr, from = 0, to = arr.length) {
  let best = -1;
  let bestV = Infinity;
  for (let i = Math.max(0, from); i < Math.min(arr.length, to); i++) {
    if (Number.isFinite(arr[i]) && arr[i] < bestV) {
      bestV = arr[i];
      best = i;
    }
  }
  return best;
}

/** Angle ABC (at B) in degrees, 2D. */
export function angle3(ax, ay, bx, by, cx, cy) {
  const v1x = ax - bx;
  const v1y = ay - by;
  const v2x = cx - bx;
  const v2y = cy - by;
  const n = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y);
  if (n < 1e-9) return NaN;
  return Math.acos(clamp((v1x * v2x + v1y * v2y) / n, -1, 1)) * DEG;
}

/** Gaussian smoothing of a series that may contain NaN gaps (NaNs are skipped). */
export function gaussianSmooth(series, sigma) {
  if (!(sigma > 0)) return series.slice();
  const r = Math.max(1, Math.ceil(sigma * 2.5));
  const w = [];
  for (let k = -r; k <= r; k++) w.push(Math.exp(-(k * k) / (2 * sigma * sigma)));
  const out = new Array(series.length);
  for (let i = 0; i < series.length; i++) {
    if (!Number.isFinite(series[i])) {
      out[i] = NaN;
      continue;
    }
    let s = 0;
    let ws = 0;
    for (let k = -r; k <= r; k++) {
      const v = series[i + k];
      if (Number.isFinite(v)) {
        s += v * w[k + r];
        ws += w[k + r];
      }
    }
    out[i] = s / ws;
  }
  return out;
}

/** Central-difference derivative (per frame). NaN-safe. */
export function derivative(series) {
  const n = series.length;
  const out = new Array(n).fill(NaN);
  for (let i = 0; i < n; i++) {
    const a = series[Math.max(0, i - 1)];
    const b = series[Math.min(n - 1, i + 1)];
    const span = Math.min(n - 1, i + 1) - Math.max(0, i - 1);
    if (Number.isFinite(a) && Number.isFinite(b) && span > 0) out[i] = (b - a) / span;
  }
  return out;
}

/** Linearly fill NaN gaps no longer than maxGap; leading/trailing gaps are held. */
export function fillGaps(series, maxGap = Infinity) {
  const out = series.slice();
  const n = out.length;
  let i = 0;
  while (i < n) {
    if (Number.isFinite(out[i])) {
      i++;
      continue;
    }
    let j = i;
    while (j < n && !Number.isFinite(out[j])) j++;
    const gap = j - i;
    const left = i > 0 ? out[i - 1] : NaN;
    const right = j < n ? out[j] : NaN;
    if (gap <= maxGap) {
      for (let k = i; k < j; k++) {
        if (Number.isFinite(left) && Number.isFinite(right)) out[k] = lerp(left, right, (k - i + 1) / (gap + 1));
        else if (Number.isFinite(left)) out[k] = left;
        else if (Number.isFinite(right)) out[k] = right;
      }
    }
    i = j;
  }
  return out;
}

/**
 * Monotone cubic Hermite interpolation (Fritsch–Carlson). xs must be strictly increasing.
 * Values outside the range are held at the ends.
 */
export function makeMonotoneInterpolator(xs, ys) {
  const n = xs.length;
  if (n === 1) return () => ys[0];
  const d = [];
  const m = new Array(n);
  for (let i = 0; i < n - 1; i++) d.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]));
  m[0] = d[0];
  m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (d[i] === 0) {
      m[i] = 0;
      m[i + 1] = 0;
      continue;
    }
    const a = m[i] / d[i];
    const b = m[i + 1] / d[i];
    const s = a * a + b * b;
    if (s > 9) {
      const t = 3 / Math.sqrt(s);
      m[i] = t * a * d[i];
      m[i + 1] = t * b * d[i];
    }
  }
  // Start and end with zero velocity so motions ease in and out of holds.
  m[0] = 0;
  m[n - 1] = 0;
  return (x) => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let i = 0;
    while (i < n - 2 && x > xs[i + 1]) i++;
    const h = xs[i + 1] - xs[i];
    const t = (x - xs[i]) / h;
    const t2 = t * t;
    const t3 = t2 * t;
    return (
      (2 * t3 - 3 * t2 + 1) * ys[i] +
      (t3 - 2 * t2 + t) * h * m[i] +
      (-2 * t3 + 3 * t2) * ys[i + 1] +
      (t3 - t2) * h * m[i + 1]
    );
  };
}

/** Deterministic PRNG (mulberry32) so synthetic data is reproducible. */
export function rng(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal sample from a uniform PRNG (Box–Muller). */
export function gauss(rand) {
  const u = Math.max(1e-12, rand());
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function round(v, digits = 2) {
  const p = 10 ** digits;
  return Math.round(v * p) / p;
}
