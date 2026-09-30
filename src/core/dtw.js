// Dynamic time warping used to line up the user's swing with the pro's.
// The pro sequence (stance → finish) must be matched completely; the user's
// sequence starts at their chosen stance but may run on after their finish
// (open end), so the best end column is chosen by normalized cost.

/**
 * @param {(i:number, j:number) => number} cost local cost between pro frame i and user frame j
 * @param {number} n pro length
 * @param {number} m user length
 * @param {{stepPenalty?: number, openEnd?: boolean, minEndRatio?: number}} opts
 * @returns {{path: number[][], cost: number, endJ: number}}
 */
export function dtw(cost, n, m, { stepPenalty = 0.05, openEnd = true, minEndRatio = 0.25 } = {}) {
  if (!n || !m) throw new Error('DTW needs two non-empty sequences');
  const D = new Float64Array(n * m).fill(Infinity);
  const B = new Uint8Array(n * m); // 0 = diag, 1 = from i-1 (pro advances), 2 = from j-1 (user advances)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      const c = cost(i, j);
      if (i === 0 && j === 0) {
        D[0] = c;
        continue;
      }
      let best = Infinity;
      let arg = 0;
      if (i > 0 && j > 0 && D[(i - 1) * m + j - 1] < best) {
        best = D[(i - 1) * m + j - 1];
        arg = 0;
      }
      if (i > 0 && D[(i - 1) * m + j] + stepPenalty < best) {
        best = D[(i - 1) * m + j] + stepPenalty;
        arg = 1;
      }
      if (j > 0 && D[i * m + j - 1] + stepPenalty < best) {
        best = D[i * m + j - 1] + stepPenalty;
        arg = 2;
      }
      D[i * m + j] = best + c;
      B[i * m + j] = arg;
    }
  }
  let endJ = m - 1;
  if (openEnd) {
    let bestNorm = Infinity;
    // Don't let the user's whole swing collapse onto a sliver of frames.
    const minJ = Math.min(m - 1, Math.floor(n * minEndRatio));
    for (let j = minJ; j < m; j++) {
      const norm = D[(n - 1) * m + j] / (n + j);
      if (norm < bestNorm) {
        bestNorm = norm;
        endJ = j;
      }
    }
  }
  const path = [];
  let i = n - 1;
  let j = endJ;
  while (true) {
    path.push([i, j]);
    if (i === 0 && j === 0) break;
    const b = B[i * m + j];
    if (i === 0) j--;
    else if (j === 0) i--;
    else if (b === 0) {
      i--;
      j--;
    } else if (b === 1) i--;
    else j--;
  }
  path.reverse();
  return { path, cost: D[(n - 1) * m + endJ] / path.length, endJ };
}

/**
 * For every pro index, the (median) user index it was aligned to, and for every
 * user index up to endJ, the median pro index (user frames after endJ map to
 * the last pro frame).
 */
export function pathMaps(path, n, m) {
  const proToUser = Array.from({ length: n }, () => []);
  const userToPro = Array.from({ length: m }, () => []);
  for (const [i, j] of path) {
    proToUser[i].push(j);
    userToPro[j].push(i);
  }
  const med = (a) => {
    const s = a.slice().sort((x, y) => x - y);
    return s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
  };
  const p2u = proToUser.map(med);
  let last = 0;
  const u2p = userToPro.map((a) => {
    if (a.length) last = med(a);
    return a.length ? last : NaN;
  });
  // Frames after the matched end map to the pro's last frame.
  for (let j = 0; j < m; j++) if (!Number.isFinite(u2p[j])) u2p[j] = j === 0 ? 0 : u2p[j - 1];
  return { proToUser: p2u, userToPro: u2p };
}
