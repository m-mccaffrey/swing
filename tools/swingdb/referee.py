"""Referee: fuse several pose models' answers joint by joint. Exact port of
src/core/referee.js (see there for the idea); standard library only."""

import math

COCO_GROUP = [0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
NGROUP = 7


def _fin(v):
    """A finite number (numpy scalars included; None or text are not)."""
    try:
        return math.isfinite(v)
    except TypeError:
        return False


def _median(values):
    v = sorted(x for x in values if _fin(x))
    if not v:
        return math.nan
    m = len(v) >> 1
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2


def consensus_height(cands):
    lo, hi = math.inf, -math.inf
    for j in range(17):
        y = _median([c[j][1] for c in cands])
        if _fin(y):
            lo, hi = min(lo, y), max(hi, y)
    return max(1.0, (hi - lo) * 1.1)


ZOOM_SCALE = 0.6
ZOOM_JOINTS = (5, 6, 7, 8, 9, 10)


def zoom_crop(cands):
    """(x0, y0, side) of the zoomed look at the arms, or None. Mirrors zoomCrop() in referee.js."""
    full = [c for c in cands if c is not None]
    if not full:
        return None
    med = [(_median([c[j][0] for c in full]), _median([c[j][1] for c in full])) for j in range(17)]
    ys = [p[1] for p in med if _fin(p[1])]
    if len(ys) < 2:
        return None
    h = max(1.0, (max(ys) - min(ys)) * 1.1)
    pts = [med[j] for j in (7, 8, 9, 10) if _fin(med[j][0]) and _fin(med[j][1])]
    if not pts:
        return None
    xs, yy = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = sum(xs) / len(xs), sum(yy) / len(yy)
    side = max(ZOOM_SCALE * h, 1.4 * max(max(xs) - min(xs), max(yy) - min(yy)))
    return cx - side / 2, cy - side / 2, side


def raw_features(cands, c, j, h):
    x, y, conf = cands[c][j]
    mx = _median([p[j][0] for p in cands])
    my = _median([p[j][1] for p in cands])
    d_med = math.hypot(x - mx, y - my) / h
    others = [math.hypot(x - cands[o][j][0], y - cands[o][j][1]) / h
              for o in range(len(cands)) if o != c and _fin(cands[o][j][0])]
    agree5 = sum(1 for d in others if d < 0.05)
    agree10 = sum(1 for d in others if d < 0.1)
    nearest = min(others) if others else 1.0
    conf_rank = sum(1 for p in cands if _fin(p[j][2]) and p[j][2] > conf)
    geo = wd = 0.0
    if j in (9, 10):
        e, s = (7, 5) if j == 9 else (8, 6)
        P = cands[c]
        fa = math.hypot(x - P[e][0], y - P[e][1])
        ua = math.hypot(P[e][0] - P[s][0], P[e][1] - P[s][1])
        geo = fa / max(ua, 1e-3)
        ow = 10 if j == 9 else 9
        wd = math.hypot(x - P[ow][0], y - P[ow][1]) / h
    return [conf, d_med, agree5, agree10, nearest, conf_rank, geo, wd]


def feature_vector(raw, c, j, K):
    conf, d_med, nearest = raw[0], raw[1], raw[4]
    M = [1.0 if k == c else 0.0 for k in range(K)]
    G = [1.0 if g == COCO_GROUP[j] else 0.0 for g in range(NGROUP)]
    ld = math.log(d_med + 1e-3)
    return M + G + list(raw) + [ld, math.log(nearest + 1e-3)] + [conf * g for g in G] + [ld * g for g in G] + [conf * m for m in M]


def fuse(cands, referee):
    """cands: list (referee candidate order) of 17x[x, y, conf] or None.
    Returns (17x[x, y, probability], chosen candidate index per joint) or None.
    With a soft_radius, answers that close to the winner are averaged (weighted
    by probability), as in referee.js."""
    K = len(cands)
    nan_row = [math.nan, math.nan, math.nan]
    full = [c if c is not None else [nan_row] * 17 for c in cands]
    if not any(_fin(q[0]) for c in full for q in c):
        return None
    if K != len(referee["candidates"]):
        raise ValueError(f"referee expects {len(referee['candidates'])} candidates, got {K}")
    h = consensus_height(full)
    radius = referee.get("soft_radius") or 0
    out, chosen = [], []
    for j in range(17):
        best, best_p = -1, -1.0
        probs = [math.nan] * K
        for c in range(K):
            if not _fin(full[c][j][0]):
                continue
            z = feature_vector(raw_features(full, c, j, h), c, j, K)
            s = referee["bias"]
            for w, v in zip(referee["weights"], z):
                s += w * v
            p = 1.0 / (1.0 + math.exp(-s))
            probs[c] = p
            if p > best_p:
                best, best_p = c, p
        chosen.append(best)
        if best < 0:
            out.append([math.nan, math.nan, 0.0])
            continue
        x, y = full[best][j][0], full[best][j][1]
        if radius > 0:
            sw = sx = sy = 0.0
            for c in range(K):
                if not _fin(probs[c]) or math.hypot(full[c][j][0] - x, full[c][j][1] - y) / h > radius:
                    continue
                sw += probs[c]
                sx += probs[c] * full[c][j][0]
                sy += probs[c] * full[c][j][1]
            x, y = sx / sw, sy / sw
        out.append([x, y, best_p])
    return out, chosen
