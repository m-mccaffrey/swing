"""Stance, pitcher-side and phase detection: a line-for-line Python port of the
parts of src/core/{math,sequence,metrics,phases}.js that the database tool
needs, so entries made here get the same defaults as the browser builder.

tools/tests/test_swingdb.py checks this module against the JavaScript on the
same inputs. If you change the detection logic, change both.
Standard library only.
"""

import math

from .body25 import KP, NUM_KP, LR_GROUPS, ARMS, HEAD, LEGS, empty_frame, swap_lr

NAN = float("nan")
MIN_CONF = 0.25  # sequence.js
P_MIN = 0.05  # metrics.js
BODY_TO_TORSO = 2.75
PHASE_KEYS = ["stance", "load", "footPlant", "contact", "extension", "finish"]


# ----------------------------------------------------------------- math.js


def isfin(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def jsround(x):
    """JavaScript Math.round (halves round up, unlike Python's round)."""
    return math.floor(x + 0.5)


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def mean(arr):
    if not arr:
        return NAN
    s = 0.0
    for v in arr:
        s += v
    return s / len(arr)


def median(arr):
    v = sorted(x for x in arr if isfin(x))
    if not v:
        return NAN
    m = len(v) >> 1
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2


def arg_max(arr, frm=0, to=None):
    to = len(arr) if to is None else to
    best, best_v = -1, -math.inf
    for i in range(max(0, frm), min(len(arr), to)):
        if isfin(arr[i]) and arr[i] > best_v:
            best_v, best = arr[i], i
    return best


def arg_min(arr, frm=0, to=None):
    to = len(arr) if to is None else to
    best, best_v = -1, math.inf
    for i in range(max(0, frm), min(len(arr), to)):
        if isfin(arr[i]) and arr[i] < best_v:
            best_v, best = arr[i], i
    return best


def gaussian_smooth(series, sigma):
    if not (sigma > 0):
        return list(series)
    r = max(1, math.ceil(sigma * 2.5))
    w = [math.exp(-(k * k) / (2 * sigma * sigma)) for k in range(-r, r + 1)]
    n = len(series)
    out = []
    for i in range(n):
        if not isfin(series[i]):
            out.append(NAN)
            continue
        s = 0.0
        ws = 0.0
        for k in range(-r, r + 1):
            idx = i + k
            if 0 <= idx < n and isfin(series[idx]):
                s += series[idx] * w[k + r]
                ws += w[k + r]
        out.append(s / ws)
    return out


def derivative(series):
    n = len(series)
    out = [NAN] * n
    for i in range(n):
        lo, hi = max(0, i - 1), min(n - 1, i + 1)
        a, b = series[lo], series[hi]
        span = hi - lo
        if isfin(a) and isfin(b) and span > 0:
            out[i] = (b - a) / span
    return out


def fill_gaps(series, max_gap=math.inf):
    out = list(series)
    n = len(out)
    i = 0
    while i < n:
        if isfin(out[i]):
            i += 1
            continue
        j = i
        while j < n and not isfin(out[j]):
            j += 1
        gap = j - i
        left = out[i - 1] if i > 0 else NAN
        right = out[j] if j < n else NAN
        if gap <= max_gap:
            for k in range(i, j):
                if isfin(left) and isfin(right):
                    t = (k - i + 1) / (gap + 1)
                    out[k] = left + (right - left) * t
                elif isfin(left):
                    out[k] = left
                elif isfin(right):
                    out[k] = right
        i = j
    return out


# ----------------------------------------------------------------- sequence.js


def _pt(f, j):
    return (f[j * 3], f[j * 3 + 1]) if f[j * 3 + 2] > MIN_CONF else None


def _mid_of(f, a, b):
    pa, pb = _pt(f, a), _pt(f, b)
    if pa and pb:
        return ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
    return pa or pb


def torso_length(frames, center=None, half_window=3):
    lo = 0 if center is None else max(0, center - half_window)
    hi = len(frames) if center is None else min(len(frames), center + half_window + 1)
    d = []
    for i in range(lo, hi):
        n, h = _pt(frames[i], KP["Neck"]), _pt(frames[i], KP["MidHip"])
        if n and h:
            d.append(math.hypot(n[0] - h[0], n[1] - h[1]))
    m = median(d)
    if isfin(m) and m > 1e-6:
        return m
    return NAN if center is None else torso_length(frames, None)


def _or1(v):
    """JavaScript `v || 1`."""
    return v if isfin(v) and v != 0 else 1


def body_scale(frames, center, half_window=2):
    lo = max(0, center - half_window)
    hi = min(len(frames), center + half_window + 1)

    def seg(f, a, b):
        p, q = _pt(f, a), _pt(f, b)
        return math.hypot(p[0] - q[0], p[1] - q[1]) if p and q else NAN

    def avg(a, b):
        if isfin(a) and isfin(b):
            return (a + b) / 2
        return a if isfin(a) else b

    d = []
    for i in range(lo, hi):
        f = frames[i]
        torso = seg(f, KP["Neck"], KP["MidHip"])
        thigh = avg(seg(f, KP["LHip"], KP["LKnee"]), seg(f, KP["RHip"], KP["RKnee"]))
        shin = avg(seg(f, KP["LKnee"], KP["LAnkle"]), seg(f, KP["RKnee"], KP["RAnkle"]))
        if all(isfin(v) for v in (torso, thigh, shin)):
            d.append((torso + thigh + shin) / BODY_TO_TORSO)
    m = median(d)
    return m if isfin(m) and m > 1e-6 else torso_length(frames, center, half_window)


def _group_cost(f, ref, pairs, swapped):
    cost, n = 0.0, 0
    for a, b in pairs:
        combos = ((a, b), (b, a)) if swapped else ((a, a), (b, b))
        for src, dst in combos:
            p, q = _pt(f, src), _pt(ref, dst)
            if p and q:
                cost += math.hypot(p[0] - q[0], p[1] - q[1])
                n += 1
    return cost / n if n else NAN


def fix_left_right_flicker(frames, ref_index=0):
    out = [list(f) for f in frames]
    if not out:
        return out

    def walk(frm, to, step):
        prev = out[frm]
        i = frm + step
        while (i <= to) if step > 0 else (i >= to):
            f = out[i]
            for pairs in LR_GROUPS:
                keep = _group_cost(f, prev, pairs, False)
                swap = _group_cost(f, prev, pairs, True)
                if isfin(keep) and isfin(swap) and swap < keep * 0.7:
                    f = swap_lr(f, pairs)
            out[i] = f
            merged = list(prev)
            for j in range(NUM_KP):
                if f[j * 3 + 2] > MIN_CONF:
                    merged[j * 3:j * 3 + 3] = f[j * 3:j * 3 + 3]
            prev = merged
            i += step

    r = clamp(ref_index, 0, len(out) - 1)
    walk(r, len(out) - 1, 1)
    walk(r, 0, -1)
    return out


def clean_sequence(frames, fps=30, max_gap_sec=0.2, smooth_sec=0.018):
    n = len(frames)
    out = [empty_frame() for _ in range(n)]
    max_gap = max(1, jsround(max_gap_sec * fps))
    sigma = smooth_sec * fps
    for j in range(NUM_KP):
        xs, ys, cs = [], [], []
        for f in frames:
            ok = f[j * 3 + 2] > MIN_CONF
            xs.append(f[j * 3] if ok else NAN)
            ys.append(f[j * 3 + 1] if ok else NAN)
            cs.append(f[j * 3 + 2])
        fx = gaussian_smooth(fill_gaps(xs, max_gap), sigma)
        fy = gaussian_smooth(fill_gaps(ys, max_gap), sigma)
        for i in range(n):
            if isfin(fx[i]) and isfin(fy[i]):
                out[i][j * 3] = fx[i]
                out[i][j * 3 + 1] = fy[i]
                out[i][j * 3 + 2] = cs[i] if isfin(xs[i]) else 0.2
    return out


def hand_speed_series(frames, fps):
    tl = _or1(torso_length(frames))
    mids = [_mid_of(f, KP["LWrist"], KP["RWrist"]) for f in frames]
    hx = [m[0] if m else NAN for m in mids]
    hy = [m[1] if m else NAN for m in mids]
    sx = gaussian_smooth(fill_gaps(hx), fps / 60)
    sy = gaussian_smooth(fill_gaps(hy), fps / 60)
    dx, dy = derivative(sx), derivative(sy)
    return [math.hypot(v, dy[i]) * fps / tl if isfin(v) and isfin(dy[i]) else NAN for i, v in enumerate(dx)]


def motion_energy(frames, fps):
    tl = _or1(torso_length(frames))
    joints = [KP[k] for k in ("LWrist", "RWrist", "LAnkle", "RAnkle", "Nose", "MidHip", "LKnee", "RKnee")]
    per = []
    for j in joints:
        xs = gaussian_smooth(fill_gaps([f[j * 3] if f[j * 3 + 2] > MIN_CONF else NAN for f in frames]), fps / 30)
        ys = gaussian_smooth(fill_gaps([f[j * 3 + 1] if f[j * 3 + 2] > MIN_CONF else NAN for f in frames]), fps / 30)
        dx, dy = derivative(xs), derivative(ys)
        per.append([math.hypot(v, dy[i]) * fps / tl if isfin(v) and isfin(dy[i]) else NAN for i, v in enumerate(dx)])
    return [mean([s[i] for s in per if isfin(s[i])]) for i in range(len(frames))]


SWING_BURST_SEC = 0.586


def _burst_of(frames, swing_fps):
    s = hand_speed_series(frames, swing_fps)
    z = lambda v: v if isfin(v) else 0  # noqa: E731  (JS `v || 0`)
    p = 0
    for i in range(1, len(s)):
        if z(s[i]) > z(s[p]):
            p = i
    thr = 0.25 * z(s[p])
    if not (thr > 0):
        return None
    a = b = p
    while a - 1 >= 0 and isfin(s[a - 1]) and s[a - 1] > thr:
        a -= 1
    while b + 1 < len(s) and isfin(s[b + 1]) and s[b + 1] > thr:
        b += 1
    fa = (s[a] - thr) / (s[a] - s[a - 1]) if a > 0 else 0.0
    fb = (s[b] - thr) / (s[b] - s[b + 1]) if b < len(s) - 1 else 0.0
    return b - a + fa + fb, a, b


def _hand_travel(frames, a, b, half_window):
    tl = _or1(torso_length(frames))

    def avg(c):
        x = y = 0.0
        n = 0
        for i in range(max(0, c - half_window), min(len(frames) - 1, c + half_window) + 1):
            h = _mid_of(frames[i], KP["LWrist"], KP["RWrist"])
            if h:
                x += h[0]
                y += h[1]
                n += 1
        return (x / n, y / n) if n else None

    pa, pb = avg(a), avg(b)
    return math.hypot(pb[0] - pa[0], pb[1] - pa[1]) / tl if pa and pb else 0.0


def estimate_swing_fps(frames, fps):
    """The swing's own clock in frames per swing-second, from the length of
    the hand burst; slow motion and frame rate need no setting. Mirrors
    estimateSwingFps() in sequence.js."""
    best = None
    for k in (0.5, 1, 2, 4, 8, 16):
        est = fps * k
        prev = est
        burst = None
        for _ in range(3):
            burst = _burst_of(frames, est)
            if not burst or not (burst[0] > 0):
                break
            prev = est
            est = burst[0] / SWING_BURST_SEC
        if not burst or not (burst[0] > 0):
            continue
        ok = abs(math.log(est / prev)) < 0.25
        travel = _hand_travel(frames, burst[1], burst[2], max(1, jsround(est * 0.02)))
        if best is None or (ok and (not best[1] or travel > best[2])):
            best = (est, ok, travel)
    return clamp(best[0], 5, 5000) if best and isfin(best[0]) else fps


def suggest_stance_frame(frames, fps):
    """End of the last quiet period before the fastest hand movement."""
    n = len(frames)
    if n < 3:
        return 0
    speed = hand_speed_series(frames, fps)
    z = lambda v: v if isfin(v) else 0  # noqa: E731  (JS `v || 0`)
    peak = 0
    for i in range(1, n):
        if z(speed[i]) > z(speed[peak]):
            peak = i
    energy = motion_energy(frames, fps)
    srt = sorted(v for v in energy if isfin(v))
    p20 = srt[math.floor(len(srt) * 0.2)] if srt else 0
    thr = max(0.3, p20 * 1.8)
    min_run = max(2, jsround(0.12 * fps))
    run = 0
    i = peak - jsround(0.1 * fps)
    while i >= 0:
        if energy[i] < thr:
            run += 1
            if run >= min_run:
                end = i + run - 1
                while end + 1 < peak and energy[end + 1] < thr:
                    end += 1
                return max(i, end - jsround(0.08 * fps))
        else:
            run = 0
        i -= 1
    return clamp(peak - jsround(1.0 * fps), 0, n - 1)


def detect_pitcher_side(frames, stance_index, fps):
    """Returns (side, confidence, votes) with side 'left' or 'right'."""
    n = len(frames)
    s = clamp(stance_index, 0, n - 1)
    tl = torso_length(frames, s)
    if not (isfin(tl) and tl != 0):
        tl = _or1(torso_length(frames))
    win = max(1, jsround(0.15 * fps))
    votes = []

    head = []
    for i in range(max(0, s - win), min(n - 1, s + win) + 1):
        nose = _pt(frames[i], KP["Nose"])
        ears = _mid_of(frames[i], KP["LEar"], KP["REar"])
        if nose and ears:
            head.append(nose[0] - ears[0])
    if head:
        votes.append(("head turn", clamp(median(head) / (0.15 * tl), -1, 1), 1.0))

    hands = _mid_of(frames[s], KP["LWrist"], KP["RWrist"])
    hip = _pt(frames[s], KP["MidHip"])
    if hands and hip:
        votes.append(("hands held back", clamp((hip[0] - hands[0]) / (0.3 * tl), -1, 1), 1.0))

    best_dx = 0.0
    for j in (KP["LAnkle"], KP["RAnkle"]):
        a0 = _pt(frames[s], j)
        if not a0:
            continue
        for i in range(s, n):
            a = _pt(frames[i], j)
            if a and abs(a[0] - a0[0]) > abs(best_dx):
                best_dx = a[0] - a0[0]
    if best_dx:
        votes.append(("stride direction", clamp(best_dx / (0.6 * tl), -1, 1), 1.5))

    if hands:
        best = 0.0
        for i in range(s, n):
            h = _mid_of(frames[i], KP["LWrist"], KP["RWrist"])
            if h and abs(h[0] - hands[0]) > abs(best):
                best = h[0] - hands[0]
        if best:
            votes.append(("hand path", clamp(best / (0.8 * tl), -1, 1), 1.5))

    wsum = sum(v[2] for v in votes) or 1
    score = sum(v[1] * v[2] for v in votes) / wsum
    return ("right" if score >= 0 else "left"), min(1.0, abs(score)), votes


MAX_ROLL = 12 * math.pi / 180
MIN_ROLL = 1 * math.pi / 180


def estimate_roll(frames, stance_index, fps=30):
    """Camera roll (radians) from the ground line through both feet at the
    stance (big toes, else ankles). 0 when unsure. Mirrors estimateRoll()."""
    n = len(frames)
    s = clamp(stance_index, 0, n - 1)
    tl = torso_length(frames, s)
    if not isfin(tl):
        return 0.0

    hw = max(2, jsround(0.15 * fps))

    def line_angles(a, b):
        out = []
        for i in range(max(0, s - hw), min(n - 1, s + hw) + 1):
            p, q = _pt(frames[i], a), _pt(frames[i], b)
            if not p or not q:
                continue
            l, r = (p, q) if p[0] <= q[0] else (q, p)
            if r[0] - l[0] < 0.5 * tl:
                continue
            out.append(math.atan2(r[1] - l[1], r[0] - l[0]))
        return out

    angles = line_angles(KP["LBigToe"], KP["RBigToe"])
    if len(angles) < 3:
        angles = line_angles(KP["LAnkle"], KP["RAnkle"])
    m = median(angles)
    if not isfin(m) or abs(m) < MIN_ROLL or abs(m) > MAX_ROLL:
        return 0.0
    return m


def _rotate_frames(frames, angle, cx, cy):
    c, sn = math.cos(angle), math.sin(angle)
    out = []
    for f in frames:
        g = list(f)
        for j in range(NUM_KP):
            if g[j * 3 + 2] > 0:
                dx, dy = f[j * 3] - cx, f[j * 3 + 1] - cy
                g[j * 3] = cx + dx * c - dy * sn
                g[j * 3 + 1] = cy + dx * sn + dy * c
        out.append(g)
    return out


def canonicalize(raw_frames, pitcher_side="right", stance_index=0, fps=30, clean=True):
    """Canonical frames (pitcher toward +x, y up, front side = L joints,
    origin = stance MidHip, unit ≈ adult torso length). Mirrors canonicalize()."""
    n = len(raw_frames)
    if not n:
        raise ValueError("No frames to analyze")
    s = clamp(stance_index, 0, n - 1)
    sx = -1 if pitcher_side == "left" else 1
    frames = []
    for f in raw_frames:
        g = list(f)
        for j in range(NUM_KP):
            g[j * 3] = sx * f[j * 3]
        frames.append(g)

    def x_of(f, j):
        return f[j * 3] if f[j * 3 + 2] > MIN_CONF else NAN

    # Level a tilted camera using the ground line at the stance.
    roll = estimate_roll(frames, s, fps)
    if roll:
        win0 = frames[max(0, s - 2):min(n, s + 3)]
        cx = median([x_of(f, KP["MidHip"]) for f in win0])
        cy = median([f[KP["MidHip"] * 3 + 1] if f[KP["MidHip"] * 3 + 2] > MIN_CONF else NAN for f in win0])
        if not (isfin(cx) and isfin(cy)):
            cx = cy = 0.0
        frames = _rotate_frames(frames, -roll, cx, cy)

    win = frames[max(0, s - 2):min(n - 1, s + 2) + 1]

    def side_score(pairs):
        score = 0.0
        for r, l in pairs:
            d = median([x_of(f, l) - x_of(f, r) for f in win])
            if isfin(d):
                score += d
        return score

    arm_score = side_score([(KP["RShoulder"], KP["LShoulder"]), (KP["RElbow"], KP["LElbow"])])
    leg_score = side_score([(KP["RHip"], KP["LHip"]), (KP["RKnee"], KP["LKnee"]), (KP["RAnkle"], KP["LAnkle"])])
    swap_arms, swap_legs = arm_score < 0, leg_score < 0
    if swap_arms or swap_legs:
        fixed = []
        for g in frames:
            if swap_arms:
                g = swap_lr(g, ARMS + HEAD)
            if swap_legs:
                g = swap_lr(g, LEGS)
            fixed.append(g)
        frames = fixed

    frames = fix_left_right_flicker(frames, s)
    if clean:
        frames = clean_sequence(frames, fps=fps)

    scale = body_scale(frames, s, 2)
    if not isfin(scale):
        raise ValueError("Could not find the hitter's torso (neck and hips) in the stance frame")
    o = frames[s]
    m = KP["MidHip"]
    if o[m * 3 + 2] > 0:
        ox, oy = o[m * 3], o[m * 3 + 1]
    else:
        ox = median([x_of(f, m) for f in frames])
        oy = median([f[m * 3 + 1] if f[m * 3 + 2] > MIN_CONF else NAN for f in frames])
    canon = []
    for f in frames:
        g = empty_frame()
        for j in range(NUM_KP):
            c = f[j * 3 + 2]
            if c > 0:
                g[j * 3] = (f[j * 3] - ox) / scale
                g[j * 3 + 1] = -(f[j * 3 + 1] - oy) / scale
                g[j * 3 + 2] = c
        canon.append(g)
    return canon


# ----------------------------------------------------------------- metrics.js / phases.js


def _P(f, j):
    return (f[j * 3], f[j * 3 + 1]) if f[j * 3 + 2] > P_MIN else None


def _hands(f):
    pa, pb = _P(f, KP["LWrist"]), _P(f, KP["RWrist"])
    if pa and pb:
        return ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
    return pa or pb


def stance_half_window(fps):
    return max(0, jsround(0.06 * fps))


def compute_series(frames, stance_index, fps):
    """The subset of computeSeries() that phase detection uses."""
    hw = stance_half_window(fps)
    sx = sy = 0.0
    cnt = 0
    for i in range(max(0, stance_index - hw), min(len(frames) - 1, stance_index + hw) + 1):
        p = _P(frames[i], KP["LAnkle"])
        if p:
            sx += p[0]
            sy += p[1]
            cnt += 1
    fa0 = (sx / cnt, sy / cnt) if cnt else (0.4, -1.6)
    S = {"handsX": [], "handsY": [], "handReach": [], "stride": [], "frontFootLift": []}
    for f in frames:
        h = _hands(f)
        neck = _P(f, KP["Neck"])
        fa = _P(f, KP["LAnkle"])
        S["handsX"].append(h[0] if h else NAN)
        S["handsY"].append(h[1] if h else NAN)
        S["handReach"].append(math.hypot(h[0] - neck[0], h[1] - neck[1]) if h and neck else NAN)
        S["stride"].append(fa[0] - fa0[0] if fa else NAN)
        S["frontFootLift"].append(fa[1] - fa0[1] if fa else NAN)
    hx = gaussian_smooth(fill_gaps(S["handsX"]), fps / 60)
    hy = gaussian_smooth(fill_gaps(S["handsY"]), fps / 60)
    vx = [v * fps if isfin(v) else NAN for v in derivative(hx)]
    vy = [v * fps if isfin(v) else NAN for v in derivative(hy)]
    S["handVx"] = vx
    S["handSpeed"] = [math.hypot(v, vy[i]) if isfin(v) and isfin(vy[i]) else NAN for i, v in enumerate(vx)]
    return S


def sanitize_phases(phases, n):
    out = {}
    last = 0
    for key in PHASE_KEYS:
        v = phases.get(key)
        v = jsround(v) if isfin(v) else last
        v = clamp(v, last, n - 1)
        out[key] = int(v)
        last = v
    return out


def find_load(S, lo, hi):
    lift = arg_max(S["frontFootLift"], lo, hi + 1)
    if lift >= 0 and S["frontFootLift"][lift] > 0.08:
        return lift
    return arg_min(S["handsX"], lo, hi + 1)


def find_foot_plant(S, lo, hi, stance, contact):
    max_stride = -math.inf
    for i in range(stance, contact + 1):
        if i < len(S["stride"]) and isfin(S["stride"][i]):
            max_stride = max(max_stride, S["stride"][i])
    if not (max_stride > 0.12):
        return -1
    for i in range(max(stance, lo), hi + 1):
        if i >= len(S["stride"]):
            break
        st, lift = S["stride"][i], S["frontFootLift"][i]
        if isfin(st) and st >= 0.9 * max_stride and not (isfin(lift) and lift > 0.12):
            return i
    return -1


def find_contact(S, lo, hi):
    """Centroid of the peak hand speed toward the pitcher (see findContact())."""
    v = S["handVx"]
    peak = arg_max(v, lo, hi + 1)
    if peak < 0:
        return peak
    thr = 0.8 * v[peak]
    a = b = peak
    while a - 1 >= max(0, lo) and isfin(v[a - 1]) and v[a - 1] >= thr:
        a -= 1
    while b + 1 <= min(len(v) - 1, hi) and isfin(v[b + 1]) and v[b + 1] >= thr:
        b += 1
    sw = st = 0.0
    for i in range(a, b + 1):
        w = v[i] - thr
        sw += w
        st += w * i
    return jsround(st / sw) if sw > 0 else peak


def detect_phases(frames, fps, stance_index=0):
    """Phase frames {stance, load, footPlant, contact, extension, finish}."""
    n = len(frames)
    s = clamp(stance_index, 0, n - 1)
    S = compute_series(frames, s, fps)

    def sec(t):
        return max(1, jsround(t * fps))

    peak = find_contact(S, s + 1, n - 1)
    contact = clamp(n - 1 if peak < 0 else peak, s + 3, n - 1)

    foot_plant = find_foot_plant(S, s, contact, s, contact)
    if foot_plant < 0:
        foot_plant = contact - sec(0.2)
    foot_plant = clamp(foot_plant, s + 2, contact - 1)

    load = clamp(find_load(S, s, foot_plant), s + 1, foot_plant - 1)

    extension = arg_max(S["handReach"], contact, contact + sec(0.2))
    if extension < 0:
        extension = contact + sec(0.08)
    extension = clamp(extension, contact + 1, n - 1)

    ps = S["handSpeed"][contact] if 0 <= contact < n else NAN
    peak_speed = ps if isfin(ps) and ps != 0 else 1
    finish = -1
    for i in range(extension + sec(0.12), min(n, contact + sec(0.6))):
        if isfin(S["handSpeed"][i]) and S["handSpeed"][i] < 0.2 * peak_speed:
            finish = i
            break
    if finish < 0:
        finish = min(n - 1, contact + sec(0.45))
    finish = clamp(finish, extension, n - 1)

    return sanitize_phases(
        {"stance": s, "load": load, "footPlant": foot_plant, "contact": contact, "extension": extension, "finish": finish},
        n,
    )


def auto_detect(frames, fps, stance_index=None, pitcher_side=None):
    """Swing clock, stance, pitcher side and beats for raw pixel frames, with
    the same defaults as the browser builder. `fps` is the video frame rate of
    the frames; slow motion is handled by the swing clock."""
    swing_fps = estimate_swing_fps(frames, fps)
    stance = suggest_stance_frame(frames, swing_fps) if stance_index is None else stance_index
    side, confidence, votes = detect_pitcher_side(frames, stance, swing_fps)
    if pitcher_side:
        side = pitcher_side
    canon = canonicalize(frames, pitcher_side=side, stance_index=stance, fps=swing_fps)
    phases = detect_phases(canon, swing_fps, stance)
    return {"stance": stance, "pitcherSide": side, "sideConfidence": confidence, "votes": votes, "phases": phases,
            "canon": canon, "swingFps": swing_fps}
