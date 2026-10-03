"""Scoring against COCO labels: distances in body heights (the labelled box height).

PCK5 / PCK10: share of labelled joints within 5% / 10% of body height of the label.
The grip is the midpoint of the two wrists, which the swing analysis uses for the hands.
OKS is COCO's keypoint similarity (1 = perfect).
"""

import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
SIGMAS = [s / 10 for s in (.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89)]
GROUPS = {"wrists": [9, 10], "elbows": [7, 8], "shoulders": [5, 6], "hips": [11, 12], "knees": [13, 14],
          "ankles": [15, 16], "head": [0, 1, 2, 3, 4]}


def load_set(name):
    return json.loads((DATA / f"{name}.json").read_text())


def labels(p):
    k = p["keypoints"]
    return [(k[j * 3], k[j * 3 + 1], k[j * 3 + 2]) for j in range(17)]


def oks(pred, p):
    gt = labels(p)
    vals = [math.exp(-((pred[j][0] - gt[j][0]) ** 2 + (pred[j][1] - gt[j][1]) ** 2) / (2 * max(p["area"], 1) * (2 * SIGMAS[j]) ** 2))
            for j in range(17) if gt[j][2] > 0 and math.isfinite(pred[j][0])]
    n = sum(1 for j in range(17) if gt[j][2] > 0)
    return sum(vals) / n if n else 0.0


def score(preds, people):
    """preds: {str(person id): 17x[x, y, conf] or None}. Missing joints count as misses."""
    errs = {g: [] for g in GROUPS}
    grip, oks_all = [], []
    for p in people:
        pred = preds.get(str(p["id"]))
        gt = labels(p)
        h = p["bbox"][3]
        if pred is None:
            for g, idx in GROUPS.items():
                errs[g] += [math.inf for j in idx if gt[j][2] > 0]
            grip.append(math.inf)
            oks_all.append(0.0)
            continue
        d = [math.hypot(pred[j][0] - gt[j][0], pred[j][1] - gt[j][1]) / h if math.isfinite(pred[j][0]) else math.inf
             for j in range(17)]
        for g, idx in GROUPS.items():
            errs[g] += [d[j] for j in idx if gt[j][2] > 0]
        if gt[9][2] > 0 and gt[10][2] > 0:
            grip.append(math.hypot((pred[9][0] + pred[10][0] - gt[9][0] - gt[10][0]) / 2,
                                   (pred[9][1] + pred[10][1] - gt[9][1] - gt[10][1]) / 2) / h)
        oks_all.append(oks(pred, p))

    def summary(v):
        v = [x if math.isfinite(x) else math.inf for x in v]
        n = max(1, len(v))
        return {"n": len(v), "pck5": sum(x < 0.05 for x in v) / n, "pck10": sum(x < 0.10 for x in v) / n,
                "mean": sum(min(x, 0.5) for x in v) / n}

    out = {g: summary(v) for g, v in errs.items()}
    out["grip"] = summary(grip)
    out["oks"] = sum(oks_all) / max(1, len(oks_all))
    out["missed"] = sum(1 for p in people if preds.get(str(p["id"])) is None)
    return out


def report(name, m):
    w, g, e = m["wrists"], m["grip"], m["elbows"]
    return (f"{name:28s} wrists PCK5 {100 * w['pck5']:4.1f}% PCK10 {100 * w['pck10']:4.1f}% | grip PCK5 {100 * g['pck5']:4.1f}% "
            f"PCK10 {100 * g['pck10']:4.1f}% | elbows PCK10 {100 * e['pck10']:4.1f}% | OKS {m['oks']:.3f} | missed {m['missed']}")
