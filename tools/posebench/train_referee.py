#!/usr/bin/env python3
"""Train the referees (models/referee-best.json, referee-fast.json) and score them.

  python tools/posebench/run.py candidates --set general
  python tools/posebench/run.py candidates --set batters
  python tools/posebench/train_referee.py [--write]

The referee is trained on general COCO people only (never on batters): for
every labelled joint, each model's answer is an example, positive when it is
within 5% of body height of the label. Features come from
tools/swingdb/referee.py itself, so training and the apps compute them the
same way. It is then scored on the batters. Needs scikit-learn.
"""

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))

from metrics import DATA, labels, load_set, report, score  # noqa: E402
from swingdb.referee import consensus_height, feature_vector, fuse, raw_features  # noqa: E402

CONFIGS = {"best": ["movenet", "movenet-mirrored", "mediapipe", "mediapipe-mirrored", "efficientpose"], "fast": ["movenet", "mediapipe"]}
LABEL = {"movenet": "MoveNet Thunder", "movenet-mirrored": "MoveNet Thunder on the mirrored crop", "mediapipe": "MediaPipe Pose (heavy)",
         "mediapipe-mirrored": "MediaPipe Pose (heavy) on the mirrored crop", "efficientpose": "EfficientPose II"}
SOFT_RADIUS = 0.1  # average the answers within 10% of body height of the winner


def candidate_sets(set_name, names):
    preds = json.loads((DATA / "preds" / set_name / "candidates.json").read_text())

    def clean(c):
        return None if c is None else [[math.nan if v is None else v for v in row] for row in c]

    return {k: [clean(v[n]) if v else None for n in names] for k, v in preds.items()}


def examples(people, cands):
    X, y = [], []
    for p in people:
        C = cands.get(str(p["id"]))
        if not C or all(c is None for c in C):
            continue
        K = len(C)
        full = [c if c is not None else [[math.nan] * 3] * 17 for c in C]
        h = consensus_height(full)
        gt = labels(p)
        for j in range(17):
            if gt[j][2] == 0:
                continue
            for c in range(K):
                if not math.isfinite(full[c][j][0]):
                    continue
                X.append(feature_vector(raw_features(full, c, j, h), c, j, K))
                y.append(math.hypot(full[c][j][0] - gt[j][0], full[c][j][1] - gt[j][1]) / p["bbox"][3] < 0.05)
    return X, y


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write models/referee-*.json")
    args = ap.parse_args(argv)
    from sklearn.linear_model import LogisticRegression

    general, batters = load_set("general"), load_set("batters")
    val = [p for p in batters if p["split"] == "val2017"]
    for quality, names in CONFIGS.items():
        X, y = examples(general, candidate_sets("general", names))
        clf = LogisticRegression(max_iter=5000, C=1.0).fit(X, y)
        referee = {
            "version": 2,
            "description": f"Logistic referee for fusing pose candidates joint by joint (src/core/referee.js). {quality}: "
                           + ", ".join(LABEL[n] for n in names),
            "candidates": names,
            "soft_radius": SOFT_RADIUS,
            "trained_on": "COCO val2017 people (no batters), candidates on a 10%-margin person crop",
            "bias": round(float(clf.intercept_[0]), 6),
            "weights": [round(float(w), 6) for w in clf.coef_[0]],
        }
        cands = candidate_sets("batters", names)
        fused = {}
        for k, C in cands.items():
            res = fuse(C, referee) if C else None
            fused[k] = res and res[0]
        print(report(f"referee-{quality} [batters]", score(fused, batters)))
        print(report(f"referee-{quality} [val2017 batters]", score(fused, val)))
        if args.write:
            out = ROOT / "models" / f"referee-{quality}.json"
            out.write_text(json.dumps(referee, indent=1) + "\n")
            print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
