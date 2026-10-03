#!/usr/bin/env python3
"""Run a pose method over a benchmark set and score it.

  python tools/posebench/run.py engine-best            # the pose engine, as the app runs it
  python tools/posebench/run.py engine-fast
  python tools/posebench/run.py mediapipe-full-frame   # the old method: MediaPipe heavy on the whole photo
  python tools/posebench/run.py candidates --set general   # each model alone, for train_referee.py

Options: --set batters|general (default batters), --procs N, --limit N.
The engine finds people with its detector and is pointed at the labelled
person (the detection overlapping the label most), as the tracker would be in
a video. Predictions are cached in data/preds/<set>/<method>.json.
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))

from metrics import DATA, load_set, report, score  # noqa: E402

CANDIDATES = ["movenet", "movenet-mirrored", "mediapipe", "mediapipe-mirrored", "efficientpose"]
MODELS = ROOT / ".cache" / "models"

_engine = None
_method = None


def _iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-6, a[2] * a[3] + b[2] * b[3] - inter)


def _init(method):
    global _engine, _method
    from swingdb.engine import PoseEngine
    from swingdb.pose import ensure_model

    _method = method
    heavy = ensure_model("heavy", MODELS)
    quality = "fast" if method == "engine-fast" else "best"
    _engine = PoseEngine(heavy, quality=quality, log=lambda *a: None)
    if method == "mediapipe-full-frame":
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options

        _engine.full = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=str(heavy)), running_mode=vision.RunningMode.IMAGE,
            num_poses=4, min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3))


def _predict(p):
    """17x[x, y, conf] (or None) for one labelled person; for "candidates", one per model."""
    import cv2
    import numpy as np
    from swingdb.body25 import COCO_TO_BODY25
    from swingdb.engine import MP_TO_COCO, square_from_box

    img = cv2.imread(str(DATA / "images" / p["file"]))
    e = _engine
    e.height, e.width = img.shape[:2]
    if _method == "candidates":
        crop = square_from_box(*p["bbox"])  # the labelled box, as the referee was trained
        out = {}
        for name in CANDIDATES:
            model, _, mirrored = name.partition("-")
            if model == "movenet":
                out[name] = e._movenet(img, crop, bool(mirrored)).tolist()
            elif model == "efficientpose":
                out[name] = e._effpose(img, crop, bool(mirrored)).tolist()
            else:
                q = e._mediapipe(img, crop, bool(mirrored))
                out[name] = None if q is None else q[MP_TO_COCO].tolist()
        return out
    if _method == "mediapipe-full-frame":
        import mediapipe as mp

        r = e.full.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
        people = [[[l.x * e.width, l.y * e.height, l.visibility] for l in (lms[i] for i in MP_TO_COCO)] for lms in r.pose_landmarks or []]
        if not people:
            return None
        from metrics import oks

        return max(people, key=lambda q: oks(q, p))  # the labelled person, if found: the old method's best case
    boxes = e.people(img)
    if not boxes:
        return None
    e.crop, e.last, e.since_check = square_from_box(*max(boxes, key=lambda b: _iou(b, p["bbox"]))[:4]), None, 1
    f = e.process(img)
    return [[f[j * 3], f[j * 3 + 1], f[j * 3 + 2]] if f[j * 3 + 2] > 0 else [float("nan")] * 3 for j in COCO_TO_BODY25]


def _run_one(p):
    try:
        return str(p["id"]), _predict(p)
    except Exception as ex:  # noqa: BLE001
        print(f"error on {p['id']}: {ex}", file=sys.stderr)
        return str(p["id"]), None


def run(method, set_name="batters", procs=4, limit=None, force=False):
    people = load_set(set_name)[:limit] if limit else load_set(set_name)
    path = DATA / "preds" / set_name / f"{method}.json"
    if path.exists() and not force and not limit:
        preds = json.loads(path.read_text())
    else:
        t0 = time.time()
        with Pool(procs, initializer=_init, initargs=(method,)) as pool:
            preds = dict(pool.map(_run_one, people, chunksize=8))
        dt = time.time() - t0
        print(f"[{method}] {len(people)} people in {dt:.0f} s ({1000 * dt * procs / len(people):.0f} ms each per process)")
        if not limit:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(preds))
    return preds, people


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=["engine-best", "engine-fast", "mediapipe-full-frame", "candidates"])
    ap.add_argument("--set", default="batters", choices=["batters", "general"])
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true", help="re-run even if predictions are cached")
    args = ap.parse_args(argv)
    preds, people = run(args.method, args.set, args.procs, args.limit, args.force)
    if args.method == "candidates":
        for name in CANDIDATES:
            print(report(f"{name} (labelled box)", score({k: v and v[name] for k, v in preds.items()}, people)))
        return
    print(report(f"{args.method} [{args.set}]", score(preds, people)))
    held_out = [p for p in people if p["split"] == "val2017"]
    if args.set == "batters" and held_out:
        print(report(f"{args.method} [val2017 batters]", score(preds, held_out)))


if __name__ == "__main__":
    main()
