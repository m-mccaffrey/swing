#!/usr/bin/env python3
"""Build the pose benchmark sets from COCO 2017 (human-labelled keypoints).

  batters.json  people holding a baseball bat (a bat box within a quarter body
                height of a wrist), both wrists labelled, at least 120 px tall:
                train2017 and val2017 (the val2017 ones were seen by no model here)
  general.json  everyone else in val2017 with 10+ labelled keypoints and a
                labelled wrist (the referee is trained on these, never on batters)

Downloads the annotations (about 250 MB) and the images used (about 1 GB) into
tools/posebench/data/.  Standard library only.
"""

import json
import math
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
COCO = "https://s3.amazonaws.com/images.cocodataset.org"
ANNOTATIONS = f"{COCO}/annotations/annotations_trainval2017.zip"
NEEDED = [f"annotations/{kind}_{split}.json" for kind in ("instances", "person_keypoints") for split in ("train2017", "val2017")]


def fetch(url, path):
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
            return
        except OSError:
            if attempt == 2:
                raise


def annotations():
    if all((DATA / n).exists() for n in NEEDED):
        return
    zpath = DATA / "annotations_trainval2017.zip"
    print("Downloading COCO 2017 annotations (about 250 MB)...")
    fetch(ANNOTATIONS, zpath)
    with zipfile.ZipFile(zpath) as z:
        for n in NEEDED:
            z.extract(n, DATA)
    zpath.unlink()


def person(a, im, split, **extra):
    return {"id": a["id"], "image_id": a["image_id"], "split": split, "file": im["file_name"], "width": im["width"],
            "height": im["height"], "bbox": a["bbox"], "area": a["area"], "keypoints": a["keypoints"],
            "num_keypoints": a["num_keypoints"], **extra}


def box_distance(px, py, b):
    bx, by, bw, bh = b
    return math.hypot(max(bx - px, 0, px - (bx + bw)), max(by - py, 0, py - (by + bh)))


def batters():
    out = []
    for split in ("train2017", "val2017"):
        inst = json.loads((DATA / f"annotations/instances_{split}.json").read_text())
        bat = next(c["id"] for c in inst["categories"] if c["name"] == "baseball bat")
        bats = {}
        for a in inst["annotations"]:
            if a["category_id"] == bat:
                bats.setdefault(a["image_id"], []).append(a["bbox"])
        del inst
        kp = json.loads((DATA / f"annotations/person_keypoints_{split}.json").read_text())
        images = {im["id"]: im for im in kp["images"]}
        for a in kp["annotations"]:
            if a["iscrowd"] or a["image_id"] not in bats or a["num_keypoints"] < 10 or a["bbox"][3] < 120:
                continue
            k = a["keypoints"]
            lw, rw = k[27:30], k[30:33]
            if lw[2] == 0 or rw[2] == 0:
                continue
            h = a["bbox"][3]
            near = [b for b in bats[a["image_id"]] if min(box_distance(*lw[:2], b), box_distance(*rw[:2], b)) < 0.25 * h]
            if near:
                out.append(person(a, images[a["image_id"]], split, bat=near[0]))
    return out


def general(batter_ids):
    kp = json.loads((DATA / "annotations/person_keypoints_val2017.json").read_text())
    images = {im["id"]: im for im in kp["images"]}
    out = []
    for a in kp["annotations"]:
        if a["iscrowd"] or a["num_keypoints"] < 10 or a["bbox"][3] < 120 or a["id"] in batter_ids:
            continue
        if a["keypoints"][29] == 0 and a["keypoints"][32] == 0:
            continue
        out.append(person(a, images[a["image_id"]], "val2017"))
    return out


def images(sets):
    (DATA / "images").mkdir(parents=True, exist_ok=True)
    files = sorted({(p["split"], p["file"]) for s in sets for p in s})
    todo = [(sp, f) for sp, f in files if not (DATA / "images" / f).exists()]
    print(f"Downloading {len(todo)} of {len(files)} images...")
    with ThreadPoolExecutor(16) as pool:
        for n, _ in enumerate(pool.map(lambda t: fetch(f"{COCO}/{t[0]}/{t[1]}", DATA / "images" / t[1]), todo)):
            if n % 500 == 499:
                print(f"  {n + 1}")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    annotations()
    bat = batters()
    gen = general({b["id"] for b in bat})
    (DATA / "batters.json").write_text(json.dumps(bat))
    (DATA / "general.json").write_text(json.dumps(gen))
    print(f"batters: {len(bat)} ({sum(b['split'] == 'val2017' for b in bat)} from val2017); general: {len(gen)}")
    images([bat, gen])
    print("Done. Next: python tools/posebench/run.py engine-best")


if __name__ == "__main__":
    sys.exit(main())
