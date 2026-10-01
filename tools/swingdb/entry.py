"""Write swing-db/v1 entries and keep data/pros/index.json in sync
(mirrors makeEntry()/validateEntry() in src/core/db.js). Standard library only."""

import json
import re
import unicodedata
from pathlib import Path

from .analysis import PHASE_KEYS, jsround

SCHEMA = "swing-db/v1"


def slugify(s):
    s = unicodedata.normalize("NFKD", str(s or "")).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:60]


def _num(v, digits):
    """Round like the JS app and print integers without a trailing .0."""
    p = 10 ** digits
    r = jsround(v * p) / p
    return int(r) if float(r).is_integer() else r


def make_entry(*, id, name, fps, pitcher_side, stance_frame, phases, frames, team="", bats="", notes="",
               source="", width=None, height=None, clip=None):
    entry = {
        "schema": SCHEMA,
        "id": id,
        "name": name,
        "team": team,
        "bats": bats,
        "notes": notes,
        "source": source,
        "keypointFormat": "BODY_25",
        "fps": _num(fps, 3),
    }
    if width and height:
        entry["image"] = {"width": int(width), "height": int(height)}
    entry["orientation"] = {"pitcherSide": pitcher_side}
    entry["stanceFrame"] = int(stance_frame)
    entry["phases"] = {k: int(phases[k]) for k in PHASE_KEYS}
    if clip:
        entry["clip"] = clip
    entry["frames"] = [
        {"pose_keypoints_2d": [_num(v, 3 if i % 3 == 2 else 2) for i, v in enumerate(f)]} for f in frames
    ]
    return entry


def validate_entry(entry):
    where = f'Entry "{entry.get("id")}"'
    for key in ("id", "name"):
        if not entry.get(key):
            raise ValueError(f'{where}: missing "{key}"')
    if not (entry.get("fps") or 0) > 0:
        raise ValueError(f'{where}: "fps" must be a positive number')
    if entry.get("orientation", {}).get("pitcherSide") not in ("left", "right"):
        raise ValueError(f'{where}: orientation.pitcherSide must be "left" or "right"')
    n = len(entry.get("frames") or [])
    if n < 10:
        raise ValueError(f"{where}: needs at least 10 frames")
    for f in entry["frames"]:
        if len(f["pose_keypoints_2d"]) != 75:
            raise ValueError(f"{where}: every frame needs 75 BODY_25 values")
    if not 0 <= entry.get("stanceFrame", -1) < n:
        raise ValueError(f"{where}: stanceFrame out of range")
    last = -1
    for k in PHASE_KEYS:
        v = entry["phases"][k]
        if not 0 <= v < n:
            raise ValueError(f'{where}: phase "{k}" out of range')
        if v < last:
            raise ValueError(f"{where}: phases must be in order ({' -> '.join(PHASE_KEYS)})")
        last = v
    return entry


def write_entry(entry, db_dir):
    """Write data/pros/<id>.json and add/replace it in index.json."""
    validate_entry(entry)
    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    path = db_dir / f"{entry['id']}.json"
    path.write_text(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    index_path = db_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"schema": "swing-db-index/v1", "entries": []}
    meta = {"id": entry["id"], "file": path.name, "name": entry["name"], "bats": entry.get("bats", "")}
    entries = [e for e in index.get("entries", []) if e.get("id") != entry["id"]]
    entries.append(meta)
    index["entries"] = entries
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
