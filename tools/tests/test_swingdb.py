"""Tests for the Python database tool (standard library only).

    python -m unittest discover -s tools/tests -v
"""

import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from swingdb import analysis  # noqa: E402
from swingdb.body25 import KP, empty_frame, mediapipe_to_body25, track_hitter  # noqa: E402
from swingdb.entry import make_entry, slugify, validate_entry, write_entry  # noqa: E402
from swingdb.video import parse_time  # noqa: E402


def js_results(*args):
    node = shutil.which("node")
    if not node:
        return None
    out = subprocess.run([node, str(ROOT / "scripts" / "parity-dump.mjs"), *args], capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(out.stdout)


def _nan(v):
    """JSON null (JavaScript NaN) -> nan, recursively."""
    if isinstance(v, list):
        return [_nan(x) for x in v]
    return math.nan if v is None else v


class ParityWithJavaScript(unittest.TestCase):
    """The Python port must make the same calls as the web app."""

    @classmethod
    def setUpClass(cls):
        cls.cases = js_results()
        if cls.cases is None:
            raise unittest.SkipTest("node not installed")

    def test_detection_matches(self):
        for c in self.cases:
            with self.subTest(case=c["name"]):
                frames = c["frames"]
                swing_fps = analysis.estimate_swing_fps(frames, c["fps"])
                self.assertAlmostEqual(swing_fps, c["swingFps"], places=6)
                stance = analysis.suggest_stance_frame(frames, swing_fps)
                self.assertEqual(stance, c["stance"])
                side, conf, _ = analysis.detect_pitcher_side(frames, stance, swing_fps)
                self.assertEqual(side, c["side"])
                self.assertAlmostEqual(conf, c["confidence"], places=9)
                canon = analysis.canonicalize(frames, pitcher_side=side, stance_index=stance, fps=swing_fps)
                worst = max(abs(a - b) for fa, fb in zip(canon, c["canon"]) for a, b in zip(fa, fb))
                self.assertLess(worst, 1e-9)
                self.assertEqual(analysis.detect_phases(canon, swing_fps, stance), c["phases"])
                det = analysis.auto_detect(frames, c["fps"])
                self.assertEqual(det["phases"], c["phases"])


class RefereeParity(unittest.TestCase):
    """tools/swingdb/referee.py must fuse candidates exactly like src/core/referee.js."""

    @classmethod
    def setUpClass(cls):
        cls.cases = js_results("referee")
        if cls.cases is None:
            raise unittest.SkipTest("node not installed")

    def test_fuse_and_body25_match(self):
        from swingdb.body25 import fused_to_body25
        from swingdb.referee import fuse

        referees = {q: json.loads((ROOT / "models" / f"referee-{q}.json").read_text()) for q in ("best", "fast")}
        self.assertGreater(len(self.cases), 20)
        for n, c in enumerate(self.cases):
            with self.subTest(case=n, quality=c["quality"]):
                res = fuse([None if cand is None else _nan(cand) for cand in c["cands"]], referees[c["quality"]])
                if c["result"] is None:
                    self.assertIsNone(res)
                    continue
                kp, chosen = res
                self.assertEqual(chosen, c["result"]["chosen"])
                for a, b in zip(kp, _nan(c["result"]["keypoints"])):
                    for u, v in zip(a, b):
                        self.assertTrue((math.isnan(u) and math.isnan(v)) or abs(u - v) < 1e-9, (a, b))
                body = fused_to_body25(kp, c["mp"])
                self.assertLess(max(abs(u - v) for u, v in zip(body, c["body25"])), 1e-9)


class Helpers(unittest.TestCase):
    def test_parse_time(self):
        self.assertEqual(parse_time("83.5"), 83.5)
        self.assertEqual(parse_time("1:23.5"), 83.5)
        self.assertEqual(parse_time("0:01:23.5"), 83.5)
        self.assertIsNone(parse_time(None))

    def test_frame_step(self):
        from swingdb.video import frame_step

        self.assertEqual(frame_step(180, 240), 1)  # 3 s at 60 fps: every frame
        self.assertEqual(frame_step(420, 240), 2)  # 7 s at 60 fps (e.g. a slow-mo replay)
        self.assertEqual(frame_step(1680, 240), 7)  # 7 s at 240 fps
        self.assertEqual(frame_step(1000, 0), 1)  # no cap

    def test_jsround(self):
        self.assertEqual(analysis.jsround(2.5), 3)
        self.assertEqual(analysis.jsround(-2.5), -2)

    def test_mediapipe_conversion(self):
        class L:
            def __init__(self, i):
                self.x, self.y, self.visibility = i / 100, 1 - i / 100, 0.9

        f = mediapipe_to_body25([L(i) for i in range(33)], 1000, 500)
        self.assertEqual(len(f), 75)
        self.assertAlmostEqual(f[KP["Neck"] * 3], 115)
        self.assertAlmostEqual(f[KP["RWrist"] * 3], 160)
        self.assertAlmostEqual(f[KP["LAnkle"] * 3], 270)

    def test_tracking_locks_on_and_ignores_jumps(self):
        def person(cx, size):
            f = empty_frame()
            for j in range(25):
                f[j * 3:j * 3 + 3] = [cx + (j % 3) * size * 0.1, 100 + j * size * 0.1, 0.9]
            f[KP["MidHip"] * 3:KP["MidHip"] * 3 + 3] = [cx, 100 + size, 0.9]
            return f

        frames = [[person(100, 60), person(500, 200)], [person(505, 200), person(110, 60)], [person(110, 60)]]
        picked = track_hitter(frames, 1000)
        self.assertEqual(picked[0][KP["MidHip"] * 3], 500)
        self.assertEqual(picked[1][KP["MidHip"] * 3], 505)
        self.assertEqual(picked[2][KP["MidHip"] * 3 + 2], 0)  # hitter lost; don't jump to the catcher
        picked = track_hitter(frames, 1000, target_x=0.1)
        self.assertEqual(picked[0][KP["MidHip"] * 3], 100)

    def test_entry_round_trip(self):
        cases = js_results()
        if cases is None:
            self.skipTest("node not installed")
        c = cases[0]
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "index.json").write_text(json.dumps({"schema": "swing-db-index/v1", "entries": [{"id": "keep", "file": "keep.json", "name": "Keep"}]}))
            entry = make_entry(id=slugify("Test Hitter"), name="Test Hitter", bats="R", fps=60, pitcher_side=c["side"],
                               stance_frame=c["phases"]["stance"], phases=c["phases"], frames=c["frames"], width=1280, height=720,
                               clip={"url": "https://example.com/v", "start": 1.0, "end": 5.0})
            path = write_entry(entry, tmp)
            data = json.loads(path.read_text())
            self.assertEqual(data["id"], "test-hitter")
            self.assertEqual(len(data["frames"][0]["pose_keypoints_2d"]), 75)
            index = json.loads((tmp / "index.json").read_text())
            self.assertEqual([e["id"] for e in index["entries"]], ["keep", "test-hitter"])
            write_entry(entry, tmp)  # re-adding replaces, doesn't duplicate
            index = json.loads((tmp / "index.json").read_text())
            self.assertEqual(len(index["entries"]), 2)
            bad = dict(entry, orientation={"pitcherSide": "up"})
            with self.assertRaises(ValueError):
                validate_entry(bad)
            # The JS validator accepts what Python wrote.
            node = shutil.which("node")
            check = "import('./src/core/db.js').then(m => { m.prepareEntry(JSON.parse(require('fs').readFileSync(process.argv[1], 'utf8'))); console.log('ok'); })"
            out = subprocess.run([node, "-e", check, str(path)], capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(out.stdout.strip(), "ok", out.stderr)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
