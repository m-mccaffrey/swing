"""Pose engine: find the hitter, follow them, and fuse several pose models.

Per frame:
  1. A square crop around the hitter. EfficientDet-Lite0 (MediaPipe object
     detector) finds people to start with, every RECHECK frames and whenever
     tracking is lost; otherwise the crop follows the previous frame's pose.
  2. On the crop, the models the referee was trained with: MoveNet Thunder
     and MediaPipe Pose (heavy), each also on the mirrored crop, plus
     EfficientPose ("best"), or MoveNet and MediaPipe once each ("fast").
  3. The referee (models/referee-*.json) picks each joint's best answer and
     averages it with the answers that agree.
  4. BODY_25: the fused joints plus MediaPipe's heels and toes, moved with the
     fused ankles.

Measured on human-labelled COCO baseball batters it never saw (tools/posebench):
wrists within 5% of body height 88% (best) / 87% (fast) of the time, against
60% for MediaPipe alone on the whole frame. The browser runs the same models
and referee (src/pose/engine.js).
"""

import json
import math
from pathlib import Path

import numpy as np

from .body25 import empty_frame, fused_to_body25, mediapipe_to_body25
from .referee import fuse

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DETECTOR_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
MOVENET_SIZE = 256
EP_SIZE = 368
MP_CROP = 384  # MediaPipe gets the crop at this size
CROP_MARGIN = 0.10  # around a detector box
POSE_MARGIN = 0.20  # around keypoints (they stop short of the head top and soles)
RECHECK = 30  # frames between detector re-checks
FLIP_COCO = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
MP_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
            (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# EfficientPose (MPII order: head, neck, r_shoulder, r_elbow, r_wrist, chest, l_shoulder, l_elbow, l_wrist,
# center, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle) -> COCO; it has no eyes or ears.
EP_TO_COCO = {0: 0, 5: 6, 6: 2, 7: 7, 8: 3, 9: 8, 10: 4, 11: 13, 12: 10, 13: 14, 14: 11, 15: 15, 16: 12}
EP_FLIP = [0, 1, 6, 7, 8, 5, 2, 3, 4, 9, 13, 14, 15, 10, 11, 12]


class _Landmark:
    """Minimal landmark object for mediapipe_to_body25()."""

    __slots__ = ("x", "y", "visibility")

    def __init__(self, x, y, v):
        self.x, self.y, self.visibility = x, y, v


def ensure_file(path, url, log=print):
    path = Path(path)
    if not path.exists() or path.stat().st_size < 100_000:
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        log(f"Downloading {path.name}...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def square_from_box(x, y, w, h, margin=CROP_MARGIN):
    half = max(w, h) / 2 * (1 + 2 * margin)
    return x + w / 2 - half, y + h / 2 - half, 2 * half


def square_from_points(pts, margin=POSE_MARGIN):
    xs, ys = pts[:, 0], pts[:, 1]
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * (1 + 2 * margin)
    return (xs.min() + xs.max()) / 2 - half, (ys.min() + ys.max()) / 2 - half, 2 * half


def heatmap_peaks(hm):
    """Peak (x, y, value) of each channel of an H x W x C heatmap, refined to
    sub-pixel by a parabola through the neighbours (same as engine.js)."""
    H, W, C = hm.shape
    out = np.zeros((C, 3))
    for c in range(C):
        m = hm[:, :, c]
        y, x = divmod(int(np.argmax(m)), W)
        v = float(m[y, x])
        dx = dy = 0.0
        if 0 < x < W - 1:
            den = m[y, x - 1] - 2 * v + m[y, x + 1]
            dx = 0.5 * (m[y, x - 1] - m[y, x + 1]) / den if den < 0 else 0.0
        if 0 < y < H - 1:
            den = m[y - 1, x] - 2 * v + m[y + 1, x]
            dy = 0.5 * (m[y - 1, x] - m[y + 1, x]) / den if den < 0 else 0.0
        out[c] = (x + dx, y + dy, v)
    return out


def warp(img, crop, size):
    import cv2

    x0, y0, side = crop
    s = size / side
    M = np.float32([[s, 0, -s * x0], [0, s, -s * y0]])
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


class PoseEngine:
    def __init__(self, pose_model_path, detector_path=None, quality="best", target_x=None, log=print):
        import mediapipe as mp
        import onnxruntime as ort
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options

        self._mp = mp
        self.quality = quality
        self.target_x = target_x
        self.referee = json.loads((MODELS / f"referee-{quality}.json").read_text())
        detector_path = detector_path or ensure_file(ROOT / ".cache" / "models" / "efficientdet_lite0.tflite", DETECTOR_URL, log)
        self.detector = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
            base_options=base_options.BaseOptions(model_asset_path=str(detector_path)),
            running_mode=vision.RunningMode.IMAGE, max_results=8, score_threshold=0.25, category_allowlist=["person"]))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=str(pose_model_path)),
            running_mode=vision.RunningMode.IMAGE, num_poses=2,
            min_pose_detection_confidence=0.1, min_pose_presence_confidence=0.1))
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.movenet = ort.InferenceSession(str(MODELS / "movenet-thunder.onnx"), so, providers=["CPUExecutionProvider"])
        self._mn_input = self.movenet.get_inputs()[0].name
        self.effpose = None
        if any(c.startswith("efficientpose") for c in self.referee["candidates"]):
            self.effpose = ort.InferenceSession(str(MODELS / "efficientpose.onnx"), so, providers=["CPUExecutionProvider"])
            self._ep_input = self.effpose.get_inputs()[0].name
        self.crop = None  # square (x0, y0, side) around the hitter for this frame
        self.last = None  # last crop that held the hitter (to find them again after losing them)
        self.since_check = 0
        self.width = self.height = None

    # ---------------------------------------------------------------- models

    def _mp_image(self, rgb):
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def people(self, bgr):
        """Person boxes (x, y, w, h, score) from the detector."""
        import cv2

        r = self.detector.detect(self._mp_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        return [(d.bounding_box.origin_x, d.bounding_box.origin_y, d.bounding_box.width, d.bounding_box.height,
                 d.categories[0].score) for d in r.detections]

    def _movenet(self, bgr, crop, mirror):
        import cv2

        rgb = cv2.cvtColor(warp(bgr, crop, MOVENET_SIZE), cv2.COLOR_BGR2RGB)
        if mirror:
            rgb = rgb[:, ::-1]
        out = self.movenet.run(None, {self._mn_input: rgb[None].astype(np.int32)})[0][0, 0]
        x0, y0, side = crop
        u = 1 - out[:, 1] if mirror else out[:, 1]
        p = np.stack([x0 + u * side, y0 + out[:, 0] * side, out[:, 2]], 1)
        if mirror:
            for a, b in FLIP_COCO:
                p[[a, b]] = p[[b, a]]
        return p

    def _effpose(self, bgr, crop, mirror):
        import cv2

        rgb = cv2.cvtColor(warp(bgr, crop, EP_SIZE), cv2.COLOR_BGR2RGB).astype(np.float32)
        if mirror:
            rgb = rgb[:, ::-1]
        pk = heatmap_peaks(self.effpose.run(None, {self._ep_input: (rgb / 255.0 * 2 - 1)[None]})[0][0])
        if mirror:
            pk[:, 0] = EP_SIZE - 1 - pk[:, 0]
            pk = pk[EP_FLIP]
        x0, y0, side = crop
        s = side / EP_SIZE
        p = np.zeros((17, 3))
        for c, e in EP_TO_COCO.items():
            p[c] = (x0 + pk[e, 0] * s, y0 + pk[e, 1] * s, pk[e, 2])
        for c in (1, 2, 3, 4):
            p[c] = (p[0, 0], p[0, 1], 0.0)  # no eyes or ears: at the head, with no confidence
        return p

    def _mediapipe(self, bgr, crop, mirror):
        """33 landmarks (x, y px, visibility) of the person nearest the crop centre."""
        import cv2

        rgb = cv2.cvtColor(warp(bgr, crop, MP_CROP), cv2.COLOR_BGR2RGB)
        if mirror:
            rgb = rgb[:, ::-1]
        r = self.pose.detect(self._mp_image(rgb))
        if not r.pose_landmarks:
            return None
        x0, y0, side = crop
        s = side / MP_CROP
        best = None
        for lms in r.pose_landmarks:
            q = np.array([[l.x * MP_CROP, l.y * MP_CROP, l.visibility] for l in lms])
            if mirror:
                q[:, 0] = MP_CROP - 1 - q[:, 0]
                for a, b in MP_PAIRS:
                    q[[a, b]] = q[[b, a]]
            d = np.hypot(q[:, 0].mean() - MP_CROP / 2, q[:, 1].mean() - MP_CROP / 2)
            if best is None or d < best[0]:
                best = (d, q)
        q = best[1]
        q[:, 0] = x0 + q[:, 0] * s
        q[:, 1] = y0 + q[:, 1] * s
        return q

    # ---------------------------------------------------------------- tracking

    def _pick(self, boxes):
        """The hitter's box: near the current crop while tracking, near the last
        place they were seen after losing them (never someone else across the
        frame), else near --target-x or the most prominent person."""
        if not boxes:
            return None
        ref = self.crop or self.last
        if ref is not None:
            cx, cy, side = ref[0] + ref[2] / 2, ref[1] + ref[2] / 2, ref[2]

            def continuity(b):
                bx, by = b[0] + b[2] / 2, b[1] + b[3] / 2
                return -math.hypot(bx - cx, by - cy) / side - abs(math.log(max(b[2], b[3]) * (1 + 2 * CROP_MARGIN) / side))

            best = max(boxes, key=continuity)
            return best if continuity(best) > (-0.6 if self.crop else -1.2) else None
        if self.target_x is not None:
            tx = self.target_x * self.width
            return min(boxes, key=lambda b: abs(b[0] + b[2] / 2 - tx))
        return max(boxes, key=lambda b: b[2] * b[3] * math.sqrt(b[4]))

    def process(self, bgr):
        """BODY_25 frame (pixels) of the hitter in this frame (empty if not found)."""
        self.height, self.width = bgr.shape[:2]
        if self.crop is None or self.since_check >= RECHECK:
            box = self._pick(self.people(bgr))
            if box is not None:
                self.crop = square_from_box(*box[:4])
            self.since_check = 0
        if self.crop is None:
            return empty_frame()
        self.since_check += 1
        cands, mp33 = [], []
        for name in self.referee["candidates"]:
            model, _, mirrored = name.partition("-")
            if model == "mediapipe":
                q = self._mediapipe(bgr, self.crop, bool(mirrored))
                mp33.append(q)
                cands.append(None if q is None else q[MP_TO_COCO].tolist())
            elif model == "movenet":
                cands.append(self._movenet(bgr, self.crop, bool(mirrored)).tolist())
            elif model == "efficientpose":
                cands.append(self._effpose(bgr, self.crop, bool(mirrored)).tolist())
            else:
                raise ValueError(f"unknown referee candidate {name}")
        res = fuse(cands, self.referee)
        if res is None:
            self.crop = None
            return empty_frame()
        fused = res[0]
        mp_ref = next((q for q in mp33 if q is not None), None)
        mp_frame = None
        if mp_ref is not None:
            mp_frame = mediapipe_to_body25([_Landmark(x / self.width, y / self.height, v) for x, y, v in mp_ref],
                                           self.width, self.height)
        frame = fused_to_body25(fused, mp_frame)
        self._follow(np.array(fused))
        return frame

    def _follow(self, fused):
        """Next frame's crop from this frame's pose, limited so one bad frame can't throw it."""
        good = fused[(fused[:, 2] > 0.3) & np.isfinite(fused[:, 0])]
        core = fused[[5, 6, 11, 12], 2]
        if len(good) < 6 or np.nanmean(core) < 0.15:
            self.crop = None  # lost: look for the hitter again next frame
            return
        x0, y0, side = square_from_points(good[:, :2])
        px0, py0, pside = self.last = self.crop
        side = min(max(side, pside * 0.8), pside * 1.25)
        cx, cy = x0 + side / 2, y0 + side / 2
        pcx, pcy = px0 + pside / 2, py0 + pside / 2
        lim = 0.3 * pside
        cx = pcx + max(-lim, min(lim, cx - pcx))
        cy = pcy + max(-lim, min(lim, cy - pcy))
        self.crop = (cx - side / 2, cy - side / 2, side)

    def close(self):
        self.pose.close()
        self.detector.close()
