"""MediaPipe Pose Landmarker (the same models the web app uses) → BODY_25."""

import urllib.request
from pathlib import Path

from .body25 import mediapipe_to_body25

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{m}/float16/1/pose_landmarker_{m}.task"


def ensure_model(name, cache_dir, log=print):
    """Download pose_landmarker_<name>.task into cache_dir if needed."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"pose_landmarker_{name}.task"
    if not path.exists() or path.stat().st_size < 1_000_000:
        url = MODEL_URL.format(m=name)
        log(f"Downloading pose model ({name})...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


class PoseExtractor:
    """Runs PoseLandmarker in VIDEO mode and returns every detected person
    as a BODY_25 frame (the caller picks the hitter)."""

    def __init__(self, model_path, num_poses=4):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options

        self._mp = mp
        opts = vision.PoseLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)
        self._last_ts = -1

    def detect(self, bgr_frame, t_seconds):
        import cv2

        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(self._last_ts + 1, int(round(t_seconds * 1000)))
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        return [mediapipe_to_body25(lms, w, h) for lms in (result.pose_landmarks or [])]

    def close(self):
        self._landmarker.close()
