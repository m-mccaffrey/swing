"""OpenPose BODY_25 helpers (Python mirror of src/core/body25.js).

A frame is a flat list of 75 floats: [x0, y0, c0, ... x24, y24, c24] in
pixels, exactly like OpenPose's ``pose_keypoints_2d``. Missing keypoints are
(0, 0, 0).
"""

import math

NUM_KP = 25

KP = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4, "LShoulder": 5,
    "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9, "RKnee": 10, "RAnkle": 11,
    "LHip": 12, "LKnee": 13, "LAnkle": 14, "REye": 15, "LEye": 16, "REar": 17,
    "LEar": 18, "LBigToe": 19, "LSmallToe": 20, "LHeel": 21, "RBigToe": 22,
    "RSmallToe": 23, "RHeel": 24,
}

LR_PAIRS = [
    (KP["RShoulder"], KP["LShoulder"]),
    (KP["RElbow"], KP["LElbow"]),
    (KP["RWrist"], KP["LWrist"]),
    (KP["RHip"], KP["LHip"]),
    (KP["RKnee"], KP["LKnee"]),
    (KP["RAnkle"], KP["LAnkle"]),
    (KP["REye"], KP["LEye"]),
    (KP["REar"], KP["LEar"]),
    (KP["RBigToe"], KP["LBigToe"]),
    (KP["RSmallToe"], KP["LSmallToe"]),
    (KP["RHeel"], KP["LHeel"]),
]

# Same order as Object.values(LR_GROUPS) in body25.js: arms, legs, head.
LR_GROUPS = [
    LR_PAIRS[0:3],
    [LR_PAIRS[3], LR_PAIRS[4], LR_PAIRS[5], LR_PAIRS[8], LR_PAIRS[9], LR_PAIRS[10]],
    [LR_PAIRS[6], LR_PAIRS[7]],
]
ARMS, LEGS, HEAD = LR_GROUPS

PAIRS = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14), (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
    (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24),
]

COLORS = [
    (255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (255, 0, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (255, 0, 170), (170, 0, 255), (255, 0, 255),
    (85, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 255, 255), (0, 255, 255),
    (0, 255, 255),
]


def empty_frame():
    return [0.0] * (NUM_KP * 3)


def swap_lr(frame, pairs=LR_PAIRS):
    f = list(frame)
    for a, b in pairs:
        for k in range(3):
            f[a * 3 + k] = frame[b * 3 + k]
            f[b * 3 + k] = frame[a * 3 + k]
    return f


# MediaPipe BlazePose landmark indices.
_MP = {
    "nose": 0, "leftEye": 2, "rightEye": 5, "leftEar": 7, "rightEar": 8,
    "leftShoulder": 11, "rightShoulder": 12, "leftElbow": 13, "rightElbow": 14,
    "leftWrist": 15, "rightWrist": 16, "leftHip": 23, "rightHip": 24, "leftKnee": 25,
    "rightKnee": 26, "leftAnkle": 27, "rightAnkle": 28, "leftHeel": 29, "rightHeel": 30,
    "leftFootIndex": 31, "rightFootIndex": 32,
}


def _conf(lm):
    v = getattr(lm, "visibility", None)
    if v is None:
        v = getattr(lm, "presence", None)
    if v is None or not math.isfinite(v):
        return 0.5
    return max(0.001, min(1.0, v))


def mediapipe_to_body25(landmarks, width, height):
    """Convert 33 MediaPipe landmarks (normalized, objects with x/y/visibility)
    to a BODY_25 frame in pixels. Mirrors mediapipeToBody25() in body25.js."""
    f = empty_frame()
    if not landmarks or len(landmarks) < 33:
        return f

    def put(j, i, scale=1.0):
        lm = landmarks[i]
        f[j * 3] = lm.x * width
        f[j * 3 + 1] = lm.y * height
        f[j * 3 + 2] = _conf(lm) * scale

    def put_mid(j, a, b):
        la, lb = landmarks[a], landmarks[b]
        f[j * 3] = (la.x + lb.x) / 2 * width
        f[j * 3 + 1] = (la.y + lb.y) / 2 * height
        f[j * 3 + 2] = min(_conf(la), _conf(lb))

    m = _MP
    put(KP["Nose"], m["nose"])
    put_mid(KP["Neck"], m["leftShoulder"], m["rightShoulder"])
    put(KP["RShoulder"], m["rightShoulder"])
    put(KP["RElbow"], m["rightElbow"])
    put(KP["RWrist"], m["rightWrist"])
    put(KP["LShoulder"], m["leftShoulder"])
    put(KP["LElbow"], m["leftElbow"])
    put(KP["LWrist"], m["leftWrist"])
    put_mid(KP["MidHip"], m["leftHip"], m["rightHip"])
    put(KP["RHip"], m["rightHip"])
    put(KP["RKnee"], m["rightKnee"])
    put(KP["RAnkle"], m["rightAnkle"])
    put(KP["LHip"], m["leftHip"])
    put(KP["LKnee"], m["leftKnee"])
    put(KP["LAnkle"], m["leftAnkle"])
    put(KP["REye"], m["rightEye"])
    put(KP["LEye"], m["leftEye"])
    put(KP["REar"], m["rightEar"])
    put(KP["LEar"], m["leftEar"])
    put(KP["LBigToe"], m["leftFootIndex"])
    put(KP["LHeel"], m["leftHeel"])
    put(KP["RBigToe"], m["rightFootIndex"])
    put(KP["RHeel"], m["rightHeel"])
    for small, big, heel in (
        (KP["LSmallToe"], m["leftFootIndex"], m["leftHeel"]),
        (KP["RSmallToe"], m["rightFootIndex"], m["rightHeel"]),
    ):
        b, h = landmarks[big], landmarks[heel]
        f[small * 3] = (b.x * 0.8 + h.x * 0.2) * width
        f[small * 3 + 1] = (b.y * 0.8 + h.y * 0.2) * height
        f[small * 3 + 2] = _conf(b) * 0.5
    return f


def person_center(f):
    for j in (KP["MidHip"], KP["Neck"]):
        if f[j * 3 + 2] > 0.05:
            return (f[j * 3], f[j * 3 + 1])
    pts = [(f[j * 3], f[j * 3 + 1]) for j in range(NUM_KP) if f[j * 3 + 2] > 0.05]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def person_bbox(f):
    pts = [(f[j * 3], f[j * 3 + 1]) for j in range(NUM_KP) if f[j * 3 + 2] > 0.05]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def person_size(f):
    box = person_bbox(f)
    if not box:
        return 0.0
    conf = sum(f[j * 3 + 2] for j in range(NUM_KP) if f[j * 3 + 2] > 0.05)
    return (box[3] - box[1] + (box[2] - box[0]) * 0.5) * (conf / NUM_KP)


def track_hitter(people_per_frame, width, target_x=None):
    """Pick one person per frame from multi-person detections.

    The first frame with anyone in it locks on to the person nearest
    ``target_x`` (a 0-1 fraction of the frame width) if given, otherwise the
    most prominent person. After that we follow whoever is closest to the last
    position, and leave a frame empty rather than jump to someone else (the
    catcher or umpire) when the hitter is briefly lost.
    """
    out = []
    prev = None
    for people in people_per_frame:
        people = [p for p in people if person_center(p)]
        if not people:
            out.append(empty_frame())
            continue
        if prev is None:
            if target_x is not None:
                tx = target_x * width
                pick = min(people, key=lambda p: abs(person_center(p)[0] - tx))
            else:
                pick = max(people, key=person_size)
        else:
            pc = person_center(prev)
            pick = min(people, key=lambda p: math.hypot(person_center(p)[0] - pc[0], person_center(p)[1] - pc[1]))
            box = person_bbox(prev)
            limit = 0.6 * max(1.0, box[3] - box[1]) if box else float("inf")
            c = person_center(pick)
            if math.hypot(c[0] - pc[0], c[1] - pc[1]) > limit:
                out.append(empty_frame())
                continue
        out.append(pick)
        prev = pick
    return out


def detection_coverage(frames, min_conf=0.25):
    """Fraction of frames where neck, hips, a wrist and an ankle were found."""
    if not frames:
        return 0.0

    def ok(f, j):
        return f[j * 3 + 2] > min_conf

    good = 0
    for f in frames:
        if ok(f, 1) and ok(f, 8) and (ok(f, 4) or ok(f, 7)) and (ok(f, 11) or ok(f, 14)):
            good += 1
    return good / len(frames)
