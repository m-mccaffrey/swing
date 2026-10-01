"""Contact sheet of the six phase frames with the detected skeleton drawn on,
so a new entry can be checked at a glance."""

from .analysis import PHASE_KEYS
from .body25 import PAIRS, COLORS, NUM_KP

LABELS = {"stance": "Stance", "load": "Load", "footPlant": "Foot plant", "contact": "Contact", "extension": "Extension", "finish": "Finish"}


def draw_skeleton(img, frame, min_conf=0.1):
    import cv2

    scale = max(1, round(img.shape[0] / 360))
    ok = lambda j: frame[j * 3 + 2] > min_conf  # noqa: E731
    pt = lambda j: (int(round(frame[j * 3])), int(round(frame[j * 3 + 1])))  # noqa: E731
    for a, b in PAIRS:
        if ok(a) and ok(b):
            r, g, bl = COLORS[b]
            cv2.line(img, pt(a), pt(b), (0, 0, 0), 4 * scale, cv2.LINE_AA)
            cv2.line(img, pt(a), pt(b), (bl, g, r), 2 * scale, cv2.LINE_AA)
    for j in range(NUM_KP):
        if ok(j):
            r, g, bl = COLORS[j]
            cv2.circle(img, pt(j), 3 * scale, (bl, g, r), -1, cv2.LINE_AA)


def contact_sheet(images_by_phase, frames, phases, fps, out_path, title=""):
    """images_by_phase: {phase_key: bgr image}; frames: BODY_25 pixel frames."""
    import cv2
    import numpy as np

    tiles = []
    for key in PHASE_KEYS:
        img = images_by_phase.get(key)
        if img is None:
            continue
        img = img.copy()
        draw_skeleton(img, frames[phases[key]])
        h = 360
        img = cv2.resize(img, (max(1, int(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)
        label = f"{LABELS[key]}  #{phases[key]}  {phases[key] / fps:.2f}s"
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        return None
    width = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, width - t.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles]
    while len(tiles) % 3:
        tiles.append(np.full_like(tiles[0], 30))
    rows = [cv2.hconcat(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    sheet = cv2.vconcat(rows)
    if title:
        bar = np.full((36, sheet.shape[1], 3), 20, dtype=sheet.dtype)
        cv2.putText(bar, title[:120], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        sheet = cv2.vconcat([bar, sheet])
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out_path
