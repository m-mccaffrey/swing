"""Get a swing clip from YouTube (yt-dlp) or a local file and read its frames
(OpenCV). Heavy imports happen inside the functions."""

import math
import shutil
from pathlib import Path


def parse_time(s):
    """'83.5', '1:23.5' or '0:01:23.5' -> seconds."""
    if s is None:
        return None
    parts = str(s).strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Bad time: {s!r}")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return secs


class _QuietLogger:
    """Keep yt-dlp's chatter out of the way; real failures still raise."""

    def debug(self, msg):
        pass

    info = warning = debug

    def error(self, msg):
        pass


def download_clip(url, start, end, workdir, cookies_from_browser=None, max_height=1080, log=print, extra_opts=None):
    """Download [start, end] seconds of a YouTube video (video only, no audio).

    With ffmpeg installed only that section is fetched and cut exactly.
    Without it the whole video stream is downloaded and the section is read
    from it. Returns (path, info, clip_start, clip_end) where clip_* are the
    times to read inside the downloaded file.
    """
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import download_range_func

    has_ffmpeg = shutil.which("ffmpeg") is not None
    opts = {
        # Video-only stream: no merging needed, and audio is useless here.
        "format": f"bv*[height<={max_height}]/b[height<={max_height}]/bv*/b",
        # Prefer H.264 (decodes everywhere), then resolution, then frame rate.
        "format_sort": ["vcodec:h264", f"res:{max_height}", "fps"],
        "outtmpl": str(Path(workdir) / "clip.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _QuietLogger(),
    }
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if has_ffmpeg:
        opts["download_ranges"] = download_range_func(None, [(start, end)])
        opts["force_keyframes_at_cuts"] = True
    else:
        log("ffmpeg not found: downloading the whole video stream, then reading your time range from it.")
    opts.update(extra_opts or {})
    sectioned = has_ffmpeg
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:  # yt_dlp.utils.DownloadError
        if not sectioned or "partially downloaded" not in str(e):
            raise
        log("This source can't be cut while downloading; fetching the whole video instead.")
        for key in ("download_ranges", "force_keyframes_at_cuts"):
            opts.pop(key, None)
        sectioned = False
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    downloads = info.get("requested_downloads") or []
    path = downloads[0].get("filepath") if downloads else None
    if not path or not Path(path).exists():
        found = sorted(Path(workdir).glob("clip.*"))
        if not found:
            raise RuntimeError("yt-dlp finished but no video file was written")
        path = str(found[0])
    if sectioned:
        return path, info, 0.0, end - start
    return path, info, start, end


def frame_step(clip_frames, max_frames=240):
    """Analyze every Nth frame so a clip yields at most ``max_frames`` frames.
    Slow-motion clips have far more frames than a swing needs; the swing's own
    clock keeps the comparison right however many are skipped."""
    if not max_frames or clip_frames <= max_frames:
        return 1
    return max(1, math.ceil(clip_frames / max_frames))


def read_frames(path, start=0.0, end=None, step=1):
    """Yield (index, time_in_clip, bgr_frame) for every ``step``-th frame in
    [start, end]. Skipped frames are only grabbed, not converted."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    k = 0
    out_i = 0
    try:
        while cap.grab():
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if end is not None and t > end + 1e-6:
                break
            if t + 1e-6 < start:
                continue
            if k % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield out_i, t - start, frame
                out_i += 1
            k += 1
    finally:
        cap.release()


def probe(path):
    """(native_fps, width, height, frame_count) of a video file."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, w, h, count
