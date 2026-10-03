#!/usr/bin/env python3
"""Add a pro swing to the Swing Match database from YouTube or a local video.

  python tools/add_pro.py "https://www.youtube.com/watch?v=..." \\
      --start 1:02.5 --end 1:06 --name "Player Name" --bats R --team "Team"

  python tools/add_pro.py --file swing.mp4 --name "Player Name" --bats L

It downloads just that clip (video only), finds the hitter's pose in every
frame with the pose engine (same pipeline as the web app: person detector,
crop tracking, MoveNet + MediaPipe, learned referee), converts it to OpenPose
BODY_25, auto-detects the pitcher side, stance and swing phases, and writes
data/pros/<id>.json plus an entry in data/pros/index.json. The video is
deleted afterwards; only keypoints are kept. A preview image of the six phase
frames is saved so you can check the result (or open the entry in
builder.html to fine-tune it). Then commit data/pros/ and push.

Install once:  pip install -r tools/requirements.txt   (ffmpeg recommended)
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from swingdb.analysis import PHASE_KEYS, auto_detect  # noqa: E402
from swingdb.body25 import detection_coverage, track_hitter  # noqa: E402
from swingdb.entry import make_entry, slugify, write_entry  # noqa: E402
from swingdb.video import parse_time  # noqa: E402

LABELS = {"stance": "Stance", "load": "Load", "footPlant": "Foot plant", "contact": "Contact", "extension": "Extension", "finish": "Finish"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Add a pro swing (YouTube or local video) to data/pros/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Install once:")[0].split("\n\n", 1)[1],
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("url", nargs="?", help="YouTube (or other yt-dlp supported) URL")
    src.add_argument("--file", help="local video file instead of a URL")
    p.add_argument("--start", help="clip start, e.g. 83.5 or 1:23.5 (default: beginning)")
    p.add_argument("--end", help="clip end (default: start + 6 s for URLs, end of file otherwise)")
    p.add_argument("--name", required=True, help="player name")
    p.add_argument("--id", help="entry id (default: from the name)")
    p.add_argument("--team", default="")
    p.add_argument("--bats", choices=["R", "L", "S"], default="")
    p.add_argument("--notes", default="")
    p.add_argument("--pitcher", choices=["left", "right"], help="side of the frame the pitcher is on (default: auto)")
    p.add_argument("--stance", type=float, help="stance time in seconds from the clip start (default: auto)")
    p.add_argument("--target-x", type=float, help="hitter's rough horizontal position (0 = left edge, 1 = right) if several people are in frame")
    p.add_argument("--model", choices=["best", "fast", "heavy", "full", "lite"], default="best",
                   help="pose tracking: best (default; MoveNet + MediaPipe on the hitter's crop and its mirror image, "
                        "cross-checked), fast (MoveNet and MediaPipe once each, about three times as fast), or heavy/full/lite (the older "
                        "MediaPipe-only method)")
    p.add_argument("--max-frames", type=int, default=240,
                   help="analyze at most this many frames of the clip, skipping evenly (default 240). Slow-motion clips have "
                        "far more frames than a swing needs; the swing's own clock keeps the comparison right either way")
    p.add_argument("--cookies-from-browser", help="pass browser cookies to yt-dlp if YouTube asks you to sign in (chrome, firefox, safari, ...)")
    p.add_argument("--keep-video", action="store_true", help="keep the downloaded clip (in .cache/previews/, not committed)")
    p.add_argument("--dry-run", action="store_true", help="analyze and make the preview, but don't write the database")
    p.add_argument("--db", default=str(ROOT / "data" / "pros"), help="database directory (default: data/pros)")
    return p.parse_args(argv)


def need(module, pip_name):
    try:
        __import__(module)
    except ImportError:
        sys.exit(f"Missing Python package '{pip_name}'. Run: pip install -r tools/requirements.txt")


def main(argv=None, *, thumb_height=None):
    """Run the tool. With ``thumb_height`` every analyzed frame is also kept as
    a JPEG at most that tall (``result["thumbs"]``), so the window can show
    them for beat review without keeping the video."""
    args = parse_args(argv)
    need("cv2", "mediapipe")
    need("mediapipe", "mediapipe")
    engine_mode = args.model in ("best", "fast")
    if engine_mode:
        need("onnxruntime", "onnxruntime")
    if args.url:
        need("yt_dlp", "yt-dlp")
    from swingdb import pose as pose_mod
    from swingdb import preview, video

    start = parse_time(args.start) or 0.0
    end = parse_time(args.end)
    entry_id = args.id or slugify(args.name)
    if not entry_id:
        sys.exit("Could not make an id from the name; pass --id.")
    cache = ROOT / ".cache"
    work = Path(tempfile.mkdtemp(prefix="swingdb-"))
    clip = None
    title = args.name

    try:
        # 1. Get the clip.
        if args.url:
            if end is None:
                end = start + 6.0
            if end <= start:
                sys.exit("--end must be after --start")
            print(f"Downloading {end - start:.1f} s from {args.url} ...")
            path, info, read_start, read_end = video.download_clip(
                args.url, start, end, work, cookies_from_browser=args.cookies_from_browser
            )
            title = info.get("title") or title
            clip = {
                "url": info.get("webpage_url") or args.url,
                "title": info.get("title", ""),
                "channel": info.get("channel") or info.get("uploader", ""),
                "start": round(start, 3),
                "end": round(end, 3),
            }
            source = f"{clip['url']}{'&' if '?' in clip['url'] else '?'}t={int(start)} ({clip['title']}, {start:.1f}-{end:.1f} s)"
        else:
            path = args.file
            if not Path(path).exists():
                sys.exit(f"No such file: {path}")
            read_start, read_end = start, end
            source = title = Path(path).name
            if args.start or args.end:
                source += f" ({start:.1f}-{end if end is not None else 'end'} s)"

        native_fps, width, height, count = video.probe(path)
        clip_len = (read_end if read_end is not None else count / native_fps) - read_start
        step = video.frame_step(clip_len * native_fps, args.max_frames)
        fps = native_fps / step  # analyzed frames per second of video
        print(f"Video: {width}x{height}, {native_fps:.0f} fps, {clip_len:.1f} s")
        frames_est = max(1, int(clip_len * fps))
        if step > 1:
            print(f"Analyzing every {step}{'nd' if step == 2 else 'rd' if step == 3 else 'th'} frame (~{frames_est} frames; --max-frames {args.max_frames})")
        else:
            print(f"Analyzing every frame (~{frames_est} frames)")

        # 2. Pose in every frame.
        model = pose_mod.ensure_model("heavy" if engine_mode else args.model, cache / "models")
        if engine_mode:
            from swingdb.engine import PoseEngine

            tracker = PoseEngine(model, quality=args.model, target_x=args.target_x)
        else:
            tracker = pose_mod.PoseExtractor(model)
        frames, people, times, thumbs, thumb_scale, t0 = [], [], [], [], 1.0, time.time()
        for i, t, frame in video.read_frames(path, read_start, read_end, step):
            if engine_mode:
                frames.append(tracker.process(frame))
            else:
                people.append(tracker.detect(frame, t))
            times.append(t)
            if thumb_height:
                thumb_scale = min(1.0, thumb_height / frame.shape[0])
                thumbs.append(preview.encode_thumb(frame, thumb_scale))
            if i % 15 == 0:
                print(f"\r  pose: frame {i + 1}", end="", flush=True)
        tracker.close()
        print(f"\r  pose: {len(times)} frames in {time.time() - t0:.1f} s")
        if len(times) < 10:
            sys.exit("Fewer than 10 frames in that range; check --start/--end.")
        if not engine_mode:
            frames = track_hitter(people, width, args.target_x)
        coverage = detection_coverage(frames)
        if coverage < 0.5:
            print(f"  warning: the hitter was found in only {coverage:.0%} of frames. "
                  "Try --target-x, a tighter time range, or a clearer side-view clip.")

        # 3. Pitcher side, stance, phases.
        stance = None if args.stance is None else max(0, min(len(frames) - 1, round(args.stance * fps)))
        det = auto_detect(frames, fps, stance_index=stance, pitcher_side=args.pitcher)
        phases = det["phases"]
        swing_fps = det["swingFps"]
        print(f"Swing clock: {swing_fps / fps:.2f}x the video's frame rate"
              + (" (slow motion or a slow swing; handled automatically)" if swing_fps > 1.6 * fps else ""))
        if swing_fps < 24:
            print(f"  warning: only about {swing_fps:.0f} frames cover each swing-second; raise --max-frames for a finer look at contact.")
        side_note = "given" if args.pitcher else f"auto, {det['sideConfidence']:.0%} confident"
        print(f"Pitcher side: {det['pitcherSide']} ({side_note})")
        for k in PHASE_KEYS:
            print(f"  {LABELS[k]:<10} frame {phases[k]:>4}  {phases[k] / fps:6.2f} s")

        # 4. Preview of the phase frames.
        images = {}
        for i, _, frame in video.read_frames(path, read_start, read_end, step):
            for k in PHASE_KEYS:
                if phases[k] == i:
                    images[k] = frame
            if i >= max(phases.values()):
                break
        (cache / "previews").mkdir(parents=True, exist_ok=True)
        preview_path = cache / "previews" / f"{entry_id}.jpg"
        preview.contact_sheet(images, frames, phases, fps, preview_path, title=f"{args.name} - {title}")
        print(f"Preview: {preview_path}")

        # 5. Database entry.
        entry = make_entry(
            id=entry_id, name=args.name, team=args.team, bats=args.bats, notes=args.notes, source=source,
            fps=fps, width=width, height=height, pitcher_side=det["pitcherSide"],
            stance_frame=phases["stance"], phases=phases, frames=frames, clip=clip,
        )
        out = None
        if args.dry_run:
            print("Dry run: database not changed.")
        else:
            out = write_entry(entry, args.db)
            print(f"Wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} and updated index.json")
            print("Check it: open the preview, or load the file in builder.html (Existing database entry).")
            print("Then: git add data/pros && git commit -m 'Add <player>' && git push")

        video_path = Path(path) if not args.url else None
        if args.keep_video and args.url:
            video_path = cache / "previews" / f"{entry_id}{Path(path).suffix}"
            os.replace(path, video_path)
            print(f"Kept clip: {video_path}")
        return {"entry": out, "entryData": entry, "preview": preview_path, "id": entry_id, "phases": phases,
                "pitcherSide": det["pitcherSide"], "fps": fps, "swingFps": swing_fps, "frames": frames,
                "times": times, "clipStart": read_start, "video": video_path, "title": title, "db": args.db,
                "thumbs": thumbs, "thumbScale": thumb_scale}
    finally:
        for f in work.glob("*"):
            f.unlink(missing_ok=True)
        work.rmdir()


if __name__ == "__main__":
    main()
