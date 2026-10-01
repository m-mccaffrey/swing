#!/usr/bin/env python3
"""Add a pro swing to the Swing Match database from YouTube or a local video.

  python tools/add_pro.py "https://www.youtube.com/watch?v=..." \\
      --start 1:02.5 --end 1:06 --name "Player Name" --bats R --team "Team"

  python tools/add_pro.py --file swing.mp4 --name "Player Name" --bats L

It downloads just that clip (video only), finds the hitter's pose in every
frame with MediaPipe (same model as the web app), converts it to OpenPose
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

from swingdb.analysis import PHASE_KEYS, auto_detect, estimate_slow_motion  # noqa: E402
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
    p.add_argument("--speed", type=float, default=1.0,
                   help="slow-motion factor of the footage (e.g. 4 for a 4x replay). Also skips frames automatically, see --ff")
    p.add_argument("--ff", "--fast-forward", type=int, metavar="N",
                   help="analyze every Nth frame (Nx fast-forward). Default: automatic from --speed so analysis runs at "
                        "about --max-fps frames per real second, e.g. a 4x replay at 60 fps -> every 4th frame")
    p.add_argument("--target-x", type=float, help="hitter's rough horizontal position (0 = left edge, 1 = right) if several people are in frame")
    p.add_argument("--model", choices=["lite", "full", "heavy"], default="heavy", help="pose model (default: heavy, most accurate)")
    p.add_argument("--max-fps", type=float, default=60, help="target frames per second of real time when skipping automatically (default 60)")
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


def main(argv=None):
    args = parse_args(argv)
    need("cv2", "mediapipe")
    need("mediapipe", "mediapipe")
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
        step = video.frame_step(native_fps, args.speed, args.max_fps, args.ff)
        fps = native_fps / step  # analyzed frames per second of video
        real_fps = fps * args.speed  # ... per second of real time
        slow = f", {args.speed:g}x slow motion" if args.speed != 1 else ""
        print(f"Video: {width}x{height}, {native_fps:.0f} fps{slow}")
        clip_len = (read_end if read_end is not None else count / native_fps) - read_start
        frames_est = max(1, int(clip_len * fps))
        if step > 1:
            print(f"Fast-forward {step}x: analyzing every {step}{'nd' if step == 2 else 'rd' if step == 3 else 'th'} frame "
                  f"(~{frames_est} frames, {real_fps:.0f} per real second)")
        else:
            print(f"Analyzing every frame (~{frames_est} frames, {real_fps:.0f} per real second)")
        if real_fps < 30:
            print(f"  warning: only {real_fps:.0f} frames per real second; fast swings may blur between frames. Try a smaller --ff.")

        # 2. Pose in every frame.
        model = pose_mod.ensure_model(args.model, cache / "models")
        extractor = pose_mod.PoseExtractor(model)
        people, t0 = [], time.time()
        for i, t, frame in video.read_frames(path, read_start, read_end, step):
            people.append(extractor.detect(frame, t))
            if i % 15 == 0:
                print(f"\r  pose: frame {i + 1}", end="", flush=True)
        extractor.close()
        print(f"\r  pose: {len(people)} frames in {time.time() - t0:.1f} s")
        if len(people) < 10:
            sys.exit("Fewer than 10 frames in that range; check --start/--end.")
        frames = track_hitter(people, width, args.target_x)
        coverage = detection_coverage(frames)
        if coverage < 0.5:
            print(f"  warning: the hitter was found in only {coverage:.0%} of frames. "
                  "Try --target-x, a tighter time range, or a clearer side-view clip.")

        if args.speed == 1:
            peak, factor = estimate_slow_motion(frames, fps)
            if factor > 1:
                print(f"  warning: this clip looks like about {factor}x slow motion (peak hand speed {peak:.1f} "
                      f"torso-lengths/s; real-time swings reach 5+). Timing in the entry will be wrong.\n"
                      f"           Re-run with --speed {factor} (that also makes the analysis ~{factor}x faster).")

        # 3. Pitcher side, stance, phases.
        stance = None if args.stance is None else max(0, min(len(frames) - 1, round(args.stance * fps)))
        det = auto_detect(frames, real_fps, stance_index=stance, pitcher_side=args.pitcher)
        phases = det["phases"]
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
            fps=fps, speed_factor=args.speed, width=width, height=height, pitcher_side=det["pitcherSide"],
            stance_frame=phases["stance"], phases=phases, frames=frames, clip=clip,
        )
        if args.dry_run:
            print("Dry run: database not changed.")
        else:
            out = write_entry(entry, args.db)
            print(f"Wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} and updated index.json")
            print("Check it: open the preview, or load the file in builder.html (Existing database entry).")
            print("Then: git add data/pros && git commit -m 'Add <player>' && git push")

        if args.keep_video and args.url:
            kept = cache / "previews" / f"{entry_id}{Path(path).suffix}"
            os.replace(path, kept)
            print(f"Kept clip: {kept}")
    finally:
        for f in work.glob("*"):
            f.unlink(missing_ok=True)
        work.rmdir()


if __name__ == "__main__":
    main()
