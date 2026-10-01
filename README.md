# Swing Match

Compare your baseball swing to MLB swings, in the browser.

1. Upload a **side-view** video of your swing (camera at 90° to the pitch path).
2. Pose estimation runs locally (MediaPipe Pose Landmarker). The landmarks are converted to the **OpenPose BODY_25** format.
3. Pick the frame with your **starting stance**. It is matched against every pro stance in the database, and the closest one is your match.
4. The rest of your swing is aligned with the pro's using dynamic time warping. Load, foot plant, contact, extension and finish are located, and about 30 checks produce phase-by-phase **feedback**: stride, head movement, hand load, front-leg brace, spine tilt, hip and shoulder turn, contact point, extension, finish, and timing.

- **Click any piece of advice** (feedback row, priority card, "matches" pill or table row). The player jumps to the exact pair of frames that check compared, rings the joints it measured, and draws the stance position with an arrow for movements measured from the stance. You can check that every claim is grounded in the poses.
- **Saved swings:** every analyzed swing is kept in the browser's IndexedDB: poses, stance pick, pitcher side, height and, if there is room, the video. Reopening one goes straight to the results with no upload and no re-analysis.

Videos never leave the device. The site is plain static files (no build step to run it), deployed to GitHub Pages by a workflow.

**Live site:** `https://m-mccaffrey.github.io/swing/` (once Pages is enabled, see below)

> The starter database holds **three synthetic placeholder swings** (`data/pros/synthetic-*.json`). A procedural 3D swing model generated them, and they are clearly labeled as synthetic in the app. Replace or extend them with real MLB swings using the [database builder](#adding-pro-swings) or OpenPose output.

## Deploying to GitHub Pages

`.github/workflows/pages.yml` runs the tests and builds `dist/` on every push. Pushes to the default branch are also deployed. The build vendors the MediaPipe runtime and downloads the pose models, so the live site does not depend on a CDN.

One-time setup, done by a repo admin:

1. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
2. Re-run the latest "Test and deploy to GitHub Pages" workflow, or push to the default branch.

The site will be at `https://<owner>.github.io/<repo>/`.

If you use "Deploy from a branch" instead, the app still works. It loads MediaPipe from jsDelivr and the models from Google's model bucket.

## Filming tips (what the analysis assumes)

- **Camera perpendicular to the pitch path.** Put it on the open side facing the hitter's chest, or behind the hitter's back; both work. The pitcher should be off to the left or right of the frame. The app detects which side; you can override it.
- Whole body in frame for the entire swing, camera still, one person in view.
- 60 fps or phone slow motion if possible. Set the slow-motion factor so timing is correct.

## How it works

| Step | Module |
| --- | --- |
| MediaPipe 33 landmarks → BODY_25 (Neck = shoulder midpoint, MidHip = hip midpoint, heels/toes mapped) | `src/core/body25.js` |
| Canonicalization: flip so the pitcher is to the right, relabel joints as **front/back** side (so left- and right-handed hitters, chest-view and back-view videos all compare directly), repair left/right label flicker, fill gaps, smooth, then normalize to the stance MidHip and **torso length** | `src/core/sequence.js` |
| Metrics a side camera can see: stance width, weight shift, head drift/drop, hand position and path, stride, apparent knee and elbow angles, spine tilt, back-shoulder drop, hip and shoulder turn (estimated from how much the hips and shoulders narrow) | `src/core/metrics.js` |
| Stance similarity: weighted RMS joint distance between normalized stances | `src/core/match.js` |
| Swing alignment: open-ended DTW on movement-from-stance features at 60 fps real time, phase transfer, then local refinement of events (peak hand speed, foot landing, leg-kick peak) calibrated against the pro | `src/core/dtw.js`, `src/core/compare.js`, `src/core/phases.js` |
| Feedback rules, tolerances, tips and drills | `src/core/feedback.js` |
| Procedural swing generator (placeholders, demo, tests) | `src/core/synth.js` |

Lengths are measured in torso lengths and shown in inches using the height you enter. Pro values are shown scaled to your size.

### Accuracy: the normalization sweep

`docs/normalization-sweep.md` (regenerate with `npm run sweep`) renders the *same* swing under 44 recording conditions (plus 4 deliberate swing changes) and checks that the comparison still says "identical". Conditions include framing, 480p to 4K, portrait, filmed from behind, left-handers, 1.5–2.0 m hitters, child proportions, 24–240 fps, 4×/8× slow motion, long lead-ins, stance picked early or late, keypoint jitter, dropouts, label flicker, camera tilt, off-axis and perspective cameras, a different pose model, and pros recorded differently. It also checks that real differences (longer stride, lower hands, slower swing, less hip turn) are caught by the right check and nothing else. The same conditions run as tests on every push.

Built-in corrections the sweep relies on:
- Camera tilt is leveled from the ground line under the feet.
- The pro is rescaled to the hitter's limb proportions.
- Slow motion that wasn't set is detected from hand speed. The app offers a one-click fix; the Python tool prints a warning.
- Contact timing is robust to keypoint jitter.

### Limitations

- Keep the camera within about 10° of perpendicular to the pitch path. Further off-axis (the sweep tests 25°), the 2D picture changes in ways one camera can't undo.
- A single side camera sees the swing in 2D. Rotation and joint angles are *apparent* values in the camera plane. They are most meaningful when you compare against pros filmed from the same kind of view.
- MediaPipe and OpenPose place some keypoints slightly differently (for example the hips and the neck). For the most consistent comparisons, build pro entries with the in-app builder, which uses the same pose model as user swings.
- The feedback is a comparison with one pro, not an absolute grade. Different good hitters do things differently. Use the ranking to pick a comparison that suits you.

## Adding pro swings

### From YouTube with the Python tool (recommended)

Run this on your own computer. YouTube often blocks downloads from cloud servers, so a home connection works best.

```sh
pip install -r tools/requirements.txt      # mediapipe + yt-dlp (Python 3.9+); ffmpeg on PATH recommended

python tools/add_pro.py "https://www.youtube.com/watch?v=VIDEO_ID" \
    --start 1:02.5 --end 1:06 --name "Player Name" --bats R --team "Team"

git add data/pros && git commit -m "Add Player Name" && git push
```

What it does:

1. Downloads only that time range, video only (the whole video if ffmpeg isn't installed). It uses H.264 at up to 1080p and the highest frame rate available.
2. Finds the hitter's pose in every frame with MediaPipe, the same model the web app uses, and converts it to OpenPose BODY_25. If several people are in the frame (catcher, umpire), it locks on to the most prominent one. Pass `--target-x 0.3` to point at the hitter instead: 0 is the left edge of the frame, 1 the right.
3. Auto-detects the pitcher side, stance and phases. The logic is the same as in the browser; `tools/tests` checks the Python port against the JavaScript.
4. Writes `data/pros/<id>.json`, adds it to `index.json` and deletes the video. Only keypoints are kept, plus the source URL and timestamps in the entry's `clip` field.
5. Saves a preview of the six phase frames with the skeleton drawn on to `.cache/previews/<id>.jpg` (not committed). Check it. If a phase is off, load the entry in `builder.html` (**Existing database entry**), fix it, and download the corrected file over the original.

Useful options:

| Option | Use |
| --- | --- |
| `--speed 4` | the clip is a 4× slow-motion replay (common on YouTube and broadcasts). Fixes timing and also fast-forwards automatically: frames are skipped so analysis runs at about 60 frames per *real* second (a 4× replay at 60 fps → every 4th frame, so 4× faster) |
| `--ff N` | fast-forward by hand: analyze every Nth frame (`--ff 2` = 2× faster). Overrides the automatic choice; warns if fewer than 30 frames per real second are left |
| `--max-fps 120` | change the automatic target (frames per real second; default 60) |
| `--pitcher left` / `--pitcher right` | override the auto-detected pitcher side |
| `--stance 0.4` | stance moment, in seconds from the clip start |
| `--file swing.mp4` | use a local video instead of a URL |
| `--cookies-from-browser chrome` | if YouTube asks you to sign in |
| `--dry-run` | analyze and make the preview without touching the database |
| `--model full` | faster, slightly less accurate pose model (default `heavy`) |

Pick clips filmed from the side, perpendicular to the pitch path, with the hitter's whole body in view. Broadcast center-field shots don't work for this.

### With the in-app builder (`builder.html`)

1. Load an MLB swing video (side view), a folder of OpenPose `--write_json` files, or an existing entry to edit.
2. Check the pitcher side, stance frame and phase frames. They are auto-detected, and you can correct them frame by frame.
3. Enter the player's details, then **Download entry JSON**. You can also **Save to this browser** to use it on the Analyze page right away.
4. Commit the file to `data/pros/` and add it to `data/pros/index.json`:

```json
{ "id": "player-name", "file": "player-name.json", "name": "Player Name", "bats": "R" }
```

### From OpenPose output on the command line

```sh
openpose.bin --video swing.mp4 --write_json out/ --display 0 --render_pose 0
node scripts/openpose-to-db.mjs out/ --name "Player Name" --fps 60 --bats R --team "Team"
# optional: --pitcher right|left --stance <frame> --speed <slow-mo factor> --source <url>
```

This writes `data/pros/<id>.json` and registers it in the index. When a frame has several people (catcher, umpire), the script locks on to the most prominent person in the first frame and follows them.

Only add footage you have the rights to use, and record the source in the entry.

### Entry format (`swing-db/v1`)

```jsonc
{
  "schema": "swing-db/v1",
  "id": "player-name",
  "name": "Player Name",
  "team": "Team",
  "bats": "R",                       // R, L or S
  "keypointFormat": "BODY_25",
  "fps": 60,                         // frames per second of the source video
  "speedFactor": 1,                  // 4 if the source was 4× slow motion
  "image": { "width": 1280, "height": 720 },
  "orientation": { "pitcherSide": "right" },
  "stanceFrame": 12,
  "phases": { "stance": 12, "load": 45, "footPlant": 61, "contact": 75, "extension": 80, "finish": 99 },
  "frames": [ { "pose_keypoints_2d": [x0, y0, c0, "…", x24, y24, c24] } ]
}
```

`frames` may also contain raw OpenPose per-frame documents (`{ "people": [...] }`) or bare keypoint arrays. COCO-18 keypoints are converted automatically. `phases` is optional and is auto-detected if missing. The same format is produced by **Download my pose data** on the Analyze page.

## Development

```sh
npm ci            # installs @mediapipe/tasks-vision (vendored into the build)
npm test          # unit tests (node:test)
npm run test:py   # Python tool tests (stdlib only; checks parity with the JS)
npm run dev       # serve the repo at http://localhost:8080 (vendor paths mapped to node_modules)
npm run build     # assemble dist/ (vendors MediaPipe, downloads pose models to .cache/)
npm run serve     # serve dist/
npm run gen       # regenerate the synthetic placeholder entries
```

No framework and no bundler: the app is ES modules loaded directly by the browser. The core analysis in `src/core/` is DOM-free and runs in Node.

```
index.html, builder.html   pages
css/styles.css             styles (light/dark)
src/app.js                 analyze page controller
src/builder.js             database builder controller
src/core/                  pose format, normalization, metrics, matching, DTW, feedback, synthetic swings
src/pose/detector.js       MediaPipe Pose Landmarker wrapper
src/ui/                    canvas drawing, SVG charts, video stage/player
data/pros/                 pro swing database (index.json + one file per swing)
scripts/                   build, dev server, OpenPose importer, synthetic generator
tools/add_pro.py           YouTube/local video → pro database entry (Python)
tools/swingdb/             its modules: pose, video, detection (port of the JS), entry writer
tests/                     unit tests
```
