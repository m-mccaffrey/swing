# Swing Match

Compare your baseball swing to MLB swings, in the browser.

1. Upload a **side-view** video of your swing (camera at 90° to the pitch path).
2. Pose tracking runs locally. A person detector finds the hitter and a tracker follows them. Several pose models (MoveNet, MediaPipe Pose, EfficientPose) look at the hitter, and a referee trained on human-labelled photos picks each joint's best answer (see [Pose tracking](#pose-tracking)). The joints are converted to the **OpenPose BODY_25** format.
3. Pick the frame with your **starting stance**. It is matched against every pro stance in the database, and the closest one is your match.
4. The rest of your swing is lined up with the pro's at six **beats**: stance, load, foot plant, contact, extension and finish. Time is stretched between them, so a slower or quicker swing, a different frame rate or slow motion all compare the same. About 28 checks then produce phase-by-phase **feedback** on positions: stride, head movement, hand load, front-leg brace, spine tilt, hip and shoulder turn, contact point, extension and finish.

- **Click any piece of advice** (feedback row, priority card, "matches" pill or table row). The player jumps to the exact pair of frames that check compared, rings the joints it measured, and draws the stance position with an arrow for movements measured from the stance. You can check that every claim is grounded in the poses.
- **Fix a joint.** If the tracking puts a joint in the wrong place, press **Fix a joint** on the stance step (or **Fix it on this frame** under the results video), scrub to the frame and drag the joint to where it really is. The frames around it follow, and the analysis goes through the point you placed (see [Fixing the tracking by hand](#fixing-the-tracking-by-hand)).
- **Adjust the beats.** The beats are detected automatically, and detection can miss. Click a beat chip under the video (or pick it in the beat editor), find the right frame, and press **Set to the frame shown**. The comparison and feedback update right away, and the edited beats are saved with the swing.
- **Saved swings:** every analyzed swing is kept in the browser's IndexedDB: poses, stance pick, pitcher side, height, adjusted beats and, if there is room, the video. Reopening one goes straight to the results with no upload and no re-analysis.

Videos never leave the device. The site is plain static files (no build step to run it), deployed to GitHub Pages by a workflow.

**Live site:** `https://m-mccaffrey.github.io/swing/` (once Pages is enabled, see below)

> The starter database holds **three synthetic placeholder swings** (`data/pros/synthetic-*.json`). A procedural 3D swing model generated them, and they are clearly labeled as synthetic in the app. Replace or extend them with real MLB swings using the [database builder](#adding-pro-swings) or OpenPose output.

## Deploying to GitHub Pages

`.github/workflows/pages.yml` runs the tests and builds `dist/` on every push. Pushes to the default branch are also deployed. The build vendors the MediaPipe runtime and TensorFlow.js and downloads the MediaPipe models, so the live site does not depend on a CDN.

One-time setup, done by a repo admin:

1. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
2. Re-run the latest "Test and deploy to GitHub Pages" workflow, or push to the default branch.

The site will be at `https://<owner>.github.io/<repo>/`.

If you use "Deploy from a branch" instead, the app still works. It loads MediaPipe and TensorFlow.js from jsDelivr and the MediaPipe models from Google's model bucket. The pose engine's own models are in `models/`, which is served as is.

## Filming tips (what the analysis assumes)

- **Camera perpendicular to the pitch path.** Put it on the open side facing the hitter's chest, or behind the hitter's back; both work. The pitcher should be off to the left or right of the frame. The app detects which side; you can override it.
- Whole body in frame for the entire swing, camera still, one person in view.
- Any frame rate works, including phone slow motion, and there is nothing to set: the swing is timed by its own hand speed. 60 fps or slow motion gives a sharper look at contact.
- **Sharp hands.** Film in bright light (daylight is best) or in slow-motion mode. Both use a short exposure, which freezes the hands. In dim light the hands smear into a blur at launch and the pose model has to guess where they are.

## Pose tracking

Everything downstream depends on where the joints are, so this is where the accuracy work went. For each frame:

1. **Find the hitter.** A person detector (EfficientDet-Lite0) looks for people at the start of the clip, every 30 frames after that, and whenever the hitter is lost. In between, a square crop follows the previous frame's pose. The crop can only move or resize a little per frame, so one bad frame can't throw it off. After losing the hitter it only accepts someone near where they were last seen, never the catcher across the frame.
2. **Several pose models look at the crop.** They are MoveNet Thunder, MediaPipe Pose (heavy) and EfficientPose. They were trained on different data and make different mistakes. In **Best** mode, MoveNet and MediaPipe also look at the mirror image of the crop, which changes their mistakes again. **Fast** mode runs MoveNet and MediaPipe once each.
3. **A close-up of the arms.** MoveNet looks again at a smaller crop around the elbows and wrists, centred where the models so far put them. With the arms filling more of the picture it makes different mistakes than on the whole body, and its shoulder, elbow and wrist answers join the others.
4. **A referee picks each joint.** It is a small logistic model (`models/referee-*.json`), trained on 2,643 people labelled by hand in COCO (no batters among them). For every joint it scores each model's answer: the model's own confidence, the distance to the other answers, how many agree with it, and which model gave it. For wrists it also looks at forearm length and the gap between the hands. The score is the probability that the answer is within 5% of body height of the truth. The most likely answer wins, averaged with the answers that agree with it, and its probability becomes the joint's confidence.
5. **Conversion to BODY_25**, then the per-joint robust spline and the hand repair described below.

Measured on COCO photos of batters, labelled by hand, against the old method (MediaPipe heavy on the whole frame):

| Batters labelled by hand | Wrists within 5% of body height | Wrists within 10% | Grip (between the hands) within 5% | Keypoint similarity (OKS) |
| --- | --- | --- | --- | --- |
| **79 never seen by any model in training** (COCO val2017) | | | | |
| Old method | 60.1% | 69.0% | 62.0% | 0.601 |
| Fast | 87.3% | 92.4% | 83.5% | 0.884 |
| Best | **88.6%** | **94.3%** | **89.9%** | **0.890** |
| **All 2,254** (most from COCO train2017, which MoveNet may have trained on) | | | | |
| Old method | 57.1% | 65.0% | 57.5% | 0.611 |
| Fast | 81.3% | 88.4% | 82.0% | 0.852 |
| Best | **83.3%** | **90.4%** | **84.8%** | **0.864** |

In a video the spline then smooths each joint over time and drops one-frame glitches, which these single-photo numbers don't include. Speed depends on the device. With no usable GPU (WebAssembly only), one frame took about 2 s in Best mode and 0.75 s in Fast mode in testing. A GPU is much faster. The Python tool took about 0.7 s per frame in Best mode and 0.2 s in Fast mode on a 4-core CPU.

Things that were tried and measured but not kept: larger or smaller crops, MoveNet on a wider crop as an extra answer, EfficientPose on the mirror image (+0.4 points for one more slow pass), a mirrored close-up (+0.2), a gradient-boosted referee (+0.5), training the referee on the engine's own answers instead of answers on the labelled box (no change), a learned wrist-correction network (worse), and MediaPipe's hand model for the wrists (found a hand for only a third of the wrists, and was less accurate when it did).

Reproduce the numbers, or retrain the referee, with `tools/posebench/` (see its README).

### Fixing the tracking by hand

No tracker is right on every frame: a hand hidden behind the body, a blurred bat at launch. When a joint is wrong, drag it to the right place (**Fix a joint**). What happens then (`src/core/fix.js`):

- The point you placed is trusted. The smoothing spline goes through it, and neither hand repair nor left/right repair moves or relabels it. If the other hand was lost on that frame, it is put next to yours.
- A mistake usually lasts several frames, so the frames around yours are re-picked from the pose models' other answers, which are kept from the analysis. Going outward from your point, each frame takes the answer nearest to where the joint is heading. This stops when the tracker's own answer agrees again for two frames, when no answer is close, half a second away, or at your next fix of that joint.
- Your fixes are saved with the swing and marked with a white square. **Undo last fix** removes the most recent one.

Models and licenses: [MoveNet](https://www.tensorflow.org/hub/tutorials/movenet) (Google, Apache 2.0) and [EfficientPose](https://github.com/daniegr/EfficientPose) (Apache 2.0), in the TensorFlow.js conversions from [@vladmandic/human-models](https://www.npmjs.com/package/@vladmandic/human-models) (MIT), with ONNX copies converted from those for the Python tool. [MediaPipe Pose Landmarker and EfficientDet-Lite0](https://ai.google.dev/edge/mediapipe/solutions/guide) (Google, Apache 2.0). The referees are trained on [COCO](https://cocodataset.org/) keypoint annotations (CC BY 4.0).

## How it works

| Step | Module |
| --- | --- |
| Pose tracking: person detector and crop tracker, several pose models and a learned referee per joint (see [Pose tracking](#pose-tracking)), then BODY_25 (Neck = shoulder midpoint, MidHip = hip midpoint, heels and toes from MediaPipe moved with the fused ankles) | `src/pose/engine.js`, `src/core/referee.js`, `src/core/body25.js` |
| Canonicalization: flip so the pitcher is to the right, relabel joints as **front/back** side (so left- and right-handed hitters, chest-view and back-view videos all compare directly), repair left/right label flicker, then fit each joint's track with a robust smoothing spline that penalizes sudden acceleration, ignores one-frame jumps and bridges short gaps along the curve (`robustSpline`), then normalize to the stance MidHip and **torso length** | `src/core/sequence.js` |
| Swing clock: the hands' fast burst (speed above 25% of its peak) lasts a fixed 0.586 *swing-seconds*, which gives the frames per swing-second for every time window in the analysis. Frame rate, slow motion and tempo need no input. When beats are known (pro entries, or beats you set by hand) and the hands were too poorly tracked to time the swing, the clock comes from the beat spacing instead | `src/core/sequence.js` (`estimateSwingFps`) |
| Hand repair: both hands hold the bat until after contact. Until shortly after the hands' peak speed, a doubtful wrist (low confidence, a forearm length far from usual, or a sudden jump while the other hand moves on smoothly) is placed next to the believable one, at the hands' spacing interpolated from frames where both were seen. Two doubtful wrists that agree are kept. Repaired points are drawn hollow, and **Pose-model points** under the player shows the model's raw output. Hand speed, which locates contact, is smoothed over 50 ms so a briefly misplaced hand doesn't look like a burst of speed | `src/core/sequence.js` (`repairHands`), `src/core/metrics.js` |
| Metrics a side camera can see: stance width, weight shift, head drift/drop, hand position and path, stride, apparent knee and elbow angles, spine tilt, back-shoulder drop, hip and shoulder turn (estimated from how much the hips and shoulders narrow) | `src/core/metrics.js` |
| Stance similarity: weighted RMS joint distance between normalized stances | `src/core/match.js` |
| Beats: open-ended DTW on movement-from-stance features proposes the user's beats from the pro's, then local refinement places them (hand-speed peak, foot landing, leg-kick peak). Comparison: time is stretched linearly between the six beats (yours, as detected or adjusted) | `src/core/dtw.js`, `src/core/compare.js`, `src/core/phases.js` |
| Feedback rules, tolerances, tips and drills | `src/core/feedback.js` |
| Procedural swing generator (placeholders, demo, tests) | `src/core/synth.js` |

Lengths are measured in torso lengths and shown in inches using the height you enter. Pro values are shown scaled to your size. The pro's ghost is redrawn with your limb proportions and stands on your ground line: each frame it is placed so its lowest foot point touches the ground under your stance, so its hip and head height come from its own leg bend at your leg lengths. If you dip and the pro doesn't, your head drops below theirs.

### Accuracy: the normalization sweep

`docs/normalization-sweep.md` (regenerate with `npm run sweep`) renders the *same* swing under 50 recording conditions (plus 3 deliberate swing changes) and checks that the comparison still says "identical". Conditions include framing, 480p to 4K, portrait, filmed from behind, left-handers, 1.5–2.0 m hitters, child proportions, 24–240 fps, 4×/8×/16× slow motion with nothing entered, swings 20–25% slower or quicker, long lead-ins, stance picked early or late, keypoint jitter, dropouts, label flicker, hands lost or misplaced by the pose model, one-frame keypoint glitches, camera tilt, off-axis and perspective cameras, a different pose model, and pros recorded differently. It also checks that real differences (longer stride, lower hands, less hip turn) are caught by the right check and nothing else. The same conditions run as tests on every push.

Built-in corrections the sweep relies on:
- Camera tilt is leveled from the ground line under the feet.
- The pro is rescaled to the hitter's limb proportions.
- Each swing runs on its own clock, measured from its hand-speed burst, so slow motion and frame rate need no input.
- Contact placement is robust to keypoint jitter.
- Hands the pose model loses or misplaces (hidden behind the body, motion blur, one hand snapped to the bat) are repaired from the other hand.

### Limitations

- Keep the camera within about 10° of perpendicular to the pitch path. Further off-axis (the sweep tests 25°), the 2D picture changes in ways one camera can't undo.
- A single side camera sees the swing in 2D. Rotation and joint angles are *apparent* values in the camera plane. They are most meaningful when you compare against pros filmed from the same kind of view.
- OpenPose places some keypoints slightly differently from the models used here (for example the hips and the neck). For the most consistent comparisons, build pro entries with the Python tool or the in-app builder. Both use the same pose engine as user swings.
- Tempo and rhythm are not judged. Time is stretched to line up the beats, so the feedback is about positions at each moment of the swing.
- The feedback is a comparison with one pro, not an absolute grade. Different good hitters do things differently. Use the ranking to pick a comparison that suits you.

## Adding pro swings

### From YouTube with the Python tool (recommended)

Run this on your own computer. YouTube often blocks downloads from cloud servers, so a home connection works best.

```sh
pip install -r tools/requirements.txt      # mediapipe, onnxruntime, yt-dlp (Python 3.9+); ffmpeg on PATH recommended

python tools/add_pro.py "https://www.youtube.com/watch?v=VIDEO_ID" \
    --start 1:02.5 --end 1:06 --name "Player Name" --bats R --team "Team"

git add data/pros && git commit -m "Add Player Name" && git push
```

Prefer a window? Run `python tools/add_pro_gui.py`. It's the same tool with a form: paste the link or pick a file, fill in the player and press **Analyze**. The **Beats** tab then shows every analyzed frame with the skeleton drawn on. Drag the bar or use ← → to find each beat, press **Set**, then **Add to database**. Changing a beat later and saving again updates the entry. There are also tabs for the six-frame contact sheet and the log, and buttons to open the database folder and copy the git commands. It needs Tkinter, which comes with Python from python.org; with Homebrew run `brew install python-tk`, on Debian/Ubuntu `sudo apt install python3-tk`.

What it does:

1. Downloads only that time range, video only (the whole video if ffmpeg isn't installed). It uses H.264 at up to 1080p and the highest frame rate available. Slow-motion replays are fine as they are. Long or slow-motion clips are thinned evenly to at most 240 analyzed frames (`--max-frames`).
2. Finds the hitter's pose in every frame with the pose engine (the same models and referee as the web app; see [Pose tracking](#pose-tracking)) and converts it to OpenPose BODY_25. If several people are in the frame (catcher, umpire), it locks on to the most prominent one and follows them. Pass `--target-x 0.3` to point at the hitter instead: 0 is the left edge of the frame, 1 the right.
3. Measures the swing clock, then auto-detects the pitcher side, stance and beats. The logic is the same as in the browser; `tools/tests` checks the Python port against the JavaScript.
4. Writes `data/pros/<id>.json`, adds it to `index.json` and deletes the video. Only keypoints are kept, plus the source URL and timestamps in the entry's `clip` field.
5. Saves a preview of the six beat frames with the skeleton drawn on to `.cache/previews/<id>.jpg` (not committed). Check it. If a beat is off, fix it in the window (`add_pro_gui.py`), or load the entry in `builder.html` (**Existing database entry**), fix it there and download the corrected file over the original.

Useful options:

| Option | Use |
| --- | --- |
| `--max-frames 480` | analyze up to this many frames (default 240; `0` = every frame). Clips with more frames are thinned evenly. Raise it for a finer look at contact in a long slow-motion clip |
| `--pitcher left` / `--pitcher right` | override the auto-detected pitcher side |
| `--stance 0.4` | stance moment, in seconds from the clip start |
| `--file swing.mp4` | use a local video instead of a URL |
| `--cookies-from-browser chrome` | if YouTube asks you to sign in |
| `--dry-run` | analyze and make the preview without touching the database |
| `--model fast` | pose tracking: `best` (default), `fast` (about three times as fast, a little less accurate), or `heavy`/`full`/`lite` (the older MediaPipe-only method) |

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
# optional: --pitcher right|left --stance <frame> --source <url>
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
  "fps": 60,                         // frames per second of the source video (slow motion needs nothing extra)
  "image": { "width": 1280, "height": 720 },
  "orientation": { "pitcherSide": "right" },
  "stanceFrame": 12,
  "phases": { "stance": 12, "load": 45, "footPlant": 61, "contact": 75, "extension": 80, "finish": 99 },
  "frames": [ { "pose_keypoints_2d": [x0, y0, c0, "…", x24, y24, c24] } ]
}
```

`frames` may also contain raw OpenPose per-frame documents (`{ "people": [...] }`) or bare keypoint arrays. COCO-18 keypoints are converted automatically. `phases` (the six beats) is optional and is auto-detected if missing. Older entries with a `speedFactor` field still load; the field is ignored. The same format is produced by **Download my pose data** on the Analyze page.

## Development

```sh
npm ci            # installs @mediapipe/tasks-vision and TensorFlow.js (vendored into the build)
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
src/pose/engine.js         pose engine: detector, crop tracker, pose models, referee
src/pose/detector.js       loads the engine (or MediaPipe alone) and runs it over a video
models/                    MoveNet and EfficientPose (TensorFlow.js; ONNX for Python) and the referees
src/ui/                    canvas drawing, SVG charts, video stage/player
data/pros/                 pro swing database (index.json + one file per swing)
scripts/                   build, dev server, OpenPose importer, synthetic generator
tools/add_pro.py           YouTube/local video → pro database entry (Python)
tools/add_pro_gui.py       the same, in a window (Tkinter)
tools/swingdb/             its modules: pose engine, video, detection (port of the JS), entry writer
tools/posebench/           pose-tracking benchmark on COCO batters, and referee training
tests/                     unit tests
```
