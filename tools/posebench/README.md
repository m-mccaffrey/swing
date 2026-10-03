# Pose-tracking benchmark

How well does the pose engine find a batter's joints? This measures it on COCO
photos whose keypoints were labelled by hand, and trains the referee that
fuses the models (`models/referee-*.json`).

Needs the tool's requirements (`pip install -r tools/requirements.txt`), plus
scikit-learn for training. About 1.3 GB of downloads and, on four CPU cores,
roughly half an hour per run of the full engine over each set.

```sh
python tools/posebench/build_sets.py                  # annotations + images into tools/posebench/data/
python tools/posebench/run.py mediapipe-full-frame    # the old method
python tools/posebench/run.py engine-fast
python tools/posebench/run.py engine-best

# retrain the referees (each model's answers on the labelled box, then logistic regression)
python tools/posebench/run.py candidates --set general
python tools/posebench/run.py candidates --set batters
python tools/posebench/train_referee.py               # add --write to update models/
```

## Sets

- **batters**: 2,254 people holding a baseball bat (a bat box within a quarter
  of their height of a wrist, both wrists labelled, at least 120 px tall).
  79 come from COCO val2017. No model here was trained on those, so they are
  the honest test. The rest come from train2017, which MoveNet may have
  trained on.
- **general**: 2,643 other people from val2017 with 10 or more labelled
  keypoints. The referee is trained on these only.

## Metrics

Distances are measured in body heights (the labelled box's height):

- **PCK5 / PCK10**: the share of labelled joints within 5% / 10% of the label.
- **Grip**: the midpoint of the two wrists, which the swing analysis uses for the hands.
- **OKS**: COCO's keypoint similarity (1 = perfect).

The engine runs its own person detector, and a frame-to-frame tracker isn't
possible on single photos. The detection that overlaps the labelled person
most is taken as the hitter, as the tracker would in a video. The old method
gets its best case: of the people MediaPipe finds, the one closest to the
label is scored.
