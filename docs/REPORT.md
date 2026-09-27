# Technical report

## What we built

A two-part pipeline for a fixed road camera:

- **Part A** (`solution.detect_events`): sample frames -> background-
  subtraction detector -> greedy-IoU tracker -> a self-built "road model"
  (convex hull of vehicle positions + dominant flow direction) -> a
  hand-written rule engine that turns finished tracks into
  `[start_sec, end_sec, label]` segments for 8 of the 14 official classes.
- **Part B** (`solution.RiskEstimator`): the same detector/tracker running
  causally, one frame at a time, feeding a time-to-collision estimate
  (plus hard-braking and pedestrian-proximity signals) into a single
  smoothed risk score per frame.
- The exact starter-kit interface (`run_submission.py`, `evaluate.py`)
  re-implemented from the challenge spec, since no starter kit files were
  provided directly to this environment -- see "What worked" for how we
  verified they behave as specified.

Everything is rule-based by design: zero training data was available (no
labels for the sample videos, and in fact no camera footage from this
specific setup was available on the build machine at all -- see
"What did not work / limitations"), so a hand-written detector+tracker+rules
pipeline was the only approach that could be built and verified without
guessing at what the hidden test set looks like.

## What worked

- **The zero-dependency baseline runs.** `MotionDetector` (OpenCV MOG2) +
  the IoU tracker need no weights, no GPU, and no internet, and were
  smoke-tested end to end (`solution.detect_events` and
  `solution.RiskEstimator.step`) against a real `.mp4` on this machine:
  both ran well inside the 3x-real-time budget (a 12s clip processed in
  ~11s combined for Part A + Part B on CPU).
- **The harness enforces the spec's contract.** `run_submission.py` runs
  each video in a separate process so the wall-clock time budget can
  actually be enforced with `terminate()` (a plain thread can't be force-
  stopped mid-C-call); a crash inside `detect_events` or `step` is caught
  and that video is scored as empty, matching the spec exactly. Verified by
  running the harness against a real video end to end and inspecting the
  resulting `predictions.json`.
- **`evaluate.py` reproduces the spec's formulas.** Verified against a
  small hand-built `examples/ground_truth.json` /
  `examples/predictions.json` pair: a correctly-matched `accident`
  prediction scored `F1 = 1.0` at all three IoU thresholds, a
  non-overlapping `red_light` prediction scored `F1 = 0.0`, and an
  unpredicted `congestion` ground-truth segment correctly dragged
  `Score_A` down via a false negative -- exactly the behavior the greedy
  IoU-matching spec describes.
- **The causal risk engine never reads ahead.** `RiskEstimator.step` only
  ever sees the frame and timestamp it's handed, and its own detector /
  tracker / road model are rebuilt from scratch in `reset()` -- there is no
  path in the code from "future frame" to "current risk score."

## What did not work / limitations

- **No real sample footage was available.** The challenge assumes teams
  start from `samples/*.mp4` from the actual camera; none existed on this
  build machine. That means: no real EDA (traffic density, lane heatmaps,
  object-count trends), no real annotated outputs for the website's
  "Results on sample videos" section, and -- most importantly -- **no way
  to tune any threshold in `src/rules.py` or `src/risk.py` against ground
  truth.** Every constant (10s for `stopped_vehicle`, 135 degrees for
  `wrong_way`, the IoU-overlap threshold for `accident`, etc.) is a
  reasoned default, not a fitted one. This is the single biggest gap
  between this baseline and a competitive score.
- **Six classes have no automatic signal**: `red_light`, `stop_line`,
  `illegal_u_turn`, `illegal_turn`, `solid_line_crossing`, `fire_smoke`.
  Each needs information a camera-agnostic motion pipeline doesn't have --
  a calibrated stop line and signal-state ROI, lane-marking geometry, or
  reliable smoke color/texture cues a coarse foreground blob doesn't carry.
  We chose not to ship guessed heuristics for these: a wrong guess only
  adds false positives (hurting precision) without any chance of recall,
  since we can't validate against real examples. `src/rules.py`'s
  `CalibrationConfig` is the wired-but-unused hook for turning these on
  once a per-camera calibration file exists.
- **Vehicle/person classification is a geometry guess.** Without a real
  object detector, `MotionDetector` splits blobs into `vehicle` / `person`
  / `unknown` using only size and aspect ratio. This is the weakest link
  in `jaywalking`, `failure_to_yield`, and indirectly `accident` /
  `near_miss` (a badly-classified blob can trigger the wrong rule, or
  none). `src/detector.py:YoloDetector` is a ready-to-use swap-in; it just
  needs `ultralytics` installed and a checkpoint in `weights/`.
- **2D bounding-box overlap has no depth cue.** `accident` and `near_miss`
  both key off box-overlap / proximity in image space. Two vehicles that
  are actually lanes apart but momentarily align with the camera's line of
  sight can register as "close" or "overlapping." A tighter detector
  reduces this (looser motion-blob boxes are more overlap-prone than tight
  YOLO boxes) but doesn't remove it -- a real fix needs either a homography
  to ground-plane coordinates (feasible for a fixed camera, not yet built)
  or a learned classifier over track pairs.
- **The starter kit's exact files weren't in hand.** `run_submission.py`
  and `evaluate.py` were re-implemented directly from the "Output format &
  interface" and "Evaluation & scoring" sections of the challenge brief.
  They match the spec as written and were verified against hand-built
  examples (above), but if the organizers' actual starter-kit files differ
  in some undocumented detail, ours should be diffed against the real ones
  before the final submission.

## Next steps (priority order)

1. Get real camera samples into `samples/`; hand-annotate a small dev set
   using the exact conventions in the challenge's "Event classes" table.
2. Run `evaluate.py --gt <dev labels>` and retune every threshold in
   `src/rules.py` / `src/risk.py` against it -- this alone is expected to
   move `Score_A` more than any single architectural change.
3. Swap in `YoloDetector` (already wired) once `weights/yolov8n.pt` is
   downloaded; re-measure `jaywalking` / `failure_to_yield` / `accident`
   precision specifically, since those are the rules most sensitive to
   detector quality.
4. Build a `calibration.json` for the real camera (stop line, signal ROI,
   lane geometry) to turn on the six currently-unimplemented classes.
5. Diff `run_submission.py` / `evaluate.py` against the organizers' actual
   starter-kit files once available, since ours were re-derived from the
   spec text rather than copied verbatim.
6. Add ground-plane homography for `accident` / `near_miss` distance
   estimates instead of raw image-space proximity.
