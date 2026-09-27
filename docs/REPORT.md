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
- `run_submission.py` and `evaluate.py` are the **organizers' own,
  unmodified `wiut_cv_scripts` starter-kit files** -- not a re-derivation.
  An earlier version of this repo shipped a hand-written stand-in for both
  (written before the real starter kit had been supplied); it was fully
  replaced the moment the real `wiut_cv_scripts.zip` arrived, per the
  spec's "unchanged" requirement.

Everything is rule-based by design: zero training data was used (no labels
for the sample videos), so a hand-written detector+tracker+rules pipeline
was the approach that could be built and verified without guessing at what
the hidden test set looks like.

## What worked

- **The zero-dependency baseline runs against the real harness.**
  `MotionDetector` (OpenCV MOG2) + the IoU tracker need no weights, no GPU,
  and no internet. Verified end to end with the *official*
  `run_submission.py`: a 12s synthetic clip processed in ~5.2s combined for
  Part A + Part B on CPU (budget 36s at the spec's 3x multiplier) --
  comfortable margin, though a real several-minutes-long 1080p clip is the
  test that actually matters (see "What we could not verify" below).
- **`evaluate.py` (the official file) reproduces the spec's formulas as
  expected.** Verified against `examples/ground_truth.json` /
  `examples/predictions.json` (also the organizers' own files): Score_A =
  0.9074, Score_B = 0.76, model score = 0.8632 -- a correctly-matched
  `accident` prediction scored F1 = 1.0 at all three IoU thresholds, and an
  extra unmatched `stopped_vehicle` prediction correctly cost precision at
  τ=0.7 only. Exactly the greedy IoU-matching behavior the spec describes.
- **The causal risk engine never reads ahead.** `RiskEstimator.step` only
  ever sees the frame and timestamp it's handed, and its own detector /
  tracker / road model are rebuilt from scratch in `reset()`.
- **An independent audit (Opus-tier agent, full transcript available on
  request) found and we fixed four real issues** before this was the
  organizers' own harness:
  1. `src/detector.py`'s default YOLO weights path was resolved relative to
     the process's current working directory, not the repo -- silently
     picking the wrong detector if invoked from elsewhere. Fixed: resolved
     from `__file__`, and the chosen detector is now logged.
  2. `accident`/`near_miss` pairwise checks fired on *any* two overlapping
     tracks, including two "unknown"-classified MOG2 noise blobs merging --
     the single biggest false-positive source for the most heavily-weighted
     class. Fixed: pairs where either track is "unknown" are excluded, and
     `accident` now requires >=0.15s of sustained overlap (see
     `MIN_ACCIDENT_OVERLAP_SEC` in `src/rules.py`) to filter single-sample
     blob-merge blips.
  3. Several docs (this file excepted) overstated `CalibrationConfig` as
     something that "switches on automatically" when a `calibration.json`
     is dropped next to a video -- it was, and still is, an unwired schema
     with no code path consuming it. Reworded everywhere to say so plainly;
     wiring it up is listed under "Next steps" below, not claimed as done.
  4. Three perf issues that would matter on a real several-minutes 1080p
     clip: `src/rules.py`'s pairwise loop rebuilt one track's timestamp
     index for every pair instead of once; `RoadModel` rebuilt its full
     convex hull on every single sample instead of throttling; `solution.py`
     fully decoded every skipped frame instead of using `cap.grab()`. All
     three fixed.
  (Earlier audit findings about queue deadlocks and AP tie-handling were
  against the pre-official harness/evaluator described above and no longer
  apply -- the official files don't share that code.)
- **Automated validators, not just manual smoke tests** (see "Validators"
  below): 31 pytest cases plus a `tools/preflight.py` gate that runs the
  real harness twice and diffs the resulting `team`/`videos` content to check
  determinism, all currently green.

## What did not work / limitations

- **No real sample footage is in the repo yet.** `Videos.pdf` (from the
  organizers) links to 4 real camera clips on Google Drive; every download
  attempt so far has hit Google Drive's "too many users have viewed or
  downloaded this file recently" throttle (a shared, host-side rate limit
  on these specific file IDs -- not a permissions problem, and not fixed by
  retrying quickly). Until they land in `samples/`, there is: no real EDA,
  no real annotated website outputs, and **no way to tune any threshold in
  `src/rules.py` / `src/risk.py` against real ground truth.** Every
  constant is a reasoned default, not a fitted one. This is the single
  biggest gap between this baseline and a competitive score, and the
  tooling to close it the moment footage arrives already exists and is
  tested (`tools/annotate.html`, `tools/tune_thresholds.py`).
- **What we could not verify**: the harness's behavior on a real, several-
  minutes-long, 1080p clip (typical per the spec) -- everything above was
  timed against a 12s synthetic clip. The perf fixes in "What worked" #4
  above are targeted at exactly this gap, but they're reasoned, not
  measured against real footage yet.
- **Six classes have no automatic signal**: `red_light`, `stop_line`,
  `illegal_u_turn`, `illegal_turn`, `solid_line_crossing`, `fire_smoke`.
  Each needs information a camera-agnostic motion pipeline doesn't have --
  a calibrated stop line and signal-state ROI, lane-marking geometry, or
  reliable smoke color/texture cues a coarse foreground blob doesn't carry.
  `src/rules.py`'s `CalibrationConfig` is a schema for that data, not yet
  wired to anything (see "What worked" #3).
- **Vehicle/person classification is a geometry guess.** Without a real
  object detector, `MotionDetector` splits blobs into `vehicle` / `person`
  / `unknown` using only size and aspect ratio. `src/detector.py:YoloDetector`
  is a ready-to-use swap-in; it just needs `ultralytics` installed and a
  checkpoint in `weights/`.
- **2D bounding-box overlap has no depth cue.** `accident` and `near_miss`
  both key off box-overlap / proximity in image space. Two vehicles that
  are actually lanes apart but momentarily align with the camera's line of
  sight can still register as "close" or "overlapping," even after the
  unknown-blob and min-duration fixes above. A real fix needs either a
  homography to ground-plane coordinates (feasible for a fixed camera, not
  yet built) or a learned classifier over track pairs.

## Validators

- `pytest tests/` -- 31 cases: interface exactness against
  `evaluate.OFFICIAL_CLASSES`, a structural (source + AST) check that
  Part B's call graph never references `cv2.VideoCapture`, format tests run
  through the actual official `evaluate.validate()`, rule-engine unit tests
  against hand-built track histories (including regression tests for the
  two `accident` false-positive fixes above), and safety tests for the
  `tuned_params.json` override loader.
- `python tools/preflight.py --videos samples --team Genz` -- repo layout
  check, a real `run_submission.py` + `evaluate.py --validate-only` run,
  a determinism check (runs the harness twice, compares the resulting
  `team`/`videos` content -- deliberately excluding run_submission.py's own
  per-run wall-clock timing log, which legitimately varies), a weights-size
  check, and a timing-margin report.

## Next steps (priority order)

1. Get the 4 real camera samples into `samples/` (currently blocked on
   Google Drive's throttle -- see "What did not work"); hand-annotate them
   with `tools/annotate.html` into a dev `ground_truth.json` using the
   exact conventions in the challenge's "Event classes" table.
2. Run `tools/tune_thresholds.py --part a` and `--part b` against that dev
   set and commit the resulting `tuned_params.json` -- expected to move
   Score_A more than any single remaining architectural change.
3. Swap in `YoloDetector` (already wired) once `weights/yolov8n.pt` is
   downloaded; re-measure `jaywalking` / `failure_to_yield` / `accident`
   precision specifically.
4. Wire `CalibrationConfig` into `detect_events` / `build_events` and write
   the five rule functions it would drive, once a real camera calibration
   (stop line, signal ROI, lane geometry) exists to validate them against.
5. Time the harness against an actual several-minutes 1080p clip once one
   exists, to confirm the perf fixes above hold at that scale.
6. Add ground-plane homography for `accident` / `near_miss` distance
   estimates instead of raw image-space proximity.
