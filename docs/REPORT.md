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

## What we found once real footage arrived (late in the timeline, ~2h before deadline)

The 4 real camera clips eventually downloaded (2-6GB each, 4K, ~30fps,
127-340s -- a fixed elevated view of a busy multi-lane signalized
intersection with a crosswalk). Running the actual pipeline against them
for the first time surfaced problems no synthetic test caught:

- **Performance was the single most serious finding.** `MotionDetector`
  ran background subtraction at native 4K: 733ms/frame, which alone
  (before tracking or rules) exceeded the 3x-duration time budget --
  every video would have scored as empty. Fixed by always downscaling
  internally to a fixed 960px processing width before detection and
  scaling boxes back up (calibrated thresholds now apply consistently
  across any input resolution, not just faster): 42ms/frame, a **17x**
  speedup. A 127.6s real video went from 1887.6s (4.9x over its 382.9s
  budget) to 96.8s (well within it).
- **`wrong_way` was structurally broken on a real divided road.** It
  compared every vehicle against one scene-wide average direction; a
  two-way avenue's legal opposing-carriageway traffic averages into an
  unstable direction and gets flagged. Replaced with a spatial-grid local
  direction model (`src/road_model.py`) -- a vehicle is now compared
  against nearby same-lane traffic. Two regression tests added.
- **`accident`/`failure_to_yield`/`near_miss`/`jaywalking` all had real
  false-positive modes**, confirmed empirically (one 127.6s real video:
  49 spurious events before fixes, 26 after): a `jaywalking` event
  spanning the entire video (RoadModel's on-road hull is broader than the
  carriageway; capped and gated on actual displacement), `near_miss`
  events lasting up to 54s (was reporting the whole proximity window
  instead of the evasive episode; now ends shortly after the last
  high-closing-speed sample), and `accident`/`failure_to_yield` firing on
  ordinary adjacent-lane visual overlap from the elevated camera angle
  (IoU threshold raised from 0.02 to 0.1, a reasoned bump, not
  empirically grid-searched -- see below).
- **Not fully solved**: given the ~2 hour window between real footage
  landing and the submission deadline, there was no time to run
  `tools/tune_thresholds.py`'s full grid search against a real annotated
  dev set, or to build proper crosswalk/carriageway-aware exclusion for
  `jaywalking`. The false-positive rate above is reduced, not eliminated
  -- an honest, measured state, not a claimed fix. This is the top item
  in "Next steps."
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

- `pytest tests/` -- 32 cases: interface exactness against
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

1. Hand-annotate the 4 real samples (now in `samples/`) with
   `tools/annotate.html` into a dev `ground_truth.json`, then run
   `tools/tune_thresholds.py --part a` and `--part b` against it and commit
   the resulting `tuned_params.json` -- the false-positive rate documented
   above (26 events on one real video with zero true events, per manual
   frame-sampled review) is the clearest remaining gap, and this is a
   grid search away from being data-driven instead of hand-picked, once
   there's time to run it.
2. Build proper crosswalk/carriageway-aware exclusion for `jaywalking`
   (RoadModel's on-road hull is currently broader than the actual
   carriageway) instead of the current duration-cap stopgap.
3. Swap in `YoloDetector` (already wired) once `weights/yolov8n.pt` is
   downloaded; re-measure `jaywalking` / `failure_to_yield` / `accident`
   precision specifically -- real object-class labels instead of
   size/aspect-ratio guessing should directly address several of the
   false-positive modes found above.
4. Wire `CalibrationConfig` into `detect_events` / `build_events` and write
   the five rule functions it would drive, once a real camera calibration
   (stop line, signal ROI, lane geometry) exists to validate them against.
5. Time the harness against an actual several-minutes 1080p clip once one
   exists, to confirm the perf fixes above hold at that scale.
6. Add ground-plane homography for `accident` / `near_miss` distance
   estimates instead of raw image-space proximity.
