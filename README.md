# Genz -- Traffic Event Detection & Accident Anticipation

A fixed-road-camera event detector (Part A) and a causal accident-risk
estimator (Part B), built as a classical detector + tracker + hand-written
rule engine -- no training data, no GPU, no external weights required to
run. `run_submission.py` and `evaluate.py` in this repo are the **exact,
unmodified files from the organizers' `wiut_cv_scripts` starter kit** --
only `solution.py` and everything under `src/` is ours. See
`docs/REPORT.md` for the full write-up (what worked, what didn't, next
steps), `HANDOFF.md` for an independent-review brief, and `website/` for
the team site + live demo (publicly hosted at https://genzwiut.com).

## Install & run

```bash
pip install -r requirements.txt

# Part A + Part B on a folder of test videos -> predictions.json
python run_submission.py --videos /data/test --out predictions.json --team Genz

# Format check (no ground truth needed)
python evaluate.py --pred predictions.json --validate-only

# Full scoring against your own labels
python evaluate.py --pred predictions.json --gt my_labels.json --per-video
```

`run_submission.py` also writes a per-video `"log"` block (duration,
timing, dropped-event warnings) into `predictions.json` alongside `"team"`
and `"videos"` -- `evaluate.py` ignores it, it's there for local debugging.

No weights are required for the default pipeline. `weights/download.sh` is
an **optional** step (run once, with internet, before an offline evaluation
run) that fetches a small YOLOv8n checkpoint if you want to swap in real
object detection -- see `weights/README.md` and "Extension points" below.

## Approach

```
Part A (offline, per video)
  video.mp4 -> [sampled frames] -> Detector -> IoU Tracker -> Road Model -> Rule Engine -> events[]

Part B (causal, per frame, never sees the future)
  frame_t -> Detector -> IoU Tracker -> Road Model (running) -> TTC / braking / pedestrian signals -> risk score
```

- **Detector** (`src/detector.py`): `MotionDetector` by default -- OpenCV
  MOG2 background subtraction + contour filtering, classifying each blob
  into `vehicle` / `person` / `unknown` from size and aspect ratio. Zero
  training data, zero GPU, and a reasonable fit for a genuinely static
  camera (which this challenge guarantees: "one angle, no camera motion").
  `YoloDetector` is a drop-in upgrade (see Extension points) that activates
  automatically if `ultralytics` + a weights file are present.
- **Tracker** (`src/tracker.py`): a minimal greedy-IoU multi-object tracker
  (the association half of SORT, without the Kalman filter -- overkill for
  a fixed camera). Keeps each track's full `(t, cx, cy, w, h)` history.
- **Road model** (`src/road_model.py`): a convex hull over every observed
  vehicle position (an implicit "where the road is" mask, no manual ROI
  needed) plus a circular-mean dominant traffic-flow direction from vehicle
  velocity samples. Used for `wrong_way` and `jaywalking`. Built offline
  from the whole video for Part A, and incrementally, frame by frame, for
  Part B (so Part B's road model never contains future information).
- **Rule engine** (`src/rules.py`): turns finished tracks into event
  segments. See "Class coverage" below for exactly which classes have an
  automatic signal today.
- **Causal risk engine** (`src/risk.py`): time-to-collision between
  converging vehicle tracks (`TTC = distance / closing_speed`, calibrated
  so `TTC == H == 5s` maps to a score of 0.5, per the challenge's own
  tip), combined with a hard-braking signal and a pedestrian-near-vehicle
  signal via `max()`, then smoothed with a backward-only EMA so the score
  doesn't flicker frame to frame.

**Everything here is rule-based; nothing is learned.** The only optional
learned component is the YOLO detector swap-in, which improves detection
quality but doesn't change any rule logic.

### Class coverage

| Class | Status | Signal |
|---|---|---|
| `stopped_vehicle` | automatic | speed <= threshold for >= 10s |
| `congestion` | automatic | >=3 vehicle tracks, >=70% near-stationary, sustained >= 8s |
| `wrong_way` | automatic | heading >=135 deg vs. dominant flow, sustained |
| `accident` | automatic | bounding-box overlap between two tracks (excludes unclassified-blob pairs; requires >=0.15s sustained overlap -- see "Known limitations") |
| `near_miss` | automatic | close approach + high closing speed, no overlap, then divergence |
| `road_obstacle` | automatic | static "unknown"-class blob on the road, no motion, >= 10s |
| `jaywalking` | automatic (coarse) | "person"-classified blob inside the road-model polygon |
| `failure_to_yield` | automatic (coarse) | vehicle-person bounding-box overlap |
| `red_light` | **not implemented** | needs a calibrated stop-line + signal-state ROI |
| `stop_line` | **not implemented** | same as above |
| `illegal_u_turn` | **not implemented** | needs calibrated no-U-turn zones |
| `illegal_turn` | **not implemented** | needs calibrated lane/turn geometry |
| `solid_line_crossing` | **not implemented** | needs calibrated lane-marking geometry |
| `fire_smoke` | **not implemented** | motion blobs alone don't carry reliable color/texture smoke cues |

`src/rules.py:CalibrationConfig` defines a schema for what per-camera
calibration (stop line polygon, signal ROI, no-U-turn zones, solid-line
segments, crosswalks) would look like for these six classes -- **it is
currently an unwired, unused extension point**: nothing in `detect_events`
constructs one or reads its fields yet. Dropping a `calibration.json` next
to a video today changes nothing. Wiring it in (loading it, passing it
through to `build_events`, writing the six rule functions that would
consume it) is real, not-yet-started work -- emitting guesses for these
classes without any ground truth to check them against would just add
false positives and hurt `Score_A`'s precision term for no offsetting
recall gain, so it's deliberately left undone rather than half-faked. This
is the first thing to build once real samples arrive (see
`docs/REPORT.md`).

### Extension points

- **Real object detection**: `weights/download.sh` + `pip install
  ultralytics` turns on `YoloDetector` (`src/detector.py`) automatically --
  no other code changes. Vehicle/person classification stops being a
  size/aspect-ratio guess and becomes a real detector's class labels, which
  should materially improve `jaywalking`, `failure_to_yield`, and reduce
  `accident`/`near_miss` false positives from ambiguous blobs.
- **Calibrated classes**: `src/rules.py:CalibrationConfig` is a schema for
  the stop-line/signal-ROI/lane geometry that `red_light`, `stop_line`,
  `illegal_u_turn`, `illegal_turn`, `solid_line_crossing` would need --
  wiring it into `detect_events`/`build_events` and writing those five rule
  functions is not-yet-started work, not a flip-a-switch config file.
- **EDA**: `python -m src.eda --samples samples --out docs/eda_output` once
  real camera footage exists in `samples/`.
- **Annotate a dev set**: open `tools/annotate.html` directly in a browser
  (no server needed) to hand-label real samples into a `ground_truth.json`
  in the exact spec format.
- **Tune thresholds**: `python -m tools.tune_thresholds --part a --samples
  samples --gt dev_ground_truth.json` (and `--part b`) grid-searches
  `src/rules.py` / `src/risk.py`'s constants against a real dev set using
  `evaluate.py`'s own scoring functions, and writes `tuned_params.json`
  (picked up automatically by both modules, see "Determinism").

## Determinism

`solution.py` seeds `random` and `numpy` at import time (seed 0). The
pipeline itself is otherwise deterministic (no sampling, no randomized
initialization) -- background subtraction and IoU tracking are fully
deterministic given the same input frames.

`src/rules.py` and `src/risk.py` each load an optional `tuned_params.json`
at import time (see `src/tunable.py`), which can override their hand-picked
threshold constants with values found by `tools/tune_thresholds.py` against
a real dev set. This is external state, but it's committed to the repo
alongside the code once tuning is done, so a run against the same commit is
still fully reproducible -- and `H_SEC` (the spec-fixed 5s horizon) is
denylisted in code (`src/risk.py`'s `apply_overrides(..., deny=...)` call)
so it can never be silently changed by that file, even by mistake. No
`tuned_params.json` ships in the repo yet -- see "Known limitations."

## Datasets & external weights

No external dataset was used for training -- there is no trained model in
this submission; every class rule is hand-written over detector/tracker
output. If `weights/yolov8n.pt` is added via `weights/download.sh`, that is
an Ultralytics-distributed open-weights YOLOv8n checkpoint (confirm current
licence terms before shipping; Ultralytics dual-licenses under AGPL-3.0 and
a commercial licence).

## Known limitations

- All thresholds in `src/rules.py` and `src/risk.py` are hand-picked
  defaults, not yet tuned against real ground truth (dev-set annotation and
  tuning is in progress -- see `docs/REPORT.md` and `HANDOFF.md` for
  current status).
- Vehicle/person classification from blob geometry alone (no detector) is
  weak -- see "Extension points" for the YOLO upgrade path.
- `accident` / `near_miss` from 2D bounding-box overlap has no depth cue,
  so two vehicles that merely pass close to the camera's line of sight can
  register as contact. Pairwise checks now exclude unclassified-blob pairs
  and require a minimum sustained overlap (0.15s) to cut MOG2 noise-blob
  false positives, but the underlying 2D-only limitation remains; see
  `docs/REPORT.md`.
- Six classes are intentionally unimplemented pending calibration
  infrastructure that doesn't exist yet -- not gated behind a config file,
  genuinely not built (see "Class coverage" and "Extension points").

## Team

Team **Genz**. See the website's Team page for member roles and links.

| Member | Role | What they did |
|---|---|---|
| [Member 1 name] | [role] | [contribution] |
| [Member 2 name] | [role] | [contribution] |
| [Member 3 name] | [role] | [contribution] |

## Repository layout

```
solution.py             # required interface: detect_events(), RiskEstimator
run_submission.py        # organizers' harness, unmodified
evaluate.py              # organizers' metric, unmodified
requirements.txt          # harness deps (numpy, opencv-python-headless)
requirements-dev.txt      # test-only deps (pytest); not needed to run the submission
weights/                 # empty by default; download.sh fetches an optional YOLO checkpoint
src/
  detector.py             # MotionDetector (default) + optional YoloDetector
  tracker.py              # greedy IoU multi-object tracker
  road_model.py           # convex-hull road mask + dominant flow direction
  rules.py                # Part A: tracks -> event segments
  risk.py                 # Part B: causal accident-risk scoring
  tunable.py              # loads tuned_params.json overrides, if present
  eda.py                  # EDA over samples/*.mp4 once real footage exists
  geometry.py
tools/
  annotate.html           # browser-only tool to hand-label samples/ into a dev ground_truth.json
  tune_thresholds.py      # grid-search src/rules.py + src/risk.py against a real dev set
  preflight.py            # local pre-submission gate (layout, harness, determinism, timing)
tests/                   # pytest: interface, causality, format, rule-engine, tunable-loader
examples/                # organizers' ground_truth.json / predictions.json (unmodified)
website/                 # team site + live demo (Dockerfile included, see website/README.md)
render.yaml              # alternative Docker-based deploy path for the demo (not part of grading; genzwiut.com is the live one)
docs/REPORT.md           # what worked, what didn't, next steps
docs/explainer.html       # published interactive architecture writeup
HANDOFF.md               # independent-review brief (for a second AI/human reviewer)
predictions_samples.json # output on samples/*.mp4 -- currently empty, see note below
```

`predictions_samples.json` is currently `{"team": "Genz", "videos": {}}`
because no real sample videos are in `samples/` yet (see "Known
limitations" -- download blocked on Google Drive's throttle, not a code
issue). Regenerate it with `python run_submission.py --videos samples --out
predictions_samples.json --team Genz` the moment real samples land there.

Before submitting: `python evaluate.py --pred predictions.json
--validate-only`, `pytest tests/`, and `python tools/preflight.py --videos
samples --team Genz`.
