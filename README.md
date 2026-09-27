# Genz -- Traffic Event Detection & Accident Anticipation

A fixed-road-camera event detector (Part A) and a causal accident-risk
estimator (Part B), built as a classical detector + tracker + hand-written
rule engine -- no training data, no GPU, no external weights required to
run. See `docs/REPORT.md` for the full write-up (what worked, what didn't,
next steps) and `website/` for the team site + live local demo.

## Install & run

```bash
pip install -r requirements.txt

# Part A + Part B on a folder of test videos -> predictions.json
python run_submission.py --videos /data/test --out predictions.json

# Format check (no ground truth needed)
python evaluate.py --pred predictions.json --validate-only

# Full scoring against your own labels
python evaluate.py --pred predictions.json --gt my_labels.json
```

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
| `accident` | automatic | bounding-box overlap between two tracks |
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

The six "not implemented" classes are wired through
`src/rules.py:CalibrationConfig`, which loads an optional
`<video_name>.calibration.json` next to a video (stop line polygon, signal
ROI, no-U-turn zones, solid-line segments, crosswalks). **We have no real
sample footage from this camera yet** (see "Known limitations" below), so
there was nothing to calibrate against -- emitting guesses for these classes
without any ground truth to check them against would just add false
positives and hurt `Score_A`'s precision term for no offsetting recall
gain. This is a scoped decision, not an oversight; it's the first thing to
fix once real samples arrive (see `docs/REPORT.md`).

### Extension points

- **Real object detection**: `weights/download.sh` + `pip install
  ultralytics` turns on `YoloDetector` (`src/detector.py`) automatically --
  no other code changes. Vehicle/person classification stops being a
  size/aspect-ratio guess and becomes a real detector's class labels, which
  should materially improve `jaywalking`, `failure_to_yield`, and reduce
  `accident`/`near_miss` false positives from ambiguous blobs.
- **Calibrated classes**: drop a `<video>.calibration.json` next to a video
  (schema in `src/rules.py:CalibrationConfig`) to enable `red_light`,
  `stop_line`, `illegal_u_turn`, `illegal_turn`, `solid_line_crossing`.
- **EDA**: `python -m src.eda --samples samples --out docs/eda_output` once
  real camera footage exists in `samples/`.

## Determinism

`solution.py` seeds `random` and `numpy` at import time (seed 0). The
pipeline itself is otherwise deterministic (no sampling, no randomized
initialization) -- background subtraction and IoU tracking are fully
deterministic given the same input frames.

## Datasets & external weights

No external dataset was used for training -- there is no trained model in
this submission; every class rule is hand-written over detector/tracker
output. If `weights/yolov8n.pt` is added via `weights/download.sh`, that is
an Ultralytics-distributed open-weights YOLOv8n checkpoint (confirm current
licence terms before shipping; Ultralytics dual-licenses under AGPL-3.0 and
a commercial licence).

## Known limitations

- No labeled data from this camera was available while building this
  baseline (see the note in the challenge about no sample videos being
  present on this machine at build time). All thresholds in `src/rules.py`
  and `src/risk.py` are hand-picked defaults, not tuned against real
  ground truth. **First priority once real `samples/*.mp4` exist**:
  annotate a dev set, run `evaluate.py --gt`, and retune.
- Vehicle/person classification from blob geometry alone (no detector) is
  weak -- see "Extension points" for the YOLO upgrade path.
- `accident` / `near_miss` from 2D bounding-box overlap has no depth cue,
  so two vehicles that merely pass close to the camera's line of sight can
  register as contact. A real detector with better box tightness reduces
  this but doesn't eliminate it; see `docs/REPORT.md`.
- Six classes are intentionally unimplemented pending calibration data (see
  "Class coverage").

## Team

Team **Genz**. See the website's Team page for member roles and links.

| Member | Role | What they did |
|---|---|---|
| [Member 1 name] | [role] | [contribution] |
| [Member 2 name] | [role] | [contribution] |
| [Member 3 name] | [role] | [contribution] |

## Repository layout

```
solution.py            # required interface: detect_events(), RiskEstimator
run_submission.py       # harness: folder of videos -> predictions.json
evaluate.py             # format check + Score_A / Score_B / model score
requirements.txt
weights/                # empty by default; download.sh fetches an optional YOLO checkpoint
src/
  detector.py            # MotionDetector (default) + optional YoloDetector
  tracker.py             # greedy IoU multi-object tracker
  road_model.py          # convex-hull road mask + dominant flow direction
  rules.py               # Part A: tracks -> event segments
  risk.py                # Part B: causal accident-risk scoring
  eda.py                 # EDA over samples/*.mp4 once real footage exists
  geometry.py
examples/                # starter-kit-format ground_truth.json / predictions.json
website/                 # team site + local live demo (see website/README.md)
docs/REPORT.md           # what worked, what didn't, next steps
predictions_samples.json # output on samples/*.mp4 -- currently empty, see note below
```

`predictions_samples.json` is currently `{"team": "Genz", "videos": {}}`
because no real sample videos were available while building this baseline.
Regenerate it with `python run_submission.py --videos samples --out
predictions_samples.json` the moment real samples land in `samples/`.

Before submitting: `python evaluate.py --pred predictions.json --validate-only`.
