# Handoff brief -- for an independent reviewer

This is written for a second reviewer (human or AI, e.g. Codex) doing a
fresh, independent pass on this repo -- not a rehash of `README.md` /
`docs/REPORT.md`, which explain the approach. This document tells you
**what to verify and why**, what's already been checked, and where the
real gaps are, so a second pass adds coverage instead of re-deriving
context from zero.

## What this is

A submission for the WIUT Hackathon 2026 Computer Vision track ("Elimination
Task"): detect traffic events in fixed-CCTV road footage (Part A, mandatory)
and anticipate accidents from a causal per-frame risk score (Part B, bonus).
Full spec: `WIUT Hackathon _ CV Track Elimination Task.pdf` (given to this
team by the organizers; not included in this repo, but every rule it states
is either implemented, or explicitly listed below as not implemented and
why).

## Provenance (why you can trust the interface files)

- `run_submission.py` and `evaluate.py` are the **exact, unmodified**
  contents of `wiut_cv_scripts.zip`, the organizers' own starter kit.
  Verify: `diff` them against a fresh copy of that zip if you have it; they
  should be byte-identical. Everything else in the repo (`solution.py`,
  `src/`, `tools/`, `tests/`, `website/`) is ours.
- `examples/ground_truth.json` and `examples/predictions.json` are also the
  organizers' own example files, unmodified.
- An earlier iteration of this repo (before the real starter kit was
  supplied) shipped a hand-written stand-in for both harness files, written
  directly from the spec's prose. It has been fully replaced. If you see
  any reference to "re-derived from spec text" anywhere, it's stale
  documentation describing that earlier state -- flag it, we may have
  missed one.

## Architecture, with exact pointers

```
Part A (offline, whole video available)
  solution.detect_events()                                    [solution.py:59]
    -> src.detector.build_default_detector()                  [src/detector.py:157]
         MotionDetector (MOG2 + contour classify)              [src/detector.py:40]
         or YoloDetector if weights/yolov8n.pt exists          [src/detector.py:106]
    -> src.tracker.IouTracker                                  [src/tracker.py:74]
    -> src.rules.build_events()                                [src/rules.py:316]
         src.road_model.RoadModel (hull + dominant flow)       [src/road_model.py:28]
         per-class rule functions                               [src/rules.py, see table below]

Part B (causal, one frame at a time, never the future)
  solution.RiskEstimator (class)                               [solution.py:105]
    .step()                                                    [solution.py:114]
    -> src.risk.CausalRiskEngine.update()                      [src/risk.py:63]
         same detector/tracker/road-model classes as Part A,
         but instantiated fresh per video in reset()            [src/risk.py:55]
         3 signals folded with max() + backward-only EMA:
           time-to-collision (_min_ttc_risk)                    [src/risk.py:103]
           hard braking (_hard_brake_risk)                       [src/risk.py:118]
           pedestrian near a vehicle (_pedestrian_risk)          [src/risk.py:132]
```

(Line numbers verified with `grep -n` at commit time; re-grep rather than
trust them blindly if this file has moved on since.)

### Class coverage (8 automatic, 6 not implemented)

| Class | Implemented? | Rule function |
|---|---|---|
| `stopped_vehicle` | yes | `_stopped_vehicle_events` (`src/rules.py`) |
| `congestion` | yes | `_congestion_events` |
| `wrong_way` | yes | `_wrong_way_events` |
| `accident` | yes | `_pairwise_events` (accident branch) |
| `near_miss` | yes | `_pairwise_events` (near-miss branch) |
| `road_obstacle` | yes | `_road_obstacle_events` |
| `jaywalking` | yes (coarse) | `_jaywalking_events` |
| `failure_to_yield` | yes (coarse) | `_pairwise_events` (overlap + person) |
| `red_light`, `stop_line`, `illegal_u_turn`, `illegal_turn`, `solid_line_crossing`, `fire_smoke` | **no** | none -- `CalibrationConfig` (`src/rules.py`) is a schema for the calibration data these would need, currently unwired to any rule |

## What's already been verified this session (don't re-derive, extend instead)

An independent Opus-tier audit agent was run against an earlier state of
this repo (before the official starter kit was swapped in) with the exact
spec text as its brief. Its full report is preserved in this session's
transcript; summarized here so a second reviewer doesn't repeat the same
ground:

**Checked and found clean** (against the *old* custom evaluate.py/harness,
re-verify against the *current* official ones -- see "Suggested focus"
below): interface exactness, causality (no video I/O reachable from
`RiskEstimator.step`), same-class overlap drop semantics ("earlier-starting
segment kept," including chained/nested/touching cases), `predictions.json`
schema on disk, greedy IoU matching correctness (hand-derived 3x3 case),
Part B frame-labeling boundary conditions (exact `<` vs `<=` at every
boundary), AP chance-normalization formula and its edge cases, alarm
construction/merging (strict `<2s` gap, measured from previous run's end),
alarm-to-accident matching (proved the "process alarms chronologically,
each claims the earliest eligible unmatched accident" implementation is
provably equivalent to "for each accident, its earliest eligible alarm"
for this problem's structure -- see the transcript for the proof sketch if
you want to re-derive it), mTTA including unmatched accidents as 0,
determinism (no unseeded randomness, no order-dependent set/dict iteration
that affects output).

**Found and fixed** (all against files still in the current repo):
1. `src/detector.py`'s YOLO weights path resolved against cwd, not the repo
   root -- fixed, now resolved from `__file__`, and the chosen detector is
   logged (`build_default_detector`).
2. `src/rules.py`'s pairwise `accident`/`near_miss`/`failure_to_yield` logic
   considered every pair of tracks regardless of label, so two
   "unknown"-classified MOG2 noise blobs merging could manufacture a false
   `accident`. Fixed: pairs are now filtered to exclude "unknown"-labelled
   tracks before the pairwise loop, and `accident` requires
   `MIN_ACCIDENT_OVERLAP_SEC` (0.15s) of sustained overlap. Regression
   tests: `tests/test_rules_synthetic.py::test_accident_does_not_fire_on_unknown_blob_overlap`
   and `::test_accident_does_not_fire_on_single_frame_overlap_blip`.
3. Several docs described `CalibrationConfig` as something that "switches
   on automatically" when it's an unwired schema. Reworded in `README.md`,
   `src/rules.py`'s docstring, and `docs/explainer.html`. Also fixed the
   loader (`CalibrationConfig.load`) to ignore unknown JSON keys, since the
   shipped `examples/calibration.example.json` has a `_comment` key that
   previously crashed it.
4. Three perf issues that only bite at real video scale: `src/rules.py`'s
   pairwise loop rebuilt a per-track timestamp index for every pair instead
   of once (now hoisted); `src/road_model.py`'s `RoadModel` rebuilt its
   full convex hull on every single sample instead of throttling (now
   throttled to every `REBUILD_EVERY_N_SAMPLES=8`); `solution.py` fully
   decoded every skipped frame in `detect_events` instead of using
   `cap.grab()` (now uses `grab()` for skipped frames).

The same audit also found a critical bug (a `multiprocessing.Queue`
deadlock) and a metric bug (AP tie-handling) in the *old, custom*
`run_submission.py`/`evaluate.py` -- both are moot now that those files
were replaced with the organizers' own, which don't share that code (this
was independently confirmed by reading the official files: they don't use
multiprocessing at all, and the official `average_precision()` explicitly
groups tied scores -- "take the whole tie group" in its own comment).

## Suggested focus for a second pass

Things the first audit either couldn't check (wrong harness at the time)
or didn't check (out of its original scope):

1. **Re-verify the official `evaluate.py`/`run_submission.py` directly**,
   now that they're the real files -- the first audit's Part A/B algorithm
   checks were against a different (custom) implementation of the same
   spec. We separately smoke-tested the official files against
   `examples/` and got Score_A=0.9074/Score_B=0.76/M=0.8632, matching hand
   expectations, but a fresh independent check costs little and catches
   what we might have missed.
2. **Timing at real scale.** Everything timed so far used a 12-20s
   synthetic clip. The spec says "typical clips are several minutes long
   at 25 fps." Nobody has yet run this against footage that long. If you
   get real samples before we do, this is the single most valuable thing
   to check.
3. **The accident/near_miss precision fix's actual effect.** We reasoned
   through why excluding "unknown"-label pairs and requiring 0.15s overlap
   should reduce false positives, and wrote regression tests for the exact
   scenarios described, but have no real footage to measure the net effect
   on Score_A.
4. **Anything in `src/risk.py`'s three signals we haven't stress-tested**:
   the hard-braking and pedestrian-proximity signals have unit-level
   sanity (they don't crash, they return values in range) but no dedicated
   correctness test the way `src/rules.py`'s synthetic tests do for Part A.

## How to run the validators

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/                                    # 31 cases, should be all green
python tools/preflight.py --videos samples --team Genz   # needs at least one .mp4 in samples/
```

`preflight.py` will refuse to run (exit code 2) if `samples/` is empty --
that's expected right now (see "Known gaps" below), drop any .mp4 in there
to exercise it, real footage not required for the check itself to work.

## Known gaps, explicitly

- **No real sample footage in `samples/` yet.** `Videos.pdf` links to 4
  real clips on Google Drive; every download attempt has hit Google
  Drive's own "too many users have viewed or downloaded this file
  recently" throttle. Not a code or permissions problem -- retry later, or
  download manually via a browser (the links work fine there) and drop the
  files into `samples/` as `sample_001.mp4`..`sample_004.mp4`.
- **No thresholds have been tuned against real data.** Every constant in
  `src/rules.py` / `src/risk.py` is a reasoned default. `tools/tune_thresholds.py`
  exists and is tested (via `--self-test`, which verifies the search loop
  runs without producing a real result), but has never been run against
  real annotations because none exist yet.
- **Six classes are unimplemented** (see table above) -- not hidden behind
  a broken feature flag, genuinely not built.
- **Team info in `README.md` and `website/index.html` is placeholder** --
  the user explicitly chose placeholders over guesses.
- **Public hosting**: `render.yaml` + `website/Dockerfile` are ready for a
  one-click Render Blueprint deploy (see `website/README.md` "Public
  hosting"). Chosen over Hugging Face Spaces because HF now gates its
  Docker/Gradio SDKs behind payment-method verification on newer accounts,
  even for the free CPU tier; Render's free web services don't require
  that. As of this writing the actual deploy step (connecting Render to
  the GitHub repo) is the user's to do in their own browser -- check
  whether `website/index.html`'s Links section still has a live URL filled
  in, or ask.
