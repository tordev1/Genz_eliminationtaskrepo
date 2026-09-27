"""Grid-search src/rules.py and src/risk.py's constants against a real,
hand-annotated dev set, using evaluate.py's own scoring functions -- so
"best params" here means "best Score_A / Score_B", not a proxy metric.

This is infrastructure, not a report: it does nothing useful without a real
`dev_ground_truth.json` (same schema as the challenge's ground_truth.json)
annotated from real camera footage. Producing that dev set is what
tools/annotate.html is for. Running this tool against fabricated or
non-camera footage would just curve-fit noise -- don't do that; the
"--self-test" mode below exists only to prove the tool's plumbing works, and
prints a loud warning instead of a score when used that way.

Usage (once you have real dev_ground_truth.json + videos in samples/):

    python -m tools.tune_thresholds --part a \
        --samples samples --gt dev_ground_truth.json --out tuned_params.json

    python -m tools.tune_thresholds --part b \
        --samples samples --gt dev_ground_truth.json --out tuned_params.json

Part A is fast: the detector+tracker pass is run once per video and cached,
then every candidate threshold combination re-scores the *same* tracks
(build_events() is pure arithmetic over track histories -- no video I/O), so
a few hundred combinations take seconds, not minutes.

Part B re-decodes and re-runs the causal detector+tracker for every
candidate combination (its constants change what the detector/tracker feed
into, frame by frame, not just a post-hoc rule), so it's slower -- keep the
grid small (this is a documented limitation, see README "Extension points").
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2  # noqa: E402

import evaluate as ev  # noqa: E402
import solution  # noqa: E402
from src import rules as rules_mod  # noqa: E402
from src import risk as risk_mod  # noqa: E402
from src.detector import build_default_detector  # noqa: E402
from src.rules import build_events  # noqa: E402
from src.tracker import IouTracker  # noqa: E402

# Search space. Kept intentionally narrow around the hand-picked defaults in
# src/rules.py -- widen it once you've seen how a real dev set responds.
RULES_GRID = {
    "MIN_STOPPED_SEC": [7.0, 10.0, 13.0],
    "STATIONARY_SPEED_PX_S": [4.0, 6.0, 9.0],
    "WRONG_WAY_ANGLE_DEG": [120.0, 135.0, 150.0],
    "WRONG_WAY_MIN_SEC": [1.0, 1.5, 2.5],
    "NEAR_MISS_CLOSE_FACTOR": [1.1, 1.4, 1.8],
    "NEAR_MISS_MIN_CLOSING_SPEED": [25.0, 40.0, 60.0],
    "CONGESTION_STATIONARY_FRACTION": [0.6, 0.7, 0.85],
    "CONGESTION_MIN_SEC": [5.0, 8.0, 12.0],
    "MERGE_GAP_SEC": [0.4, 0.75, 1.2],
}

RISK_GRID = {
    "CLOSE_FACTOR": [2.0, 3.0, 4.0],
    "BRAKE_DECEL_REF": [150.0, 250.0, 400.0],
    "EMA_ALPHA": [0.3, 0.5, 0.7],
}


def _load_gt(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_track_cache(video_names, samples_dir):
    """Runs the detector+tracker pass once per video. Reused across every
    Part A parameter combination -- see module docstring.
    """
    cache = {}
    for name in video_names:
        path = os.path.join(samples_dir, name)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"  [warn] could not open {path}, skipping")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = n_frames / fps if fps > 0 else 0.0
        stride = solution._analysis_stride(fps)

        detector = build_default_detector()
        tracker = IouTracker(iou_threshold=0.2, max_missed=int(round(2.0 * fps / stride)))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                dets = detector.detect(frame)
                tracker.update(dets, idx / fps)
            idx += 1
        cap.release()
        cache[name] = (tracker.tracks, (height, width), duration)
        print(f"  cached tracks for {name}: {len(tracker.tracks)} track(s), {duration:.1f}s")
    return cache


def _score_a_with(params, track_cache, gt):
    for key, value in params.items():
        setattr(rules_mod, key, value)
    pred_videos = {}
    for name, (tracks, shape, duration) in track_cache.items():
        pred_videos[name] = {"events": build_events(tracks, shape, duration), "risk": []}
    result = ev.evaluate_part_a(gt, pred_videos)
    return result["score_a"], result["per_class"]


def tune_part_a(samples_dir, gt, out_path):
    video_names = [v for v in gt.keys() if os.path.isfile(os.path.join(samples_dir, v))]
    missing = set(gt.keys()) - set(video_names)
    if missing:
        print(f"  [warn] in ground truth but not found under {samples_dir}: {sorted(missing)}")
    if not video_names:
        print("No ground-truth videos found under", samples_dir, "-- nothing to tune.")
        return

    print(f"Building track cache for {len(video_names)} video(s)...")
    track_cache = _build_track_cache(video_names, samples_dir)

    keys = list(RULES_GRID.keys())
    combos = list(itertools.product(*RULES_GRID.values()))
    print(f"Searching {len(combos)} combinations over {keys}...")

    best_score, best_params = -1.0, None
    for combo in combos:
        params = dict(zip(keys, combo))
        score, _ = _score_a_with(params, track_cache, gt)
        if score > best_score:
            best_score, best_params = score, params

    print(f"\nBest Score_A = {best_score:.4f}")
    print(json.dumps(best_params, indent=2))
    _write_out(out_path, "rules", best_params)


def _score_b_with(params, video_names, samples_dir, gt):
    for key, value in params.items():
        setattr(risk_mod, key, value)
    pred_videos = {}
    for name in video_names:
        path = os.path.join(samples_dir, name)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        estimator = solution.RiskEstimator()
        estimator.reset({
            "video_id": name, "fps": fps,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        })
        risk = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_sec = idx / fps
            risk.append([round(t_sec, 3), round(float(estimator.step(frame, t_sec)), 4)])
            idx += 1
        cap.release()
        pred_videos[name] = {"events": [], "risk": risk}
    result = ev.evaluate_part_b(gt, pred_videos)
    if result is None:
        return 0.0, {"note": "no accident events in gt -- nothing for Part B to score"}
    return result["score_b"], result


def tune_part_b(samples_dir, gt, out_path):
    video_names = [v for v in gt.keys() if os.path.isfile(os.path.join(samples_dir, v))]
    if not video_names:
        print("No ground-truth videos found under", samples_dir, "-- nothing to tune.")
        return

    keys = list(RISK_GRID.keys())
    combos = list(itertools.product(*RISK_GRID.values()))
    print(f"Searching {len(combos)} combinations over {keys} "
          f"(re-runs the causal pipeline on {len(video_names)} video(s) each time)...")

    best_score, best_params, best_details = -1.0, None, None
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        score, details = _score_b_with(params, video_names, samples_dir, gt)
        print(f"  [{i + 1}/{len(combos)}] {params} -> Score_B={score:.4f}")
        if score > best_score:
            best_score, best_params, best_details = score, params, details

    print(f"\nBest Score_B = {best_score:.4f}")
    print(json.dumps(best_params, indent=2))
    print(json.dumps(best_details, indent=2))
    _write_out(out_path, "risk", best_params)


def _write_out(out_path, group, params):
    data = {}
    if os.path.isfile(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data[group] = params
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nWrote {out_path} ({group}). src/{group}.py picks this up automatically on next import.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["a", "b"], required=True)
    parser.add_argument("--samples", default="samples")
    parser.add_argument("--gt", required=True, help="Path to a real, hand-annotated dev ground_truth.json")
    parser.add_argument("--out", default="tuned_params.json")
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run against whatever is in --gt/--samples even if it isn't real annotated camera "
             "footage, only to verify the search loop itself doesn't crash. The resulting "
             "tuned_params.json is meaningless and this flag refuses to write it.",
    )
    args = parser.parse_args()

    gt = _load_gt(args.gt)
    out_path = os.devnull if args.self_test else args.out
    if args.self_test:
        print("*** --self-test: verifying the tuning loop runs end-to-end. ***")
        print("*** The resulting numbers are NOT a real tuning result and will not be saved. ***\n")

    if args.part == "a":
        tune_part_a(args.samples, gt, out_path)
    else:
        tune_part_b(args.samples, gt, out_path)


if __name__ == "__main__":
    main()
