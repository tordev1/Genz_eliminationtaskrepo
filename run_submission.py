#!/usr/bin/env python3
"""
run_submission.py — the organizers' harness. Do not modify.

Runs `solution.py` over a folder of .mp4 files and writes predictions.json:

    python run_submission.py --videos /data/test --out predictions.json --team my-team

For every video it
  1. calls solution.detect_events(video_path)                      (Part A)
  2. streams every frame through solution.RiskEstimator            (Part B)
  3. enforces the time budget: TIME_FACTOR x video duration
     for Part A + Part B together (over budget -> video scored as empty)
  4. catches exceptions -> that video scored as empty, error logged
  5. drops events that would fail evaluate.py's format check (bad label,
     bad times, same-class overlap) and lists each drop in the log.

Options for local development only (the official run uses the defaults):
  --solution PATH     import a different solution module (e.g. examples/solution_demo.py)
  --no-risk           skip Part B
  --risk-stride N     call step() only every N-th frame (official: 1)
  --time-factor F     time budget multiplier (official: 3.0)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".MP4"}
TIME_FACTOR_DEFAULT = 3.0


def load_solution(path: Path):
    spec = importlib.util.spec_from_file_location("solution", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["solution"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    for name in ("CLASSES", "detect_events", "RiskEstimator"):
        if not hasattr(module, name):
            raise AttributeError(f"{path} must define `{name}`")
    return module


def video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "video_id": path.name,
        "fps": float(fps),
        "width": width,
        "height": height,
        "n_frames": n_frames,
        "duration": n_frames / float(fps) if fps else 0.0,
    }


def clean_events(events, classes: list[str], duration: float) -> tuple[list, list[str]]:
    """Coerce types and DROP events that would fail evaluate.py's format check.

    Dropped: not [start, end, label]; label outside the official list; times
    outside 0 <= start < end <= duration; a same-class segment overlapping an
    earlier-starting one. Every drop is reported in the log so you can fix
    your solution, but the written predictions.json is always valid.
    """
    out, problems = [], []
    if events is None:
        return [], ["detect_events returned None (treated as [])"]
    for i, ev in enumerate(events):
        try:
            s, e, label = ev
            s, e, label = float(s), float(e), str(label)
        except Exception:
            problems.append(f"event {i} dropped: not [start, end, label]: {ev!r}")
            continue
        if label not in classes:
            problems.append(f"event {i} dropped: label {label!r} not in the official class list")
            continue
        e = min(e, duration)
        if not (0.0 <= s < e):
            problems.append(f"event {i} dropped: bad times [{s}, {e}] for duration {duration:.2f}")
            continue
        out.append([round(s, 3), round(e, 3), label])
    # same-class overlaps: keep the earlier-starting segment
    out.sort(key=lambda x: (x[2], x[0]))
    kept, last_end = [], {}
    for s, e, label in out:
        if s < last_end.get(label, -1.0):
            problems.append(f"dropped {label} [{s}, {e}]: overlaps an earlier {label} segment")
            continue
        kept.append([s, e, label])
        last_end[label] = e
    kept.sort()
    return kept, problems


def run_risk(estimator, path: Path, meta: dict, stride: int, deadline: float) -> tuple[list, bool]:
    """Stream frames through the estimator. Returns (risk_curve, timed_out)."""
    cap = cv2.VideoCapture(str(path))
    fps = meta["fps"]
    curve, idx, last = [], 0, 0.0
    estimator.reset({k: meta[k] for k in ("video_id", "fps", "width", "height", "n_frames")})
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if idx % stride == 0:
            score = estimator.step(frame, t)
            try:
                last = min(1.0, max(0.0, float(score)))
            except Exception:
                last = 0.0
        curve.append([round(t, 4), round(last, 4)])
        idx += 1
        if idx % 100 == 0 and time.perf_counter() > deadline:
            cap.release()
            return curve, True
    cap.release()
    return curve, False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="folder with .mp4 files, or a single .mp4")
    ap.add_argument("--out", default="predictions.json")
    ap.add_argument("--team", default="unnamed-team")
    ap.add_argument("--solution", default="solution.py")
    ap.add_argument("--no-risk", action="store_true")
    ap.add_argument("--risk-stride", type=int, default=1)
    ap.add_argument("--time-factor", type=float, default=TIME_FACTOR_DEFAULT)
    args = ap.parse_args()

    sol = load_solution(Path(args.solution))
    classes = list(sol.CLASSES)
    try:  # the official list lives in evaluate.py; a solution may only shrink it
        from evaluate import OFFICIAL_CLASSES
        extra = [c for c in classes if c not in OFFICIAL_CLASSES]
        if extra:
            print(f"warning: CLASSES contains non-official ids {extra}; their events will be dropped")
        classes = [c for c in classes if c in OFFICIAL_CLASSES]
    except ImportError:
        pass

    src = Path(args.videos)
    videos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix in VIDEO_EXTS)
    if not videos:
        print(f"no .mp4 files found in {src}", file=sys.stderr)
        return 2

    result = {"team": args.team, "videos": {}, "log": {}}
    for path in videos:
        meta = video_meta(path)
        budget = args.time_factor * meta["duration"]
        t0 = time.perf_counter()
        deadline = t0 + budget
        entry = {"events": [], "risk": []}
        log = {"duration": round(meta["duration"], 2), "budget_sec": round(budget, 1), "errors": []}
        print(f"[{path.name}] {meta['duration']:.1f}s @ {meta['fps']:.2f} fps, budget {budget:.0f}s")

        # ---- Part A --------------------------------------------------
        try:
            events = sol.detect_events(str(path))
            entry["events"], problems = clean_events(events, classes, meta["duration"])
            log["errors"] += problems
        except Exception:
            log["errors"].append("detect_events raised:\n" + traceback.format_exc())
            entry["events"] = []
        log["part_a_sec"] = round(time.perf_counter() - t0, 1)

        # ---- Part B --------------------------------------------------
        timed_out = time.perf_counter() > deadline
        if not args.no_risk and not timed_out:
            t1 = time.perf_counter()
            try:
                curve, timed_out = run_risk(sol.RiskEstimator(), path, meta, max(1, args.risk_stride), deadline)
                entry["risk"] = curve
            except Exception:
                log["errors"].append("RiskEstimator raised:\n" + traceback.format_exc())
                entry["risk"] = []
            log["part_b_sec"] = round(time.perf_counter() - t1, 1)

        log["total_sec"] = round(time.perf_counter() - t0, 1)
        if timed_out or log["total_sec"] > budget:
            log["errors"].append(f"over time budget ({log['total_sec']}s > {budget:.0f}s): scored as empty")
            entry = {"events": [], "risk": []}

        result["videos"][path.name] = entry
        result["log"][path.name] = log
        status = "OK" if not log["errors"] else f"{len(log['errors'])} problem(s)"
        print(f"[{path.name}] {len(entry['events'])} events, {len(entry['risk'])} risk samples, "
              f"{log['total_sec']}s — {status}")
        for err in log["errors"]:
            print("   !", err.splitlines()[0])

    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
