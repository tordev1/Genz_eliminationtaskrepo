"""Starter-kit harness (per the challenge spec, unchanged in behaviour by
teams): walks a folder of videos, calls solution.detect_events on each,
streams every frame through solution.RiskEstimator, enforces the per-video
time budget (<= 3x video duration for Part A + Part B combined), drops
malformed events, and writes predictions.json in the exact required format.

Usage:
    python run_submission.py --videos /data/test --out predictions.json
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
import traceback

import cv2

TEAM_NAME = os.environ.get("TEAM_NAME", "Genz")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("run_submission")


def _probe(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, 0.0, 0, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = n_frames / fps if fps > 0 else 0.0
    return duration, fps, n_frames, width, height


def _worker(video_path: str, meta: dict, queue: mp.Queue):
    """Runs in a separate process so the parent can enforce a hard wall-clock
    budget with terminate() -- a plain thread can't be force-stopped while
    it's inside someone else's C-extension / numpy call.
    """
    try:
        import solution  # imported inside the worker so a crash here is isolated

        events = solution.detect_events(video_path)

        estimator = solution.RiskEstimator()
        estimator.reset(meta)
        cap = cv2.VideoCapture(video_path)
        risk = []
        fps = meta["fps"] or 25.0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_sec = idx / fps
            score = float(estimator.step(frame, t_sec))
            risk.append([round(t_sec, 3), round(score, 4)])
            idx += 1
        cap.release()

        queue.put({"ok": True, "events": events, "risk": risk})
    except Exception:
        queue.put({"ok": False, "error": traceback.format_exc()})


def _clean_events(video_name: str, events, classes: set) -> list:
    """Drop malformed / overlapping-same-class events, logging every drop."""
    cleaned = []
    by_class_intervals = {}
    try:
        events = sorted(events, key=lambda e: (e[0], e[1]))
    except Exception:
        log.warning("[%s] events not sortable, dropping all: %r", video_name, events)
        return []

    for ev in events:
        if not (isinstance(ev, (list, tuple)) and len(ev) == 3):
            log.warning("[%s] dropped malformed event (wrong shape): %r", video_name, ev)
            continue
        start, end, label = ev
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            log.warning("[%s] dropped malformed event (non-numeric bounds): %r", video_name, ev)
            continue
        if label not in classes:
            log.warning("[%s] dropped event with unknown label %r: %r", video_name, label, ev)
            continue
        if not (start < end):
            log.warning("[%s] dropped event with start >= end: %r", video_name, ev)
            continue

        overlaps = False
        for s, e in by_class_intervals.get(label, []):
            if start < e and s < end:
                overlaps = True
                break
        if overlaps:
            log.warning(
                "[%s] dropped event overlapping an earlier segment of class %r: %r",
                video_name, label, ev,
            )
            continue

        by_class_intervals.setdefault(label, []).append((start, end))
        cleaned.append([start, end, label])

    return cleaned


def run(videos_dir: str, out_path: str, team: str = TEAM_NAME, time_budget_multiplier: float = 3.0):
    import solution  # only to read CLASSES; the actual work happens in the worker process

    classes = set(solution.CLASSES)
    video_files = sorted(
        f for f in os.listdir(videos_dir) if f.lower().endswith(".mp4")
    )
    if not video_files:
        log.warning("No .mp4 files found under %s", videos_dir)

    result = {"team": team, "videos": {}}

    for fname in video_files:
        path = os.path.join(videos_dir, fname)
        duration, fps, n_frames, width, height = _probe(path)
        budget = max(5.0, duration * time_budget_multiplier)
        meta = {
            "video_id": fname,
            "fps": fps,
            "width": width,
            "height": height,
            "n_frames": n_frames,
        }

        log.info("Running %s (duration=%.1fs, budget=%.1fs)", fname, duration, budget)

        queue: mp.Queue = mp.Queue()
        proc = mp.Process(target=_worker, args=(path, meta, queue))
        start = time.time()
        proc.start()
        proc.join(timeout=budget)
        elapsed = time.time() - start

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            log.error(
                "%s exceeded the %.1fs time budget (ran %.1fs) -- scoring as empty",
                fname, budget, elapsed,
            )
            result["videos"][fname] = {"events": [], "risk": []}
            continue

        try:
            payload = queue.get_nowait()
        except Exception:
            log.error("%s produced no result (crashed before reporting) -- scoring as empty", fname)
            result["videos"][fname] = {"events": [], "risk": []}
            continue

        if not payload.get("ok"):
            log.error("%s raised an exception -- scoring as empty:\n%s", fname, payload.get("error"))
            result["videos"][fname] = {"events": [], "risk": []}
            continue

        events = _clean_events(fname, payload["events"], classes)
        risk = payload["risk"]
        result["videos"][fname] = {"events": events, "risk": risk}
        log.info("%s: %d event(s), %d risk samples, %.1fs elapsed", fname, len(events), len(risk), elapsed)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True, help="Folder containing .mp4 test videos")
    parser.add_argument("--out", required=True, help="Path to write predictions.json")
    parser.add_argument("--team", default=TEAM_NAME)
    args = parser.parse_args()
    run(args.videos, args.out, team=args.team)
