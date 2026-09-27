"""Exploratory data analysis over samples/*.mp4.

Run once real camera footage is dropped into samples/:

    python -m src.eda --samples samples --out docs/eda_output

Produces, per video and pooled:
  - basic_stats.json        resolution, fps, duration, mean brightness
  - motion_heatmap_<name>.png   where motion concentrates (proxy for lanes/ROI)
  - object_count_<name>.json    vehicle/person blob count per sampled frame,
                                 the same signal the website's "object counts
                                 over time" chart consumes.

Deliberately dependency-light (cv2 + numpy only) since this doubles as a
sanity check that the detector behaves reasonably before it's wired into
solution.py's rule engine.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import cv2

from .detector import MotionDetector

SAMPLE_STEP_SEC = 0.5


def analyze_video(path: str, out_dir: str):
    name = os.path.splitext(os.path.basename(path))[0]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  could not open {path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = n_frames / fps if fps > 0 else 0.0

    stride = max(1, round(fps * SAMPLE_STEP_SEC))
    detector = MotionDetector()
    heatmap = np.zeros((height, width), dtype=np.float32)
    counts = []
    brightness_samples = []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            dets = detector.detect(frame)
            n_vehicle = sum(1 for d in dets if d.label == "vehicle")
            n_person = sum(1 for d in dets if d.label == "person")
            counts.append({"t": idx / fps, "vehicle": n_vehicle, "person": n_person})
            for d in dets:
                x, y, w, h = [int(v) for v in d.box]
                heatmap[max(0, y):y + h, max(0, x):x + w] += 1.0
            brightness_samples.append(float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))
        idx += 1
    cap.release()

    stats = {
        "video": os.path.basename(path),
        "width": width, "height": height, "fps": fps,
        "duration_sec": round(duration, 1),
        "mean_brightness": round(float(np.mean(brightness_samples)), 1) if brightness_samples else None,
        "mean_vehicle_count": round(float(np.mean([c["vehicle"] for c in counts])), 2) if counts else 0,
        "mean_person_count": round(float(np.mean([c["person"] for c in counts])), 2) if counts else 0,
    }

    os.makedirs(out_dir, exist_ok=True)
    if heatmap.max() > 0:
        norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(out_dir, f"motion_heatmap_{name}.png"), color)
    with open(os.path.join(out_dir, f"object_count_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="samples")
    parser.add_argument("--out", default="docs/eda_output")
    args = parser.parse_args()

    videos = sorted(
        os.path.join(args.samples, f)
        for f in os.listdir(args.samples)
        if f.lower().endswith(".mp4")
    ) if os.path.isdir(args.samples) else []

    if not videos:
        print(f"No .mp4 files found in {args.samples}/ -- nothing to analyze yet.")
        return

    all_stats = []
    for v in videos:
        print(f"Analyzing {v} ...")
        stats = analyze_video(v, args.out)
        if stats:
            all_stats.append(stats)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "basic_stats.json"), "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Wrote {len(all_stats)} video summaries to {args.out}/basic_stats.json")


if __name__ == "__main__":
    main()
