"""Team Genz -- Part A + Part B reference implementation.

Implements the WIUT Hackathon 2026 CV-track interface (organizers'
`wiut_cv_scripts/solution.py` template: CLASSES, detect_events, RiskEstimator).
This file only wires together the modules in src/: a per-frame detector, an
IoU tracker, an offline rule engine (Part A) and a causal risk engine
(Part B). See README.md for the full write-up and docs/REPORT.md for what
worked / what did not.
"""
from __future__ import annotations

import random

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise RuntimeError(
        "opencv-python-headless is required. Install with: pip install -r requirements.txt"
    ) from exc

from src.detector import build_default_detector
from src.rules import build_events
from src.risk import CausalRiskEngine
from src.tracker import IouTracker

# Official class ids (14, exact order per the task spec). May only be
# shrunk (never added to) -- run_submission.py filters against
# evaluate.OFFICIAL_CLASSES anyway, so this list is never a source of truth
# for grading, only documentation.
CLASSES: list[str] = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]

# Anticipation horizon used by the metric (seconds). RiskEstimator.step()
# returns P(an `accident` starts within the next RISK_HORIZON_SEC seconds).
RISK_HORIZON_SEC = 5.0

# Determinism (see README "Determinism").
random.seed(0)
np.random.seed(0)

# Analysis frame rate: we don't need to run detection at native fps to catch
# events that last seconds, and staying well under the 3x-realtime budget
# matters more than an extra few analysis frames per second. See
# docs/REPORT.md "Runtime" for the measurements behind this number.
TARGET_ANALYSIS_FPS = 12.0


def _analysis_stride(native_fps: float) -> int:
    if native_fps <= 0:
        return 1
    return max(1, round(native_fps / TARGET_ANALYSIS_FPS))


def detect_events(video_path: str) -> list:
    """Part A. Return [[start_sec, end_sec, label], ...] for one .mp4."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (n_frames / native_fps) if native_fps > 0 and n_frames > 0 else 0.0

    stride = _analysis_stride(native_fps)

    detector = build_default_detector()
    tracker = IouTracker(iou_threshold=0.2, max_missed=int(round(2.0 * native_fps / stride)))

    frame_idx = 0
    last_t = 0.0
    while True:
        if frame_idx % stride == 0:
            # Full decode only for frames we actually analyze.
            ok, frame = cap.read()
            if not ok:
                break
            t_sec = frame_idx / native_fps
            last_t = t_sec
            detections = detector.detect(frame)
            tracker.update(detections, t_sec)
        else:
            # cap.grab() advances the stream without decoding the frame --
            # cheap compared to cap.read() on the frames the stride skips.
            if not cap.grab():
                break
        frame_idx += 1
    cap.release()

    if duration <= 0:
        duration = last_t

    if width == 0 or height == 0:
        width, height = 1280, 720

    return build_events(tracker.tracks, (height, width), duration)


class RiskEstimator:
    """Part B. Causal: step() sees frames in order and nothing else."""

    def __init__(self):
        self._engine = CausalRiskEngine()

    def reset(self, meta: dict) -> None:
        self._engine.reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return self._engine.update(frame, t_sec)
