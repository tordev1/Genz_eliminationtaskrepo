"""Per-frame object detection.

Two implementations behind one interface (`Detector.detect(frame) -> list[Detection]`):

- MotionDetector: background subtraction (MOG2) + contour filtering. Needs no
  weights, no GPU, no training data. This is the default and what the
  submission ships with.
- YoloDetector: thin wrapper around Ultralytics YOLO, used automatically
  *only* if `ultralytics` is importable AND a weights file is found under
  weights/. It gives cleaner vehicle/person boxes and class labels, which
  makes every downstream rule more precise. It is an optional upgrade path,
  not a requirement, so the baseline never depends on it being present.

Both return the same `Detection` shape so `src/rules.py` never has to know
which one produced a box.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "opencv-python is required. Install with: pip install -r requirements.txt"
    ) from exc


@dataclass
class Detection:
    box: tuple  # (x, y, w, h) in pixels
    label: str  # "vehicle", "person", or "unknown"
    score: float


class MotionDetector:
    """Background-subtraction detector.

    Assumes a static camera (true for this challenge: "one angle, no camera
    motion"), which is exactly the case MOG2 is built for. Classifies each
    foreground blob into "vehicle" / "person" / "unknown" purely from size
    and aspect ratio -- a coarse but zero-training, zero-dependency signal
    that is enough to drive the rule set in src/rules.py.
    """

    def __init__(
        self,
        min_area: int = 220,
        max_area_frac: float = 0.25,
        history: int = 400,
        var_threshold: float = 24.0,
    ):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True
        )
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self._warmed_up = 0

    def warmup(self, frame: np.ndarray) -> None:
        self.bg.apply(frame)
        self._warmed_up += 1

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        frame_area = h * w
        fg = self.bg.apply(frame)
        # 127 = shadow marker in MOG2 output; drop it, keep only solid foreground.
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > frame_area * self.max_area_frac:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bh / float(bw) if bw > 0 else 0.0
            label = self._classify(area, aspect, frame_area)
            detections.append(Detection(box=(x, y, bw, bh), label=label, score=1.0))
        return detections

    @staticmethod
    def _classify(area: float, aspect: float, frame_area: float) -> str:
        """Coarse vehicle/person split from blob geometry.

        A person-shaped blob is tall and narrow (aspect >~ 1.6) and small
        relative to the frame; a vehicle blob is wider than tall or large.
        This is intentionally simple -- swap in YoloDetector for real class
        labels once weights are available.
        """
        rel_area = area / frame_area
        if aspect >= 1.5 and rel_area < 0.02:
            return "person"
        if rel_area >= 0.002:
            return "vehicle"
        return "unknown"


class YoloDetector:
    """Optional Ultralytics YOLO detector. Only instantiate via `try_create`."""

    VEHICLE_NAMES = {"car", "truck", "bus", "motorcycle", "bicycle"}
    PERSON_NAMES = {"person"}

    def __init__(self, model, conf: float = 0.35, imgsz: int = 640):
        self.model = model
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, verbose=False
        )
        detections = []
        if not results:
            return detections
        r = results[0]
        names = r.names
        for box in r.boxes:
            cls_name = names.get(int(box.cls[0]), "")
            if cls_name in self.VEHICLE_NAMES:
                label = "vehicle"
            elif cls_name in self.PERSON_NAMES:
                label = "person"
            else:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0])
            detections.append(Detection(box=(x1, y1, x2 - x1, y2 - y1), label=label, score=score))
        return detections

    @staticmethod
    def try_create(weights_path: str) -> Optional["YoloDetector"]:
        import os

        if not os.path.isfile(weights_path):
            return None
        try:
            from ultralytics import YOLO
        except ImportError:
            return None
        try:
            model = YOLO(weights_path)
            return YoloDetector(model)
        except Exception:
            return None


def build_default_detector(weights_path: str = "weights/yolov8n.pt"):
    """Use YOLO if it's actually available and loadable, else fall back to
    the zero-dependency motion detector. Called once per video in solution.py.
    """
    yolo = YoloDetector.try_create(weights_path)
    return yolo if yolo is not None else MotionDetector()
