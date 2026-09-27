"""A cheap, self-calibrating model of "where the road is" and "which way
traffic flows", built directly from observed vehicle tracks instead of a
hand-drawn ROI. Two use sites:

  - offline (Part A, src/rules.py): built once from the *whole* video, so
    it can use every vehicle track ever seen.
  - causal (Part B, src/risk.py): built incrementally, frame by frame, using
    only vehicle tracks seen so far -- never the future.

Both share this class; the only difference is how often `add_sample` is called.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class RoadModel:
    def __init__(self, frame_shape: Tuple[int, int]):
        self.h, self.w = frame_shape[:2]
        self._points: List[Tuple[float, float]] = []
        self._vectors: List[Tuple[float, float]] = []  # (vx, vy) samples, speed-weighted
        self._hull_mask: Optional[np.ndarray] = None
        self._dominant_angle: Optional[float] = None
        self._dirty = True

    def add_sample(self, cx: float, cy: float, vx: float = 0.0, vy: float = 0.0):
        self._points.append((cx, cy))
        speed = math.hypot(vx, vy)
        if speed > 2.0:  # ignore near-zero noise so idle vehicles don't wash out direction
            self._vectors.append((vx, vy))
        self._dirty = True
        # Keep this bounded for the causal, per-frame use case.
        if len(self._points) > 20000:
            self._points = self._points[-10000:]
        if len(self._vectors) > 20000:
            self._vectors = self._vectors[-10000:]

    def _rebuild(self):
        if cv2 is not None and len(self._points) >= 3:
            pts = np.array(self._points, dtype=np.float32)
            hull = cv2.convexHull(pts)
            mask = np.zeros((self.h, self.w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
            self._hull_mask = mask
        else:
            self._hull_mask = None

        if self._vectors:
            angles = [math.atan2(vy, vx) for vx, vy in self._vectors]
            # circular mean, robust to the 0/2*pi wraparound a plain average would break on
            sin_sum = sum(math.sin(a) for a in angles)
            cos_sum = sum(math.cos(a) for a in angles)
            self._dominant_angle = math.atan2(sin_sum, cos_sum)
        else:
            self._dominant_angle = None
        self._dirty = False

    def on_road(self, x: float, y: float) -> bool:
        if self._dirty:
            self._rebuild()
        if self._hull_mask is None:
            return True  # not enough data yet: don't block downstream rules
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < self.h and 0 <= xi < self.w:
            return bool(self._hull_mask[yi, xi])
        return False

    def heading_deviation_deg(self, vx: float, vy: float) -> Optional[float]:
        """Angle (0-180) between (vx, vy) and the dominant traffic direction."""
        if self._dirty:
            self._rebuild()
        if self._dominant_angle is None or (vx == 0 and vy == 0):
            return None
        angle = math.atan2(vy, vx)
        diff = abs(math.degrees(angle - self._dominant_angle)) % 360
        if diff > 180:
            diff = 360 - diff
        return diff
