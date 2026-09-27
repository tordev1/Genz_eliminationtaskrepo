"""A cheap, self-calibrating model of "where the road is" and "which way
traffic flows", built directly from observed vehicle tracks instead of a
hand-drawn ROI. Two use sites:

  - offline (Part A, src/rules.py): built once from the *whole* video, so
    it can use every vehicle track ever seen.
  - causal (Part B, src/risk.py): built incrementally, frame by frame, using
    only vehicle tracks seen so far -- never the future.

Both share this class; the only difference is how often `add_sample` is called.

Direction is modeled **per spatial grid cell**, not as one scene-wide
average. A single global "dominant direction" breaks down on anything but a
simple one-way road: a divided two-way avenue has legal traffic flowing in
two roughly-opposite directions at once (each carriageway), and a real
intersection adds legal cross-traffic on top of that. Averaging all of it
into one angle produces an unstable, near-arbitrary direction, and flags
half of ordinary opposing-carriageway traffic as `wrong_way`. Binning
directions by where on the frame they were observed keeps each carriageway
/ lane's own direction separate, since they occupy different regions of the
frame -- confirmed against real fixed-camera footage of a divided,
multi-lane intersection during development (see docs/REPORT.md).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


REBUILD_EVERY_N_SAMPLES = 8  # throttles the O(n) hull rebuild -- see add_sample
GRID_COLS = 8
GRID_ROWS = 8
MIN_CELL_SAMPLES = 3  # below this, fall back to a wider neighborhood / global mean


class RoadModel:
    def __init__(self, frame_shape: Tuple[int, int]):
        self.h, self.w = frame_shape[:2]
        self._points: List[Tuple[float, float]] = []
        self._hull_mask: Optional[np.ndarray] = None
        self._dirty = True
        self._samples_since_rebuild = 0

        # Per-cell running circular-mean accumulators: (sin_sum, cos_sum, count).
        self._cell_w = max(1.0, self.w / GRID_COLS)
        self._cell_h = max(1.0, self.h / GRID_ROWS)
        self._cells: Dict[Tuple[int, int], List[float]] = {}
        # Scene-wide fallback accumulator, for the early frames / sparse
        # cells where a local estimate isn't trustworthy yet.
        self._global = [0.0, 0.0, 0]

    def _cell_index(self, x: float, y: float) -> Tuple[int, int]:
        ci = min(GRID_COLS - 1, max(0, int(x / self._cell_w)))
        cj = min(GRID_ROWS - 1, max(0, int(y / self._cell_h)))
        return ci, cj

    def add_sample(self, cx: float, cy: float, vx: float = 0.0, vy: float = 0.0):
        self._points.append((cx, cy))
        speed = math.hypot(vx, vy)
        if speed > 2.0:  # ignore near-zero noise so idle vehicles don't wash out direction
            angle = math.atan2(vy, vx)
            s, c = math.sin(angle), math.cos(angle)
            key = self._cell_index(cx, cy)
            cell = self._cells.setdefault(key, [0.0, 0.0, 0])
            cell[0] += s
            cell[1] += c
            cell[2] += 1
            self._global[0] += s
            self._global[1] += c
            self._global[2] += 1
        # Rebuilding the hull is an O(n) hull recompute plus a full-frame mask
        # allocation -- too expensive to do on every single sample when this
        # is called every processed frame in the causal (Part B) path.
        # Freshness at N=8 samples is more than enough for something as
        # slow-changing as "where the road is."
        self._samples_since_rebuild += 1
        if self._hull_mask is None or self._samples_since_rebuild >= REBUILD_EVERY_N_SAMPLES:
            self._dirty = True
            self._samples_since_rebuild = 0
        # Keep this bounded for the causal, per-frame use case.
        if len(self._points) > 20000:
            self._points = self._points[-10000:]

    def _rebuild(self):
        if cv2 is not None and len(self._points) >= 3:
            pts = np.array(self._points, dtype=np.float32)
            hull = cv2.convexHull(pts)
            mask = np.zeros((self.h, self.w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
            self._hull_mask = mask
        else:
            self._hull_mask = None
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

    @staticmethod
    def _circular_mean(sin_sum: float, cos_sum: float) -> float:
        return math.atan2(sin_sum, cos_sum)

    def _local_dominant_angle(self, x: float, y: float) -> Optional[float]:
        """Circular mean of the cell (x, y) falls in; widens to the 3x3
        neighborhood, then the whole scene, if the immediate cell is too
        sparse to trust -- e.g. early in a video before enough traffic has
        passed through that specific patch of road.
        """
        ci, cj = self._cell_index(x, y)
        cell = self._cells.get((ci, cj))
        if cell and cell[2] >= MIN_CELL_SAMPLES:
            return self._circular_mean(cell[0], cell[1])

        sin_sum = cos_sum = 0.0
        count = 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                neighbor = self._cells.get((ci + di, cj + dj))
                if neighbor:
                    sin_sum += neighbor[0]
                    cos_sum += neighbor[1]
                    count += neighbor[2]
        if count >= MIN_CELL_SAMPLES:
            return self._circular_mean(sin_sum, cos_sum)

        if self._global[2] > 0:
            return self._circular_mean(self._global[0], self._global[1])
        return None

    def heading_deviation_deg(self, x: float, y: float, vx: float, vy: float) -> Optional[float]:
        """Angle (0-180) between (vx, vy) and the locally-observed dominant
        traffic direction near (x, y) -- see module docstring for why this
        is local rather than one scene-wide angle.
        """
        dominant_angle = self._local_dominant_angle(x, y)
        if dominant_angle is None or (vx == 0 and vy == 0):
            return None
        angle = math.atan2(vy, vx)
        diff = abs(math.degrees(angle - dominant_angle)) % 360
        if diff > 180:
            diff = 360 - diff
        return diff
