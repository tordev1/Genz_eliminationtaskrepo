"""A minimal, dependency-free multi-object tracker.

Greedy IoU matching between frames (à la the association step of SORT,
without the Kalman filter -- fixed camera + short gaps between sampled
frames make a Kalman predictor overkill here). Each track keeps its full
history of (t_sec, cx, cy, w, h) so src/rules.py can compute velocity,
heading and dwell time directly from it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .geometry import center, distance, iou


@dataclass
class TrackPoint:
    t: float
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class Track:
    id: int
    label: str
    history: List[TrackPoint] = field(default_factory=list)
    age_missed: int = 0
    active: bool = True

    def last(self) -> TrackPoint:
        return self.history[-1]

    def velocity(self, window_sec: float = 1.0):
        """Average (vx, vy) in px/sec over the trailing `window_sec`."""
        if len(self.history) < 2:
            return 0.0, 0.0
        p1 = self.history[-1]
        t0 = p1.t - window_sec
        p0 = self.history[0]
        for p in reversed(self.history[:-1]):
            if p.t <= t0:
                p0 = p
                break
        else:
            p0 = self.history[0]
        dt = p1.t - p0.t
        if dt <= 1e-6:
            return 0.0, 0.0
        return (p1.cx - p0.cx) / dt, (p1.cy - p0.cy) / dt

    def speed(self, window_sec: float = 1.0) -> float:
        vx, vy = self.velocity(window_sec)
        return (vx ** 2 + vy ** 2) ** 0.5

    def dwell_time_stationary(self, speed_thresh_px_s: float, window_sec: float = 10.0) -> float:
        """Seconds of continuous history where speed stayed below threshold."""
        if len(self.history) < 2:
            return 0.0
        dwell = 0.0
        for i in range(len(self.history) - 1, 0, -1):
            p1, p0 = self.history[i], self.history[i - 1]
            dt = p1.t - p0.t
            v = distance((p0.cx, p0.cy), (p1.cx, p1.cy)) / dt if dt > 1e-6 else 0.0
            if v > speed_thresh_px_s:
                break
            dwell += dt
        return dwell


class IouTracker:
    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 8):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def active_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.active]

    def update(self, detections, t_sec: float) -> List[Track]:
        """detections: list of Detection(box, label, score). Returns active tracks."""
        active = self.active_tracks()
        unmatched_dets = set(range(len(detections)))
        matched_track_ids = set()

        pairs = []
        for ti, track in enumerate(active):
            tbox = self._track_box(track)
            for di, det in enumerate(detections):
                score = iou(tbox, det.box)
                if score >= self.iou_threshold:
                    pairs.append((score, track.id, di))
        pairs.sort(key=lambda x: -x[0])

        for score, tid, di in pairs:
            if tid in matched_track_ids or di not in unmatched_dets:
                continue
            det = detections[di]
            cx, cy = center(det.box)
            track = self.tracks[tid]
            track.history.append(TrackPoint(t_sec, cx, cy, det.box[2], det.box[3]))
            track.age_missed = 0
            matched_track_ids.add(tid)
            unmatched_dets.discard(di)

        for track in active:
            if track.id not in matched_track_ids:
                track.age_missed += 1
                if track.age_missed > self.max_missed:
                    track.active = False

        for di in unmatched_dets:
            det = detections[di]
            cx, cy = center(det.box)
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = Track(
                id=tid,
                label=det.label,
                history=[TrackPoint(t_sec, cx, cy, det.box[2], det.box[3])],
            )

        return self.active_tracks()

    @staticmethod
    def _track_box(track: Track):
        p = track.last()
        return (p.cx - p.w / 2.0, p.cy - p.h / 2.0, p.w, p.h)
