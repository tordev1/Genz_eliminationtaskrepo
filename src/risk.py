"""Part B: causal accident-risk scoring.

CausalRiskEngine.update(frame, t_sec) is the only entry point and it is
strictly online: it runs its own lightweight detector + tracker + road model
incrementally, one frame at a time, and never looks at a frame it hasn't
been given yet. solution.py's RiskEstimator is a thin adapter around this
class -- that split keeps the causality guarantee in one place and lets the
same engine be unit-tested outside the harness.

Signal, in order of how much it moves the score:
  1. Time-to-collision (TTC) between the two closest vehicle tracks that are
     converging. TTC = distance / closing_speed. Calibrated so TTC == H (5s,
     the challenge's horizon) maps to score 0.5, per the challenge's own tip
     ("calibrate it so 0.5 means probably within 5 s").
  2. Hard braking: a vehicle decelerating sharply while another track is
     nearby (a near-empty road braking hard is not risky by itself).
  3. A pedestrian track that has just entered the road model's hull while a
     vehicle is within a few car-lengths.
All three are folded together with a max() (worst signal wins) and a short
exponential smoothing pass so the score doesn't flicker frame to frame --
smoothing only ever looks backward, so it stays causal.
"""
from __future__ import annotations

import math
from typing import Optional

from .detector import build_default_detector
from .geometry import distance
from .road_model import RoadModel
from .tracker import IouTracker

H_SEC = 5.0  # challenge-fixed anticipation horizon
DETECT_EVERY_N_FRAMES = 2
CLOSE_FACTOR = 3.0  # "nearby" = within this many combined-box-sizes
BRAKE_DECEL_REF = 250.0  # px/s^2 treated as "hard braking" (empirical, resolution-dependent)
EMA_ALPHA = 0.5


class CausalRiskEngine:
    def __init__(self):
        self.detector = None
        self.tracker = IouTracker(iou_threshold=0.2, max_missed=10)
        self.road: Optional[RoadModel] = None
        self._frame_idx = 0
        self._last_score = 0.0
        self._prev_speed_by_track = {}

    def reset(self, meta: dict):
        self.detector = build_default_detector()
        self.tracker = IouTracker(iou_threshold=0.2, max_missed=10)
        self.road = RoadModel((meta.get("height", 720), meta.get("width", 1280)))
        self._frame_idx = 0
        self._last_score = 0.0
        self._prev_speed_by_track = {}

    def update(self, frame, t_sec: float) -> float:
        self._frame_idx += 1
        if self.detector is None:
            self.reset({"width": frame.shape[1], "height": frame.shape[0]})

        if self._frame_idx % DETECT_EVERY_N_FRAMES != 0:
            return self._last_score

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, t_sec)

        for track in tracks:
            if track.label == "vehicle" and len(track.history) >= 2:
                vx, vy = track.velocity(window_sec=0.6)
                self.road.add_sample(track.history[-1].cx, track.history[-1].cy, vx, vy)

        vehicles = [t for t in tracks if t.label == "vehicle" and len(t.history) >= 2]
        people = [t for t in tracks if t.label == "person" and len(t.history) >= 2]

        risk_ttc = self._min_ttc_risk(vehicles)
        risk_brake = self._hard_brake_risk(vehicles)
        risk_ped = self._pedestrian_risk(people, vehicles)

        raw = max(risk_ttc, risk_brake, risk_ped)
        self._last_score = EMA_ALPHA * raw + (1 - EMA_ALPHA) * self._last_score
        return float(min(1.0, max(0.0, self._last_score)))

    @staticmethod
    def _closing_speed(track_a, track_b) -> float:
        pa, pb = track_a.history[-1], track_b.history[-1]
        vax, vay = track_a.velocity(window_sec=0.6)
        vbx, vby = track_b.velocity(window_sec=0.6)
        dx, dy = pb.cx - pa.cx, pb.cy - pa.cy
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0
        rel_vx, rel_vy = vax - vbx, vay - vby
        closing = -(rel_vx * dx + rel_vy * dy) / dist
        return closing

    def _min_ttc_risk(self, vehicles) -> float:
        best = 0.0
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                a, b = vehicles[i], vehicles[j]
                pa, pb = a.history[-1], b.history[-1]
                d = distance((pa.cx, pa.cy), (pb.cx, pb.cy))
                closing = self._closing_speed(a, b)
                if closing <= 1.0:
                    continue
                ttc = d / closing
                score = max(0.0, min(1.0, 1 - ttc / (2 * H_SEC)))
                best = max(best, score)
        return best

    def _hard_brake_risk(self, vehicles) -> float:
        best = 0.0
        for track in vehicles:
            speed_now = track.speed(window_sec=0.3)
            speed_prev = self._prev_speed_by_track.get(track.id, speed_now)
            dt = 0.3
            decel = max(0.0, speed_prev - speed_now) / dt
            self._prev_speed_by_track[track.id] = speed_now
            if decel <= 0:
                continue
            score = min(1.0, decel / BRAKE_DECEL_REF) * 0.7  # capped below TTC's max weight
            best = max(best, score)
        return best

    def _pedestrian_risk(self, people, vehicles) -> float:
        if not people or not vehicles:
            return 0.0
        best = 0.0
        for person in people:
            p = person.history[-1]
            on_road = self.road.on_road(p.cx, p.cy)
            if not on_road:
                continue
            for veh in vehicles:
                v = veh.history[-1]
                d = distance((p.cx, p.cy), (v.cx, v.cy))
                thresh = CLOSE_FACTOR * (max(p.w, p.h) + max(v.w, v.h)) / 2.0
                if d <= thresh:
                    best = max(best, 0.75)
        return best
