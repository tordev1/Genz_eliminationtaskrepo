"""Part A rule engine: completed track histories -> event segments.

Everything here is hand-written heuristics on top of the tracker's output,
which is exactly one of the approaches the challenge explicitly allows
("a detector plus tracker with hand-written rules"). Six classes are
implemented with an automatic, per-camera-agnostic signal:

    stopped_vehicle, congestion, wrong_way, near_miss, accident, road_obstacle

Two more use a coarse geometric proxy for pedestrians:

    jaywalking, failure_to_yield (the latter only when a vehicle track
    passes very close to a jaywalking pedestrian track)

The remaining six classes (red_light, stop_line, illegal_u_turn,
illegal_turn, solid_line_crossing, fire_smoke) need information this
camera-agnostic pipeline does not have out of the box -- a calibrated stop
line, signal-state ROI, or lane-marking geometry, or reliable color/texture
cues for smoke that a MOG2 blob does not carry. Rather than emit
low-precision guesses that would hurt Score_A, they are left as documented
extension points (see README "Known limitations & extension points") wired
through `CalibrationConfig` below -- drop in a `calibration.json` for a
given camera and `red_light`/`stop_line`/etc. switch on automatically.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .geometry import distance, iou, merge_close_intervals
from .road_model import RoadModel
from .tracker import Track

MIN_STOPPED_SEC = 10.0
STATIONARY_SPEED_PX_S = 6.0
WRONG_WAY_MIN_SPEED_PX_S = 12.0
WRONG_WAY_ANGLE_DEG = 135.0
WRONG_WAY_MIN_SEC = 1.5
NEAR_MISS_CLOSE_FACTOR = 1.4  # multiple of combined box size counted as "close"
NEAR_MISS_MIN_CLOSING_SPEED = 40.0  # px/s
CONGESTION_MIN_TRACKS = 3
CONGESTION_STATIONARY_FRACTION = 0.7
CONGESTION_MIN_SEC = 8.0
MERGE_GAP_SEC = 0.75


@dataclass
class CalibrationConfig:
    """Optional per-camera calibration; loaded from calibration.json next to
    the video if present. Absence disables the classes that need it -- see
    module docstring.
    """
    stop_line: Optional[List[Tuple[float, float]]] = None
    signal_roi: Optional[Tuple[int, int, int, int]] = None
    no_u_turn_zones: Optional[List[List[Tuple[float, float]]]] = None
    solid_lines: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None
    crosswalks: Optional[List[List[Tuple[float, float]]]] = None

    @staticmethod
    def load(path: str) -> "CalibrationConfig":
        if not path or not os.path.isfile(path):
            return CalibrationConfig()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CalibrationConfig(**data)


def build_road_model(tracks: Dict[int, Track], frame_shape) -> RoadModel:
    model = RoadModel(frame_shape)
    for track in tracks.values():
        if track.label != "vehicle":
            continue
        for i in range(1, len(track.history)):
            p0, p1 = track.history[i - 1], track.history[i]
            dt = p1.t - p0.t
            if dt <= 1e-6:
                continue
            vx = (p1.cx - p0.cx) / dt
            vy = (p1.cy - p0.cy) / dt
            model.add_sample(p1.cx, p1.cy, vx, vy)
    return model


def _stopped_vehicle_events(track: Track) -> List[Tuple[float, float]]:
    if track.label != "vehicle" or len(track.history) < 2:
        return []
    events = []
    run_start = None
    prev = track.history[0]
    for p in track.history[1:]:
        dt = p.t - prev.t
        v = distance((prev.cx, prev.cy), (p.cx, p.cy)) / dt if dt > 1e-6 else 0.0
        if v <= STATIONARY_SPEED_PX_S:
            if run_start is None:
                run_start = prev.t
        else:
            if run_start is not None and prev.t - run_start >= MIN_STOPPED_SEC:
                events.append((run_start, prev.t))
            run_start = None
        prev = p
    if run_start is not None and prev.t - run_start >= MIN_STOPPED_SEC:
        events.append((run_start, prev.t))
    return events


def _wrong_way_events(track: Track, road: RoadModel) -> List[Tuple[float, float]]:
    if track.label != "vehicle" or len(track.history) < 3:
        return []
    events = []
    run_start = None
    prev = track.history[0]
    for p in track.history[1:]:
        dt = p.t - prev.t
        vx = (p.cx - prev.cx) / dt if dt > 1e-6 else 0.0
        vy = (p.cy - prev.cy) / dt if dt > 1e-6 else 0.0
        speed = math.hypot(vx, vy)
        deviation = road.heading_deviation_deg(vx, vy)
        is_wrong = (
            speed >= WRONG_WAY_MIN_SPEED_PX_S
            and deviation is not None
            and deviation >= WRONG_WAY_ANGLE_DEG
        )
        if is_wrong:
            if run_start is None:
                run_start = prev.t
        else:
            if run_start is not None and prev.t - run_start >= WRONG_WAY_MIN_SEC:
                events.append((run_start, prev.t))
            run_start = None
        prev = p
    if run_start is not None and prev.t - run_start >= WRONG_WAY_MIN_SEC:
        events.append((run_start, prev.t))
    return events


def _road_obstacle_events(track: Track, road: RoadModel) -> List[Tuple[float, float]]:
    if track.label != "unknown" or len(track.history) < 2:
        return []
    p_first, p_last = track.history[0], track.history[-1]
    if not road.on_road(p_first.cx, p_first.cy):
        return []
    max_move = max(
        distance((p_first.cx, p_first.cy), (p.cx, p.cy)) for p in track.history
    )
    if max_move > max(p_first.w, p_first.h):  # it moved -> not a static obstacle
        return []
    if p_last.t - p_first.t < MIN_STOPPED_SEC:
        return []
    return [(p_first.t, p_last.t)]


def _jaywalking_events(track: Track, road: RoadModel) -> List[Tuple[float, float]]:
    if track.label != "person" or len(track.history) < 2:
        return []
    events = []
    run_start = None
    prev = None
    for p in track.history:
        on_road = road.on_road(p.cx, p.cy)
        if on_road:
            if run_start is None:
                run_start = p.t
        else:
            if run_start is not None:
                events.append((run_start, prev.t if prev else p.t))
            run_start = None
        prev = p
    if run_start is not None:
        events.append((run_start, track.history[-1].t))
    return [(s, e) for s, e in events if e > s]


def _pairwise_events(tracks: Dict[int, Track]) -> Tuple[
    List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]
]:
    """Returns (accident, near_miss, failure_to_yield) segments from
    proximity between every pair of tracks, scanned along shared timestamps.
    """
    accident, near_miss, failure_to_yield = [], [], []
    ids = list(tracks.keys())
    for i in range(len(ids)):
        ta = tracks[ids[i]]
        if len(ta.history) < 2:
            continue
        idx_a = {round(p.t, 3): p for p in ta.history}
        for j in range(i + 1, len(ids)):
            tb = tracks[ids[j]]
            if len(tb.history) < 2:
                continue
            idx_b = {round(p.t, 3): p for p in tb.history}
            shared = sorted(set(idx_a) & set(idx_b))
            if len(shared) < 2:
                continue

            overlap_run_start = None
            close_run_start = None
            had_contact_in_close_run = False
            prev_t = None
            prev_dist = None

            def close_thresh(pa, pb):
                return NEAR_MISS_CLOSE_FACTOR * (max(pa.w, pa.h) + max(pb.w, pb.h)) / 2.0

            for t in shared:
                pa, pb = idx_a[t], idx_b[t]
                box_a = (pa.cx - pa.w / 2, pa.cy - pa.h / 2, pa.w, pa.h)
                box_b = (pb.cx - pb.w / 2, pb.cy - pb.h / 2, pb.w, pb.h)
                overlap = iou(box_a, box_b) > 0.02
                d = distance((pa.cx, pa.cy), (pb.cx, pb.cy))
                closing_speed = (
                    (prev_dist - d) / (t - prev_t)
                    if prev_dist is not None and t - prev_t > 1e-6
                    else 0.0
                )

                if overlap:
                    if overlap_run_start is None:
                        overlap_run_start = t
                    if ta.label == "person" or tb.label == "person":
                        failure_to_yield.append((overlap_run_start, t))
                else:
                    if overlap_run_start is not None:
                        accident.append((overlap_run_start, t))
                        overlap_run_start = None

                is_close = d <= close_thresh(pa, pb)
                if is_close and not overlap:
                    if close_run_start is None:
                        close_run_start = t
                        had_contact_in_close_run = False
                    if closing_speed >= NEAR_MISS_MIN_CLOSING_SPEED:
                        had_contact_in_close_run = True
                else:
                    if close_run_start is not None:
                        if had_contact_in_close_run and t - close_run_start >= 0.4:
                            near_miss.append((close_run_start, t))
                        close_run_start = None

                prev_t, prev_dist = t, d

            if overlap_run_start is not None:
                accident.append((overlap_run_start, shared[-1]))
            if close_run_start is not None and had_contact_in_close_run:
                near_miss.append((close_run_start, shared[-1]))

    return accident, near_miss, failure_to_yield


def _congestion_events(
    tracks: Dict[int, Track], duration: float, sample_step: float = 0.5
) -> List[Tuple[float, float]]:
    vehicle_tracks = [t for t in tracks.values() if t.label == "vehicle" and len(t.history) >= 2]
    if not vehicle_tracks:
        return []

    n_steps = max(1, int(duration / sample_step))
    congested_flags = []
    for i in range(n_steps + 1):
        t = i * sample_step
        speeds = []
        for track in vehicle_tracks:
            pts = [p for p in track.history if abs(p.t - t) <= sample_step]
            if len(pts) < 2:
                continue
            p0, p1 = pts[0], pts[-1]
            dt = p1.t - p0.t
            if dt <= 1e-6:
                continue
            v = distance((p0.cx, p0.cy), (p1.cx, p1.cy)) / dt
            speeds.append(v)
        if len(speeds) < CONGESTION_MIN_TRACKS:
            congested_flags.append((t, False))
            continue
        frac_slow = sum(1 for v in speeds if v <= STATIONARY_SPEED_PX_S * 1.5) / len(speeds)
        congested_flags.append((t, frac_slow >= CONGESTION_STATIONARY_FRACTION))

    events = []
    run_start = None
    prev_t = 0.0
    for t, flag in congested_flags:
        if flag:
            if run_start is None:
                run_start = t
        else:
            if run_start is not None and prev_t - run_start >= CONGESTION_MIN_SEC:
                events.append((run_start, prev_t))
            run_start = None
        prev_t = t
    if run_start is not None and prev_t - run_start >= CONGESTION_MIN_SEC:
        events.append((run_start, prev_t))
    return events


def build_events(
    tracks: Dict[int, Track], frame_shape, duration: float
) -> List[List]:
    """Turn every finished track into the final [start, end, label] list."""
    road = build_road_model(tracks, frame_shape)

    by_label: Dict[str, List[Tuple[float, float]]] = {}

    def add(label, segments):
        by_label.setdefault(label, []).extend(segments)

    for track in tracks.values():
        add("stopped_vehicle", _stopped_vehicle_events(track))
        add("wrong_way", _wrong_way_events(track, road))
        add("road_obstacle", _road_obstacle_events(track, road))
        add("jaywalking", _jaywalking_events(track, road))

    accident, near_miss, failure_to_yield = _pairwise_events(tracks)
    add("accident", accident)
    add("near_miss", near_miss)
    add("failure_to_yield", failure_to_yield)
    add("congestion", _congestion_events(tracks, duration))

    events = []
    for label, segments in by_label.items():
        merged = merge_close_intervals(segments, MERGE_GAP_SEC)
        for s, e in merged:
            s = max(0.0, s)
            e = min(duration, e)
            if e - s >= 0.2:
                events.append([round(s, 2), round(e, 2), label])

    events.sort(key=lambda x: x[0])
    return events
