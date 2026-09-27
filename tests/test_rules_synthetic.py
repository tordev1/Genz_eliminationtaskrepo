"""Unit tests for src/rules.py's rule logic in isolation, using hand-built
Track/TrackPoint histories instead of real or rendered video. This is
deliberately independent of whether any real camera footage exists yet --
it tests "given this track history, does the right class fire", which a
detector/tracker pass would only make slower and flakier to assert on.

Thresholds are read from the `rules` module at import time rather than
hardcoded, so these tests stay correct if tuned_params.json later changes
the defaults (see src/tunable.py) -- every synthetic scenario is built with
generous margin past whatever the current threshold is.
"""
from src import rules as rules_mod
from src.rules import build_events
from src.tracker import Track, TrackPoint

FRAME_SHAPE = (480, 640)  # (height, width)


def _track(track_id, label, points):
    """points: list of (t, cx, cy, w, h)."""
    return Track(id=track_id, label=label, history=[TrackPoint(*p) for p in points])


def test_stopped_vehicle_fires_after_min_dwell():
    dwell = rules_mod.MIN_STOPPED_SEC + 3.0
    pts = [(t, 100.0, 100.0, 30.0, 20.0) for t in _times(0.0, dwell, 0.5)]
    tracks = {1: _track(1, "vehicle", pts)}
    events = build_events(tracks, FRAME_SHAPE, duration=dwell + 1)
    labels = [e[2] for e in events]
    assert "stopped_vehicle" in labels


def test_brief_stop_does_not_fire_stopped_vehicle():
    dwell = max(0.5, rules_mod.MIN_STOPPED_SEC - 3.0)
    pts = [(t, 100.0, 100.0, 30.0, 20.0) for t in _times(0.0, dwell, 0.5)]
    tracks = {1: _track(1, "vehicle", pts)}
    events = build_events(tracks, FRAME_SHAPE, duration=dwell + 1)
    labels = [e[2] for e in events]
    assert "stopped_vehicle" not in labels


def test_wrong_way_fires_against_dominant_flow():
    duration = rules_mod.WRONG_WAY_MIN_SEC + 3.0
    speed = rules_mod.WRONG_WAY_MIN_SPEED_PX_S * 3
    # Five vehicles establishing a strong left-to-right dominant flow.
    tracks = {}
    for i in range(5):
        pts = [(t, 50.0 + speed * t, 300.0 + i, 30.0, 20.0) for t in _times(0.0, duration, 0.2)]
        tracks[i] = _track(i, "vehicle", pts)
    # One vehicle driving straight against it (right-to-left), well past the
    # angle and duration thresholds.
    pts = [(t, 600.0 - speed * t, 100.0, 30.0, 20.0) for t in _times(0.0, duration, 0.2)]
    tracks[99] = _track(99, "vehicle", pts)

    events = build_events(tracks, FRAME_SHAPE, duration=duration + 1)
    labels = [e[2] for e in events]
    assert "wrong_way" in labels


def test_accident_fires_on_sustained_vehicle_vehicle_overlap():
    duration = rules_mod.MIN_ACCIDENT_OVERLAP_SEC + 1.0
    # Two vehicle boxes centered at the same point -> full overlap -> IoU=1.
    pts_a = [(t, 100.0, 100.0, 40.0, 40.0) for t in _times(0.0, duration, 0.1)]
    pts_b = [(t, 105.0, 100.0, 40.0, 40.0) for t in _times(0.0, duration, 0.1)]
    tracks = {1: _track(1, "vehicle", pts_a), 2: _track(2, "vehicle", pts_b)}
    events = build_events(tracks, FRAME_SHAPE, duration=duration + 1)
    labels = [e[2] for e in events]
    assert "accident" in labels


def test_accident_does_not_fire_on_unknown_blob_overlap():
    """Regression test for the audit finding: unknown/unknown MOG2 noise-blob
    overlaps must not manufacture a false accident.
    """
    duration = rules_mod.MIN_ACCIDENT_OVERLAP_SEC + 1.0
    pts_a = [(t, 100.0, 100.0, 40.0, 40.0) for t in _times(0.0, duration, 0.1)]
    pts_b = [(t, 105.0, 100.0, 40.0, 40.0) for t in _times(0.0, duration, 0.1)]
    tracks = {1: _track(1, "unknown", pts_a), 2: _track(2, "unknown", pts_b)}
    events = build_events(tracks, FRAME_SHAPE, duration=duration + 1)
    labels = [e[2] for e in events]
    assert "accident" not in labels


def test_accident_does_not_fire_on_single_frame_overlap_blip():
    """A one-sample overlap (below MIN_ACCIDENT_OVERLAP_SEC) is MOG2 blob-
    merge noise, not a real collision. Uses continuous, evenly-spaced
    samples (like a real analysis pass) so the gap between "overlapping"
    and "separated" samples is realistic -- a sparse/irregular sample
    spacing would let the rule's "assume contact persisted until the next
    sample" logic manufacture an artificially long duration, which is not
    what this test is trying to isolate.
    """
    step = 0.1
    n = 20
    pts_a = [(round(i * step, 3), 100.0, 100.0, 40.0, 40.0) for i in range(n)]
    # Overlapping only at t=0.0, then immediately separated for every
    # subsequent sample.
    pts_b = [(0.0, 105.0, 100.0, 40.0, 40.0)] + [
        (round(i * step, 3), 100.0, 500.0, 40.0, 40.0) for i in range(1, n)
    ]
    tracks = {1: _track(1, "vehicle", pts_a), 2: _track(2, "vehicle", pts_b)}
    events = build_events(tracks, FRAME_SHAPE, duration=n * step)
    labels = [e[2] for e in events]
    assert "accident" not in labels


def test_congestion_fires_when_multiple_vehicles_stall():
    duration = rules_mod.CONGESTION_MIN_SEC + 3.0
    tracks = {}
    for i in range(4):
        pts = [(t, 100.0 + i * 20, 200.0, 30.0, 20.0) for t in _times(0.0, duration, 0.5)]
        tracks[i] = _track(i, "vehicle", pts)
    events = build_events(tracks, FRAME_SHAPE, duration=duration + 1)
    labels = [e[2] for e in events]
    assert "congestion" in labels


def test_empty_tracks_returns_no_events():
    assert build_events({}, FRAME_SHAPE, duration=30.0) == []


def _times(start, end, step):
    t = start
    out = []
    while t <= end:
        out.append(round(t, 3))
        t += step
    return out
