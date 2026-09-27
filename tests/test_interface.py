"""solution.py must expose exactly the interface the (official, unmodified)
run_submission.py imports and calls. These tests fail loudly if that ever
drifts, instead of the drift only surfacing as a mysterious empty-prediction
run against the hidden test set.
"""
import inspect

import evaluate
import solution


def test_classes_match_official_exactly():
    assert solution.CLASSES == evaluate.OFFICIAL_CLASSES, (
        "solution.CLASSES must match evaluate.OFFICIAL_CLASSES exactly, in order -- "
        "the spec allows removing ids, never adding or reordering them"
    )


def test_classes_are_the_14_official_strings():
    expected = [
        "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
        "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
        "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
    ]
    assert solution.CLASSES == expected


def test_detect_events_signature():
    sig = inspect.signature(solution.detect_events)
    params = list(sig.parameters)
    assert params == ["video_path"]


def test_risk_estimator_has_required_methods():
    est = solution.RiskEstimator()
    assert hasattr(est, "reset") and callable(est.reset)
    assert hasattr(est, "step") and callable(est.step)
    reset_sig = inspect.signature(est.reset)
    assert list(reset_sig.parameters) == ["meta"]
    step_sig = inspect.signature(est.step)
    assert list(step_sig.parameters) == ["frame", "t_sec"]


def test_risk_estimator_step_returns_float_in_unit_interval():
    import numpy as np

    est = solution.RiskEstimator()
    est.reset({"video_id": "x", "fps": 25.0, "width": 64, "height": 48, "n_frames": 1})
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    score = est.step(frame, 0.0)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_detect_events_on_missing_file_returns_empty_list_not_exception():
    # run_submission.py already catches exceptions from detect_events, but a
    # well-behaved implementation shouldn't need that safety net for the
    # common case of an unreadable path.
    result = solution.detect_events("this_file_does_not_exist_12345.mp4")
    assert result == []
