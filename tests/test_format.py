"""Exercises the actual official evaluate.validate() (not a reimplementation
of it) against hand-built good and bad predictions, so we know our own
output will pass the exact check the organizers run.
"""
import evaluate


def _pred(events, risk=None):
    return {"team": "Genz", "videos": {"v.mp4": {"events": events, "risk": risk or []}}}


def test_valid_predictions_pass():
    errors, warnings = evaluate.validate(_pred([[1.0, 2.0, "accident"], [5.0, 8.0, "congestion"]]))
    assert errors == []


def test_overlapping_same_class_is_rejected():
    errors, _ = evaluate.validate(_pred([[1.0, 5.0, "accident"], [3.0, 6.0, "accident"]]))
    assert errors, "overlapping same-class segments must be an error"


def test_touching_same_class_is_not_overlap():
    # [1,2] and [2,3] touch at a point but don't share any open interval --
    # the spec's matching predicate is strict, so this must be legal.
    errors, _ = evaluate.validate(_pred([[1.0, 2.0, "accident"], [2.0, 3.0, "accident"]]))
    assert errors == []


def test_different_classes_may_overlap():
    errors, _ = evaluate.validate(_pred([[1.0, 5.0, "accident"], [2.0, 6.0, "wrong_way"]]))
    assert errors == []


def test_unknown_label_is_rejected():
    errors, _ = evaluate.validate(_pred([[1.0, 2.0, "not_a_real_class"]]))
    assert errors


def test_start_not_less_than_end_is_rejected():
    errors, _ = evaluate.validate(_pred([[5.0, 5.0, "accident"]]))
    assert errors
    errors, _ = evaluate.validate(_pred([[5.0, 2.0, "accident"]]))
    assert errors


def test_malformed_event_shape_is_rejected():
    errors, _ = evaluate.validate(_pred([[1.0, 2.0]]))  # missing label
    assert errors
    errors, _ = evaluate.validate(_pred([[1.0, "two", "accident"]]))  # non-numeric end
    assert errors


def test_missing_ground_truth_video_is_rejected():
    gt = {"v.mp4": {"duration": 10.0, "fps": 25.0, "events": []}, "other.mp4": {"duration": 5.0, "fps": 25.0, "events": []}}
    errors, _ = evaluate.validate(_pred([]), gt=gt)
    assert any("other.mp4" in e for e in errors)


def test_risk_score_out_of_range_is_rejected():
    errors, _ = evaluate.validate(_pred([], risk=[[0.0, 1.5]]))
    assert errors


def test_risk_non_decreasing_time_is_rejected():
    errors, _ = evaluate.validate(_pred([], risk=[[1.0, 0.1], [0.5, 0.2]]))
    assert errors


def test_path_like_video_key_is_rejected():
    pred = {"team": "Genz", "videos": {"folder/v.mp4": {"events": [], "risk": []}}}
    errors, _ = evaluate.validate(pred)
    assert errors
