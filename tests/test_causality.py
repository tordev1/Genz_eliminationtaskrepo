"""Part B's causality guarantee ("RiskEstimator.step uses only the frames
it has received; reading the video file inside it ... is a violation") must
be structural, not just a docstring claim. This scans the actual source of
every module reachable from src/risk.py and fails if any of them ever
references cv2.VideoCapture -- the one call that would let Part B peek at
frames it hasn't been handed yet.
"""
import ast
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# The causal closure: src/risk.py and everything it imports from src/.
# Deliberately NOT solution.py or run_submission.py, which legitimately open
# video files for Part A / the harness -- this test only guards the Part B
# call graph.
CAUSAL_CLOSURE_FILES = [
    "src/risk.py",
    "src/detector.py",
    "src/tracker.py",
    "src/road_model.py",
    "src/geometry.py",
    "src/tunable.py",
]

FORBIDDEN_SUBSTRINGS = ["VideoCapture", "imageio.get_reader", "decord.VideoReader"]


def test_causal_closure_never_opens_a_video():
    offenders = []
    for rel in CAUSAL_CLOSURE_FILES:
        path = os.path.join(REPO_ROOT, rel)
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        for needle in FORBIDDEN_SUBSTRINGS:
            if needle in source:
                offenders.append(f"{rel}: contains {needle!r}")
    assert not offenders, (
        "Found a video-opening call inside Part B's causal call graph "
        f"(breaks causality if reachable from RiskEstimator.step): {offenders}"
    )


def test_risk_estimator_class_has_no_file_io_calls():
    """A second, independent check via AST rather than substring search:
    walk solution.RiskEstimator's methods and its adapter's target
    (CausalRiskEngine) for any `open(...)` call or attribute access named
    VideoCapture.
    """
    path = os.path.join(REPO_ROOT, "src", "risk.py")
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)

    bad_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in ("open", "VideoCapture"):
                bad_calls.append((name, getattr(node, "lineno", "?")))
    assert not bad_calls, f"src/risk.py calls file/video-opening functions directly: {bad_calls}"


def test_reset_rebuilds_state_from_scratch_each_video():
    """No state should survive across videos -- reset() must not merely
    reuse a previous engine's tracker/detector/road model.
    """
    import numpy as np

    import solution

    est = solution.RiskEstimator()
    est.reset({"video_id": "a", "fps": 25.0, "width": 64, "height": 48, "n_frames": 10})
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for i in range(5):
        est.step(frame, i / 25.0)
    tracker_from_video_a = est._engine.tracker  # capture the object, not just a name

    est.reset({"video_id": "b", "fps": 25.0, "width": 64, "height": 48, "n_frames": 10})
    tracker_from_video_b = est._engine.tracker

    assert tracker_from_video_a is not tracker_from_video_b, (
        "reset() must rebuild the tracker from scratch, not reuse the previous video's instance"
    )
    assert len(tracker_from_video_b.tracks) == 0
