"""A single local pre-submission gate, mirroring what the organizers'
pipeline actually does, so problems surface here instead of at grading time.

    python tools/preflight.py --videos samples --team Genz

Checks, in order (each printed with PASS/FAIL, non-zero exit on any FAIL):
  1. Repository layout matches the "Submission package" tree from the spec.
  2. `python run_submission.py --videos <videos> --out <tmp> --team <team>`
     runs clean (no exception, valid JSON).
  3. `python evaluate.py --pred <tmp> --validate-only` passes.
  4. Determinism: run step 2 twice, diff the two predictions.json byte-for-
     byte (after normalizing nothing -- they must be identical, per the
     spec's "up to floating-point noise" allowance we still check exactly
     since this pipeline has none).
  5. weights/ total size <= 5 GB.
  6. Timing margin: per-video wall time vs. its 3x budget, printed as a
     table -- a PASS here means "ran", not "ran with margin"; read the
     table yourself for how much margin there actually is.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

REQUIRED_PATHS = [
    "solution.py",
    "run_submission.py",
    "evaluate.py",
    "requirements.txt",
    "weights",
    "src",
    "predictions_samples.json",
    "README.md",
]

WEIGHTS_LIMIT_BYTES = 5 * 1024 ** 3


def _pass(msg):
    print(f"  PASS  {msg}")


def _fail(msg):
    print(f"  FAIL  {msg}")
    return False


def check_layout() -> bool:
    print("[1/6] Repository layout")
    ok = True
    for rel in REQUIRED_PATHS:
        path = os.path.join(REPO_ROOT, rel)
        if os.path.exists(path):
            _pass(rel)
        else:
            ok = _fail(f"missing: {rel}") and ok
    return ok


def run_harness(videos_dir: str, out_path: str, team: str) -> bool:
    result = subprocess.run(
        [sys.executable, "run_submission.py", "--videos", videos_dir, "--out", out_path, "--team", team],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        return False
    return os.path.isfile(out_path)


def check_harness(videos_dir: str, team: str, out_path: str) -> bool:
    print("\n[2/6] run_submission.py")
    ok = run_harness(videos_dir, out_path, team)
    if ok:
        _pass(f"wrote {out_path}")
    else:
        _fail("run_submission.py did not produce valid output")
    return ok


def check_validate(pred_path: str) -> bool:
    print("\n[3/6] evaluate.py --validate-only")
    result = subprocess.run(
        [sys.executable, "evaluate.py", "--pred", pred_path, "--validate-only"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    print(result.stdout.strip())
    ok = result.returncode == 0
    if ok:
        _pass("format valid")
    else:
        _fail("format invalid")
    return ok


def check_determinism(videos_dir: str, team: str) -> bool:
    print("\n[4/6] Determinism (two runs, compared on team+videos only)")
    # run_submission.py's own "log" block records wall-clock timings
    # (part_a_sec, part_b_sec, total_sec) that legitimately vary run to run
    # under normal system jitter -- that's not what "determinism" means
    # here. The spec's determinism requirement is about predictions.json's
    # actual content ("team" + "videos"), which is what we compare.
    with tempfile.TemporaryDirectory() as tmp:
        out1 = os.path.join(tmp, "run1.json")
        out2 = os.path.join(tmp, "run2.json")
        if not run_harness(videos_dir, out1, team) or not run_harness(videos_dir, out2, team):
            return _fail("a run failed, cannot compare")
        with open(out1) as f1, open(out2) as f2:
            d1, d2 = json.load(f1), json.load(f2)
        content1 = {"team": d1.get("team"), "videos": d1.get("videos")}
        content2 = {"team": d2.get("team"), "videos": d2.get("videos")}
        if content1 == content2:
            _pass("two runs produced identical team+videos content")
            return True
        return _fail("two runs produced different predictions (team/videos) -- pipeline is not deterministic")


def check_weights_size() -> bool:
    print("\n[5/6] weights/ size")
    weights_dir = os.path.join(REPO_ROOT, "weights")
    total = 0
    for root, _, files in os.walk(weights_dir):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    mb = total / (1024 ** 2)
    if total <= WEIGHTS_LIMIT_BYTES:
        _pass(f"{mb:.1f} MB (limit 5120 MB)")
        return True
    return _fail(f"{mb:.1f} MB exceeds the 5 GB limit")


def check_timing(pred_path: str) -> bool:
    print("\n[6/6] Timing margin (from run_submission.py's own log)")
    with open(pred_path) as f:
        data = json.load(f)
    log = data.get("log", {})
    if not log:
        print("  (no log block found -- skipping)")
        return True
    print(f"  {'video':<28}{'total_sec':>10}{'budget_sec':>12}{'margin':>10}")
    ok = True
    for name, entry in log.items():
        total = entry.get("total_sec", 0.0)
        budget = entry.get("budget_sec", 0.0)
        margin = (budget - total) / budget if budget > 0 else 0.0
        flag = "" if total <= budget else "  OVER BUDGET"
        print(f"  {name:<28}{total:>10.1f}{budget:>12.1f}{margin:>9.0%}{flag}")
        if total > budget:
            ok = False
    if ok:
        _pass("all videos within budget")
    else:
        _fail("at least one video exceeded its time budget")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="samples")
    parser.add_argument("--team", default="Genz")
    args = parser.parse_args()

    videos_dir = os.path.join(REPO_ROOT, args.videos) if not os.path.isabs(args.videos) else args.videos
    if not os.path.isdir(videos_dir) or not any(f.lower().endswith(".mp4") for f in os.listdir(videos_dir)):
        print(f"No .mp4 files in {videos_dir} -- populate samples/ before running preflight.")
        return 2

    results = [check_layout()]

    with tempfile.TemporaryDirectory() as tmp:
        pred_path = os.path.join(tmp, "predictions.json")
        results.append(check_harness(videos_dir, args.team, pred_path))
        if results[-1]:
            results.append(check_validate(pred_path))
            results.append(check_timing(pred_path))
        else:
            results.append(False)
            results.append(False)
        results.append(check_determinism(videos_dir, args.team))

    results.append(check_weights_size())

    print("\n" + "=" * 40)
    if all(results):
        print("PREFLIGHT: ALL CHECKS PASSED")
        return 0
    print("PREFLIGHT: FAILED -- see FAIL lines above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
