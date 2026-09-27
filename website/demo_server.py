"""Local live-demo server for Team Genz.

A single-file Flask app that:
  1. Serves website/index.html and website/assets/* as plain static files.
  2. Accepts an uploaded .mp4 (POST /api/detect_events, multipart field
     "video"), runs the repo's own solution.detect_events() on it in the
     background, and lets the client poll for the result
     (GET /api/detect_events/status/<job_id>) as
     {"status": "running"} / {"status": "done", "events": [...]} /
     {"status": "error", "error": "..."}.

It imports solution.py from the repository root (this file adds that root to
sys.path -- it does not duplicate or reimplement any detection logic).

Why a job queue instead of blocking the request until detection finishes:
this demo is hosted on a shared cPanel/Passenger plan whose reverse proxy
enforces its own short request timeout, independent of anything this process
sets. Empirically, processing a real ~90s 4K-sourced clip synchronously
either hung past that proxy timeout with no response at all, or failed
outright after ~15-17s -- well under what solution.detect_events() itself
actually needs. Returning a job id immediately (fast, tiny response) and
letting the client poll a status endpoint (also fast and tiny) sidesteps
that proxy timeout entirely: only the background thread's own processing
time is unbounded by HTTP, bounded instead by JOB_TIMEOUT_SEC below.

Run it:
    pip install -r website/requirements.txt   # installs flask only
    pip install -r requirements.txt            # repo root: opencv-python, numpy
    python website/demo_server.py
    # then open http://localhost:5000

This process only runs on your own machine; it is not deployed or reachable
from the internet. See website/README.md for the manual step of hosting it
publicly (Vercel / Netlify / Hugging Face Spaces / etc.) before a deadline --
that has NOT been done and is not claimed anywhere on the page.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid

from flask import Flask, jsonify, request, send_from_directory

WEBSITE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(WEBSITE_DIR, ".."))
WORKER_SCRIPT = os.path.join(WEBSITE_DIR, "_detect_worker.py")

# So `import solution` (and its own `from src...` imports) resolve to the
# repository root, exactly as run_submission.py does it.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MAX_CONTENT_BYTES = 200 * 1024 * 1024  # 200 MB, matches the limit stated on the page
# Upload-side duration cap. Detection itself runs in the background (see
# module docstring), so this is not a request-timeout constraint -- it's a
# sanity limit on how long a demo visitor should have to wait for a result.
MAX_DURATION_SEC = 150.0
# Hard ceiling on the background job itself (well above what 150s of footage
# should need with the pipeline's own internal downscaling); a safety valve
# against a pathological input hanging forever, not a value we expect to hit.
JOB_TIMEOUT_SEC = 300.0

JOBS_DIR = os.path.join(tempfile.gettempdir(), "genz_demo_jobs")
JOB_RETENTION_SEC = 3600.0  # best-effort cleanup of old job directories

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_BYTES


@app.errorhandler(413)
def too_large(_err):
    return jsonify({"error": "Upload exceeds the 200MB limit for this demo."}), 413


@app.route("/")
def index():
    return send_from_directory(WEBSITE_DIR, "index.html")


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(os.path.join(WEBSITE_DIR, "assets"), filename)


def _probe_duration_sec(video_path: str):
    """Best-effort duration check using OpenCV. Returns None if it can't be
    determined (e.g. opencv missing or an unreadable file) -- in that case we
    let detect_events() itself be the source of truth rather than blocking.
    """
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0 or n_frames <= 0:
        return None
    return n_frames / fps


def _job_paths(job_id: str):
    job_dir = os.path.join(JOBS_DIR, job_id)
    return job_dir, os.path.join(job_dir, "input.mp4"), os.path.join(job_dir, "status.json")


def _write_status(status_path: str, payload: dict):
    tmp_path = status_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, status_path)


def _cleanup_old_jobs():
    if not os.path.isdir(JOBS_DIR):
        return
    now = time.time()
    try:
        for name in os.listdir(JOBS_DIR):
            path = os.path.join(JOBS_DIR, name)
            try:
                if now - os.path.getmtime(path) > JOB_RETENTION_SEC:
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                continue
    except OSError:
        pass


def _run_job(job_id: str, video_path: str, status_path: str):
    try:
        result = subprocess.run(
            [sys.executable, WORKER_SCRIPT, video_path],
            capture_output=True, text=True, timeout=JOB_TIMEOUT_SEC,
        )
        if result.returncode != 0:
            print("detect worker failed:\n", result.stderr, file=sys.stderr)
            _write_status(status_path, {
                "status": "error",
                "error": "solution.detect_events() raised an exception while processing this file.",
            })
            return
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        _write_status(status_path, {"status": "done", "events": payload["events"]})
    except subprocess.TimeoutExpired:
        _write_status(status_path, {
            "status": "error",
            "error": f"Processing took longer than {JOB_TIMEOUT_SEC:.0f}s and was stopped. Try a shorter clip.",
        })
    except Exception:
        traceback.print_exc()
        _write_status(status_path, {
            "status": "error",
            "error": "solution.detect_events() raised an exception while processing this file.",
        })
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass


@app.route("/api/detect_events", methods=["POST"])
def detect_events_endpoint():
    upload = request.files.get("video")
    if upload is None or not upload.filename:
        return jsonify({"error": "No file uploaded (expected form field 'video')."}), 400

    if not upload.filename.lower().endswith(".mp4"):
        return jsonify({"error": "Only .mp4 files are supported by this demo."}), 400

    _cleanup_old_jobs()

    job_id = uuid.uuid4().hex
    job_dir, video_path, status_path = _job_paths(job_id)
    os.makedirs(job_dir, exist_ok=True)

    try:
        upload.save(video_path)

        duration = _probe_duration_sec(video_path)
        # +1.0s tolerance: frame-count/fps duration probing is approximate, so
        # a clip meant to be exactly at the limit can probe a hair over it.
        if duration is not None and duration > MAX_DURATION_SEC + 1.0:
            shutil.rmtree(job_dir, ignore_errors=True)
            return (
                jsonify(
                    {
                        "error": (
                            "Video is {:.0f}s long; this demo is limited to "
                            "{:.0f}s so it stays responsive on this host."
                        ).format(duration, MAX_DURATION_SEC)
                    }
                ),
                400,
            )

        _write_status(status_path, {"status": "running"})

        # Run detection in a fresh subprocess rather than in-process: OpenCV
        # and numpy often don't release memory back to the OS even after
        # Python's own GC runs, and this Flask worker is long-lived on the
        # host (Passenger reuses it across requests). A subprocess fully
        # releases its memory to the OS on exit, which matters on
        # memory-constrained shared hosting -- see the module docstring in
        # _detect_worker.py. It runs in a background thread so this request
        # can return immediately with just the job id (see module docstring).
        t = threading.Thread(target=_run_job, args=(job_id, video_path, status_path), daemon=True)
        t.start()

        return jsonify({"job_id": job_id}), 202

    except Exception:
        traceback.print_exc()
        shutil.rmtree(job_dir, ignore_errors=True)
        return jsonify({"error": "Could not accept this upload."}), 500


@app.route("/api/detect_events/status/<job_id>")
def detect_events_status(job_id):
    _, _, status_path = _job_paths(job_id)
    if not os.path.exists(status_path):
        return jsonify({"status": "error", "error": "Unknown or expired job id."}), 404
    try:
        with open(status_path) as f:
            return jsonify(json.load(f))
    except (OSError, ValueError):
        return jsonify({"status": "running"})


if __name__ == "__main__":
    # PORT is set by hosting platforms (Hugging Face Spaces' Docker SDK uses
    # 7860 by default; Render/Railway inject their own PORT). Falls back to
    # 5000 for a plain local run.
    port = int(os.environ.get("PORT", 5000))
    print("Team Genz demo server")
    print("Repo root added to sys.path:", REPO_ROOT)
    print(f"Open http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
