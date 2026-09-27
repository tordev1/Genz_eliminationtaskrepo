"""Local live-demo server for Team Genz.

A single-file Flask app that:
  1. Serves website/index.html and website/assets/* as plain static files.
  2. Accepts an uploaded .mp4 (POST /api/detect_events, multipart field
     "video"), runs the repo's own solution.detect_events() on it, and
     returns the real result as JSON: {"events": [[start_sec, end_sec, label], ...]}.

It imports solution.py from the repository root (this file adds that root to
sys.path -- it does not duplicate or reimplement any detection logic).

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

import os
import sys
import tempfile
import traceback

from flask import Flask, jsonify, request, send_from_directory

WEBSITE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(WEBSITE_DIR, ".."))

# So `import solution` (and its own `from src...` imports) resolve to the
# repository root, exactly as run_submission.py does it.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MAX_CONTENT_BYTES = 200 * 1024 * 1024  # 200 MB, matches the limit stated on the page
MAX_DURATION_SEC = 120.0  # 2 minutes, verified server-side via an actual video probe

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


@app.route("/api/detect_events", methods=["POST"])
def detect_events_endpoint():
    upload = request.files.get("video")
    if upload is None or not upload.filename:
        return jsonify({"error": "No file uploaded (expected form field 'video')."}), 400

    if not upload.filename.lower().endswith(".mp4"):
        return jsonify({"error": "Only .mp4 files are supported by this demo."}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False, dir=None
        ) as tmp_file:
            upload.save(tmp_file)
            tmp_path = tmp_file.name

        duration = _probe_duration_sec(tmp_path)
        if duration is not None and duration > MAX_DURATION_SEC:
            return (
                jsonify(
                    {
                        "error": (
                            "Video is {:.0f}s long; this demo is limited to "
                            "{:.0f}s (2 minutes) so it stays responsive on CPU."
                        ).format(duration, MAX_DURATION_SEC)
                    }
                ),
                400,
            )

        import solution  # imported lazily so a missing opencv/numpy fails per-request, not at server start

        events = solution.detect_events(tmp_path)

        # Defensive: solution.py should already return [start, end, label]
        # triples, but the demo endpoint is a good place to guarantee it,
        # the same way run_submission.py's _clean_events does for the harness.
        safe_events = []
        for ev in events:
            try:
                s, e, label = ev
                safe_events.append([float(s), float(e), str(label)])
            except Exception:
                continue

        return jsonify({"events": safe_events})

    except Exception:
        # Full traceback goes to the server log, not the HTTP response --
        # this file is meant to be deployed publicly (see website/README.md),
        # and a stack trace can leak local paths to the client.
        traceback.print_exc()
        return (
            jsonify({"error": "solution.detect_events() raised an exception while processing this file."}),
            500,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    # PORT is set by hosting platforms (Hugging Face Spaces' Docker SDK uses
    # 7860 by default; Render/Railway inject their own PORT). Falls back to
    # 5000 for a plain local run.
    port = int(os.environ.get("PORT", 5000))
    print("Team Genz demo server")
    print("Repo root added to sys.path:", REPO_ROOT)
    print(f"Open http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
