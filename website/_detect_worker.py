"""Runs solution.detect_events() in its own process and prints the result as
JSON on stdout. demo_server.py spawns this as a subprocess per request
instead of calling detect_events() in-process, so memory used by
OpenCV/numpy (which often isn't returned to the OS even after Python's own
GC runs) is fully released back to the OS when this process exits --
important on memory-constrained shared hosting, where a long-lived Flask
worker handling many requests in-process can accumulate memory across
requests until the host's process-memory limit kills it.
"""
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import solution  # noqa: E402


def main():
    video_path = sys.argv[1]
    events = solution.detect_events(video_path)
    safe_events = []
    for ev in events:
        try:
            s, e, label = ev
            safe_events.append([float(s), float(e), str(label)])
        except Exception:
            continue
    print(json.dumps({"events": safe_events}))


if __name__ == "__main__":
    main()
