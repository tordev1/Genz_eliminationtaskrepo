"""Runs solution.detect_events() in its own process and prints the result as
JSON on stdout. demo_server.py spawns this as a subprocess per request
instead of calling detect_events() in-process, so memory used by
OpenCV/numpy (which often isn't returned to the OS even after Python's own
GC runs) is fully released back to the OS when this process exits --
important on memory-constrained shared hosting, where a long-lived Flask
worker handling many requests in-process can accumulate memory across
requests until the host's process-memory limit kills it.
"""
import os

# Must be set before numpy is imported: numpy's BLAS backend (OpenBLAS) reads
# these at import time and otherwise spawns up to one thread per CPU core.
# This shared host enforces a low RLIMIT_NPROC (max processes+threads for the
# whole account); with even two of these worker processes alive at once, the
# combined thread count blew past that ceiling, causing fork()/pthread_create
# to fail and corrupting numpy's own import for whichever job lost the race
# (confirmed via passenger.log: "pthread_create failed ... Resource
# temporarily unavailable" / "RLIMIT_NPROC -1 current, -1 max" immediately
# preceding a broken numpy import). A single detect_events() call here does
# not need multithreaded BLAS, so pin every backend to one thread.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
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
