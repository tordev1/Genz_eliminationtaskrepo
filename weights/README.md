# weights/

The default pipeline (`src/detector.py` -> `MotionDetector`) needs **no
weights at all** -- it's background subtraction, which is exactly the right
tool for this challenge's fixed, non-moving camera.

## Optional upgrade: real YOLO detections

`src/detector.py` already has a `YoloDetector` wrapper. To turn it on:

1. `pip install ultralytics`
2. Put an open-weights checkpoint here, e.g. `weights/yolov8n.pt`
   (Ultralytics, AGPL-3.0 / available for commercial use per their licence --
   confirm the exact terms before shipping).
3. Nothing else changes -- `solution.py` calls
   `build_default_detector()`, which auto-detects the file and switches
   detectors transparently. Delete the file to fall back to `MotionDetector`.

`download.sh` in this folder fetches that checkpoint with internet access,
per the challenge's "run once, with internet, before evaluation" rule for
weights. It is not required for the baseline to run.
