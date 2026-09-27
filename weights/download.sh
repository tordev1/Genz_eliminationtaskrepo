#!/usr/bin/env bash
# Optional. Fetches a small open-weights YOLOv8n checkpoint so
# src/detector.py can use real object detection instead of the
# dependency-free motion-detector baseline. Run once, with internet, before
# the offline evaluation run -- the baseline works with zero weights.
set -euo pipefail
cd "$(dirname "$0")"
if [ -f yolov8n.pt ]; then
  echo "weights/yolov8n.pt already present."
  exit 0
fi
python - <<'PY'
from ultralytics import YOLO
YOLO("yolov8n.pt")  # triggers Ultralytics' own download into the cwd
PY
echo "Downloaded weights/yolov8n.pt"
