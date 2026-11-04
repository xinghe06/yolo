#!/usr/bin/env python3
"""Validate a structurally pruned YOLOv8 checkpoint on a detection dataset."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = PROJECT_ROOT / "third_party" / "ultralytics"
for source_root in (PROJECT_ROOT, ULTRALYTICS_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from pruning.yolov8_adapter import PrunableC2f  # noqa: F401
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "pruning" / "val")
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    data = args.data.expanduser().resolve()
    project = args.project.expanduser().resolve()
    for path in (weights, data):
        if not path.is_file():
            raise FileNotFoundError(path)

    result = YOLO(str(weights)).val(
        data=str(data),
        imgsz=640,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        iou=0.7,
        max_det=300,
        project=str(project),
        name=args.name,
        plots=False,
    )
    metrics = {
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
    }
    output = Path(result.save_dir) / "metrics.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {output}")


if __name__ == "__main__":
    main()

