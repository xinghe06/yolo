#!/usr/bin/env python3
"""Fine-tune a structurally pruned YOLOv8 full-object checkpoint."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = PROJECT_ROOT / "third_party" / "ultralytics"
for source_root in (PROJECT_ROOT, ULTRALYTICS_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

# Import before torch.load/YOLO so pickle can resolve the converted class.
from pruning.yolov8_adapter import PrunableC2f  # noqa: F401
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "pruning" / "finetune")
    parser.add_argument("--name", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.weights = args.weights.expanduser().resolve()
    args.data = args.data.expanduser().resolve()
    args.project = args.project.expanduser().resolve()
    for path in (args.weights, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Passing the changed model object directly prevents DetectionTrainer from
    # rebuilding the original unpruned channel dimensions from the stale YAML.
    model = YOLO(str(args.weights)).model.float()
    if hasattr(model, "criterion"):
        delattr(model, "criterion")
    overrides = {
        "model": str(args.weights),
        "data": str(args.data),
        "epochs": 1 if args.smoke else args.epochs,
        "patience": 1 if args.smoke else args.epochs,
        "batch": 2 if args.smoke else args.batch,
        "imgsz": 640,
        "device": args.device,
        "workers": 0 if args.smoke else args.workers,
        "optimizer": "SGD",
        "pretrained": True,
        "seed": 0,
        "deterministic": True,
        "close_mosaic": 0 if args.smoke else 10,
        "fraction": 0.01 if args.smoke else 1.0,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": 0.0005,
        "warmup_epochs": 0.0 if args.smoke else 1.0,
        "iou": 0.7,
        "max_det": 300,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": False,
        "mode": "train",
        "task": "detect",
    }
    trainer = DetectionTrainer(overrides=overrides)
    trainer.model = model
    trainer.train()


if __name__ == "__main__":
    main()

