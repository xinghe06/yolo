#!/usr/bin/env python3
"""Structurally prune YOLOv8 with Random, L1, or LAMP importance.

All methods use the same Torch-Pruning DepGraph backend, protected Detect
outputs, global ranking, channel rounding and target MAC speed-up. Only the
importance estimator changes, which makes their results directly comparable.
"""

import argparse
import hashlib
import json
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = PROJECT_ROOT / "third_party" / "ultralytics"
LOCAL_TP_ROOT = PROJECT_ROOT / "Torch-Pruning"
for source_root in (PROJECT_ROOT, ULTRALYTICS_ROOT, LOCAL_TP_ROOT):
    if source_root.exists() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import numpy as np
import torch
import torch_pruning as tp
from ultralytics import YOLO, __version__ as ultralytics_version

from pruning.yolov8_adapter import (
    convert_c2f_to_prunable,
    fixed_output_layers,
    forward_raw_detect,
    output_max_abs_diff,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("random", "l1", "lamp"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0", help="CUDA index such as 0, or cpu")
    parser.add_argument("--target-speedup", type=float, default=1.5)
    parser.add_argument("--iterative-steps", type=int, default=100)
    parser.add_argument("--max-pruning-ratio", type=float, default=0.8)
    parser.add_argument("--round-to", type=int, default=8)
    parser.add_argument("--trace-imgsz", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--equivalence-tolerance", type=float, default=1e-4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(value: str) -> torch.device:
    if value.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
        return torch.device(f"cuda:{value}")
    return torch.device(value)


def make_importance(method: str):
    if method == "random":
        return tp.importance.RandomImportance()
    if method == "l1":
        return tp.importance.MagnitudeImportance(p=1, group_reduction="first", normalizer=None)
    if method == "lamp":
        return tp.importance.LAMPImportance(p=2, group_reduction="first")
    raise ValueError(method)


def module_names(model, modules):
    ids = {id(module) for module in modules}
    return [name for name, module in model.named_modules() if id(module) in ids]


def save_checkpoint(model, source_checkpoint, destination: Path) -> None:
    """Save the complete changed structure in Ultralytics checkpoint format."""
    if hasattr(model, "criterion"):
        delattr(model, "criterion")
    train_args = source_checkpoint.get("train_args", {}) if isinstance(source_checkpoint, dict) else {}
    checkpoint = {
        "epoch": -1,
        "best_fitness": None,
        "model": deepcopy(model).half().cpu(),
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": train_args,
        "date": datetime.now().isoformat(),
        "version": ultralytics_version,
    }
    torch.save(checkpoint, destination)


def main():
    args = parse_args()
    args.weights = args.weights.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    if args.target_speedup <= 1.0:
        raise ValueError("--target-speedup must be greater than 1")
    if not 0 < args.max_pruning_ratio < 1:
        raise ValueError("--max-pruning-ratio must be between 0 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    wrapper = YOLO(str(args.weights))
    model = wrapper.model.float().to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    example = torch.randn(1, 3, args.trace_imgsz, args.trace_imgsz, device=device)

    # Verify the weight transfer on CPU FP32. CuDNN may choose different
    # algorithms (and TF32 accumulation paths) for one wide convolution and
    # two half-width convolutions, producing milliscale differences even when
    # their weights are exactly equivalent. CPU verification isolates the
    # structural conversion from that device-specific numerical noise.
    check_model = deepcopy(model).float().cpu().eval()
    check_example = example.detach().float().cpu()
    before_conversion = forward_raw_detect(check_model, check_example)
    check_converted = convert_c2f_to_prunable(check_model)
    after_conversion = forward_raw_detect(check_model, check_example)
    conversion_diff = output_max_abs_diff(before_conversion, after_conversion)
    del check_model, check_example, before_conversion, after_conversion
    if conversion_diff > args.equivalence_tolerance:
        raise RuntimeError(
            f"C2f conversion changed model output by {conversion_diff:.6g}; "
            f"tolerance is {args.equivalence_tolerance:.6g}"
        )

    converted = convert_c2f_to_prunable(model)
    if converted != check_converted:
        raise RuntimeError(f"C2f conversion count changed between check and target: {check_converted} -> {converted}")

    protected = fixed_output_layers(model)
    baseline_macs, baseline_params = tp.utils.count_ops_and_params(model, example)
    importance = make_importance(args.method)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example,
        importance=importance,
        global_pruning=True,
        pruning_ratio=args.max_pruning_ratio,
        iterative_steps=args.iterative_steps,
        ignored_layers=protected,
        round_to=args.round_to,
        max_pruning_ratio=args.max_pruning_ratio,
    )

    history = []
    for step in range(1, args.iterative_steps + 1):
        pruner.step()
        with torch.no_grad():
            model(example)
        macs, params = tp.utils.count_ops_and_params(model, example)
        speedup = float(baseline_macs) / float(macs)
        record = {"step": step, "macs": int(macs), "params": int(params), "speedup": speedup}
        history.append(record)
        print(
            f"step {step:03d}: MACs={macs / 1e9:.6f}G "
            f"params={params / 1e6:.6f}M speedup={speedup:.4f}"
        )
        if speedup >= args.target_speedup:
            break
    else:
        raise RuntimeError(
            f"Target speed-up {args.target_speedup} was not reached; "
            f"last speed-up was {history[-1]['speedup']:.4f}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    source_checkpoint = torch.load(args.weights, map_location="cpu")
    checkpoint_path = args.output / "pruned.pt"
    model_path = args.output / "pruned_model.pth"
    history_path = args.output / "pruning_history.pt"
    save_checkpoint(model, source_checkpoint, checkpoint_path)
    torch.save(deepcopy(model).float().cpu(), model_path)
    torch.save(pruner.pruning_history(), history_path)

    # This in-process load catches malformed checkpoint contents. The documented
    # smoke test also reloads it from a completely new Python process.
    reloaded = YOLO(str(checkpoint_path)).model.float().to(device).eval()
    with torch.no_grad():
        reloaded(example)

    scale_to_640 = (640 / args.trace_imgsz) ** 2
    final = history[-1]
    metadata = {
        "method": args.method,
        "seed": args.seed,
        "source_weights": str(args.weights),
        "source_sha256": sha256(args.weights),
        "ultralytics_version": ultralytics_version,
        "torch_pruning_version": getattr(tp, "__version__", "unknown"),
        "converted_c2f_modules": converted,
        "conversion_max_abs_diff": conversion_diff,
        "trace_imgsz": args.trace_imgsz,
        "target_speedup": args.target_speedup,
        "achieved_speedup": final["speedup"],
        "iterative_steps_configured": args.iterative_steps,
        "steps_used": len(history),
        "round_to": args.round_to,
        "max_pruning_ratio": args.max_pruning_ratio,
        "baseline_macs": int(baseline_macs),
        "baseline_params": int(baseline_params),
        "pruned_macs": final["macs"],
        "pruned_params": final["params"],
        "baseline_gflops_640": 2 * baseline_macs * scale_to_640 / 1e9,
        "pruned_gflops_640": 2 * final["macs"] * scale_to_640 / 1e9,
        "protected_layers": module_names(model, protected),
        "history": history,
    }
    (args.output / "pruning_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved pruned checkpoint: {checkpoint_path}")
    print(f"Saved pruning metadata: {args.output / 'pruning_metrics.json'}")


if __name__ == "__main__":
    main()
