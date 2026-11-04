#!/usr/bin/env python3
"""Group-L2: small local pruning step -> fine-tune -> validate -> repeat.

Adapted from the local Torch-Pruning YOLOv8 example to Ultralytics v8.0.239.
No imports from scripts/prune_yolov8.py or pruning/yolov8_adapter.py.
"""

import argparse
import csv
import hashlib
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'third_party/ultralytics', ROOT / 'Torch-Pruning'):
    if path.exists():
        sys.path.insert(0, str(path))

import numpy as np
import torch
import torch_pruning as tp
from ultralytics import YOLO, __version__
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.modules import Detect
from ultralytics.utils.torch_utils import select_device
from pruning.official_group_l2 import (
    FP32PruningTrainer, OfficialC2f, channel_shapes, convert_c2f,
    inference_checkpoint, raw_output,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='0', help='Single CUDA device or cpu')
    p.add_argument('--total-stages', type=int, default=16)
    p.add_argument('--run-stages', type=int, default=1,
                   help='Number of stages to run now; does not change the per-stage ratio')
    p.add_argument('--target-prune-rate', type=float, default=0.5)
    p.add_argument('--epochs-per-stage', type=int, default=10)
    p.add_argument('--max-map-drop', type=float, default=0.2,
                   help='Absolute mAP50-95 drop relative to original baseline (0.2 = 20 points)')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--optimizer', choices=('SGD', 'AdamW', 'auto'), default='SGD')
    p.add_argument('--lr0', type=float, default=0.01)
    p.add_argument('--warmup-epochs', type=float, default=3.0)
    p.add_argument('--close-mosaic', type=int, default=10)
    p.add_argument('--amp', action='store_true', help='Opt into AMP; FP32 is the default')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--resume', action='store_true', help='Continue after the last COMPLETED stage')
    p.add_argument('--smoke', action='store_true',
                   help='1 epoch/stage, 1%% train data, 128px, batch=2; not an accuracy experiment')
    args = p.parse_args()
    if args.total_stages < 1 or args.run_stages < 1 or args.epochs_per_stage < 1:
        p.error('Stage and epoch counts must be positive.')
    if not 0 < args.target_prune_rate < 1 or args.max_map_drop < 0:
        p.error('Require 0 < target-prune-rate < 1 and max-map-drop >= 0.')
    if args.batch < 1 or args.threads < 1 or args.imgsz < 32 or args.imgsz % 32:
        p.error('Use positive batch/threads and an imgsz divisible by 32.')
    if ',' in args.device:
        p.error('Only single-device training is supported.')
    if args.smoke:
        args.epochs_per_stage, args.imgsz, args.batch, args.workers = 1, 128, 2, 0
        args.warmup_epochs, args.close_mosaic, args.amp = 0.0, 0, False
    return args


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def metrics(result):
    return dict(precision=float(result.box.mp), recall=float(result.box.mr),
                map50=float(result.box.map50), map50_95=float(result.box.map))


def validate(checkpoint, name, args):
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    result = YOLO(str(checkpoint)).val(
        data=str(args.data), split='val', imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, conf=0.001, iou=0.7,
        max_det=300, half=False, augment=False, plots=False,
        project=str(args.output / 'val'), name=name, exist_ok=False,
    )
    value = metrics(result)
    write_json(Path(result.save_dir) / 'metrics.json', value)
    return value


def save_summary(args, state):
    rows = [dict(stage=0, phase='baseline', params=state['baseline_params'],
                 gflops=state['baseline_macs'] * 2 / 1e9, mac_ratio=1.0, **state['baseline'])]
    for step in state['stages']:
        for phase in ('pre', 'post'):
            rows.append(dict(stage=step['stage'], phase=phase, params=step['params'],
                             gflops=step['macs'] * 2 / 1e9,
                             mac_ratio=state['baseline_macs'] / step['macs'], **step[phase]))
    with (args.output / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / 'state.json', state)


def main():
    args = parse_args()
    for key in ('weights', 'data', 'output'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if __version__ != '8.0.239':
        raise RuntimeError(f'Requires local Ultralytics v8.0.239, found {__version__}')
    if not args.weights.is_file() or not args.data.is_file():
        raise FileNotFoundError('Check --weights and --data paths.')
    torch.set_num_threads(args.threads)
    # Select CUDA before initializing its RNG. Use logical device 0 afterwards.
    device = select_device(args.device, batch=args.batch)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = check_det_dataset(str(args.data), autodownload=False)
    if dataset.get('download'):
        raise ValueError('Use a local dataset YAML without a download section.')
    config = {k: v for k, v in vars(args).items()
              if k not in ('output', 'resume', 'run_stages', 'weights', 'data', 'device', 'threads', 'workers')}
    config.update(source_sha256=file_hash(args.weights), dataset_sha256=file_hash(args.data),
                  ultralytics=__version__, torch=str(torch.__version__),
                  torch_pruning=getattr(tp, '__version__', 'unknown'),
                  importance='GroupMagnitudeImportance(p=2, group_reduction=mean, normalizer=mean)',
                  global_pruning=False, round_to=None, checkpoint_dtype='float32')
    x = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    ratio = 1 - (1 - args.target_prune_rate) ** (1 / args.total_stages)
    state_path = args.output / 'state.json'
    if args.resume:
        state = json.loads(state_path.read_text())
        # Absolute dataset paths and CUDA builds may differ when moving the
        # complete run to AutoDL. Preserve algorithm/training settings and
        # source checkpoint identity; record the new runtime separately.
        runtime_keys = ('dataset_sha256', 'torch')
        comparable = lambda c: {k: v for k, v in c.items() if k not in runtime_keys}
        if comparable(state['config']) != comparable(config):
            raise ValueError('Resume configuration differs; preserve the original experiment settings.')
        if state['class_names'] != list(dataset['names'].values()):
            raise ValueError('Resume dataset classes differ from the original run.')
        state.setdefault('resume_environments', []).append(
            dict(data=str(args.data), dataset_sha256=config['dataset_sha256'],
                 torch=config['torch'], device=args.device))
        if state.get('stopped'):
            raise RuntimeError('This run already stopped on the post-finetune mAP criterion.')
        current = args.output / state['current_checkpoint']
        if not current.is_file():
            raise FileNotFoundError(f'Completed-stage checkpoint is missing: {current}')
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError('Output is nonempty. Use a new path or --resume for completed stages.')
        args.output.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(args.weights)).model.float().cpu().eval()
        if not isinstance(model.model[-1], Detect) or model.model[-1].nc != dataset['nc']:
            raise ValueError('Use a detection checkpoint trained on this dataset with matching class count.')
        test_x = x[:, :, :128, :128].detach().cpu()
        before = raw_output(model, test_x)
        converted = convert_c2f(model)
        after = raw_output(model, test_x)
        difference = max(float((a - b).abs().max()) for a, b in zip(before, after))
        if difference > 1e-4:
            raise RuntimeError(f'C2f raw-output equivalence failed: {difference}')
        current = args.output / 'converted.pt'
        inference_checkpoint(model, current, getattr(model, 'args', {}))
        model = model.to(device)
        base_macs, base_params = tp.utils.count_ops_and_params(model, x)
        state = dict(config=config, per_stage_prune_rate=ratio,
                     class_names=list(dataset['names'].values()),
                     converted_c2f=converted, conversion_max_abs_diff=difference,
                     baseline_macs=float(base_macs), baseline_params=int(base_params),
                     baseline=validate(current, 'baseline', args), stages=[],
                     current_checkpoint='converted.pt', stopped=False)
        save_summary(args, state)
        del model

    start = len(state['stages']) + 1
    last = min(args.total_stages, start + args.run_stages - 1)
    for stage in range(start, last + 1):
        if not current.is_file():
            raise FileNotFoundError(current)
        directory = args.output / f'stage_{stage:02d}'
        if directory.exists():
            raise FileExistsError(f'{directory} is an incomplete stage. Preserve it; restart in a new output path.')
        directory.mkdir()
        model = YOLO(str(current)).model.float().to(device).eval()
        if hasattr(model, 'criterion'):
            delattr(model, 'criterion')
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        head = model.model[-1]
        head_outputs = {n: m.out_channels for n, m in head.named_modules()
                        if isinstance(m, torch.nn.Conv2d)}
        pruner = tp.pruner.GroupNormPruner(
            model, x, importance=tp.importance.GroupMagnitudeImportance(p=2),
            global_pruning=False, iterative_steps=1, pruning_ratio=ratio,
            ignored_layers=[head], round_to=None,
        )
        pruner.step()
        if head_outputs != {n: m.out_channels for n, m in head.named_modules()
                            if isinstance(m, torch.nn.Conv2d)}:
            raise RuntimeError('A protected Detect output width changed.')
        macs, params = tp.utils.count_ops_and_params(model, x)
        shapes = channel_shapes(model)
        torch.save(pruner.pruning_history(), directory / 'pruning_history.pt')
        del pruner
        pre_checkpoint = directory / 'pruned.pt'
        inference_checkpoint(model, pre_checkpoint, getattr(model, 'args', {}))
        record = dict(stage=stage, input_checkpoint=str(current.relative_to(args.output)),
                      macs=float(macs), params=int(params), shapes=shapes,
                      pre=validate(pre_checkpoint, f'stage_{stage:02d}_pre', args))
        write_json(directory / 'pre_metrics.json', record)
        overrides = dict(
            model=str(pre_checkpoint), data=str(args.data), task='detect', mode='train',
            epochs=args.epochs_per_stage, patience=args.epochs_per_stage + 1,
            imgsz=args.imgsz, batch=args.batch, workers=args.workers, device=args.device,
            optimizer=args.optimizer, lr0=args.lr0, lrf=0.01, momentum=0.937,
            weight_decay=0.0005, warmup_epochs=args.warmup_epochs,
            close_mosaic=args.close_mosaic, amp=args.amp,
            fraction=0.01 if args.smoke else 1.0, seed=args.seed, deterministic=True,
            iou=0.7, max_det=300, conf=0.001, plots=False, save=True,
            project=str(directory), name='finetune', exist_ok=False,
        )
        trainer = FP32PruningTrainer(overrides=overrides)
        trainer.expected_shapes = shapes
        trainer.model = model
        trainer.train()
        best = YOLO(str(trainer.best)).model.float().cpu().eval()
        if channel_shapes(best) != shapes:
            raise RuntimeError('The trainer rebuilt or changed the pruned structure.')
        if not all(p.dtype == torch.float32 for p in torch.load(trainer.best, map_location='cpu')['ema'].parameters()):
            raise RuntimeError('Training checkpoint was not saved as FP32.')
        current = directory / 'selected.pt'
        inference_checkpoint(best, current, vars(trainer.args))
        record['post'] = validate(current, f'stage_{stage:02d}_post', args)
        record['checkpoint'] = str(current.relative_to(args.output))
        record['map50_95_drop'] = state['baseline']['map50_95'] - record['post']['map50_95']
        state['stages'].append(record)
        state['current_checkpoint'] = record['checkpoint']
        state['stopped'] = record['map50_95_drop'] > args.max_map_drop
        write_json(directory / 'metrics.json', record)
        save_summary(args, state)
        print(f"Stage {stage}: mAP50 {record['pre']['map50']:.5f} -> {record['post']['map50']:.5f}; "
              f"mAP50-95 {record['pre']['map50_95']:.5f} -> {record['post']['map50_95']:.5f}; "
              f"MAC ratio {state['baseline_macs'] / macs:.4f}", flush=True)
        del trainer, model, best
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if state['stopped']:
            print('Stopped: post-finetune mAP50-95 drop exceeds --max-map-drop.', flush=True)
            break
    print(f"Summary: {args.output / 'summary.csv'}", flush=True)


if __name__ == '__main__':
    main()
