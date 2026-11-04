#!/usr/bin/env python3
"""Unified iterative structured pruning for YOLOv8n/s.

The runner compares Random, Group-L1, Group-L2 and LAMP under one protocol:
C2f conversion, dependency-aware channel pruning, protected Detect/stem,
pre-pruning validation, fixed-budget fine-tuning and post-training validation.

It is adapted to this repository's Ultralytics v8.0.239 checkout and local
Torch-Pruning 1.6.0 source tree.  Each output directory represents exactly one
method/configuration and supports resuming after a completed stage.
"""

import argparse
import csv
import hashlib
import json
import random
import sys
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
    FP32PruningTrainer,
    channel_shapes,
    convert_c2f,
    inference_checkpoint,
    raw_output,
)


METHODS = ('random', 'group_l1', 'group_l2', 'lamp')


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='0', help='Single CUDA device or cpu')
    parser.add_argument('--allocation', choices=('global', 'local'), default='global',
                        help='Global ranks dependency groups together; local removes the same ratio per group')
    parser.add_argument('--total-stages', type=int, default=16)
    parser.add_argument('--run-stages', type=int, default=1,
                        help='Stages to run in this invocation; the per-stage rate is unchanged')
    parser.add_argument('--target-prune-rate', type=float, default=0.5,
                        help='Cumulative channel pruning target used to derive the per-stage rate')
    parser.add_argument('--target-speedup', type=float, default=None,
                        help='Optionally stop after reaching this measured baseline/current MAC ratio')
    parser.add_argument('--epochs-per-stage', type=int, default=10)
    parser.add_argument('--max-map-drop', type=float, default=0.05,
                        help='Stop when absolute mAP50-95 drop from baseline exceeds this value')
    parser.add_argument('--round-to', type=int, default=None,
                        help='Optional channel multiple; leave unset for method comparison')
    parser.add_argument('--prune-stem', action='store_true',
                        help='Allow pruning the first Conv block; protected by default')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--optimizer', choices=('SGD', 'AdamW', 'auto'), default='SGD')
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--warmup-epochs', type=float, default=3.0)
    parser.add_argument('--close-mosaic', type=int, default=10)
    parser.add_argument('--amp', action='store_true', help='Opt into AMP; FP32 is the default')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true',
                        help='Continue after the last fully completed stage')
    parser.add_argument('--smoke', action='store_true',
                        help='Use 1 epoch, 1%% train data, 128px and batch 2')
    args = parser.parse_args()

    if args.total_stages < 1 or args.run_stages < 1 or args.epochs_per_stage < 1:
        parser.error('Stage and epoch counts must be positive.')
    if not 0 < args.target_prune_rate < 1:
        parser.error('--target-prune-rate must be between 0 and 1.')
    if args.target_speedup is not None and args.target_speedup <= 1:
        parser.error('--target-speedup must be greater than 1.')
    if args.max_map_drop < 0:
        parser.error('--max-map-drop cannot be negative.')
    if args.round_to is not None and args.round_to < 2:
        parser.error('--round-to must be at least 2 when specified.')
    if args.batch < 1 or args.threads < 1 or args.imgsz < 32 or args.imgsz % 32:
        parser.error('Use positive batch/threads and an imgsz divisible by 32.')
    if ',' in args.device:
        parser.error('Only single-device training is supported.')
    if args.method == 'lamp' and args.allocation != 'global':
        parser.error('LAMP requires --allocation global; local fixed-ratio pruning removes its layer-adaptive effect.')
    if args.smoke:
        args.epochs_per_stage, args.imgsz, args.batch, args.workers = 1, 128, 2, 0
        args.warmup_epochs, args.close_mosaic, args.amp = 0.0, 0, False
    return args


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_importance(method):
    if method == 'random':
        return tp.importance.RandomImportance(), 'RandomImportance()'
    if method == 'group_l1':
        return (
            tp.importance.GroupMagnitudeImportance(
                p=1, group_reduction='mean', normalizer='mean'
            ),
            "GroupMagnitudeImportance(p=1, group_reduction='mean', normalizer='mean')",
        )
    if method == 'group_l2':
        return (
            tp.importance.GroupMagnitudeImportance(
                p=2, group_reduction='mean', normalizer='mean'
            ),
            "GroupMagnitudeImportance(p=2, group_reduction='mean', normalizer='mean')",
        )
    if method == 'lamp':
        return (
            tp.importance.LAMPImportance(p=2, group_reduction='mean', normalizer='lamp'),
            "LAMPImportance(p=2, group_reduction='mean', normalizer='lamp')",
        )
    raise ValueError(f'Unsupported method: {method}')


def result_metrics(result):
    return dict(
        precision=float(result.box.mp),
        recall=float(result.box.mr),
        map50=float(result.box.map50),
        map50_95=float(result.box.map),
    )


def validate(checkpoint, name, args):
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    result = YOLO(str(checkpoint)).val(
        data=str(args.data), split='val', imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, conf=0.001, iou=0.7,
        max_det=300, half=False, augment=False, plots=False,
        project=str(args.output / 'val'), name=name, exist_ok=False,
    )
    values = result_metrics(result)
    write_json(Path(result.save_dir) / 'metrics.json', values)
    return values


def save_summary(args, state):
    rows = [dict(
        method=args.method, stage=0, phase='baseline',
        params=state['baseline_params'],
        gflops=state['baseline_macs'] * 2 / 1e9,
        mac_ratio=1.0, param_reduction=0.0, gflops_reduction=0.0,
        **state['baseline'],
    )]
    for stage in state['stages']:
        for phase in ('pre', 'post'):
            rows.append(dict(
                method=args.method, stage=stage['stage'], phase=phase,
                params=stage['params'], gflops=stage['macs'] * 2 / 1e9,
                mac_ratio=state['baseline_macs'] / stage['macs'],
                param_reduction=1 - stage['params'] / state['baseline_params'],
                gflops_reduction=1 - stage['macs'] / state['baseline_macs'],
                **stage[phase],
            ))
    with (args.output / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / 'state.json', state)


def module_conv_outputs(module):
    return {
        name: child.out_channels
        for name, child in module.named_modules()
        if isinstance(child, torch.nn.Conv2d)
    }


def reseed_random_importance(seed, stage):
    """Make Random pruning reproducible across uninterrupted and resumed runs."""
    value = seed + stage
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def main():
    args = parse_args()
    for key in ('weights', 'data', 'output'):
        setattr(args, key, getattr(args, key).expanduser().resolve())

    if __version__ != '8.0.239':
        raise RuntimeError(f'Requires local Ultralytics v8.0.239, found {__version__}')
    if not args.weights.is_file() or not args.data.is_file():
        raise FileNotFoundError('Check --weights and --data paths.')

    torch.set_num_threads(args.threads)
    device = select_device(args.device, batch=args.batch)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = check_det_dataset(str(args.data), autodownload=False)
    if dataset.get('download'):
        raise ValueError('Use a local dataset YAML without a download section.')

    _, importance_name = make_importance(args.method)
    config = {
        key: value for key, value in vars(args).items()
        if key not in ('output', 'resume', 'run_stages', 'weights', 'data',
                       'device', 'threads', 'workers')
    }
    config.update(
        source_sha256=file_hash(args.weights),
        dataset_sha256=file_hash(args.data),
        ultralytics=__version__, torch=str(torch.__version__),
        torch_pruning=getattr(tp, '__version__', 'unknown'),
        importance=importance_name,
        global_pruning=args.allocation == 'global',
        protect_detect=True,
        protect_stem=not args.prune_stem,
        checkpoint_dtype='float32',
    )
    example = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    per_stage_rate = 1 - (1 - args.target_prune_rate) ** (1 / args.total_stages)
    state_path = args.output / 'state.json'

    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f'Resume state is missing: {state_path}')
        state = json.loads(state_path.read_text(encoding='utf-8'))
        runtime_keys = ('dataset_sha256', 'torch')
        comparable = lambda value: {
            key: item for key, item in value.items() if key not in runtime_keys
        }
        if comparable(state['config']) != comparable(config):
            raise ValueError('Resume configuration differs from the original run.')
        if state['class_names'] != list(dataset['names'].values()):
            raise ValueError('Resume dataset classes differ from the original run.')
        state.setdefault('resume_environments', []).append(dict(
            data=str(args.data), dataset_sha256=config['dataset_sha256'],
            torch=config['torch'], device=args.device,
        ))
        if state.get('stopped'):
            raise RuntimeError(
                f"This run already stopped: {state.get('stop_reason', 'unknown reason')}"
            )
        current = args.output / state['current_checkpoint']
        if not current.is_file():
            raise FileNotFoundError(f'Completed-stage checkpoint is missing: {current}')
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError('Output is nonempty. Use a new path or --resume.')
        args.output.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(args.weights)).model.float().cpu().eval()
        if not isinstance(model.model[-1], Detect) or model.model[-1].nc != dataset['nc']:
            raise ValueError('Use a detection checkpoint trained on this dataset with matching classes.')

        conversion_input = example[:, :, :128, :128].detach().cpu()
        before = raw_output(model, conversion_input)
        converted = convert_c2f(model)
        after = raw_output(model, conversion_input)
        difference = max(float((a - b).abs().max()) for a, b in zip(before, after))
        if difference > 1e-4:
            raise RuntimeError(f'C2f raw-output equivalence failed: {difference}')

        current = args.output / 'converted.pt'
        inference_checkpoint(model, current, getattr(model, 'args', {}))
        model = model.to(device)
        baseline_macs, baseline_params = tp.utils.count_ops_and_params(model, example)
        state = dict(
            config=config,
            per_stage_prune_rate=per_stage_rate,
            class_names=list(dataset['names'].values()),
            converted_c2f=converted,
            conversion_max_abs_diff=difference,
            baseline_macs=float(baseline_macs),
            baseline_params=int(baseline_params),
            baseline=validate(current, 'baseline', args),
            stages=[], current_checkpoint='converted.pt',
            last_accepted_checkpoint='converted.pt',
            stopped=False, stop_reason=None,
        )
        save_summary(args, state)
        del model

    start = len(state['stages']) + 1
    last = min(args.total_stages, start + args.run_stages - 1)
    for stage_number in range(start, last + 1):
        if not current.is_file():
            raise FileNotFoundError(current)
        stage_dir = args.output / f'stage_{stage_number:02d}'
        if stage_dir.exists():
            raise FileExistsError(
                f'{stage_dir} is incomplete or already exists; preserve it and use a new output path.'
            )
        stage_dir.mkdir()

        model = YOLO(str(current)).model.float().to(device).eval()
        if hasattr(model, 'criterion'):
            delattr(model, 'criterion')
        for parameter in model.parameters():
            parameter.requires_grad_(True)

        head = model.model[-1]
        stem = model.model[0]
        protected = [head]
        if not args.prune_stem:
            protected.append(stem)
        protected_shapes = {
            'detect': module_conv_outputs(head),
            'stem': module_conv_outputs(stem) if not args.prune_stem else None,
        }

        if args.method == 'random':
            reseed_random_importance(args.seed, stage_number)
        importance, _ = make_importance(args.method)
        pruner = tp.pruner.GroupNormPruner(
            model, example, importance=importance,
            global_pruning=args.allocation == 'global',
            iterative_steps=1, pruning_ratio=per_stage_rate,
            ignored_layers=protected, round_to=args.round_to,
        )
        pruner.step()

        if protected_shapes['detect'] != module_conv_outputs(head):
            raise RuntimeError('A protected Detect convolution width changed.')
        if not args.prune_stem and protected_shapes['stem'] != module_conv_outputs(stem):
            raise RuntimeError('A protected stem convolution width changed.')

        macs, params = tp.utils.count_ops_and_params(model, example)
        if macs >= state['baseline_macs'] and stage_number == 1:
            raise RuntimeError('Pruning did not reduce MACs; check protected layers and pruner settings.')
        shapes = channel_shapes(model)
        torch.save(pruner.pruning_history(), stage_dir / 'pruning_history.pt')
        del pruner

        pruned_checkpoint = stage_dir / 'pruned.pt'
        inference_checkpoint(model, pruned_checkpoint, getattr(model, 'args', {}))
        record = dict(
            stage=stage_number,
            input_checkpoint=str(current.relative_to(args.output)),
            macs=float(macs), params=int(params), shapes=shapes,
            pre=validate(pruned_checkpoint, f'stage_{stage_number:02d}_pre', args),
        )
        write_json(stage_dir / 'pre_metrics.json', record)

        overrides = dict(
            model=str(pruned_checkpoint), data=str(args.data), task='detect', mode='train',
            epochs=args.epochs_per_stage, patience=args.epochs_per_stage + 1,
            imgsz=args.imgsz, batch=args.batch, workers=args.workers, device=args.device,
            optimizer=args.optimizer, lr0=args.lr0, lrf=0.01, momentum=0.937,
            weight_decay=0.0005, warmup_epochs=args.warmup_epochs,
            close_mosaic=args.close_mosaic, amp=args.amp,
            fraction=0.01 if args.smoke else 1.0,
            seed=args.seed, deterministic=True,
            iou=0.7, max_det=300, conf=0.001, plots=False, save=True,
            project=str(stage_dir), name='finetune', exist_ok=False,
        )
        trainer = FP32PruningTrainer(overrides=overrides)
        trainer.expected_shapes = shapes
        trainer.model = model
        trainer.train()

        best_checkpoint = torch.load(trainer.best, map_location='cpu')
        if not all(parameter.dtype == torch.float32
                   for parameter in best_checkpoint['ema'].parameters()):
            raise RuntimeError('Training checkpoint was not saved in FP32.')
        best = YOLO(str(trainer.best)).model.float().cpu().eval()
        if channel_shapes(best) != shapes:
            raise RuntimeError('The trainer rebuilt or changed the pruned structure.')

        current = stage_dir / 'selected.pt'
        inference_checkpoint(best, current, vars(trainer.args))
        record['post'] = validate(current, f'stage_{stage_number:02d}_post', args)
        record['checkpoint'] = str(current.relative_to(args.output))
        record['map50_95_drop'] = (
            state['baseline']['map50_95'] - record['post']['map50_95']
        )
        record['mac_ratio'] = state['baseline_macs'] / record['macs']
        state['stages'].append(record)
        state['current_checkpoint'] = record['checkpoint']

        stop_reasons = []
        if record['map50_95_drop'] > args.max_map_drop:
            stop_reasons.append(
                f"mAP50-95 drop {record['map50_95_drop']:.6f} exceeded {args.max_map_drop:.6f}"
            )
        else:
            state['last_accepted_checkpoint'] = record['checkpoint']
        if args.target_speedup is not None and record['mac_ratio'] >= args.target_speedup:
            stop_reasons.append(
                f"MAC ratio {record['mac_ratio']:.6f} reached target {args.target_speedup:.6f}"
            )
        state['stopped'] = bool(stop_reasons)
        state['stop_reason'] = '; '.join(stop_reasons) if stop_reasons else None

        write_json(stage_dir / 'metrics.json', record)
        save_summary(args, state)
        print(
            f"Stage {stage_number}: mAP50 {record['pre']['map50']:.5f} -> "
            f"{record['post']['map50']:.5f}; mAP50-95 "
            f"{record['pre']['map50_95']:.5f} -> {record['post']['map50_95']:.5f}; "
            f"MAC ratio {record['mac_ratio']:.4f}", flush=True,
        )
        del best_checkpoint, trainer, model, best
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if state['stopped']:
            print(f"Stopped: {state['stop_reason']}", flush=True)
            print(f"Last accepted checkpoint: {state['last_accepted_checkpoint']}", flush=True)
            break

    print(f"Summary: {args.output / 'summary.csv'}", flush=True)


if __name__ == '__main__':
    main()
