"""Isolated YOLOv8 adaptation of Torch-Pruning/examples/yolov8.

Keep checkpoint classes importable; do not import the older pruning adapter.
The C2f split follows the reference example and preserves BN configuration.
"""

from copy import deepcopy
from datetime import datetime

import torch
from torch import nn
from ultralytics import __version__
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.modules import Bottleneck, C2f, Conv
from ultralytics.utils.torch_utils import de_parallel


class OfficialC2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(block(y[-1]) for block in self.m)
        return self.cv2(torch.cat(y, 1))


def convert_c2f(module):
    count = 0
    for name, source in list(module.named_children()):
        if isinstance(source, C2f):
            first = source.m[0] if len(source.m) else None
            target = OfficialC2f(
                source.cv1.conv.in_channels, source.cv2.conv.out_channels,
                len(source.m), bool(first is not None and first.add),
                first.cv2.conv.groups if first is not None else 1,
                source.c / source.cv2.conv.out_channels,
            ).to(source.cv1.conv.weight)
            for branch, start in ((target.cv0, 0), (target.cv1, source.c)):
                # Copy configuration (eps, momentum, act) as well as tensors.
                branch.bn.eps = source.cv1.bn.eps
                branch.bn.momentum = source.cv1.bn.momentum
                branch.act = deepcopy(source.cv1.act)
                with torch.no_grad():
                    branch.conv.weight.copy_(source.cv1.conv.weight[start:start + source.c])
                    for key in ('weight', 'bias', 'running_mean', 'running_var'):
                        getattr(branch.bn, key).copy_(getattr(source.cv1.bn, key)[start:start + source.c])
                    branch.bn.num_batches_tracked.copy_(source.cv1.bn.num_batches_tracked)
            target.cv2, target.m = source.cv2, source.m
            for key in ('i', 'f', 'type', 'np'):
                if hasattr(source, key):
                    setattr(target, key, getattr(source, key))
            target.train(source.training)
            setattr(module, name, target)
            count += 1
        elif not isinstance(source, OfficialC2f):
            count += convert_c2f(source)
    return count


def raw_output(model, x):
    head = model.model[-1]
    was_training = head.training
    head.training = True
    try:
        with torch.no_grad():
            return model(x)
    finally:
        head.training = was_training


def channel_shapes(model):
    return {name: list(m.weight.shape) for name, m in model.named_modules()
            if isinstance(m, nn.Conv2d)}


def inference_checkpoint(model, path, train_args):
    saved = deepcopy(model).float().cpu().eval()
    if hasattr(saved, 'criterion'):
        delattr(saved, 'criterion')
    torch.save({
        'model': saved, 'ema': None, 'optimizer': None, 'epoch': -1,
        'best_fitness': None, 'updates': None, 'train_args': dict(train_args),
        'date': datetime.now().isoformat(), 'version': __version__,
    }, path)


class FP32PruningTrainer(DetectionTrainer):
    """Inject the actual pruned module and never rebuild its original YAML."""

    def setup_model(self):
        if not isinstance(self.model, nn.Module):
            raise RuntimeError('Set trainer.model to the pruned module before train().')
        if channel_shapes(self.model) != self.expected_shapes:
            raise RuntimeError('Pruned channel shapes changed before training.')
        return None

    def save_model(self):
        checkpoint = {
            'epoch': self.epoch, 'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)).float().cpu(),
            'ema': deepcopy(self.ema.ema).float().cpu(), 'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(), 'train_args': vars(self.args),
            'train_metrics': {**self.metrics, 'fitness': self.fitness},
            'date': datetime.now().isoformat(), 'version': __version__,
        }
        torch.save(checkpoint, self.last)
        if self.best_fitness == self.fitness:
            torch.save(checkpoint, self.best)

    def final_eval(self):
        # The runner validates best.pt immediately after train(). Do not call
        # the base final_eval(), which strips weights to FP16 in v8.0.239.
        if not self.best.is_file():
            raise RuntimeError('Training did not produce best.pt.')
