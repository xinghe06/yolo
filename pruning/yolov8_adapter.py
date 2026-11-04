"""YOLOv8 modules and helpers used by dependency-aware structural pruning.

The pinned Ultralytics YOLOv8 C2f implementation splits one convolution with
``chunk(2, 1)``. Replacing that operation by two explicit convolutions makes
the branches visible to Torch-Pruning's dependency graph. The converted class
lives in an importable module so full-object checkpoints can be loaded later.
"""

from copy import deepcopy
from typing import Iterable, List

import torch
import torch.nn as nn

from ultralytics.nn.modules import Bottleneck, C2f, Conv, DFL, Detect


class PrunableC2f(nn.Module):
    """C2f with two explicit input branches instead of a channel chunk."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv0(x), self.cv1(x)]
        y.extend(block(y[-1]) for block in self.m)
        return self.cv2(torch.cat(y, 1))


def _copy_conv_slice(source: Conv, target: Conv, start: int, stop: int) -> None:
    """Copy one contiguous output-channel slice of an Ultralytics Conv block."""
    with torch.no_grad():
        target.conv.weight.copy_(source.conv.weight[start:stop])
        if source.conv.bias is not None and target.conv.bias is not None:
            target.conv.bias.copy_(source.conv.bias[start:stop])
        target.bn.weight.copy_(source.bn.weight[start:stop])
        target.bn.bias.copy_(source.bn.bias[start:stop])
        target.bn.running_mean.copy_(source.bn.running_mean[start:stop])
        target.bn.running_var.copy_(source.bn.running_var[start:stop])
        target.bn.num_batches_tracked.copy_(source.bn.num_batches_tracked)
    target.bn.eps = source.bn.eps
    target.bn.momentum = source.bn.momentum
    target.act = deepcopy(source.act)


def make_prunable_c2f(source: C2f) -> PrunableC2f:
    """Create a weight-equivalent :class:`PrunableC2f` from a standard C2f."""
    hidden = source.c
    if source.cv1.conv.out_channels != 2 * hidden:
        raise ValueError("Expected C2f.cv1 to contain two equal channel branches")

    first_block = source.m[0] if len(source.m) else None
    shortcut = bool(first_block is not None and getattr(first_block, "add", False))
    groups = first_block.cv2.conv.groups if first_block is not None else 1
    target = PrunableC2f(
        source.cv1.conv.in_channels,
        source.cv2.conv.out_channels,
        n=len(source.m),
        shortcut=shortcut,
        g=groups,
        e=hidden / source.cv2.conv.out_channels,
    ).to(device=source.cv1.conv.weight.device, dtype=source.cv1.conv.weight.dtype)
    _copy_conv_slice(source.cv1, target.cv0, 0, hidden)
    _copy_conv_slice(source.cv1, target.cv1, hidden, 2 * hidden)
    target.cv2 = source.cv2
    target.m = source.m
    target.train(source.training)

    # DetectionModel.forward uses these graph-routing attributes.
    for name in ("i", "f", "type", "np"):
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))
    return target


def convert_c2f_to_prunable(module: nn.Module) -> int:
    """Recursively replace every standard C2f and return the number converted."""
    converted = 0
    for name, child in list(module.named_children()):
        if isinstance(child, PrunableC2f):
            continue
        if isinstance(child, C2f):
            setattr(module, name, make_prunable_c2f(child))
            converted += 1
        else:
            converted += convert_c2f_to_prunable(child)
    return converted


def fixed_output_layers(model: nn.Module) -> List[nn.Module]:
    """Return Detect layers whose output channel counts are part of YOLO's format."""
    head = model.model[-1]
    if not isinstance(head, Detect):
        raise TypeError(f"Expected a Detect head, found {type(head).__name__}")

    ignored: List[nn.Module] = []
    for branches in (head.cv2, head.cv3):
        for branch in branches:
            convolutions = [layer for layer in branch.modules() if isinstance(layer, nn.Conv2d)]
            if not convolutions:
                raise RuntimeError("A Detect branch contains no output convolution")
            ignored.append(convolutions[-1])
    for layer in head.modules():
        if isinstance(layer, DFL):
            ignored.append(layer.conv)
    return ignored


def flatten_tensors(value) -> Iterable[torch.Tensor]:
    """Yield tensors from nested model outputs for equivalence checks."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from flatten_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from flatten_tensors(item)


def output_max_abs_diff(before, after) -> float:
    """Return the largest absolute tensor difference between nested outputs."""
    before_tensors = list(flatten_tensors(before))
    after_tensors = list(flatten_tensors(after))
    if len(before_tensors) != len(after_tensors):
        raise RuntimeError(f"Output tensor count changed: {len(before_tensors)} -> {len(after_tensors)}")
    differences = []
    for left, right in zip(before_tensors, after_tensors):
        if left.shape != right.shape:
            raise RuntimeError(f"Output shape changed during C2f conversion: {left.shape} -> {right.shape}")
        differences.append((left.float() - right.float()).abs().max().item())
    return max(differences, default=0.0)


def forward_raw_detect(model: nn.Module, inputs: torch.Tensor):
    """Run an eval model while asking only Detect to return undecoded tensors.

    Comparing decoded boxes is unnecessarily sensitive to harmless CUDA
    round-off: DFL and stride scaling amplify tiny feature-map differences.
    Setting the head's flag directly keeps all child BatchNorm modules in eval
    mode while selecting Detect's raw-output branch.
    """
    head = model.model[-1]
    if not isinstance(head, Detect):
        raise TypeError(f"Expected a Detect head, found {type(head).__name__}")
    previous = head.training
    head.training = True
    try:
        with torch.no_grad():
            return model(inputs)
    finally:
        head.training = previous
