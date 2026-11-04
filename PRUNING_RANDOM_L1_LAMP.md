# YOLOv8 Random、L1、LAMP 结构化剪枝

本实验固定使用 Ultralytics v8.0.239、Torch-Pruning v1.6.1 和同一份未剪枝基线权重。三种方法
共用 DepGraph、全局通道排序、Detect 固定输出保护、`round_to=8`、实际 MAC 加速比和微调配置，
只替换重要度函数。

## 1. 环境

在项目根目录执行：

```bash
cd /root/autodl-tmp/yolov8n-visdrone-pruning
python -m pip install --no-deps torch-pruning==1.6.1
python -m pip show torch-pruning | grep -E '^(Name|Version):'
python -c "import torch_pruning as tp; print(tp.__file__)"
```

Torch-Pruning 的 v1.6.1 发布包中，`torch_pruning.__version__` 仍可能显示 `1.6.0`；以
`pip show torch-pruning` 的发行版本为准。

运行脚本时从项目根目录启动，脚本会优先导入 `third_party/ultralytics` 中固定的 v8.0.239。

## 2. Smoke test

先对 YOLOv8n 分别只执行最小的一次结构化剪枝，仅验证结构、前向、保存和重载。由于
`round_to=8` 和依赖组大小的限制，实际加速比可能明显超过命令中的 `1.01`，以输出记录为准：

```bash
BASE=runs/baseline/baseline_yolov8n_official100/weights/best.pt

for METHOD in random l1 lamp; do
  python scripts/prune_yolov8.py \
    --method "$METHOD" \
    --weights "$BASE" \
    --output "runs/pruning/smoke/yolov8n_${METHOD}_minstep" \
    --device 0 \
    --target-speedup 1.01 \
    --iterative-steps 100 \
    --round-to 8
done
```

在新进程中逐个重载。这里显式加入固定的 Ultralytics 源码路径：

```bash
for METHOD in random l1 lamp; do
  PYTHONPATH="$PWD:$PWD/third_party/ultralytics" \
    python -c "from pruning.yolov8_adapter import PrunableC2f; from ultralytics import YOLO; m=YOLO('runs/pruning/smoke/yolov8n_${METHOD}_minstep/pruned.pt'); m.info()"
done
```

## 3. 第一阶段正式剪枝

第一阶段统一剪到实际 MAC speed-up 约 1.5 倍：

```bash
BASE=runs/baseline/baseline_yolov8n_official100/weights/best.pt

for METHOD in random l1 lamp; do
  python scripts/prune_yolov8.py \
    --method "$METHOD" \
    --weights "$BASE" \
    --output "runs/pruning/structured/yolov8n_${METHOD}_s1.5" \
    --device 0 \
    --target-speedup 1.5 \
    --iterative-steps 100 \
    --max-pruning-ratio 0.8 \
    --round-to 8 \
    --seed 0
done
```

每个目录会产生：

- `pruned.pt`：Ultralytics 格式的完整结构 checkpoint；
- `pruned_model.pth`：纯 PyTorch 完整模型对象；
- `pruning_history.pt`：Torch-Pruning 剪枝历史；
- `pruning_metrics.json`：源权重哈希、实际 MAC/GFLOPs、参数量、加速比和逐步记录。

## 4. 剪枝后立即验证

```bash
DATA=configs/VisDrone-autodl.yaml

for METHOD in random l1 lamp; do
  python scripts/validate_pruned_yolov8.py \
    --weights "runs/pruning/structured/yolov8n_${METHOD}_s1.5/pruned.pt" \
    --data "$DATA" \
    --device 0 \
    --batch 16 \
    --name "yolov8n_${METHOD}_s1.5"
done
```

## 5. 统一微调 50 轮

三种方法必须使用完全相同的微调参数：

```bash
DATA=configs/VisDrone-autodl.yaml

for METHOD in random l1 lamp; do
  python scripts/finetune_pruned_yolov8.py \
    --weights "runs/pruning/structured/yolov8n_${METHOD}_s1.5/pruned.pt" \
    --data "$DATA" \
    --device 0 \
    --epochs 50 \
    --batch 16 \
    --workers 8 \
    --lr0 0.001 \
    --name "yolov8n_${METHOD}_s1.5_ft50"
done
```

单卡顺序执行三次微调。若有三张空闲 GPU，可以把三个命令分别放入三个 tmux，并为它们设置不同的
`--device`。不要在同一块 GPU 上同时训练三个模型。

## 6. YOLOv8s

YOLOv8s 使用完全相同的脚本，只替换基线权重和输出名称：

```bash
BASE=runs/baseline/baseline_yolov8s_official100/weights/best.pt

python scripts/prune_yolov8.py \
  --method l1 \
  --weights "$BASE" \
  --output runs/pruning/structured/yolov8s_l1_s1.5 \
  --device 0 \
  --target-speedup 1.5 \
  --iterative-steps 100 \
  --round-to 8
```

应先完成 YOLOv8n 三种方法的 smoke、零微调验证和微调。确认实验表及产物完整后，再在 YOLOv8s
上重复，避免在结构或保存流程尚未验证时消耗更多训练资源。
