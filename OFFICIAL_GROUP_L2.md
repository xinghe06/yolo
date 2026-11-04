# 独立的 YOLOv8 Group-L2 渐进剪枝

入口：`scripts/prune_yolov8_official_group_l2.py`
独立模块：`pruning/official_group_l2.py`

本实现不导入之前的 `prune_yolov8.py`、`finetune_pruned_yolov8.py` 或
`pruning/yolov8_adapter.py`。参考 `Torch-Pruning/examples/yolov8/yolov8_pruning.py`，
适配当前 Ultralytics v8.0.239 和 VisDrone；不是原示例的 coco128 实验。

## 固定流程

1. 加载已训练的标准 YOLOv8 检测权重，转换 C2f，检查 CPU 原始输出等价性。
2. 验证转换后的基线，直接按指定输入尺寸统计 MACs（正式实验640，不再面积换算）。
3. 每个阶段新建 GroupNormPruner，GroupMagnitudeImportance(p=2)，默认mean聚合及mean归一化。
4. `global_pruning=False`，`round_to=None`，保护整个 Detect；其入口输入通道允许随上游变化。
5. 单步剪枝 → 保存和验证 → 微调 → 加载微调的 best.pt → 验证 → 下一阶段。
6. 根据微调后的 mAP50-95 相对初始基线下降量停止，与官方的 metric.box.map 一致。

每阶段比例为 `1 - (1 - target_prune_rate) ** (1 / total_stages)`。
默认target=0.5、total=16，每阶段约4.24%。该目标是通道比例调度，不是参数减少50%或MAC加速2倍的保证；
取整和受保护层也会影响最终实际结构。

`--run-stages 1` 只执行本次的一个阶段，**不会**把调度改为一步剪掉50%。

训练设置明确固定为SGD、lr0=0.01、lrf=0.01、momentum=0.937、weight_decay=0.0005、
warmup_epochs=3、每阶段10轮，其他参数采用固定v8.0.239默认值。
close_mosaic默认10，所以10轮的阶段从开始就关闭mosaic。
默认FP32训练，可显式加 `--amp` 使用混合精度；checkpoint始终保留FP32。
这部分是本项目明确记录的VisDrone实验设置，不宣称与旧版官方示例环境的每一个训练默认值一致。

## 本服务器：先跑一个正式阶段

```bash
conda activate leyolo
cd /mnt/sdc/lt/project/le-yolo-main
export MPLCONFIGDIR=/tmp/leyolo-mpl
export YOLO_CONFIG_DIR=/tmp/leyolo-yolo-config

python -u scripts/prune_yolov8_official_group_l2.py \
  --weights runs/baseline_yolov8n_official100/weights/best.pt \
  --data configs/VisDrone-local.yaml \
  --output runs/pruning/official_group_l2_n \
  --device 0 \
  --total-stages 16 \
  --run-stages 1 \
  --target-prune-rate 0.5 \
  --epochs-per-stage 10 \
  --batch 16 \
  --workers 8
```

正式实验以完整训练集训练，在548张验证图像上评估。选择实际空闲的GPU替换0。
推荐在tmux中执行。已有非空输出目录不会被覆盖。

完成第一阶段后，同一命令增加 `--resume` 可继续下一个阶段；其他训练设置必须保持一致。
将 `--run-stages` 改为2表示从上一次完成的位置再运行两个阶段，并不改变总调度长度。
`--resume` 支持已完成阶段间续跑，**不支持**自动恢复阶段中途被中断的训练。
若阶段目录存在但没有记录为完成，程序会拒绝覆盖，并保留中间文件。

`--max-map-drop` 默认0.2，单位是mAP50-95的绝对值（20个百分点），与示例默认值一致。
当前基线mAP50-95约0.197，该阈值实际上很宽松。可以在开始新实验时显式选0.05，
但必须记录该变化，且续跑时保持原阈值。

## 自动保存的结果

```text
official_group_l2_n/
  converted.pt                 # 转换后未剪枝的FP32模型
  state.json                   # 参数、初始指标、完整阶段记录、续跑位置
  summary.csv                  # baseline / 每阶段pre与post指标、参数与GFLOPs
  val/                         # 所有独立验证及metrics.json
  stage_01/
    pruned.pt                  # 当前阶段剪后、未微调
    pruning_history.pt         # 相对该阶段输入模型的剪枝历史
    pre_metrics.json
    finetune/weights/best.pt    # 包含完整结构、EMA和优化器的FP32 checkpoint
    finetune/weights/last.pt
    selected.pt                # 从best EMA提取的FP32推理模型，下一阶段从此加载
    metrics.json
```

评价时同时看 `pre` 和 `post`，最终效果以恢复训练后的post为准。
best使用Ultralytics的fitness选择，而非单独按mAP50选择。
保存训练checkpoint前不调用half()，也不调用默认的FP16 strip_optimizer；训练器直接注入
剪后的模块，训练前后都检查卷积形状。验证使用独立加载的模型，不融合将要继续训练的对象。

## AutoDL

同步上述两个新Python文件与本说明，保留固定Ultralytics子模块和相同Torch-Pruning版本。
从原始基线开始新实验时，替换参数即可：

```text
--weights runs/baseline/baseline_yolov8n_official100/weights/best.pt
--data configs/VisDrone-autodl.yaml
```

如搬运已完成阶段，需复制整个实验输出目录并使用 `--resume`；数据YAML中的绝对路径可以变化，
原始基线权重哈希、类别和实验参数必须一致。还应自行确保两台机器的数据内容一致。
自定义checkpoint需要项目根目录在Python导入路径上；从新入口启动会自动设置。

## 低成本连通性检查

加 `--smoke` 会使用128输入、1%训练图片、每阶段1轮、batch=2及FP32；验证集仍完整。
测试结果仅用于检查训练/保存/加载/阶段衔接，不用于判断剪枝精度。
烟雾实验与正式实验必须使用不同输出目录，也不能跨模式续跑。

本地已验证：CPU运行首阶段后退出，再以新进程 `--resume` 完成第二阶段；两个阶段都完成
训练、FP32保存、重新加载及完整验证。首阶段208个参数张量发生更新，selected.pt与保存的
best EMA完全一致。第二阶段从第一阶段selected.pt继续剪枝、训练，程序内置的
训练前后通道形状检查与FP32 checkpoint检查均通过；首阶段另行检查了参数有限性。
这些检查不等同于正式640分辨率、每阶段10轮的精度实验。
测试输出原位于 `runs/pruning/official_group_l2_smoke/`，测试完成后发现该目录在共享文件系统
中已被移除。运行日志保留在 `logs/official_group_l2_smoke.log` 和
`logs/official_group_l2_smoke_resume.log`。
