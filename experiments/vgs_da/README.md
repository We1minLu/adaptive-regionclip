# VGS＋域适应＋EMA 图像级一致性（无监督）

当前版本在固定教师 VGS＋DA 基线上加入 Adaptive Teacher 风格的强弱增强、EMA 0.9996，以及权重 1.0 的 H2FA 图像级软一致性。目标域不使用真实图像类别或框标签，没有目标伪框分类/回归监督。

源域 2,975 张，目标域三种雾浓度共 8,925 张；从相同 SourceB 与源域监督 VGS 初始化，重新训练 25K。网络、损失与结果概览见[项目首页](../../README.md)；聚合公式、教师更新和梯度边界见[实现说明](EMA_IMAGE_CONSISTENCY.md)。

## 结果

| 检查点 | 混合 AP50 | AP75 | 轻雾 AP50 | 中雾 AP50 | 浓雾 AP50 |
|---|---:|---:|---:|---:|---:|
| **8K：验证集最佳** | **58.36** | 35.00 | 62.12 | 59.38 | 53.81 |
| 25K：固定终点 | 55.19 | 33.56 | 59.62 | 55.65 | 50.20 |

精确 AP50 为 **58.356487 / 55.190529**。每 1K 在三雾混合 val 的全部 1,500 张图上评估，使用 VOC2007 十一点 AP；最佳点按混合 AP50 选择。相比[前轮固定教师 UDA](UDA_FIXED_TEACHER.md)，最佳提高 2.17 点、终点下降 0.74 点，组合改动的单项贡献尚待消融。[完整曲线](results/ema_image_25k.json)。

## 参数和复现

已验证环境：Python 3.8、PyTorch 1.9.0+cu111、torchvision 0.10.0+cu111、本仓库 Detectron2，V100 32GB。模型构造需要 GPU 和 OpenAI CLIP RN50 缓存；环境/数据安装见[原项目说明](../../docs/legacy_adapters_README.md)。

| 参数 | 设置 |
|---|---|
| 初始化 | SourceB 检测权重＋源域监督 VGS；不从旧 UDA 最佳点续训 |
| 训练 / 评估 | 25K / 每 1K，评估学生输出（保留 EMA 语义供图） |
| batch / 分辨率 | 源域 2＋目标域 2，累积 1，1024×2048 |
| 检测器 / 辅助优化器 | SGD 0.0005、momentum 0.9 / AdamW 0.0002 |
| 权重衰减 / 调度 | 均 0.0001；1K 预热，起始倍率 0.01，其后恒定 |
| EMA / 图像 BCE | 0.9996 / 1.0；软教师目标，无额外 burn-in |
| 候选 / ROI | RPN≤300＋VGS≤200；源域 ROI 采样预算 512，不追加 GT 框 |
| 推理 | ROI 分数阈值 0.001、NMS 0.5、每图最多 100 检测 |
| 执行设置 | AMP；梯度裁剪 10；学生/教师 ROI chunk 512/256；不启用 res5 checkpoint |
| 对抗 | 传统外层 0.1、GRL `r(t)`；条件外层 1.0、GRL `0.05*r(t)` |

沿用无目标标签清单；目标记录若含真实图像标签或框标注，会被拒绝。已有 UDA 清单可直接使用。若从旧有标签清单转换：

```bash
python experiments/vgs_da/prepare_uda.py \
  --input-dir /data/vgs_da/original_manifests \
  --output-dir /data/vgs_uda/manifests
```

使用新预设并指定本机资产路径：

```bash
python experiments/vgs_da/configure.py \
  --preset ema_image_25k --repo-root "$PWD" \
  --manifest-dir /data/vgs_uda/manifests --output-dir /data/vgs_ema_image/run \
  --source-checkpoint /data/weights/source_B.pth \
  --source-search-checkpoint /data/weights/source_vgs.pt \
  --rpn-checkpoint /data/weights/rpn_coco_48.pth \
  --text-embeddings /data/weights/cityscapes_8_cls_emb.pth \
  --config-out /data/vgs_ema_image/config.json

python experiments/vgs_da/train.py --config /data/vgs_ema_image/config.json \
  --mode smoke --steps 6 --output /data/vgs_ema_image/smoke
python experiments/vgs_da/train.py --config /data/vgs_ema_image/config.json \
  --mode smoke --steps 2 --output /data/vgs_ema_image/smoke \
  --resume /data/vgs_ema_image/smoke/checkpoint_last.pth
python -u experiments/vgs_da/train.py --config /data/vgs_ema_image/config.json --mode train
```

`configure.py` 校验所有固定源资产内容，不会覆盖不同配置。旧 `uda_25k` 仍是固定教师版本；新增实验必须明确选择 `ema_image_25k`。强弱增强使用可恢复的样本种子，AMP 溢出重试不推进 EMA；完整断点保存教师状态、两个优化器、scaler 与 RNG。不能把旧固定教师权重当作新版本完整断点恢复。

已通过 97 项远程 CPU 测试、10 项 GPU 测试，以及真实训练、评估和 EMA 断点恢复 smoke；新增配置发布入口另有 5 项 CPU 测试。正式运行完成 25,000 次成功更新，无跳过更新；完整记录见[结果摘要](results/ema_image_25k.json)。

## 权重

最佳权重、固定依赖和 SHA256 见[权重清单](results/ema_image_weight_manifest.json)。最佳 8K 文件包含学生与 EMA 教师，**不含优化器状态**；25K 的 `checkpoint_last.pth` 才是完整续训断点。大权重不进入 Git。

```bash
python experiments/vgs_da/train.py --config /data/vgs_ema_image/config.json \
  --mode evaluate --resume /path/to/model_best_ap5836.pth \
  --output /data/vgs_ema_image/best_evaluation
```

历史版本与成绩独立保留：[固定教师 UDA，AP56.19](UDA_FIXED_TEACHER.md)、[图像标签版本，AP55.98](../../docs/vgs_image_labels_AP5598.md)。
