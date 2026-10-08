# VGS＋域适应：目标域无标签版本

Cityscapes → Foggy Cityscapes 八类检测。源域有完整 GT；目标域仅用图像与域身份，不使用图像级类别标签或目标框。源域 2,975 张、目标域三种雾浓度共 8,925 张；从原 SourceB 与源域监督训练的 VGS 初始化开始，未从有目标标签的 AP55.98 模型续训。

## 结构与教师预测

- **固定 RPN**：最多保留前 300 个候选 B0。
- **固定 SourceB 教师**：为 B0 产生八类＋背景概率。教师由源域模型复制并冻结，没有 EMA 更新。
- **VGS**：在线 C3/C4 视觉特征 V，结合候选几何/覆盖 G 五通道、教师概率与语义分歧 S 十通道，预测目标性、遗漏概率与框偏移；按目标性×遗漏概率排序，经去重/NMS 补充最多 200 框。
- **学生 ROI**：合并原始与补充候选，学习实际 ROI 视觉特征和框回归。C3/C4/res5/注意力池化按源模型原可训练范围更新；文本/背景分类器、教师、RPN 固定。源域 GT 同时训练检测与 VGS；候选坐标及语义图停止梯度。

使用教师软概率构造语义图，并以高置信预测类别作为条件对抗的伪类别。**没有目标域伪框分类或回归损失，也没有目标域 VGS 的框监督**。UDA 仅取消目标图像类别存在性过滤，教师类别概率 ≥0.7、RPN 概率 ≥0.5 的筛选与原分组、排序和预算均保留。目标训练输入若含 `image_labels` 或框字段，会被拒绝；源域仍允许由 GT 提供类别存在性。

## 域对抗与权重

两类目标，共三个判别器网络：

| 对抗目标 | 判别器与输入 | 总损失外层权重 | GRL 系数 |
|---|---|---:|---|
| 传统域对抗 | C3、C4 各一个卷积判别器；全特征图 | 两头损失平均后 ×0.1 | `r(t)` |
| 条件域对抗 | VGS 语义融合前 p3/p4 的 ROI 视觉特征；一个条件 MLP | 1.0 | `0.05*r(t)` |

条件 MLP 含 48 个条件输出（8 类×3 尺度×2 冗余组），按每个 ROI 的条件取一个输出；不是 48 套独立网络。仅对源/目标共同出现的条件组计算组间等权、域间各半的 BCE。其对抗梯度进入 VGS lateral3/4 及学生视觉骨干。

总损失为：

`L = 1.0*L_source_detection + 1.0*L_source_VGS + 0.1*(L_C3 + L_C4)/2 + 1.0*L_conditional`

其中 `p=clip(t/25000,0,1)`，`r(t)=2/(1+exp(-10*p))-1`；每层传统损失也对源/目标取平均。GRL 只反转并缩放特征侧梯度，不缩放判别器自身梯度。因此条件损失权重是 **1.0**，不是 0.05；0.05 是其特征侧对抗强度上限。源域 VGS 的目标性、遗漏和框损失外层系数均为 1.0。

## 已完成结果

25K 更新，每 1K 在全部 1,500 张验证图上评估（500 场景×三种浓度 0.005/0.01/0.02），采用仓库原生 VOC2007 十一点 AP，混合 AP 由全部预测合并计算。

| 检查点 | 混合 AP50 | AP75 | 轻雾 AP50 | 中雾 AP50 | 浓雾 AP50 |
|---|---:|---:|---:|---:|---:|
| 9K：验证集最佳 | **56.19** | 34.30 | 60.95 | 57.58 | 50.23 |
| 25K：固定终点 | 55.93 | 33.66 | 60.43 | 57.95 | 49.32 |

精确最佳为 **56.187524**，终点为 **55.932228**；[完整曲线](results/uda_25k.json)。9K 是验证集选优的探索结果，25K 是固定预算终点。历史有图像级标签版本为 55.975298；原运行中曾调整 ROI chunk、修复 AMP，本轮沿用最终稳定设置，尚不能将差异单独归因于去掉标签。

## 运行

已验证环境：Python 3.8、PyTorch 1.9.0+cu111、对应 torchvision、本仓库 Detectron2，需要 GPU 和 OpenAI CLIP RN50 缓存。环境与数据安装参考[原项目说明](../../docs/legacy_adapters_README.md)。

先保留原源域/评估清单及目标图像顺序，并生成只含图像信息的目标训练清单；不复制标签侧文件。原清单可由 `prepare_data.py` 生成，准备步骤中产生的旧目标标签不进入 UDA 训练。

```bash
python experiments/vgs_da/prepare_uda.py \
  --input-dir /data/vgs_da/original_manifests \
  --output-dir /data/vgs_uda/manifests

python experiments/vgs_da/configure.py \
  --preset uda_25k --repo-root "$PWD" \
  --manifest-dir /data/vgs_uda/manifests --output-dir /data/vgs_uda/run \
  --source-checkpoint /data/weights/source_B.pth \
  --source-search-checkpoint /data/weights/source_vgs.pt \
  --rpn-checkpoint /data/weights/rpn_coco_48.pth \
  --text-embeddings /data/weights/cityscapes_8_cls_emb.pth \
  --config-out /data/vgs_uda/config.json

python experiments/vgs_da/train.py --config /data/vgs_uda/config.json \
  --mode smoke --steps 6 --output /data/vgs_uda/smoke
python -u experiments/vgs_da/train.py --config /data/vgs_uda/config.json --mode train
```

源域 2 张＋目标域 2 张，SGD 学习率 0.0005，辅助模块 AdamW 学习率 0.0002，预热 1K 后恒定。评估 XML 位于 `datasets/foggy_cityscapes_voc/VOC2007/Annotations/`。旧 `formal_25k`、`early_5k`、`extend_35k` 预设仍保留有图像级标签协议；无监督实验请明确使用 `uda_25k`。

完整断点用 `--resume /path/to/checkpoint_last.pth` 恢复，要求监督策略、固定权重内容和数据清单字节一致。不能从有图像级标签的旧断点恢复为 UDA。已通过 53 项测试、6 次 GPU smoke 更新、三雾评估读写与断点重载。

## 权重

9K 最佳推理权重：`/root/autodl-tmp/checkpoints/vgs_uda_ap5619_9k/model_best_ap5619.pth`。25K 推理与完整续训断点仍位于 `/root/autodl-tmp/formal_vgs_uda_20261007/run/`，文件分别为 `model_final.pth` 和 `checkpoint_last.pth`。9K 最佳文件不含优化器状态，不能作为完整训练断点。

固定 SourceB/RPN/文本/VGS 初始化依赖共用 `/root/autodl-tmp/checkpoints/vgs_da_ap5598_25k/fixed_assets/`；迁移时保持内容校验值一致。[无监督权重清单](results/uda_weight_manifest.json)记录最佳权重 SHA256。大权重不进入 Git。

```bash
python experiments/vgs_da/train.py --config /data/vgs_uda/config.json \
  --mode evaluate --resume /path/to/model_best_ap5619.pth \
  --output /data/vgs_uda/best_evaluation
```
