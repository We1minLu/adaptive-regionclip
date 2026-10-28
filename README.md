# Adaptive RegionCLIP：DeST 跨域目标检测

**Cityscapes → Foggy Cityscapes 八类无监督域适应检测**。DeST 的 25K 训练已于 2026-10-28 完成：**22K 验证最佳混合 mAP@0.5 为 62.01%，25K 固定终点为 61.69%**。源域使用框标注；目标域训练仅使用图像和域身份，不读取真实类别标签或框标注。

## 已完成结果

| 检查点 | 混合 mAP@0.5 | 混合 AP75 | 轻雾 0.005 AP50 | 中雾 0.01 AP50 | 浓雾 0.02 AP50 |
|---|---:|---:|---:|---:|---:|
| SourceB 初始化，0K | 51.20 | 30.26 | 56.63 | 52.69 | 44.59 |
| **DeST，22K 验证最佳** | **62.01** | **38.42** | **64.37** | **63.13** | **58.91** |
| DeST，25K 固定终点 | 61.69 | 37.68 | 65.01 | 63.31 | 57.72 |

评估采用 Foggy Cityscapes **val** 的 500 个场景、三种雾浓度，共 **1,500 张图**，报告八类的仓库原生 **VOC2007 十一点 AP**。表中的混合 mAP@0.5（AP50）将三种浓度的全部预测合并计算，既不是分浓度 AP 的平均，也不是仅浓雾成绩或 COCO AP50:95。类别为 person、rider、car、truck、bus、train、motorcycle、bicycle。

固定 25K 终点是预设主报告检查点。最佳权重按验证集混合 AP50 选择，候选包含 **初始 0K 与每 1K 一次的 25 个适配后评估点，共 26 次完整评估**；最佳出现在 22K，属于验证集选择结果，不是独立测试成绩。22K 和 25K 相比初始化分别提高 **10.80** 和 **10.49** 个百分点。

本次在 NVIDIA V100 32GB 上完成 **25,000 次成功更新与 25,000 次 EMA 更新**，没有跳过更新，累计 8 次 AMP 重试；记录的运行时长为 80,816.694 秒（约 22.45 小时）。详细数据见[结果摘要](experiments/dest/results/c2f_25k_summary.json)、[完整评估曲线](experiments/dest/results/c2f_evaluation_history.json)、[22K 原始指标](experiments/dest/results/c2f_best_022000_metrics.json)和[25K 原始指标](experiments/dest/results/c2f_final_025000_metrics.json)。

## 方法与训练结构

学生与 EMA 教师从同一个已完成源域训练的 **SourceB** 检查点初始化。检测器保留 RegionCLIP 的定位流与 CLIP 识别流，定位主干和 RPN 参与适配训练，文本与背景分类器保持固定。

- **教师语义证据：**弱增强目标图像产生最多 300 个 RPN 候选，并加入 284 个多尺度窗口。MRSA 使用完整前景＋背景分类概率形成前景 log-odds，以归一化区域相交核约束证据聚合，输出八类图像级软目标。窗口只提供语义证据。
- **学生图像级学习：**学生处理强增强图像，将自身最终筛选/NMS 前的 ROI 分类结果和 RPN objectness 按 H2FA 方式聚合为图像概率，再与教师软目标计算 BCE。
- **宽松空间监督：**只使用来自教师 RPN 的最终检测框，按 1.5 倍扩展其支持区域；若区域内学生锚点的最大 objectness 达到 0.7，即不再惩罚。没有教师提示的位置不因此标成背景，也不进行目标域逐框匹配或框回归。
- **源域与域对齐：**源域 GT 训练 RPN 和 ROI 分类、回归；识别流 C3/C4 保留两个 GRL 域判别器。DeST 不使用历史 VGS 补框模块或条件域判别器。

推理与评估使用**完整 EMA 教师检测器**：RPN → ROI 分类/回归 → NMS，不运行 MRSA 窗口聚合或空间监督。历史 VGS 版本评估的是学生检测输出并使用 EMA 语义图，两者模型与训练设置均有变化，不能据成绩差异单独归因于某个模块。

| 训练项 | 权重 / 设置 |
|---|---|
| 源域 RPN 与 ROI 检测损失 | 各 1.0 |
| 目标图像级语义一致性 | 1.0 |
| 目标宽松空间损失 | 0.1 |
| C3/C4 域对抗损失 | 0.1，两个判别器损失取平均 |
| 批量与预算 | 每步源域 2 张＋目标域 2 张，25K 步 |
| 优化器 | 检测器 SGD 0.0005；域判别器 AdamW 0.0002 |
| 学习率与 EMA | 1K 线性预热后恒定；EMA 0.9996 |

完整设置见[配置文件](experiments/dest/configs/c2f_mrsa_25k.json)，启动、续训、评估和实现细节见 [DeST 实验说明](experiments/dest/README.md)。配置中的数据与预训练资产路径需要按实际环境准备；目标训练集不含真实标签。

## 权重来源与保存

本次训练代码基于提交 `e089bc781e255aa47ef2b44db412de18e8a7815e`。SourceB 来自源域训练的 `model_best.pth`，检查点记录 `iteration=24499`，SHA256 为 `f199ea87709207a0b2ee23dd63dd49d4d3b540c8e9eb9c213d00f3e449a5e861`；本次未加载历史 VGS 权重或 AP58.36 的域适应权重。

22K 最佳检查点已归档至训练服务器：

```text
/root/autodl-tmp/checkpoints/dest_mrsa_c2f_ap6201_22k_20261028/model_best_ap6201_step022000.pth
SHA256: edbba29d310ae17f87d43528f95a1e4dd5459ee3d8f6da72db1090352afcee58
```

权重文件不纳入 Git；[权重清单](experiments/dest/results/c2f_weight_manifest.json)记录保存位置、来源和校验值。最佳与固定终点模型文件包含学生、教师、域判别器及来源信息，评估实际使用其中的 EMA 教师；完整续训检查点另外保存优化器、AMP scaler 和随机状态。

## 代码导航与历史记录

| 路径 | 内容 |
|---|---|
| [experiments/dest/model.py](experiments/dest/model.py) | 学生/教师检测器、训练损失与推理路径 |
| [experiments/dest/core.py](experiments/dest/core.py) | 多尺度窗口、MRSA 求解与宽松空间损失 |
| [experiments/dest/rpn_support.py](experiments/dest/rpn_support.py) | 原生 RPN 候选选择与锚点梯度通路 |
| [experiments/dest/train.py](experiments/dest/train.py) | 训练、AMP 重试、EMA、评估和检查点保存 |
| [experiments/dest/protocol.py](experiments/dest/protocol.py) | 配置检查、实验身份与续训边界 |
| [experiments/dest/preflight.py](experiments/dest/preflight.py) | 数据和初始资产校验 |
| [experiments/dest/tests](experiments/dest/tests) / [test_model.py](experiments/dest/test_model.py) | 核心数学、训练流程和模型检查 |
| [experiments/dest/results](experiments/dest/results) | 评估证据、运行摘要与权重清单 |
| [experiments/vgs_da](experiments/vgs_da) | 历史实验及 DeST 复用的数据、增强、EMA、H2FA、AMP 和评估工具 |

历史 VGS＋EMA 版本的最佳混合 AP50 为 **58.36（8K）**，固定 25K 为 **55.19**；原首页完整保存在[历史说明快照](docs/ema_image_ap5836_snapshot.md)。更早的[固定教师 UDA](experiments/vgs_da/UDA_FIXED_TEACHER.md)和[有图像级标签版本](docs/vgs_image_labels_AP5598.md)也保留独立记录。DeST 启动前的 CPU 验证文件保留当时状态；当前 GPU 训练已完成，以本次结果摘要为准。

基于 [RegionCLIP](https://github.com/microsoft/RegionCLIP) 与 [DA-Pro](https://github.com/Therock90421/DA-Pro)，复用 [Adaptive Teacher](https://github.com/facebookresearch/adaptive_teacher) 风格增强与 [H2FA R-CNN](https://github.com/XuYunqiu/H2FA_R-CNN) 聚合。历史环境与上游说明见[环境文档](docs/legacy_adapters_README.md)。
