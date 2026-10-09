# Adaptive RegionCLIP：VGS＋域适应＋EMA 图像级一致性

研究 **Cityscapes → Foggy Cityscapes 八类无监督域适应检测**。源域有框标注，目标域仅使用图像与域身份；不使用目标域真实图像级类别标签或框标注。三种雾浓度（0.005 / 0.01 / 0.02）混合训练与评估。

**出发点：**用源域 GT 学习语义条件补充搜索，找回 RPN 遗漏的候选，并让 ROI 头学习其真实视觉特征与位置。本轮在 VGS＋域对抗基础上加入强弱增强、EMA 教师及图像级一致性，旨在提高雾天类别证据的稳定性。

## 当前结构与改动

- **检测主干：**固定 RPN 最多 300 框 → VGS 融合视觉 V、几何/覆盖图 G（5 通道）与教师语义图 S（10 通道），补充最多 200 框 → 学生 RegionCLIP ROI 分类、回归和 NMS。学生 C3/C4、ROI res5/注意力池化、bbox 和 VGS 参与训练，RPN 与文本/背景分类器固定。
- **强弱视图：**教师看弱图，学生对源/目标均看强图；共享水平翻转与 1024×2048 坐标。强增强借鉴 Adaptive Teacher 的颜色扰动、灰度、模糊和随机擦除；不新增缩放/裁剪，源域保持一次 GT 监督。
- **EMA：**原固定教师改为每次成功优化后更新，系数 **0.9996**；包含 backbone、VGS 与 bbox。EMA 教师也提供语义图与条件域对抗的预测类别。
- **图像级学习：**师生各自生成合并候选，在最终检测 NMS 前按 H2FA 方式聚合为八维图像概率。学生学习教师的软概率，BCE 权重 **1.0**，无需逐框对应。**没有目标域伪框分类/回归损失。**新损失可更新视觉特征与补框 objectness，定位仍靠源域 GT。

评估报告的是**学生最终检测结果**；推理仍保留 EMA 教师 backbone/classifier 为 VGS 提供语义图，教师 VGS/bbox 不参与学生检测输出。

## 损失与参数

| 训练项 | 损失权重 | 说明 |
|---|---:|---|
| 源域检测与 VGS 监督 | 各 1.0 | 源域 GT 分类、定位和补充搜索 |
| 传统域对抗 | 0.1 | C3/C4 两个判别器损失平均；GRL=`r(t)` |
| 条件域对抗 | 1.0 | 一个条件 MLP；GRL=`0.05*r(t)` |
| 目标图像级一致性 | 1.0 | `BCE(P_student_strong, stopgrad(P_teacher_weak))` |

共 **两类域对抗、三个判别器网络**。`r(t)=2/(1+exp(-10*t/25000))-1`；0.05 是条件 GRL 系数上限，不是其损失外层权重。

从相同 **SourceB＋源域 VGS** 初始化重新训练：**25K 步，每 1K 评估**；batch 为源域 2＋目标域 2，无梯度累积。学生检测器 SGD 学习率 **0.0005**（momentum 0.9），辅助模块 AdamW **0.0002**，weight decay 均为 0.0001；预热 1K（起始倍率 0.01）后恒定。使用 AMP、梯度裁剪 10、EMA 0.9996，无额外 burn-in。完整参数见 [配置](experiments/vgs_da/configs/ema_image_25k.json)。

## 已完成结果

| 实验 / 检查点 | 混合 AP50 | AP75 | 轻雾 / 中雾 / 浓雾 AP50 |
|---|---:|---:|---|
| **本轮 EMA 图像一致性，8K 最佳** | **58.36** | **35.00** | **62.12 / 59.38 / 53.81** |
| 本轮 25K 固定终点 | 55.19 | 33.56 | 59.62 / 55.65 / 50.20 |
| 前轮固定教师 UDA，9K 最佳 | 56.19 | 34.30 | 60.95 / 57.58 / 50.23 |
| 前轮固定教师 UDA，25K 固定终点 | 55.93 | 33.66 | 60.43 / 57.95 / 49.32 |

最佳 AP50 提高 **2.17 个百分点**，但固定终点下降 **0.74 个百分点**，后期仍有波动。本轮同时改变强弱增强、EMA 和图像级学习，尚不能单独归因于某一项。

评估采用 Foggy Cityscapes **val**：500 场景×三种浓度，共 1,500 张图；原生 VOC2007 十一点 AP。混合 AP 由全部预测合并计算，不是分浓度 AP 的平均。8K 是 25 个验证点中选出的最佳点，**不是独立测试成绩**。

- [运行方法与权重](experiments/vgs_da/README.md)
- [聚合公式与实现边界](experiments/vgs_da/EMA_IMAGE_CONSISTENCY.md)
- [25 个评估点与结果摘要](experiments/vgs_da/results/ema_image_25k.json)
- [8K 原始指标](experiments/vgs_da/results/ema_image_best_008000_metrics.json)、[25K 原始指标](experiments/vgs_da/results/ema_image_final_025000_metrics.json)
- [最佳权重位置与 SHA256](experiments/vgs_da/results/ema_image_weight_manifest.json)（大权重不进入 Git）
- 历史：[固定教师 UDA，AP56.19](experiments/vgs_da/UDA_FIXED_TEACHER.md)；[有图像级标签版本，AP55.98](docs/vgs_image_labels_AP5598.md)

基于 [RegionCLIP](https://github.com/microsoft/RegionCLIP) 与 [DA-Pro](https://github.com/Therock90421/DA-Pro)，借鉴 [Adaptive Teacher](https://github.com/facebookresearch/adaptive_teacher) 的增强和 [H2FA R-CNN](https://github.com/XuYunqiu/H2FA_R-CNN) 的聚合。原 H2FA 使用真实图像标签，本项目替换为 EMA 软预测；[历史环境说明](docs/legacy_adapters_README.md)。
