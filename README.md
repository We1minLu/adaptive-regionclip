# Adaptive RegionCLIP：VGS＋域适应补漏（无监督版本）

研究 **Cityscapes → Foggy Cityscapes 八类无监督域适应检测**：源域使用框标注；目标域训练不使用图像级类别标签或框标注，只使用图像和源/目标域身份。三种雾浓度混合训练与评估。

**出发点：**RPN 遗漏会限制最终检测。我们用源域 GT 学习语义条件补充搜索，让 ROI 头学习新增候选的真实视觉特征与位置，再通过域对抗促进雾天迁移。

**网络：**固定 RPN 最多 300 框 → 固定 SourceB 教师提供区域语义 → VGS 融合视觉 V、几何/覆盖 G、语义图 S，补充最多 200 框 → 合并候选，由学生 RegionCLIP ROI 路径分类、回归和 NMS。教师与文本分类器固定，学生视觉特征、框回归、VGS 和域判别器参与训练。

已完成 **25K 更新，每 1K 评估**：

| 无监督检查点 | 混合 AP50 | AP75 | 轻雾 / 中雾 / 浓雾 AP50 |
|---|---:|---:|---|
| **9K：验证集最佳** | **56.19** | 34.30 | 60.95 / 57.58 / 50.23 |
| 25K：固定训练终点 | 55.93 | 33.66 | 60.43 / 57.95 / 49.32 |

评估为 Foggy Cityscapes val 的 500 个场景 × 三种雾浓度，共 1,500 张图，采用 VOC2007 十一点 AP。混合 AP 由全部预测合并计算，不是三种浓度的平均；9K 为按验证集混合 AP50 选出的最佳点，不是独立测试成绩。

采用**固定教师＋可训练学生**，没有 EMA。教师软概率构造语义图，预测类别作为条件对抗的伪类别；没有目标域伪框分类/回归损失。两类域对抗共三个判别器：C3/C4 两头的平均损失外乘 **0.1**，GRL 为 `r(t)`；VGS 融合前视觉特征的条件对抗损失外乘 **1.0**，GRL 为 `0.05*r(t)`。源域检测和 VGS 补漏监督外层权重均为 1.0。

- [结构、损失及运行方式](experiments/vgs_da/README.md)
- [无监督完整评估曲线与结果摘要](experiments/vgs_da/results/uda_25k.json)
- [9K 最佳原始指标](experiments/vgs_da/results/uda_best_009000_metrics.json)、[25K 终点指标](experiments/vgs_da/results/uda_final_025000_metrics.json)
- [无监督最佳权重位置及校验](experiments/vgs_da/results/uda_weight_manifest.json)（大权重不提交 Git）
- [历史有图像级标签版本：AP50 55.98](docs/vgs_image_labels_AP5598.md)

历史 55.98 运行曾调整 ROI 分块并修复 AMP 更新，本轮沿用最终稳定实现；两者是历史参照，不能据此单独证明移除标签更优。完整同版本消融仍待补充。

基于 [RegionCLIP](https://github.com/microsoft/RegionCLIP) 和 [DA-Pro](https://github.com/Therock90421/DA-Pro)；[历史适配器方案与环境说明](docs/legacy_adapters_README.md)。
