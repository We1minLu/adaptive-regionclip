> 历史版本：允许目标域图像级类别标签；不代表当前无监督结果。

# Adaptive RegionCLIP：VGS＋域适应补漏

研究 **Cityscapes → Foggy Cityscapes 八类目标检测**，允许使用目标域图像级类别标签，目标域框仅用于评估。

**出发点：**RPN 漏掉的目标无法被后续分类头恢复。我们在源域用 GT 学习语义条件补充搜索，并让 ROI 头学习新增候选的真实视觉特征和位置，再用域对抗促进这种能力向雾天迁移。

**网络结构：**固定 RPN 提供 300 个候选，固定源域 RegionCLIP 教师提供区域语义；VGS 融合视觉特征 V、候选几何/覆盖 G 和语义图 S，生成最多 200 个补充候选。合并候选进入可训练的 RegionCLIP ROI 视觉与框回归路径。训练结合源域检测/补漏监督、传统 C3/C4 域对抗（权重 0.1）和补漏分支的条件域对抗；文本分类器保持固定。

**当前最佳：25K 时混合 AP50＝55.98，AP75＝33.73。**评估使用 Foggy Cityscapes val 的 500 个场景×三种雾浓度，共 1,500 张图，采用 VOC2007 十一点 AP。轻雾/中雾/浓雾 AP50 分别为 60.80/57.33/49.12；混合 AP 不是三者平均。35K 续训未超过该混合 AP50，完整消融仍待补充。

- [代码、配置和运行方式](../experiments/vgs_da/README.md)
- [最佳结果原始指标](../experiments/vgs_da/results/best_025000_metrics.json)
- [权重位置与校验清单](../experiments/vgs_da/results/weight_manifest.json)（大权重不提交 Git）
- [历史适配器方案与原项目使用说明](legacy_adapters_README.md)

本项目基于 [RegionCLIP](https://github.com/microsoft/RegionCLIP) 和 [DA-Pro](https://github.com/Therock90421/DA-Pro)。历史方案与本轮评估协议不同，分数不直接作公平对比。
