# 强弱增强＋EMA 图像级一致性

在无监督 VGS＋DA 版本上新增教师—学生图像级学习。从相同 SourceB＋源域 VGS 初始化重新训练 25K，每 1K 评估三种雾浓度混合的 1,500 张验证图；学生在 **8K 达到最佳混合 AP50 58.36**，25K 终点为 **55.19**。推理仍保留 EMA 教师为学生 VGS 提供语义图，不是完全移除教师的单分支推理。

## 数据和结构

- 同一图像的强弱视图共享水平翻转和 1024×2048 坐标。弱视图没有额外外观变换；强视图按 Adaptive Teacher 官方顺序使用颜色扰动、灰度化、模糊及三次随机擦除。保持原分辨率，不新增 resize/crop。
- 学生对源、目标均读取强视图，保证域判别器不会仅区分增强类型。源域仍只进行一次 GT 检测和 VGS 监督；未照搬 AT 的源域强弱双份监督。
- 教师读取弱视图。师生分别通过固定 RPN300＋各自 VGS 最多 200 个补充候选，得到 ROI 分类 logits，在最终检测阈值/NMS **之前**聚合。候选无需逐框配对。
- 原固定 SourceB 语义教师改为 EMA 教师，所以 S 语义图与条件对抗所用预测类别也随 EMA 更新。学生强视图的候选在共享坐标下由弱教师特征提供语义。
- 教师 backbone、VGS、bbox 参数从学生复制，每次成功优化后更新一次：`teacher = 0.9996*teacher + 0.0004*student`。固定文本、背景分类器和 RPN 不更新；域判别器不属于教师。不额外增加 burn-in。

## H2FA 聚合与新损失

每张图的前景 logits 为 `z[N,8]`，raw objectness logits 为 `o[N]`。令 `a[n]=argmax(z[n])`，仅把 `o[n]` 放进类别 `a[n]` 的列，其余列填 **0**，得到 `o_bar[N,8]`：

`P[c] = sum_n softmax(z, dim=class)[n,c] * softmax(o_bar, dim=proposal)[n,c]`

不新增分类头。RegionCLIP 原生重复的 18 维输出只取第一组的 8 个前景 logits，背景不参与这里的类别 softmax。补框筛选仍按 `sigmoid(obj)*sigmoid(miss)`；聚合回取所选 anchor 的 raw `obj`，保留其梯度。

`L_image = BCE_mean(P_student_strong, stopgrad(P_teacher_weak))`

新损失权重 **1.0**，使用软教师概率，不使用目标域真实图像标签，也不生成目标伪框分类/回归损失。原 H2FA 用真实图像标签；本轮是借用其聚合方式的 EMA 一致性扩展。新损失能更新学生视觉特征和补框 objectness，不能直接监督框位置。源域监督仍负责定位。

总损失：

`L = L_source_detection + L_source_VGS + 0.1*L_traditional_DA + 1.0*L_conditional_DA + 1.0*L_image`

传统对抗保留 C3/C4 两个判别器，GRL=`r(t)`；条件对抗保留一个条件 MLP，GRL=`0.05*r(t)`。这两条对抗及其权重不变。

## 已完成结果

8K：混合 AP50 **58.356487**、AP75 **34.997688**；轻/中/浓雾 AP50 **62.12 / 59.38 / 53.81**。25K：混合 AP50 **55.190529**、AP75 **33.559875**。完整 25 个评估点见[结果记录](results/ema_image_25k.json)。

相较前轮固定教师 UDA，最佳提升 2.17 点，固定终点降低 0.74 点，后期存在波动。8K 是验证集选优结果；本轮同时加入强弱增强、EMA 和图像级一致性，单项贡献尚未消融。原始指标与权重清单见[运行说明](README.md)。

## 配置与验证

配置：[configs/ema_image_25k.json](configs/ema_image_25k.json)。源/目标微批各 2，SGD 学习率 0.0005，辅助 AdamW 0.0002，其余训练预算沿用 UDA。V100 32GB 上 GPU smoke 已验证：学生 ROI chunk512、教师 chunk256，关闭 res5 checkpoint；每步约 2.3 秒，峰值已分配显存约 11.2 GiB，优于初始 chunk128＋checkpoint 的约 2.65 秒。增强、batch、学习率和损失权重均不因这次吞吐量调优改变。

增强种子由样本序号确定，续训不依赖 worker 预取进度。EMA 不在前向、梯度累积或 AMP 溢出重试时更新；checkpoint 保存完整教师状态与更新次数。新版 checkpoint 与原固定教师版本区分，不能将旧权重当完整断点恢复。

监测教师/学生的 8 维概率、概率差、RPN/VGS 聚合贡献和 objectness 范围。软 BCE 在师生相同时等于教师熵，通常不为零；不能仅看损失是否接近零判断成功。图像级一致性也不保证定位正确，效果仍须看 AP。

参考：[Adaptive Teacher 官方增强](https://github.com/facebookresearch/adaptive_teacher/blob/5256463ad9ec90fd5ba84ebb8d53bed56bd369df/adapteacher/data/detection_utils.py#L20-L43)、[Adaptive Teacher 论文](https://arxiv.org/abs/2111.13216)、[H2FA 官方聚合](https://github.com/XuYunqiu/H2FA_R-CNN/blob/499c1d90833c475fd9f288bb6bb90829989def64/detectron2/modeling/roi_heads/roi_heads.py#L501-L531)、[H2FA 论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_H2FA_R-CNN_Holistic_and_Hierarchical_Feature_Alignment_for_Cross-Domain_Weakly_CVPR_2022_paper.pdf)。
