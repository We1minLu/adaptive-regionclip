# RegionCLIP：语义条件补漏与域适应

本项目研究 Cityscapes → Foggy Cityscapes 八类检测。出发点是：常规 RPN 的候选遗漏会限制最终检测，而新增候选只有经过 ROI 头正确分类和回归才有价值。我们用源域 GT 学习“根据视觉、候选覆盖和语义信息补充搜索”，再通过域对抗促进这种能力向雾天迁移。

网络由以下部分组成：

- **固定 RPN + 固定源域教师**：保留前 300 个候选；教师给出八类加背景的概率。
- **VGS 补漏头**：V 为在线 C3/C4 视觉特征；G 为覆盖数量、有效性、候选置信度和宽高五通道；S 为九类加权概率与语义分歧十通道。融合后预测目标性、遗漏概率和框偏移，去重后补充最多 200 个候选。
- **真实 ROI 学习**：合并候选进入 RegionCLIP，源域 GT 同时训练检测与补漏。更新视觉骨干、res5、注意力池化和框回归；文本分类器、教师与原 RPN 固定。候选选择及坐标停止梯度，补漏头通过自身监督学习。
- **两路域对抗**：C3/C4 上加入传统 GRL 判别器（损失权重 0.1）；补漏分支在语义融合前按类别、尺度、冗余程度进行条件对抗。目标域图像级标签仅筛除不可靠教师类别，不使用目标域框监督或伪框检测损失。

因此本实验属于**允许目标域图像级标签的域适应**。最终分类使用固定文本空间中的 ROI 得分，没有双域 prompt、熵重排或额外质量分数融合。

**当前最佳混合 AP50 为 55.98（精确值 55.975298），对应正式训练 25K。**评估包含 500 个验证场景的三种雾浓度（0.005、0.01、0.02），共 1,500 张图，采用仓库原生 VOC2007 十一点 AP；混合 AP 合并全部预测计算，不是三种浓度 AP 的平均。续训至 35K 未超过此结果。目前尚无同协议完整消融，不能把全部成绩直接归因于 VGS 或域对抗。

## 运行

已验证环境为 Python 3.8、PyTorch 1.9.0+cu111、对应 torchvision 和本仓库 Detectron2。需要 GPU；原生分类器构造还依赖 OpenAI CLIP RN50 缓存。数据、权重和环境安装见仓库原有说明。

从仓库根目录执行。下面路径请按实际存放位置修改；评估 XML 还需位于 `datasets/foggy_cityscapes_voc/VOC2007/Annotations/`。

```bash
python experiments/vgs_da/prepare_data.py \
  --repo-root "$PWD" --output-dir /data/vgs_da/manifests \
  --city-root /data/cityscapes \
  --fog-root /data/foggy_cityscapes/leftImg8bit_foggy

python experiments/vgs_da/configure.py \
  --preset formal_25k --repo-root "$PWD" \
  --manifest-dir /data/vgs_da/manifests --output-dir /data/vgs_da/run \
  --source-checkpoint /data/weights/source_B.pth \
  --source-search-checkpoint /data/weights/source_vgs.pt \
  --rpn-checkpoint /data/weights/rpn_coco_48.pth \
  --text-embeddings /data/weights/cityscapes_8_cls_emb.pth \
  --config-out /data/vgs_da/config.json

python experiments/vgs_da/train.py --config /data/vgs_da/config.json \
  --mode smoke --steps 3 --output /data/vgs_da/smoke
python -u experiments/vgs_da/train.py --config /data/vgs_da/config.json \
  --mode train
```

`formal_25k` 每 5K 评估；`early_5k` 从原始初始化训练至 5K，含 0K 评估、之后每 500 步评估；`extend_35k` 从 25K 完整断点续训，每 1K 评估。三者保留原 25K GRL 调度。正式设置为源域 2 张 + 目标域 2 张，SGD 学习率 0.0005，辅助模块 AdamW 学习率 0.0002，预热 1K 后恒定。续训需追加 `--resume /path/to/checkpoint_resume_025000.pth`。

## 最佳权重

远程归档目录：`/root/autodl-tmp/checkpoints/vgs_da_ap5598_25k/`。

- `model_best_ap5598.pth`：25K 推理权重。
- `checkpoint_resume_025000.pth`：含优化器等状态的完整恢复断点。

权重不进入 Git；仓库只记录路径、校验值及精简结果。上述权重仍依赖原 SourceB 权重、源域 VGS 初始化、RPN、文本嵌入和 `configs/source_B.yaml`；迁移路径时须保持文件内容与校验值一致。

完整断点续训还要求数据清单逐字节一致：可以搬移清单目录，但重新生成含不同图像路径的清单会被拒绝；此时应恢复原路径或仅加载推理权重评估。

```bash
python experiments/vgs_da/train.py --config /data/vgs_da/config.json \
  --mode evaluate --resume /path/to/model_best_ap5598.pth \
  --output /data/vgs_da/best_evaluation
```

## 目标域无标签对照

`uda_25k` 从相同 SourceB 和源域 VGS 初始化重新训练 25K，每 1K 评估；其余超参数沿用上述稳定配置。先运行 `prepare_uda.py --input-dir 原清单目录 --output-dir 新清单目录`，再用 `configure.py --preset uda_25k` 配置该新目录。准备程序保持源域、评估清单和目标图像顺序不变，只移除目标训练记录中的 `image_labels`，不复制标签侧文件。

目标域只取消条件域对抗的图像类别存在性过滤，仍以冻结教师类别概率 ≥0.7、RPN 概率 ≥0.5 选择候选，并保持原分组、排序和损失。源域仍使用完整 GT；目标域类别标签与框标注均不进入训练。数据加载器和模型会拒绝携带目标标签的 UDA 输入，也禁止从使用目标标签的断点完整续训。测试流程始终不需要图像级标签。

此对照沿用原实验最终稳定的 ROI chunk 与 AMP 更新实现；历史 55.98 运行中曾调整 chunk 并修复 AMP，因此它是历史参照，严格因果消融仍需同版本重跑有标签组。新的 AP 需等待训练，不能预先判断标签过滤的影响大小。
