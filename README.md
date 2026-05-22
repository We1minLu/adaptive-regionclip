# Adaptive RegionCLIP

Adaptive RegionCLIP extends RegionCLIP for domain-adaptive object detection. On
Foggy Cityscapes, the current best checkpoint reaches **54.1940 bbox AP50**.

## Current Adaptive RegionCLIP Design

This version adds multi-level quality-gated residual adapters to the RegionCLIP
C4 recognition branch. The offline RPN still generates proposals, while the
CLIP visual backbone produces C3/C4 feature maps and C5 ROI features for
recognition.

### Region quality score

For each proposal, a layer-specific ROI feature is projected into the CLIP
visual-language semantic space and compared with the RegionCLIP text classifier
weights. The class probability vector is summarized by the effective number of
classes:

```text
E_i = exp(H(p_i))
```

where `H` is entropy over the target classes. Lower `E_i` means a more certain
region prediction. The quality score combines certainty and perturbation
stability:

```text
certainty_i = (K - E_i) / (K - 1)
stability_i = exp(-abs(E_i_aug - E_i) / tau)
q_i = clamp(certainty_i * stability_i, 0, 1)
```

`K` is the number of detection classes, `E_i_aug` is computed after jittering
the same proposal box, and `tau` controls how strongly unstable regions are
penalized. In the current experiments, `tau=0.5` and the quality logit
temperature is inherited from the projection-training config.

### Zero-filled mean scatter

C3 and C4 produce region-level scores but need feature-map-level attention
maps. We view the scatter operation as a sparse matrix:

```text
A in R^(M x N)
```

where `M` is the number of feature-map locations and `N` is the maximum proposal
overlap count used as the zero-filled denominator. If proposal `j` covers
feature-map location `i`, then
`A[i, j] = q_j`; otherwise `A[i, j] = 0`. The attention value for location `i`
is the zero-filled row mean:

```text
Q_i = sum_j A[i, j] / N_max
```

The implementation computes this as `sum_map / max(count_map)`, where
`max(count_map)` is the largest number of proposals covering any location in the
same image. This is equivalent to averaging over a zero-filled proposal-overlap
matrix with a fixed per-image denominator. It deliberately dilutes isolated
high-quality regions and suppresses locations covered by many low-quality or
empty entries.

### Multi-level adapters

Adapters are inserted at C3, C4, and C5:

```text
f' = f_d + alpha_l * Q_l * Adapter_l(f_d)
```

For C3/C4, `Q_l` is the feature-map quality map from zero-filled mean scatter.
For C5, `Q_l` is the region-level quality score reshaped onto the ROI feature.
The current adapter block is a lightweight bottleneck:

```text
1x1 Conv(C_l -> 128) -> ReLU -> 1x1 Conv(128 -> C_l)
```

where `C_l` is 512 for C3, 1024 for C4, and 2048 for C5. Current main
experiments use residual scale `alpha=5.0` for C3/C4/C5.

### Projection heads

C3 and C4 use trainable linear projection heads from ROI-pooled features to the
1024-dimensional RegionCLIP semantic space:

```text
C3 ROI pooled feature -> mean pool -> Linear(512, 1024)
C4 ROI pooled feature -> mean pool -> Linear(1024, 1024)
```

These projection heads can be trained on source-domain Cityscapes supervision
using ground-truth boxes or sampled proposals, then frozen for target-domain
quality estimation. During adaptation they are used only to compute `E_i`,
`E_i_aug`, and `q_i`; they are not the detector classifier. C5 uses the normal
RegionCLIP ROI path and CLIP attention pooling/embedding for semantic scoring,
so it does not require a separate C5 projection head.

### Domain adversarial placement

The adversarial branch is attached after the quality-gated adapters. The C3
discriminator operates on the adapted C3 feature map, the C4 discriminator
operates on the adapted C4 feature map, and the C5 discriminator operates on
average-pooled adapted C5 ROI features. Source training keeps supervised
detection losses plus source-domain adversarial losses; target training keeps
target-domain adversarial losses.

The official implementation of `Learning Domain-Aware Detection Head with Prompt Tuning` ([arxiv](https://arxiv.org/abs/2306.05718)).

This codebase is based on [RegionCLIP](https://github.com/microsoft/RegionCLIP).

1. Put your dataset at './datasets/your_dataset'. Please follow the format of Pascal Voc.
For example:
- dataset
  - cityscapes_voc
    - VOC2007
      - Annotations
      - ImageSets
      - JPEGImages
  - foggy_cityscapes_voc
    - VOC2007
      - Annotations
      - ImageSets
      - JPEGImages

2. Put your pre-trained VLM model at somewhere you like, for example, './ckpt', and edit the MODEL.WEIGHTS in train_da_pro_c2f.sh.

3. Following RegionCLIP, generate class embedding and put it at somewhere you like, and edit the MODEL.CLIP.TEXT_EMB_PATH.

4. Training: train_da_pro_c2f.sh  Testing: test_da_pro_c2f.sh
Training is customizable. You can directly use the parameters of other VLMs as backbone and then adjust only domain-adaptive prompt. You can also follow the steps of Regionclip to customize a backbone on your own dataset, then conduct adaptation.


A training sample: 
1) Initial pre-trained model with VLM (like CLIP or RegionCLIP).
2) Set LEARNABLE_PROMPT.TUNING to False to fine-tune the pre-trained backbone with domain adversarial loss.
3) Set LEARNABLE_PROMPT.TUNING to True to freeze the backbone and tune a learnable domain-adaptive prompt on two domains. 
