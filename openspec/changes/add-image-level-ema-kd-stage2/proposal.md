## Why

Stage2 domain-adversarial training improves target-domain performance but leaves target-domain classification consistency weak. The next experiment should add a low-risk teacher-student signal on target images without entering Stage3 prompt tuning, without box pseudo labels, and without unfreezing the offline RPN.

## What Changes

- Add an EMA teacher for Stage2 training, initialized from the student recognition model and updated from student weights.
- Add target-domain image-level classification consistency using H2FA-style aggregation over ROI classification logits and offline RPN objectness logits.
- Use weak/strong target views with no geometric augmentation:
  - teacher weak view uses the normal preprocessing path;
  - student strong view adds Adaptive Teacher-style photometric/occlusion perturbations: color jitter, random grayscale, Gaussian blur, and cutout.
- Keep the current source supervised losses and GRL domain-adversarial losses.
- Do not add box pseudo labels, box regression pseudo loss, or instance-level distillation in this change.
- Do not unfreeze or train the offline RPN/backbone.

## Capabilities

### New Capabilities
- `stage2-image-level-ema-kd`: Stage2 training can use an EMA teacher to produce target image-level class probabilities and train the student with image-level consistency.

### Modified Capabilities

## Impact

- Affected training entry points: `tools/train_net.py`, `detectron2/engine/defaults.py`, and `detectron2/engine/train_loop.py`.
- Affected model code: `detectron2/modeling/meta_arch/clip_rcnn.py`, ROI-head/box-predictor paths needed to expose ROI logits for aggregation, and config defaults.
- Affected data/augmentation code: target-domain train mapper or training loop preprocessing for strong target augmentations.
- Dataset assumptions: source is Cityscapes train VOC; target training is Foggy Cityscapes train across three fog densities; target evaluation remains Foggy Cityscapes val across three fog densities.
- Checkpoint assumptions: Stage2 starts from the archived source-only best checkpoint unless overridden, and EMA teacher starts from the same loaded student weights.
- CLIP/text-embedding assumptions: classification logits continue to use the current RegionCLIP/CLIP text-embedding classifier; prompt tuning remains disabled.
