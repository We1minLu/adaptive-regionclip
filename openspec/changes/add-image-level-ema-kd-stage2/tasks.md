## 1. Configuration

- [x] 1.1 Add config defaults for enabling Stage2 image-level EMA KD, EMA decay, warmup, loss weight, and strong augmentation probabilities.
- [x] 1.2 Create a Stage2 experiment config that enables image-level EMA KD while preserving the existing v3 Stage2 settings.

## 2. Model Outputs and Aggregation

- [x] 2.1 Add a model/ROI-head path that returns per-image ROI class logits and proposal objectness logits without changing normal training and inference behavior.
- [x] 2.2 Implement H2FA image-level aggregation from ROI logits and offline RPN objectness logits, excluding background logits.

## 3. Teacher-Student Training

- [x] 3.1 Add EMA teacher creation and update logic for Stage2 training when image-level EMA KD is enabled.
- [x] 3.2 Add target strong-view photometric/occlusion augmentation without geometric transforms.
- [x] 3.3 Add target image-level KD loss with warmup and configurable weight while preserving existing source supervised and GRL losses.

## 4. Verification

- [x] 4.1 Add or run a focused dry-run that builds the enabled config, creates the teacher/student path, and computes one training iteration loss dictionary.
- [x] 4.2 Run a short smoke train or equivalent script check to confirm losses are finite and offline RPN parameters remain frozen.
