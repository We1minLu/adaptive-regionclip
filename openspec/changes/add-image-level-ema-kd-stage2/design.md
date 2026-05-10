## Context

The current Stage2 training path uses source supervised detection losses and target/source GRL domain-adversarial losses with `LEARNABLE_PROMPT.TUNING=False`. The offline RPN/backbone is frozen and supplies proposals plus objectness logits; the recognition branch and ROI box regression head remain trainable. Previous prompt-tuning experiments showed limited gain, so this change explores a lower-risk target-domain classification signal before adding instance-level pseudo labels.

The proposed method combines Adaptive Teacher-style EMA teacher/student training with H2FA-style image-level prediction aggregation. It keeps geometry unchanged between teacher and student target views to avoid box-coordinate mapping and to keep image-level class labels semantically consistent.

## Goals / Non-Goals

**Goals:**

- Add an optional Stage2 EMA teacher initialized from the student model.
- Generate target-domain teacher image-level class probabilities from weak views using frozen offline proposals and recognition-head ROI logits.
- Generate student image-level class probabilities from strong views using H2FA aggregation.
- Train the student with an image-level consistency loss in addition to existing Stage2 losses.
- Keep the implementation switchable through config and preserve existing Stage2 behavior when disabled.

**Non-Goals:**

- No Stage3 prompt tuning.
- No box pseudo labels, box regression pseudo loss, or instance-level distillation.
- No independent random crop, random resize crop, random horizontal flip, or box coordinate mapping between teacher and student.
- No training or unfreezing of the offline RPN/backbone.
- No external dependency on the Adaptive Teacher codebase.

## Decisions

1. **Use EMA teacher as a separate model copy.**

   The teacher will be created from the student architecture, loaded from the same checkpoint, set to eval/no-grad mode, and updated with EMA after student optimizer steps. This matches Adaptive Teacher while keeping the current Detectron2 training structure.

   Alternative considered: reuse the student with detached predictions. This would remove EMA stabilization and would not test the teacher-student hypothesis.

2. **Use no geometric difference between teacher and student target views.**

   Teacher target images use the standard mapper/preprocessing path. Student target images receive only photometric/occlusion strong augmentations: color jitter, random grayscale, Gaussian blur, and cutout. This avoids H2FA image labels becoming invalid when a crop removes an object class.

   Alternative considered: full Adaptive Teacher weak random crop/flip and strong augmentation. This is better suited for box pseudo-label training, but it introduces coordinate mapping and class-presence ambiguity that are out of scope for image-level-only consistency.

3. **Use H2FA aggregation for image-level class probabilities.**

   For each image, ROI class logits `x` and proposal objectness logits `o` are aggregated as:

   `P_c = sum_n softmax_row(x)_{n,c} * softmax_col(o_bar)_{n,c}`

   where `o_bar[n, argmax_c x[n,c]] = o[n]` and other entries are zero. Background logits are excluded before aggregation.

   Alternative considered: max or top-k mean over ROI probabilities. H2FA uses both classification confidence and proposal objectness and is the method requested for this experiment.

4. **Apply only image-level target consistency loss.**

   The initial loss should use teacher probabilities as detached soft targets for student probabilities, with a configurable weight and warmup iteration. BCE over multi-label class probabilities is preferred because each image can contain multiple classes.

   Alternative considered: hard image-level labels from thresholded teacher probabilities. Soft targets preserve uncertainty and reduce threshold sensitivity in the first experiment.

5. **Keep existing Stage2 training intact.**

   Source supervised losses, target/source adversarial losses, dataset registration, and evaluation remain unchanged. The new path only adds optional target strong augmentation, teacher forward, H2FA aggregation, image-level loss, and EMA update.

## Risks / Trade-offs

- **Risk: Teacher probabilities are noisy early in training.** → Mitigate with warmup before enabling image-level KD and by initializing from the Stage2/source checkpoint.
- **Risk: Cutout or blur may obscure small objects and make student image-level probabilities lower than teacher targets.** → Mitigate with configurable probabilities and a conservative default loss weight.
- **Risk: H2FA aggregation depends on offline RPN objectness calibrated for the target images.** → Mitigate by preserving the same frozen offline path already used by Stage1/Stage2 and logging image-level loss separately.
- **Risk: Separate teacher model increases GPU memory.** → Mitigate by freezing teacher gradients, running teacher forward under `torch.no_grad()`, and allowing the feature to be disabled by config.
- **Risk: Existing checkpoints may not contain new EMA-teacher-specific state.** → Mitigate by deriving teacher weights from the loaded student at startup rather than requiring extra checkpoint keys.
