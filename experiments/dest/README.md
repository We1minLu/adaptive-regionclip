# DeST: MRSA and relaxed spatial supervision

This is a new C2F experiment based on the paper formulation. It is not a continuation of the 8K breakthrough checkpoint, and it has no measured detection result yet. Historical `experiments/vgs_da` files are retained unchanged.

## Preserved historical result

GitHub `master` at `f9d98fc76c95e2b8e6083030169663626e11babe` already archives the previous best run. Its 12 runtime Python files match `/root/autodl-tmp/formal_vgs_ema_image_20261008/code` byte-for-byte. The observed 8K mixed AP50 was 58.356486657395756, using the native VOC07 evaluator on 1,500 foggy validation images (500 scenes, three fog densities). This was the highest observed validation checkpoint, not the fixed 25K result or an independent test-set estimate.

The historical inference checkpoint is `/root/autodl-tmp/checkpoints/vgs_ema_image_ap5836_8k/model_best_ap5836.pth`, SHA256 `1aa47033f5de2d102c01ca3329d0e173aa229eff3c0f4adff8fdc5b40f27d5d9`.

## Model and supervision

- Both student and EMA teacher start from the same completed source-only **SourceB** checkpoint used before: `model_best.pth`, SHA256 `f199ea87709207a0b2ee23dd63dd49d4d3b540c8e9eb9c213d00f3e449a5e861` (source iteration 24,499). The source-trained VGS module and the adapted 8K weights are not loaded.
- Preserve RegionCLIP's native localization backbone and RPN, including their initial weights. Unfreeze the localization stream with the standard early-stage freeze. Train it with source RPN classification/regression and the target spatial objective. The CLIP recognition stream supplies regional semantics and C3/C4 domain features.
- Teacher evidence comprises its 300 RPN proposals and 284 windows at relative scales 1/8, 1/4, 1/2 and 1. All use the same region encoder and fixed text embeddings. Window rows enter **only** MRSA, never final detections or spatial hints.
- MRSA converts full foreground-plus-background category probabilities to foreground log-odds. It solves the class-wise simplex problem `max w^T s - rho sqrt(w^T K w)`, with `K` the normalized region-intersection Gram matrix, using Frank–Wolfe and line search. A sigmoid yields eight image-level soft targets. Teacher evidence and weight solving have no gradients.
- The student retains the previous H2FA aggregation of its own ROI class logits and proposal objectness. These are its predicted regions **before** final score filtering/NMS, as in the archived implementation. Selected objectness logits retain gradients through their anchor indices. Soft BCE aligns the resulting image-level scores with the teacher targets.
- Spatial hints come only from final teacher detections whose inputs originated in the RPN. Each hint is expanded, and the loss is `max(0, tau - max sigmoid(anchor logits))` over student anchors centered inside that support. Sufficient response incurs no penalty; positions without a teacher hint are not labeled background. Source detection losses still provide ordinary foreground/background and box supervision.
- Source detection and C3/C4 GRL alignment remain active. There is no VGS/search loss, conditional discriminator, teacher-box matching, or target box regression. Inference uses the EMA detector without MRSA windows or spatial supervision.

## First-run settings

The inherited training budget is 25K updates, with two source and two target images per update, a 1K linear warmup, detector SGD at 0.0005, auxiliary AdamW at 0.0002, EMA decay 0.9996, and evaluation every 1K updates. Source and target views retain the previous augmentation and geometry protocol. The target training loader contains no ground-truth boxes or image labels.

The new method settings are initial experimental choices, not validated optima: MRSA `rho=1`, maximum 100 iterations and tolerance 1e-4; spatial expansion 1.5, response threshold 0.7 and detection threshold 0.5. Semantic/spatial/domain loss weights are 1/0.1/0.1. Diagnostics record the raw losses, semantic gap, solver convergence and active spatial supports so that ineffective or saturated supervision can be identified.

Primary reporting uses the fixed final checkpoint; the highest observed validation checkpoint is saved separately. The previous breakthrough evaluated the student using EMA semantic maps; this version evaluates the full EMA detector, as specified by the paper. A future controlled comparison must keep the evaluated model and other training choices matched.

## Run after GPU activation

Remote checkout: `/root/autodl-tmp/dest_mrsa_20261027/repo`.
Python: `/root/autodl-tmp/conda/envs/regionclip/bin/python`.

```bash
cd /root/autodl-tmp/dest_mrsa_20261027/repo
bash experiments/dest/launch_after_gpu.sh
```

The launcher exits immediately if CUDA is unavailable. After asset/data checks it runs four GPU smoke updates, restores that checkpoint for two more updates, and then starts a fresh formal run from SourceB in `../run`. Smoke metrics and weights remain in `../smoke_gpu`; they are never used to initialize the formal run. No waiting daemon or scheduled training job is installed.

Full checkpoints include both optimizers, AMP scaler, student, EMA teacher, RNG and source/data identities. Both optimizers skip together on nonfinite gradients; EMA advances exactly once per successful update. Resume refuses changes to the objective or training budget. Existing run directories are never silently restarted.

To resume an interrupted formal run explicitly:

```bash
/root/autodl-tmp/conda/envs/regionclip/bin/python -u experiments/dest/train.py \
  --config experiments/dest/configs/c2f_mrsa_25k.json --mode train \
  --resume /root/autodl-tmp/dest_mrsa_20261027/run/checkpoint_last.pth
```

Read-only CPU validation is available via `experiments/dest/preflight.py --config ... --build-cpu-model`. Mathematical and trainer tests are in `experiments/dest/tests`; native RPN parity and optional real-SourceB forward/backward checks are in `experiments/dest/test_model.py`. The latter performs zero optimizer updates. CPU checks cannot validate full-resolution CUDA memory or AMP behavior; the GPU smoke is required before the formal run.
