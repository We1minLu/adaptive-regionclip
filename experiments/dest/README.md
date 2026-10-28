# DeST: MRSA and relaxed spatial supervision

This C2F experiment implements the paper's decoupled semantic and spatial supervision. The 25K run completed on host **011** on **2026-10-28**, using training commit `e089bc781e255aa47ef2b44db412de18e8a7815e`. It starts afresh from SourceB, not from the historical 8K adapted checkpoint. Historical `experiments/vgs_da` code is retained, including the data, augmentation, H2FA, EMA and evaluation utilities that DeST still uses.

## Completed C2F result

| Checkpoint | Mixed mAP@0.5 | Mixed mAP@0.75 | Light / medium / dense mAP@0.5 |
|---|---:|---:|---:|
| SourceB initialization, 0K | 51.20 | 30.26 | 56.63 / 52.69 / 44.59 |
| **Best observed, 22K** | **62.01** | **38.42** | **64.37 / 63.13 / 58.91** |
| Fixed final, 25K | 61.69 | 37.68 | 65.01 / 63.31 / 57.72 |

These are eight-class Foggy Cityscapes **validation** results under the repository's native VOC2007 eleven-point evaluator. The mixed set has 1,500 images: 500 scenes at each of beta 0.005, 0.01 and 0.02. Pooled AP ranks all density-specific predictions together; it is not the mean of the three per-density APs. The 62.01 result is **mixed mAP@0.5**, not dense-fog-only AP or COCO AP. The evaluated network is the full EMA teacher detector.

The initial checkpoint and 25 adapted checkpoints were evaluated; 22K has the highest observed mixed AP50. Keep this validation selection separate from the prespecified fixed-25K endpoint and independent test-set performance. The historical 58.36 run used student predictions with EMA semantic maps, so the difference is not a controlled ablation of MRSA or spatial guidance.

All 25,000 updates and EMA updates completed on a Tesla V100 32GB. Eight AMP overflow retries were recovered on the same batch; no successful update was skipped. Total runtime, including evaluation, was 80,816.69 seconds (22.45 hours). Peak allocated GPU memory was 9.27 GiB. The earlier `cpu_validation.json` remains the dated pre-launch audit; it is superseded for run status by the completed records below.

- [Run summary](results/c2f_25k_summary.json), [all 26 evaluation points](results/c2f_evaluation_history.json), and [resolved configuration](results/c2f_resolved_config.json).
- Original metrics: [initial](results/c2f_initial_000000_metrics.json), [22K best](results/c2f_best_022000_metrics.json), [25K final](results/c2f_final_025000_metrics.json).
- [Weight provenance and SHA256](results/c2f_weight_manifest.json) and [post-training verification](results/c2f_release_verification.json).

## Archived weights

Best model on **011**:

```text
/root/autodl-tmp/checkpoints/dest_mrsa_c2f_ap6201_22k_20261028/model_best_ap6201_step022000.pth
SHA256: edbba29d310ae17f87d43528f95a1e4dd5459ee3d8f6da72db1090352afcee58
```

The archive includes `WEIGHTS.md`, exact evaluation records, source/data fingerprints and a Git bundle of the code used for training. Its best-model path and `run/model_best_observed.pth` are hard links to the same verified bytes. Model files stay outside Git; the repository stores their locations and hashes.

The best and final model files contain **student, teacher and domain-discriminator states**, plus configuration and source provenance. They omit optimizer/RNG state; inference reads the teacher. The final model remains at `run/model_final.pth`; the complete 25K training state remains at `run/checkpoint_last.pth`. The constructor still requires the recorded SourceB configuration, source checkpoint, RPN checkpoint and text embeddings, so these assets must be retained. Dataset manifests and actual images also remain in place.

To evaluate the archived best model without overwriting the completed run:

```bash
cd /root/autodl-tmp/dest_mrsa_20261027/repo
/root/autodl-tmp/conda/envs/regionclip/bin/python -u experiments/dest/train.py \
  --config experiments/dest/configs/c2f_mrsa_25k.json --mode evaluate \
  --resume /root/autodl-tmp/checkpoints/dest_mrsa_c2f_ap6201_22k_20261028/model_best_ap6201_step022000.pth \
  --output /root/autodl-tmp/dest_mrsa_20261027/best_recheck
```

## Code layout

| File | Responsibility |
|---|---|
| `core.py` | Multi-scale windows, class-wise robust aggregation, relaxed spatial loss |
| `model.py` | Native localization/recognition streams, student and EMA teacher, semantic/spatial/domain objectives |
| `rpn_support.py` | Native RPN selection and selected-anchor objectness gradients |
| `train.py` | Training, atomic AMP updates, EMA, checkpoints and complete evaluation |
| `protocol.py` | Configuration identity and strict resume/evaluation checks |
| `preflight.py`, `tests/`, `test_model.py` | Mathematical, training-state and native-model verification |
| `configs/c2f_mrsa_25k.json` | Exact first-run hyperparameters and asset locations |

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

## Training entry point

The recorded run is complete. Do not relaunch into its existing output directory. The commands below document the original launch; a new experiment needs its own output/configuration.

Remote checkout: `/root/autodl-tmp/dest_mrsa_20261027/repo`.
Python: `/root/autodl-tmp/conda/envs/regionclip/bin/python`.

```bash
cd /root/autodl-tmp/dest_mrsa_20261027/repo
bash experiments/dest/launch_after_gpu.sh
```

The launcher exits immediately if CUDA is unavailable. After asset/data checks it runs four GPU smoke updates, restores that checkpoint for two more updates, and then starts a fresh formal run from SourceB in `../run`. Smoke states are never used to initialize the formal run. Startup checks passed on 011; their diagnostic records remain in `../smoke_gpu`, while the disposable smoke checkpoints were removed after successful startup. No waiting daemon or scheduled training job is installed.

Full checkpoints include both optimizers, AMP scaler, student, EMA teacher, RNG and source/data identities. Both optimizers skip together on nonfinite gradients; EMA advances exactly once per successful update. Resume refuses changes to the objective or training budget. Existing run directories are never silently restarted.

To resume an interrupted formal run explicitly:

```bash
/root/autodl-tmp/conda/envs/regionclip/bin/python -u experiments/dest/train.py \
  --config experiments/dest/configs/c2f_mrsa_25k.json --mode train \
  --resume /root/autodl-tmp/dest_mrsa_20261027/run/checkpoint_last.pth
```

Read-only CPU validation is available via `experiments/dest/preflight.py --config ... --build-cpu-model`. Mathematical and trainer tests are in `experiments/dest/tests`; native RPN parity and optional real-SourceB forward/backward checks are in `experiments/dest/test_model.py`. The latter performs zero optimizer updates. CPU checks cannot validate full-resolution CUDA memory or AMP behavior; the GPU smoke is required before the formal run.
