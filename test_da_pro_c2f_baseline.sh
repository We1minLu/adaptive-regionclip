#!/usr/bin/env bash
set -euo pipefail

export DETECTRON2_DATASETS=/home/gwb/labProject/lwm/DA-Pro/datasets
export CUDA_HOME=/home/gwb/miniconda3/envs/da-pro
export PATH="${CUDA_HOME}/bin:${PATH}"

WEIGHTS=${1:-./output/da_pro_c2f_baseline/model_final.pth}

python3 ./tools/train_net.py \
  --eval-only \
  --num-gpus 1 \
  --config-file ./configs/PascalVOC-Detection/da_pro_faster_rcnn_R_50_C4_c2f_baseline.yaml \
  MODEL.WEIGHTS "${WEIGHTS}" \
  MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml \
  MODEL.CLIP.BB_RPN_WEIGHTS /home/gwb/labProject/lwm/DA-Pro-2026.4.28/pretrained_ckpt/rpn/rpn_coco_48.pth \
  MODEL.CLIP.TEXT_EMB_PATH /home/gwb/labProject/lwm/DA-Pro-2026.4.28/pretrained_ckpt/concept_emb/cityscapes_8_cls_emb.pth \
  OUTPUT_DIR ./output/da_pro_c2f_baseline_eval \
  LEARNABLE_PROMPT.TUNING False
