#!/usr/bin/env bash
set -euo pipefail

export DETECTRON2_DATASETS=/home/gwb/labProject/lwm/DA-Pro/datasets
export CUDA_HOME=/home/gwb/miniconda3/envs/da-pro
export PATH="${CUDA_HOME}/bin:${PATH}"

python3 ./tools/train_net.py \
  --num-gpus 1 \
  --config-file ./configs/PascalVOC-Detection/regionclip_faster_rcnn_R_50_C4_cityscapes_train_only.yaml \
  "$@"
