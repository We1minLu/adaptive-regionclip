#!/usr/bin/env bash
# Run only after the user has enabled the GPU. No auto-start/polling daemon.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${DEST_PYTHON:-/root/autodl-tmp/conda/envs/regionclip/bin/python}"
CONFIG="${1:-$ROOT/experiments/dest/configs/c2f_mrsa_25k.json}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "GPU not enabled; nothing started"'
"$PYTHON" experiments/dest/preflight.py --config "$CONFIG"
OUTPUT="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_dir"])' "$CONFIG")"
SMOKE="$(dirname "$OUTPUT")/smoke_gpu"
if [[ -e "$SMOKE/metrics.jsonl" || -e "$OUTPUT/metrics.jsonl" ]]; then
  printf '%s\n' 'An existing smoke/formal run exists. Inspect and explicitly resume; refusing a new launch.' >&2
  exit 1
fi
"$PYTHON" -u experiments/dest/train.py --config "$CONFIG" --mode smoke --steps 4 --output "$SMOKE"
"$PYTHON" -u experiments/dest/train.py --config "$CONFIG" --mode smoke --steps 2 --output "$SMOKE" --resume "$SMOKE/checkpoint_last.pth"
exec "$PYTHON" -u experiments/dest/train.py --config "$CONFIG" --mode train
