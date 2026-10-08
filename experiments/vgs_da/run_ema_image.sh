#!/usr/bin/env bash
# Run only on the GPU host, after the independent experiment bundle is deployed.
# Any GPU test/smoke/resume failure prevents the formal run from starting.
set -euo pipefail
ROOT="${1:-/root/autodl-tmp/formal_vgs_ema_image_20261008}"
PY="${REGIONCLIP_PYTHON:-/root/autodl-tmp/conda/envs/regionclip/bin/python}"
export PYTHONUNBUFFERED=1
exec 9>"$ROOT/launch.lock"
flock -n 9 || { echo 'Another experiment launcher is already running.'; exit 1; }
cd "$ROOT/code"
"$PY" - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
import torch
root = Path(sys.argv[1]).resolve()
cfg = json.loads((root / 'config.json').read_text())
assert torch.cuda.is_available(), 'GPU unavailable; no smoke or training started'
assert Path(cfg['output_dir']).resolve() == root / 'run'
assert cfg['image_consistency_enabled'] is True and cfg['strong_weak_enabled'] is True
assert cfg['target_image_labels_allowed'] is False
assert cfg['ema_decay'] == .9996 and cfg['image_consistency_weight'] == 1.
for name in ('run', 'smoke'):
    assert not (root / name / 'checkpoint_last.pth').exists(), 'Existing run: explicitly resume instead'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY
"$PY" -m unittest test_amp_step test_train_update -q
"$PY" train.py --config "$ROOT/config.json" --mode smoke --output "$ROOT/smoke" --steps 6
"$PY" train.py --config "$ROOT/config.json" --mode smoke --output "$ROOT/smoke" \
    --resume "$ROOT/smoke/checkpoint_last.pth" --steps 2
"$PY" - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = json.loads((root / 'smoke' / 'summary.json').read_text())
assert summary['ema_updates'] == 8, summary
assert summary['checkpoint_reload_passed'] is True
print('SMOKE_AND_RESUME_PASSED; starting a fresh source-initialized 25K run', flush=True)
PY
exec "$PY" train.py --config "$ROOT/config.json" --mode train
