"""Read-only asset/data checks. Optional CPU model check never updates weights."""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "experiments/vgs_da"))

from experiments.dest.protocol import validate_config
from experiments.vgs_da.data import read_manifest, validate_target_record

EXPECTED_ASSETS = {
    "source_cfg": "a4b7580f000e3040749648da1acd339bd5d6a74e6dbdd456fe1e333ec207d979",
    "source_checkpoint": "f199ea87709207a0b2ee23dd63dd49d4d3b540c8e9eb9c213d00f3e449a5e861",
    "rpn_checkpoint": "8b934109caae409fbf62753d1f167ea1411b75aa0b30c60f5c289aa90ada42ca",
    "text_embeddings": "c25e3d4b8c8271213929d0fe88cfe7f3e934b5ba81e9a34af3521756e74827ae",
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(cfg, build_cpu_model=False):
    validate_config(cfg)
    if Path(cfg["repo_root"]).resolve() != ROOT:
        raise ValueError("Configured repo_root is not this checkout")
    assets = {}
    for key, expected in EXPECTED_ASSETS.items():
        actual = sha(cfg[key])
        if actual != expected:
            raise ValueError("Historical initialization asset differs: " + key)
        assets[key] = {"sha256": actual, "bytes": Path(cfg[key]).stat().st_size}
    manifests = Path(cfg["manifest_dir"])
    audit = json.loads((manifests / "data_audit.json").read_text())
    if audit.get("target_image_labels_allowed") is not False:
        raise ValueError("Expected unlabeled-target manifest audit")
    for name, expected in audit["manifest_sha256"].items():
        if sha(manifests / name) != expected:
            raise ValueError("Manifest changed: " + name)
    source = read_manifest(manifests / "source_train.json")
    target = read_manifest(manifests / "target_train.json")
    evaluation = read_manifest(manifests / "eval_mixed.json")
    for row in target:
        validate_target_record(row, image_labels_allowed=False)
    if len(source) not in (2965, 2975) or len(target) != 8925 or len(evaluation) != 1500:
        raise ValueError("Unexpected C2F manifest sizes")
    train_scenes = {r["scene_id"] for r in source} | {r["scene_id"] for r in target}
    eval_scenes = {r["scene_id"] for r in evaluation}
    if train_scenes & eval_scenes:
        raise ValueError("Training and evaluation scenes overlap")
    missing = [r["file_name"] for r in source + target + evaluation if not Path(r["file_name"]).is_file()]
    if missing:
        raise ValueError("Missing dataset images: " + repr(missing[:3]))
    for row in source:
        if row.get("domain") != "source" or "annotations" not in row:
            raise ValueError("Source annotations are missing")
    # Import the exact deployed framework/compiled extension before any GPU run.
    import torch
    from detectron2 import _C
    import detectron2
    if Path(detectron2.__file__).resolve().parent != ROOT / "detectron2":
        raise ValueError("A different detectron2 checkout was imported")
    free = shutil.disk_usage(Path(cfg["output_dir"]).parent).free / 2**30
    if free < 4:
        raise RuntimeError("At least 4 GiB free space is required for recoverable checkpoints")
    result = {
        "state": "cpu_preflight_passed", "training_started": False,
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "source_initialization": "SourceB; original source-trained VGS intentionally excluded",
        "assets": assets, "manifest_sha256": audit["manifest_sha256"],
        "source_images": len(source), "target_images": len(target), "evaluation_images": len(evaluation),
        "train_eval_scene_overlap": 0, "target_labels_used": False,
        "free_disk_GiB": free, "evaluation_model": "EMA teacher",
    }
    if build_cpu_model:
        from experiments.dest.model import DeST
        cpu_cfg = dict(cfg, device="cpu")
        torch.set_num_threads(cfg.get("cpu_threads", 4))
        model = DeST(cpu_cfg)
        result["model_initialization"] = model.initialization_audit
        result["native_path"] = model.verify_native_path()
        result["teacher_has_trainable_parameters"] = any(p.requires_grad for p in model.teacher.parameters())
        if result["teacher_has_trainable_parameters"]:
            raise AssertionError("Teacher parameters are trainable")
        result["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        result["ema_updates"] = model.ema_updates
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--build-cpu-model", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    result = inspect(json.loads(Path(args.config).read_text()), args.build_cpu_model)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
