"""Full-image training entry point. Smoke and formal runs always use separate dirs."""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from amp_step import atomic_amp_step
from continuation import (validate_resume, grl_progress, format_evaluation,
                          validate_evaluation_supervision, consistency_settings)
from checkpoint_identity import same_sources


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def configure_paths(cfg):
    root = cfg["repo_root"]
    sys.path.insert(0, root)
    os.environ["DETECTRON2_DATASETS"] = str(Path(root) / "datasets")
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    torch.backends.cudnn.benchmark = False


def gradient_audit(model):
    groups = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if name.startswith(("teacher_", "student.offline_")) or not param.requires_grad:
            raise AssertionError("A fixed teacher/RPN/text parameter received a gradient: " + name)
        if not torch.isfinite(param.grad).all():
            raise FloatingPointError("Nonfinite unscaled gradient: " + name)
        depth = 3 if name.startswith(("student.backbone.", "student.roi_heads.", "global_D.heads.")) else 2
        prefix = ".".join(name.split(".")[:depth])
        group = groups.setdefault(prefix, {"parameters_with_grad": 0, "squared_norm": 0.0})
        group["parameters_with_grad"] += 1
        group["squared_norm"] += float(torch.norm(param.grad, p=2, dtype=torch.float64)) ** 2
    for group in groups.values():
        group["norm"] = math.sqrt(group.pop("squared_norm"))
    return groups


def optimizer_for(model, cfg):
    detector_params = list(model.detector_parameters())
    auxiliary_params = list(model.aux_parameters())
    ids_a, ids_b = {id(p) for p in detector_params}, {id(p) for p in auxiliary_params}
    assert not ids_a & ids_b
    assert ids_a | ids_b == {id(p) for p in model.parameters() if p.requires_grad}, \
        "Every trainable parameter must be optimized exactly once"
    return [torch.optim.SGD(detector_params, lr=cfg["detector_lr"],
                            momentum=cfg["momentum"], weight_decay=cfg["weight_decay"]),
            torch.optim.AdamW(auxiliary_params, lr=cfg["aux_lr"],
                              weight_decay=cfg["aux_weight_decay"])], detector_params + auxiliary_params


def validate_ema_step(model, step):
    """A fresh EMA run advances its teacher once per successful global step."""
    if not getattr(model, "image_consistency_enabled", False):
        return
    count = getattr(model, "ema_updates", None)
    if type(count) is not int or type(step) is not int or count < 0 or count != step:
        raise ValueError("EMA update count must equal checkpoint step: %r != %r" % (count, step))


def consistency_metadata(model, cfg):
    settings = consistency_settings(cfg)
    enabled = bool(getattr(model, "image_consistency_enabled", False))
    if enabled != settings["image_consistency_enabled"]:
        raise ValueError("Model/config image_consistency_enabled differs")
    if settings["evaluation_model"] != "student":
        raise ValueError("This evaluator only reports evaluation_model=student")
    result = {key: settings[key] for key in ("image_consistency_enabled", "strong_weak_enabled",
              "semantic_teacher_mode", "evaluation_model")}
    if enabled:
        count = getattr(model, "ema_updates", None)
        if type(count) is not int or count < 0:
            raise ValueError("EMA update count must be a nonnegative integer")
        result.update(ema_decay=settings["ema_decay"], ema_updates=count,
                      image_aggregation=settings["image_aggregation"],
                      image_consistency_weight=settings["image_consistency_weight"])
    return result


def save_checkpoint(model, optimizers, scaler, cfg, output, step, runtime, final=False):
    validate_ema_step(model, step)
    state = {"version": 1, "step": step, "model": model.checkpoint_state(),
             "optimizers": [opt.state_dict() for opt in optimizers], "scaler": scaler.state_dict(),
             "rng": rng_state(), "config": cfg, "runtime": runtime}
    checkpoint = output / "checkpoint_last.pth"
    temp = output / "checkpoint_writing.pth"
    torch.save(state, str(temp))
    if checkpoint.exists():
        checkpoint.replace(output / "checkpoint_previous.pth")
    temp.replace(checkpoint)
    if final:
        # A compact inference checkpoint; resume remains available in last.pth.
        torch.save({"step": step, "model": state["model"], "config": cfg},
                   str(output / "model_final.pth"))
    atomic_json(output / "checkpoint.json", {"step": step, "path": str(checkpoint),
                                            "final": final, "time": time.time()})


def make_loaders(cfg, start_step):
    from data import build_loaders
    batch = cfg["microbatch_per_domain"]
    return build_loaders(cfg["manifest_dir"], batch_size_source=batch,
                         batch_size_target=batch, num_workers=cfg["num_workers"],
                         seed=cfg["seed"],
                         source_samples_consumed=start_step * cfg["accumulation_steps"] * batch,
                         target_samples_consumed=start_step * cfg["accumulation_steps"] * batch,
                         target_image_labels_allowed=cfg.get("target_image_labels_allowed", True),
                         strong_weak_enabled=cfg.get("strong_weak_enabled", False))


def train_update(model, batches, optimizers, scaler, cfg, progress, step, output):
    """Retry overflow on identical data/RNG; count only a successful atomic update."""
    before_rng = rng_state()
    before_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    named_parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    retry_limit = int(cfg.get("amp_max_retries", 4))
    for attempt in range(retry_limit + 1):
        if attempt:
            restore_rng(before_rng)
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(before_buffers[name])
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        sums, density_counts, support_per_micro = {}, {}, []
        for source, target in batches:
            assert all("instances" in record for record in source)
            assert all("instances" not in record and "annotations" not in record for record in target)
            if cfg.get("target_image_labels_allowed", True) is False:
                assert all("image_labels" not in record for record in target), "UDA target input contains image labels"
            for record in target:
                beta = str(record["beta"])
                density_counts[beta] = density_counts.get(beta, 0) + 1
            with torch.cuda.amp.autocast(enabled=cfg["amp"]):
                losses, stats = model.training_losses(source, target, progress=progress)
                total = sum(losses.values()) / cfg["accumulation_steps"]
            support_per_micro.append({"active_groups": stats["conditional_domain"]["active_groups"],
                                      "roi": stats["roi"]})
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("Nonfinite forward loss at step %d: %r" % (step, losses))
            scaler.scale(total).backward()
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.) + float(value.detach()) / cfg["accumulation_steps"]
            del total, losses
        result = atomic_amp_step(optimizers, scaler, named_parameters,
                                 max_norm=cfg["gradient_clip_norm"])
        if result["did_step"]:
            # Use the just-updated student once per successful optimizer step.
            # Updating in forward, per microbatch, or on overflow would break
            # the existing same-batch/RNG retry and teacher time scale.
            teacher_updated = bool(getattr(model, "image_consistency_enabled", False))
            if teacher_updated:
                model.update_teacher()
                stats["ema_teacher_update_applied"] = True
                if hasattr(model, "ema_updates"):
                    stats["ema_updates"] = int(model.ema_updates)
            return dict(losses=sums, stats=stats, density_counts=density_counts,
                        support_per_micro=support_per_micro, retries=attempt,
                        teacher_updated=teacher_updated, **result)
        event = dict(step=step, attempt=attempt + 1, retry_limit=retry_limit,
                     time=time.time(), image_ids=[{
                         "source": [r["image_id"] for r in source],
                         "target": [r["image_id"] for r in target]} for source, target in batches], **result)
        with open(Path(output) / "amp_overflow_events.jsonl", "a", buffering=1) as stream:
            stream.write(json.dumps(event) + "\n")
        print("AMP overflow at step=%d; both optimizers unchanged; scale %.1f -> %.1f; attempt %d/%d; %s" %
              (step, result["old_scale"], result["new_scale"], attempt + 1, retry_limit + 1,
               "retry same batch" if attempt < retry_limit else "retry limit reached"), flush=True)
        if attempt == retry_limit:
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            restore_rng(before_rng)
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(before_buffers[name])
            raise FloatingPointError("AMP overflow persisted at step %d after %d retries; no optimizer update" %
                                     (step, retry_limit))


def publish_evaluation(model, cfg, output, step, result, scope="current"):
    """Publish only complete evaluation; preserve one explicitly selected best model."""
    validate_ema_step(model, step)
    print(format_evaluation(step, result, scope=scope), flush=True)
    if not (result.get("complete") is True and result.get("images") == 1500
            and result.get("expected_images") == 1500):
        raise ValueError("Partial evaluation cannot update formal latest/best metrics")
    if result.get("manifest_sha256") != cfg["manifest_sha256"]["eval_mixed.json"]:
        raise ValueError("Evaluation manifest differs from the training protocol")
    scores = [result["pooled"]["AP50"], result["pooled"]["AP75"]]
    scores += [result["per_density"][beta]["AP50"] for beta in ("0.005", "0.01", "0.02")]
    if any(isinstance(v, bool) or not math.isfinite(float(v)) or not 0 <= float(v) <= 100
           for v in scores):
        raise ValueError("Formal evaluation scores must be finite percentages in [0,100]")
    output = Path(output)
    record = {"step": int(step), "scope": scope, "images": result["images"],
              "mixed_AP50": result["pooled"]["AP50"], "mixed_AP75": result["pooled"]["AP75"],
              "density_AP50": {b: v["AP50"] for b,v in result["per_density"].items()},
              "recorded_at": time.time()}
    history_path = output / "evaluation_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history = [r for r in history if r["step"] != step] + [record]
    atomic_json(history_path, sorted(history, key=lambda r:r["step"]))
    atomic_json(output / "latest_evaluation.json", record)
    best_path = output / "best_observed.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    if best is None or record["mixed_AP50"] > best["mixed_AP50"]:
        if scope == "parent_reference":
            checkpoint_path = cfg["parent_inference_checkpoint"]
        else:
            checkpoint_path = str(output / "model_best_observed.pth")
            temporary = output / "model_best_observed_writing.pth"
            torch.save({"step":step,"model":model.checkpoint_state(),"config":cfg,
                        "selection":"exploratory_validation_mixed_AP50","evaluation":record}, str(temporary))
            temporary.replace(checkpoint_path)
        atomic_json(best_path, dict(record, checkpoint=checkpoint_path,
                    selection="highest_observed_validation_mixed_AP50; separate from fixed final"))
        print("BEST_OBSERVED step=%d mixed_AP50=%.4f scope=%s checkpoint=%s" %
              (step,record["mixed_AP50"],scope,checkpoint_path),flush=True)
    return record


def run_training(cfg, smoke=False, smoke_steps=6, resume=None, eval_smoke=True):
    from detectron2.utils.events import EventStorage
    from model import FormalVGSDA
    configure_paths(cfg)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any((output / name).exists() for name in ("metrics.jsonl", "checkpoint_last.pth")):
        raise RuntimeError("Output already contains a run; use --resume or a new --output")
    set_seed(cfg["seed"])
    begin = time.time()
    assert digest(cfg["source_checkpoint"]) == cfg["source_checkpoint_sha256"]
    if not isinstance(cfg.get("target_image_labels_allowed"), bool):
        raise ValueError("target_image_labels_allowed must be an explicit boolean")
    data_audit = json.loads((Path(cfg["manifest_dir"]) / "data_audit.json").read_text())
    if data_audit.get("target_image_labels_allowed", True) != cfg["target_image_labels_allowed"]:
        raise ValueError("Manifest and training target supervision policies differ")
    cfg["manifest_sha256"] = data_audit["manifest_sha256"]
    for name, expected in cfg["manifest_sha256"].items():
        assert digest(Path(cfg["manifest_dir"]) / name) == expected, "Manifest changed: " + name
    atomic_json(output / "config.json", cfg)
    atomic_json(output / "status.json", {"state": "initializing", "pid": os.getpid(), "smoke": smoke})
    model = FormalVGSDA(cfg).to("cuda")
    model.train()
    atomic_json(output / "initialization_audit.json", model.initialization_audit)
    atomic_json(output / "native_path_verification.json", model.verify_native_path())
    optimizers, all_params = optimizer_for(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"], init_scale=1024.0)
    start_step = 0
    previous_runtime = {}
    if resume:
        checkpoint = torch.load(str(resume), map_location="cpu")
        old = checkpoint["config"]
        if not same_sources(checkpoint["model"]["sources"], model.source_identity):
            raise ValueError("Resume fixed-source fingerprints differ")
        resume_audit = validate_resume(old, cfg, checkpoint["step"], verified_sources=True)
        model.load_checkpoint_state(checkpoint["model"])
        validate_ema_step(model, checkpoint["step"])
        for opt, saved in zip(optimizers, checkpoint["optimizers"]):
            opt.load_state_dict(saved)
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"])
        start_step = checkpoint["step"]
        previous_runtime = checkpoint.get("runtime", {})
        atomic_json(output / "resume_audit.json", dict(resume_audit, checkpoint=str(resume),
                    restored_step=start_step, scaler=scaler.state_dict(), runtime=previous_runtime,
                    **consistency_metadata(model, cfg),
                    optimizer_state_counts=[len(o.state) for o in optimizers],
                    restored_optimizer_lrs=[o.param_groups[0]["lr"] for o in optimizers],
                    source_samples_consumed=start_step*cfg["accumulation_steps"]*cfg["microbatch_per_domain"],
                    target_samples_consumed=start_step*cfg["accumulation_steps"]*cfg["microbatch_per_domain"]))
        del checkpoint
    latest_evaluation = None
    if not smoke:
        latest_path = output / "latest_evaluation.json"
        if latest_path.exists():
            latest_evaluation = json.loads(latest_path.read_text())
            if latest_evaluation["step"] > start_step:
                raise ValueError("Latest evaluation is ahead of the resumed checkpoint")
        if start_step == 0 and not resume:
            # A recoverable exact initial state, before any evaluation or training.
            save_checkpoint(model, optimizers, scaler, cfg, output, 0,
                            dict(successful_updates=0, skipped_updates=0,
                                 amp_retry_attempts=0, elapsed_seconds=time.time()-begin))
        latest_evaluation = prepare_initial_evaluation(model, cfg, output, start_step, latest_evaluation)
    elif eval_smoke and cfg.get("evaluate_initial", False):
        # Exercise evaluation before the first fresh update without publishing a full metric.
        run_evaluation(model, cfg, output / "eval_initial_smoke", max_images=6)
    atomic_json(output / "status.json", {"state": "initializing", "pid": os.getpid(),
                "smoke": smoke, "step": start_step, "max_steps": cfg["max_steps"], "resume_from_step": start_step})
    source_loader, target_loader = make_loaders(cfg, start_step)
    source_iter, target_iter = iter(source_loader), iter(target_loader)
    stop_request = {"requested": False}
    def graceful_stop(sig, frame):
        stop_request["requested"] = True
        print("Stop requested: finishing current update and saving checkpoint.", flush=True)
    signal.signal(signal.SIGTERM, graceful_stop)
    signal.signal(signal.SIGINT, graceful_stop)
    last_step = start_step
    limit = start_step + smoke_steps if smoke else cfg["max_steps"]
    durations = []
    stats_latest = {}
    successful_updates = previous_runtime.get("successful_updates", 0)
    skipped_updates = previous_runtime.get("skipped_updates", 0)
    amp_retry_attempts = previous_runtime.get("amp_retry_attempts", 0)
    torch.cuda.reset_peak_memory_stats()
    with EventStorage(start_step) as storage, open(output / "metrics.jsonl", "a", buffering=1) as metrics:
        for iteration in range(start_step, limit):
            now = time.time()
            step = iteration + 1
            storage.iter = iteration
            warm = min(1.0, step / max(cfg["warmup_steps"], 1))
            lr_factor = cfg["warmup_factor"] + (1.0 - cfg["warmup_factor"]) * warm
            for opt, base_lr in zip(optimizers, (cfg["detector_lr"], cfg["aux_lr"])):
                for group in opt.param_groups:
                    group["lr"] = base_lr * lr_factor
            # All presets retain the original 25K GRL horizon.
            progress = grl_progress(step, cfg)
            batches = [(next(source_iter), next(target_iter)) for _ in range(cfg["accumulation_steps"])]
            result = train_update(model, batches, optimizers, scaler, cfg, progress, step, output)
            del batches
            sums, stats_latest = result["losses"], result["stats"]
            density_counts, support_per_micro = result["density_counts"], result["support_per_micro"]
            grad_norm = result["global_grad_norm"]
            amp_retry_attempts += result["retries"]
            if step == start_step + 1:
                audit = gradient_audit(model)
                for required in ("student.backbone.layer4", "student.backbone.attnpool",
                                 "student.roi_heads.box_predictor", "search.lateral3", "search.lateral4"):
                    assert audit.get(required, {}).get("norm", 0.0) > 0, "Missing online gradient: " + required
                atomic_json(output / "gradient_audit.json", audit)
            skipped = False
            successful_updates += 1
            torch.cuda.synchronize()
            elapsed = time.time() - now
            durations.append(elapsed)
            last_step = step
            row = dict(step=step, max_steps=cfg["max_steps"], smoke=smoke,
                       elapsed_seconds=elapsed, total_loss=sum(sums.values()), losses=sums,
                       lr_detector=optimizers[0].param_groups[0]["lr"], lr_aux=optimizers[1].param_groups[0]["lr"],
                       gradient_norm=float(grad_norm), amp_scale=scaler.get_scale(), skipped=bool(skipped),
                       amp_retries=result["retries"], successful_updates=successful_updates,
                       ema_teacher_updated=result["teacher_updated"],
                       peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                       peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
                       grl_progress=progress, latest_evaluation=latest_evaluation,
                       target_density_counts=density_counts, stats=stats_latest,
                       stats_scope="last_microbatch", support_per_microbatch=support_per_micro)
            metrics.write(json.dumps(row, ensure_ascii=False) + "\n")
            atomic_json(output / "status.json", dict(state="training", pid=os.getpid(), **row))
            if step % cfg["log_period"] == 0 or smoke or step <= 3:
                eval_text = (" latest_eval_step=%d latest_mixed_AP50=%.4f" %
                             (latest_evaluation["step"],latest_evaluation["mixed_AP50"])) if latest_evaluation else ""
                print(("step=%d/%d loss=%.5f time=%.2fs GPU_alloc=%.2fGiB GPU_reserved=%.2fGiB lr=%.7f" %
                      (step, cfg["max_steps"], row["total_loss"], elapsed, row["peak_allocated_GiB"],
                       row["peak_reserved_GiB"], row["lr_detector"])) + eval_text, flush=True)
            runtime = dict(successful_updates=successful_updates, skipped_updates=skipped_updates,
                           amp_retry_attempts=amp_retry_attempts,
                           elapsed_seconds=time.time() - begin)
            checkpoint_due = step % cfg["checkpoint_period"] == 0
            if checkpoint_due or step == limit or stop_request["requested"]:
                save_checkpoint(model, optimizers, scaler, cfg, output, step, runtime,
                                final=not smoke and step == cfg["max_steps"])
            eval_due = not smoke and (step % cfg["eval_period"] == 0 or step == cfg["max_steps"])
            if eval_due and not stop_request["requested"]:
                atomic_json(output / "status.json", dict(state="evaluating",pid=os.getpid(), **row))
                print("EVAL_START step=%d images=1500 last_mixed_AP50=%s" %
                      (step,latest_evaluation["mixed_AP50"] if latest_evaluation else "none"),flush=True)
                evaluation = run_evaluation(model, cfg, output / ("eval_step_%06d" % step))
                latest_evaluation = publish_evaluation(model,cfg,output,step,evaluation)
                model.train()
                row["latest_evaluation"] = latest_evaluation
                atomic_json(output / "status.json", dict(state="training",pid=os.getpid(), **row))
            if stop_request["requested"]:
                break
    summary = dict(state="stopped" if stop_request["requested"] else "complete", pid=os.getpid(),
                   smoke=smoke, step=last_step, max_steps=cfg["max_steps"], successful_updates=successful_updates,
                   latest_evaluation=latest_evaluation,
                   skipped_updates=skipped_updates, duration_seconds=time.time() - begin,
                   amp_retry_attempts=amp_retry_attempts,
                   mean_update_seconds=float(np.mean(durations[1:] or durations)),
                   peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
                   effective_images_per_domain=cfg["microbatch_per_domain"] * cfg["accumulation_steps"])
    summary.update(consistency_metadata(model, cfg))
    if smoke and eval_smoke and not stop_request["requested"]:
        # Three fog densities, two scenes. Metric here only validates evaluator I/O.
        summary["evaluation_smoke"] = run_evaluation(model, cfg, output / "eval_smoke", max_images=6)
        reloaded = torch.load(str(output / "checkpoint_last.pth"), map_location="cpu")
        model.load_checkpoint_state(reloaded["model"])
        assert reloaded["step"] == last_step
        validate_ema_step(model, last_step)
        summary["checkpoint_reload_passed"] = True
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "status.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "evaluation_smoke"}, ensure_ascii=False), flush=True)


def prepare_initial_evaluation(model, cfg, output, step, latest):
    """Evaluate the fresh0K state, or finish a checkpoint's interrupted evaluation."""
    if latest is not None and latest["step"] == step:
        return latest
    initial_due = step == 0 and cfg.get("evaluate_initial", False)
    pending_due = step > 0 and step % cfg["eval_period"] == 0
    if not (initial_due or pending_due):
        return latest
    atomic_json(Path(output) / "status.json", dict(state="evaluating", pid=os.getpid(),
                step=step, max_steps=cfg["max_steps"], smoke=False, latest_evaluation=latest))
    print("EVAL_START step=%d images=1500 scope=%s" %
          (step, "initial" if step == 0 else "resumed_pending"), flush=True)
    result = run_evaluation(model, cfg, Path(output) / ("eval_step_%06d" % step))
    return publish_evaluation(model, cfg, output, step, result,
                              scope="initial" if step == 0 else "resumed_pending")


def run_evaluation(model, cfg, output, max_images=None):
    from evaluation import evaluate_model
    state, was_training = rng_state(), model.training
    try:
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg["amp"]):
            result = evaluate_model(model, Path(cfg["manifest_dir"]) / "eval_mixed.json",
                                    output, max_images=max_images)
        result.update(training_budget_extension_informed_by_validation="extension_from_step" in cfg,
                      experiment_design_informed_by_prior_validation=cfg.get("experiment_scope") != "formal_fixed_25k",
                      experiment_scope=cfg.get("experiment_scope", "formal_fixed_25k"),
                      primary_checkpoint_policy="fixed_%d" % cfg["max_steps"],
                      auxiliary_best_checkpoint_policy="highest_observed_validation_mixed_AP50",
                      target_image_labels_used_in_training=cfg["target_image_labels_allowed"],
                      evaluation_GT_used_for_training=False,
                      validation_metrics_used_for_exploration_and_auxiliary_best=True)
        result.update(consistency_metadata(model, cfg))
        result.pop("target_used_for_selection",None)
        result.pop("target_used_for_training",None)
        atomic_json(Path(output)/"metrics.json",result)
        # Keep the evaluator's hash reference synchronized after adding scope metadata.
        eval_status_path = Path(output)/"status.json"
        if eval_status_path.exists():
            eval_status=json.loads(eval_status_path.read_text())
            eval_status["metrics_sha256"]=digest(Path(output)/"metrics.json")
            atomic_json(eval_status_path,eval_status)
        return result
    finally:
        restore_rng(state)
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "train", "evaluate"), required=True)
    parser.add_argument("--output")
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--resume")
    parser.add_argument("--skip-eval-smoke", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.output:
        cfg["output_dir"] = args.output
    if args.microbatch:
        cfg["microbatch_per_domain"] = args.microbatch
    if args.accumulation:
        cfg["accumulation_steps"] = args.accumulation
    configure_paths(cfg)
    if args.mode == "smoke" and cfg["output_dir"].endswith("/run"):
        raise ValueError("Smoke requires its own --output directory")
    try:
        if args.mode == "evaluate":
            from model import FormalVGSDA
            state = torch.load(args.resume, map_location="cpu")
            validate_evaluation_supervision(state.get("config", {}), cfg)
            validate_evaluation_supervision(state["model"].get("config", {}), cfg)
            model = FormalVGSDA(cfg).to("cuda")
            model.load_checkpoint_state(state["model"])
            print(json.dumps(run_evaluation(model, cfg, Path(cfg["output_dir"]) / "evaluation")), flush=True)
        else:
            run_training(cfg, smoke=args.mode == "smoke", smoke_steps=args.steps,
                         resume=args.resume, eval_smoke=not args.skip_eval_smoke)
    except BaseException as error:
        output = Path(cfg["output_dir"])
        failure = {"error": repr(error), "traceback": traceback.format_exc(), "time": time.time()}
        atomic_json(output / "failure.json", failure)
        status_path = output / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        atomic_json(status_path, dict(status, state="failed", pid=os.getpid(),
                                     error=repr(error), failure_time=failure["time"],
                                     last_successful_step=status.get("step")))
        raise


if __name__ == "__main__":
    main()
