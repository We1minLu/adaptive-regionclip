"""Explicit DeST identity and strict resume boundaries; CPU-only checks."""
import math

from experiments.vgs_da.continuation import format_evaluation

VERSION = "dest-mrsa-relaxed-spatial-v1"
SOURCE_SHA256 = "f199ea87709207a0b2ee23dd63dd49d4d3b540c8e9eb9c213d00f3e449a5e861"

# Relocation and logging changes are allowed. Objective/data/optimizer changes
# are new experiments, never silently admitted as full-state continuation.
OPERATIONAL_KEYS = {
    "repo_root", "manifest_dir", "output_dir", "source_cfg", "source_checkpoint",
    "rpn_checkpoint", "text_embeddings", "num_workers", "cpu_threads",
    "teacher_roi_chunk", "train_roi_chunk", "inference_roi_chunk", "log_period",
    "eval_period", "checkpoint_period", "evaluate_initial", "device",
}


def _number(cfg, name, lower=0., upper=None, strict=False):
    value = cfg.get(name)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("Expected numeric " + name)
    if not math.isfinite(value) or value < lower or (strict and value == lower):
        raise ValueError("Invalid " + name)
    if upper is not None and value > upper:
        raise ValueError("Invalid " + name)
    return value


def validate_config(cfg):
    if cfg.get("method_version") != VERSION:
        raise ValueError("This entry point accepts only the DeST protocol")
    if cfg.get("source_checkpoint_sha256") != SOURCE_SHA256:
        raise ValueError("DeST C2F must start from the verified source-only SourceB")
    required = {
        "target_image_labels_allowed": False,
        "strong_weak_enabled": True,
        "image_consistency_enabled": True,
        "semantic_teacher_mode": "ema",
        "image_aggregation": "mrsa_teacher_h2fa_student",
        "evaluation_model": "teacher",
        "source_view": "strong",
        "score_fusion": "roi_only",
        "rpn_mode": "native_trainable",
        "lr_schedule": "linear_warmup_then_constant",
        "gradient_step_policy": "atomic_amp_retry_v1",
    }
    for name, expected in required.items():
        value = cfg.get(name)
        if value != expected or (type(expected) is bool and type(value) is not bool):
            raise ValueError("Invalid DeST protocol field: " + name)
    for name in ("source_search_checkpoint", "supplemental_proposals",
                 "search_loss_weight", "conditional_domain_weight", "conditional_grl_max"):
        if name in cfg:
            raise ValueError("Historical VGS/conditional-adaptation field is not allowed: " + name)
    for name in ("max_steps", "microbatch_per_domain", "accumulation_steps", "initial_proposals",
                 "source_roi_batch", "mrsa_max_iter", "grl_schedule_steps",
                 "teacher_roi_chunk", "train_roi_chunk", "inference_roi_chunk",
                 "checkpoint_period", "eval_period", "log_period"):
        if type(cfg.get(name)) is not int or cfg[name] < 1:
            raise ValueError("Expected positive integer: " + name)
    if (cfg.get("fixed_height"), cfg.get("fixed_width")) != (1024, 2048):
        raise ValueError("The inherited C2F data protocol uses 1024x2048")
    scales = cfg.get("window_scales")
    if not isinstance(scales, list) or not scales:
        raise ValueError("window_scales must be a nonempty list")
    if any(isinstance(s, bool) or not isinstance(s, (int, float)) or
           not math.isfinite(s) or not 0 < s <= 1 for s in scales):
        raise ValueError("Window scales must lie in (0,1]")
    for name in ("mrsa_rho", "image_consistency_weight", "spatial_loss_weight", "traditional_domain_weight"):
        _number(cfg, name)
    _number(cfg, "spatial_gamma", lower=1.)
    _number(cfg, "spatial_tau", lower=0., upper=1., strict=True)
    if cfg["spatial_tau"] == 1:
        raise ValueError("spatial_tau must be less than one")
    _number(cfg, "spatial_score_threshold", lower=0., upper=1.)
    _number(cfg, "mrsa_tolerance", strict=True)
    _number(cfg, "ema_decay", upper=1.)
    if cfg["ema_decay"] == 1:
        raise ValueError("EMA decay must be less than one")
    for name in ("detector_lr", "aux_lr", "gradient_clip_norm"):
        _number(cfg, name, strict=True)
    return cfg


def consistency_settings(cfg):
    validate_config(cfg)
    return {k: cfg[k] for k in (
        "image_consistency_enabled", "strong_weak_enabled", "semantic_teacher_mode",
        "evaluation_model", "ema_decay", "image_aggregation", "image_consistency_weight")}


def grl_progress(step, cfg):
    return min(1., max(0., float(step) / cfg["grl_schedule_steps"]))


def validate_resume(old, cfg, saved_step, verified_sources=False):
    validate_config(old)
    validate_config(cfg)
    if type(saved_step) is not int or not 0 <= saved_step <= cfg["max_steps"]:
        raise ValueError("Checkpoint step is outside the configured schedule")
    allowed = OPERATIONAL_KEYS if verified_sources else (OPERATIONAL_KEYS - {
        "source_cfg", "source_checkpoint", "rpn_checkpoint", "text_embeddings", "manifest_dir"})
    for key in set(old) | set(cfg):
        if key not in allowed and old.get(key) != cfg.get(key):
            raise ValueError("DeST resume changes experiment identity: " + key)
    return {"saved_step": saved_step, "max_steps": cfg["max_steps"],
            "verified_sources": bool(verified_sources), "budget_extended": False}


def validate_evaluation_supervision(saved, cfg):
    validate_resume(saved, cfg, 0, verified_sources=False)
