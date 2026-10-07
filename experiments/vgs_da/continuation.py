"""CPU-only guards and reporting for a controlled training continuation."""

import math
import re


# These settings define the data, objective, optimizer, model identity and
# prediction protocol. Operational settings (paths for output, log/eval/save
# periods, workers and student ROI chunks) may change without resetting them.
UNCHANGED_KEYS = (
    "source_checkpoint_sha256", "source_search_checkpoint",
    "microbatch_per_domain", "accumulation_steps", "source_roi_batch", "seed",
    "manifest_sha256", "detector_lr", "aux_lr", "warmup_steps", "warmup_factor",
    "momentum", "weight_decay", "aux_weight_decay", "gradient_clip_norm", "amp",
    "traditional_domain_weight", "conditional_grl_max", "conditional_domain_weight",
    "source_loss_weight", "search_loss_weight", "checkpoint_res5", "score_fusion",
    "initial_proposals", "supplemental_proposals", "score_threshold", "nms_threshold",
    "detections_per_image", "lr_schedule", "fixed_height", "fixed_width",
    "teacher_roi_chunk", "source_cfg", "source_checkpoint", "rpn_checkpoint",
    "text_embeddings", "repo_root", "manifest_dir", "target_image_labels_allowed",
    "gradient_step_policy",
)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def _horizon(cfg):
    return _positive_int(cfg.get("grl_schedule_steps", cfg["max_steps"]),
                         "grl_schedule_steps")


def validate_resume(old, cfg, saved_step, verified_sources=False):
    """Validate a full-state resume; never mutate either configuration.

    A larger budget is allowed only at the completed parent budget and with an
    explicit extension_from_step marker. Resuming a checkpoint already produced
    by the extension does not need to be at that original boundary again.
    """
    old_max = _positive_int(old["max_steps"], "checkpoint max_steps")
    new_max = _positive_int(cfg["max_steps"], "max_steps")
    if isinstance(saved_step, bool) or not isinstance(saved_step, int):
        raise ValueError("saved_step must be an integer")
    if not 0 <= saved_step <= old_max:
        raise ValueError("saved_step lies outside the checkpoint training budget")
    missing = object()
    relocated_paths = {"repo_root", "source_cfg", "source_checkpoint", "rpn_checkpoint",
                       "text_embeddings", "source_search_checkpoint", "manifest_dir"}
    for key in UNCHANGED_KEYS:
        # The caller must first verify every fixed source hash against the checkpoint.
        # Manifest content hashes remain immutable even when its directory is moved.
        if verified_sources and key in relocated_paths:
            continue
        if old.get(key, missing) != cfg.get(key, missing):
            raise ValueError("Resume config changed: " + key)
    old_horizon, new_horizon = _horizon(old), _horizon(cfg)
    if old_horizon != new_horizon:
        raise ValueError("Resume config changed: grl_schedule_steps (%s -> %s)" %
                         (old_horizon, new_horizon))
    extension = new_max != old_max
    if extension:
        marker = cfg.get("extension_from_step")
        if (new_max <= old_max or saved_step != old_max
                or isinstance(marker, bool) or not isinstance(marker, int)
                or marker != saved_step):
            raise ValueError("Budget extension requires new max_steps > old max_steps and "
                             "extension_from_step == saved_step == old max_steps")
    return dict(saved_step=saved_step, parent_max_steps=old_max, max_steps=new_max,
                budget_extended=extension, grl_schedule_steps=new_horizon,
                grl_progress=grl_progress(saved_step, cfg),
                detector_lr=cfg.get("detector_lr"), aux_lr=cfg.get("aux_lr"),
                lr_schedule=cfg.get("lr_schedule"), verified_sources=bool(verified_sources))


def grl_progress(step, cfg):
    """Continue the original GRL ramp, saturated beyond its original horizon."""
    value = float(step)
    if not math.isfinite(value):
        raise ValueError("GRL step must be finite")
    return max(0.0, min(1.0, value / _horizon(cfg)))


def _score(value):
    if isinstance(value, bool):
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "%.4f" % numeric if math.isfinite(numeric) else "NA"


def format_evaluation(step, result, scope="current"):
    """Single-line report; a smoke/partial run cannot masquerade as 1500 images.

    ``parent_reference`` explicitly marks an old result carried into a new log.
    Mixed AP comes from pooled predictions, never a mean of per-density APs.
    """
    if not isinstance(scope, str) or re.fullmatch(r"[A-Za-z0-9_-]+", scope) is None:
        raise ValueError("Evaluation scope must be a single safe label")
    images, expected = result.get("images"), result.get("expected_images")
    complete = (result.get("complete") is True and images == 1500 and expected == 1500)
    pooled = result.get("pooled") or {}
    fields = ["EVAL", "step=%s" % step, "scope=" + scope,
              "status=" + ("full" if complete else "partial"),
              "complete=" + ("true" if complete else "false"),
              "images=%s/%s" % (images if images is not None else "unknown",
                                 expected if expected is not None else "unknown"),
              "mixed_AP50=" + _score(pooled.get("AP50")),
              "mixed_AP75=" + _score(pooled.get("AP75"))]
    densities = result.get("per_density") or {}
    for beta in ("0.005", "0.01", "0.02"):
        fields.append("fog_%s_AP50=%s" % (beta, _score((densities.get(beta) or {}).get("AP50"))))
    return " ".join(fields)
