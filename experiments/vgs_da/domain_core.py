"""Conditional domain adversary outside the unchanged supplementary SearchHead.

The domain path observes visual p3/p4 BEFORE map fusion. Conditions and sample
reliability come only from the same frozen B0 teacher in both domains, filtered
by image-level class presence. Neither GT boxes nor per-ROI GT labels are read.

Typical integration (reuse an existing full SearchHead forward):
    tap = DomainFeatureTap(head)
    source_predictions = head(source_f3, source_f4, source_maps)
    source_visual = tap.pop()
    target_predictions = head(target_f3, target_f4, target_maps)
    target_visual = tap.pop()
    loss, stats = adapter(source_visual, source_samples,
                          target_visual, target_samples, grl_coeff=0.05)

For a separate domain-only pass, instead use:
    source_visual = prefusion_features(head, source_f3, source_f4)
    target_visual = prefusion_features(head, target_f3, target_f4)
Do not install a FeatureTap for this alternative path.

The adapter's discriminator parameters must be optimized/checkpointed separately
from SearchHead. The original head's state_dict and forward API are unchanged.
Python 3.8 compatible; no dependency on RegionCLIP or dataset GT interfaces.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align

try:
    from search_core import SearchHead, box_iou
except ImportError:
    sibling = Path(__file__).resolve().parent.parent / "learned_search"
    if not (sibling / "search_core.py").is_file():
        raise
    sys.path.insert(0, str(sibling))
    from search_core import SearchHead, box_iou


NUM_CLASSES = 8
NUM_SIZES = 3
NUM_REDUNDANCY_BINS = 2
NUM_GROUPS = NUM_CLASSES * NUM_SIZES * NUM_REDUNDANCY_BINS


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, coefficient):
        ctx.coefficient = float(coefficient)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


def gradient_reverse(value, coefficient=1.0):
    coefficient = float(coefficient)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("GRL coefficient must be finite and nonnegative")
    return _GradientReverse.apply(value, coefficient)


def prefusion_features(model, f3, f4):
    """Domain-only forward through lateral3/4, without any map/tower path.

    This is an alternative to FeatureTap, not an extra call needed when a full
    SearchHead forward is already captured. Autocast may be set by the caller.
    """
    p4 = model.lateral4(f4)
    p3 = model.lateral3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="nearest")
    return {"p3": p3, "p4": p4}


class DomainFeatureTap:
    """Non-Module hooks that preserve all SearchHead parameter/state names."""
    def __init__(self, model):
        self._captured = {}
        self._closed = False
        self._handles = [model.lateral3.register_forward_hook(self._capture("lateral3")),
                         model.lateral4.register_forward_hook(self._capture("lateral4"))]

    def _capture(self, name):
        def hook(module, inputs, output):
            if name in self._captured:
                raise RuntimeError("FeatureTap must be popped/cleared before the next forward")
            self._captured[name] = output
        return hook

    def pop(self):
        if self._closed:
            raise RuntimeError("FeatureTap has been closed")
        if set(self._captured) != {"lateral3", "lateral4"}:
            raise RuntimeError("FeatureTap requires one complete SearchHead forward")
        lateral3 = self._captured["lateral3"]
        p4 = self._captured["lateral4"]
        self._captured = {}
        p3 = lateral3 + F.interpolate(p4, size=lateral3.shape[-2:], mode="nearest")
        return {"p3": p3, "p4": p4}

    def clear(self):
        self._captured = {}

    def close(self):
        if not self._closed:
            for handle in self._handles:
                handle.remove()
            self._handles = []
            self._captured = {}
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def group_components(group_id):
    group_id = int(group_id)
    if not 0 <= group_id < NUM_GROUPS:
        raise ValueError("Invalid class/size/redundancy group")
    return {"class": group_id // 6, "size": (group_id % 6) // 2,
            "redundancy": group_id % 2}


def _counts(values):
    if not len(values):
        return {}
    groups, counts = torch.unique(values.detach().cpu(), sorted=True, return_counts=True)
    return {str(int(group)): int(count) for group, count in zip(groups.tolist(), counts.tolist())}


@torch.no_grad()
def select_domain_rois(boxes, probs, rpn_logits, image_labels,
                       max_rois=64, min_probability=0.7, min_rpn_probability=0.5,
                       redundancy_threshold=0.5, k0=300):
    """Choose stop-gradient teacher ROIs with the identical rule in both domains.

    image_labels must be an 8-element binary presence vector. It filters the
    teacher's original argmax class AFTER confidence thresholding; probabilities
    are never renormalized and a rejected class is never replaced by another.

    Group = (predicted_class * 3 + size_bin) * 2 + redundancy_bin.
    Size is original-image box area: <32^2, [32^2,96^2), >=96^2.
    Redundancy is max IoU with another B0 box, thresholded at 0.5, computed BEFORE
    reliability/image-label filtering. The threshold is not target-tuned.
    """
    if max_rois < 1 or k0 < 1:
        raise ValueError("max_rois and k0 must be positive")
    for value in (min_probability, min_rpn_probability, redundancy_threshold):
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError("Reliability and redundancy thresholds must be probabilities in [0,1]")
    boxes = torch.as_tensor(boxes).detach().cpu().float()[:k0].reshape(-1, 4)
    probabilities = torch.as_tensor(probs).detach().cpu().float()[:len(boxes)]
    logits = torch.as_tensor(rpn_logits).detach().cpu().float().reshape(-1)[:len(boxes)]
    presence = torch.as_tensor(image_labels).detach().cpu().reshape(-1)
    if presence.shape != (NUM_CLASSES,) or not bool(((presence == 0) | (presence == 1)).all()):
        raise ValueError("Both domains require an eight-element binary image_labels vector")
    if probabilities.shape != (len(boxes), 9) or len(logits) != len(boxes):
        raise ValueError("B0 boxes, nine-class probabilities and RPN logits disagree")
    if not (torch.isfinite(boxes).all() and torch.isfinite(probabilities).all() and torch.isfinite(logits).all()):
        raise ValueError("Nonfinite frozen-teacher inputs")
    if len(boxes) and not bool(((boxes[:, 2:] - boxes[:, :2]) > 0).all()):
        raise ValueError("B0 contains an empty or inverted box")
    if (probabilities < -1e-6).any() or (probabilities > 1 + 1e-6).any():
        raise ValueError("Expected teacher probabilities, not logits")
    if len(boxes) and not torch.allclose(probabilities.sum(1), torch.ones(len(boxes)), atol=2e-3, rtol=0):
        raise ValueError("Nine-class teacher probabilities must sum to one")
    if not len(boxes):
        return {"boxes": boxes, "indices": torch.empty(0, dtype=torch.long),
                "groups": torch.empty(0, dtype=torch.long), "confidence": torch.empty(0),
                "stats": {"b0_rois": 0, "reliable_before_presence": 0, "presence_rejected": 0,
                          "eligible": 0, "selected": 0, "eligible_group_counts": {},
                          "selected_group_counts": {}, "redundancy_all_counts": [0, 0],
                          "redundancy_eligible_counts": [0, 0], "raw_foreground_winners": 0,
                          "raw_background_winners": 0, "raw_group_counts": {}, "raw_class_counts": {},
                          "reliable_group_counts": {}, "eligible_class_counts": {},
                          "selected_class_counts": {}}}
    confidence, classes = probabilities[:, :NUM_CLASSES].max(dim=1)
    objectness = logits.sigmoid()
    reliable = (confidence >= min_probability) & (objectness >= min_rpn_probability)
    # Background cannot win when a foreground probability is at least 0.7, but
    # this explicit guard preserves semantics if a threshold is configured lower.
    foreground_winner = probabilities.argmax(dim=1) < NUM_CLASSES
    reliable = reliable & foreground_winner
    allowed = presence.bool()[classes]
    eligible = reliable & allowed
    pairwise = box_iou(boxes, boxes)
    pairwise.fill_diagonal_(0)
    redundancy = pairwise.max(dim=1)[0]
    redundancy_bin = (redundancy >= redundancy_threshold).long()
    area = (boxes[:, 2:] - boxes[:, :2]).prod(dim=1)
    size_bin = (area >= 32 ** 2).long() + (area >= 96 ** 2).long()
    groups = (classes * NUM_SIZES + size_bin) * NUM_REDUNDANCY_BINS + redundancy_bin
    eligible_indices = torch.where(eligible)[0].tolist()
    ranking_score = (confidence * objectness).numpy()
    by_group = {}
    for index in eligible_indices:
        by_group.setdefault(int(groups[index]), []).append(index)
    for group in by_group:
        by_group[group].sort(key=lambda index: (-float(ranking_score[index]), index))
    group_order = sorted(by_group, key=lambda group: (-float(ranking_score[by_group[group][0]]), group))
    # With 48 possible groups and a 64-ROI budget, every available condition gets
    # one slot before any group receives a second. No additional NMS is applied.
    chosen = []
    cursor = 0
    while len(chosen) < max_rois:
        added = False
        for group in group_order:
            if cursor < len(by_group[group]):
                chosen.append(by_group[group][cursor])
                added = True
                if len(chosen) == max_rois:
                    break
        if not added:
            break
        cursor += 1
    selected = torch.tensor(chosen, dtype=torch.long)
    stats = {"b0_rois": len(boxes), "reliable_before_presence": int(reliable.sum()),
             "presence_rejected": int((reliable & ~allowed).sum()), "eligible": int(eligible.sum()),
             "selected": len(selected), "eligible_group_counts": _counts(groups[eligible]),
             "selected_group_counts": _counts(groups[selected]),
             "raw_foreground_winners": int(foreground_winner.sum()),
             "raw_background_winners": int((~foreground_winner).sum()),
             "raw_group_counts": _counts(groups[foreground_winner]),
             "raw_class_counts": _counts(classes[foreground_winner]),
             "reliable_group_counts": _counts(groups[reliable]),
             "eligible_class_counts": _counts(classes[eligible]),
             "selected_class_counts": _counts(classes[selected]),
             "redundancy_all_counts": [int((redundancy_bin == value).sum()) for value in (0, 1)],
             "redundancy_eligible_counts": [int(((redundancy_bin == value) & eligible).sum()) for value in (0, 1)]}
    return {"boxes": boxes[selected], "indices": selected, "groups": groups[selected],
            "confidence": confidence[selected], "stats": stats}


def describe_domain_metadata(boxes, probs, rpn_logits, image_labels, **selection_config):
    """Support audit using only frozen-teacher metadata; never reads features."""
    return select_domain_rois(boxes, probs, rpn_logits, image_labels, **selection_config)["stats"]


def describe_cache_metadata(path, domain, image_labels=None, **selection_config):
    """Read only teacher arrays from NPZ, without decompressing f3/f4 or gt.

    Target requires explicit image_labels or an image_labels field. Source may
    derive image-level presence from its class-ID list if that field is absent;
    no GT coordinates or per-ROI labels are ever used for domain grouping.
    """
    if domain not in ("source", "target"):
        raise ValueError("domain must be source or target")
    with np.load(str(path), allow_pickle=False) as saved:
        boxes = saved["boxes"]
        probs = saved["probs"]
        rpn_logits = saved["rpn_logits"]
        if image_labels is not None:
            presence = image_labels
            origin = "explicit_image_labels"
        elif "image_labels" in saved:
            presence = saved["image_labels"]
            origin = "cached_image_labels"
        elif domain == "source" and "labels" in saved:
            classes = np.asarray(saved["labels"], np.int64).reshape(-1)
            if len(classes) and ((classes < 0) | (classes >= NUM_CLASSES)).any():
                raise ValueError("Source annotation classes outside the native eight labels")
            presence = np.zeros(NUM_CLASSES, np.int64)
            presence[classes] = 1
            origin = "source_annotation_class_presence_only"
        else:
            raise ValueError("Image-level presence missing; target annotation fallback is forbidden")
        stats = describe_domain_metadata(boxes, probs, rpn_logits, presence, **selection_config)
    return {"domain": domain, "cache_path": str(path), "presence_origin": origin,
            "metadata_only": True, **stats}


def aggregate_support(rows):
    """Sum support counts without changing thresholds or selection rules."""
    scalar_keys = ("b0_rois", "raw_foreground_winners", "raw_background_winners",
                   "reliable_before_presence", "presence_rejected", "eligible", "selected")
    count_keys = ("raw_group_counts", "raw_class_counts", "reliable_group_counts",
                  "eligible_group_counts", "eligible_class_counts", "selected_group_counts", "selected_class_counts")
    result = {"images": len(rows), **{name: 0 for name in scalar_keys},
              **{name: {} for name in count_keys}, "redundancy_all_counts": [0, 0],
              "redundancy_eligible_counts": [0, 0]}
    for row in rows:
        for name in scalar_keys:
            result[name] += int(row[name])
        for name in count_keys:
            for key, count in row[name].items():
                result[name][key] = result[name].get(key, 0) + int(count)
        for name in ("redundancy_all_counts", "redundancy_eligible_counts"):
            for index in (0, 1):
                result[name][index] += int(row[name][index])
    return result


def pool_visual_rois(features, selections, channels=64):
    """Average 3x3 ROIAlign pools from pre-fusion p3/p4, yielding visual Z."""
    p3, p4 = features["p3"], features["p4"]
    if p3.ndim != 4 or p4.ndim != 4 or p3.shape[0] != len(selections) or p4.shape[0] != len(selections):
        raise ValueError("Feature batch and per-image domain selections disagree")
    if p3.shape[1] != channels or p4.shape[1] != channels:
        raise ValueError("Unexpected pre-fusion feature channels")
    if p3.device != p4.device:
        raise ValueError("p3/p4 must be on the same device")
    roi_rows, group_rows = [], []
    for index, selection in enumerate(selections):
        boxes = selection["boxes"].to(device=p3.device, dtype=torch.float32).detach()
        if len(boxes):
            roi_rows.append(torch.cat((boxes.new_full((len(boxes), 1), index), boxes), dim=1))
            group_rows.append(selection["groups"].to(device=p3.device, dtype=torch.long).detach())
    if not roi_rows:
        # Keep a finite, zero-gradient connection to both lateral feature graphs.
        connected_zero = (p3.float().sum() + p4.float().sum()) * 0
        return p3.new_zeros((0, channels), dtype=torch.float32) + connected_zero, torch.empty(0, dtype=torch.long, device=p3.device)
    rois = torch.cat(roi_rows)
    # ROI pooling and the small discriminator intentionally use float32. Casting
    # retains gradients into FP16 AMP activations without a half-sum overflow.
    with torch.cuda.amp.autocast(enabled=False):
        z3 = roi_align(p3.float(), rois, output_size=(3, 3), spatial_scale=1.0 / 8,
                       sampling_ratio=2, aligned=True).mean(dim=(2, 3))
        z4 = roi_align(p4.float(), rois, output_size=(3, 3), spatial_scale=1.0 / 16,
                       sampling_ratio=2, aligned=True).mean(dim=(2, 3))
    return 0.5 * (z3 + z4), torch.cat(group_rows)


class DomainDiscriminator(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        # Group-specific output boundaries make D_g(z) conditional. A single
        # scalar D(z), even with balanced group losses, only aligns a reweighted
        # mixture and cannot support the claimed conditional alignment objective.
        self.net = nn.Sequential(nn.Linear(channels, 64), nn.ReLU(inplace=False), nn.Linear(64, NUM_GROUPS))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, features, groups):
        groups = groups.detach().to(features.device, dtype=torch.long).reshape(-1)
        if len(groups) != len(features):
            raise ValueError("D_g requires one condition ID per feature")
        if len(groups) and bool(((groups < 0) | (groups >= NUM_GROUPS)).any()):
            raise ValueError("Invalid D_g condition ID")
        return self.net(features).gather(1, groups[:, None]).reshape(-1)


def _balanced_logits_loss(source_logits, source_groups, target_logits, target_groups):
    """Each shared condition has equal weight; each domain has half its weight."""
    source_groups = source_groups.detach().to(source_logits.device, dtype=torch.long)
    target_groups = target_groups.detach().to(target_logits.device, dtype=torch.long)
    for groups in (source_groups, target_groups):
        if len(groups) and bool(((groups < 0) | (groups >= NUM_GROUPS)).any()):
            raise ValueError("Invalid condition group")
    count_source = torch.bincount(source_groups, minlength=NUM_GROUPS)
    count_target = torch.bincount(target_groups, minlength=NUM_GROUPS)
    common = (count_source > 0) & (count_target > 0)
    common_ids = torch.where(common)[0]
    stats = {"active_groups": len(common_ids), "source_rois": len(source_logits),
             "target_rois": len(target_logits), "common_group_ids": common_ids.detach().cpu().tolist(),
             "group_counts_source": _counts(source_groups), "group_counts_target": _counts(target_groups),
             "class_counts_source": _counts(source_groups // 6), "class_counts_target": _counts(target_groups // 6),
             "common_classes": sorted({int(group) // 6 for group in common_ids.detach().cpu().tolist()}),
             "domain_bce": 0.0, "balanced_domain_accuracy": None}
    if not len(common_ids):
        return (source_logits.float().sum() + target_logits.float().sum()) * 0, stats
    loss_source = F.binary_cross_entropy_with_logits(source_logits.float(), torch.zeros_like(source_logits).float(), reduction="none")
    loss_target = F.binary_cross_entropy_with_logits(target_logits.float(), torch.ones_like(target_logits).float(), reduction="none")
    source_sums = loss_source.new_zeros(NUM_GROUPS).scatter_add(0, source_groups, loss_source)
    target_sums = loss_target.new_zeros(NUM_GROUPS).scatter_add(0, target_groups, loss_target)
    group_loss = 0.5 * (source_sums / count_source.clamp(min=1).float()
                        + target_sums / count_target.clamp(min=1).float())
    loss = group_loss[common].mean()
    with torch.no_grad():
        correct_source = (source_logits < 0).float()
        correct_target = (target_logits >= 0).float()
        accuracy_source = correct_source.new_zeros(NUM_GROUPS).scatter_add(0, source_groups, correct_source) / count_source.clamp(min=1).float()
        accuracy_target = correct_target.new_zeros(NUM_GROUPS).scatter_add(0, target_groups, correct_target) / count_target.clamp(min=1).float()
        stats["balanced_domain_accuracy"] = float((0.5 * (accuracy_source + accuracy_target))[common].mean().item())
        stats["domain_bce"] = float(loss.item())
        stats["group_losses"] = {str(int(group)): float(group_loss[group].item()) for group in common_ids.cpu().tolist()}
    return loss, stats


def balanced_domain_bce(discriminator, source_z, source_groups, target_z, target_groups, grl_coeff=1.0):
    """Stop-gradient groups; GRL for Z only, normal gradient for D parameters."""
    if source_z.ndim != 2 or target_z.ndim != 2 or source_z.shape[1] != target_z.shape[1]:
        raise ValueError("Expected source/target ROI feature matrices with equal channel count")
    if len(source_z) != len(source_groups) or len(target_z) != len(target_groups):
        raise ValueError("ROI feature and group counts disagree")
    with torch.cuda.amp.autocast(enabled=False):
        joined = torch.cat((source_z.float(), target_z.float()), dim=0)
        groups = torch.cat((source_groups.detach().to(joined.device, dtype=torch.long),
                            target_groups.detach().to(joined.device, dtype=torch.long)))
        logits = discriminator(gradient_reverse(joined, grl_coeff), groups)
        return _balanced_logits_loss(logits[:len(source_z)], source_groups,
                                     logits[len(source_z):], target_groups)


class ConditionalDomainAdapter(nn.Module):
    """External D + reliable common-condition loss; no SearchHead ownership."""
    def __init__(self, channels=64, max_rois=64, min_probability=0.7,
                 min_rpn_probability=0.5, redundancy_threshold=0.5, k0=300):
        super().__init__()
        self.channels = int(channels)
        self.discriminator = DomainDiscriminator(channels)
        self.selection_config = {"max_rois": int(max_rois), "min_probability": float(min_probability),
                                 "min_rpn_probability": float(min_rpn_probability),
                                 "redundancy_threshold": float(redundancy_threshold), "k0": int(k0)}

    def _prepare(self, features, samples):
        selections = []
        for sample in samples:
            # Deliberately no access to sample['gt'] or sample['labels'].
            selections.append(select_domain_rois(sample["boxes"], sample["probs"],
                sample["rpn_logits"], sample["image_labels"], **self.selection_config))
        z, groups = pool_visual_rois(features, selections, channels=self.channels)
        return z, groups, [selected["stats"] for selected in selections]

    def forward(self, source_features, source_samples, target_features, target_samples, grl_coeff=1.0):
        source_z, source_groups, source_stats = self._prepare(source_features, source_samples)
        target_z, target_groups, target_stats = self._prepare(target_features, target_samples)
        loss, stats = balanced_domain_bce(self.discriminator, source_z, source_groups,
                                          target_z, target_groups, grl_coeff)
        stats["source_selection"] = source_stats
        stats["target_selection"] = target_stats
        stats["grl_coeff"] = float(grl_coeff)
        return loss, stats


def self_test():
    import json
    torch.set_num_threads(2)
    torch.manual_seed(17)
    x = torch.tensor([1., 2.], requires_grad=True)
    gradient_reverse(x, .37).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -.37))

    # Compare GRL gradients with an ordinary positive-domain-BCE reference.
    d1, d2 = DomainDiscriminator(4), DomainDiscriminator(4)
    d2.load_state_dict(d1.state_dict())
    s1 = torch.randn(3, 4, requires_grad=True)
    t1 = torch.randn(2, 4, requires_grad=True)
    s2, t2 = s1.detach().clone().requires_grad_(), t1.detach().clone().requires_grad_()
    sg, tg = torch.tensor([0, 0, 7]), torch.tensor([0, 7])
    ordinary, ordinary_stats = _balanced_logits_loss(d1(s1, sg), sg, d1(t1, tg), tg)
    reversed_loss, reversed_stats = balanced_domain_bce(d2, s2, sg, t2, tg, .2)
    ordinary.backward()
    reversed_loss.backward()
    assert torch.allclose(s2.grad, -.2 * s1.grad, atol=1e-8, rtol=1e-5)
    assert torch.allclose(t2.grad, -.2 * t1.grad, atol=1e-8, rtol=1e-5)
    for a, b in zip(d1.parameters(), d2.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-8, rtol=1e-5)
    # Duplicate a single-domain group many times: its domain/group weight cannot
    # change. Unshared groups must not affect loss or reported balanced accuracy.
    a = torch.tensor([.3, -.9])
    b = torch.tensor([-.5, .8])
    original, _ = _balanced_logits_loss(a, torch.tensor([0, 7]), b, torch.tensor([0, 7]))
    duplicated, _ = _balanced_logits_loss(torch.tensor([.3] * 20 + [-.9, 9.]),
        torch.tensor([0] * 20 + [7, 17]), b, torch.tensor([0, 7]))
    assert torch.allclose(original, duplicated, atol=1e-7)
    assert ordinary_stats["active_groups"] == reversed_stats["active_groups"] == 2

    boxes = torch.tensor([[8., 8., 24., 24.], [10., 10., 26., 26.], [40., 40., 56., 56.]])
    probabilities = torch.zeros(3, 9)
    probabilities[0:2, 2], probabilities[0:2, 8] = .8, .2
    probabilities[2, 7], probabilities[2, 8] = .9, .1
    presence = torch.zeros(8, dtype=torch.long)
    presence[2] = 1
    teacher_probs = probabilities.requires_grad_()
    selection = select_domain_rois(boxes, teacher_probs, torch.ones(3) * 2, presence)
    assert len(selection["boxes"]) == 2 and selection["stats"]["presence_rejected"] == 1
    assert selection["stats"]["redundancy_all_counts"] == [1, 2]
    assert selection["groups"].tolist() == [13, 13]  # car, small, redundant
    assert not selection["confidence"].requires_grad
    support = aggregate_support([selection["stats"], selection["stats"]])
    assert support["images"] == 2 and support["selected"] == 4
    assert support["raw_class_counts"] == {"2": 4, "7": 2}
    class MetadataOnly:
        def __init__(self):
            self.data = {"boxes": boxes.numpy(), "probs": probabilities.detach().numpy(),
                         "rpn_logits": np.ones(3) * 2, "labels": np.array([2])}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __contains__(self, key):
            return key in self.data
        def __getitem__(self, key):
            if key in ("f3", "f4", "gt"):
                raise AssertionError("Metadata audit attempted to read visual features or GT coordinates")
            return self.data[key]
    metadata_only = MetadataOnly()
    original_np_load = np.load
    try:
        np.load = lambda *args, **kwargs: metadata_only
        source_audit = describe_cache_metadata("metadata_only_test.npz", "source")
        assert source_audit["presence_origin"] == "source_annotation_class_presence_only"
        target_fallback_rejected = False
        try:
            describe_cache_metadata("metadata_only_test.npz", "target")
        except ValueError:
            target_fallback_rejected = True
        assert target_fallback_rejected
        metadata_only.data["image_labels"] = presence.numpy()
        target_audit = describe_cache_metadata("metadata_only_test.npz", "target")
        assert target_audit["presence_origin"] == "cached_image_labels"
    finally:
        np.load = original_np_load
    # Identical visual vectors can have different conditional domain boundaries.
    conditional = DomainDiscriminator(4)
    with torch.no_grad():
        conditional.net[-1].weight.zero_()
        conditional.net[-1].bias.zero_()
        conditional.net[-1].bias[7] = 2.0
    conditional_logits = conditional(torch.zeros(2, 4), torch.tensor([0, 7]))
    assert conditional_logits.tolist() == [0., 2.]

    class GuardedSample(dict):
        def __getitem__(self, key):
            if key in ("gt", "labels"):
                raise AssertionError("Domain grouping attempted to read GT")
            return super().__getitem__(key)
    sample = GuardedSample(boxes=boxes, probs=teacher_probs, rpn_logits=torch.ones(3) * 2,
                           image_labels=presence, gt=None, labels=None)
    model = SearchHead(f3_channels=8, f4_channels=16, channels=64, arm="vgs")
    state_keys = list(model.state_dict())
    f3s, f4s = torch.randn(1, 8, 8, 8), torch.randn(1, 16, 4, 4)
    f3t, f4t = torch.randn(1, 8, 8, 8), torch.randn(1, 16, 4, 4)
    maps = torch.randn(1, 15, 8, 8)
    with DomainFeatureTap(model) as tap:
        model(f3s, f4s, maps)
        source_features = tap.pop()
        model(f3t, f4t, maps)
        target_features = tap.pop()
    assert list(model.state_dict()) == state_keys
    direct = prefusion_features(model, f3s, f4s)
    assert torch.equal(direct["p3"], source_features["p3"])
    assert torch.equal(direct["p4"], source_features["p4"])
    adapter = ConditionalDomainAdapter()
    loss, stats = adapter(source_features, [sample], target_features, [sample], grl_coeff=.1)
    loss.backward()
    assert stats["active_groups"] == 1 and stats["source_rois"] == stats["target_rois"] == 2
    nonzero = []
    for name, parameter in model.named_parameters():
        if name.startswith(("lateral3.", "lateral4.")):
            assert parameter.grad is not None
            if bool((parameter.grad != 0).any()):
                nonzero.append(name)
        else:
            assert parameter.grad is None, "Domain gradient leaked into " + name
    assert any(name.startswith("lateral3.") for name in nonzero)
    assert any(name.startswith("lateral4.") for name in nonzero)
    assert teacher_probs.grad is None
    assert all(parameter.grad is not None for parameter in adapter.parameters())

    # No shared condition/ROIs: a finite graph-connected zero can be backpropagated.
    model.zero_grad(set_to_none=True)
    adapter.zero_grad(set_to_none=True)
    empty_source = dict(sample)
    empty_source["image_labels"] = torch.zeros(8)
    sf, tf = prefusion_features(model, f3s, f4s), prefusion_features(model, f3t, f4t)
    empty_loss, empty_stats = adapter(sf, [empty_source], tf, [sample], grl_coeff=.1)
    assert empty_loss.requires_grad and float(empty_loss) == 0 and empty_stats["active_groups"] == 0
    empty_loss.backward()
    assert all(p.grad is not None for p in adapter.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print(json.dumps({"passed": True, "checks": ["GRL_negative_feature_gradient",
        "D_positive_gradient_unchanged", "per_domain_and_group_balance", "unshared_group_ignored",
        "teacher_presence_filter_without_relabeling", "observable_redundancy",
        "conditional_group_specific_D_boundaries", "metadata_support_aggregation",
        "metadata_does_not_read_features_or_boxes_GT", "target_GT_presence_fallback_forbidden",
        "GT_read_guard", "SearchHead_state_dict_unchanged", "tap_equals_direct_prefusion",
        "domain_gradients_only_lateral3_lateral4", "teacher_conditions_stop_gradient",
        "empty_shared_group_backward"], "example_active_groups": stats["active_groups"],
        "domain_loss": float(loss.detach()), "domain_parameter_count": sum(p.numel() for p in adapter.parameters())}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.print_help()
