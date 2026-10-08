"""H2FA Eq. (2) aggregation and soft EMA image consistency, without new layers.

H2FA uses foreground ROI logits, class-routed *raw* objectness logits and
proposal-wise softmax. Non-routed entries are zero, not negative infinity.
Reference: https://openaccess.thecvf.com/content/CVPR2022/papers/
Xu_H2FA_R-CNN_Holistic_and_Hierarchical_Feature_Alignment_for_Cross-Domain_Weakly_CVPR_2022_paper.pdf
Official implementation: https://github.com/XuYunqiu/H2FA_R-CNN/blob/
499c1d90833c475fd9f288bb6bb90829989def64/detectron2/modeling/roi_heads/roi_heads.py#L501-L531

Replacing H2FA's true image labels with detached EMA probabilities is our
consistency extension. Equal soft predictions have zero consistency gradient,
but their BCE is the teacher's entropy and is generally not zero.
"""
import torch
import torch.nn.functional as F


NUM_CLASSES = 8
PROBABILITY_EPS = 1e-6


def _float_tensor(value, name):
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise ValueError("%s must be a floating-point tensor" % name)
    value = value.float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("%s must be finite in float32" % name)
    return value


def _aggregation_terms(foreground_logits, objectness_logits):
    logits = _float_tensor(foreground_logits, "foreground_logits")
    objectness = _float_tensor(objectness_logits, "objectness_logits")
    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
        raise ValueError("foreground_logits must have shape (N, 8), without background")
    if logits.shape[0] == 0:
        raise ValueError("H2FA aggregation requires at least one proposal")
    if objectness.ndim != 1 or objectness.shape[0] != logits.shape[0]:
        raise ValueError("objectness_logits must have shape (N,) matching the proposals")
    if logits.device != objectness.device:
        raise ValueError("ROI and objectness logits must be on the same device")
    # The official hard argmax routing is not differentiated; scatter retains
    # gradients to each selected proposal's continuous objectness logit.
    route = logits.argmax(dim=1, keepdim=True)
    routed = torch.zeros_like(logits).scatter(1, route, objectness[:, None])
    class_probs = F.softmax(logits, dim=1)
    proposal_weights = F.softmax(routed, dim=0)
    contributions = class_probs * proposal_weights
    return contributions, proposal_weights, objectness


def h2fa_aggregate(foreground_logits, objectness_logits):
    """Return an eight-class float32 image probability vector for one image.

Inputs are (N, 8) pre-background-softmax ROI logits and (N,) raw proposal
objectness logits. The caller supplies ROI-input proposals, before detection
score filtering/NMS. Empty images are rejected rather than assigned fake labels.
    """
    with torch.cuda.amp.autocast(enabled=False):
        contributions, _, _ = _aggregation_terms(foreground_logits, objectness_logits)
        probabilities = contributions.sum(dim=0)
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError("Nonfinite H2FA image probabilities")
        # The weighted mean is mathematically in [0, 1]. Float32 summation of
        # many nearly certain proposals can overshoot one by a few ulps.
        return probabilities.clamp(0., 1.)


def image_consistency(student_probs, teacher_probs):
    """Mean BCE over images/classes, with detached soft teacher targets.

Accepts shape (8,) or (B, 8). Only student probabilities are clamped for stable
logs, following H2FA; valid teacher targets including zero/one remain unchanged.
The reduction is a mean, so the requested external loss weight is exactly 1.0.
    """
    with torch.cuda.amp.autocast(enabled=False):
        student = _float_tensor(student_probs, "student_probs")
        teacher = _float_tensor(teacher_probs, "teacher_probs").detach()
        if student.shape != teacher.shape or student.ndim not in (1, 2):
            raise ValueError("Student and teacher probabilities need matching (8,) or (B, 8) shapes")
        if student.shape[-1] != NUM_CLASSES or student.numel() == 0:
            raise ValueError("Image probabilities must contain eight classes and a nonempty batch")
        if student.device != teacher.device:
            raise ValueError("Student and teacher probabilities must be on the same device")
        if not bool(((student >= 0) & (student <= 1)).all()):
            raise ValueError("student_probs must be in [0, 1]")
        if not bool(((teacher >= 0) & (teacher <= 1)).all()):
            raise ValueError("teacher_probs must be in [0, 1]")
        return F.binary_cross_entropy(
            student.clamp(PROBABILITY_EPS, 1.0 - PROBABILITY_EPS), teacher,
            reduction="mean")


@torch.no_grad()
def h2fa_group_diagnostics(foreground_logits, objectness_logits, base_count):
    """Return JSON-friendly RPN/VGS contribution and raw-score diagnostics.

Assumes base proposals precede supplements. Contributions add up to the image
probabilities; group weights add up to one independently for each class.
This logging helper is detached and adds no trainable path or extra loss.
    """
    with torch.cuda.amp.autocast(enabled=False):
        contributions, weights, objectness = _aggregation_terms(foreground_logits, objectness_logits)
        if type(base_count) is not int or not 0 <= base_count <= len(objectness):
            raise ValueError("base_count must be an integer within the proposal count")
        # Contributions retain their raw weighted sums for diagnostics; their
        # total differs from the bounded probabilities by float32 roundoff only.
        result = {"probabilities": contributions.sum(dim=0).clamp(0., 1.).cpu().tolist()}
        for name, region in (("base", slice(0, base_count)),
                             ("supplement", slice(base_count, None))):
            values = objectness[region]
            result[name] = {
                "count": len(values),
                "contribution": contributions[region].sum(dim=0).cpu().tolist(),
                "weight": weights[region].sum(dim=0).cpu().tolist(),
                "objectness_min": float(values.min()) if len(values) else None,
                "objectness_mean": float(values.mean()) if len(values) else None,
                "objectness_max": float(values.max()) if len(values) else None,
            }
        return result
