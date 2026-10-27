"""DeST teacher aggregation and positive-only RPN spatial constraints.

This module deliberately depends on PyTorch alone, including on CPU. Coordinates
are continuous XYXY; image_size is (height, width). Teacher targets, coordinates,
and optimization weights are detached. Only student objectness is differentiated
by relaxed_spatial_loss. The caller averages its returned per-image losses.
"""
import math

import torch
from torch.nn import functional as F


def _number(value, name, minimum=None, maximum=None):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % name)
    if minimum is not None and value < minimum:
        raise ValueError("%s must be >= %s" % (name, minimum))
    if maximum is not None and value > maximum:
        raise ValueError("%s must be <= %s" % (name, maximum))
    return value


def _size(image_size):
    if len(image_size) != 2:
        raise ValueError("image_size must be (height, width)")
    height, width = (_number(v, "image dimension", minimum=0.) for v in image_size)
    if height == 0 or width == 0:
        raise ValueError("Image dimensions must be positive")
    return height, width


def _floating(value, name):
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise ValueError("%s must be a floating-point tensor" % name)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("%s must contain finite values" % name)
    return value


def _boxes(boxes, name="boxes"):
    _floating(boxes, name)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("%s must have shape (N,4)" % name)
    return boxes


def collect_windows(image_size, scales, device=None, dtype=torch.float32):
    """Build a half-stride grid, adding no redundant boundary-only windows.

    At each scale the top-left grid starts at zero. The last window along each
    axis is the first to reach the boundary; if needed, that last window is
    clipped. Thus {1/8,1/4,1/2,1} yields 225+49+9+1=284 windows. Repeated scales
    and coincident clipped windows are removed without changing their order.
    """
    height, width = _size(image_size)
    rows, seen = [], set()
    for scale in scales:
        scale = _number(scale, "window scale", minimum=0., maximum=1.)
        if scale == 0:
            raise ValueError("Window scales must be positive")
        # All coordinates are generated in Python, not by GPU scalar accesses.
        last = int(math.ceil(max(0., 2. / scale - 2.) - 1e-10))
        size_x, size_y = scale * width, scale * height
        for iy in range(last + 1):
            top = iy * size_y / 2.
            for ix in range(last + 1):
                left = ix * size_x / 2.
                row = (left, top, min(left + size_x, width), min(top + size_y, height))
                key = tuple(round(v, 10) for v in row)
                if key not in seen and row[2] > row[0] and row[3] > row[1]:
                    rows.append(row)
                    seen.add(key)
    return torch.tensor(rows, dtype=dtype, device=device).reshape(-1, 4)


@torch.no_grad()
def overlap_kernel(boxes):
    """Gram kernel of normalized rectangle indicators, not IoU/covariance."""
    boxes = _boxes(boxes).detach()
    boxes = boxes.to(dtype=torch.float64 if boxes.dtype == torch.float64 else torch.float32)
    extent = boxes[:, 2:] - boxes[:, :2]
    if len(boxes) == 0 or not bool((extent > 0).all()):
        raise ValueError("overlap_kernel requires nonempty positive-area regions")
    area = extent.prod(dim=1)
    low = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    high = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    intersection = (high - low).clamp_min(0).prod(dim=2)
    # sqrt(a)*sqrt(b) avoids unnecessary overflow from first multiplying areas.
    root_area = area.sqrt()
    return (intersection / (root_area[:, None] * root_area[None, :])).clamp(0., 1.)


def _line_search(nominal_direction, norm_sq, cross_term, direction_norm_sq, rho):
    """Batched exact one-dimensional concave maximization, up to bisection.

    Along w+t(e_j-w), support is linear and squared norm is a quadratic.
    Thirty bisections avoid a generic optimizer and host/device synchronization.
    """
    tiny = torch.finfo(norm_sq.dtype).tiny

    def derivative(step):
        square = norm_sq + 2. * step * cross_term + step.square() * direction_norm_sq
        return nominal_direction - rho * (cross_term + step * direction_norm_sq) / square.clamp_min(tiny).sqrt()

    lower, upper = torch.zeros_like(norm_sq), torch.ones_like(norm_sq)
    at_zero, at_one = derivative(lower), derivative(upper)
    for _ in range(30):
        middle = .5 * (lower + upper)
        positive = derivative(middle) > 0
        lower = torch.where(positive, middle, lower)
        upper = torch.where(positive, upper, middle)
    interior = .5 * (lower + upper)
    return torch.where(at_zero <= 0, torch.zeros_like(interior),
                       torch.where(at_one >= 0, torch.ones_like(interior), interior))


@torch.no_grad()
def robust_support(support, kernel, rho=1., max_iter=100, tol=1e-4,
                   return_weights=False):
    """Maximize s'w-rho*sqrt(w'Kw) independently for every category.

    support is N x C and kernel is N x N. The public teacher entry point below
    constructs the PSD kernel from regions. This lower-level function also
    permits the identity/all-ones kernels for controlled mathematical tests.
    No matrix square root, inverse, autograd graph, or external solver is needed.
    K*w is updated in O(N*C) per Frank--Wolfe iteration after computing K once.
    """
    support, kernel = _floating(support, "support").detach(), _floating(kernel, "kernel").detach()
    rho = _number(rho, "rho", minimum=0.)
    tol = _number(tol, "tol", minimum=0.)
    if type(max_iter) is not int or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if support.ndim != 2 or min(support.shape) == 0:
        raise ValueError("support must have nonempty shape (N,C)")
    regions, classes = support.shape
    if kernel.shape != (regions, regions) or kernel.device != support.device:
        raise ValueError("kernel must be (N,N) on the support device")
    dtype = torch.float64 if support.dtype == torch.float64 else torch.float32
    support, kernel = support.to(dtype), kernel.to(dtype)
    # All valid overlap, identity, and all-ones kernels satisfy these checks.
    if not bool((kernel.diagonal() > 0).all()) or not torch.allclose(kernel, kernel.t(), atol=1e-6, rtol=1e-6):
        raise ValueError("kernel must be symmetric with positive diagonal")
    columns = torch.arange(classes, device=support.device)
    winner = support.argmax(dim=0)
    weights = torch.zeros_like(support)
    weights[winner, columns] = 1.
    kw = kernel[:, winner].clone()
    nominal = support[winner, columns].clone()
    iterations = torch.zeros(classes, dtype=torch.long, device=support.device)
    tiny = torch.finfo(dtype).tiny
    # Freezing converged categories avoids changing their certified solution.
    active = torch.ones(classes, dtype=torch.bool, device=support.device)
    for _ in range(max_iter):
        norm_sq = (weights * kw).sum(dim=0).clamp_min(tiny)
        norm = norm_sq.sqrt()
        gradient = support - rho * kw / norm[None, :]
        best_grad, vertex = gradient.max(dim=0)
        objective = nominal - rho * norm
        gap = (best_grad - objective).clamp_min(0.)
        active = active & (gap > tol)
        # One early-stop synchronization per iteration, never per region/class.
        if not bool(active.any()):
            break
        delta_nominal = support[vertex, columns] - nominal
        kw_at_vertex = kw[vertex, columns]
        cross = kw_at_vertex - norm_sq
        delta_sq = (kernel[vertex, vertex] - 2. * kw_at_vertex + norm_sq).clamp_min(0.)
        if rho == 0:
            step = active.to(dtype)
        else:
            step = _line_search(delta_nominal, norm_sq, cross, delta_sq, rho)
            step = step * active.to(dtype)
        weights.mul_(1. - step[None, :])
        weights[vertex, columns] += step
        kw = kw * (1. - step[None, :]) + kernel[:, vertex] * step[None, :]
        nominal = nominal + step * delta_nominal
        iterations += active.to(torch.long)
    # Recompute once to avoid accumulation error in the incremental K*w state.
    kw = kernel @ weights
    nominal = (support * weights).sum(dim=0)
    norm = (weights * kw).sum(dim=0).clamp_min(tiny).sqrt()
    objective = nominal - rho * norm
    gradient = support - rho * kw / norm[None, :]
    gap = (gradient.max(dim=0).values - objective).clamp_min(0.)
    stats = {
        "regions": regions, "classes": classes, "rho": rho,
        "max_iterations": max_iter, "tolerance": tol,
        "iterations": iterations.cpu().tolist(),
        "max_duality_gap": float(gap.max()),
        "duality_gap": gap.cpu().tolist(),
        "converged": (gap <= tol).cpu().tolist(),
        "nominal_support": nominal.cpu().tolist(),
        "uncertainty_penalty": (rho * norm).cpu().tolist(),
        "robust_support": objective.cpu().tolist(),
        "effective_regions": (1. / weights.square().sum(dim=0)).cpu().tolist(),
    }
    if return_weights:
        # Optional diagnostics remain JSON serializable and detached.
        stats["weights"] = weights.cpu().tolist()
    return objective, stats


@torch.no_grad()
def mrsa_aggregate(boxes, all_class_logits, rho=1., max_iter=100, tol=1e-4,
                   return_weights=False, probability_eps=1e-6):
    """Return (C-vector detached soft targets, JSON-friendly diagnostics).

    all_class_logits contains C foreground columns followed by background.
    Degenerate regions are excluded together with their score rows. An empty
    evidence bank is an error: it must not silently become a negative label.
    """
    boxes = _boxes(boxes).detach()
    logits = _floating(all_class_logits, "all_class_logits").detach()
    if logits.ndim != 2 or logits.shape[0] != len(boxes) or logits.shape[1] < 2:
        raise ValueError("Logits must be (N,C+1), with a final background column")
    if logits.device != boxes.device:
        raise ValueError("Boxes and logits must share a device")
    probability_eps = _number(probability_eps, "probability_eps", minimum=0., maximum=.49)
    if probability_eps == 0:
        raise ValueError("probability_eps must be positive")
    valid = ((boxes[:, 2:] - boxes[:, :2]) > 0).all(dim=1)
    if not bool(valid.any()):
        raise ValueError("MRSA requires at least one positive-area evidence region")
    original_count = len(boxes)
    boxes, logits = boxes[valid], logits[valid]
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    # Explicit full precision even when called under CUDA autocast.
    with torch.cuda.amp.autocast(enabled=False):
        probability = F.softmax(logits.to(dtype), dim=1)[:, :-1]
        probability = probability.clamp(probability_eps, 1. - probability_eps)
        support = probability.log() - torch.log1p(-probability)
        kernel = overlap_kernel(boxes.to(dtype))
        robust, stats = robust_support(support, kernel, rho, max_iter, tol, return_weights)
        targets = robust.sigmoid()
    stats["excluded_empty_regions"] = original_count - len(boxes)
    stats["probabilities"] = targets.cpu().tolist()
    return targets, stats


def relaxed_spatial_loss(raw_logits, anchor_centers, final_boxes, image_size,
                         gamma=1.5, tau=.7):
    """Per-image positive-only max-response hinge and detached diagnostics.

    raw_logits is (A,), anchor_centers is (A,2). final_boxes must already have
    passed teacher confidence filtering and NMS; extra semantic windows must
    never be passed here. Empty/degenerate/clipped-empty supports do not count
    in the denominator. Overlapping supports may share a responding anchor.
    """
    raw_logits = _floating(raw_logits, "raw_logits")
    anchor_centers = _floating(anchor_centers, "anchor_centers").detach()
    final_boxes = _boxes(final_boxes, "final_boxes").detach()
    height, width = _size(image_size)
    gamma = _number(gamma, "gamma", minimum=1.)
    tau = _number(tau, "tau", minimum=0., maximum=1.)
    if tau == 0 or tau == 1:
        raise ValueError("tau must lie strictly between zero and one")
    if raw_logits.ndim != 1 or anchor_centers.shape != (len(raw_logits), 2):
        raise ValueError("Expected logits (A,) and matching anchor_centers (A,2)")
    if anchor_centers.device != raw_logits.device or final_boxes.device != raw_logits.device:
        raise ValueError("Student logits, centers, and teacher boxes must share a device")
    # Float32 preserves the hinge under AMP; double precision is useful for tests.
    dtype = torch.float64 if raw_logits.dtype == torch.float64 else torch.float32
    with torch.cuda.amp.autocast(enabled=False):
        logits = raw_logits.to(dtype)
        boxes = final_boxes.to(dtype)
        extent = boxes[:, 2:] - boxes[:, :2]
        positive = (extent > 0).all(dim=1)
        center = .5 * (boxes[:, 2:] + boxes[:, :2])
        support = torch.cat((center - .5 * gamma * extent,
                             center + .5 * gamma * extent), dim=1)
        support[:, 0::2].clamp_(0., width)
        support[:, 1::2].clamp_(0., height)
        positive = positive & ((support[:, 2:] - support[:, :2]) > 0).all(dim=1)
        support = support[positive]
        centers = anchor_centers.to(dtype)
        inside = ((centers[None, :, :] >= support[:, None, :2]) &
                  (centers[None, :, :] <= support[:, None, 2:])).all(dim=2)
        valid = inside.any(dim=1)
        if len(logits) == 0 or not bool(valid.any()):
            # Keep a zero-valued autograd connection to the student.
            loss = logits.sum() * 0.
            return loss, {"teacher_boxes": len(final_boxes), "valid_supports": 0,
                          "active_supports": 0, "active_fraction": 0.,
                          "mean_max_objectness": 0., "loss": 0.}
        inside = inside[valid]
        probabilities = logits.sigmoid()
        responses = probabilities[None, :].expand(len(inside), -1).masked_fill(~inside, -1.).max(dim=1).values
        hinges = F.relu(tau - responses)
        loss = hinges.mean()
    stats = {
        "teacher_boxes": len(final_boxes), "valid_supports": len(responses),
        "active_supports": int((hinges.detach() > 0).sum()),
        "active_fraction": float((hinges.detach() > 0).to(dtype).mean()),
        "mean_max_objectness": float(responses.detach().mean()),
        "loss": float(loss.detach()),
    }
    return loss, stats
