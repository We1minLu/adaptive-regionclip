"""Overflow-safe coordinated AMP steps for the formal detector optimizers.

The helper makes one all-or-none *gradient-validity decision*. It cannot roll
back an unrelated exception raised inside an optimizer's own step method.
It never retries, clears gradients, advances a data loader, or saves a model.
The caller must replay the same batch/RNG after ``retry_required=True``.
Compatible with public torch.cuda.amp.GradScaler APIs in PyTorch 1.9.
"""
import math

import torch


def _gradient_records(optimizers, named_parameters):
    names = {id(parameter): name for name, parameter in named_parameters}
    records, seen, with_grad = [], set(), []
    for optimizer_index, optimizer in enumerate(optimizers):
        count = 0
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                identity = id(parameter)
                if identity in seen:
                    raise ValueError("An optimized parameter occurs more than once")
                seen.add(identity)
                if identity not in names:
                    raise ValueError("Every optimized parameter needs a diagnostic name")
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise ValueError("Formal AMP helper requires dense gradients")
                if not gradient.is_floating_point():
                    raise ValueError("Expected floating-point gradients: " + names[identity])
                records.append((names[identity], optimizer_index, parameter))
                count += 1
        with_grad.append(count)
    if not records or not all(with_grad):
        raise ValueError("Each optimizer must have at least one gradient before coordinated stepping")
    devices = {parameter.grad.device for _, _, parameter in records}
    if len(devices) != 1:
        raise ValueError("Formal AMP helper expects all gradients on one device")
    return records


def _nonfinite_diagnostics(records):
    # Only one host synchronization on the common finite path. On failure,
    # detailed element counts are collected for the affected parameters only.
    flags = torch.stack([torch.isfinite(parameter.grad).all()
                         for _, _, parameter in records]).cpu().tolist()
    failures = []
    for finite, (name, optimizer_index, parameter) in zip(flags, records):
        if finite:
            continue
        gradient = parameter.grad
        failures.append({"parameter": name, "optimizer_index": optimizer_index,
                         "shape": list(gradient.shape), "dtype": str(gradient.dtype),
                         "nan_count": int(torch.isnan(gradient).sum().item()),
                         "positive_inf_count": int((torch.isinf(gradient) & (gradient > 0)).sum().item()),
                         "negative_inf_count": int((torch.isinf(gradient) & (gradient < 0)).sum().item())})
    return failures


def _stable_global_norm(records):
    # FP64 reduction is per gradient tensor, not a concatenated FP64 copy of all
    # model parameters/gradients. Only one scalar per parameter is retained.
    # FP32 finite values have ample FP64 squared-sum range, including values
    # whose FP32 sum of squares or resulting FP32 norm would overflow.
    norms = [torch.norm(parameter.grad.detach(), p=2, dtype=torch.float64)
             for _, _, parameter in records]
    return torch.norm(torch.stack(norms), p=2)


def _backoff_and_reset(scaler, old_scale):
    state = scaler.state_dict()
    new_scale = old_scale * float(state["backoff_factor"])
    if not math.isfinite(new_scale) or new_scale <= 0:
        raise FloatingPointError("AMP backoff produced an invalid scale: %r" % new_scale)
    # update(new_scale=...) clears per-optimizer unscale/found-inf bookkeeping,
    # but does not reset the growth counter. Reset it through the public state
    # round-trip, never by modifying GradScaler's private tensor attributes.
    scaler.update(new_scale=new_scale)
    state = scaler.state_dict()
    if not math.isfinite(float(state["scale"])) or float(state["scale"]) <= 0:
        raise FloatingPointError("AMP scale underflowed during backoff")
    state["_growth_tracker"] = 0
    scaler.load_state_dict(state)
    return float(scaler.get_scale())


@torch.no_grad()
def atomic_amp_step(optimizers, scaler, named_parameters, max_norm=10.0):
    """Unscale, check all gradients, then jointly step or request a replay.

    Call after ``scaler.scale(loss).backward()`` for the complete accumulated
    loss. ``named_parameters`` may contain frozen/unused parameters, but must
    name every optimized parameter. Optimizer parameter sets must be disjoint.

    On overflow with AMP enabled: no optimizer is stepped; scale is backed off
    and its growth counter reset; caller must clear gradients and replay the
    same batch/RNG. Retry limits/minimum scale belong to that caller.
    With AMP disabled, nonfinite gradients raise immediately before any step.
    The returned norm is the unclipped FP64 global L2 norm (None on overflow).
    """
    optimizers = list(optimizers)
    if not optimizers:
        raise ValueError("Expected at least one optimizer")
    max_norm = float(max_norm)
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be finite and positive")
    records = _gradient_records(optimizers, named_parameters)
    old_scale = float(scaler.get_scale())
    for optimizer in optimizers:
        scaler.unscale_(optimizer)
    # Unscale may replace gradient tensors, so records deliberately retain
    # parameters rather than references to their pre-unscale .grad values.
    failures = _nonfinite_diagnostics(records)
    result = {"did_step": False, "retry_required": False,
              "old_scale": old_scale, "new_scale": old_scale,
              "global_grad_norm": None, "clip_coefficient": None,
              "nonfinite_gradients": failures,
              "parameters_with_grad": len(records), "optimizers": len(optimizers)}
    if failures:
        if not scaler.is_enabled():
            raise FloatingPointError("Nonfinite gradients with AMP disabled; no optimizer updated: "
                                     + ", ".join(row["parameter"] for row in failures))
        result["new_scale"] = _backoff_and_reset(scaler, old_scale)
        result["retry_required"] = True
        return result
    norm_tensor = _stable_global_norm(records)
    norm = float(norm_tensor.item())
    if not math.isfinite(norm):
        # Finite FP32 gradient elements cannot exhaust FP64 range in this model.
        # A nonfinite FP64 norm therefore warrants investigation, not silently
        # masking an unrelated implementation/dtype problem by skipping a step.
        raise FloatingPointError("Nonfinite FP64 global gradient norm despite finite elements; no optimizer updated")
    coefficient = min(1.0, max_norm / (norm + 1e-6))
    if coefficient < 1.0:
        for _, _, parameter in records:
            parameter.grad.mul_(coefficient)
    # Every optimizer has passed the same global preflight before the first
    # step. GradScaler's per-optimizer found_inf flags are all zero here.
    for optimizer in optimizers:
        scaler.step(optimizer)
    scaler.update()
    result.update(did_step=True, new_scale=float(scaler.get_scale()),
                  global_grad_norm=norm, clip_coefficient=coefficient)
    return result
