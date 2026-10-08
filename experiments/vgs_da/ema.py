"""Small, strict EMA updates; called only after a successful student step."""
import math

import torch


@torch.no_grad()
def update_ema(teacher, student, decay):
    """Update an already frozen module from an identically shaped student.

    Floating parameters and persistent buffers receive EMA. Integer/bool
    buffers are copied, since averaging counters or masks is undefined. All
    compatibility checks run before the first mutation. No optimizer state,
    training mode, or update counter is owned by this helper.
    """
    if isinstance(decay, bool) or not isinstance(decay, (float, int)):
        raise ValueError("EMA decay must be a real number in [0, 1)")
    decay = float(decay)
    if not math.isfinite(decay) or not 0. <= decay < 1.:
        raise ValueError("EMA decay must be a real number in [0, 1)")
    if teacher is student:
        raise ValueError("EMA teacher and student must be distinct modules")
    if any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
        raise ValueError("EMA teacher parameters must be frozen and have no gradients")
    target, source = teacher.state_dict(), student.state_dict()
    if set(target) != set(source):
        raise ValueError("EMA teacher/student state keys differ")
    for key, value in target.items():
        other = source[key]
        if not torch.is_tensor(value) or not torch.is_tensor(other):
            raise ValueError("EMA state must contain tensors: " + key)
        if value.shape != other.shape or value.dtype != other.dtype or value.device != other.device:
            raise ValueError("EMA teacher/student state shape/dtype/device differs: " + key)
        if value.numel() and value.data_ptr() == other.data_ptr():
            raise ValueError("EMA teacher/student state aliases: " + key)
    for key, value in target.items():
        if value.is_floating_point():
            value.mul_(decay).add_(source[key], alpha=1. - decay)
        else:
            value.copy_(source[key])
