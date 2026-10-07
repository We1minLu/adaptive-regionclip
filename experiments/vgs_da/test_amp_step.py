"""Meaningful CPU/GPU checks for atomic_amp_step; run directly with Python.

GPU cases use real torch.cuda.amp.GradScaler, including the PyTorch 1.9 public
state APIs. CPU cases validate disabled-AMP behavior and stable clipping.
No dataset, detector checkpoint, training directory or external GPU job used.
"""
import copy
import math
import unittest

import torch

from amp_step import atomic_amp_step


def assert_nested_equal(test, left, right):
    test.assertEqual(type(left), type(right))
    if torch.is_tensor(left):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_nested_equal(test, a, b)
    else:
        test.assertEqual(left, right)


def setup_pair(device, enabled, init_scale=128.0):
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0], device=device))
    second = torch.nn.Parameter(torch.tensor([-1.0, .5], device=device))
    optimizers = [torch.optim.SGD([first], lr=.1, momentum=.9),
                  torch.optim.AdamW([second], lr=.1, weight_decay=1e-4)]
    scaler = torch.cuda.amp.GradScaler(enabled=enabled, init_scale=init_scale,
                                       growth_interval=100, backoff_factor=.5)
    named = [("first.weight", first), ("second.weight", second)]
    return first, second, optimizers, scaler, named


def backward_pair(first, second, optimizers, scaler):
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    loss = .5 * (first.square().sum() + second.square().sum())
    scaler.scale(loss).backward()


class CPUChecks(unittest.TestCase):
    def test_disabled_amp_finite_clipping_and_both_updates(self):
        p = torch.nn.Parameter(torch.tensor([3.0]))
        q = torch.nn.Parameter(torch.tensor([4.0]))
        opts = [torch.optim.SGD([p], lr=.1), torch.optim.SGD([q], lr=.1)]
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        scaler.scale(.5 * (p.square().sum() + q.square().sum())).backward()
        report = atomic_amp_step(opts, scaler, [("p", p), ("q", q)], max_norm=2.0)
        self.assertTrue(report["did_step"])
        self.assertFalse(report["retry_required"])
        self.assertAlmostEqual(report["global_grad_norm"], 5.0, places=10)
        self.assertAlmostEqual(report["clip_coefficient"], 2.0 / (5.0 + 1e-6), places=10)
        self.assertAlmostEqual(float(p), 2.88, places=6)
        self.assertAlmostEqual(float(q), 3.84, places=6)
        self.assertEqual(p.dtype, torch.float32)

    def test_disabled_amp_failure_changes_neither_optimizer(self):
        p, q, opts, scaler, named = setup_pair("cpu", False)
        backward_pair(p, q, opts, scaler)
        atomic_amp_step(opts, scaler, named)
        weights = [p.detach().clone(), q.detach().clone()]
        states = copy.deepcopy([opt.state_dict() for opt in opts])
        backward_pair(p, q, opts, scaler)
        q.grad[0] = float("nan")
        with self.assertRaises(FloatingPointError):
            atomic_amp_step(opts, scaler, named)
        for weight, parameter in zip(weights, (p, q)):
            self.assertTrue(torch.equal(weight, parameter))
        assert_nested_equal(self, states, [opt.state_dict() for opt in opts])

    def test_huge_finite_gradient_norm_does_not_overflow(self):
        p, q, opts, scaler, named = setup_pair("cpu", False)
        backward_pair(p, q, opts, scaler)
        p.grad.fill_(3e38)
        q.grad.fill_(3e38)
        element = float(p.grad[0])
        report = atomic_amp_step(opts, scaler, named, max_norm=10.0)
        self.assertTrue(report["did_step"])
        self.assertTrue(math.isfinite(report["global_grad_norm"]))
        self.assertAlmostEqual(report["global_grad_norm"] / (2.0 * element), 1.0, places=12)
        clipped_norm = math.sqrt(float(p.grad.double().square().sum() + q.grad.double().square().sum()))
        self.assertAlmostEqual(clipped_norm, 10.0, places=5)
        self.assertTrue(torch.isfinite(p).all() and torch.isfinite(q).all())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for real AMP scale/backoff checks")
class GPUChecks(unittest.TestCase):
    def test_one_optimizer_inf_or_nan_blocks_both_and_retry_succeeds(self):
        for broken_optimizer in (0, 1):
            for invalid in (float("inf"), float("nan")):
                with self.subTest(optimizer=broken_optimizer, invalid=str(invalid)):
                    p, q, opts, scaler, named = setup_pair("cuda", True)
                    # Warm up actual momentum buffers and a nonzero growth counter.
                    backward_pair(p, q, opts, scaler)
                    warmup = atomic_amp_step(opts, scaler, named)
                    self.assertTrue(warmup["did_step"])
                    self.assertGreater(scaler.state_dict()["_growth_tracker"], 0)
                    weights = [p.detach().clone(), q.detach().clone()]
                    states = copy.deepcopy([opt.state_dict() for opt in opts])
                    old_scale = scaler.get_scale()
                    backward_pair(p, q, opts, scaler)
                    (p, q)[broken_optimizer].grad[0] = invalid
                    report = atomic_amp_step(opts, scaler, named)
                    self.assertFalse(report["did_step"])
                    self.assertTrue(report["retry_required"])
                    self.assertEqual(report["new_scale"], old_scale * .5)
                    self.assertEqual(scaler.get_scale(), old_scale * .5)
                    self.assertEqual(scaler.state_dict()["_growth_tracker"], 0)
                    self.assertEqual(len(report["nonfinite_gradients"]), 1)
                    failure = report["nonfinite_gradients"][0]
                    self.assertEqual(failure["optimizer_index"], broken_optimizer)
                    self.assertEqual(failure["parameter"], named[broken_optimizer][0])
                    self.assertEqual(failure["nan_count"] + failure["positive_inf_count"], 1)
                    for weight, parameter in zip(weights, (p, q)):
                        self.assertTrue(torch.equal(weight, parameter))
                    assert_nested_equal(self, states, [opt.state_dict() for opt in opts])
                    # Must not raise "unscale_ already called" on this replay.
                    backward_pair(p, q, opts, scaler)
                    retried = atomic_amp_step(opts, scaler, named)
                    self.assertTrue(retried["did_step"])
                    self.assertFalse(retried["retry_required"])
                    self.assertEqual(scaler.state_dict()["_growth_tracker"], 1)
                    for weight, parameter in zip(weights, (p, q)):
                        self.assertFalse(torch.equal(weight, parameter))

    def test_real_amp_clipping_uses_unscaled_global_norm(self):
        p = torch.nn.Parameter(torch.tensor([3.0], device="cuda"))
        q = torch.nn.Parameter(torch.tensor([4.0], device="cuda"))
        opts = [torch.optim.SGD([p], lr=.1), torch.optim.SGD([q], lr=.1)]
        scaler = torch.cuda.amp.GradScaler(init_scale=128.0)
        scaler.scale(.5 * (p.square().sum() + q.square().sum())).backward()
        report = atomic_amp_step(opts, scaler, [("p", p), ("q", q)], max_norm=2.0)
        self.assertAlmostEqual(report["global_grad_norm"], 5.0, places=10)
        self.assertAlmostEqual(float(p), 2.88, places=6)
        self.assertAlmostEqual(float(q), 3.84, places=6)

    def test_real_amp_huge_finite_gradient_norm_and_step(self):
        p, q, opts, scaler, named = setup_pair("cuda", True, init_scale=1.0)
        backward_pair(p, q, opts, scaler)
        p.grad.fill_(3e38)
        q.grad.fill_(3e38)
        expected_norm = 2.0 * float(p.grad[0])
        report = atomic_amp_step(opts, scaler, named, max_norm=10.0)
        self.assertTrue(report["did_step"])
        self.assertAlmostEqual(report["global_grad_norm"] / expected_norm, 1.0, places=12)
        clipped_norm = math.sqrt(float(p.grad.double().square().sum() + q.grad.double().square().sum()))
        self.assertAlmostEqual(clipped_norm, 10.0, places=5)
        self.assertTrue(torch.isfinite(p).all() and torch.isfinite(q).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
