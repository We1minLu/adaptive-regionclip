"""CPU checks for mathematical behavior, teacher detachment and RPN gradients."""
import importlib.util
from pathlib import Path
import unittest

import torch


spec = importlib.util.spec_from_file_location("dest_core", Path(__file__).resolve().parents[1] / "core.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class RegionTests(unittest.TestCase):
    def test_default_window_count_coverage_and_duplicates(self):
        windows = core.collect_windows((1024, 2048), [.125, .25, .5, 1., .25])
        self.assertEqual(windows.shape, (284, 4))
        self.assertEqual(len(torch.unique(windows, dim=0)), 284)
        torch.testing.assert_close(windows[-1], torch.tensor([0., 0., 2048., 1024.]))
        self.assertTrue(bool((windows[:, :2] >= 0).all()))
        self.assertTrue(bool((windows[:, 2] <= 2048).all()))
        self.assertTrue(bool((windows[:, 3] <= 1024).all()))

    def test_fractional_grid_clips_first_boundary_crossing(self):
        windows = core.collect_windows((10, 20), [.3])
        self.assertEqual(len(windows), 36)
        torch.testing.assert_close(windows[-1], torch.tensor([15., 7.5, 20., 10.]))
        self.assertEqual(core.collect_windows((10, 20), []).shape, (0, 4))
        with self.assertRaises(ValueError):
            core.collect_windows((10, 20), [0])

    def test_kernel_matches_normalized_intersection_and_is_psd(self):
        boxes = torch.tensor([[0., 0., 2., 2.], [1., 0., 3., 2.], [4., 0., 5., 1.]], dtype=torch.double)
        kernel = core.overlap_kernel(boxes)
        expected = torch.tensor([[1., .5, 0.], [.5, 1., 0.], [0., 0., 1.]], dtype=torch.double)
        torch.testing.assert_close(kernel, expected)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(kernel).min()), -1e-12)


class MRSATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_zero_rho_equals_max_full_softmax_and_is_detached(self):
        boxes = torch.tensor([[0., 0., 1., 1.], [2., 0., 3., 1.]], requires_grad=True)
        logits = torch.tensor([[3., 1., 2.], [1., 2., 4.]], requires_grad=True)
        target, stats = core.mrsa_aggregate(boxes, logits, rho=0)
        expected = logits.softmax(dim=1)[:, :-1].max(dim=0).values
        torch.testing.assert_close(target, expected)
        self.assertFalse(target.requires_grad)
        self.assertEqual(stats["iterations"], [0, 0])
        self.assertEqual(stats["max_duality_gap"], 0.)

    def test_two_region_solution_matches_independent_dense_search(self):
        support = torch.tensor([[2., -1.], [1.5, .1]], dtype=torch.double)
        kernel = torch.tensor([[1., .2], [.2, 1.]], dtype=torch.double)
        q, stats = core.robust_support(support, kernel, rho=1.3, max_iter=30, tol=1e-8, return_weights=True)
        t = torch.linspace(0., 1., 100001, dtype=torch.double)
        w = torch.stack((t, 1.-t), dim=1)
        dense = w @ support - 1.3 * ((w @ kernel) * w).sum(dim=1).sqrt()[:, None]
        torch.testing.assert_close(q, dense.max(dim=0).values, atol=1e-7, rtol=0.)
        weights = torch.tensor(stats["weights"], dtype=torch.double)
        torch.testing.assert_close(weights.sum(dim=0), torch.ones(2, dtype=torch.double))
        self.assertTrue(bool((weights >= 0).all()))
        self.assertLess(stats["max_duality_gap"], 1e-7)

    def test_identical_regions_give_shifted_maximum(self):
        support = torch.randn(7, 3, dtype=torch.double)
        q, stats = core.robust_support(support, torch.ones(7, 7, dtype=torch.double), rho=.7)
        torch.testing.assert_close(q, support.max(dim=0).values - .7)
        self.assertEqual(stats["iterations"], [0, 0, 0])

    def test_exact_duplicate_regions_do_not_add_support(self):
        boxes = torch.tensor([[0., 0., 2., 2.], [1., 0., 3., 2.], [0., 3., 2., 5.]], dtype=torch.double)
        logits = torch.tensor([[1., 1.2, 0.], [1.1, 1.15, .1], [.9, 1.25, 0.]], dtype=torch.double)
        kwargs = dict(rho=1.5, max_iter=4000, tol=1e-7)
        reference, stats = core.mrsa_aggregate(boxes, logits, **kwargs)
        copied, _ = core.mrsa_aggregate(torch.cat((boxes, boxes[:1].repeat(5, 1))),
                                         torch.cat((logits, logits[:1].repeat(5, 1))), **kwargs)
        torch.testing.assert_close(reference, copied, atol=1e-7, rtol=0.)
        self.assertTrue(all(stats["converged"]))

    def test_independent_equal_evidence_spreads_weights(self):
        support = torch.zeros(4, 2, dtype=torch.double)
        q, stats = core.robust_support(support, torch.eye(4, dtype=torch.double),
                                      rho=1., max_iter=2000, tol=1e-7, return_weights=True)
        torch.testing.assert_close(q, torch.tensor([-.5, -.5], dtype=torch.double), atol=1e-7, rtol=0.)
        torch.testing.assert_close(torch.tensor(stats["weights"]), torch.full((4, 2), .25), atol=2e-4, rtol=0.)

    def test_empty_regions_excluded_and_nonfinite_rejected(self):
        boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 0., 2.]])
        target, stats = core.mrsa_aggregate(boxes, torch.tensor([[1., 0.], [100., 0.]]), rho=0)
        torch.testing.assert_close(target, torch.tensor([1.]).sigmoid())
        self.assertEqual(stats["excluded_empty_regions"], 1)
        with self.assertRaises(ValueError):
            core.mrsa_aggregate(boxes[:0], torch.empty((0, 2)))
        with self.assertRaises(ValueError):
            core.mrsa_aggregate(boxes[:1], torch.tensor([[float('nan'), 0.]]))

    def test_iteration_cap_reports_uncertainty_not_false_convergence(self):
        _, stats = core.robust_support(torch.zeros((20, 2)), torch.eye(20), max_iter=1, tol=1e-8)
        self.assertFalse(any(stats["converged"]))
        self.assertGreater(stats["max_duality_gap"], 1e-8)


class SpatialTests(unittest.TestCase):
    def test_only_winning_in_support_anchor_gets_gradient(self):
        logits = torch.tensor([-.5, 0., 5.], requires_grad=True)
        centers = torch.tensor([[1., 1.], [2., 2.], [9., 9.]], requires_grad=True)
        boxes = torch.tensor([[0., 0., 3., 3.]], requires_grad=True)
        loss, stats = core.relaxed_spatial_loss(logits, centers, boxes, (10, 10), gamma=1., tau=.7)
        torch.testing.assert_close(loss, torch.tensor(.2))
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([0., -.25, 0.]))
        self.assertIsNone(centers.grad)
        self.assertIsNone(boxes.grad)
        self.assertEqual(stats["active_fraction"], 1.)

    def test_sufficient_response_stops_loss_and_gradient(self):
        logits = torch.tensor([2., 8.], requires_grad=True)
        loss, stats = core.relaxed_spatial_loss(logits, torch.tensor([[1., 1.], [9., 9.]]),
                                               torch.tensor([[0., 0., 2., 2.]]), (10, 10), gamma=1., tau=.7)
        loss.backward()
        self.assertEqual(float(loss), 0.)
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        self.assertEqual(stats["active_supports"], 0)

    def test_empty_supports_zero_is_differentiable(self):
        for boxes in (torch.empty(0, 4), torch.tensor([[20., 20., 30., 30.]]),
                      torch.tensor([[5., 5., 6., 6.]])):
            logits = torch.tensor([0.], requires_grad=True)
            loss, stats = core.relaxed_spatial_loss(logits, torch.tensor([[1., 1.]]), boxes,
                                                   (10, 10), gamma=1., tau=.7)
            loss.backward()
            self.assertEqual(float(loss), 0.)
            self.assertEqual(float(logits.grad), 0.)
            self.assertEqual(stats["valid_supports"], 0)

    def test_expansion_allows_response_outside_original_box(self):
        logits = torch.tensor([2.], requires_grad=True)
        centers = torch.tensor([[3.5, 3.5]])
        boxes = torch.tensor([[4., 4., 6., 6.]])
        loss, stats = core.relaxed_spatial_loss(logits, centers, boxes, (10, 10), gamma=2., tau=.7)
        self.assertEqual(stats["valid_supports"], 1)
        self.assertEqual(float(loss), 0.)

    def test_only_valid_supports_enter_mean_and_shared_response_is_allowed(self):
        logits = torch.tensor([0.], requires_grad=True)
        boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 3., 3.], [6., 6., 7., 7.]])
        loss, stats = core.relaxed_spatial_loss(logits, torch.tensor([[1., 1.]]), boxes, (10, 10), gamma=1., tau=.7)
        loss.backward()
        torch.testing.assert_close(loss, torch.tensor(.2))
        torch.testing.assert_close(logits.grad, torch.tensor([-.25]))
        self.assertEqual(stats["valid_supports"], 2)

    def test_response_is_permutation_invariant_within_support(self):
        boxes = torch.tensor([[0., 0., 5., 5.]])
        centers = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])
        a, _ = core.relaxed_spatial_loss(torch.tensor([-.2, 0., -.7]), centers, boxes, (10, 10), gamma=1., tau=.8)
        b, _ = core.relaxed_spatial_loss(torch.tensor([-.7, -.2, 0.]), centers, boxes, (10, 10), gamma=1., tau=.8)
        torch.testing.assert_close(a, b)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
