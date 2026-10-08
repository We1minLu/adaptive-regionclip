"""CPU checks of H2FA axes, routing, gradients and detached EMA soft targets."""
import json
import math
import unittest

import torch
import torch.nn.functional as F

from image_consistency import (
    PROBABILITY_EPS, h2fa_aggregate, h2fa_group_diagnostics, image_consistency,
)


def reference(logits, objectness):
    # Explicit scalar reference, independent of the vectorized scatter/softmax.
    logits = logits.detach().double()
    objectness = objectness.detach().double()
    categories = logits.argmax(dim=1).tolist()
    rows = logits.exp() / logits.exp().sum(dim=1, keepdim=True)
    result = []
    for category in range(8):
        raw = [float(objectness[n]) if categories[n] == category else 0.
               for n in range(len(logits))]
        denominator = sum(math.exp(value) for value in raw)
        result.append(sum(float(rows[n, category]) * math.exp(raw[n]) / denominator
                          for n in range(len(logits))))
    return torch.tensor(result)


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([
            [3., .4, -.5, .1, -.9, .2, .3, -.2],
            [.1, 2., -.4, .3, -.5, .5, .7, -.8],
            [1., -.2, .5, .4, -.3, -.6, .2, -.1],
            [-.3, .6, 1.5, .2, -.1, .4, -.7, .1],
        ])
        self.objectness = torch.tensor([2., -1.5, .4, .8])

    def test_matches_explicit_reference(self):
        result = h2fa_aggregate(self.logits, self.objectness)
        self.assertEqual(result.shape, (8,))
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.allclose(result, reference(self.logits, self.objectness), atol=1e-7))
        self.assertTrue(bool(((result >= 0) & (result <= 1)).all()))
        self.assertFalse(torch.allclose(result.sum(), torch.tensor(1.)))

    def test_permutation_invariant(self):
        order = torch.tensor([2, 0, 3, 1])
        self.assertTrue(torch.allclose(h2fa_aggregate(self.logits, self.objectness),
            h2fa_aggregate(self.logits[order], self.objectness[order]), atol=1e-7))

    def test_unassigned_routes_are_zero_logits_not_masked(self):
        actual = h2fa_aggregate(self.logits, self.objectness)
        # No proposal is routed to class 7: its weights must be uniform, not zero.
        self.assertAlmostEqual(float(actual[7]), float(self.logits.softmax(1)[:, 7].mean()), places=7)

    def test_single_proposal_has_no_objectness_selection_effect(self):
        actual = h2fa_aggregate(self.logits[:1], torch.tensor([-10.]))
        self.assertTrue(torch.allclose(actual, self.logits[:1].softmax(1)[0]))

    def test_float32_even_for_half_inputs(self):
        result = h2fa_aggregate(self.logits.half(), self.objectness.half())
        self.assertEqual(result.dtype, torch.float32)
        expected = h2fa_aggregate(self.logits.half().float(), self.objectness.half().float())
        self.assertTrue(torch.equal(result, expected))

    def test_many_extreme_proposals_remain_in_probability_domain(self):
        for count in (500, 1000):
            for pattern in ("uniform", "varied"):
                with self.subTest(count=count, pattern=pattern):
                    logits = torch.full((count, 8), -100.)
                    logits[:, 0] = 100.
                    objectness = (torch.zeros(count) if pattern == "uniform"
                                  else torch.linspace(-30., 30., count))
                    probabilities = h2fa_aggregate(logits, objectness)
                    self.assertTrue(bool(((probabilities >= 0) & (probabilities <= 1)).all()))
                    loss = image_consistency(probabilities, probabilities)
                    self.assertTrue(bool(torch.isfinite(loss)))
                    diagnostic = h2fa_group_diagnostics(logits, objectness, 300)
                    self.assertLessEqual(max(diagnostic["probabilities"]), 1.)

    def test_raw_objectness_and_roi_gradients_match_formula(self):
        logits = self.logits.clone().requires_grad_()
        objectness = self.objectness.clone().requires_grad_()
        probabilities = h2fa_aggregate(logits, objectness)
        category = 0
        probabilities[category].backward()
        q = logits.detach().softmax(1)
        route = logits.detach().argmax(1)
        mask = (route == category).float()
        weight = (objectness.detach() * mask).softmax(0)
        expected_objectness = mask * weight * (q[:, category] - probabilities[category].detach())
        expected_roi = -weight[:, None] * q[:, category, None] * q
        expected_roi[:, category] += weight * q[:, category]
        self.assertTrue(torch.allclose(objectness.grad, expected_objectness, atol=1e-7))
        self.assertTrue(torch.allclose(logits.grad, expected_roi, atol=1e-7))
        self.assertGreater(float(objectness.grad.abs().sum()), 0.)
        self.assertTrue(torch.equal(objectness.grad[route != category], torch.zeros(2)))

    def test_diagnostics_reconstruct_total_and_are_serializable(self):
        diagnostic = h2fa_group_diagnostics(self.logits.requires_grad_(), self.objectness, 2)
        json.dumps(diagnostic, allow_nan=False)
        base = diagnostic["base"]
        supplement = diagnostic["supplement"]
        total = torch.tensor(base["contribution"]) + torch.tensor(supplement["contribution"])
        weights = torch.tensor(base["weight"]) + torch.tensor(supplement["weight"])
        self.assertTrue(torch.allclose(total, torch.tensor(diagnostic["probabilities"]), atol=1e-7))
        self.assertTrue(torch.allclose(weights, torch.ones(8), atol=1e-7))
        self.assertEqual(base["count"], 2)
        self.assertEqual(base["objectness_min"], -1.5)
        empty = h2fa_group_diagnostics(self.logits, self.objectness, 4)["supplement"]
        self.assertEqual(empty["count"], 0)
        self.assertIsNone(empty["objectness_mean"])

    def test_aggregation_rejects_malformed_or_nonfinite_inputs(self):
        cases = [
            (torch.empty(0, 8), torch.empty(0)),
            (torch.zeros(2, 9), torch.zeros(2)),
            (torch.zeros(2, 8), torch.zeros(2, 1)),
            (torch.zeros(2, 8), torch.zeros(1)),
            (torch.zeros(2, 8, dtype=torch.long), torch.zeros(2)),
            (torch.full((2, 8), float("nan")), torch.zeros(2)),
            (torch.zeros(2, 8), torch.tensor([float("inf"), 0.])),
            (torch.full((2, 8), 1e300, dtype=torch.double), torch.zeros(2)),
        ]
        for logits, objectness in cases:
            with self.subTest(shape=logits.shape, objectness_shape=objectness.shape):
                with self.assertRaises(ValueError):
                    h2fa_aggregate(logits, objectness)
        for base_count in (-1, 5, True, 1.5):
            with self.assertRaises(ValueError):
                h2fa_group_diagnostics(self.logits, self.objectness, base_count)


class ConsistencyTests(unittest.TestCase):
    def test_matches_mean_soft_bce_for_batch(self):
        student = torch.linspace(.1, .9, 16).reshape(2, 8).requires_grad_()
        teacher = torch.linspace(.8, .2, 16).reshape(2, 8).requires_grad_()
        actual = image_consistency(student, teacher)
        expected = F.binary_cross_entropy(student, teacher.detach(), reduction="mean")
        self.assertTrue(torch.equal(actual, expected))
        actual.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(student.grad.abs().sum()), 0.)

    def test_teacher_roi_and_objectness_receive_no_consistency_gradient(self):
        torch.manual_seed(13)
        student_roi = torch.randn(5, 8, requires_grad=True)
        student_obj = torch.randn(5, requires_grad=True)
        teacher_roi = torch.randn(7, 8, requires_grad=True)
        teacher_obj = torch.randn(7, requires_grad=True)
        loss = image_consistency(h2fa_aggregate(student_roi, student_obj),
                                 h2fa_aggregate(teacher_roi, teacher_obj))
        loss.backward()
        self.assertGreater(float(student_roi.grad.abs().sum()), 0.)
        self.assertGreater(float(student_obj.grad.abs().sum()), 0.)
        self.assertIsNone(teacher_roi.grad)
        self.assertIsNone(teacher_obj.grad)

    def test_equal_soft_predictions_have_entropy_loss_and_zero_gradient(self):
        student = torch.linspace(.1, .8, 8).requires_grad_()
        teacher = student.detach().clone().requires_grad_()
        loss = image_consistency(student, teacher)
        entropy = -(teacher.detach() * teacher.detach().log()
                    + (1 - teacher.detach()) * (1 - teacher.detach()).log()).mean()
        self.assertGreater(float(loss), 0.)
        self.assertTrue(torch.allclose(loss, entropy))
        loss.backward()
        self.assertTrue(torch.equal(student.grad, torch.zeros(8)))
        self.assertIsNone(teacher.grad)

    def test_extreme_probabilities_are_finite_and_clamped_in_float32(self):
        student = torch.tensor([0., 1.] * 4, dtype=torch.float16, requires_grad=True)
        teacher = 1 - student.detach()
        loss = image_consistency(student, teacher)
        expected = F.binary_cross_entropy(student.float().clamp(PROBABILITY_EPS, 1 - PROBABILITY_EPS),
                                           teacher.float())
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(torch.equal(loss, expected))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(student.grad).all()))

    def test_probability_validation(self):
        correct = torch.full((8,), .5)
        for malformed in (torch.ones(7), torch.ones(1, 8), torch.full((8,), -0.1),
                          torch.full((8,), 1.1), torch.full((8,), float("nan")),
                          torch.full((8,), float("inf")), torch.ones(8, dtype=torch.long)):
            for left, right in ((malformed, correct), (correct, malformed)):
                with self.assertRaises(ValueError):
                    image_consistency(left, right)
        for invalid in (torch.empty(0, 8), torch.ones(1, 1, 8)):
            with self.assertRaises(ValueError):
                image_consistency(invalid, invalid)


if __name__ == "__main__":
    unittest.main()
