"""CPU regression checks for the explicit target image-label ablation boundary."""
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from domain_core import ConditionalDomainAdapter, select_domain_rois
from model import FormalVGSDA, IMAGE_SIZE, target_image_labels_allowed


class FakeInstances:
    def __init__(self, classes):
        self.gt_classes = torch.tensor(classes, dtype=torch.long)
        self.gt_boxes = SimpleNamespace(tensor=torch.ones(len(classes), 4))

    def to(self, device):
        return self


def sample_only_model(allowed):
    # _samples and input guards need no RegionCLIP/CUDA construction.
    model = FormalVGSDA.__new__(FormalVGSDA)
    nn.Module.__init__(model)
    model.student = SimpleNamespace(backbone=nn.Linear(1, 1))
    model.target_image_labels_allowed = allowed
    return model


def proposal():
    return SimpleNamespace(proposal_boxes=SimpleNamespace(tensor=torch.tensor([[0., 0., 16., 16.]])),
                           objectness_logits=torch.tensor([2.]))


def input_image():
    return {"image": torch.zeros(3, 1, 1).expand((3,) + IMAGE_SIZE)}


class DomainSelectionTests(unittest.TestCase):
    def setUp(self):
        self.boxes = torch.tensor([[0., 0., 16., 16.], [2., 2., 18., 18.],
            [40., 40., 90., 90.], [120., 100., 250., 250.],
            [300., 0., 340., 40.], [350., 0., 390., 40.], [400., 0., 440., 40.]])
        self.probs = torch.zeros(7, 9)
        for i, (label, confidence) in enumerate(((2, .8), (2, .85), (7, .9),
                (0, .7), (1, .69), (3, .95), (8, .9))):
            self.probs[i, label] = confidence
            self.probs[i, 8 if label != 8 else 2] = 1 - confidence
        self.logits = torch.tensor([2., 2., 2., 0., 2., -.1, 2.])

    def test_none_is_exact_all_classes_selection_without_fake_presence(self):
        # Includes confidence and RPN boundaries, background, groups and budget.
        for budget in (1, 3, 64):
            with self.subTest(budget=budget):
                unlabeled = select_domain_rois(self.boxes, self.probs, self.logits, None, max_rois=budget)
                all_present = select_domain_rois(self.boxes, self.probs, self.logits,
                                                torch.ones(8), max_rois=budget)
                for key in ("boxes", "indices", "groups", "confidence"):
                    self.assertTrue(torch.equal(unlabeled[key], all_present[key]), key)
                a, b = copy.deepcopy(unlabeled["stats"]), copy.deepcopy(all_present["stats"])
                self.assertFalse(a.pop("presence_filter_enabled"))
                self.assertTrue(b.pop("presence_filter_enabled"))
                self.assertEqual(a, b)
                self.assertEqual(a["reliable_before_presence"], 4)
                self.assertEqual(a["presence_rejected"], 0)

    def test_original_presence_filter_rejects_without_relabeling(self):
        presence = torch.zeros(8)
        presence[2] = 1
        selected = select_domain_rois(self.boxes, self.probs, self.logits, presence)
        self.assertEqual(selected["indices"].tolist(), [1, 0])
        self.assertEqual(selected["groups"].tolist(), [13, 13])
        self.assertEqual(selected["stats"]["presence_rejected"], 2)
        self.assertTrue(selected["stats"]["presence_filter_enabled"])
        empty = select_domain_rois(self.boxes, self.probs, self.logits, torch.zeros(8))
        self.assertEqual(len(empty["indices"]), 0)

    def test_empty_roi_audit_retains_explicit_filter_mode(self):
        for presence, expected in ((None, False), (torch.zeros(8), True)):
            result = select_domain_rois(torch.empty(0, 4), torch.empty(0, 9), torch.empty(0), presence)
            self.assertEqual(result["stats"]["presence_filter_enabled"], expected)
            self.assertEqual(result["stats"]["selected"], 0)

    def test_presence_shape_and_binary_values_still_validated(self):
        for malformed in (torch.ones(7), torch.ones(9), torch.full((8,), .5), torch.full((8,), float("nan"))):
            with self.subTest(shape=malformed.shape):
                with self.assertRaises(ValueError):
                    select_domain_rois(self.boxes, self.probs, self.logits, malformed)

    def test_missing_sample_key_fails_instead_of_implicitly_disabling_filter(self):
        sample = {"boxes": self.boxes, "probs": self.probs, "rpn_logits": self.logits}
        with self.assertRaises(KeyError):
            ConditionalDomainAdapter()._prepare({}, [sample])

    def test_uda_adapter_audits_filter_mode_and_keeps_domain_gradients(self):
        presence = torch.zeros(8)
        presence[2] = 1
        teacher = self.probs.clone().requires_grad_()
        common = {"boxes": self.boxes, "probs": teacher, "rpn_logits": self.logits}
        source = dict(common, image_labels=presence)
        target = dict(common, image_labels=None)
        sf = {"p3": torch.randn(1, 4, 64, 64, requires_grad=True),
              "p4": torch.randn(1, 4, 32, 32, requires_grad=True)}
        tf = {key: torch.randn_like(value, requires_grad=True) for key, value in sf.items()}
        adapter = ConditionalDomainAdapter(channels=4)
        loss, stats = adapter(sf, [source], tf, [target], grl_coeff=.05)
        loss.backward()
        self.assertEqual(stats["source_rois"], 2)
        self.assertEqual(stats["target_rois"], 4)
        self.assertTrue(stats["source_presence_filter_enabled"])
        self.assertFalse(stats["target_presence_filter_enabled"])
        self.assertFalse(stats["target_image_labels_used"])
        self.assertIsNone(teacher.grad)
        for value in list(sf.values()) + list(tf.values()):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
        self.assertTrue(all(parameter.grad is not None for parameter in adapter.parameters()))


class ModelBoundaryTests(unittest.TestCase):
    def test_default_mode_is_original_labeled_protocol(self):
        self.assertIs(target_image_labels_allowed({}), True)
        for allowed in (True, False):
            self.assertIs(target_image_labels_allowed({"target_image_labels_allowed": allowed}), allowed)

    def test_non_boolean_configuration_rejected_before_model_construction(self):
        for value in ("false", "true", 0, 1, None, [], {}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be a boolean"):
                    FormalVGSDA({"target_image_labels_allowed": value})

    def test_uda_inputs_reject_any_image_label_key(self):
        for value in (None, torch.ones(8), [0] * 8):
            record = dict(input_image(), image_labels=value)
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must not contain image_labels"):
                    FormalVGSDA._validate_inputs([record], False, False)
                with self.assertRaisesRegex(ValueError, "must not contain image_labels"):
                    sample_only_model(False)._samples([record], [proposal()], [torch.zeros(1, 9)], 0)

    def test_uda_inputs_allow_images_but_forbid_target_boxes(self):
        FormalVGSDA._validate_inputs([input_image()], False, False)
        for key in ("instances", "annotations", "gt", "gt_boxes", "gt_classes", "labels", "targets"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "region annotations"):
                    FormalVGSDA._validate_inputs([dict(input_image(), **{key: None})], False, False)

    def test_uda_target_sample_contains_explicit_none(self):
        records = sample_only_model(False)._samples([input_image()], [proposal()], [torch.zeros(1, 9)], 0)
        self.assertIn("image_labels", records[0])
        self.assertIsNone(records[0]["image_labels"])
        self.assertNotIn("gt", records[0])
        self.assertNotIn("labels", records[0])

    def test_original_target_mode_still_requires_eight_binary_labels(self):
        model = sample_only_model(True)
        with self.assertRaisesRegex(ValueError, "explicitly permitted"):
            model._samples([input_image()], [proposal()], [torch.zeros(1, 9)], 0)
        expected = torch.tensor([0, 0, 1, 0, 0, 0, 0, 1])
        records = model._samples([dict(input_image(), image_labels=expected)], [proposal()], [torch.zeros(1, 9)], 0)
        self.assertTrue(torch.equal(records[0]["image_labels"], expected))
        with self.assertRaisesRegex(ValueError, "eight binary"):
            model._samples([dict(input_image(), image_labels=[1] * 7)], [proposal()], [torch.zeros(1, 9)], 0)

    def test_source_presence_always_comes_from_gt_with_unchanged_values(self):
        class SourceRecord(dict):
            def __getitem__(self, key):
                if key == "image_labels":
                    raise AssertionError("Source must derive presence from source GT")
                return super().__getitem__(key)
        record = SourceRecord(input_image(), instances=FakeInstances([2, 2, 7]), image_labels=None)
        expected = torch.tensor([0., 0., 1., 0., 0., 0., 0., 1.])
        for allowed in (True, False):
            with self.subTest(allowed=allowed):
                FormalVGSDA._validate_inputs([record], True, allowed)
                source = sample_only_model(allowed)._samples([record], [proposal()], [torch.zeros(1, 9)], 1)[0]
                self.assertTrue(torch.equal(source["image_labels"], expected))
                self.assertTrue(torch.equal(source["labels"], torch.tensor([2, 2, 7])))
                self.assertIs(source["gt"], record["instances"].gt_boxes.tensor)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
