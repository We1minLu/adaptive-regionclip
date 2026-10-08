"""CPU integration checks for the actual EMA model paths with tiny ROI fixtures."""
import contextlib
import copy
import sys
from types import MethodType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from checkpoint_identity import REQUIRED
from image_consistency import h2fa_aggregate
from model import EMA_VERSION, FormalVGSDA, FixedSemanticClassifier, IMAGE_SIZE
from search_core import flatten_outputs, select_candidates


class Boxes:
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, item):
        return Boxes(self.tensor[item])

    def to(self, device):
        return Boxes(self.tensor.to(device))


class Instances:
    def __init__(self, image_size=IMAGE_SIZE):
        self.image_size = image_size

    def __len__(self):
        value = self.proposal_boxes if hasattr(self, "proposal_boxes") else self.gt_boxes
        return len(value)

    def __getitem__(self, item):
        result = Instances(self.image_size)
        for key, value in self.__dict__.items():
            if key != "image_size":
                setattr(result, key, value[item])
        return result

    def to(self, device):
        result = Instances(self.image_size)
        for key, value in self.__dict__.items():
            if key != "image_size":
                setattr(result, key, value.to(device))
        return result


def structure_patch():
    module = ModuleType("detectron2.structures")
    module.Boxes, module.Instances = Boxes, Instances
    return patch.dict(sys.modules, {"detectron2.structures": module})


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([.7, 1.1, 1.4]))
        self.raw_scale = nn.Parameter(torch.tensor([1.2, .8, 1.1]))
        self.register_buffer("counter", torch.tensor(0, dtype=torch.long))

    def forward(self, images):
        value = images.mean(dim=(2, 3), keepdim=True) * self.scale[None, :, None, None]
        return {"res3": value.expand(-1, -1, 2, 2), "res4": value.expand(-1, -1, 2, 2)}


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_score = nn.Linear(3, 8, bias=False).requires_grad_(False)
        self.cls_bg_score = nn.Linear(3, 1, bias=False).requires_grad_(False)
        self.bbox_pred = nn.Linear(3, 4)
        self.temperature, self.use_bias = .3, False
        self.loss_batches = []

    def losses(self, predictions, proposals, is_source):
        self.loss_batches.append((is_source, len(proposals), [len(p) for p in proposals]))
        return {"loss_cls": predictions[0].square().mean(),
                "loss_box_reg": predictions[1].square().mean()}


class TinyROIHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_predictor = TinyPredictor()
        self.sample_batches = []

    def label_and_sample_proposals(self, proposals, targets):
        self.sample_batches.append((len(proposals), len(targets)))
        for proposal, target in zip(proposals, targets):
            proposal.gt_classes = target.gt_classes[0].expand(len(proposal))
        return proposals


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.roi_heads = TinyROIHeads()
        self.offline_backbone = nn.Identity()
        self.offline_proposal_generator = nn.Identity()

    def preprocess_image(self, inputs):
        return SimpleNamespace(tensor=torch.stack([r["image"].mean((1, 2), keepdim=True) for r in inputs]))


class TinySearch(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral3 = nn.Conv2d(3, 3, 1, bias=False)
        self.lateral4 = nn.Conv2d(3, 3, 1, bias=False)
        self.obj = nn.Parameter(torch.tensor([4., 3., 2., 1., .5]))
        self.miss = nn.Parameter(torch.tensor([1., 3., 1., 2., 1.]))
        self.delta = nn.Parameter(torch.zeros(5, 4))
        self.register_buffer("anchor_boxes", torch.tensor([
            [0., 0., 8., 8.], [12., 0., 20., 8.], [24., 0., 32., 8.],
            [36., 0., 44., 8.], [48., 0., 56., 8.]]))

    def forward(self, res3, res4, maps):
        value = (self.lateral3(res3) + self.lateral4(res4)).mean((1, 2, 3))
        return [{"obj": self.obj[None, :] + value[:, None],
                 "miss": self.miss[None, :].expand(len(value), -1),
                 "delta": self.delta[None, :, :].expand(len(value), -1, -1)}]

    def anchors(self, outputs):
        return self.anchor_boxes


class TinyDomain(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(.1))
        self.calls = []

    def forward(self, source, target, coefficient):
        self.calls.append((len(source["res3"]), len(target["res3"])))
        return self.value.square(), {}


class TinyConditional(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(.1))
        self.calls = []

    def forward(self, source, source_samples, target, target_samples, grl_coeff):
        self.calls.append((len(source_samples), len(target_samples), target_samples))
        return self.value.square(), {}


def fixture():
    torch.manual_seed(72)
    model = FormalVGSDA.__new__(FormalVGSDA)
    nn.Module.__init__(model)
    model.student = TinyStudent()
    model.search = TinySearch()
    model.teacher_backbone = copy.deepcopy(model.student.backbone).requires_grad_(False)
    model.teacher_search = copy.deepcopy(model.search).requires_grad_(False)
    model.teacher_bbox = copy.deepcopy(model.student.roi_heads.box_predictor.bbox_pred).requires_grad_(False)
    model.teacher_classifier = FixedSemanticClassifier(model.student.roi_heads.box_predictor)
    model.global_D, model.conditional_D = TinyDomain(), TinyConditional()
    model.image_consistency_enabled, model.target_image_labels_allowed = True, False
    model.version, model.ema_decay, model.ema_updates = EMA_VERSION, .9996, 0
    model.source_loss_weight = model.search_loss_weight = model.conditional_domain_weight = 1.
    model.image_consistency_weight, model.traditional_domain_weight = 1., .1
    model.conditional_grl_max = .05
    model.teacher_roi_chunk = model.train_roi_chunk = model.inference_roi_chunk = 2
    model.checkpoint_res5 = False
    model.config, model.initialization_audit = {"image_consistency_enabled": True}, {"fixture": True}
    model.source_identity = {key: {"path": "/fixture/" + key, "sha256": "a" * 64} for key in REQUIRED}
    model.base_calls, model.bundle_calls = [], []

    def raw(self, features, boxes, backbone):
        value = features.mean((2, 3)).expand(len(boxes), -1)
        offsets = boxes[:, :1] / 40. * boxes.new_tensor([[.7, -.3, .2]])
        return (value + offsets) * backbone.raw_scale

    def base(self, inputs):
        self.base_calls.append([float(r["image"].mean()) for r in inputs])
        result = []
        for r in inputs:
            shift = float(r["image"].mean())
            proposal = Instances()
            proposal.proposal_boxes = Boxes(torch.tensor([[0., 0., 8., 8.], [80., 0., 88., 8.]]))
            # Weak and strong images deliberately yield different proposal geometry.
            proposal.proposal_boxes.tensor[1] += shift
            proposal.objectness_logits = torch.tensor([1., 2.], requires_grad=True)
            result.append(proposal)
        return result

    def bundle(self, inputs, teacher_features=None):
        self.bundle_calls.append(([float(r["image"].mean()) for r in inputs], teacher_features))
        images = self.student.preprocess_image(inputs)
        features = self.student.backbone(images.tensor)
        proposals = self._base_proposals(inputs)
        semantics = self._teacher_semantics(images, proposals, teacher_features)
        return images, features, proposals, semantics, torch.zeros(len(inputs), 15, 2, 2)

    model._raw_roi_features = MethodType(raw, model)
    model._base_proposals = MethodType(base, model)
    model._bundle = MethodType(bundle, model)
    model._event_context = lambda: contextlib.nullcontext()
    model.train()
    return model


def record(strong, weak, source=False):
    result = {"image": torch.full((3, 1, 1), strong).expand((3,) + IMAGE_SIZE),
              "image_weak": torch.full((3, 1, 1), weak).expand((3,) + IMAGE_SIZE)}
    if source:
        instance = Instances()
        instance.gt_classes = torch.tensor([2])
        instance.gt_boxes = Boxes(torch.tensor([[12., 0., 20., 8.]]))
        result["instances"] = instance
    return result


def zero_search_loss(outputs, anchors, samples):
    value = outputs[0]["obj"].sum() * 0.
    return {"obj": value, "miss": value, "box": value}, {"fixture_source_count": len(samples)}


class IntegratedEMATests(unittest.TestCase):
    def test_actual_image_probability_path_uses_first_eight_logits_and_image_splits(self):
        proposals = [Instances(), Instances()]
        for p, count in zip(proposals, (2, 3)):
            p.proposal_boxes = Boxes(torch.zeros(count, 4))
            p.aggregation_objectness_logits = torch.linspace(-1, 2, count)
        torch.manual_seed(3)
        logits = torch.randn(5, 18)
        expected = torch.stack([h2fa_aggregate(logits[:2, :8], proposals[0].aggregation_objectness_logits),
                                h2fa_aggregate(logits[2:, :8], proposals[1].aggregation_objectness_logits)])
        logits[:, 8:] = 1e5
        actual = FormalVGSDA._image_probabilities((logits,), proposals)
        self.assertTrue(torch.equal(actual, expected))

    def test_real_selection_returns_surviving_raw_anchor_indices_and_gradients(self):
        model = fixture()
        inputs = [record(.4, .7)]
        features = model.student.backbone(model.student.preprocess_image(inputs).tensor)
        outputs = model.search(features["res3"], features["res4"], torch.zeros(1, 15, 2, 2))
        base = model._base_proposals(inputs)
        samples = [{"boxes": base[0].proposal_boxes.tensor, "image_size": IMAGE_SIZE}]
        selected = select_candidates(outputs, model.search.anchors(outputs), samples[0], return_indices=True)
        self.assertNotIn(0, selected[3].tolist())  # Exact duplicate of the first RPN proposal.
        with structure_patch():
            merged, _ = model._merged(outputs, model.search.anchors(outputs), samples, base)
        raw = flatten_outputs(outputs)["obj"][0, selected[3]]
        self.assertTrue(torch.equal(merged[0].aggregation_objectness_logits[2:], raw))
        self.assertFalse(merged[0].proposal_boxes.tensor.requires_grad)
        self.assertFalse(merged[0].objectness_logits.requires_grad)
        merged[0].aggregation_objectness_logits.sum().backward()
        expected_gradient = torch.zeros(5)
        expected_gradient[selected[3]] = 1.
        self.assertTrue(torch.equal(model.search.obj.grad, expected_gradient))
        self.assertIsNone(model.search.miss.grad)
        self.assertIsNone(model.search.delta.grad)
        self.assertIsNone(base[0].objectness_logits.grad)

    def test_actual_training_routes_two_source_one_target_and_image_only_target_gradients(self):
        model = fixture()
        source = [record(.2, .3, True), record(.4, .5, True)]
        target = [record(.6, .9)]
        with structure_patch(), patch("model.training_loss", side_effect=zero_search_loss):
            losses, stats = model.training_losses(source, target, .2)
        self.assertEqual(set(losses), {"loss_source_cls", "loss_source_box", "loss_search_obj",
            "loss_search_miss", "loss_search_box", "loss_traditional_domain", "loss_conditional_domain",
            "loss_target_image_consistency"})
        self.assertEqual(model.student.roi_heads.sample_batches, [(2, 2)])
        self.assertEqual(model.student.roi_heads.box_predictor.loss_batches[0][:2], (True, 2))
        self.assertEqual(model.global_D.calls, [(2, 1)])
        self.assertEqual(model.conditional_D.calls[0][:2], (2, 1))
        target_sample = model.conditional_D.calls[0][2][0]
        self.assertNotIn("gt", target_sample)
        self.assertIsNone(target_sample["image_labels"])
        self.assertTrue(torch.allclose(torch.tensor(model.base_calls[0]), torch.tensor([.2, .4, .6])))
        self.assertTrue(torch.allclose(torch.tensor(model.base_calls[1]), torch.tensor([.9])))
        self.assertFalse(model.bundle_calls[0][1]["res4"].requires_grad)
        self.assertEqual(stats["search"]["fixture_source_count"], 2)
        self.assertEqual(len(stats["image_consistency"]["student_image_probabilities"]), 1)
        self.assertFalse(stats["image_consistency"]["target_pseudo_box_losses"])
        self.assertFalse(stats["target_image_labels_used"])
        losses["loss_target_image_consistency"].backward()
        self.assertGreater(float(model.student.backbone.scale.grad.abs().sum()), 0.)
        self.assertGreater(float(model.student.backbone.raw_scale.grad.abs().sum()), 0.)
        self.assertGreater(float(model.search.obj.grad.abs().sum()), 0.)
        self.assertIsNone(model.search.delta.grad)
        self.assertIsNone(model.search.miss.grad)
        self.assertIsNone(model.student.roi_heads.box_predictor.bbox_pred.weight.grad)
        for teacher in (model.teacher_backbone, model.teacher_search, model.teacher_bbox, model.teacher_classifier):
            self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_ema_checkpoint_roundtrip_restores_teacher_and_update_count(self):
        model = fixture()
        with torch.no_grad():
            model.student.backbone.scale.add_(2.)
            model.search.obj.sub_(1.)
            model.student.roi_heads.box_predictor.bbox_pred.weight.add_(.5)
            model.student.backbone.counter.fill_(9)
        old = model.teacher_backbone.scale.detach().clone()
        model.update_teacher()
        self.assertTrue(torch.allclose(model.teacher_backbone.scale,
            old * .9996 + model.student.backbone.scale.detach() * .0004))
        self.assertEqual(model.ema_updates, 1)
        self.assertEqual(int(model.teacher_backbone.counter), 9)
        state = model.checkpoint_state()
        restored = fixture()
        restored.load_checkpoint_state(state)
        self.assertEqual(restored.ema_updates, 1)
        restored_state = restored.checkpoint_state()
        for key in ("student_backbone", "student_bbox", "search", "teacher_backbone", "teacher_search", "teacher_bbox"):
            for name in state[key]:
                self.assertTrue(torch.equal(state[key][name], restored_state[key][name]), (key, name))
        self.assertFalse(restored.teacher_backbone.training)
        self.assertFalse(restored.teacher_search.training)
        self.assertFalse(restored.teacher_bbox.training)
        self.assertTrue(all(not p.requires_grad for p in restored.teacher_backbone.parameters()))
        with torch.no_grad():
            restored.teacher_search.obj.add_(100.)
        self.assertFalse(torch.equal(state["teacher_search"]["obj"], restored.teacher_search.obj))
        invalid = dict(state, ema_updates=-1)
        with self.assertRaises(ValueError):
            fixture().load_checkpoint_state(invalid)
        missing = dict(state)
        del missing["teacher_search"]
        with self.assertRaises(KeyError):
            fixture().load_checkpoint_state(missing)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
