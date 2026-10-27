"""CPU tests for live RPN scores, detached selection, and DeST integration.

The native Detectron2 parity test runs in the project's existing environment.
The full checkpoint smoke is opt-in via DEST_INTEGRATION_CONFIG; it performs
one small synthetic forward/backward, never an optimizer step or training run.
"""
import json
import os
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from .model import DeST, SpatialDomainDiscriminators, _Reverse, _legacy_helpers
from .rpn_support import flatten_rpn_outputs, _select_indices, run_rpn


class MiniBoxes:
    def __init__(self, tensor):
        self.tensor = tensor

    def clip(self, image_size):
        height, width = image_size
        self.tensor[:, 0::2].clamp_(0, width)
        self.tensor[:, 1::2].clamp_(0, height)

    def nonempty(self, threshold):
        size = self.tensor[:, 2:] - self.tensor[:, :2]
        return (size > threshold).all(dim=1)

    def __getitem__(self, indices):
        return type(self)(self.tensor[indices])


class MiniInstances:
    def __init__(self, image_size):
        self.image_size = image_size


def minimal_d2():
    structures = types.ModuleType("detectron2.structures")
    structures.Boxes, structures.Instances = MiniBoxes, MiniInstances
    layers = types.ModuleType("detectron2.layers")
    # Test boxes do not overlap, so this is the exact NMS outcome.
    layers.batched_nms = lambda boxes, scores, levels, threshold: scores.argsort(descending=True)
    return patch.dict(sys.modules, {"detectron2.structures": structures, "detectron2.layers": layers})


class ToyRPN(nn.Module):
    in_features = ["res4"]
    nms_thresh, min_box_size = .7, 0.
    pre_nms_topk = {False: 20, True: 20}

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([[[[.2, 1., -.8]]], [[[1.5, -.7, .5]]]]))
        self.deltas = nn.Parameter(torch.zeros(2, 4, 1, 3))
        self.anchor_generator = types.SimpleNamespace(box_dim=4)
        # Callable instance rather than the namespace's special method.
        class Anchors:
            box_dim = 4
            def __call__(self, features):
                return [MiniBoxes(torch.tensor([[0., 0., 5., 5.], [10., 0., 15., 5.], [20., 0., 25., 5.]]))]
        self.anchor_generator = Anchors()

    def rpn_head(self, features):
        return [self.logits], [self.deltas]

    def _decode_proposals(self, anchors, deltas):
        return [anchors[0].tensor[None] + deltas[0]]

    def label_and_sample_anchors(self, anchors, targets):
        self.source_count = len(targets)
        return [torch.ones(3, dtype=torch.long) for _ in targets], [anchors[0].tensor for _ in targets]

    def losses(self, anchors, scores, labels, deltas, boxes):
        self.loss_batch = scores[0].shape[0]
        return {"loss_rpn_cls": scores[0].square().mean(), "loss_rpn_loc": deltas[0].sum() * 0}


class ModelUnitTests(unittest.TestCase):
    def test_flatten_anchor_order(self):
        scores = torch.arange(12.).reshape(1, 2, 2, 3)
        deltas = torch.arange(48.).reshape(1, 8, 2, 3)
        flat_s, flat_d = flatten_rpn_outputs([scores], [deltas])
        expected_s = [scores[0, anchor, y, x] for y in range(2) for x in range(3) for anchor in range(2)]
        expected_d = [deltas[0, anchor*4:anchor*4+4, y, x] for y in range(2) for x in range(3) for anchor in range(2)]
        self.assertTrue(torch.equal(flat_s[0][0], torch.stack(expected_s)))
        self.assertTrue(torch.equal(flat_d[0][0], torch.stack(expected_d)))

    def test_live_objectness_detached_boxes_source_only_labels(self):
        rpn = ToyRPN()
        images = types.SimpleNamespace(image_sizes=[(10, 30), (10, 30)])
        with minimal_d2():
            proposals, losses, raw, centers = run_rpn(rpn, images, {"res4": torch.zeros(2, 1, 1, 3)},
                source_targets=[object()], proposal_budget=2)
        self.assertEqual(rpn.source_count, 1)
        self.assertEqual(rpn.loss_batch, 1)
        self.assertFalse(proposals[1].proposal_boxes.tensor.requires_grad)
        self.assertFalse(proposals[1].objectness_logits.requires_grad)
        self.assertTrue(proposals[1].aggregation_objectness_logits.requires_grad)
        self.assertEqual(proposals[1].anchor_indices.tolist(), [0, 2])
        self.assertEqual(centers.tolist(), [[2.5, 2.5], [12.5, 2.5], [22.5, 2.5]])
        source_grad = torch.autograd.grad(losses["loss_rpn_cls"], rpn.logits, retain_graph=True)[0]
        self.assertEqual(float(source_grad[1].abs().sum()), 0.)
        proposals[1].aggregation_objectness_logits.sum().backward()
        self.assertEqual(rpn.logits.grad[1].flatten().tolist(), [1., 0., 1.])
        self.assertIsNone(rpn.deltas.grad)

    def test_h2fa_propagates_to_roi_and_selected_objectness(self):
        aggregate, consistency, _ = _legacy_helpers()
        roi = torch.tensor([[2., 1., 0., -1., 0., 0., 0., 0.],
                            [.3, 2., 0., 0., 0., 0., 0., 0.]], requires_grad=True)
        objectness = torch.tensor([.5, -.4], requires_grad=True)
        teacher = torch.full((8,), .6, requires_grad=True)
        loss = consistency(aggregate(roi, objectness), teacher)
        loss.backward()
        self.assertGreater(float(roi.grad.abs().sum()), 0.)
        self.assertGreater(float(objectness.grad.abs().sum()), 0.)
        self.assertIsNone(teacher.grad)

    def test_grl_sign_and_zero_ramp(self):
        value = torch.tensor([2., 3.], requires_grad=True)
        _Reverse.apply(value, .4).sum().backward()
        self.assertTrue(torch.allclose(value.grad, torch.full_like(value, -.4)))
        value.grad = None
        _Reverse.apply(value, 0.).sum().backward()
        self.assertEqual(float(value.grad.abs().sum()), 0.)

    def test_teacher_remains_eval_and_no_parameter_alias(self):
        # Exercise the actual DeST lifecycle without requiring a checkpoint.
        model = DeST.__new__(DeST); nn.Module.__init__(model)
        model.student = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        model.student[0].bias.requires_grad_(False)
        import copy
        model.teacher = copy.deepcopy(model.student).requires_grad_(False)
        model.global_D = nn.Linear(2, 1)
        model.ema_decay, model.ema_updates = .5, 0
        model.train()
        self.assertFalse(model.teacher.training)
        self.assertFalse(model.student[1].training)
        before = model.teacher[0].weight.detach().clone()
        with torch.no_grad():
            model.student[0].weight.add_(2.)
        model.update_teacher()
        self.assertTrue(torch.allclose(model.teacher[0].weight, before + 1.))
        self.assertEqual(model.ema_updates, 1)
        self.assertTrue(torch.equal(model.teacher[0].bias, model.student[0].bias))
        for teacher, student in zip(model.teacher.parameters(), model.student.parameters()):
            self.assertNotEqual(teacher.data_ptr(), student.data_ptr())
            self.assertFalse(teacher.requires_grad)

    def test_target_label_leakage_rejected(self):
        image = torch.zeros(3, 16, 32)
        with self.assertRaises(ValueError):
            DeST._validate([{"image": image, "image_weak": image, "image_labels": torch.zeros(8)}], weak=True)

    def test_native_rpn_selection_parity(self):
        try:
            from detectron2.modeling.proposal_generator.proposal_utils import find_top_rpn_proposals
        except (ImportError, OSError) as exc:
            self.skipTest("Native Detectron2 environment required: " + str(exc))
        decoded = [torch.tensor([[[0., 0., 8., 8.], [1., 1., 7., 7.], [12., 0., 20., 8.], [-4., 0., -1., 8.]]]),
                   torch.tensor([[[0., 0., 8., 8.], [22., 2., 40., 12.]]])]
        scores = [torch.tensor([[.7, .8, .4, .9]]), torch.tensor([[.6, .5]])]
        indexed = _select_indices(decoded, scores, [(10, 30)], .5, 4, 4, 0.)
        native = find_top_rpn_proposals(decoded, scores, [(10, 30)], .5, 4, 4, 0., False)
        self.assertTrue(torch.equal(indexed[0][0].tensor, native[0].proposal_boxes.tensor))
        self.assertTrue(torch.equal(torch.cat(scores, dim=1)[0, indexed[0][1]], native[0].objectness_logits))

    @unittest.skipUnless(os.environ.get("DEST_INTEGRATION_CONFIG"), "Real SourceB CPU smoke is opt-in")
    def test_real_checkpoint_forward_backward(self):
        from detectron2.structures import Boxes, Instances
        with open(os.environ["DEST_INTEGRATION_CONFIG"], encoding="utf-8") as stream:
            config = json.load(stream)
        # Test-only geometry/budgets: full real model, with no formal config edit.
        # Keep sampling budget >=4: the native .25 foreground fraction would
        # round down to zero at budget2 and remove the bbox gradient check.
        config.update(device="cpu", amp=False, initial_proposals=2, source_roi_batch=4,
            window_scales=[1.], teacher_roi_chunk=1, train_roi_chunk=1, inference_roi_chunk=1,
            rpn_pre_nms_topk=16, mrsa_max_iter=20, checkpoint_res5=True)
        torch.set_num_threads(2)
        torch.backends.mkldnn.enabled = False
        torch.manual_seed(3)
        print("[CPU integration] building full SourceB model", flush=True)
        model = DeST(config)
        self.assertTrue(model.verify_native_path()["passed"])
        print("[CPU integration] exact initialization and native forward verified", flush=True)
        height, width = 64, 128
        source_image = torch.rand(3, height, width) * 255.
        target_image = torch.rand(3, height, width) * 255.
        print("[CPU integration] preparing synthetic source GT via native RPN", flush=True)
        with torch.no_grad():
            proposals, _, _, _ = model._rpn(model.student, [{"image": source_image}])
        gt = Instances((height, width))
        gt.gt_boxes = Boxes(proposals[0].proposal_boxes.tensor[:2].clone())
        self.assertGreater(len(gt.gt_boxes), 0)
        gt.gt_classes = torch.arange(len(gt.gt_boxes), dtype=torch.long)
        del proposals
        original_teacher_targets = model._teacher_targets
        def logged_teacher_targets(*args, **kwargs):
            print("[CPU integration] teacher evidence and targets begin", flush=True)
            result = original_teacher_targets(*args, **kwargs)
            print("[CPU integration] teacher targets complete; student forward begins", flush=True)
            return result
        model._teacher_targets = logged_teacher_targets
        print("[CPU integration] complete training loss forward begin", flush=True)
        losses, stats = model.training_losses([{"image": source_image, "instances": gt}],
            [{"image": target_image, "image_weak": target_image.clone()}], progress=.2)
        print("[CPU integration] forward complete; backward begin", flush=True)
        sum(losses.values()).backward()
        print("[CPU integration] backward complete; checking all key gradients", flush=True)
        nonzero = {}
        for prefix in ("student.backbone", "student.offline_backbone", "student.offline_proposal_generator",
                       "student.roi_heads.box_predictor.bbox_pred", "global_D"):
            gradients = [p.grad for name, p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
            self.assertTrue(gradients, prefix)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients), prefix)
            nonzero[prefix] = sum(float(g.abs().sum()) for g in gradients)
            self.assertGreater(nonzero[prefix], 0., prefix)
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
        loss_values = {k: float(v.detach()) for k, v in losses.items()}
        del losses, gradients
        model.zero_grad(set_to_none=True); model.eval()
        import gc
        gc.collect()
        print("[CPU integration] gradient audit passed; teacher inference begin", flush=True)
        result = model.predict([{"image": target_image}], return_proposals=True)
        self.assertEqual(len(result), 1)
        self.assertGreater(len(result[0]["base_proposals"]), 0)
        self.assertLessEqual(len(result[0]["base_proposals"]), 2)
        report = {"cpu_forward_backward": "passed", "optimizer_steps": 0,
            "test_image_size": [height, width], "test_proposal_budget": 2,
            "losses": loss_values, "gradient_l1": nonzero,
            "teacher_mrsa": stats["teacher_mrsa"], "spatial": stats["spatial"],
            "initialization_audit": model.initialization_audit}
        output = os.environ.get("DEST_INTEGRATION_REPORT")
        if output:
            with open(output, "w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
        print(json.dumps({k: v for k, v in report.items() if k != "initialization_audit"}), flush=True)


if __name__ == "__main__":
    unittest.main()
