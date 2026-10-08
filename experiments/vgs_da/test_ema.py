"""CPU checks for EMA mathematics, atomic-update integration, and resuming.

Overflow branches here mock only the AMP overflow decision; they verify the
trainer's EMA call boundary, not real CUDA overflow detection (covered by the
existing GPU AMP tests).
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from continuation import (CONSISTENCY_DEFAULTS, consistency_settings, validate_resume,
                          validate_evaluation_supervision)
from ema import update_ema
import train as trainer


class StateModule(nn.Module):
    def __init__(self, width=2):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(width, dtype=torch.float32))
        self.register_buffer("running", torch.ones(width))
        self.register_buffer("counter", torch.tensor(3, dtype=torch.int64))
        self.register_buffer("mask", torch.tensor([True, False]))


class EMAHelperTests(unittest.TestCase):
    def test_parameters_and_float_buffers_average_integer_buffers_copy(self):
        student = StateModule()
        teacher = copy.deepcopy(student).requires_grad_(False)
        before = copy.deepcopy(teacher.state_dict())
        with torch.no_grad():
            student.weight.add_(4.)
            student.running.add_(8.)
            student.counter.add_(5)
            student.mask.logical_not_()
        student_before = copy.deepcopy(student.state_dict())
        update_ema(teacher, student, .9996)
        for key, value in teacher.state_dict().items():
            expected = (before[key] * .9996 + student_before[key] * .0004
                        if value.is_floating_point() else student_before[key])
            self.assertTrue(torch.equal(value, expected), key)
            self.assertTrue(torch.equal(student.state_dict()[key], student_before[key]), key)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))

    def test_zero_decay_copies_all_state(self):
        student, teacher = StateModule(), StateModule().requires_grad_(False)
        with torch.no_grad():
            student.weight.fill_(9.)
        update_ema(teacher, student, 0.)
        for key, value in teacher.state_dict().items():
            self.assertTrue(torch.equal(value, student.state_dict()[key]))

    def test_invalid_decay_is_rejected(self):
        student, teacher = StateModule(), StateModule().requires_grad_(False)
        for decay in (-.1, 1., float("nan"), float("inf"), True, "0.9996"):
            with self.subTest(decay=decay), self.assertRaisesRegex(ValueError, "decay"):
                update_ema(teacher, student, decay)

    def test_trainable_teacher_and_alias_are_rejected(self):
        student = StateModule()
        with self.assertRaisesRegex(ValueError, "frozen"):
            update_ema(StateModule(), student, .9)
        frozen = copy.deepcopy(student).requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "distinct"):
            update_ema(frozen, frozen, .9)
        aliased = copy.deepcopy(student).requires_grad_(False)
        aliased.running = student.running
        with self.assertRaisesRegex(ValueError, "aliases"):
            update_ema(aliased, student, .9)

    def test_incompatible_state_fails_before_any_mutation(self):
        teacher = StateModule().requires_grad_(False)
        before = copy.deepcopy(teacher.state_dict())
        cases = [StateModule(3), StateModule().double()]
        missing = StateModule()
        del missing.counter
        cases.append(missing)
        for student in cases:
            with self.subTest(student=repr(student)), self.assertRaises(ValueError):
                update_ema(teacher, student, .9)
            for key, value in teacher.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]), key)


class ToyModel(nn.Module):
    def __init__(self, enabled=True):
        super().__init__()
        self.student = nn.ModuleDict({"detector": nn.Linear(2, 2), "auxiliary": nn.Linear(2, 1)})
        self.teacher = copy.deepcopy(self.student).requires_grad_(False).eval()
        self.register_buffer("forward_count", torch.tensor(0, dtype=torch.int64))
        self.image_consistency_enabled = enabled
        self.ema_updates = 0
        self.force_nonfinite = False
        self.trace = []

    def training_losses(self, source, target, progress):
        self.forward_count.add_(1)
        value = source[0]["image"] + torch.rand(2) * .1
        self.trace.append(value.detach().clone())
        hidden = self.student["detector"](value)
        loss = self.student["auxiliary"](hidden).square().mean() + hidden.square().mean()
        if self.force_nonfinite:
            loss = loss * float("nan")
        return {"toy": loss}, {"conditional_domain": {"active_groups": 1}, "roi": {"sampled": 2}}

    def update_teacher(self):
        update_ema(self.teacher, self.student, .9996)
        self.ema_updates += 1

    def checkpoint_state(self):
        return {"state": copy.deepcopy(self.state_dict()), "ema_updates": self.ema_updates}

    def load_checkpoint_state(self, state):
        self.load_state_dict(state["state"], strict=True)
        self.ema_updates = state["ema_updates"]


def make_optimizers(model):
    return [torch.optim.SGD(model.student["detector"].parameters(), lr=.01, momentum=.9),
            torch.optim.AdamW(model.student["auxiliary"].parameters(), lr=.002)]


class EMATrainIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(418)
        self.temp = tempfile.TemporaryDirectory(prefix="vgs_ema_cpu_")
        self.output = Path(self.temp.name)
        self.cfg = {"amp": False, "accumulation_steps": 2, "amp_max_retries": 2,
                    "gradient_clip_norm": 10., "max_steps": 25000,
                    "target_image_labels_allowed": False,
                    "image_consistency_enabled": True, "strong_weak_enabled": True}
        self.batches = [([{"image_id": "source_%d" % i, "instances": object(),
                           "image": torch.tensor([1., 2.])}],
                         [{"image_id": "target_%d" % i, "beta": "0.005"}]) for i in range(2)]

    def tearDown(self):
        self.temp.cleanup()

    def call_update(self, model, optimizers, step=1):
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        return trainer.train_update(model, self.batches, optimizers, scaler, self.cfg,
                                    progress=.1, step=step, output=self.output)

    @staticmethod
    def overflow_result():
        return {"did_step": False, "retry_required": True, "old_scale": 2., "new_scale": 1.,
                "nonfinite_gradients": [{"parameter": "toy", "optimizer_index": 0}]}

    def assert_state_equal(self, a, b):
        self.assertEqual(set(a), set(b))
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_success_updates_teacher_once_after_both_student_optimizers(self):
        model = ToyModel()
        before = copy.deepcopy(model.teacher.state_dict())
        result = self.call_update(model, make_optimizers(model))
        self.assertEqual(model.ema_updates, 1)
        self.assertEqual(int(model.forward_count), 2)
        self.assertTrue(result["teacher_updated"])
        for key, value in model.teacher.state_dict().items():
            expected = before[key] * .9996 + model.student.state_dict()[key] * .0004
            self.assertTrue(torch.equal(value, expected), key)

    def test_mocked_overflow_retry_updates_once_and_matches_clean_replay(self):
        retried = ToyModel()
        clean = copy.deepcopy(retried)
        retry_opts, clean_opts = make_optimizers(retried), make_optimizers(clean)
        rng_before = trainer.rng_state()
        real_atomic, calls = trainer.atomic_amp_step, []

        def first_overflow(*args, **kwargs):
            calls.append(1)
            return self.overflow_result() if len(calls) == 1 else real_atomic(*args, **kwargs)

        with patch.object(trainer, "atomic_amp_step", side_effect=first_overflow):
            result = self.call_update(retried, retry_opts)
        self.assertEqual(result["retries"], 1)
        self.assertEqual(retried.ema_updates, 1)
        self.assertEqual(int(retried.forward_count), 2)
        self.assertEqual(len(retried.trace), 4)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(retried.trace[:2], retried.trace[2:])))
        trainer.restore_rng(rng_before)
        self.call_update(clean, clean_opts)
        self.assert_state_equal(retried.state_dict(), clean.state_dict())

    def test_exhausted_retries_never_update_teacher_or_student(self):
        model = ToyModel()
        before = copy.deepcopy(model.state_dict())
        with patch.object(trainer, "atomic_amp_step", return_value=self.overflow_result()):
            with self.assertRaisesRegex(FloatingPointError, "no optimizer update"):
                self.call_update(model, make_optimizers(model))
        self.assertEqual(model.ema_updates, 0)
        self.assert_state_equal(model.state_dict(), before)

    def test_nonfinite_forward_never_updates_teacher(self):
        model = ToyModel()
        before = copy.deepcopy(model.teacher.state_dict())
        model.force_nonfinite = True
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite forward"):
            self.call_update(model, make_optimizers(model))
        self.assertEqual(model.ema_updates, 0)
        self.assert_state_equal(model.teacher.state_dict(), before)

    def test_baseline_does_not_update_teacher(self):
        model = ToyModel(enabled=False)
        before = copy.deepcopy(model.teacher.state_dict())
        result = self.call_update(model, make_optimizers(model))
        self.assertFalse(result["teacher_updated"])
        self.assertEqual(model.ema_updates, 0)
        self.assert_state_equal(model.teacher.state_dict(), before)

    def test_checkpoint_resume_preserves_teacher_counter_and_next_step(self):
        model = ToyModel()
        optimizers = make_optimizers(model)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.call_update(model, optimizers)
        trainer.save_checkpoint(model, optimizers, scaler, self.cfg, self.output, 1, {}, final=True)
        saved = torch.load(str(self.output / "checkpoint_last.pth"), map_location="cpu")
        self.call_update(model, optimizers, step=2)
        resumed = ToyModel()
        resumed.load_checkpoint_state(saved["model"])
        self.assertEqual(resumed.ema_updates, 1)
        trainer.validate_ema_step(resumed, saved["step"])
        resumed_optimizers = make_optimizers(resumed)
        for optimizer, state in zip(resumed_optimizers, saved["optimizers"]):
            optimizer.load_state_dict(state)
        trainer.restore_rng(saved["rng"])
        self.call_update(resumed, resumed_optimizers, step=2)
        self.assertEqual(resumed.ema_updates, 2)
        self.assert_state_equal(model.state_dict(), resumed.state_dict())

    def test_checkpoint_rejects_inconsistent_teacher_count_before_writing(self):
        model = ToyModel()
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        with self.assertRaisesRegex(ValueError, "count must equal checkpoint step"):
            trainer.save_checkpoint(model, make_optimizers(model), scaler, self.cfg,
                                    self.output, 1, {})
        self.assertFalse((self.output / "checkpoint_last.pth").exists())
        trainer.validate_ema_step(model, 0)
        model.ema_updates = True
        with self.assertRaisesRegex(ValueError, "count must equal checkpoint step"):
            trainer.validate_ema_step(model, 1)

    def test_resume_counter_mismatch_is_detected(self):
        model = ToyModel()
        saved = model.checkpoint_state()
        model.load_checkpoint_state(saved)
        with self.assertRaisesRegex(ValueError, "count must equal checkpoint step"):
            trainer.validate_ema_step(model, 17)
        # Baseline checkpoints predate EMA and need no teacher counter.
        baseline = ToyModel(enabled=False)
        trainer.validate_ema_step(baseline, 17)

    def test_evaluation_metrics_record_teacher_protocol_and_student_identity(self):
        model = ToyModel()
        self.call_update(model, make_optimizers(model))
        cfg = dict(self.cfg, manifest_dir=str(self.output), experiment_scope="ema_cpu_test")
        result_stub = {"pooled": {"AP50": 0.}, "complete": False}
        with patch("evaluation.evaluate_model", return_value=result_stub):
            result = trainer.run_evaluation(model, cfg, self.output / "evaluation", max_images=1)
        self.assertEqual(result["ema_updates"], 1)
        self.assertEqual(result["ema_decay"], .9996)
        self.assertTrue(result["image_consistency_enabled"])
        self.assertTrue(result["strong_weak_enabled"])
        self.assertEqual(result["semantic_teacher_mode"], "ema")
        self.assertEqual(result["evaluation_model"], "student")
        written = json.loads((self.output / "evaluation" / "metrics.json").read_text())
        self.assertEqual(written, result)
        self.assertTrue(model.training)

    def test_baseline_metadata_does_not_claim_ema_and_rejects_mislabeled_model(self):
        model = ToyModel(enabled=False)
        cfg = {"target_image_labels_allowed": False}
        metadata = trainer.consistency_metadata(model, cfg)
        self.assertFalse(metadata["image_consistency_enabled"])
        self.assertEqual(metadata["semantic_teacher_mode"], "fixed")
        self.assertNotIn("ema_updates", metadata)
        self.assertNotIn("ema_decay", metadata)
        with self.assertRaisesRegex(ValueError, "Model/config"):
            trainer.consistency_metadata(model, self.cfg)
        with self.assertRaisesRegex(ValueError, "only reports"):
            trainer.consistency_metadata(model, dict(cfg, evaluation_model="teacher"))


class EMAResumeContractTests(unittest.TestCase):
    def test_legacy_baseline_missing_defaults_resumes_with_explicit_defaults(self):
        old = {"max_steps": 25000}
        new = dict(old, **CONSISTENCY_DEFAULTS, semantic_teacher_mode="fixed")
        self.assertFalse(validate_resume(old, new, 300)["budget_extended"])

    def test_new_objective_cannot_be_enabled_by_full_resume(self):
        old = {"max_steps": 25000}
        changed = dict(old, image_consistency_enabled=True, strong_weak_enabled=True)
        with self.assertRaisesRegex(ValueError, "image_consistency_enabled"):
            validate_resume(old, changed, 300)

    def test_all_new_settings_are_locked(self):
        old = {"max_steps": 25000, "image_consistency_enabled": True, "strong_weak_enabled": True}
        replacements = {"image_consistency_enabled": False, "strong_weak_enabled": False,
                        "ema_decay": .99, "image_consistency_weight": 2., "image_aggregation": "max",
                        "semantic_teacher_mode": "fixed", "source_view": "weak", "evaluation_model": "teacher"}
        for key, value in replacements.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_resume(old, dict(old, **{key: value}), 300)

    def test_implicit_semantic_mode_depends_on_enabled_branch(self):
        self.assertEqual(consistency_settings({})["semantic_teacher_mode"], "fixed")
        self.assertEqual(consistency_settings({"image_consistency_enabled": True})["semantic_teacher_mode"], "ema")
        with self.assertRaisesRegex(ValueError, "boolean"):
            consistency_settings({"strong_weak_enabled": "false"})

    def test_evaluation_cannot_silently_change_consistency_protocol(self):
        saved = {"target_image_labels_allowed": False,
                 "image_consistency_enabled": True, "strong_weak_enabled": True}
        validate_evaluation_supervision(saved, dict(saved, semantic_teacher_mode="ema"))
        for key, value in (("image_consistency_enabled", False), ("strong_weak_enabled", False),
                           ("semantic_teacher_mode", "fixed"), ("ema_decay", .99),
                           ("image_consistency_weight", .5), ("image_aggregation", "mean"),
                           ("evaluation_model", "teacher"), ("source_view", "weak")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_evaluation_supervision(saved, dict(saved, **{key: value}))
        baseline = {"target_image_labels_allowed": False}
        validate_evaluation_supervision(baseline, dict(baseline, **CONSISTENCY_DEFAULTS,
                                                    semantic_teacher_mode="fixed"))


if __name__ == "__main__":
    unittest.main()
