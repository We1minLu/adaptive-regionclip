"""CPU-only DeST protocol and optimizer/EMA transaction contracts.

These tests run no detector, image pipeline, or GPU training. Actual CUDA AMP
numerics still require the separate GPU smoke after the user enables hardware.
"""
import copy
from contextlib import ExitStack
import itertools
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dest import protocol
from experiments.dest import train


def config():
    path = Path(__file__).resolve().parents[1] / "configs" / "c2f_mrsa_25k.json"
    cfg = json.loads(path.read_text())
    cfg["amp"] = False
    return cfg


def batches(count=1):
    return [([{"image_id": "source_%d" % i, "instances": object(), "value": .7}],
             [{"image_id": "target_%d" % i, "beta": ("0.005", "0.01", "0.02")[i % 3],
               "value": 1.2}]) for i in range(count)]


class ToyDeST(nn.Module):
    """Two real optimization partitions, a frozen EMA copy, and spatial stats."""
    def __init__(self):
        super().__init__()
        self.student = nn.Linear(1, 1)
        self.global_D = nn.Linear(1, 1)
        self.teacher = copy.deepcopy(self.student)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("seen", torch.zeros(()))
        self.image_consistency_enabled = True
        self.ema_updates = 0
        self.trace = []
        self.teacher_update_trace = []

    def detector_parameters(self):
        return self.student.parameters()

    def aux_parameters(self):
        return self.global_D.parameters()

    def training_losses(self, source, target, progress):
        noise = random.random() + float(np.random.random()) + float(torch.rand(()))
        self.trace.append((noise, float(self.seen), source[0]["image_id"], target[0]["image_id"]))
        self.seen.add_(1)
        value = torch.tensor([[source[0]["value"] + target[0]["value"] + .01 * noise]])
        regional = self.student(value)
        losses = {
            "source": (regional - .2).square().mean(),
            "semantic": .1 * (regional + .3).square().mean(),
            "spatial": .2 * (regional - .5).square().mean(),
            "domain": self.global_D(regional).square().mean(),
        }
        return losses, {"spatial": {"valid_supports": 3, "active_supports": 2},
                        "roi": {"regions": 4}}

    @torch.no_grad()
    def update_teacher(self):
        self.teacher_update_trace.append(self.student.weight.detach().clone())
        for teacher, student in zip(self.teacher.parameters(), self.student.parameters()):
            teacher.mul_(.9).add_(student, alpha=.1)
        self.ema_updates += 1

    def checkpoint_state(self):
        return {"student": self.student.state_dict(), "teacher": self.teacher.state_dict(),
                "ema_updates": self.ema_updates, "version": protocol.VERSION}


class ProtocolTests(unittest.TestCase):
    def test_default_is_source_initialized_native_rpn_teacher_protocol(self):
        cfg = config()
        self.assertIs(protocol.validate_config(cfg), cfg)
        self.assertEqual(cfg["source_checkpoint_sha256"], protocol.SOURCE_SHA256)
        settings = protocol.consistency_settings(cfg)
        self.assertEqual(settings["evaluation_model"], "teacher")
        self.assertEqual(settings["image_aggregation"], "mrsa_teacher_h2fa_student")

    def test_old_vgs_and_conditional_options_are_rejected(self):
        for name in ("source_search_checkpoint", "supplemental_proposals", "search_loss_weight",
                     "conditional_domain_weight", "conditional_grl_max"):
            with self.subTest(name=name):
                cfg = config()
                cfg[name] = 0
                with self.assertRaises(ValueError):
                    protocol.validate_config(cfg)

    def test_target_labels_legacy_pooling_student_eval_and_wrong_source_rejected(self):
        for name, value in (("target_image_labels_allowed", True), ("evaluation_model", "student"),
                            ("image_aggregation", "h2fa_iir"), ("rpn_mode", "frozen"),
                            ("method_version", "formal-regionclip-vgs-da-ema-image-v2"),
                            ("source_checkpoint_sha256", "0" * 64)):
            with self.subTest(name=name):
                cfg = config()
                cfg[name] = value
                with self.assertRaises(ValueError):
                    protocol.validate_config(cfg)

    def test_invalid_uncertainty_geometry_and_solver_values_rejected(self):
        for name, value in (("mrsa_rho", -1), ("mrsa_rho", float("nan")),
                            ("spatial_gamma", .9), ("spatial_tau", 0), ("spatial_tau", 1),
                            ("mrsa_tolerance", 0), ("mrsa_max_iter", True),
                            ("window_scales", []), ("window_scales", [float("inf")]),
                            ("window_scales", [True]), ("ema_decay", 1)):
            with self.subTest(name=name, value=value):
                cfg = config()
                cfg[name] = value
                with self.assertRaises(ValueError):
                    protocol.validate_config(cfg)

    def test_resume_rejects_changes_to_objective_data_optimizer_and_budget(self):
        old = config()
        for name, value in (("mrsa_rho", 2.), ("spatial_loss_weight", .2),
                            ("window_scales", [.25, .5]), ("spatial_tau", .5),
                            ("ema_decay", .99), ("seed", 1), ("detector_lr", .001),
                            ("max_steps", old["max_steps"] + 1000),
                            ("manifest_sha256", {"eval_mixed.json": "changed"})):
            with self.subTest(name=name):
                new = copy.deepcopy(old)
                new[name] = value
                with self.assertRaises(ValueError):
                    protocol.validate_resume(old, new, 10, verified_sources=True)

    def test_resume_allows_logging_but_requires_verified_asset_relocation(self):
        old = config()
        new = copy.deepcopy(old)
        new.update(output_dir="some/new/output", log_period=5, num_workers=0, train_roi_chunk=64)
        self.assertEqual(protocol.validate_resume(old, new, 123)["saved_step"], 123)
        new["source_checkpoint"] = "relocated/source.pth"
        with self.assertRaises(ValueError):
            protocol.validate_resume(old, new, 123)
        self.assertTrue(protocol.validate_resume(old, new, 123, verified_sources=True)["verified_sources"])

    def test_resume_step_must_be_valid_integer_within_fixed_schedule(self):
        for step in (-1, True, 1.5, config()["max_steps"] + 1):
            with self.subTest(step=step), self.assertRaises(ValueError):
                protocol.validate_resume(config(), config(), step)

    def test_grl_schedule_is_bounded(self):
        cfg = config()
        self.assertEqual(protocol.grl_progress(-1, cfg), 0.)
        self.assertEqual(protocol.grl_progress(cfg["grl_schedule_steps"] / 2, cfg), .5)
        self.assertEqual(protocol.grl_progress(10 * cfg["grl_schedule_steps"], cfg), 1.)


class TrainingTransactionTests(unittest.TestCase):
    def setUp(self):
        train.set_seed(71)
        self.cfg = config()
        self.model = ToyDeST()
        self.optimizers, _ = train.optimizer_for(self.model, self.cfg)
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def update(self, batch_list=None):
        return train.train_update(self.model, batch_list or batches(), self.optimizers,
                                  self.scaler, self.cfg, .1, 1, self.output)

    def test_real_cpu_update_advances_both_optimizers_and_teacher_once(self):
        student_before = self.model.student.weight.detach().clone()
        domain_before = self.model.global_D.weight.detach().clone()
        teacher_before = self.model.teacher.weight.detach().clone()
        result = self.update()
        self.assertTrue(result["did_step"])
        self.assertEqual(result["retries"], 0)
        self.assertEqual(self.model.ema_updates, 1)
        self.assertFalse(torch.equal(student_before, self.model.student.weight))
        self.assertFalse(torch.equal(domain_before, self.model.global_D.weight))
        torch.testing.assert_close(self.model.teacher.weight.detach(),
                                   .9 * teacher_before + .1 * self.model.student.weight)
        self.assertTrue(all(p.grad is None for p in self.model.teacher.parameters()))

    def test_accumulation_retains_spatial_stats_without_old_conditional_stats(self):
        self.cfg["accumulation_steps"] = 2
        result = self.update(batches(2))
        self.assertEqual(self.model.ema_updates, 1)
        self.assertEqual(len(result["support_per_micro"]), 2)
        self.assertEqual(result["support_per_micro"][0]["spatial"]["active_supports"], 2)
        self.assertNotIn("conditional_domain", result["stats"])
        self.assertEqual(result["density_counts"], {"0.005": 1, "0.01": 1})
        self.assertEqual(result["stats"]["ema_updates"], 1)

    def test_target_annotations_and_image_labels_rejected_before_update(self):
        for forbidden in ("instances", "annotations", "image_labels"):
            records = batches()
            records[0][1][0][forbidden] = []
            before = self.model.student.weight.detach().clone()
            with self.subTest(field=forbidden), self.assertRaises(AssertionError):
                self.update(records)
            torch.testing.assert_close(before, self.model.student.weight.detach())
        self.assertEqual(self.model.ema_updates, 0)

    def test_nonfinite_aux_gradient_leaves_both_optimizers_and_teacher_unchanged(self):
        before = [p.detach().clone() for p in self.model.parameters()]
        hook = self.model.global_D.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        try:
            with self.assertRaises(FloatingPointError):
                self.update()
        finally:
            hook.remove()
        for expected, parameter in zip(before, self.model.parameters()):
            torch.testing.assert_close(expected, parameter.detach())
        self.assertEqual(self.model.ema_updates, 0)
        self.assertTrue(all(len(opt.state) == 0 for opt in self.optimizers))

    def test_retry_replays_rng_buffers_and_inputs_before_one_committed_update(self):
        original = train.atomic_amp_step
        calls = []
        def simulated_overflow(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return dict(did_step=False, retry_required=True, old_scale=2., new_scale=1.,
                            global_grad_norm=None, clip_coefficient=None, nonfinite_gradients=[])
            return original(*args, **kwargs)
        with mock.patch.object(train, "atomic_amp_step", side_effect=simulated_overflow):
            result = self.update()
        self.assertEqual(result["retries"], 1)
        self.assertEqual(self.model.trace[0], self.model.trace[1])
        self.assertEqual(float(self.model.seen), 1.)
        self.assertEqual(self.model.ema_updates, 1)
        self.assertEqual(len((self.output / "amp_overflow_events.jsonl").read_text().splitlines()), 1)

    def test_exhausted_retries_restore_buffers_and_do_not_update_teacher(self):
        self.cfg["amp_max_retries"] = 1
        before_rng = torch.get_rng_state().clone()
        failure = dict(did_step=False, retry_required=True, old_scale=2., new_scale=1.,
                       global_grad_norm=None, clip_coefficient=None, nonfinite_gradients=[])
        with mock.patch.object(train, "atomic_amp_step", return_value=failure):
            with self.assertRaises(FloatingPointError):
                self.update()
        self.assertEqual(float(self.model.seen), 0.)
        self.assertEqual(self.model.ema_updates, 0)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))

    def test_teacher_step_and_metadata_reject_desynchronization(self):
        train.validate_ema_step(self.model, 0)
        with self.assertRaises(ValueError):
            train.validate_ema_step(self.model, 1)
        metadata = train.consistency_metadata(self.model, self.cfg)
        self.assertEqual(metadata["evaluation_model"], "teacher")
        self.model.image_consistency_enabled = False
        with self.assertRaises(ValueError):
            train.consistency_metadata(self.model, self.cfg)

    def test_full_checkpoint_preserves_two_optimizers_rng_teacher_and_source_identity(self):
        self.update()
        train.save_checkpoint(self.model, self.optimizers, self.scaler, self.cfg,
                              self.output, 1, {"successful_updates": 1}, final=True)
        saved = torch.load(str(self.output / "checkpoint_last.pth"), map_location="cpu")
        self.assertEqual(len(saved["optimizers"]), 2)
        self.assertEqual(saved["model"]["ema_updates"], 1)
        self.assertEqual(saved["config"]["source_checkpoint_sha256"], protocol.SOURCE_SHA256)
        self.assertEqual(set(saved["rng"]), {"python", "numpy", "torch", "cuda"})
        self.assertTrue(all(len(state["state"]) > 0 for state in saved["optimizers"]))
        compact = torch.load(str(self.output / "model_final.pth"), map_location="cpu")
        self.assertNotIn("optimizers", compact)
        self.assertIn("teacher", compact["model"])

    def test_evaluation_uses_teacher_metadata_and_restores_rng_and_training_mode(self):
        before = torch.get_rng_state().clone()
        module = types.ModuleType("experiments.vgs_da.evaluation")
        def fake_evaluate(model, manifest, output, max_images=None):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            torch.rand(3)
            random.random()
            np.random.random()
            return {"complete": False, "images": 6}
        module.evaluate_model = fake_evaluate
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            result = train.run_evaluation(self.model, self.cfg, self.output, max_images=6)
        self.assertEqual(result["evaluation_model"], "teacher")
        self.assertEqual(result["image_aggregation"], "mrsa_teacher_h2fa_student")
        self.assertFalse(result["evaluation_GT_used_for_training"])
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


class OutputOwnershipTests(unittest.TestCase):
    """Exercise real CLI/trainer admission with CPU stand-ins for the detector."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "run"
        self.output.mkdir()
        self.cfg = config()
        self.cfg.update(output_dir=str(self.output), manifest_dir=str(self.root / "data"))
        manifests = Path(self.cfg["manifest_dir"])
        manifests.mkdir()
        hashes = {}
        for name in ("source_train.json", "target_train.json", "eval_mixed.json"):
            (manifests / name).write_text("[]")
            hashes[name] = train.digest(manifests / name)
        (manifests / "data_audit.json").write_text(json.dumps({
            "target_image_labels_allowed": False, "manifest_sha256": hashes}))
        self.saved_cfg = copy.deepcopy(self.cfg)
        train.verify_manifest_identity(self.saved_cfg)
        for name in ("config.json", "status.json", "initialization_audit.json"):
            (self.output / name).write_text(json.dumps({"preserved": name, "step": 4}))
        self.original = self.snapshot()
        self.sources = {key: {"path": key, "sha256": "a" * 64}
                        for key in ("source_cfg", "source_checkpoint", "rpn_checkpoint", "text_embeddings")}

    def snapshot(self):
        return {str(p.relative_to(self.output)): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}

    def environment(self, constructor=None):
        case = self
        class Model(ToyDeST):
            def __init__(self, cfg):
                super().__init__()
                self.config = copy.deepcopy(cfg)
                self.source_identity = copy.deepcopy(case.sources)
                self.initialization_audit = {"verified": True}
            def to(self, device):
                return self
            def checkpoint_state(self):
                return {"version": protocol.VERSION, "config": self.config,
                        "sources": self.source_identity, "ema_updates": self.ema_updates,
                        "weights": copy.deepcopy(self.state_dict())}
            def load_checkpoint_state(self, state):
                self.load_state_dict(state["weights"])
                self.ema_updates = state["ema_updates"]
        class Storage:
            def __init__(self, step):
                self.iter = step
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        model_module = types.ModuleType("experiments.dest.model")
        model_module.DeST = constructor or Model
        events = types.ModuleType("detectron2.utils.events")
        events.EventStorage = Storage
        original_digest = train.digest
        stack = ExitStack()
        stack.enter_context(mock.patch.dict(sys.modules, {
            model_module.__name__: model_module, events.__name__: events}))
        stack.enter_context(mock.patch.object(train, "configure_paths"))
        stack.enter_context(mock.patch.object(train, "digest", side_effect=lambda p:
            protocol.SOURCE_SHA256 if str(p) == self.cfg["source_checkpoint"] else original_digest(p)))
        stack.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=True))
        for name in ("reset_peak_memory_stats", "synchronize"):
            stack.enter_context(mock.patch.object(torch.cuda, name))
        for name in ("max_memory_allocated", "max_memory_reserved"):
            stack.enter_context(mock.patch.object(torch.cuda, name, return_value=0))
        stack.enter_context(mock.patch.object(train, "gradient_audit", return_value={
            key: {"norm": 1.} for key in ("student.backbone.layer4", "student.backbone.attnpool",
                "student.roi_heads.box_predictor", "student.offline_proposal_generator.rpn_head")}))
        source, target = batches()[0]
        stack.enter_context(mock.patch.object(train, "make_loaders", side_effect=lambda cfg, step:
            (itertools.cycle([source]), itertools.cycle([target]))))
        return stack, Model

    def invoke_cli(self, mode, checkpoint=None):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.cfg))
        args = ["train.py", "--config", str(config_path), "--mode", mode]
        if checkpoint is not None:
            args.extend(["--resume", str(checkpoint)])
        with mock.patch.object(sys, "argv", args):
            train.main()

    def test_four_asset_identity_accepts_relocation_and_rejects_wrong_hash(self):
        relocated = copy.deepcopy(self.sources)
        relocated["source_checkpoint"]["path"] = "/elsewhere/source.pth"
        self.assertTrue(train.same_sources(self.sources, relocated))
        relocated["source_checkpoint"]["sha256"] = "b" * 64
        self.assertFalse(train.same_sources(self.sources, relocated))
        relocated["source_search_checkpoint"] = None
        with self.assertRaises(ValueError):
            train.same_sources(self.sources, relocated)

    def test_rejected_fresh_launch_does_not_overwrite_existing_run(self):
        (self.output / "metrics.jsonl").write_text('{"step":4}\n')
        before = self.snapshot()
        environment, _ = self.environment()
        with environment, self.assertRaisesRegex(RuntimeError, "already contains"):
            self.invoke_cli("train")
        self.assertEqual(self.snapshot(), before)

    def test_bad_resume_and_objective_change_do_not_touch_existing_metadata(self):
        for failure in ("incomplete", "objective", "source"):
            with self.subTest(failure=failure):
                old = copy.deepcopy(self.saved_cfg)
                sources = copy.deepcopy(self.sources)
                if failure == "objective":
                    old["mrsa_rho"] += 1
                if failure == "source":
                    sources["source_checkpoint"]["sha256"] = "b" * 64
                state = {"version": 1, "step": 4, "config": old, "optimizers": [{}, {}],
                         "model": {"config": old, "sources": sources}}
                if failure == "incomplete":
                    state["optimizers"] = [{}]
                path = self.root / (failure + ".pth")
                torch.save(state, str(path))
                environment, _ = self.environment()
                with environment, self.assertRaises(ValueError):
                    self.invoke_cli("train", path)
                self.assertEqual(self.snapshot(), self.original)

    def test_model_construction_failure_does_not_claim_existing_output(self):
        constructor = mock.Mock(side_effect=RuntimeError("model initialization failed"))
        environment, _ = self.environment(constructor)
        with environment, self.assertRaisesRegex(RuntimeError, "model initialization failed"):
            self.invoke_cli("train")
        self.assertEqual(self.snapshot(), self.original)

    def test_rejected_evaluation_does_not_overwrite_existing_run(self):
        old = copy.deepcopy(self.saved_cfg)
        old["mrsa_rho"] += 1
        path = self.root / "mismatched.pth"
        torch.save({"config": old, "model": {"config": old}}, str(path))
        environment, _ = self.environment()
        with environment, self.assertRaisesRegex(ValueError, "mrsa_rho"):
            self.invoke_cli("evaluate", path)
        self.assertEqual(self.snapshot(), self.original)

    def test_evaluation_resolves_runtime_manifest_identity_before_checkpoint_check(self):
        environment, Model = self.environment()
        model = Model(self.saved_cfg)
        state = {"config": self.saved_cfg, "model": model.checkpoint_state(), "step": 0}
        path = self.root / "valid.pth"
        torch.save(state, str(path))
        ownership = train.RunOwnership()
        with environment, mock.patch.object(train, "run_evaluation", return_value={"ok": True}) as evaluate:
            result = train.run_checkpoint_evaluation(copy.deepcopy(self.cfg), path, ownership)
        self.assertTrue(result["ok"])
        self.assertEqual(evaluate.call_args[0][1]["manifest_sha256"], self.saved_cfg["manifest_sha256"])
        self.assertEqual(ownership.output, self.output / "evaluation")
        self.assertEqual(self.snapshot(), self.original)

    def test_modified_manifest_is_rejected_without_claiming_output(self):
        (Path(self.cfg["manifest_dir"]) / "target_train.json").write_text('[{"changed":true}]')
        ownership = train.RunOwnership()
        with self.assertRaisesRegex(ValueError, "Manifest changed"):
            train.verify_manifest_identity(self.cfg)
        self.assertIsNone(ownership.output)
        self.assertEqual(self.snapshot(), self.original)

    def test_claimed_failure_is_recorded_only_in_claimed_evaluation_directory(self):
        ownership = train.RunOwnership()
        ownership.claim(self.output / "evaluation")
        ownership.report_failure(RuntimeError("evaluation failed"))
        self.assertTrue((self.output / "evaluation" / "failure.json").is_file())
        for name, data in self.original.items():
            self.assertEqual((self.output / name).read_bytes(), data)

    def test_four_step_smoke_then_two_step_resume_keeps_full_state_and_appends(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["output_dir"] = str(self.root / "smoke")
        output = Path(cfg["output_dir"])
        environment, _ = self.environment()
        with environment:
            train.run_training(copy.deepcopy(cfg), smoke=True, smoke_steps=4, eval_smoke=False)
            original_metrics = (output / "metrics.jsonl").read_bytes()
            train.run_training(copy.deepcopy(cfg), smoke=True, smoke_steps=2, eval_smoke=False,
                               resume=output / "checkpoint_last.pth")
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["step"] for row in rows], list(range(1, 7)))
        self.assertTrue((output / "metrics.jsonl").read_bytes().startswith(original_metrics))
        checkpoint = torch.load(str(output / "checkpoint_last.pth"), map_location="cpu")
        previous = torch.load(str(output / "checkpoint_previous.pth"), map_location="cpu")
        self.assertEqual(checkpoint["step"], 6)
        self.assertEqual(checkpoint["model"]["ema_updates"], 6)
        self.assertEqual(previous["step"], 4)
        self.assertEqual(len(checkpoint["optimizers"]), 2)
        self.assertEqual(checkpoint["runtime"]["successful_updates"], 6)
        resume = json.loads((output / "resume_audit.json").read_text())
        self.assertEqual(resume["restored_step"], 4)
        self.assertEqual(resume["source_samples_consumed"], 8)


if __name__ == "__main__":
    unittest.main()
