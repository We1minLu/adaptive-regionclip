"""Small CUDA-only integration tests for train.train_update; no detector imports.

Run with the same Python/PyTorch runtime as the formal experiment::

    python test_train_update.py --output train_update_tests.json

No images, detection code, checkpoints, or production output directory are read.
A missing CUDA device is an error, not a silently skipped successful test run.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("formal_train_update_subject", HERE / "train.py")
TRAIN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TRAIN)


class CountingSGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_calls = 0

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


class CountingAdamW(torch.optim.AdamW):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_calls = 0

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


class CountingLoader:
    """Iterator whose consumption remains visible after train_update returns."""
    def __init__(self, domain):
        self.domain = domain
        self.consumed = 0

    def __next__(self):
        index = self.consumed
        self.consumed += 1
        record = {
            "image_id": "%s_%d" % (self.domain, index),
            "image": torch.full((2, 4), .1 + .05 * index, device="cuda"),
        }
        if self.domain == "source":
            record["instances"] = object()  # Only the source-presence contract is needed.
        else:
            record["beta"] = ("0.005", "0.01", "0.02")[index % 3]
        return [record]


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = nn.Linear(4, 4)
        self.auxiliary = nn.Linear(4, 2)
        self.register_buffer("running", torch.zeros(2))
        self.trace = []
        self.backward_calls = 0
        self.force_forward_nonfinite = False

    def training_losses(self, source, target, progress):
        # Consume all four RNG streams. The buffer affects the loss, so a missing
        # buffer rollback cannot pass by merely preserving the RNG sequence.
        python_value = random.random()
        numpy_value = np.random.random(3)
        cpu_value = torch.rand(3)
        cuda_value = torch.rand(2, 4, device="cuda")
        before = self.running.detach().cpu().tolist()
        with torch.no_grad():
            self.running[0].add_(1)
            self.running[1].add_(python_value + float(numpy_value.sum()))
        self.trace.append({
            "python": python_value, "numpy": numpy_value.tolist(),
            "cpu": cpu_value.tolist(), "cuda": cuda_value.cpu().tolist(),
            "buffer_before": before, "progress": progress,
            "source_ids": [row["image_id"] for row in source],
            "target_ids": [row["image_id"] for row in target],
            "source_objects": [id(row) for row in source],
            "target_objects": [id(row) for row in target],
        })
        shift = .01 * (python_value + float(numpy_value.sum()) + float(cpu_value.sum()))
        shift += .001 * float(self.running.sum())
        value = source[0]["image"] + .2 * target[0]["image"] + .1 * cuda_value + shift
        hidden = self.detector(value)
        prediction = self.auxiliary(hidden)
        loss_detector = .5 * hidden.float().square().mean()
        loss_auxiliary = (prediction.float() - .25).square().mean()
        if self.force_forward_nonfinite:
            loss_detector = loss_detector * float("nan")
        return {"detector": loss_detector, "auxiliary": loss_auxiliary}, {
            "conditional_domain": {"active_groups": 2},
            "roi": {"sampled": 4, "foreground": 2},
        }

    def inject_gradient_inf(self, optimizer_index, always=False):
        parameter = (self.detector if optimizer_index == 0 else self.auxiliary).weight

        def hook(gradient):
            self.backward_calls += 1
            if always or self.backward_calls == 1:
                result = gradient.clone()
                result.reshape(-1)[0] = float("inf")
                return result
            return gradient

        return parameter.register_hook(hook)


def make_optimizers(model):
    optimizers = [
        CountingSGD(model.detector.parameters(), lr=.025, momentum=.9, weight_decay=.001),
        CountingAdamW(model.auxiliary.parameters(), lr=.002, weight_decay=.01),
    ]
    # Populate both momentum and Adam state. Atomicity must preserve existing
    # state, rather than only leaving two initially empty dictionaries unchanged.
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, .125)
    for optimizer in optimizers:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer.step_calls = 0
    return optimizers


def make_scaler(scale, growth_tracker=0):
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=scale,
                                      growth_factor=2., backoff_factor=.5,
                                      growth_interval=2000)
    state = scaler.state_dict()
    state["_growth_tracker"] = growth_tracker
    scaler.load_state_dict(state)
    return scaler


class TrainUpdateTests(unittest.TestCase):
    def setUp(self):
        TRAIN.set_seed(71903)
        self.temp = tempfile.TemporaryDirectory(prefix="formal_amp_unit_")
        self.output = Path(self.temp.name)
        self.cfg = {"amp": True, "accumulation_steps": 2,
                    "gradient_clip_norm": .2, "amp_max_retries": 2}
        self.source_loader, self.target_loader = CountingLoader("source"), CountingLoader("target")
        self.batches = [(next(self.source_loader), next(self.target_loader))
                        for _ in range(self.cfg["accumulation_steps"])]

    def tearDown(self):
        self.temp.cleanup()

    def assert_tree_equal(self, actual, expected, path="root"):
        if torch.is_tensor(expected):
            self.assertTrue(torch.is_tensor(actual), path)
            self.assertEqual(actual.dtype, expected.dtype, path)
            self.assertEqual(tuple(actual.shape), tuple(expected.shape), path)
            self.assertTrue(torch.equal(actual, expected), "Tensor mismatch: " + path)
        elif isinstance(expected, np.ndarray):
            self.assertTrue(np.array_equal(actual, expected), path)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected), path)
            for key in expected:
                self.assert_tree_equal(actual[key], expected[key], path + "." + str(key))
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected), path)
            for index, (left, right) in enumerate(zip(actual, expected)):
                self.assert_tree_equal(left, right, path + "[%d]" % index)
        else:
            self.assertEqual(actual, expected, path)

    def call_update(self, model, optimizers, scaler):
        return TRAIN.train_update(model, self.batches, optimizers, scaler, self.cfg,
                                  progress=.337, step=8424, output=self.output)

    def assert_data_consumed_once(self):
        self.assertEqual(self.source_loader.consumed, 2)
        self.assertEqual(self.target_loader.consumed, 2)

    def overflow_rows(self):
        path = self.output / "amp_overflow_events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def once_overflow_matches_clean(self, optimizer_index):
        retry_model = DummyModel().cuda()
        retry_optimizers = make_optimizers(retry_model)
        initial_model = copy.deepcopy(retry_model.state_dict())
        initial_optimizers = [copy.deepcopy(opt.state_dict()) for opt in retry_optimizers]
        clean_model = DummyModel().cuda()
        clean_optimizers = make_optimizers(clean_model)
        clean_model.load_state_dict(initial_model)
        for opt, state in zip(clean_optimizers, initial_optimizers):
            opt.load_state_dict(copy.deepcopy(state))
        # A real overflow must reset an already advancing growth tracker.
        retry_scaler = make_scaler(1024., growth_tracker=11)
        clean_scaler = make_scaler(512., growth_tracker=0)
        before_rng = copy.deepcopy(TRAIN.rng_state())
        handle = retry_model.inject_gradient_inf(optimizer_index)
        try:
            retried = self.call_update(retry_model, retry_optimizers, retry_scaler)
        finally:
            handle.remove()
        after_retry_rng = TRAIN.rng_state()
        self.assertEqual(retried["retries"], 1)
        self.assertTrue(retried["did_step"])
        self.assertEqual([opt.step_calls for opt in retry_optimizers], [1, 1])
        self.assertEqual(retry_scaler.get_scale(), 512.)
        self.assertEqual(len(retry_model.trace), 4)
        self.assert_tree_equal(retry_model.trace[:2], retry_model.trace[2:])
        self.assertEqual(float(retry_model.running[0]), 2.)
        rows = self.overflow_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attempt"], 1)
        self.assertEqual(rows[0]["old_scale"], 1024.)
        self.assertEqual(rows[0]["new_scale"], 512.)
        self.assertFalse(rows[0]["did_step"])
        TRAIN.restore_rng(before_rng)
        clean = self.call_update(clean_model, clean_optimizers, clean_scaler)
        self.assertEqual(clean["retries"], 0)
        self.assertEqual([opt.step_calls for opt in clean_optimizers], [1, 1])
        self.assert_tree_equal(retry_model.trace[2:], clean_model.trace)
        self.assert_tree_equal(retry_model.state_dict(), clean_model.state_dict(), "model")
        for index, (retry_opt, clean_opt) in enumerate(zip(retry_optimizers, clean_optimizers)):
            self.assert_tree_equal(retry_opt.state_dict(), clean_opt.state_dict(), "optimizer_%d" % index)
        self.assert_tree_equal(retry_scaler.state_dict(), clean_scaler.state_dict(), "scaler")
        self.assert_tree_equal(after_retry_rng, TRAIN.rng_state(), "rng_after_success")
        self.assert_tree_equal(retried["losses"], clean["losses"], "successful_losses")
        self.assert_tree_equal(retried["density_counts"], clean["density_counts"])
        self.assertEqual(retried["density_counts"], {"0.005": 1, "0.01": 1})
        self.assertEqual(len(retried["support_per_micro"]), 2)
        self.assert_data_consumed_once()

    def test_sgd_only_overflow_retries_whole_accumulation_exactly(self):
        self.once_overflow_matches_clean(0)

    def test_adam_only_overflow_never_partially_updates_sgd(self):
        self.once_overflow_matches_clean(1)

    def test_continuous_overflow_exhaustion_never_updates_either_optimizer(self):
        model = DummyModel().cuda()
        optimizers = make_optimizers(model)
        initial_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        initial_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
        initial_optimizers = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        scaler = make_scaler(1024.)
        initial_rng = copy.deepcopy(TRAIN.rng_state())
        handle = model.inject_gradient_inf(1, always=True)
        try:
            with self.assertRaisesRegex(FloatingPointError, "after 2 retries.*no optimizer update"):
                self.call_update(model, optimizers, scaler)
        finally:
            handle.remove()
        self.assertEqual(len(model.trace), 6)  # First attempt plus two retries, each two micros.
        for start in (2, 4):
            self.assert_tree_equal(model.trace[:2], model.trace[start:start + 2])
        self.assertEqual([opt.step_calls for opt in optimizers], [0, 0])
        self.assert_tree_equal(dict(model.named_parameters()), initial_parameters, "parameters_after_failure")
        self.assert_tree_equal(dict(model.named_buffers()), initial_buffers, "buffers_after_failure")
        self.assert_tree_equal(TRAIN.rng_state(), initial_rng, "rng_after_failure")
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for index, opt in enumerate(optimizers):
            self.assert_tree_equal(opt.state_dict(), initial_optimizers[index], "optimizer_after_failure_%d" % index)
        self.assertEqual(scaler.get_scale(), 128.)
        rows = self.overflow_rows()
        self.assertEqual([row["attempt"] for row in rows], [1, 2, 3])
        self.assertTrue(all(not row["did_step"] for row in rows))
        expected_ids = [{"source": [s[0]["image_id"]], "target": [t[0]["image_id"]]}
                        for s, t in self.batches]
        self.assertTrue(all(row["image_ids"] == expected_ids for row in rows))
        self.assert_data_consumed_once()

    def test_nonfinite_forward_loss_fails_without_rescaling_or_optimizer_update(self):
        model = DummyModel().cuda()
        optimizers = make_optimizers(model)
        initial = copy.deepcopy(model.state_dict())
        initial_optimizers = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.force_forward_nonfinite = True
        scaler = make_scaler(1024.)
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite forward loss"):
            self.call_update(model, optimizers, scaler)
        self.assertEqual(len(model.trace), 1)
        self.assertEqual([opt.step_calls for opt in optimizers], [0, 0])
        self.assertEqual(scaler.get_scale(), 1024.)
        for name, parameter in model.named_parameters():
            self.assert_tree_equal(parameter, initial[name], name)
        for index, optimizer in enumerate(optimizers):
            self.assert_tree_equal(optimizer.state_dict(), initial_optimizers[index])
        self.assertEqual(self.overflow_rows(), [])
        self.assert_data_consumed_once()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional JSON test report (outside production run preferred)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required: do not interpret CPU skips as a passing AMP integration test")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TrainUpdateTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "passed": result.wasSuccessful() and result.testsRun == 4 and not result.skipped,
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "device": torch.cuda.get_device_name(0),
        "scope": "Small DummyModel integration with the actual train_update and atomic_amp_step; no detector/data",
        "comparisons": "Bitwise tensors, all four RNG streams, both optimizer states, scalar/loss/buffer state",
        "cases": [test.id() for test, _ in result.failures + result.errors],
    }
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
