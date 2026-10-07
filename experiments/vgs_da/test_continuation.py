"""Pure CPU contract checks: python -m unittest test_continuation -v."""

import copy
import unittest

from continuation import UNCHANGED_KEYS, format_evaluation, grl_progress, validate_resume


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.old = {key: "fixed-" + key for key in UNCHANGED_KEYS}
        self.old.update(max_steps=25000, detector_lr=0.0005, aux_lr=0.0002,
                        warmup_steps=1000, warmup_factor=0.01,
                        lr_schedule="linear_warmup_then_constant",
                        manifest_sha256={"source_train.json": "a", "target_train.json": "b"})
        self.new = copy.deepcopy(self.old)
        self.new.update(max_steps=35000, extension_from_step=25000,
                        grl_schedule_steps=25000, eval_period=1000,
                        output_dir="new-run", experiment="extension")

    def test_extension_preserves_lr_and_grl(self):
        old_before, new_before = copy.deepcopy(self.old), copy.deepcopy(self.new)
        audit = validate_resume(self.old, self.new, 25000)
        self.assertTrue(audit["budget_extended"])
        for step in (25000, 25001, 26000, 35000):
            self.assertEqual(grl_progress(step, self.new), 1.0)
            warm = min(1.0, step / max(self.new["warmup_steps"], 1))
            factor = self.new["warmup_factor"] + (1 - self.new["warmup_factor"]) * warm
            self.assertEqual(self.new["detector_lr"] * factor, self.old["detector_lr"])
            self.assertEqual(self.new["aux_lr"] * factor, self.old["aux_lr"])
        self.assertEqual(self.old, old_before)
        self.assertEqual(self.new, new_before)

    def test_existing_resume_and_extension_checkpoint_resume(self):
        self.assertFalse(validate_resume(self.old, self.old, 8000)["budget_extended"])
        resumed = copy.deepcopy(self.new)
        resumed.update(log_period=20, eval_period=1000, checkpoint_period=1000)
        self.assertFalse(validate_resume(self.new, resumed, 27000)["budget_extended"])
        self.assertEqual(grl_progress(27000, resumed), 1.0)

    def test_all_immutable_settings_are_guarded(self):
        for key in UNCHANGED_KEYS:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.new)
                changed[key] = "changed"
                with self.assertRaisesRegex(ValueError, key):
                    validate_resume(self.old, changed, 25000)

    def test_extension_requires_explicit_boundary(self):
        for changes, step in (({"extension_from_step": None}, 25000),
                              ({"extension_from_step": 24000}, 24000),
                              ({"extension_from_step": 25000.0}, 25000),
                              ({"max_steps": 24000}, 25000),
                              ({}, 24000), ({}, 25001)):
            with self.subTest(changes=changes, step=step):
                changed = dict(self.new, **changes)
                with self.assertRaises(ValueError):
                    validate_resume(self.old, changed, step)

    def test_extended_budget_cannot_reset_grl_ramp(self):
        for horizon in (35000, 0, -1):
            changed = dict(self.new, grl_schedule_steps=horizon)
            with self.assertRaises(ValueError):
                validate_resume(self.old, changed, 25000)
        changed = dict(self.new)
        del changed["grl_schedule_steps"]
        with self.assertRaisesRegex(ValueError, "grl_schedule_steps"):
            validate_resume(self.old, changed, 25000)

    def test_grl_clamp_and_original_fallback(self):
        self.assertEqual(grl_progress(-1, self.old), 0.0)
        self.assertEqual(grl_progress(12500, self.old), 0.5)
        self.assertEqual(grl_progress(35000, self.new), 1.0)
        with self.assertRaises(ValueError):
            grl_progress(float("nan"), self.new)


class EvaluationLogTests(unittest.TestCase):
    def setUp(self):
        self.result = dict(complete=True, images=1500, expected_images=1500,
                           pooled=dict(AP50=55.98, AP75=33.73),
                           per_density={"0.005": dict(AP50=60.80),
                                        "0.01": dict(AP50=57.33),
                                        "0.02": dict(AP50=49.12)})

    def test_full_result_reports_pooled_not_mean(self):
        line = format_evaluation(26000, self.result)
        self.assertIn("scope=current status=full complete=true images=1500/1500", line)
        self.assertIn("mixed_AP50=55.9800 mixed_AP75=33.7300", line)
        self.assertIn("fog_0.02_AP50=49.1200", line)
        self.assertNotIn("\n", line)

    def test_partial_missing_or_inconsistent_metadata_is_explicit(self):
        cases = [dict(self.result, complete=False), dict(self.result, images=6),
                 dict(self.result, expected_images=6), dict(self.result, complete=1)]
        for key in ("complete", "images", "expected_images"):
            case = dict(self.result)
            del case[key]
            cases.append(case)
        for case in cases:
            with self.subTest(case=case):
                line = format_evaluation(25002, case)
                self.assertIn("status=partial complete=false", line)
                self.assertNotIn("status=full", line)

    def test_parent_reference_cannot_look_like_current_result(self):
        line = format_evaluation(25000, self.result, scope="parent_reference")
        self.assertIn("step=25000 scope=parent_reference", line)
        self.assertNotIn("scope=current", line)

    def test_missing_nonfinite_values_and_unsafe_label(self):
        line = format_evaluation(25000, {"pooled": {"AP50": float("nan")}})
        self.assertIn("status=partial", line)
        self.assertIn("mixed_AP50=NA mixed_AP75=NA", line)
        with self.assertRaises(ValueError):
            format_evaluation(25000, self.result, scope="current\nEVAL")


if __name__ == "__main__":
    unittest.main()
