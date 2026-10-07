"""CPU-only checks for portable immutable dependencies and published presets."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import checkpoint_identity
import configure as publication_configure
import continuation


HERE = Path(__file__).resolve().parent


def sha(value):
    return hashlib.sha256(value).hexdigest()


class CheckpointIdentityTests(unittest.TestCase):
    def setUp(self):
        self.sources = {
            key: {"path": "/original/" + key, "sha256": sha(key.encode("ascii"))}
            for key in checkpoint_identity.REQUIRED
        }

    def test_path_relocation_preserves_all_content_identities(self):
        moved = copy.deepcopy(self.sources)
        for key in moved:
            moved[key]["path"] = "/new/mount/" + key
        self.assertTrue(checkpoint_identity.same_sources(self.sources, moved))
        self.assertEqual(set(checkpoint_identity.fingerprints(moved)),
                         checkpoint_identity.REQUIRED)
        self.assertEqual(self.sources["source_cfg"]["path"], "/original/source_cfg")

    def test_each_changed_dependency_is_rejected(self):
        for key in checkpoint_identity.REQUIRED:
            with self.subTest(dependency=key):
                changed = copy.deepcopy(self.sources)
                changed[key]["sha256"] = sha(b"different contents")
                self.assertFalse(checkpoint_identity.same_sources(self.sources, changed))

    def test_missing_or_extra_dependency_is_rejected(self):
        for key in checkpoint_identity.REQUIRED:
            with self.subTest(missing=key):
                incomplete = copy.deepcopy(self.sources)
                del incomplete[key]
                with self.assertRaises(ValueError):
                    checkpoint_identity.same_sources(self.sources, incomplete)
        extra = dict(self.sources, unexpected={"sha256": sha(b"extra")})
        with self.assertRaises(ValueError):
            checkpoint_identity.same_sources(self.sources, extra)

    def test_malformed_sha_and_records_are_rejected(self):
        for malformed in (None, "", "a" * 63, "g" * 64, "A" * 64, 123, [],
                          "a" * 64 + "\n"):
            with self.subTest(value=malformed):
                invalid = copy.deepcopy(self.sources)
                invalid["source_checkpoint"]["sha256"] = malformed
                with self.assertRaises(ValueError):
                    checkpoint_identity.fingerprints(invalid)
        for invalid_record in (None, "not a record", {"path": "/missing/hash"}):
            invalid = dict(self.sources, source_cfg=invalid_record)
            with self.assertRaises(ValueError):
                checkpoint_identity.fingerprints(invalid)
        with self.assertRaises(ValueError):
            checkpoint_identity.fingerprints([])

    def test_explicit_random_search_initialization_remains_distinct(self):
        random_search = dict(self.sources, source_search_checkpoint=None)
        self.assertTrue(checkpoint_identity.same_sources(random_search, random_search))
        self.assertIsNone(checkpoint_identity.fingerprints(random_search)["source_search_checkpoint"])
        self.assertFalse(checkpoint_identity.same_sources(self.sources, random_search))


class PublishedConfigureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.package = self.root / "published package"
        (self.package / "configs").mkdir(parents=True)
        (self.package / "results").mkdir()
        self.templates = {}
        for preset in ("formal_25k", "early_5k", "extend_35k"):
            template = json.loads((HERE / "configs" / (preset + ".json")).read_text())
            self.templates[preset] = template
            (self.package / "configs" / (preset + ".json")).write_text(json.dumps(template))
        self.repo = self.root / "repo relocated"
        (self.repo / "detectron2").mkdir(parents=True)
        self.manifest = self.root / "manifests relocated"
        self.manifest.mkdir()
        for name in ("source_train.json", "target_train.json", "eval_mixed.json",
                     "target_image_labels.json", "data_audit.json"):
            (self.manifest / name).write_text("{}\n")
        self.assets = {}
        self.fixed_sources = {}
        for key in checkpoint_identity.REQUIRED:
            asset = self.root / (key + ".fixture")
            content = ("distinct immutable contents: " + key).encode("ascii")
            asset.write_bytes(content)
            self.assets[key] = asset
            self.fixed_sources[key] = {"path": "/old/" + key, "sha256": sha(content)}
        (self.package / "results" / "weight_manifest.json").write_text(
            json.dumps({"fixed_sources": self.fixed_sources}))
        self.patch = mock.patch.object(publication_configure, "HERE", self.package)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def args(self, preset="formal_25k"):
        return argparse.Namespace(
            preset=preset, repo_root=str(self.repo), manifest_dir=str(self.manifest),
            output_dir=str(self.root / "new outputs" / preset),
            source_checkpoint=str(self.assets["source_checkpoint"]),
            source_search_checkpoint=str(self.assets["source_search_checkpoint"]),
            rpn_checkpoint=str(self.assets["rpn_checkpoint"]),
            text_embeddings=str(self.assets["text_embeddings"]),
            source_config=str(self.assets["source_cfg"]),
            config_out=str(self.root / "generated configs" / (preset + ".json")))

    def test_presets_keep_training_budget_and_original_grl_horizon(self):
        settings = {"formal_25k": (25000, 5000, False),
                    "early_5k": (5000, 500, True),
                    "extend_35k": (35000, 1000, False)}
        for preset, (budget, period, initial_eval) in settings.items():
            with self.subTest(preset=preset):
                args = self.args(preset)
                output = publication_configure.configure(args)
                cfg = json.loads(output.read_text())
                self.assertEqual(cfg["max_steps"], budget)
                self.assertEqual(cfg["grl_schedule_steps"], 25000)
                self.assertEqual(cfg["eval_period"], period)
                self.assertEqual(cfg.get("evaluate_initial", False), initial_eval)
                self.assertEqual(cfg["repo_root"], str(self.repo.resolve()))
                self.assertEqual(cfg["manifest_dir"], str(self.manifest.resolve()))
                self.assertEqual(cfg["output_dir"], str(Path(args.output_dir).resolve()))
                for key, asset in self.assets.items():
                    self.assertEqual(cfg[key], str(asset.resolve()))
                self.assertEqual(cfg["source_checkpoint_sha256"],
                                 self.fixed_sources["source_checkpoint"]["sha256"])
                for key in ("detector_lr", "aux_lr", "warmup_steps", "seed",
                            "microbatch_per_domain", "traditional_domain_weight"):
                    self.assertEqual(cfg[key], self.templates[preset][key])
                for key in ("parent_run", "parent_eval_metrics", "parent_inference_checkpoint"):
                    self.assertNotIn(key, cfg)

    def test_every_asset_hash_is_checked_before_writing_config(self):
        for key, asset in self.assets.items():
            with self.subTest(dependency=key):
                original = asset.read_bytes()
                asset.write_bytes(original + b"changed")
                try:
                    with self.assertRaisesRegex(ValueError, "immutable source asset: " + key):
                        publication_configure.configure(self.args())
                    self.assertFalse(Path(self.args().config_out).exists())
                finally:
                    asset.write_bytes(original)

    def test_identical_config_can_be_regenerated(self):
        args = self.args()
        output = publication_configure.configure(args)
        original = output.read_bytes()
        self.assertEqual(publication_configure.configure(args), output)
        self.assertEqual(output.read_bytes(), original)

    def test_different_existing_config_is_preserved(self):
        args = self.args()
        output = publication_configure.configure(args)
        existing = json.loads(output.read_text())
        existing["detector_lr"] = 999.0
        output.write_text(json.dumps(existing))
        protected_bytes = output.read_bytes()
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            publication_configure.configure(args)
        self.assertEqual(output.read_bytes(), protected_bytes)

    def test_missing_manifest_is_rejected_before_writing_config(self):
        (self.manifest / "eval_mixed.json").unlink()
        with self.assertRaisesRegex(AssertionError, "Missing manifest: eval_mixed.json"):
            publication_configure.configure(self.args())
        self.assertFalse(Path(self.args().config_out).exists())


class VerifiedResumeRelocationTests(unittest.TestCase):
    PATH_KEYS = ("repo_root", "source_cfg", "source_checkpoint", "rpn_checkpoint",
                 "text_embeddings", "source_search_checkpoint", "manifest_dir")

    def setUp(self):
        self.old = json.loads((HERE / "configs" / "formal_25k.json").read_text())
        self.old["manifest_sha256"] = {
            name: sha(name.encode("ascii")) for name in
            ("source_train.json", "target_train.json", "target_image_labels.json", "eval_mixed.json")
        }
        self.current = copy.deepcopy(self.old)
        self.current.update(max_steps=35000, extension_from_step=25000, eval_period=1000)
        for key in self.PATH_KEYS:
            self.current[key] = "/relocated/" + key

    def test_unverified_resume_keeps_path_guards(self):
        with self.assertRaisesRegex(ValueError, "Resume config changed"):
            continuation.validate_resume(self.old, self.current, 25000)
        with self.assertRaisesRegex(ValueError, "Resume config changed"):
            continuation.validate_resume(self.old, self.current, 25000, verified_sources=False)

    def test_verified_identical_assets_allow_path_relocation(self):
        before_old, before_current = copy.deepcopy(self.old), copy.deepcopy(self.current)
        audit = continuation.validate_resume(self.old, self.current, 25000, verified_sources=True)
        self.assertTrue(audit["budget_extended"])
        self.assertEqual(audit["grl_schedule_steps"], 25000)
        self.assertEqual(audit["grl_progress"], 1.0)
        self.assertEqual(self.old, before_old)
        self.assertEqual(self.current, before_current)

    def test_verified_assets_do_not_allow_changed_manifest_bytes(self):
        self.current["manifest_sha256"]["target_train.json"] = sha(b"new image paths")
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            continuation.validate_resume(self.old, self.current, 25000, verified_sources=True)

    def test_verified_assets_do_not_bypass_training_or_content_guards(self):
        changes = {"detector_lr": 0.5, "microbatch_per_domain": 8,
                   "traditional_domain_weight": 99, "grl_schedule_steps": 35000,
                   "source_checkpoint_sha256": sha(b"changed detector")}
        for key, value in changes.items():
            with self.subTest(setting=key):
                altered = dict(self.current, **{key: value})
                with self.assertRaisesRegex(ValueError, key):
                    continuation.validate_resume(self.old, altered, 25000, verified_sources=True)


if __name__ == "__main__":
    unittest.main()
