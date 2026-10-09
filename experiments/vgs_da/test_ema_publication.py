"""CPU checks for publishing the exact EMA experiment with relocated assets."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import configure as publication_configure


HERE = Path(__file__).resolve().parent
ASSETS = ('source_cfg', 'source_checkpoint', 'source_search_checkpoint',
          'rpn_checkpoint', 'text_embeddings')
PATH_KEYS = set(ASSETS) | {'repo_root', 'manifest_dir', 'output_dir',
                         'source_checkpoint_sha256'}


class EmaPublishedConfigureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.package = self.root / 'package'
        (self.package / 'configs').mkdir(parents=True)
        (self.package / 'results').mkdir()
        self.template = json.loads((HERE / 'configs/ema_image_25k.json').read_text())
        (self.package / 'configs/ema_image_25k.json').write_text(json.dumps(self.template))
        self.repo = self.root / 'relocated repo'
        (self.repo / 'detectron2').mkdir(parents=True)
        self.manifests = self.root / 'unlabeled manifests'
        self.manifests.mkdir()
        for name in ('source_train.json', 'target_train.json', 'eval_mixed.json'):
            (self.manifests / name).write_text('[]\n')
        (self.manifests / 'data_audit.json').write_text(
            json.dumps({'target_image_labels_allowed': False}))
        self.assets, expected = {}, {}
        for key in ASSETS:
            path = self.root / (key + '.fixture')
            content = ('immutable fixture: ' + key).encode('ascii')
            path.write_bytes(content)
            self.assets[key] = path
            expected[key] = {'path': '/original/' + key,
                             'sha256': hashlib.sha256(content).hexdigest()}
        # The EMA experiment shares the existing source assets. It must not
        # require an unpublished EMA result or another downloaded weight file.
        (self.package / 'results/weight_manifest.json').write_text(
            json.dumps({'fixed_sources': expected}))
        self.expected = expected
        patch = mock.patch.object(publication_configure, 'HERE', self.package)
        patch.start()
        self.addCleanup(patch.stop)

    def args(self):
        return argparse.Namespace(
            preset='ema_image_25k', repo_root=str(self.repo),
            manifest_dir=str(self.manifests), output_dir=str(self.root / 'training outputs'),
            source_config=str(self.assets['source_cfg']),
            source_checkpoint=str(self.assets['source_checkpoint']),
            source_search_checkpoint=str(self.assets['source_search_checkpoint']),
            rpn_checkpoint=str(self.assets['rpn_checkpoint']),
            text_embeddings=str(self.assets['text_embeddings']),
            config_out=str(self.root / 'generated/ema.json'))

    def test_cli_accepts_ema_preset_and_passes_explicit_paths(self):
        args = self.args()
        argv = ['configure.py']
        for key, value in vars(args).items():
            argv.extend(['--' + key.replace('_', '-'), value])
        with mock.patch('sys.argv', argv), mock.patch('builtins.print'):
            publication_configure.main()
        self.assertTrue(Path(args.config_out).is_file())

    def test_relocation_preserves_every_non_path_training_setting(self):
        args = self.args()
        cfg = json.loads(publication_configure.configure(args).read_text())
        for key, value in self.template.items():
            if key not in PATH_KEYS:
                self.assertEqual(cfg[key], value, key)
        for key, path in self.assets.items():
            self.assertEqual(cfg[key], str(path.resolve()), key)
        self.assertEqual(cfg['repo_root'], str(self.repo.resolve()))
        self.assertEqual(cfg['manifest_dir'], str(self.manifests.resolve()))
        self.assertEqual(cfg['output_dir'], str(Path(args.output_dir).resolve()))
        self.assertEqual(cfg['source_checkpoint_sha256'],
                         self.expected['source_checkpoint']['sha256'])
        self.assertFalse((self.manifests / 'target_image_labels.json').exists())
        self.assertIs(cfg['target_image_labels_allowed'], False)
        settings = {
            'image_consistency_enabled': True, 'strong_weak_enabled': True,
            'semantic_teacher_mode': 'ema', 'ema_decay': 0.9996,
            'image_consistency_weight': 1.0, 'image_aggregation': 'h2fa_iir',
            'source_view': 'strong', 'evaluation_model': 'student',
            'max_steps': 25000, 'eval_period': 1000, 'grl_schedule_steps': 25000,
            'train_roi_chunk': 512, 'checkpoint_res5': False,
            'teacher_roi_chunk': 256, 'microbatch_per_domain': 2,
            'detector_lr': 0.0005, 'aux_lr': 0.0002,
            'traditional_domain_weight': 0.1, 'conditional_domain_weight': 1.0,
            'conditional_grl_max': 0.05,
        }
        for key, value in settings.items():
            self.assertEqual(cfg[key], value, key)

    def test_changed_source_asset_fails_without_materializing_config(self):
        for key, path in self.assets.items():
            with self.subTest(asset=key):
                original = path.read_bytes()
                path.write_bytes(original + b'wrong initialization')
                try:
                    with self.assertRaisesRegex(ValueError, 'immutable source asset: ' + key):
                        publication_configure.configure(self.args())
                    self.assertFalse(Path(self.args().config_out).exists())
                finally:
                    path.write_bytes(original)

    def test_conflicting_existing_ema_or_supervision_config_is_not_overwritten(self):
        args = self.args()
        path = publication_configure.configure(args)
        cfg = json.loads(path.read_text())
        for key, value in (('ema_decay', 0.99), ('target_image_labels_allowed', True),
                           ('image_consistency_enabled', False), ('train_roi_chunk', 64)):
            with self.subTest(setting=key):
                path.write_text(json.dumps(dict(cfg, **{key: value})))
                original = path.read_bytes()
                with self.assertRaisesRegex(ValueError, 'Refusing to overwrite a different config'):
                    publication_configure.configure(args)
                self.assertEqual(path.read_bytes(), original)

    def test_identical_ema_config_can_be_regenerated(self):
        path = publication_configure.configure(self.args())
        original = path.read_bytes()
        self.assertEqual(publication_configure.configure(self.args()), path)
        self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
