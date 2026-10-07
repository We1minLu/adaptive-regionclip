"""Boundary tests: UDA loaders cannot receive target annotation-derived labels."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from data import DetectionDataset, IMAGE_FIELDS, validate_target_record
from prepare_uda import prepare
from continuation import validate_resume, validate_evaluation_supervision


def target_record():
    return dict(file_name='/unused.png', image_id='scene_foggy_beta_0.005',
                scene_id='scene', domain='target', beta='0.005', height=1024, width=2048)


class UDABoundaryTests(unittest.TestCase):
    def test_uda_requires_label_free_target_schema(self):
        clean = target_record()
        validate_target_record(clean, False)
        for field in ('image_labels', 'annotations', 'annotation_file', 'gt_boxes'):
            with self.assertRaises(ValueError):
                validate_target_record(dict(clean, **{field: []}), False)
        with self.assertRaises(ValueError):
            validate_target_record(clean, True)

    def test_weakly_supervised_schema_is_unchanged(self):
        record = dict(target_record(), image_labels=[0, 0, 1, 0, 0, 0, 0, 0])
        validate_target_record(record)
        DetectionDataset([record], 'target', True)
        with self.assertRaises(ValueError):
            DetectionDataset([record], 'target', True, target_image_labels_allowed=False)

    def test_uda_image_loader_emits_no_class_metadata(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'image.png'
            Image.new('RGB', (2048, 1024)).save(path)
            record = dict(target_record(), file_name=str(path))
            dataset = DetectionDataset([record], 'target', True, target_image_labels_allowed=False)
            sample = dataset[(0, True)]
            self.assertEqual(set(sample), IMAGE_FIELDS | {'image'})
            self.assertEqual(tuple(sample['image'].shape), (3, 1024, 2048))

    def test_materialization_preserves_all_image_data_and_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / 'original'
            original.mkdir()
            source = [dict(scene_id='source', annotations=[{'category_id': 2}])]
            evaluation = [dict(target_record(), annotations=[{'category_id': 2}])]
            targets = [dict(target_record(), image_labels=[0, 0, 1, 0, 0, 0, 0, 0])]
            datasets = {'source_train.json': source, 'target_train.json': targets,
                        'eval_mixed.json': evaluation}
            hashes = {}
            for name, value in datasets.items():
                content = json.dumps(value).encode('utf8')
                (original / name).write_bytes(content)
                hashes[name] = hashlib.sha256(content).hexdigest()
            (original / 'data_audit.json').write_text(json.dumps(dict(manifest_sha256=hashes,
                target_presence_counts_by_density={'0.005': {'car': 1}})))
            (original / 'target_image_labels.json').write_text('must not be copied')
            audit = prepare(original, root / 'uda')
            clean = json.loads((root / 'uda/target_train.json').read_text())
            self.assertEqual(clean, [target_record()])
            self.assertFalse(audit['target_image_labels_allowed'])
            self.assertNotIn('target_presence_counts_by_density', audit)
            self.assertFalse((root / 'uda/target_image_labels.json').exists())
            for name in ('source_train.json', 'eval_mixed.json'):
                self.assertEqual((root / 'uda' / name).read_bytes(), (original / name).read_bytes())
            self.assertEqual(audit, prepare(original, root / 'uda'))

    def test_resume_cannot_cross_supervision_regimes(self):
        cfg = json.loads((Path(__file__).parent / 'configs/formal_25k.json').read_text())
        uda = copy.deepcopy(cfg)
        uda['target_image_labels_allowed'] = False
        with self.assertRaisesRegex(ValueError, 'target_image_labels_allowed'):
            validate_resume(cfg, uda, 1000)

    def test_evaluation_cannot_mislabel_old_checkpoint_as_uda(self):
        uda = dict(target_image_labels_allowed=False)
        validate_evaluation_supervision(uda, uda)
        for old in ({}, {'target_image_labels_allowed': True}, {'target_image_labels_allowed': 0}):
            with self.assertRaisesRegex(ValueError, 'supervision differs'):
                validate_evaluation_supervision(old, uda)


if __name__ == '__main__':
    unittest.main()
