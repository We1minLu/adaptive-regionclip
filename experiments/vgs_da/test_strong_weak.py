"""CPU checks for AT view semantics, deterministic resume, and target isolation."""
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
from PIL import Image, ImageFilter
import torch
from torchvision import transforms as T

import augmentation
import data


def take(sampler, n):
    return list(itertools.islice(iter(sampler), n))


def target_records(file_name='/unused.png', scenes=3):
    return [dict(file_name=file_name, image_id='s%d_foggy_beta_%s' % (i, beta),
                 scene_id='s%d' % i, domain='target', beta=beta,
                 height=1024, width=2048)
            for i in range(scenes) for beta in data.BETAS]


class SamplerTests(unittest.TestCase):
    def test_source_order_and_flip_are_unchanged(self):
        old = take(data.InfiniteSourceSampler(7, 42), 101)
        new = take(data.InfiniteSourceSampler(7, 42, enable_aug_seed=True), 101)
        self.assertEqual(old, [x[:2] for x in new])
        self.assertEqual(len(set(x[2] for x in new)), len(new))
        self.assertEqual(new[37:], take(data.InfiniteSourceSampler(
            7, 42, start_index=37, enable_aug_seed=True), 64))

    def test_target_order_density_flip_and_resume_are_unchanged(self):
        records = target_records()
        old = take(data.EqualDensitySampler(records, 42), 101)
        new = take(data.EqualDensitySampler(records, 42, enable_aug_seed=True), 101)
        self.assertEqual(old, [x[:2] for x in new])
        self.assertEqual(new[37:], take(data.EqualDensitySampler(
            records, 42, start_index=37, enable_aug_seed=True), 64))
        self.assertEqual(len(set(x[2] for x in new)), len(new))

    def test_seed_depends_on_domain_sample_count_and_run_seed(self):
        seeds = {data.augmentation_seed(s, d, n)
                 for s in (42, 43) for d in ('source', 'target') for n in (0, 1)}
        self.assertEqual(len(seeds), 8)


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.fromarray(np.arange(64 * 128 * 3, dtype=np.uint8).reshape(64, 128, 3))
        self.transform = augmentation.build_strong_augmentation()

    def test_exact_reference_pipeline_parameters(self):
        transforms = self.transform.transforms
        self.assertEqual(len(transforms), 4)
        self.assertIsInstance(transforms[0], T.RandomApply)
        self.assertEqual(transforms[0].p, .8)
        jitter = transforms[0].transforms[0]
        # torchvision 0.10 exposes these ranges as lists; newer versions use
        # tuples. Assert parameter values independently of container spelling.
        self.assertEqual(tuple(jitter.brightness), (.6, 1.4))
        self.assertEqual(tuple(jitter.contrast), (.6, 1.4))
        self.assertEqual(tuple(jitter.saturation), (.6, 1.4))
        self.assertEqual(tuple(jitter.hue), (-.1, .1))
        self.assertEqual(transforms[1].p, .2)
        self.assertEqual(transforms[2].p, .5)
        self.assertEqual(tuple(transforms[2].transforms[0].sigma), (.1, 2.0))
        inner = transforms[3].transforms
        self.assertIsInstance(inner[0], T.ToTensor)
        self.assertIsInstance(inner[-1], T.ToPILImage)
        expected = [(.7, (.05, .2), (.3, 3.3)),
                    (.5, (.02, .2), (.1, 6)),
                    (.3, (.02, .2), (.05, 8))]
        for erase, (probability, scale, ratio) in zip(inner[1:-1], expected):
            self.assertIsInstance(erase, T.RandomErasing)
            self.assertEqual((erase.p, tuple(erase.scale), tuple(erase.ratio), erase.value),
                             (probability, scale, ratio, 'random'))

    def test_sample_seed_reproduces_pixels_independent_of_global_rng(self):
        first = augmentation.apply_strong_augmentation(self.image, 32, self.transform)
        random.random()
        torch.rand(8)
        second = augmentation.apply_strong_augmentation(self.image, 32, self.transform)
        self.assertTrue(np.array_equal(np.asarray(first), np.asarray(second)))
        self.assertEqual(first.size, self.image.size)
        self.assertFalse(np.array_equal(np.asarray(first), np.asarray(self.image)))

    def test_python_and_cpu_rng_restored_without_cuda_access(self):
        py_before = random.getstate()
        torch_before = torch.random.get_rng_state()
        with mock.patch.object(torch.cuda, 'manual_seed_all', side_effect=AssertionError('CUDA seed')), \
                mock.patch.object(torch.cuda, 'get_rng_state_all', side_effect=AssertionError('CUDA RNG')):
            augmentation.apply_strong_augmentation(self.image, 64, self.transform)
        self.assertEqual(random.getstate(), py_before)
        self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))

    def test_rng_restored_on_failure(self):
        def fail(image):
            random.random()
            torch.rand(1)
            raise RuntimeError('transform failed')
        before_python, before_torch = random.getstate(), torch.random.get_rng_state()
        with self.assertRaisesRegex(RuntimeError, 'transform failed'):
            augmentation.apply_strong_augmentation(self.image, 8, fail)
        self.assertEqual(before_python, random.getstate())
        self.assertTrue(torch.equal(before_torch, torch.random.get_rng_state()))

    def test_blur_matches_pil_reference_and_requires_true_rgb(self):
        with mock.patch('augmentation.random.uniform', return_value=.7):
            actual = augmentation.GaussianBlur()(self.image)
        expected = self.image.filter(ImageFilter.GaussianBlur(radius=.7))
        self.assertTrue(np.array_equal(np.asarray(actual), np.asarray(expected)))
        with self.assertRaisesRegex(ValueError, 'RGB'):
            augmentation.apply_strong_augmentation(self.image.convert('L'), 1, self.transform)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.path = cls.root / 'image.png'
        cls.array = np.zeros((1024, 2048, 3), dtype=np.uint8)
        cls.array[:, :, 0] = np.arange(2048, dtype=np.uint16)[None, :] % 256
        cls.array[:, :, 1] = 100
        cls.array[:, :, 2] = 200
        Image.fromarray(cls.array).save(cls.path)
        cls.records = target_records(str(cls.path), scenes=1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def dataset(self, enabled=True):
        return data.DetectionDataset(self.records, 'target', True,
                                     target_image_labels_allowed=False,
                                     strong_weak_enabled=enabled)

    def test_baseline_output_schema_and_pixels_unchanged(self):
        sample = self.dataset(False)[(0, True)]
        self.assertEqual(set(sample), data.IMAGE_FIELDS | {'image'})
        expected = torch.from_numpy(self.array[:, ::-1, :].transpose(2, 0, 1).copy())
        self.assertTrue(torch.equal(sample['image'], expected))

    def test_target_two_views_share_geometry_and_have_no_gt(self):
        sample = self.dataset()[(0, True, 1234)]
        self.assertEqual(set(sample), data.IMAGE_FIELDS | {'image', 'image_weak'})
        expected = torch.from_numpy(self.array[:, ::-1, :].transpose(2, 0, 1).copy())
        self.assertTrue(torch.equal(sample['image_weak'], expected))
        self.assertEqual(sample['image'].shape, sample['image_weak'].shape)
        self.assertEqual(sample['image'].dtype, torch.uint8)
        self.assertFalse(torch.equal(sample['image'], sample['image_weak']))

    def test_eval_never_augments_or_retains_annotation_fields(self):
        annotated = [dict(self.records[0], image_labels=[1] * 8, annotations=[{'bbox': [1, 2, 3, 4]}])]
        with mock.patch('augmentation.build_strong_augmentation', side_effect=AssertionError('eval augmentation')):
            dataset = data.DetectionDataset(annotated, 'target', False, strong_weak_enabled=True)
            sample = dataset[0]
        self.assertEqual(set(sample), data.IMAGE_FIELDS | {'image'})
        self.assertTrue(torch.equal(sample['image'], torch.from_numpy(self.array.transpose(2, 0, 1).copy())))
        with self.assertRaises(ValueError):
            dataset[(0, True)]

    def test_training_requires_a_sample_seed(self):
        dataset = self.dataset()
        for index in (0, (0, False)):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'sampler-provided'):
                dataset[index]

    def test_target_annotation_or_presence_leakage_is_rejected(self):
        for field in ('annotations', 'instances', 'gt_boxes', 'image_labels'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                data.DetectionDataset([dict(self.records[0], **{field: []})], 'target', True,
                                      target_image_labels_allowed=False, strong_weak_enabled=True)

    def test_source_gt_uses_shared_flip_and_only_one_instance_set(self):
        class Boxes:
            def __init__(self, tensor):
                self.tensor = tensor
        class Instances:
            def __init__(self, image_size):
                self.image_size = image_size
        fake = types.ModuleType('detectron2.structures')
        fake.Boxes, fake.Instances = Boxes, Instances
        source = dict(self.records[0], domain='source', annotations=[{'bbox': [10, 20, 40, 60], 'category_id': 2}])
        with mock.patch.dict(sys.modules, {'detectron2.structures': fake}):
            sample = data.DetectionDataset([source], 'source', True, strong_weak_enabled=True)[(0, True, 1234)]
        self.assertEqual(sample['instances'].gt_boxes.tensor.tolist(), [[2008., 20., 2038., 60.]])
        self.assertEqual(sample['image_labels'].tolist(), [0., 0., 1., 0., 0., 0., 0., 0.])
        self.assertIn('image_weak', sample)
        self.assertNotIn('instances_weak', sample)

    def test_loader_seed_wiring_and_resumed_pixels(self):
        source = dict(self.records[0], domain='source', annotations=[])
        (self.root / 'source_train.json').write_text(json.dumps([source]), encoding='utf8')
        (self.root / 'target_train.json').write_text(json.dumps(self.records), encoding='utf8')
        options = dict(batch_size_source=1, batch_size_target=1, num_workers=0,
                       seed=42, target_image_labels_allowed=False, strong_weak_enabled=True)
        source_loader, target_loader = data.build_loaders(self.root, **options)
        self.assertTrue(source_loader.sampler.enable_aug_seed)
        self.assertTrue(source_loader.dataset.strong_weak_enabled)
        original = iter(target_loader)
        next(original)
        expected = next(original)[0]
        _, resumed = data.build_loaders(self.root, target_samples_consumed=1, **options)
        actual = next(iter(resumed))[0]
        self.assertEqual(actual['image_id'], expected['image_id'])
        self.assertTrue(torch.equal(actual['image'], expected['image']))
        self.assertTrue(torch.equal(actual['image_weak'], expected['image_weak']))

    def test_worker_count_does_not_change_views(self):
        from torch.utils.data import DataLoader
        dataset = self.dataset()
        samples = take(data.EqualDensitySampler(self.records, 73, enable_aug_seed=True), 2)
        serial = [dataset[item] for item in samples]
        loader = DataLoader(dataset, sampler=samples, batch_size=1, num_workers=2,
                            collate_fn=data.list_collate,
                            generator=torch.Generator().manual_seed(789))
        parallel = [batch[0] for batch in loader]
        self.assertEqual(len(serial), len(parallel))
        for expected, actual in zip(serial, parallel):
            self.assertEqual(actual['image_id'], expected['image_id'])
            self.assertTrue(torch.equal(actual['image'], expected['image']))
            self.assertTrue(torch.equal(actual['image_weak'], expected['image_weak']))


if __name__ == '__main__':
    unittest.main()
