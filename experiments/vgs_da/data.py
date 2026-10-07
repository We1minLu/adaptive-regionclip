"""Formal data loaders with an explicit target-box exclusion boundary.

Sampler-controlled flips make transforms independent of worker scheduling and
reproducible after resuming with the number of already-consumed domain samples.
The mixed evaluation loader drops annotations before reading image tensors.
"""
import json
import random
from collections import Counter
from pathlib import Path

CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
BETAS = ('0.005', '0.01', '0.02')
TARGET_FIELDS = {'file_name', 'image_id', 'scene_id', 'image_labels', 'domain', 'beta', 'height', 'width'}
IMAGE_FIELDS = {'file_name', 'image_id', 'scene_id', 'domain', 'beta', 'height', 'width'}


def read_manifest(path):
    records = json.loads(Path(path).read_text(encoding='utf8'))
    if not isinstance(records, list) or not records:
        raise ValueError('Expected a nonempty list manifest: ' + str(path))
    if len({r['image_id'] for r in records}) != len(records):
        raise ValueError('Duplicate image IDs in ' + str(path))
    return records


def validate_target_record(record):
    if set(record) != TARGET_FIELDS:
        raise ValueError('Target record has forbidden/missing fields: ' + repr(set(record) ^ TARGET_FIELDS))
    if record['domain'] != 'target' or record['beta'] not in BETAS:
        raise ValueError('Invalid target domain or density')
    if record['image_id'] != record['scene_id'] + '_foggy_beta_' + record['beta']:
        raise ValueError('Target scene/density ID mismatch')
    if len(record['image_labels']) != 8 or any(x not in (0, 1) for x in record['image_labels']):
        raise ValueError('Target labels must be eight binary image-presence values')
    if (record['height'], record['width']) != (1024, 2048):
        raise ValueError('Fixed 1024x2048 input required')


class DetectionDataset:
    def __init__(self, records, domain, training=True):
        self.records, self.domain, self.training = records, domain, bool(training)
        if domain == 'target' and training:
            for record in records:
                validate_target_record(record)
        elif domain == 'source' and training:
            for record in records:
                if record['domain'] != 'source' or 'annotations' not in record:
                    raise ValueError('Source records must contain source GT')
        elif not training:
            # Eval annotations are not retained in the image dataset at all.
            self.records = [{k: record[k] for k in IMAGE_FIELDS} for record in records]
        else:
            raise ValueError('Unknown dataset domain')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        import numpy as np
        import torch
        from PIL import Image
        if isinstance(item, tuple):
            index, flip = item
        else:
            index, flip = item, False
        if flip and not self.training:
            raise ValueError('Validation augmentation is forbidden')
        record = self.records[index]
        with Image.open(record['file_name']) as image:
            array = np.asarray(image.convert('RGB'))
        if array.shape != (1024, 2048, 3):
            raise ValueError('Unexpected image shape: ' + record['image_id'] + str(array.shape))
        if flip:
            array = array[:, ::-1, :]
        output = {k: record[k] for k in IMAGE_FIELDS}
        output['image'] = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        if not self.training:
            return output
        if self.domain == 'target':
            output['image_labels'] = torch.tensor(record['image_labels'], dtype=torch.float32)
            # No annotations, Instances, boxes, proposals, or annotation paths.
            assert set(output) == IMAGE_FIELDS | {'image', 'image_labels'}
            return output
        from detectron2.structures import Boxes, Instances
        annotations = record['annotations']
        boxes = torch.tensor([a['bbox'] for a in annotations], dtype=torch.float32).reshape(-1, 4)
        classes = torch.tensor([a['category_id'] for a in annotations], dtype=torch.int64)
        if flip and len(boxes):
            boxes[:, [0, 2]] = 2048 - boxes[:, [2, 0]]
        instances = Instances((1024, 2048))
        instances.gt_boxes, instances.gt_classes = Boxes(boxes), classes
        presence = torch.zeros(8, dtype=torch.float32)
        if len(classes):
            presence[classes.unique()] = 1
        output['instances'], output['image_labels'] = instances, presence
        return output


def shuffled_forever(indices, rng):
    indices = list(indices)
    if not indices:
        raise ValueError('Cannot sample an empty index set')
    while True:
        order = list(indices)
        rng.shuffle(order)
        yield from order


class InfiniteSourceSampler:
    def __init__(self, size, seed, start_index=0):
        if size <= 0 or start_index < 0:
            raise ValueError('Invalid source sampler size/start')
        self.size, self.seed, self.start_index = int(size), int(seed), int(start_index)

    def __iter__(self):
        choices = shuffled_forever(range(self.size), random.Random(self.seed))
        flips = random.Random(self.seed + 9137)
        for count, index in enumerate(choices):
            flip = flips.random() < .5
            if count >= self.start_index:
                yield index, flip


class EqualDensitySampler:
    def __init__(self, records, seed, start_index=0):
        self.indices = {b: [i for i, r in enumerate(records) if r['beta'] == b] for b in BETAS}
        if len({len(v) for v in self.indices.values()}) != 1 or not all(self.indices.values()):
            raise ValueError('Equal density sampler requires a nonempty complete triplet manifest')
        scenes = {b: {records[i]['scene_id'] for i in self.indices[b]} for b in BETAS}
        if not all(scenes[b] == scenes[BETAS[0]] for b in BETAS):
            raise ValueError('Fog density base scenes differ')
        if start_index < 0:
            raise ValueError('Invalid target sampler start')
        self.seed, self.start_index = int(seed), int(start_index)

    def __iter__(self):
        streams = {b: shuffled_forever(self.indices[b], random.Random(self.seed + 1009 * (i + 1)))
                   for i, b in enumerate(BETAS)}
        density_rng, flips = random.Random(self.seed), random.Random(self.seed + 9137)
        count = 0
        while True:
            order = list(BETAS)
            density_rng.shuffle(order)
            for beta in order:
                sample = next(streams[beta]), flips.random() < .5
                if count >= self.start_index:
                    yield sample
                count += 1


def list_collate(batch):
    return batch


def build_loaders(manifest_dir, batch_size_source=2, batch_size_target=2, num_workers=2,
                  seed=20261004, source_samples_consumed=0, target_samples_consumed=0):
    """Return infinite (source_loader, target_loader); samples are list-of-dicts.

    Resume counts refer to samples consumed by training, excluding prefetch.
    No eval manifest is opened by this function.
    """
    import torch
    from torch.utils.data import DataLoader
    root = Path(manifest_dir)
    source = DetectionDataset(read_manifest(root / 'source_train.json'), 'source', True)
    target = DetectionDataset(read_manifest(root / 'target_train.json'), 'target', True)
    source_sampler = InfiniteSourceSampler(len(source), seed + 101, source_samples_consumed)
    target_sampler = EqualDensitySampler(target.records, seed + 202, target_samples_consumed)
    options = dict(num_workers=num_workers, collate_fn=list_collate, pin_memory=True)
    if num_workers:
        options['persistent_workers'] = True
        options['prefetch_factor'] = 1
    # Creating an iterator asks PyTorch for a worker/base seed, including with
    # zero workers. Keep that draw away from the model's checkpointed CPU RNG.
    source_generator = torch.Generator().manual_seed(seed + 301)
    target_generator = torch.Generator().manual_seed(seed + 302)
    return (DataLoader(source, batch_size=batch_size_source, sampler=source_sampler,
                       generator=source_generator, **options),
            DataLoader(target, batch_size=batch_size_target, sampler=target_sampler,
                       generator=target_generator, **options))


def make_eval_loader(manifest_dir, batch_size=1, num_workers=2, beta=None):
    import torch
    from torch.utils.data import DataLoader
    records = read_manifest(Path(manifest_dir) / 'eval_mixed.json')
    if beta is not None:
        if beta not in BETAS:
            raise ValueError('Unknown fog density')
        records = [r for r in records if r['beta'] == beta]
    dataset = DetectionDataset(records, 'target', False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      collate_fn=list_collate, pin_memory=True,
                      generator=torch.Generator().manual_seed(918273))


def self_test():
    """CPU-only checks for density balance, resume, leakage rejection, IDs."""
    import itertools
    records = [{'file_name': '/dummy.png', 'image_id': 'scene_%d_leftImg8bit_foggy_beta_%s' % (j, b),
                'scene_id': 'scene_%d_leftImg8bit' % j, 'beta': b, 'height': 1024, 'width': 2048,
                'domain': 'target', 'image_labels': [1, 0, 1, 0, 0, 0, 0, 0]}
               for j in range(7) for b in BETAS]
    for r in records:
        validate_target_record(r)
    sequence = list(itertools.islice(iter(EqualDensitySampler(records, 12)), 210))
    for start in range(0, 210, 3):
        assert Counter(records[i]['beta'] for i, _ in sequence[start:start+3]) == Counter(BETAS)
    for b in BETAS:
        first_epoch = [i for i, _ in sequence[:21] if records[i]['beta'] == b]
        assert len(set(first_epoch)) == 7
    resumed = list(itertools.islice(iter(EqualDensitySampler(records, 12, 47)), 163))
    assert sequence[47:] == resumed
    source = list(itertools.islice(iter(InfiniteSourceSampler(7, 12)), 100))
    assert source[23:] == list(itertools.islice(iter(InfiniteSourceSampler(7, 12, 23)), 77))
    for field in ['annotations', 'instances', 'gt_boxes', 'annotation_file', 'proposals']:
        invalid = dict(records[0], **{field: []})
        try:
            validate_target_record(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Target leakage field accepted: ' + field)
    evaluation = DetectionDataset([dict(records[0], annotations=[{'bbox': [1, 2, 3, 4]}])], 'target', False)
    assert set(evaluation.records[0]) == IMAGE_FIELDS
    return {'passed': True, 'checks': ['equal-density blocks', 'without-replacement scene epochs',
            'target/source sampler resume', 'target GT field rejection', 'eval image dataset strips GT']}


if __name__ == '__main__':
    print(json.dumps(self_test(), indent=2))
