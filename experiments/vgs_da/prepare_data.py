"""Prepare isolated formal-run manifests without changing existing datasets.

Source training boxes retain the host's historical polygon-to-VOC convention.
Target class-presence labels are extracted separately by reading names only.
Evaluation boxes live only in eval_mixed.json, never in target_train.json.
"""
import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
BETAS = ('0.005', '0.01', '0.02')
TARGET_FIELDS = {'file_name', 'image_id', 'scene_id', 'image_labels', 'domain', 'beta', 'height', 'width'}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_locked(path, value):
    """A re-run can verify the same manifest, but cannot change a locked split."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf8')) != value:
            raise RuntimeError('Refusing to overwrite a different manifest: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    temporary.replace(path)


def scene_group(scene):
    return '_'.join(scene.split('_')[:2])


def presence_from_names(path):
    """Do not access polygons, extents, counts, or boxes for target supervision."""
    raw = json.loads(Path(path).read_text(encoding='utf8'))
    names = {obj.get('label') for obj in raw.get('objects', []) if not obj.get('deleted', False)}
    return [int(name in names) for name in CLASSES]


def conversion_module(repo_root):
    path = Path(repo_root) / 'tools/prepare_cityscapes_voc_exp1.py'
    spec = importlib.util.spec_from_file_location('_formal_historical_conversion', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.CLASS_NAMES) != CLASSES:
        raise ValueError('Historical source class order changed')
    return module, path


def source_annotations(record):
    return [{'bbox': [int(round(x)) for x in obj['bbox']], 'category_id': CLASSES.index(obj['name']),
             'bbox_mode': 0, 'difficult': 0} for obj in record['objects']]


def read_eval_xml(path):
    tree = ET.parse(str(path))
    annotations = []
    for obj in tree.findall('object'):
        name = obj.findtext('name')
        if name not in CLASSES:
            continue
        bbox = obj.find('bndbox')
        annotations.append({'bbox': [int(bbox.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')],
                            'category_id': CLASSES.index(name), 'bbox_mode': 0,
                            'difficult': int(obj.findtext('difficult', '0'))})
    return int(tree.findtext('./size/height')), int(tree.findtext('./size/width')), annotations


def prepare_manifests(repo_root='/root/regionclip/DA-Pro', output_dir=None,
                      city_root='/root/autodl-tmp/data/cityspace',
                      fog_root='/root/autodl-tmp/data/foggycity/leftImg8bit_foggy'):
    if output_dir is None:
        raise ValueError('An isolated output_dir is required')
    repo_root, output_dir, city_root, fog_root = map(Path, (repo_root, output_dir, city_root, fog_root))
    conversion, conversion_path = conversion_module(repo_root)
    source_raw, counts, groups, invalid = conversion.read_annotations(city_root / 'gtFine/train')
    train_keys = set(source_raw)
    val_jsons = sorted((city_root / 'gtFine/val').glob('*/*_gtFine_polygons.json'))
    val_keys = {p.name.replace('_gtFine_polygons.json', '_leftImg8bit') for p in val_jsons}
    if len(train_keys) != 2975 or len(val_keys) != 500:
        raise ValueError('Expected the complete 2975/500 Cityscapes annotation split')
    if train_keys & val_keys or {scene_group(k) for k in train_keys} & {scene_group(k) for k in val_keys}:
        raise ValueError('Training and validation scene/sequence leakage')

    clear_images = {p.stem: p for p in sorted((city_root / 'leftImg8bit/train').glob('*/*_leftImg8bit.png')) if p.is_file()}
    source = []
    for scene in sorted(train_keys & set(clear_images)):
        raw = source_raw[scene]
        if (raw['height'], raw['width']) != (1024, 2048):
            raise ValueError('Unexpected source dimensions: ' + scene)
        annotations = source_annotations(raw)
        if any(a['bbox'][2] <= a['bbox'][0] or a['bbox'][3] <= a['bbox'][1] for a in annotations):
            raise ValueError('Rounding creates a nonpositive source box: ' + scene)
        source.append({'image_id': scene, 'scene_id': scene, 'file_name': str(clear_images[scene]),
                       'height': 1024, 'width': 2048, 'domain': 'source', 'beta': 'clear',
                       'annotations': annotations})
    fog_maps = {beta: {p.stem.split('_foggy_beta_')[0]: p
                       for p in sorted((fog_root / 'train').glob('*/*_leftImg8bit_foggy_beta_' + beta + '.png')) if p.is_file()}
                for beta in BETAS}
    usable_target_scenes = train_keys.intersection(*(set(fog_maps[b]) for b in BETAS))
    if not source or not usable_target_scenes:
        raise ValueError('Source or target training images are unavailable')
    # Each scene contributes exactly one image per density. Missing variants do
    # not silently turn equal-density training into a density-dependent subset.
    target, image_labels = [], {}
    for scene in sorted(usable_target_scenes):
        annpath = city_root / 'gtFine/train' / scene.split('_')[0] / (scene.replace('_leftImg8bit', '') + '_gtFine_polygons.json')
        presence = presence_from_names(annpath)
        for beta in BETAS:
            path = fog_maps[beta][scene]
            record = {'file_name': str(path), 'image_id': path.stem, 'scene_id': scene,
                      'image_labels': presence, 'domain': 'target', 'beta': beta, 'height': 1024, 'width': 2048}
            assert set(record) == TARGET_FIELDS
            target.append(record)
            image_labels[path.stem] = presence

    eval_records = []
    eval_root = repo_root / 'datasets/foggy_cityscapes_voc/VOC2007'
    for scene in sorted(val_keys):
        for beta in BETAS:
            image_id = scene + '_foggy_beta_' + beta
            image_path = fog_root / 'val' / scene.split('_')[0] / (image_id + '.png')
            annotation_path = eval_root / 'Annotations' / (image_id + '.xml')
            if not image_path.is_file() or not annotation_path.is_file():
                raise FileNotFoundError('Validation image/XML missing: ' + image_id)
            height, width, annotations = read_eval_xml(annotation_path)
            if (height, width) != (1024, 2048):
                raise ValueError('Unexpected validation dimensions: ' + image_id)
            eval_records.append({'file_name': str(image_path), 'image_id': image_id, 'scene_id': scene,
                                 'domain': 'target', 'beta': beta, 'height': height, 'width': width,
                                 'annotations': annotations, 'annotation_file': str(annotation_path)})
    all_eval_ids = {r['image_id'] for r in eval_records}
    if len(all_eval_ids) != 1500 or len(eval_records) != 1500:
        raise ValueError('Mixed validation IDs are not collision-free')
    if {r['scene_id'] for r in source + target} & {r['scene_id'] for r in eval_records}:
        raise ValueError('Train/validation scene collision')
    contents = {'source_train.json': source, 'target_train.json': target,
                'target_image_labels.json': image_labels, 'eval_mixed.json': eval_records}
    for name, value in contents.items():
        save_locked(output_dir / name, value)
    audit = {
        'version': 'formal-city-fog-v1', 'classes': list(CLASSES), 'betas': list(BETAS),
        'source_annotation_scenes': len(train_keys), 'source_usable_images': len(source),
        'source_empty_gt_images_retained': sum(not r['annotations'] for r in source),
        'source_empty_gt_scene_ids': [r['scene_id'] for r in source if not r['annotations']],
        'source_missing_image_scenes': sorted(train_keys - set(clear_images)),
        'source_extra_image_scenes': sorted(set(clear_images) - train_keys),
        'source_classes_before_missing_image_filter': dict(counts),
        'source_classes_used': dict(Counter(CLASSES[a['category_id']] for r in source for a in r['annotations'])),
        'source_group_regions_excluded': dict(groups), 'source_invalid_polygons': invalid,
        'source_conversion': {'path': str(conversion_path), 'sha256': sha_file(conversion_path),
                              'method': 'read_annotations; int(round) coordinates; historical group exclusion; no xmin/ymin shift'},
        'target_usable_images': len(target), 'target_usable_scenes': len(usable_target_scenes),
        'target_density_counts': dict(Counter(r['beta'] for r in target)),
        'target_missing_image_scenes_by_beta': {b: sorted(train_keys - set(fog_maps[b])) for b in BETAS},
        'target_incomplete_triplet_scenes': sorted(set.union(*(set(fog_maps[b]) for b in BETAS)) - usable_target_scenes),
        'target_image_label_provenance': {'kind': 'allowed image-level annotation simulation from paired clear train JSON',
            'root': str(city_root / 'gtFine/train'), 'fields_accessed': ['objects[].label', 'objects[].deleted'],
            'groups': 'group-suffixed labels excluded', 'polygon_coordinates_accessed_for_target_labels': False,
            'target_manifest_has_box_annotations_or_annotation_paths': False},
        'target_presence_counts_by_density': {b: {c: sum(r['image_labels'][i] for r in target if r['beta'] == b)
                                               for i, c in enumerate(CLASSES)} for b in BETAS},
        'eval_images': len(eval_records), 'eval_base_scenes': len(val_keys),
        'eval_density_counts': dict(Counter(r['beta'] for r in eval_records)),
        'eval_id_scheme': 'full raw fog image stem including _foggy_beta_<density>',
        'eval_voc_root': str(eval_root), 'eval_metric': 'native Pascal VOC 2007 historical implementation',
        'train_val_scene_overlap': 0, 'train_val_sequence_overlap': 0,
        'source_target_train_scene_overlap': len({r['scene_id'] for r in source} & usable_target_scenes),
        'source_target_train_overlap_role': 'paired clear/fog train counterparts are permitted; no validation counterpart is used',
        'target_sampler': 'permutation of all three densities per block; independent shuffled scene streams; equal totals every three samples',
        'dimensions': [1024, 2048], 'augmentation': 'independent seeded horizontal flip for source and target; no evaluation flip',
        'manifest_sha256': {name: sha_file(output_dir / name) for name in contents},
    }
    save_locked(output_dir / 'data_audit.json', audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', default='/root/regionclip/DA-Pro')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--city-root', default='/root/autodl-tmp/data/cityspace')
    parser.add_argument('--fog-root', default='/root/autodl-tmp/data/foggycity/leftImg8bit_foggy')
    args = parser.parse_args()
    print(json.dumps(prepare_manifests(**vars(args)), indent=2, ensure_ascii=False))
