"""Remove target image labels while preserving image order, source GT and evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

from data import IMAGE_FIELDS, TARGET_FIELDS, validate_target_record


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def write_locked(path, content):
    if path.exists() and path.read_bytes() != content:
        raise ValueError('Refusing to overwrite different manifest: ' + str(path))
    path.write_bytes(content)


def prepare(input_dir, output_dir):
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if source == output:
        raise ValueError('UDA manifests must have their own directory')
    previous = json.loads((source / 'data_audit.json').read_text())
    contents = {}
    for name in ('source_train.json', 'target_train.json', 'eval_mixed.json'):
        contents[name] = (source / name).read_bytes()
        if digest_bytes(contents[name]) != previous['manifest_sha256'][name]:
            raise ValueError('Original manifest hash differs: ' + name)
    original = json.loads(contents['target_train.json'])
    records = []
    for record in original:
        if set(record) != TARGET_FIELDS:
            raise ValueError('Unexpected original target fields')
        # Copy image metadata only. Never use presence values to select images,
        # classes, proposals, losses, or paired-source annotation lookups.
        clean = {key: value for key, value in record.items() if key in IMAGE_FIELDS}
        validate_target_record(clean, image_labels_allowed=False)
        records.append(clean)
    if not records:
        raise ValueError('Empty target manifest')
    contents['target_train.json'] = (json.dumps(records, indent=2) + '\n').encode('utf8')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'target_image_labels.json').exists():
        raise ValueError('Label sidecar must not be present in UDA data directory')
    for name, content in contents.items():
        write_locked(output / name, content)
    audit = {key: value for key, value in previous.items()
             if key not in ('manifest_sha256', 'target_image_label_provenance',
                            'target_presence_counts_by_density')}
    audit.update(version='formal-city-fog-uda-v1', target_image_labels_allowed=False,
                 target_image_label_provenance={'kind': 'none; image metadata only',
                     'label_sidecar_copied': False, 'paired_source_label_lookup': False},
                 controlled_ablation={'parent_manifest_dir': str(source),
                     'parent_manifest_sha256': {name: previous['manifest_sha256'][name]
                                               for name in contents},
                     'source_and_eval_bytes_unchanged': True,
                     'target_metadata_order_unchanged': True,
                     'only_removed_target_field': 'image_labels'},
                 manifest_sha256={name: digest_bytes(value) for name, value in contents.items()})
    write_locked(output / 'data_audit.json', (json.dumps(audit, indent=2) + '\n').encode('utf8'))
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input_dir, args.output_dir), indent=2))
