"""Materialize an experiment preset using explicit data and immutable-weight paths."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def configure(args):
    cfg = json.loads((HERE / 'configs' / (args.preset + '.json')).read_text())
    repo = Path(args.repo_root).resolve(strict=True)
    manifest = Path(args.manifest_dir).resolve(strict=True)
    assert (repo / 'detectron2').is_dir(), 'repo-root must contain this checkout'
    for name in ('source_train.json', 'target_train.json', 'eval_mixed.json', 'data_audit.json'):
        assert (manifest / name).is_file(), 'Missing manifest: ' + name
    cfg.update(repo_root=str(repo), manifest_dir=str(manifest),
               output_dir=str(Path(args.output_dir).resolve()))
    for key in ('source_checkpoint', 'source_search_checkpoint', 'rpn_checkpoint', 'text_embeddings'):
        cfg[key] = str(Path(getattr(args, key)).resolve(strict=True))
    cfg['source_cfg'] = str(Path(args.source_config).resolve(strict=True))
    cfg['source_checkpoint_sha256'] = digest(cfg['source_checkpoint'])
    expected = json.loads((HERE / 'results/weight_manifest.json').read_text())['fixed_sources']
    for key in ('source_cfg', 'source_checkpoint', 'source_search_checkpoint', 'rpn_checkpoint', 'text_embeddings'):
        if digest(cfg[key]) != expected[key]['sha256']:
            raise ValueError('This preset requires the recorded immutable source asset: ' + key)
    # These historical host locations are not used by the published generic trainer.
    for key in ('parent_run', 'parent_eval_metrics', 'parent_inference_checkpoint'):
        cfg.pop(key, None)
    output = Path(args.config_out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and json.loads(output.read_text()) != cfg:
        raise ValueError('Refusing to overwrite a different config: ' + str(output))
    output.write_text(json.dumps(cfg, indent=2) + '\n')
    return output

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--preset', choices=('formal_25k', 'extend_35k', 'early_5k', 'uda_25k'), default='formal_25k')
    parser.add_argument('--repo-root', default=str(HERE.parents[1]))
    parser.add_argument('--source-config', default=str(HERE / 'configs/source_B.yaml'))
    for name in ('manifest-dir', 'output-dir', 'source-checkpoint', 'source-search-checkpoint',
                 'rpn-checkpoint', 'text-embeddings', 'config-out'):
        parser.add_argument('--' + name, required=True)
    print(configure(parser.parse_args()))

if __name__ == '__main__':
    main()
