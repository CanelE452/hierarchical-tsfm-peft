"""Fetch only two pinned Bolt checkpoints and bind official config/weight bytes."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from runtime import CACHE, HERE, ROOT, LIMITS, Job, artifact, digest, read_json, save_json


MODELS = {
    'SMALL': ('amazon/chronos-bolt-small', '772f3d25d38aec6d914c8949dab4462e2d46f5d8'),
    'MINI': ('amazon/chronos-bolt-mini', '251268337516a88e253628c43e1d26ec577b376b'),
    'TINY': ('amazon/chronos-bolt-tiny', 'a0e552de83495b5c28c14c71c374f3e33280b340'),
}


def main():
    if (HERE / 'model_manifest.json').exists():
        record = read_json(HERE / 'model_manifest.json')
        for model in record['models'].values():
            for entry in model['files'].values():
                artifact(entry['path'], entry['sha256'])
        print('Pinned model files already complete; no duplicate download.')
        return
    manifest = {'scope': 'Official pinned Chronos-Bolt family; original channels independently forecast',
                'models': {}, 'actual_loaded_parameter_counts': 'Recorded by the preflight receipt before TEST seal'}
    with Job('cpu', 'pinned-model-preparation', reserve_s=300) as job:
        for key, (model_id, revision) in MODELS.items():
            info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
            if info.sha != revision:
                raise ValueError('Official revision identity mismatch')
            wanted = {s.rfilename: s for s in info.siblings if s.rfilename in ('config.json', 'model.safetensors')}
            if set(wanted) != {'config.json', 'model.safetensors'}:
                raise ValueError('Required official model files not found')
            folder = ROOT / '.cache/huggingface/models--amazon--chronos-bolt-small/snapshots' / revision if key == 'SMALL' else CACHE / 'models' / key.lower()
            files = {}
            for name, remote in wanted.items():
                destination = folder / name
                existed = destination.is_file()
                if key == 'SMALL':
                    if not existed:
                        raise FileNotFoundError('Existing small anchor is missing: ' + name)
                elif not existed:
                    ledger = read_json(HERE / 'ledger.json')
                    if ledger['download_bytes'] + remote.size > LIMITS['download_bytes']:
                        raise RuntimeError('STOP_RESOURCE: model byte cap')
                    destination = Path(hf_hub_download(model_id, name, revision=revision, local_dir=folder))
                    ledger['download_bytes'] += destination.stat().st_size
                    ledger['downloads'].append({'model': key, 'revision': revision, 'file': name,
                                                'bytes_charged': destination.stat().st_size,
                                                'accounting': 'Conservative newly materialized file bytes; not network wire telemetry'})
                    save_json(HERE / 'ledger.json', ledger)
                entry = artifact(destination)
                if remote.size != entry['bytes']:
                    raise ValueError('Official model file size mismatch')
                if remote.lfs is not None and remote.lfs.sha256 != entry['sha256']:
                    raise ValueError('Official LFS digest mismatch')
                if remote.lfs is None:
                    content = destination.read_bytes()
                    git_blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
                    if git_blob != remote.blob_id:
                        raise ValueError('Official config Git blob mismatch')
                entry.update(official_git_blob=remote.blob_id, official_lfs_sha256=None if remote.lfs is None else remote.lfs.sha256,
                             reused_existing_file=existed)
                files[name] = entry
                job.heartbeat({'model': key, 'file': name})
            config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
            chronos = config['chronos_config']
            if config['architectures'] != ['ChronosBoltModelForForecasting'] or chronos['context_length'] < 512 or chronos['prediction_length'] < 48 or 0.5 not in chronos['quantiles']:
                raise ValueError('Official config violates Bolt/L512/H48/median contract')
            manifest['models'][key] = {'id': model_id, 'revision': revision,
                'official_url': f'https://huggingface.co/{model_id}/tree/{revision}',
                'checkpoint_dir': str(folder), 'files': files,
                'config': {k: config.get(k) for k in ('architectures', 'd_model', 'd_ff', 'num_layers', 'num_decoder_layers', 'num_heads')},
                'chronos_config': chronos, 'pretraining_overlap': 'unverified'}
            save_json(HERE / 'model_manifest_partial.json', manifest)
        save_json(HERE / 'model_manifest.json', manifest)
        print(json.dumps({'models': list(manifest['models']), 'download_bytes_charged': read_json(HERE / 'ledger.json')['download_bytes']}))


if __name__ == '__main__':
    main()
