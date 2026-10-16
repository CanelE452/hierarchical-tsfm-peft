"""Render sealed representative-learning metadata; never open weights or arrays."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for _variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '2'

DEFAULT_IDS = (
    'primary_beijing_E1_DIRECT_G_k0_s92601__e1__e3',
    'primary_beijing_E1_B0_k3_s92601__e1__e3',
    'primary_beijing_E1_B2_k3_s92601__e1__e3',
    'primary_beijing_E1_DIRECT_B2_k0_s92601__e3',
    'primary_beijing_E1_DIRECT_B4_k0_s92601__e3',
    'primary_beijing_E1_LORA_R8_k0_s92601__e3',
)
LABELS = {'DIRECT_G': 'G', 'B0': 'B0', 'B2': 'B2', 'DIRECT_B2': 'Direct B2',
          'DIRECT_B4': 'Direct B4', 'LORA_R8': 'LoRA r8', 'B4': 'B4',
          'ADAPTS_BOLT_PORT': 'AdaPTS port', 'LAG_CAUSAL': 'Lag causal',
          'LAG_BIDIR': 'Lag bidirectional', 'ORDERED_MULTI_K': 'Shared ordered K',
          'MAXK_PCA_TRAINED': 'Max-K trained'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def new_json(path, value):
    with path.open('x', encoding='utf-8', newline='') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


COHORT = 'beijing:primary:V_select:stride48'
MUTABLE_REGISTRY = {'execution_status', 'training_status', 'validity_status', 'scientific_status',
                    'publication_status', 'reason', 'resume_command'}


def canonical_v_paths(identity):
    return ('predictions/' + identity + '/V_select/PREDICTION_MANIFEST.json',
            'ordered_predictions/' + identity + '/PREDICTION_MANIFEST.json')


def validate_manifest(manifest, identity, receipt, relative):
    if any(manifest.get(key) != value for key, value in
           {'run_id': identity, 'split': 'V_select', 'cohort': COHORT}.items()):
        raise ValueError('V run/split/cohort differs: ' + relative)
    metadata = manifest.get('metadata', {})
    if any(metadata.get(key) != manifest[key] for key in ('run_id', 'split', 'cohort')):
        raise ValueError('V metadata identity differs: ' + relative)
    if manifest.get('scaler_sha') != receipt['refit_scaler_sha']:
        raise ValueError('V canonical selected-artifact scaler differs: ' + relative)
    ordered = relative == canonical_v_paths(identity)[1]
    field = 'parent_artifact_sha256' if ordered else 'artifact_sha256'
    if metadata.get(field) != receipt['artifact_sha256']:
        raise ValueError('V selected artifact differs: ' + relative)
    if ordered and (metadata.get('parent_run_id') != identity or metadata.get('variant') not in {'shared', 'max_only'}):
        raise ValueError('Ordered representative must be the exact full-width parent: ' + relative)
    flag = metadata.get('reused_prediction_cache')
    if flag is not None and not isinstance(flag, bool):
        raise ValueError('Cache reuse record must be boolean or absent')
    score = manifest.get('score', {})
    if any(score.get(key) != manifest[key] for key in ('run_id', 'split', 'cohort')):
        raise ValueError('V score identity differs: ' + relative)
    if score.get('num_channels') != 11 or score.get('horizon') != 48:
        raise ValueError('Beijing representative V axes differ: ' + relative)


def selected_science(public_root, identities):
    rows = list(csv.DictReader(io.StringIO((public_root / 'EXPERIMENT_REGISTRY.csv').read_text(encoding='utf-8-sig'))))
    result = {}
    for identity in identities:
        own = [row for row in rows if row['experiment_id'] == identity]
        if len(own) != 1:
            raise ValueError('Exactly one selected scientific registry row required: ' + identity)
        result[identity] = {key: value for key, value in own[0].items() if key not in MUTABLE_REGISTRY}
    return result


def display_family(scientific_row, source, receipt):
    spec = source['spec']
    family = scientific_row['family']
    underlying = {'ORDERED_MULTI_K': 'B0', 'MAXK_PCA_TRAINED': 'B0', 'LORA_R8': 'LORA',
                  'ADAPTS_BOLT_PORT': 'ADAPTS'}.get(family, family)
    if (spec.get('run_id') != scientific_row['experiment_id'] or receipt.get('run_id') != scientific_row['experiment_id']
            or spec.get('family') != receipt['family'] or receipt['family'] != underlying
            or spec.get('fold', 'primary') != scientific_row['fold'] or int(spec.get('seed', -1)) != int(scientific_row['seed'])):
        raise ValueError('Registered display family/source/BASE identity differs')
    if family in {'ORDERED_MULTI_K', 'MAXK_PCA_TRAINED'}:
        expected = [3, 6, 11] if family == 'ORDERED_MULTI_K' else [11]
        if (int(scientific_row['num_channels']) != 11 or int(scientific_row['latent_k']) != 11
                or spec.get('latent_k') != 11 or spec.get('k_grid') != expected):
            raise ValueError('Registered ordered display family differs from actual source K setting')
    if family == 'LORA_R8' and spec.get('lora_rank') != 8:
        raise ValueError('Registered LoRA r8 differs from actual source rank')
    return family


def capture(public_root, run_ids):
    public_root = Path(public_root)
    if len(set(run_ids)) != len(run_ids):
        raise ValueError('Duplicate representative run ID')
    files = {}
    manifests, identities = {}, set(run_ids)

    def take(relative, optional=False):
        path = public_root / relative
        if not path.exists():
            if not optional:
                raise FileNotFoundError(relative)
            files[relative] = {'exists': False, 'sha256': None, 'bytes': 0}
            return None
        raw = path.read_bytes()
        files[relative] = {'exists': True, 'sha256': sha(raw), 'bytes': len(raw),
                           'content_utf8': raw.decode('utf-8-sig')}
        # Decode UTF-8 BOM without losing original bytes for fingerprint replay.
        if raw.startswith(b'\xef\xbb\xbf'):
            files[relative]['utf8_bom'] = True
        return json.loads(raw.decode('utf-8-sig')) if path.suffix == '.json' else raw

    for run_id in run_ids:
        if not run_id.startswith('primary_beijing_') or '/' in run_id or '\\' in run_id:
            raise ValueError('Only explicit Beijing representative IDs are accepted')
        prefix = 'runs/' + run_id + '/'
        base = take(prefix + 'BASE_FIT_RECEIPT.json')
        if base.get('run_id') != run_id or base['status'] != 'COMPLETE_VALID' or max(base['epochs']) != 3:
            raise ValueError('Incomplete or non-3-epoch run: ' + run_id)
        take(prefix + 'SOURCE_CONTRACT.json')
        extension = take(prefix + 'LEARNING_EXTENSION_RECEIPT.json')
        take(prefix + 'search/CURVE.csv')
        parent = extension['extension_provenance']['identity']['parent_run_id']
        identities.add(parent)
        parent_receipt = take('runs/' + parent + '/BASE_FIT_RECEIPT.json')
        take('runs/' + parent + '/SOURCE_CONTRACT.json')
        for identity in (parent, run_id):
            receipt = parent_receipt if identity == parent else base
            found = []
            for relative in canonical_v_paths(identity):
                value = take(relative, optional=True)
                if value is not None:
                    validate_manifest(value, identity, receipt, relative)
                    found.append((relative, value))
            if len(found) > 1:
                raise ValueError('Multiple canonical V manifests; no implicit preference: ' + identity)
            relative, value = found[0] if found else (None, None)
            manifests[identity] = {'relative_path': relative, 'sha256': None if relative is None else files[relative]['sha256'],
                'artifact_sha256': receipt['artifact_sha256'], 'cohort': COHORT,
                'cache_reuse_recorded': None if value is None else value['metadata'].get('reused_prediction_cache')}
    science = selected_science(public_root, identities)
    for identity in identities:
        source = json.loads(files['runs/' + identity + '/SOURCE_CONTRACT.json']['content_utf8'])
        receipt = json.loads(files['runs/' + identity + '/BASE_FIT_RECEIPT.json']['content_utf8'])
        display_family(science[identity], source, receipt)
    for relative, record in files.items():
        path = public_root / relative
        if record['exists']:
            if not path.exists() or sha(path.read_bytes()) != record['sha256']:
                raise RuntimeError('Input changed during sealing: ' + relative)
        elif path.exists():
            raise RuntimeError('Previously absent V appeared while sealing; recapture in a new snapshot')
    if selected_science(public_root, identities) != science:
        raise RuntimeError('Selected scientific registry identities changed while sealing')
    science_sha = sha(json.dumps(science, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    return {'format_version': 3, 'run_ids': list(run_ids), 'files': files,
            'prediction_manifest_paths': manifests, 'selected_scientific_rows': science,
            'selected_scientific_rows_sha256': science_sha,
            'scope': 'Public BASE/extension/search curve/selected V metadata only; no model/NPZ/raw data/T.',
            'created_unix': time.time()}


def load_tables(bundle):
    def text(relative):
        record = bundle['files'][relative]
        if not record['exists']:
            return None
        raw = record['content_utf8'].encode('utf-8')
        if record.get('utf8_bom'):
            raw = b'\xef\xbb\xbf' + raw
        if sha(raw) != record['sha256'] or len(raw) != record['bytes']:
            raise ValueError('Sealed metadata changed: ' + relative)
        return record['content_utf8']

    def obj(relative):
        value = text(relative)
        return None if value is None else json.loads(value)

    v3 = bundle.get('format_version') == 3
    if v3:
        science = bundle['selected_scientific_rows']
        if sha(json.dumps(science, sort_keys=True, separators=(',', ':')).encode('utf-8')) != bundle['selected_scientific_rows_sha256']:
            raise ValueError('Sealed selected registry science changed')

    def manifest(identity, receipt):
        if not v3:
            relative = canonical_v_paths(identity)[0]
            value = obj(relative)
        else:
            entry = bundle['prediction_manifest_paths'][identity]
            relative = entry['relative_path']
            present = [path for path in canonical_v_paths(identity) if bundle['files'].get(path, {}).get('exists')]
            if (entry['artifact_sha256'] != receipt['artifact_sha256'] or entry['cohort'] != COHORT
                    or (relative is None and (present or entry['sha256'] is not None))
                    or (relative is not None and (relative not in canonical_v_paths(identity) or present != [relative]
                        or bundle['files'][relative]['sha256'] != entry['sha256']))):
                raise ValueError('Sealed canonical V map differs: ' + identity)
            value = None if relative is None else obj(relative)
            if value is not None and entry['cache_reuse_recorded'] != value['metadata'].get('reused_prediction_cache'):
                raise ValueError('Sealed cache reuse flag differs')
        if value is not None:
            validate_manifest(value, identity, receipt, relative)
        return value

    curves, accounting, comparisons = [], [], []
    for run_id in bundle['run_ids']:
        base = obj('runs/' + run_id + '/BASE_FIT_RECEIPT.json')
        ext = obj('runs/' + run_id + '/LEARNING_EXTENSION_RECEIPT.json')
        prov = ext['extension_provenance']
        parent_id = prov['identity']['parent_run_id']
        parent = obj('runs/' + parent_id + '/BASE_FIT_RECEIPT.json')
        family_for_display = display_family(science[run_id], obj('runs/' + run_id + '/SOURCE_CONTRACT.json'), base) if v3 else base['family']
        label = LABELS.get(family_for_display, family_for_display)
        if family_for_display in {'B0', 'B2', 'B4'}:
            label += ' (K=3)'
        curve = list(csv.DictReader(io.StringIO(text('runs/' + run_id + '/search/CURVE.csv'))))
        account = ext['extension_accounting']
        if account['new_search_updates'] + account['inherited_search_updates'] != base['search_updates']:
            raise ValueError('Search accounting mismatch')
        if account['new_total_updates'] != account['new_search_updates'] + account['new_refit_updates']:
            raise ValueError('Incremental total accounting mismatch')
        if account['inherited_search_updates'] != parent['search_updates']:
            raise ValueError('Parent update mismatch')
        if not prov['payload_unchanged'] or not prov['fresh_A60_refit'] or prov['copied_old_A60_refit']:
            raise ValueError('Fresh-refit/continuation provenance mismatch')
        endpoints = [float(x['epoch']) for x in curve]
        if endpoints != [0, .25, 1, 3]:
            raise ValueError('Exact .25/1/3 curve required')
        mse = [float(x['astop_mse']) for x in curve]
        if not all(math.isfinite(x) and x >= 0 for x in mse):
            raise ValueError('Nonfinite A_stop score')
        if endpoints[mse.index(min(mse))] != float(base['selected_epoch']):
            raise ValueError('Recorded A_stop selection differs from curve')
        if float(curve[-1]['updates']) != base['search_updates']:
            raise ValueError('Search endpoint update count mismatch')
        for point in curve:
            curves.append({'run_id': run_id, 'label': label, 'family': base['family'],
                           'epoch': float(point['epoch']), 'astop_mse': float(point['astop_mse']),
                           'exposures': int(point['exposures']), 'updates': int(point['updates']),
                           'segment_seconds': float(point['segment_seconds']),
                           'clipped_steps': int(point['clipped_steps']),
                           'inherited_endpoint': float(point['epoch']) <= max(parent['epochs']),
                           'selected_endpoint': float(point['epoch']) == base['selected_epoch']})
        accounting.append({'run_id': run_id, 'label': label, 'family': base['family'],
                           'parent_max_search_epoch': max(parent['epochs']),
                           'new_max_search_epoch': max(base['epochs']),
                           'selected_refit_epoch': base['selected_epoch'],
                           'inherited_search_updates': account['inherited_search_updates'],
                           'new_search_updates': account['new_search_updates'],
                           'inclusive_search_updates': account['inclusive_search_updates'],
                           'fresh_refit_updates': account['new_refit_updates'],
                           'new_total_updates': account['new_total_updates'],
                           'old_parent_refit_updates_preserved_excluded': account['preserved_parent_refit_updates'],
                           'new_search_exposures': base['search_exposures'] - parent['search_exposures'],
                           'fresh_refit_exposures': base['refit_exposures'],
                           'new_search_F_streams': account['new_search_F_logical_streams'],
                           'new_refit_F_streams': account['new_refit_F_logical_streams'],
                           'training_status': base['training_status'],
                           'fresh_A60_refit_recorded': prov['fresh_A60_refit'],
                           'copied_old_A60_refit_recorded': prov['copied_old_A60_refit']})
        old_v = manifest(parent_id, parent)
        new_v = manifest(run_id, base)
        v_available = old_v is not None and new_v is not None
        row = {'run_id': run_id, 'label': label, 'family': base['family'], 'parent_run_id': parent_id,
               'parent_max_search_epoch': max(parent['epochs']), 'new_max_search_epoch': max(base['epochs']),
               'selected_epoch': base['selected_epoch'], 'best_astop_mse': base['selected_astop_mse'],
               'last_astop_mse': mse[-1], 'training_status': base['training_status'],
               'parent_model_state_sha': parent['model_state_sha'], 'new_model_state_sha': base['model_state_sha'],
               'model_state_sha_equal': parent['model_state_sha'] == base['model_state_sha'],
               'refit_scaler_sha_equal': parent['refit_scaler_sha'] == base['refit_scaler_sha'],
               'whole_artifact_sha_equal': parent['artifact_sha256'] == base['artifact_sha256'],
               'parent_V_present': old_v is not None, 'new_V_present': new_v is not None,
               'V_status': 'V_METADATA_AVAILABLE' if v_available else 'TRAIN_ONLY',
               'parent_V_mse': None if old_v is None else old_v['score']['mse'],
               'new_V_mse': None if new_v is None else new_v['score']['mse'],
               'parent_V_mae': None if old_v is None else old_v['score']['mae'],
               'new_V_mae': None if new_v is None else new_v['score']['mae'],
               'V_cohort_equal': None, 'V_data_equal': None, 'V_scaler_equal': None,
               'V_source_equal': None, 'V_config_equal': None, 'V_model_identity_equal': None,
               'V_prediction_fingerprint_equal': None, 'V_reference_fingerprints_equal': None,
               'V_scores_equal': None, 'new_V_cache_reused_recorded': None}
        if new_v is not None:
            row['new_V_cache_reused_recorded'] = new_v['metadata'].get('reused_prediction_cache')
        if v_available:
            for field in ('cohort', 'data', 'scaler', 'source', 'config'):
                key = field if field == 'cohort' else field + '_sha'
                row['V_' + field + '_equal'] = old_v[key] == new_v[key]
            row['V_model_identity_equal'] = old_v['model_sha'] == new_v['model_sha']
            row['V_prediction_fingerprint_equal'] = old_v['arrays']['pred_normalized'] == new_v['arrays']['pred_normalized']
            row['V_reference_fingerprints_equal'] = all(old_v['arrays'][key] == new_v['arrays'][key]
                for key in ('target_normalized', 'observed', 'origins', 'channel_ids'))
            row['V_scores_equal'] = all(old_v['score'][key] == new_v['score'][key] for key in ('mse', 'mae', 'n_observed'))
            if not all(row[key] for key in ('V_cohort_equal', 'V_data_equal', 'V_scaler_equal', 'V_reference_fingerprints_equal')):
                raise ValueError('Parent/new V cohort contract differs')
        comparisons.append(row)
    return {'CURVES.csv': curves, 'ACCOUNTING.csv': accounting, 'COMPARISONS.csv': comparisons}


def table_bytes(rows):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode('utf-8')


def render(tables, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'informs-representative-learning-v1', 'svg.fonttype': 'none'})
    rows = tables['COMPARISONS.csv']
    count = len(rows)
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00', '#332288', '#882255']

    def save(fig, name):
        fig.savefig(directory / (name + '.png'), dpi=180, facecolor='white', metadata={'Software': 'Representative learning metadata renderer'})
        fig.savefig(directory / (name + '.svg'), facecolor='white', metadata={'Date': None, 'Creator': 'Representative learning metadata renderer'})
        plt.close(fig)

    ncols = 3
    nrows = math.ceil(count / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.35 * nrows + 1.2), squeeze=False)
    fig.subplots_adjust(top=.84 if nrows == 2 else .88, bottom=.14, hspace=.55, wspace=.28)
    for index, row in enumerate(rows):
        ax = axes[index // ncols][index % ncols]
        points = [x for x in tables['CURVES.csv'] if x['run_id'] == row['run_id']]
        color = colors[index % len(colors)]
        ax.plot([p['epoch'] for p in points], [p['astop_mse'] for p in points], '-o', color=color, lw=2, ms=5)
        chosen = next(p for p in points if p['selected_endpoint'])
        ax.scatter([chosen['epoch']], [chosen['astop_mse']], marker='*', s=190, color='#111111', zorder=5)
        ax.set_title(row['label'] + '  |  selected ' + str(row['selected_epoch']) + ' epoch\n' + row['training_status'], loc='left', fontsize=9, fontweight='bold')
        ax.set_xticks([0, .25, 1, 3], ['0', '.25', '1', '3'])
        ax.set_xlabel('Cumulative search epochs')
        ax.set_ylabel('A_stop normalized MSE')
        ax.grid(axis='y', alpha=.25)
        ax.text(.99, .97, 'last: ' + format(row['last_astop_mse'], '.6f') + '\nbest: ' + format(row['best_astop_mse'], '.6f'),
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none'})
        ax.margins(x=.08, y=.2)
    for index in range(count, nrows * ncols):
        axes[index // ncols][index % ncols].set_visible(False)
    fig.suptitle('Beijing representative search: A_stop checkpoint selection', y=.97, fontsize=18, fontweight='bold')
    fig.text(.5, .915 if nrows == 2 else .935, 'A_stop search / preliminary | one seed (92601) | stars mark selected endpoints', ha='center', color='#555555', fontsize=11)
    fig.text(.5, .025, str(count) + '/12 representative conditions shown; all-family / dataset / K coverage unfinished.\nRecorded condition status is shown above; a worse last checkpoint alone does not establish convergence.', ha='center', color='#555555', fontsize=10)
    save(fig, 'search_curves')

    accounting = tables['ACCOUNTING.csv']
    height = .56 * count + 3
    fig, ax = plt.subplots(figsize=(13.2, height))
    fig.subplots_adjust(left=.19, right=.95, top=.78, bottom=.23)
    for index, row in enumerate(accounting):
        left = 0
        for field, color, hatch in [('inherited_search_updates', '#BDBDBD', '//'),
                                     ('new_search_updates', '#0072B2', None),
                                     ('fresh_refit_updates', '#E69F00', None)]:
            value = row[field]
            ax.barh(index, value, left=left, color=color, height=.66, hatch=hatch, edgecolor='white', linewidth=.8)
            if value >= 150:
                ax.text(left + value / 2, index, str(value), ha='center', va='center', color='white' if field == 'new_search_updates' else '#111111', fontsize=10)
            left += value
        ax.text(left + 30, index, 'new ' + str(row['new_total_updates']), va='center', fontsize=10)
    ax.set_yticks(range(count), [x['label'] for x in accounting])
    ax.invert_yaxis()
    ax.set_xlim(0, max(x['inclusive_search_updates'] + x['fresh_refit_updates'] for x in accounting) * 1.17)
    ax.set_xlabel('Optimizer updates (gray occurred before this extension)')
    ax.grid(axis='x', alpha=.2)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor='#BDBDBD', hatch='//', label='Inherited search'), Patch(facecolor='#0072B2', label='New search'), Patch(facecolor='#E69F00', label='Fresh A60 refit')], loc='upper center', bbox_to_anchor=(.55, .88), ncol=3, frameon=False)
    fig.suptitle('Actual added learning and preserved search history', y=.98, fontsize=17, fontweight='bold')
    fig.text(.5, .06, 'Blue + orange = newly executed updates. The old parent A60 refit is preserved and excluded from these bars.\nThe maximum search budget is 3 epochs; the fresh refit follows the selected endpoint, which may be shorter.\nBeijing / one seed / preliminary; all-family, dataset and K coverage remains incomplete.', ha='center', fontsize=10, color='#555555')
    save(fig, 'training_accounting')

    fig, ax = plt.subplots(figsize=(15, .62 * count + 3.6))
    ax.axis('off')
    labels = ['Condition', 'Parent search', 'New search /\nselected refit', 'State SHA vs\nparent (metadata)', 'Saved V vs parent', 'New V MSE / MAE']
    cells = []
    for row in rows:
        state = 'SAME' if row['model_state_sha_equal'] else 'DIFFERENT'
        if row['V_status'] == 'TRAIN_ONLY':
            vstatus, metrics = 'TRAIN_ONLY', 'not available'
        else:
            vstatus = 'SAME fingerprints' if row['V_prediction_fingerprint_equal'] else 'DIFFERENT predictions'
            metrics = format(row['new_V_mse'], '.6f') + ' / ' + format(row['new_V_mae'], '.6f')
        cells.append([row['label'], str(row['parent_max_search_epoch']), '3 / ' + str(row['selected_epoch']), state, vstatus, metrics])
    table = ax.table(cellText=cells, colLabels=labels, cellLoc='center', colLoc='center', bbox=[0, .2, 1, .65], colWidths=[.16, .11, .14, .18, .2, .21])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (y, x), cell in table.get_celld().items():
        cell.set_edgecolor('#DDDDDD')
        if y == 0:
            cell.set_facecolor('#E8EEF3'); cell.set_text_props(weight='bold')
        elif y % 2 == 0:
            cell.set_facecolor('#F7F8FA')
    fig.suptitle('Selected model state and saved V metadata', y=.97, fontsize=18, fontweight='bold')
    fig.text(.5, .88, 'V selection / preliminary | exact values and SHA fields: COMPARISONS.csv', ha='center', color='#555555')
    fig.text(.5, .06, 'SAME state means recorded model_state_sha equality; whole artifact identities can differ.\nV fingerprints are compared on the same cohort/data/scaler. This renderer performs no weight/NPZ replay or independent rescore.\nA repeated selected model is evidence of actual search with unchanged selection, not a 3-epoch accuracy gain or no training.', ha='center', fontsize=10, color='#555555')
    fig.subplots_adjust(left=.025, right=.975, top=.88, bottom=.05)
    save(fig, 'selected_state_and_v')
    font_path = Path(font_manager.findfont('DejaVu Sans'))
    return {'matplotlib_version': matplotlib.__version__, 'font_name': 'DejaVu Sans',
            'font_filename': font_path.name, 'font_sha256': sha(font_path.read_bytes()),
            'dpi': 180, 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
            'thread_limits': {key: os.environ[key] for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-root', type=Path)
    parser.add_argument('--input-bundle', type=Path)
    parser.add_argument('--output-parent', type=Path, required=True)
    parser.add_argument('--run-id', action='append')
    args = parser.parse_args()
    if bool(args.public_root) == bool(args.input_bundle):
        parser.error('Use exactly one of --public-root or --input-bundle')
    if args.input_bundle and args.run_id:
        parser.error('--run-id is only for a new live metadata capture')
    if args.input_bundle:
        bundle = json.loads(args.input_bundle.read_text(encoding='utf-8'))
    else:
        bundle = capture(args.public_root.resolve(), args.run_id or DEFAULT_IDS)
    tables = load_tables(bundle)
    directory = args.output_parent / ('snapshot_' + str(time.time_ns()))
    directory.mkdir(parents=True, exist_ok=False)
    new_json(directory / 'INPUT_BUNDLE.json', bundle)
    new_json(directory / 'INPUTS_SHA256.json', {key: {k: v for k, v in record.items() if k != 'content_utf8'} for key, record in bundle['files'].items()})
    for name, rows in tables.items():
        with (directory / name).open('xb') as handle:
            handle.write(table_bytes(rows))
    environment = render(tables, directory)
    # Parse sealed input again, then re-render into new paths. No live reread is used.
    repeat = directory / 'repeat_render'
    repeat.mkdir(exist_ok=False)
    repeated_tables = load_tables(json.loads((directory / 'INPUT_BUNDLE.json').read_text(encoding='utf-8')))
    for name, rows in repeated_tables.items():
        with (repeat / name).open('xb') as handle:
            handle.write(table_bytes(rows))
    render(repeated_tables, repeat)
    repeated = {}
    for name in list(tables) + [stem + suffix for stem in ('search_curves', 'training_accounting', 'selected_state_and_v') for suffix in ('.png', '.svg')]:
        first, second = (directory / name).read_bytes(), (repeat / name).read_bytes()
        repeated[name] = {'first_sha256': sha(first), 'repeat_sha256': sha(second), 'byte_exact': first == second}
    if not all(value['byte_exact'] for value in repeated.values()):
        raise RuntimeError('Repeat render or table reproduction differs')
    new_json(directory / 'REPRODUCIBILITY_CHECK.json', {'status': 'PASS', 'checks': repeated,
             'scope': 'Two parses of the sealed metadata and two CPU renders; no training, forecast replay or scorer rerun.'})
    new_json(directory / 'RENDER_RECEIPT.json', {'status': 'COMPLETE_DERIVED_METADATA_FIGURES', 'run_ids': bundle['run_ids'],
             'displayed_count': len(bundle['run_ids']), 'planned_representative_count': 12,
             'capture_format_version': bundle.get('format_version', 2),
             'selected_scientific_rows_sha256': bundle.get('selected_scientific_rows_sha256'),
             'canonical_V_manifest_map_sha256': None if 'prediction_manifest_paths' not in bundle else sha(json.dumps(bundle['prediction_manifest_paths'], sort_keys=True).encode('utf-8')),
             'V_available': sum(x['new_V_present'] for x in tables['COMPARISONS.csv']),
             'source_sha256': sha(Path(__file__).read_bytes()), 'environment': environment,
             'training_status_preserved': sorted(set(x['training_status'] for x in tables['COMPARISONS.csv'])),
             'weights_NPZ_rawdata_T_GPU_accessed': False, 'independent_training_forecast_replay_rescore': False})
    shutil.copyfile(__file__, directory / 'generate.py')
    readme = '''# Beijing 대표 학습 심화: 입력 봉인 그림\n\n[확인] 완료된 대표 조건의 공개 search CURVE·BASE·learning-extension·V manifest만 읽었다. 최대 검색량 3 epoch와 A_stop이 선택한 refit 길이는 다르다. `ACCOUNTING.csv`는 inherited search, 이번 신규 search, 새 A60 refit, 보존한 이전 refit을 구분하며 이전 refit은 그림 막대에 더하지 않는다. 동일 origin 반복 노출을 새 독립 데이터로 세지 않는다.\n\n![A_stop 검색 곡선](search_curves.png)\n\n![실제 신규 학습량](training_accounting.png)\n\n![선택 상태와 저장 V 메타데이터](selected_state_and_v.png)\n\nG/B0/B2의 기존 1 epoch 부모와 새 3 epoch 검색은 선택된 model_state_sha·scaler와 저장 V fingerprints가 같은 경우를 표시한다. 이는 실제 추가 검색 및 fresh refit을 하지 않았다는 뜻이 아니다. 새 bundle은 run/spec/search 계약을 포함해 달라질 수 있다. model_state SHA는 가중치 재열람 없이 기록을 비교한 것이므로 독립 checkpoint replay를 했다는 뜻도 아니다. V가 없으면 TRAIN_ONLY이며 0점이나 실패로 대체하지 않는다.\n\n마지막 checkpoint가 선택된 checkpoint보다 나쁘다는 이유로 수렴을 판정하지 않는다. 각 조건의 실제 training_status를 그대로 유지하며, UNDERTRAINED_OR_BUDGET_LIMIT 등 다른 상태를 하나의 문구로 바꾸지 않는다. 이번 그림은 Beijing 한 seed의 선택 단계 중간 결과다. 모든 가족·자료·K의 최소 학습 완료나 독립 TEST 일반화 증거가 아니다.\n\n모든 원수치는 CURVES.csv·ACCOUNTING.csv·COMPARISONS.csv에 전체 정밀도로 저장했다. INPUT_BUNDLE.json에는 읽은 공개 메타데이터의 정확한 문자열과 SHA가 있어 원 프로젝트 자산 없이 다시 그릴 수 있다. INPUTS_SHA256.json은 부재했던 V도 명시한다. REPRODUCIBILITY_CHECK.json은 봉인 입력을 두 번 독립 파싱하고 PNG/SVG/CSV를 새 경로에 두 번 생성해 바이트 SHA가 일치했는지 기록한다. 이는 모델 재학습·추론·독립 오차 재계산 검사와 다르다.\n\n재현: `python -X utf8 -u generate.py --input-bundle INPUT_BUNDLE.json --output-parent <새_출력_부모>`\n\n나중에 전체 대표 12조건을 포착하려면 `--public-root <campaign> --run-id <완료된_ID>`를 각 ID에 반복하고 새 --output-parent를 지정한다. 미완료 fit은 오류로 멈추며, 기존 결과를 덮어쓰지 않는다. 완성된 입력 묶음을 재현할 때에는 원 V가 나중에 추가돼도 이전 그림이 바뀌지 않는다.\n'''
    with (directory / 'README_KO.md').open('x', encoding='utf-8') as handle:
        handle.write(readme)
        handle.write('\nV3는 일반 predictions 및 ordered_predictions의 두 실제 canonical 위치를 exact run/split/cohort/선택 artifact로 검사하고, 유일한 경로와 SHA map을 INPUT_BUNDLE에 봉인한다. load_tables는 그 map만 읽으며 당시 없던 V를 이후 live 파일로 보충하지 않는다. ordered metadata의 parent_artifact_sha256와 일반 metadata의 artifact_sha256를 각각 검사한다. reused_prediction_cache가 기록되지 않으면 None/UNKNOWN으로 보존한다.\n\n')
        handle.write('표시 family는 봉인한 registry scientific row와 SOURCE spec을 대조한다. E4의 underlying BASE family B0는 full-C PCA/ordered 모델 구현을 뜻하며, 실제 k_grid가 [3,6,11]인지 [11]인지 확인해 Shared ordered K와 Max-K trained를 구분한다. underlying family 수치는 바꾸지 않는다. 전체 registry mutable byte는 고정하지 않고 선택한 scientific row만 전후 동일성을 확인한다.\n\n')
        handle.write('V2 봉인 입력도 읽을 수 있으며, 그 입력에 기록되지 않은 ordered V는 소급 추정하지 않는다. backward 재현의 숫자·그림 동일성 여부는 별도의 V3 검산 proof에서 명시한다. 새 전체 대표 그림 생성은 실제 완료 조건과 ROOT trigger 이후에만 수행한다.\n')
    new_json(directory / 'OUTPUTS_SHA256.json', {p.relative_to(directory).as_posix(): {'sha256': sha(p.read_bytes()), 'bytes': p.stat().st_size} for p in sorted(directory.rglob('*')) if p.is_file()})
    print(json.dumps({'output': directory.as_posix(), 'displayed_runs': len(bundle['run_ids']),
                      'V_available': sum(x['new_V_present'] for x in tables['COMPARISONS.csv']),
                      'repeat_render': 'BYTE_EXACT_PASS', 'scope': 'CPU metadata only'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
