"""Audit only saved public A03 JSON; never imports scientific/model code."""
import argparse
import collections
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import struct


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    report = repo/'reports/adapter_diagnostics_20261017_v2'
    plan_path = report/'EXPERIMENT_PLAN.json'
    plan_file_sha256 = sha(plan_path)
    plan = read(plan_path)
    sealed_plan_sha256 = plan['plan_sha256']
    body = {key: value for key, value in plan.items() if key != 'plan_sha256'}
    recomputed_seal = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                               ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    jobs = [job for job in plan['jobs'] if job['question_id'] == 'A03']
    families = ('B0', 'B2', 'B2_CAP_POINTWISE')
    seeds = (92601, 92602)
    splits = ('A_fit', 'A_stop', 'V_select')
    epochs = (0., .25, 1., 3.)
    expected = set(itertools.product(families, seeds, range(4), splits))
    actual = [(job['family'], job['seed'], job['endpoint_index'], job['split']) for job in jobs]
    problems = []

    def check(ok, message):
        if not ok:
            problems.append(message)

    check(recomputed_seal == sealed_plan_sha256, 'Plan canonical sealed identity mismatch')
    check(len(jobs) == 72 and set(actual) == expected and len(set(actual)) == 72,
          'A03 cartesian coverage differs from 3x2x4x3')
    legal = list(range(512, 10482-48+1))
    exposures = [math.ceil(epoch*len(legal)) for epoch in epochs]
    updates = [0]
    for lo, hi in zip(exposures, exposures[1:]):
        updates.append(updates[-1]+math.ceil((hi-lo)/16))
    bounds = {'A_fit': (0, 10482), 'A_stop': (10482, 12579), 'V_select': (14675, 16772)}
    origins = {split: list(range(max(512, bound[0]), bound[1]-48+1, 1 if split == 'A_fit' else 48))
               for split, bound in bounds.items()}
    origin_hash = {split: hashlib.sha256(struct.pack('<'+'q'*len(value), *value)).hexdigest()
                   for split, value in origins.items()}
    columns = read(repo/'research/informs_broad_validation_20261015_v1/DATA_CONTRACT.json')['assets']['jena']['columns']
    width = max(map(len, columns))
    channel_hash = hashlib.sha256(b''.join(column.ljust(width, '\0').encode('utf-32le')
                                          for column in columns)).hexdigest()
    audit = read(report/'DATA_AUDIT.json')['datasets']['jena']['checks']
    scalers = {split: read(repo/'reports/informs_b2_why_where_20261017/same_update/s92601'/
                          ('B0_e0_'+split+'.json'))['scaler_sha256'] for split in splits}
    check(scalers['V_select'] == audit['A60_scaler_sha256'],
          'V scaler differs from independent DATA_AUDIT canonical A60')
    results = {}
    groups = collections.defaultdict(list)
    parity = []
    public_bindings = {}
    max_rescore = 0.
    artifact_count = 0
    for job in jobs:
        identifier = job['job_id']
        score_path = report/'scores'/(identifier+'.json')
        receipt_path = report/'receipts'/(identifier+'.json')
        score, receipt = read(score_path), read(receipt_path)
        index, split = job['endpoint_index'], job['split']
        results[(job['family'], job['seed'], index, split)] = score
        check(receipt.get('status') == 'COMPLETE' and score.get('status') == 'COMPLETE', identifier+': not COMPLETE')
        check(receipt.get('scientific_identity') == job['scientific_identity'], identifier+': plan/receipt identity mismatch')
        check(receipt.get('result') == score, identifier+': score differs from receipt result')
        for relative, expected_sha in receipt.get('artifact_hashes', {}).items():
            path = (report/relative).resolve()
            check(path.is_relative_to(report.resolve()) and path.is_file() and sha(path) == expected_sha,
                  identifier+': public artifact SHA mismatch '+relative)
            artifact_count += 1
        check(receipt.get('artifact_hashes', {}).get('scores/'+identifier+'.json') == sha(score_path),
              identifier+': score not bound by receipt')
        check((score.get('model'), score.get('seed'), score.get('eval')) == (job['family'], job['seed'], split),
              identifier+': job axes mismatch')
        check((score.get('epoch'), score.get('updates'), score.get('exposure')) ==
              (epochs[index], updates[index], exposures[index]), identifier+': registered progress mismatch')
        check((score.get('C'), score.get('K'), score.get('L'), score.get('H')) == (21, 6, 512, 48),
              identifier+': registered dimensions mismatch')
        check(score.get('device') == 'cuda' and score.get('dtype') == 'float32' and score.get('new_forward') is True,
              identifier+': actual CUDA FP32 route mismatch')
        check(score.get('fit') == 'A_fit_search' and score.get('selection') == 'NOT_SELECTED_COMMON_ENDPOINT',
              identifier+': selected/refit mixed into endpoint')
        check(score.get('metric_scale') == ('canonical_A60' if split == 'V_select' else 'search_A50'),
              identifier+': wrong stage scale')
        check(score.get('scaler_sha256') == scalers[split], identifier+': scaler differs from registered stage')
        check(score.get('source_code_sha256') == job['scientific_source_hashes'][
            'research/adapter_diagnostics_20261017_v2/gpu_endpoints.py'], identifier+': replay source differs from plan')
        arrays = score['array_fingerprints']
        shape = [len(origins[split]), 48, 21]
        for name in ('pred_normalized', 'target_normalized', 'observed'):
            check(arrays[name]['shape'] == shape, identifier+': array shape mismatch '+name)
        check(arrays['origins'] == dict(shape=[len(origins[split])], dtype='int64', sha256=origin_hash[split]),
              identifier+': origin fingerprint differs from independent registered origins')
        check(arrays['channel_ids'] == dict(shape=[21], dtype='<U'+str(width), sha256=channel_hash),
              identifier+': channel fingerprint differs from registered order')
        check(arrays['target_normalized']['dtype'] == 'float32' and arrays['pred_normalized']['dtype'] == 'float32'
              and arrays['observed']['dtype'] == 'bool', identifier+': stored dtypes mismatch')
        signature = {key: arrays[key] for key in ('target_normalized', 'observed', 'origins', 'channel_ids')}
        signature['scaler_sha256'] = score['scaler_sha256']
        signature['metric_scale'] = score['metric_scale']
        groups[split].append((identifier, signature))
        proof = score['parity']
        check(proof['status'] == 'PASS' and proof['atol'] == 1e-5 and proof['rtol'] == 1e-4,
              identifier+': parity status/tolerance mismatch')
        check(proof['origin'] == origins[split][0] and proof['checkpoint_sha256'] == score['checkpoint_sha256'],
              identifier+': parity origin/checkpoint mismatch')
        check(proof.get('bitwise_equivalence_claimed') is False, identifier+': unsupported bitwise claim')
        parity.append((identifier, proof['max_absolute_difference']))
        counts = score['channel_counts']
        mse = math.fsum(value/count for value, count in zip(score['channel_sse'], counts))/len(counts)
        mae = math.fsum(value/count for value, count in zip(score['channel_sae'], counts))/len(counts)
        difference = max(abs(mse-score['MSE']), abs(mae-score['MAE']))
        max_rescore = max(max_rescore, difference)
        check(difference <= 1e-10, identifier+': public FP64 macro mismatch')
        public_bindings['scores/'+identifier+'.json'] = sha(score_path)
        public_bindings['receipts/'+identifier+'.json'] = sha(receipt_path)
    for split, group in groups.items():
        check(len(group) == 24 and all(value == group[0][1] for _, value in group),
              split+': target/mask/origin/channel/scaler not common across 24 conditions')
    cp0 = []
    for seed, split in itertools.product(seeds, splits):
        values = [results[(family, seed, 0, split)] for family in families]
        reference = values[0]
        difference = max(abs(value[key]-reference[key]) for value in values for key in ('MSE', 'MAE'))
        same = len({value['array_fingerprints']['pred_normalized']['sha256'] for value in values}) == 1
        check(difference == 0 and same, 'CP0 output/score mismatch seed'+str(seed)+' '+split)
        cp0.append(dict(seed=seed, split=split, families=3, max_score_difference=difference,
                        stored_prediction_fingerprint_equal=same))
    check(sha(plan_path) == plan_file_sha256, 'Plan changed during independent check')
    max_parity = max(parity, key=lambda value: value[1])
    summary = dict(status='PASS' if not problems else 'FAIL', checked_jobs=len(jobs), coverage_shape=[3, 2, 4, 3],
                   fingerprint_groups=len(groups), conditions_per_fingerprint_group=24, CP0_family_groups=len(cp0),
                   parity_pass_jobs=sum(1 for _ in parity), max_CPU_CUDA_absolute_difference=max_parity[1],
                   max_CPU_CUDA_difference_job=max_parity[0], max_public_FP64_rescore_absolute_difference=max_rescore,
                   CP0_max_score_difference=max(value['max_score_difference'] for value in cp0),
                   CP0_stored_prediction_fingerprint_equal=all(value['stored_prediction_fingerprint_equal'] for value in cp0),
                   checked_public_artifact_hashes=artifact_count, problems=problems)
    exit_code = 0 if not problems else 1
    value = dict(summary, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 plan_file_sha256=plan_file_sha256, sealed_plan_sha256=sealed_plan_sha256,
                 sealed_plan_canonical_recompute='PASS' if recomputed_seal == sealed_plan_sha256 else 'FAIL',
                 verification_script_sha256=sha(Path(__file__)),
                 verification_command='python -B reports/adapter_diagnostics_20261017_v2/code/verify_a03_snapshot.py --repo . --cache-root PRIVATE_CACHE',
                 verification_exit_code=exit_code,
                 method='Independent stdlib-only public JSON/plan/receipt audit; no model/checkpoint/NPZ loaded',
                 registered_epochs=epochs, registered_exposures=exposures, registered_updates=updates,
                 split_fingerprints={split: group[0][1] for split, group in groups.items()}, CP0_groups=cp0,
                 public_source_bindings=public_bindings,
                 claim_scope='CP0 C21K6 equal stored endpoint predictions among B0/B2/CAP only; no K=C/Raw equivalence claim')
    path = args.cache_root/'A03_INDEPENDENT_COHORT_CHECK.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('PRIVATE_CHECK_SHA256', sha(path))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
