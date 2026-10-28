"""Seal the approved five-seed study and verify reusable Full-P evidence."""
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from . import runtime, run_one


CODE = Path(__file__).resolve().parent
SEEDS = list(range(92601, 92606))
FAMILIES = ['NO_F_LINEAR:jena', 'NO_F_LINEAR_NO_P:jena', 'NO_F_LINEAR:appliances', 'NO_ANCHOR:ETTh1']
OPTIONAL = ['A5_LAG0:ETTh1', 'NO_ANCHOR:appliances']
LEGACY_NO_SCALER = {'HISTORICAL_jena_K6_s92601', 'HISTORICAL_jena_K6_s92602'}


def write(path, value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf8') != text:
            raise FileExistsError('Refusing to overwrite registered preparation: ' + str(path))
        return
    with path.open('x', encoding='utf8', newline='\n') as stream:
        stream.write(text)


def jobs_for(families, extended=False):
    rows = []
    for family in families:
        method, dataset = family.split(':')
        k = runtime.KS[dataset]
        for seed in SEEDS:
            rows.append(dict(job_id=f'PJ_{method}_{dataset}_K{k}_s{seed}' + ('_ext10' if extended else ''),
                             method_id=method, dataset=dataset, K=k, seed=seed, extended=extended))
    return rows


def reference_evidence():
    source = runtime.REPO / 'reports/comparative_ablation_followup_20261027_v1/RUNS.csv'
    with source.open(encoding='utf-8-sig', newline='') as stream:
        source_rows = list(csv.DictReader(stream))
    rows = [r for r in source_rows if int(r['seed']) in SEEDS and
            ((r['method_id'] == 'OURS' and r['dataset'] in runtime.KS) or
             (r['method_id'] == 'A1' and r['dataset'] == 'jena' and str(r['extended']).lower() == 'true'))]
    bounds = {r['dataset']: r['split_ranges'] for r in runtime.read(
        runtime.DEFAULT_SOURCE / 'reports/k_tau_rule_20261023_v1/CURRENT_DATA_FINGERPRINTS.json')}
    checks, baseline_extension = [], {}
    for dataset in runtime.KS:
        full = sorted((r for r in rows if r['dataset'] == dataset and r['method_id'] == 'OURS'), key=lambda r: int(r['seed']))
        if [int(r['seed']) for r in full] != SEEDS:
            raise ValueError('Full-P reference five-seed coverage differs: ' + dataset)
        group = [r for r in rows if r['dataset'] == dataset]
        canonical_row = next(r for r in full if int(r['seed']) == 92603)
        with np.load(canonical_row['prediction_path'], allow_pickle=False) as archive:
            canonical = {key: archive[key] for key in ('target_normalized', 'observed', 'origins', 'channel_ids', 'mean', 'std')}
        for row in group:
            if (int(row['K']) != runtime.KS[dataset]
                    or int(row['normalization_fit_end']) != runtime.FIT_ENDS[dataset][1]
                    or float(row['training_cap']) != (10 if row['method_id'] == 'A1' else 3)
                    or (str(row['extended']).lower() == 'true') != (row['method_id'] == 'A1')):
                raise ValueError('Reference protocol differs: ' + row['job_id'])
            path = Path(row['prediction_path'])
            if runtime.sha(path) != row['prediction_sha256']:
                raise ValueError('Reference prediction hash mismatch: ' + row['job_id'])
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            for key, target in canonical.items():
                if key not in arrays:
                    if key in ('mean', 'std') and row['job_id'] in LEGACY_NO_SCALER:
                        continue
                    raise ValueError('Reference array missing: ' + key)
                if not np.array_equal(arrays[key], target, equal_nan=target.dtype.kind in 'fc'):
                    raise ValueError('Reference cohort/scaler differs: ' + dataset + '/' + key)
            p, y, mask = arrays['pred_normalized'], arrays['target_normalized'], arrays['observed'].astype(bool)
            if p.shape != y.shape or mask.shape != y.shape or y.shape[1] != 48:
                raise ValueError('Reference prediction shape differs')
            target_end = int(arrays['origins'].max()) + 48
            if target_end > bounds[dataset]['V_select'][1] or target_end >= bounds[dataset]['T_final'][0]:
                raise ValueError('Reference target exceeds the safe V cohort')
            difference = p.astype(np.float64) - y.astype(np.float64)
            if not np.isfinite(difference[mask]).all():
                raise ValueError('Nonfinite observed reference residual')
            counts = mask.sum(axis=(0, 1))
            active = counts > 0
            if not active.any():
                raise ValueError('Reference cohort has no observed channel')
            sse = np.where(mask, difference ** 2, 0).sum(axis=(0, 1))
            sae = np.where(mask, np.abs(difference), 0).sum(axis=(0, 1))
            mse, mae = float(np.mean(sse[active]/counts[active])), float(np.mean(sae[active]/counts[active]))
            if max(abs(mse-float(row['MSE'])), abs(mae-float(row['MAE']))) > 1e-9:
                raise ValueError('Reference FP64 saved-array metrics differ')
            checks.append(dict(job_id=row['job_id'], dataset=dataset, status='PASS', shape=list(p.shape),
                               prediction_sha256=row['prediction_sha256'], origin_max=int(arrays['origins'].max()),
                               target_end=target_end, T_start=bounds[dataset]['T_final'][0],
                               normalization_arrays_present=all(k in arrays for k in ('mean', 'std'))))
        cap_count = sum(float(r['selected_epoch']) == 3. for r in full)
        baseline_extension[dataset] = dict(selected_epochs=[float(r['selected_epoch']) for r in full],
                                           cap_selected_count=cap_count, trigger=3, required=cap_count >= 3)
        if cap_count >= 3:
            raise ValueError('Matched Full-P baseline needs an unregistered extension: ' + dataset)
    no_p = sorted((r for r in rows if r['method_id'] == 'A1'), key=lambda r: int(r['seed']))
    if [int(r['seed']) for r in no_p] != SEEDS:
        raise ValueError('No-P reference extended five-seed coverage differs')
    no_p_evidence = []
    for row in no_p:
        result_path = runtime.REPO / 'reports/comparative_ablation_20261025_v1/jobs' / row['job_id'] / 'RESULT.json'
        result = runtime.read(result_path)
        selection_path = Path(result['selection_path'])
        selection = runtime.read(selection_path)
        original = next(r for r in source_rows if r['job_id'] == row['job_id'].removesuffix('_ext10'))
        if (result['status'] != 'COMPLETE' or result['row']['selected_epoch'] != float(row['selected_epoch'])
                or runtime.sha(selection_path) != result['selection_sha256'] or selection['V_used'] is not False
                or [float(r['epoch']) for r in selection['curves']] != [0., .25, 1., 3., 6., 10.]):
            raise ValueError('Historical No-P extension evidence differs')
        no_p_evidence.append(dict(job_id=row['job_id'], original_cap3_selected_epoch=float(original['selected_epoch']),
                                  extended_selected_epoch=float(row['selected_epoch']), result_path=str(result_path),
                                  result_sha256=runtime.sha(result_path), selection_path=str(selection_path),
                                  selection_sha256=runtime.sha(selection_path)))
    return dict(status='VERIFIED', rows=rows, source=str(source), source_sha256=runtime.sha(source),
                validation=checks, baseline_extension=baseline_extension,
                no_p_extension_evidence=no_p_evidence,
                legacy_without_scaler=sorted(LEGACY_NO_SCALER), T_numeric_values_parsed=0)


def main():
    seal_path = runtime.REPORT / 'REGISTRATION_SEAL.json'
    if seal_path.exists():
        plan = runtime.read(runtime.REPORT / 'RUN_PLAN.json')
        run_one.validate_registration(plan['jobs'][0])
        print(json.dumps(dict(status='EXISTING_SEAL_VERIFIED')))
        return
    references = reference_evidence()
    write(runtime.REPORT / 'REFERENCE_RUNS.json', references)
    preserved = set()
    for name in ('comparative_ablation_20261025_v1', 'comparative_ablation_followup_20261027_v1', 'jena_p_branch_20261027_v1'):
        preserved.update(p for p in (runtime.REPO / 'reports' / name).rglob('*') if p.is_file())
    preserved.update(p for p in (runtime.REPO / 'research').rglob('*.py') if not p.is_relative_to(CODE))
    preserved.update(Path(r['prediction_path']) for r in references['rows'])
    baseline = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    scope='Existing research Python, three completed campaign report trees, and reused prediction archives; hashing does not parse T.',
                    files=[dict(path=str(p.resolve()), sha256=runtime.sha(p)) for p in sorted(preserved)])
    write(runtime.REPORT / 'PRESERVATION_BASELINE.json', baseline)
    sources = {p.relative_to(runtime.REPO).as_posix(): runtime.sha(p) for p in
               [CODE / name for name in (*run_one.SOURCE_NAMES, 'prepare.py')]}
    protocol = dict(campaign=runtime.NAME, status='PREREGISTERED_BEFORE_TRAINING',
        authority='User explicitly approved further justified experiments and main publication; parent orchestrator owns Git and launch.',
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime.REPO, text=True).strip(),
        registered_sources=sources, training_deadline='2026-10-28T17:30:00+09:00',
        delivery_deadline='2026-10-28T20:00:00+09:00', seeds=SEEDS, base_runs=20, optional_base_runs=10,
        datasets=runtime.KS,
        methods=dict(NO_F_LINEAR='Fresh original P/E/D plus channel-shared Linear(512,48,bias=True) instead of F; all jointly trained. Head initialized to latent persistence, adds 24624 parameters; zero foundation construction/calls.',
                     NO_F_LINEAR_NO_P='Same fresh original B2 initialization, then replace P by Identity; retain initialized E/D and add the identical trainable shared Linear512-to48. With existing Full-P/No-P Bolt and new Full-P Linear, forms an exploratory 2x2 with disclosed cap schedules.',
                     NO_ANCHOR='Remove only the explicit x_last output addition. Keep P/E/frozen F/lag7 D, latent differencing, original PCA initialization, and no new decoder bias.',
                     A5_LAG0='Keep x_last anchor and only lag0 decoder (future-z_last)D0, removing lag1/2/4/8/16/32 coefficients.'),
        initialization='Construct original full B2 at each original seed and stage. Common P/E/D tensors are unchanged before the registered intervention. NO_F head is added only after P/E/D initialization.',
        training=dict(L=512, H=48, optimizer='AdamW', lr=.001, weight_decay=0, gradient_clip=1,
                      effective_batch=16, microbatch=4, F_row_microbatch=128, checkpoint_inputs=False,
                      device='cuda', dtype='float32', revin=False,
                      base_epochs=[0, .25, 1, 3], extension_epochs=[0, .25, 1, 3, 6, 10],
                      selection='Minimum A_stop MSE, earlier exact tie, sealed before fresh A60 refit and V evaluation.'),
        data_bounds=dict(FIT_ENDS=runtime.FIT_ENDS, T_numeric_values_parsed=0,
                         chronology='A_fit 0-50%; A_stop 50-60%; fresh A60 refit; V70-80%; bounded loaders exclude protected T80-100%.'),
        extension_policy=dict(trigger='At least 3/5 base seeds select cap3', terminal='At least 3/5 cap10 selections imply UNDERTRAINED_AT_10; no further extension.',
                              action='Whole five-seed family cap10 only if time admission fits; no partial family is primary.',
                              order='Required base20, required family extensions, optional A5_LAG0 ETTh1 plus its extension, optional NO_ANCHOR Appliances plus its extension.',
                              decision_data='A_stop selected epochs and wall time only; no V score controls execution.',
                              baseline_extension=references['baseline_extension']),
        controller_policy=dict(concurrency=1, base_job_seconds=1080, extension_job_seconds=1800,
                               admission_margin_seconds=120, no_F_reestimate='max completed same-method/same-dataset base wall time *1.35; cap10 multiplier10/3; floor60s',
                               admission='Whole family five seeds before any family admission; worker hard deadline17:30KST.',
                               retries=1, retry='Technical same-configuration failure only; verified completed jobs reused, scientific failures stop.'),
        analysis=dict(primary_metric='Normalized development-V MSE', secondary_metric='Normalized development-V MAE',
                      contrasts=[dict(comparison_id=key, dataset=dataset, variant_method_id=variant, reference_method_id=reference)
                                 for key, dataset, variant, reference in [
                                     ('F_WITH_P_JENA', 'jena', 'NO_F_LINEAR', 'OURS'),
                                     ('F_WITHOUT_P_JENA', 'jena', 'NO_F_LINEAR_NO_P', 'A1'),
                                     ('P_WITHOUT_F_JENA', 'jena', 'NO_F_LINEAR_NO_P', 'NO_F_LINEAR'),
                                     ('P_WITH_F_JENA', 'jena', 'A1', 'OURS'),
                                     ('F_WITH_P_APPLIANCES', 'appliances', 'NO_F_LINEAR', 'OURS'),
                                     ('ANCHOR_ETTH1', 'ETTh1', 'NO_ANCHOR', 'OURS'),
                                     ('LAG_ETTH1', 'ETTh1', 'A5_LAG0', 'OURS'),
                                     ('ANCHOR_APPLIANCES', 'appliances', 'NO_ANCHOR', 'OURS')]],
                      interaction='Per matched Jena seed: (NO_F_LINEAR_NO_P - NO_F_LINEAR) - (A1 - OURS) in V MSE; report all differences and their median. Descriptive schedule-conditional interaction, not a causal interaction test.',
                      comparator='Reuse Full-P same dataset first five seeds; no new Full baseline needed (all cap3 counts0/5). Reuse Jena No-P cap10 first5; original cap3 selections5/5, extended cap10 selections0/5.',
                      summaries=['all seed points', 'median', 'Q1/Q3', 'paired seed difference', 'wall time', 'parameters', 'selected epochs'],
                      decision='Descriptive median gap versus maximum IQR, unchanged from preceding campaign; not significance or equivalence.',
                      limits=['Exploratory development V repeatedly viewed; no independent T claim.',
                              'n=5 is a time-limited pilot and does not establish equivalence.',
                              'F comparison jointly retrains P/E/D; replacement has extra24624 trainable parameters.',
                              'Jena 2x2 reveals a descriptive interaction only; historical Full-P cap3 and No-P cap10 schedules are disclosed and not a fully balanced compute control.',
                              'No-anchor retains latent differences and no new bias; an optimization/parameterization intervention, not universal necessity of a direct path.',
                              'A5 tests decoder lag removal only, not an input temporal-branch claim.',
                              'Budget-excluded or cap-limited comparisons stay explicitly incomplete/provisional.']))
    write(runtime.REPORT / 'PREREGISTRATION.json', protocol)
    write(runtime.REPORT / 'RUN_PLAN.json', dict(campaign=runtime.NAME,
          jobs=jobs_for(FAMILIES), optional_jobs=jobs_for(OPTIONAL), extension_jobs=jobs_for(FAMILIES+OPTIONAL, True),
          family_order=FAMILIES, optional_family_order=OPTIONAL,
          training_deadline=protocol['training_deadline'], delivery_deadline=protocol['delivery_deadline']))
    markdown = '''# PPT 당위성 보완 실험 사전등록

기본20회: NO_F_LINEAR Jena5 → NO_F_LINEAR_NO_P Jena5 → NO_F_LINEAR Appliances5 → NO_ANCHOR ETTh1 5. 모든 seed는92601–92605다.
선택10회: A5_LAG0 ETTh1 5 → NO_ANCHOR Appliances5를 고정 순서로 예산이 허용할 때만 수행한다.
학습은10월28일17:30KST까지, 검증·발표자료 전달은20:00KST까지다. 승인된 새 캠페인만 수정한다.

F 제거: 기존 R5b 설계처럼 전체 P/E/D를 같은 초기화에서 새로 학습하고 F만 공유 Linear512→48으로 대체한다.
head는 마지막 latent를 반복하는 persistence 초기화이며24624개 파라미터가 추가된다. F 생성·호출은0이다.
NO_F_LINEAR_NO_P는 같은 fullB2 초기화 후P를Identity로 바꾸고 동일head를 추가한다. 기존Full/NoP Bolt와 함께P유무×예측기2×2 탐색을 구성한다.
기존NoP Jena 첫5는 cap3에서5/5 상한선택 후cap10으로 연장되어[3,6,6,6,6]을 선택했다. Full의cap3과 이 일정 차이를 공개한다.
Jena 사전 대비는 NO_F_LINEAR−OURS, NO_F_LINEAR_NO_P−A1, NO_F_LINEAR_NO_P−NO_F_LINEAR, A1−OURS다.
seed별 조건부 interaction=(NO_F_LINEAR_NO_P−NO_F_LINEAR)−(A1−OURS)의 전체 값과 중앙값을 기술한다. 일정/용량이 완전히 균형 잡힌 인과 상호작용 검정으로 부르지 않는다.
직접 수준 가산 제거: x_last 더하기만 없애며7lag D와 latent 차분·P/E/F·초기화를 유지한다. 새 bias는 없다.
입력 observed history와 z_last 및 lag별 관측 latent 차분은 여전히 남는다. 모든 과거 관측값을 제거한 실험으로 해석하지 않는다.
A5: x_last 가산은 유지하고 lag0만 사용한다. 세 질문을 서로 대체하거나 하나의 효과로 묶지 않는다.

원래 A_fit/A_stop, AdamW(.001,wd0), clip1, effective16/micro4, FP32, K(Jena6/App10/ETTh1 4), L512/H48을 유지한다.
checkpoint0/.25/1/3에서 A_stop 최소값으로 선택하고 동률은 이른 epoch, 선택 봉인 후 A60 fresh refit, 마지막 V 평가를 수행한다.
전체5개중3개이상 cap3 선택 시 해당5개 전체를0/.25/1/3/6/10으로 연장한다. 연장5개가 모두 끝나야 주 결과가 된다.
cap10도3/5이상 선택하면 학습 상한 미해결로 표시하고 더 반복하지 않는다. Full-P 첫5개는 모든 데이터 cap3선택0/5라 연장이 필요 없다.
우선순위는 기본20 → 그 연장 → 선택A5 ETTh1와 그 연장 → 선택NO_ANCHOR App와 그 연장이다.
각 묶음은 남은 시간에5개 모두 끝낼 수 있어야 시작한다. F유지 기본18분/연장30분으로 보수 추정한다.
NO_F는 첫 성공실행을 포함한 같은 데이터 완료시간 최댓값×1.35로 갱신하며 연장은10/3배, 하한60초다.
V의 좋고 나쁨으로 실행 범위·seed·선택·임계값을 바꾸지 않는다. 기술적 실패는 같은 설정1회 재시도, 과학 게이트 실패는 중단한다.

기존 Full-P 같은5seed의 예측 해시·cohort·정규화·FP64 지표를 먼저 검산해 재사용한다. 보호T 숫자는 읽지 않는다.
모든 seed, 중앙값/IQR, pair별 차이, 파라미터·시간·선택epoch를 공개한다. 중앙값/max-IQR 규칙은 기술적 비교이며 유의성·동등성 검정이 아니다.
n5·기존V 반복사용·새head 용량차이·no-anchor 매개화/최적화 차이·미완료/상한 상태를 숨기지 않는다.
결과가 방법론을 지지하지 않으면 주장 범위를 줄이고 기존 부정/불명확 결과를 보존한다.
'''
    write(runtime.REPORT / 'PREREGISTRATION.md', markdown)
    names = ('PREREGISTRATION.json', 'PREREGISTRATION.md', 'RUN_PLAN.json', 'PRESERVATION_BASELINE.json', 'REFERENCE_RUNS.json')
    write(seal_path, dict(created_utc=datetime.now(timezone.utc).isoformat(),
                         method='Immutable-before-training local SHA256 seal; parent owns Git publication.',
                         files={name: runtime.sha(runtime.REPORT / name) for name in names}))
    print(json.dumps(dict(status='SEALED', base_jobs=20, optional_jobs=10, extension_candidates=30,
                         preserved_files=len(baseline['files']), reference_rows=len(references['rows']))))


if __name__ == '__main__':
    main()
