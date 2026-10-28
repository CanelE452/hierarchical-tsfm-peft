"""Prospective bounded cross-dataset screen; root seals before training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from . import runtime, run_one


CODE = Path(__file__).resolve().parent
METHODS = ('OURS', 'TIME_ONLY', 'A5_LAG0', 'E_FIXED')
SOURCE_NAMES = (*run_one.SOURCE_NAMES, 'host_gpu_gate.py', 'prepare.py',
                'tests/test_models.py', 'tests/test_execution.py', 'tests/test_controller.py')


def jobs(seeds):
    return [dict(job_id=f'XC_{method}_{dataset}_K{axes["K"]}_s{seed}',
                 method_id=method, dataset=dataset, K=axes['K'], seed=seed,
                 training_cap=axes['cap'])
            for dataset, axes in runtime.DATASETS.items() for seed in seeds for method in METHODS]


def write(path, value):
    content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf8') != content:
            raise FileExistsError('Prepared document already differs: '+str(path))
        return
    with path.open('x', encoding='utf8', newline='\n') as stream:
        stream.write(content)


def sources():
    entries = {f'research/{runtime.NAME}/{name}': runtime.sha(CODE/name) for name in SOURCE_NAMES}
    for parent in ('ppt_e_fixed_20261028_v1', 'ppt_justification_20261028_v1'):
        for path, digest in runtime.read(runtime.REPO/'reports'/parent/'PREREGISTRATION.json')['registered_sources'].items():
            if runtime.sha(runtime.REPO/path) != digest:
                raise ValueError('Prior registered source changed: '+path)
            entries[path] = digest
    broad = runtime.REPO/'research/informs_broad_validation_20261015_v1'
    extras = ['fit_base.py', 'data_contract.py', 'training.py', 'evaluation.py', 'networks.py', 'gpu_runtime.py',
              'priority_b2_validation_20261016_v1/safe_jena_prefix.py',
              'priority_pems_subset_20261016_v1/subset_minimum_runner.py',
              'priority_nested_cost_20261016_v1/prefix_nested_cost_runner.py']
    extras += list(runtime.read(broad/'priority_pems_subset_20261016_v1/PLAN.json')['dependencies'])
    for name in extras:
        path = broad/name
        entries[path.relative_to(runtime.REPO).as_posix()] = runtime.sha(path)
    for name in ('research/jena_p_branch_20261027_v1/models.py',
                 'research/comparative_ablation_20261025_v1/models.py',
                 'research/adapter_diagnostics_20261017_v2/existing_jobs.py'):
        entries[name] = runtime.sha(runtime.REPO/name)
    return dict(sorted(entries.items()))


def preservation():
    inherited = runtime.read(runtime.REPO/'reports/ppt_e_fixed_20261028_v1/PRESERVATION_BASELINE.json')
    entries = {row['path']: row['sha256'] for row in inherited['files']}
    names = subprocess.check_output(['git', 'ls-files', '--',
        'research/ppt_justification_20261028_v1', 'reports/ppt_justification_20261028_v1',
        'research/ppt_e_fixed_20261028_v1', 'reports/ppt_e_fixed_20261028_v1'],
        cwd=runtime.REPO, text=True).splitlines()
    for name in names:
        if '/presentation/' not in name:
            path = (runtime.REPO/name).resolve()
            entries[str(path)] = runtime.sha(path)
    broad = runtime.REPO/'research/informs_broad_validation_20261015_v1'
    for relative in ('DATA_CONTRACT.json', 'priority_pems_subset_20261016_v1/PLAN.json'):
        path = broad/relative
        entries[str(path.resolve())] = runtime.sha(path)
    changed = [name for name,digest in entries.items() if not Path(name).is_file() or runtime.sha(name)!=digest]
    if changed:
        raise ValueError('Prior protected input changed: '+str(changed))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Inherited frozen data/code/results plus published preceding scientific campaigns; presentation files excluded.',
        files=[dict(path=name,sha256=digest) for name,digest in sorted(entries.items())])


def protocol(source_hashes):
    return dict(campaign=runtime.NAME, status='PREREGISTERED_BEFORE_TRAINING',
        authority='User explicitly requested option2: explore other datasets to find conditional component benefit; all attempted outcomes retained.',
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime.REPO,text=True).strip(),
        registered_sources=source_hashes, unit_test_receipt_sha256=runtime.sha(runtime.REPORT/'UNIT_TESTS.json'),
        host_gpu_gate=runtime.host_gpu_registration(),
        training_deadline='2026-10-28T17:30:00+09:00',delivery_deadline='2026-10-28T20:00:00+09:00',
        base_runs=16,optional_runs=8,extension_runs=0,base_seeds=[92601,92602],optional_seeds=[92603],
        datasets=runtime.DATASETS,
        hypotheses=dict(TIME_ONLY='Removing the channel branch worsens forecasting in some multivariate conditions.',
                        A5_LAG0='Removing extra decoder lags, while retaining lag0 and last-context addition, worsens forecasting.',
                        E_FIXED='Freezing stage-local PCA E, while P/D remain trainable, worsens forecasting.'),
        design=dict(baseline='Fresh OURS for every matched seed; no historical baseline prediction reuse.',
                    contrasts='Three ablations versus same-seed OURS; positive ablation-minus-OURS MSE supports the component in this condition.',
                    time_only='Remove channel residual branch only; retain temporal branch and shared joint LayerNorm. Not a channel-independent whole model.',
                    E_fixed='Same initial state; E excluded from actual optimizer, checked unchanged within search and within fresh A60 refit separately.',
                    lag0='Keep direct last-context addition; use only lag0 decoder features.',
                    scope='Conditional implementation-level effects; capacity/optimization differences are part of these interventions.'),
        selection=dict(datasets='Available safe-prefix traffic and air-quality domains beyond earlier Jena/Appliances/ETT; chosen with historical results known.',
                       pems32='Fixed32 sensors chosen by sensor-ID SHA ordering; derived from PEMS08, not an independent second dataset.',
                       K='Inherited fixed K8/K6 to isolate components, not a test of K-tau optimality.',
                       caps='PeMS6 prospectively because prior Full hit3 and6; Beijing3. Same cap within each dataset; no outcome-based extensions.',
                       alternatives='ETTm1 is slower and related to ETTh1; solar137 is heavier; synthetic phase-shift stress tests could diagnose mechanisms but do not answer current real-data scope.'),
        training=dict(L=512,optimizer='AdamW',lr=.001,weight_decay=0,gradient_clip=1,
                      effective_batch=16,microbatch=4,F_row_microbatch=128,checkpoint_inputs=False,
                      device='cuda',dtype='float32',revin=False,
                      checkpoints=dict(pems32=[0,.25,1,3,6],beijing=[0,.25,1,3]),
                      selection='Minimum A_stop MSE, earlier exact tie; seal before fresh A60 refit and V evaluation.'),
        controller_policy=dict(concurrency=1,dataset_job_seconds=dict(pems32=1020,beijing=650),
            reestimate_floor_seconds=dict(pems32=850,beijing=500),reestimate_multiplier=1.35,
            admission_margin_seconds=120,retries=1,
            admission='Whole remaining4-arm group must fit. After first OURS success use maximum successful job walltime times1.35 with fixed floors. Time only, never V.',
            order=['pems32_base','beijing_base','beijing_seed3','pems32_seed3'],
            partial='If time prediction changes, unfinished jobs markedNOT_RUN_BUDGET. Incomplete optional4-arm seed not included in primary cohort.',
            retry='One identical-setting technical retry; scientific gate failure stops.',
            locking='Shared controller/worker locks and identity-verified live parent lease; no concurrent GPU experiments.'),
        analysis=dict(primary_metric='Normalized development-V channel-macro FP64 MSE',secondary_metric='MAE',
            primary_cohort='All4arms complete on both base seeds; add92603 only when all4arms complete. Show any partial results separately.',
            summaries='All seed points, paired differences, median/Q1/Q3, direction count and relative median change.',
            descriptive_rule='Absolute difference of medians must exceed maximum arm IQR, with at least ceil(2*n/3) paired seeds in same direction; otherwise UNCLEAR. Not a significance test.',
            provisional='n2 always PILOT. If either arm selects its cap in at least ceil(2*n/3) seeds, UNDERTRAINED_PROVISIONAL. n3 remains exploratory.',
            reporting='Publish every dataset, intervention, failed attempt, budget exclusion and null/negative outcome. No universal necessity, equivalence or final-test claim.',
            forbidden='No protectedT numeric reads, no winner-only reporting, no outcome-based seed/cap/dataset additions within this campaign.'),
        literature=[dict(title='TSMixer',url='https://arxiv.org/abs/2303.06053',role='Time/feature mixing motivates component comparison; does not guarantee benefit.'),
                    dict(title='PatchTST',url='https://arxiv.org/abs/2211.14730',role='Channel-independent successes motivate testing rather than assuming mixing necessity.'),
                    dict(title='DCRNN',url='https://arxiv.org/abs/1707.01926',role='Traffic spatial-temporal dependency motivates traffic domain; our model does not use its graph.')])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');args=parser.parse_args()
    if not args.seal:
        print(json.dumps(dict(status='PREVIEW',base_jobs=16,optional_jobs=8,source_sha256=sources())))
        return
    seal_path=runtime.REPORT/'REGISTRATION_SEAL.json'
    if seal_path.exists():
        run_one.validate_registration(jobs([92601])[0]);print('EXISTING_SEAL_VERIFIED');return
    source_hashes=sources();tests=runtime.read(runtime.REPORT/'UNIT_TESTS.json')
    if tests.get('status')!='PASS' or tests.get('source_sha256')!=source_hashes:
        raise ValueError('CPU test receipt absent or bound to different sources')
    registry=dict(datasets={name:runtime.describe_task(runtime.load_task(name,require_registry=False)) for name in runtime.DATASETS},
                  T_numeric_values_parsed=0,baseline_reuse=False)
    registration=protocol(source_hashes)
    write(runtime.REPORT/'DATA_REGISTRY.json',registry)
    write(runtime.REPORT/'REFERENCE_RUNS.json',dict(rows=[],reason='All4arms including Full freshly trained under matched current protocol.'))
    write(runtime.REPORT/'PRESERVATION_BASELINE.json',preservation())
    write(runtime.REPORT/'PREREGISTRATION.json',registration)
    write(runtime.REPORT/'RUN_PLAN.json',dict(campaign=runtime.NAME,jobs=jobs([92601,92602]),optional_jobs=jobs([92603]),
          extension_jobs=[],group_order=['pems32_base','beijing_base','beijing_seed3','pems32_seed3']))
    write(runtime.REPORT/'PREREGISTRATION.md','''# 다른 데이터에서 구성요소의 조건부 기여를 찾는 탐색\n\n사용자가 다른 데이터에서도 도움이 되는 조건을 찾아보도록 승인했다. PeMS32(K8,H12,cap6)와 Beijing(K6,H48,cap3)을 고정한다. 각 데이터의 OURS/TIME_ONLY/A5_LAG0/E_FIXED를 seed92601/92602로 새로 학습한다. 과거 Full 예측은 재사용하지 않는다. 두 기본 묶음 다음 Beijing, PeMS32 순으로 seed92603의 네 구성을 시간이 허락할 때만 함께 추가한다. 기본16회, 선택8회이며 학습17:30·전달20:00KST가 마감이다.\n\nTIME_ONLY는 채널 분기만 제거하며 공동 LayerNorm은 남는다. A5_LAG0는 마지막 관측값 더하기를 유지하고 추가 lag만 제거한다. E_FIXED는 각 단계의 PCA E만 고정하고 실제 optimizer 제외와 불변성을 검증한다. 선택은 A_stop에서 0/.25/1/3(PeMS는6추가) 중 최소오차·동률이면 이른 epoch이며, 봉인 후 fresh A60 refit으로 개발V를 평가한다. 보호T 숫자는 열지 않는다.\n\n전체 묶음이 남은 시간에 들어갈 때만 시작한다. 최초작업시간상한은 PeMS1020초/Beijing650초, 여유120초다. 첫 OURS 완료 후 실제 성공실행 최대시간×1.35(하한850/500초)를 적용한다. 시간으로만 선택 seed를 허용하며 V를 보고 늘리지 않는다. 동일설정 기술 재시도는1회, 과학게이트 실패는중단, epoch연장은없다.\n\n기본 두 seed의 네 구성 모두 완료되어야 주비교를 제시한다. 세번째seed도 네 구성 모두 완료일 때만 합치며, 나머지 부분결과는 따로 공개한다. 모든 seed·차이·중앙값/IQR·실제epoch·예산제외를 보고한다. 중앙값 차이가 두 IQR 중 큰 값보다 크고 ceil(2n/3)개 이상 seed가 같은 방향일 때만 기술적 방향판정을 붙인다. n2는 항상 PILOT, 비교 양쪽 중 어느 한쪽이 ceil(2n/3)개 이상 cap을 고르면 UNDERTRAINED_PROVISIONAL이다. 이 규칙은 유의성 검정이 아니다.\n\n기존 V 결과를 알고 데이터/질문을 골랐으므로 새 데이터셋에 대한 확증시험으로 부르지 않는다. PeMS32는 PEMS08의 고정 부분집합이다. K는 기존 고정값이며 K_tau의 최적성을 검사하지 않는다. 좋은 결과뿐 아니라 불명확/불리한 결과도 모두 공개한다. 작은표본·개발V·용량/최적화 차이 때문에 모든 데이터에서 필수라는 주장을 하지 않는다.\n\n선행근거: TSMixer https://arxiv.org/abs/2303.06053 ; PatchTST https://arxiv.org/abs/2211.14730 ; DCRNN https://arxiv.org/abs/1707.01926 . 이들은 비교 질문과 데이터 선택의 근거이지 우리 구성의 효과에 대한 증거가 아니다.\n''')
    names=('PREREGISTRATION.json','PREREGISTRATION.md','RUN_PLAN.json','DATA_REGISTRY.json','PRESERVATION_BASELINE.json','REFERENCE_RUNS.json')
    write(seal_path,dict(created_utc=datetime.now(timezone.utc).isoformat(),method='Prospective root SHA256 seal before training',
                        files={name:runtime.sha(runtime.REPORT/name) for name in names}))
    run_one.validate_registration(jobs([92601])[0])
    print(json.dumps(dict(status='SEALED',base_jobs=16,optional_jobs=8,T_numeric_values_parsed=0)))


if __name__=='__main__':
    main()
