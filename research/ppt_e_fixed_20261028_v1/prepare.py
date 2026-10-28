"""Prepare a separate prospective E-only intervention; sealing is a root action."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from . import runtime, run_one


CODE = Path(__file__).resolve().parent
SEEDS = list(range(92601, 92606))
SOURCE_NAMES = (*run_one.SOURCE_NAMES, 'prepare.py', 'tests/test_contract.py')


def jobs(extended=False):
    return [dict(job_id=f'PE_E_FIXED_jena_K6_s{seed}' + ('_ext10' if extended else ''),
                 method_id='E_FIXED', dataset='jena', K=6, seed=seed, extended=extended) for seed in SEEDS]


def write(path, value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf8') != text:
            raise FileExistsError('Do not overwrite a prepared registration: ' + str(path))
        return
    with path.open('x', encoding='utf8', newline='\n') as stream:
        stream.write(text)


def sources():
    return {f'research/{runtime.NAME}/{name}': runtime.sha(CODE / name) for name in SOURCE_NAMES}


def bind_references():
    parent = runtime.UPSTREAM_REPORT
    seal = runtime.read(parent / 'REGISTRATION_SEAL.json')
    for name, digest in seal['files'].items():
        if runtime.sha(parent / name) != digest:
            raise ValueError('Upstream sealed input differs: ' + name)
    inherited = runtime.read(parent / 'REFERENCE_RUNS.json')
    rows = sorted([r for r in inherited['rows'] if r['method_id'] == 'OURS' and r['dataset'] == 'jena'],
                  key=lambda r: int(r['seed']))
    if [int(row['seed']) for row in rows] != SEEDS:
        raise ValueError('Exactly the original matched five Full-P seeds are required')
    for row in rows:
        if (int(row['K']) != 6 or float(row['training_cap']) != 3
                or int(row['normalization_fit_end']) != 12579
                or runtime.sha(row['prediction_path']) != row['prediction_sha256']):
            raise ValueError('Full-P reference differs from its registered K6/A60/prediction binding')
    if sum(float(row['selected_epoch']) == 3. for row in rows) >= 3:
        raise ValueError('Full-P baseline would require an unregistered extension')
    ids = {row['job_id'] for row in rows}
    validation = [row for row in inherited['validation'] if row['job_id'] in ids]
    if len(validation) != 5 or any(row['status'] != 'PASS' for row in validation):
        raise ValueError('Inherited cohort/scaler/FP64 checks are incomplete')
    return dict(status='BOUND_TO_VERIFIED_UPSTREAM_REFERENCE', rows=rows,
                inherited_validation=validation, inherited_validation_scope='Upstream sealed reference audit; only byte hashes rechecked here, no new numeric V scoring.',
                source=str(parent / 'REFERENCE_RUNS.json'), source_sha256=runtime.sha(parent / 'REFERENCE_RUNS.json'),
                source_seal_sha256=runtime.sha(parent / 'REGISTRATION_SEAL.json'), T_numeric_values_parsed=0,
                baseline_cap_selected_count=0, selected_epochs=[float(r['selected_epoch']) for r in rows])


def preservation():
    parent = runtime.UPSTREAM_REPORT
    baseline = runtime.read(parent / 'PRESERVATION_BASELINE.json')
    entries = {row['path']: row['sha256'] for row in baseline['files']}
    upstream_protocol = runtime.read(parent / 'PREREGISTRATION.json')
    for path, digest in upstream_protocol['registered_sources'].items():
        entries[str((runtime.REPO / path).resolve())] = digest
    upstream_seal = runtime.read(parent / 'REGISTRATION_SEAL.json')
    for name, digest in upstream_seal['files'].items():
        entries[str((parent / name).resolve())] = digest
    entries[str((parent / 'REGISTRATION_SEAL.json').resolve())] = runtime.sha(parent / 'REGISTRATION_SEAL.json')
    changed = [path for path, digest in entries.items() if not Path(path).is_file() or runtime.sha(path) != digest]
    if changed:
        raise ValueError('Protected input changed: ' + ', '.join(changed))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),
                scope='Inherit upstream protected6172 entries and add only its sealed sources/input documents; do not bind mutable ongoing progress/results/cost/PPT.',
                files=[dict(path=path, sha256=digest) for path, digest in sorted(entries.items())])


def protocol(registered_sources, references, tests_sha):
    return dict(campaign=runtime.NAME, status='PREREGISTERED_BEFORE_TRAINING',
        authority='User authorized PPT justification experiments; root alone reviews, seals, commits, and launches this separate study.',
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=runtime.REPO).strip(),
        registered_sources=registered_sources, unit_test_receipt_sha256=tests_sha,
        host_gpu_gate=runtime.host_gpu_registration(),
        training_deadline='2026-10-28T17:30:00+09:00', delivery_deadline='2026-10-28T20:00:00+09:00',
        seeds=SEEDS, datasets=dict(jena=6), base_runs=5, optional_base_runs=0, extension_runs=5,
        hypothesis='At the selected K6/PCA initialization and original P/frozen F/lag7 D/anchor, allowing E to learn improves development forecasting versus freezing only E.',
        intervention='Construct the same original fresh B2 at each seed/stage, then only model.E.requires_grad_(False). P, D, anchor, frozen F, initial tensors, losses and optimizer schedule otherwise unchanged.',
        identification='Learning E at this chosen architecture/K/initialization; not PCA optimality, K optimality, or universal necessity. F remains frozen; P and D still adapt.',
        reference=dict(source_sha256=references['source_sha256'], Full_seed_count=5, cap3_selected_count=0,
                       selected_epochs=references['selected_epochs']),
        training=dict(L=512,H=48,K=6,C=21,optimizer='AdamW',lr=.001,weight_decay=0,gradient_clip=1,
                      effective_batch=16,microbatch=4,F_row_microbatch=128,checkpoint_inputs=False,
                      device='cuda',dtype='float32',revin=False,base_epochs=[0,.25,1,3],extension_epochs=[0,.25,1,3,6,10],
                      selection='Minimum A_stop MSE, earlier exact tie; selection sealed before fresh A60 refit and V evaluation.'),
        data_bounds=dict(A_fit=[0,10482],A_stop=[10482,12579],A60_refit=[0,12579],V_select=[14675,16772],T_start=16772,
                         chronology='Only bounded pre-T Jena prefix; no protected T numeric values.'),
        invariants=['Exact initial/live/selected E equality separately within search and fresh refit.',
                    'Original PCA is independently re-fitted on each allowed training endpoint; E need not equal between A_fit and A60.',
                    'E absent from actual optimizer groups at every training segment, E gradient absent.',
                    'All other parameter values match original seeded B2 immediately before training.',
                    'Full-P matched predictions are reused; completed E-fixed jobs are hash-verified, never repeated.'],
        extension_policy=dict(trigger='At least3/5 base seeds select cap3',action='Run all five cap10 candidates if triggered.',
                              terminal='At least3/5 cap10 selections means UNDERTRAINED_AT_10; no further runs.',
                              partial='Incomplete five-seed groups never become the primary result; required incomplete extension remains provisional.'),
        controller_policy=dict(concurrency=1,base_job_seconds=1080,extension_job_seconds=1800,admission_margin_seconds=120,
                               admission='Before any new study training reserve all unfinished base jobs plus every possible unfinished extension: first start requires14520s (4h2m). Time and A_stop only; V never changes admission.',
                               retries=1,retry='One same-setting technical retry; scientific integrity failure stops.',
                               prerequisites='Upstream training terminal and actual controller/run_one/cost processes absent; explicit root gate tied to this seal.',
                               locking='Parent holds upstream controller.lock; GPU child holds upstream worker.lock plus own worker.lock for its full lifetime. Lease verifies owner PID/creation/module and limited matching Python ancestry. Before CUDA loading, the shared host gate permits only prospectively sealed exact desktop identities and rejects unknown/new/Python/WSL GPU contexts.'),
        analysis=dict(primary_metric='Normalized development-V channel-macro FP64 MSE',secondary_metric='Normalized development-V channel-macro FP64 MAE',
                      contrast='E_FIXED minus matched OURS for each seed; positive means learning E helped.',
                      descriptive='All five points, paired differences, median/Q1/Q3, maximum-IQR descriptive rule inherited from parent. No significance or equivalence claim.',
                      report=['Actual selected epochs/caps', 'Frozen126 E parameters; expected35591 trainable versus Full35717', 'All attempts and wall time', 'Budget and optimization status'],
                      limits=['Jena selected prospectively from prior known P benefit; development V already viewed repeatedly.',
                              'n5 is a time-limited exploratory pilot, not an independent final test.',
                              'A null or worse result must remain visible and cannot be converted into a necessity claim.',
                              'Freezing changes the optimization space by126 parameters; no claim that E learning is the only possible effective parameterization.']))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', action='store_true', help='Root invokes only after code/test/protocol review.')
    args = parser.parse_args(argv)
    if not args.seal:
        print(json.dumps(dict(status='PREVIEW_ONLY_NOT_SEALED', base_jobs=5, extension_candidates=5,
                              training_deadline='2026-10-28T17:30:00+09:00', required_initial_seconds=14520,
                              latest_default_admission='2026-10-28T13:28:00+09:00',
                              source_files=list(sources()), next='Root review, then prepare --seal; root launch gate still separately required.')))
        return
    seal_path = runtime.REPORT / 'REGISTRATION_SEAL.json'
    if seal_path.exists():
        run_one.validate_registration(jobs()[0])
        print(json.dumps(dict(status='EXISTING_SEAL_VERIFIED')))
        return
    tests = runtime.read(runtime.REPORT / 'UNIT_TESTS.json')
    source_hashes = sources()
    if tests['status'] != 'PASS' or tests['source_sha256'] != source_hashes:
        raise ValueError('CPU tests are absent or refer to different sources')
    references, preserved = bind_references(), preservation()
    registration = protocol(source_hashes, references, runtime.sha(runtime.REPORT / 'UNIT_TESTS.json'))
    write(runtime.REPORT / 'REFERENCE_RUNS.json', references)
    write(runtime.REPORT / 'PRESERVATION_BASELINE.json', preserved)
    write(runtime.REPORT / 'PREREGISTRATION.json', registration)
    write(runtime.REPORT / 'RUN_PLAN.json', dict(campaign=runtime.NAME,jobs=jobs(),optional_jobs=[],extension_jobs=jobs(True),
          family_order=['E_FIXED:jena'],optional_family_order=[],training_deadline=registration['training_deadline'],
          delivery_deadline=registration['delivery_deadline']))
    write(runtime.REPORT / 'PREREGISTRATION.md', '''# E 학습 기여를 분리하는 Jena 5-seed 대조

이 연구는 기존 RUN_PLAN의 확장이 아닌 별도 사전등록이다. Jena K6, seed92601–92605에서 원 B2를 같은 초기화로 만든 뒤 E.requires_grad만 False로 바꾼다. P/frozen F/lag7 D/anchor는 그대로 두고 학습한다. 같은 기존 Full5를 재사용한다.

A_fit0–50%/A_stop50–60%, checkpoint0/.25/1/3 최소 A_stop 선택(정확동률이면 이른 epoch), 선택봉인, fresh A60 refit, 개발 V70–80% 평가를 유지한다. Full5는 cap3선택0/5다. E-fixed5중3개이상 cap3이면 전5 cap10 연장, cap10도3개이상이면 학습상한 미해결로 표시한다. 보호T는 열지 않는다.

search와 refit 각각 PCA를 허용 학습구간에서 새로 계산하고 E 초기/최종/선택값이 정확히 같아야 한다. 두 단계의 PCA가 서로 같은 값이어야 하는 것은 아니다. 실제 optimizer parameter groups에서 E 제외와 gradient 부재를 검증한다.

시작 전 기본5×18분+가능연장5×30분+2분=4시간2분을 남겨야 한다. 기본 추정으로13:28KST 이후에는 새 study를 시작하지 않는다. 학습17:30, 전달20:00 마감이다. 시간/A_stop만 입장·연장을 결정한다. 예산으로 시작하지 못하면 NOT_RUN_BUDGET이며 근거가 확보됐다고 말하지 않는다. 기술적 동일설정 재시도는1회, 과학 게이트 실패는 중단한다.

주캠페인 종료기록과 실제 프로세스 부재, root gate, 공유GPU락, 살아있는 parent lease/ancestry를 모두 확인한다. parent는 기존controller.lock, 실제 GPU child는 기존worker.lock과 자기worker.lock을 보유해 parent가 종료돼도 GPU 중복을 막는다. CUDA 로딩 전에 공통 host gate로 사전봉인된 정확한 desktop PID·생성시각·실행경로만 허용하며 새 프로세스·미확인·Python·WSL GPU context는 거부한다. 이 연구는 성능 benchmark가 아니므로 비용 측정용 60초 부하 상한은 적용하지 않고 desktop 동시 부하 정책과 매 attempt 검사 영수증을 보존한다.

비교는 같은 seed의 E_FIXED−OURS이며 양수이면 E 학습이 도움됐다는 뜻이다. 모든5점·paired차이·median/IQR·실제선택epoch/시간/파라미터를 보존한다. 새 결과는 반복 사용한 개발V의 탐색이며 n5로 동등성/독립일반화/모든설계의필연성을 주장하지 않는다. PCA나K의 최적성은 이 실험의 질문이 아니다. 불리하거나 불명확한 결과도 그대로 보고한다.
''')
    names = ('PREREGISTRATION.json','PREREGISTRATION.md','RUN_PLAN.json','PRESERVATION_BASELINE.json','REFERENCE_RUNS.json')
    write(seal_path, dict(created_utc=datetime.now(timezone.utc).isoformat(),method='Root-reviewed immutable prospective SHA256 seal before any study training',
                         files={name:runtime.sha(runtime.REPORT/name) for name in names}))
    print(json.dumps(dict(status='SEALED',base_jobs=5,extension_candidates=5,preserved_files=len(preserved['files']))))


if __name__ == '__main__':
    main()
