"""Seal the approved Jena branch study before any new training starts."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[2]
NAME = 'jena_p_branch_20261027_v1'
REPORT = REPO / 'reports' / NAME
CODE = REPO / 'research' / NAME
OLD = REPO / 'reports/comparative_ablation_followup_20261027_v1'
METHODS = ('P_TIME_ONLY', 'P_CHANNEL_ONLY')
SEEDS = list(range(92601, 92611))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf8') != text:
            raise FileExistsError(f'Refusing to replace sealed content: {path}')
        return
    with path.open('x', encoding='utf8', newline='\n') as f:
        f.write(text)


def main():
    if (REPORT / 'REGISTRATION_SEAL.json').exists():
        seal = json.loads((REPORT / 'REGISTRATION_SEAL.json').read_text(encoding='utf8'))
        for name, digest in seal['files'].items():
            if sha(REPORT / name) != digest:
                raise ValueError(f'Sealed registration changed: {name}')
        print(json.dumps({'status': 'EXISTING_SEAL_VERIFIED', 'created_utc': seal['created_utc']}))
        return
    sources = {p.relative_to(REPO).as_posix(): sha(p) for p in
               [CODE / n for n in ('models.py', 'runtime.py', 'run_one.py', 'controller.py', 'prepare.py')]}
    jobs = [dict(job_id=f'JPB_{m}_jena_K6_s{s}', method_id=m, dataset='jena', K=6,
                 seed=s, extended=False) for s in SEEDS for m in METHODS]
    extensions = [dict(job_id=f'JPB_{m}_jena_K6_s{s}_ext10', method_id=m, dataset='jena', K=6,
                       seed=s, extended=True) for m in METHODS for s in SEEDS]
    with (OLD / 'RUNS.csv').open(encoding='utf-8-sig', newline='') as f:
        references = [r for r in csv.DictReader(f) if r['dataset'] == 'jena' and
                      (r['method_id'] in ('OURS', 'A2') or
                       (r['method_id'] == 'A1' and r['extended'].lower() == 'true'))]
    for method in ('OURS', 'A1', 'A2'):
        rows = [r for r in references if r['method_id'] == method]
        if sorted(int(r['seed']) for r in rows) != SEEDS:
            raise ValueError(f'Reference seed coverage changed: {method}')
    for row in references:
        if int(float(row['K'])) != 6 or int(float(row['normalization_fit_end'])) != 12579:
            raise ValueError('Reference K/scaler range changed')
        if sha(row['prediction_path']) != row['prediction_sha256']:
            raise ValueError('Reference prediction hash mismatch')
    write(REPORT / 'REFERENCE_RUNS.json', dict(status='VERIFIED', rows=references,
          source=str(OLD / 'RUNS.csv'), source_sha256=sha(OLD / 'RUNS.csv')))
    preserved = set()
    for folder in ('comparative_ablation_20261025_v1', 'comparative_ablation_followup_20261027_v1'):
        preserved.update(p for p in (REPO / 'reports' / folder).rglob('*') if p.is_file())
    preserved.update(p for p in (REPO / 'research').rglob('*.py') if not p.is_relative_to(CODE))
    preserved.update(Path(r['prediction_path']) for r in references)
    baseline = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    scope='All existing Python research sources; both prior campaign report trees; reused prediction archives. Hashing only does not parse T.',
                    files=[dict(path=str(p.resolve()), sha256=sha(p)) for p in sorted(preserved)])
    write(REPORT / 'PRESERVATION_BASELINE.json', baseline)
    protocol = dict(
        campaign=NAME, status='PREREGISTERED_LOCAL_HASH_SEAL',
        authority='User explicitly approved Jena P branch 20 runs and presentation revision; no additional commit/push requested.',
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        registered_sources=sources,
        training_deadline='2026-10-28T16:00:00+09:00',
        delivery_deadline='2026-10-28T20:00:00+09:00',
        deadline='2026-10-28T16:00:00+09:00',
        dataset='jena', K=6, seeds=SEEDS, base_runs=20,
        methods={
            'P_TIME_ONLY': 'Retain the original temporal LayerNorm + temporal residual MLP; remove the feature branch.',
            'P_CHANNEL_ONLY': 'Retain the original feature LayerNorm + feature residual MLP; remove the temporal branch.'},
        normalization='Original joint LayerNorm over L x C is retained. These are branch ablations, not end-to-end axis independence.',
        initialization='Construct the original full model with the original seed; retain the corresponding original branch without new random initialization. Common E/D PCA initialization and data-order rules remain unchanged.',
        trainable_parameters=dict(P_TIME_ONLY=34320, P_CHANNEL_ONLY=2405, OURS=35717, A1=1008, A2=35708),
        backbone='Frozen Chronos-Bolt-small; original state SHA and checkpoint; FP32; L512 H48.',
        training=dict(optimizer='AdamW', lr=0.001, weight_decay=0, gradient_clip=1,
                      effective_batch=16, microbatch=4, F_row_microbatch=128,
                      checkpoint_inputs=False, revin=False, base_epochs=[0, .25, 1, 3],
                      extension_epochs=[0, .25, 1, 3, 6, 10],
                      selection='Minimum A_stop MSE; earlier epoch on exact tie; selection sealed before fresh A60 refit and V evaluation.'),
        data_bounds=dict(A_fit_end=10482, A60_end=12579, V_end=16772, T_start=16772,
                         T_numeric_values_parsed=0, loader='Unchanged safe Jena pre-T prefix loader'),
        extension_policy=dict(trigger='At least 5 of the 10 base seeds choose cap 3.',
                              action='Whole family 10 seeds freshly searched/refit to cap10 only if deadline budget permits.',
                              priority=list(METHODS),
                              incomplete='Retain base result as provisional; never mix partial extension seeds into the primary result.',
                              terminal='If at least 5/10 choose cap10 mark UNDERTRAINED_AT_10; no further extension.',
                              decision_data='A_stop selected epochs and elapsed time only; never V metric values.'),
        controller_policy=dict(concurrency=1, base_job_seconds=900,
                               extension_job_seconds=2700, admission_margin_seconds=120,
                               retries=1, retry='Same job/scientific settings for recoverable execution failure only; completed verified jobs reused.'),
        analysis=dict(primary_metric='V normalized MSE', secondary_metric='V normalized MAE',
                      primary_contrasts=['OURS vs P_TIME_ONLY: incremental feature-branch contribution in current architecture.',
                                         'OURS vs P_CHANNEL_ONLY: incremental temporal-branch contribution in current architecture.'],
                      references=['OURS 10', 'A1 extended 10', 'A2 10'],
                      summaries=['all seed points', 'median', 'Q1/Q3', 'paired per-seed differences', 'selected epochs', 'parameter counts'],
                      decision='Descriptive original median/max-IQR rule only; no statistical significance or equivalence claims.',
                      interpretive_limits=['Unequal branch capacities: direct time-only/channel-only contrast does not isolate axis superiority.',
                                           'A2 is additional context, not a complete capacity/FLOP control.',
                                           'Jena was chosen after observing the existing P effect; all new V results are exploratory.',
                                           'No generalization claim to other datasets or protected T.',
                                           'All base and extended results remain available; extension replaces base only as a complete 10-seed family.']),
        deliverables=['20 base RESULT records', 'branch comparison CSV and figure', 'Korean findings/limitations',
                      'revised problem-definition/method wording', 'updated slides 16 and 18',
                      'revised local PPTX/PDF presentation and notes', 'validation and bundle'])
    write(REPORT / 'PREREGISTRATION.json', protocol)
    write(REPORT / 'RUN_PLAN.json', dict(campaign=NAME, jobs=jobs, extension_jobs=extensions,
          training_deadline=protocol['training_deadline'], delivery_deadline=protocol['delivery_deadline']))
    markdown = '''# Jena P 분기 실험 사전 기록

사용자가 승인한 Jena 분기 두 종류 × seed 92601–92610 = 기본 20회와 발표 수정 작업이다.
학습 마감은 2026-10-28 16:00 KST, 검증된 발표 자료 전달 목표는 같은 날 20:00 KST다.
사용자는 추가 질문 없이 진행하도록 승인했다. 이번에는 commit/push를 요청하지 않아 로컬 해시 봉인으로 실행 전 설정을 기록한다.

## 질문과 구조

- P_TIME_ONLY: 원래 시간 LayerNorm+잔차 MLP만 유지.
- P_CHANNEL_ONLY: 원래 채널 LayerNorm+잔차 MLP만 유지.
- 원래 전체 모델을 동일 seed로 생성한 뒤 해당 분기를 그대로 보존한다. E, D, F, PCA 초기화, K=6, L=512, H=48은 유지한다.
- 공동 LayerNorm 및 E/D의 채널 결합이 남는다. 완전한 시간/채널 독립성 실험이 아니다.
- 시간 분기 33,312개, 채널 분기 1,397개 파라미터 차이를 공개한다. 시간축 자체의 우월성을 판정하지 않는다.

## 실행·선택

기본 checkpoint 0/.25/1/3, AdamW lr=.001, weight decay=0, gradient clip=1, effective batch16, microbatch4, FP32.
A_fit은 10,482행까지, A_stop 선택 후 A60 12,579행까지 새로 학습한다. V 끝과 T 시작은 16,772행이며 T 수치는 읽지 않는다.
seed별 두 방법을 순회한다. A_stop가 상한을 고른 seed가 5/10 이상이면 해당 방법 10개 전체를 0/.25/1/3/6/10으로 새 실행한다.
연장 순서는 P_TIME_ONLY, P_CHANNEL_ONLY이며 전체 10개를 마감 전에 수행할 예산이 있는 경우에만 허용한다.
부분 연장은 주 결과에 섞지 않으며, 상한 미해결은 잠정으로 남긴다. V 값을 보고 연장하거나 범위를 고르지 않는다.
같은 설정의 기술적 실패 재시도는 1회, 검증된 완료 학습은 반복하지 않는다. 실제 과학적 게이트 실패는 설정을 바꾸어 우회하지 않는다.

## 분석·공개

원 OURS10, 연장을 완료한 A1/No-P10, A2 10을 재사용한다. 원 A1 기본10을 더하여 n=20으로 만들지 않는다.
모든 seed 점, MSE/MAE 중앙값·IQR, seed별 차이, 파라미터 수, 선택 epoch를 보고한다.
원 중앙값/max-IQR 규칙은 기술적 비교이며 통계적 유의성이나 동등성 검정이 아니다.
Jena 선택이 기존 결과를 본 뒤 이루어졌음을 공개한다. 이 결과는 개발 V의 탐색 결과이며 다른 데이터나 보호 T로 일반화하지 않는다.
P 전체 대비 각 분기의 추가 기여를 질문한다. A2를 함께 보여주지만 용량·연산량 통제가 완결됐다고 주장하지 않는다.
기존 비교 어댑터의 잠정 결과를 유지한다. 결과에 맞춰 과거 결과를 삭제하거나 문구를 과장하지 않는다.

## 산출물

기본 20회와 허용된 연장의 실행 기록, 비교표·그림·한계, 문제정의·방법론 문구 수정, 16·18장 업데이트,
편집 가능한 수정 발표 자료/PDF/발표 대본, 검증 및 ZIP을 새 캠페인 내부에 보존한다.
'''
    write(REPORT / 'PREREGISTRATION.md', markdown)
    seal_files = ('PREREGISTRATION.json', 'PREREGISTRATION.md', 'RUN_PLAN.json',
                  'PRESERVATION_BASELINE.json', 'REFERENCE_RUNS.json')
    write(REPORT / 'REGISTRATION_SEAL.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
          method='Local immutable-before-training SHA256 seal; no Git commit claimed.',
          files={n: sha(REPORT / n) for n in seal_files}))
    print(json.dumps(dict(status='SEALED', base_jobs=len(jobs), extension_candidates=len(extensions),
                         preserved_files=len(baseline['files']), reference_rows=len(references))))


if __name__ == '__main__':
    main()
