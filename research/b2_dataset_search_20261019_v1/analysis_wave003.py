"""Validate the completed ETTm1 wave without changing sealed wave-2 analysis."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from research.b2_dataset_search_20261019_v1 import analysis as base


NAME = base.NAME
REPO = base.REPO


def _terminal_attempt(repo, report, cache, wave_sha, source_sha):
    path = cache / 'TERMINAL_003.json'
    recovery_paths = [report / f'EXECUTION_RECOVERY_WAVE_003_ATTEMPT_{attempt:03d}.json' for attempt in (2, 3)]
    if not any(candidate.exists() for candidate in recovery_paths):
        return path, None
    selected_attempt = 3 if recovery_paths[1].exists() else 2
    lineage = []
    for attempt in range(2, selected_attempt + 1):
        recovery_path = recovery_paths[attempt - 2]
        base._require(recovery_path.is_file(), 'Execution recovery preceding registration is missing')
        recovery = base.read(recovery_path)
        next_name = f'TERMINAL_003_ATTEMPT_{attempt:03d}.json'
        base._require(recovery.get('status') == 'REGISTERED_BEFORE_RESUMED_PRODUCER'
            and type(recovery.get('wave')) is int and recovery['wave'] == 3
            and type(recovery.get('attempt')) is int and recovery['attempt'] == attempt
            and recovery.get('previous_terminal') == dict(reference='cache://' + path.name, sha256=base.sha(path))
            and recovery.get('next_terminal_reference') == 'cache://' + next_name
            and recovery.get('unchanged_producer_sources') is True
            and recovery.get('wave_sha256') == wave_sha and recovery.get('producer_sha256') == source_sha,
            'Execution recovery registration path, prior terminal or producer binding differs')
        if attempt == 3:
            base._require(recovery.get('preceding_recovery') == dict(
                reference='report://' + recovery_paths[0].name, sha256=base.sha(recovery_paths[0])),
                'Execution recovery preceding registration binding differs')
            adapter = (repo / 'research' / NAME / 'resume_wave003.py').resolve()
            base._require(adapter.is_relative_to(repo) and adapter.is_file()
                and recovery.get('execution_adapter') == dict(
                    reference='repo://research/' + NAME + '/resume_wave003.py', sha256=base.sha(adapter))
                and recovery.get('identity_serialization_repair') == 'JSON_ROUNDTRIP_ONLY_EXACT_TRAIN_IDENTITY',
                'Execution recovery identity adapter binding differs')
        previous = base.read(path)
        base._require(previous.get('status') == 'FAILED' and type(previous.get('exit_code')) is int
            and previous['exit_code'] != 0 and type(previous.get('wave')) is int and previous['wave'] == 3,
            'Execution recovery does not preserve a failed wave-3 terminal')
        for key in ('started_unix', 'finished_unix', 'wall_seconds'):
            base._require(type(previous.get(key)) in (int, float) and math.isfinite(previous[key]),
                'Execution recovery prior terminal timing is missing or nonfinite')
        registered = recovery.get('registered_unix')
        base._require(previous['finished_unix'] >= previous['started_unix'] and previous['wall_seconds'] >= 0
            and type(registered) in (int, float) and math.isfinite(registered)
            and registered > previous['finished_unix'], 'Execution recovery registration timing differs')
        next_path = cache / next_name
        base._require(next_path.is_file(), 'Execution recovery retry terminal is missing')
        resumed = base.read(next_path)
        base._require(type(resumed.get('started_unix')) in (int, float) and math.isfinite(resumed['started_unix'])
            and registered <= resumed['started_unix'] and resumed['started_unix'] >= previous['finished_unix'],
            'Execution recovery was not registered before the resumed terminal attempt')
        lineage.append(dict(binding=dict(reference='repo://' + recovery_path.relative_to(repo).as_posix(), sha256=base.sha(recovery_path)),
            registration=recovery, previous_terminal=dict(previous, reference='cache://' + path.name, sha256=base.sha(path))))
        path = next_path
    return path, dict(lineage[-1], preceding_recoveries=lineage[:-1])


def terminal_evidence(repo, wave=3):
    if wave != 3:
        raise ValueError('This analysis wrapper is registered only for wave 3')
    repo = Path(repo).resolve()
    report, cache = repo / 'reports' / NAME, repo / '.cache' / NAME
    wave_path = report / 'waves' / 'WAVE_003.json'
    if not any((report / f'EXECUTION_RECOVERY_WAVE_003_ATTEMPT_{attempt:03d}.json').exists() for attempt in (2, 3)):
        base._require((cache / 'TERMINAL_003.json').is_file(),
            'Terminal success is required before reading actual scores')
        original = base.read(cache / 'TERMINAL_003.json')
        base._require(original.get('status') == 'COMPLETE' and type(original.get('exit_code')) is int
            and original['exit_code'] == 0 and type(original.get('wave')) is int and original['wave'] == 3,
            'Terminal driver did not complete wave 3 successfully')
    wave_sha, source_sha = base.sha(wave_path), base.sha(repo / 'research' / NAME / 'run.py')
    terminal_path, recovery = _terminal_attempt(repo, report, cache, wave_sha, source_sha)
    if not terminal_path.is_file():
        raise ValueError('Terminal success is required before reading actual scores')
    terminal = base.read(terminal_path)
    base._require(terminal.get('status') == 'COMPLETE' and type(terminal.get('exit_code')) is int
        and terminal['exit_code'] == 0 and type(terminal.get('wave')) is int and terminal['wave'] == 3,
        'Terminal driver did not complete wave 3 successfully')
    for key in ('started_unix', 'finished_unix', 'wall_seconds'):
        base._require(type(terminal.get(key)) in (int, float) and math.isfinite(terminal[key]),
            'Terminal timing evidence is missing or nonfinite')
    base._require(terminal['finished_unix'] >= terminal['started_unix'] and terminal['wall_seconds'] >= 0,
        'Terminal timing order differs')
    if recovery is not None:
        base._require(recovery['registration']['registered_unix'] <= terminal['started_unix']
            and terminal['started_unix'] >= recovery['previous_terminal']['finished_unix'],
            'Execution recovery was not registered before the resumed terminal attempt')
    journal_path = cache / 'PROCESS_ATTEMPTS.jsonl'
    if not journal_path.is_file():
        raise ValueError('Stage journal is required before reading actual scores')
    events = [json.loads(line) for line in journal_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    events = [entry for entry in events if entry.get('wave') == wave]
    selected = {}
    for stage in ('train', 'baselines', 'costs'):
        stage_events = [event for event in events if event.get('stage') == stage]
        base._require(bool(stage_events) and stage_events[-1].get('status') == 'COMPLETE'
            and any(event.get('status') == 'STARTED' for event in stage_events),
            'Stage journal does not prove terminal completion: ' + stage)
        for event in stage_events:
            base._require(event.get('wave_sha256') == wave_sha and event.get('producer_sha256') == source_sha
                and isinstance(event.get('observed_unix'), (int, float)) and math.isfinite(event['observed_unix']),
                'Stage journal wave/source identity differs: ' + stage)
        last = stage_events[-1]
        base._require(terminal['started_unix'] <= last['observed_unix'] <= terminal['finished_unix'],
            'Stage completion is outside the successful terminal driver: ' + stage)
        selected[stage] = last
    base._require(selected['train']['observed_unix'] <= selected['baselines']['observed_unix']
        <= selected['costs']['observed_unix'], 'Stage terminal completion order differs')
    return dict(terminal, source_reference=terminal_path.relative_to(repo).as_posix(),
        sha256=base.sha(terminal_path), stage_completions=selected,
        wave_journal_events_sha256=base.fingerprint(events),
        **({'execution_recovery': recovery} if recovery is not None else {}))


def _activation(repo):
    report = repo / 'reports' / NAME
    wave = base.read(report / 'waves' / 'WAVE_003.json')
    path = report / 'ACTIVATION_AMENDMENT_WAVE_003.json'
    amendment = base.read(path)
    plan_path = report / 'CANDIDATE_PLAN_003.json'
    plan = base.read(plan_path)
    base._require(amendment.get('status') == 'REGISTERED_BEFORE_NEW_FORECASTS'
        and amendment.get('activated_wave') == 3 and amendment.get('activated_datasets') == ['ETTm1']
        and amendment.get('new_authorization_overrides_prior_conditional_stop') is True
        and amendment.get('preserve_prior_activation_rule') is True
        and amendment.get('unchanged_recipe') is True and amendment.get('unchanged_K_grid') is True
        and amendment.get('V_scores_used_to_choose_wave') is False and amendment.get('protected_T_access') is False
        and wave.get('activation_amendment_sha256') == base.sha(path)
        and wave.get('candidate_plan_sha256') == base.sha(plan_path)
        and wave.get('V_forecast_scores_used') is False and wave.get('T_consumed_by_this_wave_training') is False
        and [(entry['dataset'], entry['K_grid']) for entry in wave.get('datasets', [])] == [('ETTm1', [1, 2, 4])],
        'Wave-3 activation amendment or complete registered ETTm1 grid differs')
    base._require(amendment.get('prior_plan_binding') == dict(
        reference='repo://reports/' + NAME + '/CANDIDATE_PLAN_003.json', sha256=base.sha(plan_path))
        and plan.get('numeric_forecast_scores_used_to_choose_candidates') is False
        and plan.get('unchanged_recipe') is True and plan.get('unchanged_K_rule') is True
        and amendment['registered_at_unix'] <= wave['registered_at_unix'],
        'Wave-3 activation does not bind the preserved pre-score candidate plan')
    return amendment, plan, base.sha(path), base.sha(plan_path)


def _pending_registry(repo, plan, positive):
    rows = []
    for entry in plan['ordered_candidates']:
        if entry['dataset'] == 'ETTm1':
            continue
        base._require(entry['dataset'] == 'solar_full137', 'Unregistered pending wave-3 candidate')
        path = repo / '.cache' / NAME / 'data' / entry['dataset'] / 'DATA_DIAGNOSTIC.json'
        metadata = base.read(path)
        base._require(metadata.get('status') == 'READY' and all(metadata.get(key) == entry[key]
            for key in ('dataset', 'C', 'columns', 'prefix_rows', 'interval_seconds')),
            'Pending Solar preparation metadata differs from its registered full-channel plan')
        rows.append(dict(wave='NEXT_UNREGISTERED', originally_planned_wave=3, dataset=entry['dataset'],
            status='READY_NOT_EXECUTED' if positive else 'READY_NEXT_WAVE_AFTER_NO_POSITIVE',
            K_grid=','.join(map(str, entry['K_grid'])), C=entry['C'],
            conditional=True, measured_in_this_report=False, forecast_results_counted=0,
            preparation_status='READY', config_sha256=base.sha(path), prefix_sha256=metadata['prefix_sha256'],
            source_reference='reports/' + NAME + '/CANDIDATE_PLAN_003.json', plan_sha256=base.sha(repo / 'reports' / NAME / 'CANDIDATE_PLAN_003.json'),
            detail='Unexecuted full137-channel Solar; metadata only, numeric prefix not opened or hashed. '
                + ('Additional example found in wave3; retain prepared candidate.' if positive else
                   'No wave3 positive; retained next candidate under the latest continuing-search authorization.')))
    return rows


def collect(repo=REPO, wave=3):
    repo = Path(repo).resolve()
    terminal = terminal_evidence(repo, wave)
    amendment, plan, activation_sha, plan_sha = _activation(repo)
    rows, proof = base.collect(repo, wave)
    base._require(set(proof['datasets']) == {'ETTm1'} and set(proof['judgments']) == {'ETTm1'}
        and proof['datasets']['ETTm1']['C'] == 7
        and proof['datasets']['ETTm1']['interval_seconds'] == 900
        and proof['datasets']['ETTm1']['original_rows'] == 69680
        and proof['datasets']['ETTm1']['prefix_rows'] == 55744,
        'Wave-3 ETTm1 native frequency, original cohort or full prefix differs')
    pending = _pending_registry(repo, plan, proof['judgments']['ETTm1']['positive'])
    candidates = [dict(row) for row in proof['candidates']]
    for row in candidates:
        if row['wave'] == 2 and row['dataset'] in amendment.get('prior_positive_datasets', []):
            row.update(status='PRESERVED_PRIOR_POSITIVE_EXPLORATORY',
                detail='Prior wave2 positive preserved by activation amendment; not rerun or counted in wave3.')
    candidates.extend(pending)
    proof.update(terminal_evidence=terminal, activation_amendment=amendment,
        activation_amendment_sha256=activation_sha, candidate_plan_sha256=plan_sha,
        candidates=candidates, pending_candidates=pending, conditional_candidates=pending,
        conditional_candidate_prefixes_opened_or_hashed=False,
        analysis_numeric_scope='Only activated ETTm1 prefix and saved V predictions; no protected T or whole raw numeric hash',
        prior_ETTh2_amendment_scope='Inherited historical provenance only; unrelated to fresh ETTm1 prefix preparation')
    return rows, proof


def _table(summary, policy):
    lines = ['```text', 'Method       seeds  MSE mean [min,max]           MAE mean [min,max]           B8 ms mean [min,max]         allocated MiB [min,max]      reserved MiB [min,max]']
    for row in summary:
        if row['row_policy'] != policy:
            continue
        name = 'Raw' if row['family'] == 'RAW' else f'B2 K={row["K"]}' if row['family'] == 'B2' else 'Direct MLP'
        values = '  '.join(f'{row[key+"_mean"]:.6f} [{row[key+"_min"]:.6f},{row[key+"_max"]:.6f}]' for key in base.METRICS)
        lines.append(f'{name:<12} {row["seeds"]:<5}  {values}')
    return lines + ['```', '']


def _texts(rows, proof, synthetic=False):
    judged = proof['judgments']['ETTm1']
    verdict = 'POSITIVE_EXPLORATORY' if judged['positive'] else 'NEGATIVE_EXPLORATORY'
    summary = base.aggregate(rows)
    prefix = 'SYNTHETIC — ' if synthetic else ''
    ko = [f'# {prefix}ETTm1 데이터·K 탐색 wave 3', '',
        f'판정: {verdict}. ETTm1 C7의 K1/2/4와 두 seed92601/92602를 모두 실제 실행한 wave다.', '',
        '등록 조건은 DEFAULT128에서 두 seed의 MSE가 각각 Raw보다 낮고 평균 MSE가 1% 이상 낮으며, '
        '두 seed의 whole B8 중앙 지연과 allocated 피크도 각각 Raw보다 낮아야 한다. '
        'MAE 개선이나 Direct MLP 우위까지 뜻하지 않으며 이 두 축도 함께 공개한다.', '',
        '원래 N69680 중 허용 prefix55744행과 7개 원채널을 유지했다. native 15분 간격이므로 H48=12시간, '
        'L512=128시간이다. ETTh1은 같은 transformer station의 hourly 변형으로, 서로 독립적인 새 도메인으로 세지 않는다. '
        'ETTh1 H48=48시간·L512=512시간, 과거 Jena H48=8시간과 실제 예측 시간은 다르다. '
        '강제 리샘플링이나 서로 다른 데이터 간 MSE 숫자의 직접 비교는 하지 않는다.', '']
    en = [f'# {prefix}ETTm1 registered K trade-offs — wave 3', '',
        f'Outcome: {verdict}. ETTm1 was executed at all registered K=1, K=2 and K=4 with seeds92601/92602.', '',
        'Positive requires both seeds below Raw MSE, their mean at least1% below Raw, and both seeds below Raw '
        'whole B8 median latency and allocated peak under DEFAULT128. MAE and the practical Direct MLP comparison remain separate disclosed axes.', '',
        'Full original C7, N69680, consumed prefix55744, native15min; H48=12h and L512=128h. '
        'ETTm1 and ETTh1 share the same transformer station; they are related frequency variants, not independent domains. '
        'ETTh1 H48=48h; historical Jena H48=8h. Compare methods within this dataset; no frequency resampling.', '']
    for policy in base.POLICIES:
        ko.extend([f'## {policy}', ''] + _table(summary, policy))
        en.extend([f'## {policy}', ''] + _table(summary, policy))
    direct = next(row for row in summary if row['row_policy'] == 'DEFAULT128' and row['family'] == 'DIRECT_G')
    for k, result in judged['widths'].items():
        arm = next(row for row in summary if row['row_policy'] == 'DEFAULT128' and row['family'] == 'B2' and str(row['K']) == k)
        seeds = sorted((row for row in rows if row['row_policy'] == 'DEFAULT128' and row['family'] == 'B2' and str(row['K']) == k), key=lambda row: row['seed'])
        raw = next(row for row in rows if row['row_policy'] == 'DEFAULT128' and row['family'] == 'RAW')
        mse_vs_direct = 100 * (arm['MSE_mean'] / direct['MSE_mean'] - 1)
        direct_latency_ratio = arm['p50_ms_mean'] / direct['p50_ms_mean']
        direct_memory_ratio = arm['peak_allocated_MiB_mean'] / direct['peak_allocated_MiB_mean']
        seed_text = ', '.join(f'{row["seed"]}: {row["MSE"]:.6f} (Raw보다 낮음={row["MSE"] < raw["MSE"]})' for row in seeds)
        seed_en = ', '.join(f'{row["seed"]}: {row["MSE"]:.6f} (below Raw={row["MSE"] < raw["MSE"]})' for row in seeds)
        ko.extend([f'K={k}: positive={result["positive"]}; 두 seed MSE {seed_text}. Raw MSE={raw["MSE"]:.6f}. '
            f'평균 MSE 감소 {result["MSE_gain_pct_vs_raw"]:+.2f}%, MAE 변화 {result["mae_change_pct_vs_raw"]:+.2f}%, '
            f'지연 속도비 {result["speedup_vs_raw"]:.3f}배, allocated 변화 {result["allocated_change_pct_vs_raw"]:+.2f}%. '
            f'조건별 판정={json.dumps(result["predicates"], sort_keys=True)}.', '',
            f'Direct MLP 대비 K={k} 평균 MSE 변화 {mse_vs_direct:+.2f}%, 지연 비 {direct_latency_ratio:.3f}배, '
            f'allocated 비 {direct_memory_ratio:.3f}배다. Direct MLP의 MSE·지연·allocated 동시 지배는 '
            f'평균={result["direct_dominates_mean"]}, 짝지은 seed={json.dumps(result["direct_dominates_paired_seeds"], sort_keys=True)}다.', ''])
        en.extend([f'- K={k}: positive={result["positive"]}; seed MSE {seed_en}; Raw={raw["MSE"]:.6f}. '
            f'Mean MSE gain vs Raw {result["MSE_gain_pct_vs_raw"]:+.2f}%; MAE change {result["mae_change_pct_vs_raw"]:+.2f}%; '
            f'default speed ratio {result["speedup_vs_raw"]:.3f}x; allocated change {result["allocated_change_pct_vs_raw"]:+.2f}%. '
            f'Predicates: {json.dumps(result["predicates"], sort_keys=True)}.', '',
            f'  Direct MLP comparison: B2 MSE change {mse_vs_direct:+.2f}%; B2/Direct latency {direct_latency_ratio:.3f}x; '
            f'allocated {direct_memory_ratio:.3f}x. Direct dominates all three mean axes={result["direct_dominates_mean"]}; '
            f'paired seed dominance={json.dumps(result["direct_dominates_paired_seeds"], sort_keys=True)}.', ''])
    first = rows[0]
    ko.extend(['## 학습·검산과 해석 범위', '',
        f'전체 V origin {first["n_origins"]}개, 관측 목표가 있는 origin {first["scored_origins"]}개, '
        f'관측 scalar target {first["observed_targets"]}개다. 목표를 보간하거나 결과에 따라 원점을 제외하지 않았다.', '',
        'A_fit 구조 진단: ' + json.dumps(proof['datasets']['ETTm1'].get('train_structure') or {}, ensure_ascii=False, sort_keys=True), '',
        '```text', 'run_id  selected_epoch  search_train_origins  training_status'])
    en.extend(['## Training and scope', '',
        f'Full V: {first["n_origins"]} origins, {first["scored_origins"]} with observed targets, {first["observed_targets"]} observed scalar targets. '
        'No target imputation or outcome-dependent origin removal.', '',
        '```text', 'run_id  selected_epoch  search_train_origins  training_status'])
    for job, item in sorted(proof['training'].items()):
        line = f'{job}  {item["selected_epoch"]}  {item["search_train_origins"]}  {item["training_status"]}'
        ko.append(line); en.append(line)
    ko.extend(['```', '',
        'TRAIN은 stride1 전체 origin이다. A_stop의 strict minimum·earlier tie로 고른 전체 B2 선택을 봉인한 뒤 '
        'fresh A60 refit/V를 수행했고, Direct MLP는 자체 고정 A_stop 선택 후 fresh refit했다. '
        'cap3에서 계속 개선하는 arm을 수렴했다고 주장하지 않는다. RevIN OFF, FP32, L512/H48 및 고정 recipe는 유지했다.', '',
        f'B2 검색 {proof["B2_searches"]}개·fresh refit {proof["B2_fresh_refits"]}개, Direct MLP search/refit '
        f'{proof["Direct_MLP_search_refits"]}쌍, 비용 {proof["route_facets"]}facet, warmup {proof["warmup_trials"]}회·timed '
        f'{proof["timed_trials"]}회를 검산했다. driver exit0과 train/baselines/costs 완료 journal을 확인한 뒤에만 결과를 읽었다.', '',
        '각 facet은 별도 (PID, 생성 시각)의 fresh process다. 첫 8개 고정 V origin의 physical+mask→causal fill→고정 scaler→실제 모델→inverse→CPU 반환을 측정했다. '
        'slicing/load/hash/검산은 시간에서 제외한다. 전체 V 예측과 두 row 정책의 parity, FP64 관측 equal-channel macro MSE/MAE, '
        'A60 scaler·physical history/target mask, raw trial quantile·allocated/reserved 최대, source/checkpoint·선택 봉인을 독립 검산했다. '
        'physical 출력 byte 배열이 저장되지 않아 그 hash 재계산은 주장하지 않는다.', '',
        '두 seed min–max는 신뢰구간이 아니며 timed20회 p95는 서비스 tail 보장이 아니다. CUDA allocated/reserved 피크는 프로세스 hard cap이 아니다. '
        'COMMON_CAP=16과 DEFAULT128은 같은 선택 모델·V를 사용하지만 별도 비용 정책이다. MAE와 reserved 메모리도 표에 남겼다.', '',
        '순차적 development V 후보 탐색이며 독립 TEST나 통계적 유의성 결론이 아니다. 이전 데이터 사용·관련 주파수 원천 공유와 '
        '기반모델 사전학습 중복 가능성은 배제하지 못한다. K·seed·채널·recipe는 새 V 결과에 따라 바꾸지 않았다. LoRA/Ridge는 새 wave에서 실행하지 않았다.', '',
        'JENA_HISTORICAL_CONTEXT.csv는 과거 memory_budget_20261018_v1 B8 confirm의 K3/6/12·Raw·merged LoRA 별도 실행을 역사적 문맥으로 보존한다. '
        '새 ETTm1 판정에는 쓰지 않으며 기존 ancestry/TEST 노출 한계가 유지된다. 과거 실제 CUBLAS 값은 미기록이고 새 비용은 부모 상속값을 receipt에 기록했다.', '',
        '## 활성화 정정과 역사적 준비 provenance', '',
        'ACTIVATION_AMENDMENT_WAVE_003은 사용자의 추가 데이터 사례 요청을 새 점수 전에 등록해 ETTm1을 활성화했다. '
        '이 최신 승인은 wave2 positive가 있으면 중단한다는 이전 조건을 이번 wave에 한해 대체한다. wave2 Appliances·ETTh1 positive와 원래 계획은 보존한다.', '',
        'PREPARATION_AMENDMENT_001의 최초 ETTh2 prefix13936 중 16시간과 같은 물리 원천 ETTm2의 더 이른 보호 경계 겹침 가능성은 역사적 provenance다. '
        '당시 수치 materialize 사실과 superseded WAVE001·원본을 보존했고 그 16시간의 새 학습/forecast 사용은 0이었다. '
        '교체 ETTh2_safe17400의 prefix13920은 원본에서 다시 준비한 custom cohort다. '
        '이 ETTh2/ETTm2 준비 사건은 fresh ETTm1 prefix와 무관하며, 이번 wave에 상속된 후보 이력으로만 공개한다. '
        'ETTm1은 N69680/prefix55744와 같은 station의 ETTh1 cutoff를 따로 검증한 native 15분 자료다.', '',
        '새 보호 T 수치나 전체 raw 파일의 numeric hash는 읽거나 재계산하지 않았다. 소비 prefix의 SHA/array 지문과 '
        '과거 metadata의 full-source SHA를 구분한다. 분석은 미실행 Solar numeric NPZ도 열거나 hash하지 않는다.', '',
        '```text', 'wave dataset status detail'])
    en.extend(['```', '',
        f'{proof["B2_searches"]} B2 searches, {proof["B2_fresh_refits"]} fresh A60 refits, {proof["Direct_MLP_search_refits"]} '
        f'Direct MLP search/refit pairs; {proof["route_facets"]} cost facets, {proof["warmup_trials"]} warmups and {proof["timed_trials"]} timed trials. '
        'Actual score reading follows driver exit0 and completed train/baselines/costs journal entries.', '',
        'Full stride1 TRAIN, strict minimum A_stop selection with earliest ties, all B2 selections sealed before fresh A60 refit/V; '
        'Direct MLP selects separately under the fixed recipe. Cap3 does not prove convergence. FP32, RevIN OFF, L512/H48.', '',
        'Each cost facet has a distinct (PID, process creation time). Whole B8 physical+mask -> causal fill -> fixed scaler -> '
        'online model -> inverse -> CPU output excludes slicing/load/hash/verification. Full V forecasts, row-policy parity, '
        'FP64 observed equal-channel macro MSE/MAE, A60 scaler/history/target-mask, raw trial timing quantiles, tensor memory peaks '
        'and source/model/checkpoint/selection bindings are independently checked. Physical output bytes were not saved; their hash is not recomputed.', '',
        'Figure rows: DEFAULT128 and COMMON_CAP=16; columns: normalized MSE, whole B8 latency, allocated peak. '
        'Bars show means and whiskers seed min-max, not confidence intervals. Raw is one deterministic model. '
        'MAE and reserved peaks remain in tables. Timed20 p95 is not a service-tail guarantee; tensor peaks are not process hard caps.', '',
        'Adaptive development V and two seeds do not establish independent TEST performance or statistical significance. '
        'Prior data exposure, shared station ancestry and possible foundation pretraining overlap remain limitations. '
        'No new LoRA/Ridge comparison. Historical Jena B8 confirm K3/6/12, Raw and merged LoRA remain a separate prior execution '
        'with existing ancestry/TEST-exposure limits; they do not enter ETTm1 classification. Prior CUBLAS values were not recorded; new receipts record inherited settings.', '',
        '## Activation and preserved provenance', '',
        'ACTIVATION_AMENDMENT_WAVE_003 registers the latest explicit request before new scores and activates ETTm1 despite '
        'the old conditional stop after wave2 positives. Prior Appliances and ETTh1 positives and the original plan are preserved.', '',
        'Historical preparation amendment: initial ETTh2 prefix13936 materialized16 potentially overlapping protected hours '
        'relative to same-source ETTm2; zero new training/forecast use. Superseded WAVE001 and source preparation are preserved. '
        'Replacement ETTh2_safe17400 uses a custom cohort with freshly streamed prefix13920. This historical ETTh2/ETTm2 incident '
        'is unrelated to ETTm1 fresh prefix preparation beyond inherited cohort history. ETTm1 uses its separately checked '
        'N69680/prefix55744/native15min and same-station ETTh1 cutoff.', '',
        'No new protected T numeric access or whole-raw numeric hashing. Consumed prefix/array hashes are separate from '
        'historical full-source metadata hashes. Unexecuted Solar numeric archives are not opened or hashed.', '',
        '```text', 'wave dataset status detail'])
    candidates = proof['candidates'] + [row for row in proof.get('pending_candidates', [])
        if row['dataset'] not in {item['dataset'] for item in proof['candidates']}]
    for item in candidates:
        line = f'{item["wave"]} {item["dataset"]} {item["status"]} {item["detail"]}'
        ko.append(line); en.append(line)
    solar_ko = ('Solar full137은 준비된 READY_NOT_EXECUTED로 보존한다.' if judged['positive'] else
        '이번 ETTm1 wave에 positive가 없으므로 Solar full137을 다음 후보로 유지한다.')
    solar_en = ('Solar full137 remains READY_NOT_EXECUTED after the additional example.' if judged['positive'] else
        'With no wave3 positive, Solar full137 remains the next prepared candidate.')
    ko.extend(['```', '', solar_ko + ' Solar는 이번 wave 성능 실험 수에 포함하지 않는다. 별도 RawContext 캠페인의 실행·예산과도 분리한다.', '',
        '전체 seed는 RUNS.csv, 평균·범위는 SUMMARY.csv, 후보 이력은 CANDIDATES.csv, 두 비용 정책과 그림은 ETTm1별 파일을 따른다.'])
    en.extend(['```', '', solar_en + ' Solar has zero performance experiments counted in this report. '
        'The separately authorized RawContext campaign has a separate execution and budget.', '',
        'RUNS.csv discloses every seed; SUMMARY.csv gives means/ranges; CANDIDATES.csv preserves all candidate outcomes.'])
    return '\n'.join(ko).encode('utf-8'), '\n'.join(en).encode('utf-8')


def render_outputs(rows, proof, synthetic=False):
    """Build deterministic bytes from already validated rows (or explicit CPU fixtures)."""
    ko, en = _texts(rows, proof, synthetic)
    summary = base.aggregate(rows)
    candidates = proof['candidates'] + [row for row in proof.get('pending_candidates', [])
        if row['dataset'] not in {item['dataset'] for item in proof['candidates']}]
    outputs = {'RUNS.csv': base._csv(rows), 'SUMMARY.csv': base._csv(summary),
        'CANDIDATES.csv': base._csv(candidates), 'JENA_HISTORICAL_CONTEXT.csv': base._csv(proof['historical']),
        'RESULTS_KO.md': ko, 'SLIDE_CONTENT_EN.md': en}
    for policy in base.POLICIES:
        outputs['ETTm1_' + policy + '.csv'] = base._csv([row for row in summary if row['row_policy'] == policy])
    for extension, data in base.render_figures(rows, 'ETTm1', proof['datasets']['ETTm1'], synthetic=synthetic).items():
        outputs['ETTm1_COMPARISON.' + extension] = data
    import matplotlib
    validation = dict(proof, analysis_source_sha256=base.sha(__file__),
        reused_analysis_source_sha256=base.sha(base.__file__), numpy_version=np.__version__,
        matplotlib_version=matplotlib.__version__, synthetic_fixture=synthetic,
        outputs={name: hashlib.sha256(value).hexdigest() for name, value in outputs.items()})
    outputs['VALIDATION.json'] = (json.dumps(validation, sort_keys=True, ensure_ascii=False, indent=2,
        allow_nan=False) + '\n').encode('utf-8')
    return outputs, validation


def write_outputs(outputs, destination, repo):
    repo, destination = Path(repo).resolve(), Path(destination).resolve()
    base._require(any(destination.is_relative_to(root) for root in (repo / 'reports' / NAME, repo / '.cache' / NAME)),
        'Output escaped dataset-search namespace')
    for name, value in outputs.items():
        base._require(Path(name).name == name, 'Output filename escaped namespace')
        path = destination / name
        if path.exists() and path.read_bytes() != value:
            raise FileExistsError('Preserve existing analysis output before rendering: ' + name)
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        path = destination / name
        if not path.exists():
            with path.open('xb') as stream:
                stream.write(value)


def generate(repo=REPO, wave=3, output_dir=None):
    repo = Path(repo).resolve()
    rows, proof = collect(repo, wave)
    outputs, validation = render_outputs(rows, proof)
    destination = Path(output_dir) if output_dir else repo / 'reports' / NAME / 'waves' / 'WAVE_003_results'
    write_outputs(outputs, destination, repo)
    return validation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--wave', type=int, choices=(3,), default=3)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    if args.check_only:
        rows, proof = collect(args.repo, args.wave)
        result = dict(status=proof['status'], wave=3, route_facets=len(rows), judgments=proof['judgments'],
            terminal_sha256=proof['terminal_evidence']['sha256'])
    else:
        proof = generate(args.repo, args.wave, args.output_dir)
        result = dict(status=proof['status'], wave=3, outputs=proof['outputs'])
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
