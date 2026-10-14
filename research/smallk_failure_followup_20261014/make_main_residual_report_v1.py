"""Guarded terminal-JSON-only MAIN residual report; no arrays or models opened.

Source preparation is not an execution claim. The fresh CPU resource factory is
constructed before any inputs, output directory, source copies, or plotting.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASETS = ('robin', 'beijing_aotizhongxin')
SEEDS = (92601, 92602)
FINAL = {'robin': 'val', 'beijing_aotizhongxin': 'eval'}
ROUTES = ('RAW33C', 'E33K', 'E512K')
METHODS = ('RAW_BOLT', 'ORIGINAL_MAIN', 'MEAN_CORRECTION') + ROUTES
IDS = ('ZERO_CORRECTION', 'MEAN_CORRECTION') + tuple('RIDGE_ALPHA_'+str(i) for i in range(7))
ALPHAS = (None, None, 0., 1e-4, .001, .01, .1, 1., 10.)
LABELS = {'RAW_BOLT':'RAW-F', 'ORIGINAL_MAIN':'MAIN', 'MEAN_CORRECTION':'Mean',
          'RAW33C':'RAW33C', 'E33K':'E33K', 'E512K':'E512K'}
COLORS = ('#65758b', '#a87924', '#8a6198', '#278f8e', '#3374b6', '#bd6154')
FIGURES = ('01_final_seed_scores', '02_mean_and_MAIN_deltas', '03_candidate_tune_grid',
           '04_fit_tune_final', '05_channel_delta_vs_mean', '06_whole_path_cost')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''): h.update(block)
    return h.hexdigest()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def record(path, role):
    path = resolve(path)
    return {'role':role, 'path':str(path), 'sha256':sha(path)}


def score(p, method, split):
    if method in ('RAW_BOLT', 'ORIGINAL_MAIN'): return p['references'][split][method]
    if method == 'MEAN_CORRECTION': return p['controls'][method]['splits'][split]
    return p['families'][method]['splits'][split]


def validate(producers, records, audit, cost, cost_record, cost_audit):
    expected = {(d,s) for d in DATASETS for s in SEEDS}
    if len(producers)!=4 or {(p['dataset'],p['seed']) for p in producers}!=expected:
        raise ValueError('EXACT_FOUR_UNIQUE_CONDITIONS_REQUIRED')
    if audit.get('status')!='PASS' or audit.get('failed_count')!=0 or len(audit.get('conditions',[]))!=4:
        raise ValueError('INDEPENDENT_SCIENTIFIC_AUDIT_PASS_REQUIRED')
    if (audit.get('candidate_rows_checked')!=108 or audit.get('new_ridge_candidates_checked')!=84
            or audit.get('selected_predictions_checked')!=36):
        raise ValueError('FULL_108_POOL_AND_84_RIDGE_AUDIT_REQUIRED')
    if audit.get('source_before_after_SHA_pass') is not True:
        raise ValueError('SCIENTIFIC_AUDIT_EXECUTION_SOURCE_UNCHANGED_REQUIRED')
    for p,r in zip(producers, records):
        if (p.get('status')!='COMPLETE' or p.get('GPU_execution') is not False
                or p.get('future_oracle_read') is not False or p.get('protected_TEST_accessed') is not False
                or p.get('independent_holdout') is not False):
            raise ValueError('COMPLETE_PAST_ONLY_CPU_PRODUCER_REQUIRED')
        binding = [c for c in audit['conditions'] if (c.get('dataset'),c.get('seed'))==(p['dataset'],p['seed'])]
        if len(binding)!=1 or binding[0].get('status')!='PASS' or binding[0]['producer_sha256']!=r['sha256']:
            raise ValueError('EXACT_SCIENTIFIC_PRODUCER_SHA_BINDING_REQUIRED')
        for route in ROUTES:
            f = p['families'][route]; grid = f['candidate_grid']
            if f['status']!='COMPLETE' or tuple(v['candidate_id'] for v in grid)!=IDS or tuple(v['alpha'] for v in grid)!=ALPHAS:
                raise ValueError('COMPLETE_FIXED_NINE_CANDIDATES_REQUIRED')
            if any(v['status']!='COMPLETE' or not math.isfinite(v['tune_actual_Y_mse']) for v in grid):
                raise ValueError('FINITE_COMPLETE_ACTUAL_Y_TUNE_POOL_REQUIRED')
            ix = min(range(9), key=lambda j:grid[j]['tune_actual_Y_mse'])
            if f['selected_index']!=ix or f['selected_candidate_id']!=IDS[ix] or f['selected_alpha']!=ALPHAS[ix]:
                raise ValueError('EARLIEST_TUNE_SELECTION_MISMATCH')
    if cost.get('status')!='COMPLETE' or len(cost.get('rows',[]))!=24 or any(r['status']!='COMPLETE' for r in cost['rows']):
        raise ValueError('COMPLETE_24_WHOLE_COST_ROWS_REQUIRED')
    expected_cost = {(d,92601,b,m) for d in DATASETS for b in (1,8)
                     for m in ('RAW_BOLT','ORIGINAL_MAIN','MEAN_CONTROL')+ROUTES}
    if {(r['dataset'],r['seed'],r['batch'],r['route']) for r in cost['rows']}!=expected_cost:
        raise ValueError('EXACT_BALANCED_COST_COHORT_REQUIRED')
    if cost_audit.get('status')!='PASS' or cost_audit.get('failed_count')!=0:
        raise ValueError('INDEPENDENT_COST_AUDIT_PASS_REQUIRED')
    if cost_audit.get('source_before_after_SHA_pass') is not True or cost_audit.get('whole_rows_checked')!=24:
        raise ValueError('COST_AUDIT_SOURCE_UNCHANGED_AND_24_ROWS_REQUIRED')
    # The cost audit must explicitly bind the measured RESULT, not just its cohort.
    binding = cost_audit.get('producer') or cost_audit.get('cost_producer') or cost_audit.get('cost_result')
    bound_sha = (binding or {}).get('sha256') or cost_audit.get('producer_sha256') or cost_audit.get('cost_producer_sha256')
    if bound_sha!=cost_record['sha256']:
        raise ValueError('EXACT_COST_RESULT_SHA_BINDING_REQUIRED')
    for p,r in zip(producers,records):
        if p['seed']==92601 and not any(v.get('sha256')==r['sha256'] for v in cost.get('producers',[])):
            raise ValueError('COST_SCIENTIFIC_PRODUCER_SHA_BINDING_REQUIRED')
    return {'passed':True,'scientific_producer_SHA_bound':True,'cost_RESULT_SHA_bound':True,
            'conditions':4,'candidate_rows':108,'new_ridge_candidates':84,'whole_cost_rows':24,
            'audit_source':audit.get('audit_source'),'audit_source_after':audit.get('audit_source_after')}


def metrics_rows(producers):
    rows=[]
    for p in producers:
        for method in METHODS:
            for split in ('fit','tune',FINAL[p['dataset']]):
                for scale in ('normalized','physical_per_channel_diagnostic'):
                    s=score(p,method,split)[scale]
                    rows.append({'dataset':p['dataset'],'seed':p['seed'],'method':method,'split':split,
                        'split_role':{'fit':'training','tune':'selection','val':'reused_development','eval':'exposed_evaluation'}[split],
                        'metric_scale':scale,'mse':s['mse'],'mae':s['mae'],'active_channels':s['active_channels'],
                        'observed_counts':s['observed_counts'],'per_channel_mse':s['per_channel_mse'],
                        'per_channel_mae':s['per_channel_mae'],'physical_macro_comparable_units_guaranteed':False})
    return rows


def candidate_rows(producers):
    rows=[]
    for p in producers:
        for route in ROUTES:
            f=p['families'][route]
            for i,e in enumerate(f['candidate_grid']):
                rows.append({'dataset':p['dataset'],'seed':p['seed'],'method':route,'candidate_index':i,
                    'candidate_id':e['candidate_id'],'alpha':e['alpha'],'status':e['status'],'selected':i==f['selected_index'],
                    'new_ridge':i>=2,'model_sha256':e['model_sha256'],
                    'fit_normalized_mse':e['fit_scores']['normalized']['mse'],
                    'fit_normalized_mae':e['fit_scores']['normalized']['mae'],
                    'tune_normalized_mse':e['tune_scores']['normalized']['mse'],
                    'tune_normalized_mae':e['tune_scores']['normalized']['mae'],
                    'fit_physical_diagnostic_mse':e['fit_scores']['physical_per_channel_diagnostic']['mse'],
                    'tune_physical_diagnostic_mse':e['tune_scores']['physical_per_channel_diagnostic']['mse']})
    return rows


def capacity_rows(producers):
    rows=[]
    for p in producers:
        for route in ROUTES:
            f=p['families'][route]; m=f['selected_model_metadata']; g=f['candidate_grid']
            current=score(p,'ORIGINAL_MAIN','tune')['normalized']['mse']
            mean=score(p,'MEAN_CORRECTION','tune')['normalized']['mse']; chosen=g[f['selected_index']]['tune_actual_Y_mse']
            final=score(p,route,FINAL[p['dataset']])['normalized']
            base=score(p,'ORIGINAL_MAIN',FINAL[p['dataset']])['normalized']
            control=score(p,'MEAN_CORRECTION',FINAL[p['dataset']])['normalized']
            raw=score(p,'RAW_BOLT',FINAL[p['dataset']])['normalized']
            rows.append({'dataset':p['dataset'],'seed':p['seed'],'method':route,'feature_count':f['feature_state']['feature_count'],
                'C':m['channels'],'K':4 if p['dataset']=='robin' else 2,
                'potential_ridge_coefficients':f['feature_state']['feature_count']*48*m['channels'],
                'selected_registered_coefficients':m['registered_coefficients'],'selected_registered_intercepts':m['registered_intercepts'],
                'rank_min':m.get('rank_min'),'rank_max':m.get('rank_max'),
                'factorization_groups':f.get('preparation',{}).get('factorization_groups'),
                'selected_candidate_id':f['selected_candidate_id'],'selected_alpha':f['selected_alpha'],
                'selected_model_sha256':f['selected_model_sha256'],
                'tune_mse_margin_vs_ZERO':current-chosen,'tune_mse_margin_vs_MEAN':mean-chosen,
                'final_mse_delta_vs_MAIN':final['mse']-base['mse'],'final_mae_delta_vs_MAIN':final['mae']-base['mae'],
                'final_mse_delta_vs_MEAN':final['mse']-control['mse'],'final_mae_delta_vs_MEAN':final['mae']-control['mae'],
                'final_mse_delta_vs_RAW':final['mse']-raw['mse'],'final_mae_delta_vs_RAW':final['mae']-raw['mae'],
                'representation_correction_selected':f['selected_index']>=2,'strict_K_only_inference':False})
    return rows


def cpu_rows(producers, receipts):
    rows=[]
    for p,receipt in zip(producers,receipts):
        for route in ROUTES:
            f=p['families'][route]
            rows.append({'dataset':p['dataset'],'seed':p['seed'],'method':route,
                'FIT_feature_state_wall_seconds':f['FIT_feature_state_wall_seconds'],
                'new_FIT_factorization_wall_seconds':f['new_FIT_factorization_wall_seconds'],
                'family_including_apply_save_guards_wall_seconds':f['total_family_wall_seconds'],
                'shared_mean_FIT_wall_seconds':p['controls']['MEAN_CORRECTION']['new_FIT_mean_wall_seconds'],
                'producer_body_wall_seconds':p['body_wall_seconds'],'producer_process_wall_seconds':receipt['wall_seconds'],
                'historical_E_MAIN_generation_wall_seconds':None,'historical_learning_wall_seconds':None,
                'process_wall_is_GPU_kernel_time':False})
    return rows


def whole_rows(cost):
    rows=[]
    for r in cost['rows']:
        rows.append({'dataset':r['dataset'],'seed':r['seed'],'batch':r['batch'],
            'method':'MEAN_CORRECTION' if r['route']=='MEAN_CONTROL' else r['route'],'status':r['status'],
            'median_ms':1000*r['median_seconds'],'median_ms_per_window':1000*r['median_seconds_per_window'],
            'paired_RAW_median_ratio':r['paired_RAW_median_ratio'],'seconds':r['seconds'],
            'paired_RAW_seconds':r['paired_RAW_seconds'],'paired_RAW_ratios':r['paired_RAW_ratios'],
            'block':r['block'],'position':r['position'],'selected_candidate_id':r.get('selected_candidate_id'),
            'selected_alpha':r.get('selected_alpha'),'selected_model_sha256':r.get('selected_model_sha256'),
            'scientific_route_status':r.get('scientific_route_status'),
            'selected_zero_skips_feature_and_CPU_transfers':r.get('selected_zero_skips_feature_and_CPU_transfers'),
            'stage_decomposition':False,'reference_recipe_is_independent_math_replay':False})
    return rows


def summary(producers):
    return {d:{m:{v:statistics.mean(score(p,m,FINAL[d])['normalized'][v] for p in producers if p['dataset']==d)
                  for v in ('mse','mae')} for m in METHODS} for d in DATASETS}


def draw(producers, cost, plt, out, full):
    def finish(fig, stem, note):
        fig.text(.01,-.035,note,fontsize=8,color='#444444'); fig.tight_layout()
        for extension in ('png','svg'):
            full(); fig.savefig(out/'figures'/(stem+'.'+extension),dpi=180,bbox_inches='tight'); full()
        plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    for di,d in enumerate(DATASETS):
        subset=sorted((p for p in producers if p['dataset']==d),key=lambda p:p['seed'])
        for vi,v in enumerate(('mse','mae')):
            ax=axs[di,vi]
            for mi,m in enumerate(METHODS):
                vals=[score(p,m,FINAL[d])['normalized'][v] for p in subset]
                ax.plot([mi-.09,mi+.09],vals,'o',color=COLORS[mi],ms=6)
                ax.plot([mi-.22,mi+.22],[statistics.mean(vals)]*2,color=COLORS[mi],lw=2)
            ax.set_xticks(range(6),[LABELS[m] for m in METHODS],rotation=25); ax.set_ylabel('Normalized '+v.upper())
            ax.set_title(d+' / '+FINAL[d]); ax.grid(axis='y',alpha=.2)
    finish(fig,FIGURES[0],'Two original encoder seeds; descriptive means. Same exposed subsets; no independent holdout.')
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    for di,d in enumerate(DATASETS):
        for vi,v in enumerate(('mse','mae')):
            ax=axs[di,vi]
            for j,p in enumerate(sorted((p for p in producers if p['dataset']==d),key=lambda p:p['seed'])):
                base=score(p,'ORIGINAL_MAIN',FINAL[d])['normalized'][v]; mean=score(p,'MEAN_CORRECTION',FINAL[d])['normalized'][v]
                values=[mean-base]+[score(p,r,FINAL[d])['normalized'][v]-mean for r in ROUTES]
                ax.plot(range(4),values,'o-',label=str(p['seed']),alpha=.8)
            ax.axhline(0,color='#333333',lw=.8); ax.set_xticks(range(4),['Mean-MAIN','RAW33-Mean','E33-Mean','E512-Mean'],rotation=20)
            ax.set_title(d+' / '+v.upper()+' delta'); ax.set_ylabel('Negative is lower error'); ax.legend(); ax.grid(alpha=.2)
    finish(fig,FIGURES[1],'Selected ZERO can beat an adverse mean without using features. Interpret selected candidate type alongside deltas.')
    fig,axs=plt.subplots(2,2,figsize=(14,9))
    for ax,p in zip(axs.flat,sorted(producers,key=lambda p:(DATASETS.index(p['dataset']),p['seed']))):
        for ri,r in enumerate(ROUTES):
            f=p['families'][r]; vals=[v['tune_actual_Y_mse'] for v in f['candidate_grid']]
            ax.plot(range(9),vals,'o-',label=r,color=COLORS[ri+3]); ix=f['selected_index']
            ax.scatter([ix],[vals[ix]],marker='*',s=170,color=COLORS[ri+3],edgecolor='black',zorder=4)
        ax.set_xticks(range(9),['ZERO','MEAN','a=0','1e-4','.001','.01','.1','1','10'],rotation=30)
        ax.set_yscale('log'); ax.set_ylabel('Actual-Y TUNE normalized MSE'); ax.set_title(p['dataset']+' / '+str(p['seed'])); ax.legend(); ax.grid(alpha=.2)
    finish(fig,FIGURES[2],'Fixed nine-candidate pool; earliest actual-Y TUNE tie wins. ZERO is distinct from ridge alpha=0. Log axis retains poor candidates.')
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    for ax,p in zip(axs.flat,sorted(producers,key=lambda p:(DATASETS.index(p['dataset']),p['seed']))):
        for mi,m in enumerate(METHODS):
            ax.plot(range(3),[score(p,m,s)['normalized']['mse'] for s in ('fit','tune',FINAL[p['dataset']])],'o-',label=LABELS[m],color=COLORS[mi])
        ax.set_xticks(range(3),['FIT','TUNE',FINAL[p['dataset']].upper()]); ax.set_ylabel('Normalized MSE')
        ax.set_title(p['dataset']+' / '+str(p['seed'])); ax.legend(ncol=2,fontsize=8); ax.grid(alpha=.2)
    finish(fig,FIGURES[3],'FIT trains coefficients; TUNE alone selects; final split was already exposed. Distribution differences are not causal evidence.')
    fig,axs=plt.subplots(2,1,figsize=(14,9))
    for ax,d in zip(axs,DATASETS):
        matrix=[]; names=[]
        for p in sorted((p for p in producers if p['dataset']==d),key=lambda p:p['seed']):
            base=score(p,'MEAN_CORRECTION',FINAL[d])['normalized']['per_channel_mse']
            for r in ROUTES:
                row=score(p,r,FINAL[d])['normalized']['per_channel_mse']
                matrix.append([float('nan') if a is None or b is None else a-b for a,b in zip(row,base)])
                names.append(r+' '+str(p['seed'])+' '+p['families'][r]['selected_candidate_id'].replace('_CORRECTION',''))
        bound=max((abs(x) for row in matrix for x in row if math.isfinite(x)),default=1) or 1
        im=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-bound,vmax=bound)
        ax.set_yticks(range(6),names,fontsize=8); ax.set_xlabel('Channel index'); ax.set_title(d+' normalized MSE delta vs mean')
        fig.colorbar(im,ax=ax,label='Negative: lower error')
    finish(fig,FIGURES[4],'Observed-channel normalization; missing channels are blank. Selected ZERO does not use correction features.')
    fig,axs=plt.subplots(2,2,figsize=(14,8))
    for di,d in enumerate(DATASETS):
        for bi,b in enumerate((1,8)):
            order=('RAW_BOLT','ORIGINAL_MAIN','MEAN_CONTROL')+ROUTES
            ax=axs[di,bi]; rs=sorted((r for r in cost['rows'] if r['dataset']==d and r['batch']==b),key=lambda r:order.index(r['route']))
            for i,r in enumerate(rs):
                ax.plot([i]*6,[v*1000 for v in r['seconds']],'.',color=COLORS[i],alpha=.55)
                ax.plot([i-.2,i+.2],[1000*r['median_seconds']]*2,color=COLORS[i],lw=2)
            ax.set_xticks(range(6),[LABELS['MEAN_CORRECTION' if r['route']=='MEAN_CONTROL' else r['route']] for r in rs],rotation=25)
            ax.set_ylabel('Whole-path wall ms'); ax.set_title(d+' / B'+str(b)+' / seed92601'); ax.grid(axis='y',alpha=.2)
    finish(fig,FIGURES[5],'Latin6, warmup2, repeats6. Selected ZERO routes execute MAIN identically and skip features; their timing differences are measurement variation.')


def write_report(out, producers, audit, cost, capacities, attempts, values):
    lines=['# MAIN 예측을 유지한 과거 정보 기반 잔차 보정','',
        '이번 실험은 기존 인코더·Chronos-Bolt·디코더와 MAIN 예측을 고정하고, 과거 표현이 그 예측의 오차를 추가로 줄일 수 있는지 확인했다. '
        'TUNE에서 보정을 거부할 수 있게 원 MAIN 그대로인 ZERO를 항상 포함했다. 성능의 조건부 개선, 단순 평균 보정, 표현의 추가 효과를 분리한다.','',
        f'[확인] 실제 조건 4개, 새 ridge 계수 84개, 고정 후보 비교 108개, 선택 예측 36개를 수행했다. 독립 CPU 수치 검산 상태는 `{audit["status"]}`이며 비용은 whole-path 24행이다. '
        '검산 통과는 성능 개선이나 새로운 데이터 일반화 검증을 뜻하지 않는다.','',
        '## 먼저 답해야 할 질문','']
    for d in DATASETS:
        lines+=['### '+d,'','```text','방법       최종 정규화 MSE   최종 정규화 MAE   (원 인코더 seed 2개 평균)']
        for m in METHODS: lines.append(f'{LABELS[m]:<10} {values[d][m]["mse"]:.10f}       {values[d][m]["mae"]:.10f}')
        lines+=['```','']
        chosen=[c for c in capacities if c['dataset']==d]
        for c in chosen:
            lines.append(f'- seed {c["seed"]} {c["method"]}: `{c["selected_candidate_id"]}` (α={c["selected_alpha"]}), '
                f'TUNE MSE 여유 ZERO 대비 {c["tune_mse_margin_vs_ZERO"]:.10g}, MEAN 대비 {c["tune_mse_margin_vs_MEAN"]:.10g}; '
                f'최종 MSE 변화 MAIN 대비 {c["final_mse_delta_vs_MAIN"]:+.10g}, MEAN 대비 {c["final_mse_delta_vs_MEAN"]:+.10g}; '
                f'MAE 변화 MAIN 대비 {c["final_mae_delta_vs_MAIN"]:+.10g}, MEAN 대비 {c["final_mae_delta_vs_MEAN"]:+.10g}.')
        lines.append('')
        ridge=[c for c in chosen if c['representation_correction_selected']]
        if not ridge:
            lines+=['[확인] 이 데이터의 세 경로·두 seed 모두 TUNE에서 새 표현 보정을 선택하지 않았다. ZERO 선택은 기존 예측을 그대로 보존했다는 뜻이다. '
                    '이 결과만으로 과거 표현에 정보가 없거나 더 나은 학습이 불가능하다고 결론낼 수 없다.','']
        else:
            lines+=['[판단] 일부 ridge 보정은 MSE를 줄였지만 원수치와 MAE를 함께 봐야 한다. RAW33C는 원 채널을 사용하며 p가 다르므로 '
                    'E33K 대비 우위를 인코더 실패의 단독 원인으로 해석할 수 없다. 아주 작은 TUNE 선택 여유는 조건부 결과이며 반복 사용한 평가 구간에서의 안정성은 미확정이다.','']
    lines+=['## 비교 계약과 남는 한계','',
        '동일한 원 MAIN 체크포인트와 FIT/TUNE/기존 노출 최종 구간을 사용했다. Robin C=17,K=4, Beijing C=11,K=2, L=512,H=48이다. '
        'RAW33C는 원 C 채널의 32개 블록 평균과 마지막 값, E33K는 같은 요약의 K 표현, E512K는 전체 과거 표현이다. '
        'FIT 중심·스케일만 사용하며 실제 Y−MAIN 잔차는 FP32로 먼저 계산했다. 관측 마스크의 채널 전체 N_c를 사용하는 정규화 채널 macro MSE ridge이며 절편은 벌점에서 제외했다.','',
        '후보 순서는 ZERO → 공유 FIT 평균 잔차 MEAN → α=0,10⁻⁴,10⁻³,10⁻²,10⁻¹,1,10의 ridge다. 실제-Y TUNE MSE만으로 선택하며 동률이면 앞 후보다. '
        'ZERO는 ridge α=0과 다르다. 모든 새 계수를 학습한 뒤 ZERO를 선택할 수 있다. MEAN 우위는 평균 오차 보정이며 표현의 추가 효과와 구분한다.','',
        '원 MAIN 자체가 현재 C 채널 anchor를 사용한다. 따라서 세 보정 경로 모두 엄격한 K-only 배포가 아니다. RAW33C는 추가 원 채널 과거 정보를 제공한다. '
        'p, 계수 개수, rank 및 정규화가 다르므로 같은 α가 같은 용량·같은 prior를 보장하지 않는다. 원 인코더·F·D는 이번 CPU 계수 학습에서 로드하거나 갱신하지 않았다.','',
        '보호 TEST를 사용하지 않았다. 최종 구간은 이미 노출된 Robin VAL/Beijing EVAL이며 독립 holdout이 아니다. RAW 입력은 두 인코더 seed에서 동일한 결정적 경로이므로 '
        '두 RAW 결과를 독립 데이터셋 두 개로 세지 않는다. 시점별 인코더 인과성, 대조학습·그래프 필요성, Chronos-Bolt의 일반적 유용성은 이번 보정 비교로 입증되지 않는다.','',
        '## 그림과 원자료','']
    captions=('최종 MSE·MAE와 두 seed 원수치','MEAN의 변화와 선택 보정의 추가 변화','고정 9개 후보의 TUNE 점수와 선택',
              'FIT/TUNE/기존 최종 구간의 차이','채널별 MSE 변화: MEAN 대비','whole-path 비용: 같은 batch와 균형 순서')
    for stem,caption in zip(FIGURES,captions):
        lines+=['### '+caption,'',f'![{caption}](figures/{stem}.png)','',f'[SVG](figures/{stem}.svg)','']
    lines+=['- [전체 점수·채널 관측 수](METRICS.csv)','- [108개 후보](CANDIDATE_GRID.csv)',
        '- [용량·rank·선택 여유](CAPACITY_SELECTION.csv)','- [새 계수 학습 CPU 비용](CPU_COST.csv)',
        '- [whole-path 24행](WHOLE_COST.csv)','',
        '## 비용 해석과 보존한 실패','',
        '새 FIT factorization, 경로 전체 CPU 실행, 프로세스 wall 비용을 따로 남겼다. 과거 E/MAIN 캐시 생성과 원 학습 비용은 알 수 없으므로 0으로 쓰지 않았다. '
        'whole-path는 seed92601, B1/B8, Latin6 순서, warmup2/repeat6이다. CPU fill·전송·E/F/D·선택 보정·복원까지 포함하지만 '
        '로드·SHA·검산·저장·timer 밖 guard는 제외한다. stage 분해나 GPU kernel 시간으로 해석하지 않는다. ZERO는 추가 feature·CPU 전송을 생략한다. '
        'ZERO로 선택된 RAW33C/E33K/E512K 비용 라벨은 MAIN과 동일한 계산 경로다. 이들 수치의 작은 차이는 별개의 학습 방법 차이로 해석하지 않는다. '
        '같은 batch 원시 예측과의 출력 parity를 검산했고 thermal/carryover 영향을 완전히 제거했다고 주장하지 않는다. '
        '이번 B1/B8 실행의 원수치이며 이전 세션의 RAW 비용과 차이가 있어 안정적 속도 우위를 일반화하지 않는다. '
        '독점 연구 job lease는 병행 연구 프로세스를 막지만 Windows WDDM·데스크톱의 배경 GPU 활동까지 제거한 것은 아니다.','']
    if attempts:
        for a in attempts:
            lines.append(f'- 보존한 시도 `{a["name"]}`: `{a["status"]}`, 새 ridge {a["new_ridge_candidates_completed"]}, '
                f'비교 {a["candidate_comparisons_completed"]}. 실패 설명: {a["failure"]}. [원 JSON]({a["copy_relative"]})')
        lines+=['','이 시도들은 완료 조건 4개와 84/108 개수에 포함하지 않는다. `scientific_execution=true` 표시는 프로세스 진입이며 '
                '그 자체가 자산 로드·학습·추론을 실행했다는 증거가 아니다. 자산 전 경로 오류와 실제 학습 실패를 구분해 보존했다.','']
    lines+=['## 출처와 실행 상태','',
        '[독립 수치 검산](provenance/inputs/INDEPENDENT_AUDIT.json), [비용 검산](provenance/inputs/COST_AUDIT.json), '
        '[승인](provenance/inputs/AUTHORIZATION.json), [고정 실행 manifest](provenance/inputs/RUN_MANIFEST.json), '
        '[보고서 manifest](MANIFEST.json), [보고서 실제 종료 상태](RESULT.json).','',
        'provenance/sources의 계획·lock에 있는 SOURCE_ONLY 문구는 실행 전 역사적 snapshot을 그대로 보존한 것이다. '
        '실제 완료 여부는 승인·실행 manifest·생산자·검산·보고서 RESULT의 종료 상태로 판단한다. '
        '보고서 생성기는 종료 JSON과 코드만 읽었으며 별도 검증기가 저장 예측 배열→CSV를 CPU로 검산한다.','',
        '다음 판단은 조건부다. Robin은 새 보정의 TUNE 근거가 없어 원 MAIN을 보존했다. Beijing에서 작은 MSE 변화와 MAE 악화를 '
        '분리해 보고, 단순 MEAN과 원 채널 RAW33C를 넘어서는 일관된 표현의 추가 이득이 있는지 확인해야 한다. '
        '현재 결과는 전체 연구 방법을 해결했다는 증거가 아니다.','']
    with (out/'REPORT_KO.md').open('x',encoding='utf-8') as handle: handle.write('\n'.join(lines))


def execute(args):
    import main_residual_runtime_v1 as runtime
    limit=runtime.ContinuationLimit('main_residual_report_v1',gpu=False)
    out=resolve(args.out); full=limit.checkfull
    result={'status':'RUNNING','terminal_JSON_only':True,'NPZ_or_models_read':False,
            'whole_goal_complete':False,'authorization':limit.authorization_record,'run_manifest':limit.run_manifest_record}
    initialized=False
    try:
        full(); producers=[]; records=[]
        for path in args.results:
            full(); r=record(path,'producer'); records.append(r); producers.append(read(r['path']))
        audit_r=record(args.audit,'independent_audit'); audit=read(audit_r['path'])
        cost_r=record(args.cost,'whole_cost'); cost=read(cost_r['path'])
        cost_audit_r=record(args.cost_audit,'cost_audit'); cost_audit=read(cost_audit_r['path']); full()
        result['audit_gate']=validate(producers,records,audit,cost,cost_r,cost_audit)
        receipts=[]; receipt_records=[]
        for p in producers:
            full(); r=p['resource_receipt']; path=resolve(r['path'])
            if sha(path)!=r['sha256']: raise ValueError('PRODUCER_RECEIPT_HASH')
            receipt=read(path)
            if receipt['status']!='COMPLETE' or receipt['gpu'] is not False: raise ValueError('COMPLETE_CPU_RECEIPT_REQUIRED')
            receipts.append(receipt); receipt_records.append(record(path,'producer_receipt'))
        out.mkdir(parents=True,exist_ok=False); initialized=True
        for p in ('figures','provenance/inputs','provenance/sources'): (out/p).mkdir(parents=True)
        copied=[]
        def copy(r, relative):
            full(); destination=out/relative; destination.parent.mkdir(parents=True,exist_ok=True)
            with Path(r['path']).open('rb') as src,destination.open('xb') as dst: shutil.copyfileobj(src,dst)
            if sha(destination)!=r['sha256']: raise ValueError('REPORT_COPY_SHA')
            copied.append({**r,'copy_relative':relative}); full(); return relative
        for p,r,receipt_r in zip(producers,records,receipt_records):
            name=p['dataset']+'_s'+str(p['seed'])
            copy(r,'provenance/inputs/'+name+'.json'); copy(receipt_r,'provenance/inputs/'+name+'_RESOURCE.json')
        for r,name in ((audit_r,'INDEPENDENT_AUDIT.json'),(cost_r,'WHOLE_COST.json'),(cost_audit_r,'COST_AUDIT.json'),
                ({**limit.authorization_record,'role':'authorization'},'AUTHORIZATION.json'),
                ({**limit.run_manifest_record,'role':'run_manifest'},'RUN_MANIFEST.json')):
            copy(r,'provenance/inputs/'+name)
        for value,role in ((cost,'cost_receipt'),(audit,'audit_receipt'),(cost_audit,'cost_audit_receipt')):
            receipt=value.get('resource_receipt')
            if receipt and receipt.get('path'):
                if sha(resolve(receipt['path']))!=receipt['sha256']: raise ValueError('AUX_RECEIPT_SHA')
                copy(record(receipt['path'],role),'provenance/inputs/'+role.upper()+'.json')
        if cost_audit.get('metrics_csv'):
            metric_record=cost_audit['metrics_csv']
            if sha(resolve(metric_record['path']))!=metric_record['sha256']: raise ValueError('COST_AUDIT_METRICS_SHA')
            copy(record(metric_record['path'],'cost_audit_metrics'),'provenance/inputs/COST_AUDIT_METRICS.csv')
        attempts=[]
        for index,path in enumerate(args.attempts or []):
            r=record(path,'preserved_attempt'); a=read(r['path']); relative=copy(r,'provenance/inputs/ATTEMPT_'+str(index)+'.json')
            attempts.append({'name':Path(path).parent.name,'status':a.get('status'),'scientific_execution_flag':a.get('scientific_execution'),
                'new_ridge_candidates_completed':a.get('new_ridge_candidates_completed',0),
                'candidate_comparisons_completed':a.get('candidate_comparisons_completed',0),
                'failure':a.get('failure'),'copy_relative':relative})
        for index,path in enumerate(args.fixtures or []): copy(record(path,'fixture_evidence'),'provenance/inputs/FIXTURE_'+str(index)+'.json')
        sources=['make_main_residual_report_v1.py','verify_main_residual_report_v1.py','run_main_residual_v1.py','run_main_residual_v2.py',
            'main_forecast_residual_readout_v1.py','matched_readout_math_v1.py','stage_separation_math_v1.py','main_residual_runtime_v1.py',
            'test_main_forecast_residual_readout_v1.py','test_run_main_residual_v1.py','test_run_main_residual_v2.py',
            'PLAN_MAIN_FORECAST_RESIDUAL_KO_v1.md','METHOD_LOCK_MAIN_RESIDUAL_v1.md','profile_main_residual_cost_v1.py',
            'test_profile_main_residual_cost_v1.py','test_audit_main_residual_cost_v1.py',
            'test_audit_main_residual_v1.py','test_main_residual_runtime_v1.py',
            'test_main_residual_runtime_v2.py','test_main_residual_runtime_v3.py',
            'activate_main_residual_fixtures_v1.py','run_main_residual_aux_fixtures_v1.py',
            'fast_storage_scan_v1.py','run_condition_probes.py','prepare_inputs.py','frozen_decoder_refit.py',
            'run_matched_readout_v1.py','matched_readout_contract_v1.py']
        if args.audit_source: sources.append(args.audit_source)
        else: sources.append('audit_main_residual_v1.py')
        sources.append(args.cost_audit_source or 'audit_main_residual_cost_v1.py')
        source_names=set()
        for path in sources:
            path=Path(path); source=path if path.is_absolute() else HERE/path
            if source.name in source_names: continue
            source_names.add(source.name); copy(record(source,'source'),'provenance/sources/'+source.name)
        guides=[]
        for original,relative,target in (
            (HERE/'MATCHED_READOUT_INTERPRETATION_KO_v1.md','provenance/sources/MATCHED_READOUT_INTERPRETATION_KO_v1.md','../../../MATCHED_READOUT_INTERPRETATION_KO_v1.md'),
            (HERE/'report_matched_readout_v2/REPORT_KO.md','provenance/sources/report_matched_readout_v2/REPORT_KO.md','../../../../report_matched_readout_v2/REPORT_KO.md')):
            full(); digest=sha(original); path=out/relative; path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x',encoding='utf-8') as handle:
                handle.write('# 원 문서 상대 링크 안내\n\n이 파일은 연결 안내이며 원 문서의 바이트 사본이나 새 결과가 아니다.\n\n'
                    '[원 문서]('+target+')\n\n원 SHA256: `'+digest+'`\n')
            guides.append({'path':relative,'sha256':sha(path),'original_source':{'path':str(original),'sha256':digest},'byte_identical_source_copy':False})
        metrics=metrics_rows(producers); candidates=candidate_rows(producers); capacities=capacity_rows(producers)
        cpu=cpu_rows(producers,receipts); whole=whole_rows(cost)
        for filename,rows in (('METRICS.csv',metrics),('CANDIDATE_GRID.csv',candidates),('CAPACITY_SELECTION.csv',capacities),('CPU_COST.csv',cpu),('WHOLE_COST.csv',whole)):
            full()
            with (out/filename).open('x',encoding='utf-8',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader()
                for row in rows: writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
            full()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
        draw(producers,cost,plt,out,full); values=summary(producers)
        full(); write_report(out,producers,audit,cost,capacities,attempts,values); full()
        for r in copied:
            if sha(r['path'])!=r['sha256'] or sha(out/r['copy_relative'])!=r['sha256']: raise ValueError('REPORT_SOURCE_OR_INPUT_CHANGED')
        for g in guides:
            if sha(g['original_source']['path'])!=g['original_source']['sha256']: raise ValueError('GUIDE_TARGET_CHANGED')
        artifacts=[{'path':str(p.relative_to(out)).replace('\\','/'),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()]
        manifest={'status':'COMPLETE','input_records':copied,'artifact_records':artifacts,'source_link_guides':guides,
            'audit_gate':result['audit_gate'],'summary':values,'metrics_rows':len(metrics),'candidate_rows':len(candidates),
            'capacity_rows':len(capacities),'CPU_cost_rows':len(cpu),'whole_cost_rows':len(whole),
            'new_ridge_candidates':84,'candidate_comparisons':108,'shared_mean_conditions':4,'selected_split_predictions':36,
            'figure_png_count':6,'figure_svg_count':6,'conditions':[{'dataset':p['dataset'],'seed':p['seed']} for p in producers],
            'preserved_attempts':attempts,'report_scope':'terminal JSON/code only; no NPZ/model replay',
            'independent_holdout':False,'strict_K_only_inference':False,'whole_goal_complete':False,
            'historical_learning_and_cache_cost_unknown_not_zero':True}
        full(); runtime.write_new(out/'MANIFEST.json',manifest); full()
        result.update(status='COMPLETE',manifest={'path':'MANIFEST.json','sha256':sha(out/'MANIFEST.json')},summary=values,figures=list(FIGURES))
    except Exception as error:
        result.update(status='BUDGET_HELD_PRESERVED' if isinstance(error,runtime.BudgetHeld) else 'FAILED_PRESERVED',
            failure={'type':type(error).__name__,'message':str(error)})
    finally:
        receipt=limit.close(result['status'],result.get('failure')); result['resource_receipt']=receipt
        if receipt.get('terminal_status')!='COMPLETE': result.update(status=receipt['terminal_status'],resource_terminal_error=receipt.get('terminal_error'))
        if initialized: runtime.write_new(out/'RESULT.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute',action='store_true'); mode.add_argument('--describe',action='store_true')
    parser.add_argument('--results',nargs=4); parser.add_argument('--audit'); parser.add_argument('--cost'); parser.add_argument('--cost-audit')
    parser.add_argument('--attempts',nargs='*'); parser.add_argument('--fixtures',nargs='*')
    parser.add_argument('--audit-source'); parser.add_argument('--cost-audit-source')
    parser.add_argument('--out',default=str(HERE/'report_main_residual_v1')); args=parser.parse_args()
    if args.describe:
        print(json.dumps({'status':'SOURCE_ONLY_NOT_EXECUTED','figures':list(FIGURES),'canonical_auth_required':True})); return 0
    if not args.results or not args.audit or not args.cost or not args.cost_audit: parser.error('execute requires four results, audit, cost and cost-audit')
    result=execute(args); print(json.dumps({'status':result['status'],'failure':result.get('failure')})); return 0 if result['status']=='COMPLETE' else 1


if __name__=='__main__': raise SystemExit(main())
