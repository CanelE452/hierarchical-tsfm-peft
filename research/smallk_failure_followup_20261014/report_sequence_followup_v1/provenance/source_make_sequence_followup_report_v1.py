"""SOURCE-only default; CPU report from terminal RF1/RF3 JSON/CSV evidence.

Never open scientific NPZ/checkpoints or construct F. Actual generation and
fixtures require root GO after the live fit queue and cost jobs have ended.
"""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import make_followup_report as common
import make_joint_followup_report_v4 as prior

DATASETS=('robin','beijing_aotizhongxin')
SEEDS=(92601,92602)
HEADS=('STATIC_RF1','TEMPORAL_RF3')
FINAL={'robin':'val','beijing_aotizhongxin':'eval'}
EXPECTED={(d,s,h) for d in DATASETS for s in SEEDS for h in HEADS}
TERMINAL={'COMPLETE','FAILED_PRESERVED','BUDGET_HELD_PRESERVED'}
COLOR={'RAW_BOLT':'#333333','ORIGINAL_MAIN':'#777777','STATIC_RF1':'#0072B2','TEMPORAL_RF3':'#D55E00'}
FIGURES=('01_selected_forecast_scores','02_actual_Y_TUNE_checkpoints','03_E_D_head_learning_proof',
         '04_position_balanced_whole_cost','05_separate_synchronized_stages','06_APP_stronger_RAW_R_context')


def producer_refs(queue,require_complete=False):
    if queue.get('status') not in TERMINAL|{'PARTIAL_WITH_FAILURES'}:raise ValueError('TERMINAL_SEQUENCE_QUEUE_REQUIRED')
    seen=set();records=[]
    for row in queue.get('conditions',[]):
        key=(row.get('dataset'),row.get('seed'),row.get('head'))
        if key not in EXPECTED or key in seen:raise ValueError('UNEXPECTED_OR_DUPLICATE_SEQUENCE_QUEUE_CONDITION')
        if row.get('status') not in TERMINAL:raise ValueError('TERMINAL_FIT_STATUS_REQUIRED')
        seen.add(key);record=row.get('producer_result',{})
        if not record.get('path') or not record.get('sha256'):raise ValueError('ACTUAL_QUEUE_PRODUCER_PATH_AND_SHA_REQUIRED')
        records.append({**record,'dataset':key[0],'seed':key[1],'head':key[2],'status':row['status']})
    if require_complete and (seen!=EXPECTED or any(r['status']!='COMPLETE' for r in records)):raise ValueError('COST_REQUIRES_ALL8_COMPLETE_ACTUAL_PRODUCERS')
    return records


def cost_cli_arguments(queue,bridge,audit,tag):
    refs=producer_refs(queue,require_complete=True)
    return ['--execute','--input-bridge',str(bridge),'--sequence-results',*[r['path'] for r in refs],
            '--sequence-audit',str(audit),'--tag',tag,'--no-contention-attested']


def validate_complete(p):
    learning=p.get('learning_executed',{})
    if (p.get('status')!='COMPLETE' or p.get('objective')!='Y' or learning.get('optimizer_updates')!=600
            or learning.get('full_learning_proof_pass') is not True or not all(learning.get(g+'_updated') is True for g in ('E','D','Head'))
            or learning.get('frozen_F_input_gradient_nonzero_diagnostic_count',0)<=0
            or p.get('F_frozen_pass') is not True or p.get('F_has_no_parameter_gradients') is not True
            or p.get('initial_fixed_basis_unchanged') is not True):raise ValueError('COMPLETE600_E_D_HEAD_F_PROOF_REQUIRED')
    rows=p['checkpoints'];index=p['selected_index']
    if ([r['new_update'] for r in rows]!=[0,100,200,400,600] or not all(np.isfinite(r['tune_actual_Y_mse']) for r in rows)
            or index!=min(range(5),key=lambda i:rows[i]['tune_actual_Y_mse']) or rows[index]['checkpoint']!=p['selected_checkpoint']
            or p['selected_new_update']!=rows[index]['new_update'] or p['selected_is_initial_original_MAIN']!=(index==0)):
        raise ValueError('ACTUAL_Y_TUNE_EARLIEST_SELECTED_CP_REQUIRED')
    keys=('state_sha_old_algorithm','state_sha_Bi_algorithm','head_state_sha_old_algorithm','head_state_sha_Bi_algorithm','combined_state_sha_Bi_algorithm')
    if any(p['selected_state_hashes'][k]!=p['selected_checkpoint'][k] for k in keys):raise ValueError('SELECTED_CODEC_HEAD_HASH_METADATA_MISMATCH')


def collect_sequence(producers,records):
    tables={name:[] for name in ('scores','checkpoints','curve','learning','gradients','status','asset_references')};seen=set()
    for p,r in zip(producers,records):
        key=(p.get('dataset'),p.get('seed'),p.get('head'));seen.add(key)
        if key!=tuple(r[k] for k in ('dataset','seed','head')) or p.get('status')!=r['status']:raise ValueError('QUEUE_PRODUCER_IDENTITY_MISMATCH')
        base={'dataset':key[0],'seed':key[1],'head':key[2]}
        tables['status'].append({**base,'status':p['status'],'actual_updates':len(p.get('curve',[])),
            'selected_new_update':p.get('selected_new_update'),'failure':json.dumps(p.get('failure'),ensure_ascii=False),
            'producer_path':r['path'],'producer_sha256':r['sha256']})
        tables['curve'].extend({**base,**v} for v in p.get('curve',[]))
        for row in p.get('checkpoints',[]):
            tables['checkpoints'].append({**base,'new_update':row['new_update'],'actual_Y_TUNE_mse':row['tune_actual_Y_mse'],
                'selected':p.get('selected_new_update')==row['new_update'],'checkpoint_path':row['checkpoint']['path'],'checkpoint_sha256':row['checkpoint']['sha256']})
        for name in ('original_MAIN_prediction_source','original_RAW_prediction_source','warmstart_all_splits'):
            if name in p:tables['asset_references'].append({**base,'role':name,**p[name],'report_opened_scientific_array':False})
        if p['status']!='COMPLETE':continue
        validate_complete(p)
        for split,v in p['selected_inference'].items():
            for method,scores in ((p['head'],v['scores']),('ORIGINAL_MAIN',p['original_MAIN_scores'][split]),('RAW_BOLT',p['original_RAW_scores'][split])):
                score=scores['normalized_actual_Y']
                tables['scores'].append({**base,'method':method,'split':split,'mse':score['mse'],'mae':score['mae'],
                    'physical_mse':scores['physical_actual_Y']['mse'],'physical_mae':scores['physical_actual_Y']['mae'],
                    'selected_new_update':p['selected_new_update'],'selected0':p['selected_is_initial_original_MAIN'],
                    'prediction_source_path':(v['artifact'] if method==p['head'] else p['original_MAIN_prediction_source'] if method=='ORIGINAL_MAIN' else p['original_RAW_prediction_source'])['path'],
                    'prediction_source_sha256':(v['artifact'] if method==p['head'] else p['original_MAIN_prediction_source'] if method=='ORIGINAL_MAIN' else p['original_RAW_prediction_source'])['sha256']})
        change=p['new_update_parameter_proof'];capacity=p['head_capacity']
        tables['learning'].append({**base,**p['learning_executed'],
            **{g+'_changed_tensors':change[g]['changed_parameter_tensors'] for g in ('E','D','Head')},
            'head_registered_parameters':p['registered_head_parameters'],'head_potentially_active':capacity['potentially_active'][p['head']],
            'registered_capacity_relative_gap':capacity['registered_symmetric_relative_difference'],
            'active_capacity_relative_gap':capacity['active_symmetric_relative_difference'],
            'selected_new_update':p['selected_new_update'],'selected0':p['selected_is_initial_original_MAIN'],
            'training_checkpoint_wall_seconds':p['training_checkpoint_wall_seconds'],
            'guard_full_wall_seconds':p['guard_cadence']['explicit_wall_seconds']['full']})
        tables['gradients'].extend({**base,'new_update':v['new_update'],'F_input_gradient_norm':v['frozen_F_input_gradient_norm'],
            'Head_gradient_norm':v['Head_gradients']['norm'],'Head_gradient_finite':v['Head_gradients']['finite']} for v in p['gradient_diagnostics'])
    for d,s,h in sorted(EXPECTED-seen):tables['status'].append({'dataset':d,'seed':s,'head':h,'status':'NOT_EXECUTED','actual_updates':0,'failure':'not present in terminal queue; no result inferred'})
    return tables


def audit_gate(audit,producers,records):
    declared={(v['dataset'],v['seed'],v['head']):v['producer_sha256'] for v in audit.get('conditions',[])}
    expected={(p['dataset'],p['seed'],p['head']):r['sha256'] for p,r in zip(producers,records)}
    bound=set(declared)==set(expected)==EXPECTED and all(declared.get(k)==v for k,v in expected.items()) and all(p.get('status')=='COMPLETE' for p in producers)
    return {'passed':str(audit.get('status','')).startswith('PASS') and audit.get('failed_count')==0 and bound,
        'producer_SHA_bound':bound,'status':audit.get('status'),'checks':audit.get('checks'),'failed_count':audit.get('failed_count'),
        'scope':'stored-latent CPU decoder/head and metrics reconstruction; no F/past E forward or per-update replay'}


def cost_binding(cost,records):
    expected={r['sha256'] for r in records}
    actual={r['sha256'] for r in cost.get('sequence_producers',[])}
    if len(records)!=8 or len(cost.get('sequence_producers',[]))!=8 or actual!=expected:
        raise ValueError('BALANCED_COST_ALL8_PRODUCER_SHA_BINDING_REQUIRED')


def flat_binding(diagnostic,records):
    sources={r['sha256'] for r in diagnostic.get('source_records',[])}
    rows=diagnostic.get('rows',[])
    expected={(r['dataset'],r['seed'],r['head'],split) for r in records for split in ('fit','tune',FINAL[r['dataset']])}
    actual={(r['dataset'],r['seed'],r['head'],r['split']) for r in rows}
    if (diagnostic.get('status')!='COMPLETE' or len(rows)!=24 or actual!=expected
            or any(r['sha256'] not in sources for r in records) or diagnostic.get('E_or_F_replayed') is not False
            or diagnostic.get('causal_failure_identified') is not False):
        raise ValueError('COMPLETE_BOUND24_DESCRIPTIVE_FLAT_DIAGNOSTIC_REQUIRED')
    return rows


def paired_differences(rows):
    values={(r['dataset'],r['seed'],r['method']):r for r in final_scores(rows)};out=[]
    for dataset in DATASETS:
        for seed in SEEDS:
            for reference in ('STATIC_RF1','RAW_BOLT','ORIGINAL_MAIN'):
                candidate=values.get((dataset,seed,'TEMPORAL_RF3'));control=values.get((dataset,seed,reference))
                if candidate and control:out.append({'dataset':dataset,'seed':seed,'candidate':'TEMPORAL_RF3','reference':reference,
                    'mse_difference':float(candidate['mse'])-float(control['mse']),'mae_difference':float(candidate['mae'])-float(control['mae']),
                    'interpretation':'negative is lower error; paired reused development split, not independent generalization'})
    return out


def collect_cost(value):
    if value.get('status') not in TERMINAL:raise ValueError('TERMINAL_BALANCED_COST_REQUIRED')
    rows=[];stage_rows=[]
    for r in value.get('rows',[]):
        seconds=r.get('seconds',[]);ratios=r.get('paired_RAW_ratios',[])
        rows.append({'dataset':r['dataset'],'batch':r['batch'],'route':r['route'],'status':r['status'],'repeats':len(seconds),
            'method':r.get('method'),'seed':r.get('seed'),'selected_new_update':r.get('selected_new_update'),
            'median_ms':float(np.median(seconds))*1000 if seconds else None,
            'median_ms_per_window':float(np.median(seconds))*1000/r['batch'] if seconds else None,
            'paired_RAW_ratios':json.dumps(ratios),'paired_RAW_median_ratio':r.get('paired_RAW_median_ratio'),
            'seconds':json.dumps(seconds),'blocks':json.dumps(r.get('block',[])),'positions':json.dumps(r.get('position',[])),
            'timed_output_path':r.get('timed_outputs',{}).get('path'),'timed_output_sha256':r.get('timed_outputs',{}).get('sha256')})
    for r in value.get('stage_rows',[]):
        for stage,seconds in r.get('stage_median_seconds',{}).items():
            stage_rows.append({'dataset':r['dataset'],'batch':r['batch'],'route':r['route'],'status':r['status'],
                'repeats':r['repeats'],'stage':stage,'median_ms_with_sync':seconds*1000,'stage_sum_is_whole_latency':False})
    return rows,stage_rows


def final_scores(rows):
    selected={}
    for r in rows:
        if r['split']!=FINAL[r['dataset']]:continue
        key=(r['dataset'],r['seed'],r['method'])
        if key in selected and any(not np.isclose(float(r[m]),float(selected[key][m]),atol=1e-12,rtol=1e-12) for m in ('mse','mae')):
            raise ValueError('SHARED_BASELINE_METRIC_DUPLICATION_MISMATCH')
        selected[key]=r
    return list(selected.values())


def figures(t,folder,plt,sequence_verified,cost_status):
    rows=final_scores(t['scores']);methods=('RAW_BOLT','ORIGINAL_MAIN')+HEADS
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for di,d in enumerate(DATASETS):
        for mi,m in enumerate(('mse','mae')):
            ax=axes[di,mi]
            for i,method in enumerate(methods):
                values=[float(r[m]) for r in rows if r['dataset']==d and r['method']==method]
                if values:ax.bar(i,np.mean(values),color=COLOR[method],alpha=.7);ax.scatter(np.linspace(i-.08,i+.08,len(values)),values,c='black',s=20)
                else:ax.text(i,.1,'미실행',ha='center')
            ax.set_xticks(range(4),['원 RAW','원 MAIN','RF1','RF3']);ax.set_title(common.DISPLAY[d]+' · '+m.upper());ax.set_ylabel('normalized channel-macro '+m.upper())
    common.finish_figure(fig,FIGURES[0],folder,plt,'seed metrics 평균+개별 점; 반복 VAL/노출 EVAL. 독립 검산 gate='+str(sequence_verified)+'; 미통과면 producer 수치이며 판단 보류.',bottom=.075)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for di,d in enumerate(DATASETS):
        for hi,h in enumerate(HEADS):
            ax=axes[di,hi]
            for si,s in enumerate(SEEDS):
                values=[r for r in t['checkpoints'] if r['dataset']==d and r['seed']==s and r['head']==h]
                if values:
                    ax.plot([r['new_update'] for r in values],[r['actual_Y_TUNE_mse'] for r in values],marker='o',linestyle='-' if si==0 else '--',label=str(s),color=COLOR[h])
                    for r in values:
                        if r['selected']:ax.scatter(r['new_update'],r['actual_Y_TUNE_mse'],marker='*',s=150,c=COLOR[h],edgecolor='black')
            ax.set_title(common.DISPLAY[d]+' / '+h);ax.set_xlabel('새 update');ax.set_ylabel('actual-Y TUNE MSE');ax.legend()
    common.finish_figure(fig,FIGURES[1],folder,plt,'0/100/200/400/600, earliest TUNE minimum. 부분 fit은 존재하는 checkpoint만 표시; 선택0은 새 성능 개선이 아님.',bottom=.075)
    fig,axes=plt.subplots(2,2,figsize=(14,8))
    for di,d in enumerate(DATASETS):
        learning=[r for r in t['learning'] if r['dataset']==d];labels=[r['head'].replace('STATIC_','').replace('TEMPORAL_','')+'\n'+str(r['seed']) for r in learning]
        x=np.arange(len(learning));ax=axes[di,0]
        for gi,g in enumerate(('E','D','Head')):ax.bar(x+(gi-1)*.22,[r[g+'_changed_tensors'] for r in learning],.21,label=g)
        ax.set_xticks(x,labels);ax.set_title(common.DISPLAY[d]+' · u0→u600 변경 증거');ax.set_ylabel('parameter tensor 수');ax.legend()
        ax=axes[di,1]
        for i,r in enumerate(learning):
            values=[v['F_input_gradient_norm'] for v in t['gradients'] if v['dataset']==d and v['seed']==r['seed'] and v['head']==r['head']]
            ax.scatter(np.repeat(i,len(values)),values,c=COLOR[r['head']],s=20)
            if values:ax.text(i,max(values)*1.4,'선택 u'+str(r['selected_new_update']),ha='center',fontsize=8)
        if learning:
            ax.set_yscale('log');lower,upper=ax.get_ylim();maximum=max(v['F_input_gradient_norm'] for v in t['gradients'] if v['dataset']==d);ax.set_ylim(lower,max(upper,maximum*2))
            from matplotlib.ticker import FuncFormatter,NullFormatter
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.0e}'));ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(x,labels);ax.set_ylabel('동결 F branch input gradient norm');ax.set_title('F weight 고정 · 입력 gradient 전달')
    common.finish_figure(fig,FIGURES[2],folder,plt,'실제600 proof가 있는 완료 조건만. E/D도 함께 갱신했으므로 head만의 인과효과를 분리한 비교가 아님.',bottom=.075)
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    for di,d in enumerate(DATASETS):
        for bi,b in enumerate((1,8)):
            ax=axes[di,bi];cost=[r for r in t['whole_cost'] if r['dataset']==d and r['batch']==b]
            if not cost:ax.text(.5,.5,'실제 비용 NOT_MEASURED\n'+cost_status['reason'],ha='center',va='center',transform=ax.transAxes)
            for i,r in enumerate(cost):
                ratios=[v for v in json.loads(r['paired_RAW_ratios']) if v is not None]
                if ratios:ax.scatter(np.repeat(i,len(ratios)),ratios,s=20,c=COLOR.get(r['method'],'#777777'));ax.plot([i-.16,i+.16],[np.median(ratios)]*2,c='black')
            ax.axhline(1,c='black',linewidth=.6);ax.set_xticks(range(len(cost)),[r['route'].replace('_s','\n') for r in cost],rotation=20,fontsize=8)
            ax.set_title(common.DISPLAY[d]+f' / batch{b}');ax.set_ylabel('같은 block RAW 대비 whole latency ratio')
    common.finish_figure(fig,FIGURES[3],folder,plt,'각7 route가 각 위치1회: cyclic Latin square; carryover 쌍/thermal 원인 제거 아님. 점=block, 선=중앙값, 신뢰구간 아님.',bottom=.10)
    fig,axes=plt.subplots(2,2,figsize=(15,9));stages=('fill','normalization','H2D','E','F','oldD','head','anchor_inverse','D2H')
    for di,d in enumerate(DATASETS):
        for bi,b in enumerate((1,8)):
            ax=axes[di,bi];values=[r for r in t['stages'] if r['dataset']==d and r['batch']==b]
            routes=list(dict.fromkeys(r['route'] for r in values));left=np.zeros(len(routes))
            if not values:ax.text(.5,.5,'Stage NOT_MEASURED\n'+cost_status['stage_reason'],ha='center',va='center',transform=ax.transAxes)
            for stage in stages:
                widths=[sum(float(r['median_ms_with_sync']) for r in values if r['route']==route and r['stage']==stage)/b for route in routes]
                ax.barh(range(len(routes)),widths,left=left,label=stage);left+=np.asarray(widths)
            ax.set_yticks(range(len(routes)),[r.replace('_s','\n') for r in routes],fontsize=8);ax.invert_yaxis();ax.set_title(common.DISPLAY[d]+f' / batch{b}');ax.set_xlabel('별도 sync stage median 합 / window (ms)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=9,loc='lower center',bbox_to_anchor=(.5,.045),fontsize=8)
    common.finish_figure(fig,FIGURES[4],folder,plt,'전체 균형 trials 뒤 별도로 측정. stage sum ≠ 실제 whole latency; 원 RAW/MAIN head는 host identity+sync overhead.',bottom=.12)
    prior.figure_app_or_budget({'app_scores':t['APP_context'],'channel_budget':[]},folder,plt)
    for suffix in ('png','svg'):
        oldfile=folder/(prior.FIGURES[5]+'.'+suffix);newfile=folder/(FIGURES[5]+'.'+suffix);oldfile.rename(newfile)


def markdown(t,sequence_gate,cost_status,prior_link,memo_link):
    rows=final_scores(t['scores']);lines=['# RF1/RF3 예측 시퀀스 복원의 실제 후속 결과','',
        '과학검산 gate: '+str(sequence_gate['passed'])+'. 모델8개를 실행한 것과 성능 개선·독립 일반화를 확인한 것은 다르다. 아래 평가는 반복 개발 Robin VAL30·이미 노출된 Beijing EVAL146·seed2개이며 추가 holdout을 만들지 않았다.','',
        '## 1. 시간 정보를 보는 residual decoder가 성능 격차를 닫았나?','', '```text','자료                   방법          n  normalized MSE / MAE']
    for d in DATASETS:
        for method in ('RAW_BOLT','ORIGINAL_MAIN')+HEADS:
            selected=[r for r in rows if r['dataset']==d and r['method']==method]
            if selected:lines.append(f'{d:22} {method:15} {len(selected)}  {np.mean([r["mse"] for r in selected]):.9f} / {np.mean([r["mae"] for r in selected]):.9f}')
            else:lines.append(f'{d:22} {method:15} 0  NOT_EXECUTED_OR_NO_SELECTED_OUTPUT')
    lines+=['```','각 seed와 모든 split은 [점수 표](tables/scores.csv)에 보존한다. 검산 gate 미통과 시 이 값은 producer 기록이며 성능 판단을 보류한다. RAW의 n=2는 같은 저장 RAW 예측을 두 seed 비교에 재사용한 것이며 독립 RAW 학습·추론 2회를 뜻하지 않는다.','',
        '동일 seed의 RF3−대조군 차이(음수=낮은 오차):','', '```text','자료                   대조군          n   ΔMSE / ΔMAE']
    for d in DATASETS:
        for reference in ('STATIC_RF1','RAW_BOLT','ORIGINAL_MAIN'):
            paired=[r for r in t['paired_differences'] if r['dataset']==d and r['reference']==reference]
            if paired:lines.append(f'{d:22} {reference:15} {len(paired)}   {np.mean([r["mse_difference"] for r in paired]):+.9f} / {np.mean([r["mae_difference"] for r in paired]):+.9f}')
    lines+=['```']
    for d in DATASETS:
        paired=[r for r in t['paired_differences'] if r['dataset']==d and r['reference']=='RAW_BOLT']
        if paired:
            delta=float(np.mean([r['mse_difference'] for r in paired]));mae_delta=float(np.mean([r['mae_difference'] for r in paired]))
            verdict='평균 MSE에서 RAW 격차를 닫지 못했다' if delta>=0 else '평균 MSE가 RAW보다 낮았으나 반복 개발 split 결과다'
            lines.append(f'{common.DISPLAY[d]}: {verdict}. 같은 seed {len(paired)}개 ΔMSE={delta:+.9f}, ΔMAE={mae_delta:+.9f}.')
    lines+=['','seed2개 차이로 일반화 불확실성을 추정하기 어렵다. RF3가 RF1 또는 RAW보다 낮더라도 반복 개발 split 개선으로 범위를 제한한다. 다음 판단은 두 seed의 방향과 MSE/MAE tradeoff 및 비용을 함께 확인하는 것이다.','',
        '## 2. 학습을 못 했나, 선택된 상태가 개선을 못 했나?','',
        '완료된 fit의 실제600 update·E/D/head 변화·동결 F branch gradient를 확인한다. final600와 selected checkpoint는 다르며 선택0은 원 MAIN 출력 유지다. 같은 dataset/seed 두 head는 동일 원 CP와 batch order를 사용하지만 E/D도 함께 학습하므로 temporal head의 순수 인과효과로 읽지 않는다. 새 head 등록 capacity 차이는 약2% 이내이고 Beijing의 output-bias 제외 active capacity 차이는 약2.06%이다.','',
        f'완료600 학습 proof가 있는 조건={len(t["learning"])}/8, 선택0={sum(bool(r["selected0"]) for r in t["learning"])}/8. 학습이 실행됐다는 증거와 예측 적합성 개선 증거를 분리한다. TUNE 선택은 독립 EVAL 개선을 보장하지 않는다. [모든 선택 checkpoint](tables/checkpoints.csv)를 원 MAIN과 함께 남겼다.','',
        '## 3. RF3 비교 자체에 어떤 구조적 한계가 있나?','',
        '과거 prefix와 예측 future에 적용한 R에서, 같은 과거 prefix와 마지막 latent를 반복한 R을 뺀다. future latent가 마지막 값으로 지속하면 residual은 강제로0이다. 그러나 실제 지연 채널은 latent가 지속해도 첫 미래 시점에서 갱신될 수 있다. 이 반례는 reference residual의 조건부 제약이며, 전체 E가 지연 정보를 K에 표현할 수 없다는 증거도 실제 실패 원인 확정도 아니다. RF3의 실패를 모든 temporal decoder의 실패로 일반화하거나 성공을 지연 원인 해결의 증거로 쓰지 않는다. '+memo_link,'',
        ('저장 latent의 flat 빈도는 [24개 split 진단](tables/flat_latents.csv)에 기록했다. threshold는 encoder 간 정규화하지 않은 absolute latent 1e-5이며 E/F 독립 replay·원인 검증이 아니다.' if t['flat_latents'] else '실제 flat latent 빈도: NOT_MEASURED. 합성 반례가 실제로 빈번한 조건인지 추정하지 않는다.'),'',
        '## 4. 같은 순서 조건에서 batch1 효율은 해결됐나?','',
        'whole cost 상태: '+cost_status['status']+' / '+cost_status['reason']+'. stage 상태: '+cost_status['stage_status']+' / '+cost_status['stage_reason']+'.','',
        '7경로 위치를 균형화하고 같은 block RAW와 비교한다. RAW absolute latency와 원 MAIN·선택 head의 whole 시간은 [비용 표](tables/whole_cost.csv)에 보존한다. 이전 Robin RAW가24.87ms에서48.58ms로 달라진 기준 변동을 숨기지 않는다. 위치 균형만으로 clock/thermal/carryover 원인을 배제하지 못하며 batch8 처리량을 batch1 latency로 바꾸어 말하지 않는다. 별도 synchronized stage sum은 whole latency가 아니다.','',
        '## 5. APP에서는 강한 단순 RAW+R 대안까지 이겼나?','',
        '이번 새8 fit에 APP를 추가한 것이 아니다. '+prior_link+'의 실제 APP6 MSE/MAE/Huber_delta1 및 RAW/RAW+R/원 LEARNED 결과를 [APP 맥락 표](tables/APP_context.csv)와 마지막 그림에 복사했다. 강한 RAW+R accuracy reference를 유지하며 그 배포 비용은 NOT_MEASURED다. MSE 개선과 MAE 악화, RAW-error-fixed body/tail의 결과를 함께 읽는다.','',
        '## 6. 무엇이 남았고 Chronos-Bolt를 유지할 근거는 무엇인가?','',
        'RAW Chronos-Bolt가 원 MAIN보다 강했던 근거는 남지만, 이것이 압축된 latent에서의 예측 적합성이나 경량화를 입증하지 않는다. E가 보존한 정보·F의 latent 미래 예측·현재 reference를 가진 복원 함수의 제약을 분리해 판단해야 한다. 새 비교의 양·음성 결과와 비용으로 다음 가설을 좁히며, 대조학습·dynamic graph·큰 decoder가 필요하다고 자동 결론 내리지 않는다.','',
        '독립 CPU audit는 저장 latent와 CP로 D/head 산술·관측 mask 기반 점수·selection·SHA를 재구성했다. F/past E 재추론과 매 update forward를 독립 replay했다고 하지 않는다. CP/NPZ는 ignored cache literal path+SHA만 보존하고 존재하지 않는 GitHub 링크를 만들지 않는다.','',
        '## 실제 완료·미완료 상태','', '```text','자료                   seed   head            status                 updates selected']
    for r in t['status']:lines.append(f'{r["dataset"]:22} {r["seed"]}  {r["head"]:15} {r["status"]:23} {r["actual_updates"]} {r.get("selected_new_update")}')
    lines+=['```','[실패·재개 정보](tables/status.csv)에는 원수치와 미완료 이유를 남긴다. 같은 tag/CP에서 자동 재개하지 않으며 추가 자원이 필요하면 승인 범위와 새 실행 조건부터 확인한다.','']
    for name in FIGURES:lines.extend([f'![{name}](figures/{name}.png)','',f'[SVG](figures/{name}.svg)',''])
    lines+=['[입력 SHA 목록](provenance/input_manifest.csv) · [checkpoint](tables/checkpoints.csv) · [학습 proof](tables/learning.csv) · [별도 stages](tables/stages.csv)','']
    return '\n'.join(lines)


def generate(args):
    if not args.no_contention_attested:raise ValueError('ROOT_LIVE_FITS_AUDITS_COST_JOBS_ENDED_ATTESTATION_REQUIRED')
    output=HERE/('report_'+args.tag);output.mkdir(exist_ok=False);limit=None;inputs=[];started=time.perf_counter()
    result={'status':'RUNNING','whole_goal_complete':False,'GPU_execution':False,'scientific_NPZ_or_CP_opened':False,'independent_holdout':False}
    import continuation_runtime_v2 as runtime
    try:
        limit=runtime.ContinuationLimit('report_'+args.tag,gpu=False)
        import prepare_inputs
        prepare_inputs.install_protected_test_guard()
        for folder in ('tables','figures','provenance','provenance/producers','provenance/audits'):(output/folder).mkdir(exist_ok=False)
        queue,qr=common.checked_json(args.sequence_queue);refs=producer_refs(queue)
        prior.copy_input(qr,output,'SEQUENCE_QUEUE.json',inputs);producers=[];records=[]
        for i,r in enumerate(refs):
            p,checked=common.checked_json(r['path'],r['sha256']);producers.append(p);records.append(r)
            prior.copy_input(checked,output,f'producers/sequence_{i+1:02d}.json',inputs)
        t=collect_sequence(producers,records);t['paired_differences']=paired_differences(t['scores']);t['flat_latents']=[]
        audit,ar=common.checked_json(args.sequence_audit);gate=audit_gate(audit,producers,records)
        if not gate['passed'] and not args.allow_incomplete:raise ValueError('PASSED_BOUND_SEQUENCE_AUDIT_REQUIRED_OR_EXPLICIT_HONEST_PARTIAL_MODE')
        prior.copy_input(ar,output,'audits/SEQUENCE.json',inputs)
        cost_status={'status':'NOT_MEASURED','reason':args.cost_not_measured_reason or 'no actual cost producer supplied',
            'stage_status':'NOT_MEASURED','stage_reason':args.cost_not_measured_reason or 'no actual stages supplied'};t['whole_cost']=[];t['stages']=[]
        if args.balanced_cost:
            cost,cr=common.checked_json(args.balanced_cost);cost_binding(cost,records);t['whole_cost'],t['stages']=collect_cost(cost)
            prior.copy_input(cr,output,'producers/BALANCED_COST.json',inputs)
            cost_status.update(status=cost['status'],reason=cost.get('failure',{}).get('message','actual producer'),
                stage_status=cost.get('stage_status','NOT_MEASURED'),stage_reason=cost.get('stage_reason') or 'separate protocol')
            if args.cost_audit:
                ca,car=common.checked_json(args.cost_audit);prior.copy_input(car,output,'audits/BALANCED_COST.json',inputs)
                if ca.get('cost_producer',{}).get('sha256')!=cr['sha256']:raise ValueError('COST_AUDIT_PRODUCER_SHA_BINDING_REQUIRED')
                cost_status['audit_passed']=str(ca.get('status','')).startswith('PASS') and ca.get('failed_count')==0
            if cost.get('status')=='COMPLETE' and (len(t['whole_cost'])!=28 or len(cost.get('stage_rows',[]))!=28 or not cost_status.get('audit_passed')):
                raise ValueError('COMPLETE_COST_REQUIRES28_WHOLE28_STAGE_AND_BOUND_AUDIT')
        if args.flat_diagnostic:
            fd,fdr=common.checked_json(args.flat_diagnostic);t['flat_latents']=flat_binding(fd,records)
            prior.copy_input(fdr,output,'audits/FLAT_LATENTS.json',inputs)
            source=common.resolve(fd['csv']['path'])
            if common.sha(source)!=fd['csv']['sha256']:raise ValueError('FLAT_DIAGNOSTIC_CSV_SHA_MISMATCH')
            prior.copy_input(fd['csv'],output,'FLAT_LATENTS.csv',inputs)
        context=common.resolve(args.prior_report);manifest,mr=common.checked_json(context/'MANIFEST.json')
        if manifest.get('generator_result_status')!='COMPLETE':raise ValueError('ACTUAL_PRIOR_REPORT_COMPLETE_REQUIRED')
        prior.copy_input(mr,output,'PRIOR_MANIFEST.json',inputs)
        for name in ('app_scores.csv','simple_tail.csv','joint_scores.csv','joint_learning.csv'):
            rec=next(r for r in manifest['artifacts'] if r['path']=='tables/'+name);source=context/'tables'/name
            if common.sha(source)!=rec['sha256']:raise ValueError('PRIOR_NUMERIC_TABLE_SHA_MISMATCH')
            destination=output/'tables'/('prior_'+name);shutil.copyfile(source,destination)
            inputs.append({'path':str(source),'sha256':rec['sha256'],'copied_to':'tables/prior_'+name})
        t['APP_context']=common.csv_rows(output/'tables/prior_app_scores.csv')
        if args.counterexample_memo:
            memo=common.resolve(args.counterexample_memo);prior.copy_input({'path':str(memo),'sha256':common.sha(memo)},output,'TEMPORAL_REFERENCE_COUNTEREXAMPLE_KO.md',inputs)
            memo_link='[반례·적용 범위 메모](provenance/TEMPORAL_REFERENCE_COUNTEREXAMPLE_KO.md)'
        else:memo_link='별도 반례 메모 미제공; 수학적 제약과 실제 원인 미확정을 구분한다.'
        for path in (Path(__file__),HERE/'make_followup_report.py',HERE/'make_joint_followup_report_v4.py',HERE/'continuation_runtime_v1.py',HERE/'continuation_runtime_v2.py',HERE/'fast_storage_scan_v1.py'):
            prior.copy_input({'path':str(path),'sha256':common.sha(path)},output,'source_'+path.name,inputs)
        for name,record in (('AUTHORIZATION.json',limit.authorization_record),('RUN_MANIFEST.json',limit.run_manifest_record)):prior.copy_input(record,output,name,inputs)
        for name,rows in t.items():
            if rows:common.save_csv(output/'tables'/(name+'.csv'),rows)
            else:
                with (output/'tables'/(name+'.csv')).open('x',encoding='utf-8') as f:f.write('status,reason\nNOT_MEASURED,No actual rows\n')
        common.save_csv(output/'provenance/input_manifest.csv',inputs);limit.check();plt,font=common.configure_plotting()
        figures(t,output/'figures',plt,gate['passed'],cost_status);limit.check()
        text=markdown(t,gate,cost_status,'[이전 joint·APP 실제 보고서](../'+context.name+'/REPORT_KO.md)',memo_link)
        text+='\n검산 원문: '+' · '.join(f'[{p.name}](provenance/audits/{p.name})' for p in sorted((output/'provenance/audits').glob('*.json')))+'\n'
        text+='\n[사용자 자원 승인](provenance/AUTHORIZATION.json) · [고정 실행 manifest](provenance/RUN_MANIFEST.json)\n'
        text,links=prior.checked_links(text,output)
        with (output/'REPORT_KO.md').open('x',encoding='utf-8') as f:f.write(text)
        for r in inputs:
            if common.sha(r['path'])!=r['sha256'] or common.sha(output/r['copied_to'])!=r['sha256']:raise ValueError('ACTUAL_INPUT_OR_COPY_CHANGED_DURING_REPORT')
        result.update(status='COMPLETE',report_generation_complete=True,sequence_scientific_gate=gate,cost_status=cost_status,
            figures_count=6,font=font,tables_rows={k:len(v) for k,v in t.items()},relative_links_checked=links,
            scientific_all8_complete=sum(r['status']=='COMPLETE' for r in t['status'])==8,
            source_sha256=common.sha(__file__),report_body_wall_seconds=time.perf_counter()-started)
        limit.check()
    except Exception as error:
        result.update(status='FAILED_PRESERVED',error={'type':type(error).__name__,'message':str(error)});raise
    finally:
        receipt=limit.close(result['status'],result.get('error')) if limit else getattr(runtime,'LAST_CONSTRUCTOR_TERMINAL',None)
        if receipt:
            result['shared_resource_receipt']=receipt
            if receipt.get('terminal_status',result['status'])!='COMPLETE':result.update(status=receipt.get('terminal_status',result['status']),error=receipt.get('terminal_error',result.get('error')))
        common.write(output/'RESULT.json',result)
        common.write(output/'MANIFEST.json',{'generator_result_status':result['status'],'artifacts':[
            {'path':str(p.relative_to(output)).replace('\\','/'),'sha256':common.sha(p),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()]})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe',action='store_true');mode.add_argument('--generate',action='store_true')
    for name in ('tag','sequence-queue','sequence-audit','balanced-cost','cost-audit','cost-not-measured-reason','prior-report','counterexample-memo','flat-diagnostic'):parser.add_argument('--'+name)
    parser.add_argument('--allow-incomplete',action='store_true');parser.add_argument('--no-contention-attested',action='store_true');args=parser.parse_args()
    if args.describe:print(json.dumps({'status':'SOURCE_ONLY_NOT_EXECUTED','actual_GPU_or_NPZ_CP_reads':False,'figures':6,'cost_whole_and_stages_separate':True,'honest_partial_mode_explicit':True}));return
    if not all((args.tag,args.sequence_queue,args.sequence_audit,args.prior_report,args.no_contention_attested)) or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.tag):parser.error('generate needs terminal queue/audit/prior actual report/fresh tag/no-contention attestation')
    if not args.balanced_cost and not args.cost_not_measured_reason:parser.error('missing actual cost needs explicit NOT_MEASURED reason')
    result=generate(args);print(json.dumps({'status':result['status'],'figures':result.get('figures_count'),'whole_goal_complete':False}))
    if result['status']!='COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()
