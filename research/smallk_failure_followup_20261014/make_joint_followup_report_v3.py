"""Source-only by default; NEW v2 terminal joint/APP/cost report adapter.

No NPZ/checkpoint/model reads, GPU inference, or scientific execution on import.
Actual --generate is held until root confirms all training/cost jobs have ended.
"""
from pathlib import Path
import argparse
import json
import re
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:sys.path.insert(0, str(HERE))
import make_followup_report as old
import profile_joint_cost_v1 as contract

DATASETS = contract.DATASETS
SEEDS = contract.SEEDS
OBJECTIVES = contract.OBJECTIVES
COLOR = {'RAW_BOLT':'#333333','OLD_MAIN':'#777777','Y':'#0072B2','RAW':'#009E73','MIXED':'#CC79A7',
         'MSE':'#0072B2','MAE':'#D55E00','HUBER_DELTA1':'#009E73','RAW_PLUS_R':'#E69F00'}
FINAL = {'robin':'val','beijing_aotizhongxin':'eval'}
FIGURES = ('01_joint_actual_scores','02_actual_Y_TUNE_selection','03_selected_whole_cost',
           '04_appliances_RAW_fixed_body_tail','05_learning_and_selection_proof','06_APP_or_channel_menu')


def passed_audit(value):
    if not str(value.get('status','')).startswith('PASS') or value.get('failed_count') != 0:
        raise ValueError('ACTUAL_PASSED_INDEPENDENT_AUDIT_REQUIRED')


def joint_tables(producers):
    contract.validate_joint_producers(producers)
    tables = {'joint_scores':[],'joint_checkpoints':[],'joint_curve':[],'joint_learning':[],'joint_gradients':[],'resources':[]}
    for p in producers:
        base = {'dataset':p['dataset'],'seed':p['seed'],'objective':p['objective']}
        for split, selected in p['selected_inference'].items():
            for method, score in ((p['objective'],selected['scores']),('OLD_MAIN',p['original_MAIN_scores'][split]),('RAW_BOLT',p['original_RAW_scores'][split])):
                tables['joint_scores'].append({**base,'split':split,'method':method,
                    **score['normalized_actual_Y'],'selected_new_update':p['selected_new_update'],
                    'selected_is_initial_original_MAIN':p['selected_is_initial_original_MAIN']})
        for row in p['checkpoints']:
            tables['joint_checkpoints'].append({**base,'new_update':row['new_update'],
                'actual_Y_TUNE_mse':row['tune_actual_Y_mse'],'selected':row['new_update']==p['selected_new_update'],
                'checkpoint_path':row['checkpoint']['path'],'checkpoint_sha256':row['checkpoint']['sha256']})
        tables['joint_curve'].extend({**base,**row} for row in p['curve'])
        proof = p['learning_executed']; finite = p['FIT_target_finite_contract']
        tables['joint_learning'].append({**base,**proof,'E_changed_tensors':p['new_update_parameter_proof']['E']['changed_parameter_tensors'],
            'D_changed_tensors':p['new_update_parameter_proof']['D']['changed_parameter_tensors'],
            'selected_new_update':p['selected_new_update'],'selected0':p['selected_is_initial_original_MAIN'],
            'FIT_unobserved_nonfinite_count':finite['actual_Y_unobserved_nonfinite_count'],
            'training_wall_seconds':p['training_wall_seconds'],'runner_wall_seconds':p['runner_wall_seconds'],
            'guard_full_check_wall_seconds':p['guard_cadence']['explicit_wall_seconds']['full'],
            'guard_fast_base_wall_seconds':p['guard_cadence']['explicit_wall_seconds']['fast_base'],
            'historical_teacher_generation_seconds':p['cached_teacher_cost']['historical_teacher_generation_compute_wall_seconds']})
        tables['joint_gradients'].extend({**base,'new_update':row['new_update'],
            'F_input_gradient_norm':row['frozen_F_input_gradient_norm'],'F_input_gradient_finite':row['frozen_F_input_gradient_finite']}
            for row in p['gradient_diagnostics'])
        receipt = p.get('shared_resource_receipt',{})
        tables['resources'].append({**base,'role':'joint_training',**receipt})
    return tables


def cost_tables(value):
    if value.get('status') != 'COMPLETE' or value.get('actual_complete_rows') != 28:
        raise ValueError('ACTUAL_COST28_TERMINAL_COMPLETE_REQUIRED')
    expected = {(d,s,o,b) for d in DATASETS for s in SEEDS for o in OBJECTIVES for b in (1,8)}
    expected |= {(d,None,None,b) for d in DATASETS for b in (1,8)}
    rows = value.get('rows',[])
    actual = {(r.get('dataset'),r.get('seed'),r.get('objective'),r.get('batch')) for r in rows}
    if len(rows)!=28 or actual!=expected or value.get('F_before')!=value.get('F_after'):
        raise ValueError('COST28_COVERAGE_OR_F_PRESERVATION_MISMATCH')
    output=[]
    for row in rows:
        if (row['status']!='COMPLETE' or row['warmup']!=2 or row['repeats']!=7 or len(row['seconds'])!=7
                or len(row['parity'])!=7 or any(not r['matched_batch_normalized']['pass'] or not r['matched_batch_physical']['pass'] for r in row['parity'])):
            raise ValueError('COST28_REPEAT7_MATCHED_BATCH_GATE_REQUIRED')
        output.append({'dataset':row['dataset'],'seed':row['seed'],'objective':row['objective'],'method':row['method'],
            'batch':row['batch'],'selected_new_update':row['selected_new_update'],'selected0':row['selected_is_original_MAIN'],
            'median_ms':row['median_seconds']*1000,'median_ms_per_window':row['median_seconds_per_window']*1000,
            'min_ms_per_window':min(row['seconds'])*1000/row['batch'],'max_ms_per_window':max(row['seconds'])*1000/row['batch'],
            'seconds':json.dumps(row['seconds']),'original_bulk_normalized_failed_repeats':sum(not p['original_bulk_normalized']['pass'] for p in row['parity']),
            'original_bulk_physical_failed_repeats':sum(not p['original_bulk_physical']['pass'] for p in row['parity']),
            'timed_output_path':row['timed_outputs']['path'],'timed_output_sha256':row['timed_outputs']['sha256'],
            'memory':json.dumps(row['memory']),'whole_preprocessing':row['preprocessing']})
    return output


def app_tables(producers, missing_reason):
    if len(producers)>6:raise ValueError('APP_AT_MOST6_ACTUAL_PRODUCERS')
    expected={(s,o) for s in SEEDS for o in ('MSE','MAE','HUBER_DELTA1')}
    seen=set();rows=[];statuses=[]
    for p in producers:
        key=(p.get('seed'),p.get('objective'))
        if key not in expected or key in seen:raise ValueError('APP_DUPLICATE_OR_UNEXPECTED_CONDITION')
        seen.add(key);status=p.get('status')
        if status not in {'COMPLETE','FAILED_PRESERVED','BUDGET_HELD_PRESERVED'}:raise ValueError('APP_TERMINAL_STATUS_REQUIRED')
        statuses.append({'seed':key[0],'objective':key[1],'status':status,'reason':p.get('failure')})
        if status=='COMPLETE':
            proof=p['learning_executed']
            if proof.get('optimizer_updates')!=600 or proof.get('full_learning_proof_pass') is not True:
                raise ValueError('APP_ACTUAL600_PROOF_REQUIRED')
            for split,selected in p['selected_inference'].items():
                for method,score in ((p['objective'],selected['scores']),('OLD_LEARNED',p['original_MAIN_scores'][split]),('RAW_BOLT',p['original_RAW_scores'][split]),('RAW_PLUS_R',p['original_RAW_PLUS_R_scores'][split])):
                    rows.append({'dataset':p['dataset'],'seed':p['seed'],'objective':p['objective'],'split':split,
                        'method':method,**score['normalized_actual_Y'],'selected_new_update':p['selected_new_update']})
    missing=expected-seen
    if missing and not missing_reason:raise ValueError('APP_NOT_EXECUTED_REASONS_REQUIRED')
    statuses.extend({'seed':s,'objective':o,'status':'NOT_EXECUTED','reason':missing_reason} for s,o in sorted(missing))
    return rows,statuses


def app_cost_tables(value):
    from profile_appliances_cost_v1 import SEEDS as app_seeds, OBJECTIVES as app_objectives
    expected={(s,o,b) for s in app_seeds for o in app_objectives for b in (1,8)}|{(None,None,b) for b in (1,8)}
    rows=value.get('rows',[])
    if (value.get('status')!='COMPLETE' or value.get('actual_complete_rows')!=14 or len(rows)!=14
            or {(r.get('seed'),r.get('objective'),r.get('batch')) for r in rows}!=expected
            or value.get('F_before')!=value.get('F_after') or value.get('RAW_PLUS_R_cost_measured') is not False):raise ValueError('APP_ACTUAL_COST14_COMPLETE_REQUIRED')
    output=[]
    for r in rows:
        if (r['status']!='COMPLETE' or r['warmup']!=2 or r['repeats']!=7 or len(r['seconds'])!=7 or len(r['parity'])!=7
                or any(not p['matched_batch_normalized']['pass'] or not p['matched_batch_physical']['pass'] for p in r['parity'])):raise ValueError('APP_COST14_MATCHED_BATCH_REPEAT7_REQUIRED')
        output.append({'dataset':'appliances','seed':r['seed'],'objective':r['objective'],'method':r['method'],'batch':r['batch'],
            'selected_new_update':r['selected_new_update'],'selected0':r['selected_is_original_MAIN'],
            'median_ms':r['median_seconds']*1000,'median_ms_per_window':r['median_seconds_per_window']*1000,
            'min_ms_per_window':min(r['seconds'])*1000/r['batch'],'max_ms_per_window':max(r['seconds'])*1000/r['batch'],
            'seconds':json.dumps(r['seconds']),'timed_output_path':r['timed_outputs']['path'],'timed_output_sha256':r['timed_outputs']['sha256'],
            'original_bulk_normalized_failed_repeats':sum(not p['original_bulk_normalized']['pass'] for p in r['parity']),
            'original_bulk_physical_failed_repeats':sum(not p['original_bulk_physical']['pass'] for p in r['parity']),
            'memory':json.dumps(r['memory']),'RAW_PLUS_R_cost':'NOT_MEASURED'})
    return output


def metric_groups(rows, split_final=True):
    groups={}
    for row in rows:
        if split_final and row['split']!=FINAL.get(row['dataset'],'eval'):continue
        key=(row['dataset'],row['method'],row['seed'])
        if key in groups:
            if not all(np.isclose(float(groups[key][m]),float(row[m]),atol=1e-12,rtol=1e-12) for m in ('mse','mae')):
                raise ValueError('BASELINE_DUPLICATION_HAS_DIFFERENT_METRICS')
        else:groups[key]=row
    return list(groups.values())


def copy_input(record, output, name, inputs):
    destination=output/'provenance'/name
    shutil.copyfile(record['path'],destination)
    if old.sha(destination)!=record['sha256']:raise ValueError('COPIED_INPUT_SHA_MISMATCH')
    inputs.append({**record,'copied_to':str(destination.relative_to(output)).replace('\\','/')})


def checked_links(markdown, output):
    checked=[]
    for prefix,label,target in re.findall(r'(!?)\[([^\]]+)\]\(([^)]+)\)',markdown):
        if '://' in target or target.startswith('#'):continue
        path=(Path(output)/target).resolve()
        if (not path.is_relative_to(ROOT.resolve()) or '.cache' in path.parts
                or prefix and not path.is_relative_to(Path(output).resolve()) or not path.is_file()):
            raise ValueError('MISSING_OR_UNSAFE_GITHUB_RELATIVE_LINK:'+target)
        checked.append(target)
    return markdown,checked


def figure_scores(t,folder,plt):
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    rows=metric_groups(t['joint_scores'])
    methods=('RAW_BOLT','OLD_MAIN','Y','RAW','MIXED')
    for di,d in enumerate(DATASETS):
        for mi,m in enumerate(('mse','mae')):
            ax=axes[di,mi]
            for x,method in enumerate(methods):
                vals=[float(r[m]) for r in rows if r['dataset']==d and r['method']==method]
                ax.bar(x,np.mean(vals),color=COLOR[method],alpha=.7)
                ax.scatter(np.array([x-.09,x+.09]),vals,color='#111111',s=22,zorder=3)
            ax.set_xticks(range(5),['원 RAW','원 MAIN','joint Y','joint RAW','joint MIXED'])
            ax.set_title(old.DISPLAY[d]+' · '+m.upper());ax.set_ylabel('normalized channel-macro '+m.upper())
    old.finish_figure(fig,FIGURES[0],folder,plt,'막대=seed별 metric 평균, 점=seed92601/92602. 같은 원 CP에서600학습 후 actual-Y TUNE 선택; 반복 VAL/노출 EVAL.',bottom=.075)


def figure_selection(t,folder,plt):
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True)
    for di,d in enumerate(DATASETS):
        for oi,o in enumerate(OBJECTIVES):
            ax=axes[di,oi]
            for si,s in enumerate(SEEDS):
                rows=[r for r in t['joint_checkpoints'] if r['dataset']==d and r['objective']==o and r['seed']==s]
                ax.plot([r['new_update'] for r in rows],[r['actual_Y_TUNE_mse'] for r in rows],marker='o',label=str(s),color=COLOR[o],linestyle='-' if si==0 else '--')
                selected=next(r for r in rows if r['selected']);ax.scatter(selected['new_update'],selected['actual_Y_TUNE_mse'],marker='*',s=180,edgecolor='black',color=COLOR[o],zorder=4)
            ax.set_title(old.DISPLAY[d]+' / '+o);ax.set_ylabel('actual-Y TUNE MSE');ax.set_xlabel('새 optimizer update');ax.legend()
    old.finish_figure(fig,FIGURES[1],folder,plt,'0/100/200/400/600 모두 보존. 별=최소 actual-Y TUNE MSE(동률이면 앞 checkpoint); 선택0은 새 학습 이득이 아님.',bottom=.065)


def figure_cost(t,folder,plt):
    datasets=DATASETS+(('appliances',) if t.get('app_cost') else ())
    allrows=t['joint_cost']+t.get('app_cost',[])
    fig,axes=plt.subplots(len(datasets),2,figsize=(15,4.4*len(datasets)))
    for di,d in enumerate(datasets):
        for bi,b in enumerate((1,8)):
            rows=[r for r in allrows if r['dataset']==d and r['batch']==b]
            ax=axes[di,bi]
            labels=['RAW 공유']+[str(r['objective'])+'\n'+str(r['seed'])+' / u'+str(r['selected_new_update']) for r in rows[1:]]
            for x,r in enumerate(rows):
                v=r['median_ms_per_window'];ax.bar(x,v,color=COLOR.get(r['objective'],'#333333'),alpha=.7)
                ax.vlines(x,r['min_ms_per_window'],r['max_ms_per_window'],color='black',linewidth=1)
            ax.set_xticks(range(len(rows)),labels,fontsize=9);ax.set_title(old.DISPLAY.get(d,'Appliances C26/K6 · 노출 EVAL')+f' · batch{b}')
            ax.set_ylabel('실제 전체 path ms/window');ax.grid(axis='y',alpha=.2)
    old.finish_figure(fig,FIGURES[2],folder,plt,f'{len(allrows)} 실제 행: warm2/repeat7 중앙값·최솟값-최댓값(신뢰구간 아님). 각 process F+모든 해당 codec resident. RAW+R cost 미측정.',bottom=.075)


def figure_tail(t,folder,plt):
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for mi,m in enumerate(('mse','mae')):
        ax=axes[mi]
        for ri,region in enumerate(('body','tail')):
            rows=[r for r in t['simple_tail'] if r['split']=='eval' and r['metric']==m and r['comparison']=='LEARNED_SEED_MEAN_vs_RAW' and r['region']==region]
            rows=sorted(rows,key=lambda r:float(r['quantile']))
            ax.bar(np.arange(3)+ri*.35-.175,[float(r['gap_contribution']) for r in rows],width=.33,label=region,color=('#0072B2','#D55E00')[ri])
        ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(range(3),['RAW q=.90','RAW q=.95','RAW q=.99'])
        ax.set_title('Appliances 노출 EVAL · '+m.upper());ax.set_ylabel('LEARNED - RAW signed contribution');ax.legend()
    old.finish_figure(fig,FIGURES[3],folder,plt,'같은 RAW |error| 경계로 영역 고정; 전체 split/채널 denominator 유지. 음수=오차 감소, 양수=증가. seed metric 평균.',bottom=.12)


def figure_learning(t,folder,plt):
    fig,axes=plt.subplots(2,2,figsize=(14,8))
    for di,d in enumerate(DATASETS):
        rows=[r for r in t['joint_learning'] if r['dataset']==d]
        labels=[r['objective']+'\n'+str(r['seed']) for r in rows];x=np.arange(6)
        ax=axes[di,0]
        ax.bar(x-.17,[r['E_changed_tensors'] for r in rows],.33,label='E changed tensors',color='#0072B2')
        ax.bar(x+.17,[r['D_changed_tensors'] for r in rows],.33,label='D changed tensors',color='#D55E00')
        ax.set_xticks(x,labels);ax.set_ylabel('u0→u600 바뀐 parameter tensor 수');ax.legend();ax.set_title(old.DISPLAY[d]+' · 실제600 proof')
        ax=axes[di,1]
        for i,r in enumerate(rows):
            values=[v['F_input_gradient_norm'] for v in t['joint_gradients'] if v['dataset']==d and v['seed']==r['seed'] and v['objective']==r['objective']]
            ax.scatter(np.repeat(i,len(values)),values,color=COLOR[r['objective']],s=20)
            ax.text(i,max(values)*1.4,'선택 u'+str(r['selected_new_update']),ha='center',fontsize=8)
        ax.set_yscale('log');ax.set_xticks(x,labels);ax.set_ylabel('동결 F branch input gradient norm');ax.set_title('F weight 동결 · input gradient는 전달')
    old.finish_figure(fig,FIGURES[4],folder,plt,'변경 증거는600 학습 완료를 확인하며 성능 개선을 뜻하지 않음. gradient는5개 사전 지정 진단 시점; 선택0과 final600 분리.',bottom=.07)


def figure_app_or_budget(t,folder,plt):
    complete=metric_groups(t['app_scores']) if t['app_scores'] else []
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    if complete:
        methods=[m for m in ('RAW_BOLT','RAW_PLUS_R','OLD_LEARNED','MSE','MAE','HUBER_DELTA1') if any(r['method']==m for r in complete)]
        for ax,m in zip(axes,('mse','mae')):
            for i,method in enumerate(methods):
                vals=[float(r[m]) for r in complete if r['method']==method]
                ax.bar(i,np.mean(vals),color=COLOR.get(method,'#777777'),alpha=.7);ax.scatter(np.repeat(i,len(vals)),vals,color='black',s=20)
            ax.set_xticks(range(len(methods)),methods,rotation=20);ax.set_ylabel('normalized '+m.upper());ax.set_title('APP 실제 완료 조건만 · '+m.upper())
        note='Appliances 실제 COMPLETE 결과. 모든 후보 actual-Y TUNE MSE 선택; q aux 공통 제거. RAW+R 강한 accuracy reference 유지(그 비용 미측정).'
    else:
        for ax,d in zip(axes,DATASETS):
            rows=[r for r in t['channel_budget'] if r['dataset']==d and r['split']==FINAL[d] and r['scale']=='normalized' and r['channel']=='macro']
            for i,r in enumerate(rows):ax.bar(i,float(r['mse']),color='#CC79A7' if r['method']=='FUTURE_ORACLE_TOPK' else '#777777')
            ax.set_xticks(range(len(rows)),[r['method'].replace('_','\n') for r in rows],fontsize=8);ax.set_ylabel('normalized MSE');ax.set_title(old.DISPLAY[d]+' · 제한 RAW/LAST 메뉴')
        note='APP 신규 fit 그림이 아님. 캐시 full-C RAW/LAST CPU 진단; FUTURE_ORACLE는 미래 실제 Y를 사용하여 배포 불가. 실제 K채널 F 비용/보편적 한계 아님.'
    old.finish_figure(fig,FIGURES[5],folder,plt,note,bottom=.15)


def report_markdown(t,app_statuses,context_link,inputs):
    rows=metric_groups(t['joint_scores']);lines=['# 작은 K 후속 진단: joint 목적과 실제 배포 비용','',
        '같은 원 MAIN checkpoint에서 FIT600 updates를 수행하고 actual-Y TUNE으로 선택한 결과다. seed2개·반복 Robin VAL·이미 노출된 Beijing EVAL이며 독립 일반화 검증이 아니다. 새 학습을 모두 수행한 것과 선택 checkpoint가 개선한 것은 구분한다.','']
    def mean(d,method,metric):return float(np.mean([float(r[metric]) for r in rows if r['dataset']==d and r['method']==method]))
    lines+=['## 먼저 답하는 여섯 질문','']
    for title in ('1. 표현을 갱신하고 RAW teacher를 쓰면 원 RAW와의 격차를 닫았나?',):lines += [title,'','```text','자료                방법      normalized MSE / MAE          MSE gap vs RAW']
    for d in DATASETS:
        for method in ('RAW_BOLT','OLD_MAIN','Y','RAW','MIXED'):
            lines.append(f'{d:21} {method:10} {mean(d,method,"mse"):.9f} / {mean(d,method,"mae"):.9f}  {mean(d,method,"mse")-mean(d,"RAW_BOLT","mse"):+.9f}')
    lines+=['```','원수치는 seed별 표에 보존한다. RAW 목적의 개선 여부는 actual-Y로 읽으며 teacher 모방 손실 감소만으로 실제 예측 개선이라 부르지 않는다.','',
        '2. decoder만 바꿀 때와 joint E/D 학습의 차이는?', '',
        '동결 E/F/hidden-D의 output ridge 재학습은 이전 D2이고, 이번에는 E와 D를 모두 갱신했다. 비교할 D2 원수치는 [동결 decoder 표](tables/context_d2_selected_mean.csv)에 복사했다. D2의 REFIT_RAW는 teacher 목적의 출력 head 재학습이며 원 RAW_BOLT baseline과 다르다. 최종600 변화 증거와 실제 선택 checkpoint를 함께 보아야 한다.','',
        '3. MSE 개선과 MAE 악화는 어디서 생겼나?', '',
        'Appliances 기존 LEARNED와 RAW의 차이를 같은 RAW-error 경계 body/tail로 분해했다. [signed contribution 표](tables/simple_tail_region_comparisons.csv)는 q=.9/.95/.99·MSE/MAE·전체 split denominator를 보존한다. 이는 오차 분포의 위치를 보여주며 학습 문제의 인과 원인을 확정하지 않는다. 신규 APP 목표 비교 상태는 아래에 따로 기록한다.','',
        '4. 물리 채널 K개만 배분하는 대안은 가능한가?', '',
        'FIT 고정·과거128 분산·미래 oracle 선택을 같은 RAW/LAST 메뉴에서 모두 보존했다. [채널 예산 표](tables/channel_budget_ALL_METRICS.csv)를 기준으로 gate를 판단했다. FUTURE_ORACLE는 실제 미래가 필요해 배포할 수 없다. 현재 단순 물리 routing 실제 구현은 진행하지 않았으며 다른 잠재 표현의 가능성을 닫는 결과로 읽지 않는다.','',
        '5. batch1에서 실제 효율 이득이 있었나?', '', '```text','자료                 B    조건/seed             whole ms/window  raw ratio']
    for r in t['joint_cost']:
        raw=next(v for v in t['joint_cost'] if v['dataset']==r['dataset'] and v['batch']==r['batch'] and v['method']=='RAW_BOLT')
        lines.append(f'{r["dataset"]:21} {r["batch"]:1} {str(r["objective"]):7}/{str(r["seed"]):5}  {r["median_ms_per_window"]:12.6f}  {r["median_ms_per_window"]/raw["median_ms_per_window"]:.4f}')
    lines+=['```','비율1 미만이 이 계측에서 빠른 경로다. warm2/repeat7의 동일 batch 비교이며 단일 장비·한 프로세스·모두 co-resident 조건이다. load/reference 준비·teacher 생성·학습 비용을 제외했고 반복 범위를 신뢰구간이라 하지 않는다. batch8 처리량을 batch1 응답 속도로 바꾸어 말하지 않는다.','',
        '6. 무엇이 남았고 다음 판단은?', '',
        '실제600 E/D 학습, 동결 F input gradient, 원 CP 보존과 selected inference를 검사했다. 표현의 미래 예측 적합성·동결 F와 표현 좌표의 호환성·복원 오차를 완전히 분리한 인과 결론은 아직 아니다. 가장 작은 목적 변경으로 격차가 닫히는지, MAE tradeoff가 달라지는지, batch1 비용을 포함해 유지할 가치가 있는지 판단한다. Chronos-Bolt 유지나 새 대조학습을 실행 횟수만으로 정당화하지 않는다.','',
        '## APP 신규 목표 비교의 실제 상태','', '```text','seed    objective status                    reason']
    for r in app_statuses:lines.append(f'{r["seed"]}  {r["objective"]:8} {r["status"]:24} {r.get("reason","")}')
    lines+=['```','미실행/부분 실패는 완료 실험으로 집계하지 않는다. APP가 없을 때 여섯 번째 그림은 기존 캐시의 채널 예산 진단이다.','',
        '## 학습·비용·검산의 범위','',
        '원 joint v1은150초 process cap에서510 updates까지만 진행한 기록과 약151초 실패 비용을 보존했다. v2는 안전 검사 cadence만 바꿨으며 원수치·손실·batch order·상한을 바꾸거나 u400부터 이어 학습하지 않았다. full guard CPU/wall은 GPU kernel 시간과 다르다. 현재 FIT 유한성 원수치를 [학습 proof 표](tables/joint_learning.csv)에 보존하며, NaN masking 구현 수정이 기존 나쁜 MSE의 원인이었다고 주장하지 않는다.','',
        '원 full-C teacher precompute 비용은 unknown이며 이번 비용 영수증에 포함되지 않았다. 모든 방법에 현재 C anchor가 있어 strict K-only 병목이 아니다. 검사 통과는 누출 부재·성능 개선·독립 일반화의 증명이 아니다. 이전 D0의 입력 복원과 예측 호환성, D1의 full-C rotation/압축 구분, D2/D3 계측 맥락은 '+context_link+'에서 확인한다.','',
        '## 그림과 원수치','']
    for name in FIGURES:lines += [f'![{name}](figures/{name}.png)','',f'[SVG](figures/{name}.svg)','']
    lines += ['[입력 경로·SHA 목록](provenance/input_manifest.csv) · [28행 비용](tables/joint_cost.csv) · [seed별 모든 split](tables/joint_scores.csv) · [checkpoint 선택](tables/joint_checkpoints.csv)','',
        '검산 원문과 작은 producer JSON은 provenance 아래 복사한다. scientific NPZ/checkpoint는 ignored cache에 있으므로 GitHub 링크를 만들지 않고 원 path/SHA를 표와 JSON의 literal 값으로 보존한다. 보고서 생성은 CPU 집계이며 모델 추론을 재실행하지 않는다.','']
    teacher=[];decoder=[];routing=[];bodytail=[];speed=[]
    for d in DATASETS:
        gap=mean(d,'RAW','mse')-mean(d,'RAW_BOLT','mse')
        teacher.append(f'[확인] {d}: joint RAW teacher의 actual-Y MSE gap은 {gap:+.9f}, MAE gap은 {mean(d,"RAW","mae")-mean(d,"RAW_BOLT","mae"):+.9f}이다. '+
            ('이 점추정에서 MSE 격차가 남았다.' if gap>1e-8 else '이 점추정에서 MSE 격차가 남지 않았으나 독립 일반화는 확인하지 않았다.'))
        for o in OBJECTIVES:
            frozen=next(v for v in t['context_d2_selected_mean'] if v['dataset']==d and v['split']==FINAL[d] and v['method']=='REFIT_'+o)
            decoder.append(f'{d} {o}: frozen-output-refit MSE/MAE {float(frozen["mse"]):.9f}/{float(frozen["mae"]):.9f} → joint {mean(d,o,"mse"):.9f}/{mean(d,o,"mae"):.9f}.')
        selected=[v for v in t['channel_budget'] if v['dataset']==d and v['split']==FINAL[d] and v['scale']=='normalized' and v['channel']=='macro']
        raw=next(v for v in selected if v['method']=='FULL_C_RAW_REFERENCE')
        routing.append(d+': '+', '.join(f'{v["method"]} MSE {float(v["mse"]):.6f}' for v in selected)+f'; full RAW {float(raw["mse"]):.6f}.')
        costs=[v for v in t['joint_cost'] if v['dataset']==d and v['batch']==1 and v['method']=='JOINT_SELECTED']
        raw=next(v for v in t['joint_cost'] if v['dataset']==d and v['batch']==1 and v['method']=='RAW_BOLT')
        ratios=[v['median_ms_per_window']/raw['median_ms_per_window'] for v in costs]
        speed.append(f'{d}: batch1 selected6개/공유 RAW latency 비율 범위 {min(ratios):.4f}~{max(ratios):.4f}. 개별 조건과 반복 범위는 비용 표에 보존했다.')
    for m in ('mse','mae'):
        contributions=[v for v in t['simple_tail'] if v['split']=='eval' and v['metric']==m and v['quantile']=='0.95' and v['comparison']=='LEARNED_SEED_MEAN_vs_RAW']
        bodytail.append('Appliances q=.95 '+m.upper()+': '+', '.join(v['region']+' signed gap '+f'{float(v["gap_contribution"]):+.9f}' for v in contributions)+'.')
    text='\n'.join(lines)
    text=text.replace('2. decoder만 바꿀 때와 joint E/D 학습의 차이는?', '\n\n'.join(teacher)+'\n\n2. decoder만 바꿀 때와 joint E/D 학습의 차이는?')
    text=text.replace('3. MSE 개선과 MAE 악화는 어디서 생겼나?', '\n\n'.join(decoder)+'\n\n3. MSE 개선과 MAE 악화는 어디서 생겼나?')
    text=text.replace('4. 물리 채널 K개만 배분하는 대안은 가능한가?', '\n\n'.join(bodytail)+'\n\n4. 물리 채널 K개만 배분하는 대안은 가능한가?')
    text=text.replace('5. batch1에서 실제 효율 이득이 있었나?', '\n\n'.join(routing)+'\n\n5. batch1에서 실제 효율 이득이 있었나?')
    text=text.replace('6. 무엇이 남았고 다음 판단은?', '\n\n'.join(speed)+'\n\n6. 무엇이 남았고 다음 판단은?')
    drift=[]
    for d in DATASETS:
        previous=[float(v['median_ms_per_window']) for v in t['context_d3_previous_cost'] if v['dataset']==d and str(v['batch'])=='1' and v['method']=='RAW_BOLT' and v['preprocessing']=='vectorized']
        current=next(v for v in t['joint_cost'] if v['dataset']==d and v['batch']==1 and v['method']=='RAW_BOLT')
        drift.append(f'{d} RAW batch1: 이전 D3 seed 평균 {np.mean(previous):.6f} ms/window → 이번 공유 RAW {current["median_ms_per_window"]:.6f} ms/window.')
    caveat='\n\n'.join(drift)+'\n\n이번 공유 RAW는 고정된 단일 순서에서 먼저 측정했다. 특히 Robin RAW 기준의 측정 차이가 크다. warm2/repeat7만으로 thermal·clock·route-order 영향을 배제하지 못했으며 원인은 확인하지 않았다. 현재 RAW 대비 비율만으로 batch1 효율 문제가 해결됐다고 주장할 수 없다. 재측정은 이번 상한에서 수행하지 않았다.'
    text=text.replace('6. 무엇이 남았고 다음 판단은?',caveat+'\n\n6. 무엇이 남았고 다음 판단은?')
    app_rows=metric_groups(t['app_scores']) if t['app_scores'] else []
    if app_rows:
        app_lines=['APP actual-Y TUNE MSE로 선택된 노출 EVAL 결과(seed metric 평균):','', '```text','방법             normalized MSE / MAE']
        for method in ('RAW_BOLT','RAW_PLUS_R','OLD_LEARNED','MSE','MAE','HUBER_DELTA1'):
            selected=[r for r in app_rows if r['method']==method]
            if selected:app_lines.append(f'{method:16} {np.mean([float(r["mse"]) for r in selected]):.9f} / {np.mean([float(r["mae"]) for r in selected]):.9f}')
        app_lines+=['```','모든 신규 loss는 actual Y만 학습하며 RAW forecast는 진단 reference다. 원 CP는 LEARNED_REGIME(q auxiliary w=.1로 과거 학습됨), 신규6개에서는 q를 공통 제거했다. original_MAIN이라는 저장 key를 새 MAIN baseline으로 부르지 않는다.','']
        text=text.replace('## 학습·비용·검산의 범위','\n'.join(app_lines)+'\n## 학습·비용·검산의 범위')
    cost_text='APP 신규14행 비용: **NOT_MEASURED**. RAW+R 배포 비용도 **NOT_MEASURED**.'
    if t.get('app_cost'):
        cost_text='APP 신규14행 전체 비용을 [실제 APP 비용 표](tables/app_cost.csv)에 보존했다. RAW+R 비용은 **NOT_MEASURED**다.\n\n```text\nB  조건/seed                    whole ms/window'
        cost_text+='\n'+'\n'.join(f'{r["batch"]}  {str(r["objective"]):13}/{str(r["seed"]):5} {r["median_ms_per_window"]:.6f}' for r in t['app_cost'])+'\n```'
    text=text.replace('## 학습·비용·검산의 범위',cost_text+'\n\n## 학습·비용·검산의 범위')
    return text


def generate(args):
    if not args.no_contention_attested:raise ValueError('ROOT_TRAINING_AND_LATENCY_JOBS_ENDED_ATTESTATION_REQUIRED')
    output=HERE/('report_'+args.tag);output.mkdir(exist_ok=False)
    result={'status':'RUNNING','GPU_execution':False,'new_model_inference':False,'NPZ_or_checkpoint_opened':False,
            'independent_holdout':False,'whole_goal_complete':False,'actual_report_generated':False}
    limit=None;inputs=[];started=time.perf_counter()
    import continuation_runtime_v1 as runtime
    try:
        limit=runtime.ContinuationLimit('report_'+args.tag,gpu=False)
        result.update(authorization_record=limit.authorization_record,run_manifest_record=limit.run_manifest_record,
            runtime_adaptation_scope='new continuation resource adapter only; frozen v2 scientific/report adapters unchanged')
        import prepare_inputs
        prepare_inputs.install_protected_test_guard()
        for name in ('tables','figures','provenance','provenance/producers','provenance/audits'):(output/name).mkdir(exist_ok=False)
        producers=[];records=[]
        for i,path in enumerate(args.joint_results):
            p,r=old.checked_json(path);producers.append(p);records.append(r)
            copy_input(r,output,f'producers/joint_{i+1:02}.json',inputs)
        audit,auditrec=old.checked_json(args.joint_audit);passed_audit(audit)
        contract.verify_audit_binding(audit,producers,records);copy_input(auditrec,output,'audits/JOINT.json',inputs)
        t=joint_tables(producers)
        cost,costrec=old.checked_json(args.cost_result);t['joint_cost']=cost_tables(cost)
        if {r['sha256'] for r in cost['joint_producers']}!={r['sha256'] for r in records}:raise ValueError('COST_INPUT_JOINT12_BINDING_MISMATCH')
        copy_input(costrec,output,'producers/COST28.json',inputs)
        if args.cost_audit:
            ca,cr=old.checked_json(args.cost_audit);passed_audit(ca);copy_input(cr,output,'audits/COST28.json',inputs)
        simple,simplerec=old.checked_json(args.simple_result)
        if simple.get('status')!='COMPLETE':raise ValueError('SIMPLE_PHASE_V2_COMPLETE_REQUIRED')
        copy_input(simplerec,output,'producers/SIMPLE_PHASE_V2.json',inputs)
        for name,r in simple['tables'].items():
            source=old.resolve(r['path']);digest=old.sha(source)
            if digest!=r['sha256']:raise ValueError('SIMPLE_TABLE_SHA_MISMATCH')
            dest=output/'tables'/('simple_'+name+'.csv');shutil.copyfile(source,dest)
            if old.sha(dest)!=digest:raise ValueError('COPIED_SIMPLE_TABLE_SHA_MISMATCH')
            inputs.append({'path':str(source),'sha256':digest,'copied_to':str(dest.relative_to(output)).replace('\\','/')})
        t['simple_tail']=old.csv_rows(output/'tables/simple_tail_region_comparisons.csv')
        budget,br=old.checked_json(args.channel_result)
        if budget.get('status')!='COMPLETE':raise ValueError('CHANNEL_CPU_DIAGNOSTIC_COMPLETE_REQUIRED')
        ba,bar=old.checked_json(args.channel_audit);passed_audit(ba)
        copy_input(br,output,'producers/CHANNEL_BUDGET.json',inputs);copy_input(bar,output,'audits/CHANNEL_BUDGET.json',inputs)
        source=old.resolve(args.channel_result).parent/'ALL_METRICS.csv';digest=old.sha(source)
        shutil.copyfile(source,output/'tables/channel_budget_ALL_METRICS.csv')
        inputs.append({'path':str(source),'sha256':digest,'copied_to':'tables/channel_budget_ALL_METRICS.csv'})
        t['channel_budget']=old.csv_rows(source)
        context=old.resolve(args.context_report);manifest,mr=old.checked_json(context/'MANIFEST.json')
        if manifest.get('generator_result_status')!='COMPLETE':raise ValueError('PREVIOUS_ACTUAL_REPORT_COMPLETE_REQUIRED')
        copy_input(mr,output,'CONTEXT_MANIFEST.json',inputs)
        for artifact in manifest['artifacts']:
            name=artifact['path']
            if name in ('REPORT_KO.md','RESULT.json') or name.startswith('provenance/audits/'):
                source=context/name
                if old.sha(source)!=artifact['sha256']:raise ValueError('CONTEXT_REPORT_OR_AUDIT_SHA_MISMATCH')
                target='audits/CONTEXT_'+source.name if name.startswith('provenance/audits/') else 'CONTEXT_'+source.name
                copy_input({'path':str(source),'sha256':artifact['sha256']},output,target,inputs)
        for name in ('d2_selected_mean.csv','d0_summary_mean.csv','d1.csv','d3.csv'):
            artifact=next(r for r in manifest['artifacts'] if r['path']=='tables/'+name)
            source=context/'tables'/name;digest=old.sha(source)
            if digest!=artifact['sha256']:raise ValueError('CONTEXT_CSV_SHA_MISMATCH')
            shutil.copyfile(source,output/'tables'/('context_'+name));inputs.append({'path':str(source),'sha256':digest,'copied_to':'tables/context_'+name})
            if name=='d2_selected_mean.csv':t['context_d2_selected_mean']=old.csv_rows(source)
            if name=='d3.csv':t['context_d3_previous_cost']=old.csv_rows(source)
        app=[]
        for i,path in enumerate(args.app_results or []):
            p,r=old.checked_json(path);app.append(p);copy_input(r,output,f'producers/APP_{i+1:02}.json',inputs)
        t['app_scores'],app_statuses=app_tables(app,args.app_not_executed_reason)
        if any(p['status']=='COMPLETE' for p in app):
            if not args.app_audit:raise ValueError('ACTUAL_APP_INDEPENDENT_AUDIT_REQUIRED')
            aa,ar=old.checked_json(args.app_audit);passed_audit(aa);copy_input(ar,output,'audits/APP.json',inputs)
            app_records=[r for r in inputs if r['copied_to'].startswith('provenance/producers/APP_')]
            declared={(r['dataset'],r['seed'],r['objective']):r['producer_sha256'] for r in aa.get('conditions',[])}
            expected={(p['dataset'],p['seed'],p['objective']):r['sha256'] for p,r in zip(app,app_records) if p['status']=='COMPLETE'}
            if not expected or any(declared.get(k)!=v for k,v in expected.items()):raise ValueError('APP_AUDIT_COMPLETE_PRODUCER_BINDING_MISMATCH')
        t['app_cost']=[]
        if args.app_cost_result:
            ac,acr=old.checked_json(args.app_cost_result);t['app_cost']=app_cost_tables(ac)
            if {r['sha256'] for r in ac['APP_producers']}!={r['sha256'] for r in inputs if r['copied_to'].startswith('provenance/producers/APP_')}:raise ValueError('APP_COST_PRODUCER_BINDING_MISMATCH')
            copy_input(acr,output,'producers/APP_COST14.json',inputs)
            if args.app_cost_audit:
                aca,acar=old.checked_json(args.app_cost_audit);passed_audit(aca);copy_input(acar,output,'audits/APP_COST14.json',inputs)
        for path in (Path(__file__),HERE/'make_followup_report.py',HERE/'profile_joint_cost_v1.py',HERE/'continuation_runtime_v1.py'):
            copy_input({'path':str(path),'sha256':old.sha(path)},output,'source_'+path.name,inputs)
        for name,record in (('CONTINUATION_AUTHORIZATION.json',limit.authorization_record),('CONTINUATION_RUN_MANIFEST.json',limit.run_manifest_record)):
            copy_input(record,output,name,inputs)
        for name,rows in t.items():
            if rows and name!='context_d2_selected_mean':old.save_csv(output/'tables'/(name+'.csv'),rows)
        old.save_csv(output/'provenance/input_manifest.csv',inputs);old.write(output/'provenance/APP_STATUS.json',app_statuses)
        limit.check();plt,font=old.configure_plotting()
        for plot in (figure_scores,figure_selection,figure_cost,figure_tail,figure_learning,figure_app_or_budget):
            plot(t,output/'figures',plt);limit.check()
        context_link='[이전 D0-D3 보고서](../'+context.name+'/REPORT_KO.md)'
        markdown=report_markdown(t,app_statuses,context_link,inputs)
        markdown+='\n검산 원문: '+ ' · '.join(f'[{p.name}](provenance/audits/{p.name})' for p in sorted((output/'provenance/audits').glob('*.json')))+'\n'
        markdown+='\n추가 자원 근거: [사용자 승인](provenance/CONTINUATION_AUTHORIZATION.json) · [고정 실행 manifest](provenance/CONTINUATION_RUN_MANIFEST.json) · [실제 runtime source](provenance/source_continuation_runtime_v1.py)\n'
        markdown,links=checked_links(markdown,output)
        with (output/'REPORT_KO.md').open('x',encoding='utf-8') as f:f.write(markdown)
        for r in inputs:
            if old.sha(r['path'])!=r['sha256'] or old.sha(output/r['copied_to'])!=r['sha256']:raise ValueError('INPUT_OR_COPY_CHANGED_DURING_REPORT')
        result.update(status='COMPLETE',actual_report_generated=True,source_sha256=old.sha(__file__),font=font,
            figures_count=6,tables_rows={k:len(v) for k,v in t.items()},relative_links_checked=links,
            report_body_wall_seconds=time.perf_counter()-started,APP_statuses=app_statuses,
            scientific_scope='joint12/cost28 complete; APP statuses separate; repeated/exposed evaluation only')
        limit.check()
    except Exception as error:
        result.update(status='FAILED_PRESERVED',error={'type':type(error).__name__,'message':str(error)});raise
    finally:
        receipt=limit.close(result['status'],result.get('error')) if limit else getattr(runtime,'LAST_CONSTRUCTOR_TERMINAL',None)
        if receipt is not None:
            result['shared_resource_receipt']=receipt
            if receipt.get('terminal_status',result['status'])!='COMPLETE':result.update(status=receipt.get('terminal_status',result['status']),error=receipt.get('terminal_error',result.get('error')))
        old.write(output/'RESULT.json',result)
        old.write(output/'MANIFEST.json',{'generator_result_status':result['status'],'artifacts':[
            {'path':str(p.relative_to(output)).replace('\\','/'),'sha256':old.sha(p),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()]})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--describe',action='store_true');mode.add_argument('--generate',action='store_true')
    parser.add_argument('--tag');parser.add_argument('--joint-results',nargs='+')
    for name in ('joint-audit','cost-result','cost-audit','simple-result','channel-result','channel-audit','context-report','app-audit','app-cost-result','app-cost-audit'):parser.add_argument('--'+name)
    parser.add_argument('--app-results',nargs='+');parser.add_argument('--app-not-executed-reason');parser.add_argument('--no-contention-attested',action='store_true')
    args=parser.parse_args()
    if args.describe:
        print(json.dumps({'status':'PREPARATION_ONLY','actual_GPU_or_scientific_NPZ_reads':False,'figures':6,
            'requires_root_GO':True,'expected_report_CPU_wall_seconds':[10,30],'APP_missing_is_NOT_EXECUTED':True}));return
    required=('tag','joint_results','joint_audit','cost_result','simple_result','channel_result','channel_audit','context_report')
    if any(not getattr(args,k) for k in required) or not args.no_contention_attested or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.tag):parser.error('generate requires all terminal actual inputs, safe fresh tag and no-contention attestation')
    result=generate(args)
    print(json.dumps({'status':result['status'],'figures':result.get('figures_count'),'whole_goal_complete':False}))
    if result['status']!='COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()
