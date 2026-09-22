from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from .io import *
from .metrics import score, scalar_primary, gain, paired_bootstrap, first_target


def markdown_table(rows,columns):
    def fmt(x):
        if x is None:return '—'
        if isinstance(x,float):return f'{x:.6f}' if abs(x)<1 else f'{x:.3f}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(fmt(r.get(c)) for c in columns)+' |' for r in rows])


def render_panel(ctx,name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result,cache=ctx.paths(name); panel=ctx.panel(name); cfg=ctx.cfg
    manifest=read_json(result/'PREDICTIONS_MANIFEST.json')
    if not manifest.get('all_saved_before_scoring'): raise Blocked('PREDICTIONS_NOT_SEALED')
    for p,h in manifest['predictions'].items():
        if sha(ctx.repo/p)!=h: raise Blocked('PREDICTION_CHANGED_BEFORE_SCORING')
    seal=read_json(result/'SELECTION_SEAL.json')
    if sha(result/'SELECTION_SEAL.json')!=manifest['selection_sha256']: raise Blocked('SELECTION_HASH_MISMATCH')
    levels=np.asarray(read_json(result/'MODEL_RECEIPT.json')['levels']); y=panel.targets('test')
    pred_dir=cache/'test_predictions'; score_rows=[]; per={}; per_rows=[]
    fits=[read_json(result/'fits'/f'{name}_{a}_{s}.json') for s in cfg['seeds'] for a in cfg['arms']]
    for label in ['F0_LONG','F0_SHORT','SEASONAL_NAIVE']:
        z=np.load(pred_dir/(label+'.npz')); m,p=score(z['q'],y,panel.sigma,levels)
        per[(label,0,'selected')]=p
        score_rows.append(dict(panel=name,arm=label,seed=0,variant='selected',step=0,**{k:v for k,v in m.items() if k!='per_channel'}))
        for c,v in zip(panel.columns,m['per_channel']): per_rows.append(dict(panel=name,arm=label,seed=0,variant='selected',channel=c,primary=v))
    for f in fits:
        c=seal['choices'][f['key']]; variants={'selected':c['selected_step'],'fixed_final':cfg['max_steps']}
        variants.update({f'budget_{k}':v for k,v in c['budget_steps'].items() if v is not None})
        for variant,step in variants.items():
            z=np.load(pred_dir/f'{f["key"]}_{step}.npz'); q=z['q']; np.testing.assert_array_equal(z['origins'],panel.origins['test'])
            m,p=score(q,y,panel.sigma,levels); per[(f['arm'],f['seed'],variant)]=p
            # Independent scalar audit avoids making the reporting formula its own oracle.
            check=scalar_primary(q[:2,:2],y[:2,:2],panel.sigma[:2],levels)
            expected=score(q[:2,:2],y[:2,:2],panel.sigma[:2],levels)[0]['primary']
            if not np.isclose(check,expected,rtol=1e-10,atol=1e-12): raise Blocked('INDEPENDENT_METRIC_REPLAY_FAILED')
            score_rows.append(dict(panel=name,arm=f['arm'],seed=f['seed'],variant=variant,step=step,
                **{k:v for k,v in m.items() if k!='per_channel'}))
            for ch,v in zip(panel.columns,m['per_channel']): per_rows.append(dict(panel=name,arm=f['arm'],seed=f['seed'],variant=variant,channel=ch,primary=v))
    write_csv(result/'SCORES.csv',score_rows);write_csv(result/'CHANNEL_SCORES.csv',per_rows)
    origin_rows=[]
    for (a,s,v),p in per.items():
        for i,o in enumerate(panel.origins['test']):
            origin_rows.append(dict(panel=name,arm=a,seed=s,variant=v,origin=int(o),
                                   timestamp=str(np.datetime64(int(panel.timestamps[o]),'ns')),primary=float(p[i].mean())))
    write_csv(result/'ORIGIN_SCORES.csv',origin_rows)
    summaries=[]
    for a in ['F0_SHORT','F0_LONG','SEASONAL_NAIVE']+cfg['arms']:
        rr=[r for r in score_rows if r['arm']==a and r['variant']=='selected']
        fs=[f for f in fits if f['arm']==a]
        summaries.append(dict(arm=a,primary=float(np.mean([r['primary'] for r in rr])),
            nmae=float(np.mean([r['nmae'] for r in rr])),
            selected_steps='/'.join(str(r['step']) for r in rr),
            cold_ready_s=float(np.mean([f['selected_cold_ready_s'] for f in fs])) if fs else None,
            trainable_parameters=fs[0]['trainable_parameters'] if fs else 0,
            train_peak_mib=float(max(f['training_peak_allocated_bytes'] for f in fs)/2**20) if fs else None))
    reference=read_json(result/'F0_REFERENCE.json')
    for rr in summaries:
        if rr['arm']=='F0_LONG': rr['cold_ready_s']=reference['f0_cold_ready_s']
        if rr['arm']=='SEASONAL_NAIVE': rr['cold_ready_s']=reference['naive_cold_ready_s']
    summary={r['arm']:r for r in summaries}
    eff=[]
    pairs=[('LORA','HEAD'),('LORAPLUS','HEAD'),('LORAPLUS','LORA'),('HEAD','F0_LONG')]
    pair=(seal['internal_reference'],seal['cheap_reference'])
    if pair not in pairs:pairs.append(pair)
    for a,b in pairs:
        aa=np.stack([per[(a,s,'selected')] for s in cfg['seeds']])
        bb=np.stack([per[(b,s,'selected')] if b in cfg['arms'] else per[(b,0,'selected')] for s in cfg['seeds']])
        # Natural-day evaluation strides. Block labels preserve gaps instead of pretending consecutive rows are days.
        labels=panel.timestamps[panel.origins['test']]//int(cfg['bootstrap_block_days']*86400*10**9)
        bs=paired_bootstrap(aa,bb,cfg['bootstrap_repetitions'],cfg['bootstrap_block_days'],cfg['bootstrap_seed'],labels=labels)
        eff.append(dict(candidate=a,baseline=b,mean_gain_pct=bs['gain_pct'],ci_low=bs['ci95'][0],ci_high=bs['ci95'][1],
                        seed1_gain_pct=bs['seed_gains'][0],seed2_gain_pct=bs['seed_gains'][1]))
    write_csv(result/'EFFECTS.csv',eff)
    e={(r['candidate'],r['baseline']):r for r in eff}
    threshold=float(np.mean([f['curves'][-1]['val_primary'] for f in fits if f['arm']==seal['internal_reference']]))*(1+cfg['target_relative_tolerance'])
    ttt=[]
    for f in fits: ttt.append(dict(panel=name,arm=f['arm'],seed=f['seed'],threshold=threshold,**first_target(f['curves'],threshold)))
    write_csv(result/'TIME_TO_TARGET.csv',ttt)
    resources=[]
    for f in fits:
        times=np.asarray([r['active_step_s'] for r in f['trace']]); trimmed=times[min(16,len(times)//4):]
        cv=float(trimmed.std()/max(trimmed.mean(),1e-12))
        resources.append(dict(panel=name,arm=f['arm'],seed=f['seed'],selected_step=f['selected_step'],
            selected_cold_ready_s=f['selected_cold_ready_s'],cache_build_s_charged=f['cache_build_s_charged'],
            model_load_s=f['model_load_s'],data_prepare_s_charged=f['data_prepare_s_charged'],
            active_training_s=f['active_training_s'],full_job_wall_s=f['full_job_wall_s'],
            train_peak_allocated_bytes=f['training_peak_allocated_bytes'],active_step_cv=cv,
            timing_noisy=cv>cfg['timing_noise_cv_limit']))
    write_csv(result/'RESOURCES.csv',resources)
    curves=[dict(panel=name,arm=f['arm'],seed=f['seed'],**{k:v for k,v in r.items() if k not in ['state_path','state_sha256','val_prediction']}) for f in fits for r in f['curves']]
    write_csv(result/'CURVES.csv',curves)
    # Decision is about a problem gap, not a new PEFT method or an equivalence proof.
    internal=seal['internal_reference']; cheap=seal['cheap_reference']; gap=e[(internal,cheap)]
    head_fits=[f for f in fits if f['arm']=='HEAD']
    boundary=[]
    for f in head_fits:
        a,b=f['curves'][-2:]; boundary.append(f['selected_step']==cfg['max_steps'] and gain(b['val_primary'],a['val_primary'])>cfg['head_boundary_improvement_pct'])
    ratio=summary[internal]['cold_ready_s']/max(summary[cheap]['cold_ready_s'],1e-9)
    if gap['mean_gain_pct']>=cfg['quality_gap_screen_pct'] and gap['seed1_gain_pct']>0 and gap['seed2_gain_pct']>0:
        decision='QUALITY_GAP_BUDGET_LIMITED' if any(boundary) else ('INTERNAL_GAP_WORTH_FOLLOWUP' if ratio>=cfg['cost_gap_screen_ratio'] else 'QUALITY_GAP_NO_COST_BOTTLENECK')
    elif gap['mean_gain_pct']<=cfg['quality_gap_screen_pct'] and ratio>=1 and not any(boundary):
        decision='OUTPUT_SUFFICIENT_AT_CURRENT_RESOLUTION'
    else: decision='HOLD_NO_NEW_METHOD_AUTHORIZATION'
    if any(r['timing_noisy'] for r in resources) and decision=='INTERNAL_GAP_WORTH_FOLLOWUP': decision='QUALITY_GAP_TIMING_UNCERTAIN'
    conclusions=dict(status=decision,internal_reference_from_validation=internal,cheap_reference_from_validation=cheap,internal_gain_vs_cheap_pct=gap['mean_gain_pct'],internal_gain_vs_head_pct=e[(internal,'HEAD')]['mean_gain_pct'],
        seed_gains=[gap['seed1_gain_pct'],gap['seed2_gain_pct']],cold_ready_ratio_internal_over_cheap=ratio,cold_ready_ratio_internal_over_head=summary[internal]['cold_ready_s']/max(summary['HEAD']['cold_ready_s'],1e-9),
        head_at_improving_boundary=boundary,novel_method_claim=False,automatic_successor=False,
        scope='Two-seed, one fixed recipe per method; previously used data; head family preselected historically, no full HPO',
        thresholds_are_project_screen_not_scientific_laws=True)
    write_json(result/'DECISION.json',conclusions)
    figdir=result/'figures';figdir.mkdir(exist_ok=True)
    is_demo=read_json(ctx.out/'ENVIRONMENT.json').get('authenticity')!='REAL_MODEL_RUN'
    stamp='SYNTHETIC TEST — ' if is_demo else ''
    for xkey,fname,xlabel in [('step','val_updates.png','Optimizer updates'),('cold_ready_s','val_total_cost.png','Cold-start cost incl. preparation (s)')]:
        fig,ax=plt.subplots(figsize=(7.6,4.7))
        for arm in cfg['arms']:
            ff=[f for f in fits if f['arm']==arm]
            # Plot each actual trace; a seed mean on irregular time points would invent equal-time observations.
            for j,f in enumerate(ff): ax.plot([r[xkey] for r in f['curves']],[r['val_primary'] for r in f['curves']],
                    marker='o',linestyle='-' if j==0 else '--',label=f'{arm}, seed {f["seed"]}')
        ax.set(xlabel=xlabel,ylabel='Validation scaled 2-pinball (lower better)',title=stamp+name)
        ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figdir/fname,dpi=155);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.6,4.7))
    labels=[r['arm'] for r in summaries]; ax.bar(labels,[r['primary'] for r in summaries])
    for i,a in enumerate(labels):
        vals=[r['primary'] for r in score_rows if r['arm']==a and r['variant']=='selected'];ax.scatter([i]*len(vals),vals,marker='x')
    ax.set(ylabel='TEST scaled 2-pinball (lower better)',title=stamp+name+' — selected by validation');ax.tick_params(axis='x',rotation=25)
    fig.tight_layout();fig.savefig(figdir/'test_scores.png',dpi=155);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.6,4.7))
    for r in summaries:
        if r['cold_ready_s'] is not None:
            ax.scatter(r['cold_ready_s'],r['primary']);ax.annotate(r['arm'],(r['cold_ready_s'],r['primary']),xytext=(5,5),textcoords='offset points')
    ax.set(xlabel='Cold-start ready-to-select cost (s)',ylabel='TEST scaled 2-pinball',title=stamp+name+' — measured quality/cost')
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(figdir/'quality_cost.png',dpi=155);plt.close(fig)
    phead=np.stack([per[('HEAD',s,'selected')] for s in cfg['seeds']]).mean((0,1))
    pint=np.stack([per[(internal,s,'selected')] for s in cfg['seeds']]).mean((0,1))
    fig,ax=plt.subplots(figsize=(max(7.6,len(panel.columns)*.4),4.7));ax.bar(range(len(panel.columns)),100*(phead-pint)/phead)
    ax.axhline(0,linestyle='--');ax.set_xticks(range(len(panel.columns)),panel.columns,rotation=70,ha='right')
    ax.set(ylabel='Internal versus HEAD gain (%)',title=stamp+name+' — channel-wise trade-offs')
    fig.tight_layout();fig.savefig(figdir/'channel_gains.png',dpi=155);plt.close(fig)
    text=f'''# {name}: 내부 적응의 정확도–비용 격차 확인

{'**SYNTHETIC CPU TEST. 실제 TSFM 성능이 아닌 보고서 경로 검사다.**' if is_demo else '[확인] 이 보고서는 실행 후 저장된 예측·선택·자원 기록에서 자동 작성됐다.'}

## 1. 질문과 판정

**{decision}**

같은 8일 과거와 같은 원자료를 사용했을 때, 고정된 출력 보정으로 얻지 못하는 예측 이득이 내부 LoRA에 남는지 확인한다.
새 어댑터의 성능이나 논문 PASS를 판정하는 실험이 아니다. 검증으로 고른 내부 기준은 **{internal}**다.
검증으로 선택한 저비용 대조 **{cheap}** 대비 오차 감소는 **{gap['mean_gain_pct']:.4f}%**, 선택 모델까지의 준비비 포함 비용비는 **{ratio:.3f}배**다.
양수 개선이 없거나 간단한 방법으로 충분하면 새 모듈을 자동 생성하지 않는다.

## 2. 실제 설정

- 데이터: {panel.metadata['source_first']} ~ {panel.metadata['source_last']}; {len(panel.columns)}개 계열, 원 간격 {panel.metadata['frequency_seconds']}초.
- 입력 {panel.context}개(8일), 미래 {panel.horizon}개(1일). F0_SHORT만 4일이며 비교상의 유리한 기준선으로 대체하지 않는다.
- TRAIN/VAL/TEST의 합법적 원점: {len(panel.origins['train'])}/{len(panel.origins['val'])}/{len(panel.origins['test'])}.
- HEAD는 과거 S1에서 선택됐던 **{cfg['head_kind'][name]}**를 사전 고정. 이번 TEST로 구조를 고르지 않았다.
- 모든 군에 같은 native 훈련 손실, 같은 반복 seed·표본 순서·512-update 상한을 제공한다.
- LoRA+는 A/B 학습률 비 {cfg['loraplus_ratio']}를 적용한 기존 방법이며 우리의 새 기법이 아니다.
- TEST는 모든 선택을 고정한 뒤 예측·저장하고 채점했다. 이전에 사용된 데이터의 개발 재점검이며 독립 외부 확증이 아니다.

## 3. 점수: 낮을수록 좋음

주지표는 원단위 예측을 TRAIN 표준편차로 나눈 평균 2-pinball이다. native 훈련 공간의 손실과 구분한다.
정렬 전 분위수 교차율도 CSV에 보존한다. 각 원점·계열은 모든 군에서 같다.

{markdown_table(summaries,['arm','primary','nmae','selected_steps','cold_ready_s','train_peak_mib'])}

![TEST 점수](figures/test_scores.png)

{markdown_table(eff,['candidate','baseline','mean_gain_pct','seed1_gain_pct','seed2_gain_pct','ci_low','ci_high'])}

## 4. 학습곡선과 비용

![업데이트별 검증](figures/val_updates.png)

![준비비 포함 검증](figures/val_total_cost.png)

![성능 비용](figures/quality_cost.png)

HEAD는 각 독립 적용의 cold-start 비용에 전체 TRAIN/VAL 표현 캐시를 청구한다. 캐시는 물리적으로 공유해도 무료로 취급하지 않는다.
모델 로딩, 데이터 준비, 캐시, 역전파·optimizer, 각 선택 시점까지의 검증·체크포인트 I/O를 분리한다.
full_job_wall에는 장부 fsync 등도 포함된다. 테스트용 캐시는 평가 비용이며 적응에 쓰지 않는다.
GPU peak는 훈련 구간만 집계하고 캐시 구축 peak는 별도 receipt에 둔다. 파라미터 수만으로 속도를 주장하지 않는다.

## 5. 목표 품질 도달

목표는 검증으로 고른 내부 기준의 두 seed 평균 마지막 VAL 점수 × {1+cfg['target_relative_tolerance']:.3f}다.
동일한 목표를 모든 군에 적용한다. 첫 도달은 저장 checkpoint의 상한이며 정확한 연속 시점을 추정하지 않는다.
F0가 이미 만족하면 NO_ADAPTATION_HEADROOM, 미도달이면 CENSORED로 기록한다.

{markdown_table(ttt,['arm','seed','status','step','time_s','lower_step','lower_time_s'])}

## 6. 계열별 손익

![계열별 이득](figures/channel_gains.png)

## 7. 한계와 다음 결정

[판단] 현재 결과는 고정된 head 종류·학습률·시간 분할에서의 제한된 비교다. 완전한 HPO나 충분한 수렴을 보장하지 않는다.
HEAD가 마지막까지 의미 있게 개선 중이면 품질 격차를 OUTPUT이 본질적으로 부족하다는 증거로 확정하지 않는다.
두 seed·일주일 블록 구간은 새로운 도메인·모델·seed 모집단 전체의 불확실성을 보장하지 않는다.
시간 측정은 실제 순차 실행이며 측정 편차가 크면 알고리즘적 가속을 주장하지 않는다.
과거 S1의 절대 점수와 이번 점수를 직접 차감하지 않는다: 문맥, 전체 시간 범위, 분할, checkpoint가 달라졌다.
현재 판정은 **{decision}**다. 후속 새 모델·데이터·학습은 자동으로 시작하지 않는다.

## 8. 재검산 링크

[설정·자료](DATA_AUDIT.json) · [실모델 점검](PREFLIGHT.json) · [모델](MODEL_RECEIPT.json) · [선택 봉인](SELECTION_SEAL.json)

[점수](SCORES.csv) · [차이](EFFECTS.csv) · [원점별 점수](ORIGIN_SCORES.csv) · [계열별 점수](CHANNEL_SCORES.csv) · [비용](RESOURCES.csv) · [학습곡선](CURVES.csv)

[목표 도달](TIME_TO_TARGET.csv) · [예측 hash](PREDICTIONS_MANIFEST.json) · [판정](DECISION.json)

원자료·가중치·전체 예측은 로컬 ignored cache에 있다. GitHub의 집계 CSV만으로 모든 원본을 재생할 수 있다고 주장하지 않는다.
'''
    (result/'REPORT_KO.md').write_text(text,encoding='utf-8')
    (result/'FINAL_DECISION.md').write_text(f'# {decision}\n\n'+json.dumps(conclusions,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_json(result/'REPORT_RECEIPT.json',dict(complete=True,created_utc=time.time(),predictions_manifest_sha=sha(result/'PREDICTIONS_MANIFEST.json'),
        independent_scalar_replay=True,figures=[p.name for p in figdir.glob('*.png')]))
    return conclusions


def summary_report(ctx):
    rows=[]
    for name in ctx.cfg['panels']:
        result,_=ctx.paths(name)
        if (result/'DECISION.json').exists():
            d=read_json(result/'DECISION.json');rows.append(dict(panel=name,status=d['status'],gain_pct=d['internal_gain_vs_cheap_pct'],cost_ratio=d['cold_ready_ratio_internal_over_cheap']))
        else:
            err=read_json(result/'ERROR.json') if (result/'ERROR.json').exists() else {}
            rows.append(dict(panel=name,status=err.get('classification','NOT_EXECUTED'),gain_pct=None,cost_ratio=None))
    text='# 내부 적응 격차 재확인 — 통합 보고서\n\n'
    text+='[목적] 단순한 출력 보정 뒤에도 실제 내부 적응의 정확도 이득과 비용 문제가 함께 남는지 판단한다. 새 방법론 성공 보고서가 아니다.\n\n'
    text+=markdown_table(rows,['panel','status','gain_pct','cost_ratio'])+'\n\n'
    for name in ctx.cfg['panels']:
        result,_=ctx.paths(name)
        if (result/'REPORT_KO.md').exists():
            text+=f'## {name}\n\n[전체 보고서]({name}/REPORT_KO.md) · [판정]({name}/DECISION.json)\n\n![선택 모델]({name}/figures/test_scores.png)\n\n![비용]({name}/figures/quality_cost.png)\n\n'
        else:
            text+=f'## {name}\n\n미완료. 성능 결과를 만들지 않았다. '
            if (result/'ERROR.json').exists():text+=f'[실행 기록]({name}/ERROR.json)'
            text+='\n\n'
    text+='## 판정 경계\n\n두 자료를 합쳐 우승 점수 하나로 만들지 않는다. 출력 보정으로 충분한 설정은 별도로 남긴다. 격차가 남아도 해당 고정 recipe에서의 개발 근거이며 LoRA+·HEAD의 충분한 튜닝이나 새로운 PEFT 기여가 입증된 것은 아니다.\n\n'
    text+='[환경](ENVIRONMENT.json) · [소스 봉인](SOURCE_SEAL.json) · [검산](VERIFICATION.json) · [학습 장부](UPDATE_LEDGER.jsonl)\n'
    (ctx.out/'SUMMARY_KO.md').write_text(text,encoding='utf-8');write_json(ctx.out/'SUMMARY.json',rows)
