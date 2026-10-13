"""Independent saved-prediction scoring and readable Stage A report."""
from pathlib import Path
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/past_only_repr_diagnosis_20261012'))
from common import access_guard, sha, read
import numpy as np
import psutil
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE/'stage_a_checkpoint_readout_v1'


def independent_metrics(prediction,target,observed,mean,std):
    p=np.asarray(prediction*np.asarray(std,dtype=np.float32)+np.asarray(mean,dtype=np.float32),dtype=np.float64)
    y=np.asarray(target*np.asarray(std,dtype=np.float32)+np.asarray(mean,dtype=np.float32),dtype=np.float64)
    errors=np.where(observed,p-y,0.)
    counts=observed.sum((0,1))
    usable=counts>0
    mse=np.sum(errors**2,axis=(0,1))[usable]/counts[usable]
    mae=np.sum(np.abs(errors),axis=(0,1))[usable]/counts[usable]
    return float(mse.mean()),float(mae.mean())


def main():
    start=time.perf_counter()
    proc=psutil.Process();cpu_start=sum(proc.cpu_times()[:2])
    access_guard()
    result=read(OUT/'RESULT.json')
    contracts={d:read(OUT/(d+'_contract.json')) for d in ('jena','robin')}
    checks=[]
    for relative,digest in result['sources'].items():
        assert sha(ROOT/relative)==digest,relative
        checks.append({'type':'source_sha256','path':relative,'pass':True})
    for dataset,c in contracts.items():
        assert sha(ROOT/c['source'])==c['source_sha256']
        assert sha(ROOT/c['contract_path'])==c['source_contract_sha256']
        for split,bounds in (('fit',c['fit_bounds']),('tune',c['tune_bounds']),('val',c['val_bounds'])):
            origins=np.asarray(c['origins'][split])
            assert np.all(origins>=bounds[0]) and np.all(origins+48<=bounds[1])
        checks.append({'type':'bounded_source_and_target_origins','dataset':dataset,'pass':True})
    records=[(v['dataset'],v) for v in result['raw_baselines']]
    for case in result['cases']:
        training=read(ROOT/case['source_training_record'])
        assert sha(ROOT/case['source_training_record'])==case['source_training_sha256']
        for cp in case['checkpoints']:
            assert sha(ROOT/cp['checkpoint_path'])==cp['checkpoint_sha256']
        expected=min(case['checkpoints'],key=lambda v:(v['metrics']['tune']['mse'],v['update']))['update']
        assert expected==case['selected_update']
        assert len(case['unique_training_origins'])==case['unique_training_origin_count']
        assert sum(x['input_samples'] for x in training['training_exposures'])==case['sample_exposures']
        checks.append({'type':'checkpoint_selection_and_exposure','dataset':case['dataset'],'seed':case['seed'],
                       'checkpoint_count':len(case['checkpoints']),'pass':True})
        records += [(case['dataset'],v) for v in case['comparisons']]
    for dataset,record in records:
        c=contracts[dataset]
        path=ROOT/record['prediction']['path']
        assert sha(path)==record['prediction']['sha256']
        with np.load(path,allow_pickle=False) as z:
            assert z['prediction'].shape==z['target'].shape==z['observed'].shape
            assert z['origins'].tolist()==c['origins']['val']
            assert np.isfinite(z['prediction']).all()
            mse,mae=independent_metrics(z['prediction'],z['target'],z['observed'],c['mean'],c['std'])
        assert np.isclose(mse,record['val']['mse'],rtol=1e-12,atol=1e-9),(path,mse,record['val']['mse'])
        assert np.isclose(mae,record['val']['mae'],rtol=1e-12,atol=1e-9)
        checks.append({'type':'independent_prediction_rescore','path':record['prediction']['path'],'pass':True})

    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,case in zip(axes.ravel(),result['cases']):
        updates=[c['update'] for c in case['checkpoints']]
        for split,label,color in [('original_train','Original training origins','#2b8cbe'),('fit','Common FIT 256','#a6bddb'),('tune','TUNE 64','#d95f0e')]:
            values=[c['metrics'][split]['normalized']['mse'] for c in case['checkpoints']]
            ax.plot(updates,values,marker='o',label=label,color=color)
        ax.axvline(case['selected_update'],ls='--',color='#555555',alpha=.7,label='Selected by physical TUNE MSE')
        ax.set(title=f"{case['dataset'].title()} / seed {case['seed']} / original origins n={case['unique_training_origin_count']}",
               xlabel='Saved update',ylabel='Normalized channel-macro MSE')
        ax.grid(alpha=.2)
        if case['dataset']=='robin':
            ax.set_ylim(-.05,2.2)
        ax.legend(fontsize=8)
    fig.suptitle('Saved original FORECAST head: fitting vs development performance',fontsize=14)
    curves=OUT/'stage_a_head_curves.png'
    if curves.exists():raise FileExistsError(curves)
    fig.savefig(curves,dpi=180);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    labels=['Original\nhead','Last-z ridge\nsparse FIT','Last-z ridge\nFIT 256','Last33 ridge\nFIT 256','Raw joint\n63 steps','Raw joint\n95 steps']
    for ax,case in zip(axes.ravel(),result['cases']):
        records600=[v for v in case['comparisons'] if v['update']==600]
        vals=[v['val']['mse'] for v in records600]
        vals.extend(v['val']['mse'] for v in result['raw_baselines'] if v['dataset']==case['dataset'] and v['method'].startswith('RAW_'))
        bars=ax.bar(np.arange(6),vals,color=['#d95f0e','#fdbb84','#2b8cbe','#74a9cf','#777777','#aaaaaa'])
        for bar,value in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{value:,.0f}',ha='center',va='bottom',fontsize=8)
        ax.set(title=f"{case['dataset'].title()} / seed {case['seed']} / checkpoint 600",ylabel='Physical channel-macro MSE',ylim=(0,max(vals)*1.15))
        ax.set_xticks(np.arange(6),labels,fontsize=8)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Readout and fitting-origin changes: common VAL, lower is better',fontsize=14)
    comparisons=OUT/'stage_a_readout_comparison.png'
    if comparisons.exists():raise FileExistsError(comparisons)
    fig.savefig(comparisons,dpi=180);plt.close(fig)

    lines=['# 저장된 FORECAST 실패 위치 진단 — Stage A','',
           '[확인] 기존 encoder/head checkpoint 28개를 실제 재추론했다. 신규 신경망 학습은 없으며, 새 ridge만 FIT에 적합했다. 결과는 기존 VAL을 다시 사용하는 개발 진단이다. 독립 일반화 검증이나 방법군 전체의 실패 판정이 아니다.','',
           '## 먼저 얻은 답','',
           '- **원래 head를 버렸기 때문에만 실패한 것은 아니다.** Jena 최종 checkpoint의 원래 head MSE 32648.064/35010.134보다 공통 FIT256의 last33 ridge 25749.831/25892.494가 낮다. 다만 학습 origin 수, 과거 길이, 정규화/규제가 동시에 달라지므로 ridge 구조만의 이득으로 해석하면 안 된다.',
           '- **Robin에서 원래 예측 모델의 학습은 관측된 TUNE/VAL 일반화를 악화시켰다.** 두 seed 모두 TUNE가 학습 전 update0을 선택했다. 기존 고유 학습 origin 2개는 거의 정확히 맞추지만, 공통 FIT256과 TUNE에서는 좋아지지 않았다. 이는 현재 표본 구성에서의 과적합/분포 차이와 일치하며, 예측 목적 전체가 불가능하다는 뜻이 아니다.',
           '- **학습 표본 증가와 readout 교체를 분리하면 표본 효과가 크다.** 최종 encoder의 last-z ridge를 원래 sparse origins로 fit한 것보다 같은 feature를 공통 FIT256으로 fit한 것이 네 조건에서 모두 MSE가 낮다. encoder 재학습은 하지 않았으므로 새로운 temporal feature 학습의 효과는 아니다.',
           '- **Jena의 저장 head checkpoint를 TUNE로 고르면 final600보다 조금 개선된다.** 그러나 두 seed 모두 raw joint63/95 ridge보다 여전히 나쁘다. early stopping만으로 해결되지 않았다.',
           '- **긴 latent readout의 효과는 일관적이지 않다.** final600의 last33은 last-z보다 Jena 두 seed와 Robin seed92602에서 좋지만 Robin seed92601에서는 나쁘다. 입력길이를 늘리면 항상 좋아진다는 근거가 아니다.','',
           '## 원래 head: TUNE 선택과 final600','',
           '```text','자료   seed    선택 update   선택 VAL MSE/MAE         final600 VAL MSE/MAE']
    for case in result['cases']:
        chosen=next(v for v in case['comparisons'] if v['method']=='ORIGINAL_FORECAST_HEAD' and v['update']==case['selected_update'])
        final=next(v for v in case['comparisons'] if v['method']=='ORIGINAL_FORECAST_HEAD' and v['update']==600)
        lines.append(f"{case['dataset']:<6} {case['seed']}  {case['selected_update']:>7}      {chosen['val']['mse']:>10.3f}/{chosen['val']['mae']:>7.3f}      {final['val']['mse']:>10.3f}/{final['val']['mae']:>7.3f}")
    lines+=['```','',
            '선택은 공통 TUNE64의 물리 단위 channel-macro MSE다. 과거 로그의 소수 고정 batch 정규화 학습 손실 최저점과 선택 기준 및 표본이 다르다. Robin TUNE가 L512보다 짧더라도 목표 구간을 TUNE 안에 두고 입력 과거가 FIT를 포함하도록 허용하는 기존 readout 계약에서는 64개 평가 origin을 확보할 수 있다.','',
            '![원래 head의 학습 창·공통 FIT·TUNE 성능](stage_a_head_curves.png)','',
            '## 같은 최종 encoder에서 head·readout 표본·과거 길이 분리','',
            '```text','자료   seed      head       z1/sparse   z1/FIT256   z33/FIT256   raw63      raw95']
    for case in result['cases']:
        vals=[v['val']['mse'] for v in case['comparisons'] if v['update']==600]
        vals += [v['val']['mse'] for v in result['raw_baselines'] if v['dataset']==case['dataset'] and v['method'].startswith('RAW_')]
        lines.append(f"{case['dataset']:<6} {case['seed']} "+' '.join(f'{v:>11.3f}' for v in vals))
    lines+=['```','',
            'z1/sparse는 encoder를 원래 학습했던 고유 origin(Jena37/Robin2)에 새 ridge를 적합한다. z1/FIT256은 같은 feature와 ridge 목적을 공통 FIT256에 적합한다. 두 조건 모두 같은 TUNE λ grid를 사용하지만 선택 λ는 달라질 수 있다. z33/FIT256은 마지막33 latent를 평탄화한다. Raw joint는 모든 원 채널의 마지막63 또는95 값을 같은 ridge 절차로 사용한다. 신경망과 raw ridge는 함수 용량까지 동일하지 않으며, 단순 대조군의 역할이다.','',
            '![최종 checkpoint의 readout 비교](stage_a_readout_comparison.png)','',
            '## 데이터·검사·재사용 계약','',
            '- 기존 bounded TRAIN/VAL 원자료 SHA와 데이터 계약 SHA를 확인했다. FIT-only fill/scaler, H48, 공통 FIT256/TUNE64, VAL Jena371/Robin30을 그대로 사용했다. TEST 경로는 접근 차단 audit hook으로 보호한다.',
            '- 28 checkpoint의 실제 byte SHA와 step/seed/condition을 확인했다. 원래 head는 모든 checkpoint에서 원래 학습 origin·공통 FIT·TUNE로 평가하고, VAL은 TUNE로 선택한 update와 사전 비교 대상 final600에만 평가했다. 이후 다른 readout도 이 두 checkpoint를 사용하며 VAL로 checkpoint를 재선택하지 않았다.',
            '- 저장 head는 마지막 z 하나만 입력하므로 네트워크 유효 과거 길이는63이다. 마지막33 latent는95이다. full512 대비 tail95/63 parity, 95 이전 변경 무반응, 최근 관측 변경 양성대조를 실제 실행했다. 단, 이는 전처리된 입력에 대한 receptive field이며 missing-value forward-fill은 더 오래된 관측값을 전달할 수 있다. Raw와 latent 모두 같은 과거 전처리를 사용했다.',
            '- 기존 RESULT의 공통ridge 점수는 REUSED_RECORDED_RESULT로 보존했다. 이번 CPU 재추론·ridge 재적합 결과는 별도다. Jena 기존 prediction과 최대 정규화 차이는 약8e-8이다. Robin은 이전 GPU 결과와 최대0.00275/0.00323, 물리 MSE0.05055/0.19304 차이가 있어 byte/수치 완전동일 재현이라고 말하지 않는다. 새 비교들은 모두 동일 CPU 경로다.',
            '- 학습/평가 origin들은 시간적으로 겹칠 수 있다. 256/64/371/30은 독립 표본 수가 아니다. 물리 MSE는 단위가 다른 채널들의 평균이므로 [원수치 CSV](metrics.csv)에 정규화 MSE/MAE도 동반했다.',
            '- 38개 신규 VAL 예측 파일의 SHA, target/origin/mask 형상, 독립적인 MSE/MAE 재계산을 검사했다. 원자료/체크포인트 및 결과 연결은 [검산 기록](VERIFICATION.json)과 [전체 결과](RESULT.json)에 남겼다.','',
            '## 다음 실험에 주는 제약','',
            '먼저 신경망 FORECAST를 동일 FIT256 origin과 충분한 causal context로 학습하고, 원래 head를 유지한 TUNE 선택을 포함해야 한다. 그 뒤 적은 성분→동결 Bolt→복원과 잔차 보정을 비교한다. 원자료 직접 예측, 잔차 모델 단독, 압축 Bolt 단독, 결합, 원자료 Bolt+동일 보정 모델을 두어 보정 모델이 혼자 이득을 만든 경우와 압축 경로의 추가 기여를 구분한다. 이 Stage A는 해당 전체 결합의 가능성이나 비용 개선을 아직 입증하지 않았다.','',
            '## 실행과 재현','',
            '```powershell',
            '.venv/Scripts/python.exe research/forecast_component_study_20261013/diagnose_forecast.py --tag NEW_UNIQUE_TAG',
            '.venv/Scripts/python.exe research/forecast_component_study_20261013/stage_a_verify_and_report.py',
            '```','',
            '실험 실행은 기존 파일을 덮지 않는 고유 tag가 필요하다. 보고/검산 스크립트는 봉인된 checkpoint_readout_v1을 대상으로 하며 기존 보고 출력이 있으면 중단한다. 체크포인트와 원자료는 로컬 비공개 cache에 있어 저장소만으로 모델 재현이 완료되지는 않는다.','',
            f"실제 분석 본체 비용: CPU {result['resource']['cpu_seconds']:.3f}초, wall {result['resource']['wall_seconds']:.3f}초, GPU0; PyTorch threads2. 실행 정상 종료. 이 수치는 모델 warm inference 지연 비교가 아니라 데이터 로딩·checkpoint 평가·ridge 적합·결과 저장을 포함한 Stage A 작업 시간이며 Python import/프로세스 시작 시간은 제외한다.",
            '']
    report=OUT/'REPORT_KO.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    receipt={'status':'PASS','result_sha256':sha(OUT/'RESULT.json'),'script_sha256':sha(Path(__file__)),
             'checks':checks,'prediction_count':len(records),'checkpoint_count':28,
             'figures':{p.name:sha(p) for p in [curves,comparisons]},'report_sha256':sha(report),
             'resources':{'cpu_seconds':sum(proc.cpu_times()[:2])-cpu_start,'wall_seconds':time.perf_counter()-start,'gpu_seconds':0},
             'scope':'Saved prediction rescoring and binding; no independent TEST or new model training'}
    with (OUT/'VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2,ensure_ascii=False)
    print(json.dumps({'status':'PASS','checks':len(checks),'predictions':len(records),'resources':receipt['resources']}))


if __name__=='__main__':
    main()
