"""Verify sparse/dense paired FORECAST experiment and write its report."""
from pathlib import Path
import csv
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/past_only_repr_diagnosis_20261012'))
from common import access_guard,read,sha,ahash
from stage_a_verify_and_report import independent_metrics
import numpy as np
import psutil
from matplotlib import pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE/'stage_a2_dense_pool_v1'


def main():
    start=time.perf_counter();proc=psutil.Process();cpu_start=sum(proc.cpu_times()[:2])
    access_guard()
    result=read(OUT/'RESULT.json')
    checks=[];metrics=[]
    for path,digest in result['source_sha256'].items():
        assert sha(ROOT/path)==digest
        checks.append({'type':'source_sha256','path':path,'pass':True})
    for dataset,contract in result['contracts'].items():
        assert sha(ROOT/contract['source'])==contract['source_sha256']
        for split,bounds in contract['bounds'].items():
            origins=np.asarray(contract['origins'][split])
            assert np.all(origins>=bounds[0]) and np.all(origins+48<=bounds[1])
        checks.append({'type':'source_and_origin_bounds','dataset':dataset,'pass':True})
    predictions=0;checkpoints=0
    for case in result['cases']:
        source=case['source_training_record']
        assert sha(ROOT/source['path'])==source['sha256']
        old=read(ROOT/source['path'])
        assert sha(case['initial_checkpoint']['path'])==case['initial_checkpoint']['sha256']
        assert case['initial_checkpoint']['sha256']==old['checkpoints'][0]['sha256']
        assert case['gradient_finite_checks']==600
        assert len(case['losses'])==600 and np.isfinite(case['losses']).all()
        batch_file=case['batch_file']
        assert sha(ROOT/batch_file['path'])==batch_file['sha256']
        with np.load(ROOT/batch_file['path'],allow_pickle=False) as z:batches=z['origins']
        assert ahash(batches)==case['batch_order_hash']
        assert len(np.unique(batches))==case['unique_origin_count']
        if case['pool']=='sparse_replay':
            actual=np.asarray([v['origins'] for v in old['training_exposures']],dtype=np.int64)
            assert np.array_equal(batches,actual)
        else:
            assert sorted(np.unique(batches).tolist())==result['contracts'][case['dataset']]['origins']['fit']
        assert len(case['checkpoints'])==7
        for cp in case['checkpoints']:
            assert sha(ROOT/cp['path'])==cp['sha256'];checkpoints+=1
        selected=min(case['curves'],key=lambda v:(v['tune']['mse'],v['update']))['update']
        assert selected==case['selected_update']
        assert {v['update'] for v in case['evaluations']}=={selected,600}
        checks.append({'type':'training_and_TUNE_selection','dataset':case['dataset'],'seed':case['seed'],'pool':case['pool'],'pass':True})
        contract=result['contracts'][case['dataset']]
        for evaluation in case['evaluations']:
            p=evaluation['prediction'];assert sha(ROOT/p['path'])==p['sha256']
            with np.load(ROOT/p['path'],allow_pickle=False) as z:
                assert z['origins'].tolist()==contract['origins']['val']
                mse,mae=independent_metrics(z['prediction'],z['target'],z['observed'],contract['mean'],contract['std'])
            assert np.isclose(mse,evaluation['val']['mse'],rtol=1e-12,atol=1e-9)
            assert np.isclose(mae,evaluation['val']['mae'],rtol=1e-12,atol=1e-9)
            predictions+=1
            checks.append({'type':'independent_prediction_rescore','path':p['path'],'pass':True})
            metrics.append({'dataset':case['dataset'],'seed':case['seed'],'pool':case['pool'],
                            'unique_training_origins':case['unique_origin_count'],'update':evaluation['update'],
                            'selected_by_TUNE':evaluation['update']==selected,
                            'val_mse':mse,'val_mae':mae,'val_normalized_mse':evaluation['val']['normalized']['mse'],
                            'val_normalized_mae':evaluation['val']['normalized']['mae'],
                            'tune_mse':evaluation['tune']['mse']})
    pairs=[]
    for dataset in ('jena','robin'):
        for seed in (92601,92602):
            pair={c['pool']:c for c in result['cases'] if c['dataset']==dataset and c['seed']==seed}
            sparse,dense=pair['sparse_replay'],pair['dense256']
            for key in ('initial_checkpoint','encoder_initial_state_hash','head_initial_state_hash','actual_sample_exposures','batch_size'):
                assert sparse[key]==dense[key],key
            chosen={p:next(v for v in case['evaluations'] if v['update']==case['selected_update']) for p,case in pair.items()}
            final={p:next(v for v in case['evaluations'] if v['update']==600) for p,case in pair.items()}
            pairs.append({'dataset':dataset,'seed':seed,'cases':pair,'selected':chosen,'final':final})
            checks.append({'type':'paired_initialization_and_exposure','dataset':dataset,'seed':seed,'pass':True})
    with (OUT/'metrics.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)

    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,pair in zip(axes.ravel(),pairs):
        for pool,color,label in [('sparse_replay','#d95f0e','Original sparse pool'),('dense256','#2b8cbe','Common FIT256 pool')]:
            case=pair['cases'][pool]
            xs=[v['update'] for v in case['curves']];ys=[v['tune']['mse'] for v in case['curves']]
            ax.plot(xs,ys,marker='o',color=color,label=label)
            k=xs.index(case['selected_update']);ax.scatter([xs[k]],[ys[k]],s=150,facecolor='none',edgecolor=color,linewidth=2)
        ax.set(title=f"{pair['dataset'].title()} / seed {pair['seed']}",xlabel='Update',ylabel='Physical TUNE MSE')
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Same FORECAST model and budget: change only the training-origin pool',fontsize=14)
    curve=OUT/'stage_a2_tune_curves.png'
    if curve.exists():raise FileExistsError(curve)
    fig.savefig(curve,dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,dataset in zip(axes,('jena','robin')):
        selected_pairs=[p for p in pairs if p['dataset']==dataset]
        positions=np.arange(2)
        for offset,pool,color,label in [(-.18,'sparse_replay','#d95f0e','Sparse pool'),(.18,'dense256','#2b8cbe','FIT256 pool')]:
            ys=[p['selected'][pool]['val']['mse'] for p in selected_pairs]
            bars=ax.bar(positions+offset,ys,width=.34,color=color,label=label)
            for bar,y in zip(bars,ys):ax.text(bar.get_x()+bar.get_width()/2,y,f'{y:,.1f}',ha='center',va='bottom',fontsize=9)
        ax.set_xticks(positions,['seed92601','seed92602']);ax.set(title=dataset.title(),ylabel='Physical VAL MSE')
        ax.set_ylim(0,max(p['selected']['sparse_replay']['val']['mse'] for p in selected_pairs)*1.15)
        ax.grid(axis='y',alpha=.2);ax.legend()
    fig.suptitle('TUNE-selected checkpoints: same original forecasting head',fontsize=14)
    bar=OUT/'stage_a2_selected_val.png'
    if bar.exists():raise FileExistsError(bar)
    fig.savefig(bar,dpi=180);plt.close(fig)

    lines=['# FORECAST 학습 origin pool의 영향 — Stage A2','',
           '[확인] 원래 FORECAST 구조·초기 encoder/head·loss·batch size·600 update를 유지하고 학습 origin pool만 기존 sparse(Jena37/Robin2)에서 공통 FIT256으로 바꿨다. 두 자료·두 seed 모두 TUNE로 선택한 원래 head의 VAL MSE가 개선됐다. **현재 부진에서 학습 origin pool 설계가 상당한 역할을 했다는 직접적인 대조 증거**다. 미래 표현 정렬이나 새 압축기를 넣어서 얻은 개선이 아니다.','',
           '## TUNE 선택 결과','',
           '```text','자료   seed   sparse update  dense update   sparse VAL MSE/MAE       dense VAL MSE/MAE       MSE 변화']
    for pair in pairs:
        a,b=pair['selected']['sparse_replay'],pair['selected']['dense256']
        delta=100*(b['val']['mse']/a['val']['mse']-1)
        lines.append(f"{pair['dataset']:<6} {pair['seed']} {a['update']:>13} {b['update']:>13}  {a['val']['mse']:>10.3f}/{a['val']['mae']:>7.3f}    {b['val']['mse']:>10.3f}/{b['val']['mae']:>7.3f}     {delta:>7.2f}%")
    lines+=['```','',
            '선택은 공통 TUNE64의 물리 단위 channel-macro MSE로만 했다. dense 네 조건 모두 update500이 선택됐다. Robin sparse는 두 seed 모두 update0을 선택했고 dense에서는 학습 후 모델이 선택됐다. VAL은 선택 checkpoint와 사전에 정한 final600에만 평가했다.','',
            '![TUNE로 선택한 원래 head의 VAL 성능](stage_a2_selected_val.png)','',
            '## 동일한 final600 비교','',
            '```text','자료   seed      sparse MSE      dense MSE       sparse MAE    dense MAE']
    for pair in pairs:
        a,b=pair['final']['sparse_replay']['val'],pair['final']['dense256']['val']
        lines.append(f"{pair['dataset']:<6} {pair['seed']}    {a['mse']:>11.3f}    {b['mse']:>11.3f}       {a['mae']:>8.3f}     {b['mae']:>8.3f}")
    lines+=['```','',
            'final600에서도 네 조건 모두 MSE/MAE가 개선되어 checkpoint 선택 차이만으로 설명되는 결과가 아니다. dense가 최종까지 계속 좋아지는 것은 아니므로 더 오래 학습하면 무조건 개선된다고 해석하지 않는다.','',
            '![같은 예측 head와 학습 예산에서 origin pool만 바꾼 TUNE 곡선](stage_a2_tune_curves.png)','',
            '## 무엇을 고정했고 무엇이 달라졌는가','',
            '- 초기값: 각 seed의 기존 E_u0000 실제 encoder/head와 byte SHA를 그대로 사용했다. paired sparse/dense의 encoder/head 초기 state hash가 동일하다.',
            '- 구조: causal RF63, latent16, 마지막 z→48*C 선형 head, 입력512. 채널 수·데이터·FIT-only 정규화·missing fill·H48을 유지했다.',
            '- 학습: Adam lr0.001, 600 update, Jena batch8 / Robin batch2, 기존 observed-element masked MSE. gradient clipping이나 목적 함수를 추가하지 않았다. 총4800개 update의 loss·parameter gradient finite 검사를 통과했다.',
            '- sparse: 저장된 기존 600개 batch origin 배열을 실제 그대로 재생했다. 원래 make_batches와 seed+31로 만든 배열과도 동일함을 확인했다. dense: 같은 make_batches·seed+31을 공통 FIT256 pool에 적용했다. 배치당 표본 수와 총 표본 노출 수는 paired 조건에서 같다.',
            '- **origin pool의 크기·위치·중첩도가 바뀐다.** 이를 256개 독립 표본 확보나 오직 독립 표본 수만의 효과로 바꾸어 말하지 않는다. 같은 원자료에서 더 많은 예측 시작점을 학습에 노출한 효과다.',
            '- 신경망 재학습 경로는 모두 새 CPU 실행이다. 기존 Jena CPU/Robin GPU 결과를 그대로 새 sparse 결과로 사용하지 않았다. 기존 sparse 최종 파라미터와 CPU replay의 실제 차이는 RESULT의 original_final_comparison에 남겼다. 두CPUthread/기존GPU 커널의 수치 궤적 차이를 원본 결과 오류로 단정하지 않는다.','',
            '## 해석과 남은 문제','',
            '[판단] 기존 FORECAST가 나빴다는 결과만으로 예측 목적 학습을 기각하거나, 재구성 대신 복잡한 대조학습이 반드시 필요하다고 판단하면 안 된다. 동일 예측 목적도 기존 sparse pool을 고치면 크게 개선된다. 이 결과는 목적 함수보다 학습 origin 설계가 먼저 점검되어야 했다는 근거다.',
            '',
            'Jena dense의 선택 MSE12887.576/15812.935는 앞 Stage A raw joint63/95 ridge(22766.847/20010.149)보다 낮다. 하지만 Robin dense646.548/819.041은 기존 raw SELF33 ridge512.196보다 높아, 단순 대안보다 항상 좋은 단계는 아니다. rawSELF33은 관측 채널/과거 범위/모델 용량이 동일한 비교가 아니라 저비용 실용 대조군이다.',
            '',
            '이 모델은 원래 채널의 미래를 직접 출력한다. **잠재 시계열의 미래를 동결 Bolt가 잘 예측한다는 증거는 아직 아니다.** 따라서 Stage B에서 예측 성분→동결 Bolt→복원/보정 경로를 별도로 시험하고, 이 dense 직접 예측 모델을 원자료 신경망 대조군으로 남기는 것이 타당하다.',
            '',
            '## 보존·검산·비용','',
            f"8개 신규 학습이 정상 종료했다. 모델56개 checkpoint, 신규 VAL예측{predictions}개와 source/data/초기값 SHA를 검증하고 예측의 물리 MSE/MAE를 별도 계산으로 검산했다. [검산 기록](VERIFICATION.json), [전체 결과](RESULT.json), [원수치 CSV](metrics.csv).",'',
            f"실제 학습/평가 본체 CPU {result['resource']['cpu_seconds']:.3f}초, wall {result['resource']['wall_seconds']:.3f}초, GPU0, PyTorch threads2. CPU300초 상한 이내. Python import/프로세스 시작은 별도이며 모델 순수 추론 지연으로 해석하지 않는다.",
            '',
            'TEST 접근 차단, 기존 source/checkpoint/결과 보존, 새 cache만 기록했다. VAL은 반복 개발 자료이며 독립 일반화 검증은 아니다. 신경망 학습 초기값·배치·새 checkpoint는 로컬 cache 자산이므로 저장소 코드만으로 모든 자산이 배포되는 것은 아니다.','',
            '```powershell',
            '.venv/Scripts/python.exe research/forecast_component_study_20261013/diagnose_dense_forecast.py --tag NEW_UNIQUE_TAG',
            '```','',
            '기존 결과를 덮지 않도록 새 tag를 사용해야 한다. 이번 보고는 dense_pool_v1의 봉인된 결과를 대상으로 한다.','']
    report=OUT/'REPORT_KO.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    receipt={'status':'PASS','result_sha256':sha(OUT/'RESULT.json'),'script_sha256':sha(Path(__file__)),
             'checks':checks,'checkpoint_count':checkpoints,'prediction_count':predictions,
             'gradient_finite_checks':sum(c['gradient_finite_checks'] for c in result['cases']),
             'paired_initialization_checks':4,'report_sha256':sha(report),
             'figures':{p.name:sha(p) for p in (curve,bar)},
             'resource':{'cpu_seconds':sum(proc.cpu_times()[:2])-cpu_start,'wall_seconds':time.perf_counter()-start,'gpu_seconds':0}}
    with (OUT/'VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2,ensure_ascii=False)
    print(json.dumps({'status':'PASS','checks':len(checks),'checkpoint_count':checkpoints,'prediction_count':predictions,'resource':receipt['resource']}))


if __name__=='__main__':main()
