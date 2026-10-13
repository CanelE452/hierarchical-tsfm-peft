"""Re-evaluate saved codecs on fixed past windows; no Bolt or new training."""
from pathlib import Path
import csv
import json
import os
import sys
import time
import traceback

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import torch
from component_experiment import Codec
from diagnose_forecast import ROOT,HERE,OLD,Limit,write_new
from common import access_guard,sha,read,ahash,load_data,windows,masks,score

OUT=HERE/'reconstruction_audit_v1'
CACHE=ROOT/'.cache/forecast_component_study_20261013/reconstruction_audit_v1'
STAGE_B=HERE/'stage_b_full_v1'


class ReconstructionLimit(Limit):
    def status(self):
        return dict(super().status(),cpu_limit_seconds=180,wall_limit_seconds=300)

    def check(self):
        v=self.status()
        if v['cpu_seconds']>=180 or v['wall_seconds']>=300:
            raise RuntimeError('RECONSTRUCTION_CPU180_WALL300_LIMIT')
        if v['peak_sampled_rss_bytes']>6*1024**3:
            raise RuntimeError('RECONSTRUCTION_RSS6GIB_LIMIT')


def independent_score(pred,target,observed):
    channel_mse=[];channel_mae=[]
    for channel in range(target.shape[-1]):
        valid=observed[:,:,channel]
        if not np.any(valid):continue
        difference=pred[:,:,channel][valid].astype(np.float64)-target[:,:,channel][valid].astype(np.float64)
        channel_mse.append(float(np.dot(difference,difference)/len(difference)))
        channel_mae.append(float(np.sum(np.abs(difference))/len(difference)))
    return {'mse':float(np.mean(channel_mse)),'mae':float(np.mean(channel_mae))}


def reconstruction_scores(pred,x,mask,data):
    normalized=score(pred,x,mask)
    physical_pred=pred*data['std']+data['mean']
    physical_target=x*data['std']+data['mean']
    physical=score(physical_pred,physical_target,mask)
    independent={'normalized':independent_score(pred,x,mask),
                 'physical':independent_score(physical_pred,physical_target,mask)}
    for scale,metric in [('normalized',normalized),('physical',physical)]:
        for name in ('mse','mae'):
            if not np.isclose(metric[name],independent[scale][name],rtol=2e-12,atol=1e-9):
                raise AssertionError('Independent reconstruction score mismatch')
    return {'normalized':normalized,'physical':physical,'independent_numpy':independent,'rescore_pass':True}


def latent_stats(z):
    flat=z.reshape(-1,z.shape[-1]).astype(np.float64)
    variance=np.var(flat,axis=0)
    std=np.sqrt(variance)
    return {'finite':bool(np.isfinite(z).all()),'per_channel_variance':variance.tolist(),
            'per_channel_std':std.tolist(),'min_std':float(std.min()),'max_abs':float(np.abs(z).max()),
            'mean':flat.mean(0).tolist(),
            'near_constant_channels_std_lt_1e_minus6':int(np.sum(std<1e-6)),
            'scope':'All positions in fixed windows, repeated timestamps retained; nonzero variance does not prove useful information'}


@torch.inference_mode()
def codec_outputs(codec,x,limit):
    latent=[];reconstructed=[]
    for start in range(0,len(x),32):
        z=codec.encode(torch.from_numpy(x[start:start+32]))
        p=codec.decode(z)
        latent.append(z.numpy());reconstructed.append(p.numpy());limit.check()
    return np.concatenate(reconstructed),np.concatenate(latent)


@torch.inference_mode()
def causal_fixture(codec,x):
    original=torch.from_numpy(x[:2].copy())
    changed=original.clone();changed[:,256:]=changed[:,256:].flip(1)+101
    before=codec.encode(original);after=codec.encode(changed)
    delta=float((before[:,:256]-after[:,:256]).abs().max())
    reconstructed_delta=float((codec.decode(before[:,:256])-codec.decode(after[:,:256])).abs().max())
    positive=original.clone();positive[:,255]+=1
    positive_delta=float((before[:,255]-codec.encode(positive)[:,255]).abs().max())
    if max(delta,reconstructed_delta)>2e-5 or positive_delta<=0:
        raise AssertionError('Past-only reconstruction fixture failed')
    return {'future_suffix_to_earlier_latent_max_abs':delta,'future_suffix_to_earlier_reconstruction_max_abs':reconstructed_delta,
            'change_observed_present_positive_max_abs':positive_delta,'pass':True}


def save_array(path,**arrays):
    with path.open('xb') as f:np.savez(f,**arrays)
    return {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size}


def write_report(result):
    rows=[]
    for case in result['cases']:
        for cp in case['snapshots']:
            for split,values in cp['reconstruction'].items():
                rows.append({'dataset':case['dataset'],'objective':case['objective'],'seed':case['seed'],'update':cp['update'],
                    'split':split,'normalized_mse':values['normalized']['mse'],'normalized_mae':values['normalized']['mae'],
                    'physical_mse':values['physical']['mse'],'physical_mae':values['physical']['mae'],
                    'latent_min_std':cp['latent'][split]['min_std'],'latent_max_abs':cp['latent'][split]['max_abs'],
                    'reused_forecast_tune_physical_mse':cp['reused_forecast_tune']['mse']})
    with (OUT/'metrics.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rec=[c for c in result['cases'] if c['objective']=='REC']
    fit_better=sum(c['snapshots'][-1]['reconstruction']['fit']['normalized']['mse']<c['snapshots'][0]['reconstruction']['fit']['normalized']['mse'] for c in rec)
    tune_better=sum(c['snapshots'][-1]['reconstruction']['tune']['normalized']['mse']<c['snapshots'][0]['reconstruction']['tune']['normalized']['mse'] for c in rec)
    forecast_worse=sum(c['snapshots'][-1]['reused_forecast_tune']['mse']>c['snapshots'][0]['reused_forecast_tune']['mse'] for c in rec)
    lines=['# 고정 과거 창의 재구성 학습 재평가','',
        f"[확인] 40개 저장 codec을 다시 실행했다. REC final600의 정규화 재구성 MSE는 초기값 대비 고정 FIT에서 {fit_better}/4, 고정 TUNE에서 {tune_better}/4 조건이 낮았다. 같은 checkpoint의 미래 TUNE 물리 MSE는 {forecast_worse}/4 조건에서 높아졌다. 아래 원수치와 지표별 차이를 함께 보아야 한다.",'',
        '## REC 초기값과 final600: 같은 입력에서의 비교','',
        '```text','자료   seed     FIT norm MSE (0→600)    TUNE norm MSE (0→600)    미래 TUNE phys MSE (0→600)']
    for c in rec:
        a,b=c['snapshots'][0],c['snapshots'][-1]
        lines.append(f"{c['dataset']:<6} {c['seed']}   {a['reconstruction']['fit']['normalized']['mse']:.6f}→{b['reconstruction']['fit']['normalized']['mse']:.6f}       {a['reconstruction']['tune']['normalized']['mse']:.6f}→{b['reconstruction']['tune']['normalized']['mse']:.6f}       {a['reused_forecast_tune']['mse']:.3f}→{b['reused_forecast_tune']['mse']:.3f}")
    lines+=['```','',
        '학습 목적은 정규화 channel-macro MSE다. 물리 MSE는 채널별 단위가 달라 같은 순위를 보장하지 않는다. [모든 snapshot의 정규화·물리 MSE/MAE](metrics.csv), [채널별 수치와 latent 통계](RESULT.json)에 원값을 남겼다.','',
        '## 해석 범위','',
        '[판단] 같은 고정 입력에서 재구성 오차가 낮아졌는데 미래 예측 오차가 높아진 조건은, 단순한 loss 미감소보다 재구성 목적과 실제 예측 목적이 어긋날 수 있다는 구체적 증거다. 단, 이 결과만으로 encoder의 정보 손실, 잠재 미래의 예측 불가능성, decoder 민감도 중 하나를 유일 원인으로 특정하지는 못한다. 재구성도 나빠진 조건은 학습/일반화 문제를 별도로 남긴다.','',
        'PRED snapshot도 같은 방식으로 평가했지만, PRED는 미래 예측 오차로 E/D를 함께 최적화한 모델이다. 그 D를 과거의 실제 latent에 적용한 재구성 오차는 학습 목표가 아니며, 나쁘다고 PRED의 예측 목적 실패로 판정하지 않는다.','',
        '원래 REC 네 조건의 checkpoint 선택은 미래 TUNE 예측 MSE로 초기0을 선택했다. 이번 고정 재구성 검사로 checkpoint나 방법을 다시 선택하지 않는다. latent의 분산·최소std가 0보다 큰 것은 상수 출력 붕괴 검사이며, 유용한 정보 보존이나 동결 Bolt 호환의 증거는 아니다.','',
        '## 정보 경계와 검산','',
        '- FIT256/TUNE64의 과거512만 E/D에 입력했다. 목표는 각 창의 같은 과거값이고 observed mask만 채점했다. TUNE 과거 창이 FIT와 겹치는 기존 계약을 유지하므로 독립적인 TUNE 시점만의 재구성 평가라고 부르지 않는다.',
        '- 원자료는 기존 단일 bounded TRAIN/VAL 파일을 읽었지만 VAL 창·target·새 점수는 만들지 않았다. TEST guard를 등록했다. Bolt 모델 로드/실행, 추가 학습은 0이다.',
        '- data/source/checkpoint byte SHA 및 Stage B의 고정 origin·FIT mean/std 일치를 확인했다. 저장 배열을 재독해 별도 NumPy 채널 루프로 MSE/MAE를 계산했다. 40개 snapshot의 causal 변형·최근 관측 양성대조도 수행했다.',
        '- 미래 TUNE 예측 값은 기존 Stage B RESULT의 checkpoint metadata를 SHA로 연결해 재사용했다. 이번 CPU 재구성 실행과 GPU에서 생성됐던 미래 예측 실행은 같은 새 실행으로 표현하지 않는다.',
        '- 재구성/latent 배열과 고정 입력/mask는 새 로컬 cache에 저장하고 SHA를 기록했다. 원자료·모델은 공개 저장소 산출물에 복사하지 않았다.',
        '',f"실제 본체 CPU {result['resource']['cpu_seconds']:.3f}초, wall {result['resource']['wall_seconds']:.3f}초, GPU0, threads2. Python import/프로세스 시작은 제외한다. 기존 파일·결과 수정 및 commit/push는 없다.",'',
        '```powershell','.venv/Scripts/python.exe research/forecast_component_study_20261013/diagnose_reconstruction.py','```','',
        '고정 v1 산출물이 존재하면 덮지 않고 중단한다. 새 실행은 별도 버전이 필요하다.','']
    with (OUT/'REPORT_KO.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    limit=ReconstructionLimit();protected=access_guard()
    OUT.mkdir(exist_ok=False);CACHE.mkdir(parents=True,exist_ok=False)
    source=STAGE_B/'RESULT.json';old=read(source)
    result={'status':'RUNNING','source_stage_b':{'path':str(source.relative_to(ROOT)),'sha256':sha(source)},
        'new_training_updates':0,'bolt_instances':0,'new_val_evaluations':0,'device':'CPU',
        'protected_test_guard':protected,'cases':[],'contracts':{},
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'RECONSTRUCTION_PROTOCOL.md',
            HERE/'component_experiment.py',HERE/'diagnose_forecast.py',OLD/'common.py']}}
    try:
        for p,digest in old['source_hashes'].items():
            if p.endswith('component_experiment.py') or p.endswith('common.py'):assert sha(ROOT/p)==digest
        sanity_target=np.array([[[1.,2.],[3.,4.]]],dtype=np.float32)
        sanity_mask=np.ones_like(sanity_target,dtype=bool)
        sanity=independent_score(sanity_target+.25,sanity_target,sanity_mask)
        assert sanity=={'mse':.0625,'mae':.25}
        result['score_fixture']={'known_positive_error':sanity,'pass':True}
        for dataset in ('jena','robin'):
            data=load_data(dataset)
            data['origins']={s:data['origins'][s] for s in ('fit','tune')}
            contract=read(STAGE_B/f'{dataset}_contract.json')
            assert contract['source_sha256']==data['source_hash']
            assert np.array_equal(np.asarray(contract['mean'],dtype=np.float32),data['mean'])
            assert np.array_equal(np.asarray(contract['std'],dtype=np.float32),data['std'])
            for split in ('fit','tune'):
                assert ahash(data['origins'][split])==contract['origin_hashes'][split]
            histories={s:windows(data,s) for s in ('fit','tune')}
            observed={s:masks(data,s,False) for s in ('fit','tune')}
            input_cache=save_array(CACHE/f'{dataset}_fixed_inputs.npz',fit=histories['fit'],tune=histories['tune'],
                                  fit_mask=observed['fit'],tune_mask=observed['tune'],
                                  fit_origins=data['origins']['fit'],tune_origins=data['origins']['tune'],mean=data['mean'],std=data['std'])
            result['contracts'][dataset]={'source':str(Path(data['source']).relative_to(ROOT)),'source_sha256':data['source_hash'],
                'stage_b_contract_sha256':sha(STAGE_B/f'{dataset}_contract.json'),'origins':{s:v.tolist() for s,v in data['origins'].items()},
                'fixed_inputs':input_cache,'normalization':data['normalization_fit_scope'],'channel_names':data['columns'],
                'mean':data['mean'].tolist(),'std':data['std'].tolist(),'evaluation_targets':'Same observed past values only; no new forecast labels'}
            for case in [c for c in old['cases'] if c['dataset']==dataset and c['label'] in ('REC','PRED')]:
                record={'dataset':dataset,'objective':case['label'],'seed':case['seed'],'k':case['k'],'snapshots':[],
                        'original_forecast_selected_update':case['training']['selected_update']}
                codec=Codec(np.zeros(data['x'].shape[1],dtype=np.float32),
                            np.zeros((data['x'].shape[1],case['k']),dtype=np.float32),case['seed'],True).eval().requires_grad_(False)
                for cp in case['training']['checkpoints']:
                    path=ROOT/cp['path'];assert sha(path)==cp['sha256']
                    state=torch.load(path,map_location='cpu',weights_only=True)
                    assert (state['update'],state['seed'],state['objective'])==(cp['update'],case['seed'],case['label'])
                    codec.load_state_dict(state['codec'])
                    outputs={s:codec_outputs(codec,histories[s],limit) for s in ('fit','tune')}
                    for s,(p,z) in outputs.items():
                        assert p.shape==histories[s].shape==observed[s].shape and z.shape==(*p.shape[:2],case['k'])
                        assert np.isfinite(p).all() and np.isfinite(z).all()
                    cached=save_array(CACHE/f"{dataset}_s{case['seed']}_{case['label']}_u{cp['update']}.npz",
                        fit_reconstruction=outputs['fit'][0],tune_reconstruction=outputs['tune'][0],
                        fit_latent=outputs['fit'][1],tune_latent=outputs['tune'][1])
                    with np.load(ROOT/cached['path'],allow_pickle=False) as saved:
                        metrics={s:reconstruction_scores(saved[s+'_reconstruction'],histories[s],observed[s],data) for s in ('fit','tune')}
                        latents={s:latent_stats(saved[s+'_latent']) for s in ('fit','tune')}
                    snap={'update':cp['update'],'checkpoint':{'path':cp['path'],'sha256':cp['sha256']},'arrays':cached,
                          'reconstruction':metrics,'latent':latents,'causal_fixture':causal_fixture(codec,histories['fit']),
                          'reused_forecast_tune':cp['tune'],'forecast_metric_protocol':'REUSED_STAGE_B_GPU_METRIC; no Bolt execution in this audit'}
                    record['snapshots'].append(snap);limit.check()
                result['cases'].append(record)
                write_new(OUT/f"{dataset}_s{case['seed']}_{case['label']}.json",record)
                print(json.dumps({'dataset':dataset,'seed':case['seed'],'objective':case['label'],'snapshots':len(record['snapshots']),
                                  'fit_initial_final':[record['snapshots'][i]['reconstruction']['fit']['normalized']['mse'] for i in (0,-1)],
                                  'tune_initial_final':[record['snapshots'][i]['reconstruction']['tune']['normalized']['mse'] for i in (0,-1)]}),flush=True)
        result['status']='COMPLETE';result['resource']=limit.status()
        write_new(OUT/'RESULT.json',result)
        write_report(result)
        checks={'status':'PASS','snapshot_count':sum(len(c['snapshots']) for c in result['cases']),
                'independent_split_reconstructions':sum(len(cp['reconstruction']) for c in result['cases'] for cp in c['snapshots']),
                'causal_positive_fixtures':40,'known_sign_scale_fixture':True,'result_sha256':sha(OUT/'RESULT.json'),
                'report_sha256':sha(OUT/'REPORT_KO.md'),'metrics_sha256':sha(OUT/'metrics.csv'),
                'all_saved_arrays_sha256_checked':True}
        for c in result['cases']:
            for cp in c['snapshots']:
                assert sha(ROOT/cp['arrays']['path'])==cp['arrays']['sha256']
                limit.check()
        checks['resource_including_report']=limit.status()
        write_new(OUT/'VERIFICATION.json',checks)
        print(json.dumps({'status':'COMPLETE','result':str(OUT/'RESULT.json'),'resource':limit.status()}),flush=True)
    except BaseException as error:
        result['status']='FAILED_PRESERVED';result['error']=str(error);result['traceback']=traceback.format_exc();result['resource']=limit.status()
        write_new(OUT/'FAILURE.json',result);raise


if __name__=='__main__':main()
