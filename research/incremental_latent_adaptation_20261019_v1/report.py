"""Public-table-only scientific report and deterministic figure replay."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import common


CORE=('G','ALPHA_fitted','READOUT_ONLY','RESIDUAL_F','RESIDUAL_Q','RESIDUAL_RAW_MLP')
ORDER=(*CORE,'S','RAW','ZERO','MEAN','OWN','ALL','SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX')
COLORS=('#000000','#E69F00','#009E73','#0072B2','#D55E00','#CC79A7','#56B4E9')
LABELS={'ALPHA_fitted':'G + alpha (S-G)','READOUT_ONLY':'Readout','RESIDUAL_F':'Residual F',
        'RESIDUAL_Q':'Residual Q','RESIDUAL_RAW_MLP':'Raw residual MLP'}
NA=('','NA','N/A','None','null','nan')


def _csv(path):
    if not path.is_file():
        return []
    with path.open(encoding='utf8',newline='') as stream:
        return list(csv.DictReader(stream))


def _num(row,*keys):
    for key in keys:
        value=row.get(key)
        if value is not None and str(value) not in NA:
            try:
                value=float(value)
            except (ValueError,TypeError):
                continue
            if math.isfinite(value):
                return value
    return None


def _fmt(value,digits=6):
    return 'N/A' if value is None else format(float(value),'.'+str(digits)+'g')


def _display(value):
    if value is None or (isinstance(value,str) and value in NA):
        return 'N/A'
    return json.dumps(value,ensure_ascii=False) if isinstance(value,(list,dict)) else str(value)


def _actual(rows):
    return [r for r in rows if str(r.get('seed','')).lower() not in ('mean','average','two_seed_mean')
            and r.get('row_type','') not in ('mean','two_seed_mean')]


def _v(rows):
    return [r for r in _actual(rows) if r.get('split','V_select') in ('V','V_select')]


def _series(rows,dataset,method,metric):
    return [(_num(r,metric),r) for r in rows if r.get('dataset')==dataset and r.get('method')==method
            and _num(r,metric) is not None]


def _mean(rows,dataset,method,metric='mse'):
    values=[v for v,r in _series(rows,dataset,method,metric)]
    return math.fsum(values)/len(values) if values else None


def _style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,
        'svg.hashsalt':'incremental_latent_adaptation_20261019_v1','svg.fonttype':'none',
        'savefig.facecolor':'white','figure.facecolor':'white'})


def _save(fig,out,name,caption,figures):
    fig.tight_layout(rect=(0,.07,1,.95))
    fig.text(.01,.006,textwrap.fill(caption,145),fontsize=7,va='bottom')
    for extension in ('png','svg'):
        path=out/'figures'/(name+'.'+extension)
        metadata={'Software':'incremental-public-replay'} if extension=='png' else {
            'Date':'2026-10-19T00:00:00Z','Creator':'incremental-public-replay','Title':name}
        fig.savefig(path,dpi=300,metadata=metadata)
        figures.append(dict(file=path.relative_to(out).as_posix(),sha256=common.sha(path),caption=caption))
    plt.close(fig)


def _point(points,figure,panel,**row):
    points.append(dict(figure=figure,panel=panel,**row))


def _empty(ax,label='N/A: no measured public values'):
    ax.text(.5,.5,label,ha='center',va='center',transform=ax.transAxes)
    ax.set_xticks([])


def _accuracy(rows,dataset,out,points,figures):
    name='F1_accuracy_'+dataset
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    fig.suptitle(dataset+' | fixed development V')
    for ax,metric in zip(axes,('mse','mae')):
        n=0
        for index,method in enumerate(CORE):
            values=_series(rows,dataset,method,metric)
            for j,(value,row) in enumerate(values):
                offset=(j-(len(values)-1)/2)*.12
                ax.scatter(index+offset,value,color=COLORS[index],marker=('o','s')[j%2],s=32,zorder=3)
                _point(points,name,metric,dataset=dataset,method=method,seed=row.get('seed','NA'),
                       split='V_select',metric=metric,value=value,x=index+offset,source='RUNS.csv')
            if values:
                vv=[v for v,r in values]
                ax.plot([index,index],[min(vv),max(vv)],color=COLORS[index],linewidth=1)
                ax.plot(index,math.fsum(vv)/len(vv),marker='_',color=COLORS[index],markersize=14)
                n+=1
        if not n:
            _empty(ax)
        ax.set_xticks(range(len(CORE)),[LABELS.get(m,m) for m in CORE],rotation=28,ha='right')
        ax.set_ylabel('Normalized '+metric.upper()+' (lower is better)')
        ax.set_title(metric.upper())
    _save(fig,out,name,'Markers: individual seeds. Vertical span: observed seed min/max, not a confidence interval. No V reselection.',figures)


def _cost(rows,dataset,device,batch,out,points,figures):
    name=f'F1_cost_{dataset}_{device}_B{batch}'
    selected=[r for r in _actual(rows) if r.get('dataset')==dataset and str(r.get('batch_size'))==str(batch)
              and r.get('device','').split(':')[0]==device]
    methods=list(CORE) if device=='cuda' else [m for m in ('G','MEAN','OWN','ALL') if any(r.get('method')==m for r in selected)]
    if not methods:
        methods=['G','MEAN','OWN','ALL']
    fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    fig.suptitle(f'{dataset} | {device.upper()} | batch {batch} | whole physical path')
    for ax,metric,keys,label in (
        (axes[0],'median_batch_ms',('median_batch_ms','median_ms_per_batch','p50_batch_ms'),'Median time (ms/batch)'),
        (axes[1],'peak_allocated_MiB',('peak_allocated_MiB','GPU_peak_allocated_MiB') if device=='cuda' else ('CPU_RSS_sampled_peak_MiB','CPU_RSS_MiB'),
         'Peak allocated tensor memory (MiB)' if device=='cuda' else 'Sampled process RSS (MiB)')):
        n=0
        for index,method in enumerate(methods):
            candidates=[r for r in selected if r.get('method')==method]
            for j,row in enumerate(candidates):
                value=_num(row,*keys)
                if value is None:
                    continue
                x=index+(j-(len(candidates)-1)/2)*.12
                ax.scatter(x,value,color=COLORS[index%len(COLORS)],marker=('o','s')[j%2],s=30)
                _point(points,name,metric,dataset=dataset,device=device,batch_size=batch,method=method,
                       seed=row.get('seed','NA'),metric=metric if device=='cuda' or ax is axes[0] else 'CPU_RSS_sampled_peak_MiB',
                       value=value,x=x,source='COST.csv')
                n+=1
        if not n:
            _empty(ax)
        ax.set_xticks(range(len(methods)),[LABELS.get(m,m) for m in methods],rotation=28,ha='right')
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    caption='Separate device/batch facet. Fresh processes; 10 warmups, 20 timed trials. B8/origin is throughput-normalized, not single-request latency.'
    _save(fig,out,name,caption,figures)


def _sufficient(path):
    """Aggregate public per-cell sums; raw targets/forecasts are never opened."""
    grouped={}
    if not path.is_file():
        return grouped
    with gzip.open(path,'rt',encoding='utf8',newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('split') not in ('V','V_select'):
                continue
            key=(row['dataset'],row['method'],row.get('seed','NA'),row['run_id'])
            if key not in grouped:
                grouped[key]=dict(h=defaultdict(lambda:np.zeros(3)),o=defaultdict(lambda:np.zeros(3)),channels={})
            record=grouped[key]
            h,c,o=int(row['horizon']),int(row['channel_index']),int(row['origin'])
            sums=np.array([float(row['sse']),float(row['sae']),int(row['count'])])
            if sums[2]<0 or not np.isfinite(sums).all():
                raise ValueError('Invalid public sufficient statistic')
            record['h'][(h,c)]+=sums
            record['o'][(o,c)]+=sums
            record['channels'][c]=row.get('channel_id',str(c))
    return grouped


def _macro(stats):
    return math.fsum(float(s[0])/float(s[2]) for s in stats if s[2]>0)/sum(s[2]>0 for s in stats) if any(s[2]>0 for s in stats) else None


def _dimensions(record):
    hc,oc=record['h'],record['o']
    channels=sorted({c for h,c in hc})
    horizons=sorted({h for h,c in hc})
    origins=sorted({o for o,c in oc})
    channel={c:sum((s for (h,cc),s in hc.items() if cc==c),np.zeros(3)) for c in channels}
    horizon={h:[hc[(h,c)] for c in channels if (h,c) in hc] for h in horizons}
    blocks={}
    for b,chunk in enumerate(np.array_split(origins,4),1):
        cohort=set(chunk.tolist())
        blocks[b]=[sum((s for (o,cc),s in oc.items() if cc==c and o in cohort),np.zeros(3)) for c in channels]
    return {'channel':{c:_macro([s]) for c,s in channel.items()},
            'horizon':{h:_macro(s) for h,s in horizon.items()},
            'time_block':{b:_macro(s) for b,s in blocks.items()}}


def _heterogeneity(stats,dataset,out,points,figures,warnings):
    name='F2_gains_'+dataset
    fig,axes=plt.subplots(1,3,figsize=(11,3.5))
    fig.suptitle(dataset+' | signed development-V error changes')
    candidates=[(k,v) for k,v in stats.items() if k[0]==dataset and k[1] in ('S','OWN','ALL')]
    baseline=next((v for k,v in stats.items() if k[0]==dataset and k[1]=='G'),None)
    references=[]
    if baseline is not None:
        references=[('G',baseline,k,v) for k,v in candidates]
    own=next((v for k,v in candidates if k[1]=='OWN'),None)
    if own is not None:
        references += [('OWN',own,k,v) for k,v in candidates if k[1]=='ALL']
    for j,(base,b,k,value) in enumerate(references):
        if set(b['o'])!=set(value['o']) or any(b['o'][idx][2]!=value['o'][idx][2] for idx in b['o']):
            warnings.append('Unpaired heterogeneity cohort: '+str(k))
            continue
        bd,vd=_dimensions(b),_dimensions(value)
        label=f'{base} -> {k[1]} / {k[2]}'
        for ax,dimension in zip(axes,('channel','horizon','time_block')):
            x,y=[],[]
            for index,v in vd[dimension].items():
                bv=bd[dimension].get(index)
                if bv is None or v is None:
                    continue
                x.append(index if dimension=='time_block' else index+1)
                y.append(bv-v)
                _point(points,name,dimension,dataset=dataset,method=k[1],seed=k[2],baseline=base,
                       metric='MSE_gain',value=bv-v,x=x[-1],label=value['channels'].get(index,'') if dimension=='channel' else '',
                       source='ERROR_SUFFICIENT_STATS.csv.gz')
            if x:
                ax.plot(x,y,label=label,color=COLORS[j%len(COLORS)],linestyle=('-','--',':','-.')[j%4],
                        marker=('o','s','^','D')[j%4],markersize=3)
    for ax,dimension in zip(axes,('channel','horizon','time_block')):
        ax.axhline(0,color='#777777',linewidth=.7)
        ax.set_xlabel({'channel':'Original channel index (1-based)','horizon':'Forecast step (1-based)',
                       'time_block':'Chronological origin block (1-4)'}[dimension])
        ax.set_ylabel('Baseline MSE - candidate MSE')
        if references:
            ax.legend(frameon=False,fontsize=6)
        else:
            _empty(ax)
    _save(fig,out,name,'Positive helps; negative harms. Equal-channel observed macro scores; blocks are descriptive, never reselection. OWN/ALL dimensions differ.',figures)


def _curves(report):
    rows=[]
    for path in sorted((report/'runs').rglob('search_cap*.json')):
        match=re.search(r'search_cap(\d+)',path.name)
        relative=path.relative_to(report).as_posix()
        dataset=next((d for d in common.DATASETS if d in path.parts),None)
        method=next((m for m in common.METHODS if any(part.startswith(m+'_s') for part in path.parts)),None)
        if dataset is None or method is None:
            continue
        result=common.read(path)
        for curve in result.get('curves',[]):
            rows.append(dict(curve,dataset=dataset,method=method,seed=result.get('seed','NA'),
                             cap=int(match.group(1)),source=relative,selected_epoch=result.get('selected_epoch')))
    latest={}
    for row in rows:
        key=(row['dataset'],row['method'],str(row['seed']))
        latest[key]=max(latest.get(key,0),row['cap'])
    return [r for r in rows if r['cap']==latest[(r['dataset'],r['method'],str(r['seed']))]]


def _training_receipts(report):
    latest={}
    for path in sorted((report/'runs').rglob('*_cap*.json')):
        match=re.fullmatch(r'(search|refit)_cap(\d+)\.json',path.name)
        if match is None:
            continue
        row=common.read(path)
        key=(row.get('dataset'),row.get('method'),row.get('seed'),match.group(1))
        cap=int(match.group(2))
        if key not in latest or cap>latest[key][0]:latest[key]=(cap,row)
    return [(key,row) for key,(cap,row) in sorted(latest.items(),key=lambda item:str(item[0]))]


def _learning(curves,dataset,out,points,figures):
    name='F4_learning_'+dataset
    fig,axes=plt.subplots(2,2,figsize=(9,6))
    fig.suptitle(dataset+' | correction search; actual M_fit / M_select curves')
    for ax,method in zip(axes.flat,common.METHODS):
        rows=[r for r in curves if r['dataset']==dataset and r['method']==method]
        seeds=sorted({str(r['seed']) for r in rows})
        for i,seed in enumerate(seeds):
            selected=sorted([r for r in rows if str(r['seed'])==seed],key=lambda r:_num(r,'epoch','endpoint_epochs'))
            for split,style in (('M_fit','-'),('M_select','--')):
                valid=[r for r in selected if _num(r,split+'_mse') is not None]
                x=[_num(r,'epoch','endpoint_epochs') for r in valid]
                y=[_num(r,split+'_mse') for r in valid]
                ax.plot(x,y,color=COLORS[(i+3)%len(COLORS)],linestyle=style,marker=('o','s')[i%2],markersize=3,label=seed+' '+split)
                for r,xx,yy in zip(valid,x,y):
                    chosen=_num(r,'selected_epoch')
                    _point(points,name,method,dataset=dataset,method=method,seed=seed,split=split,
                           metric='mse',value=yy,x=xx,cap=r['cap'],selected=chosen==xx,source=r['source'])
                    if chosen==xx and split=='M_select':
                        ax.scatter(xx,yy,facecolors='none',edgecolors='black',s=85,zorder=5)
        if not rows:
            _empty(ax,'N/A: no completed measured search curve')
        ax.set_title(LABELS.get(method,method))
        ax.set_xlabel('Correction epoch (M_fit origins)')
        ax.set_ylabel('Normalized MSE')
        if rows:
            ax.legend(frameon=False,fontsize=7)
    _save(fig,out,name,'Solid: M_fit; dashed: M_select. Ring: selected checkpoint. CP0 retained. Reaching the cap is not evidence of convergence.',figures)


def _prefix(rows,costs,dataset,out,points,figures):
    selected=[r for r in _v(rows) if r.get('dataset')==dataset]
    if not selected:
        return
    name='F3_prefix_'+dataset
    fig,axes=plt.subplots(2,2,figsize=(9,6))
    fig.suptitle(dataset+' | conditional block comparisons; all reported prefixes')
    for i,method in enumerate(('SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX')):
        for metric,ax in zip(('mse','mae'),axes[0]):
            method_rows=[r for r in selected if r.get('method')==method]
            for seed in sorted({str(r.get('seed','NA')) for r in method_rows}):
                rr=sorted([r for r in method_rows if str(r.get('seed','NA'))==seed],key=lambda r:_num(r,'prefix') or 0)
                xx=[_num(r,'prefix') for r in rr];yy=[_num(r,metric) for r in rr]
                if all(v is not None for v in (*xx,*yy)):
                    ax.plot(xx,yy,marker='o',linestyle=('-','--','-.')[i],color=COLORS[i+2],label=method+'/'+seed)
                    for r,x,y in zip(rr,xx,yy):
                        _point(points,name,metric,dataset=dataset,method=method,seed=seed,prefix=x,metric=metric,value=y,x=x,
                               selected=r.get('prefix_is_selected',r.get('selected_prefix',r.get('selected','N/A'))),source='PREFIX_RESULTS.csv')
        for metric,ax in zip(('median_batch_ms','peak_allocated_MiB'),axes[1]):
            for r in costs:
                if (r.get('dataset')==dataset and r.get('method')==method and r.get('device','').startswith('cuda')
                        and str(r.get('batch_size'))=='8' and _num(r,'prefix') is not None and _num(r,metric) is not None):
                    ax.scatter(_num(r,'prefix'),_num(r,metric),color=COLORS[i+2],marker=('o','s','^')[i])
                    _point(points,name,metric,dataset=dataset,method=method,seed=r.get('seed','NA'),device='cuda',batch_size=8,
                           prefix=_num(r,'prefix'),metric=metric,value=_num(r,metric),x=_num(r,'prefix'),source='COST.csv')
    for ax,label in zip(axes.flat,('Normalized MSE','Normalized MAE','CUDA B8 time (ms/batch)','CUDA B8 peak allocated (MiB)')):
        ax.set_xlabel('Selected / reported prefix K')
        ax.set_ylabel(label)
        if not ax.has_data():
            _empty(ax)
    axes[0,0].legend(frameon=False,fontsize=6)
    _save(fig,out,name,'V prefixes are descriptive; deployment K was sealed on M_select. Prefix savings do not imply lower fixed F weight memory.',figures)


def _write(path,text):
    path.write_text(text.rstrip()+'\n',encoding='utf8',newline='\n')


def _texts(report,out,runs,costs,inputs,warnings):
    validation=common.read(report/'NUMERICAL_VALIDATION.json') if (report/'NUMERICAL_VALIDATION.json').is_file() else {'status':'N/A'}
    decisions=common.read(report/'C_DECISIONS.json') if (report/'C_DECISIONS.json').is_file() else {'status':'N/A'}
    ledger=common.read(report/'RESOURCE_LEDGER.json') if (report/'RESOURCE_LEDGER.json').is_file() else {}
    analysis=['# 기준 예측 이후 추가 보정: 실제 결과','',
        '이 문서는 공개 RUNS/COST/충분통계/학습곡선에서 다시 생성했다. V는 반복 사용한 개발 평가이며 독립 TEST가 아니다. M_select도 반복 선택에 사용했다.',
        '두 seed의 점과 min/max는 관측 범위이며 신뢰구간이 아니다. 겹치는 예측 원점을 독립 표본으로 취급한 p-value는 계산하지 않았다.','',
        'G는 자료별 seed 92601의 기존 부모 하나이며 모든 S/보정 seed에서 공유한다. S는 Jena K12 cap3, PeMS32 K8 cap6의 기존 A60 부모다. A_stop도 A60 refit에 포함되므로 새 보정 선택에 재사용하지 않았다.',
        '새 보정은 G/S가 학습·선택하지 않은 M_cal에서 M_fit 학습, 경계 정답을 purge한 M_select 선택을 거쳤다. 선택 후 전체 M_cal에서 같은 A60 초기 상태로 fresh refit했다. 보정 경로에 x_last를 중복 가산하지 않는다.','']
    for dataset in common.DATASETS:
        analysis += ['## '+dataset,'','```text','method                 seed      K   MSE          MAE        selected_epoch  selected_prefix']
        rr=sorted([r for r in runs if r.get('dataset')==dataset],key=lambda r:(ORDER.index(r['method']) if r.get('method') in ORDER else 99,str(r.get('seed',''))))
        for r in rr:
            analysis.append(f"{r.get('method','N/A'):<22} {str(r.get('seed','NA')):<8} {_fmt(_num(r,'prefix')):>3} {_fmt(_num(r,'mse')):>11} {_fmt(_num(r,'mae')):>11} {_display(r.get('selected_epoch')):>14} {_display(r.get('artifact_selected_prefix',r.get('selected_prefix'))):>14}")
        if not rr:
            analysis.append('N/A: 공개 최종 평가 행이 아직 없다.')
        analysis += ['```','']
        f=_mean(runs,dataset,'RESIDUAL_F');g=_mean(runs,dataset,'G')
        if f is not None and g is not None:
            analysis.append(f'RESIDUAL_F 평균 MSE는 {_fmt(f)}, G는 {_fmt(g)}이다. 차이(F−G)는 {_fmt(f-g)}이며 음수이면 개선이다.')
        else:
            analysis.append('RESIDUAL_F 대 G 평균 비교는 N/A이다.')
        for method in ('ALPHA_fitted','READOUT_ONLY','RESIDUAL_Q','RESIDUAL_RAW_MLP','OWN','ALL'):
            value=_mean(runs,dataset,method)
            analysis.append(f'{method} 대비 F 평균 MSE 차이: {_fmt(None if f is None or value is None else f-value)}. 각 방법의 MAE와 seed별 음의 결과도 위 표에 보존했다.')
        analysis += ['','서로 다른 자료의 정규화 MSE 절대값으로 난이도 순위를 정하지 않는다. OWN/ALL은 차원이 다르므로 차이를 순수 교차채널 효과로 해석하지 않는다.','']
    analysis += ['## 전체 경로 비용','',
        '비용은 physical 과거와 원래 past mask에서 시작해 과거만 채우기, A60 scaler, G와 실제 보정/P/E/F/Q/D, 역표준화, CPU 출력을 포함한다. 캐시 읽기와 head-only 시간을 전체 비용으로 쓰지 않는다. 모델 로드와 window slicing은 별도 기록이다.',
        '배치·장치별 실측을 아래에 분리했다. B8/origin은 단일 요청 latency가 아니다. allocated/reserved, per-PID GPU, CPU RSS는 서로 다른 메모리 좌표다. per-PID 불가 사유는 COST 원자료를 따른다.','',
        '```text','dataset method                 seed   K device B  ms/batch  ms/origin allocatedMiB reservedMiB RSSMiB measurement_origin']
    for r in costs:
        analysis.append(f"{r.get('dataset','NA'):<7} {r.get('method','NA'):<22} {str(r.get('seed','NA')):<6} {_fmt(_num(r,'prefix')):>3} {r.get('device','NA'):<6} {r.get('batch_size','NA'):<2} {_fmt(_num(r,'median_batch_ms','median_ms_per_batch')):>9} {_fmt(_num(r,'median_origin_ms','median_ms_per_origin')):>9} {_fmt(_num(r,'peak_allocated_MiB')):>11} {_fmt(_num(r,'peak_reserved_MiB')):>11} {_fmt(_num(r,'CPU_RSS_sampled_peak_MiB')):>8} {r.get('measurement_origin','N/A')}")
    if not costs:
        analysis.append('N/A: 실측 비용 표가 아직 없다.')
    analysis += ['```','','## 선택·조건부 단계·검증','',
        'C 판정 봉인 상태: '+str(decisions.get('status','N/A'))+'. 상세 수치·진입/미진입·예산 사유는 [C_DECISIONS.json](C_DECISIONS.json)을 따른다. 조건 미충족은 방법의 불가능성 증명이 아니다.',
        '고정된 수치 검산 상태: '+str(validation.get('status','N/A'))+'. [NUMERICAL_VALIDATION.json](NUMERICAL_VALIDATION.json)을 수치 근거로 사용한다. 정규화·마스크·부모·기울기·재개·그림의 최종 상태는 [VALIDATION_REPORT.json](VALIDATION_REPORT.json)을 따른다. 원격 게시 여부는 별도 PUBLICATION_VERIFICATION.json의 main SHA·파일 검증으로 확인한다. 누락 자료는 PASS로 바꾸지 않았다.',
        '자원 스냅샷까지 경과시간(초): '+_fmt(_num(ledger,'elapsed_seconds'))+'; GPU 소유 프로세스 누적(초): '+_fmt(_num(ledger,'GPU_owned_process_seconds'))+'. 실제 프로세스 계보와 종료 상태는 [RESOURCE_LEDGER.json](RESOURCE_LEDGER.json)을 따른다. 이후 보고·게시까지 전체 경과시간은 게시 검증 영수증에 별도로 남긴다.',
        '추가 search/refit·캐시·부모 학습비용은 RUNS/학습 영수증의 각 열을 사용한다. 누락된 비용을 0으로 채우지 않는다.',
        '상류 Jena 역정규화/FP32 반올림, 과거 보호자료 노출, F 사전학습 중복은 unresolved다. 이 결과로 모든 데이터 우위·독립 검증·인과 설명·원천적 신규성을 주장하지 않는다.',
        '가까운 선행·기존 V9와 실제 확인 범위는 [PRIOR_ART_AND_DELTA.md](PRIOR_ART_AND_DELTA.md)에 있다. 직접 CATS 전체 재현 없이 선행 대비 우위를 판정하지 않는다.','']
    analysis += ['## 추가 학습·캐시 비용','',
        '아래는 새 보정기의 실제 단계 영수증이다. search/refit의 epoch·노출·update와 시간을 분리했다. 과거 G/S 부모학습은 새 보정 비용에 중복 합산하지 않는다. cache 생성은 별도 열이며 shared 자산을 model 수만큼 새 학습했다고 세지 않는다.','',
        '```text','dataset method                 seed    phase  updates exposures train_s eval_s ckpt_s cache_s worker_s']
    receipts=_training_receipts(report)
    for (dataset,method,seed,phase),r in receipts:
        analysis.append(f"{str(dataset):<7} {str(method):<22} {str(seed):<7} {phase:<6} {_fmt(_num(r,'updates')):>7} {_fmt(_num(r,'exposures')):>9} {_fmt(_num(r,'training_seconds')):>7} {_fmt(_num(r,'evaluation_seconds')):>7} {_fmt(_num(r,'checkpoint_seconds')):>7} {_fmt(_num(r,'cache_generation_seconds')):>7} {_fmt(_num(r,'worker_method_seconds')):>8}")
    if not receipts:analysis.append('N/A: 공개 단계 비용 영수증이 아직 없다.')
    analysis += ['```','']
    c_paths=sorted((report/'C').rglob('*.json'))
    if c_paths:
        analysis += ['C 단계는 각 블록 update·목표노출·F forward/backward rows와 실제 wall을 별도로 공개했다. SEQ는 두 내부 학습 단계가 있어 단순 optimizer step 총합만으로 joint와 비용 동등이라고 쓰지 않는다.','', '```text']
        for path in c_paths:
            r=common.read(path)
            analysis.append(path.relative_to(report).as_posix())
            for key in ('status','selected_prefix','selected_epoch','search_accounting','refit_accounting'):
                analysis.append(key+': '+json.dumps(r.get(key,'N/A'),ensure_ascii=False,sort_keys=True))
        analysis += ['```','']
    analysis += ['## 현재 비교가 뒷받침하는 판단','',
        'M_select의 조건부 진입은 자원 배정 규칙이다. 그곳의 개선이 이후 V에도 유지된다는 보장이 아니다. V를 본 뒤 checkpoint, 학습량, prefix 또는 후보를 바꾸지 않았다.','']
    c_comparisons=[]
    for dataset in common.DATASETS:
        full_k=12 if dataset=='jena' else 8
        full=[r for r in runs if r.get('dataset')==dataset and _num(r,'prefix')==full_k]
        means={method:_mean(full,dataset,method) for method in ('SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX')}
        if all(v is not None for v in means.values()):
            c_comparisons.append((dataset,means))
            analysis += [f"{dataset} K={full_k}: SEQ 평균 MSE {_fmt(means['SEQ_BLOCKS'],10)}, JOINT_FULL {_fmt(means['JOINT_FULL'],10)}, JOINT_PREFIX {_fmt(means['JOINT_PREFIX'],10)}.",
                '이 세 방법의 같은 K·seed별 값과 모든 미선택 prefix도 위 표에 남겼다. 평균 비교를 모든 seed의 동일한 방향으로 확대하지 않는다.','']
            g=_mean(runs,dataset,'G')
            nonzero=[_num(r,'mse') for r in runs if r.get('dataset')==dataset and r.get('method') in ('SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX') and (_num(r,'prefix') or 0)>0]
            if g is not None and nonzero and all(v is not None and v>g for v in nonzero):
                analysis += ['V에서는 모든 비영 prefix의 MSE가 G-only K0보다 높았다. 이는 선택 구간에서 정한 K의 전이 실패다. K0로 사후 재선택하지 않았다.','']
        f=_mean(runs,dataset,'RESIDUAL_F');a=_mean(runs,dataset,'ALPHA_fitted')
        if f is not None and a is not None and f>=a:
            analysis.append(dataset+': 이번 등록 비교는 기존 fitted alpha 결합을 새 잔차 F로 교체할 근거를 주지 못했다.')
    reuse=[r for r in costs if str(r.get('reused_original_measurement','')).lower()=='true']
    fresh=[r for r in costs if str(r.get('new_fresh_process_executed','True')).lower()=='true']
    analysis += ['',f'비용 {len(costs)}개 facet 중 기존 동일 계약 실측 {len(reuse)}개를 원 영수증·trial·PID·시간과 함께 재사용했고, {len(fresh)}개를 이번에 새 독립 프로세스로 측정했다. 새로 측정한 것처럼 재사용 시간을 원장에 합산하지 않았다.',
        '각 timed trial 직전 GPU peak를 재설정해 timed20회의 최대 allocated/reserved를 요약했다. CPU RSS·per-PID 샘플 peak는 warmup과 timed 구간의 별도 관측이며 순간 최대의 보장이 아니다.',
        'Q도 A60 S의 P/E 초기화를 사용하므로 S 부모의 과거 F 학습 정보를 물려받는다. no-F는 새 보정 학습·배포에서 F를 호출하지 않는다는 뜻이며, F를 한 번도 이용하지 않은 학습 이력이라는 뜻이 아니다.',
        'B의 dataset 묶음 학습 worker에는 F가 계속 상주한다. Q/원본 MLP의 그 worker GPU peak를 독립 no-F 학습 메모리로 해석하지 않는다. 독립 배포 메모리는 COST.csv의 fresh-process 경로에서 확인한다.',
        '실제 source-array의 보호 T 숫자 접근은 없었다. 과거 공개 역사 문서의 TEST 숫자가 우연히 노출됐는지는 현재 증거로 부재를 확정하지 못한다. [INFORMATION_ACCESS_AUDIT.json](INFORMATION_ACCESS_AUDIT.json)에 범위를 구분했다.',
        'C 전체작업의 legacy calls=0 필드는 미계측 값을 뜻해 공개 사본에서 N/A로 해석했다. 원 기록은 보존했다. 훈련 F rows와 추론의 실측 F calls는 이 필드와 별개다. [COUNTER_INTERPRETATION.json](COUNTER_INTERPRETATION.json)을 따른다.','']
    if warnings:
        analysis += ['공개 입력 주의: '+'; '.join(warnings)]
    _write(out/'RESULT_ANALYSIS_KO.md','\n'.join(analysis))
    criteria=[]
    for dataset in common.DATASETS:
        ff=_mean(runs,dataset,'RESIDUAL_F')
        gg=_mean(runs,dataset,'G')
        ro=_mean(runs,dataset,'READOUT_ONLY')
        simple=[_mean(runs,dataset,m) for m in ('ALPHA_fitted','RESIDUAL_Q','RESIDUAL_RAW_MLP','OWN','ALL')]
        criteria.append((dataset,ff,gg,ro,simple))
    extra=any(f is not None and g is not None and f<g for _,f,g,_,_ in criteria)
    pe=any(f is not None and ro is not None and f<ro for _,f,_,ro,_ in criteria)
    fv=any(f is not None and all(v is not None and f<v for v in simple) for _,f,_,_,simple in criteria)
    prefix_summary='; '.join(dataset+': 봉인된 K='+','.join(str(k) for k in sorted({int(r['selected_prefix']) for r in decision.get('models',[]) if r.get('selected_prefix') is not None}))
        for dataset,decision in decisions.get('datasets',{}).items() if decision.get('models')) or '실행된 C 선택 없음'
    claim=['# 주장 상태','',
        '상태는 반복 개발 평가의 기술적 근거 범위다. 평균에서 유리한 자료가 있어도 보편적 채택이나 유의성을 뜻하지 않는다. 채택·prefix 선택은 V를 보기 전에 봉인했다.','',
        '1. 추가 보정 자료의 개선: '+('조건부' if extra else '보류')+'. 자료·방법·seed별 MSE/MAE는 RESULT_ANALYSIS_KO에 모두 보존했다.',
        '2. 출력부 이상 P/E 학습 필요성: '+('조건부' if pe else '보류')+'. F와 READOUT_ONLY의 개발 오차 차이는 초기정보·누적 학습량까지 같은 순수 인과 효과가 아니다.',
        '3. no-F 직접 예측기 이상 F 가치: '+('조건부' if fv else '보류')+'. alpha/Q/원본 MLP/OWN/ALL과 정확도 및 전체 실행비용을 함께 본다.',
        '4. 순차 순서의 joint-final/joint-prefix 대비 효과: '+('기각(이번 등록 비교 범위)' if c_comparisons and all(m['SEQ_BLOCKS']>=min(m['JOINT_FULL'],m['JOINT_PREFIX']) for _,m in c_comparisons) else '보류')+'. 같은 K의 실제 평균 비교에서 판정한다. 한 seed의 소폭 예외를 숨기거나 순차 원리 자체의 불가능성으로 일반화하지 않는다. JOINT_FULL의 임의 앞 절반을 우위 근거로 쓰지 않는다.',
        '5. 사용 prefix와 F 계산/메모리 절감: 보류. '+prefix_summary+'. 선택된 K와 미선택 prefix의 기술통계를 구분한다. K0는 실제 G-only/F 미로드 경로다. 작은 비영 K의 적은 F rows는 같은 상주 F의 고정 가중치 메모리 감소를 뜻하지 않는다.',
        '6. 자료별 차이의 인과 설명: 기각. 현재 근거는 관측·연관이며 공분산 설명분산은 미래 잔차 예측 가능성의 보장이 아니다.',
        '7. 가까운 선행·V9 이후 남는 신규 기여: 보류. 잔차 가산·잠재 어댑터·prefix 자체는 기존 원리와 겹친다. [PRIOR_ART_AND_DELTA.md](PRIOR_ART_AND_DELTA.md)를 따른다.','']
    _write(out/'CLAIM_STATUS.md','\n'.join(claim))
    _write(out/'README_KO.md','\n'.join(['# 고정 G 이후 잠재 잔차 적응','',
        '실제 공개 입력에서 재생성한 결과다. 고정된 독립 수치 검산 상태: **'+str(validation.get('status','N/A'))+'**. 최종 완료 상태는 [VALIDATION_REPORT.json](VALIDATION_REPORT.json)을 확인한다.',
        '필수 비교는 G/기존 alpha/출력부만/F/no-F 잠재/원본 잔차 MLP와 선형 프로브다. 음의 자료·seed를 제외하지 않는다.',
        '자료별 F 대 기존 fitted alpha의 평균 MSE 차이(F−alpha): '+ '; '.join(dataset+' '+_fmt(None if _mean(runs,dataset,'RESIDUAL_F') is None or _mean(runs,dataset,'ALPHA_fitted') is None else _mean(runs,dataset,'RESIDUAL_F')-_mean(runs,dataset,'ALPHA_fitted')) for dataset in common.DATASETS)+'. 양수이면 기존 alpha가 더 정확하다. 순차 블록의 평균 우위도 확보하지 못했다.',
        '실제 모델 수·내부 단계 수·조건부 미진입·미완료 여부는 [EXECUTION_STATUS.csv](EXECUTION_STATUS.csv)와 [RESUME_QUEUE.md](RESUME_QUEUE.md), 원격 게시 증거는 [PUBLICATION_VERIFICATION.json](PUBLICATION_VERIFICATION.json)을 따른다. 이 문서는 실험 검산과 원격 게시 검증을 별도로 연결한다.',
        '', '- [전체 분석](RESULT_ANALYSIS_KO.md)', '- [주장 상태](CLAIM_STATUS.md)',
        '- [부모·자료 계약](PARENT_AND_DATA_CONTRACT.json)','- [고정 수치 검산](NUMERICAL_VALIDATION.json)','- [최종 검증](VALIDATION_REPORT.json)',
        '- [그림·공개 원점](FIGURE_MANIFEST.json)','- [실측 값](FIGURE_POINTS.csv)',
        '- [재현](REPRODUCE.md)','',
        'V는 반복 개발 평가다. 두 seed 범위는 신뢰구간이 아니다. 보호 T source-array 숫자 접근은 없었으며, 역사 문서의 우연한 숫자 노출·상류 전처리·사전학습 중복은 INFORMATION_ACCESS_AUDIT의 한계를 따른다.']))
    _write(out/'REPRODUCE.md','\n'.join(['# 공개 표·그림 재현','',
        '저장소 루트에서 실행한다. 아래 재생성은 공개 CSV/JSON/충분통계만 읽고 raw data, private 예측, 모델 또는 GPU를 사용하지 않는다.','',
        '```powershell',
        'python -X utf8 -m research.incremental_latent_adaptation_20261019_v1.report --report-dir reports/incremental_latent_adaptation_20261019_v1 --output-dir reports/incremental_latent_adaptation_20261019_v1/replayed_public',
        'python -X utf8 -m research.incremental_latent_adaptation_20261019_v1.report --report-dir reports/incremental_latent_adaptation_20261019_v1 --verify-replay',
        '```','',
        '학습 재개는 실제 [RESUME_QUEUE.md](RESUME_QUEUE.md) 명령과 sealed EXPERIMENT_PLAN을 따른다. 이 보고서가 새 학습/선택 명령을 추정해 만들지 않는다.',
        '이번 실행 환경은 Python 3.11, PyTorch 2.10+cu128, RTX 4070 12 GiB다. 공개 그림은 NumPy와 Matplotlib 및 저장된 공개 파일로 재생성한다. 실제 학습·원 예측 재검산은 공개하지 않은 원자료/A60 부모/checkpoint/F 자산과 .cache의 봉인·manifest가 있어야 한다. 공개 충분통계 재생성은 원 예측의 신규 실행과 구분한다.',
        '등록 시계와 deadline은 재개해도 초기화하지 않는다. 이후 날짜의 새 학습은 같은 캠페인의 시계를 재설정하는 방식으로 실행하지 않는다. 이미 완료된 결과와 현재 공개 표·그림의 재생성은 이 학습 상한과 구분한다.',
        '배포 비용 fresh worker: `python -X utf8 -m research.incremental_latent_adaptation_20261019_v1.costs --request <sealed_request.json> --asset-root <local_asset_root> --output <new_receipt.json>`.',
        '비용 요청에는 정확한 artifact SHA와 final prediction reference SHA가 필요하며 C_DECISIONS의 SEALED_BEFORE_V 봉인 전에는 V loader가 거부한다.']))


def build(reportdir=common.REPORT,output_dir=None):
    report=Path(reportdir).resolve();out=Path(output_dir).resolve() if output_dir else report
    out.mkdir(parents=True,exist_ok=True)
    (out/'figures').mkdir(exist_ok=True)
    _style()
    runs=_v(_csv(report/'RUNS.csv'));cost_rows=_actual(_csv(report/'COST.csv'))
    points,figures,warnings=[],[],[]
    required=('RUNS.csv','COST.csv','COST_TRIALS.csv','ERROR_SUFFICIENT_STATS.csv.gz','C_DECISIONS.json','NUMERICAL_VALIDATION.json')
    missing=[name for name in required if not (report/name).is_file()]
    warnings.extend('Missing '+name for name in missing)
    for dataset in common.DATASETS:
        for method in CORE:
            if not _series(runs,dataset,method,'mse') or not _series(runs,dataset,method,'mae'):
                warnings.append('Missing accuracy values '+dataset+'/'+method)
        for batch in (1,8):
            for method in CORE:
                found=[r for r in cost_rows if r.get('dataset')==dataset and r.get('method')==method
                       and r.get('device','').startswith('cuda') and str(r.get('batch_size'))==str(batch)]
                if not found or any(_num(r,'median_batch_ms','median_ms_per_batch') is None
                                    or _num(r,'peak_allocated_MiB','GPU_peak_allocated_MiB') is None for r in found):
                    warnings.append(f'Missing GPU cost value {dataset}/{method}/B{batch}')
    if (report/'C_DECISIONS.json').is_file() and common.read(report/'C_DECISIONS.json').get('status')!='SEALED_BEFORE_V':
        warnings.append('B/C decisions are not sealed before V')
    if (report/'NUMERICAL_VALIDATION.json').is_file() and common.read(report/'NUMERICAL_VALIDATION.json').get('status')!='PASS':
        warnings.append('Independent numerical validation is not PASS')
    inputs={name:common.sha(report/name) for name in required if (report/name).is_file()}
    for name in ('RESOURCE_LEDGER.json','PARENT_AND_DATA_CONTRACT.json','PRIOR_ART_AND_DELTA.md',
                 'PREFIX_RESULTS.csv','B_DECISIONS.json','C_ADMISSION.json','HETEROGENEITY.csv',
                 'COVARIANCE_SPECTRUM.csv','TRAINING_COST.csv','COST_REUSE_VALIDATION.json',
                 'REFERENCE_REPRODUCTION.json','COUNTER_INTERPRETATION.json','INFORMATION_ACCESS_AUDIT.json'):
        if (report/name).is_file():inputs[name]=common.sha(report/name)
    for path in sorted((report/'runs').rglob('*_cap*.json')):
        inputs[path.relative_to(report).as_posix()]=common.sha(path)
    for path in sorted((report/'C').rglob('*.json')):
        inputs[path.relative_to(report).as_posix()]=common.sha(path)
    curves=_curves(report)
    stats=_sufficient(report/'ERROR_SUFFICIENT_STATS.csv.gz')
    for dataset in common.DATASETS:
        _accuracy(runs,dataset,out,points,figures)
        for device in ('cuda','cpu'):
            for batch in (1,8):_cost(cost_rows,dataset,device,batch,out,points,figures)
        _heterogeneity(stats,dataset,out,points,figures,warnings)
        _learning(curves,dataset,out,points,figures)
        if (report/'PREFIX_RESULTS.csv').is_file():
            _prefix(_csv(report/'PREFIX_RESULTS.csv'),cost_rows,dataset,out,points,figures)
    _texts(report,out,runs,cost_rows,inputs,warnings)
    fields=list(dict.fromkeys(key for row in points for key in row)) or ['figure','panel','metric','value']
    with (out/'FIGURE_POINTS.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(points)
    artifacts={name:common.sha(out/name) for name in ('README_KO.md','RESULT_ANALYSIS_KO.md','CLAIM_STATUS.md','REPRODUCE.md','FIGURE_POINTS.csv')}
    manifest=dict(schema='INCREMENTAL_PUBLIC_FIGURES_v1',status='COMPLETE' if not warnings else 'PARTIAL',
        input_files=inputs,source_sha256=common.sha(Path(__file__)),figures=figures,artifacts=artifacts,
        figure_point_count=len(points),warnings=warnings,
        statistical_scope='Two fixed seeds; min/max ranges are not confidence intervals; overlapping origins do not give independent sample size',
        independence='Public aggregate tables only; no raw source, private prediction, checkpoint, torch, CUDA or network access',
        deterministic_metadata='Fixed SVG hashsalt/date; PNG Software fixed; no current timestamp or output-directory identifiers')
    common.atomic(out/'FIGURE_MANIFEST.json',manifest)
    return manifest


def replay(reportdir=common.REPORT,output_dir=None):
    report=Path(reportdir).resolve()
    destination=Path(output_dir) if output_dir else report/'replayed_public'
    current=common.read(report/'FIGURE_MANIFEST.json')
    rebuilt=build(report,destination)
    differences=[]
    for field in ('input_files','figures','artifacts','figure_point_count','warnings'):
        if current.get(field)!=rebuilt.get(field):differences.append(field)
    result=dict(status='PASS' if not differences else 'FAIL',differences=differences,
        figure_manifest_sha256=common.sha(report/'FIGURE_MANIFEST.json'),
        values_and_text_identical=current.get('artifacts')==rebuilt.get('artifacts'),
        figure_bytes_identical=current.get('figures')==rebuilt.get('figures'),
        source_inputs_unchanged=current.get('input_files')==rebuilt.get('input_files'),
        protected_numeric_reads=0,private_prediction_reads=0,GPU_operations=0)
    common.atomic(report/'REPLAY_VALIDATION.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',default=str(common.REPORT))
    parser.add_argument('--output-dir')
    parser.add_argument('--verify-replay',action='store_true')
    args=parser.parse_args(argv)
    result=replay(args.report_dir,args.output_dir) if args.verify_replay else build(args.report_dir,args.output_dir)
    print(json.dumps({k:result[k] for k in ('status',) if k in result}))
    return 0 if result['status'] in ('COMPLETE','PASS') else 1


if __name__=='__main__':raise SystemExit(main())
