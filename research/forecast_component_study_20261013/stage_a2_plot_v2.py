"""Preserve original figure; move the shared legend outside the data panels."""
from pathlib import Path
import json
import time
import psutil
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from diagnose_forecast import read,sha,write_new

started=time.perf_counter();proc=psutil.Process();cpu_start=sum(proc.cpu_times()[:2])
out=Path(__file__).resolve().parent/'stage_a2_dense_pool_v1'
r=read(out/'RESULT.json')
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for ax,dataset in zip(axes,('jena','robin')):
    position=np.arange(2)
    high=[]
    for offset,pool,color,label in [(-.18,'sparse_replay','#d95f0e','Sparse pool'),(.18,'dense256','#2b8cbe','FIT256 pool')]:
        values=[]
        for seed in (92601,92602):
            c=next(c for c in r['cases'] if c['dataset']==dataset and c['seed']==seed and c['pool']==pool)
            values.append(next(v['val']['mse'] for v in c['evaluations'] if v['update']==c['selected_update']))
        high.extend(values)
        bars=ax.bar(position+offset,values,width=.34,color=color,label=label)
        for bar,y in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,y,f'{y:,.1f}',ha='center',va='bottom',fontsize=9)
    ax.set_xticks(position,['seed92601','seed92602']);ax.set(title=dataset.title(),ylabel='Physical VAL MSE',ylim=(0,max(high)*1.12))
    ax.grid(axis='y',alpha=.2)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=2)
fig.suptitle('TUNE-selected checkpoints: same original forecasting head',fontsize=14)
path=out/'stage_a2_selected_val_v2.png'
if path.exists():raise FileExistsError(path)
fig.savefig(path,dpi=180);plt.close(fig)
write_new(out/'FIGURE_READABILITY_v2.json',{'status':'PASS','source_sha256':sha(out/'RESULT.json'),
    'figure_sha256':sha(path),'script_sha256':sha(Path(__file__)),'change':'Shared legend placed below both panels; original image and report preserved',
    'resource':{'cpu_seconds':sum(proc.cpu_times()[:2])-cpu_start,'wall_seconds':time.perf_counter()-started,'gpu_seconds':0}})
with (out/'FIGURE_ADDENDUM_KO.md').open('x',encoding='utf-8') as f:
    f.write('# 그림 가독성 보완\n\n원래 수치·보고서·그림을 보존하고, 막대 위 수치와 범례의 간격을 확보한 개선판을 추가했다. 원자료는 동일한 RESULT.json이다.\n\n![TUNE 선택 후 paired VAL 비교: 범례 위치 개선](stage_a2_selected_val_v2.png)\n\n[수치 보고서](REPORT_KO.md) · [바이트 연결 기록](FIGURE_READABILITY_v2.json)\n')
print(path)
