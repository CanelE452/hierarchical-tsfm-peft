"""Visualization only: read saved CSV/JSON/Markdown; never import model code.

Run with the repository Python environment. Outputs stay beside this script.
All measured coordinates and caption statistics are derived from source files.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from PIL import Image

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = BASE.parents[1]
PANELS = ['ettm2', 'jena']
NAMES = {'ettm2': 'ETTm2', 'jena': 'Jena'}
ARMS = ['HEAD', 'LORA', 'LORAPLUS']
LABEL = {'HEAD': 'HEAD', 'LORA': 'LoRA', 'LORAPLUS': 'LoRA+',
         'F0_LONG': 'F0 long', 'F0_SHORT': 'F0 short'}
COLORS = {'HEAD': '#555555', 'LORA': '#477A98', 'LORAPLUS': '#AA7945',
          'F0_LONG': '#8A8A8A', 'F0_SHORT': '#A0A0A0'}
MARKERS = {'HEAD': 'o', 'LORA': 's', 'LORAPLUS': '^', 'F0_LONG': 'D', 'F0_SHORT': 'v'}
LINES = {'HEAD': '-', 'LORA': '--', 'LORAPLUS': '-.'}
FILES = ['fig1_internal_adaptation_gap', 'fig2_quality_cost_tradeoff',
         'fig3_validation_learning_curves', 'fig4_channelwise_internal_gain',
         'fig5_problem_motivation_summary', 'fig6_context_control']
FONT = next(n for n in ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
            if n in {f.name for f in font_manager.fontManager.ttflist})
mpl.rcParams.update({'font.family': FONT, 'font.size': 8, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': .55, 'xtick.major.width': .5, 'ytick.major.width': .5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'svg.hashsalt': 'internal-adaptation-paper-v1'})

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def close(a, b):
    assert np.isclose(float(a), float(b), rtol=1e-10, atol=2e-12), (a, b)

def source_snapshot():
    return {p.relative_to(BASE).as_posix(): sha(p) for p in BASE.rglob('*')
            if p.is_file() and OUT not in p.parents}

def tables(text):
    result = []
    lines = text.splitlines()
    for i, line in enumerate(lines[:-1]):
        if line.startswith('|') and re.match(r'^\|[\s:|\-]+$', lines[i+1]):
            header = [x.strip() for x in line.strip('|').split('|')]
            rows = []
            for row in lines[i+2:]:
                if not row.startswith('|'):
                    break
                rows.append(dict(zip(header, [x.strip() for x in row.strip('|').split('|')])))
            result.append(rows)
    return result

def rounded_equal(actual, text):
    decimals = len(text.partition('.')[2])
    assert abs(actual-float(text)) <= .50001*10**(-decimals), (actual, text)

DATA = {}
VALUES = []
CHECKS = {}
FORMATS = {}

def audit_sources():
    summary = BASE.joinpath('SUMMARY_KO.md').read_text(encoding='utf-8')
    summary_rows = tables(summary)[0]
    for panel in PANELS:
        folder = BASE/panel
        d = {n: pd.read_csv(folder/(n+'.csv'), float_precision='round_trip')
             for n in ['SCORES', 'EFFECTS', 'CURVES', 'RESOURCES', 'CHANNEL_SCORES']}
        d.update(seal=read(folder/'SELECTION_SEAL.json'), f0=read(folder/'F0_REFERENCE.json'),
                 decision=read(folder/'DECISION.json'), audit=read(folder/'DATA_AUDIT.json'))
        d['selected'] = d['SCORES'].query("variant == 'selected'").copy()
        d['internal'] = d['seal']['internal_reference']
        assert d['internal'] == {'ettm2': 'LORA', 'jena': 'LORAPLUS'}[panel]
        d['seeds'] = sorted(d['selected'].query("arm == 'HEAD'")['seed'].tolist())
        assert len(d['seeds']) == 2
        assert not d['SCORES'].duplicated(['arm','seed','variant']).any()
        for arm in ARMS:
            assert sorted(d['selected'].query('arm == @arm').seed) == d['seeds']
            for seed in d['seeds']:
                key = f'{panel}_{arm}_{seed}'
                fit_path = folder/'fits'/(key+'.json')
                assert sha(fit_path) == d['seal']['fit_hashes'][fit_path.name]
                fit = read(fit_path)
                score = d['selected'].query('arm == @arm and seed == @seed').iloc[0]
                resource = d['RESOURCES'].query('arm == @arm and seed == @seed').iloc[0]
                curve = d['CURVES'].query('arm == @arm and seed == @seed').sort_values('step')
                step = d['seal']['choices'][key]['selected_step']
                assert step == fit['selected_step'] == score.step == resource.selected_step
                assert step == int(curve.loc[curve.val_primary.idxmin(), 'step'])
                assert len(curve) == len(fit['curves'])
                for row, saved in zip(curve.to_dict('records'), fit['curves']):
                    for col in ['step','val_primary','val_nmae','active_train_s',
                                'validation_s','checkpoint_io_s','cold_ready_s']:
                        close(row[col], saved[col])
                close(curve.query('step == @step').iloc[0].cold_ready_s,
                      resource.selected_cold_ready_s)
        means = d['selected'].groupby('arm').primary.mean()
        for row in d['EFFECTS'].itertuples():
            close(100*(1-means[row.candidate]/means[row.baseline]), row.mean_gain_pct)
            for i, seed in enumerate(d['seeds']):
                a = d['selected'].query('arm == @row.candidate and seed == @seed').primary.iloc[0]
                b = d['selected'].query('arm == @row.baseline')
                b = b[b.seed == (0 if row.baseline.startswith('F0') else seed)].primary.iloc[0]
                close(100*(1-a/b), getattr(row, f'seed{i+1}_gain_pct'))
        effect = d['EFFECTS'][(d['EFFECTS'].candidate == d['internal']) & (d['EFFECTS'].baseline == 'HEAD')]
        d['effect'] = effect.iloc[0]
        # User-specified three-decimal claims are assertions, never plot data.
        assert f"{d['effect'].mean_gain_pct:.3f}" == {'ettm2':'2.172', 'jena':'1.695'}[panel]
        close(d['effect'].mean_gain_pct, d['decision']['internal_gain_vs_head_pct'])
        ratio = d['RESOURCES'].groupby('arm').selected_cold_ready_s.mean()
        d['cost_ratio'] = ratio[d['internal']]/ratio['HEAD']
        close(d['cost_ratio'], d['decision']['cold_ready_ratio_internal_over_head'])
        for arm in d['selected'].arm.unique():
            for seed in d['selected'].query('arm == @arm').seed:
                ch = d['CHANNEL_SCORES'].query("arm == @arm and seed == @seed and variant == 'selected'")
                assert ch.channel.tolist() == d['audit']['columns']
                close(ch.primary.mean(), d['selected'].query('arm == @arm and seed == @seed').primary.iloc[0])
        report = folder.joinpath('REPORT_KO.md').read_text(encoding='utf-8')
        ts = tables(report)
        for row in ts[0]:
            arm = row['arm']
            ss = d['selected'].query('arm == @arm')
            for metric in ['primary','nmae']:
                rounded_equal(ss[metric].mean(), row[metric])
            expected_steps = '/'.join(str(int(x)) for x in ss.sort_values('seed').step)
            assert expected_steps == row['selected_steps']
            if arm in ARMS:
                rr = d['RESOURCES'].query('arm == @arm')
                rounded_equal(rr.selected_cold_ready_s.mean(), row['cold_ready_s'])
                rounded_equal(rr.train_peak_allocated_bytes.mean()/2**20, row['train_peak_mib'])
            elif arm != 'F0_SHORT':
                rounded_equal(d['f0']['f0_cold_ready_s' if arm == 'F0_LONG' else 'naive_cold_ready_s'], row['cold_ready_s'])
        for row in ts[1]:
            ee = d['EFFECTS'][(d['EFFECTS'].candidate == row['candidate']) & (d['EFFECTS'].baseline == row['baseline'])].iloc[0]
            for key in ['mean_gain_pct','seed1_gain_pct','seed2_gain_pct','ci_low','ci_high']:
                rounded_equal(ee[key], row[key])
        sr = next(x for x in summary_rows if x['panel'] == panel)
        rounded_equal(d['effect'].mean_gain_pct, sr['gain_pct'])
        rounded_equal(d['cost_ratio'], sr['cost_ratio'])
        assert sr['status'] == d['decision']['status']
        assert f"{d['effect'].mean_gain_pct:.4f}%" in report
        close(d['audit']['context']*d['audit']['frequency_seconds']/86400, 8)
        close(d['audit']['short_context']*d['audit']['frequency_seconds']/86400, 4)
        DATA[panel] = d
        CHECKS[panel] = {'report_score_effect_resource_tables_match': True,
            'summary_matches': True, 'fit_hashes_and_selected_steps_match': True,
            'curve_values_match_fit_json': True, 'channel_mean_matches_score': True,
            'all_three_arms_two_seeds_retained': True, 'gain_recalculated_from_raw_scores': float(d['effect'].mean_gain_pct),
            'selected_checkpoint_cost_ratio': float(d['cost_ratio']),
            'unchanged_scientific_decision': d['decision']['status']}

def record(fig, panel, arm, seed, role, metric, value, source, selector, operation='identity', channel=''):
    VALUES.append(dict(figure=fig,panel=panel,arm=arm,seed=str(seed),role=role,metric=metric,
        value=float(value),source=source,selector=json.dumps(selector,sort_keys=True),operation=operation,channel=channel))

def score_points(fig, panel, arm):
    d = DATA[panel]
    rows = d['selected'].query('arm == @arm').sort_values('seed')
    for row in rows.itertuples():
        record(fig,panel,arm,row.seed,'TEST','primary',row.primary,f'{panel}/SCORES.csv',
               dict(arm=arm,seed=int(row.seed),variant='selected'))
    mean = rows.primary.mean()
    if arm in ARMS:
        record(fig,panel,arm,'mean','TEST','primary',mean,f'{panel}/SCORES.csv',
               dict(arm=arm,variant='selected'),'mean')
    return rows, mean

def panel_label(ax, letter, subtitle):
    ax.text(0,1.06,f'({letter})',transform=ax.transAxes,fontsize=9.5,fontweight='bold',va='bottom')
    ax.text(.095,1.06,subtitle,transform=ax.transAxes,fontsize=8.5,va='bottom')

def axes_style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='#E6E6E6',linewidth=.4)
    ax.yaxis.set_major_locator(MaxNLocator(6, steps=[1,2,4,5,10]))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

def point_panel(ax, fig, panel, arms, label_override=None):
    for i, arm in enumerate(arms):
        rows, mean = score_points(fig,panel,arm)
        if len(rows) > 1:
            for j, row in enumerate(rows.itertuples()):
                ax.plot(i+[-.12,.12][j],row.primary,MARKERS[arm],ms=3.2,
                        mec=COLORS[arm],mfc='white' if j==0 else COLORS[arm],mew=.7)
            ax.plot([i-.19,i+.19],[mean,mean],color=COLORS[arm],lw=1.8,solid_capstyle='butt')
        else:
            ax.plot(i,mean,MARKERS[arm],ms=4,mfc='white',mec=COLORS[arm],mew=1)
    ax.set_xticks(range(len(arms)), [label_override.get(a,LABEL[a]) if label_override else LABEL[a] for a in arms])
    ax.set_xlim(-.48,len(arms)-.52)
    ax.margins(y=.2)
    ax.set_ylabel('TEST scaled twice-pinball')
    axes_style(ax)

def seed_legend(fig):
    handles=[Line2D([],[],marker='o',mfc='white',mec='#555555',ls='',ms=3,label='Seed 1'),
             Line2D([],[],marker='o',color='#555555',ls='',ms=3,label='Seed 2'),
             Line2D([],[],color='#555555',lw=1.8,label='Two-seed mean')]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.52,.005))

def export(fig, number):
    name = FILES[number-1]
    fig.canvas.draw()
    # The same Figure object and fixed page size are used for all three formats.
    for ext in ['pdf','svg','png']:
        meta = {'Title': name, 'Creator':'Matplotlib; saved-result visualization only'} if ext != 'png' else {'Description':name}
        if ext == 'pdf':
            meta.update(CreationDate=None,ModDate=None)
        elif ext == 'svg':
            meta['Date'] = None
        fig.savefig(OUT/f'{name}.{ext}',dpi=600,metadata=meta)
        if ext == 'svg':
            path = OUT/f'{name}.svg'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    FORMATS[name] = {'inches':fig.get_size_inches().tolist(), 'same_figure_object':True,
                    'axes_limits':[dict(x=list(a.get_xlim()),y=list(a.get_ylim())) for a in fig.axes]}
    plt.close(fig)

def make_figures():
    fig, axs = plt.subplots(1,2,figsize=(7,2.7))
    fig.subplots_adjust(left=.09,right=.985,bottom=.22,top=.85,wspace=.32)
    for i,p in enumerate(PANELS):
        point_panel(axs[i],1,p,['F0_LONG']+ARMS)
        panel_label(axs[i],chr(97+i),NAMES[p])
    seed_legend(fig)
    export(fig,1)

    fig, axs = plt.subplots(1,2,figsize=(7,2.85))
    fig.subplots_adjust(left=.09,right=.985,bottom=.25,top=.85,wspace=.32)
    for i,p in enumerate(PANELS):
        ax=axs[i];d=DATA[p]
        for arm in ['F0_LONG']+ARMS:
            rows,mean=score_points(2,p,arm)
            if arm=='F0_LONG':
                x=d['f0']['f0_cold_ready_s']
                record(2,p,arm,0,'cost','f0_cold_ready_s',x,f'{p}/F0_REFERENCE.json',{})
            else:
                rr=d['RESOURCES'].query('arm == @arm').sort_values('seed')
                for j,r in enumerate(rr.itertuples()):
                    y=rows.query('seed == @r.seed').primary.iloc[0]
                    ax.plot(r.selected_cold_ready_s,y,MARKERS[arm],ms=3,mec=COLORS[arm],
                            mfc='white' if j==0 else COLORS[arm],mew=.7)
                    record(2,p,arm,r.seed,'cost','selected_cold_ready_s',r.selected_cold_ready_s,
                           f'{p}/RESOURCES.csv',dict(arm=arm,seed=int(r.seed)))
                x=rr.selected_cold_ready_s.mean()
                record(2,p,arm,'mean','cost','selected_cold_ready_s',x,f'{p}/RESOURCES.csv',dict(arm=arm),'mean')
            ax.plot(x,mean,MARKERS[arm],ms=5,mfc=COLORS[arm],mec='white',mew=.45)
            offset={'F0_LONG':(7,3),'HEAD':(7,5),'LORA':(6,-13),'LORAPLUS':(7,6)}[arm]
            ax.annotate(LABEL[arm],(x,mean),xytext=offset,textcoords='offset points',fontsize=7)
        ax.set_xlim(left=0);ax.margins(x=.15,y=.2)
        ax.set_xlabel('Cold-ready cost to selected checkpoint (s)')
        ax.set_ylabel('TEST scaled twice-pinball');axes_style(ax)
        panel_label(ax,chr(97+i),NAMES[p])
    fig.text(.53,.02,'Small markers: individual seeds; large markers: arithmetic means',ha='center',fontsize=7)
    export(fig,2)

    fig,axs=plt.subplots(2,2,figsize=(7,4.8))
    fig.subplots_adjust(left=.09,right=.985,bottom=.14,top=.91,hspace=.56,wspace=.32)
    for i,p in enumerate(PANELS):
        d=DATA[p]
        for arm in ARMS:
            for j,seed in enumerate(d['seeds']):
                cc=d['CURVES'].query('arm == @arm and seed == @seed').sort_values('step')
                for row in cc.itertuples():
                    for col in ['step','val_primary','cold_ready_s']:
                        record(3,p,arm,seed,'VAL',col,getattr(row,col),f'{p}/CURVES.csv',
                               dict(arm=arm,seed=int(seed),step=int(row.step)))
                for k,xcol in enumerate(['step','cold_ready_s']):
                    axs[i,k].plot(cc[xcol],cc.val_primary,color=COLORS[arm],ls=LINES[arm],lw=.7,
                        marker=MARKERS[arm],ms=2.8,mfc='white' if j==0 else COLORS[arm],mew=.6)
        for k,lab in enumerate(['Optimizer updates','Cold-ready cost (s)']):
            ax=axs[i,k];ax.set_xlabel(lab);ax.set_ylabel('VAL scaled twice-pinball');axes_style(ax)
            ax.set_xlim(left=0)
            if k==0:ax.set_xticks([0,128,256,384,512])
            panel_label(ax,chr(97+i*2+k),NAMES[p])
    handles=[Line2D([],[],color=COLORS[a],marker=MARKERS[a],ls=LINES[a],ms=3,lw=.8,label=LABEL[a]) for a in ARMS]
    handles += [Line2D([],[],marker='o',mfc='white',mec='#555555',ls='',ms=3,label='Seed 1'),
                Line2D([],[],marker='o',color='#555555',ls='',ms=3,label='Seed 2')]
    fig.legend(handles=handles,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.52,.005))
    export(fig,3)

    fig,axs=plt.subplots(1,2,figsize=(7,3.3))
    fig.subplots_adjust(left=.09,right=.985,bottom=.36,top=.87,wspace=.32)
    for i,p in enumerate(PANELS):
        ax=axs[i];d=DATA[p];arm=d['internal'];ch=d['CHANNEL_SCORES'].query("variant == 'selected'")
        for k,channel in enumerate(d['audit']['columns']):
            h=ch[(ch.arm=='HEAD') & (ch.channel==channel)].primary.mean()
            a=ch[(ch.arm==arm) & (ch.channel==channel)].primary.mean()
            gain=100*(h-a)/h
            record(4,p,arm,'mean','TEST','channel_gain_pct',gain,f'{p}/CHANNEL_SCORES.csv',
                   dict(arm=arm,baseline='HEAD',channel=channel,variant='selected'),'ratio_of_channel_means',channel)
            ax.vlines(k,0,gain,color=COLORS[arm],lw=.8)
            ax.plot(k,gain,MARKERS[arm],ms=3.8,color=COLORS[arm],mfc=COLORS[arm] if gain>=0 else 'white',mew=.8)
        ax.axhline(0,color='#444444',lw=.7)
        ax.set_xticks(range(len(d['audit']['columns'])),d['audit']['columns'],rotation=90 if p=='jena' else 45,ha='center' if p=='jena' else 'right')
        ax.tick_params(axis='x',labelsize=7)
        ax.set_ylabel('Internal gain over HEAD (%)');ax.margins(x=.05,y=.18)
        ax.grid(axis='y',color='#E6E6E6',lw=.4);ax.set_axisbelow(True)
        panel_label(ax,chr(97+i),f"{NAMES[p]}: {LABEL[arm]} vs HEAD")
    export(fig,4)

    fig=plt.figure(figsize=(7,4.65))
    gs=fig.add_gridspec(2,2,height_ratios=[1.1,1],left=.09,right=.985,bottom=.13,top=.91,hspace=.62,wspace=.38)
    ax=fig.add_subplot(gs[0,:]);ax.set_axis_off();panel_label(ax,'a','Conceptual motivation — not a proposed method')
    boxes=[(.14,'Frozen TSFM\n+ output adaptation'),(.48,'Lower adaptation cost'),(.84,'Limited observed accuracy'),
           (.14,'TSFM + internal PEFT'),(.48,'Higher adaptation computation'),(.84,'Better observed accuracy')]
    for j,(x,text) in enumerate(boxes):
        y=.82 if j<3 else .43
        ax.text(x,y,text,transform=ax.transAxes,ha='center',va='center',fontsize=7.5,
                bbox=dict(boxstyle='square,pad=.4',facecolor='white',edgecolor='#888888',lw=.6))
    for y in [.82,.43]:
        for a,b in [(.285,.34),(.65,.72)]:
            ax.annotate('',xy=(b,y),xytext=(a,y),xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='->',lw=.7,color='#555555'))
    ax.text(.5,-.02,'Research target\nCan we retain the accuracy benefit of internal adaptation at a lower adaptation cost?',
            transform=ax.transAxes,ha='center',va='center',fontsize=8)
    for j,p in enumerate(['jena','ettm2']):
        aa=fig.add_subplot(gs[1,j]);point_panel(aa,5,p,['HEAD',DATA[p]['internal']])
        panel_label(aa,chr(98+j),f'{NAMES[p]}: measured TEST error')
    seed_legend(fig)
    export(fig,5)

    fig,axs=plt.subplots(1,2,figsize=(7,2.95))
    fig.subplots_adjust(left=.09,right=.985,bottom=.27,top=.85,wspace=.32)
    for i,p in enumerate(PANELS):
        point_panel(axs[i],6,p,['F0_SHORT','F0_LONG','HEAD',DATA[p]['internal']],
                    {'F0_SHORT':'F0 short\n(4 days)','F0_LONG':'F0 long\n(8 days)'})
        panel_label(axs[i],chr(97+i),NAMES[p])
    seed_legend(fig)
    export(fig,6)

def captions():
    e,j=[DATA[p]['effect'] for p in PANELS]
    boundary='The output-only HEAD continued to improve at the maximum training budget, so this comparison does not establish an intrinsic limitation of output adaptation.'
    cost='Costs correspond to the selected checkpoints and do not represent equal-quality time-to-target ratios.'
    jena='The reported cost difference is not an equal-quality time-to-target comparison.'
    common='Errors are TRAIN-standard-deviation-scaled twice-pinball losses (lower is better). ETTm2 and Jena are development datasets, not independent external replications. LoRA+ is an existing method, not a method proposed here.'
    seed=f"Small open and filled markers denote seeds {DATA['ettm2']['seeds'][0]} and {DATA['ettm2']['seeds'][1]}, respectively; short horizontal strokes denote their arithmetic mean. Deterministic F0 references have one marker."
    effect=f"Validation-selected LoRA improves over HEAD by {e.mean_gain_pct:.3f}% on ETTm2, and validation-selected LoRA+ by {j.mean_gain_pct:.3f}% on Jena (ratios of seed-mean TEST losses)."
    ci=f"Saved paired block-bootstrap effect intervals are [{e.ci_low:.3f}%, {e.ci_high:.3f}%] and [{j.ci_low:.3f}%, {j.ci_high:.3f}%], respectively; these are not confidence intervals on the raw-score points. No new intervals were computed."
    content=[
      ('Internal adaptation accuracy gap', f'{seed} {effect} {ci} The numerical y-axis limits are displayed and do not start at zero. {common} {boundary} {jena}',
       '고정된 학습 조건에서 출력 HEAD와 내부 적응의 정확도 차이를 보여준다. 평균과 두 seed를 모두 표시한다. 신뢰구간은 저장된 개선율 구간이며 원점수 오차막대가 아니다. ETTm2 HEAD의 학습이 더 진행되면 격차가 줄 가능성을 배제하지 않는다.'),
      ('Quality–cost trade-off', f'Small markers show each seed at its own validation-selected checkpoint; large markers show arithmetic means of their costs and TEST errors. F0_LONG is a deterministic reference. Lower-left indicates lower error and lower adaptation cost. Cold-ready costs include the recorded preparation, model loading, charged feature cache, training, validation and checkpoint I/O through selection. F0 uses the saved reference-ready cost, including its reference evaluation. Timing variation is visible in seed points; ETTm2 LoRA+ fits were flagged as noisy in the source receipts. {cost} {jena} {common} {boundary}',
       '왼쪽 아래일수록 정확도와 비용 모두 유리하다. 비용은 서로 다른 품질의 선택 체크포인트까지 걸린 값이다. ETTm2 LoRA+의 시간 변동도 숨기지 않는다. 같은 정확도에 도달하는 속도비로 해석하면 안 된다.'),
      ('Adaptation trajectories', f'Each thin line follows the recorded validation checkpoints of one method and seed: (a,b) ETTm2 and (c,d) Jena, against optimizer updates and actual cold-ready cost. Method is encoded by color, marker and line style; open/filled markers distinguish the two seeds. No time coordinates were averaged or interpolated into new observations. In Jena, internal methods reached their selected validation checkpoints near 64–128 updates; later updates degraded validation performance in this fixed run. This observation does not identify the cause. {boundary} The Jena legacy TIME_TO_TARGET threshold, based on degraded final-step internal performance, is not used as adaptation-acceleration evidence. {jena} {common}',
       'Jena 내부 적응은 64–128 step의 선택 시점 이후 검증 성능이 악화했다. 이것만으로 과적합의 원인을 입증하지 않는다. ETTm2 HEAD는 512 step에서도 개선 중이다. 시간축은 seed별 실제 기록이며 느슨해진 기존 목표 도달 표는 가속 증거로 사용하지 않는다.'),
      ('Channel-wise internal gain', f'For every recorded channel, gain is 100 times (HEAD mean loss minus selected internal mean loss), divided by HEAD mean loss. Means are taken across the two seeds before computing the ratio. Internal methods were selected on validation: LoRA for ETTm2 and LoRA+ for Jena. Positive values favor internal adaptation; negative values favor HEAD. All channels are retained in source order, and no channel-level confidence intervals are invented. {common} {boundary} {jena}',
       '채널별 두 seed 평균 원점수의 비율을 사용했다. 양수와 음수 채널을 모두 남겨 전체 평균 개선을 모든 채널의 개선으로 오해하지 않도록 했다.'),
      ('Research problem motivation', f'Panel (a) is a conceptual summary of the observed trade-off, not an evaluated architecture or a proposed method; the research target has no measured performance. Panels (b,c) are measured Jena and ETTm2 TEST errors of HEAD and their validation-selected internal comparator. {seed} {effect} These results motivate investigating whether the observed accuracy benefit can be retained at lower adaptation cost; they do not establish a solution. {common} {boundary} {jena}',
       '교수님께 먼저 보여줄 문제 정의 그림이다. 위쪽은 개념도, 아래쪽은 실제 수치다. 새로운 방법의 성공을 주장하지 않고 내부 적응의 관찰된 이득을 더 낮은 비용으로 유지할 수 있는지를 연구 질문으로 제시한다.'),
      ('Context length control', f'F0_SHORT uses four days of history and is a context-length diagnostic. F0_LONG and all adaptation arms use the same eight-day context. The internal comparator is validation-selected LoRA on ETTm2 and LoRA+ on Jena. {seed} The comparison controls this specific context-length alternative rather than establishing an intrinsic requirement for internal adaptation. {common} {boundary} {jena}',
       'HEAD와 내부 적응은 모두 같은 8일 입력을 받는다. 4일 F0는 문맥 길이 진단이며 주 비교의 유리한 기준선으로 바꾸지 않았다. 관찰된 내부 적응 이득을 단순한 입력 길이 차이와 구분하는 보조 그림이다.')]
    text='# Publication captions / 미팅 설명\n\n'
    for i,(title,en,ko) in enumerate(content,1):
        text+=f'## Figure {i}. {title}\n\n**English.** {en}\n\n**한국어.** {ko}\n\n'
    (OUT/'CAPTIONS.md').write_text(text.rstrip()+'\n',encoding='utf-8')
    for p in PANELS:
        row=DATA[p]['effect']
        for col in ['mean_gain_pct','ci_low','ci_high']:
            record('captions',p,DATA[p]['internal'],'mean','TEST',col,row[col],f'{p}/EFFECTS.csv',
                   dict(candidate=DATA[p]['internal'],baseline='HEAD'))
    messages=['Output-only vs internal accuracy','Accuracy–adaptation cost trade-off','Adaptation dynamics',
              'Channel-wise heterogeneity','Research problem summary','Longer-context control']
    sections=['Motivation / Results','Motivation','Analysis','Analysis','Introduction','Ablation']
    index='# Internal adaptation gap — paper figures\n\n'
    index+='Visualization only; scientific decisions are unchanged. All figures are 7 inches wide, generated with Matplotlib from saved results, with editable-text SVG, embedded-font PDF and 600 dpi PNG. Font files are not distributed.\n\n'
    index+='| Figure | Main message | Possible paper section | Files |\n| --- | --- | --- | --- |\n'
    for i,name in enumerate(FILES):
        links=' · '.join(f'[{ext.upper()}]({name}.{ext})' for ext in ['pdf','svg','png'])
        index+=f'| Fig. {i+1} | {messages[i]} | {sections[i]} | {links} |\n'
    index+='\n## 미팅에서 보여줄 순서\n\n1. Fig.5: 실제 정확도 격차에서 출발한 연구 질문이며 아직 제안 방법은 없다.\n2. Fig.1과 Fig.2: 정확도 이득과 선택 시점 비용을 함께 보되 동일 품질 가속으로 해석하지 않는다.\n3. Fig.3: ETTm2 HEAD의 예산 경계 개선과 Jena의 후반 악화를 확인해 주장의 범위를 제한한다.\n\n'
    index+='## Reproduction and audit\n\nRun `python results/internal_adaptation_gap_v1_20260922/paper_figures/generate_figures.py` from the repository root using its existing Python environment. This script reads CSV/JSON/Markdown only; it never loads data arrays, weights, predictions or checkpoints and performs no training/inference.\n\n'
    index+='[Captions](CAPTIONS.md) · [All plotted values and provenance](figure_values.csv) · [Verification](FIGURE_VERIFICATION.json) · [Generation script](generate_figures.py) · [Original summary](../SUMMARY_KO.md)\n\n'
    index+='## Previews\n\n'
    for i in [4,0,1,2,3,5]:index+=f'![Figure {i+1}: {messages[i]}]({FILES[i]}.png)\n\n'
    (OUT/'FIGURE_INDEX.md').write_text(index.rstrip()+'\n',encoding='utf-8')

def verify_values():
    frame=pd.read_csv(OUT/'figure_values.csv',float_precision='round_trip',keep_default_na=False)
    for r in frame.itertuples():
        sel=json.loads(r.selector);path=BASE/r.source
        if path.suffix=='.json':expected=read(path)[r.metric]
        else:
            df=pd.read_csv(path,float_precision='round_trip')
            if r.operation=='ratio_of_channel_means':
                df=df[(df.variant==sel['variant']) & (df.channel==sel['channel'])]
                a=df[df.arm==sel['arm']].primary.to_numpy()
                b=df[df.arm==sel['baseline']].primary.to_numpy()
                expected=100*(b.mean()-a.mean())/b.mean()
            else:
                for key,val in sel.items():df=df[df[key]==val]
                assert len(df)>0
                expected=df[r.metric].mean() if r.operation=='mean' else df[r.metric].iloc[0]
                if r.operation=='identity':assert len(df)==1
        close(r.value,expected)
    # Exact method/seed/step coverage in each figure, including every channel/curve checkpoint.
    for p,d in DATA.items():
        for f,arms in [(1,['F0_LONG']+ARMS),(2,['F0_LONG']+ARMS),(5,['HEAD',d['internal']]),
                       (6,['F0_SHORT','F0_LONG','HEAD',d['internal']])]:
            actual=frame[(frame.figure==str(f)) & (frame.panel==p) & (frame.metric=='primary') & (frame.seed!='mean')]
            expected=d['selected'][d['selected'].arm.isin(arms)]
            assert set(zip(actual.arm,actual.seed))==set(zip(expected.arm,expected.seed.astype(str)))
        curve=frame[(frame.figure=='3') & (frame.panel==p) & (frame.metric=='step')]
        assert set(zip(curve.arm,curve.seed,curve.value))==set(zip(d['CURVES'].arm,d['CURVES'].seed.astype(str),d['CURVES'].step))
        channel=frame[(frame.figure=='4') & (frame.panel==p)]
        assert channel.channel.tolist()==d['audit']['columns']
    return len(frame)

def verify_outputs(before):
    assert source_snapshot()==before, 'Existing results changed'
    count=verify_values()
    for name in FILES:
        for ext in ['pdf','svg','png']:
            p=OUT/f'{name}.{ext}';assert p.is_file() and p.stat().st_size>1000
        with Image.open(OUT/f'{name}.png') as im:
            dpi=im.info['dpi'];assert all(abs(x-600)<.05 for x in dpi)
            size=FORMATS[name]['inches'];assert im.size==tuple(round(x*600) for x in size)
            FORMATS[name].update(png_pixels=list(im.size),png_dpi=list(dpi))
        svg=ET.parse(OUT/f'{name}.svg').getroot()
        width,height=[float(svg.attrib[k].replace('pt','')) for k in ['width','height']]
        for a,b in zip([width,height],np.array(size)*72):close(a,b)
        raw=(OUT/f'{name}.pdf').read_bytes()
        box=re.search(rb'/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]',raw)
        assert box and b'/FontFile2' in raw
        for a,b in zip(map(float,box.groups()),np.array(size)*72):close(a,b)
        assert '<image' not in (OUT/f'{name}.svg').read_text(encoding='utf-8')
    for md in ['CAPTIONS.md','FIGURE_INDEX.md']:
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',(OUT/md).read_text(encoding='utf-8')):
            if link != 'FIGURE_VERIFICATION.json':
                assert (OUT/link).is_file(),link
    captions_text=(OUT/'CAPTIONS.md').read_text(encoding='utf-8')
    for p in PANELS:assert f"{DATA[p]['effect'].mean_gain_pct:.3f}%" in captions_text
    assert all(not re.search(r'\b'+re.escape(x)+r'\b',captions_text.lower())
               for x in ['significantly better','proves','universally','state-of-the-art','lora is necessary','faster by'])
    result=dict(status='PASS',source_result_hashes=before,source_unchanged=True,
        original_git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_audits=CHECKS,figure_value_rows_verified=count,formats=FORMATS,
        no_new_training=True,no_new_inference=True,no_new_bootstrap=True,
        source_scope='Saved CSV/JSON/Markdown only; no raw arrays, prediction NPZ or model state read',
        numeric_assertion_tolerance=dict(relative=1e-10,absolute=2e-12),
        method_seed_channel_curve_coverage=True,markdown_links_valid=True,
        font=FONT,matplotlib=mpl.__version__,numpy=np.__version__,pandas=pd.__version__,
        pdf_fonts_embedded=True,svg_text_editable=True,vector_has_no_raster_images=True,
        visual_review='Inspect all previews at final size; redundant method marker/line encodings are present',
        artifact_hashes={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='FIGURE_VERIFICATION.json'})
    (OUT/'FIGURE_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert (OUT/'FIGURE_VERIFICATION.json').is_file()
    print(json.dumps({'status':'PASS','value_rows':count,'figures':len(FILES),'font':FONT,'audits':CHECKS},indent=2))

def main():
    before=source_snapshot()
    audit_sources()
    make_figures()
    captions()
    pd.DataFrame(VALUES).to_csv(OUT/'figure_values.csv',index=False)
    verify_outputs(before)

if __name__=='__main__':main()
