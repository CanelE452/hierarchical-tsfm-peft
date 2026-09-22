"""Paper figures for emulator_response_gonogo_v1_20260922 (visualization only).

Reads the result CSVs written by the experiment, plus this folder's config.json for the
pre-registered screening thresholds drawn in Fig. 2. Never imports model or experiment code,
never trains or predicts. Every plotted coordinate and every number in CAPTIONS.md is computed
from those files and written to FIGURE_VALUES.csv.

    python experiments/emulator_response_gonogo_v1_20260922/figures.py [--results DIR]

Outputs: DIR/figures/{name}.png (600 dpi) / .svg / .pdf, DIR/FIGURE_VALUES.csv, DIR/CAPTIONS.md.
Exit code 0 only when all of them exist; a missing column or row is a hard error.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE/'config.json').read_text(encoding='utf-8'))
DEFAULT_RESULTS = HERE.parents[1]/'results'/CFG['experiment_id']

# Same rcParams as results/internal_adaptation_gap_v1_20260922/paper_figures/generate_figures.py;
# only the SVG hash salt differs. Font files are not shipped with the repository.
FONT = next(n for n in ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
            if n in {f.name for f in font_manager.fontManager.ttflist})
mpl.rcParams.update({'font.family': FONT, 'font.size': 8, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': .55, 'xtick.major.width': .5, 'ytick.major.width': .5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'svg.hashsalt': 'emulator-response-gonogo-v1'})

# Arm display: the only place for labels, colours (Okabe-Ito, as in the viz palette) and markers.
# The candidate (E_RESPONSE_DELTA) gets no extra weight, size or emphasis.
STYLE = {
    'F_FULL':           dict(label='Full LoRA+',     color='#0072B2', marker='o'),
    'F_TOP3':           dict(label='Top-3 LoRA+',    color='#56B4E9', marker='s'),
    'E_EMLOC':          dict(label='EMLoC-style',    color='#009E73', marker='^'),
    'E_DELTA':          dict(label='Delta',          color='#E69F00', marker='D'),
    'E_VALUE_DELTA':    dict(label='Delta+value',    color='#CC79A7', marker='v'),
    'E_RESPONSE_DELTA': dict(label='Delta+response', color='#D55E00', marker='h'),
}
REF_COLOR = '0.35'          # F0 and prior-run references
ARMS = CFG['arms']
E_ARMS = [a for a in ARMS if a.startswith('E_')]
EMULATOR_ARM = {'svd': 'E_DELTA', 'value': 'E_VALUE_DELTA', 'response': 'E_RESPONSE_DELTA'}  # delta arm trained on each emulator
EMULATOR_LABEL = {'svd': 'SVD\n(uncalibrated)', 'value': 'Value\ncalibrated', 'response': 'Response\ncalibrated'}
DELTA_ARMS = list(EMULATOR_ARM.values())
FIGURES = ['fig1_transferred_quality', 'fig2_total_cost_frontier', 'fig3_cost_breakdown',
           'fig4_transfer_mismatch', 'fig5_response_ablation']
ERR = 'TEST scaled twice-pinball'

# Fig. 3 categories of the cold-start ledger: (label, tab20c colour index, components). Every charged
# component must belong to exactly one category, so no cost can silently drop out of the bars.
COST_CATEGORIES = [
    ('Data preparation', 16, ['prepare:data_prepare']),
    ('Model loads', 17, ['compress:full_model_load', 'full_model_load', 'emulator_load', 'full_model_load_transfer']),
    ('Activation stats + SVD + emulator build', 12,
     ['compress:activation_collection', 'compress:svd_factorization', 'compress:emulator_build']),
    ('Teacher probes + gamma calibration', 13,
     ['calibrate:teacher_zero_calibration', 'calibrate:teacher_probe_calibration',
      'calibrate:initial_error_calibration', 'calibrate:calibration_VALUE', 'calibrate:calibration_RESPONSE']),
    ('F0/E0 forecast caches', 14, [f'caches:{m}_cache{k}_{role}' for m, ks in (('f0', ['']), ('e0', ['_svd', '_value', '_response']))
                                   for k in ks for role in ('train', 'val')]),
    ('Main updates', 0, ['main_updates']),
    ('VAL evaluation (original model)', 1, ['val_eval']),
    ('Checkpoint I/O', 2, ['checkpoint_io']),
    ('Transfer/correction + selection', 8, ['transfer', 'selection_export']),
]

REQUIRED = {
    'SCORES.csv': ['arm', 'seed', 'variant', 'step', 'primary'],
    'RESOURCES.csv': ['arm', 'seed', 'test_primary', 'total_s', 'preparation_s', 'fit_procedure_s',
                      'transfer_selection_s', 'lora_modules'],
    'COST_COMPONENTS.csv': ['ledger', 'arm', 'seed', 'group', 'component', 'seconds'],
    'TRANSFER_GAP.csv': ['arm', 'seed', 'step', 'emulator_training_val_primary', 'transferred_val_primary',
                         'transfer_gap', 'train_vs_actual_sigma_rmse', 'naive_val_primary'],
    'CALIBRATION_RESPONSE.csv': ['origin', 'label', 'variant', 'L_value', 'L_response'],
}

VALUES = []


def fail(msg):
    sys.exit(f'figures.py: ERROR: {msg}')


def rec(fig, panel, series, seed, x, y, unit, source):
    """One plotted (or captioned) value. x/y are the plotted coordinates; a category label where the axis is categorical."""
    num = lambda v: float(v) if isinstance(v, (int, float, np.number)) else v
    VALUES.append(dict(figure=fig, panel=panel, series=series, seed=seed, x=num(x), y=num(y), unit=unit,
                       source_file=source))


def reduction(value, reference):
    """Percent reduction relative to the reference; positive = lower than the reference."""
    return 100*(reference-value)/reference


def load(results):
    data = {}
    for name, cols in REQUIRED.items():
        path = results/name
        if not path.is_file():
            fail(f'{path} is missing')
        df = pd.read_csv(path, float_precision='round_trip')
        missing = [c for c in cols if c not in df.columns]
        if missing:
            fail(f'{name} lacks required column(s) {missing}; found {list(df.columns)}')
        data[name[:-4]] = df
    return data


def check(D):
    """Coverage and cross-file consistency; returns the two main seeds read from SCORES.csv."""
    S = D['SCORES']; sel = S[S.variant == 'selected']
    if set(sel.arm) != set(ARMS):
        fail(f'SCORES.csv selected arms {sorted(set(sel.arm))} differ from config arms {ARMS}')
    seeds = sorted(int(s) for s in sel.seed.unique())
    if len(seeds) != 2:
        fail(f'expected two main seeds in SCORES.csv, found {seeds}')
    for arm in ARMS:
        got = sorted(sel[sel.arm == arm].seed.astype(int))
        if got != seeds:
            fail(f'SCORES.csv arm {arm} has selected seeds {got}, expected {seeds}')
    if (S.variant == 'reference_f0').sum() != 1:
        fail('SCORES.csv needs exactly one reference_f0 row')
    R = D['RESOURCES']
    for arm in ARMS:
        for s in seeds:
            r = R[(R.arm == arm) & (R.seed == s)]
            if len(r) != 1:
                fail(f'RESOURCES.csv needs one row for {arm} seed {s}, found {len(r)}')
            p = sel[(sel.arm == arm) & (sel.seed == s)].primary.iloc[0]
            if not np.isclose(r.test_primary.iloc[0], p, rtol=1e-12, atol=0):
                fail(f'RESOURCES.test_primary differs from SCORES.primary for {arm} seed {s}')
    G = D['TRANSFER_GAP']
    steps = sorted(int(x) for x in G.step.unique())
    for arm in E_ARMS:
        for s in seeds:
            got = sorted(G[(G.arm == arm) & (G.seed == s)].step.astype(int))
            if got != steps:
                fail(f'TRANSFER_GAP.csv {arm} seed {s} has steps {got}, expected {steps}')
    if max(steps) != CFG['main_steps']:
        fail(f'TRANSFER_GAP.csv last step {max(steps)} differs from config main_steps {CFG["main_steps"]}')
    C = D['CALIBRATION_RESPONSE']; H = C[C.label == 'heldout']
    origins = {v: sorted(H[H.variant == v].origin) for v in EMULATOR_ARM}
    if any(not o or o != origins['svd'] for o in origins.values()):
        fail('CALIBRATION_RESPONSE.csv needs the same held-out origins for variants svd/value/response')
    per_seed = cost_by_category(D, seeds)       # Fig. 3 must add up to the Fig. 2 totals
    for arm in ARMS:
        for s in seeds:
            total = float(R[(R.arm == arm) & (R.seed == s)].total_s.iloc[0])
            part = float(per_seed[(per_seed.index.get_level_values(0) == arm)
                                  & (per_seed.index.get_level_values(1) == s)].sum())
            if not np.isclose(part, total, rtol=1e-9, atol=1e-9):
                fail(f'{arm} seed {s}: charged components sum to {part} s but RESOURCES.total_s is {total} s')
    return seeds, steps


def panel_label(ax, letter, text='', dy=4):
    ax.annotate(f'({letter})', (0, 1), xycoords='axes fraction', xytext=(0, dy), textcoords='offset points',
                fontsize=9, fontweight='bold', va='bottom', ha='left')
    if text:
        ax.annotate(text, (0, 1), xycoords='axes fraction', xytext=(17, dy), textcoords='offset points',
                    fontsize=8, va='bottom', ha='left')


def value_axis(ax, grid=True):
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis='y', color='#E6E6E6', linewidth=.4)
    ax.ticklabel_format(axis='y', useOffset=False)


def seed_points(ax, fig, panel, x, xlabel, rows, value_col, color, marker, seeds, unit, source, series):
    """Two seeds (open = first, filled = second) at x -/+ 0.12 and their mean as a stroke; returns the mean."""
    for j, s in enumerate(seeds):
        v = float(rows[rows.seed == s][value_col].iloc[0])
        ax.plot(x+(-.12, .12)[j], v, marker, ms=3.8, mec=color, mfc=('white', color)[j], mew=.8, ls='')
        rec(fig, panel, series, s, xlabel, v, unit, source)
    mean = float(rows[rows.seed.isin(seeds)][value_col].mean())
    ax.plot([x-.22, x+.22], [mean, mean], color=color, lw=1.8, solid_capstyle='butt')
    rec(fig, panel, series, 'mean', xlabel, mean, unit, source)
    return mean


def seed_handles(seeds, color='0.3'):
    return [Line2D([], [], marker='o', mfc='white', mec=color, mew=.8, ls='', ms=3.8, label=f'Seed {seeds[0]}'),
            Line2D([], [], marker='o', mfc=color, mec=color, ls='', ms=3.8, label=f'Seed {seeds[1]}')]


def export(fig, out, name):
    for ext in ('pdf', 'svg', 'png'):
        meta = {'Description': name} if ext == 'png' else {'Title': name, 'Creator': 'Matplotlib; saved-result visualization only'}
        if ext == 'pdf':
            meta.update(CreationDate=None, ModDate=None)
        elif ext == 'svg':
            meta['Date'] = None
        path = out/f'{name}.{ext}'
        fig.savefig(path, dpi=600, metadata=meta)
        if ext == 'svg':
            path.write_text('\n'.join(l.rstrip() for l in path.read_text(encoding='utf-8').splitlines())+'\n',
                            encoding='utf-8')
    plt.close(fig)


def method_key(D):
    R = D['RESOURCES']
    mods = {a: int(R[R.arm == a].lora_modules.iloc[0]) for a in ('F_FULL', 'F_TOP3')}
    L = {a: STYLE[a]['label'] for a in ARMS}
    return (f"Methods: {L['F_FULL']} (F_FULL) trains LoRA+ on {mods['F_FULL']} modules of the original Chronos-2; "
            f"{L['F_TOP3']} (F_TOP3) on {mods['F_TOP3']} modules in the last three encoder blocks and the output projection; "
            f"{L['E_EMLOC']} (E_EMLOC) trains on the SVD emulator and applies an EMLoC-style correction when moving the LoRA "
            f"to the original model; {L['E_DELTA']} (E_DELTA), {L['E_VALUE_DELTA']} (E_VALUE_DELTA) and "
            f"{L['E_RESPONSE_DELTA']} (E_RESPONSE_DELTA, the candidate) compute the training loss on the frozen "
            f"original-model forecast plus the forecast change that the LoRA causes in the uncalibrated, value-calibrated "
            f"or response-calibrated emulator, respectively, and copy the LoRA to the original model unchanged. All scores "
            f"are of the original model carrying the transferred LoRA.")


def common(seeds, errors=True):
    metric = (" Errors are quantile twice-pinball losses scaled by the TRAIN standard deviation of each series "
              "(lower is better).") if errors else ''
    return (f"Jena only, seeds {seeds[0]} and {seeds[1]}, a fixed {CFG['main_steps']}-update budget.{metric} "
            f"Two seeds are not a significance test.")


# ---------------------------------------------------------------------------------------------- Fig. 1
def fig1(D, seeds, out):
    S = D['SCORES']; sel = S[S.variant == 'selected']; src = 'SCORES.csv'; name = FIGURES[0]
    f0 = float(S[S.variant == 'reference_f0'].primary.iloc[0])
    prior = S[S.variant == 'reference_prior_run'].sort_values('seed')
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    fig.subplots_adjust(left=.085, right=.99, bottom=.2, top=.97)
    means = {}
    for i, arm in enumerate(ARMS):
        st = STYLE[arm]
        means[arm] = seed_points(ax, name, 'a', i, st['label'], sel[sel.arm == arm], 'primary', st['color'],
                                 st['marker'], seeds, ERR, src, arm)
    labels = [STYLE[a]['label'] for a in ARMS]
    if len(prior):
        x = len(ARMS)
        ax.axvline(x-.5, color='0.7', lw=.5, ls=':')
        for j, r in enumerate(prior.itertuples()):
            ax.plot(x+(-.12, .12)[j % 2], r.primary, 'x', ms=3.8, color=REF_COLOR, mew=.8)
            rec(name, 'a', 'HEAD_PRIOR_RUN', int(r.seed), 'HEAD (prior run)', r.primary, ERR, src)
        pm = float(prior.primary.mean())
        ax.plot([x-.22, x+.22], [pm, pm], color=REF_COLOR, lw=1.8, solid_capstyle='butt')
        rec(name, 'a', 'HEAD_PRIOR_RUN', 'mean', 'HEAD (prior run)', pm, ERR, src)
        labels.append('HEAD\n(prior run, reference)')
    ax.axhline(f0, color=REF_COLOR, lw=.8, ls='--')
    rec(name, 'a', 'F0', 0, '', f0, ERR, src)
    ax.set_xticks(range(len(labels)), labels)
    if len(prior):
        ax.get_xticklabels()[-1].set_color(REF_COLOR)
    ax.set_xlim(-.55, len(labels)-.45)
    ax.margins(y=.12)
    ax.set_ylabel(ERR)
    value_axis(ax)
    handles = seed_handles(seeds) + [
        Line2D([], [], color='0.3', lw=1.8, label='Two-seed mean'),
        Line2D([], [], color=REF_COLOR, lw=.8, ls='--', label='F0: original model, no adaptation')]
    if len(prior):
        handles.append(Line2D([], [], marker='x', color=REF_COLOR, ls='', ms=3.8, mew=.8,
                              label=f"HEAD prior run, seeds {'/'.join(str(int(s)) for s in prior.seed)}"))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, bbox_to_anchor=(.53, .0))
    export(fig, out, name)

    full = means['F_FULL']
    red = {a: reduction(means[a], full) for a in ARMS if a != 'F_FULL'}
    for a, v in red.items():
        rec(name, 'caption', f'{a}|mean error reduction vs F_FULL', 'mean', '', v, '%', src)
    steps = '; '.join(f"{STYLE[a]['label']} {'/'.join(str(int(sel[(sel.arm == a) & (sel.seed == s)].step.iloc[0])) for s in seeds)}"
                      for a in ARMS)
    prior_txt = ''
    if len(prior):
        prior_txt = (f" Crosses on the right are the output-head adaptation (HEAD) of an earlier run with other seeds "
                     f"({', '.join(str(int(s)) for s in prior.seed)}; mean {pm:.4f}); it is shown for orientation only and "
                     f"is not part of this comparison.")
    text = (f"TEST error of the original Chronos-2 model after the LoRA of each method is placed on it. Open and filled "
            f"markers are seeds {seeds[0]} and {seeds[1]}; horizontal strokes are two-seed means. Each seed uses its own "
            f"validation-selected checkpoint (selected update, seed {seeds[0]}/{seeds[1]}: {steps}). The dashed line is "
            f"the unadapted original model (F0, {f0:.4f}). Mean error reduction relative to {STYLE['F_FULL']['label']} "
            f"({full:.4f}; positive = lower error): "
            + ', '.join(f"{STYLE[a]['label']} {v:+.2f}%" for a, v in red.items()) + '.'
            + prior_txt + f" The y-axis does not start at zero. {common(seeds)} {method_key(D)}")
    return 'Transferred TEST quality', text


# ---------------------------------------------------------------------------------------------- Fig. 2
def fig2(D, seeds, out):
    R = D['RESOURCES']; src = 'RESOURCES.csv'; name = FIGURES[1]
    fig, ax = plt.subplots(figsize=(3.5, 3.6))
    fig.subplots_adjust(left=.19, right=.97, bottom=.34, top=.98)
    means = {}
    for arm in ARMS:
        st = STYLE[arm]; rr = R[R.arm == arm]
        mx, my = float(rr.total_s.mean()), float(rr.test_primary.mean())
        # Mean underneath, seeds on top, so a mean never hides a seed.
        ax.plot(mx, my, st['marker'], ms=6.5, mfc=st['color'], mec='white', mew=.5, alpha=.55, zorder=2)
        for j, s in enumerate(seeds):
            r = rr[rr.seed == s].iloc[0]
            ax.plot([mx, r.total_s], [my, r.test_primary], color=st['color'], lw=.45, zorder=2)
            ax.plot(r.total_s, r.test_primary, st['marker'], ms=3.2, mec=st['color'], mfc=('white', st['color'])[j],
                    mew=.7, zorder=3)
            rec(name, 'a', arm, s, r.total_s, r.test_primary, f's | {ERR}', src)
        rec(name, 'a', arm, 'mean', mx, my, f's | {ERR}', src)
        means[arm] = (mx, my)
    tf, ef = means['F_FULL']
    saving, margin = CFG['minimum_total_time_saving'], CFG['quality_relative_margin']
    tx, ty = (1-saving)*tf, (1+margin)*ef
    ax.set_xlim(0, 1.08*R.total_s.max())
    ax.margins(y=.18)
    y0 = ax.get_ylim()[0]
    ax.fill_between([0, tx], y0, ty, color='0.5', alpha=.12, lw=0, zorder=0)
    ax.axvline(tx, color='0.45', lw=.6, ls=':', zorder=1)
    ax.axhline(ty, color='0.45', lw=.6, ls=':', zorder=1)
    ax.set_ylim(bottom=y0)
    rec(name, 'a', 'threshold|total_s', 'mean', tx, np.nan, 's', 'RESOURCES.csv + config.json')
    rec(name, 'a', 'threshold|test_primary', 'mean', np.nan, ty, ERR, 'RESOURCES.csv + config.json')
    ax.set_xlabel('Total adaptation time incl. preparation (s)')
    ax.set_ylabel(ERR)
    value_axis(ax)
    ax.grid(axis='x', color='#E6E6E6', linewidth=.4)
    arm_h = [Line2D([], [], marker=STYLE[a]['marker'], color=STYLE[a]['color'], ls='', ms=5, label=STYLE[a]['label'])
             for a in ARMS]
    key_h = seed_handles(seeds) + [
        Line2D([], [], marker='o', mfc='0.3', mec='white', ls='', ms=6.5, alpha=.55, label='Two-seed mean')]
    region_h = [Patch(facecolor='0.5', alpha=.12, edgecolor='none',
                      label=f"Screening region for means: time ≤ {1-saving:.2f}×, error ≤ {1+margin:.3f}× "
                            f"{STYLE['F_FULL']['label']}")]
    for handles, ncol, y in ((arm_h, 3, .105), (key_h, 3, .045), (region_h, 1, -.005)):
        fig.add_artist(fig.legend(handles=handles, loc='lower center', ncol=ncol, frameon=False,
                                  bbox_to_anchor=(.53, y), columnspacing=1.0, handletextpad=.4))
    export(fig, out, name)

    saved = {a: reduction(means[a][0], tf) for a in ARMS if a != 'F_FULL'}
    for a, v in saved.items():
        rec(name, 'caption', f'{a}|mean total-time saving vs F_FULL', 'mean', '', v, '%', src)
    inside = [STYLE[a]['label'] for a in ARMS if a != 'F_FULL' and means[a][0] <= tx and means[a][1] <= ty]
    text = (f"Total adaptation time versus TEST error of the original model carrying the transferred LoRA. Total time "
            f"charges each method its full cold-start cost: data preparation, every compression, calibration and cache "
            f"stage it depends on (charged in full even when physically shared between methods), all "
            f"{CFG['main_steps']} updates, checkpoint I/O, transfer or correction and original-model VAL at every "
            f"checkpoint, and final selection/export; diagnostic-only computation is excluded. Small open and filled "
            f"markers are seeds {seeds[0]} and {seeds[1]}; large markers are two-seed means, joined to their seeds by thin "
            f"lines. Dotted lines are the pre-registered screening thresholds relative to the {STYLE['F_FULL']['label']} "
            f"means ({tf:.1f} s, {ef:.4f}): total time at most {tx:.1f} s ({100*saving:.0f}% saving) and TEST error at "
            f"most {ty:.4f} (+{100*margin:.1f}%). The shaded region satisfies both for the means only; the screening "
            f"decision also requires both seeds to satisfy the conditions and requires timing-reliability checks, which "
            f"are not drawn. Methods whose means fall in the shaded region: {', '.join(inside) if inside else 'none'}. "
            f"Mean total-time saving relative to {STYLE['F_FULL']['label']} (positive = faster): "
            + ', '.join(f"{STYLE[a]['label']} {v:+.1f}%" for a, v in saved.items())
            + f". The x-axis starts at zero; the y-axis does not. {common(seeds)}")
    return 'Total adaptation time and transferred TEST error', text


# ---------------------------------------------------------------------------------------------- Fig. 3
def cost_by_category(D, seeds):
    """Charged cold-start seconds per (arm, seed, Fig. 3 category)."""
    C = D['COST_COMPONENTS']
    C = C[(C.ledger == 'cold_start_charged') & C.arm.isin(ARMS) & C.seed.isin(seeds)]
    where = {comp: label for label, _, comps in COST_CATEGORIES for comp in comps}
    unknown = sorted(set(C.component) - set(where))
    if unknown:
        fail(f'COST_COMPONENTS.csv has charged components without a Fig. 3 category: {unknown}')
    return C.assign(category=C.component.map(where)).groupby(['arm', 'seed', 'category']).seconds.sum()


def fig3(D, seeds, out):
    R = D['RESOURCES']; src = 'COST_COMPONENTS.csv'; name = FIGURES[2]
    per_seed = cost_by_category(D, seeds)
    colors = plt.get_cmap('tab20c').colors
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    fig.subplots_adjust(left=.12, right=.64, bottom=.17, top=.97)
    ys = np.arange(len(ARMS))
    for yi, arm in zip(ys, ARMS):
        left = 0.
        for label, ci, _ in COST_CATEGORIES:
            v = float(np.mean([per_seed.get((arm, s, label), 0.) for s in seeds]))
            if v > 0:
                ax.barh(yi, v, left=left, height=.62, color=colors[ci], edgecolor='white', linewidth=.4)
            rec(name, 'a', label, 'mean', v, STYLE[arm]['label'], 's', src)
            left += v
        totals = [float(R[(R.arm == arm) & (R.seed == s)].total_s.iloc[0]) for s in seeds]
        for j, (s, t) in enumerate(zip(seeds, totals)):
            ax.plot(t, yi+(-.14, .14)[j], 'o', ms=2.6, mec='black', mfc=('white', 'black')[j], mew=.6, zorder=4)
            rec(name, 'a', 'total', s, t, STYLE[arm]['label'], 's', 'RESOURCES.csv')
        ax.annotate(f'{np.mean(totals):.0f} s', (max(totals), yi), xytext=(5, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=7)
        rec(name, 'a', 'total', 'mean', float(np.mean(totals)), STYLE[arm]['label'], 's', 'RESOURCES.csv')
    ax.set_yticks(ys, [STYLE[a]['label'] for a in ARMS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.13*R[R.arm.isin(ARMS)].total_s.max())
    ax.set_xlabel('Cold-start time per method, two-seed mean (s)')
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#E6E6E6', linewidth=.4)
    handles = [Patch(facecolor=colors[ci], edgecolor='none', label=label) for label, ci, _ in COST_CATEGORIES]
    handles += seed_handles(seeds, 'black')
    for h in handles[-2:]:
        h.set_label(h.get_label()+' total')
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.655, .56), frameon=False, handlelength=1.4)
    export(fig, out, name)

    means = {a: float(R[(R.arm == a) & R.seed.isin(seeds)].total_s.mean()) for a in ARMS}
    upd = {a: float(np.mean([per_seed.get((a, s, 'Main updates'), 0.) for s in seeds])) for a in ARMS}
    share = {a: 100*upd[a]/means[a] for a in ARMS}
    for a in ARMS:
        rec(name, 'caption', f'{a}|main-update share of mean total', 'mean', '', share[a], '%', src)
    text = (f"Where the cold-start time of each method goes (two-seed mean, seconds only). Components are those charged "
            f"in Fig. 2: grey = data preparation and model loads (the original-model load for compression, the model load "
            f"at the start of each fit and the original-model reload for transfer); purple = emulator-only preparation "
            f"(activation statistics, SVD and emulator construction; teacher probe forecasts, initial error and gamma "
            f"calibration; F0/E0 training and VAL forecast caches); blue = the fixed {CFG['main_steps']}-update procedure "
            f"(updates, original-model VAL at every checkpoint, checkpoint I/O); green = transfer or correction plus final "
            f"selection/export. Open and filled black markers are the per-seed totals of seeds {seeds[0]} and {seeds[1]}; "
            f"numbers are two-seed mean totals ("
            + ', '.join(f"{STYLE[a]['label']} {means[a]:.1f} s" for a in ARMS)
            + "). For every method and seed the components sum to the total used in Fig. 2 (checked when this figure "
            f"is generated). Main updates account for "
            + ', '.join(f"{share[a]:.0f}%" for a in ARMS)
            + f" of the mean totals, in the order above. {common(seeds, errors=False)}")
    return 'Cold-start cost breakdown', text


# ---------------------------------------------------------------------------------------------- Fig. 4
def fig4(D, seeds, steps, out):
    G = D['TRANSFER_GAP']; src = 'TRANSFER_GAP.csv'; name = FIGURES[3]
    fig, axs = plt.subplots(2, len(E_ARMS), figsize=(7.2, 4.1), sharex=True, sharey='row')
    fig.subplots_adjust(left=.085, right=.99, bottom=.2, top=.9, hspace=.42, wspace=.1)
    for k, arm in enumerate(E_ARMS):
        st = STYLE[arm]; top, bottom = axs[0, k], axs[1, k]
        for j, s in enumerate(seeds):
            g = G[(G.arm == arm) & (G.seed == s)].sort_values('step')
            kw = dict(marker=st['marker'], ms=3.2, mec=st['color'], mfc=('white', st['color'])[j], mew=.7)
            top.plot(g.step, g.transferred_val_primary, color=st['color'], lw=.8, ls='-', **kw)
            top.plot(g.step, g.emulator_training_val_primary, color=st['color'], lw=.8, ls='--', **kw)
            bottom.plot(g.step, g.train_vs_actual_sigma_rmse, color=st['color'], lw=.8, ls='-', **kw)
            if g.naive_val_primary.notna().any():
                top.plot(g.step, g.naive_val_primary, color=REF_COLOR, lw=.8, ls=':', marker='x', ms=3, mew=.7)
            for r in g.itertuples():
                rec(name, 'a', f'{arm}|transferred', s, int(r.step), r.transferred_val_primary, 'VAL scaled twice-pinball', src)
                rec(name, 'a', f'{arm}|training-time', s, int(r.step), r.emulator_training_val_primary,
                    'VAL scaled twice-pinball', src)
                if pd.notna(r.naive_val_primary):
                    rec(name, 'a', f'{arm}|uncorrected transfer', s, int(r.step), r.naive_val_primary,
                        'VAL scaled twice-pinball', src)
                rec(name, 'b', f'{arm}|rms difference', s, int(r.step), r.train_vs_actual_sigma_rmse, 'TRAIN sigma', src)
        top.annotate(st['label'], (.5, 1), xycoords='axes fraction', xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=8)
        for ax in (top, bottom):
            value_axis(ax)
            ax.set_xticks(steps)
            ax.tick_params(axis='x', labelsize=7)
        bottom.set_xlabel('Optimizer updates')
    panel_label(axs[0, 0], 'a', dy=14)
    panel_label(axs[1, 0], 'b')
    axs[0, 0].set_ylabel('VAL scaled twice-pinball')
    axs[1, 0].set_ylabel('RMS forecast difference (TRAIN σ)')
    axs[1, 0].set_ylim(bottom=0)
    handles = [Line2D([], [], color='0.3', lw=.8, ls='-', label='Transferred to original model'),
               Line2D([], [], color='0.3', lw=.8, ls='--', label='Training-time forecast'),
               Line2D([], [], color=REF_COLOR, lw=.8, ls=':', marker='x', ms=3, label='EMLoC-style without correction')]
    handles += seed_handles(seeds)
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, bbox_to_anchor=(.53, .0))
    export(fig, out, name)

    last = max(steps)
    L = G[G.step == last]
    gap = {a: float(L[(L.arm == a) & L.seed.isin(seeds)].transfer_gap.mean()) for a in E_ARMS}
    rms = {a: float(L[(L.arm == a) & L.seed.isin(seeds)].train_vs_actual_sigma_rmse.mean()) for a in E_ARMS}
    for a in E_ARMS:
        rec(name, 'caption', f'{a}|mean transfer gap at step {last}', 'mean', last, gap[a], 'VAL scaled twice-pinball', src)
        rec(name, 'caption', f'{a}|mean rms difference at step {last}', 'mean', last, rms[a], 'TRAIN sigma', src)
    text = (f"Forecast used during training versus the forecast actually obtained after transfer, for the four "
            f"emulator-trained methods at every recorded checkpoint ({', '.join(str(s) for s in steps)} updates). "
            f"(a) VAL error of the training-time forecast (dashed; for {STYLE['E_EMLOC']['label']} the emulator's own "
            f"forecast, for the Delta methods the frozen original forecast plus the emulator's change) and of the original "
            f"model carrying the transferred LoRA (solid). For {STYLE['E_EMLOC']['label']} the dotted grey line is the "
            f"same LoRA transferred without correction (diagnostic). (b) Root-mean-square difference between the "
            f"training-time and transferred forecasts over VAL origins, series, quantiles and horizon, in units of the "
            f"TRAIN standard deviation. Only the transferred VAL error is used to select checkpoints; the other curves are "
            f"diagnostics. Open and filled markers are seeds {seeds[0]} and {seeds[1]}. At update {last}, the two-seed "
            f"mean transfer gap (transferred minus training-time VAL error) is "
            + ', '.join(f"{STYLE[a]['label']} {gap[a]:+.4f}" for a in E_ARMS)
            + '; the mean RMS difference is '
            + ', '.join(f"{STYLE[a]['label']} {rms[a]:.4f}" for a in E_ARMS)
            + f". Panels in a row share the y-axis. {common(seeds)}")
    return 'Training-time versus transferred forecasts', text


# ---------------------------------------------------------------------------------------------- Fig. 5
def fig5(D, seeds, out):
    Cal = D['CALIBRATION_RESPONSE']; S = D['SCORES']; name = FIGURES[4]
    H = Cal[Cal.label == 'heldout']; sel = S[S.variant == 'selected']
    variants = list(EMULATOR_ARM)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    fig.subplots_adjust(left=.075, right=.99, bottom=.25, top=.9, wspace=.42)
    cal_means = {}
    for ax, col, letter, title, ylabel in (
            (axs[0], 'L_value', 'a', 'Held-out value error', 'Value error ratio, L_value'),
            (axs[1], 'L_response', 'b', 'Held-out response error', 'Response error ratio, L_response')):
        wide = H.pivot(index='origin', columns='variant', values=col)[variants]
        for o, row in wide.iterrows():
            ax.plot(range(len(variants)), row.to_numpy(), color='0.72', lw=.5, marker='o', ms=2, mfc='0.6', mec='none')
            for i, v in enumerate(variants):
                rec(name, letter, f'{v}|origin {int(o)}', '', v, row[v], 'ratio', 'CALIBRATION_RESPONSE.csv')
        for i, v in enumerate(variants):
            m = float(wide[v].mean()); cal_means[(col, v)] = m
            ax.plot([i-.24, i+.24], [m, m], color=STYLE[EMULATOR_ARM[v]]['color'], lw=1.8, solid_capstyle='butt')
            rec(name, letter, v, 'mean_over_heldout_origins', v, m, 'ratio', 'CALIBRATION_RESPONSE.csv')
        ax.set_xticks(range(len(variants)), [EMULATOR_LABEL[v] for v in variants])
        ax.set_xlim(-.5, len(variants)-.5)
        ax.set_ylim(bottom=0, top=1.08*float(np.nanmax(wide.to_numpy())))
        ax.set_ylabel(ylabel)
        value_axis(ax)
        panel_label(ax, letter, title)
    ax = axs[2]; test = {}
    for i, arm in enumerate(DELTA_ARMS):
        st = STYLE[arm]
        test[arm] = seed_points(ax, name, 'c', i, st['label'], sel[sel.arm == arm], 'primary', st['color'],
                                st['marker'], seeds, ERR, 'SCORES.csv', arm)
    ax.set_xticks(range(len(DELTA_ARMS)), [STYLE[a]['label'].replace('+', '\n+') for a in DELTA_ARMS])
    ax.set_xlim(-.5, len(DELTA_ARMS)-.5)
    ax.margins(y=.2)
    ax.set_ylabel(ERR)
    value_axis(ax)
    panel_label(ax, 'c', 'Transferred TEST error')
    handles = [Line2D([], [], color='0.72', lw=.5, marker='o', ms=2, mfc='0.6', mec='none', label='One held-out origin'),
               Line2D([], [], color='0.3', lw=1.8, label='Mean over origins (a, b) / seeds (c)')] + seed_handles(seeds)
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, bbox_to_anchor=(.53, .0))
    export(fig, out, name)

    rv = {s: reduction(float(sel[(sel.arm == 'E_RESPONSE_DELTA') & (sel.seed == s)].primary.iloc[0]),
                       float(sel[(sel.arm == 'E_VALUE_DELTA') & (sel.seed == s)].primary.iloc[0])) for s in seeds}
    rv_mean = reduction(test['E_RESPONSE_DELTA'], test['E_VALUE_DELTA'])
    rd_mean = reduction(test['E_RESPONSE_DELTA'], test['E_DELTA'])
    rec(name, 'caption', 'E_RESPONSE_DELTA|mean error reduction vs E_VALUE_DELTA', 'mean', '', rv_mean, '%', 'SCORES.csv')
    rec(name, 'caption', 'E_RESPONSE_DELTA|mean error reduction vs E_DELTA', 'mean', '', rd_mean, '%', 'SCORES.csv')
    for s in seeds:
        rec(name, 'caption', 'E_RESPONSE_DELTA|error reduction vs E_VALUE_DELTA', s, '', rv[s], '%', 'SCORES.csv')
    n_held = H.origin.nunique(); n_cal = Cal[Cal.label == 'calibration'].origin.nunique()
    svd_one = bool(np.allclose(H[H.variant == 'svd'].L_value, 1, rtol=0, atol=1e-6))
    text = (f"Response calibration versus value calibration of the emulator, and what it changes after transfer. "
            f"(a, b) Calibration diagnostics on {n_held} held-out TRAIN origins that were not among the {n_cal} "
            f"calibration origins and use a different probe seed: (a) L_value, the emulator's output error relative to "
            f"that of the uncalibrated SVD emulator on the same origin"
            + (' (1 for the SVD emulator by construction)' if svd_one else '')
            + "; (b) L_response, the error of the emulator's response to a small LoRA probe relative to the original "
            f"model's response magnitude. Grey points joined by lines follow one origin across the three emulators; "
            f"coloured strokes are means over origins (L_value: "
            + ', '.join(f"{EMULATOR_LABEL[v].replace(chr(10), ' ')} {cal_means[('L_value', v)]:.3f}" for v in variants)
            + '; L_response: '
            + ', '.join(f"{EMULATOR_LABEL[v].replace(chr(10), ' ')} {cal_means[('L_response', v)]:.3f}" for v in variants)
            + f"). Both seeds use the same calibrated emulators, so (a, b) have no seed dimension. (c) TEST error of the "
            f"original model carrying the transferred LoRA of the three Delta methods (open and filled markers: seeds "
            f"{seeds[0]} and {seeds[1]}; strokes: two-seed means). Mean error reduction of "
            f"{STYLE['E_RESPONSE_DELTA']['label']} relative to {STYLE['E_VALUE_DELTA']['label']} is {rv_mean:+.2f}% "
            f"(seed {seeds[0]} {rv[seeds[0]]:+.2f}%, seed {seeds[1]} {rv[seeds[1]]:+.2f}%; positive = lower error) and "
            f"relative to {STYLE['E_DELTA']['label']} {rd_mean:+.2f}%. Proxy improvement is not forecast success: a lower "
            f"calibration error in (a, b) does not by itself imply a lower forecast error, and (c) is the outcome that "
            f"counts. {common(seeds)}")
    return 'Response versus value calibration', text


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--results', type=Path, default=DEFAULT_RESULTS,
                    help='folder holding the result CSVs (default: %(default)s)')
    results = ap.parse_args().results.resolve()
    D = load(results)
    seeds, steps = check(D)
    out = results/'figures'
    out.mkdir(exist_ok=True)
    captions = [fig1(D, seeds, out), fig2(D, seeds, out), fig3(D, seeds, out), fig4(D, seeds, steps, out),
                fig5(D, seeds, out)]
    pd.DataFrame(VALUES, columns=['figure', 'panel', 'series', 'seed', 'x', 'y', 'unit', 'source_file']).to_csv(
        results/'FIGURE_VALUES.csv', index=False)
    md = [f"# Figure captions — {CFG['experiment_id']}", '',
          f"Generated by `experiments/{CFG['experiment_id']}/figures.py` from the CSV files in this folder. Every number "
          f"below is computed from those files; plotted values are listed in [FIGURE_VALUES.csv](FIGURE_VALUES.csv).", '']
    for i, (fname, (title, text)) in enumerate(zip(FIGURES, captions), 1):
        md += [f'## Figure {i}. {title}', '', f'![Figure {i}](figures/{fname}.png)', '',
               f'Fig. {i}. {text}', '',
               f'Files: [PDF](figures/{fname}.pdf) · [SVG](figures/{fname}.svg) · [PNG](figures/{fname}.png)', '']
    (results/'CAPTIONS.md').write_text('\n'.join(md), encoding='utf-8')
    missing = [f'{n}.{e}' for n in FIGURES for e in ('png', 'svg', 'pdf')
               if not (out/f'{n}.{e}').is_file() or (out/f'{n}.{e}').stat().st_size == 0]
    missing += [p for p in ('FIGURE_VALUES.csv', 'CAPTIONS.md') if not (results/p).is_file()]
    if missing:
        fail(f'outputs missing after generation: {missing}')
    print(f'figures.py: wrote {3*len(FIGURES)} figure files to {out}, {len(VALUES)} rows to FIGURE_VALUES.csv, CAPTIONS.md')


if __name__ == '__main__':
    main()
