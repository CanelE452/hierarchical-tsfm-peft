#!/usr/bin/env python
"""Figures for gap atlas Stage 2 (contract section 7).

Visualization only: this script reads the saved result CSVs and never imports experiment,
data or model code. Every plotted coordinate - bar heights, confidence whiskers, curve
points, scatter points and the reference lines - is exported to FIGURE_VALUES.csv, so a
reader can recompute any mark of any figure from the source CSVs without rerunning the
study. One figure is one file and no subplots are used (contract section 7).

    python figures.py [--results DIR] [--period P1|P2] [--configs C1,C2] [--no-localization]

Inputs (all read from --results):
    STABILITY.csv                              figure 1
    GAP_TABLE.csv                              consistency check and caption numbers
    LOCALIZATION/L1_horizon_{id}_{period}.csv  figure 2
    LOCALIZATION/L6_calibration_{id}_{period}.csv  figure 3

Exit codes: 0 every contracted figure was written, 2 an input is missing or inconsistent -
in which case nothing at all is written, not even a partial figure.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------------- style
# These are diagnostic figures: they carry the evidence for the contract verdict, they are
# not paper figures. The style file is the viz-expert analysis-mode style, vendored here
# byte-identical to ~/.claude/agents/viz-expert/assets/analysis.mplstyle (same practice as
# ga.py vendoring the official GIFT-Eval loader) so that the appearance is fixed by a file
# rather than by rcParams scattered through this script, and so that the script keeps
# working on a machine that does not have the agent assets.
STYLE = HERE/'analysis.mplstyle'
if not STYLE.is_file():
    raise SystemExit(f'ERROR: figure style file is missing: {STYLE}')
plt.style.use(STYLE)
# The one addition on top of the style file: a fixed salt makes the element ids inside the
# SVG deterministic, so that rerunning this script produces byte-identical output.
mpl.rcParams['svg.hashsalt'] = 'gap-atlas-stage2-20260924'

# Colours follow the viz-expert single source of truth
# (~/.claude/agents/viz-expert/assets/palette.py, Okabe-Ito, colour-vision safe).
# color_for / colors_for are mirrored rather than imported because this script may only
# depend on matplotlib, numpy and pandas. The mapping is name -> colour through an md5 of
# the lowercased name, so a series keeps its colour across sessions and machines and does
# not move when the drawing order changes.
_FALLBACK = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00', '#CC79A7']
_REGISTRY = {'ours': '#D55E00', 'baseline': '#0072B2', 'gt': '#000000', 'pred': '#E69F00',
             'train': '#0072B2', 'val': '#E69F00', 'test': '#009E73',
             'before': '#56B4E9', 'after': '#D55E00'}
SEMANTIC = {'mean': '0.4', 'threshold': '#D55E00', 'baseline': '0.6', 'zero': '0.5',
            'annotation': '0.25'}
LINESTYLE = {'mean': '-', 'threshold': ':', 'baseline': '-.', 'zero': '-'}
SEPARATOR = '0.80'          # group divider: layout furniture, not a data reference


def color_for(label: str) -> str:
    key = str(label).strip().lower()
    if key in _REGISTRY:
        return _REGISTRY[key]
    digest = hashlib.md5(key.encode('utf-8')).digest()
    return _FALLBACK[int.from_bytes(digest[:4], 'big') % len(_FALLBACK)]


def colors_for(labels) -> list:
    out, used = [], set()
    for label in labels:
        colour = color_for(label)
        if colour in used:
            colour = next((c for c in _FALLBACK if c not in used), colour)
        out.append(colour)
        used.add(colour)
    return out


# ---------------------------------------------------------------------------- vocabulary
FIG1 = 'fig1_stability_gap'
FIG2 = 'fig2_horizon_curve'
FIG3 = 'fig3_interval_coverage'

G_THRESHOLD = 0.10          # contract 6 S2a/S2b; drawn here, decided in score.py
PERIODS = ['P1', 'P2']
PERIOD_NOTE = {'P1': 'P1 (windows before the official test span)',
               'P2': 'P2 (official GIFT-Eval test windows)'}
PERIOD_COLOR = {'P1': color_for('P1'), 'P2': color_for('P2')}
ARM_COLOR = {'F0': color_for('baseline'), 'ACH': color_for('ACH')}
CONFIG_MARKERS = ['o', '^', 's', 'D', 'v', 'P', 'X']

GAP_LABEL = ('Gap  G = (CRPS_F0 - CRPS_ACH) / CRPS_F0\n'
             '[dimensionless, > 0 = training helped]')
RATIO_LABEL = ('CRPS ratio  ACH / F0  at step h\n'
               '[dimensionless, < 1 = training helped]')
STEP_LABEL = 'Forecast horizon step  h   [1 = first step of the window]'
COVERAGE_LABEL = ('Empirical coverage of the 80% interval\n'
                  '[share of points inside the 0.1-0.9 quantile band]')
WIDTH_LABEL = 'Mean interval width / mean |y|   [dimensionless]'

REQUIRED = {
    'STABILITY.csv': ['id', 'config', 'role', 'verdict', 'G_P1', 'G_P2', 'ci_P1', 'ci_P2',
                      'ach_P1', 'ach_P2', 'note'],
    'GAP_TABLE.csv': ['id', 'config', 'role', 'period', 'arm', 'is_ach', 'crps',
                      'crps_official', 'mase', 'G_crps', 'G_crps_ci_low', 'G_crps_ci_high',
                      'G_mase', 'n_units', 'unit', 'val_crps', 'selected', 'seconds'],
}
L1_COLUMNS = ['step', 'crps_f0', 'crps_ach', 'ratio', 'gap']
L6_COLUMNS = ['arm', 'nominal_coverage', 'empirical_coverage', 'mean_width',
              'mean_width_over_mean_abs_y']
VERDICTS = {'STABLE_GAP', 'UNSTABLE_GAP', 'NO_GAP', 'INFEASIBLE'}
CONTROL_TAGS = {'CONTROL_OK', 'CONTROL_FAIL'}
RATIO_TOL = 1e-9            # L1 ratio against crps_ach / crps_f0
GAP_TOL = 1e-5              # STABILITY rounds G to five decimals


class DataError(Exception):
    """Raised when the saved results cannot support the contracted figures."""


def fail(message):
    raise DataError(message)


def fmt(value, digits=3) -> str:
    value = float(value)
    return 'n/a' if not np.isfinite(value) else f'{value:.{digits}f}'


# ------------------------------------------------------------------------------- loading
def sha8(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:8]


def to_float(value, where: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        fail(f'{where}: expected a number, found {value!r}')
    if not np.isfinite(out):
        fail(f'{where}: expected a finite number, found {value!r}')
    return out


def parse_interval(text, where: str):
    """'[-0.0123, 0.0456]' as written by score.py -> (low, high)."""
    raw = str(text).strip()
    if not (raw.startswith('[') and raw.endswith(']')):
        fail(f'{where}: expected an interval like "[-0.0123, 0.0456]", found {text!r}')
    parts = raw[1:-1].split(',')
    if len(parts) != 2:
        fail(f'{where}: expected two bounds, found {text!r}')
    low, high = (to_float(p, where) for p in parts)
    if high < low:
        fail(f'{where}: the upper bound is below the lower bound ({text!r})')
    return low, high


def read_table(path: Path, columns: list, label: str) -> pd.DataFrame:
    if not path.is_file():
        fail(f'missing input: {path}  ({label})')
    frame = pd.read_csv(path, float_precision='round_trip')
    absent = [c for c in columns if c not in frame.columns]
    if absent:
        fail(f'{path}: missing column(s) {", ".join(absent)}')
    if frame.empty:
        fail(f'{path}: no data rows')
    return frame


def read_stability(results: Path) -> list:
    frame = read_table(results/'STABILITY.csv', REQUIRED['STABILITY.csv'],
                       'written by score.py')
    rows = []
    for position, raw in enumerate(frame.to_dict('records')):
        where = f'STABILITY.csv row {position + 2}'
        cid = str(raw['id']).strip()
        if not cid or cid.lower() == 'nan':
            fail(f'{where}: empty configuration id')
        role = str(raw['role']).strip()
        if role not in ('candidate', 'control'):
            fail(f'{where} ({cid}): role must be "candidate" or "control", found {role!r}')
        parts = [p.strip() for p in str(raw['verdict']).split('/')]
        if not parts or parts[0] not in VERDICTS:
            fail(f'{where} ({cid}): unknown verdict {raw["verdict"]!r}; '
                 f'expected one of {", ".join(sorted(VERDICTS))}')
        if len(parts) > 2:
            fail(f'{where} ({cid}): cannot read verdict {raw["verdict"]!r}')
        if len(parts) == 2:
            if role != 'control':
                fail(f'{where} ({cid}): a control tag on a {role} row ({raw["verdict"]!r})')
            if parts[1] not in CONTROL_TAGS:
                fail(f'{where} ({cid}): unknown control tag {parts[1]!r}')
        row = dict(id=cid, config=str(raw['config']).strip(), role=role,
                   verdict=parts[0], control_tag=parts[1] if len(parts) == 2 else '',
                   feasible=parts[0] != 'INFEASIBLE', gap={}, ci={}, ach={})
        for period in PERIODS:
            gap_cell, ci_cell, ach_cell = raw[f'G_{period}'], raw[f'ci_{period}'], raw[f'ach_{period}']
            if not row['feasible']:
                for name, cell in ((f'G_{period}', gap_cell), (f'ci_{period}', ci_cell),
                                   (f'ach_{period}', ach_cell)):
                    if not (pd.isna(cell) or str(cell).strip() == ''):
                        fail(f'{where} ({cid}): INFEASIBLE row carries {name}={cell!r}')
                continue
            row['gap'][period] = to_float(gap_cell, f'{where} ({cid}) G_{period}')
            row['ci'][period] = parse_interval(ci_cell, f'{where} ({cid}) ci_{period}')
            arm = str(ach_cell).strip()
            if not arm or arm.lower() == 'nan':
                fail(f'{where} ({cid}): ach_{period} is empty')
            row['ach'][period] = arm
        rows.append(row)
    ids = [r['id'] for r in rows]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        fail(f'STABILITY.csv: repeated configuration id(s) {", ".join(duplicated)}')
    if not any(r['feasible'] for r in rows):
        fail('STABILITY.csv: every configuration is INFEASIBLE, so there is nothing to plot')
    rows.sort(key=lambda r: (r['role'] != 'candidate', ids.index(r['id'])))
    return rows


def check_gap_table(results: Path, stability: list) -> pd.DataFrame:
    frame = read_table(results/'GAP_TABLE.csv', REQUIRED['GAP_TABLE.csv'],
                       'written by score.py')
    frame['id'] = frame['id'].astype(str).str.strip()
    frame['period'] = frame['period'].astype(str).str.strip()
    frame['arm'] = frame['arm'].astype(str).str.strip()
    frame['is_ach'] = frame['is_ach'].astype(str).str.strip().str.lower().isin(('true', '1'))
    unknown = sorted(set(frame['period']) - set(PERIODS))
    if unknown:
        fail(f'GAP_TABLE.csv: unexpected period value(s) {", ".join(unknown)}')
    for row in stability:
        if not row['feasible']:
            continue
        for period in PERIODS:
            block = frame[(frame['id'] == row['id']) & (frame['period'] == period)]
            if block.empty:
                fail(f'GAP_TABLE.csv: no rows for {row["id"]} {period}, '
                     f'but STABILITY.csv reports a gap for it')
            if 'F0' not in set(block['arm']):
                fail(f'GAP_TABLE.csv: {row["id"]} {period} has no F0 row')
            chosen = block[block['is_ach']]
            if len(chosen) != 1:
                fail(f'GAP_TABLE.csv: {row["id"]} {period} marks {len(chosen)} rows as '
                     f'is_ach, expected exactly one')
            arm = str(chosen.iloc[0]['arm'])
            if arm != row['ach'][period]:
                fail(f'{row["id"]} {period}: STABILITY.csv names ACH {row["ach"][period]!r} '
                     f'but GAP_TABLE.csv marks {arm!r}')
            table_gap = to_float(chosen.iloc[0]['G_crps'],
                                 f'GAP_TABLE.csv {row["id"]} {period} G_crps')
            if abs(table_gap - row['gap'][period]) > GAP_TOL:
                fail(f'{row["id"]} {period}: G disagrees between files '
                     f'(STABILITY {row["gap"][period]:.6f}, GAP_TABLE {table_gap:.6f}); '
                     f'report the disagreement instead of plotting it')
    return frame


def read_horizon(results: Path, cid: str, period: str) -> pd.DataFrame:
    path = results/'LOCALIZATION'/f'L1_horizon_{cid}_{period}.csv'
    frame = read_table(path, L1_COLUMNS, 'written by localize.py, contract L1')
    steps = frame['step'].to_numpy()
    if not np.array_equal(steps, np.arange(1, len(steps) + 1)):
        fail(f'{path}: step column must run 1..H without gaps, found '
             f'{steps[:5].tolist()}...{steps[-1] if len(steps) else "empty"}')
    for column in ('crps_f0', 'crps_ach', 'ratio'):
        values = np.asarray([to_float(v, f'{path} {column}') for v in frame[column]])
        frame[column] = values
    if (frame['crps_f0'].to_numpy() <= 0).any():
        fail(f'{path}: crps_f0 is not positive at some step, so the ratio is not meaningful')
    recomputed = frame['crps_ach'].to_numpy()/frame['crps_f0'].to_numpy()
    worst = float(np.max(np.abs(recomputed - frame['ratio'].to_numpy())))
    if worst > RATIO_TOL:
        fail(f'{path}: the ratio column disagrees with crps_ach / crps_f0 by {worst:.3g} '
             f'(tolerance {RATIO_TOL:g})')
    return frame


def read_calibration(results: Path, cid: str, period: str, ach_arm: str) -> pd.DataFrame:
    path = results/'LOCALIZATION'/f'L6_calibration_{cid}_{period}.csv'
    frame = read_table(path, L6_COLUMNS, 'written by localize.py, contract L6')
    frame['arm'] = frame['arm'].astype(str).str.strip()
    if len(frame) != 2:
        fail(f'{path}: expected two rows (F0 and the ACH arm), found {len(frame)}')
    arms = list(frame['arm'])
    for arm in ('F0', ach_arm):
        if arm not in arms:
            fail(f'{path}: no row for arm {arm!r} (found {", ".join(arms)})')
    for column in ('nominal_coverage', 'empirical_coverage', 'mean_width',
                   'mean_width_over_mean_abs_y'):
        frame[column] = [to_float(v, f'{path} {column}') for v in frame[column]]
    nominal = sorted(set(frame['nominal_coverage']))
    if len(nominal) != 1:
        fail(f'{path}: the two rows report different nominal coverage {nominal}')
    coverage = frame['empirical_coverage'].to_numpy()
    if ((coverage < 0) | (coverage > 1)).any():
        fail(f'{path}: empirical_coverage outside [0, 1]: {coverage.tolist()}')
    if (frame['mean_width_over_mean_abs_y'].to_numpy() < 0).any():
        fail(f'{path}: negative interval width')
    return frame


def load_inputs(results: Path, period: str, wanted: list, want_localization: bool) -> dict:
    """Read and check every input before a single output file is written."""
    if not results.is_dir():
        fail(f'results directory not found: {results}')
    stability = read_stability(results)
    gap_table = check_gap_table(results, stability)
    by_id = {row['id']: row for row in stability}
    data = dict(results=results, period=period, stability=stability, gap_table=gap_table,
                want_localization=want_localization, configs=[], horizon={},
                calibration={}, nominal=float('nan'),
                digest={name: sha8(results/name) for name in REQUIRED})
    if not want_localization:
        return data

    if wanted:
        unknown = [c for c in wanted if c not in by_id]
        if unknown:
            fail(f'--configs names {", ".join(unknown)}, which are not in STABILITY.csv '
                 f'(it holds {", ".join(sorted(by_id))})')
        blocked = [c for c in wanted if not by_id[c]['feasible']]
        if blocked:
            fail(f'--configs names INFEASIBLE configuration(s) {", ".join(blocked)}')
        configs = list(wanted)
    else:
        configs = [row['id'] for row in stability if row['verdict'] == 'STABLE_GAP']
        if not configs:
            fail('no configuration came out STABLE_GAP, so contract L1/L6 produced no '
                 'localization tables. Name the configurations explicitly with --configs, '
                 'or pass --no-localization to write figure 1 only')
    for cid in configs:
        data['horizon'][cid] = read_horizon(results, cid, period)
        data['calibration'][cid] = read_calibration(results, cid, period,
                                                    by_id[cid]['ach'][period])
    nominal = sorted({float(f['nominal_coverage'].iloc[0]) for f in data['calibration'].values()})
    if len(nominal) != 1:
        fail(f'the L6 tables report different nominal coverage across configurations: {nominal}')
    data['configs'] = configs
    data['nominal'] = nominal[0]
    data['digest'].update({f'L1_horizon_{c}_{period}.csv':
                           sha8(results/'LOCALIZATION'/f'L1_horizon_{c}_{period}.csv')
                           for c in configs})
    data['digest'].update({f'L6_calibration_{c}_{period}.csv':
                           sha8(results/'LOCALIZATION'/f'L6_calibration_{c}_{period}.csv')
                           for c in configs})
    return data


# ------------------------------------------------------------------------------- helpers
VALUES = []


def record(figure, element, series, label, x, y, y_low=np.nan, y_high=np.nan,
           unit='', source_file=''):
    """Every coordinate that reaches the canvas is written to FIGURE_VALUES.csv."""
    VALUES.append({'figure': figure, 'element': element, 'series': series, 'label': label,
                   'x': float(x), 'y': float(y), 'y_low': float(y_low),
                   'y_high': float(y_high), 'unit': unit, 'source_file': source_file})


def config_label(row) -> str:
    """Three short lines: id, dataset with frequency, forecast term."""
    key = row['config']
    head, _, tail = key.rpartition('/')
    return f'{row["id"]}\n{head or key}\n{tail}' if head else f'{row["id"]}\n{key}'


def combine(digests) -> str:
    """One short digest over several input files, so the footer stays on one line."""
    joined = ''.join(sorted(digests))
    return hashlib.sha256(joined.encode('utf-8')).hexdigest()[:8]


def footer(fig, results: Path, label: str, digest: str):
    """Provenance: where the numbers came from, and which bytes they were.

    The analysis-mode style asks for a source path and a timestamp here. A wall-clock
    timestamp would change the output on every run, which would break the reproducibility
    requirement of this contract, so the input digest takes its place: it identifies the
    exact result files the figure was drawn from.
    """
    right = f'inputs: {label}  sha256 {digest}'
    left = f'src: {results}'
    budget = int(fig.get_size_inches()[0]*72/4.6) - 4     # DejaVu Sans at 8 pt, with slack
    if len(left) + len(right) > budget:
        keep = max(12, budget - len(right) - 8)
        left = 'src: ...' + str(results)[-keep:]
    fig.text(0.005, 0.004, left, ha='left', va='bottom', fontsize=8, color='gray')
    fig.text(0.995, 0.004, right, ha='right', va='bottom', fontsize=8, color='gray')


def export(fig, outdir: Path, name: str) -> list:
    """PNG for reading on screen plus vector PDF and SVG, all byte-reproducible."""
    fig.canvas.draw()
    written = []
    for ext in ('png', 'pdf', 'svg'):
        meta = ({'Description': name} if ext == 'png'
                else {'Title': name, 'Creator': 'Matplotlib; saved-result visualization only'})
        if ext == 'pdf':
            meta.update(CreationDate=None, ModDate=None)
        elif ext == 'svg':
            meta['Date'] = None
        path = outdir/f'{name}.{ext}'
        fig.savefig(path, metadata=meta)
        written.append(path)
    plt.close(fig)
    return written


def free_corners(ax, xs, ys) -> list:
    """Corner names ordered from emptiest to busiest, so legends do not cover points."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xm, ym = (x0 + x1)/2, (y0 + y1)/2
    counts = {'upper left': 0, 'upper right': 0, 'lower left': 0, 'lower right': 0}
    for x, y in zip(xs, ys):
        counts[('upper ' if y >= ym else 'lower ') + ('right' if x >= xm else 'left')] += 1
    return sorted(counts, key=lambda k: (counts[k], k))


# ------------------------------------------------------------------------------ figure 1
def figure1(data, outdir: Path) -> dict:
    """Gap per configuration and period, candidates and negative controls side by side."""
    stability, results = data['stability'], data['results']
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    fig.get_layout_engine().set(rect=(0, 0.028, 1, 0.972))   # (left, bottom, width, height)

    positions, gap_between_roles = {}, 0.9
    offset = 0.0
    for index, row in enumerate(stability):
        if index and row['role'] == 'control' and stability[index - 1]['role'] == 'candidate':
            offset += gap_between_roles
            divider = index + offset - gap_between_roles/2 - 0.5
            ax.axvline(divider, color=SEPARATOR, lw=0.8, zorder=0)
        positions[row['id']] = index + offset

    width, lows, highs, infeasible = 0.38, [], [], []
    for row in stability:
        x = positions[row['id']]
        if not row['feasible']:
            infeasible.append(x)
            continue
        for period in PERIODS:
            gap = row['gap'][period]
            low, high = row['ci'][period]
            shift = -width/2 - 0.02 if period == 'P1' else width/2 + 0.02
            error = np.array([[max(0.0, gap - low)], [max(0.0, high - gap)]])
            ax.bar(x + shift, gap, width=width, color=PERIOD_COLOR[period], zorder=3,
                   yerr=error, capsize=2.5, ecolor='0.25',
                   error_kw=dict(lw=0.9, zorder=4))
            lows.append(min(gap, low))
            highs.append(max(gap, high))
            record(FIG1, 'bar', f'{row["id"]} {period}', row['config'], x + shift, gap,
                   low, high, 'gap ratio (dimensionless)', 'STABILITY.csv')

    span = max(highs) - min(min(lows), 0.0)
    top = max(max(highs), G_THRESHOLD) + 0.30*span
    bottom = min(min(lows), 0.0) - (0.30*span if min(lows) < 0 else 0.04*span)
    ax.set_ylim(bottom, top)
    pad = 0.02*(top - bottom)

    clipped = 0
    for row in stability:
        x = positions[row['id']]
        if not row['feasible']:
            ax.text(x, 0.0 + pad, 'INFEASIBLE', rotation=90, ha='center', va='bottom',
                    fontsize=8, color='0.35')
            continue
        for period in PERIODS:
            gap = row['gap'][period]
            low, high = row['ci'][period]
            clipped += int(gap < low or gap > high)
            shift = -width/2 - 0.02 if period == 'P1' else width/2 + 0.02
            if gap >= 0:
                ax.text(x + shift, max(gap, high) + pad, row['ach'][period], rotation=90,
                        ha='center', va='bottom', fontsize=7.5, color='0.15')
            else:
                ax.text(x + shift, min(gap, low) - pad, row['ach'][period], rotation=90,
                        ha='center', va='top', fontsize=7.5, color='0.15')

    ax.axhline(0.0, color=SEMANTIC['zero'], ls=LINESTYLE['zero'], lw=0.8, zorder=1)
    ax.axhline(G_THRESHOLD, color=SEMANTIC['threshold'], ls=LINESTYLE['threshold'], lw=1.2,
               zorder=2)
    record(FIG1, 'reference_line', 'zero', 'no change against F0', np.nan, 0.0,
           unit='gap ratio (dimensionless)', source_file='contract 5.4')
    record(FIG1, 'reference_line', 'decision threshold', 'decision threshold 0.10', np.nan,
           G_THRESHOLD, unit='gap ratio (dimensionless)', source_file='contract 6 S2a/S2b')

    ax.set_xticks([positions[r['id']] for r in stability])
    ax.set_xticklabels([config_label(r) for r in stability], fontsize=8.5)
    ax.set_xlim(-0.75, max(positions.values()) + 0.75)
    ax.set_ylabel(GAP_LABEL)
    ax.grid(axis='y')
    ax.grid(axis='x', visible=False)
    ax.yaxis.set_major_locator(MaxNLocator(8, steps=[1, 2, 2.5, 5, 10]))

    groups = {}
    for row in stability:
        groups.setdefault(row['role'], []).append(positions[row['id']])
    names = {'candidate': 'candidates', 'control': 'negative controls (no leaderboard gap)'}
    secondary = ax.secondary_xaxis(-0.20)
    secondary.set_xticks([float(np.mean(v)) for v in groups.values()])
    secondary.set_xticklabels([names[k] for k in groups], fontsize=9.5)
    secondary.tick_params(length=0)
    secondary.spines['bottom'].set_visible(False)

    stable = [r for r in stability if r['verdict'] == 'STABLE_GAP']
    candidates = [r for r in stability if r['role'] == 'candidate']
    controls = [r for r in stability if r['role'] == 'control' and r['feasible']]
    worst = max(((r, p) for r in controls for p in PERIODS),
                key=lambda t: t[0]['gap'][t[1]], default=None)
    control_note = ('no feasible negative control' if worst is None else
                    f'largest control gap {worst[0]["gap"][worst[1]]:+.3f} '
                    f'({worst[0]["id"]} {worst[1]})')
    ax.set_title('Does the Chronos-2 gap survive in a second period? '
                 'Gap of the best trained arm (ACH) against zero-shot F0, per period\n'
                 f'{len(stable)} of {len(candidates)} candidates reach G >= {G_THRESHOLD:.2f} '
                 f'with a positive CI lower bound in both periods; {control_note}', pad=26)
    handles = [Patch(facecolor=PERIOD_COLOR[p], label=PERIOD_NOTE[p]) for p in PERIODS]
    handles += [Line2D([], [], color='0.25', lw=0.9, label='95% bootstrap CI of G'),
                Line2D([], [], color=SEMANTIC['threshold'], ls=LINESTYLE['threshold'], lw=1.2,
                       label=f'decision threshold {G_THRESHOLD:.2f}')]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.005), ncols=4,
              frameon=False, fontsize=9)
    footer(fig, results, 'STABILITY.csv', data['digest']['STABILITY.csv'])
    export(fig, outdir, FIG1)
    return dict(stable=[r['id'] for r in stable], clipped=clipped,
                control_note=control_note)


# ------------------------------------------------------------------------------ figure 2
def figure2(data, outdir: Path) -> dict:
    """Per-step CRPS ratio: is the gap in the near steps or the far steps (contract L1)."""
    results, period, configs = data['results'], data['period'], data['configs']
    by_id = {row['id']: row for row in data['stability']}
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    fig.get_layout_engine().set(rect=(0, 0.028, 1, 0.972))

    colours = dict(zip(configs, colors_for(configs)))
    # One marker rule for every line: markers help at short horizons and clutter at long
    # ones, and mixing the two styles would read as a difference between configurations.
    longest = max(len(data['horizon'][c]) for c in configs)
    marker = 'o' if longest <= 40 else None
    horizons, ends, xs, ys = [], {}, [], []
    for cid in configs:
        frame = data['horizon'][cid]
        steps = frame['step'].to_numpy(dtype=float)
        ratio = frame['ratio'].to_numpy(dtype=float)
        horizons.append(len(steps))
        ends[cid] = (float(ratio[0]), float(ratio[-1]), int(steps[-1]))
        source = f'LOCALIZATION/L1_horizon_{cid}_{period}.csv'
        ax.plot(steps, ratio, color=colours[cid], lw=1.5, zorder=3, marker=marker,
                markersize=3.5,
                label=f'{cid} {by_id[cid]["config"]} - ACH {by_id[cid]["ach"][period]}')
        xs.extend(steps.tolist())
        ys.extend(ratio.tolist())
        for x, y in zip(steps, ratio):
            record(FIG2, 'curve_point', cid, by_id[cid]['config'], x, y,
                   unit='CRPS ratio ACH/F0 (dimensionless)', source_file=source)

    ax.axhline(1.0, color=SEMANTIC['baseline'], ls=LINESTYLE['baseline'], lw=1.0, zorder=2)
    record(FIG2, 'reference_line', 'parity', 'ACH equals F0 at that step', np.nan, 1.0,
           unit='CRPS ratio ACH/F0 (dimensionless)', source_file='contract 6 S2c L1')
    ax.set_xlim(0.5, max(horizons) + 0.5)
    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel(RATIO_LABEL)
    ax.xaxis.set_major_locator(MaxNLocator(10, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(8, steps=[1, 2, 2.5, 5, 10]))
    ax.set_title('Where along the horizon does the gap sit? Per-step CRPS of the ACH arm '
                 f'relative to F0, {PERIOD_NOTE[period]}\n'
                 'Values below the parity line mean training helped at that step; the '
                 'curve is descriptive and was not used for any selection', pad=12)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=SEMANTIC['baseline'], ls=LINESTYLE['baseline'],
                          lw=1.0))
    labels.append('parity with F0 (ratio = 1)')
    ax.legend(handles, labels, loc=free_corners(ax, xs, ys)[0], fontsize=9)
    footer(fig, results, f'LOCALIZATION/L1_horizon_*_{period}.csv',
           combine(data['digest'][f'L1_horizon_{c}_{period}.csv'] for c in configs))
    export(fig, outdir, FIG2)
    return ends


# ------------------------------------------------------------------------------ figure 3
def figure3(data, outdir: Path) -> dict:
    """Coverage and width of the 80% interval, F0 against ACH (contract L6)."""
    results, period, configs = data['results'], data['period'], data['configs']
    by_id = {row['id']: row for row in data['stability']}
    nominal = data['nominal']
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    fig.get_layout_engine().set(rect=(0, 0.028, 1, 0.972))

    markers = dict(zip(configs, CONFIG_MARKERS*(1 + len(configs)//len(CONFIG_MARKERS))))
    xs, ys, summary = [], [], {}
    for cid in configs:
        frame = data['calibration'][cid]
        source = f'LOCALIZATION/L6_calibration_{cid}_{period}.csv'
        point = {}
        for role, arm in (('F0', 'F0'), ('ACH', by_id[cid]['ach'][period])):
            row = frame[frame['arm'] == arm].iloc[0]
            x = float(row['empirical_coverage'])
            y = float(row['mean_width_over_mean_abs_y'])
            point[role] = (x, y)
            xs.append(x)
            ys.append(y)
            ax.plot([x], [y], marker=markers[cid], markersize=8, zorder=4,
                    color=ARM_COLOR[role], markeredgecolor='white', markeredgewidth=0.8,
                    linestyle='none')
            record(FIG3, 'point', f'{cid} {role} ({arm})', by_id[cid]['config'], x, y,
                   unit='x: coverage share, y: mean width / mean |y| (dimensionless)',
                   source_file=source)
        ax.annotate('', xy=point['ACH'], xytext=point['F0'], zorder=3,
                    arrowprops=dict(arrowstyle='-|>', color=SEMANTIC['annotation'], lw=0.9,
                                    shrinkA=7, shrinkB=4))
        summary[cid] = point

    ax.axvline(nominal, color=SEMANTIC['threshold'], ls=LINESTYLE['threshold'], lw=1.2,
               zorder=2)
    record(FIG3, 'reference_line', 'nominal coverage', f'nominal coverage {nominal:.2f}',
           nominal, np.nan, unit='coverage share', source_file='contract 6 S2c L6')
    pad_x = max(0.02, 0.10*(max(xs + [nominal]) - min(xs + [nominal])))
    pad_y = max(0.02, 0.10*(max(ys) - min(ys)))
    ax.set_xlim(max(0.0, min(xs + [nominal]) - pad_x), min(1.0, max(xs + [nominal]) + pad_x))
    ax.set_ylim(max(0.0, min(ys) - pad_y), max(ys) + pad_y)
    ax.set_xlabel(COVERAGE_LABEL)
    ax.set_ylabel(WIDTH_LABEL)
    ax.set_title('Is the gap a calibration problem? Coverage and relative width of the 80% '
                 f'interval, {PERIOD_NOTE[period]}\n'
                 'Each arrow runs from the zero-shot arm to the trained arm of the same '
                 'configuration; left of the dotted line the interval is too narrow', pad=22)

    arm_handles = [Line2D([], [], color=ARM_COLOR[r], marker='o', markersize=8,
                          markeredgecolor='white', linestyle='none',
                          label={'F0': 'F0 (Chronos-2 zero-shot)',
                                 'ACH': 'ACH (best trained arm)'}[r]) for r in ('F0', 'ACH')]
    arm_handles.append(Line2D([], [], color=SEMANTIC['threshold'], ls=LINESTYLE['threshold'],
                              lw=1.2, label=f'nominal coverage {nominal:.2f}'))
    config_handles = [Line2D([], [], color='0.35', marker=markers[c], markersize=8,
                             markeredgecolor='white', linestyle='none',
                             label=f'{c} {by_id[c]["config"]}') for c in configs]
    corners = free_corners(ax, xs, ys)
    first = ax.legend(handles=arm_handles, loc=corners[0], fontsize=9)
    ax.add_artist(first)
    ax.legend(handles=config_handles, loc=corners[1], fontsize=9, title='configuration',
              title_fontsize=9)
    footer(fig, results, f'LOCALIZATION/L6_calibration_*_{period}.csv',
           combine(data['digest'][f'L6_calibration_{c}_{period}.csv'] for c in configs))
    export(fig, outdir, FIG3)
    return summary


# ----------------------------------------------------------------------------- captions
def captions(data, fig1_stats, fig2_ends, fig3_points, results: Path):
    stability, period = data['stability'], data['period']
    by_id = {row['id']: row for row in stability}
    lines = ['# Figure captions - gap atlas Stage 2', '',
             'English captions for the figures of contract section 7. Every number below is '
             'computed by `figures.py` from the CSVs in this directory, and every plotted '
             'coordinate is exported to `FIGURE_VALUES.csv`, so any mark can be recomputed '
             'without rerunning the study. The gap is '
             'G = (CRPS_F0 - CRPS_ACH) / CRPS_F0, positive when the trained arm beats the '
             'zero-shot arm. No verdict is recomputed here; the verdicts are read from '
             '`STABILITY.csv`.', '']

    feasible = [r for r in stability if r['feasible']]
    candidates = [r for r in feasible if r['role'] == 'candidate']
    controls = [r for r in feasible if r['role'] == 'control']
    stable = [r for r in feasible if r['verdict'] == 'STABLE_GAP']
    unstable = [r for r in feasible if r['verdict'] == 'UNSTABLE_GAP']
    blocked = [r for r in stability if not r['feasible']]
    spread = [(r['id'], r['gap']['P2'] - r['gap']['P1']) for r in candidates]
    widest = max(spread, key=lambda t: abs(t[1])) if spread else ('n/a', float('nan'))
    lines += [f'## Figure 1 - `{FIG1}`', '',
              'Gap of the best trained arm against Chronos-2 zero-shot for every '
              'configuration of the contract, with the earlier period P1 and the official '
              'test period P2 side by side; whiskers are the 95% bootstrap interval of G '
              'and the arm named above each bar is the arm that the validation block '
              'selected. Read it by column: a configuration supports the gap claim only '
              'when both of its bars clear the dotted decision threshold at '
              f'{G_THRESHOLD:.2f} and both whiskers stay above zero, and the negative '
              'controls on the right should stay below it. '
              + ' '.join(f'{r["id"]} ({r["config"]}): G = {fmt(r["gap"]["P1"])} in P1 '
                         f'[{fmt(r["ci"]["P1"][0])}, {fmt(r["ci"]["P1"][1])}] and '
                         f'{fmt(r["gap"]["P2"])} in P2 '
                         f'[{fmt(r["ci"]["P2"][0])}, {fmt(r["ci"]["P2"][1])}], '
                         f'verdict {r["verdict"]}'
                         + (f' / {r["control_tag"]}' if r['control_tag'] else '') + '.'
                         for r in feasible)
              + f' The largest movement between the two periods among the candidates is '
                f'{fmt(widest[1])} ({widest[0]}); '
              + f'{len(stable)} of {len(candidates)} feasible candidates are STABLE_GAP and '
                f'{len(unstable)} {"is" if len(unstable) == 1 else "are"} UNSTABLE_GAP. '
              + (f'{len(blocked)} configuration{"" if len(blocked) == 1 else "s"} '
                 f'{"is" if len(blocked) == 1 else "are"} INFEASIBLE under contract 5.2 '
                 f'({", ".join(r["id"] for r in blocked)}) and carr'
                 f'{"ies" if len(blocked) == 1 else "y"} no bar. ' if blocked else '')
              + 'Limits: two periods cannot separate a stable gap from a slowly drifting '
                'one, the bootstrap resamples evaluation units and so understates the '
                'shared-window correlation, and the ACH arm is only as strong as the three '
                'trained arms of contract 5.3 - a small gap can mean a weak reference '
                'rather than an easy dataset.', '']
    if fig1_stats['clipped']:
        lines[-2] += (f' Note: for {fig1_stats["clipped"]} bar(s) the point estimate falls '
                      'outside its bootstrap interval; the whisker on that side is drawn at '
                      'the bar height while `FIGURE_VALUES.csv` keeps the original bounds.')

    if not data['want_localization']:
        lines += ['## Figures 2 and 3', '',
                  'Not produced: `figures.py` ran with `--no-localization`, so the contract '
                  'L1 and L6 tables were not read. Rerun without that flag once '
                  '`LOCALIZATION/` holds the tables for the configurations of interest.', '']
        (results/'CAPTIONS.md').write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')
        return

    configs = data['configs']
    horizon_bits = []
    for cid in configs:
        first, last, steps = fig2_ends[cid]
        frame = data['horizon'][cid]
        ratio = frame['ratio'].to_numpy(dtype=float)
        best = int(np.argmin(ratio)) + 1
        half = len(ratio)//2
        horizon_bits.append(
            f'{cid}: ratio {fmt(first)} at step 1 and {fmt(last)} at step {steps}, '
            f'lowest at step {best} ({fmt(float(ratio.min()))}), '
            f'mean {fmt(float(ratio[:half].mean()))} over the first half against '
            f'{fmt(float(ratio[half:].mean()))} over the second half')
    lines += [f'## Figure 2 - `{FIG2}`', '',
              'Per-step CRPS of the selected trained arm divided by the CRPS of the '
              f'zero-shot arm, over the forecast horizon of the {period} evaluation '
              f'windows ({PERIOD_NOTE[period].split("(", 1)[1].rstrip(")")}), one line per '
              'configuration. Read it against '
              'the parity line: a point below 1 means the trained arm was better at that '
              'step, so a curve that rises towards 1 says the gap lives in the early steps '
              'while a flat curve says the whole horizon carries it. '
              + '; '.join(horizon_bits) + '. '
              'Limits: this is a descriptive breakdown of the evaluation windows computed '
              'after selection and it carries no interval, steps are not independent of '
              'each other, and the denominator is the zero-shot CRPS of the same step, so '
              'a step where both arms are nearly perfect can still show a large ratio.', '']

    cal_bits = []
    for cid in configs:
        (cx, cy), (ax_, ay) = fig3_points[cid]['F0'], fig3_points[cid]['ACH']
        cal_bits.append(f'{cid} ({by_id[cid]["ach"][period]}): coverage {fmt(cx)} -> '
                        f'{fmt(ax_)} against a nominal {data["nominal"]:.2f}, relative '
                        f'width {fmt(cy)} -> {fmt(ay)}')
    lines += [f'## Figure 3 - `{FIG3}`', '',
              'Calibration of the 80% prediction interval (the 0.1 to 0.9 quantile band): '
              'empirical coverage against mean width divided by the mean absolute target, '
              'with one arrow per configuration running from the zero-shot arm to the '
              'trained arm. Read the horizontal axis first - points left of the dotted '
              'nominal line have intervals that are too narrow - then the vertical axis, '
              'which says whether the arm bought that coverage with a wider band. '
              + '; '.join(cal_bits) + '. '
              'This is the check of contract section 9 item four: if training mainly moves '
              'the interval width, a training-free recalibration could close the gap and '
              'the finding would not be about parameter-efficient fine-tuning. '
              'Limits: coverage is averaged over every step, variate and window, so a '
              'configuration can look calibrated on average while being wrong in each '
              'regime, and width is scaled by the mean absolute target of the same '
              'window, which is itself noisy for bursty series.', '']

    lines += ['---', '',
              f'Generated by `figures.py` from `{results}` (period {period}, configurations '
              f'{", ".join(configs)}). Style: `analysis.mplstyle`; PNG at 150 dpi plus '
              'vector PDF and SVG. Input digests: '
              + ', '.join(f'{n} sha256 {d}' for n, d in sorted(data['digest'].items())) + '.']
    (results/'CAPTIONS.md').write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')


# ---------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    default = Path(__file__).resolve().parents[2]/'results'/'gap_atlas_stage2_20260924'
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results', type=Path, default=default,
                        help=f'directory holding the result CSVs (default: {default})')
    parser.add_argument('--period', default='P2', choices=PERIODS,
                        help='period whose localization tables feed figures 2 and 3')
    parser.add_argument('--configs', default='',
                        help='comma separated configuration ids for figures 2 and 3 '
                             '(default: every STABLE_GAP configuration)')
    parser.add_argument('--no-localization', action='store_true',
                        help='write figure 1 only, when no localization table exists')
    args = parser.parse_args(argv)
    results = args.results.expanduser().resolve()
    wanted = [c.strip() for c in args.configs.split(',') if c.strip()]
    if wanted and args.no_localization:
        print('ERROR: --configs and --no-localization contradict each other', file=sys.stderr)
        return 2
    try:
        data = load_inputs(results, args.period, wanted, not args.no_localization)
    except DataError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2

    outdir = results/'figures'
    outdir.mkdir(parents=True, exist_ok=True)
    written = [FIG1]
    fig1_stats = figure1(data, outdir)
    fig2_ends, fig3_points = {}, {}
    if data['want_localization']:
        fig2_ends = figure2(data, outdir)
        fig3_points = figure3(data, outdir)
        written += [FIG2, FIG3]
    pd.DataFrame(VALUES, columns=['figure', 'element', 'series', 'label', 'x', 'y',
                                  'y_low', 'y_high', 'unit', 'source_file']) \
        .to_csv(results/'FIGURE_VALUES.csv', index=False)
    captions(data, fig1_stats, fig2_ends, fig3_points, results)
    for name in written:
        print('wrote', ', '.join(str(outdir/f'{name}.{e}') for e in ('png', 'svg', 'pdf')))
    print('wrote', results/'FIGURE_VALUES.csv', f'({len(VALUES)} rows)')
    print('wrote', results/'CAPTIONS.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
