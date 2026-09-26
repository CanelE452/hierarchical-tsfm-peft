"""Rebuild descriptive C0 figures and audit stored evidence; never call a model."""
from pathlib import Path
import hashlib
import json
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
OUT = ROOT / 'results/text_event_favorita_v2_20260925'


def main():
    counts = json.loads((OUT/'EVENT_COUNTS.json').read_text(encoding='utf8'))
    independent = json.loads((OUT/'C0_INDEPENDENT.json').read_text(encoding='utf8'))
    rows = pd.read_csv(OUT/'C0_EVENT_STORE_DETAILS.csv')
    retained = rows.loc[rows.passed_c0]
    events = pd.read_csv(OUT/'C0_EVENT_DETAILS.csv')
    actual = set(zip(retained.concept, retained.eval_date, retained.store_nbr.astype(int)))
    checks = {'exact_823_tuple_match': actual == {tuple(v) for v in independent['kept_store_events']},
              '44_unique_events': len(events)==44 and not events.duplicated(['concept','eval_date']).any(),
              '823_store_units': int(events.event_store_units.sum()) == len(actual) == 823,
              '29_concepts': events.concept.nunique()==29,
              'C0_pass': counts['pass_all'] is True,
              'T0_stopped': json.loads((OUT/'T0_REPORT.json').read_text(encoding='utf8'))['status']=='STOP_DEBUG',
              'no_scoring_seal': not (OUT/'SEAL.json').exists()}
    archive = EXP/'.cache/store-sales-time-series-forecasting.zip'
    manifest = json.loads((OUT/'DATA_MANIFEST.json').read_text(encoding='utf8'))
    checks['raw_archive_hash'] = hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['sha256']
    with zipfile.ZipFile(archive) as z:
        for member in manifest['members']:
            checks['raw_member_'+member['path']] = hashlib.sha256(z.read(member['path'])).hexdigest()==member['sha256']
        holidays = pd.read_csv(z.open('holidays_events.csv'))
    used_ids = {int(i) for value in retained.holiday_row_ids.astype(str) for i in value.split('|')}
    checks['no_transferred_source_rows'] = not holidays.loc[sorted(used_ids), 'transferred'].any()
    checks['no_workday_source_rows'] = not holidays.loc[sorted(used_ids), 'type'].eq('Work Day').any()
    prior_audit = json.loads((OUT/'C0_VERIFICATION.json').read_text(encoding='utf8'))
    checks['C0_stored_artifact_hashes'] = all(hashlib.sha256((OUT/name).read_bytes()).hexdigest()==sha
        for name,sha in prior_audit['artifact_sha256'].items())
    assert all(checks.values()), checks

    figures = OUT/'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
        'axes.spines.top':False, 'axes.spines.right':False, 'svg.fonttype':'none'})
    def save(fig, name):
        fig.savefig(figures/(name+'.png'), dpi=180, bbox_inches='tight', facecolor='white')
        fig.savefig(figures/(name+'.svg'), bbox_inches='tight', facecolor='white')
        svg_path = figures/(name+'.svg')
        svg_path.write_bytes(('\n'.join(line.rstrip() for line in svg_path.read_text(encoding='utf8').splitlines())+'\n').encode('utf8'))
        plt.close(fig)

    labels = ['Event occurrences', 'Distinct concepts', 'Local / regional concepts',
              'Event-store units', 'Sibling concept families']
    observed = [44,29,20,823,3]
    minimum = [40,25,10,300,3]
    source = pd.DataFrame({'criterion':labels,'observed':observed,'minimum':minimum})
    source['ratio'] = source.observed/source.minimum
    source.to_csv(figures/'c0_gate_counts.csv', index=False)
    fig, ax = plt.subplots(figsize=(9,4.5))
    ax.barh(labels[::-1], source.ratio[::-1], color='#0072B2', height=.58)
    ax.axvline(1, color='#333333', linestyle='--', label='Minimum = 1.0')
    for y,ratio,obs,threshold in zip(range(5),source.ratio[::-1],observed[::-1],minimum[::-1]):
        ax.text(ratio+.05,y,f'{obs:,} / {threshold:,}',va='center')
    ax.set_xlim(0,3.35); ax.set_xlabel('Observed count / prespecified minimum')
    ax.set_title('C0: all five data-volume criteria passed',loc='left',pad=16)
    ax.legend(loc='lower right',frameon=False)
    fig.text(.02,-.01,'Descriptive counts only. This is not a model-performance result.',fontsize=10)
    save(fig,'c0_gate_counts')

    stage = counts['stage_counts_event_concept_date']
    values = [stage[k] for k in ['active_grouped_event_concept_date_in_eval_before_exclusions',
        'after_earthquake_exclusion_event_concept_date','after_closure_exclusion_event_concept_date',
        'after_overlap_exclusion_event_concept_date']]
    names = ['Before exclusions','After earthquake','After closure','After overlap']
    pd.DataFrame({'stage':names,'events':values}).to_csv(figures/'c0_event_retention.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.bar(names,values,color=['#999999','#56B4E9','#56B4E9','#0072B2'],width=.55)
    for n,v in zip(names,values): ax.text(n,v+1,str(v),ha='center',fontweight='bold')
    ax.set_ylim(0,75); ax.set_ylabel('Unique (concept, date) occurrences')
    ax.set_title('65 candidate occurrences → 44 retained',loc='left',pad=16)
    fig.text(.02,-.01,'Store-level exclusions are applied before recounting unique occurrences.',fontsize=10)
    save(fig,'c0_event_retention')

    coverage=events.groupby('concept').agg(occurrences=('eval_date','size'),
        event_store_units=('event_store_units','sum')).sort_values(['event_store_units','concept'])
    coverage.to_csv(figures/'c0_concept_coverage.csv')
    fig,ax=plt.subplots(figsize=(10,10))
    ax.barh(coverage.index,coverage.event_store_units,color='#0072B2',height=.65)
    for y,row in enumerate(coverage.itertuples()):
        ax.text(row.event_store_units+.8,y,f'{row.event_store_units} ({row.occurrences} events)',va='center',fontsize=9)
    ax.set_xlim(0,140); ax.set_xlabel('Retained event-store units (not independent events)')
    ax.tick_params(axis='y',labelsize=9)
    ax.set_title('Store coverage varies widely across 29 concepts',loc='left',pad=16)
    fig.text(.02,-.005,'Statistical unit in the proposed screen: occurrence, not each store replication.',fontsize=10)
    save(fig,'c0_concept_coverage')
    audit={'status':'PASS_PUBLICATION_EVIDENCE_CHECKS','checks':checks,
        'scope':'Stored C0 evidence and raw source hashes; not full T0 or model validation',
        'new_model_calls':0,'new_training_updates':0,
        'figures':[p.name for p in sorted(figures.glob('*.png'))],
        'claim':'C0 passed; T0 implementation failed; predictive performance unmeasured'}
    (OUT/'PUBLICATION_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
