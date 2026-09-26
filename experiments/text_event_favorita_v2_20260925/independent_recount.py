"""Independent C0 recount directly from the original CSVs, without c0_count imports."""
from pathlib import Path
import collections
import json
import re
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/text_event_favorita_v2_20260925'
with zipfile.ZipFile(Path(__file__).parent/'.cache/store-sales-time-series-forecasting.zip') as z:
    holidays = pd.read_csv(z.open('holidays_events.csv'), parse_dates=['date'])
    stores = pd.read_csv(z.open('stores.csv'))
    sales = pd.read_csv(z.open('train.csv'), usecols=['date', 'store_nbr', 'sales'], parse_dates=['date'])
totals = sales.groupby(['store_nbr', 'date']).sales.sum().unstack(0).sort_index()
last_sales_date = sales.date.max()
prior = totals.shift(1).rolling(28, min_periods=28).mean()
calendar = collections.defaultdict(lambda: collections.defaultdict(set))
local = set()
for row in holidays.itertuples():
    if row.transferred or row.type == 'Work Day':
        continue
    concept = row.description
    for prefix in ['Traslado ', 'Puente ', 'Recupero ']:
        if concept.startswith(prefix):
            concept = concept[len(prefix):]
    concept = re.sub(r'[+\-−]\d+$', '', concept).split(':')[0].strip()
    if row.locale != 'National':
        local.add(concept)
    ids = (stores.store_nbr if row.locale == 'National' else
           stores.loc[stores['state' if row.locale == 'Regional' else 'city'] == row.locale_name, 'store_nbr'])
    for store in ids:
        calendar[store][concept].add(row.date)
kept = []
reasons = collections.Counter()
before = 0
for store, concepts in calendar.items():
    for concept, dates in concepts.items():
        for date in sorted(dates):
            if date-pd.Timedelta(days=1) in dates:
                continue
            if not pd.Timestamp('2016-01-08') <= date <= last_sales_date:
                continue
            before += 1
            excluded = []
            if 'terremoto' in concept.lower():
                excluded.append('disaster')
            if totals.loc[date, store] < .1 * prior.loc[date, store]:
                excluded.append('closure')
            if any(other != concept and any(date-pd.Timedelta(days=4) <= day <= date+pd.Timedelta(days=3)
                    for day in other_dates) for other, other_dates in concepts.items()):
                excluded.append('overlap')
            if excluded:
                reasons.update(excluded)
            else:
                kept.append((concept, str(date.date()), int(store)))
names = {row[0] for row in kept}
families = {prefix: sorted(c for c in names if c.startswith(prefix))
            for prefix in ['Fundacion de ', 'Cantonizacion de ', 'Provincializacion de ']}
counts = {'events': len({(c,d) for c,d,s in kept}), 'concepts': len(names),
          'local_regional_concepts': len(names & local), 'store_events': len(kept),
          'sibling_families': sum(len(members) >= 2 for members in families.values())}
thresholds = dict(events=40, concepts=25, local_regional_concepts=10, store_events=300, sibling_families=3)
result = {'counts': counts, 'thresholds': thresholds,
          'pass_all': all(counts[k] >= v for k,v in thresholds.items()),
          'raw_store_events': before, 'exclusion_reasons_nonexclusive': dict(reasons),
          'sibling_members': families, 'kept_store_events': kept,
          'source': 'Direct original CSV recount; no c0_count functions/results read',
          'overlap_bounds': '[d-4,d+3] on active days of any other concept including earthquake',
          'model_calls': 0}
with (OUT/'C0_INDEPENDENT.json').open('x', encoding='utf8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps(counts), result['pass_all'])
