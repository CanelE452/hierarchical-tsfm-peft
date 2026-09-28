"""Pre-TEST continuation decision from completed base fits, without new scoring."""
import argparse
import copy
import json
from runtime_v7 import HERE, CACHE, Job, budget_snapshot, save_json

parser=argparse.ArgumentParser()
parser.add_argument('--dataset',choices=['robin','jena'],required=True)
args=parser.parse_args()
with Job('extension_decision_'+args.dataset,category='cpu_analysis',reserve_seconds=30,metadata={'dataset':args.dataset}):
    target=HERE/f'extension_decision_{args.dataset}.json'
    if target.exists() or (HERE/f'test_exposure_{args.dataset}.json').exists():
        raise RuntimeError('Do not replace an existing decision or decide extensions after TEST')
    protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    specs=[s for s in protocol['fits'] if s['dataset']==args.dataset]
    results=[json.loads((HERE/'runs'/s['id']/'result.json').read_text(encoding='utf-8')) for s in specs]
    if any(r['status']!='complete' for r in results):
        raise RuntimeError('Resolve invalid required fit before continuation decision')
    eligible=[r for r in results if r['epochs_completed']==120 and r['final_stale']<6 and 120-r['selected_epoch']<6]
    budget=budget_snapshot()
    future_base=12 if args.dataset=='robin' else 0
    remaining=20-budget['real_fit_attempts']-future_base
    authorize=bool(eligible) and len(eligible)<=remaining
    ids={r['id'] for r in eligible}
    effective=copy.deepcopy(specs)
    extensions=[]
    if authorize:
        for spec in effective:
            if spec['id'] in ids:
                parent=spec['id']
                spec.update(id=parent+'_extended240',epochs=240,
                            resume_from=str(CACHE/'runs'/parent/'restart.pt'),
                            decision_record=target.name)
                extensions.append(spec)
    decision=dict(dataset=args.dataset,test_open=False,
                  criterion='Reached 120, not patience-terminated, strict best within final six epochs; all eligible or none',
                  eligible=[r['id'] for r in eligible],
                  authorized_continuations=[s['id'] for s in extensions],
                  remaining_fit_reserve=remaining,future_base_fits_protected=future_base,
                  decision='continue_all_eligible' if authorize else 'none_eligible' if not eligible else 'insufficient_reserve_for_all_eligible',
                  base_results=[dict(id=r['id'],epochs=r['epochs_completed'],best_epoch=r['selected_epoch'],stale=r['final_stale']) for r in results],
                  budget=budget)
    save_json(target,decision)
    save_json(HERE/f'effective_specs_{args.dataset}.json',effective)
    print(json.dumps(decision,indent=2))

