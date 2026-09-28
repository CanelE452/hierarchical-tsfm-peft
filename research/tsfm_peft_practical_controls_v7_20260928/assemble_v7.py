"""Small source-linked tables and derived values, no fitting or inference."""
import csv
import json
from runtime_v7 import HERE, Job, digest, budget_snapshot, save_json

def load(name):
    return json.loads((HERE/name).read_text(encoding='utf-8'))

def write_csv(name,rows):
    path=HERE/name
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return {'path':str(path),'sha256':digest(path),'rows':len(rows)}

with Job('assemble_report01',category='cpu_analysis',reserve_seconds=120):
    import numpy as np
    accuracy,training,cost_rows=[],[],[]
    comparisons={}
    for dataset in ('robin','jena'):
        evaluation=load(dataset+'_eval01.json')
        if evaluation['status']!='complete':
            raise RuntimeError('Incomplete required evaluation')
        comparisons[dataset]=evaluation['comparisons']
        for role,periods in evaluation['role_scores'].items():
            for period,score in periods.items():
                accuracy.append(dict(dataset=dataset,role=role,period=period,aggregation='mean_individual_seed_loss',
                                     mse=score['mse'],mae=score['mae'],model_count=score['count']))
        for identifier,row in evaluation['models'].items():
            for period,score in row['scores'].items():
                accuracy.append(dict(dataset=dataset,role=row['role'],period=period,
                                     aggregation='individual_model',id=identifier,seed=row['spec'].get('seed'),
                                     mse=score['mse'],mae=score['mae'],model_count=1))
    for path in sorted((HERE/'runs').glob('*/result.json')):
        row=json.loads(path.read_text(encoding='utf-8'))
        if row.get('deterministic'):
            continue
        training.append(dict(id=row['id'],dataset=row['spec']['dataset'],family=row['spec']['family'],
                             seed=row['spec']['seed'],lr=row['spec']['lr'],
                             epochs_completed=row['epochs_completed'],selected_epoch=row['selected_epoch'],
                             initial_train_probe_mse=row['initial_train_probe']['mse'],
                             selected_train_probe_mse=row['selected_train_probe']['mse'],
                             initial_val_mse=row['initial_val_mse'],best_val_mse=row['best_val_mse'],
                             updates=row['attempt_updates'],replay_max_abs=row['replay_error'],
                             trainable_parameters=row['trainable_parameters'],total_parameters=row['total_parameters'],
                             elapsed_s=row['elapsed_s']))
    cost=load('cost_summary.json')
    for group in cost['groups']:
        fits=[r for r in cost['fits'] if r['family']==group['family'] and r['batch']==group['batch']
              and r['variant']==group['variant'] and r['micro']==group['micro'] and r['device']==group['device']]
        reserved=[p for r in fits for p in r['peak_reserved_bytes'] if p is not None]
        item=dict(family=group['family'],batch=group['batch'],device=group['device'],
                  variant=group['variant'],series_microbatch=group['micro'],
                  mean_seed_median_ms_per_origin=group['mean_seed_median_ms'],
                  mean_seed_median_origins_per_second=group['mean_seed_median_throughput'],
                  typical_peak_allocated_MiB=group['mean_seed_median_peak_allocated_bytes']/1024**2
                  if group['mean_seed_median_peak_allocated_bytes'] is not None else None,
                  observed_peak_allocated_min_MiB=group['observed_peak_allocated_range_bytes'][0]/1024**2
                  if group['observed_peak_allocated_range_bytes'] else None,
                  observed_peak_allocated_max_MiB=group['observed_peak_allocated_range_bytes'][1]/1024**2
                  if group['observed_peak_allocated_range_bytes'] else None,
                  observed_peak_reserved_min_MiB=min(reserved)/1024**2 if reserved else None,
                  observed_peak_reserved_max_MiB=max(reserved)/1024**2 if reserved else None,
                  trainable_parameters=fits[0]['model']['trainable_parameters_before_merge'],
                  deployment_parameters=fits[0]['model']['deployed_parameters'],
                  deployment_tensor_bytes=fits[0]['model']['deployment_tensor_bytes'])
        cost_rows.append(item)
    paths={name:digest(HERE/name) for name in ('robin_eval01.json','jena_eval01.json','cost_summary.json',
                                               'microbatch_parity.json','robin_diagnostics_corrected_v7.json')}
    diagnostic=load('robin_diagnostics_corrected_v7.json')
    space_path=HERE/'diagnostics_corrected/robin_complete_row_space_summary_corrected.csv'
    with space_path.open(newline='',encoding='utf-8') as stream:
        space=[r for r in csv.DictReader(stream) if r['member_id']=='mean_of_member_losses']
    space_deltas=[]
    for period in ('test_a','test_b'):
        roles={r['role']:r for r in space if r['period']==period}
        for reference in ('old_res','f0','lora','direct_nlinear'):
            left,right=roles['level_res'],roles[reference]
            space_deltas.append(dict(period=period,reference=reference,
                complete_rows=int(left['complete_origin_horizon_rows']),
                delta_retained=float(left['p_retained_space_mse'])-float(right['p_retained_space_mse']),
                delta_discarded=float(left['q_residual_space_mse'])-float(right['q_residual_space_mse']),
                delta_total=float(left['total_complete_row_mse'])-float(right['total_complete_row_mse'])))
    cost_relative=[]
    for batch in (1,4):
        reference=next(r for r in cost_rows if r['family']=='level_res' and r['batch']==batch
                       and r['variant']=='original' and r['device']=='cuda')
        for row in (r for r in cost_rows if r['batch']==batch and r['device']=='cuda'):
            cost_relative.append(dict(family=row['family'],batch=batch,variant=row['variant'],
                series_microbatch=row['series_microbatch'],reference='LEVEL original, same-session',
                ms_relative_pct=100*(row['mean_seed_median_ms_per_origin']/reference['mean_seed_median_ms_per_origin']-1),
                throughput_relative_pct=100*(row['mean_seed_median_origins_per_second']/reference['mean_seed_median_origins_per_second']-1),
                allocated_relative_pct=100*(row['typical_peak_allocated_MiB']/reference['typical_peak_allocated_MiB']-1)))
    report=dict(status='complete',evidence='confirmed from saved result files, not new predictions',
                sources=paths,accuracy=accuracy,comparisons=comparisons,training=training,cost=cost_rows,
                diagnostic=dict(coverage=diagnostic['coverage'],space_deltas=space_deltas,
                                space_source=dict(path=str(space_path),sha256=digest(space_path)),
                                scope='Complete-row energy per channel, seed losses averaged; distinct from primary masked macro MSE.'),
                same_session_cost_changes=cost_relative,
                sentinels=cost['sentinels'],budget=budget_snapshot(),files={
                    'accuracy':write_csv('accuracy_comparison.csv',accuracy),
                    'training':write_csv('training_summary.csv',training),
                    'cost':write_csv('cost_comparison.csv',cost_rows)})
    save_json(HERE/'report_values.json',report)

