"""Summarize only complete dataset grids after the fixed cost guard stopped."""
import argparse

from runtime_v15 import HERE, DATASETS, Job, artifact, digest, protocol, read_json, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', required=True, type=float)
    args = parser.parse_args()
    with Job('cpu_analysis', args.job, reserve_s=args.reserve_s,
             metadata={'purpose': 'saved complete-dataset cost aggregation after budget guard',
                       'model_runs': 0, 'new_measurements': 0, 'fits': 0}) as job:
        from cost_v15 import helpers
        output = HERE / 'cost_available_summary.json'
        if output.exists():
            raise FileExistsError(output)
        raw = read_json(HERE / 'cost_cuda_rows.json')
        grid = read_json(HERE / 'cost_cuda_grid.json')
        guard = read_json(HERE / 'cost_guard.json')
        assert raw['status'] == 'budget_stopped'
        assert [r['key'] for r in raw['rows']] == [r['key'] for r in grid['rows'][:len(raw['rows'])]]
        assert len(raw['rows']) == guard['complete_rows'] < len(grid['rows']) == guard['expected_rows']
        complete = [ds for ds in DATASETS if
                    {r['key'] for r in raw['rows'] if r['dataset'] == ds} ==
                    {r['key'] for r in grid['rows'] if r['dataset'] == ds}]
        rows = [r for r in raw['rows'] if r['dataset'] in complete]
        subset = protocol()
        subset['units'] = {ds: subset['units'][ds] for ds in complete}
        summary = helpers().summarize(rows, 'cuda', subset)
        summary['status'] = 'complete_for_listed_datasets'
        summary['complete_datasets'] = complete
        save_json(output, {
            'status': 'partial_cost', 'complete_campaign': False,
            'selection_rule': 'Include a dataset only when every fixed grid row exists; no timing or accuracy filter',
            'original_cost_summary': artifact(HERE / 'cost_summary.json'),
            'raw_gpu': artifact(HERE / 'cost_cuda_rows.json'),
            'grid': artifact(HERE / 'cost_cuda_grid.json'), 'guard': artifact(HERE / 'cost_guard.json'),
            'source': artifact(__file__),
            'source_files': {name: digest(HERE / name) for name in
                             ('summarize_available_cost_v15.py', 'cost_v15.py', 'runtime_v15.py')},
            'complete_datasets': complete,
            'summarized_rows': len(rows), 'complete_rows': len(raw['rows']),
            'omitted_dataset_rows': {ds: sum(r['dataset'] == ds for r in raw['rows'])
                                     for ds in DATASETS if ds not in complete},
            'missing_grid_rows': grid['rows'][len(raw['rows']):], 'gpu': summary,
            'scope': 'Original raw rows remain intact; no partial-block median, new run, score, fit or budget extension',
            'job_id': job.id})
        job.heartbeat('Deterministic complete-dataset subset summarized; full campaign remains incomplete')


if __name__ == '__main__':
    main()
