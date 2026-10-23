"""Rerunnable scalar-contract audit. Never imports the campaign or opens binaries."""
import argparse
import copy
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys


NAME = 'k_tau_rule_20261023_v1'
CORE = ('fit_base.py', 'training.py', 'networks.py', 'data_contract.py',
        'gpu_runtime.py', 'evaluation.py')
FIXED_SPEC = dict(family='B2', init_kind='PCA', revin=False, device='cuda',
                  fold='primary', core_dtype='float32', lr=.001,
                  effective_batch=16, microbatch=4, F_row_microbatch=128,
                  checkpoint_inputs=False, epochs=[.25, 1., 3.])
PROVENANCE = {'run_id', 'scientific_identity', 'new_scientific_source_hashes',
              'custom_source_sha256'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


class Audit:
    def __init__(self, repo):
        self.repo = repo.resolve()
        self.report = self.repo/'reports'/NAME
        self.cache = self.repo/'.cache'/NAME
        self.inputs = {}
        self.source_hashes = {}
        self.prereg = self.read(self.report/'PREREGISTRATION.json')
        self.commit = self.read(self.report/'PREREGISTRATION_COMMIT.json')
        self.history = self.read(self.report/'HISTORICAL_FINGERPRINTS.json')
        self.plan = self.read(self.report/'RUN_PLAN.json')
        self.preflight = self.read(self.report/'RUN_PLAN_PREFLIGHT.json')

    def relative(self, path):
        return path.resolve().relative_to(self.repo).as_posix()

    def read(self, path):
        path = path.resolve()
        if path.suffix != '.json' or not path.is_relative_to(self.repo):
            raise ValueError('Only repository JSON metadata may be read: '+str(path))
        self.inputs[self.relative(path)] = sha(path)
        return json.loads(path.read_text(encoding='utf-8-sig'))

    def source_sha(self, name):
        path = (self.repo/name).resolve()
        if path.suffix != '.py' or not path.is_relative_to(self.repo):
            raise ValueError('Only repository Python source bytes may be hashed')
        result = sha(path)
        self.source_hashes[self.relative(path)] = result
        return result

    def bound_json(self, binding, identity):
        scheme, relative = binding['reference'].split('://', 1)
        root = Path(identity['roots'][scheme]).resolve()
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or path.suffix != '.json':
            raise ValueError('Expected a contained scalar JSON binding')
        return path, self.read(path)

    def reference(self, dataset):
        run = min(self.history['datasets'][dataset]['baseline_runs'],
                  key=lambda item: (item['K'], item['seed']))
        source_path = self.repo/run['source_contract']
        source = self.read(source_path)
        result = self.read(source_path.with_name('REFIT_RESULT.json'))
        return run, source_path, source, result

    @staticmethod
    def check(checks, field, expected, actual, evidence):
        checks.append(dict(field=field, expected=expected, actual=actual,
                           pass_=expected == actual, evidence=evidence))

    def check_sources(self, checks, source):
        for name in CORE:
            self.check(checks, 'core_source.'+name, self.prereg['core6'][name],
                       source['files'].get(name), 'Preregistered shared core source identity')
            self.check(checks, 'current_core_source.'+name, self.prereg['core6'][name],
                       self.source_sha('research/informs_broad_validation_20261015_v1/'+name),
                       'Current source bytes only; no import/execution')

    def audit_job(self, entry):
        job, role = entry['job'], entry['role']
        dataset, identifier = job['cohort'], job['job_id']
        directory = self.report/'datasets'/dataset
        public = directory/'jobs'/identifier
        run = directory/'runs'/identifier/'cap3'
        identity_path, source_path = public/'IDENTITY.json', run/'SOURCE_CONTRACT.json'
        if not identity_path.is_file() or not source_path.is_file():
            return None
        identity, source = self.read(identity_path), self.read(source_path)
        reference_run, reference_path, oldsource, oldresult = self.reference(dataset)
        checks = []
        check = lambda field, expected, actual, evidence: self.check(checks, field, expected, actual, evidence)
        expected_spec = dict(FIXED_SPEC, dataset=dataset, latent_k=job['K'], seed=job['seed'])
        for name, value in expected_spec.items():
            check('spec.'+name, value, source['spec'].get(name), 'Registered recipe and exact planned K/seed')
        unknown = sorted(set(source['spec']) - set(expected_spec) - PROVENANCE)
        check('spec.unclassified_fields', [], unknown, 'G5 forbids unclassified contract fields')
        check('identity.job', job, identity['job'], 'Exact admitted public plan job')
        check('identity.role', role, identity['role'], 'Exact admitted public plan role')
        check('identity.spec', source['spec'], dict(identity['spec'], epochs=identity['epochs']),
              'Pre-search identity plus separately registered epochs')
        check('identity.cap', 3, identity['cap'], 'Fixed cap')
        check('identity.T_access', False, identity['T_access'], 'Declared metadata only; no trace claim')
        check('identity.protocol_sha256', self.commit['preregistration_sha256'], identity['protocol_sha256'],
              'Immutable preregistration binding')
        check('source.data_sha', oldsource['data_sha'], source['data_sha'], 'Historical exact dataset identity')
        check('source.model_sha', oldsource['model_sha'], source['model_sha'], 'Historical frozen foundation identity')
        check('identity.data_sha256', source['data_sha'], identity['data_sha256'], 'Exact source binding')
        check('identity.foundation_sha256', source['model_sha'], identity['foundation_sha256'], 'Exact source binding')
        check('identity.split_ranges', self.history['datasets'][dataset]['split_ranges'], identity['split_ranges'],
              'Historical boundary metadata only, no numeric T access')
        check('identity.wrapper_sha256', self.prereg['registered_sources']['research/'+NAME+'/runner.py'],
              identity['wrapper_sha256'], 'Registered wrapper source identity')
        self.check_sources(checks, source)
        for name, expected in identity['source_hashes'].items():
            check('identity.producer_current.'+name, expected, self.source_sha(name),
                  'Byte-bound producer source; no model or data access')
        reference_science = {name: oldsource['spec'].get(name) for name in FIXED_SPEC}
        for name, expected in FIXED_SPEC.items():
            check('historical_effective_recipe.'+name, expected, reference_science[name],
                  'Minimum-K baseline has full explicit recipe; K/seed are permitted differences')
        complete = (public/'RESULT.json').is_file()
        result = None
        artifacts = dict(identity=self.relative(identity_path), source=self.relative(source_path),
                         historical_source=self.relative(reference_path))
        if complete:
            result = self.read(public/'RESULT.json')
            selection = self.read(public/'SELECTION.json')
            search = self.read(run/'SEARCH_RESULT.json')
            row, oldrow = result['rows'][0], oldresult['rows'][0]
            check('result.row_count', 1, len(result['rows']), 'Exactly one V result')
            check('result.status', 'COMPLETE', result['status'], 'Completed public result')
            for name, value in oldresult['recipe'].items():
                check('result.recipe.'+name, value, result['recipe'].get(name), 'Historical full optimization recipe')
            for name in ('L','H','C','P_params','RevIN_params','evaluation_stride','core_dtype',
                         'prediction_dtype','metric_accumulation_dtype','point_statistic','units',
                         'fit','evaluation','selection_split'):
                check('result.row.'+name, oldrow[name], row.get(name), 'Historical fixed scientific field')
            channels, k = oldrow['C'], job['K']
            for name, expected in dict(E_params=channels*k, D_params=7*channels*k,
                                      target_trained_params=oldrow['P_params']+8*channels*k).items():
                check('result.row.'+name, expected, row.get(name),
                      'Unchanged ForecastAdapter structure with approved new K; no tensor inspection')
            for name, expected in dict(dataset=dataset,K=k,seed=job['seed'],family='B2',revin=False,
                                      device='cuda',cap=3).items():
                check('result.row.'+name, expected, row.get(name), 'Exact planned scientific axes')
            curve = [(float(r['epoch']),float(r['astop_mse'])) for r in search['curves']
                     if r.get('astop_mse') is not None]
            check('search.curve_epochs',[0.,.25,1.,3.],[r[0] for r in curve], 'Registered A_stop endpoints')
            best = min(curve,key=lambda value:(value[1],value[0]))
            for name, expected in dict(status='SEALED',cap=3,V_used=False,selection_split='A_stop',
                                       job_id=identifier,cohort=dataset,selected_epoch=best[0],
                                       selected_astop_mse=best[1],spec=source['spec'],contract=search['contract'],
                                       protocol_sha256=identity['protocol_sha256']).items():
                check('selection.'+name,expected,selection.get(name), 'Actual byte-bound pre-refit selection')
            receipt, oldreceipt = result['fit_receipt'],oldresult['fit_receipt']
            for name in ('search_fit_end','refit_fit_end','normalization_fit_end','search_train_origins',
                         'search_stop_origins','optimizer','weight_decay','refit_scaler_sha'):
                check('fit_receipt.'+name,oldreceipt[name],receipt.get(name), 'Same fit/refit/scaler scientific scope')
            check('fit_receipt.fresh_A60',True,receipt['fresh_A60'],'Fresh refit actual receipt')
            check('fit_receipt.selected_epoch',best[0],receipt['selected_epoch'],'Selected-epoch propagation')
            check('refit.completed_epochs',[0.]+([best[0]] if best[0] else []),
                  result['refit_summary']['completed_epochs'],'Selected epoch only, with fresh A60 initialization')
            check('scaler_sha256',oldrow['scaler_sha256'],row['scaler_sha256'],'Identical fixed A60 scaler')
            check('independent_FP64','PASS',row['independent_FP64'],'Existing independent scorer receipt')
            check('selection_sealed_before_V',True,result['selection_sealed_before_V'],'Actual helper receipt')
            check('selection.sha256',sha(public/'SELECTION.json'),result['selection_decision_binding']['sha256'],
                  'Only public JSON is opened/hashed')
            check('selection.identity_sha256',sha(identity_path),selection['identity_binding']['sha256'],
                  'Only public JSON is opened/hashed')
            bound, canonical = self.bound_json(selection['search_binding'],identity)
            check('selection.search_sha256',sha(bound),selection['search_binding']['sha256'],
                  'Canonical scalar JSON in cache, not checkpoint')
            check('selection.search_public_projection',search,
                  {name:canonical.get(name) for name in search},'Public redacted search matches canonical fields')
            for name in ('data_sha','model_sha'):
                check('result.'+name,source[name],result[name],'Result/source exact identity')
            for name in ('source_sha','config_sha','data_sha','model_sha'):
                check('selection.source_contract.'+name,source[name],selection['contract'][name],
                      'Actual source aggregate bound by selection; aggregate may differ from historical producer')
            artifacts.update(result=self.relative(public/'RESULT.json'),selection=self.relative(public/'SELECTION.json'),
                             canonical_search_metadata=self.relative(bound))
        return dict(dataset=dataset,job_id=identifier,K=job['K'],seed=job['seed'],role=role,
                    complete=complete,status='PASS' if all(c['pass_'] for c in checks) else 'FAIL',
                    checks=checks,artifacts=artifacts,
                    comparison_reference=dict(K=reference_run['K'],seed=reference_run['seed'],
                                              note='Different K/seed are explicitly allowed; parameter counts scale by current K.'),
                    excluded_provenance_fields=sorted(PROVENANCE))

    def audit_reused(self, record):
        dataset,k,seed = record['dataset'],record['K'],record['seed']
        matches = [r for r in self.history['datasets'][dataset]['baseline_runs'] if r['K']==k and r['seed']==seed]
        if len(matches)!=1:
            raise ValueError('Expected exactly one historical reused K/seed')
        run = matches[0]
        source_path=self.repo/run['source_contract'];source=self.read(source_path)
        receipt=self.read(source_path.with_name('BASE_FIT_RECEIPT.json'))
        reuse_control = None
        oldresult = None
        if dataset=='jena' and k==6:
            path=self.repo/f'reports/adapter_diagnostics_20261017_v2/receipts/C02_s{seed}.json'
            rows=[r for r in self.read(path)['result']['rows'] if r['method']=='B2_PCA']
            if len(rows)!=1:raise ValueError('Expected original Jena PCA seed row')
            row=rows[0]
            reuse=self.read(self.repo/'reports/b2_revin_20261018_v1/REUSE_AUDIT.json')
            reuse_control=next(r for r in reuse['controls'] if r['cohort']==dataset and r['seed']==seed)
        else:
            path=source_path.with_name('REFIT_RESULT.json');oldresult=self.read(path);row=oldresult['rows'][0]
        checks=[]
        check=lambda field,expected,actual,evidence:self.check(checks,field,expected,actual,evidence)
        check('G4.reuse_allowed',True,self.plan['G4'][dataset]['reused_historical_results'],'Admitted G4 fallback decision')
        check('K_main',self.plan['K_main'][dataset],k,'Preselected A_fit K, no V reselection')
        check('seed_allowed',True,seed in (92601,92602),'Only original two seeds may be reused')
        for name,expected in dict(dataset=dataset,K=k,seed=seed).items():check('row.'+name,expected,row[name],'Original exact scored row')
        check('plan.MSE',row['MSE'],record['mse'],'Plan score exactly equals original execution JSON')
        check('plan.MAE',row['MAE'],record['mae'],'Plan score exactly equals original execution JSON')
        check('plan.source_paths_present',True,bool(record.get('source_paths')),'Public baseline provenance retained')
        if reuse_control is not None:
            check('reuse_control.status','EXACT_OFF_REUSE_VERIFIED',reuse_control['status'],'Original Jena K6 reuse proof')
            check('reuse_control.source_contract_sha256',sha(source_path),reuse_control['source_contract']['sha256'],
                  'Hash of original public source contract JSON only')
            check('reuse_control.receipt_sha256',sha(source_path.with_name('BASE_FIT_RECEIPT.json')),
                  reuse_control['receipt']['sha256'],'Hash of original public receipt JSON only')
            check('reuse_control.bundle_binding_sha256',receipt['artifact_sha256'],reuse_control['bundle']['sha256'],
                  'Stored binary binding metadata compared; bundle not opened')
            check('reuse_control.prediction_binding_sha256',row['prediction_sha256'],reuse_control['prediction']['sha256'],
                  'Stored binary binding metadata compared; prediction not opened')
        else:
            check('historical_result.bundle_binding_sha256',receipt['artifact_sha256'],oldresult['bundle_binding']['sha256'],
                  'Stored binary binding metadata compared; bundle not opened')
            verified_path=self.repo/f'reports/b2_dataset_search_20261019_v1/datasets/{dataset}/VERIFIED_ROWS.json'
            verified=next(r for r in self.read(verified_path) if r['K']==k and r['seed']==seed)
            check('verified_row.prediction_binding',row['prediction_binding'],verified['prediction_binding'],
                  'Original public verified row and original refit row bind the same prediction')
            check('verified_row.bundle_binding',oldresult['bundle_binding'],verified['bundle_binding'],
                  'Original public verified row and original refit result bind the same bundle')
        check('data_sha',run['historical_data_sha'],source['data_sha'],'Pinned historical scientific fingerprint')
        check('model_sha',run['historical_model_sha'],source['model_sha'],'Pinned historical foundation fingerprint')
        check('recipe.lr',.001,source['spec']['lr'],'Original spec')
        for name in ('effective_batch','microbatch','F_row_microbatch','checkpoint_inputs','epochs','family','init_kind'):
            check('recipe.'+name,FIXED_SPEC[name],source['spec'].get(name),'Original explicit recipe')
        for name in ('revin','core_dtype'):
            check('recipe.'+name,FIXED_SPEC[name],source['spec'].get(name,FIXED_SPEC[name]),
                  'Jena K6 missing explicit field uses previously audited unchanged original PCA factory; REUSE_AUDIT records OFF reuse')
        check('scaler_sha',receipt['refit_scaler_sha'],row['scaler_sha256'],'Original saved fixed A60 scaler')
        self.check_sources(checks,source)
        binary_bindings={name:receipt.get(name) for name in ('artifact_sha256','artifact_cache_relative_path','model_state_sha')}
        return dict(dataset=dataset,K=k,seed=seed,job_id=record['job_id'],status='PASS' if all(c['pass_'] for c in checks) else 'FAIL',
                    checks=checks,artifacts=dict(source=self.relative(source_path),result=self.relative(path)),
                    stored_binary_bindings_not_opened=binary_bindings,
                    scope='Binding and scalar metadata only; no checkpoint/prediction file bytes inspected')

    def run(self):
        output=self.report/'ACTUAL_CONTRACT_AUDIT.json'
        backup=self.cache/'initial_contract_audit_394.json'
        if not backup.exists():
            original=self.read(output)
            self.inputs.pop(self.relative(output),None)
            if sum(len(r['checks']) for r in original['audited_jobs'])!=394:
                raise ValueError('Initial 394-check audit must be preserved before expansion')
            with backup.open('x',encoding='utf-8') as stream:
                json.dump(original,stream,ensure_ascii=False,indent=2);stream.write('\n')
        initial=self.read(backup)
        baseline_checks=[]
        self.check(baseline_checks,'preregistration.sha256',self.commit['preregistration_sha256'],
                   sha(self.report/'PREREGISTRATION.json'),'Immutable committed preregistration')
        for name,value in self.prereg['registered_sources'].items():
            self.check(baseline_checks,'registered_source.'+name,value,self.source_sha(name),'Registered source unchanged')
        jobs=[];pending=[];seen=set()
        for entry in self.preflight['jobs']+self.plan['jobs']:
            identifier=entry['job']['job_id']
            if identifier in seen:raise ValueError('Duplicate admitted job identity')
            seen.add(identifier)
            current=self.audit_job(entry)
            if current is None:pending.append(identifier)
            else:jobs.append(current)
        reused=[self.audit_reused(r) for r in self.plan['reused']]
        # Initial recorded input hashes remain immutable evidence.
        for name,value in initial['input_public_metadata_sha256'].items():
            self.check(baseline_checks,'initial_metadata.'+name,value,sha(self.repo/name),'Original 394-check audit input unchanged')
        failures=[dict(job_id=case['job_id'],field=check['field']) for case in jobs+reused
                  for check in case['checks'] if not check['pass_']]
        failures.extend(dict(job_id='GLOBAL',field=c['field']) for c in baseline_checks if not c['pass_'])
        if not all(sha(self.repo/name)==value for name,value in self.inputs.items()):
            raise ValueError('Metadata changed during audit; retry after completion')
        # Retain original 394 checks without reclassifying them as checks of later jobs.
        value=copy.deepcopy(initial)
        value.update(status='FAIL' if failures else 'PASS_FOR_AUDITED_CONTRACTS',
            updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            scope='Listed completed jobs, present in-progress source contracts and six reused result bindings only; pending jobs are not certified.',
            initial_394_checks_backup=self.relative(backup),global_current_checks=baseline_checks,
            current_job_audits=jobs,reused_historical_bindings=reused,pending_job_ids=pending,
            remaining_scope=pending+[r['job_id'] for r in jobs if not r['complete']],
            failure_fields=failures,input_public_metadata_sha256=self.inputs,
            source_byte_hashes=self.source_hashes,
            counts=dict(initial_preserved_checks=394,completed_public_jobs=sum(r['complete'] for r in jobs),
                        in_progress_public_source_jobs=sum(not r['complete'] for r in jobs),
                        reused_historical_rows=len(reused),planned_new_main_jobs=len(self.plan['jobs']),
                        pending_without_source=len(pending)),
            audit_script=dict(path=self.relative(Path(__file__)),sha256=sha(Path(__file__))),
            access=dict(model_or_checkpoint_files_opened=0,prediction_NPZ_opened=0,
                        numeric_dataset_or_T_files_opened=0,new_training_or_PCA_executions=0,GPU_queries=0))
        save(output,value)
        print(json.dumps(dict(status=value['status'],counts=value['counts'],failure_fields=failures,
                              output=self.relative(output)),ensure_ascii=False))
        return 2 if failures else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    return Audit(args.repo).run()


if __name__=='__main__':
    sys.exit(main())
